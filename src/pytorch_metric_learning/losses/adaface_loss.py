import math
from typing import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor

from ..types import IndicesTuple, LossDict, ProxyLossOptions
from ..utils import _common, indexing
from .large_margin_softmax_loss import _CosineMarginSoftmaxLoss

_NORM_MIN = 0.001
_NORM_MAX = 100
_STABILITY_EPS = 1e-3


class AdaFaceLoss(_CosineMarginSoftmaxLoss):
    batch_mean: Tensor
    batch_std: Tensor

    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin: float = 0.4,
        scale: float = 64,
        h: float = 0.333,
        t_alpha: float = 0.01,
        **options: Unpack[ProxyLossOptions],
    ) -> None:
        super().__init__(num_classes, embedding_size, margin, scale, **options)
        self.h = h
        self.t_alpha = t_alpha
        self.register_buffer("batch_mean", torch.ones(1) * 20, persistent=True)
        self.register_buffer("batch_std", torch.ones(1) * 100, persistent=True)

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        _common.ref_not_supported(ref_emb)
        _common.check_parameter_placement(self, self.weight, embeddings)
        cosine = self.get_cosine(embeddings)
        safe_norms = self.distance.get_norm(embeddings).clamp(
            _NORM_MIN, _NORM_MAX
        ).detach()
        if self.training:
            with torch.no_grad():
                mean = safe_norms.mean().to(self.batch_mean.dtype)
                self.batch_mean.mul_(1 - self.t_alpha).add_(mean * self.t_alpha)
                if len(safe_norms) > 1:
                    std = safe_norms.std().to(self.batch_std.dtype)
                    self.batch_std.mul_(1 - self.t_alpha).add_(std * self.t_alpha)
        margin_scaler = (
            (safe_norms - self.batch_mean) / (self.batch_std + _STABILITY_EPS) * self.h
        ).clamp(-1, 1)
        targets = labels.unsqueeze(1)
        target_cosine = cosine.gather(1, targets).squeeze(1)
        angles = torch.acos(
            target_cosine.clamp(-1 + _STABILITY_EPS, 1 - _STABILITY_EPS)
        )
        margin_angles = (angles - self.margin * margin_scaler).clamp(
            _STABILITY_EPS, math.pi - _STABILITY_EPS
        )
        modified = torch.cos(margin_angles) - self.margin - self.margin * margin_scaler
        margined = cosine.scatter(1, targets, modified.unsqueeze(1).to(cosine.dtype))
        logits = self.scale_logits(margined, embeddings)
        weights = indexing.convert_to_weights(indices_tuple, labels, dtype=logits.dtype)
        loss = F.cross_entropy(logits, labels, reduction="none") * weights
        return {"loss": {"losses": loss, "indices": _common.arange_like(embeddings)}}
