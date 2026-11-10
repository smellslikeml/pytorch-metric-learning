import math
import unittest

import torch
import torch.nn.functional as F

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import AdaFaceLoss, CosFaceLoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers

MARGIN = 0.4
H = 0.333
SCALE = 64
LABELS = [0, 1, 2, 3, 0, 3]


def batch_scaler(embeddings, batch_mean, batch_std, detach=True):
    norms = torch.norm(embeddings, 2, dim=1).clamp(0.001, 100)
    if detach:
        norms = norms.detach()
    return ((norms - batch_mean) / (batch_std + 1e-3) * H).clamp(-1, 1)


def reference_losses(embeddings, weight, labels, batch_mean, batch_std, detach=True):
    cosine = F.normalize(embeddings, dim=1) @ F.normalize(weight, dim=1).T
    scaler = batch_scaler(embeddings, batch_mean, batch_std, detach=detach)
    losses = []
    for row, label in enumerate(labels.tolist()):
        target = cosine[row, label].clamp(-1 + 1e-3, 1 - 1e-3)
        angle = (torch.acos(target) - MARGIN * scaler[row]).clamp(
            1e-3, math.pi - 1e-3
        )
        logits = cosine[row].clone()
        logits[label] = torch.cos(angle) - MARGIN - MARGIN * scaler[row]
        losses.append(-F.log_softmax(logits * SCALE, dim=0)[label])
    return torch.stack(losses)


class TestAdaFaceLoss(unittest.TestCase):
    def test_values_and_gradients_match_reference(self):
        torch.manual_seed(2204)
        loss_func = AdaFaceLoss(4, 8, t_alpha=0.5).to(TEST_DEVICE, torch.float64)
        with torch.no_grad():
            loss_func.batch_mean.fill_(2.5)
            loss_func.batch_std.fill_(0.75)
        embeddings = torch.randn(
            6, 8, dtype=torch.float64, device=TEST_DEVICE, requires_grad=True
        )
        labels = torch.tensor(LABELS, device=TEST_DEVICE)
        losses = loss_func.loss_terms(embeddings, labels)["loss"]["losses"]
        norms = torch.norm(embeddings, 2, dim=1).clamp(0.001, 100).detach()
        batch_mean = 0.5 * norms.mean() + 0.5 * 2.5
        batch_std = 0.5 * norms.std() + 0.5 * 0.75
        expected = reference_losses(
            embeddings, loss_func.weight, labels, batch_mean, batch_std
        )
        self.assertTrue(torch.allclose(losses, expected, atol=1e-6))
        gradient = torch.autograd.grad(losses.sum(), embeddings)[0]
        expected_gradient = torch.autograd.grad(expected.sum(), embeddings)[0]
        self.assertTrue(torch.allclose(gradient, expected_gradient, atol=1e-6))

    def test_gradients_flow_through_the_cosine_only(self):
        torch.manual_seed(9)
        loss_func = AdaFaceLoss(4, 8).to(TEST_DEVICE, torch.float64)
        embeddings = torch.randn(
            6, 8, dtype=torch.float64, device=TEST_DEVICE, requires_grad=True
        )
        labels = torch.tensor(LABELS, device=TEST_DEVICE)
        losses = loss_func.loss_terms(embeddings, labels)["loss"]["losses"]
        gradient = torch.autograd.grad(losses.sum(), embeddings)[0]
        batch_mean = loss_func.batch_mean.item()
        batch_std = loss_func.batch_std.item()
        through_norms = reference_losses(
            embeddings, loss_func.weight, labels, batch_mean, batch_std, detach=False
        )
        through_norms_gradient = torch.autograd.grad(through_norms.sum(), embeddings)[0]
        self.assertGreater((gradient - through_norms_gradient).abs().max().item(), 1e-3)

    def test_constant_norms_reduce_to_cosface(self):
        torch.manual_seed(64)
        labels = torch.tensor(LABELS, device=TEST_DEVICE)
        directions = F.normalize(torch.randn(6, 8, dtype=torch.float64), dim=1)
        for dtype in TEST_DTYPES:
            embeddings = (directions * 5).to(TEST_DEVICE, dtype)
            ada = AdaFaceLoss(4, 8, t_alpha=0.0).to(TEST_DEVICE, dtype)
            with torch.no_grad():
                ada.batch_mean.fill_(5.0)
            cosface = CosFaceLoss(4, 8, margin=MARGIN, scale=SCALE).to(
                TEST_DEVICE, dtype
            )
            with torch.no_grad():
                cosface.weight.copy_(ada.weight)
            helpers.assert_close(
                ada(embeddings, labels), cosface(embeddings, labels), dtype
            )

    def test_converged_statistics_match_cosface(self):
        torch.manual_seed(65)
        ada = AdaFaceLoss(4, 8, t_alpha=1.0).to(TEST_DEVICE, torch.float64)
        cosface = CosFaceLoss(4, 8, margin=MARGIN, scale=SCALE).to(
            TEST_DEVICE, torch.float64
        )
        with torch.no_grad():
            cosface.weight.copy_(ada.weight)
        embeddings = F.normalize(
            torch.randn(6, 8, dtype=torch.float64), dim=1
        ).to(TEST_DEVICE, torch.float64) * 5
        labels = torch.tensor(LABELS, device=TEST_DEVICE)
        loss = ada(embeddings, labels)
        self.assertAlmostEqual(ada.batch_mean.item(), 5.0, places=6)
        self.assertAlmostEqual(ada.batch_std.item(), 0.0, places=6)
        cosface_loss = cosface(embeddings, labels)
        self.assertAlmostEqual(loss.item(), cosface_loss.item(), places=6)

    def test_batch_statistics_follow_the_exponential_average(self):
        for t_alpha in (0.0, 0.5, 1.0):
            loss_func = AdaFaceLoss(4, 8, t_alpha=t_alpha).to(
                TEST_DEVICE, torch.float64
            )
            expected_mean = torch.tensor(20.0, dtype=torch.float64)
            expected_std = torch.tensor(100.0, dtype=torch.float64)
            for _ in range(3):
                embeddings = torch.randn(5, 8, dtype=torch.float64, device=TEST_DEVICE)
                labels = torch.randint(0, 4, (5,), device=TEST_DEVICE)
                loss_func(embeddings, labels)
                norms = torch.norm(embeddings, 2, dim=1).clamp(0.001, 100)
                expected_mean = t_alpha * norms.mean() + (1 - t_alpha) * expected_mean
                expected_std = t_alpha * norms.std() + (1 - t_alpha) * expected_std
            with self.subTest(t_alpha=t_alpha):
                self.assertAlmostEqual(
                    loss_func.batch_mean.item(), expected_mean.item(), places=5
                )
                self.assertAlmostEqual(
                    loss_func.batch_std.item(), expected_std.item(), places=5
                )

    def test_eval_mode_and_single_embeddings_leave_statistics_unchanged(self):
        loss_func = AdaFaceLoss(4, 8, t_alpha=1.0).to(TEST_DEVICE, torch.float64).eval()
        for batch in (5, 1):
            embeddings = torch.randn(batch, 8, dtype=torch.float64, device=TEST_DEVICE)
            labels = torch.randint(0, 4, (batch,), device=TEST_DEVICE)
            before = (loss_func.batch_mean.clone(), loss_func.batch_std.clone())
            loss_func(embeddings, labels)
            self.assertTrue(torch.equal(loss_func.batch_mean, before[0]))
            self.assertTrue(torch.equal(loss_func.batch_std, before[1]))
        loss_func.train()
        single = torch.randn(1, 8, dtype=torch.float64, device=TEST_DEVICE)
        before = loss_func.batch_std.clone()
        loss_func(single, torch.zeros(1, dtype=torch.long, device=TEST_DEVICE))
        self.assertTrue(torch.equal(loss_func.batch_std, before))

    def test_statistics_persist_in_state_dict_and_keep_identity(self):
        loss_func = AdaFaceLoss(4, 8, t_alpha=1.0).to(TEST_DEVICE, torch.float64)
        mean, std = loss_func.batch_mean, loss_func.batch_std
        embeddings = torch.randn(5, 8, dtype=torch.float64, device=TEST_DEVICE)
        labels = torch.randint(0, 4, (5,), device=TEST_DEVICE)
        loss_func(embeddings, labels)
        self.assertIs(loss_func.batch_mean, mean)
        self.assertIs(loss_func.batch_std, std)
        self.assertIn("batch_mean", loss_func.state_dict())
        restored = AdaFaceLoss(4, 8).to(TEST_DEVICE, torch.float64)
        restored.load_state_dict(loss_func.state_dict())
        self.assertTrue(torch.equal(restored.batch_mean, loss_func.batch_mean))
        self.assertTrue(torch.equal(restored.batch_std, loss_func.batch_std))

    def test_statistics_follow_module_placement(self):
        loss_func = AdaFaceLoss(3, 4)
        self.assertEqual(loss_func.batch_mean.dtype, torch.float32)
        moved = loss_func.to(TEST_DEVICE, torch.float64)
        self.assertEqual(moved.batch_mean.dtype, torch.float64)
        self.assertEqual(moved.batch_std.device.type, TEST_DEVICE.type)

    def test_larger_norms_get_a_larger_margin(self):
        torch.manual_seed(22)
        direction = F.normalize(
            torch.randn(1, 4, dtype=torch.float64), dim=1
        ).to(TEST_DEVICE, torch.float64)
        embeddings = torch.cat([direction * 5, direction * 15])
        labels = torch.zeros(2, dtype=torch.long, device=TEST_DEVICE)
        for h, differs in ((H, True), (0.0, False)):
            loss_func = AdaFaceLoss(3, 4, h=h, t_alpha=0.0).to(
                TEST_DEVICE, torch.float64
            ).eval()
            with torch.no_grad():
                loss_func.batch_mean.fill_(10.0)
                loss_func.batch_std.fill_(5.0)
            losses = loss_func.loss_terms(embeddings, labels)["loss"]["losses"]
            with self.subTest(h=h):
                if differs:
                    self.assertGreater(losses[1].item(), losses[0].item())
                else:
                    self.assertAlmostEqual(losses[1].item(), losses[0].item(), places=8)

    def test_clamped_norms_and_extreme_cosines_stay_finite(self):
        loss_func = AdaFaceLoss(3, 2, t_alpha=0.0).to(TEST_DEVICE, torch.float64).eval()
        weight = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        with torch.no_grad():
            loss_func.weight.copy_(weight)
        embeddings = torch.tensor(
            [[1e-4, 0.0], [1e3, 0.0], [5.0, 0.0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        labels = torch.tensor([0, 1, 1], device=TEST_DEVICE)
        losses = loss_func.loss_terms(embeddings, labels)["loss"]["losses"]
        self.assertTrue(torch.isfinite(losses).all())
        single_label = torch.zeros(1, dtype=torch.long, device=TEST_DEVICE)
        for first, second in ((5e-4, 1e-3), (1e3, 100.0)):
            left = loss_func(
                torch.tensor([[first, 0.0]], dtype=torch.float64, device=TEST_DEVICE),
                single_label,
            )
            right = loss_func(
                torch.tensor([[second, 0.0]], dtype=torch.float64, device=TEST_DEVICE),
                single_label,
            )
            with self.subTest(norms=(first, second)):
                self.assertAlmostEqual(left.item(), right.item(), places=10)

    def test_get_logits_returns_scale_times_the_cosine(self):
        for dtype in TEST_DTYPES:
            loss_func = AdaFaceLoss(3, 4).to(TEST_DEVICE, dtype)
            embeddings = torch.randn(5, 4, device=TEST_DEVICE, dtype=dtype)
            cosine = F.normalize(embeddings.double(), dim=1) @ F.normalize(
                loss_func.weight.detach().double(), dim=1
            ).T
            helpers.assert_close(
                loss_func.get_logits(embeddings), cosine * SCALE, dtype
            )

    def test_empty_indices_tuple_gives_a_differentiable_zero(self):
        loss_func = AdaFaceLoss(4, 8, t_alpha=0.0).to(TEST_DEVICE, torch.float64)
        empty = torch.empty(0, dtype=torch.long, device=TEST_DEVICE)
        embeddings = torch.randn(
            6, 8, dtype=torch.float64, device=TEST_DEVICE, requires_grad=True
        )
        labels = torch.tensor(LABELS, device=TEST_DEVICE)
        loss = loss_func(embeddings, labels, (empty, empty, empty, empty))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_labels_and_reference_sets_are_rejected(self):
        loss_func = AdaFaceLoss(4, 8, t_alpha=0.0).to(TEST_DEVICE, torch.float64)
        embeddings = torch.randn(4, 8, dtype=torch.float64, device=TEST_DEVICE)
        labels = torch.tensor([0, 1, 2, 3], device=TEST_DEVICE)
        with self.assertRaises(ValueError):
            loss_func(embeddings)
        with self.assertRaises(ValueError):
            loss_func(embeddings, labels, ref_emb=embeddings, ref_labels=labels)

    def test_other_distances_are_rejected(self):
        with self.assertRaises(TypeError):
            AdaFaceLoss(3, 4, distance=LpDistance())

    def test_weight_options_are_forwarded(self):
        loss_func = AdaFaceLoss(3, 4, weight_init_func=torch.nn.init.zeros_)
        self.assertTrue(torch.equal(loss_func.weight, torch.zeros(3, 4)))

    def test_unplaced_module_raises(self):
        loss_func = AdaFaceLoss(4, 8).to(TEST_DEVICE, torch.float32)
        embeddings = torch.randn(4, 8, device=TEST_DEVICE, dtype=torch.float64)
        with self.assertRaises(ValueError):
            loss_func(embeddings, torch.arange(4, device=TEST_DEVICE))
