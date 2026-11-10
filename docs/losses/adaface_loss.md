# AdaFaceLoss

[AdaFace: A Quality Adaptive Margin Loss for Face Recognition](https://arxiv.org/abs/2204.00964)

```python {.signature}
losses.AdaFaceLoss(num_classes, embedding_size, margin=0.4, scale=64, h=0.333, t_alpha=0.01,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
```

**Equation**:

$$
t_i = \operatorname{clip}\left(\frac{h \left(\lVert x_i \rVert_2 - \mu\right)}{\varepsilon + \sigma}, -1, 1\right), \qquad
L = \frac{1}{N} \sum_{i} -\log \frac{e^{s \left(\cos\phi_{y_i,i} - m (1 + t_i)\right)}}{e^{s \left(\cos\phi_{y_i,i} - m (1 + t_i)\right)} + \sum_{j \neq y_i} e^{s \cos\left(\theta_{j,i}\right)}}
$$

Here $N$ is the batch size, $\theta_{j,i}$ is the angle between embedding $i$
and the weight vector of class $j$, and
$\phi_{y_i,i} = \operatorname{clip}(\theta_{y_i,i} - m t_i, \varepsilon,
\pi - \varepsilon)$ combines an angular margin $-m t_i$ with an additive margin
$m (1 + t_i)$ on the target class. A larger embedding norm, which the paper
treats as higher sample quality, increases $t_i$ and both parts of the margin.

**Implementation notes**: The embeddings must reach the loss un-normalized: the
norm $\lVert x_i \rVert_2$ is the quality signal, so normalizing embeddings
elsewhere in the pipeline removes it. $\mu$ and $\sigma$ are exponential moving
averages of the batch mean and standard deviation of the clamped norms, held in
the `batch_mean` and `batch_std` buffers and initialized to 20 and 100. A
training forward updates them in place, like the running statistics of batch
normalization; an eval forward leaves them unchanged, and a training batch of
one embedding updates only `batch_mean`, because a single norm has no spread.
The buffers are saved in `state_dict`, and `.to()` moves and casts them with
the module. The norms are detached, so gradients flow only through the cosine.
Norms are clamped to $[0.001, 100]$, target cosines to
$[-1 + 10^{-3}, 1 - 10^{-3}]$ before `acos`, and angles to
$[10^{-3}, \pi - 10^{-3}]$. With an `indices_tuple`, each embedding's loss is
multiplied by its [miner weight](../losses.md#miner-weights).
`get_logits(embeddings)` returns $s \cos\left(\theta_{j,i}\right)$ without the
margin; see [classification logits](../losses.md#classification-logits). With
`h=0`, or with constant norms, $t_i = 0$ and the loss equals
[CosFaceLoss](cosface_loss.md) with the same `margin` and `scale`. This loss
derives from [BaseProxyLoss](base_proxy_loss.md).

**Paper settings**: The defaults are the paper's margin 0.4, scale 64, and
$h = 0.333$. `t_alpha=0.01` follows the paper's training setting, where the
statistics settle slowly across batches; the reference implementation's
constructor default is 1.0, which uses each batch's own statistics. Pass
`t_alpha=1.0` to match it.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.AdaFaceLoss(num_classes=10, embedding_size=128)
optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 10, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
optimizer.step()
```

**Parameters**:

- **num_classes**: The number of classes in the training set. Labels must be in
  `0..num_classes - 1`.
- **embedding_size**: The size of the embeddings passed to the loss.
- **margin**: The margin $m$ shared by the angular and additive parts. The paper
  uses 0.4.
- **scale**: The scale $s$. The paper uses 64.
- **h**: The adaptivity strength $h$. The paper uses 0.333.
- **t_alpha**: The momentum of the batch statistics: each training forward
  moves `batch_mean` and `batch_std` a fraction `t_alpha` toward the batch's
  clamped-norm mean and standard deviation. The paper uses 0.01.
- **weight_regularizer**, **weight_reg_weight**, **weight_init_func**: See
  [learnable weight arguments](../losses.md#learnable-weight-arguments). The
  class weights are `weight`, shaped `(num_classes, embedding_size)` and
  initialized with `torch.nn.init.normal_`. Include `loss_func.parameters()` in
  an optimizer.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). This is the only
  compatible distance; another distance class raises `TypeError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per embedding, indexed by embedding position.
