---
title: "Note: nano-vllm 学习笔记"
date: 2025-09-09 17:36:53
updated: 2025-09-20 0:01:14
home_cover: https://p.sda1.dev/26/0186e079ea9478e12f463e4d80a9d5c3/cover.jpg
post_cover: https://p.sda1.dev/26/ac89d2ec92626ebc7dd79366d9b9da98/post.JPG
copyright_info: true
tags:
    - MLSys
categories:
    - Notes
mathjax: true
excerpt: 第一次学习 MLSys, 谨做记录和总结.
---

第一次学习 MLSys, 谨做记录和总结.

### Tensor Parallelism

考虑一个矩阵乘法: $Y=W\times X$, 其中 $W$ 是 weight, $X$ 是输入.

第一种是行分割:

$$
\begin{bmatrix}W_1&W_2\end{bmatrix}\times\begin{bmatrix}X_1\\\\X_2\end{bmatrix}=\begin{bmatrix}W_1\times X_1+W_2\times X_2\end{bmatrix}=Y.
$$

最后要在所有卡上做一个 all_reduce 的操作对结果做汇总.

第二种是列分割:

$$
\begin{bmatrix}W_1\\\\W_2\end{bmatrix}\times X=\begin{bmatrix}W_1\times X\\\\W_2\times X\end{bmatrix}=Y.
$$

最后要在所有卡上做一个 all_gather.

在训练阶段, 无论是何种方式切分, 都会在 forward 和 backward 各产生一次额外通信. 如果有连续的矩阵乘法, 通过先列分割再行分割可以把一对 all_gather/split 抵消.

<details>
  <summary>model weight loader 的实现:</summary>

```python
> nanovllm/engine/model_runner.py:l 32 --> ModelRunner.__init__
load_model(self.model, config.model)

> nanovllm/utils/loader.py:def load_model
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

> nanovllm/models/qwen3.py:l 185 --> Qwen3ForCausalLM
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}

> nanovllm/models/qwen3.py:l 41 --> Qwen3Attention.__init__
self.qkv_proj = QKVParallelLinear(
    hidden_size,
    self.head_dim,
    self.total_num_heads,
    self.total_num_kv_heads,
    bias=qkv_bias,
)

> nanovllm/models/qwen3.py:l 97 --> Qwen3MLP.__init__
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size] * 2,
    bias=False,
)

> nanovllm/layers/linear.py:def QKVParallelLinear.weight_loader
def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
    param_data = param.data
    assert loaded_shard_id in ["q", "k", "v"]
    if loaded_shard_id == "q":
        shard_size = self.num_heads * self.head_size
        shard_offset = 0
    elif loaded_shard_id == "k":
        shard_size = self.num_kv_heads * self.head_size
        shard_offset = self.num_heads * self.head_size
    else:
        shard_size = self.num_kv_heads * self.head_size
        shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
    param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
    loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
    param_data.copy_(loaded_weight)

> nanovllm/layers/linear.py:l 23 --> LinearBase.__init__
self.tp_dim = tp_dim
self.tp_rank = dist.get_rank()
self.tp_size = dist.get_world_size()
```

</details>

<details>
  <summary>forward 的实现:</summary>

Embedding:

```python
> nanovllm/layers/embed_head.py:l 35 --> VocabParallelEmbedding.forward
def forward(self, x: torch.Tensor):
    if self.tp_size > 1:
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        x = mask * (x - self.vocab_start_idx)
    y = F.embedding(x, self.weight)
    if self.tp_size > 1:
        y = mask.unsqueeze(1) * y
        dist.all_reduce(y)
    return y
```

用 y = mask.unsqueeze(1) * y 把不属于本卡的归零,用 dist.all_reduce(y) 汇总.

在 Attention 之前, 每个卡都有完整的 embedding. 每个 Attention 都采用 Pre-Norm 归一化. 每个 Attention 在 num_heads 维度上做列切分, 进一步提高并行化程度. 叠加 QKNorm, 每一层都添加 RoPE. 最后的 attn_o MLP 是单层线性层, 采用行切分和 dist.all_reduce().

```python
> nanovllm/layers/linear.py:l 120 --> QKVParallelLinear.__init__
self.num_heads = divide(self.total_num_heads, tp_size)
self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)

> nanovllm/layers/linear.py:def RowParallelLinear.forward
def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
    if self.tp_size > 1:
        dist.all_reduce(y)
    return y
```

Attention 后的 MLP 是两层线性层, 激活函数是 SiLU, 元素之间独立, 线性层采用先列切分再行切分的方法减少一次通信.

</details>

### 关于 Scheduler

prefill 会让 kv_cache_block_manager try_allocate 而 decode 会让 kv_cache_block_manager try_append.

如果 try_append 失败, 则会把已经保存的 kv_cache_block deallocate.

### 关于 Transformer

<details>
  <summary>模型的初始化与标准化</summary>

用二阶矩来衡量输出的稳定性, 对于一个单层的无激活函数的全连接线性网络层来说 (假设输入 channel 数为 $m$ , 输出 channel 数为 $n$ ), 简单起见, 我们用零初始化 bias, 并且将 $w_{ij}$ 的均值也设为 $0$. 我们计算二阶矩:

$$
\mathbb{E}[y_j^2]=\mathbb{E}[(\sum_{i=1}w_{ij}x_i)^2]=\sum_{i_1,i_2}\mathbb{E}[w_{i_1,j}w_{i_2,j}]\mathbb{E}[x_{i_1}x_{i_2}]=\sum_{i}\mathbb{E}[x_i^2]\mathbb{E}[w_{i,j}^2]=m\mathbb{E}[w_{i,j}^2].
$$

所以为了使 $\mathbb{E}[y_j^2]$ 为 $1$, 那么 $\mathbb{E}[w_{i,j}^2]=\displaystyle\frac{1}{m}$, 这就是 LeCun 初始化.

如果考虑激活函数, 比如采用 relu, 那么可以假设有大概一般的输出 $y_j$ 被归零了, 从而初始化的方差为 $\displaystyle\frac{2}{m}$, 这就是专门针对 relu 网络的 He 初始化.

对于其他的激活函数, 有可能无论如何修改初始化都无法控制二阶矩, 这时需要"微调"激活函数.

以 sigmoid 为例, 假设我们依然以均值为 $0$, 方差为 $\displaystyle\frac{1}{m}$ 来初始化, 那么激活前的输出也是均值为 $0$, 方差为 $1$, 用标准正态分布估计 sigmoid 后的二阶矩:

$$
\int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}\frac{1}{(1+e^{-x})^2}dx\approx0.293379.
$$

``` mathematica
NIntegrate[1/Sqrt[2*Pi]*Exp[-x^2/2]*1/(1+Exp[-x])^2, {x, -Infinity, Infinity}]

0.293379
```

所以, 如果我们希望保持输出的二阶矩不变, 那么可以把输出结果除以 $\sqrt{0.293379}$.

2017 这篇论文 [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515) 提出了 SELU 激活函数, 其定义为

$$
\text{SELU}(x)=\lambda\begin{cases}x&x>0\\\alpha e^x-\alpha&x\le0\end{cases},
$$

其中 $\lambda \approx 1.0507, \alpha \approx 1.6733$. 论文中给出的 $\lambda, \alpha$ 的值可以使得标准正态分布经过 SELU 激活函数后, 均值和方差都不变. 只能算得上一种好的初始化方法.

``` mathematica
F[x_] = Exp[-x^2/2]/Sqrt[2*Pi];
Selu[x_] = Piecewise[{{\[Lambda]*x, x > 0}, {\[Lambda]*\[Alpha]*(Exp[x] - 1), x <= 0}}];
x1 = Integrate[F[x]*Selu[x], {x, -Infinity, Infinity}];
x2 = Integrate[F[x]*Selu[x]^2, {x, -Infinity, Infinity}];
N[Solve[{x1 == 0, x2 == 1}, {\[Lambda], \[Alpha]}], 20]

{{\[Lambda] -> -1.0507009873554804934, \[Alpha] -> 1.6732632423543772848}}
```

当然相比于这种"微调", 更直接的是各种 Normalization 方法, 通过直接计算当前数据的均值和方差来归一化, 而非预先估计积分. 虽然 Normalization 都包含 centering 和 scaling 两个步骤, 但越来越多的工作逐渐尝试去掉 centering 这一步, 甚至有些工作表明去掉 centering 反而能提升模型的性能.

比如 <a href="https://arxiv.org/abs/1910.07467">Root Mean Square Layer Normalization</a> 提出的 RMSNorm, 就表明相比 LayerNorm 更快且保持基本一致的效果.

类似地, 同样是 2019 年的文章, <a href="https://arxiv.org/abs/1912.04958">Analyzing and Improving the Image Quality of StyleGAN</a> 发现使用了 InstanceNorm 后图片会带有"水滴", 而保留 InstanceNorm 单去掉 centering 能改善这个现象, 这也为 centering 有可能带来负面影响提供了佐证.

关于残差连接 $x+F(x)$, 假设 $x$ 与 $F(x)$ 两者独立, 那么 $x+F(x)$ 的方差为 $\sigma_1^2+\sigma_2^2$, 会进一步放大方差, 一种朴素的方法是直接在残差相加之后加入 Normalization 操作:

$$
x_{t+1}=Norm(x_t+F(x_t)).
$$

这种 `Post Norm` 的结构, 是原版 Transformer 和 BERT 所采用的, 这种虽然稳定了正向传播的方差, 但是会削弱残差连接中的恒定项, 所以反而失去了残差易于训练的优点, 通常要 Warmup 并设置足够小的学习率才能收敛.

一个针对性的改进是 `Pre Norm`, 其形式为:

$$
x_{t+1}=x_t+F(Norm(x_t)).
$$

迭代展开后有:

$$
x_{t+1}=x_0+F_0(x_0)+F_1(x_1/\sqrt{2})+\dots+F_t(x_t/\sqrt{t+1}).
$$

至少每一个残差项都是平权的, 作用会相比 `Post Norm` 更大, 所以也更容易优化. 当然, 这样的输出方差会很大, 在预测层之前需要加一个 Normalization.

为什么 `Pre Norm` 的效果会不如 `Post Norm`? 回顾我们的迭代展开式, 每一项都是同一量级的, 因为有 $x_{t+1}=O(t+1)$, 当深度很深的时候, $x_{t+1}$ 与 $x_t$ 的相对差别是比较小的, 因此:

$$
F_{t-1}(Norm(x_{t-1}))+F_t(Norm(x_t))\approx F_{t-1}(Norm(x_{t-1}))+F_t(Norm(x_{t-1})).
$$

因此原本一个 $t-1$ 层模型的输出和 $t$ 输出的结果相加, 近似于一个更宽的 $t$ 层模型, 所以在 `Pre Norm` 中多层叠加的结果在更深的模型中是增加宽度而非增加深度, 层数越多, 层数越"虚".

</details>

### Prepare

```python
> nanovllm/engine/scheduler.py:l 34 --> Scheduler.schedule
self.block_manager.allocate(seq)

> nanovllm/engine/block_manager.py:def BlockManager.allocate
def allocate(self, seq: Sequence):
    assert not seq.block_table
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```
在这里实现了 prefix caching, 每一次处理请求, 都会逐块计算哈希, 判断是否在缓存块中.

```python
> nanovllm/engine/model_runner.py:l 209 --> ModelRunner.run
input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
```

如果是 prefill, model 需要做 varlen 的 attention, 首先将所有未处理的 input_ids 拼接在一起, 作为输入 q 的总长度, 每一个序列的总长度相加是 kv 的长度, positions 是相对每一个序列的开始到结束的区间. slot_mapping 记录了每一个新进入的 q 对应的 kv cache 在 pool 里的位置, 通过 block_id * block_size 得到开始位置.






