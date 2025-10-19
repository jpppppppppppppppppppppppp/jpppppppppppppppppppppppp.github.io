---
title: "Note: triton 学习笔记"
date: 2025-09-16 17:39:53
updated: 2025-09-17 21:02:04
home_cover: https://p.sda1.dev/27/9fa85a2c8d0fdfae2fc5082170299317/cover.jpg
post_cover: https://p.sda1.dev/27/946a46691e4fea8fca7063f89d75fbe8/post.png
copyright_info: true
tags:
    - Machine Learning Systems
categories:
    - Notes
mathjax: true
excerpt: Openai-triton tutorial 学习笔记.
---

第一次学习 Triton 是在今年八月份, 看到了 <a href="Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention">NSA</a>, 才去学习了 triton 的写法, 虽然自己实现了一个包括正向和反向的 kernel, 但是还是有很多问题, 比如数值上和 PyTorch 对拍结果不一致, 以及性能上还没有十分满意. 所以借此机会, 依靠官方的材料, 重新学习一遍.

Grouped ordering matmul: 一种 L2 Cache Optimizations, 在计算矩阵乘法的时候, 并非简单的遍历边, 而是以一种 Zigzag 的方式遍历, 这样可以在计算相同数量的结果的情况下, 减少访存的数量, 从而提高缓存的命中率. 比如计算乘法 $C=A @ B$, 其中 $A\in\mathbb{R}^{m\times k}, B\in\mathbb{R}^{k\times n}, C\in\mathbb{R}^{m\times n}$, 这里的形状都以分块后考虑. 如果采用行列遍历的方式, 我们计算 $C$ 的第一行, 需要访问 $A$ 的第一行和完整的矩阵 $B$, 访存数量是 $k+kn$; 而如果采用 grouped 的方式, 计算 $C$ 的 $\sqrt{n}\times\sqrt{n}$ 大小的 group, 访存需求是 $2\sqrt{n}k$, 远小于 $k(1+n)$. 更容易命中缓存, 会有 ~10% 的提升.

LayerNorm: 为了方便, 我们省去 $\varepsilon$ 的书写, $Y_{i,j}=\displaystyle\frac{X_{i,j}-\mu_i}{\sigma_i}*w_j+b_j$. 正向传播比较容易, 在 row 上做任务的分发. 下面主要记录一下反向传播过程的推导. 我们把归一化后的 $X_{i,j}$ 记为 $\widehat{X_{i,j}}=\displaystyle\frac{X_{i,j}-\mu_i}{\sigma_i}$.

求 $w$ 和 $b$ 的梯度比较直接:
$$
\frac{\partial L}{\partial w_j} = \sum_{i=1}^T\frac{\partial L}{\partial Y_{i,j}} * \widehat{X_{i,j}}, \quad \frac{\partial L}{\partial b_j} = \sum_{i=1}^T\frac{\partial L}{\partial Y_{i,j}}.
$$

对 $X$ 的梯度计算就比较麻烦, 可以先分别对 $\widehat{X}, \mu, \sigma$ 分别算梯度再求和:
$$
\frac{\partial L}{\partial \sigma_i}=\sum_{j=1}^D\frac{\partial L}{\partial Y_{i,j}}\frac{\partial Y_{i,j}}{\partial \sigma_i}=\sum_{j=1}^D\frac{\partial L}{\partial Y_{i,j}} w_j (X_{i,j}-\mu_i)\frac{-1}{\sigma_i^2} = -\sum_{j=1}^D \frac{\partial L}{\partial Y_{i,j}}\frac{w_j}{\sigma_i}\widehat{X_{i,j}},
$$
$$
\frac{\partial L}{\partial \mu_i}=\sum_{j=1}^D\frac{\partial L}{\partial Y_{i,j}}\frac{\partial Y_{i,j}}{\partial \mu_i}=-\sum_{j=1}^D\frac{\partial L}{\partial Y_{i,j}}\frac{w_j}{\sigma_i},
$$
$$
\begin{aligned}
\frac{\partial L}{\partial X_{i,j}}&=\frac{\partial L}{\partial \widehat{X_{i,j}}}\frac{\partial \widehat{X_{i,j}}}{\partial X_{i,j}}+\frac{\partial L}{\partial \mu_i}\frac{\partial \mu_i}{\partial X_{i,j}}+\frac{\partial L}{\partial \sigma_i}\frac{\partial \sigma_i}{\partial X_{i,j}}\\\\&=\frac{\partial L}{\partial Y_{i,j}}\frac{w_j}{\sigma_i}-\sum_{j=1}^D\frac{\partial L}{\partial Y_{i,j}}\frac{w_j}{\sigma_i}\frac{1}{D}-\sum_{j=1}^D \frac{\partial L}{\partial Y_{i,j}}\frac{w_j}{\sigma_i}\widehat{X_{i,j}}\frac{1}{\sigma_i D}(X_{i,j}-\mu_i)\\\\&=\frac{1}{\sigma_i}\left[\frac{\partial L}{\partial Y_{i,j}}w_j-\frac{1}{D}\sum_{j=1}^D\frac{\partial L}{\partial Y_{i,j}}w_j-\frac{1}{D}\widehat{X_{i,j}}\sum_{j=1}^D\frac{\partial L}{\partial Y_{i,j}}w_j\widehat{X_{i,j}}\right].
\end{aligned}
$$

为了计算 $w$ 和 $b$ 的梯度, 需要在 row 上求和, 可以分两个阶段, 在第一个阶段上分组求和, 然后在第二阶段 all_reduce. 在第一阶段的求和中, 需要用锁保证只有一个线程可以做加和.

```python
@triton.jit
def _layer_norm_bwd_dx_fused(DY,  # pointer to the output gradient
                             DX,  # pointer to the input gradient
                             DW,  # pointer to the partial sum of weights gradient
                             DB,  # pointer to the partial sum of biases gradient
                             X,  # pointer to the input
                             W,  # pointer to the weights
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             Lock,  # pointer to the lock
                             stride,  # how much to increase the pointer when moving by 1 row
                             N,  # number of columns in X
                             GROUP_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr):
    row_pid = tl.program_id(0)
    X += row_pid * stride
    DY += row_pid * stride
    DX += row_pid * stride

    off_col = tl.arange(0, BLOCK_SIZE_COL)
    mask_col = off_col < N

    lock_id = row_pid % GROUP_SIZE_ROW
    Lock += lock_id
    Count = Lock + GROUP_SIZE_ROW # indicates whether it is the first time to accumulate
    DW = DW + lock_id * N + off_col
    DB = DB + lock_id * N + off_col

    x = tl.load(X + off_col, mask=mask_col, other=0.0).to(tl.float32)
    dy = tl.load(DY + off_col, mask=mask_col, other=0.0).to(tl.float32)
    w = tl.load(W + off_col, mask=mask_col).to(tl.float32)
    mean = tl.load(Mean + row_pid)
    rstd = tl.load(Rstd + row_pid)

    x_hat = (x - mean) * rstd
    x_hat = tl.where(mask_col, x_hat, 0.0)
    wdy = w * dy
    wdy = tl.where(mask_col, wdy, 0.0)

    term1 = tl.sum(wdy, axis=0) / N  # sum_j dy * w
    term2 = tl.sum(wdy * x_hat, axis=0) / N  # sum_j dy * w * x_hat
    dx = (wdy - (xhat * term2 + term1)) * rstd
    tl.store(DX + off_col, dx.to(DX.dtype), mask=mask_col)

    group_partial_dw = tl.sum(dy * x_hat, axis=0)
    group_partial_db = tl.sum(dy, axis=0)
    while tl.atomic_cas(Count, 0, 1) == 1:
        pass  # acquire lock
    count = tl.load(Count)
    if Count == 0:
        tl.atomic_xchg(Count, 1)  # indicate that it is not the first time to accumulate
    else:
        group_partial_dw += tl.load(DW, mask=mask_col).to(tl.float32)
        group_partial_db += tl.load(DB, mask=mask_col).to(tl.float32)
    tl.store(DW, group_partial_dw, mask=mask_col)
    tl.store(DB, group_partial_db, mask=mask_col)

    tl.debug_barrier()

    tl.atomic_xchg(Lock, 0)

@triton.jit
def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)
```

官方教程的实现和 <a href="https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py">FlashAttention2</a> 的实现略有不同, 主要区别在反向传播中每个 Block 负责连续的若干行, 非官方教程中不连续的若干行, 相比之下更加简单.

后面的教程大多都是矩阵乘法的各种变体, <a href="https://arxiv.org/abs/2301.03598">Stream-K</a> 的 <a href="https://github.com/triton-lang/triton/issues/1393#issuecomment-1518038216">实现</a> 贴一下, 具体的太过冗长懒得看了.
