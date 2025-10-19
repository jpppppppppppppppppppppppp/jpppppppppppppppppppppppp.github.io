---
title: 'PaperReading: Sparse VideoGen2'
date: 2025-10-01 20:57:16
updated: 2025-10-09 17:51:53
home_cover: https://p.sda1.dev/27/869d472a6fed782d7cd472c73fadd420/cover.jpeg
post_cover: https://p.sda1.dev/27/00617cf5060c0a25763cef4a0cb40e27/post.jpg
copyright_info: true
tags:
  - Machine Learning Systems
categories:
  - PaperReading
mathjax: true
excerpt: "[NIPS2025] Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation."
---

Link: <a href="https://arxiv.org/abs/2505.18875">Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation</a>.

针对 Sparse Attention 提出两个现有的问题: 一是寻找关键 Token 准确率低, 基于位置成块不如基于语义成块; 二是稀疏性对硬件不友好, 不能够连续地计算.

他们的做法是先对 Q 和 K 分别做 K-means 聚类, 根据聚类从而假设 semantic-aware 的成块, 每个块内部用 pool 提出一个做总体性决策, 比用位置 patchify 更合理, 第一是语义更加通顺, 第二是稀疏性结果更加连续. 在 Attention 部分用的是 Top-p. 总之, 很多内容都是我知其然而不知其所以然的, 所以借此机会结合代码学习一下.

从 Wan 模型的架构开始吧. 首先是 Wan-VAE, 一种 Causal 3D-VAE 架构, 其实结构上挺简单的, 为了实现无限长视频的编码, 采用了长度为 3 的类似滑动窗口的编码方式, 从而保证了因果性. 再配合三次降采样, 减少卷积的计算. 文章中提到的训练方式比较有趣, 先在图片上训练, 掌握一定的 2d 降采样能力, 再在视频上训练, 加快 3d 降采样的训练速度. 由于显存限制, 我用的是 1.3B 的模型, 一些参数如下: 最终的 VAE channels=384, 隐空间维数 z_dim=16.

在 Diffusion 阶段, 采用的推理更快的 Rectified Flow. 采样 time_step_num=50, 采用 CFG 的方式, 其中 guidance_scale=5.0.

这篇文章主要加速了 Full Self-Attention 的部分, 我们来先看一下 Diffusers 的官方实现 (已经把不相关的代码删除了):

```python
class WanAttnProcessor2_0:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
```

可以看到整体上和正常的 Self-Attention 一模一样, 过 qkv 矩阵, 然后是 qk_norm, 然后是 RoPE, 最后是直接调用 sdpa. 我们来看看修改后的主逻辑:

```python
def attention_core_logic(self, query, key, value, timestep):
    cfg, num_heads, seq_len, dim = query.size()
    assert cfg == 1, "Batch size must be 1 for kmeans block sparse attention"
    context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size
    q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices = self.semantic_aware_permutation(
        query, key, value
    )
    output_permuted = dynamic_block_sparse_fwd_flashinfer(
        q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, is_cpu=False
    )
    attn_output = apply_inverse_permutation_triton(output_permuted, q_sorted_indices, dim=2)
    return attn_output.reshape(cfg, num_heads, seq_len, dim)
```

先看 kmeans 的聚类部分:

```python
def _euclid_iter(x, x_sq, centroids):
    # cent_sq = (centroids ** 2).sum(dim=-1)
    # cross = torch.einsum('bnd,bkd->bnk', x, centroids)
    # dist_sq = (x_sq[:,:,None] + cent_sq[:,None,:] - 2.0 * cross).clamp_min_(0.0)

    # cluster_ids = dist_sq.argmin(dim=-1)
    cluster_ids = euclid_assign_triton(x, centroids, x_sq)
    centroids_new, cluster_sizes = triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)
    # centroids_new = triton_centroid_update_euclid(x, cluster_ids, centroids)

    # centroids_new = centroids_new.clone()  # avoid CUDA graphs aliasing

    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids, cluster_sizes
```

第一步聚类, 计算距离肯定不是麻烦的事情, 比较关心的是 top-1 的计算过程. 好吧, 直接用的 tl.argmin. 那后面的内容也就很简单了.

第二步更新质心, 这个比较有趣, 因为每个聚类的分布是不连续的, 如果是我的话, 会先计算一个反向的索引, 即每个聚类包括哪些点, 然后在聚类的数量上面做并行切分. 仔细想了一下, 我的方案有两个问题, 第一是重复读数据, 按照聚类数量遍历的时候, 每一个 workload 都会遍历所有的样本, 而实际上每一个样本点只会被归为一类; 第二是中间变量很大, 存储这个 mask 需要 $[B, N, K]$ 的矩阵, 虽然是 bool 类型, 实际上增加了读取的开销.

他们的做法是先对 cluster_ids 排序, 然后在样本数量上做并行切分, 每个 workload 负责若干段同类的样本点, 使用 tl.atomic_add() 避免处理同类样本信息的写冲突.

```python
@triton.jit
def _centroid_update_chunk_kernel(
    x_ptr,                  # [B, N, D] – original order
    sorted_idx_ptr,         # [B, N]    – indices after sort
    sorted_cluster_ptr,     # [B, N]    – cluster ids in sorted order
    sum_ptr,                # [B, K, D]
    count_ptr,              # [B, K]
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,  # how many tokens (points) each program processes
):
    pid_chunk, pid_b = tl.program_id(axis=0), tl.program_id(axis=1)
    chunk_start = pid_chunk * BLOCK_N  # position of the first token handled by this program

    # base pointers for this batch
    idx_batch_base = sorted_idx_ptr + pid_b * N
    cid_batch_base = sorted_cluster_ptr + pid_b * N
    x_batch_base = x_ptr + pid_b * N * D

    # helper aranges
    offs_token = tl.arange(0, BLOCK_N)
    offs_dim = tl.arange(0, D)

    # first token index & validity mask
    token_idx = chunk_start + offs_token
    valid_tok = token_idx < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    # Load first cluster id to initialise the running accumulator
    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id = tl.load(cid_batch_base + last_token_idx)
    all_ids = tl.load(cid_batch_base + token_idx, mask=valid_tok, other=-1)

    all_tokens_idxs = tl.load(idx_batch_base + token_idx, mask=valid_tok, other=-1)  # [BLOCK_N]
    load_off = all_tokens_idxs[:, None] * D + offs_dim[None, :]

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            cluster_feats = tl.load(x_batch_base + load_off, mask=cluster_mask[:, None], other=0.0)  # [BLOCK_N, D]
            cluster_feats = cluster_feats.to(tl.float32)
            sum_feats = tl.sum(cluster_feats, axis=0)
            dest_ptr = sum_ptr + (pid_b * K + cid) * D + offs_dim
            tl.atomic_add(dest_ptr, sum_feats)
            tl.atomic_add(count_ptr + pid_b * K + cid, cluster_size)
```

permutation 也很简单, 接下来我们来看 top-p 的 Sparse Attention Map 的实现. 因为只需要对质心做 Attention, 所以也没有必要做太多的优化, 直接用 matmul 和 weigted_softmax (因为要考虑 k_cluster 中每个聚类的大小), 排序后计算累计概率 cumsum, 根据 top-p 截断, 然后再根据排序的位置把布尔值写回去.

```python
def weighted_softmax(scores, weights):
    input_dtype = scores.dtype
    scores = scores.float()
    weights = weights.float()
    max_score = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - max_score)
    weighted_exp = weights * exp_scores
    softmax_out = weighted_exp / torch.sum(weighted_exp, dim=-1, keepdim=True).clamp(min=1e-12)
    return softmax_out.to(input_dtype)

def identify_dynamic_map(
    query_centroids,
    key_centroids,
    k_cluster_sizes,
    p,
    min_kc_ratio=0,
):
    B, H, qc_num, D = query_centroids.shape
    kc_num = key_centroids.shape[2]
    device = query_centroids.device

    attn_scores = torch.matmul(query_centroids, key_centroids.transpose(-2, -1)) / (D**0.5)
    k_weights = k_cluster_sizes.unsqueeze(-2).float()

    weighted_attn_probs = weighted_softmax(attn_scores, k_weights)
    sorted_probs, sorted_indices = torch.sort(weighted_attn_probs, dim=-1, descending=True)

    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_indices = cumsum_probs > p
    remove_indices[..., 1:] = remove_indices[..., :-1].clone() # shift right to include the first above-threshold
    remove_indices[..., 0] = False

    if min_kc_ratio > 0:
        preserve_length = int(min_kc_ratio * kc_num)
        remove_indices[..., :preserve_length] = False

    sorted_clusters_to_keep = ~remove_indices

    dynamic_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
    dynamic_map.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    return dynamic_map
```

去噪后下一次 K-means 聚类直接用上一次迭代的结果作为初始化可以加快收敛.

后面的 Block Sparse Attention 是直接依赖 FlashInfer 的实现. 另开专题学习.
