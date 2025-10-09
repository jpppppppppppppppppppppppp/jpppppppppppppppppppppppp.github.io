---
title: 'PaperReading: Sparse VideoGen2'
date: 2025-10-01 20:57:16
updated: 2025-10-08 22:54:06
home_cover: https://p.sda1.dev/27/869d472a6fed782d7cd472c73fadd420/cover.jpeg
post_cover: https://p.sda1.dev/27/00617cf5060c0a25763cef4a0cb40e27/post.jpg
copyright_info: true
tags:
  - Video Generation
  - Diffusion
  - Efficient Inference
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



