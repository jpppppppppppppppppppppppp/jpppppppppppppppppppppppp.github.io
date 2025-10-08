---
title: 'PaperReading: Sparse VideoGen2'
date: 2025-10-01 20:57:16
updated: 2025-10-01 20:57:16
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



