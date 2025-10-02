---
title: 'PaperReading: Sparse VideoGen2'
date: 2025-10-01 20:57:16
updated: 2025-10-01 20:57:16
home_cover: https://p.sda1.dev/27/be11eba4b34fc4fdf7394442f90dbfb5/cover.jpg
post_cover: https://p.sda1.dev/27/fbc8d67219a2abc03a75ff380f6d4fc1/post.PNG
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



