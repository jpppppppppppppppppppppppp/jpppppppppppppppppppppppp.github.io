---
title: "Note: triton 学习笔记"
date: 2025/09/16 17:39:53
updated: 2025/09/16 23:05:33
home_cover: https://p.sda1.dev/27/9fa85a2c8d0fdfae2fc5082170299317/cover.jpg
post_cover: https://p.sda1.dev/27/946a46691e4fea8fca7063f89d75fbe8/post.png
copyright_info: true
tags:
    - MLSys
categories:
    - Notes
mathjax: true
excerpt: Openai-triton tutorial 学习笔记.
---

第一次学习 Triton 是在今年八月份, 看到了 <a href="Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention">NSA</a>, 才去学习了 triton 的写法, 虽然自己实现了一个包括正向和反向的 kernel, 但是还是有很多问题, 比如数值上和 PyTorch 对拍结果不一致, 以及性能上还没有十分满意. 所以借此机会, 依靠官方的材料, 重新学习一遍.


