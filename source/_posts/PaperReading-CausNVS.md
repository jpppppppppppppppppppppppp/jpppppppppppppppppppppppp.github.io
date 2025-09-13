---
title: 'PaperReading: CausNVS'
date: 2025-09-11 15:17:23
updated: 2025-09-11 18:41:40
home_cover: https://p.sda1.dev/26/c916c973e873eb2986d59e381be3f460/cover.jpeg
post_cover: https://p.sda1.dev/26/f0b4dd4c2dbff6c30a3d9e466daf54ec/post.PNG
copyright_info: true
tags:
    - 2D Generation
    - AutoRegressive
    - Diffusion
    - Multi-view
categories:
    - PaperReading
mathjax: true
excerpt: "[2025.09.08] CausNVS: Autoregressive Multi-view Diffusion for Flexible 3D Novel View Synthesis."
---

Link: <a href="https://arxiv.org/abs/2509.06579">CausNVS: Autoregressive Multi-view Diffusion for Flexible 3D Novel View Synthesis</a>.

很新的文章, 趁着现在没卡训练, 有空来学习一下 autoregressive 的 diffusion model. 第一次知道 Multi-view diffusion model 是在祖师爷 Barron 的论文 <a href="https://arxiv.org/abs/2405.10314">CAT3D: Create Anything in 3D with Multi-View Diffusion Models</a> 中, 趁着这个机会, 了解一下最新的进步.

过去的固定长度的 diffusion paradigm 没有良好的因果关系, 不能适应 on-the-fly 的场景, 使用自回归有以下几个困难:

<details>
  <summary>Recently Generation Works:</summary>

首先是 Multi-view Diffusion Model 的相关工作:

最经典的 Multi-view Diffusion Model 是固定长度的, 通常包括若干个已知的视角及图片输入和预定义的输出视角, 整体一起去噪来增强不同视角的 consistency.

<a href="https://arxiv.org/abs/2411.04928">[ICCV2025] DimensionX: Create Any 3D and 4D Scenes from a Single Image with Decoupled Video Diffusion</a> 训练 spatial 和 temporal diffusion model, 来生成 3D 场景或 4D 场景. <a href="https://arxiv.org/abs/2409.02048">ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis</a> 则是利用 Dust3R 获取 PointCloud 来迭代式地生成.

<a href="https://arxiv.org/abs/2503.14489">Stable Virtual Camera: Generative View Synthesis with Diffusion Models</a> 采用 Two-passing Stage 的方法, 先生成 Anchor Frames, 再生成中间帧.

下面是 Autoregressive Diffusion Model 的相关工作:

<a href="http://arxiv.org/abs/2412.07772">From Slow Bidirectional to Fast Autoregressive Video Diffusion Models</a> 将双向注意力的视频生成模型蒸馏成单向的自回归模型.

</details>

---

第一个要解决的问题是 Autoregressive drift: 因为在训练中已知的视角都是 ground truth, 而在推理时使用过去的输出, 产生的误差会逐步积累, 逐渐影响生成的质量.

所以, 在训练时, 一个 batch 内的 frames 可以加上不同程度的噪声, 这样鼓励模型从不确定的 context 里学习, 在推理时也可以在已有的输入里加上少量的噪声, 也可以在一定程度上减少对整体相机轨迹的依赖 (文章中举例: 比如在 <a href="https://arxiv.org/abs/2407.07860">[ICLR2025] 4DiM: Controlling Space and Time with Diffusion Models</a> 中,  生成的道路倾向于和相机轨迹平行, 笔者没有验证).

第二个可以解决的问题是 KV-Cache 的兼容性.

使用自回归架构之后, 就可以使用 KV-Cache 来加速, 但是要先解决一些问题. 关于相机的 encoding, 可以使用 intrinsic-extrinsic matrix 或者 Pl&uuml;cker rays 来表示, 第一个问题是这会引入额外的参数去学习, 比如把这些 representation 映射到 condition 空间. 第二个问题是这些表示都依赖一个世界坐标系, 如果没有世界坐标系, 外参就无从谈起, 所以当生成的轨迹变得很长的时候, 就有可能出现 out-of-distribution 的问题. 而一旦我们移动了世界坐标系, 所有的 KV-Cache 都失效, 需要重新计算. 文章采用的方法是用相机参数来实现相对位置编码 CaPE (<a href="https://arxiv.org/abs/2402.03908">[CVPR2024] Eschernet: A generative model for scalable view synthesis.</a>), 这样就不需要一个全局的世界坐标系, 只需要相对位置, 这样就能保证 KV-Cache 的有效性. 但是 CaPE 好像只用了相机的外参, 没有用内参, 估计要等到开源了才能确定, 具体的做法是用相机 Pose 矩阵 $P\in\mathbb{R}^{4\times4}$ 来作为 RoPE 中的对角阵元素.

$$
\pi(v, P)=\phi(P)v,\quad \phi(P)=I_{d/4}\otimes\Psi,\quad \Psi=\begin{cases}P&\text{if key,}\\\\ P^{-\mathsf{T}}&\text{if query.}\end{cases}
$$

另一个有趣的设计是把 KV-Cache 作为 Spatial Memory, 具体做法是生成新的图片时, 仅使用距离对应视角最近的几个视角的 KV-Cache. 这样就不需要额外的空间记忆模块.

