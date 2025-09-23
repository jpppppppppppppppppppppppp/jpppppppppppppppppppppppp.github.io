---
title: "Note: Rectified Flow"
date: 2025-09-22 23:26:05
updated: 2025-09-23 0:27:43
home_cover: https://p.sda1.dev/27/be11eba4b34fc4fdf7394442f90dbfb5/cover.jpg
post_cover: https://p.sda1.dev/27/fbc8d67219a2abc03a75ff380f6d4fc1/post.PNG
copyright_info: true
tags:
    - Flow
categories:
    - Notes
mathjax: true
excerpt: 学习刘强老师的 "Let us Flow Together" 笔记.
---

原文链接: <a href="https://www.cs.utexas.edu/~lqiang/PDF/flow_book.pdf">Let us Flow Together</a>.

主要内容围绕 Rectified Flow 展开, 我谨记录一些关键内容, 以及尽可能地补充一些旁支末节.

### 1.1 Overview

对 Flow-based 建模做符号约定:

$$
\dot{Z}_t=v_t(Z_t),\quad \forall t\in[0,1],\quad Z_0\sim\pi_0.
\tag{1.1}
$$

在推理阶段需要在精度和速度之间 trade-off, 一个理想的场景是沿直线传播, 在这种情况随机过程可以由输入和输出直接线性插值出来, $X_t=tX_1+(1-t)X_0$, 所以这种过程不是一个 casual 的过程. 所以要转换成一个 casual 的过程的同时保持边际分布.

$$
\min_v \int_0^1\mathbb{E}\left[\left\Vert\dot{X}_t-v_t(X_t)\right\Vert^2\right]dt.
\tag{1.2}
$$

最优解为:

$$
v^*_t(x)=\mathbb{E}\left[\dot{X}_t|X_t=x\right].
$$

在直线插值的交叉点, 速度场由各条插值的期望组成, 由于速度场不会发生交叉, 所以最终得到的 $v$ 会重写部分轨迹来避免交叉.

对于一个随机过程 $\\{X_t:t\in[0,1]\\}$, 把由它的边际分布导出的 Rectified Flow 记为 $\\{Z_t\\}=\mathtt{Rectify}(\\{X_t\\})$. 满足 $\dot{Z}_t=v^*_t(Z_t)$, 其中如上文所述 $v^\*_t(z)=\mathbb{E}\left[\dot{X}_t|X_t=z\right]$, 并且符合初始条件 $Z_0=X_0$.

Rectified Flow 有两个性质:
1. 边际分布不变: $Z_t\stackrel{d}{=}X_t,\forall t\in[0,1]$.
2. 运输成本最低: 对任意凸价值函数 $c$ 成立 $\mathbb{E}[c(Z_1-Z_0)]\leq\mathbb{E}[c(X_1-X_0)]$.


