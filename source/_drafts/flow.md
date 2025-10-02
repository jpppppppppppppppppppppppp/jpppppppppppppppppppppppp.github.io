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
draft: true
---

原文链接: <a href="https://www.cs.utexas.edu/~lqiang/PDF/flow_book.pdf">Let us Flow Together</a>.

主要内容围绕 Rectified Flow 展开, 我谨记录一些关键内容, 以及尽可能地补充一些旁支末节.

### 1. Rectified Flow

这一章节主要介绍了 Rectified Flow 的基本概念, 以及它的性质.

#### 1.1 Overview

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

#### 1.2 Reflow

虽然 Rectified Flow 鼓励直线传播, 但是由于交叉和重写的存在, 并不总是直线, 但是我们可以观察到, 用 Rectified Flow 产生的路径做插值, 产生的交叉数量比原始路径要少. 所以, 在新插值上训练新的 Rectified Flow 可以产生更直的轨迹, 从而有更快的推理速度. 形式化表达如下:

$$
(Z_0^0, Z_1^0) = (X_0, X_1),\quad
\\{Z_t^{k+1}\\}=\mathtt{Rectify}(\mathtt{Interp}(Z_0^k, Z_1^k)).
\tag{1.3}
$$

用 Loss 来量化轨迹的直线程度:
$$
S(\\{Z_t\\}) = \int_0^1\mathbb{E}[\left\Vert Z_1-Z_0-\dot{Z}_t \right\Vert^2]dt.
$$

用 Reflow 进行迭代, 可以作为一种普适的直线化方法, 并有如下保证:

$$
\underset{k\sim U\\{1,\dots,K\\}}{\mathbb{E}}[S(\\{Z_t^k\\})]=\mathcal{O}(1/k).
\tag{1.4}
$$

#### 1.3 Loss Function

不同的 Loss Function 可以被视为不同的 Time-weighted Loss. 例如直接根据 $X_t$ 学习噪声输入 $X_0$ 或者直接学习目标 $X_1$.

$$
\hat{x}\_{0|t}(x)=\mathbb{E}[X_0|X_t=x],\quad\hat{x}\_{1|t}(x)=\mathbb{E}[X_1|X_t=x].
\tag{1.5}
$$

只需要简单的变换, $\hat{x}\_{1|t}(x)=x+(1-t)v_t(x)$, $\hat{x}\_{0|t}(x)=x-tv_t(x)$,

$$
\int_0^1\mathbb{E}\left[\left\Vert X_1 - \hat{x}\_{1|t}(x) \right\Vert^2\right]dt=\int_0^1(1-t)^2\mathbb{E}\left[\left\Vert X_1 - X_0 - v_t(X_t) \right\Vert^2\right]dt.
$$

### 2. Marginals and Errors

这一章基本围绕边际分布和误差估计展开.

{% note success %}
对于一个连续可微的随机过程 $\\{X_t:t\in[0,1]\\}$, 它的期望速度, 或者说 Rectified Flow 速度场 $v^X$, 定义如下:
$$
v_t^X(x)=\mathbb{E}[\dot{X}_t|X_t=x],\quad \forall x\in supp(X_t).
$$
{% endnote %}

{% note info %}
Theorem 1:
$\\{Z_t\\}$ 是从 $\\{X_t\\}$ 导出的 Rectified Flow, 那么 $Z_t\stackrel{d}{=}X_t$ for $\forall t\in[0,1]$.
{% endnote %}

