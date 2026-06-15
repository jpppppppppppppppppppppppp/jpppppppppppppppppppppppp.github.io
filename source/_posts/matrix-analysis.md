---
title: "Matrix Analysis"
date: 2026-04-24 23:22:22
updated: 2026-04-26 23:52:05
home_cover: https://p.sda1.dev/32/4b7f08b1be244fdbffa12865bec5f918/cover.jpg
post_cover: https://p.sda1.dev/32/fc5f5ce56aa068a6fecf50b560c26bf9/post.jpg
copyright_info: true
tags:
    - Math
categories:
    - Notes
mathjax: true
tikzjax: true
excerpt: 张跃辉老师的《矩阵理论教程》一书的笔记.
---

### 第一章: 线性代数概要

(Sylvester\'s inequality) $A\in\mathbb{R}^{m\times p}$, $B\in\mathbb{R}^{p\times n}$, 则有
$$
r(A)+r(B)-p\leq r(AB)\leq \min(r(A), r(B)).
$$

用下面矩阵等式证明左侧不等式:
$$
\begin{pmatrix}I_m & A \\\\0 & I_p\end{pmatrix}
\begin{pmatrix}AB & 0 \\\\0 & I_p\end{pmatrix}
\begin{pmatrix}I_n & 0 \\\\-B & I_p\end{pmatrix}=
\begin{pmatrix}0 & A \\\\-B & I_p\end{pmatrix}.
$$

矩阵的张量积: 设 $A=(a_{ij})\_{m\times n}$, $B=(b_{st})\_{p\times q}$, 则 $A\otimes B\in\mathbb{C}^{mp\times nq}$ 定义为:
$$
A\otimes B = \begin{pmatrix}
a_{11}B & a_{12}B & \cdots & a_{1n}B \\\\
a_{21}B & a_{22}B & \cdots & a_{2n}B \\\\
\vdots  & \vdots  & \ddots & \vdots  \\\\
a_{m1}B & a_{m2}B & \cdots & a_{mn}B
\end{pmatrix}.
$$

下面列举一部分张量积的基本性质:

1. 结合律: $(A\otimes B)\otimes C = A\otimes (B\otimes C)$.
2. 分配率: $(A\otimes B)(C\otimes D) = (AC)\otimes (BD)$.
3. 保可逆: $(A\otimes B)^{-1} = A^{-1} \otimes B^{-1}$.

利用矩阵的张量积可以求一类线性矩阵方程问题:
$$
A_1 X B_1 + A_2 X B_2 + \cdots + A_s X B_s = C,
$$

其中 $A_i\in M_m(\mathbb{C}), B_i\in M_n(\mathbb{C})$ 均为方阵, $C\in\mathbb{C}^{m\times n}$ 是已知矩阵, $X\in \mathbb{C}^{m\times n}$ 是未知矩阵.

为此引入两个操作, 设矩阵 $A=(a_{ij})\_{m\times n}$, 将 $A$ 的各列依次竖排得到 $mn$ 维的向量, 记为列展开 $vec(A)$, 类似地, 行展开 $rvec(A)$.

其重要的性质是: $rvec(ABC)=rvec(B)(A^T\otimes C)$. 证明如下:

先确定形状 $A=(a_{ij})\_{m\times n}, B=(b_{ij})\_{n\times s}, C=(c_{ij})\_{s\times t}$, 拆解矩阵 $B=\displaystyle\sum_{i,j} b_{ij}e_i^{(n)}{e_j^{(m)}}^T$.

那么 $ABC=\displaystyle\sum_{i,j} b_{ij} Ae_i^{(n)} \left(C^T e_j^{(s)}\right)^T$, $rvec(ABC)=\displaystyle\sum_{i,j}b_{ij}\cdot rvec\left(Ae_i^{(n)} \left(C^T e_j^{(s)}\right)^T\right)$.

而 $rvec\left(Ae_i^{(n)} \left(C^T e_j^{(s)}\right)^T\right)$ 形式简单, 可以直接写成 $\left(A e_i^{(n)}\right)^T\otimes\left(C^T e_j^{(s)}\right)^T=\left(\left(e_i^{(n)}\right)^T\otimes\left(e_j^{(s)}\right)^T\right)\left(A^T\otimes C\right)$.

所以 $rvec(ABC)=\displaystyle\sum_{i,j}b_{ij}\left(\left(e_i^{(n)}\right)^T\otimes\left(e_j^{(s)}\right)^T\right)\left(A^T\otimes C\right)=rvec(B)(A^T\otimes C)$.

只需解方程 $vec\(C\)=\displaystyle\sum_{i=1}^s \left(B_i ^T \otimes A_i\right) vec(X)$.

