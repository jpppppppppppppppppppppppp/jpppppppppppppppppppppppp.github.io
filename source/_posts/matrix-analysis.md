---
title: "Matrix Analysis"
date: 2026-04-24 23:22:22
updated: 2026-07-03 0:31:14
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

(Cayley-Hamilton 定理): 矩阵 $A$ 的特征多项式为 $p_A(\lambda)$, 则有 $p_A(A)=O$.

首先需要伴随矩阵的性质: $S\operatorname{adj}(S)=\operatorname{det}(S)I_n$. 只需要展开 $(S \operatorname{adj}(S))\_{ij}=\displaystyle\sum\_{k}S\_{ik}M\_{jk}$. 如果 $i=j$, 那么 $(S \operatorname{adj}(S))\_{ij}=\operatorname{det}(S)$. 如果 $i\neq j$, 那么可以视为 $S$ 的第 $i$ 行复制到了第 $j$ 行, 所以等于 $0$.

取 $S=\lambda I_n - A$, $B=\operatorname{adj}(\lambda I_n - A)$, 那么 $p_A(\lambda) I_n=(\lambda I_n-A)B=(\lambda I_n-A)\displaystyle\sum_{i=0}^{n-1}\lambda^i B_i$.

$\operatorname{RHS}=\lambda^n B_{n-1}+\displaystyle\sum_{i=1}^{n-1}\lambda^i (B_{i-1}-AB_i) - AB_0$.

$\operatorname{LHS}=\lambda^n I_n + \displaystyle\sum_{i=1}^{n-1}\lambda^i c_i I_n+c_0 I_n$.

对比系数得到:

$B_{n-1}=I_n,\quad B_{i-1}-AB_i=c_i I_n,\quad -AB_0=c_0 I_n$.

等式两边分别乘上 $A^i$ 并相加, 得到 $A^n+ c_{n-1} A^{n-1} + \cdots + c_1 A + c_0 = p_A(A) = O$.

Sylvester 降幂: 设 $A, B$ 分别是 $m\times n, n\times m$ 的矩阵, $m\geq n$, 则 $\operatorname{det}(\lambda I_m - AB)=\lambda^{m-n}\operatorname{det}(\lambda I_n - BA)$.

使用下述分块恒等式: 
$$
\begin{pmatrix}I_n & B \\\\ O & I_m\end{pmatrix}
\begin{pmatrix}O & O \\\\ A & AB\end{pmatrix}=
\begin{pmatrix}BA & BAB \\\\ A & AB\end{pmatrix}=
\begin{pmatrix}BA & O \\\\ A & O\end{pmatrix}
\begin{pmatrix}I_n & B \\\\ O & I_m\end{pmatrix}.
$$

所以矩阵 $C_1=\begin{pmatrix}O & O \\\\ A & AB\end{pmatrix}$ 与 $C_2=\begin{pmatrix}BA & O \\\\ A & O\end{pmatrix}$ 相似, 且非零的特征值相同.

设矩阵 $A, B$ 无公共特征值, 则 $AX=XB\Leftrightarrow X=O$.

记 $p_A(\lambda) = \operatorname{det}(\lambda I_n - A)$, 那么 $p_A(A)=O$ 且 $p_A(B)$ 可逆. 这是因为 $p_A(B)$ 的特征值是 $p_A(\lambda_i)$, 其中 $\lambda_i$ 是 $B$ 的特征值, 而 $A, B$ 无公共特征值, 所以 $p_A(\lambda_i)\neq 0$, 因此 $p_A(B)$ 可逆.

现设 $AX=XB$, 那么 $A^kX=XB^k,\forall k\geq0$, 于是对于任意 $g(x)\in\mathbb{F}[x]$, 均有 $g(A)X=Xg(B)$. 特别的, $O=p_A(A)X=Xp_A(B)$, 所以 $X=O$.

思考题: 设 $A, B$ 都是 $n\times n$ 的方阵, 由降幂公式知道 $AB$ 与 $BA$ 有相同的特征多项式, 这是否说明它们相似?

不一定相似, 相同的特征多项式不足以推出相似, 要求更强, 例如 Jordan 标准型相同, 或者最小多项式相同. 反例例如:

$A=\begin{pmatrix}1 & 0 \\\\ 0 & 0\end{pmatrix}, \quad B=\begin{pmatrix}0 & 1 \\\\ 0 & 0\end{pmatrix}$. 则 $AB=\begin{pmatrix}0 & 1 \\\\ 0 & 0\end{pmatrix}$, $BA=\begin{pmatrix}0 & 0 \\\\ 0 & 0\end{pmatrix}$. 它们的特征多项式都是 $\lambda^2$, 但是不相似.


