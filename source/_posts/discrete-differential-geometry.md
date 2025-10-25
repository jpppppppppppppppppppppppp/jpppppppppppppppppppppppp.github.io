---
title: "Discrete Differential Geometry"
date: 2025-10-23 21:20:51
updated: 2025-10-24 23:44:39
home_cover: https://p.sda1.dev/28/4758f7df8a1e40db126c60e74da62de0/cover.png
post_cover: https://p.sda1.dev/28/c9db327e7e9ec33ba2973c63a6eed6f3/post.PNG
copyright_info: true
tags:
    - Math
categories:
    - Notes
mathjax: true
tikzjax: true
excerpt: "[Keenan Crane] Discrete Differential Geometry"
---

Homepage: <a href="https://www.cs.cmu.edu/~kmcrane/Projects/DDG/">Discrete Differential Geometry: An Applied Introduction</a>

### A Quick and Dirty Introduction to Differential Geometry

一个三维空间的曲面可以描述为映射 $f:M\mapsto\mathbb{R}^3$, 它的微分 $df$ 则是将一个 $M$ 中的切向量映射到 $fM$ 的切空间. 我们用 $TM$ 表示 $M$ 上所有的切向量.

有了切空间, 我们可以定义法向量 $N$, 对于一个可定向的曲面, 可以把法向量理解成一个连续映射 $N:M\mapsto\mathbb{S}^2$ (Gauss Map), 单位球面显然也是三维空间的一个曲面, 我们可以把 $N$ 看做 $f$, 它的微分 $dN$ (Weingarten Map) 告诉了我们法向量变化的方向.

在处理曲线的时候, 我们往往会用到 isometric reparameterization 的想法, 也可以叫做共形映射, $|df(X)|=|X|$, 但是对于曲面而言, 这种映射有可能不存在, 甚至是局部的也不存在, 作为替代品, 我们可以使用保角映射 (conformal coordinates), 要求 $df(X)\cdot df(Y)=a\left<X,Y\right>$, 这里要求 $X,Y$ 是 $M$ 中的所有切向量, $a$ 是一个关于 $M$ 的正标量函数. 这表明曲面仍然可以被拉伸, 但是不能切变, 因为正交的切向量映射后仍然是正交的.

一个重要的事实是, 由<a href="https://en.wikipedia.org/wiki/Uniformization_theorem">uniformization theorem</a>保证, 每一个点的领域都可以找到保角映射. 就像曲线一样, 我们往往只需要知道直线的共形映射存在, 并不需要显式地写出它的表达式, 类似地, 对于曲面, 我们也只需要知道在一个领域内保角映射存在, 并且只需要知道该点的函数值 $a(p)$, 就可以知道这个曲面是如何拉伸的, 而不需要完整的雅可比矩阵.

下面介绍一下有关曲线曲率的内容, 不加说明的, 我们用 $\gamma$ 表示一般地正则曲线, 用 $s$ 表示弧长重参数化.

我们先假设曲线在二维平面, 对于弧长参数化 $s$, 对它做泰勒展开 $s(t+\Delta t)=s(t)+\dot{s}(t)\Delta t+\displaystyle\frac{1}{2}\ddot{s}(t)\Delta t^2+O(\Delta t^3)$, 曲率则是考虑它关于切线的偏离程度, 所以 $(s(t+\Delta t)-s(t))\cdot N=\displaystyle\frac{1}{2}\ddot{s}(t)\cdot N(\Delta t)^2+O(\Delta t^3)$. 由于 $\|\dot{s}(t)\|^2=1$, 所以 $\dot{s}(t)\cdot\ddot{s}(t)=0$, 所以 $\ddot{s}(t)$ 与 $N(t)$ 平行, 于是我们定义曲率为 $\kappa(t)=\|\ddot{s}(t)\|$. 我们可以把这个定义推广到任意高维空间. 对于一般曲线的曲率计算, 我们有
$$
\kappa=\frac{\|\ddot{\gamma}\times\dot{\gamma}\|}{\|\dot{\gamma}\|^3}.
$$

法向量也可以通过曲率定义, $N(t)=\displaystyle\frac{1}{\kappa(t)}\ddot{s}(t)$, 它是一个与切向量 $T(t)=\dot{s}(t)$ 垂直的单位向量, 我们定义附法向量(binormal) $B=T\times N$, 这样 $\\{T,N,B\\}$ 构成了一组右手正交基.

由于 $B(t)$ 是单位向量, 所以 $\dot{B}(t)$ 与 $B(t)$ 垂直, 又因为 $\dot{B}=\dot{T}\times N+T\times\dot{N}=T\times\dot{N}$, 所以 $\dot{B}(t)$ 与 $T(t)$ 也垂直, 因此 $\dot{B}(t)$ 与 $N(t)$ 平行, 我们定义挠率 (torsion) $\tau(t)$ 使得 $\dot{B}(t)=-\tau(t)N(t)$. 挠率只有在曲率不为零的点有定义. 对于一般曲线的挠率计算, 我们有
$$
\tau=\frac{(\dot{\gamma}\times\ddot{\gamma})\cdot\dddot{\gamma}}{\|\dot{\gamma}\times\ddot{\gamma}\|^2}.
$$

如果正则曲线的曲率处处为零, 那么它就是一条直线; 如果它挠率处处为零(假设处处有定义), 那么它可以被一个平面包含.

利用右手正交基, 可以进一步得到 $\dot{N}=\dot{B}\times T+B\times\dot{T}=-\tau N\times T+\kappa B\times N=\tau B-\kappa T$. 总结为 Frenet–Serret equations:
$$
\begin{bmatrix}\dot{T}\\\\\dot{N}\\\\\dot{B}\end{bmatrix}=\begin{bmatrix}0 & \kappa & 0\\\\-\kappa & 0 & \tau\\\\0 & -\tau & 0\end{bmatrix}\begin{bmatrix}T\\\\N\\\\B\end{bmatrix}.
$$

这个反对称矩阵非常好理解, 可以把式子写成 $\dot{Q}=AQ$, 其中 $QQ^T=I$ 因为这是单位正交基, 所以 $\dot{Q}Q^T=-(\dot{Q}Q^T)^T$, 因此 $A$ 是反对称的.


