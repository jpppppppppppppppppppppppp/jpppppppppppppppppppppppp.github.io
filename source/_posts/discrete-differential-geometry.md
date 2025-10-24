---
title: "Discrete Differential Geometry"
date: 2025-10-23 21:20:51
updated: 2025-10-23 23:32:54
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



