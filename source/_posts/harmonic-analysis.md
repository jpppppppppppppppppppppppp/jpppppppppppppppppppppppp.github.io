---
title: "Note: Harmonic Analysis"
date: 2025-09-20 16:07:09
updated: 2025-09-21 16:46:14
home_cover: https://p.sda1.dev/27/3b163beb87dacac2e7af5d12fa1e5c27/cover.PNG
post_cover: https://p.sda1.dev/27/112269185d77bddf4f1efd879257d4c2/post.JPG
copyright_info: true
tags:
    - Math
categories:
    - Notes
mathjax: true
excerpt: 出于纯粹的好奇心, 选了吴耀琨老师的调和分析课程, 谨做笔记.
---

第一节课介绍了这节课使用的主要<a href="#ref1">教材</a>, 内容部分涵盖了这本书的前若干章, 简要介绍傅里叶分析的正确性, 各种意义上的收敛性等; 后半程则根据情况介绍调和分析的应用, 包括小波分析和组合学应用等.

傅里叶是在尝试解决固体热传导问题时, 为了解傅里叶热传导方程: $\displaystyle\frac{\partial u}{\partial t}=\alpha\Delta u$, 提出将 $u(x, t)$ 分解为 $u(x,t)=a(x)b(t)$, 提出一组基本解, 从而假定所有的初始条件都可以分解为三角函数的和. 傅里叶方法提出的时候也是一众哗然, 缺少严格的数学证明, 直到 <a href="#ref2">1966</a> 年才有工作证明了逐点收敛性.

拿 Taylor series 作为更加熟悉的例子, $f(x)\sim\displaystyle\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n$, 在十八世纪, 数学家发现在传统微积分里的函数, 比如对数指数幂, 都能在各点有很好的收敛性, 直到 1821 年, Cauchy 给出反例 $f(x)=\begin{cases}e^{-1/x^2}&x\neq0;\\\\0&x=0.\end{cases}$, 这个函数在 $x=0$ 的任意阶导数都为零, 所以展开式只在这一点收敛. 原因是在复数域上 $z=0$ 是它的奇点, 但是在实数域上是无穷次可微的. 类似地, 我们要发问: 已知傅里叶系数, 或者说一个函数在这组基上的投影, 能否重建原函数? 如果能重建, 收敛的速度如何呢?

对于第一个问题, 答案是肯定的, 如果重建的方法是 partial sum, $S_N(f)(x)=\displaystyle\sum_{n=-N}^N\hat{f}(n)e^{2\pi inx/L}$, 是有反例说明不收敛的, Fej&eacute;r 说使用前项和的平均可以重建原函数, 具体内容在教材的第四章, 等我看到了再补充具体内容.

第一节课吴老师举了一个具体的应用例子, 是 Weyl's Equidistribution Theorem.

{% note info %}
Weyl's Equidistribution Theorem:

取一个无理数 $\gamma\in\mathbb{R}\backslash\mathbb{Q}$, 一个周期为 $2\pi$ 的连续函数 $f:\mathbb{T}\mapsto\mathbb{C}$, $\mathbb{T}=\mathbb{R}/2\pi\mathbb{Z}$, 那么

1. $n^{-1}\displaystyle\sum_{r=1}^nf(2\pi r\gamma)\to\displaystyle\frac{1}{2\pi}\displaystyle\int_{\mathbb{T}}f(t)dt$,
2. $0\leq a\leq b\leq1$, $n^{-1}\displaystyle\sum_{r=1}^n\mathbb{1}_{2\pi r\gamma\in[a,b]}\to b-a$.
{% endnote %}

第一问就是验证每一个傅里叶级数的基都是满足该极限的, 再利用 Fej&eacute;r 求和说明任意函数可以用三角级数逼近; 第二问则是将第一问要求的连续函数换成了不连续的指示函数, 用线性函数上下夹逼即可.

吴老师说这是在 Weyl 做表示论的时候提出的该定理, 很难想象是啥表示论的难题需要这个..

---

推荐阅读:

- <span id="ref1"></span> Pereyra, M. C., & Ward, L. A. (2012). <a href="https://fig.if.usp.br/~marchett/fismat2/harmonic-analysis-fourier-wavelet_pereyra-ward.pdf"> Harmonic analysis: from Fourier to wavelets (Vol. 63).</a> American Mathematical Soc.

- <span id="ref2"></span> Carleson, L. (1966). <a href="https://projecteuclid.org/journals/acta-mathematica/volume-116/issue-none/On-convergence-and-growth-of-partial-sums-of-Fourier-series/10.1007/BF02392815.full">On convergence and growth of partial sums of Fourier series.</a>

- Lacey, M., & Thiele, C. (2000). <a href="https://archive.intlpress.com/site/pub/files/_fulltext/journals/mrl/2000/0007/0004/MRL-2000-0007-0004-a001.pdf">A proof of boundedness of the Carleson operator.</a> Mathematical Research Letters, 7(4), 361-370.



