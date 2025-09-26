---
title: "Note: Harmonic Analysis"
date: 2025-09-20 16:07:09
updated: 2025-09-23 21:43:22
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

### 第一节课

第一节课介绍了这节课使用的主要<a href="#ref1">教材</a>, 内容部分涵盖了这本书的前若干章, 简要介绍傅里叶分析的正确性, 各种意义上的收敛性等; 后半程则根据情况介绍调和分析的应用, 包括小波分析和组合学应用等.

傅里叶是在尝试解决固体热传导问题时, 为了解傅里叶热传导方程: $\displaystyle\frac{\partial u}{\partial t}=\alpha\Delta u$, 提出将 $u(x, t)$ 分解为 $u(x,t)=a(x)b(t)$, 提出一组基本解, 从而假定所有的初始条件都可以分解为三角函数的和. 傅里叶方法提出的时候也是一众哗然, 缺少严格的数学证明, 直到 <a href="#ref2">1966</a> 年才有工作证明了逐点收敛性.

拿 Taylor series 作为更加熟悉的例子, $f(x)\sim\displaystyle\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n$, 在十八世纪, 数学家发现在传统微积分里的函数, 比如对数指数幂, 都能在各点有很好的收敛性, 直到 1821 年, Cauchy 给出反例 $f(x)=\begin{cases}e^{-1/x^2}&x\neq0,\\\\0&x=0.\end{cases}$, 这个函数在 $x=0$ 的任意阶导数都为零, 所以展开式只在这一点收敛. 原因是在复数域上 $z=0$ 是它的奇点, 但是在实数域上是无穷次可微的. 类似地, 我们要发问: 已知傅里叶系数, 或者说一个函数在这组基上的投影, 能否重建原函数? 如果能重建, 收敛的速度如何呢?

对于第一个问题, 答案是肯定的, 如果重建的方法是 partial sum, $S_N(f)(x)=\displaystyle\sum_{n=-N}^N\hat{f}(n)e^{2\pi inx/L}$, 是有反例不收敛的, Fej&eacute;r 说使用前项和的平均可以重建原函数, 具体内容在教材的第四章, 等我看到了再补充具体内容.

#### 一个简单的应用例子

第一节课吴老师举了一个具体的应用例子, 是 Weyl's Equidistribution Theorem.

{% note info %}
Weyl\'s Equidistribution Theorem:

取一个无理数 $\gamma\in\mathbb{R}\backslash\mathbb{Q}$, 一个周期为 $2\pi$ 的连续函数 $f:\mathbb{T}\mapsto\mathbb{C}$, $\mathbb{T}=\mathbb{R}/2\pi\mathbb{Z}$, 那么

1. $n^{-1}\displaystyle\sum_{r=1}^nf(2\pi r\gamma)\to\displaystyle\frac{1}{2\pi}\displaystyle\int_{\mathbb{T}}f(t)dt$,
2. $0\leq a\leq b\leq1$, $n^{-1}\displaystyle\sum_{r=1}^n\mathbb{1}_{2\pi r\gamma\in[a,b]}\to b-a$.
{% endnote %}

第一问就是验证每一个傅里叶级数的基都是满足该极限的, 再利用 Fej&eacute;r 求和说明任意函数可以用三角级数逼近; 第二问则是将第一问要求的连续函数换成了不连续的指示函数, 用线性函数上下夹逼即可.

吴老师说这是在 Weyl 做表示论的时候提出的该定理, 很难想象是啥表示论的难题需要这个..

### 教材阅读

#### 收敛性相关:

##### 定理 1.1

{% note info %}
Theorem 1.1:

取连续且二阶连续可导函数, $f\in C^2(\mathbb{T})$, 其中 $\mathbb{T}$ 是单位圆, 那么对于任意 $\theta\in\mathbb{T}$, 部分和的极限
$$
\lim_{N\to\infty}S_N(f)(\theta)=\sum_{n=-\infty}^\infty\hat{f}(n)e^{in\theta}=: S(f)(\theta)
$$
存在且一致收敛.
{% note success %}
$f$ 在 $\mathbb{T}$ 上连续, 是指 $f(-\displaystyle\frac{T}{2})=f(\displaystyle\frac{T}{2})$ 且 $f'(-\displaystyle\frac{T}{2})=f'(\displaystyle\frac{T}{2})$.
{% endnote %}
{% endnote %}

{% note primary %}
Proof:

函数 $f$ 的部分和定义如下: $S_N(f)(\theta)=\displaystyle\sum_{|n|\leq N}\hat{f}(n)e^{in\theta}$.
$$
\sum_{n=-\infty}^\infty|\hat{f}(n)e^{in\theta}|=\sum_{n=-\infty}^\infty|\hat{f}(n)|<\infty,
$$
如果傅里叶级数系数和绝对收敛, 那么根据 Weierstrass M-test, 级数本身一定也收敛, 且是一致收敛的.
{% note info %}
Weierstrass M-test:

设 $\\{f_n\\}$ 是定义在集合 $E\subset\mathbb{R}$ 上的函数列, 如果存在数列 $\\{a_n\\}$ 使得对任意 $x\in E$ 和任意 $n$, 都有 $|f_n(x)|\leq a_n$, 且 $\displaystyle\sum_{n=1}^\infty a_n<\infty$, 那么 $\displaystyle\sum_{n=1}^\infty f_n(x)$ 在 $E$ 上一致收敛.
{% endnote %}

下面就是证明傅里叶系数绝对收敛. 可以用分部积分法对系数进行改造:
<span id="1.1"></span>
$$
\begin{aligned}
\hat{f}(n)&=\frac{1}{2\pi}\int_{-\pi}^\pi f(\theta)e^{-in\theta}d\theta\\\\
&=\frac{1}{2\pi}\left[-\frac{1}{in}f(\theta)e^{-in\theta}\right]^\pi_{-\pi}-\frac{1}{2\pi}\int_{-\pi}^\pi -\frac{1}{in}f'(\theta)e^{-in\theta}d\theta\\\\
&=\frac{1}{in}\frac{1}{2\pi}\int_{-\pi}^\pi f'(\theta)e^{-in\theta}d\theta=\frac{1}{in}\hat{f'}(n).
\end{aligned}
\tag{1.1}
$$

类似地, 我们有 $\hat{f}(n)=-\displaystyle\frac{\hat{f'\'}(n)}{n^2}$. 所以对系数展开估计,
$$
\begin{aligned}
|\hat{f}(n)|&=\frac{1}{n^2}|\hat{f'\'}(n)|=\frac{1}{n^2}\left|\frac{1}{2\pi}\int_{-\pi}^\pi f'\'(\theta)e^{-in\theta}d\theta\right|\\\\
&\leq\frac{1}{n^2}\left(\frac{1}{2\pi}\int_{-\pi}^\pi |f'\'(\theta)|d\theta\right),
\end{aligned}
$$
由于 $f'\'$ 在 $\mathbb{T}$ 上连续, 所以有界, 故而 $|\hat{f}(n)|<C/n^2$, 对于任意的 $n\neq0$都成立.

从而傅里叶系数绝对收敛, 进一步原傅里叶级数一致收敛.
{% endnote %}

##### 定理 1.2

假定我们先承认 Fej&eacute;r 定理, 即连续函数的部分和的 Ces&agrave;ro 一致收敛于原函数, 那么我们可以得到:

{% note info %}
Theorem 1.2:

$f\in C^2(\mathbb{T})$, 那么 $S_N(f)$ 一致收敛于 $f$.
{% endnote %}

{% note primary %}
对于一个收敛的数列 $\\{a_n\\}$, 有:
$$
\lim_{n\to\infty}(a_1+a_2+\dots+a_n)/n=\lim_{n\to\infty}a_n.
$$
{% endnote %}

##### 定理 1.3

从式子 <a href="#1.1">(1.1)</a>, 对于函数 $f\in C^k$, 可以对他求 k 阶导, 所以:

$$
|\hat{f}(n)|=|(in)^{-k}\hat{f^{(k)}}(n)|\leq C/n^k.
$$

而如果傅里叶系数像 $n^{-k}$ 同等速率衰减, 那是否说明 $f\in C^k$. 下面是一个 $k=1$ 的反例, 在 $\mathbb{T}$ 上的指示函数 $\chi_{[0,1]}$ 甚至不是连续函数, $\chi_{[0,1]}\not\in C^0$, 但是傅里叶系数是 $O(1/n)$ 的.

$$
\widehat{\chi_{[0,1]}}(n)=\frac{1}{2\pi}\int_0^1e^{-in\theta}d\theta=\frac{1}{2\pi}\frac{e^{-in}-1}{-in}.
$$

类似地, 我们可以仿照这个例子构造 $k\geq 2$ 的反例. 但是, 下面这个定理及其推论表明, 这样的函数一定在 $C^{k-2}(\mathbb{T})$ 里.

{% note info %}
Theorem 1.3:

可积函数 $f:\mathbb{T}\mapsto\mathbb{C}$ 是以 $2\pi$ 为周期, 且 $l\geq 0$. 如果 $\displaystyle\sum_{n\in\mathbb{Z}}|\hat{f}(n)||n|^l<+\infty$, 那么函数 $f\in C^l$.

推论:

如果对于 $k\geq 2$ 和不考虑常数项 $n\neq 0$, 有 $|\hat{f}(n)|\leq C|n|^k$, 那么 $f\in C^l$. 如果 $k\in\mathbb{Z}$ 是整数, 那么 $l=k-2$; 如果 $k\not\in\mathbb{Z}$ 不是整数, 那么 $l=\lfloor k \rfloor-1$.
{% endnote %}

{% note primary %}
Proof:

对 $l$ 做数学归纳法.
当 $l=0$ 时, 由上文所述定理, 先得到 $S_N(f)$ 一致收敛到 $S(f)$, 其中, $S(f)$ 是函数项级数, 每个函数都是连续的可以推出和函数也是连续的.

{% note success %}
设函数 $u_n(x)$ 定义在闭区间 $X=[a,b]$ 上, 并在一点 $x=x_0\in X$ 上都连续, 若 $\displaystyle\sum_{n=1}^\infty u_n(x)\rightrightarrows f(x)$, 则和函数 $f(x)$ 在 $x=x_0$ 处也连续.
设 $f(x)=f_n(x)+\phi_n(x)$, 那么 $|f(x)-f(x_0)|\leq |f_n(x)-f_n(x_0)|+|\phi_n(x)|+|\phi_n(x_0)|$, 根据一致收敛和连续性, 可以放缩到 $3\varepsilon$.
{% endnote %}

再由 Fej&eacute;r 定理和均值极限等式, $S(f)=f$, 所以 $f\in C^0$.

当 $l=1$ 时, 除了 $S(f)\rightrightarrows f$ 之外, 还有 $(S(f))\'\rightrightarrows f'$, 这个依赖于这个结论:
{% note success %}
$\\{f_n:[a,b]\mapsto\mathbb{C}\\}$ 为一组连续可微函数且一致收敛于函数 $f$, 那么:
$$
\lim_{n\to\infty}(f_n)' = \left(\lim_{n\to\infty}f_n\right)'.
$$
{% endnote %}
后面的归纳类似.
{% endnote %}

##### Riemann-Lebesgue Lemma

从上面可以知道, 对于连续可微函数 (换言之, 光滑的函数), 傅里叶级数都会快速地衰减到零. 如果限制条件只有连续呢, 下面的引理说明其傅里叶级数也会衰减到零.

{% note info %}
Riemann-Lebesgue Lemma:

连续函数 $f\in C(\mathbb{T})$ 的傅里叶级数收敛至零, $\displaystyle\lim_{|n|\to\infty}\hat{f}(n)=0$.
{% endnote %}

{% note primary %}
Proof for Lemma:

$$
\begin{aligned}
\hat{f}(n)&=\frac{1}{2\pi}\int_{-\pi}^\pi f(\theta)e^{-in\theta}d\theta = -\frac{1}{2\pi}\int_{-\pi}^\pi f(\theta) e^{-in\theta} e^{i\pi}d\theta\\\\
&=-\frac{1}{2\pi}\int_{-\pi}^\pi f(\theta) e^{-in(\theta-\pi/n)}d\theta\\\\
&=-\frac{1}{2\pi}\int_{-\pi-\pi/n}^{\pi-\pi/n}f(\alpha+\pi/n)e^{-in\alpha}d\alpha\\\\
&=-\frac{1}{2\pi}\int_{-\pi}^\pi f(\alpha+\pi/n)e^{-in\alpha}d\alpha.
\end{aligned}
$$

所以两式相加, 得到:
$$
\hat{f}(n)=\frac{1}{4\pi}\int_{-\pi}^\pi [f(\theta)-f(\theta+\pi/n)]e^{-in\theta}d\theta.
$$

两边取范数, 有:
$$
|\hat{f}(n)|=\frac{1}{4\pi}\int_{-\pi}^\pi |f(\theta)-f(\theta+\pi/n)|d\theta.
$$

因为函数 $f$ 在闭区间 $\mathbb{T}$ 上连续, 所以一致连续, 所以 $g_n(\theta)=|f(\theta)-f(\theta+\pi/n)|$ 一致收敛到零, 所以可以交换极限和积分次序, 得到:
$$
0\leq \lim_{|n|\to\infty} |\hat{f}(n)| \leq \frac{1}{4\pi}\int_{-\pi}^\pi \lim_{|n|\to\infty} |f(\theta)-f(\theta+\pi/n)|d\theta=0.
$$
{% endnote %}



---

推荐阅读:

- <span id="ref1"></span> Pereyra, M. C., & Ward, L. A. (2012). <a href="https://fig.if.usp.br/~marchett/fismat2/harmonic-analysis-fourier-wavelet_pereyra-ward.pdf"> Harmonic analysis: from Fourier to wavelets (Vol. 63).</a> American Mathematical Soc.

- <span id="ref2"></span> Carleson, L. (1966). <a href="https://projecteuclid.org/journals/acta-mathematica/volume-116/issue-none/On-convergence-and-growth-of-partial-sums-of-Fourier-series/10.1007/BF02392815.full">On convergence and growth of partial sums of Fourier series.</a>

- Lacey, M., & Thiele, C. (2000). <a href="https://archive.intlpress.com/site/pub/files/_fulltext/journals/mrl/2000/0007/0004/MRL-2000-0007-0004-a001.pdf">A proof of boundedness of the Carleson operator.</a> Mathematical Research Letters, 7(4), 361-370.



