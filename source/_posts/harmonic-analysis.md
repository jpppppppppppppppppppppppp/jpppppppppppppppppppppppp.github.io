---
title: "[MATH3610] Harmonic Analysis"
date: 2025-09-20 16:07:09
updated: 2025-11-17 23:14:33
home_cover: https://p.sda1.dev/27/3b163beb87dacac2e7af5d12fa1e5c27/cover.PNG
post_cover: https://p.sda1.dev/27/112269185d77bddf4f1efd879257d4c2/post.JPG
copyright_info: true
tags:
    - Math
categories:
    - Notes
mathjax: true
tikzjax: true
excerpt: 出于纯粹的好奇心, 选了吴耀琨老师的调和分析课程, 谨做笔记.
---

### 第一节课

第一节课介绍了这节课使用的主要<a href="#ref1">教材</a>, 内容部分涵盖了这本书的前若干章, 简要介绍傅里叶分析的正确性, 各种意义上的收敛性等; 后半程则根据情况介绍调和分析的应用, 包括小波分析和组合学应用等.

傅里叶是在尝试解决固体热传导问题时, 为了解傅里叶热传导方程: $\displaystyle\frac{\partial u}{\partial t}=\alpha\Delta u$, 提出将 $u(x, t)$ 分解为 $u(x,t)=a(x)b(t)$, 提出一组基本解, 从而假定所有的初始条件都可以分解为三角函数的和. 傅里叶方法提出的时候也是一众哗然, 缺少严格的数学证明, 直到 <a href="#ref2">1966</a> 年才有工作证明了逐点收敛性.

拿 Taylor series 作为更加熟悉的例子, $f(x)\sim\displaystyle\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n$, 在十八世纪, 数学家发现在传统微积分里的函数, 比如对数指数幂, 都能在各点有很好的收敛性, 直到 1821 年, Cauchy 给出反例 $f(x)=\begin{cases}e^{-1/x^2}&x\neq0,\\\\0&x=0.\end{cases}$, 这个函数在 $x=0$ 的任意阶导数都为零, 所以展开式只在这一点收敛. 原因是在复数域上 $z=0$ 是它的奇点, 但是在实数域上是无穷次可微的. 类似地, 我们要发问: 已知傅里叶系数, 或者说一个函数在这组基上的投影, 能否重建原函数? 如果能重建, 收敛的速度如何呢?

对于第一个问题, 答案是肯定的, 如果重建的方法是 partial sum, $S_N(f)(x)=\displaystyle\sum_{n=-N}^N\hat{f}(n)e^{2\pi inx/L}$, 是有反例不收敛的, Fej&eacute;r 说使用前项和的平均可以重建原函数, 具体内容在教材的第四章, 等我看到了再补充具体内容.

第一节课吴老师举了一个具体的应用例子, 是 Weyl's Equidistribution Theorem.

{% note info %}
Weyl\'s Equidistribution Theorem:

取一个无理数 $\gamma\in\mathbb{R}\backslash\mathbb{Q}$, 一个周期为 $2\pi$ 的连续函数 $f:\mathbb{T}\mapsto\mathbb{C}$, $\mathbb{T}=\mathbb{R}/2\pi\mathbb{Z}$, 那么

1. $n^{-1}\displaystyle\sum_{r=1}^nf(2\pi r\gamma)\to\displaystyle\frac{1}{2\pi}\displaystyle\int_{\mathbb{T}}f(t)dt$,
2. $0\leq a\leq b\leq1$, $n^{-1}\displaystyle\sum_{r=1}^n\mathbb{1}[2\pi r\gamma\in[a,b]]\to b-a$.
{% endnote %}

第一问就是验证每一个傅里叶级数的基都是满足该极限的, 再利用 Fej&eacute;r 求和说明任意函数可以用三角级数逼近; 第二问则是将第一问要求的连续函数换成了不连续的指示函数, 用线性函数上下夹逼即可.

吴老师说这是在 Weyl 做表示论的时候提出的该定理, 很难想象是啥表示论的难题需要这个..

---

### 第二节课

一开始吴老师回顾了在不同的群上, 时域和频域是什么: 比如在时域为 $\mathbb{R}/L\mathbb{Z}$ 上, 频域是 $\mathbb{Z}$; 在时域为 $\mathbb{Z}$ 上, 频域是 $\mathbb{R}/L\mathbb{Z}$; 在时域为 $\mathbb{R}$ 上, 频域也是 $\mathbb{R}$; 在时域为 $\mathbb{Z}/N\mathbb{Z}$ 上, 频域也是 $\mathbb{Z}/N\mathbb{Z}$. 我们可以发现, 不管在什么群上, 都会出现 $e^{2\pi inx/N}$ 这样的表示, 后面吴老师向我们推导了一般有限阿贝尔群上的傅里叶变换.

有一个有限阿贝尔群 $A$, 我们需要找到一个群 $A$ 上的 "简单的信号" 组合. 我们考虑 $A$ 的对偶群 $\hat{A}=Hom(A, \mathbb{C}^*)$, 意思是群 $A$ 和复数乘法群的同态群, 即满足条件 $f(a+b)=f(a)f(b)$ 的函数构成的群. 很显然如果 $f, g\in\hat{A}$, 那么 $fg\in\hat{A}$. 而且 $\hat{A}$ 也是一个有限阿贝尔群, 其阶和 $A$ 相同. 因为 $A$ 是个循环群, 所以 $f$ 一定由单位根组成的 $f(n)=\omega^n$.

$\hat{A}\subseteq \mathbb{C}^A$, 可以在 $\mathbb{C}^A$ 上定义内积: $\left<f, g\right>\_A=\displaystyle\frac{1}{|A|}\displaystyle\sum_{a\in A}f(a)\overline{g(a)}=\mathbb{E}_{A}[f\bar{g}]$.

先证明几个小性质: 取 $\hat{A}$ 中的一个函数 $\chi\in\hat{A}$, 如果 $\chi$ 为常函数 $\mathbb{1}$, 那么 $\mathbb{E}\_A[\chi]=1$; 反之, 则存在 $b\in A$, 使得 $\chi(b)\neq 1$. 那么
$$
\begin{aligned}
\mathbb{E}\_A[\chi]&=\frac{1}{|A|} \sum_{a\in A}\chi(a)\\\\
&=\frac{1}{|A|}\sum_{a\in A}\chi(a+b)\\\\
&=\chi(b)\frac{1}{|A|}\sum_{a\in A}\chi(a)=0.
\end{aligned}
$$

下面可以很容易地得到对偶群里的正交关系: $\left<\chi,\chi'\right>\_A=\mathbb{E}\_A[\chi\overline{\chi'}]=\delta_{\chi,\chi'}$. 吴老师说这个叫做不可约表示. 所以傅里叶变化 $f\in \mathbb{C}^A \mapsto \displaystyle\sum_{\chi\in\hat{A}}\mathcal{F}(f)_\chi\chi$ 就是构建了算子 $\mathcal{F}:\mathbb{C}^A\mapsto\mathbb{C}^{\hat{A}}$.

如果 $A$ 和 $\hat{A}$ 在某种意义上相同, 我们可以研究傅里叶变换的特征值. 比如如果 $A=\mathbb{R}$ 上, 特征向量是 $f(s)=e^{-\pi s^2}$. 今天讲有限阿贝尔群上的特征值.

取函数 $\hat{f}\in\mathbb{C}^{\hat{A}}$, 做这个函数在 $\hat{\hat{A}}=A$ 上的分解, 因为 $a(\chi_1\chi_2)=a(\chi_1)a(\chi_2)$, 所以 $A$ 是 $\hat{\hat{A}}$ 上的一种表示.
$$
\hat{f}=\sum_{a\in A}\hat{\hat{f}}(a)a,
$$

带入内积
$$
\begin{aligned}
\hat{\hat{f}}(a)&=\left<\hat{f},a\right>_{\hat{A}}=\mathbb{E}\_{\hat{A}}\left[\hat{f}\bar{a}\right]=\displaystyle\frac{1}{|\hat{A}|}\displaystyle\sum\_{\chi\in\hat{A}}\hat{f}(\chi)\bar{a}(\chi)\\\\
&=\frac{1}{|\hat{A}|}\sum\_{\chi\in\hat{A}}a^{-1}(\chi)\frac{1}{|A|}\sum\_{b\in A}f(b)\overline{\chi(b)}\\\\
&=\frac{1}{|A|}\sum\_{b\in A}f(b)\frac{1}{|\hat{A}|}\sum\_{\chi\in\hat{A}}\chi(a^{-1}b^{-1})\\\\
&=\frac{1}{|A|}\sum\_{b\in A}f(b)\mathbb{E}\_{\chi\in\hat{A}}\chi(a^{-1}b^{-1})\\\\
&=\frac{1}{|A|}\sum\_{b\in A}f(b)\delta\_{a^{-1}, b^{-1}}=\frac{1}{|A|}f(a^{-1}).
\end{aligned}
$$

所以 $\hat{\hat{\hat{\hat{f}}}}(a)=\displaystyle\frac{1}{|A|^2}f(a)$, 特征值是 $|A|^{-1/2}(-i)^n$.

吴老师用矩阵的方法推有限循环群的傅里叶变换, $A=\mathbb{Z}/N\mathbb{Z}$.

$$
\begin{pmatrix}
\hat{f}(0) \\\\
\hat{f}(1) \\\\
\vdots \\\\
\hat{f}(N-1)
\end{pmatrix}=\frac{1}{N}\begin{pmatrix}
\omega^{-0\cdot 0} & \omega^{-0\cdot 1} & \cdots & \omega^{-0\cdot (N-1)} \\\\
\omega^{-1\cdot 0} & \omega^{-1\cdot 1} & \cdots & \omega^{-1\cdot (N-1)} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
\omega^{-(N-1)\cdot 0} & \omega^{-(N-1)\cdot 1} & \cdots & \omega^{-(N-1)(N-1)}
\end{pmatrix}\begin{pmatrix}
f(0) \\\\
f(1) \\\\
\vdots \\\\
f(N-1)
\end{pmatrix}
$$

那就是求中间矩阵 $\Omega$ 的特征值. Schur 定理我们说, 这个矩阵的特征值是 $\sqrt{N}$, $i\sqrt{N}$, $-\sqrt{N}$, $-i\sqrt{N}$. 而且甚至重数也是确定的: $\lfloor\displaystyle\frac{N+4}{4}\rfloor$, $\lfloor\displaystyle\frac{N+1}{4}\rfloor$, $\lfloor\displaystyle\frac{N+2}{4}\rfloor$, $\lfloor\displaystyle\frac{N-1}{4}\rfloor$.

一个很重要的特征向量是:

取 $N$ 是一个奇素数 $p$, $A=\mathbb{Z}/p\mathbb{Z}$, 函数 $h_p(x)=\left(\displaystyle\frac{x}{p}\right)=\begin{cases}0 & a\equiv0(\bmod p),\\\\+1&a\not\equiv0(\bmod p), \exists x\in\mathbb{Z}, x^2\equiv a(\bmod p)\\\\-1&\forall x\in\mathbb{Z}, x^2\not\equiv a(\bmod p). \end{cases}$ 是傅里叶变换的特征函数.

可以用<a href="https://en.wikipedia.org/wiki/Euler%27s_criterion">欧拉判别法</a>验证它的乘法性质 $h_p\in \hat{A}$. 下面是它的傅里叶变换的性质:

(A) $\widehat{h_p}=\widehat{h_p}(1) h_p$;

$$
\widehat{h_p}(m)=\frac{1}{p}\sum_{k=0}^{p-1}h_p(k)e^{-2\pi ikm/p}
$$

1. $m=0$ 时, $\widehat{f}(m)=\displaystyle\frac{1}{p}\displaystyle\sum_{k=0}^{p-1}h_p(k)=0$. 因为 $\mathbb{E}_{\hat{A}}[\chi]=0$.
2. $m\neq0$ 时, 换元 $b=km$,
$$
\begin{aligned}
\widehat{h_p}(m)&=\displaystyle\frac{1}{p}\displaystyle\sum_{b=0}^{p-1}h_p(bm^{-1})e^{-2\pi ib/p}\\\\
&=h_p(m)\frac{1}{p}\sum_{b=0}^{p-1}h_p(b)e^{-2\pi ib/p}\\\\
&=h_p(m)\widehat{h_p}(1).
\end{aligned}
$$

(B) 高斯核函数:
$$
\begin{aligned}
g(m, p)&=\sum_{k=0}^{p-1}e^{2\pi im k^2/p}=1+\sum_{k=1}^{p-1}e^{2\pi im k^2/p}\\\\
&=1+\sum_{k=1}^{p-1}(1+h_p(k))e^{2\pi im k/p},
\end{aligned}
$$

最后一个等号成立是因为, 我们知道 $\mathbb{E}_{\hat{A}}[h_p]=0$, 这意味着 $k^2$ 在 $\bmod p$ 意义下会遍历二次剩余群两次, 因为 $k^2\equiv (p-k)^2(\bmod p)$. 所以

$$
g(m, p)=\sum_{k=1}^{p-1}h_p(k)e^{2\pi im k/p}=p\cdot\widehat{h_p}(m).
$$

\(C\) 如果 $(r,s)=1$ 互素, 那么 $g(mr,s)g(ms,r)=g(m,rs)$. 需要使用中国剩余定理.

综合这三条, 可以证明二次互反律: $p$, $q$ 是奇素数, 那么 $\left(\displaystyle\frac{p}{q}\right)\left(\displaystyle\frac{q}{p}\right)=(-1)^{\frac{(p-1)(q-1)}{4}}$.

$$
\begin{aligned}
trace(\Omega_{pq})&=g(1, pq)\overset{\(C\)}{=}g(p,q)g(q,p)\overset{(B)}{=}pq\widehat{h_p}(q)\widehat{h_q}(p)\\\\
&\overset{(A)}{=}pq\widehat{h_p}(1)h_p(q)\widehat{h_q}(1)h_q(p)\overset{(B)}{=}\left(\displaystyle\frac{p}{q}\right)\left(\displaystyle\frac{q}{p}\right)g(1,p)g(1,q)\\\\
&=\left(\displaystyle\frac{p}{q}\right)\left(\displaystyle\frac{q}{p}\right)trace(\Omega_p)trace(\Omega_q)
\end{aligned}
$$

最后根据 Schur 定理计算矩阵 $\Omega$ 的迹, 从而得证.

吴老师的碎碎念: 这个矩阵的迹高斯不会求, 但是我们的重点并不是用简单的方法来计算它. 高斯用了很多种方法去证明二次互反律, 这是他觉得最漂亮的结果, 所以给了很多的证明, 因为不同的证明意味着后面有不同非常深刻的推广, 最复杂的证明才有可能走到最深刻的数学里面去. 很简单的证明不一定比复杂的证明来的好, 因为简单的证明在一般的场景基本用不下去.

---

### 第三节课

今天主要围绕有限阿贝尔群上的傅里叶变换上的性质.

第一部分是卷积的性质. $f\*g(x)=\mathbb{E}\_{y\in G}f(y)g(x-y)$, 傅里叶变换 $\mathcal{F}:\mathbb{C}^G\mapsto\mathbb{C}^{\hat{G}}$ 给出了 $(\mathbb{C}^G,\*)$ 和 $(\mathbb{C}^{\hat{G}},\cdot)$ 的同构:

$$
\begin{aligned}
\widehat{f\*g}(\chi)&=\mathbb{E}\_{x\in G}f\*g(x)\overline{\chi(x)}\\\\
&=\mathbb{E}\_{x\in G}\mathbb{E}\_{y\in G}f(y)g(x-y)\overline{\chi(y)}\overline{\chi(x-y)}\\\\
&=\hat{f}\cdot\hat{g}(\chi).
\end{aligned}
$$

反过来, 傅里叶变换把逐点乘变成类似的卷积: $\widehat{f\cdot g}(\chi)=|G|\hat{f}\*\hat{g}(\chi)$. 实际上卷积算子 $\mathcal{R}=\\{R_h:h\in\mathbb{C}^G\\}$ 是一个可交换代数, 且与 $L(G)=(\mathbb{C}^G,\*)$ 同构, $R_{h_1}R_{h_2}=R_{h2}R_{h_1}=R_{h_1\*h_2}$.

如果 $R\in\text{End}(\mathbb{C}^G)$, 也就是说 $R:\mathbb{C}^G\mapsto\mathbb{C}^G$ 是线性映射, 那么下面三个条件等价:

1. $R$ 关于 translation 可交换, 即 $\tau_x(Rf)=R(\tau_x f),\forall x\in G, f\in\mathbb{C}^G$, 其中 $\tau_x f(y)=f(y-x)$.
2. $R$ 是卷积算子, 即存在 $g\in\mathbb{C}^G$, 使得 $Rf=f\*g,\forall f\in\mathbb{C}^G$.
3. $\hat{G}$ 中的每一个 $\psi_a$ 都是 $R$ 的特征函数, 即存在 $\lambda_a\in\mathbb{C}$, 使得 $R\psi_a=\lambda_a\psi_a,\forall \psi_a\in\hat{G}$.

假设满足(1), 即 $\tau_x(Rf)=R(\tau_x f)$, 取任意函数 $f$, $f(x)=\displaystyle\sum_{a\in G}f(a)\delta(x-a)$, 那么
$$
\begin{aligned}
Rf(x)&=R(\sum_{a\in G}f(a)\tau_a\delta)(x)=\sum_{a\in G}f(a)R(\tau_a\delta)(x)\\\\
&=\sum_{a\in G}f(a)R\delta(x-a)=f\*g(x),
\end{aligned}
$$

其中 $g(x)=|G|R\delta(x)$.

假设满足(3), 即 $R\psi_a=\lambda_a\psi_a,\forall\psi_a\in\hat{G}$. 对于一个函数 $f$, 先做分解 $f=\displaystyle\sum_{\psi_a\in\hat{G}}\hat{f}(\psi_a)\psi_a$. 那么:

$$
\begin{aligned}
Rf(x)&=\sum_{\psi_a\in\hat{G}}\hat{f}(\psi_a)R\psi_a(x)=\sum_{\psi_a\in\hat{G}}\hat{f}(\psi_a)\lambda\_a\psi_a(x)\\\\
&=\frac{1}{|G|}\sum_{\psi_a\in\hat{G}}\lambda\_a\psi_a(x)\sum_{y\in G}f(y)\overline{\psi_a(y)}\\\\
&=\frac{1}{|G|}\sum_{y\in G}f(y)\sum_{\psi_a\in\hat{G}}\lambda\_a\psi_a(x-y)=f\*g(x),
\end{aligned}
$$

其中 $g(x)=\displaystyle\sum_{\psi_a\in\hat{G}}\lambda\_a\psi_a(x)$.

这里用到的想法是利用空间 $\mathbb{C}^G$ 的两组基, 一组是常见的 delta function, 另一组是满足傅里叶反演的 character function.

从(1)推(3), 可以利用 delta function 构成基的特性, 将 translation 推展到任意函数, $R(f)\*g=R(f*g)$, 从而取 $g=\psi_a\in\hat{G}$, 可以推出 $\widehat{R(f)}(\psi_a)\psi_a=\hat{f}(\psi_a)R(\psi_a)$, 这说明 $\psi_a$ 是 $R$ 的特征函数.

另一个有趣的结论是, 如果把卷积算子写成矩阵形式 $R_h$, 它的迹可以用结论 (3) 来计算: $\text{trace}(R_h)=\displaystyle\sum_{\psi\in\hat{G}}\hat{h}(\psi)=h(0)$.

下面话题进入了离散的泊松求和公式, 是在描述在一个 lattice 上定义的函数求和的一些性质. 首先 $A$ 是一个有限的阿贝尔群, $B\leq A$ 是它的一个子群, 对于商群 $A/B$ 上的函数 $f\in L(A/B)$, 可以把它提升到 $A$ 上, $\tilde{f}=f\circ\pi$, 其中 $\pi;A\mapsto A/B$ 是自然的投影映射. 

```tikz
\usepackage{tikz-cd}
\begin{document}
\Large
\begin{tikzcd}[column sep=huge, row sep=huge]
    A \arrow[d, "\pi"] \arrow[rd, "\tilde{f}"] & \\
    A/B \arrow[r, "f"] & C
\end{tikzcd}
\end{document}
```

当 $\psi\in\widehat{A/B}$, 由于投影映射 $\pi$ 是群同构, 对偶群也是群同构, 所以 $\tilde{\psi}\in\hat{A}$ 也是特征标. 泊松求和公式就是说, 函数 $f\in L(A)$ 定义在 $A$ 上, 那么它在子群上的求和和它在商群上的傅里叶变换求和是有关系的:

$$
\frac{1}{|B|}\sum_{b\in B}f(b)=\sum_{\psi\in\widehat{A/B}}\hat{f}(\tilde{\psi}).
$$

构建函数 $f^\sharp(a)=\displaystyle\sum_{b\in B}f(a+b)$, 观察它的傅里叶系数:

$$
\begin{aligned}
\widehat{f^\sharp}(\chi)&=\frac{1}{|A|}\sum_{a\in A}f^\sharp(a)\overline{\chi(a)}\\\\
&=\sum_{b\in B}\frac{1}{|A|}\sum_{a\in A}f(a+b)\overline{\chi(a+b)}\chi(b)\\\\
&=\hat{f}(\chi)\sum_{b\in B}\chi(b)=|B|\hat{f}(\chi)\left<1,\chi\right>_B\\\\
&=\begin{cases}
|B|\hat{f}(\chi) & \text{if }\chi\vert_B=1,\\\\
0 & \text{otherwise}.
\end{cases}
\end{aligned}
$$

那么做傅里叶反演: $f^\sharp(a)=|B|\displaystyle\sum_{\psi\in\widehat{A/B}}\hat{f}(\tilde{\psi})\tilde{\psi}(a)$.

设 $S$ 是 $B$ 陪集的代表元, 那么有:

$$
\begin{aligned}
\frac{1}{|A|}\sum_{c\in S}|B|\cdot\left\vert\sum_{b\in B}f(c+b)\right\vert^2&=\frac{1}{|A|}\sum_{c\in S}\sum_{b\in B}\left\vert f^\sharp(c+b)\right\vert^2\\\\
&=\frac{1}{|A|}\sum_{a\in A}\left\vert f^\sharp(a)\right\vert^2\stackrel{Plancherel}{=}\sum_{\chi\in\hat{A}}\left\vert\widehat{f^\sharp}(\chi)\right\vert^2\\\\
&=|B|^2\sum_{\psi\in\widehat{A/B}}\left\vert\hat{f}(\tilde{\psi})\right\vert^2.
\end{aligned}
$$

下面介绍了一下不确定性原理 Uncertainty Principle. 见 <a href="#Uncertainty_Principle">Discrete Uncertainty Principle</a>.

---

### 第四节课

这节课首先吴老师先回顾了一下关于 $L^p$ 空间的一些性质.

首先是 H&ouml;lder 不等式, 两个正数 $s,t$ 满足 $\displaystyle\frac{1}{s}+\displaystyle\frac{1}{t}=1$, 且函数 $f\in L^s(I), g\in L^t(I)$, 那么不等式成立:

$$
\left\vert\int_{I}f(x)g(x)dx\right\vert\leq\|f\|\_{s}\|g\|\_{t}.
$$

通过这个可以推出 $L_p$ 空间的包含关系: 即 $1\leq p<q<\infty$, 那么 $h\in L^q(I)\Rightarrow h\in L^p(I)$. 这是因为 $\|h\|\_{p}\leq |I|^{\frac{1}{p}-\frac{1}{q}}\|h\|\_{q}$.

另一个不等式是 Young's Convolution 不等式: $f\in L^p(\mathbb{R}^n), g\in L^q(\mathbb{R}^n)$, 那么 $f*g\in L^r(\mathbb{R}^n)$, 需要满足关系 $\displaystyle\frac{1}{r}=\frac{1}{p}+\frac{1}{q}-1$. 这说明可以把 $L^1(I)$ 看成一个代数结构, 它关于卷积是封闭的.

考虑一维的 torus $\mathbb{T}=\mathbb{R}/2\pi\mathbb{Z}$, 它的对偶群是 $\mathbb{T}^\*=\mathbb{Z}$. 函数 $f$ 的傅里叶级数部分和:

$$
\begin{aligned}
S_N(f)&=\sum_{n=-N}^N\hat{f}(n)e^{in\theta}\\\\
&=\sum_{n=-N}^N\left(\frac{1}{2\pi}\int_{-\pi}^\pi f(t)e^{-int}dt\right)e^{in\theta}\\\\
&=\frac{1}{2\pi}\int_{-\pi}^\pi f(t)\left(\sum_{n=-N}^Ne^{in(\theta-t)}\right)dt\\\\
&=\frac{1}{2\pi}\int_{-\pi}^\pi f(t)D_N(\theta-t)dt=f\*D_N(\theta),
\end{aligned}
$$

这个叫做 Dirichlet Kernel. 我们要问 $(D_N\*f)(\theta)$ 是否等于 $f(\theta)$, 我们可以假定 $\delta_0\*f=f$. 下面给出 good kernels 的定义, 用来逼近理想中的 $\delta_0$: 有一族 kernels $\\{K_n(x)\\}^\infty_{n=1}$ 定义在 $\mathbb{T}$ 上, 且满足以下性质:

1. 归一化: $\forall n, \mathbb{E}K_n=1$.
2. 有界性: 存在常数 $M>0$, 使得 $\forall n, \displaystyle\int_{-\pi}^\pi |K_n(x)|dx<M$.
3. 集中性: 对于任意 $\delta>0$, 有 $\displaystyle\lim_{n\to\infty}\displaystyle\int_{\delta\leq|x|\leq\pi}|K_n(x)|dx\to0$.

一个重要的定理是: 函数 $f:\mathbb{T}\mapsto\mathbb{C}$ 可积且有界, 那么 $\displaystyle\lim_{n\to\infty}(K_n\*f)(\theta)=f(\theta)$, 其中 $f$ 在 $\theta$ 处连续. 更进一步, 如果 $f$ 处处连续, 那么是一致收敛的.

而 Dirichlet Kernel 并不是一个 good kernel:

$$
\begin{aligned}
D_N(\theta)&=\sum_{n=-N}^N\omega^n=\omega^{-N}\frac{1-\omega^{2N+1}}{1-\omega}\\\\
&=\frac{\omega^{-(N+1/2)}-\omega^{N+1/2}}{\omega^{-1/2}-\omega^{1/2}}=\frac{\sin(\frac{2N+1}{2}\theta)}{\sin(\frac{\theta}{2})}\leq 2N+1.
\end{aligned}
$$

Dirichlet Kernel 不满足要求(2). 它的积分结果 $\displaystyle\int_{-\pi}^\pi |D_N(x)|dx\sim O(\log N)$. 当然这并非说明 Dirichlet Kernel 一定就不收敛了, 只是不能用这个方法来说明它的收敛性.

下面要介绍 Fej&eacute;r Kernel: $\displaystyle\frac{S_0(f)+\dots+S_{N-1}(f)}{N}=\sigma_N(f)=f\*F_N$.

由于卷积的线性结合性: $F_N=\displaystyle\frac{D_0+\dots+D_{N-1}}{N}$.

它满足要求(1)很容易验证. 可以先计算出来 $F_N(\theta)=\displaystyle\frac{(\sin\frac{N\theta}{2}/\sin\frac{\theta}{2}^2)}{N}$, 要求(2)只需要先验证 $F_N\geq0$ 恒成立, 要求(3)通过放缩即可.

通过可微性, 可以得到 Dirichlet Kernel 的收敛性定理:

$$
\begin{aligned}
S_N(f)(\theta_0)-f(\theta_0)&=\frac{1}{2\pi}\int_{-\pi}^\pi \left(f(\theta_0-t)-f(\theta_0)\right)D_N(t)dt\\\\
&=\frac{1}{2\pi}\int_{-\pi}^\pi tF(t)D_N(t)dt,
\end{aligned}
$$

其中定义 $F(t)=\begin{cases}\displaystyle\frac{f(\theta_0-t)-f(\theta_0)}{t}&t\neq0\\\\-f'(\theta_0)&t=0\end{cases}$, 可微性是为了让 $F(t)$ 在 $t=0$ 处有定义, 该条件可以弱化为 Lipschitz 条件. 根据 Reimann-Lebesgue 引理, 对于 $f\in L^1$, 那么 $\hat{f}(n)\to0$. 现在我们要把上面的式子拆成两个在 $L^1$ 内的函数:

$$
g_1=\frac{t\cdot F(t)\cdot\cos\frac{t}{2}}{\sin\frac{t}{2}},\quad g_2=t\cdot F(t),
$$

并且:

$$
\begin{aligned}
tD_N(t)&=t\cdot\frac{\sin((N+\frac{1}{2})t)}{\sin\frac{t}{2}}\\\\
&=\frac{t}{\sin\frac{t}{2}}\left[\sin(Nt)\cos\frac{t}{2}+\cos(Nt)\sin\frac{t}{2}\right]
\end{aligned}
$$

所以原式等于:

$$
=\frac{1}{2\pi}\int_{-\pi}^\pi [g_1(t) \sin(Nt)+g_2(t)\cos(Nt)]dt
$$

可以通过 $\widehat{g_1}$ 和 $\widehat{g_2}$ 组合得到, 所以在 $\theta_0$ 处收敛.

---

### 第五节课

今天上课首先补全了上周的 Reimann-Lebesgue 引理的证明. 命题是: 对于一个可积函数 $f\in L^1(\mathbb{T})$, 有 $\displaystyle\lim_{n\to\infty}\hat{f}(n)=0$. 相比于连续函数, 可以用一致连续来证明, 但是对于可积函数, 我们需要使用关于积分的性质.

我们要利用 $L^1(\mathbb{T})$ 中 step function 是稠密的, 然后我们用它作为简单函数去逼近任意的函数, 从而证明命题.

step function 是 indicator function 的有限线性组合, 即 $I=[a,b)\subseteq[-\pi,\pi)$, 导出函数 $\chi\_I$. 我们首先验证 $\chi\_I$ 满足该引理:

$$
\begin{aligned}
\left\|\widehat{\chi\_I}(n)\right\|&=\left\|\frac{1}{2\pi}\int_{-\pi}^\pi\chi\_I(t)e^{-int}dt\right\|\\\\
&=\frac{1}{2\pi}\left\|\int_a^b e^{-int}dt\right\|=\frac{1}{2\pi|n|}\left\|e^{-inb}-e^{-ina}\right\|\leq\frac{1}{\pi|n|},
\end{aligned}
$$

显然成立. 接下来用它来逼近给定的函数 $f=f_m+\varepsilon\_m$, 其中 $f_m$ 是 step function, $\varepsilon\_m$ 是残差项, 满足 $\displaystyle\int_{-\pi}^\pi \|\varepsilon\_m(x)\|dx<\displaystyle\frac{1}{m}$, 从而经过简单的方所可以得到 $f$ 的傅里叶系数是收敛至零的.

作为这个引理的应用, 我们可以证明 $(L^1(T), \*)$ 这个代数结构上没有单位元 ($\*$ 是卷积符号).

下面话题进入了 Possion Kernel, $\left\\{P_r(\theta)=\displaystyle\sum_{r\in\mathbb{Z}}r^{\|n\|}e^{in\theta}\right\\}\_{r\to1^-}$, 我们要说明 Possion Kernel 是一族 good kernels.

1. 归一化容易验证.
2. 有界性: 换元 $\omega=re^{i\theta}$, 那么 $P_r(\theta)=\displaystyle\sum_{n=0}^\infty\omega^n+\displaystyle\sum_{n=1}^\infty\overline{\omega}^n=\displaystyle\frac{1-r^2}{1-2r\cos\theta+r^2}>0$.
3. 在 $\delta\leq\|\theta\|\leq\pi$ 且 $r\to1^-$ 时, $P_r(\theta)$ 可以任意接近零.

下面话题转入了另一种关于积分的收敛性: 对于 $f\in L^1(\mathbb{T})$, $\displaystyle\lim_{N\to\infty}\displaystyle\int_\alpha^\beta S_N(f)(\gamma)d\gamma=\displaystyle\int_\alpha^\beta f(\gamma)d\gamma$.

构造函数 $g(\gamma)=\displaystyle\frac{1}{2\pi}\displaystyle\int_0^\gamma\left(f(\lambda)-\hat{f}[0]\right)d\lambda$, 我们有 $0=\displaystyle\int_0^{2\pi}\left(e^{-in\gamma}g(\gamma)\right)\'d\gamma$, 这是因为 $0=g(0)=e^{-2\pi in}g(2\pi)$. 展开得到

$$
0=\frac{1}{2\pi}\int_0^{2\pi}e^{-in\gamma}(f(\gamma)-\hat{f}[0])d\gamma-in\int_0^{2\pi}e^{-in\gamma}g(\gamma)d\gamma,
$$

所以我们有, 对于 $\underline{n\in\mathbb{Z}\backslash\\{0\\}, \hat{f}[n]=2\pi in\hat{g}[n]}$. 这里其实是应用了等式 $\widehat{f\'}[n]=in\hat{f}[n]$, 因为 $g\'=\displaystyle\frac{1}{2\pi}f\'$.

因为 $g(\gamma)$ 处处可导, 应用 Dirichlet Kernel 的收敛性定理, 我们有:

$$
\frac{1}{2\pi}\int_0^\gamma f(\lambda)d\lambda-\frac{\gamma}{2\pi}\hat{f}[0]=g(\gamma)=\hat{g}[0]+\sum_{n\in\mathbb{Z}\backslash\\{0\\}}\hat{g}[n]e^{in\gamma},
$$

取 $\gamma=0$, 得到 $\underline{-\hat{g}[0]=\displaystyle\sum_{n\in\mathbb{Z}\backslash\\{0\\}}\displaystyle\frac{\hat{f}[n]}{2\pi in}}$, 这个结论与 $f$ 的具体形式无关.

将 $g(\gamma)$ 展开并移项可以得到: $\displaystyle\frac{1}{2\pi}\displaystyle\int_0^\gamma f(\lambda)d\lambda = \hat{g}[0]+\sum_{n\in\mathbb{Z}\backslash\\{0\\}}\hat{g}[n]e^{in\gamma}+\displaystyle\frac{\gamma}{2\pi}\hat{f}[0]$. 对于级数的部分和, 我们有:

$$
\begin{aligned}
\frac{1}{2\pi}\int_0^\gamma S_N(f)(\lambda)d\lambda&=\frac{1}{2\pi}\sum_{n=-N}^N\hat{f}[n]\int_0^\gamma e^{in\lambda}d\lambda\\\\
&=\frac{\gamma}{2\pi}\hat{f}[0]+\sum_{|n|<N,n\neq0}\hat{f}[n]\frac{e^{in\gamma}-1}{2\pi in}\\\\
&\to\frac{\gamma}{2\pi}\hat{f}[0]+\sum_{n\in\mathbb{Z}\backslash\\{0\\}}\frac{\hat{f}[n] e^{in\gamma}}{2\pi in} + \hat{g}[0]\\\\
&=\frac{1}{2\pi}\int_0^\gamma f(\lambda)d\lambda.
\end{aligned}
$$

下面我们的话题来到了等周不等式, 对于一个 arc-length parametrized 曲线 $\gamma:[0,2pi]\mapsto\mathbb{R}^2$, $\gamma(s)=(x(s), y(s))$, 满足 $\gamma(0)=\gamma(2\pi)$, 它的长度 $l=\displaystyle\int_0^{2\pi}\|x\'(s)\|ds=2\pi$.

定义曲线围出的面积为 $Area=\displaystyle\frac{1}{2}\left\|\int_0^{2\pi}x(s)y\'(s)-x\'(s)y(s)ds\right\|$, 我们要找到它的最大值.

由于曲线是连续的, 对 $x(s)$ 和 $y(s)$ 分别做傅里叶变换: $x(s)=\displaystyle\sum_na_ne^{ins}$, $y(s)=\displaystyle\sum_nb_ne^{ins}$.

利用 $x\'(s)\sim\displaystyle\sum_na_iine^{ins}$, $y\'(s)\sim\displaystyle\sum_nb_iine^{ins}$, 展开等式 $1=\displaystyle\frac{1}{2\pi}\displaystyle\int_0^{2\pi}x\'(s)^2+y\'(s)^2ds$:

$$
1=\displaystyle\sum_n (|a_n|^2+|b_n|^2)|n|^2.
$$

而面积 $Area=\displaystyle\frac{1}{2}\left\|\int_0^{2\pi}x(s)y\'(s)-x\'(s)y(s)ds\right\|=\pi\left\|\displaystyle\sum_{n}n(a_nb_{-n}-a_{-n}b_n)\right\|=\pi\left\|\displaystyle\sum_{n}n(a_n\overline{b_n}-\overline{a_n}b_n)\right\|$.

而 $|n|\leq n^2$, $|a_n\overline{b_n}-\overline{a_n}b_n|\leq 2|a_n||b_n|\leq |a_n|^2+|b_n|^2$. 所以有 $Area\leq \pi$. 取等条件也容易验证, 就是圆形.

---

### 第六节课

关于等周不等式的新的证明: 上节课我们是考虑从 x 轴, y 轴来参数化这个区域或曲线, 新的证明方法是用 suopport function 来参数化凸区域. Support function $h_K(\theta)$ 是指一个凸区域 $K$ 的支撑线到原点的有向距离. 通过 support function 可以把周长和面积写做: $P(K)=\displaystyle\int_0^{2\pi} h_K(\theta)d\theta$, $A(K)=\displaystyle\int_0^{2\pi}(h_K(\theta)^2+h'_K(\theta)^2)d\theta$.

为了方便, 将 $h_K(\theta)$ 展开成三角级数:
\begin{align*}
h_K(\theta)&\sim a_0+\displaystyle\sum_{k=1}^\infty a_k\cos k\theta+b_k\sin k\theta,\\\\
h'\_K(\theta)&\sim \displaystyle\sum_{k=1}^\infty -k a_k\sin k\theta + k b_k\cos k\theta.
\end{align*}

那么 $P(K)=2\pi a_0$, $A(K)=\pi a_0^2-\displaystyle\frac{\pi}{2}\displaystyle\sum_{k=1}^\infty(k^2-1)(a_k^2+b_k^2)$. 所以 $P(K)^2-4\pi A(K)\geq0$.

这个证明可以很快得到稳定性, 定义距离函数 $\delta(M,N)=\left(\displaystyle\int_0^{2\pi}(h_M-h_N)^2d\theta\right)^{1/2}$. 那么距离一个凸区域 $K$ 的最近的圆形区域 $B(K)$ 只需要截断高阶项, 而且残差可以得到估计: $P(K)^2-4\pi A(K)\geq6\pi\cdot\delta(K,B(K))^2$. 更多关于稳定性的内容可以参考论文: <a href="#ref4">A Stability Result for the Isoperimetric Inequality</a>.

下面介绍了 Wiener 代数. 定义 $A(\mathbb{T})$ 是定义在 $\mathbb{T}=\mathbb{R}/2\pi\mathbb{Z}$ 上的连续函数构成的集合, 满足其傅里叶级数绝对收敛. 首先我们要在 $A(\mathbb{T})$ 上给一个范数, 让它成为 Banach 代数: $f\in A(\mathbb{T}),||f||\_A=\displaystyle\sum\_{n\in\mathbb{Z}}\|\hat{f}(n)\|$. 下面要验证这个范数满足 Banach 代数的性质: $||f\cdot g||\_A\leq||f||\_A\cdot||g||\_A$. 如果满足的话, 则可以说明 $A(\mathbb{T})$ 关于逐点乘是封闭的, 从而是一个代数结构.

证明展开很像卷积的性质:
\begin{align*}
f(\theta)g(\theta)&=\sum_{m\in\mathbb{Z}}\hat{f}(m)e^{im\theta} \cdot \sum_{n\in\mathbb{Z}}\hat{g}(n)e^{in\theta}\\\\
&=\sum_{n,m\in\mathbb{Z}}\hat{f}(m)\hat{g}(n)e^{i(m+n)\theta}\\\\
&=\sum_{n\in\mathbb{Z}}e^{in\theta}\sum_{m\in\mathbb{Z}}\hat{f}(m)\hat{g}(n-m),
\end{align*}

因此:
\begin{align*}
||f\cdot g||\_A&=\sum_{n\in\mathbb{Z}}\left\|\sum_{m\in\mathbb{Z}}\hat{f}(m)\hat{g}(n-m)\right\|\\\\
&\leq\sum_{n\in\mathbb{Z}}\sum_{m\in\mathbb{Z}}\|\hat{f}(m)\|\cdot\|\hat{g}(n-m)\|\\\\
&=\left(\sum_{m\in\mathbb{Z}}\|\hat{f}(m)\|\right)\cdot\left(\sum_{k\in\mathbb{Z}}\|\hat{g}(k)\|\right)=||f||\_A\cdot||g||\_A.
\end{align*}

下面引入一个引理: 假设 $f\in L^1(\mathbb{T})$, 其导函数 $f'$ 有界, 那么 $||f||\_\infty\leq||f||\_A\leq||f||\_\infty+2||f'||\_\infty$.

左边的不等式显然,

\begin{align*}
||f(\theta)||\_\infty=|\sum_{n\in\mathbb{Z}}\hat{f}(n)e^{in\theta}|\leq \sum_{n\in\mathbb{Z}}|\hat{f}(n)|=||f||\_A.
\end{align*}

对于右边的不等式, 我们需要对傅里叶级数的每一项做不等式, 首先是第零项, $|\hat{f}(0)|\leq ||f||_\infty$ 显然成立. 我们只需要证明 $\displaystyle\sum\_{n\neq0}|\hat{f}(n)|\leq 2||f'||\_\infty$. 使用 Cauchy-Schwarz 不等式:

\begin{align*}
\sum_{n\neq0}|\hat{f}(n)|\leq\left(\sum_{n\neq0}\frac{1}{n^2}\right)^{1/2}\cdot\left(\sum_{n\neq0}|\hat{f}(n)|^2 n^2\right)^{1/2}.
\end{align*}

Basel 等式告诉我们右边不等于正无穷: $\displaystyle\sum_{n=1}^\infty\displaystyle\frac{1}{n^2}=\displaystyle\frac{\pi^2}{6}$. 

> 考虑锯齿函数 $f(x)=x$, 其中 $x\in[-\pi,\pi)$. 当 $n\neq0$ 时, $\hat{f}(n)=\displaystyle\frac{(-1)^n i}{n}$. 利用 Parseval 等式, 有:
> \begin{align*}
\frac{\pi^2}{3}=\frac{1}{\pi}\int_0^\pi x^2 dx&=\frac{1}{2\pi}\int_{-\pi}^\pi |f(x)|^2dx\\\\
&=\|\hat{f}\|\_2^2=\sum_{n=-\infty}^\infty ||\hat{f}(n)||^2=2\sum_{n=1}^\infty \frac{1}{n^2}\\\\
\end{align*}

再使用 $\widehat{f'}(n)=in\hat{f}(n)$. 所以 $\displaystyle\sum_{n\neq0}|\hat{f}(n)|\leq \sqrt{\displaystyle\frac{\pi^2}{3}}\left(\displaystyle\sum_{n\in\mathbb{Z}}|\widehat{f'}(n)|^2\right)^{1/2}$, 再使用 Parseval 等式, 有:
\begin{align*}
\sum_{n\neq0}|\hat{f}(n)|\leq \sqrt{\frac{\pi^2}{3}}\left(\frac{1}{2\pi}\int_{-\pi}^\pi |f'(\theta)|^2 d\theta\right)^{1/2}\leq 2||f'||_\infty.
\end{align*}

通过这个引理, 我们可以证明 Wieber Inversion Theorem: 对于 $f\in A(\mathbb{T})$, 且 $f$ 处处不等于零, 那么 $\displaystyle\frac{1}{f}\in A(\mathbb{T})$. 不妨假设 $\displaystyle\min_{\theta}|f(\theta)|=1$. 我们可以取一个逼近的足够好的部分和, 取 $N$, 使得 $||S_N-f||\_A\leq \displaystyle\frac{1}{3}$, 那么 $|S_N(\theta)|\geq\displaystyle\frac{2}{3}$. 那么

\begin{align*}
\frac{1}{f(\theta)}&=\frac{1}{S_N(\theta)-(S_N(\theta)-f(\theta))}\\\\
&=\frac{1}{S_N(\theta)}\cdot\frac{1}{1-\frac{S_N(\theta)-f(\theta)}{S_N(\theta)}}\\\\
&=\sum_{n=0}^\infty\frac{(S_N(\theta)-f(\theta))^{n}}{S_N(\theta)^{n+1}}.
\end{align*}

那么根据三角不等式:
\begin{align*}
||\frac{1}{f}||\_A&\leq\sum_{n=0}^\infty||\frac{(S_N-f)^{n}}{S_N^{n+1}}||\_A\\\\
&\leq\sum_{n=0}^\infty ||S_N-f||\_A^{n}\cdot||\frac{1}{S_N^{n+1}}||\_A\\\\
&\leq\sum_{n=0}^\infty (\frac{1}{3})^n\cdot ||\frac{1}{S_N^{n+1}}||\_A
\end{align*}

我们把 $\displaystyle\frac{1}{S_N^{n+1}}$ 视为引理中的 $f$, 则 $||(\displaystyle\frac{1}{S_N^{n+1}})'||\_\infty=||\displaystyle\frac{-(n+1)S_N'}{S_N^{n+2}}||\_\infty\leq C\cdot (n+1)\cdot (\frac{3}{2})^{n+2}$, 其中 $C=||S_N'||\_\infty$. 带入得到: $||\displaystyle\frac{1}{f}||\_A\leq\displaystyle\sum_{n=0}^\infty 3\cdot (\frac{1}{2})^{n+1}+2\cdot C\cdot (n+1)\cdot 9 \cdot (\frac{1}{2})^{n+2}<+\infty$. 因此 $\displaystyle\frac{1}{f}\in A(\mathbb{T})$.

下面约定 $\mathbb{T}=\mathbb{R}/\mathbb{Z}$, 一个实值偶函数 $K\in C(\mathbb{T})$ 是正定的, 是指对于任意函数 $f\in L^1(\mathbb{T})$, 都有 $\displaystyle\int_0^1\displaystyle\int_0^1f(x)K(x-y)\overline{f(y)}dxdy\geq0$.

---

### 教材阅读

#### 周期函数的傅里叶级数的收敛性:

##### 连续性与收敛性

{% note info %}
Theorem 1.1:

取二阶连续可导函数, $f\in C^2(\mathbb{T})$, 其中 $\mathbb{T}$ 是单位圆, 那么对于任意 $\theta\in\mathbb{T}$, 部分和的极限
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

假定我们先承认 Fej&eacute;r 定理, 即连续函数的部分和的 Ces&agrave;ro 一致收敛于原函数, 那么我们可以得到:

{% note info %}
<span id="theo1.2"></span>Theorem 1.2:

$f\in C^2(\mathbb{T})$, 那么 $S_N(f)$ 一致收敛于 $f$.
{% endnote %}

{% note primary %}
对于一个收敛的数列 $\\{a_n\\}$, 有:
$$
\lim_{n\to\infty}(a_1+a_2+\dots+a_n)/n=\lim_{n\to\infty}a_n.
$$
{% endnote %}

更强的结论是:

{% note info %}
<span id="theo1.2.2"></span>Theorem 1.2.2:

$f\in C^1(\mathbb{T})$, 那么 $S_N(f)$ 一致收敛于 $f$.
{% endnote %}

##### 衰减速度与光滑性

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
<span id="theo1.3"></span>Theorem 1.3:

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

##### 收敛速度与光滑性

根据上面的内容, 我们可以得到, 函数越光滑, 其傅里叶系数衰减越快, 那能否得到傅里叶级数收敛越快呢? 下面的定理给出了肯定的答复.

{% note info %}
Theorem 1.4:

对于一个连续可微的函数 $f\in C^k(\mathbb{T})$, 其中 $k\geq2$, 那傅里叶级数的收敛速度是有保障的, 且不依赖于 $\theta$ :
$$
|S_N(f)(\theta) - f(\theta)|\leq C/N^{k-1}.
\tag{1.2}
$$

如果 $k=1$, 既 $f\in C^1(\mathbb{T})$, 那么也是能保证的:
$$
|S_N(f)(\theta) - f(\theta)|\leq C/\sqrt{N}.
$$
{% endnote %}

证明略. 下面给出一个逆定理的反例, 如果函数的部分和收敛速度符合式子 <a href="#1.2">(1.2)</a>, 那么函数不一定在 $C^k(\mathbb{T})$ 里, 甚至不在 $C^{k-1}(\mathbb{T})$ 里.

函数 $f:\mathbb{T}\mapsto\mathbb{R}$ 如此设计: $f(\theta)=\displaystyle\frac{\pi}{2}-|\theta|$, 这个函数连续但是在 $\theta\in \pi\mathbb{Z}$ 处不可导. 所以不属于 $C^2$ (甚至不属于 $C^1$). 它的傅里叶系数如下:
$$
\hat{f}(n)=\begin{cases}
0,&\text{if }n\text{ is even;}\\\\
2/(\pi n^2),&\text{if }n\text{ is odd.}
\end{cases}
$$

所以 $\displaystyle\sum_{n\in\mathbb{Z}}|\hat{f}(n)|<\infty$, 所以根据<a href="#theo1.3">定理 1.3</a>, $S_N(f)$ 一致收敛到 $f$, 对于残差有以下估计:
$$
|S_N(f)(\theta)-f(\theta)|\leq\frac{2\pi^{-1}}{N-1}.
$$

这构成了逆定理的一个反例.

##### 总结与其他

根据<a href="#theo1.2">定理 1.2</a> 和 <a href="#theo1.2.2">定理 1.2.2</a> 的内容, 我们可以知道, 周期函数的傅里叶级数的收敛性要求略强于连续, 但是弱于可微. 下面这个定理证明了这一点.

{% note info %}
Theorem by Du Bois-Reymond, 1873:

存在一个连续函数 $f:\mathbb{T}\mapsto\mathbb{R}$, 使得傅里叶级数的部分和在 $x=0$ 处不收敛.
$$
\lim\sup_{N\to\infty} |S_N(f)(0)| = \infty.
$$
{% endnote %}

跟进一步, 数学家证明了存在一个勒贝格可积的函数处处不收敛:

{% note info %}
Theorem by Kolmogorov, 1926:

存在一个可积函数 $f:\mathbb{T}\mapsto\mathbb{C}$, 使得傅里叶级数的部分和处处不收敛.
$$
\lim\sup_{N\to\infty} |S_N(f)(\theta)| = \infty,\quad\forall\theta\in\mathbb{T}.
$$
{% endnote %}

尽管他的构造是勒贝格可积的函数, 但是并不是连续函数, 实际上在任何一个区间内该函数都是无界的. 人们觉得这个结果离构造出连续函数的例子已经不远了, 直到半个世纪后, Carleson 于1966年证明了对于连续函数( 更强地, 是黎曼可积的函数 )或者平方可积的函数, 傅里叶级数几乎处处收敛. 而 Kolmogorov 的例子并不是平方可积的, 也不是黎曼可积的. 收敛性问题就此完结.

#### 傅里叶变换的天堂

在非周期条件下的函数, 也可以定义傅里叶变换和傅里叶逆变换为:
$$
\hat{f}(\xi)=\int_{\mathbb{R}}f(x)e^{-2\pi i\xi x}dx,\quad (g)^\vee(x)=\int_{\mathbb{R}}g(\xi)e^{2\pi i\xi x}d\xi.
$$

那么傅里叶反演定理就是问, 在什么条件下, $(\hat{f})^\vee=f$ 成立.

我们可以逐步扩大积分范围, 先仅考虑函数 $f$ 在 $[-L/2,L/2)$ 上的情况, 假设它是以 $L$ 为周期的函数, 假设 $f$ 的性质足够好, 那么展开的傅里叶级数收敛. $f(x)=\displaystyle\sum_{n\in\mathbb{Z}}a_L(n)e^{2\pi inx/L}$, 这里系数为 $a_L(n)=\displaystyle\frac{1}{L}\int_{-L/2}^{L/2}f(y)e^{-2\pi iny/L}dy$. 可以做换元 $\xi_n=n/L$, 且 $\Delta\xi=1/L$, 那么可以重写为 $f(x)=\displaystyle\sum_{n\in\mathbb{Z}}F_L(\xi_n)\Delta\xi$, 其中 $F_L(\xi)=e^{2\pi i\xi x}\displaystyle\int_{-L/2}^{L/2}f(y)e^{-2\pi i\xi y}dy$.

这个表达式像极了黎曼和, 但是这个函数内部和区间划分都和 $1/L$ 有关. 如果函数 $f$ 是紧支撑的, 即存在数 $M$, 在区间 $[-M, M]$ 外函数的取值都是零. 那么在充分大的情况下, 函数 $F_L$ 就与 $L$ 无关了. 那么我们也可以期待求和式也会收敛成积分式. 为了更加严谨, 我们需要给出 $f$ 性质"良好"的定义.

##### Schwartz Class

我们给出 Schwartz Class 的定义, 并在后文给出傅里叶反演定理在 Schwartz Class 上成立的证明. Schwartz Class $\mathcal{S}(\mathbb{R})\subset C^{\infty}(\mathbb{R})$ 包含了各阶导函数都快速衰减的光滑函数, 即对于任意的整数 $k,l\geq0$, 都有 $\displaystyle\lim_{|x|\to\infty}|x|^k|f^{(l)}(x)|=0$. 容易验证, 这个条件与下面这个条件等价: $\displaystyle\sup_{x\in\mathbb{R}}|x|^k|f^{(l)}(x)|<\infty$. 这个函数空间对于很多操作都是封闭的, 例如乘法和求导, 以及和多项式函数的乘法, 以及卷积. 因为衰减的很快, 所以总是可积的.

值得注意的是, 如果仅仅是函数 $f$ 快于任何多项式函数衰减, 其本身可能并不属于 Schwartz Class, 例如 $f(x)=e^{-x}\cdot e^{-ie^{2x}}$, 其导函数并不收敛. 下面是一些属于 Schwartz Class 的函数:

一个是存在紧支撑的函数: $B(x)=\begin{cases}e^{-1/(x-a)}e^{-1/(b-x)},&x\in[a,b];\\\\0,&\text{otherwise}.\end{cases}$, 其光滑性容易验证. 另一个非常重要的例子是高斯函数 $G(x)=e^{-\pi x^2}$, 它的傅里叶变换是其本身: $\hat{G}(\xi)=G(\xi)=e^{-\pi \xi^2}$.

#### 天堂之外

这一章将放松对函数必须是 Schwartz Class 的要求, 讨论快速衰减的连续函数, 以及用分布的角度去看更加一般的函数.

##### 快速衰减的连续函数

定义: 连续函数 $f$, 存在常数 $A,\varepsilon>0$, 使得对于任意 $x\in\mathbb{R}$, 有 $|f(x)|\leq \displaystyle\frac{A}{1+|x|^{1+\varepsilon}}$. 分母即要求在无穷远处衰减得足够快, 所以可积, 又要求在零附近有界. 这些函数一定属于 $L^p(\mathbb{R})$, 因为当 $p=2$ 时,
$$
\Vert f\Vert_2^2=\int_{\mathbb{R}}|f(x)|^2dx\leq \left(\sup_{x\in\mathbb{R}}|f(x)|\right)\int_{\mathbb{R}}|f(x)|dx=\Vert f\Vert_\infty\Vert f\Vert_1.
$$

$p$ 更高时同理.

##### 缓增分布

我不知道这个 (Tempered Distribution) 的中文名是什么, 只能机翻了, 后面都用分布来替代.

分布 $T:\mathcal{S}(\mathbb{R})\mapsto\mathbb{C}$ 是 Schwartz Class 上的一个连续线性泛函, 这个线性泛函构成的空间记为 $\mathcal{S}'(\mathbb{R})$, 这构成了 Schwartz 空间的对偶空间. 两个泛函在分布意义下相同是指对于所有的 $\phi\in\mathcal{S}(\mathbb{R})$, 有 $T(\phi)=U(\phi)$. 一列泛函在分布意义下收敛是指 $\displaystyle\lim_{n\to\infty}T_n(\phi)=T(\phi)$.

泛函的连续性需要更加详细的说明, 首先要定义 $\mathcal{S}(\mathbb{R})$ 上的收敛, 对于函数 $\phi\in\mathcal{S}(\mathbb{R})$, 对于任意自然数 $k,l$, $\rho_{k,l}(\phi)=\displaystyle\sup_{x\in\mathbb{R}}|x|^k|\phi^{(l)}(x)|$ 都是有限的, $\rho_{k,l}$ 构成了半范数, 它是正的, 齐次的, 满足三角不等式的. 我们用这个定义 $\mathcal{S}(\mathbb{R})$ 上的函数收敛: 一列函数 $\\{\phi_n\\}$ 收敛到 $\phi\in\mathcal{S}(\mathbb{R})$, 当且仅当对于任意的 $k,l\geq0$, 都有 $\displaystyle\lim_{n\to\infty}\rho_{k,l}(\phi_n-\phi)=0$.

从而定义泛函 $T$ 的连续性: 对于任意收敛到 $\phi\in\mathcal{S}(\mathbb{R})$ 的函数列 $\\{\phi_n\\}$, 都有 $\displaystyle\lim_{n\to\infty}T(\phi_n)=T(\phi)$.

一个经典的分布构造是 $T_f(\phi)=\displaystyle\int_{\mathbb{R}}f(x)\phi(x)dx$. 这里只要求 $f$ 不是增长过快的函数即可. 如果 $f$ 是一个有界连续函数, 或者是多项式函数, 那么 $T_f$ 都是连续的, 从而构成一个分布. 并且由 $\varepsilon(f)=T_f$ 给出的映射 $\varepsilon:\mathcal{S}(\mathbb{R})\mapsto\mathcal{S}'(\mathbb{R})$ 是双射.

##### 分布上的时域与频域的关系

以求导举例, 我们可以定义对分布的操作: $T'\_f(\phi)=T_{f\'}(\phi)=\displaystyle\int_{\mathbb{R}}f'(x)\phi(x)dx=-\displaystyle\int_{\mathbb{R}}f(x)\phi'(x)dx=-T_f(\phi')$.

下面举一个更加复杂的例子: $\widehat{\psi*T}(\phi)=\mathcal{M}_{\hat{\psi}}\hat{T}(\phi)$.

首先我们有 $\widehat{\psi\*T}(\phi)=\psi\*T(\hat{\phi})=T(\widetilde{\psi}\*\hat{\phi})$.

$$
\begin{aligned}
\widetilde{\psi}\*\hat{\phi}(\xi)&=\int_{\mathbb{R}}\widetilde{\psi}(y)\hat{\phi}(\xi-y)dy\\\\
&=\int_{\mathbb{R}}\psi(-y)\left(\int_{\mathbb{R}}\phi(t)e^{-2\pi i(\xi-y)t}dt\right)dy\\\\
&=\int_{\mathbb{R}}\left(\int_{\mathbb{R}}\psi(-y)e^{-2\pi i(-y)t}dy\right)\phi(t)e^{-2\pi i\xi t}dt\\\\
&=\int_{\mathbb{R}}\hat{\psi}(t)\phi(t)e^{-2\pi i\xi t}dt=\widehat{\hat{\psi}\phi}(\xi).
\end{aligned}
$$

所以 $\widehat{\psi\*T}(\phi)=T(\widetilde{\psi}\*\hat{\phi})=T(\widehat{\hat{\psi}\phi})=\hat{T}(\hat{\psi}\phi)=\mathcal{M}_{\hat{\psi}}\hat{T}(\phi)$.

##### Delta Distribution

分布里有一些对象是无法用函数导出的, 最经典的例子是 Delta Distribution, 我们考察常函数 $f(x)\equiv1$, 试着计算它在分布意义下的傅里叶变换:

$$
\widehat{T_1}(\phi)=T_1(\hat{\phi})=\int_{\mathbb{R}}\hat{\phi}(\xi)d\xi=\int_{\mathbb{R}}\hat{\phi}(\xi)\cdot e^{2\pi i\xi0} d\xi=(\hat{\phi})^\vee(0)=\phi(0),
$$

其中最后一个等号成立是因为在 Schwartz Class 上傅里叶反演定理成立. 我们可以定义 Delta Distribution 为 $\delta(\phi)=\phi(0)$. 容易证明它满足连续性, 所以确实是一个分布. 我们甚至可以计算它的傅里叶变换 $\hat{\delta}$:

$$
\hat{\delta}(\phi)=\delta(\hat{\phi})=\hat{\phi}(0)=\int_{\mathbb{R}}\phi(x)dx=T_1(\phi).
$$

下面我们来计算一下 $\delta(\phi)$ 的各阶导数以及它们的傅里叶变换. 导数很容易计算 $\delta^{(k)}(\phi)=(-1)^k\delta(\phi^{(k)})=(-1)^k\phi^{(k)}(0)$. 它的傅里叶变化为: $\widehat{\delta^{(k)}}(\phi)=(2\pi i\xi)^k\hat{\delta}(\phi)=(2\pi i\xi)^kT_1(\phi)=T_{(2\pi i\xi)^k}(\phi)$, 所以在分布的意义下可以认为 $\delta^{(k)}$ 等于 $(2\pi i\xi)^k$.

接下来要验证 $\delta$ 本身恰好是 Heaviside 函数的导数 $H'$. 函数 $H$ 定义如下: $H(x)=\begin{cases}1,&x>0;\\\\0,&x\leq0.\end{cases}$, 它的分布意义下的导数为 $T'\_{H}(\phi)=-T_H(\phi')=-\displaystyle\int_0^\infty 1\cdot\phi'(\xi)d\xi=-[\phi(\xi)]^\infty_0=\phi(0)$, 这说明 $\delta=T'\_H$. 类似地, 对于每一个分布, 虽然它本身可能并不是一个函数, 但是它一定是某个函数的若干阶导数.

另一个有趣的性质是 $\delta$ 是卷积的单位元, 即对于任意的 $\psi\in\mathcal{S}(\mathbb{R})$, 都有 $\psi\*\delta=T\_\psi$.

##### Exercise: 离散版本的不确定性原理

<span id="Uncertainty_Principle"></span>Discrete Uncertainty Principle: 在一个有限阿贝尔群 $G$ 上, 实函数 $f\in L^2(G)$, 那么该函数的支撑集和其傅里叶变换的支撑集大小满足以下不等式:

$$
\\#\text{supp}(f)\times\\#\text{supp}(\hat{f})\geq\\#G=n.
$$

首先定义符号函数 $s(a)=\text{sign}(f(a))$, 可以观察每一项傅里叶变换的系数: $\hat{f}(e)=\mathbb{E}\_{g\in G}[f\overline{e}]\leq\mathbb{E}\_{g\in G}[|f|]=\mathbb{E}\_{g\in G}[f\overline{s}]$, 两边同时取最大值:

$$
\begin{aligned}
\max_{e\in\hat{G}}\hat{f}(e) &\leq\mathbb{E}\_{g\in G}[f\overline{s}]=\left<f,s\right>\\\\
&\leq {\left<f,f\right>}^{\frac{1}{2}}{\left<s,s\right>}^{\frac{1}{2}}\\\\
&=|G|^{\frac{1}{2}}{\left<\hat{f},\hat{f}\right>}^{\frac{1}{2}}\sqrt{\frac{1}{|G|}\sum_{g\in G}s(g)^2}\\\\
&=\frac{1}{\sqrt{G}}\sqrt{\sum_{e\in\hat{G}}\hat{f}(e)^2}\sqrt{\sum_{g\in G}s(g)^2}\\\\
&\leq \frac{\max\hat{f}(e)}{\sqrt{G}}\sqrt{\\#\text{supp}(\hat{f})}\sqrt{\\#\text{supp}(f)},
\end{aligned}
$$

这里用到了有限群上的 Parseval 恒等式: $\left<f, f\right>=\displaystyle\sum_{e\in\hat{G}}\left<f, e\right>\hat{f}(e)=|G|\left<\hat{f},\hat{f}\right>$. 两边同时除以 $\displaystyle\max_{e\in\hat{G}}\hat{f}(e)$ 即可.

---

### 推荐阅读:

- <span id="ref1"></span> Pereyra, M. C., & Ward, L. A. (2012). <a href="https://fig.if.usp.br/~marchett/fismat2/harmonic-analysis-fourier-wavelet_pereyra-ward.pdf"> Harmonic analysis: from Fourier to wavelets (Vol. 63).</a>

- <span id="ref2"></span> Carleson, L. (1966). <a href="https://projecteuclid.org/journals/acta-mathematica/volume-116/issue-none/On-convergence-and-growth-of-partial-sums-of-Fourier-series/10.1007/BF02392815.full">On convergence and growth of partial sums of Fourier series.</a>

- Lacey, M., & Thiele, C. (2000). <a href="https://archive.intlpress.com/site/pub/files/_fulltext/journals/mrl/2000/0007/0004/MRL-2000-0007-0004-a001.pdf">A proof of boundedness of the Carleson operator.</a>

- <span id="ref4"></span> Fuglede, Bent. (1986). <a href="https://math.univ-lyon1.fr/~santambrogio/fuglede.pdf">Stability in the isoperimetric problem.</a>

