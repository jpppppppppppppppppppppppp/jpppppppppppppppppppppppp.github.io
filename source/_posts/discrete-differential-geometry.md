---
title: "Discrete Differential Geometry"
date: 2025-10-23 21:20:51
updated: 2025-12-13 19:01:03
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

为了引出曲面的曲率, 现在我们先考虑曲面上曲线的曲率, 可以想象 $M$ 中的一条直线 $X$, 这个时候可以把 $f(X)$ 看做是一条曲线, 它在某点的曲率可以根据 $d{N}=-\kappa T+\tau B$ 来提取出来:
$$
\kappa_n(X)=\frac{df(X)\cdot dN(X)}{|df(X)|^2}.
$$

怎么理解这个式子我其实想了很久, 假设 $s$ 是 $f(X)$ 的弧长参数化, 那么 $T(X)=\displaystyle\frac{df}{ds}=\displaystyle\frac{df(X)}{ds/dt}=\displaystyle\frac{df(X)}{|df(X)|}$, 同样的 $\displaystyle\frac{dN(s)}{ds}=\displaystyle\frac{dN(X)}{ds/dt}=\displaystyle\frac{dN(X)}{|df(X)|}$. 这样可以理解为什么分布会带有平方, 因为 $dN(X)$ 也会收到 $|f(X)|$ 的拉伸而放缩.

再观察公式 $dN=-\kappa T+\tau B$, 可以发现 $dN$ 一定落在曲面的切平面内的, 所以我们引入 Shape Operator $S:TM\mapsto TM$, 满足 $df(SX)=dN(X)$. Shape Operator 是一个线性映射, 因为方向导数可以表示成关于基的导数的线性组合.

假设一点 $x\in M$ 的邻域用 $\\{u,v\\}$ 坐标系参数化为 $X(u,v):\mathbb{R}^2\mapsto\mathbb{R}^3$, 那么 $\\{x_u,x_v\\}$ 就是该点切空间 $T_xM$ 的一组基,根据定义: $df(Sx_u)=dN(u)$, $df(Sx_v)=dN(v)$. 由于 $(N,X_u)=(N,X_v)=0$, 分别对 $v,u$ 求导, 得到:

$(dN(v),X_u)+(N,X_{uv})=0$, $(dN(u),X_v)+(N,X_{vu})=0$. 所以 $(dN(v), X_u) = (dN(u), X_v)$ 具有对称性.

因为 Shape Operator 是一个二维的线性映射, 所以它有两个特征值 $\kappa_1,\kappa_2$, 叫做主曲率 (principal curvatures), 即 $SX_i=\kappa_i X_i$.

根据上面的对称等式, 我们有 $(dN(X_1), df(X_2))=(dN(X_2), df(X_1))$. 而$dN(X_i)=df(SX_i)=\kappa_1df(X_i)$, 所以 $\kappa_1(df(X_1),df(X_2))=\kappa_2(df(X_1),df(X_2))$. 只要 $\kappa_1\neq\kappa_2$, 就有 $(df(X_1),df(X_2))=0$, 也就是说主曲率对应的切向量是垂直的.

所以对于一般方向上的曲率, 可以表示成主曲率的线性组合. 假设 $df(Y)=\cos\theta df(X_1)+\sin\theta df(X_2)$, 那么:
$$
\begin{aligned}
\kappa_n(Y)=(df(SY),df(Y))&=(\kappa_1\cos\theta df(X_1)+\kappa_2\sin\theta df(X_2),\cos\theta df(X_1)+\sin\theta df(X_2))\\\\
&=\kappa_1\cos^2\theta+\kappa_2\sin^2\theta.
\end{aligned}
$$

对于曲面的整体曲率, 可以定义平均曲率(mean curvature) $H=\displaystyle\frac{\kappa_1+\kappa_2}{2}$, 高斯曲率(Gaussian curvature) $K=\kappa_1\kappa_2$.

总结一下就是两个对称式, 这分别叫做第一基本形式和第二基本形式:

$$
I(X,Y)=(df(X),df(Y)),\quad II(X,Y)=(dN(X),df(Y)).
$$

对于一个曲面 $f:M\subset\mathbb{R}^2\mapsto\mathbb{R}^3$, 它的切向量 $df(X)=\displaystyle\lim_{h\to0}\displaystyle\frac{f(p+hX)-f(p)}{h}$ 可以写成雅可比矩阵的形式,
$$
J=\begin{bmatrix}
    {\partial f^1}/{\partial x^1} & {\partial f^1}/{\partial x^2} \\\\
    {\partial f^2}/{\partial x^1} & {\partial f^2}/{\partial x^2} \\\\
    {\partial f^3}/{\partial x^1} & {\partial f^3}/{\partial x^2}
\end{bmatrix},
$$

那么 $df(X)=JX=J\begin{bmatrix}X^1&X^2\end{bmatrix}^T$. 我们可以尝试把其他的几何对象也用矩阵表示, 比如第一基本形式: $I(X,Y)=(df(X),df(Y))$. 我们如果用矩阵 $I\in\mathbb{R}^{2\times2}$ 表示, 那么

$$
X^TIY=(JX)^T(JY),\quad I=J^TJ=\begin{bmatrix}
E & F \\\\
F & G
\end{bmatrix}.
$$

我们定义了 Shape Operator $S:TM\mapsto TM$, 和第二基本形式 $II(X,Y)=(dN(X),df(Y))=I(SX,Y)$. 如果我们用矩阵 $S,II\in\mathbb{R}^{2\times2}$ 表示, 那么

$$
II=IS=\begin{bmatrix}
e & f \\\\
f & g
\end{bmatrix}.
$$

其中每个元素可以写成: $II_{i,j}=(dN(X_i),df(X_j))=-(N, f_{ji})$. 最后我们可以通过矩阵来验证通过微分得到的结论, 比如 $\kappa_n(X)=\displaystyle\frac{II(X, X)}{I(X,X)}$, 我们可以重写为:
$$
\frac{X^TIIX}{X^TIX}=\frac{X^TISX}{X^TIX}=\frac{(JX)^T(JSX)}{(JX)^T(JX)}=\frac{df(X)\cdot dN(X)}{|df(X)|^2}.
$$

### A Quick and Dirty Introduction to Exterior Calculus

外代数研究的基本对象是 k-vectors, 1-vector 就是向量, 2-vector 可以看做是一个有向的面积元, 写为 $u\wedge v$. 它的方向可以通过右手定则确定, 而其本身满足双线性条件. 即 $u\wedge v=-v\wedge u$, $(au_1+bu_2)\wedge v=a(u_1\wedge v)+b(u_2\wedge v)$.

对于 3-vector $u\wedge v\wedge w$, 满足结合律 $(u\wedge v)\wedge w=u\wedge(v\wedge w)$.

在线性空间中, 一个子空间的正交补是另一个子空间, 类似的, 一个 k-vector 的正交补是一个 (n-k)-vector, 用 Hodge star 符号表示. 因为 k-vector 的维数是 $\displaystyle{n\choose k}$, 而 (n-k)-vector 的维数是 $\displaystyle{n\choose{n-k}}$, Hodge star 在这两个空间上定义的对偶关系. 对于 $\mathbb{R}^n$ 中的标准正交基 $\\{e_1,\dots,e_n\\}$, 选取其中 $k$ 个 $u_1,\dots,u_k$, 定义: $(u_1\wedge\cdots\wedge u_k)\wedge\star(u_1\wedge\cdots\wedge u_k)=e_1\wedge\cdots\wedge e_n$. 而对于一般的 k-vector, 则由加性和齐次性推广: $\star(u+v)=\star u+\star v$.

一个重要的例子是 $\mathbb{R}^2$ 中 1-vector, $\star u$ 就是原始向量逆时针旋转九十度. 而 $\mathbb{R}^3$ 中的 2-vector, $\star(u\wedge v)=u\times v$, 和上面的对偶思想类似.

我们可以把线性代数里的内积引入到外代数中, 在这里叫做 k-form. 用 $\sharp$ 可以把 k-form 转换到 k-vector, 用 $\flat$ 可以把 k-vector 转换到 k-form. 在曲面上, 它们的定义是: $u^\flat(v)=I(u,v)$.

在不引起混淆的情况下, 我们用这套符号表示坐标: 对于 vector, 表示为:
$$
v=v^1\frac{\partial}{\partial x^1}+\dots+v^n\frac{\partial}{\partial x^n},
$$
对于 1-form, 表示为:
$$
\alpha=\alpha_1dx^1+\dots+\alpha_ndx^n.
$$

一对基 $dx^i$ 和 $\displaystyle\frac{\partial}{\partial x^i}$ 有时也被称做对偶基, 满足 $dx^i\left(\displaystyle\frac{\partial}{\partial x^j}\right)=\delta_j^i$. 所以: $\alpha(v)=\displaystyle\sum_i \alpha_i v^i$.

上面已经讨论过了一维向量的投影内积, 下面我们来看一下二维的情况. 假设有一个有向的平面元 $u\wedge v$, 我们想让它投影到 $\alpha\times\beta$ 上, 可以让；两条边分别投影到这个平面上: $u'=(\alpha(u),\beta(u))^T$, $v'=(\alpha(v),\beta(v))^T$. 所以投影后的平面元的面积由叉积的长度给出: $\alpha(u)\beta(v)-\alpha(v)\beta(u)$.

这样我们推导出了二维 k-form 的操作:
$$
\alpha\wedge\beta(u,v)=\alpha(u)\beta(v)-\alpha(v)\beta(u).
$$
同样地, 这里的 $\wedge$ 也是满足双线性和反对称和结合律的. 具体地来说, 对于 k-form $\alpha$, l-form $\beta$:
1. $\alpha\wedge\beta=(-1)^{kl}\beta\wedge\alpha$,
2. $\alpha\wedge(\beta\wedge\gamma)=(\alpha\wedge\beta)\wedge\gamma$, 
3. $\alpha\wedge(\beta+\gamma)=\alpha\wedge\beta+\alpha\wedge\gamma$.

现在我们一直在讨论实值的向量, 实际上外代数可以讨论任何向量空间的值, 包括复数值和向量值. 比如向量值的情况: 我们的曲面映射 $f:M\mapsto\mathbb{R}^3$, 其本身 $f$ 就可以视为 0-form, 对于任一点 $p$, 不需要任何的输入, 就可以输出一个三维向量 $f(p)$. 类似地, 它的微分 $df$ 则是一个 1-form, 需要输入一个方向向量 $X$, 映射到一个三维向量 $df_p(X)$.

更加一般的, 假设我们在向量空间 $E$ 上构建外代数, 考虑最简单的 2-form: $\alpha\wedge\beta(u,v)=\alpha(u)\beta(v)-\alpha(v)\beta(u)$, 此时 $\alpha(u)$ 和 $\beta(v)$ 都是 $E$ 中的向量, 要如何定义两个向量的乘法是一个问题. 并非每一个向量空间都有自然的乘法. 如果是复数空间, 可以用复数的乘法; 如果是 $\mathbb{R}^3$, 向量的叉积可以作为乘法.

曲面上的体积元. 回到空间上的曲面 $f:M\mapsto\mathbb{R}^3$, 如果选取 $M$ 中的两个垂直的单位向量 $u,v\in\mathbb{R}^2$, 简单的面积元 $dx^1\wedge dx^2(u,v)=1$ 是度量它们在平面上的面积, 如果想要度量它们在曲面上的面积, 我们可以计算扭曲后的面积大小:
$$
|df(u)\times df(v)|=\sqrt{I(u,u)I(v,v)-I(u,v)^2}=\sqrt{\det(I)}.
$$

所以在任何一个邻域上, 曲面的面积元和平面的面积元都只差一个缩放 $\sqrt{\det(I)}$, 例如:
$$
\star 1=\sqrt{\det(I)}dx^1\wedge\cdots\wedge dx^n:=\omega.
$$

在曲面中, 我们也可以在 k-form 上做 Hodge star: $\alpha\wedge\star\beta=\left<\alpha,\beta\right>\omega$. 通过这个表达式, 我们可以把欧氏空间的内积写成外代数的形式:
$$
u\cdot v=\star\left(u^\flat\wedge\star v^\flat\right),
$$
其中 $u,v\in\mathbb{R}^3$ 是三维空间的向量. 类似地, 向量的叉积也可以写成外代数的形式:
$$
u\times v=\left(\star\left(u^\flat\wedge v^\flat\right)\right)^\sharp.
$$

下面我们要仔细讨论什么是微分, 对于一个标量函数, $\phi:\mathbb{R}^n\mapsto\mathbb{R}$, 可以认为它是一个 0-form, 它的梯度 $\nabla\phi$ 是一个向量场, 第一种定义梯度的方式是指这唯一的向量场, 使得对于任意向量 $X$, $\left<\nabla\phi, X\right>=D_X\phi$ 它们的内积等于方向导数, 另一种则是把微分看做 1-form, 使得 $d\phi(X)=D_X\phi$. 这里的区别则是在于后者不依赖内积的定义, 而是把微分直接看做是一个 $\Omega^k\mapsto\Omega^{k+1}$ 的映射, 这里 $\Omega^k$ 是指所有 k-form 构成的集合. 因此对于梯度, 我们有 $\nabla\phi=(d\phi)^\sharp$.

下面要递归地定义其他 k-form 上的微分, 对于一个 k-form $\alpha$, $d(\alpha\wedge\beta)=d\alpha\wedge\beta+(-1)^k\alpha\wedge d\beta$.

例如, 对于一个三维空间的 1-form $\alpha=\alpha_1dx^1+\alpha_2dx^2+\alpha_3dx^3$, 它的微分: $d\alpha=d(\alpha_1 dx^1+\alpha_2 dx^2+\alpha_3 dx^3)$. 对于 $d(\alpha_1 dx^1)$, 我们可以视 $\alpha_1$ 为一个 0-form, $dx^1$ 是一个 1-form 的基, 所以

$$
\begin{aligned}
d(\alpha_1\wedge dx^1)&=d\alpha_1\wedge dx^1-\alpha_1\wedge ddx^1=d\alpha_1\wedge dx^1\\\\
&=\frac{\partial\alpha_1}{\partial x^1} dx^1\wedge dx^1+\frac{\partial\alpha_1}{\partial x^2} dx^2\wedge dx^1+\frac{\partial\alpha_1}{\partial x^3} dx^3\wedge dx^1.
\end{aligned}
$$

完整展开后计算, 会发现这个和计算旋度是一样的, 我们得到: $\nabla\times X=(\star dX^\flat)^\sharp$. 向量场 $X$ 首先通过 $\flat$ 转换成 1-form, 然后通过微分算子 $d$ 得到 2-form, 再通过 Hodge star 得到 1-form, 最后通过 $\sharp$ 转换回向量场.

下面一个性质是 correctness: $d\circ d=0$. 有点类似于梯度的旋度是零, $d\circ dX=(\nabla\times(\nabla X))^\flat=0$.

类似地, 我们可以对 1-form $\alpha$ 取 Hodge star, 得到一个 (n-1)-form 后计算它的微分
$$
\begin{aligned}
d\star\alpha&=d(\alpha_1 dx^2\wedge dx^3)+d(\alpha_2 dx^3\wedge dx^1)+d(\alpha_3 dx^1\wedge dx^2)\\\\
&=(\frac{\partial\alpha_1}{dx^1}+\frac{\partial\alpha_2}{dx^2}+\frac{\partial\alpha_3}{dx^3})dx^1\wedge dx^2\wedge dx^3.
\end{aligned}
$$

所以散度可以写成: $\nabla\cdot X=\star d\star X^\flat$. 也有一些人喜欢定义 codifferential $\delta=\star d\star$. 我们可以画出下面的图来总结这些算子之间的关系:

<center>
<img src="https://i.upmath.me/svgb/hZDNCoJAFIX3PsXFhW0mMqVd9gpB267B6Fx1aBzjeqVIfPcolHa1Pt_54WBBtfWj2OuzNFOAx5ZqfYkD1Mzd_bxfHxQwMysIsRfN4UpBQd4A27qRfOE-gAkV9I2tBBxVkkMQAUQwR25hYb9BrutuCoztRfuSsoRaBdZnSbpT0A2SpfHuZ8OiOfeeZ8iJ_jUhgX88IuJsabU0RTGepg0KPWTsB650SVOA5M3y1ws" width="40%" />
</center>

综上, 外微分是唯一一个 $\Omega^k\mapsto\Omega^{k+1}$ 的线性算子, 满足:
1. $k=0$ 时, $d\phi(X)=D_X\phi$,
2. 对于 $\alpha\in\Omega^k$, $d(\alpha\wedge\beta)=d\alpha\wedge\beta+(-1)^k\alpha\wedge d\beta$,
3. $d\circ d=0$.

在传统的微积分学中, 积分和微分是通过牛顿-莱布尼兹基本定理联系起来的, 而在外微分中, 则是由斯托克斯定理联系起来, 我们的最终目的是在 Mesh 上得到离散外微分算子.

$$
\int_\Omega d\alpha=\int_{\partial\Omega}\alpha
$$

式子的左边是在 k-dim 的区域 $\Omega$ 上对 k-form $d\alpha$ 做积分, 右边是在 k-1-dim 的边界 $\partial\Omega$ 上对 k-1-form $\alpha$ 做积分.

一个例子是散度定理. $\displaystyle\int_{\Omega}\nabla\cdot XdA=\displaystyle\int_{\partial\Omega}n\cdot X dl$.

定义 1-form $\alpha=X^\flat$, 散度 $\nabla\cdot X=\star d\star \alpha$, 但是写成 2-form 则是 $d\star\alpha$. 所以运用 Stokes 定理:
$$
\int_\Omega d\star\alpha=\int_{\partial\Omega}\star\alpha=\int_{\partial\Omega}n\cdot X dl.
$$

另一个例子是 Green 公式: $\displaystyle\int_\Omega \nabla\times XdA=\displaystyle\int_{\partial\Omega}t\cdot Xdl$.

同样的, 定义 1-form $\alpha=X^\flat$, 旋度 $\nabla\times X=(\star d\alpha)^\sharp$, 但是写成 2-form 则是 $d\alpha$. 所以运用 Stokes 定理:
$$
\int_\Omega d\alpha=\int_{\partial\Omega}\alpha=\int_{\partial\Omega}t\cdot X dl.
$$

在连续光滑的情况, 我们已经见到了很多算子, 而在离散情况下, 想要计算这些微分方程, 最常见的算子是离散微分算子 $d_k$ 和离散 Hodge star $\star_k$. 最终我们要在单纯形网格中去计算它们, 使用的是稀疏的邻接矩阵. 基本的流程是: load mesh -> 构建一些矩阵 -> 求解线性系统. 将连续和离散系统联系起来的是两个操作: 离散化和插值.

离散化的过程并不是简单的采样, 这并不能让我们还原出更多的信息, 我们选择在 k 维的单纯形上对 k-form 做积分, 记为 $\hat{\omega}\_\sigma=\displaystyle\int\_\sigma\omega$. 从连续 form 映射到离散 form 的过程叫做离散化或者 de Rham map, 记为 $R:\Omega^k\mapsto C^{|K|}$, 这里 $C^{|K|}$ 是所有离散 k-form 构成的空间.

以直线段为例, 在实践中我们往往通过采样的方式来近似积分, $\hat{\alpha}\_e=\displaystyle\int\_e \alpha\approx\frac{|e|}{N}\displaystyle\sum_{i=1}^N\alpha_{p_i}(T)$.

首先对于单纯形定义有向的边界: 对于一个有向的 k 维单纯形 $\sigma=(v_0, \dots, v_k)$, 它的边界是由 k-1 维的单纯形组成的链: $\partial\sigma=\displaystyle\sum_{p=0}^k (-1)^p (v_0, \dots, \cancel{v_p}, \dots, v_k)$. 自然地, 我们可以把边界算子扩展到链空间 $\partial\displaystyle\sum c_i\sigma_i=\displaystyle\sum c_i\partial\sigma_i$. 一个 k 维的有向单纯形的上边缘是所有包含它的 k+1 维单纯形, 它们有着相同的相对方向.

和 covector 类似, k 维的 cochain 是一个输入 k 维单纯形链, 输出实数的线性映射. $\alpha(c_1\sigma_1+\dots+c_n\sigma_n)=\displaystyle\sum c_i\alpha(\sigma_i)$. 一个离散化的 k-form 就是一个 k-cochain. 简单地来说, differential k-form 的离散化就是在每一个定向的 k 维单纯形上赋予一个值, 定向的改变会影响值的符号.

有了离散化下面要介绍插值. 0-form 的离散化是在每个顶点上赋值, 它的插值就是利用质心坐标进行线性插值, $u(x)=\displaystyle\sum_{i\in V}u_i\phi_i(x)$. 其中 $\phi_i$ 是针对顶点 $i$ 的 hat function. 这个线性插值通过一组基, 把离散的 form 转为连续的 form. 对于更高维的情况, 比如 1-form, 每一条有向边定义一个基函数, $\phi_{ij}=\phi_id\phi_j-\phi_jd\phi_i$. 这个基函数本身也满足有向的性质, $\phi_{ji}=-\phi_{ij}$, 一个离散 1-form 的插值可以得出: $\displaystyle\sum_{ij}\hat{\omega}\_{ij}\phi_{ij}$. 对于更高维的情况, 基函数是这样定义的: 对于 k 维的单纯形 $(i_0,\dots,i_k)$, 基函数为 $\displaystyle\sum_{p=0}^k(-1)^p\phi_{i_p}d\phi_{i_0}\wedge\dots\wedge\cancel{d\phi_{i_p}}\wedge\dots\wedge d\phi_{i_k}$. 这个基函数叫做 Whitney form. 为什么选取这个为基, 是因为对于任意一个离散的 k-form, 通过 Whitney form 插值后, 再通过 de Rham map 离散化, 可以还原出原始的离散 k-form.

在离散 form 上的微分算子, 通过 Stokes 定理, 可以转变为多个离散值之间的求和. 如果把所有 k 维单纯形对应的离散 k-form 写成向量 $\hat{e}_k$, 微分算子 $\hat{d_k}:\hat{e}\_k\mapsto\hat{e}\_{k+1}$ 写成矩阵形式就是一个有向的邻接矩阵. 同样的, 离散微分算子也满足 $\hat{d}\circ\hat{d}=0$.

```tikz
\usepackage{tikz-cd}
\begin{document}
\Large
\begin{tikzcd}[column sep=huge, row sep=huge]
    \alpha \arrow[r, "d"] \arrow[d, "\int"'] & d\alpha \arrow[d, "\int"] \\
    \hat{\alpha} \arrow[r, "\hat{d}"'] & \widehat{d\alpha}
\end{tikzcd}
\end{document}
```

为了在离散 form 下定义 Hodge star, 首先要在 mesh 上定义对偶. 一个 mesh 的对偶是指每个单纯形的对偶. n 维空间的 k 维单纯形的对偶是一个 n-k 维的形状 (这个单纯形并不一定在一个 n-k 维的子空间). 而离散 form 的对偶就是对每个单纯形的对偶赋予一个值. 和离散 form 不一样的是, 在对偶离散 form 中, 基函数不能用 hat function 来定义, 因为每个面不一定是单纯形, 甚至不一定是凸的形状.

在微分算子中, 我们只需要 mesh 的连接性, 或者说拓扑, 而在 Hodge star 中, 我们还需要 mesh 的几何信息. 最一般地, 我们利用外心来定义单纯形的对偶中顶点的位置. 考虑 k 维单纯形 $\sigma$ 和它的对偶 $\sigma^\star$, k form $\alpha$ 在 $\sigma$ 上的积分值为 $\hat{\alpha}$, 它的对偶 n-k form 在 $\sigma^\star$ 上的积分值为 $\widehat{\star\alpha}$. 很显然没有任何定理能够描述这两个值之间的关系, 如果 $\alpha$ 是常值, 那么很显然, 他们的比值和体积成正比, 再假设 form 足够光滑或者 mesh 足够精细, 我们可以近似地认为这个比例关系成立: $\displaystyle\frac{\widehat{\star\alpha}}{\hat{\alpha}}=\displaystyle\frac{|\sigma^\star|}{|\sigma|}$. 这种定义方式也叫做 *diagonal Hodge star*, 这是因为写作矩阵形式的话是一个对角矩阵. 这些体积的比值一般只和角度和长度有关, 并不需要真正地计算体积.

<img src="https://p.sda1.dev/29/46fa40785e17e4e2e936ab4107b8a56d/3d_example_hodge_star.jpg" />
