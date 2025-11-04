---
title: "Discrete Differential Geometry"
date: 2025-10-23 21:20:51
updated: 2025-11-03 14:13:31
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


