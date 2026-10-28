---
title: "Oriented Matroids"
date: 2026-08-26 20:52:47
updated: 2026-10-28 00:32:29
home_cover: https://p.sda1.dev/34/a02d8ca62759e5d9983ed95a3e822782/cover.jpg
post_cover: https://p.sda1.dev/34/1f2656e074b6bfc49e46aed500b4d614/post.jpg
copyright_info: true
tags:
    - Math
categories:
    - Notes
mathjax: true
tikzjax: true
excerpt: Oriented Matroids by Anders Bj&ouml;rner, etc.
---

eBook: <a href="https://www.cambridge.org/core/books/oriented-matroids/A34966F40E168883C68362886EF5D334">Oriented Matroids</a>.

---

Before discussing oriented matroids, we first recall the definition of a matroid. A matroid abstracts the notion of independence that appears in many settings, such as linear independence in a vector space and acyclicity in a graph.

Formally, a matroid is a pair $M=(E,I)$, where $E$ is a ground set and $I\subseteq 2^E$ is a family of subsets of $E$, called the *independent sets*. The family $I$ must satisfy the following three axioms:

First, the empty set is independent: $\emptyset\in I$.

Second, every subset of an independent set is independent. That is, if $A\in I$ and $B\subseteq A$, then $B\in I$.

Third, if two independent sets have different cardinalities, the smaller one can be enlarged by an element of the larger one. More precisely, if $A,B\in I$ and $|A|<|B|$, then there exists an element $e\in B\setminus A$ such that $A\cup \\{e\\}\in I$.

These axioms allow us to define the rank of any subset $X\subseteq E$ as the maximum size of an independent subset of $X$:
$$r(X)=\max\\{|A|:A\subseteq X,\ A\in I\\}.$$

Any subset of $E$ that does not belong to $I$ is called dependent. A minimal dependent set is called a **circuit**. Thus, a matroid can be described equivalently by either its independent sets or its circuits.

### 1.1 Oriented matroids from directed graphs

Consider a directed graph $D=(V,E)$ with vertex set $V$ and arc set $E$. Each **simple cycle**, being a minimal dependent set, can be regarded as a signed subset of $E$. After choosing a direction in which to traverse the cycle, we place each arc in the positive or negative part according to whether its orientation agrees or disagrees with the direction of traversal. The resulting signed subset is called a **signed circuit** of $D$.

An oriented matroid obtained from a digraph $D$ can be described by its signed circuits. The collection of all signed circuits of $D$ is
$$
\mathcal{C}=\\{X=(X^+,X^-):X\text{ is a signed circuit of }D\\}.
$$
We denote the resulting oriented matroid by $\mathcal{M}_D=\mathcal{M}(E)=(E,\mathcal{C})$. Note that $\mathcal{C}$ contains both signed circuits in every opposite pair $\pm X$. By forgetting the signs, we obtain the underlying matroid $\underline{\mathcal{M}}(E)=(E,\underline{\mathcal{C}})$, where $\underline{\mathcal{C}}=\\{\underline{X}=X^+\cup X^-:X\in\mathcal{C}\\}$.

A second description uses minimal cuts. Given a partition $V=V^1\dot\cup V^2$ of the vertex set, the arcs between $V^1$ and $V^2$ form a **minimal cut** if removing them increases the number of connected components of the underlying undirected graph by one. Let $Y^+$ consist of the arcs directed from $V^1$ to $V^2$, and let $Y^-$ consist of those directed from $V^2$ to $V^1$. The resulting signed sets $Y=(Y^+,Y^-)$ are called the **signed cocircuits** of $D$. We write
$$
\mathcal{C}^\*=\\{Y=(Y^+,Y^-):Y\text{ is a signed cocircuit of }D\\}.
$$

Properties of $D$ can be expressed in terms of the collections $\mathcal{C}$ and $\mathcal{C}^\*$. For example, the digraph $D$ is acyclic—that is, it contains no directed cycle—if and only if $\mathcal{C}$ contains no positive circuit, where a circuit $X$ is positive if $X^-=\emptyset$. The same property can be characterized in terms of cocircuits: $D$ is acyclic if and only if every arc belongs to a positive cocircuit. In other words, for every $e\in E$, there exists a cocircuit $Y=(Y^+,\emptyset)$ such that $e\in Y^+$.

<details>
    <summary>Sketch of Proof.</summary>

Suppose that $D$ is acyclic, and fix an arc $e=(u,v)\in E$. Let $S=\\{x\in V:x\leadsto u\\}$ be the set of vertices from which $u$ is reachable. Since $D$ is acyclic, $v\notin S$; otherwise, a path from $v$ to $u$, together with the arc $(u,v)$, would form a directed cycle. Hence, $e$ belongs to the cut induced by the partition $V=S\mathbin{\dot\cup}(V\setminus S)$. Moreover, no arc enters $S$: if $(x,y)$ were an arc with $x\notin S$ and $y\in S$, then $y\leadsto u$ would imply $x\leadsto u$, contradicting $x\notin S$. Thus, this cut is positive. By restricting it to a minimal cut containing $e$, we obtain a positive cocircuit $Y=(Y^+,\emptyset)$ with $e\in Y^+$.

</details>

We now introduce the dual oriented matroid, whose collection of circuits is $\mathcal{C}^\*$. To motivate this definition, suppose that $D$ is a planar digraph. There is a canonical way to orient its dual graph $D^\*$ such that the circuits $\mathcal{C}(D^\*)$ of the dual graph correspond exactly to the cocircuits $\mathcal{C}^\*(D)$ of the original graph, and conversely.

The next important property is **orthogonality**: if a directed cycle crosses a cut in one direction, then it must also cross the cut in the opposite direction.

- If $X\in\mathcal{C}$ is a circuit and $Y\in\mathcal{C}^\*$ is a cocircuit of an oriented matroid, then
$$
 (X^+\cap Y^+)\cup(X^-\cap Y^-)\neq\emptyset
 \quad\Longleftrightarrow\quad
 (X^+\cap Y^-)\cup(X^-\cap Y^+)\neq\emptyset.
$$

For two signed sets $X$ and $Y$, let
$$
S(X,Y)=(X^+\cap Y^-)\cup(X^-\cap Y^+)
$$
denote their **separation set**. The orthogonality condition can then be written as $X\perp Y$ for every $X\in\mathcal{C}$ and every $Y\in\mathcal{C}^\*$.

### 1.2 Point configurations and hyperplane arrangements

#### Vector configurations

Linear dependence and independence in vector spaces provide alternative ways to view oriented matroids. Given a finite set of vectors that spans a vector space of dimension $r$ over an arbitrary field, the minimal linear dependences yield the circuits of a matroid of rank $r$. Over $\mathbb{R}$, a minimal linear dependence may be written as
$$
\sum_{i=1}^n \lambda_i \boldsymbol{v}_i=\boldsymbol{0}
$$
with $\lambda_i\in\mathbb{R}$, not all zero. Here the sets $\underline{X}=\\{i:\lambda_i\neq0\\}$ are the circuits of the underlying matroid. For the associated oriented matroid, we consider the signed sets $X=(X^+,X^-)$.

This yields the oriented matroid $\mathcal{M}=(E,\mathcal{C})$ of a vector configuration $E=\\{\boldsymbol{v}_1,\dots,\boldsymbol{v}_n\\}\subsetneq \mathbb{R}^r$ in terms of its collection $\mathcal{C}$ of signed circuits.

If $E$ provides the same list of signed circuits for a given oriented matroid $\mathcal{M}_0$, we say $E$ is a realization of $\mathcal{M}_0$.

The basis orientation or chirotope of a vector configuration is given by the signs of the determinants of ordered $r$-subsets of $E$.
$$
\chi(i_1,\dots,i_r)=\operatorname{sign} \operatorname{det} (\boldsymbol{v}\_{i_1},\dots,\boldsymbol{v}\_{i_r})\in\\{+,-,0\\}.
$$
The function $\chi$ is antisymmetric.

In addition to antisymmetry, the determinants of a configuration of vectors also satisfy Grassmann-Pl&uuml;cker relations. To describe these relations, consider the Pl&uuml;cker embedding $\eta$ of the Grassmannian of $k$-dimensional subspaces of an $n$-dimensional vector space $V$ into the projectivization of the $k$-th exterior power of $V$.
$$
\eta:\quad\begin{gather}
\operatorname{Gr}(k,V)\to \mathbb{P}(\wedge^k V)\\\\
\operatorname{Span}(w_1,\dots,w_k)\mapsto [w_1\wedge\dots\wedge w_k]
\end{gather}
$$
Here $w_1,\dots,w_k$ form a basis of the chosen subspace. The homogeneous coordinates of the image under this embedding satisfy a simple set of homogeneous quadratic relations. After choosing a basis of $V$, let $[w]$ be the $n\times k$ matrix whose columns are the coordinates of $w_1,\dots,w_k$, and define $\Delta_{i_1,\dots,i_k}$ to be the determinant of the $k\times k$ submatrix obtained by selecting rows $i_1,\dots,i_k$ in that order. Then for any two ordered sequences
$$
1\leq\quad\begin{gather}
    i_1<i_2<\dots<i_{k-1}\\\\
    j_1<j_2<\dots<j_{k+1}
\end{gather}\quad\leq n,
$$
we have
$$
\sum_{l=1}^{k+1}(-1)^l\Delta_{i_1,\dots,i_{k-1},j_l}\Delta_{j_1,\dots,\hat{j}\_l,\dots,j_{k+1}}=0.
$$

For example, in rank $3$, with $i=(1,2)$ and $j=(1,3,4,5)$, and writing determinants in brackets, we have
$$
-\underbrace{[121]}_{=0}[345]+[123][145]-[124][135]+[125][134]=0.
$$
The corresponding oriented matroid axiom thus requires that the six signs of the brackets on the left-hand side allow the equality to hold, even when the actual scalars are not given. In other words, the three terms $[123][145]$, $-[124][135]$, and $[125][134]$ must either all be zero or include both a positive and a negative term.

We can also interpret the cocircuits of a vector configuration: consider any linear hyperplane $H$ spanned by a subset of $E$, and let $Y_H^+$ and $Y_H^-$ be the sets of vectors in $E$ on either side of $H$. These signed sets are the cocircuits of the configuration. However, this is hard to visualize, which motivates the passage to affine space to reduce the dimension by one.

#### Point configurations

Every vector configuration in $\mathbb{R}^r\backslash\\{0\\}$ corresponds to an affine point configuration in an $(r-1)$-dimensional affine space. We choose a linear form $l_0$ such that $l_0(\boldsymbol{v}_i)\neq 0$ for all $\boldsymbol{v}_i$, define
$$
\mathbb{A}^{r-1}=\\{\boldsymbol{x}\in\mathbb{R}^r:l_0(\boldsymbol{x})=1\\},
$$
as our affine space of dimension $r-1$, and associate with every vector $\boldsymbol{v}_i$ the point $\frac{1}{l_0(\boldsymbol{v}_i)}\boldsymbol{v}_i\in\mathbb{A}^{r-1}$.

```tikz
\usetikzlibrary{decorations.markings}
\begin{document}
\begin{tikzpicture}[
    x=1.3cm, y=1.3cm, font=\large, line width=1.2pt,
    directed/.style={
        decoration={markings, mark=at position 0.5 with {\arrow[scale=1.8]{stealth}}},
        postaction={decorate}
    }
]
    \coordinate (v1) at (0,0);
    \coordinate (v2) at (4,0);
    \coordinate (v3) at (2,1.15);
    \coordinate (v4) at (2,3.45);

    \draw[directed] (v1) -- node[below=5pt] {$1$} (v2);
    \draw[directed] (v1) -- node[pos=0.4, below right=2pt] {$2$} (v3);
    \draw[directed] (v1) -- node[pos=0.4, left=5pt] {$3$} (v4);
    \draw[directed] (v2) -- node[pos=0.6, above right=2pt] {$4$} (v3);
    \draw[directed] (v2) -- node[pos=0.4, right=5pt] {$5$} (v4);
    \draw[directed] (v3) -- node[pos=0.25, left=5pt] {$6$} (v4);

    \fill (v1) circle (3.5pt) node[left=3pt] {$v_1$};
    \fill (v2) circle (3.5pt) node[right=3pt] {$v_2$};
    \fill (v3) circle (3.5pt) node[below=3pt] {$v_3$};
    \fill (v4) circle (3.5pt) node[above=3pt] {$v_4$};
\end{tikzpicture}
\end{document}
```

For example, let $E=(\boldsymbol{v}\_1,\dots,\boldsymbol{v}\_6)$ be the vectors in $\mathbb{R}^3$ given by the columns of the matrix $A=\begin{pmatrix}1&1&1&0&0&0\\\\0&1&1&1&1&0\\\\0&0&1&0&1&1\end{pmatrix}$. From $A$ we get an oriented matroid of rank $3$ with signed circuits $1\overline{2}4,1\overline{2}5\overline{6}$, and so on. This vector configuration has the same signed circuits as the digraph shown above, with column $i$ corresponding to arc $i$. Thus, $A$ is a realization of the oriented matroid of this digraph.

With $l_0(\boldsymbol{x})=x_1+x_2+x_3$, we get the point configuration in the affine plane, $\overline{A}=\begin{pmatrix}0&\frac12&\frac13&1&\frac12&0\\\\0&0&\frac13&0&\frac12&1\end{pmatrix}$, which is obtained by deleting the first coordinates from the vectors $\frac{1}{l_0(\boldsymbol{v}_i)}\boldsymbol{v}_i$.

```tikz
\begin{document}
\begin{tikzpicture}[x=4.8cm, y=4.8cm, >=stealth, font=\normalsize]
    % Coordinates are (x_2,x_3), the columns of the normalized matrix.
    \coordinate (p1) at (0,0);
    \coordinate (p2) at (0.5,0);
    \coordinate (p3) at ({1/3},{1/3});
    \coordinate (p4) at (1,0);
    \coordinate (p5) at (0.5,0.5);
    \coordinate (p6) at (0,1);

    \draw[->, gray!70] (-0.07,0) -- (1.17,0) node[right] {$x_2$};
    \draw[->, gray!70] (0,-0.07) -- (0,1.17) node[above] {$x_3$};

    % The four collinear triples are 124, 456, 135, and 236.
    \draw[black, thick] (p1) -- (p4);
    \draw[black, thick] (p4) -- (p6);
    \draw[black, thick] (p1) -- (p5);
    \draw[black, thick] (p2) -- (p6);

    \fill (p1) circle (2.5pt) node[below left] {$1$};
    \fill (p2) circle (2.5pt) node[below] {$2$};
    \fill (p3) circle (2.5pt) node[right, xshift=2pt] {$3$};
    \fill (p4) circle (2.5pt) node[below] {$4$};
    \fill (p5) circle (2.5pt) node[above right] {$5$};
    \fill (p6) circle (2.5pt) node[left] {$6$};
\end{tikzpicture}
\end{document}
```

Note that signed circuits have a simple geometric interpretation in the affine plane. Every circuit is a minimal signed set of points such that the convex hull of the positive elements intersects the convex hull of the negative elements. Mathematically speaking, the signed circuits of an affine point configuration are exactly the minimal Radon partitions of the point set. For example, from $1\overline{2}4$ we see that point $2$ must lie on the open line segment $\overset{\frown}{14}$. Similarly, from $2\overline{34}5$ we see that the open line segments $\overset{\frown}{25}$ and $\overset{\frown}{34}$ must intersect (in the ambient real affine plane, not necessarily in the oriented matroid itself).

The sign patterns of arbitrary (not necessarily minimal) dependences can be derived by composing circuits. The composition of two signed sets is defined as
$$
\begin{gather}
X\circ Y=(X^+\cup(Y^+\backslash X^-),X^-\cup(Y^-\backslash X^+)).\\\\
(X\circ Y)_e=\begin{cases}
    X_e & \text{if }X_e\neq0,\\\\
    Y_e & \text{otherwise}.
\end{cases}
\end{gather}
$$
For example, if $X=(+,+,-,0)$ and $Y=(-,-,0,+)$, then $X\circ Y=(+,+,-,+)$. The signed sets obtained by repeated composition of circuits are called the vectors of this oriented matroid. Note that a dependent set may correspond to different vectors: $Y\circ X=(-,-,-,+)$ has the same support as $X\circ Y$, but a different sign pattern.

By cutting the point configuration with an arbitrary affine hyperplane (not necessarily spanned by points of this configuration), we can record which points lie on the positive side, on the negative side, or on the hyperplane itself, assigning them the signs $+$, $-$, or $0$, respectively. The resulting sign patterns are called covectors and can be obtained by repeated compositions of cocircuits.

The chirotope records the orientation of an ordered $r$-tuple of points. Writing the affine coordinates of the configuration as column vectors $\bar{\boldsymbol{v}}_1,\dots,\bar{\boldsymbol{v}}_n$, we have
$$
\chi(i_1,\dots,i_r)=\operatorname{sign}\det\begin{pmatrix}
1 & \cdots & 1\\\\
\bar{\boldsymbol{v}}\_{i\_1} & \cdots  & \bar{\boldsymbol{v}}\_{i\_r}
\end{pmatrix}.
$$

Thus, for example, $\chi(2,3,4)=-$.

#### Hyperplane arrangements

Let $\mathcal{A}=\\{H_1,\dots,H_n\\}$ be a real hyperplane arrangement, which is a finite set of hyperplanes through the origin in $\mathbb{R}^r$. Each hyperplane $H_i$ is defined by $l_i(\boldsymbol{x})=0$, where $l_i(\boldsymbol{x})=\boldsymbol{a}_i^T\boldsymbol{x}$ is a linear function. We can interpret the $l_i$ as vectors in the dual space $(\mathbb{R}^r)^\*$, where they form a vector configuration.

In fact, once we choose $l_i$, we orient each hyperplane $H_i$ by specifying its positive side $H_i^+=\\{\boldsymbol{x}:l_i(\boldsymbol{x})\geq0\\}$. Equivalently, by taking $S_i^+=H_i^+\cap S^{r-1}$, we get an arrangement of hemispheres on the sphere $S^{r-1}$ that represents the oriented matroid.

```tikz
\begin{document}
\begin{tikzpicture}[x=1.3cm, y=1.3cm, font=\small,
    line width=0.9pt, every node/.style={inner sep=1.5pt},
    positive side/.style={->, >=stealth, line width=0.8pt}]
    % Equator and the five hyperplane traces in the upper hemisphere.
    \coordinate (T) at (0,4);
    \coordinate (B) at (0,-4);
    \coordinate (L) at (-4,0);
    \coordinate (R) at (4,0);
    \coordinate (U) at ({-sqrt(8)},{sqrt(8)});
    \coordinate (S) at ({sqrt(8)},{-sqrt(8)});
    \coordinate (O) at (0,0);
    \coordinate (P) at (-1.52,0);
    \coordinate (Q) at (0,-1.52);
    \coordinate (D) at ({1.52/sqrt(1+1.52*1.52/16)},
                            {-1.52/sqrt(1+1.52*1.52/16)});

    \draw (0,0) circle (4);
    \draw (T) -- (B);
    \draw (L) -- (R);
    \draw (U) -- (S);
    \draw (L) arc[start angle=180, end angle=360,
                  x radius=4, y radius=1.52];
    \draw (U) .. controls (-2.8,1.75) and (-2.08,0.7) .. (P)
        .. controls (-1.05,-0.6) and (-0.58,-1.08) .. (Q)
        .. controls (0.95,-2.25) and (1.95,-2.73) .. (S);

    % Coorientations: one normal arrow points into each positive hemisphere.
    % These are not tangent arrows giving a direction of travel on the curves.
    \draw[positive side] (0,-2.8) -- (0.5,-2.8);
    \node at (0.46,-2.61) {$1$};
    \draw[positive side] (2.45,-2.45) -- (2.8,-2.1);
    \node at (2.89,-1.94) {$2$};
    % Evaluate the cubic and its normal at t=0.4 so the arrow starts on it.
    \pgfmathsetmacro{\curveX}{3*0.6*0.6*0.4*0.95+3*0.6*0.4*0.4*1.95+0.4*0.4*0.4*sqrt(8)}
    \pgfmathsetmacro{\curveY}{0.6*0.6*0.6*(-1.52)+3*0.6*0.6*0.4*(-2.25)+3*0.6*0.4*0.4*(-2.73)-0.4*0.4*0.4*sqrt(8)}
    \pgfmathsetmacro{\tangentX}{3*0.6*0.6*0.95+6*0.6*0.4*(1.95-0.95)+3*0.4*0.4*(sqrt(8)-1.95)}
    \pgfmathsetmacro{\tangentY}{3*0.6*0.6*(-2.25+1.52)+6*0.6*0.4*(-2.73+2.25)+3*0.4*0.4*(-sqrt(8)+2.73)}
    \pgfmathsetmacro{\tangentLength}{sqrt(\tangentX*\tangentX+\tangentY*\tangentY)}
    \draw[positive side] (\curveX,\curveY) --
        ({\curveX-0.45*\tangentY/\tangentLength},
         {\curveY+0.45*\tangentX/\tangentLength});
    \node at (1.53,-1.98) {$3$};
    \draw[positive side] (2.9,0) -- (2.9,0.5);
    \node at (3.08,0.45) {$4$};
    % The ellipse normal (-dy/dx,1) points toward the positive fifth side.
    \pgfmathsetmacro{\ellipseY}{-1.52*sqrt(1-2.8*2.8/16)}
    \pgfmathsetmacro{\ellipseSlope}{1.52*2.8/(16*sqrt(1-2.8*2.8/16))}
    \pgfmathsetmacro{\normalLength}{sqrt(1+\ellipseSlope*\ellipseSlope)}
    \draw[positive side] (2.8,\ellipseY) --
        ({2.8-0.45*\ellipseSlope/\normalLength},
         {\ellipseY+0.45/\normalLength});
    \node at (2.52,-0.57) {$5$};
    \draw[positive side] ({-sqrt(8)},{-sqrt(8)}) --
        ({-sqrt(8)+0.34},{-sqrt(8)+0.34});
    \node at (-2.34,-2.62) {$6$};

    % Two-dimensional cells: all six signs are nonzero.
    \node at (1.8,1.3) {$123456$};
    \node at (-1,2.3) {$\overline{1}23456$};
    \node at (-3.05,0.9) {$\overline{123}456$};
    \node at (-1.5,0.9) {$\overline{12}3456$};
    \node at (-2.2,-0.62) {$\overline{1234}56$};
    \node at (-0.48,-0.44) {$\overline{12}3\overline{4}56$};
    \node at (1.96,-0.62) {$123\overline{4}56$};
    \node at (0.58,-1.25) {$1\overline{2}3\overline{4}56$};
    \node at (2.65,-1.5) {$123\overline{45}6$};
    \node at (1.7,-2.24) {$1\overline{2}3\overline{45}6$};
    \node at (-1.8,-2.08) {$\overline{12345}6$};
    \node at (0.9,-3.1) {$1\overline{2345}6$};

    % One-dimensional cells on the equator (the sixth sign is zero).
    \node[rotate=-45] at (2.65,2.72) {$12345$};
    \node[rotate=20] at (-1.45,3.43) {$\overline{1}2345$};
    \node[rotate=65] at (-3.9,1.55) {$\overline{123}45$};
    \node[rotate=-58] at (-3.2,-2.12) {$\overline{12345}$};
    \node[rotate=20] at (1.46,-3.47) {$1\overline{2345}$};
    \node[rotate=65] at (3.45,-1.65) {$123\overline{45}$};

    % One-dimensional cells on the interior traces.
    \node[rotate=-90] at (0.16,2.08) {$23456$};
    \node[rotate=-90] at (0.17,-0.7) {$\overline{2}3\overline{4}56$};
    \node[rotate=-90] at (-0.17,-2.6) {$\overline{2345}6$};
    \node at (-3.55,0.2) {$\overline{123}56$};
    \node at (-0.83,0.2) {$\overline{12}356$};
    \node at (1.93,0.2) {$12356$};
    \node[rotate=-45] at (-1.22,1.48) {$\overline{1}3456$};
    \node[rotate=-45] at (0.87,-0.6) {$13\overline{4}56$};
    \node[rotate=-45] at (2.15,-1.86) {$13\overline{45}6$};
    \node[rotate=-67] at (-2.32,1.55) {$\overline{12}456$};
    \node[rotate=-45] at (-0.96,-0.93) {$\overline{124}56$};
    \node[rotate=-25] at (1,-2.4) {$1\overline{245}6$};
    \node[rotate=-20] at (-2.1,-1.4) {$\overline{1234}6$};
    \node at (0.85,-1.72) {$1\overline{2}3\overline{4}6$};
    \node[rotate=35] at (3.26,-0.56) {$123\overline{4}6$};

    % Zero-dimensional cells; omitted digits have sign zero.
    \fill (T) circle (2.7pt) node[below right=4pt] {$2345$};
    \fill (B) circle (2.7pt) node[above left=4pt] {$\overline{2345}$};
    \fill (L) circle (2.7pt) node[left=5pt] {$\overline{123}$};
    \fill (R) circle (2.7pt) node[above left=4pt] {$123$};
    \fill (U) circle (2.7pt) node[right=6pt] {$\overline{1}45$};
    \fill (S) circle (2.7pt) node[above=7pt] {$1\overline{45}$};
    \fill (O) circle (2.7pt) node[above right=4pt] {$356$};
    \fill (P) circle (2.7pt) node[above left=5pt] {$\overline{12}56$};
    \fill (Q) circle (2.7pt) node[below left=5pt] {$\overline{24}6$};
    \fill (D) circle (2.7pt) node[above right=5pt] {$13\overline{4}6$};
\end{tikzpicture}
\end{document}
```

Reading the basis orientations is simple when $r=3$. The rule is that $[i,j,k]=+$ if the boundary arcs labeled $i,j,k$ occur in counterclockwise order around the spherical triangle $S_i^+\cap S_j^+\cap S_k^+$, $[i,j,k]=-$ if they occur in clockwise order, and $[i,j,k]=0$ if they do not determine a triangle. For example, $[1,2,6]=+$, $[2,3,5]=-$, and $[2,3,6]=0$.
