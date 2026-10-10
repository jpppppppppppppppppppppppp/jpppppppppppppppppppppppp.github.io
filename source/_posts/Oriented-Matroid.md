---
title: "Oriented Matroids"
date: 2026-08-26 20:52:47
updated: 2026-10-10 01:09:20
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
