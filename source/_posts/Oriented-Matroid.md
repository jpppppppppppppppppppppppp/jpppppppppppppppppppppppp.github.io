---
title: "Oriented Matroids"
date: 2026-08-26 20:52:47
updated: 2026-09-06 16:34:31
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
\mathcal{C}^*=\\{Y=(Y^+,Y^-):Y\text{ is a signed cocircuit of }D\\}.
$$


