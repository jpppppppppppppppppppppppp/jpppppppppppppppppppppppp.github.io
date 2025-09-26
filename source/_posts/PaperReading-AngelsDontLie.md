---
title: "PaperReading: Angels Don't Lie"
date: 2025-09-26 15:05:07
updated: 2025-09-26 17:06:26
home_cover: https://p.sda1.dev/27/be11eba4b34fc4fdf7394442f90dbfb5/cover.jpg
post_cover: https://p.sda1.dev/27/fbc8d67219a2abc03a75ff380f6d4fc1/post.PNG
copyright_info: true
tags:
    - Reinforcement Learning
    - Post Training
    - Large Language Model
categories:
    - PaperReading
mathjax: true
excerpt: "[NIPS2025Spotlight] Angles don't lie: Unlocking Training-Efficient RL Through the Model's Own Signals."
---

Link: <a href="https://arxiv.org/abs/2506.02281">Angles Don\'t Lie: Unlocking Training-Efficient RL Through the Model\'s Own Signals</a>.

是和我关系非常好的博一学长的工作, 借着机会学习一下如何做研究, 认真读了一下, 觉着思路非常好, 写一下感受. 笔者也不懂 RL 和 Post Training, 很多地方只能靠感受来理解.

文章针对 Post Training 中训练样本效率问题, 例如, 使用 GRPO 微调 7B 模型需要消耗 240 卡时 (16卡$\times$15h), 只能在八千个样本上跑 100 个 step. 虽然笔者没有跑过强化学习和后训练, 但是也知道 Rollout 的计算成本之大. 对训练数据做操作是减少训练成本的重要方法, 现有的方法可以分为两类: 一类是精选数据, 如 <a href="https://arxiv.org/abs/2502.03387">LIMO: Less is More for Reasoning</a>, 说明了在精选的高质量数据集上微调效率更好, 效果更好; 另一类是类似于课程学习的方法, 例如 <a href="https://arxiv.org/abs/2504.05520">Efficient Reinforcement Finetuning via Adaptive Curriculum Learning</a>, 通过动态调整问题难度来加速收敛. 这些方法有两个缺陷:
1. 设计数据时没有考虑模型的反应. 无论是数据的难度和多样性, 都是假定模型不可知的, 但是不同的模型对于同样的数据反应不同, 所以这类不考虑模型的方法并不能达到最好的效果.
2. 数据准备成本高. 无论是精选数据亦或是依据难度分层, 都需要模型对问题的正确率或者人类标注.

### Which signal should we focus on?

如果想要在 decoding 阶段得到模型的 feedback 是消耗很多计算的, 约等于直接依赖正确率, 相反, pre-filling 阶段的计算更少, 需要一次正向输入. 下面我们来理解正向过程是如何影响反向传播的.

对于一个权重矩阵 $W\in\mathbb{R}^{d\times h}$, 输入序列为 $X\in\mathbb{R}^{m\times d}$, 输入为 $Y=XW\in\mathbb{R}^{m\times h}$, 所以矩阵 $W$ 的梯度为
$$
\nabla_W\mathcal{L}=X^T\nabla_Y\mathcal{L}=\sum_{i=1}^m x_i^T(\nabla_{Y}\mathcal{L})_i.
$$

下面考虑梯度的范数, 这里用 Frobenius 范数,
$$
\|\|\nabla_W\mathcal{L}\|\|_F^2=\left< \sum\_{i=1}^m x_i^T(\nabla_Y\mathcal{L})_i, \sum\_{j=1}^m x_j^T(\nabla_Y\mathcal{L})_j \right>=\sum\_{i=1}^m\sum\_{j=1}^m \left< x_i^T(\nabla_Y\mathcal{L})_i, x_j^T(\nabla_Y\mathcal{L})_j \right>.
$$

又因为 $x_i$, $x_j$, $(\nabla_Y\mathcal{L})_i$, $(\nabla_Y\mathcal{L})_j$ 都是列向量, 所以
$$
\left< x_i^T(\nabla_Y\mathcal{L})_i, x_j^T(\nabla_Y\mathcal{L})_j \right>=trace((x_i^T(\nabla_Y\mathcal{L})_i)^T x_j^T(\nabla_Y\mathcal{L})_j)=\|\|x_i\|\| \|\|x_j\|\|\cos\theta\_{ij}((\nabla_Y\mathcal{L})_i(\nabla_Y\mathcal{L})_j^T).
$$

因为一般情况下, 输入都是经过归一化的, 这里经过变化的 $\|\|x_i\|\|$ 已经不能反应它们包含的信息了, 所以文章提出, 隐空间状态的相对角度 $\cos\theta\_{ij}$ 会影响梯度的范数, 因此选择这个作为信号.

在包含激活函数和 Attention Mechanism 的网络结构中, 类似的结果也成立, 具体证明结果在附录中.

### Which signal should we focus on?

文章发现, 相对角度在网络中呈凝聚现象, 层数越深, 越呈现分块凝聚. 并且分块的部位准确的分割了输入: system prompt, few-shot examples, question 三个部分.

<img src="https://p.sda1.dev/27/72e2923d85d0e0669fdb7d79baa8a5c0/angle_layer.jpg" />

假设输入总长度为 $m$, system prompt 和 few-shot examples 总长度为 $n$, 那么问题的长度就是 $m-n$. 因为在所有问题上前者都是一样的, 所以文章提出两个信号:

$$
\mathcal{C}\_{intra}=\frac{1}{(m-n)^2}\sum_{i=n+1}^m\sum_{j=n+1}^m\cos\theta\_{ij},\quad \mathcal{C}\_{inter}=\frac{1}{(m-n)n}\sum_{i=n+1}^m\sum_{j=1}^n\cos\theta\_{ij}.
$$

### How Can This Signal Be Leveraged to Accelerate Training?

然后, 文章发现, 在训练过程中, 上述两个信号都在稳定增加, 且信号越强的样本, 学习的越快.

<img src="https://p.sda1.dev/27/d42024a9c26366c040f10d56b60093bd/angle_train.jpg" />

这也很好解释, 因为信号越强, 梯度越大, 模型的更新越大, 越容易在训练的早期阶段学会. 同时, 文章也可视化了神经元的激活情况, 可以发现神经激活情况也逐渐收敛, 且远离收敛簇的样本正确率更低.

<img src="https://p.sda1.dev/27/844e4a6b14f7028ff45ec05f8feacee9/neuron_train.jpg" />

### Framework

总结一下上面的发现, 文章提出了一个基于角度的信号, 信号越强, 学习越快.

所以先用 pre-filling 阶段对数据集进行排序, 并根据每个 step 的训练结果调整数据的采样概率.

<img src="https://p.sda1.dev/27/844e4a6b14f7028ff45ec05f8feacee9/neuron_train.jpg" />

总体上我感觉这篇文章内容十分流畅清晰.

