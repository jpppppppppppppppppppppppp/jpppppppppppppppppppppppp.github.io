---
title: 'PaperReading: DiffusionNFT'
date: 2026-07-30 23:21:28
updated: 2026-07-30 23:21:28
home_cover: https://p.sda1.dev/34/21f46e32fba900619c2ac475bbe619d9/cover.jpg
post_cover: https://p.sda1.dev/34/b54abdf7b0deb3a21c19309b43e508d8/post.jpg
copyright_info: true
tags:
    - Reinforcement Learning
    - Diffusion
categories:
    - PaperReading
mathjax: true
excerpt: "[2025.09.19] DiffusionNFT: Online Diffusion Reinforcement with Forward Process."
---

Link: <a href="https://arxiv.org/abs/2509.16117v2">DiffusionNFT: Online Diffusion Reinforcement with Forward Process</a>.

DiffusionNFT 提出了一种在 forward process 中用 RL 训练 Diffusion Model 的方法, 在除了数据收集阶段, 不需要任何 reverse process. 所以不依赖任何 Solver 求解 ODE. 由于本人对于 Diffusion Model 的了解还非常的浅层, 也希望能够在未来回过头来继续补充本篇.

Diffusion 的加噪过程可以写为 $\pi_{t|0}(x_t|x_0)=\mathcal{N}(x_t;\alpha_t x_0,\sigma_t^2 I)$, 并且可以重参数化为 $x_t=\alpha_t x_0 + \sigma_t\epsilon, \epsilon\sim\mathcal{N}(\epsilon;0,I)$.

而 Flow Model 的训练目标是训练一个速度场 $v_\theta(x_t,t,c)$, 并最小化误差的期望
$$
\min \underset{c,t,x_0,\epsilon}{\mathbb{E}} \left[w(t)\cdot \|\|v_\theta(x_t,t,c)-(\dot{\alpha_t}x_0+\dot{\sigma_t}\epsilon)\|\|_2^2\right].
$$

那么训练得到的最优速度场为后验期望

$$
\begin{aligned}
v_\theta^*(x_t,t,c)&=\mathbb{E}\left[\dot{x_t}\|x_t, t, c\right]\\\\
&=\mathbb{E}\left[\dot{\alpha}x_0 + \dot{\sigma_t}\frac{x_t-\alpha_t x_0}{\sigma_t} \| x_t, t, c \right]\\\\
&=\frac{\dot{\sigma_t}}{\sigma_t} x_t + (\dot{\alpha_t}-\frac{\alpha_t\dot{\sigma_t}}{\sigma_t})\cdot\underset{x_0\sim\pi(x_0|x_t,c)}{\mathbb{E}}[x_0]
\end{aligned}
$$

下面回到正题, 对于 Online RL, 我们从 prompt 集合中 $c\in\{c\}$ 采样 $K$ 张清晰图像 $x_0^{1:K}$, 并用给定的 reward model 给出分数, 表示它质量最优的概率 $r(x_0, c)=p(o=1|x_0, c)\in[0,1]$. 通过 $r$ 可以把采样得到的图像分为两个虚构的子集, 每张图像 $x_0$ 会有 $r$ 的概率进入 $D^+$ 并有 $1-r$ 的概率进入 $D^-$. 那么在这两个数据集上的数据分布为:

$$
\begin{gathered}
\pi^+(x_0|c)=\pi^{\text{old}}(x_0|o=1,c)=\frac{p(o=1|x_0,c)\cdot\pi^{\text{old}}(x_0|c)}{p_{\pi^{\text{old}}}(o=1|c)}=\frac{r(x_0,c)}{p_{\pi^{\text{old}}}(o=1|c)}\cdot \pi^{\text{old}}(x_0|c),
\\\\
\pi^-(x_0|c)=\pi^{\text{old}}(x_0|o=0,c)=\frac{p(o=0|x_0,c)\cdot\pi^{\text{old}}(x_0|c)}{p_{\pi^{\text{old}}}(o=0|c)}=\frac{1-r(x_0,c)}{1-p_{\pi^{\text{old}}}(o=1|c)}\cdot \pi^{\text{old}}(x_0|c).
\end{gathered}
$$

RL 的每一步要求 $\pi^\*\succ\pi^{\text{old}}\Leftrightarrow \underset{\pi^\*(x_0|c)}{\mathbb{E}}[r(x_0,c)] \geq \underset{\pi^{\text{old}}(x_0|c)}{\mathbb{E}}[r(x_0,c)]$. 我们根据定义可以得到 $\pi^+\succ\pi^{\text{old}}\succ\pi^-$.

这是因为 $\pi^{\text{old}}$ 可以表示为 $\pi^+$ 和 $\pi^-$ 的凸组合:
$$
\pi^{\text{old}}(x_0|c)=p_{\pi^{\text{old}}}(o=1|c)\pi^+(x_0|c) + [1-p_{\pi^{\text{old}}}(o=1|c)]\pi^-(x_0|c), \tag{1}
$$

我们只需要说明 $\pi^+\succ\pi^-$.

我们记 $p_{\pi^{\text{old}}}(o=1|c)=\underset{\pi^{\text{old}}(x_0|c)}{\mathbb{E}}\left[r(x_0, c)\right]=\tilde{r}$, 那么:

$$
\begin{gathered}
\underset{\pi^+(x_0|c)}{\mathbb{E}}\left[r(x_0, c)\right]=\frac{\underset{\pi^{\text{old}}(x_0|c)}{\mathbb{E}}\left[r^2(x_0,c)\right]}{\tilde{r}}=\frac{\mathbb{E}\left[r^2\right]}{\tilde{r}},
\\\\
\underset{\pi^-(x_0|c)}{\mathbb{E}}\left[r(x_0, c)\right]=\frac{\tilde{r}-\underset{\pi^{\text{old}}(x_0|c)}{\mathbb{E}}\left[r^2(x_0,c)\right]}{1-\tilde{r}}=\frac{\tilde{r}-\mathbb{E}\left[r^2\right]}{1-\tilde{r}},
\\\\
\frac{\mathbb{E}\left[r^2\right]}{\tilde{r}}-\frac{\tilde{r}-\mathbb{E}\left[r^2\right]}{1-\tilde{r}}=\frac{\mathbb{E}\left[r^2\right]-\tilde{r}^2}{\tilde{r}(1-\tilde{r})}\geq 0.
\end{gathered}
$$

继而 $\pi^+\succ\pi^-$. 那么很显然, 一个简单可行的微调策略就是每次用 $\pi^{\text{old}}$ 生成若干张图片, 根据 $r(x_0, c)$ 的打分情况构造 $D^+$, 并引导模型走向这个分布. 例如 <a href="https://arxiv.org/abs/2302.12192v1">Aligning Text-to-Image Models using Human Feedback</a> 中也明确提到, 用这种拒绝采样的方法效果提升很大, 具体实现方法是从生成的 $16$ 张图片中选取 top-$4$ 进行训练. 这种方法也别称作 Rejection FineTuning (RFT).

然而作者 argue, 并希望能够充分利用 $D^-$ 内的样本. 为此, 我们需要更进一步地了解 $\pi^+,\pi^{\text{old}},\pi^-$ 的性质.

我们已经利用了它们的凸组合性质, 下面要证明它们的后验分布也是凸组合.

$$
\begin{gathered}
\pi^{\text{old}}(x_0|x_t,c)=\alpha(x_t)\pi^+(x_0|x_t,c)+[1-\alpha(x_t)]\pi^-(x_0|x_t,c)\\\\
\text{where}\quad\alpha(x_t)=\frac{\pi^+_t(x_t|c)}{\pi^{\text{old}}_t(x_t|c)} \underset{\pi^{\text{old}}(x_0|c)}{\mathbb{E}}\left[r(x_0,c)\right].
\end{gathered}
$$

<details open>
    <summary>这一步的目的是为了和最优速度场相联系, 从而得到训练目标.</summary>

用 $\pi^\*(x_0|c)=\displaystyle\frac{\pi^\*_t(x_t|c)\pi^\*\_{0|t}(x_0|x_t,c)}{\pi(x_t|x_0)}$ 替换式 (1), 得到

$$
\begin{aligned}
\frac{\pi^{\text{old}}\_t(x_t|c)\pi^{\text{old}}\_{0|t}(x_0|x_t,c)}{\pi(x_t|x_0)}=p\_{\pi^{\text{old}}}(o=1|c)\frac{\pi^+\_t(x_t|c)\pi^+\_{0|t}(x_0|x_t,c)}{\pi(x_t|x_0)}+p_{\pi^{\text{old}}}(o=0|c) \frac{\pi^-\_t(x_t|c)\pi^-\_{0|t}(x_0|x_t,c)}{\pi(x_t|x_0)}.
\end{aligned}
$$

从而得到了

$$
\pi^{\text{old}}\_{0|t}(x_0|x_t,c)=p_{\pi^{\text{old}}}(o=1|c)\frac{\pi^+\_t(x_t|c)}{\pi^{\text{old}}\_t(x_t|c)}\pi^+\_{0|t}(x_0|x_t,c) + p_{\pi^{\text{old}}}(o=0|c)\frac{\pi^-\_t(x_t|c)}{\pi^{\text{old}}\_t(x_t|c)}\pi^-\_{0|t}(x_0|x_t,c).
$$

在式 (1) 的两边同时作用 $\displaystyle\int\pi(x_t|x_0)(\cdot)dx_0$, 我们得到中间状态的凸组合:

$$
\pi^{\text{old}}\_t(x_t|c)=p_{\pi^{\text{old}}}(o=1|c)\pi^+\_t(x_t|c) + [1-p_{\pi^{\text{old}}}(o=1|c)]\pi^-\_t(x_t|c).
$$

并结合 $\tilde{r}$ 符号, 我们也就得到的后验分布的凸组合以及 $\alpha(x_t)$ 的表达式.

</details>

有了这一步, 并结合 flow model 的最优速度场, 我们可以得到在 $D^+$ 和 $D^-$ 下训练的得到的速度场可以凸组合得到 $v^{\text{old}}$.

$$
[1-\alpha(x_t)]\left[v^{\text{old}}(x_t,c,t)-v^-(x_t,c,t)\right]=\alpha(x_t)\left[v^+(x_t,c,t)-v^{\text{old}}(x_t,c,t)\right]=\Delta(x_t,c,t).
$$

从而, 为了实现 $v^*\leftarrow v^{\text{old}} + \displaystyle\frac{1}{\beta}\Delta$, 我们可以同时利用 $v^+$ 和 $v^-$:

$$
\begin{gathered}
\mathcal{L}(\theta)=\underset{c,t,\pi^{\text{old}}(x_0|c)}{\mathbb{E}}\left[ r\|\| v_\theta^+(x_t,c,t)-v \|\|\_2^2 + (1-r) \|\| v\_\theta^-(x_t,c,t)-v \|\|_2^2 \right],
\\\\
\text{where}\quad v^+\_\theta(x_t,c,t)=(1-\beta) v^{\text{old}}(x_t,c,t) + \beta v\_\theta(x_t,c,t),
\\\\
\text{and}\quad v^-\_\theta(x_t,c,t)=(1+\beta) v^{\text{old}}(x_t,c,t) - \beta v\_\theta(x_t,c,t).
\end{gathered}
\tag{2}
$$

且此目标函数训练下的最优速度场为 $v_\theta^*(x_t,c,t)=v^{\text{old}}(x_t,c,t)+\displaystyle\frac{2}{\beta}\Delta(x_t,c,t)$.

<details open>
    <summary>证明如下: </summary>

首先对 $\mathcal{L}$ 进行变形, 用 $\pi^{\text{old}}_{0|t}(x_0|x_t,c)$ 替代 $\pi^{\text{old}}(x_0|c)$, 这是为了和速度场联系起来.

$$
\mathcal{L}(\theta)=\underset{c,t,\pi^{\text{old}}\_t(x_t|c)}{\mathbb{E}} \left\\{
\underset{\pi^{\text{old}}\_{0|t}(x\_0|x\_t,c)}{\mathbb{E}}\left[
r(x_0,c) \|\| v\_\theta^+(x_t,c,t)-v\|\|\_2^2
\right] +
\underset{\pi^{\text{old}}\_{0|t}(x\_0|x\_t,c)}{\mathbb{E}}\left[
(1-r(x_0,c)) \|\| v\_\theta^-(x_t,c,t)-v\|\|\_2^2
\right]
\right\\}.
$$

对内部的期望进行改造:

$$
\begin{aligned}
\textcolor{red}{\pi^{\text{old}}\_{0|t}(x_0|x_t,c)}r(x_0, c) &=
\textcolor{blue}{r(x_0,c) \cdot} \frac{\textcolor{blue}{\pi^{\text{old}}(x_0|c)} \pi(x_t|x_0) }{\pi^{\text{old}}\_t(x_t|c)} \\\\
&=p(o=1|c)\frac{\textcolor{red}{\pi^+(x_0|c) \pi(x_t|x_0)}}{\pi^{\text{old}}\_t(x_t|c)} \\\\
&=\textcolor{blue}{p(o=1|c)\frac{\pi^+\_t(x_t|c)}{\pi^{\text{old}}\_t(x_t|c)}}\cdot \pi^+\_{0|t}(x_0|x_t,c) \\\\
&=\alpha(x_t) \cdot \pi^+\_{0|t}(x_0|x_t,c).
\end{aligned}
$$

同理 $\pi^{\text{old}}\_{0|t}(x_0|x_t,c)[1-r(x_0, c)] = [1-\alpha(x_t)]\cdot\pi^-\_{0|t}(x_0|x_t,c)$. 因此内部的期望可以重写为

$$
\begin{aligned}
\underset{\pi^{\text{old}}\_{0|t}(x\_0|x\_t,c)}{\mathbb{E}}\left[
r(x_0,c) \|\| v\_\theta^+(x_t,c,t)-v\|\|\_2^2
\right]&=
\underset{\pi^+\_{0|t}(x_0|x_t,c)}{\mathbb{E}}\left[
\alpha(x_t)\|\|v\_\theta^+(x_t,c,t)-v\|\|\_2^2
\right]\\\\
&=\alpha(x_t)\left\Vert v\_\theta^+(x_t,c,t) - \underset{\pi^+\_{0|t}(x_0|x_t,c)}{\mathbb{E}}\left[v\right]\right\Vert_2^2 + C_1\\\\
&=\alpha(x_t)\left\Vert v\_\theta^+(x_t,c,t) - v^+(x_t,c,t) \right\Vert_2^2.
\end{aligned}
$$

同理, 另一部分可以写为 $[1-\alpha(x_t)]\Vert v\_\theta^-(x_t,c,t)-v^-(x_t,c,t) \Vert_2^2$. 而利用 $\Delta$ 的定义, 可以继续写为

$$
\begin{aligned}
v\_\theta^+(x_t,c,t)-v^+(x_t,c,t)&=(1-\beta)v^{\text{old}}(x_t,c,t)+\beta v_\theta(x_t,c,t)-v^+(x_t,c,t)\\\\
&=\beta\left[v_\theta-v^{\text{old}}-\frac{1}{\beta}\frac{\Delta(x_t,c,t)}{\alpha(x_t)}\right],\\\\
v\_\theta^-(x_t,c,t)-v^-(x_t,c,t)&=(1+\beta)v^{\text{old}}(x_t,c,t)-\beta v_\theta(x_t,c,t)-v^-(x_t,c,t)\\\\
&=-\beta\left[v_\theta-v^{\text{old}}-\frac{1}{\beta}\frac{\Delta(x_t,c,t)}{1-\alpha(x_t)}\right].
\end{aligned}
$$

因此, 最后 $\mathcal{L}(\theta)$ 可以写为:

$$
\mathcal{L}(\theta)=\beta^2 \underset{c,t,\pi^{\text{old}}\_t(x\_t|c)}{\mathbb{E}}\left\Vert v\_\theta-(v^{\text{old}}+\frac{2}{\beta}\Delta) \right\Vert_2^2 + C.
$$

</details>

因此, 式 (2) 提出了一种 off-policy 的强化学习方法, 并于监督学习相结合. 下面是一些实践中的细节.

首先是 reward 的计算, 借鉴了 GRPO 的思路, 对 reward 做了归一化处理:

$$
r(x_0, c)=\frac{1}{2}+\frac{1}{2} \operatorname{clip}\left[\frac{r^{\text{raw}}(x_0,c)-\operatorname{mean}r^{\text{raw}}}{Z_c}, -1, 1\right],
$$

其中 $Z_c>0$ 是一个和 $\operatorname{std}$ 有关的量, 有点类似于 temperature.

其二是关于 $\pi^{\text{old}}$, 做了 soft EMA: $\theta^{\text{old}}\leftarrow \eta \theta^{\text{old}} + (1-\eta)\theta$. 并且明确指出了 $\eta$ 的取值是训练稳定性和收敛速度的平衡.

另一个小改动是启发式地设定 time weight function $w(t)$. 基模使用 rectifed flow model, 所以可以直接用 $v\_\theta$ 得到 $x\_\theta$.
$$
w(t)\Vert v\_\theta(x_t,c,t)-v \Vert_2^2\leftarrow \frac{\Vert x\_\theta(x_t,c,t)-x_0 \Vert_2^2}{\operatorname{stop-grad}(\operatorname{mean}(\operatorname{abs}(x\_\theta(x_t,c,t)-x_0)))}.
$$

训练采用 $r=32$ 的 LoRA 微调, 每个 epoch 采样 48 个 prompts, 每个 prompt 采样 24 张图片.
