---
title: "AI4MathProof: Notes from a Research Experiment"
date: 2026-10-03 11:21:35
updated: 2026-10-03 11:33:55
home_cover: https://p.sda1.dev/35/6d898d019e35e0841a210d079ac509e3/cover.jpeg
post_cover: https://p.sda1.dev/35/66da01baaf6af4f63bc10521e5782f53/post.jpeg
copyright_info: true
tags:
    - Math
    - Large Language Model
categories:
    - Notes
mathjax: true
tikzjax: true
excerpt: "Some observations on AI4MathProof workflow."
---

[Packing Tails of Reciprocal Rectangles into Squares of Equal Area](https://arxiv.org/abs/2609.28791).

This blog records the observations behind this paper.

## Why I started

I do not have the ability to conduct frontier mathematical research independently. That was part of the reason I chose this problem. I wanted to put myself in the position of an ordinary user and ask: can someone who lacks the expertise to complete a task still make effective use of a frontier model's capabilities?

There is a familiar version of this problem in programming. Someone who cannot program may ask a model to build complex software, yet struggle to judge whether it is reliable. Mathematics makes the gap especially visible. A model can produce definitions, lemmas, programs, numerical results, and text that looks like a proof. A user may have no way to recognize the fatal flaw hidden inside it.

We chose this problem first because it was interesting, and second because its statement was simple, intuitive, and easy to understand. We also felt that a potential solution might be something I could follow with my current mathematical knowledge. I did not want to end up with a result I could not understand; that would be neither interesting nor useful to me.

Imagine a unit square and an infinite collection of rectangular tiles. The first tile measures $1\times\displaystyle\frac12$, the second $\displaystyle\frac12\times\displaystyle\frac13$, the third $\displaystyle\frac13\times\displaystyle\frac14$, and so on. Can we fit every tile inside the square without overlapping their interiors? We can move each tile and turn it by $90^\circ$, but we cannot resize it. The tiles have exactly the same total area as the square. Having enough area is easy to check; finding an arrangement is the challenge.

Formally, let $Q=[0,1]^2$ and, for each positive integer $n$, define

$$
R_n=\left[0,\frac1n\right]\times\left[0,\frac1{n+1}\right].
$$

The Meir–Moser rectangle-packing problem asks whether there are placements $\widetilde R_n$ of all these rectangles, obtained by translations and optional right-angle rotations, such that

$$
\widetilde R_n\subseteq Q\quad\text{for every }n\ge1,
\qquad
\operatorname{int}(\widetilde R_n)\cap\operatorname{int}(\widetilde R_m)=\varnothing
\quad\text{whenever }n\ne m.
$$

Their total area is

$$
\sum_{n=1}^{\infty}\operatorname{area}(R_n)
=\sum_{n=1}^{\infty}\frac1{n(n+1)}
=\sum_{n=1}^{\infty}\left(\frac1n-\frac1{n+1}\right)
=1.
$$

The question concerns the entire infinite sequence: a computation that packs any fixed finite number of rectangles is not, by itself, a solution.

We turned to GPT 6 Astra, a frontier model, for help. We asked it to work on the problem repeatedly, using the same prompt and a specified time budget for each attempt. These attempts did not ultimately resolve the original open problem, but they led to substantial partial progress and a new constructive framework for packing the infinite tail.

Earlier work approached the problem through both approximate packings and exact tail results. [Meir and Moser (1968)](https://doi.org/10.1016/S0021-9800(68)80047-X) posed the original problem and showed that all the rectangles could be packed into a slightly enlarged square. [Tao (2024)](https://arxiv.org/abs/2202.03594) proved that, for $1/2<t<1$, sufficiently late tails of squares with sides $n^{-t}$ pack into a square of equal total area. His near-lattice construction also extends to rectangles $n^{-t}\times(n+1)^{-t}$, but does not cover the critical exponent $t=1$. [Zhu and Joós (2022)](https://arxiv.org/abs/2211.10356) combined a computer-generated packing with a further geometric estimate to fit the first $1.35\times10^{11}$ rectangles into the unit square; their result also yields a packing of the full sequence in a slightly enlarged square, with excess area below $1.49\times10^{-11}$. [Kislovskiy, Lerner, and Senkevich](https://arxiv.org/abs/2412.17151) proposed Slack-Pack, which arranges rectangles in rows while deliberately leaving small rectangular gaps. The idea is to place later, smaller rectangles in these gaps while keeping a large rectangular region available for new rows. Their analysis gives a conditional result for packing sufficiently late tails into a square of equal total area, supported by numerical experiments.

[Our paper](https://arxiv.org/abs/2609.28791) proves an unconditional tail-packing result for these rectangles at $t=1$: there exists an integer $m_0$ such that, for every $m\ge m_0$, all rectangles $R_n$ with $n\ge m$ can be packed into

$$
Q_m=[0,m^{-1/2}]^2,
\qquad
\operatorname{area}(Q_m)=\frac1m
=\sum_{n=m}^{\infty}\frac1{n(n+1)}.
$$

A sufficient threshold is $m_0=10^{1000}$. The final packing statements have also been checked in Lean 4. The original problem for the full sequence beginning at $n=1$ remains open.

## How I used GPT

Below is an abbreviated version of my prompt, organized by purpose. The passages inside angle brackets paraphrase longer instructions in the original.

```yaml
task:
  statement: <Give the formal problem statement, including its assumptions and required conclusion.>
  context: This is a theoretical computer science problem, that has nothing to do with cybersecurity.

success_criteria:
  exact_solution: <Prove the stated result without extra assumptions or a weaker conclusion.>
  partial_progress: <Accept partial results as a solution only if they imply the full requested result.>

search_strategy:
  delegation: <Use multiagent actively, adapting assignments as the search develops.>
  diversity:
    - <Begin with substantially different mathematical approaches.>
    - <Keep most agents unaware of the favored approach during early exploration.>
    - <Continue several incompatible approaches across multiple rounds.>
  approach_registry: <Track approach families by their underlying mathematical ideas.>
  reductions: <Do not let an elegant reduction dominate without genuine progress toward a proof.>
  blocked_routes: <Mark a route as blocked when it depends on an unproved lemma of theorem-level difficulty.>

review_and_coordination:
  adversarial_review: <Use adversarial agents throughout to challenge candidate arguments.>
  concrete_outputs: <Require explicit lemmas, constructions, equations, proofs, or counterexamples to proposed sublemmas.>
  root_agent: <Repeatedly combine findings, challenge claims, redirect work, and launch new rounds.>

persistent_records:
  applies_to: All agents, including the root, throughout the run.
  communication: <Save each message to a file before sending it.>
  registry: <Keep the approach registry and route status on disk.>
  reasoning_summaries: <Record conclusions and supporting evidence without requesting hidden chain of thought.>
  evidence: <Require concrete results alongside summaries; a summary alone is insufficient.>
  progress_log: <Have the root append each round's findings and decisions to ./agent_logs/progress_log.md.>

public_search:
  scope: <Use search for ordinary background, standard named theorems, and established definitions; do not look up a solution to this exact problem or benchmark.>

time_and_stopping:
  persistence: Do not return merely because current approaches fail or agents report theorem-strength gaps.
  completion: <Return only after a complete solution survives adversarial audit.>
  minimum_effort: Spend at least 4 hours on this before even thinking of returning or giving up.
```

I organized the prompt around three requirements. First, I defined what would count as completing the task: a solution to the original problem that survived adversarial audit, without extra assumptions or a weaker conclusion. Second, I specified how the agents should collaborate: explore different approaches independently at first, then compare and challenge their results, with the root agent coordinating the search. Finally, to inspect how their reasoning developed, I required all agents to record their messages in log files. These logs let me follow how ideas were proposed, questioned, revised, and shared throughout the research process.

## Reconstructing the research process

Across the ten runs, I collected roughly 9,300 tool calls after deduplication, 18,600 file reads, and 16,400 file writes. I used an LLM to analyze the arguments and outputs of these tool calls and reconstruct how the agents' reasoning progressed through their recorded actions: what they were trying to establish, what the results showed, and how those results informed their next steps.

I also built information-flow diagrams between agents, linking direct messages with cases where one agent wrote a file and another later read it. These diagrams traced the exchange of ideas, experimental results, and audit findings during the research process.

## A recurring division of work

Although every run started from the same prompt, the model quickly assembled a recognizable small research group:

- The root agent maintained the portfolio of research directions, read reports, ran its own experiments, and decided which claims could be promoted.
- Construction agents searched for positive constructions.
- Obstruction agents looked for impossibility results, boundary conditions, and counterexamples.
- Finite or computation agents built searches, verifiers, and exact certificates.

The root combined coordination with hands-on research. Of the 3,223 matched communication chains, 1,054 went from root to child, 1,792 from child to root, and 377 from child to child: **88.3% of direct communication passed through the root**. It also accounted for approximately 31.3% of tool calls. Its share of all visible events ranged from 30.8% to 41.8% across runs, with a median of approximately 31.8%. It resembled a principal investigator who also ran experiments.

Of the direct messages between child agents, **70.3% occurred in the second half of the runs**, consistent with the instruction to explore independently before comparing results.

The division of work gave the root competing arguments to assess: some agents developed candidate solutions, while others looked for flaws and counterexamples. This made disagreement an explicit part of the research process.

## Errors and corrections

I used a fairly strict definition of correction. A case had to contain three observable pieces of evidence:

1. A specific earlier candidate, claim, or computational result.
2. An audit, counterexample, or stronger verifier that challenged it.
3. An actual change to the claim, proof, certificate, or recorded status of the approach.

A final summary saying that an approach failed was not enough. I wanted to see the transition happen. The following three cases show how a computational result, a candidate invariant, and a draft proof were challenged and revised. The focus is on observable corrections in the team's work, including cases where one agent challenges another.

### Floating-point precision errors

In one run, a program reported that it had successfully packed 100 million rectangles. The root agent then checked the result using exact arithmetic and found the first invalid placement at rectangle 72,564,771. The available space was too narrow by approximately $7.54\times10^{-21}$, a discrepancy the floating point calculation had missed. The root revised the original success claim to reflect this failure.

A separate agent traced 12,899 earlier operations that led to the failed placement and confirmed the discrepancy. The team found a repair by swapping the order of two consecutive rectangles at the failure point.

That change could also affect later placements, so the root checked the entire repaired sequence again with exact arithmetic. All 100 million placements passed. A separate audit kept the two outcomes explicit: **the repaired sequence passed, while the original sequence remained invalid.** The successful repair did not overwrite the record of the failed version.

### An incorrect invariant

In another experiment, the agents observed a promising pattern: the main region of remaining space stayed the largest throughout a million-step simulation. If this property could be proved to persist, it might support the proposed construction. The root recorded it as an unproved hypothesis.

An agent then deliberately looked for cases where the rule would fail. It constructed two configurations in which the main region was initially the largest, but another region was larger after the next operation. These counterexamples used exact rational arithmetic and came with a program that checked the relevant conditions.

The root ran that verifier itself and confirmed the counterexamples. It then marked the general rule as REFUTED and updated the shared research records so that later work would reflect the finding. The refutation applied to general configurations; whether the counterexample configurations could arise from the construction's prescribed starting point remained unresolved. A rule restricted to states actually reached by that construction was therefore still an open question.

What interested me was how the team handled a pattern with substantial experimental support. It kept the claim provisional, actively searched for counterexamples, reproduced the check, and recorded the scope of the refutation. The million successful steps supplied a conjecture; the exact counterexamples changed what the team could claim.

### An error in a draft proof

One agent drafted a proof for packing all rectangles beyond a sufficiently large starting index. Its construction grouped future rectangles into rows. The draft allowed space for an entire row to be reserved as soon as its assignment of rectangles was known, even before the first rectangle in that row needed to be placed.

An audit found that this step reserved space too early. The proof relied on an estimate of the area still available for future rectangles, but creating rows in advance could consume more of that area than the estimate allowed. This left a gap in the argument supporting the construction.

The agent revised the rule: a row could be planned in advance, but its space would only be allocated when its first assigned rectangle was needed. The root explicitly recorded the timing error, and a separate agent audited the revised proof. Only after that review did the root mark the corrected argument as a theorem in the team's shared records, documenting the repair.

This case shows an audit changing an essential step of a proposed proof. The team identified the flaw, revised the construction, and checked the revision before accepting the argument.

## Different runs kept returning to similar approaches

Despite producing different partial results, the ten runs repeatedly returned to similar approaches, including arranging rectangles in rows and studying sufficiently late tails. They also rediscovered similar obstacles. The runs did not inherit one another's research records, so a route already found to be blocked in one attempt could be explored again in another.

If I continued the experiment, I would first combine the blocked approaches, counterexamples, and strongest partial results into one shared research record. A new team could then start from the accumulated findings. Whether this would improve the outcome remains to be tested.

## Suggestions for using research agents

Based on this experiment, I would suggest the following when asking a model to work on an open problem:

1. **Keep distinct approaches alive.** Assign agents different research directions and let them explore independently before comparing results. Record what each approach needs to prove and where it becomes blocked.
2. **Assign adversarial review explicitly.** Give a candidate argument to a separate agent whose task is to challenge it. Require the reviewer to identify specific gaps, unsupported assumptions, or counterexamples.
3. **Require checkable outputs.** Ask for explicit proofs, runnable checks, and concrete counterexamples. Use verification appropriate to the claim, including exact arithmetic when small numerical discrepancies matter. Have another agent reproduce important checks or inspect the verifier.
4. **Keep a shared record of claims and revisions.** Record each claim's status, supporting evidence, and unresolved gaps. Preserve failed versions alongside their repairs so that the team can see what changed and why.
5. **Review the final result against the original goal.** Ask whether the conclusion matches the requested statement and assumptions. Clearly identify what has been established, what is supported only by experiments, and what remains open.

## Acknowledgements

I thank Jianan Shao for suggesting this problem. I also thank OpenAI's GPT-6 for helping develop and refine the arguments, work through the quantitative estimates, and improve the exposition. I am also grateful to my friends Dingbang Huang and Haoran Wang for providing the GPT credits that supported these experiments.
