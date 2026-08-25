---
id: 23txgler3omebqci59os94n
title: Graph Prior Probe
desc: ''
updated: 1787671502063
created: 1787671502063
---

## 2026.07.26 - Graph-prior probe: does network proximity predict which reporters respond

Design for a probe that has NOT been run. It tests the assumption the graph channel rests on:
that deleting gene $X$ perturbs the genes near $X$ on graph $k$. Companion to
[[experiments.019-simb-multimodal.expression-round-retrospective]] and
[[experiments.019-simb-multimodal.multiplicative-perturbation-conditioning]]; the kNN probe it
contrasts against is [[experiments.019-simb-multimodal.scripts.knn_embedding_probe]].

**Status:** unbuilt. No script in `experiments/019-simb-multimodal/scripts/` implements it, and
every number below is a design quantity, not a measurement.

### 1. Notation

- $N = 6607$ genes. The transformer carries **one token per gene**.
- Strain $s$ deletes gene $p(s)$ (single deletions throughout).
- $y_{s,g} \in \mathbb{R}$ = measured $\log_2$ ratio for **reporter** gene $g$ in strain
  $s$. Expression targets are $\sim\!1{,}484$ strains $\times$ $6{,}169$ reporters.
- Node embedding $e_g \in \mathbb{R}^{d_e}$, projected to the initial token
  $h^{(0)}_g = \phi(e_g) \in \mathbb{R}^{d}$.
- $L$ layers, $H = 9$ heads. Attention at layer $\ell$, head $h$ is row-stochastic
  $\alpha^{(\ell,h)} \in \mathbb{R}^{N\times N}$, with $\sum_j \alpha^{(\ell,h)}_{ij} = 1$.
- Graph $k$ has row-normalized adjacency $\tilde{A}_k$, $\sum_j (\tilde{A}_k)_{ij} = 1$.
  There are $K = 9$ (physical, regulatory, tflink, 6 $\times$ string12_0).

**Readout is S0 / per-token:** reporter $g$'s prediction comes from **its own** final token,

$$
\hat{y}_{s,g} = R_\phi\!\left(h^{(L)}_{s,g}\right).
$$

### 2. Where the networks enter: they shape $\alpha$, and nothing else

Graph regularization is a KL penalty pulling **one designated head** toward **one graph**:

$$
\mathcal{L}_{\text{graph}}
= \sum_{k=1}^{K} \lambda_k \sum_{\ell \in L_k}
  \mathrm{KL}\!\left(\tilde{A}_k[i,:] \,\Big\|\, \alpha^{(\ell,h_k)}[i,:]\right),
\qquad
\mathrm{KL}(P\|Q) = \sum_j P_{ij}\log\frac{P_{ij}}{Q_{ij}} .
$$

This is **forward KL** (mode-covering): it forces $\alpha$ to put mass wherever
$\tilde{A}_k$ does. The graph **never injects features** -- it only constrains *where each
token is allowed to look*.

### 3. Consequence -- the model's implicit prior on how a perturbation spreads

Deleting $p(s)$ modifies that gene's token. For reporter $g$ to be predicted correctly,
information about $p(s)$ must **reach $g$'s token**, and attention is the only route.
Stacking $L$ layers composes the attention matrices, so to first order the influence of
gene $X$ on gene $Y$'s final token is

$$
\frac{\partial h^{(L)}_Y}{\partial h^{(0)}_X}
\;\sim\;
\left(\alpha^{(L)}\alpha^{(L-1)}\cdots\alpha^{(1)}\right)_{YX}.
$$

Under the regularizer $\alpha^{(\ell,h_k)} \to \tilde{A}_k$, so on head $h_k$ this becomes

$$
\text{influence}(X \to Y) \;\approx\; \left(\tilde{A}_k^{\,L}\right)_{YX}
$$ -- the **$L$-step random-walk reachability of $Y$ from $X$ on graph $k$.**

That is the entire content of the graph channel, and it is a **falsifiable assumption
about the data**:

> deleting $X$ perturbs the genes that are close to $X$ on graph $k$.

If that is false -- if the genes that actually respond to deleting $X$ are not its graph
neighbors -- then pulling attention toward $\tilde{A}_k$ pushes the model toward the
**wrong prior**, and no value of $\lambda$ rescues it. Tuning $\lambda$ only controls *how
hard* we push toward a target nobody has checked.

### 4. Why the "readout-side probe" was the wrong probe

What I described was, over reporter pairs $(i,j)$, with
$R \in \mathbb{R}^{n_{\text{strain}} \times n_{\text{reporter}}}$, $R_{s,g}=y_{s,g}$:

$$
\cos(e_i,e_j) \quad\text{vs.}\quad r\!\left(R_{:,i},R_{:,j}\right).
$$

That asks *do reporter genes with similar embeddings co-respond?* -- a legitimate question
about the **embedding** on the readout side. But $\tilde{A}_k$ appears **nowhere in it**.
It cannot say anything about the networks, and the networks are what `_006` is spending 6
GPUs on. That was the gap.

On "readout is important after the model trains": right, and the probe is a
**necessary-condition** test, not a substitute for training. If the structure is absent
from the data, no amount of training recovers it; if present, training may still fail to
use it -- which is exactly what the kNN probe showed for the perturbed side. A data-side
probe can only *rule out*, never *confirm*. Still worth a lot when 6 GPUs are committed
for three days.

### 5. The probe that DOES use the networks

**Question.** Does graph proximity on $\tilde{A}_k$ predict which reporters actually
respond to a deletion?

**Data.** For every (strain $s$, reporter $g$) pair we have $y_{s,g}$, with $X = p(s)$.
Response magnitude $m_{X,g} = |y_{s,g}|$, or a $z$-score against the per-reporter noise SD.

**Predictor.** For graph $k$ and walk length $t \in \{1,2,3\}$:

$$
W^{(k,t)} = \tilde{A}*k^{\,t}, \qquad w^{(k,t)}*{X,g} = \left(W^{(k,t)}\right)_{gX}.
$$

**Statistic.** Per deleted gene $X$, the across-reporter rank correlation

$$
\rho^{(k,t)}*X = \mathrm{Spearman}*g\!\left(w^{(k,t)}*{X,g},\; m*{X,g}\right),
$$

reported as the mean over $X$, alongside an AUC form: the probability that a randomly
chosen *responding* reporter ($|y| > \tau$) is closer to $X$ on graph $k$ than a randomly
chosen non-responder. AUC is the readable one -- **0.5 means the graph says nothing.**

**Controls, so the number means something.**

1. **Degree-preserving rewiring** (configuration model): destroys topology, keeps degree.
   Anything above this is real structure rather than a hub artifact.
2. **Random graph** at matched edge count -- the floor.
3. **Per graph, not pooled**, so the 9 are *ranked*. The regularizer spends one head each;
   if 3 of 9 carry the signal, the other 6 heads are constrained toward noise and are
   worse than free heads.

**What each outcome implies.**

| result | reading |
|---|---|
| AUC $\gg 0.5$ for several graphs | the prior is sound; `_006`'s widening is well-founded; $\lambda$ is the right remaining knob |
| AUC $\approx 0.5$ everywhere | the KL target is the wrong prior; more graphs and more $\lambda$ cannot help, and the 9-head budget is wasted |
| AUC $\gg 0.5$ for a few only | regularize just those heads, leave the rest free |

**Cost.** $\tilde{A}_k$ dense is $6607^2 \approx 4.4\times10^7$ floats $\approx$ 175 MB in
float32; $t \le 3$ is three matmuls per graph. Minutes on CPU, no GPU needed. Read-only
against the existing LMDB -- touches nothing `_006` depends on.

### 6. Three open hypotheses, and their order

None of these could be addressed by the kNN probe:

1. **DNA $\times$ network** (the reason not to drop the DNA embeddings): a promoter
   embedding may only pay off *given* the regulatory graph. **Being tested right now** -- `fudt_upstream` and `nt_window_5979` are in `_006`'s `NODE_EMBEDDINGS` alongside 9
   enforced graphs, for the first time (every earlier round had $\lambda^2 \approx$ off).
2. **Readout-side embedding** (§4): does a reporter's own promoter predict its response
   profile. Untested, graph-free.
3. **Is the graph prior even right** (§5): does $\tilde{A}_k$ proximity predict response.
   Untested, and it **gates** whether the graph channel can work at all.

Value order: **§5 first** -- it is the assumption the running 6-GPU sweep rests on, and it
is the cheapest of the three.

### Carried over from the scratch original

`nt_window_three_prime_5979` still fails to load: `Invalid model_name
'window_three_prime_5979'`. The `nt_` prefix is lost before `BaseEmbeddingDataset`'s
`MODEL_TO_WINDOW` check.
