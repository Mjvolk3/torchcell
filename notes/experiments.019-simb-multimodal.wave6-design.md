---
id: tzg8tb5s027abr2ntmkolfe
title: Wave 6 Design
desc: ''
updated: 1787671509266
created: 1787671509266
---

## 2026.07.30 - Wave 6 Experimental Design (019 expression)

**Where this landed.** Components A to C were implemented and run: the arm ladder and its naming
contract are in [[experiments.019-simb-multimodal.scripts.gh_expr_008_arm]], the masked
objective's pre-launch gate in
[[experiments.019-simb-multimodal.scripts.verify_masked_objective]], and the oracle bound of
section 3.2 in [[experiments.019-simb-multimodal.scripts.masked_conditioning_oracle]]. The
measured facts of section 1 come from
[[experiments.019-simb-multimodal.expression-round-retrospective]], and the operator argument
from [[experiments.019-simb-multimodal.multiplicative-perturbation-conditioning]]. Sections 7
and 8, the generalization bound and the per-epoch cost decomposition, are recorded nowhere else,
which is why this design is kept whole rather than folded into those notes.

Design for the fresh restart. One seed, 10,000-epoch cap, arms not replicates. Every
component below is stated as math first; the implementation must match the math and the
math must match the implementation.

### 0. Notation

| symbol | meaning | value here |
|---|---|--:|
| $N$ | gene tokens in the graph | 6,607 |
| $\mathcal{G}$ | measured (reporter) genes | 6,127 |
| $d$ | model width | 90 |
| $H$ | attention heads | 9 |
| $L$ | encoder layers | 4 → **6** |
| $b$ | strain index | 1,244 train |
| $S_b$ | perturbed gene set of strain $b$ | $\lvert S_b \rvert = 1$ |
| $y_{b,g}$ | measured $\log_2$ ratio, gene $g$, strain $b$ | |
| $h_i \in \mathbb{R}^d$ | encoder output for gene $i$ | **strain-independent** |

The last row is the fact everything else hangs on. The encoder runs on the **wildtype**
graph at batch 1, so $h_i$ carries no information about which gene was deleted. All
strain dependence enters at the perturbation operator and later.

### 1. Measured facts that constrain this design

These are already established; the design exists to respect them.

1. **The additive operator has no pair term.** At $|S_b| = 1$ the cross-attention softmax
   runs over a one-element key set, so $\alpha \equiv 1$ and
   $$H^{\text{pert}}_{b,i} = g(h_i + c_b), \qquad c_b = W_O(W_V h_p + b_V) + b_O .$$
   Re-drawing $W_Q, W_K$ at std 10 changes the output by exactly $0.0$ (16,200 of 32,760
   attention parameters dead). Nothing depends on the pair $(p,i)$.

2. **The null sink is a rank-9 fix and came back null.** $n=8$ paired, fixed split:
   $\Delta = -0.0023$, sd $0.0091$, 95% CI $\pm 0.0076$.

3. **Post-perturbation Perceiver mixing alone is null.** $n=8$: $\Delta = -0.0023 \pm
   0.0076$ (`X_mix`), $-0.0014 \pm 0.0080$ (`X_mix_rezero`, gate starts closed). Mixing
   with **nothing requiring it** does not get used.

4. **Nothing had converged at 1,500 epochs.** Eval-mode train Pearson rose to 0.643/0.645
   (dropout 0.1) and 0.721/0.723 (dropout 0), still climbing, max $=$ last in 4 of 4.

5. **Validation dips, then climbs past its early peak.** Smoothed val Pearson peaks
   $\approx 0.14$ at epoch 85-136, falls to 0.08-0.11 by epoch 200-300, then rises to a
   project-best **0.1980 at epoch 1367**. Every arm ever scored at 300-400 epochs was
   scored *in the dip*.

6. **Loss and metric move in opposite directions.** `val/expression/loss` bottoms at epoch
   103-136 then rises, while Pearson climbs throughout; `pred_sd_ratio` goes $0.001 \to
   0.52$. 010 does **not** do this (train and val loss both fall monotonically, val-loss
   min at the last epoch), so it is specific to this target/loss pair.

7. **Reproducibility ceiling is 0.775**, from 82 deletions measured independently in
   Kemmeren and Sameith:
   $$\hat\rho_g = r\big(x_{g,\cdot}, y_{g,\cdot}\big), \qquad
     \widehat{\text{ceiling}} = \frac{1}{|\mathcal{G}|}\sum_g \sqrt{\operatorname{clip}(\hat\rho_g,0,1)} = 0.775 .$$
   **This bounds generalization, not fitting** -- see §7.

8. **Residual gene-gene structure is real, reproducible, and low-rank.** This is the
   quantitative case for the masked objective; see §3.

### 2. Component A -- how the perturbation is asserted

Four arms, differing only in how strain identity reaches gene $i$. Ordered by the **rank
of the pair-$(p,i)$ term** they can express.

**A0 `add` -- additive (reference), pair-rank 0.**
$$H^{\text{pert}}_{b,i} = g\big(h_i + c_b\big), \qquad c_b \in \mathbb{R}^{d}\ \text{shared by all } i .$$
Gene identity and strain identity meet once, by addition. No pair term exists.

**A1 `sink` -- null sink, pair-rank 9.** Append a zero-valued key/value column with a
learned scalar $\beta$ in the attention mask. The real-key weight becomes query-dependent:
$$\alpha_i = \sigma\!\left(\frac{q_i^\top k_p}{\sqrt{d_k}} - \beta\right), \qquad
  c_{b,i} = \alpha_i \, W_O W_V h_p .$$
Per head this is one bounded scalar, so across $H=9$ heads gene $i$ receives exactly **9**
pair-dependent numbers against a 90-dimensional gene-independent context. Measured: the
numerical rank of the gene-dependent context variation is exactly 9.

**A2 `film` -- FiLM at the readout, pair-rank $d = 90$.**
$$\tilde h_{b,i} = h^{\text{pert}}_{b,i} \odot \big(1 + \gamma(z_{S_b})\big) + \beta(z_{S_b}),
  \qquad \gamma,\beta \in \mathbb{R}^{d},$$
with the generator's final layer zero-initialized so the block is the **exact identity at
init** ($\gamma = \beta = 0 \Rightarrow \tilde h = h$). The product $h_i^\top
\operatorname{diag}(\gamma)w$ is a genuine rank-$d$ bilinear interaction. Conditioner input
must be $z_{S_b}$ **only** -- the current code passes $[z_{S_b}; h_{\text{CLS}}]$ and
$h_{\text{CLS}}$ is byte-identical across strains (across-strain sd $0.0$ vs $0.973$ for
$z_S$), so half the conditioner is fed a constant.

**A3 `hadamard` -- multiplicative at the operator, pair-rank $d = 90$.**
$$H^{\text{pert}}_{b,i} = g\big(h_i \odot (1 + \gamma(c_b))\big), \qquad \gamma \in \mathbb{R}^d .$$
The perturbation is **asserted** into every gene's channels rather than retrieved by
similarity. Same identity-at-init requirement. This is the closest reading of "Hadamard
assertion instead of cross-attention".

*Why zeroing $h_p$ is not on this list.* Zeroing after the transform changes one row of
6,607, and $\approx 99.98\%$ of scored predictions are unaffected; measured, the
self-indicator arm gave $-0.0010$ at $n=1$. Zeroing *before* the transform does change
$c_b$ for every gene (since $h_p$ is the K/V), but $c_b$ is gene-independent, so it still
cannot create a pair term. Neither site buys pair-dependence.

### 3. Component B -- the masked objective (the priority)

#### 3.1 Why it should work: measured residual structure

Let $R_{b,g} = y_{b,g} - \mu_g$ be the residual after removing each gene's mean. From
`residual_covariance_diagnostic.json` ($n_{\text{strains}} = 1482$, $n_{\text{genes}} = 6169$):

| quantity | value |
|---|--:|
| split-half $r$ of $\operatorname{offdiag}\operatorname{corr}(R)$, kNN baseline | **0.8687** |
| permutation null | $8.45\times10^{-5}$ |
| effective rank | **32.78** |
| variance in rank-32 subspace | 59.1 % |
| variance in rank-128 subspace | 76.0 % |

So the residual correlation pattern **replicates on held-out strains** and lives in
$\approx 33$ effective components. A per-gene independent readout emits 6,127 marginals
and discards all of it. Masked conditioning is precisely the mechanism that can use it:
observing $m$ genes constrains the $\approx 33$ latent components, which constrain the
other $\approx 6{,}000$.

**This also sets the reveal schedule.** To identify $\approx 33$ components you need
$m \gtrsim 33$ observations. Hence $m = 10$ (under-determined, $10 < 33$), $m = 100$
(over-determined $3\times$), $m = 1000$ (saturated).

#### 3.2 The simpler test to run first -- no GPU, no training

Before building the objective, bound what it can possibly buy. Under a Gaussian model
$R \sim \mathcal{N}(0,\Sigma)$ with $\Sigma$ estimated on **train** strains, the best
possible use of an observed set $\mathcal{M}$ is the conditional mean
$$\mathbb{E}\big[R_{\mathcal{U}} \mid R_{\mathcal{M}}\big]
  = \Sigma_{\mathcal{U}\mathcal{M}} \, \Sigma_{\mathcal{M}\mathcal{M}}^{-1} R_{\mathcal{M}} .$$
Evaluate its `pearson_per_feature` on the **val** strains at $m \in \{0,10,100,1000\}$.
This is minutes of CPU and it is an **oracle upper bound** on any masked-conditioning
architecture that uses only linear residual structure.

- If the oracle is $\approx 0.20$ (today's val), masked conditioning cannot help and we
  should not build it.
- If the oracle rises steeply in $m$, the ceiling is real and the schedule above is
  calibrated to it.

Run this first. It is the cheapest decisive experiment in the round.

#### 3.3 The objective

One label block for now ($\mathcal{Y}_n$ = expression); $\mathcal{Y}_u$ (whole-cell) and
$\mathcal{Y}_e$ (edge/interaction) are deferred -- noted in §9.

Two mask types, and conflating them is the main correctness risk:

- **Hard mask** $\mathcal{H}_b$ -- structurally absent for strain $b$ (never measured).
  Never predicted, never scored, never revealed. For `fig3_core` expression
  $\mathcal{H}_b = \emptyset$, but the code path must exist for multimodal.
- **Soft mask** -- deliberately hidden for the objective, revealed as $k$ grows.

Let $\mathcal{M}_k$ be the **observed** set at step $k$, nested
$$\emptyset = \mathcal{M}_0 \subset \mathcal{M}_1 \subset \mathcal{M}_2 \subset \mathcal{M}_3
  \subset \mathcal{M}_4 = \mathcal{G}\setminus\mathcal{H}_b,$$
with $|\mathcal{M}_k| = m_k$, $m = (0, 10, 100, 1000, |\mathcal{G}|)$ drawn uniformly at
random per strain per epoch. The scored set at step $k$ is everything still hidden:
$$\mathcal{U}_k = \mathcal{G} \setminus (\mathcal{M}_k \cup \mathcal{H}_b).$$

**Teacher forcing.** The values fed back at step $k$ are ground truth $y_{b,g}$ for
$g \in \mathcal{M}_k$, not the model's own predictions.

**Loss.** Scored only on hidden entries -- this is the part that must not be got wrong,
because scoring revealed entries lets the model copy them and inflates train fit for free:
$$\mathcal{L}_b = \sum_{k=0}^{K} w_k \,\frac{1}{|\mathcal{U}_k|}
  \sum_{g \in \mathcal{U}_k} \ell\!\left(\hat y^{(k)}_{b,g},\, y_{b,g}\right),
  \qquad w_k \ge 0,\ \textstyle\sum_k w_k = 1 .$$
Default $w_k$ uniform over the $K+1$ steps. $\ell$ stays the quantile loss (§6).

**How observed values enter the tokens.** With $m_{b,i} = \mathbb{1}[i \in \mathcal{M}_k]$,
$$h^{\text{obs}}_{b,i} = h^{\text{pert}}_{b,i}
  + g_{\text{obs}} \cdot \mathrm{MLP}\big(\big[\, y_{b,i}\, m_{b,i},\; m_{b,i} \,\big]\big).$$
The second channel is load-bearing: without the indicator, "observed and happens to be 0"
is identical to "not observed".

**Inference / validation is unchanged.** At $k=0$, $\mathcal{M}_0 = \emptyset$, every
encoded feature is zero, and the forward pass is *identical* to today's unconditioned
model. So `val/mean/pearson_per_feature` at $k=0$ stays directly comparable to every
previous arm, and the $k>0$ numbers are a strict addition.

**Mixing is mandatory.** Gene $i$'s prediction is a function of $h^{\text{obs}}_{b,i}$
alone unless something routes between tokens; then $y_{b,j}$ can never reach gene $i$ and
the objective is inert. Every masked arm carries post-perturbation mixing. This is also
the reason fact (3) in §1 is *not* evidence against mixing: mixing was tested with
**nothing requiring it**.

**Open question, to settle by arm.** Whether the perturbation is re-asserted at each
unmasking step. Two variants:
$$\text{(i) once:}\quad h^{\text{obs},(k)} = \mathcal{F}\big(h^{\text{pert}}, \mathcal{M}_k\big),
  \qquad
  \text{(ii) per-step:}\quad h^{\text{obs},(k)} = \mathcal{F}\big(g(h \odot \gamma(c_b)), \mathcal{M}_k\big).$$
(i) applies the perturbation operator once and only re-runs the mixing/readout per step;
(ii) re-applies it inside every step so the deletion is re-asserted alongside each new
evidence set. (ii) costs $K\times$ the operator; (i) risks the perturbation being washed
out by accumulated evidence.

#### 3.4 Metrics during unfolding

Log per step $k$, in project **v8**:
$$\texttt{val/expression/pearson\_per\_feature@k},\quad
  \texttt{spearman@k},\quad \texttt{nmse@k},\quad |\mathcal{U}_k| .$$
The shape of $\text{Pearson}(k)$ *is* the result: flat means conditioning is unused,
rising means the residual structure is being exploited, and its slope between $m=10$ and
$m=100$ is directly comparable to the §3.2 oracle.

### 4. Component C -- where the graph acts

Currently the mask is applied inside the **encoder** at layer $\ell$, as a hard additive
attention mask per head $h$ keyed to relation $E_h$:
$$\text{logits}^{(h)}_{ij} \leftarrow \frac{q_i^\top k_j}{\sqrt{d_k}}
  + \begin{cases} 0 & (i,j) \in E_h \\ -\infty & \text{otherwise.}\end{cases}$$

**C0 `pre`** (current). Acts on $h_i$, which is strain-independent. It can shape gene
representations but **cannot express deletion-neighbor coupling**, because no quantity it
touches depends on $S_b$. This is provable from the architecture, not a hypothesis.

**C1 `post`.** The same masked attention applied to the **perturbed** tokens
$H^{\text{pert}}_b \in \mathbb{R}^{B\times N\times d}$. Now the graph routes strain-dependent
information, so gene $i$ learns about the deletion through its actual neighbors. Requires
new code: a batched masked attention layer after the operator.

**C2 `both`.** C0 and C1 together.

All three keep the organism-transfer property, because the graph enters as *structure on
attention* rather than as an input feature: "no graph" degrades to "no mask" and the model
still runs. This is the criterion that disqualified the propagation arms.

KL regularization on attention stays **off** -- locked, replaced by hard edge masking.

### 5. Component D -- capacity

$L = 6$ (from 4). Fixed for the round, not swept. Rationale: with train fit at 0.64-0.72
and val at 0.20 the deficit is generalization, not capacity, so capacity is set once at a
value 007's top runs support and spent elsewhere.

### 6. Component E -- loss form

Quantile loss retained. 007 settled this at $n \approx 60$ per mode: the five
distributional modes are within noise of each other on accuracy (means 0.017-0.028 against
within-mode sd 0.030-0.040) while calibration *succeeded* (coverage within 0.03 of nominal
at both levels). The distributional axis is not the lever.

One caveat now on record: because the loss and metric diverge (§1.6), `mse` and `nmse` are
logged alongside Pearson, with
$$\texttt{nmse} = \frac{1}{|\mathcal{G}|}\sum_g
  \frac{\sum_b (\hat y_{b,g} - y_{b,g})^2}{\sum_b (y_{b,g} - \bar y_{g})^2},$$
so $\texttt{nmse} = 1$ is exactly "predict each gene's mean". Pearson is invariant to a
global rescale; `nmse` is not. Read together they separate right-ordering from
right-magnitude, which is what the divergence turns on.

### 7. Component F -- regularization, and the generalization bound

**Write this down so it can be referenced later.** The ceiling of §1.7 bounds
**generalization only**. On training strains a model may memorize the noise realisation
$e_{g,b}$ itself, so training Pearson has no bound below 1. Consequently:

- Train Pearson of 0.72 against a ceiling of 0.775 is **not** evidence of approaching a
  wall. It is positive evidence that the model has the capacity to fit noise.
- The meaningful statement is the **validation** one: 0.198 against 0.775 is
  $\approx 26\%$ of achievable.
- Therefore any train/val gap beyond the ceiling is definitionally memorization, and the
  correct response is more regularisation, not more capacity.

Measured and consistent: dropout $0 \to$ train 0.72 / val 0.1415-0.1635; dropout $0.1 \to$
train 0.64 / val 0.1780-0.1980. Paired ref $-$ nodrop $= +0.0145, +0.0565$. **Dropout goes
up, not down**: arms at $\{0.1, 0.2, 0.3\}$.

### 8. Throughput -- hitting 5,000 epochs in 24 h

Target $= 24\times3600/5000 = 17.3$ s/epoch. Measured today: 32 s/epoch (mmli, 2 runs/GPU,
7 workers), 270 s/epoch (mmli, 3 workers -- dataloader-starved, A100 at 0 % util).

Per-epoch cost decomposes as
$$T_{\text{epoch}} \;\approx\; \Big\lceil \tfrac{n_{\text{train}}}{B} \Big\rceil
  \big(C_{\text{enc}} + B \cdot C_{\text{pert}}\big) + T_{\text{val}} + \tfrac{1}{\texttt{train\_eval\_every}} T_{\text{traineval}} .$$

$C_{\text{enc}}$ -- the $L$-layer encoder over $N = 6{,}607$ tokens -- is paid **once per
step regardless of $B$**, because the encoder runs on the wildtype graph. With
$B = 8$ that is 155 steps/epoch, so the design is paying the dominant cost 155 times to
process 1,244 strains.

Levers, in order of leverage:

1. **Batch size $8 \to 32$.** Cuts steps $4\times$ and hence the $C_{\text{enc}}$ term
   $\approx 4\times$. Caveat to state plainly: $4\times$ fewer optimizer updates per epoch
   at fixed lr is a real change to the optimization, not a free speedup -- lr should be
   re-checked, and "epoch" stops meaning the same amount of learning.
2. **Cache the 5,125 processed records in host RAM.** Removes per-epoch LMDB reads over
   GPFS entirely and makes the worker count stop mattering -- which is what makes 3
   runs/GPU viable at all.
3. **`train_eval_every` $= 25$** at a 10,000-epoch budget (still 400 curve points). The
   eval-mode pass costs $\approx 65$ s alone; at every-5 that is $+13$ s on every epoch.
4. **Validate every $v$ epochs** rather than every epoch.

Ordering matters: (2) is a prerequisite for the packing, (1) is the largest single win.
**Neither is a scientific choice**, so both land before any arm runs. If 17.3 s/epoch is
still not reached after (1)-(4), the honest options are fewer runs per GPU or a smaller
epoch target -- not a silently truncated wave.

### 9. Deferred, with intent recorded

- **Multi-block labels.** The general form has $\mathcal{Y}_u$ (whole-cell),
  $\mathcal{Y}_n$ (per-gene), $\mathcal{Y}_e$ (edge/interaction) concatenated to a fixed
  width, hard-masked where a dataset lacks a block, soft-masked by the schedule. Built
  gene-block-only now because `fig3_core` has one label type; the hard/soft distinction is
  implemented from the start so the generalization is a widening, not a rewrite.
- **Self-conditioned (non-teacher-forced) unmasking**, i.e. feeding back predictions
  rather than truth, which turns the scheme into a sampler.
- **Post-perturbation graph masking** is in this round (C1); *graph-structured* mixing
  inside the unmasking loop is not.
