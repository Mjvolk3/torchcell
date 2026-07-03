---
id: jtq206kop1jam9em8cv9b7t
title: 131106 Sga Normalization Sd Se Pvalue
desc: ''
updated: 1783102267916
created: 1783102267916
---

# SGA fitness, normalization, and the genetic-interaction p-value

Goal: understand exactly how Costanzo 2016 / Baryshnikova 2010 go from **colony
sizes** to a **genetic-interaction score** and its **p-value**, and pin down the
relationships between SD, SE, bootstrapped SD, and colony-vs-plate/screen variance.
Sources (sha256-pinned in `[[torchcell.datasets.scerevisiae.costanzo2016.noise-computation]]`):
Baryshnikova 2010 Supplementary Note 1 + Supplementary Software 1 (`compute_sgascore.m`,
`output_interaction_data.m`).

## 1. The measurement hierarchy (this is the crux)

What is physically measured is **colony area**. Nothing else. Fitness is colony
size, normalized. But colonies are **nested**:

$$\text{colony } k \;\subset\; \text{plate / screen} \;\subset\; \text{strain (query or array)}$$

- One **plate/screen** = one query crossed into a 384-array, each pinned in
  **quadruplicate** (4 colonies). A double mutant $(i,j)$ is measured with
  $N_{ij} = 4$ colonies per screen, up to two screens, so $N_{ij}\in\{4,\dots,8\}$.
- Colonies **within** a plate share the same day, plate, spatial position batch,
  nutrient field $\Rightarrow$ their errors are **correlated**. They are
  *technical* replicates / **pseudoreplicates**.
- **Screens** (different days/plates) are the level that captures biological +
  batch variation. They are the *independent* replication level.

Everything below hinges on: **colony $\neq$ independent sample.** The number that
goes under a $\sqrt{n}$ depends on *which level* you are averaging over.

## 2. Normalization: colony size $\to$ interaction estimate

Multiplicative model (Supp. Note 1, Eq. 1):

$$C_{ij} = \gamma\, f_{ij}\, t\, s_{ij}\, e, \qquad f_{ij} = f_i f_j + \varepsilon_{ij}$$

where $C_{ij}$ = colony area, $f_{ij}$ = double-mutant fitness, $t$ = time,
$s_{ij}$ = systematic factors, $e$ = log-normal error, and $\varepsilon_{ij}$ =
the **genetic interaction** (the thing we want). Because $f_{ij}=f_i f_j+\varepsilon_{ij}$,

$$\boxed{\;\varepsilon_{ij} = f_{ij} - f_i f_j\;}\qquad(\text{observed} - \text{expected, multiplicative})$$

Normalization removes the nuisance factors so residual colony size reflects only
$\varepsilon_{ij}$:

1. **Plate normalization** (Eq. 3): divide each colony by the *plate middle mean*
   $\mathrm{PMM}_k$ (mean of the middle 60% of colonies). Since $\mathrm{PMM}_k\approx
   f_j t_k$, this removes the query fitness $\times$ time effect at once. Also
   spatial (row/col), competition, and batch corrections.
2. **Colony residual** (Eq. 12): for replicate colony $k$,
   $$R_{ijk} = \tilde C'_{ijk} - \operatorname{median}_{j}\tilde C'_{ijk}$$
   i.e. subtract the array strain's **expected** size (its median across *all*
   queries crossed into it = the empirical "no-interaction" background).
3. **Interaction estimate** (Eq. 13): average the residuals over the colonies,
   $$\varepsilon_{ij}\equiv I_{ij} = \frac{1}{N_{ij}}\sum_{k=1}^{N_{ij}} R_{ijk}.$$

## 3. SD vs SE vs bootstrapped SD (the vocabulary you asked for)

Let $x_1,\dots,x_n$ be observations of a quantity with mean $\bar x$.

- **Standard deviation (SD)** — spread of the *observations*:
  $$\mathrm{SD} = s = \sqrt{\tfrac{1}{n-1}\sum_k (x_k-\bar x)^2}.$$
  Describes how variable individual measurements are. Does **not** shrink with $n$.
- **Standard error of the mean (SE / SEM)** — precision of the *estimate* $\bar x$:
  $$\mathrm{SE}(\bar x) = \frac{\mathrm{SD}}{\sqrt{n}}\quad\textbf{if the }x_k\textbf{ are independent.}$$
  Shrinks as $\sqrt{n}$. This is what you want as an ML training weight (inverse-variance).
- **Bootstrapped SD/SE** — resample the data $B$ times (here $B=800$), recompute
  the statistic $\hat\theta^{(b)}$ (mean or median) each time, and take
  $$\mathrm{SE}_{\text{boot}}(\hat\theta) = \operatorname{SD}_b\!\big(\hat\theta^{(b)}\big).$$
  Because it is the SD of the **sampling distribution of the estimate**, a bootstrapped
  SD *of the mean/median* **is already an SE** — you do **not** divide it by $\sqrt n$
  again. It also needs no normality assumption and handles the median.

**The trap** (the whole reason our first reproduction failed): the published SMF
`stddev` column is an **SD of the observations**, but the value used *in scoring*
is the **bootstrapped SEM** (small). Different quantities. Mixing them, or dividing
an already-SE quantity by $\sqrt n$, is the class of error we are fixing in the
schema ontology.

## 4. Two variances for the interaction (colony vs plate/screen)

For every interaction the pipeline computes **two** error estimates:

### 4a. Local colony SD (Eq. 14) — the published `DMF standard deviation`

$$\sigma_{I_{ij}} = \sqrt{\frac{1}{N_{ij}-1}\sum_{k}\big(R_{ijk}-I_{ij}\big)^2}.$$

This is the SD of the $N_{ij}\!=\!4$–$8$ **colony** residuals. The authors warn it
*"can dramatically underestimate the true variance"* — because the colonies are
adjacent pseudoreplicates, $\sigma_{I_{ij}}$ measures within-plate noise, **not**
the real (between-screen) uncertainty. If you (wrongly) treated the colonies as
independent, $\mathrm{SE}\approx\sigma_{I_{ij}}/\sqrt{N_{ij}}$ — but that $N_{ij}$ is
pseudoreplication.

### 4b. Pooled "expected" SD (Eq. 16) — from control screens

Model the null ($\varepsilon_{ij}\!=\!0$) colony size as log-normal with variance
from *both* strains:

$$C_{ij}\sim \alpha f_i f_j\, e^{X},\qquad X\sim\mathcal N\!\big(0,\;\sigma_i^2+\sigma_j^2\big)$$

- $\sigma_i^2$ = **array**-strain variance, measured across many **wild-type control
  screens** (independent-replication level = screen).
- $\sigma_j^2$ = **query**-strain variance, pooled within-array across the query screen.

The reported multiplicative (geometric) SD is

$$\sigma_{ij,\text{expected}} = \exp\!\Big(\sqrt{\sigma_i^2+\sigma_j^2}\Big),
\qquad \log \sigma_{ij,\text{expected}} = \sqrt{\sigma_i^2+\sigma_j^2}.$$

This is the *honest* uncertainty: variance estimated at the **screen** level, not
the colony level. **It is computed from raw control-screen data and is NOT written
to the published output** — only $\sigma_{I_{ij}}$ (4a) is released.

## 5. How the p-value comes about (exact, from `output_interaction_data.m`)

$$p_{ij} = \sqrt{\;\underbrace{\Phi\!\Big(-\Big|\tfrac{\varepsilon_{ij}}{\sigma_{I_{ij}}}\Big|\Big)}_{\text{Term A: local, linear}}\;\cdot\;
\underbrace{\Phi\!\Big(-\Big|\tfrac{\log(f^{\text{act}}_{ij}/f^{\text{exp}}_{ij})}{\sqrt{\sigma_i^2+\sigma_j^2}}\Big|\Big)}_{\text{Term B: pooled, log/geometric}}\;}$$

where $\Phi$ is the standard-normal CDF, $f^{\text{exp}}_{ij}$ = expected DMF
(background mean), and $f^{\text{act}}_{ij}=f^{\text{exp}}_{ij}+\varepsilon_{ij}$ =
actual DMF.

- **Term A**: a one-sided z-test of $\varepsilon$ against its **local colony SD** (§4a), on the linear scale.
- **Term B**: a one-sided z-test of the **log-fold-change** actual-vs-expected DMF
  against the **pooled screen-level SD** (§4b) — log scale because the model is
  multiplicative ($\log\sigma_{ij,\text{expected}}=\sqrt{\sigma_i^2+\sigma_j^2}$).
- **Combine** by **geometric mean** ($\sqrt{\text{A}\cdot\text{B}}$): the interaction
  must look significant *both* against local reproducibility *and* against the
  expected between-screen variability.

**Consequence for us:** Term B needs $\sigma_i^2,\sigma_j^2$, which live only in the
raw control-screen pipeline and are **not** in the released data. So Costanzo's
exact $p$ is **not reconstructable** from the summary tables (our best fit with
Term A alone tops out at $\mathrm{corr}=0.95$ on $-\log_{10}p$). To reproduce it we
would have to re-run `compute_sgascore.m` on the raw colony data.

## 6. Summary — which quantity is which

| symbol | what it is | level | shrinks with $\sqrt n$? | in our data? |
|---|---|---|---|---|
| $s$ (colony SD) | SD of colony residuals, Eq. 14 = `DMF standard deviation` | colony | is an SD (not divided) | yes |
| $\sigma_{I_{ij}}/\sqrt{N_{ij}}$ | naive SE from colonies | colony | pseudoreplicated $\Rightarrow$ too small | derivable, wrong |
| $\sqrt{\sigma_i^2+\sigma_j^2}$ | pooled log-SD, Eq. 16 | **screen** | correct independent level | **no** |
| SMF `stddev` column | SD across screens | screen | is an SD | yes |
| SMF bootstrap SEM | precision of SMF median, $B{=}800$ | screen | already an SE | no (only SD released) |

**One-line takeaways**
- Fitness = normalized colony size; the interaction $\varepsilon = f_{ij}-f_i f_j$.
- The *right* $n$ is **screens**, not colonies (colonies are pseudoreplicates).
- A bootstrapped SD *of an estimator* already IS an SE — never $\div\sqrt n$ it.
- The p-value combines a **colony-level** and a **screen-level** test by geometric
  mean; the screen-level piece is unpublished, so exact $p$ needs the raw pipeline.
