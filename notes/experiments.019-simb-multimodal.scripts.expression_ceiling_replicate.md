---
id: qt1r1fky5qzw6mum3q40cfy
title: Expression_ceiling_replicate
desc: ''
updated: 1785092695134
created: 1785092695134
---

## 2026.07.26 - Replicate-Based Expression Ceiling (0.775), Superseding the p-Value Inversion

Script: `experiments/019-simb-multimodal/scripts/expression_ceiling_replicate.py`
Results: `experiments/019-simb-multimodal/results/expression_ceiling_replicate.json`
Supersedes: `experiments/019-simb-multimodal/scripts/expression_noise_ceiling.py` (0.092)

### 1. The estimand — what `pearson_per_feature` actually averages

Let $\mathcal{S}$ be the strains in an evaluation split and $\mathcal{G}$ the reporter
genes, $|\mathcal{G}| = 6{,}169$. Write $y_{g,i}$ for the measured $\log_2$ ratio of
reporter gene $g$ in strain $i$, and $\hat{y}_{g,i}$ for the model's prediction. The
ranked metric is

$$
\texttt{pearson\_per\_feature}
= \frac{1}{|\mathcal{G}|}\sum_{g\in\mathcal{G}}
  r\!\left(\hat{y}_{g,\cdot},\, y_{g,\cdot}\right),
\qquad
r(a,b)=\frac{\operatorname{Cov}(a,b)}{\sigma_a\sigma_b},
$$

where each $r$ is taken **across strains** $i\in\mathcal{S}$ at fixed $g$, and the
outer average is **over genes**. Two things follow that we must respect: the
correlation axis is strains, and the aggregation is a plain mean over genes.

(The companion metric `pearson_per_instance` transposes this — correlate across genes
at fixed strain, then average over strains. It is a different number: measured
cross-study, $0.666$ vs $0.611$ on the same data. Do not mix them.)

### 2. Measurement model

Decompose each measurement into signal plus independent noise,

$$
y_{g,i} = s_{g,i} + e_{g,i},
\qquad
\operatorname{Var}_i(s_{g,\cdot}) = \sigma_{s,g}^2,
\quad
\operatorname{Var}_i(e_{g,\cdot}) = \sigma_{e,g}^2,
\quad
e \perp s .
$$

Define the per-gene **reliability** (broad-sense $H^2$)

$$
\rho_g \;=\; \frac{\sigma_{s,g}^2}{\sigma_{s,g}^2+\sigma_{e,g}^2}\;\in[0,1].
$$

### 3. Two lemmas — the test–retest correlation is $\rho_g$, the ceiling is $\sqrt{\rho_g}$

**(a) Two independent measurements of the same strains.** Let
$x = s + e_1$ and $y = s + e_2$ with $e_1 \perp e_2$, both of variance
$\sigma_{e,g}^2$. Then $\operatorname{Cov}(x,y) = \operatorname{Var}(s) = \sigma_{s,g}^2$
and $\sigma_x = \sigma_y = \sqrt{\sigma_{s,g}^2+\sigma_{e,g}^2}$, so

$$
r(x,y) \;=\; \frac{\sigma_{s,g}^2}{\sigma_{s,g}^2+\sigma_{e,g}^2} \;=\; \rho_g .
$$

A test–retest correlation estimates the **reliability itself**.

**(b) A perfect predictor scored against a noisy target.** The best possible predictor
of $y$ is $\mathbb{E}[y\mid s]=s$. Its correlation with the measured target is

$$
r(s,y)=\frac{\operatorname{Cov}(s,\,s+e)}{\sigma_s\,\sigma_y}
=\frac{\sigma_{s,g}^2}{\sigma_{s,g}\sqrt{\sigma_{s,g}^2+\sigma_{e,g}^2}}
=\frac{\sigma_{s,g}}{\sqrt{\sigma_{s,g}^2+\sigma_{e,g}^2}}
=\sqrt{\rho_g}.
$$

So $\boxed{\text{ceiling}_g=\sqrt{\rho_g}}$ — **the square root of the test–retest
correlation, not the correlation itself.** At $\rho_g=0.611$ that is the difference
between $0.611$ and $0.782$.

### 4. Aggregation — mean-of-root, not root-of-mean

Because the metric averages per-gene correlations (§1), its ceiling is the average of
the per-gene ceilings:

$$
\text{ceiling} \;=\; \frac{1}{|\mathcal{G}|}\sum_{g}\sqrt{\rho_g}
\;\neq\; \sqrt{\frac{1}{|\mathcal{G}|}\sum_{g}\rho_g}.
$$

By Jensen's inequality ($\sqrt{\cdot}$ concave) the left side is the smaller one.
Numerically $0.7746$ vs $0.7818$ — small here, but it is the correct order and the
one that matches how the metric is computed.

### 5. Estimation — the 82 shared deletions

Kemmeren 2014 (GSE42527/42526, 1,484 single deletions) and Sameith 2015 (GSE42536)
were run independently. **All 82 Sameith single deletions also appear in Kemmeren**,
giving paired independent measurements of the same genotypes. For each reporter gene
$g$, with $x$ = Kemmeren and $y$ = Sameith over those $n=82$ strains,

$$
\hat{\rho}_g = r\!\left(x_{g,\cdot},\,y_{g,\cdot}\right),
\qquad
\widehat{\text{ceiling}} = \frac{1}{|\mathcal{G}|}\sum_g \sqrt{\operatorname{clip}(\hat{\rho}_g,0,1)} .
$$

| quantity | value |
|---|--:|
| $\hat\rho_g$ mean / median | **0.611** / 0.620 |
| $\hat\rho_g$ IQR | [0.521, 0.710] |
| genes with $\hat\rho_g<0$ | 0.1 % |
| **$\text{mean}_g\sqrt{\hat\rho_g}$** | **0.775** |
| per-gene ceiling median / IQR | 0.788 / [0.722, 0.843] |
| observed best (`expr_002`) | 0.109 → **14 % of ceiling** |

### 6. Assumptions, and which were verified

- **A1 — $e_1\perp e_2$ (independent noise).** Cross-study, so strain rebuild, batch,
  operator and hybridisation are all independent. *Not* fully verified: both studies
  share a platform and normalisation lineage, and any shared systematic bias inflates
  $\hat\rho_g$. Direction of error: ceiling too high.
- **A2 — equal noise variance in the two studies.** Assumed. Only matters for the
  variance-decomposition cross-check (§7), not for $r(x,y)=\rho$, which is symmetric.
- **A3 — the 82 shared deletions represent the evaluation split's effect-size
  distribution.** **Verified**: their median across-strain variance is $0.70\times$ the
  full 1,484-strain panel, i.e. slightly *smaller* spread, so if anything $\hat\rho_g$
  is conservative. The initial worry that TF/kinase mutants would be an unrepresentative
  high-effect subset does not hold.
- **A4 — sampling error.** With $n=82$, $\operatorname{SE}(\hat\rho_g)\approx
  (1-\rho^2)/\sqrt{n-3}\approx 0.07$ at $\rho=0.61$. Large per gene; the mean over 6,169
  genes is far tighter, though genes are not independent so the effective $n$ is smaller
  than 6,169.

### 7. Why the superseded estimate failed (0.092), and why it took a measurement to prove

`expression_noise_ceiling.py` never observed a replicate. It *inferred* the noise SD by
inverting a p-value under a normal law,

$$
\hat\sigma_g = \frac{|M|}{\Phi^{-1}(1-p/2)},
\qquad
\hat\rho_g = 1-\frac{\hat\sigma_g^2}{\operatorname{Var}_i(M_{g,\cdot})},
$$

using limma **moderated-$t$** p-values. Two symptoms said it was broken before any
mechanism was known:

1. **It was violated.** Observed $0.109$ against a "ceiling" of $0.092$ — 118 % realized.
2. **Its distribution was degenerate.** Median ceiling exactly $0.0000$, IQR $[0,0]$;
   84 % of genes clipped to zero reliability — while 41.6 % of that same file's p-values
   are $\le 0.05$. Both cannot describe the same data.

The mechanism, measured rather than argued: if the released per-gene SE were a complete
noise model, the observed cross-study disagreement would satisfy
$\operatorname{Var}(x-y) = \sigma_{e,\text{kem}}^2+\sigma_{e,\text{sam}}^2$. Measured
over the 82 pairs,

$$
\frac{\operatorname{Var}_i(x-y)}{\overline{\mathrm{se}^2_{\text{kem}}+\mathrm{se}^2_{\text{sam}}}}
= 0.10 \quad (\text{IQR }[0.08,\,0.15]).
$$

The reported SE overstates real noise by $\approx 10\times$ in variance ($\approx 3.2\times$
in SD). Feeding an inflated $\hat\sigma_g^2$ into $1-\hat\sigma_g^2/\sigma^2_{\text{total}}$
drives the estimate negative and it clips to zero — exactly the median-0 pathology. Since
limma's p is computed from that same SE, the p-value inversion (0.092) and a direct
reported-SE route (0.061) are **one artifact, not two independent estimates**.

Reading: a moderated SE is shrunk toward a gene-wise prior and answers *"how confident is
the DE call"*, not *"how reproducible is this value"*. Using it as a noise model is a
category error.

### 8. Comparison to the morphology ceiling — now methodologically matched

`morphology_noise_ceiling.py` was always replicate-based: $\sigma^2_{e,k}$ from Ohya's
**122 independent his3 WT replicates**, $\sigma^2_{\text{total},k}$ across the 4,718
mutants, $\text{ceiling}_k=\sqrt{1-\sigma^2_{e,k}/\sigma^2_{\text{total},k}}$. With this
note the expression ceiling is estimated the same way (empirical replicate noise), so
the two are comparable:

| | ceiling | median | observed | realized |
|---|--:|--:|--:|--:|
| morphology (CalMorph, 278 feats) | 0.611 | 0.566 | 0.040 | 6.5 % |
| **expression (Kemmeren, 6,169 genes)** | **0.775** | 0.788 | 0.109 | **14 %** |

**Both modalities are early, not saturated.** The previously claimed near-saturation of
expression was an artifact of the inflated SE.

### 9. Consequence for the ceiling as an overfitting diagnostic

The ceiling bounds **generalization**, not fitting. On strains used for fitting, a model
can memorise the realisation $e_{g,i}$ itself, so training Pearson has no bound below 1.
A morphology run reaching train per-feature $\approx 0.9$ against a $0.611$ ceiling is
therefore not evidence against the ceiling — it is positive evidence that the model has
the capacity to fit noise, i.e. that it is memorising. Read together with validation loss
rising from the first epochs, that is memorisation from the start, not late overfitting.

Related: [[experiments.019-simb-multimodal.decoder-distributional-plan]],
[[experiments.019-simb-multimodal.scripts.morphology_noise_ceiling]]
