# Agent 04: the masked-reveal objective in 019

Read of `experiments/019-simb-multimodal/scripts/train_cgt_multitask.py`,
`torchcell/losses/distributional.py`, `torchcell/models/equivariant_cell_graph_transformer.py`
(`ObservedLabelEncoder`), `conf/cgt_expr_v9_mask.yaml` and the six oracle/diagnosis notes,
plus fresh W&B history pulls (entity `zhao-group`) made 2026-09-17 for this review.

Everything below that is a number was either read out of the repo or pulled from W&B in this
session. Curve statistics use a centered 25-epoch rolling mean unless stated; a max over
epochs is an upward-biased order statistic and is labeled as a peak.

---

## 0. What the objective actually is, in code

`MultitaskCGTTask._masked_step` (train_cgt_multitask.py, lines 1226 to 1313):

1. One **extra no-grad forward** of the whole batch (`with torch.no_grad(): _, reps0 = self(batch)`)
   is run first, purely to decode the targets and shapes.
2. A per-row random key `scores` is drawn, NaN entries set to `+inf`, and
   `_observed_feature_mask` takes the `n_reveal` smallest, so observed sets are nested across k.
3. Training: `ks = [int(torch.randint(len(self.mask_schedule), (1,)).item())]` -- **exactly one k,
   sampled uniformly, per batch.** Validation and test: `ks = list(range(len(...)))`, all k.
4. The revealed true values (in the per-feature z-scored training space) are scattered into token
   space and fed through `ObservedLabelEncoder`, which adds `gate * proj([value*mask, mask])` to
   every post-perturbation gene token.
5. The loss is `feature_masks={head: ~obs_feat}`, so only still-hidden genes are scored
   (`masked_mean` in distributional.py). `verify_masked_objective.json` contract C3 pins the
   revealed-gene gradient at exactly 0.0.
6. At k = 0 the run additionally caches the standard `val/expression/pearson_per_feature`, which is
   the leaderboard metric and the `metric_monitor` checkpoint key.

Config lineage: `cgt_expr_v9_mask.yaml` sets `mask_schedule: [0, 10, 100, 1000]`, and
**v10, v11, v12, v13, v14, v15 all inherit it**. W&B confirms: 68 + 12 + 42 + 24 + 16 + 4 = 166
runs in v10 through v15, and every single one carries `mask_schedule [0, 10, 100, 1000]`. No
unmasked run exists anywhere after v9.

One code note, not a bug but the docstrings overstate it. `ObservedLabelEncoder.forward` with an
empty observed set feeds `[0, 0]` into a two-layer MLP with biases, so the output is a **constant
nonzero vector added to every gene token**, not zero. The claims in the model docstring
("the forward pass is identical to the unconditioned model") and in the trainer comment at line
1182 are true only up to that learned constant offset. It is the same constant at k0 and at
validation, so k0-versus-val comparability is unaffected; a comparison against a model built
without the encoder is not exactly an identity.

---

## 1. The training signal at k0 versus k>0, and the gradient-step budget

**Literal step accounting.** v13 expression: 1,244 supervised train records, batch 32,
158,879 optimizer steps over 4,073 epochs = **39.0 steps/epoch**. With `len(mask_schedule) = 4`
and a uniform draw, **25% of optimizer steps (about 9.8 per epoch) run with an empty observed
set**, which is the k0 pathway the leaderboard scores.

**But "25% of steps train k0" is the wrong frame, and this matters.** The loss at step k is
taken over the *hidden* genes, which is 6127, 6117, 6027, 5127 genes at k = 0, 1, 2, 3 out of
6,127 measured. So **every step supervises genotype -> expression on at least 83.7% of the gene
set**. What the k>0 steps add is an input shortcut, not a removal of the genotype task.

**Whether that shortcut is actually used is measurable, and after a few hundred epochs it is not.**
The `traineval` pass (eval mode, dropout off, over the train loader, `train_eval_every: 10`) runs
the full k sweep, so `traineval/expression/pearson_per_feature@k{0..3}` says how much the revealed
labels buy *on the data the gradient sees*. Run `wq8y8nd5` (V_ref_s0_seed0):

| epoch | traineval @k0 | @k1 | @k2 | @k3 | gap k3 - k0 |
|--:|--:|--:|--:|--:|--:|
| 100 | 0.047 | 0.243 | 0.283 | 0.292 | **+0.245** |
| 200 | 0.189 | 0.316 | 0.366 | 0.377 | +0.188 |
| 500 | 0.379 | 0.387 | 0.404 | 0.413 | +0.034 |
| 1,000 | 0.629 | 0.629 | 0.630 | 0.631 | **+0.002** |
| 2,000 | 0.727 | 0.727 | 0.727 | 0.727 | 0.000 |
| 4,000 | 0.773 | 0.773 | 0.773 | 0.773 | 0.000 |

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/wq8y8nd5

Same shape on the proteome (`wg15r736`, P_ref_s2_seed0): traineval @k0 / @k3 is 0.154 / 0.429 at
epoch 100, 0.565 / 0.583 at 500, 0.610 / 0.615 at 1,000.

https://wandb.ai/zhao-group/torchcell_019_prot_v14/runs/wg15r736

**Reading.** Up to about epoch 300 the reveals carry a large share of the training signal. After
about epoch 1,000 the trunk reproduces the training strains' whole profile from genotype alone, so
the revealed values are redundant input on train and carry no residual to explain. From there the
masked objective is **inert**: the k>0 steps are, in gradient terms, plain supervised steps on a
random 84 to 100% subset of the genes. So over the bulk of a 4,000-epoch v13 run or a 10,000-epoch
v9 run, the answer to "what fraction of steps train the k0 pathway" is effectively **all of them**,
and over the first few hundred epochs it is **25%**.

---

## 2. Was a no-mask control ever run under the current trunk/head?

**No. Not once.** 166 runs across v10 to v15 are all masked (W&B config audit above). The only
no-mask control in the masked era is the v9 `M_off` arm, and it is the *old* trunk:
`calm` embeddings only, 1,194,509 parameters, shared-MLP readout. The current trunk (v13) is
`E_full` = [fudt_upstream, calm, prot_T5_all, fudt_downstream], 6,671,629 parameters, and v13 also
carries the `H_concat` readout as its second arm. So the honest statement is:

- **Measured, n = 1, old trunk, and null**: `M_off` (no `mask_schedule`) versus the seven masked
  schedule arms, all seed 0, all 9,999 epochs, all on GilaHyper.
- **Never measured at the current trunk/head.** This is the single biggest confound in the strand,
  and it has been carried silently through six project versions.

The v9 mask round, recomputed here (5-epoch centered rolling mean of
`val/expression/pearson_per_feature`, plus first epoch reaching a threshold, plus total runtime):

| arm | run | schedule | k0 share | roll_max @ epoch | ep to 0.10 | ep to 0.15 | ep to 0.18 | runtime |
|---|---|---|--:|---|--:|--:|--:|--:|
| M_off | `tow1z48n` | none | 1 | 0.2045 @ 5,906 | 71 | 771 | 1,168 | 61.9 h |
| M_sched | `8r5ewoaq` | [0,10,100,1000] | 1/4 | 0.1890 @ 3,742 | 149 | 531 | 3,057 | 76.9 h |
| M_nomix | `rb3bhryq` | [0,10,100,1000], mixing OFF | 1/4 | 0.2109 @ 4,444 | 63 | 248 | 985 | 86.8 h |
| M_fine | `hx8pxdic` | [0,10,30,100,300,1000] | 1/6 | 0.2427 @ 9,188 | 148 | 550 | 1,225 | 91.4 h |
| M_coarse | `ebkzn1ao` | [0,100,1000] | 1/3 | 0.2129 @ 9,682 | 133 | 600 | 4,182 | 83.8 h |
| M_hi | `da5g4o9v` | [0,1000,3000] | 1/3 | 0.1925 @ 3,376 | 115 | 493 | 2,177 | 83.6 h |
| M_lo | `f2wf23oy` | [0,5,10,30] | 1/4 | 0.1858 @ 8,789 | 112 | 865 | 2,222 | 85.6 h |
| M_gate_rezero | `u1vuznme` | [0,10,100,1000], rezero gate | 1/4 | 0.1688 @ 2,201 | 58 | 102 | never | 85.3 h |

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/tow1z48n

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/8r5ewoaq

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/rb3bhryq

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/hx8pxdic

Three things fall out.

- **Cost is measured and it is 24%.** `tow1z48n` (M_off) and `8r5ewoaq` (M_sched) were
  **co-resident in the same slurm job 1444 on the same card**, same seed, same 9,999 epochs, same
  config except the schedule. Wall clock 222,666 s versus 276,815 s, so masking costs **+24.3%**
  at equal epochs. Every masked round since v9 has paid that.
- **The k0 share does not order the k0 score.** 1/6 (M_fine 0.2427) > 1/3 (M_coarse 0.2129) >
  1 (M_off 0.2045) > 1/4 (M_sched 0.1890). One draw per arm against a pooled within-config
  replicate sd of 0.0246 (v10, 16 df), so this is a **null on the k0-budget axis**, not a ranking.
  It directly contradicts the "masking starves the k0 pathway" story.
- **M_nomix is the fastest riser.** With cross-gene mixing off, a revealed value at gene j can only
  touch gene j's own token, and gene j is not scored, so the conditioning is inert by construction
  (this is the model docstring's own argument). That arm reaches 0.15 at epoch 248 versus M_off's
  771 and M_sched's 531, and its roll_max 0.2109 is second best. n = 1; treat as a lead, not a
  result.

The retrospective already recorded the M_off null
(`notes/experiments.019-simb-multimodal.expression-strand-retrospective.md`, 2026.09.08 correction:
"at k = 0 scoring the masked-label objective has not been shown to help. One draw per arm; a
hypothesis until replicated"). It was never replicated, and the trunk changed twice since.

---

## 3. loss@k0 rising while loss@k3 falls (v14); pf@k3 peaking at ~1,000 then falling while k0 rises (v13)

Both observations reproduce exactly. Measured, 25-epoch centered rolling mean:

**v13 expression, `wq8y8nd5`:**

| series | minimum / peak | at epoch | last (ep 4,072) |
|---|--:|--:|--:|
| `val/mask/loss@k0` | 0.2612 | 208 | 0.2792 (+6.9%) |
| `val/mask/loss@k1` | 0.2445 | 277 | 0.2780 |
| `val/mask/loss@k2` | 0.2317 | 618 | 0.2706 |
| `val/mask/loss@k3` | 0.2146 | 955 | 0.2553 |
| `val/expression/pearson_per_feature@k0` | 0.2058 | 3,848 | 0.1969 |
| `...@k1` | 0.3075 | 231 | 0.2032 |
| `...@k2` | 0.3961 | 619 | 0.2401 |
| `...@k3` | 0.4946 | 959 | 0.3174 |

**v14 proteome, `wg15r736`:** loss@k0 min 0.2597 @ 112 then rises to 0.2835 (+9.2%); loss@k3 min
0.1984 @ 773 and still 0.2004 at the end (flat). pf@k0 peaks 0.1074 @ 114 and decays to 0.0902;
pf@k3 peaks 0.5557 @ 773 and holds 0.5481.

Six runs, four expression and two proteome, all the same shape:

| run | k0 peak @ ep | k1 peak @ ep | k2 peak @ ep | k3 peak @ ep | k0,k1,k2,k3 AT the k0 peak |
|---|---|---|---|---|---|
| `wq8y8nd5` V_ref_s0_s0 | 0.206 @ 3,848 | 0.308 @ 231 | 0.396 @ 619 | 0.495 @ 959 | 0.206, 0.213, 0.253, 0.332 |
| `bn37i9vs` V_concat_s0_s0 | 0.200 @ 4,061 | 0.311 @ 263 | 0.396 @ 562 | 0.484 @ 854 | 0.200, 0.208, 0.250, 0.339 |
| `58qsybms` V_ref_s1_s0 | 0.166 @ 4,059 | 0.293 @ 261 | 0.399 @ 677 | 0.498 @ 1,154 | 0.166, 0.172, 0.210, 0.312 |
| `lp6guytz` V_ref_s2_s0 | 0.135 @ 1,745 | 0.315 @ 201 | 0.390 @ 368 | 0.455 @ 897 | 0.135, 0.150, 0.245, 0.431 |
| `uc0pm2pv` P_ref_s0_s0 | 0.116 @ 200 | 0.355 @ 182 | 0.501 @ 303 | 0.565 @ 661 | 0.116, 0.350, 0.484, 0.514 |
| `b95dpt9u` P_ref_s1_s0 | 0.084 @ 446 | 0.311 @ 143 | 0.470 @ 310 | 0.545 @ 906 | 0.084, 0.193, 0.460, 0.534 |

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/bn37i9vs

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/58qsybms

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/lp6guytz

https://wandb.ai/zhao-group/torchcell_019_prot_v14/runs/uc0pm2pv

https://wandb.ai/zhao-group/torchcell_019_prot_v14/runs/b95dpt9u

**What this implies about what the model learns from revealed genes versus from genotype.**

1. **The conditioning capability peaks early and then is destroyed.** On expression, k1 peaks at
   epoch ~230 to 260 and decays to within 0.007 of the k0 curve by the end. Revealing 10 true gene
   values buys the final model **+0.007** of per-feature Pearson. The closed-form linear oracle on
   the same reveal budget buys **+0.408** from a floor of 0
   (`results/masked_conditioning_oracle.json`). The trained conditioner is not weak, it has been
   unlearned.
2. **Section 1's traineval table is the mechanism, and it is measured, not inferred.** The reveals
   stop adding anything on the training set at around epoch 500 to 1,000 (gap +0.245 -> +0.002).
   An input channel that explains no residual on train receives no gradient, so the encoder and
   the mixing path drift. The validation decay of k1/k2/k3 is the visible consequence.
3. **This is not a k0-versus-k3 capacity trade.** Both pathways overfit, on their own clocks:
   loss@k0 bottoms at 208 and rises, loss@k3 bottoms at 955 and rises. The k3 task has far more
   signal per step (it is the easier problem), so it reaches its generalization turn later in
   epochs while decaying faster afterward. The k0 *Pearson* keeps rising on a rising loss for the
   known uncalibration reason (section 4), so a rising loss@k0 is not evidence that k0 degrades.
4. **On the proteome the ordering flips and the genotype pathway is the one that dies.**
   pf@k0 peaks at epoch 114 to 446 and decays while pf@k3 holds. That is consistent with the v14
   note's own read (one measurement per strain, low ceiling, early noise fitting) and means the
   proteome round is currently reporting a k0 number taken from a decayed curve.
5. **Checkpoint consequence, and it is concrete.** The `metric_monitor` checkpoint is
   `val/expression/pearson_per_feature` = the k0 metric, so the saved model is the one at epoch
   ~3,800 to 4,100. At that epoch the run delivers 0.332 at k3 against its own peak of 0.495, so
   **the released checkpoint carries about two thirds of the imputation ability the same run had at
   epoch 959.** If the imputation capability is going to be claimed at all, it is currently being
   thrown away by the selection rule.

---

## 4. Does pinball on z-scored targets produce a shrunken predictor?

**The premise needs correcting: the measured failure is over-dispersion, not shrinkage.**

Setup. Per gene g the target is z-scored on the train split (`standardize_per_feature_target:
[per_gene]`), so var(y) = 1 across strains. Let the point estimate be yhat, with
s = sd(yhat)/sd(y) (`pred_sd_ratio`) and r = corr(yhat, y). Then

    NMSE = E(y - yhat)^2 / var(y) = 1 + s^2 - 2 r s,

a parabola in s minimized at s* = r with value 1 - r^2, and NMSE > 1 exactly when s > 2r.
Measured at the Pearson peak:

| run | objective | r | s | s / r | NMSE | 1 + s^2 - 2rs |
|---|---|--:|--:|--:|--:|--:|
| `tow1z48n` M_off (NO mask) | quantile | 0.2017 | 0.4511 | **2.24** | 1.0359 | 1.0215 |
| `8r5ewoaq` M_sched | quantile | 0.1855 | 0.4584 | 2.47 | 1.0600 | 1.0400 |
| `rb3bhryq` M_nomix | quantile | 0.2082 | 0.4774 | 2.29 | 1.0444 | 1.0291 |
| `hx8pxdic` M_fine | quantile | 0.2378 | 0.4562 | 1.92 | 1.0077 | 0.9912 |
| `wq8y8nd5` v13 V_ref_s0 | quantile | ~0.20 | 0.457 (peak) | ~2.3 | 1.0523 (peak) | |

So predictions are 1.9 to 2.5 times **more** spread than the correlation supports, matching
`expression_objective_diagnosis.json` (1.95x on v9, 2.21x on v8). Crucially the **no-mask control
shows the same 2.24x**, so masking is not the cause of the uncalibration.

**Derivation of why pinball does not correct it.**

(i) *The population optimum is fine.* Pinball is minimized by the true conditional quantile
function. Write y = r u + sqrt(1 - r^2) e with u the model's latent score and e independent, both
standard normal, and let the point prediction be a u. Then y - a u ~ N(0, V(a)) with
V(a) = (a - r)^2 + (1 - r^2). The median-knot pinball is 0.5 E|y - a u| = 0.5 sqrt(2 V(a) / pi),
minimized at a = r, the same place MSE puts it. **The pathology is in the curvature, not the
optimum.**

(ii) *The curvature is roughly half of MSE's, in relative terms.* At r = 0.2, going from a = r to
a = 2r raises V from 0.96 to 1.00. MSE pays V(2r) - V(r) = r^2 = 0.04, which is 4.2% of its own
loss. Pinball pays proportionally to sqrt(V), so it pays (1.000 - 0.980) = 0.0202, which is 2.0% of
its own loss. Halving the relative penalty halves the restoring force on the scale.

(iii) *And the point estimate holds 1/19 of the objective.* The quantile head emits K = 19 knots
and `masked_mean` divides by B x F x K, so the median knot, which is the only output
`DistHead.point()` returns and the only one the metric ever sees, carries **5.3% of the gradient**.
The other 18 knots are fit to spread and nothing ties them to the median. Combining (ii) and (iii),
a 2x scale error on the reported prediction changes the total training loss by roughly 0.1%. That
is the known failure mode: **pinball is nearly blind to the scale of the quantity we score.**

(iv) *The shrunken predictor the question describes is the MSE arm's failure, and it is total.*
The objective round measured `dist: point` (MSE) **collapsing 6 of 6 runs** to r = 0, versus
`crps` 4 of 6 and `laplace_crps` and `quantile` 0 of 6 and 0 of 20
(expression-strand-retrospective, objective round table). Mechanism, **hypothesis (untested) but
consistent with those counts**: on a per-feature z-scored target, the constant predictor 0 is the
conditional mean under an uninformative trunk, sits at NMSE exactly 1.0, and its MSE gradient
toward a strain-dependent signal is proportional to cov(feature, residual), which is O(r) with
R^2 = 0.04. Pinball's gradient is sign-valued and does not vanish there, which is why the quantile
head never collapses. The objective axis is therefore a trade between collapse (MSE) and
over-dispersion (pinball), not a free choice.

(v) *The free fix is arithmetic, already noted in the diagnosis and still not applied.* Multiplying
the point prediction by r/s changes no correlation and moves NMSE from 1.037 to 1 - r^2 = 0.958 on
`wq8y8nd5`. Fit r/s on the tune split, not on val.

---

## 5. Warm-up / curriculum: would annealing change the epoch-to-peak?

**Measured evidence in the v9 mask round: none supports a k0-budget effect.** The eight arms in
section 2 span k0 shares of 1, 1/3, 1/4 and 1/6, and the score does not order by that share; the
best arm has the *smallest* k0 share. Epoch-to-threshold does not order by it either
(M_nomix at 1/4 reaches 0.15 at epoch 248, M_off at share 1 reaches it at 771). One draw per arm
against sd 0.0246, so this is **measured and null at the round's resolution**, not "never tested".

**No curriculum has ever been run.** `mask_schedule` is a static list and `ks` is drawn uniformly
with no epoch dependence (`_masked_step` line 1261). There is no annealing code anywhere.

**Which direction has a mechanism.** The traineval table in section 1 says the reveals carry real
signal only in the first few hundred epochs. So "k>0 first, anneal to k0" is the only direction with
a stated mechanism: it front-loads the phase where the conditioning channel is actually informative
and then stops paying for it. The reverse (k0 first, anneal to k>0) has no measured support, since
by the time k>0 would switch on the trunk has already memorized the train strains and the reveals
add 0.002.

**Hypothesis (untested):** an anneal from k>0-heavy to k0-only would recover most of the 24.3% wall
clock (the extra probe forward and the four-k validation sweep become unnecessary after the anneal)
and would not move the k0 peak, because the k0 peak did not move across a 6x range of k0 budget in
v9. Worth ~12 runs only if the cheaper section-6 experiments say the mask matters at all.

---

## 6. Is the masked objective the right thing for the deployment question?

The deployment question is genotype -> expression for a never-measured strain, which is k0. On the
evidence, the masked objective is **orthogonal to it, by measurement, and it was documented as such
before the build**:

- `conditioning_gain_after_genotype.json`: removing a genuine genotype-conditional predictor
  (kNN on prot_T5, k = 25, leave-one-out) leaves **97.5 / 99.2 / 100.6%** of the conditioning gain
  at m = 10 / 100 / 1000. The config comment in `cgt_expr_v9_mask.yaml` states plainly
  "this is NOT expected to improve the k=0 score".
- `cross_study_conditioning_oracle.json`: about 40% of the within-study oracle is same-array
  technical structure (0.4838 cross-study versus 0.7832 within, against a cross-study ceiling of
  0.611). So part of what a masked objective can learn is Kemmeren's hybridization, which is
  precisely what must not leak into a genotype-only claim.
- `M_off` measured at 0.2045 against the masked arms' 0.1663 to 0.2427: null at n = 1.
- The capability it does buy is, at the checkpoint the leaderboard selects, **+0.007 at m = 10**
  and +0.126 at m = 1000, versus an oracle floor-to-value of +0.408 and +0.793.

**What would be lost by dropping it.**

1. The imputation figure (predict a strain's unmeasured genes from a partial measurement). That is a
   real, separate scientific claim, and the honest target for it is the cross-study 0.4838 at
   m = 1000 capped by 0.611, not the within-study 0.7932.
2. The only reason the Perceiver cross-gene mixing channel exists in the model
   (the model docstring says so explicitly). Note `M_nomix`, which turns mixing off, scored 0.2109,
   the second best of the eight v9 arms, so mixing is not obviously paying for itself either.
3. Nothing else. No result in the strand currently depends on the mask.

**Recommendation.** Do not silently delete it. Demote it: make the default expression/proteome
config unmasked, and keep masking as **one declared arm with its own checkpoint monitor**
(`val/expression/pearson_per_feature@k3`) so the imputation claim is taken at epoch ~950 where it
is real, and the k0 claim at epoch ~3,900 where it is real. Right now one checkpoint is being asked
to serve both and serves the second only.

---

## 7. Cheapest decisive experiments

Ordered by information per GPU-hour. Config-only unless a file and function is named.

### E1. The missing no-mask control at the current trunk. CONFIG ONLY. Run this first.

Three arms on the v13 config, paired within card and seed:

| arm | override | isolates |
|---|---|---|
| `X_mask` | none (inherits `[0, 10, 100, 1000]`) | the incumbent |
| `X_k0only` | `multitask.mask_schedule=[0]` | **the objective alone** |
| `X_nomask` | `~multitask.mask_schedule` (hydra delete, or `mask_schedule: null`) | objective **plus** plumbing |

`mask_schedule=[0]` is the exact control the round never had: `ks` is always `[0]`, `n_reveal = 0`,
`obs_feat` all False, `feature_mask` all True, so the training signal is identical to unmasked
while the probe forward, the `ObservedLabelEncoder` constant offset and the k-sweep validation all
stay. `X_nomask` versus `X_k0only` then prices the plumbing separately. Verified against the code:
`_step` dispatches on `self.mask_schedule is not None` (line 1396), and `main()` only sets it when
`cfg.multitask.get("mask_schedule", None)` is truthy (line 3141), so a null and an empty list both
take the unmasked path.

Design: 3 arms x split seeds {0, 1, 2, 3} x init seed {0} = 12 runs, four per A40 as in v13, so each
card holds three arms of one split plus one spare. Budget 1,400 epochs (the v12 budget), which is
past every measured k1/k2/k3 peak and past the traineval inflection, and which the v13 note says
lands well inside a 48-hour Delta task. Four paired differences at the v13 within-split replicate
sd of about 0.010 to 0.025 resolves about 0.015 to 0.025.

**Score it twice: at matched epochs and at matched wall clock.** Masking costs a measured +24.3%,
so at a fixed GPU budget the unmasked arm gets 1.24x the epochs, and that is the number the strand
actually cares about.

### E2. k0-fraction ladder, replicated. CONFIG ONLY. Only if E1 shows a mask effect.

`mask_schedule` in `[0]`, `[0, 1000]`, `[0, 10, 100, 1000]`, `[0, 10, 30, 100, 300, 1000]`
(k0 share 1, 1/2, 1/4, 1/6), split 0, seeds {0, 1, 2}, 1,400 epochs = 12 runs. This is the v9 round
with replicates, which is the only thing it was missing. v9's unreplicated ranking was non-monotone
in the k0 share, so the prior is a null.

### E3. An imputation checkpoint. CONFIG ONLY, near-zero cost, do it on every masked run from now on.

Set in the round config:

```yaml
trainer:
  checkpoint:
    monitor: val/expression/pearson_per_feature      # k0, unchanged
    metric_monitor: val/expression/pearson_per_feature@k3
```

`metric_monitor` is already read at train_cgt_multitask.py line 3208 and feeds the second
`ModelCheckpoint`, so this needs no code. Without it the imputation claim is reported from a
checkpoint that has lost a third of the capability (section 3, point 5).

### E4. Post-hoc scale calibration. SCRIPT ONLY, no GPU.

Fit `c = r / s` on the tune or val split and rescale the per-gene dumps under
`$DATA_ROOT/.../val-predictions/` and `test-predictions/`. Correlation is invariant, NMSE goes from
1.037 to 0.958 on `wq8y8nd5`. New script in `experiments/019-simb-multimodal/scripts/`, per the
repo rule that any reported number comes from a committed script in the experiment folder.

### E5. Curriculum over k. SMALL CODE CHANGE. Only if E2 says the k mixture matters.

File `experiments/019-simb-multimodal/scripts/train_cgt_multitask.py`:

- `MultitaskCGTTask._masked_step`, the line
  `ks = [int(torch.randint(len(self.mask_schedule), (1,)).item())]` (line 1261): replace the uniform
  draw with `torch.multinomial(self._mask_k_weights(self.current_epoch), 1)`.
- `MultitaskCGTTask.__init__` (near line 932): add `self.mask_k_anneal: dict | None = None`.
- `main()` (near line 3141, beside the existing `mask_schedule` block): read
  `multitask.mask_k_anneal: {start: [...], end: [...], epochs: N}` and assign it.

About 15 lines. Anneal direction: k>0-heavy to k0-only, which is the only direction with a
mechanism (section 5).

### E6. What I would NOT spend cards on

A longer masked run, a finer schedule, or a second gate mode. The v9 round already spans those axes
and is a null at its resolution, and the traineval table says the objective is inert over most of
the epochs being bought.

---

## Summary

**How it works.** `_masked_step` samples one unmasking step k per training batch, uniformly over
`mask_schedule`; validation sweeps all k. Revealed values enter as an additive token encoding and
are excluded from the loss (contract C3: revealed-gene gradient exactly 0.0). k0 is the leaderboard
metric and the checkpoint monitor.

**(1) The step budget.** With `[0, 10, 100, 1000]` and 39 steps per epoch (v13), 25% of optimizer
steps run with an empty observed set. That framing misleads: the loss at k = 3 still covers 5,127
of 6,127 genes, so every step supervises genotype -> expression on at least 83.7% of the genes.
More decisively, the `traineval` pass shows the reveals stop buying anything **on the training
set**: the k3 minus k0 gap is +0.245 at epoch 100, +0.034 at 500, **+0.002 at 1,000** and 0.000
thereafter (`wq8y8nd5`). Once the trunk memorizes the train strains the objective is inert.

**(2) No-mask control.** Never run at the current trunk. 166 runs across v10 to v15 are all masked.
The only control is v9 `M_off` (`tow1z48n`), old trunk, n = 1, 0.2045 against the seven masked arms'
0.1663 to 0.2427: null. Measured cost of masking: `tow1z48n` and `8r5ewoaq` were co-resident on one
card in job 1444 for 9,999 epochs each, 222,666 s versus 276,815 s, **+24.3%**.

**(3) The k0 / k3 divergence.** Both observations reproduce in all six runs I pulled. v13: loss@k0
bottoms at epoch 208, loss@k3 at 955; pf@k3 peaks 0.495 at 959 then falls to 0.317 while pf@k0 rises
to 0.206 at 3,848. v14: loss@k0 rises from epoch 112 while loss@k3 is still flat at its minimum at
the end. This is not a capacity trade; both pathways overfit on their own clocks, and the k>0
pathway decays because it has no residual left to explain on train. Consequence: at the k0-selected
checkpoint the model scores 0.332 at k3 against its own peak of 0.495, and revealing 10 genes buys
**+0.007** against the linear oracle's +0.408.

**(4) Pinball.** The premise is inverted: the measured state is over-dispersion, not shrinkage,
s/r = 1.9 to 2.5 with NMSE > 1, and the **no-mask control shows the same 2.24x**, so masking is not
the cause. Pinball's population optimum does put s = r; the failure is curvature. Its relative
penalty for a 2x scale error is half of MSE's (sqrt(V) versus V) and the median knot carries only
1/19 of the gradient, so a 2x scale error costs about 0.1% of the loss. The shrunken predictor is
the MSE arm's failure: `dist: point` collapsed 6 of 6, quantile 0 of 20. Free fix: rescale by r/s,
NMSE 1.037 -> 0.958.

**(5) Curriculum.** Never run; the code has no annealing. The v9 round spans k0 shares of 1, 1/3,
1/4, 1/6 and the score does not order by it (the best arm has the smallest share), so the k0 budget
is measured not to be the binding constraint. Only "k>0 first, anneal to k0" has a mechanism.

**(6) Deployment.** The masked objective is orthogonal to the k0 question by measurement
(97.5 to 100.6% of the conditioning gain survives removing a genotype predictor), was documented as
such before the build, and its own capability is largely unlearned by the selected checkpoint.
Dropping it costs the imputation figure and the only justification for the Perceiver channel, and
frees 24% wall clock. Demote rather than delete: unmasked default, masking as one declared arm with
its own `@k3` checkpoint.

**Cheapest decisive test:** E1, three arms (`mask_schedule` incumbent / `[0]` / null) x four splits,
1,400 epochs, 12 runs, config-only, scored at matched epochs **and** matched wall clock.

---

Files read or produced:

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/losses/distributional.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/models/equivariant_cell_graph_transformer.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_expr_v9_mask.yaml

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-strand-retrospective.md

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/agent-04-masking.md
