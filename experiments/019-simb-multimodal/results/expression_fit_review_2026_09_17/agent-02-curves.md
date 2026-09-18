# agent-02: training-dynamics forensics from the W&B histories

Question put to me: **"are we suffering from slow runs because masking needs some warm-up to
fit properly?"** Answered below from the actual per-epoch curves of 175 runs across
`torchcell_019_expr_v9/v10/v11/v12/v13/v15` and `torchcell_019_prot_v14` (entity
`zhao-group`).

**Verdict up front: REFUTED as stated, at n = 1 for the decisive control and n = 8 for the
dose-response.** There is no warm-up anywhere in the masked objective to be slow about: the
reveal counts are constants, not a curriculum, and the conditioned branches (k1, k2, k3) are
the *fast* part of the curve, peaking at epochs 214 / 606 / 1,099 while the unconditioned
k0 metric peaks at 3,531. Turning masking off entirely (`M_off`) does not speed the k0 curve
up. What *does* move time-to-plateau is the **readout**: a per-gene output row
(`H_state` / `H_pergene` / `R_pergene`) reaches 90% of its own plateau at mean epoch 234 to 394
where the shared-MLP family needs 810 to 954, a 2.6x difference pooled; on the long v9 budget
the per-gene arm peaks at epoch 1,767 against the reference's 8,414, a 4.8x difference.

Pull code and raw curves (read-only artifacts of this review) are listed at the end.

---

## 0. What the code actually does (read before the numbers)

From `experiments/019-simb-multimodal/scripts/train_cgt_multitask.py` and the config chain
`cgt_expr_008 -> 009/010/011/012 -> v9_mask -> v10/v11 -> v12 -> v13 -> v14/v15`:

| fact | evidence |
|---|---|
| `mask_schedule` is a **static list** of reveal counts, e.g. `[0, 10, 100, 1000]`. It never changes with epoch. | `train/mask/n_revealed@k0` has exactly **1 unique value (0)** and `@k3` exactly 1 unique value (1000) over the whole run, in all 15 runs I pulled it for. |
| Training draws **one k uniformly per batch**: `ks = [int(torch.randint(len(self.mask_schedule), (1,)).item())]`. So with a 4-step schedule only **25% of gradient steps** are the k = 0 regime the leaderboard scores. | train_cgt_multitask.py, `_masked_step` |
| Validation runs the **whole sweep** k = 0..3 every epoch; `pearson_per_feature` (no @k) is an alias for @k0 (verified across 175 runs: the two series agree to a median maximum absolute difference of 6.5e-8, largest 0.0084). | `_masked_step`, `_cache_epoch_metric` |
| **There is no LR schedule and no warmup.** `lr_scheduler.type: null` in 175/175 runs pulled; `lr` = 3e-4 constant, AdamW, `weight_decay` 1e-8 (except the v10 wd arms at 1e-4 and the v15 arms at 1e-2 / 1e-1). `lr` is not logged at all. | run configs; no `lr` key in any run summary |
| `observed_labels.gate_mode: "on"` and `post_perturbation_mixing.gate_mode: "on"` in the v9+ lineage, so there is no ReZero gate that has to open either. | `cgt_expr_v9_mask.yaml` |
| Masking costs **+6% per epoch and +24% total wall clock**, not 2x. | `M_off` 16.3 s/epoch, 61.9 h for 9,999 epochs; `M_sched` 17.3 s/epoch, 76.9 h |

So the literal hypothesis has no mechanism: nothing in the mask path warms up.

**Correction to the premise in the task.** Expression trains on `n_train_supervised` = **1,244**
strains at batch 32, which is **39 steps/epoch**, not 116. 4,000 epochs is **156k steps**, not
460k. The 3,581-strain / 112-steps-per-epoch figure is the **proteome** (v14).

---

## (a) The shape of val Pearson vs epoch

Mean over the 23 non-collapsed v13 runs (one validation draw each, 4 split seeds), rolling
mean w = 9:

| epoch | 10 | 50 | 100 | 150 | 200 | 300 | 500 | 700 | 900 | 1200 | 2000 | 3000 | 4000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val pf@k0 | 0.004 | 0.021 | 0.076 | **0.089** | 0.083 | **0.076** | 0.093 | 0.132 | 0.146 | 0.151 | 0.158 | 0.160 | 0.153 |
| val pred_sd_ratio | 0.000 | 0.003 | 0.043 | 0.097 | 0.141 | 0.193 | 0.270 | 0.358 | 0.402 | 0.440 | 0.465 | 0.469 | 0.467 |
| val pf@k3 | 0.061 | 0.227 | 0.311 | 0.365 | 0.375 | 0.384 | 0.404 | 0.455 | **0.479** | 0.475 | 0.415 | 0.365 | 0.316 |
| val/mask/loss@k0 | 0.260 | 0.260 | **0.258** | 0.258 | 0.259 | 0.260 | 0.263 | 0.267 | 0.270 | 0.276 | 0.280 | 0.281 | 0.283 |
| val/mask/loss@k3 | 0.260 | 0.250 | 0.240 | 0.232 | 0.230 | 0.229 | 0.226 | 0.217 | **0.214** | 0.215 | 0.229 | 0.240 | 0.250 |

**It is a three-phase curve, not a slow monotone rise:** a fast climb to ~0.089 by epoch
150 (55% of the eventual plateau), a **dip and plateau from epoch 150 to 500**, then a second
slow climb that finishes around epoch 1,500 to 2,000.

Time to fractions of each run's own value at the matched reference epoch (median over runs):

| round | n | pf(ref) | ep to 25% | 50% | 75% | 90% | steps to 90% |
|---|---|---|---|---|---|---|---|
| v13 expression (ref ep 3,700) | 23 | 0.162 | 64 | 118 | 594 | **825** | 32,175 |
| v12 expression (ref ep 1,350) | 32 | 0.163 | 54 | 91 | 257 | **426** | 16,594 |
| v14 proteome (ref ep 1,100) | 12 | 0.078 | 16 | 34 | 50 | **57** | 6,384 |

Against each run's own rolling **max** instead (which is the biased order statistic, so it
pushes the 90% point later):

| round | n | median ep to 50% | median ep to 90% | median peak position (fraction of run) | final/max |
|---|---|---|---|---|---|
| v9 | 42 | 140 | 954 | 0.70 | 0.93 |
| v10 | 28 | 90 | 990 | 0.98 | 0.98 |
| v11 | 11 | 101 | 890 | 0.92 | 0.96 |
| v12 | 32 | 103 | 594 | 0.89 | 0.96 |
| v13 | 23 | 128 | 1,149 | 0.87 | 0.95 |
| v15 | 4 | 79 | 327 | 0.86 | 0.79 |
| **v14 proteome** | 16 | **40** | **98** | **0.19** | **0.82** |

v14 is the only round whose peak sits in the first fifth of the run. Every expression round
peaks at 87% to 98% of the way through whatever budget it was given, which is the signature
of a budget-truncated curve rather than of convergence.

---

## (b) Is the rise tied to the reveal schedule, the LR schedule, or the train fit?

**Reveal schedule: no.** `n_revealed@k` is constant (uniq = 1) in every run checked. There is
no curriculum to warm up. What the k-resolved metrics show is the **opposite** of a warm-up:
the more genes are revealed, the **earlier** that branch peaks.

Median peak epoch of `val/<pheno>/pearson_per_feature@k`:

| | k0 (reveal 0) | k1 (10) | k2 (100) | k3 (1000) |
|---|---|---|---|---|
| v13 expression, peak epoch | **3,531** | 214 | 606 | 1,099 |
| v13, peak value | 0.167 | 0.315 | 0.399 | 0.492 |
| v14 proteome, peak epoch | 152 | 156 | 299 | 749 |
| v14, peak value | 0.097 | 0.334 | 0.486 | 0.554 |

**loss@k3 vs loss@k0 trend in opposite directions, and the crossover is at ~epoch 100 for k0
and ~epoch 900 for k3.** `val/mask/loss@k0` bottoms at 0.258 around epoch 100 to 150 and then
degrades monotonically to 0.283 by epoch 4,000, that is, **the k0 pinball loss on held-out
data never improves after epoch 150** while the k0 Pearson doubles from 0.076 to 0.160.
`val/mask/loss@k3` keeps falling until epoch ~900 (0.214) and only then turns. The conditioned
capability is learned fast and then overfits; the unconditioned one is what crawls.

**LR schedule: there is none.** `lr_scheduler.type` is `null` in all 175 runs; lr is a
constant 3e-4 for the entire run. There is nothing to "sit at" when Pearson rises. This axis
is therefore **untestable from the logs as they stand**, and it is the one axis with a real
prior: the perturbation head is `residual: postln` and post-LN transformers are the classic
case that wants LR warmup. `train/grad_norm` is flat at 0.031 to 0.053 across epochs 10 to
9,000 against a clip of 10.0, so nothing is clipping and nothing is exploding.

`train/grad_norm` at epoch (clip = 10.0):

| run | 10 | 100 | 400 | 1,500 | 4,000 | 9,000 |
|---|---|---|---|---|---|---|
| v13 `wq8y8nd5` V_ref_s0 | 0.033 | 0.040 | 0.036 | 0.047 | 0.053 | -- |
| v9 `tow1z48n` M_off | 0.037 | 0.039 | 0.048 | 0.049 | 0.051 | 0.051 |
| v9 `8r5ewoaq` M_sched | 0.035 | 0.041 | 0.042 | 0.049 | 0.053 | 0.054 |
| v9 `vqek7ali` R_pergene | 0.038 | 0.054 | 0.089 | 0.047 | 0.035 | 0.037 |

**Train fit: yes, val lags train by roughly a factor 3 in epochs.** v13 means over 24 runs
(`traineval`, eval mode, dropout off):

| epoch | 100 | 300 | 500 | 1,000 | 2,000 | 3,000 | 3,700 |
|---|---|---|---|---|---|---|---|
| traineval pf | 0.078 | 0.292 | 0.384 | 0.615 | 0.702 | 0.727 | 0.739 |
| val pf | 0.072 | 0.073 | 0.089 | 0.141 | 0.152 | 0.154 | 0.156 |
| val pred_sd_ratio | 0.040 | 0.185 | 0.259 | 0.402 | 0.446 | 0.449 | 0.449 |

Between epoch 300 and 1,000 the train fit goes 0.29 -> 0.62 while val goes 0.073 -> 0.141.
So the honest description is "the whole optimization is slow", not "memorization first, then
a delayed generalization event": both move in the same window, val at about a fifth of the
amplitude.

**The tightest single correlate of the val rise is `pred_sd_ratio`,** the ratio of predicted
to observed standard deviation. Over epochs, corr(val pf, val sdr) is 0.79 to 0.95 in 23 of
24 v13 runs (the 24th is the collapsed run `825on260`, 0.04) (`curves.json`, already in the scratchpad). The model spends its first ~75 epochs
emitting an essentially constant prediction (sdr 0.003 at epoch 50, 0.043 at epoch 100) and
then slowly inflates its output scale to ~0.47 by epoch 2,000, where it stops. The val
Pearson curve is that shrinkage unwinding.

---

## (c) Which runs rose FAST, and what distinguished them

Fastest 14 risers across all expression rounds, restricted to runs with rolling max >= 0.15
so this is not a list of collapsed runs. **12 of the top 13 carry a per-gene output row.**

| ep to 90% | max | round | run | arm | mechanism |
|---|---|---|---|---|---|
| 203 | 0.156 | v12 | `1at9ti2a` | H_pergene | per-gene readout row |
| 221 | 0.173 | v12 | `n3pibzjx` | H_state | per-gene row over strain context |
| 228 | 0.172 | v12 | `1acw1jm4` | H_state | per-gene row over strain context |
| 234 | 0.181 | v12 | `mfgjmq57` | H_state | per-gene row over strain context |
| 238 | 0.181 | v12 | `ahgltcx6` | H_pergene | per-gene readout row |
| 244 | 0.186 | v9 | `pgd9ho8e` | RR_pergene | per-gene readout row |
| 255 | 0.164 | v12 | `3ba6vxof` | H_state | per-gene row over strain context |
| 284 | 0.181 | v12 | `ethkld5d` | H_pergene_basis64 | per-gene row + rank-64 basis |
| 308 | 0.158 | v12 | `s5zs2ytk` | H_gears | per-gene row + cross-gene |
| **311** | **0.219** | v9 | `vqek7ali` | R_pergene | per-gene readout row |
| 343 | 0.196 | v9 | `q3x9qrzu` | RR_pergene | per-gene readout row |
| 449 | 0.169 | v12 | `spc3squv` | H_gears | per-gene row + cross-gene |
| 492 | 0.202 | v12 | `i494qkse` | H_pergene_basis64 | per-gene row + rank-64 basis |
| 536 | 0.180 | v11 | `t8n7rmax` | E_full | (no per-gene row) |

Slowest, same filter: `8r5ewoaq` M_sched 2,927; `hx8pxdic` M_fine 4,188; `r2s57bvt` R_ref
4,907; `0fymu4py` W_ref 10,004; `d94cy5az` W_ref 12,237.

The v12 readout round is the controlled version of this: same 1,399-epoch budget, split seed
0 pinned, 4 init seeds per arm, everything else identical. Collapsed runs (max < 0.10)
excluded.

| family | arm | n | mean ep to 90% | mean max | pf@100 | pf@200 | pf@400 | pf@800 | pf@1350 |
|---|---|---|---|---|---|---|---|---|---|
| per-gene row | H_state | 4 | **234** | 0.172 | 0.094 | 0.142 | 0.131 | 0.153 | 0.163 |
| per-gene row | H_pergene_basis64 | 3 | 363 | 0.177 | 0.072 | 0.118 | 0.153 | 0.140 | 0.165 |
| per-gene row | H_gears | 3 | 340 | 0.157 | 0.073 | 0.111 | 0.137 | 0.128 | 0.142 |
| per-gene row | H_pergene | 4 | 394 | 0.163 | 0.074 | 0.132 | 0.135 | 0.143 | 0.151 |
| shared map | H_basis64 | 3 | 810 | 0.184 | 0.078 | 0.096 | 0.113 | 0.165 | 0.176 |
| shared map | H_linear | 4 | 821 | 0.175 | 0.052 | 0.088 | 0.087 | 0.155 | 0.171 |
| shared map | H_concat | 4 | 845 | 0.186 | 0.105 | 0.086 | 0.101 | 0.163 | 0.181 |
| shared map | H_ref | 4 | **954** | 0.172 | 0.063 | 0.096 | 0.098 | 0.141 | 0.168 |
| | **per-gene pooled** | 14 | **330** (median 265) | 0.167 | | | | | |
| | **shared pooled** | 15 | **860** (median 842) | 0.179 | | | | | |

The train fit moves the same way, so this is an optimization-speed effect and not a
regularization effect: `traineval` Pearson at epoch 400 / 800 is 0.600 / 0.757 for H_state
`mfgjmq57`, 0.449 / 0.707 for H_pergene `ahgltcx6`, against 0.342 / 0.572 and 0.350 / 0.578
for the two H_ref runs `vahi7g0w` and `9pujw6j2`.

**Important caveat, and it cuts both ways.** At the 1,399-epoch cap the shared family has a
*higher* mean max (0.179 vs 0.167) because it is **still climbing** at the cap while the
per-gene family has stopped. At a long budget in v9 the per-gene arms reach a higher peak and
*then decline*:

| arm | run | last ep | ep 100 | 300 | 600 | 1,000 | 1,800 | 3,000 | 5,000 | 8,000 | rollmax @ ep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R_pergene | `vqek7ali` | 8,498 | 0.073 | 0.195 | 0.180 | 0.189 | 0.209 | 0.191 | 0.164 | 0.169 | **0.219 @ 1,767** |
| R_pergene | `4u0txt3s` | 8,498 | 0.058 | 0.150 | 0.168 | 0.172 | 0.169 | 0.168 | 0.149 | 0.139 | 0.188 @ 1,193 |
| R_pergene_basis64 | `nrkorko0` | 4,079 | 0.055 | 0.134 | 0.179 | 0.195 | 0.204 | 0.199 | -- | -- | 0.213 @ 2,490 |
| R_ref | `r2s57bvt` | 8,498 | 0.092 | 0.095 | 0.121 | 0.153 | 0.154 | 0.161 | 0.168 | 0.181 | 0.191 @ 8,414 |
| R_ref | `i63dbvha` | 4,139 | 0.077 | 0.105 | 0.161 | 0.165 | 0.171 | 0.173 | -- | -- | 0.185 @ 3,865 |

So the per-gene row buys a 4.8x shorter time to peak (epoch 1,767 vs 8,414), at a peak that is
as high or higher, at the cost of needing early stopping. The incumbent (`V_ref` = H_ref, `V_concat` = H_concat) is one of
the two **slowest** readouts in the panel.

Nothing else separated fast from slow. Batch size: only 4 batch-64 runs exist and at matched
steps they are indistinguishable from batch 32 on the same loss (`g755dzhn` / `teaeym0k`
listmle b32 reach 0.134 / 0.143 at 40k steps; `6b33ftkt` / `k79hvcqe` listmle b64 reach
0.141 / 0.128). LR: constant everywhere in the long rounds; the only lr arms
(`P_lr1e4`, `P_lr3e5`, `L*`) are short v8-era runs. Weight decay: the v15 ladder at 440
epochs shows nothing (below).

---

## (d) Steps, not epochs

With `n_train_supervised` = 1,244 and batch 32, an expression epoch is **39 steps**. The
proteome (3,581 strains) is **112 steps**. Matched on optimizer steps, mean over runs:

| optimizer steps | 2,000 | 5,000 | 10,000 | 20,000 | 40,000 | 80,000 | 120,000 |
|---|---|---|---|---|---|---|---|
| v13 expression val pf (n=23) | 0.022 | 0.086 | 0.079 | 0.093 | 0.147 | 0.157 | 0.160 |
| v14 proteome val pf (n=16) | 0.025 | 0.054 | 0.082 | 0.082 | 0.079 | 0.079 | 0.075 |
| v13 expression pred_sd_ratio | 0.003 | 0.067 | 0.172 | 0.268 | 0.419 | 0.464 | 0.469 |
| v14 proteome pred_sd_ratio | 0.006 | 0.032 | 0.099 | 0.213 | 0.357 | 0.379 | 0.391 |

The **output-scale unwinding is step-paced and nearly task-independent**: sdr passes 0.2 at
about 12k to 20k steps and saturates at 40k to 80k steps on both datasets, despite a 2.9x
difference in strains per epoch. Median steps to 90% of the reference value: expression
**32,175**, proteome **6,384**.

Doubling the batch does not buy per-step progress (listmle pair above), so the wall-clock
lever is not batch size. The reason expression "feels" slow in epochs is partly arithmetic:
39 steps per epoch means the headline "4,000 epochs" is only 156k steps, which is a small
optimization budget by any transformer standard.

---

## (e) Proteome early peak vs expression slow rise

Same model, same mask schedule, same lr, same batch, different label:

| | v14 proteome (16 runs) | v13 expression (23 runs) |
|---|---|---|
| n_train | 3,581 | 1,244 |
| steps/epoch | 112 | 39 |
| median epoch to 90% of ref | **57** | 825 |
| median steps to 90% of ref | **6,384** | 32,175 |
| median peak position | 0.19 of the run | 0.87 of the run |
| val pf at peak (median of run maxima) | 0.085 | 0.167 |
| val pf at last epoch / max | 0.82 | 0.95 |
| traineval pf @ 1,000 ep | 0.614 | 0.615 |
| label ceiling | 0.42 duplicate strains, 0.61 HIS3 replicate | 0.775 replicate |

The proteome is **not slower and does not need a longer warm-up; it peaks 5x earlier in
steps and then declines** while its train fit keeps climbing to 0.61. Its pred_sd_ratio keeps
growing after its Pearson has peaked (0.213 at 20k steps, 0.391 at 120k steps), so the
output-scale story and the generalization story are separable: scale keeps inflating, the
signal does not.

The same k-ordering holds in v14 (k1 peaks at 156, k2 at 299, k3 at 749, k0 at 152) and
`val/mask/loss@k0` again rises monotonically from epoch ~75. This is the strongest single
argument against a mask warm-up: **the identical masked objective plateaus in 57 epochs on
one label and 825 on another.** Whatever is slow is dataset-specific, not objective-specific.

Representative runs:

https://wandb.ai/zhao-group/torchcell_019_prot_v14/runs/uc0pm2pv

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/wq8y8nd5

---

## (f) Is the v13 late rise real, or an order-statistic artifact?

Real. Measured as a **paired within-run difference at matched epochs of a rolling mean**
(w = 9), which is not a max and therefore not an order statistic. One run (`825on260`,
V_ref_s0_seed1) is excluded because it never left 0.001.

| contrast | n | mean | sd | paired t | positive | sign-test p |
|---|---|---|---|---|---|---|
| pf(3,700) - pf(1,000) | 23 | **+0.0156** | 0.0164 | 4.55 | 18/23 | 0.011 |
| pf(2,000) - pf(1,000) | 23 | **+0.0107** | 0.0107 | 4.79 | 21/23 | 0.0001 |

By split group: fold90 +0.0259 (4/4 positive), split0 +0.0166 (6/7), split1 +0.0078 (2/4),
split2 +0.0169 (3/4), split3 +0.0099 (3/4). So the late rise reproduces across all four
validation draws and both readout arms, at about +0.015 Pearson over 2,700 epochs, which is
roughly a third of the between-split spread and about a tenth of the total score. It is real
and it is small. Per-run values (rolling w = 9):

| run | id | ep 300 | 500 | 1,000 | 2,000 | 3,700 | d(3700-1000) |
|---|---|---|---|---|---|---|---|
| V_concat_s0_90_seed1 | `wmmw6ff9` | 0.113 | 0.160 | 0.188 | 0.220 | 0.237 | +0.049 |
| V_concat_s2_seed0 | `qp8cazj9` | 0.048 | 0.040 | 0.112 | 0.125 | 0.147 | +0.035 |
| V_concat_s0_seed0 | `bn37i9vs` | 0.083 | 0.131 | 0.160 | 0.175 | 0.194 | +0.033 |
| V_ref_s3_seed0 | `bw7mxqcl` | 0.043 | 0.045 | 0.102 | 0.115 | 0.131 | +0.029 |
| V_ref_s0_seed0 | `wq8y8nd5` | 0.088 | 0.098 | 0.190 | 0.197 | 0.197 | +0.007 |
| V_concat_s1_seed0 | `bg4pzjub` | 0.096 | 0.103 | 0.155 | 0.156 | 0.143 | -0.012 |
| V_ref_s3_seed1 | `i4uw04mb` | 0.049 | 0.054 | 0.132 | 0.134 | 0.121 | -0.011 |

Two caveats that do belong here. First, every number above is one validation draw of 155
strains, so a per-run difference of 0.015 is inside the sampling noise of the draw; the
evidence is the **pairing across 23 runs**, not any single curve. Second, the run-level
`val_pf_rollmax` that the round leaderboards use *is* a max over ~4,000 epochs and therefore
upward-biased; the v13 readout should be quoted at a matched epoch, which is what
`v13_split_readout.json` does.

---

## The warm-up hypothesis, adjudicated

| reading of "masking needs warm-up" | verdict | evidence |
|---|---|---|
| The reveal schedule ramps over training and the ramp is slow | **Refuted, definitively** | `n_revealed@k` has 1 unique value per k in 15/15 runs checked. There is no ramp. |
| The masked branch must be learned before the k0 branch can improve | **Refuted** | k1/k2/k3 peak at 214/606/1,099 and then decline; k0 peaks at 3,531 and is still rising. The conditioned branch is learned first and fastest. |
| Spending 75% of gradient steps at k > 0 starves the k0 objective and that is why it is slow | **Refuted at n = 1 per arm** | Turning masking off entirely (`M_off`, 100% of steps at k = 0) reaches 90% of its own max at epoch 1,809, in the middle of the mask arms (817 to 5,417), and its k = 0-fraction dose-response is flat/wrong-signed. See table below. |
| The optimization has a genuine warm-up problem (post-LN head, constant LR, no warmup) | **Untestable from the logged data** | `lr` is never logged; `lr_scheduler.type` is `null` in every long run, so there is no warmup arm to read. `grad_norm` is flat at 0.03 to 0.05 against a clip of 10.0, which rules out the explosion failure mode but not the slow-start one. |

The `M_off` dose-response, all 8 arms run to 9,999 epochs on split seed 0, one seed each. These
numbers come from a separate higher-sample history pull (`mask_curves.json`, 5,000 samples per
run) and differ from the `curves2.json` figures quoted elsewhere by under 1.5%, which is
W&B history downsampling, not a discrepancy:

| arm | mask_schedule | fraction of steps at k = 0 | ep to 50% of own max | ep to 90% | rolling max @ epoch | pf @ 1,000 | pf @ 4,000 |
|---|---|---|---|---|---|---|---|
| `M_off` | none | **1.00** | 76 | **1,809** | 0.203 @ 3,733 | 0.169 | 0.192 |
| `M_coarse` | [0,100,1000] | 0.33 | 136 | 5,417 | 0.209 @ 9,678 | 0.152 | 0.170 |
| `M_hi` | [0,1000,3000] | 0.33 | 117 | 1,715 | 0.190 @ 3,364 | 0.167 | 0.182 |
| `M_sched` | [0,10,100,1000] | 0.25 | 143 | 2,800 | 0.185 @ 3,722 | 0.141 | 0.174 |
| `M_nomix` | [0,10,100,1000] | 0.25 | 66 | 1,508 | 0.209 @ 4,441 | 0.176 | 0.200 |
| `M_lo` | [0,5,10,30] | 0.25 | 102 | 1,087 | 0.182 @ 8,810 | 0.156 | 0.162 |
| `M_gate_rezero` | [0,10,100,1000] | 0.25 | 55 | 817 | 0.167 @ 2,180 | 0.147 | 0.148 |
| `M_fine` | [0,10,30,100,300,1000] | **0.17** | 224 | 4,192 | 0.240 @ 9,183 | 0.165 | 0.209 |

Spearman of (k0 fraction, ep-to-90%) across these 8 arms is essentially zero and the sign is
wrong for the hypothesis: the arm with the *fewest* k0 steps (`M_fine`, 1/6) has the
*highest* final score (0.233 at epoch 9,900). `M_off`'s train fit is also not faster
(`traineval` pf 0.289 at epoch 400 and 0.497 at 800, against `M_sched` 0.360 and 0.596).

Run links, one per line:

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/tow1z48n

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/8r5ewoaq

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/hx8pxdic

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/ebkzn1ao

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/da5g4o9v

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/f2wf23oy

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/rb3bhryq

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/u1vuznme

The fast-readout runs:

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/vqek7ali

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/4u0txt3s

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/nrkorko0

https://wandb.ai/zhao-group/torchcell_019_expr_v12/runs/mfgjmq57

https://wandb.ai/zhao-group/torchcell_019_expr_v12/runs/n3pibzjx

https://wandb.ai/zhao-group/torchcell_019_expr_v12/runs/ahgltcx6

The slow reference runs they should be paired against:

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/r2s57bvt

https://wandb.ai/zhao-group/torchcell_019_expr_v12/runs/vahi7g0w

https://wandb.ai/zhao-group/torchcell_019_expr_v12/runs/9pujw6j2

v13 runs used for the late-rise test:

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/wmmw6ff9

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/bn37i9vs

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/qp8cazj9

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/bw7mxqcl

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/825on260

v15, which is simply at epoch 444 of a 6,000-epoch curve and carries no information about
weight decay yet (pf 0.087 to 0.102, exactly the v13 split-1 trajectory at that epoch):

https://wandb.ai/zhao-group/torchcell_019_expr_v15/runs/dabvh5po

https://wandb.ai/zhao-group/torchcell_019_expr_v15/runs/223ce8px

---

## What to log so the remaining reading is testable

The one reading I cannot adjudicate is "the optimizer has a slow start because the head is
post-LN and the LR is a flat 3e-4 with no warmup". To make it readable:

1. **Log the learning rate.** Add Lightning's `LearningRateMonitor(logging_interval="epoch")`.
   Today no run in any of the seven projects logs `lr`, so a future scheduler arm cannot even
   be confirmed to have taken effect.
2. **Log the update-to-weight ratio per parameter group** (`||lr * update|| / ||w||`, target
   band 1e-3), separately for the encoder, the perturbation head and the readout. This is the
   quantity that distinguishes "the optimizer is crawling" from "the gradient signal is
   weak"; `grad_norm` alone cannot, because a small grad_norm is consistent with both.
3. **Log the effective rank of the prediction matrix** (participation ratio of the singular
   values of `yhat[strain, gene]` on the val set, one number per epoch). The `pred_sd_ratio`
   curve says the output is inflating; effective rank would say whether the model is adding
   *directions* or only amplitude. The measured rank-32 residual structure
   (`residual_covariance_diagnostic.json`) gives it a reference line.
4. **Log `train/mask/loss@k` restricted to the k that was actually drawn** as a per-k running
   mean, and the count of steps drawn per k. It is logged now, but only on the steps where
   that k came up, so the series is ragged and a reader cannot tell a rate from a sample.
5. **Log the head's output standard deviation before and after the per-feature
   de-standardization**, so the shrinkage can be attributed to the head or to the target
   normalization.

---

## The two cheapest experiments that would settle it

Both are **config-only** (no code), both run on the v13 apparatus, and both are paired within
split and init seed so they are readable at n = 4 rather than the n = 1 the v9 arms give.

### E1. The k0-step-fraction ladder, 12 runs, ~1,500 epochs

The direct falsification. Three doses of "fraction of gradient steps at k = 0", nothing else
changed, on the two split seeds where the model has the most to gain:

| arm | override | steps at k = 0 |
|---|---|---|
| `K_off` | `~multitask.mask_schedule` | 100% |
| `K_half` | `multitask.mask_schedule=[0,1000]` | 50% |
| `K_ref` | (v13 default `[0,10,100,1000]`) | 25% |

2 split seeds (1 and 2) x 2 init seeds x 3 arms = 12 runs, `trainer.max_epochs=1500`
(the median run reaches 90% of its 3,700-epoch value at epoch 825, so 1,500 is enough to read
the *rate*). Read out: epoch to 50/75/90% of the arm's own value at epoch 1,500, and pf at
matched epoch 1,500, both paired within (split, seed). Cost: v13 ran 88.7 s/epoch at four
runs per A40; 1,500 epochs is about 37 h per run, 12 runs on 3 cards is about 37 h wall. The
`K_off` arm is 24% cheaper still, because it does one forward per train step and one
validation pass instead of four. **If ep-to-90% is flat in the dose, the mask-starvation
reading is dead with n = 4 pairs instead of n = 1.** If it is not flat, the fix is a
one-line change to the k sampler.

### E2. The readout-speed confirmation, 8 runs, ~1,500 epochs

The competing explanation, which the data already points at. Both flags exist:
`multitask.context_readout=true` is H_state and `multitask.per_gene_weight=true` is
H_pergene; `cgt_expr_v12_head.yaml` currently sets `context_readout: false`.

| arm | override |
|---|---|
| `S_ref` | (v13 default, shared MLP) |
| `S_state` | `multitask.context_readout=true` |

2 split seeds (1 and 2) x 2 init seeds x 2 arms = 8 runs at 1,500 epochs, checkpoint on
`val/expression/pearson_per_feature` with early stopping enabled (the v9 long runs show the
per-gene arms decline after their peak, so an unstopped run would under-report them). Read
out: epoch to 90%, peak value, and the peak epoch. Cost about 37 h wall on 2 cards.
**Prediction from the existing curves (untested at these splits): `S_state` reaches its peak
by epoch ~250 and `S_ref` by epoch ~950, at peaks within 0.01 of each other.** If that
replicates, the campaign's epoch budget can drop from 6,000 to about 800 for every future
arm, which is a 7x increase in arms per GPU-week and is worth more than any single
architecture delta measured so far.

I would run E1 and E2 as one 20-run launch on 5 cards: they share the reference arm, so
`S_ref` and `K_ref` are the same configuration and can be pooled.

---

## Summary (600 words)

The masked objective has no warm-up in it. `mask_schedule` is a static list of reveal counts
and `train/mask/n_revealed@k` takes exactly one value per k for the entire run in all 15 runs
checked, so there is no ramp to be slow about. Training draws one k uniformly per batch, so
with the standard `[0, 10, 100, 1000]` schedule 25% of gradient steps are the k = 0 regime the
leaderboard scores; the natural version of the hypothesis is that this starves the scored
objective. It does not. In v9, eight arms ran to 9,999 epochs with the k = 0 step fraction
varying from 1.00 (`M_off`, masking removed entirely) to 0.17 (`M_fine`), and epoch to 90% of
each arm's own max ranges 817 to 5,417 with no ordering by dose: `M_off` sits at 1,809, in the
middle, and `M_fine`, with the fewest k = 0 steps, ends highest at 0.233. Masking costs 24%
wall clock, not 2x.

The k-resolved metrics say the opposite of a warm-up. In v13 the conditioned branches peak
early and then decay (k1 at epoch 214, k2 at 606, k3 at 1,099) while the unconditioned k0
metric peaks at 3,531. `val/mask/loss@k0` bottoms at epoch ~100 and degrades monotonically
thereafter while k0 Pearson doubles, so the scored metric and the loss it is trained on move
in opposite directions from epoch 150 onward.

The curve shape is three-phase, not a slow monotone rise: a fast climb to 0.089 by epoch 150
(55% of the plateau), a dip and plateau to epoch 500, then a second climb finishing around
epoch 1,500 to 2,000. Median epochs to 25/50/75/90% of the value at epoch 3,700 are
64/118/594/825. The tightest correlate of the rise is `pred_sd_ratio`, which goes 0.003 at
epoch 50 to 0.47 at epoch 2,000 with corr(pf, sdr) of 0.79 to 0.95 across epochs in 23 of 24
runs. Progress is paced by optimizer steps, not epochs or examples: with 1,244 strains at
batch 32 an epoch is 39 steps, so 4,000 epochs is 156k steps, and doubling the batch buys
nothing per step.

The proteome is the cleanest refutation. Same model, same schedule, same lr, same batch, on
3,581 strains: it reaches 90% of its plateau at epoch 57 (6,384 steps) against expression's
825 (32,175 steps), peaks at 19% of the way through the run, and then declines while its
train fit climbs to 0.61. The identical masked objective plateaus in 57 epochs on one label
and 825 on another, so whatever is slow is the label, not the objective.

What actually changes the timescale is the readout. In the v12 round (matched budget, pinned
split, four init seeds, collapsed runs excluded) the per-gene-output-row family reaches 90%
of its max at a mean epoch of 330 against 860 for the shared-map family, a 2.6x difference,
and the same ordering shows in the train fit, so it is optimization speed and not
regularization. At a long budget in v9 the per-gene arms peak higher (0.219 at epoch 1,767)
and then decline, while the shared reference is still climbing at 8,414. The current
incumbent, `V_ref` / `V_concat`, is one of the two slowest readouts in the panel.

The v13 late rise is real: paired within-run, at matched epochs of a rolling mean, pf(3,700)
minus pf(1,000) is +0.0156, sd 0.0164, t = 4.55, 18/23 positive, sign-test p = 0.011, and it
reproduces on all four validation draws. It is not an order statistic, and it is small,
about a tenth of the score.

One reading remains open and is untestable from the logs: `lr_scheduler.type` is null in all
175 runs, lr is a flat 3e-4, the perturbation head is post-LN, and `lr` is never logged. Log
the learning rate and the per-group update-to-weight ratio before asking that question again.

---

## Artifacts written by this review (scratchpad only, nothing under the repo)

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/pull2.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/curves2.json

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/pull_mask.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/mask_curves.json

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/pull_grad.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/grad_curves.json

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/an_mask.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/an3.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/an4.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/an5.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/an6.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/an7.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/agent-02-curves.md
