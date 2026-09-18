# Agent 05 -- optimization and regularization of the 019 expression rounds

Read-only review, 2026-09-17. Every number below traces to a config line, a source line, a
results JSON, or a W&B history I pulled myself. Hypotheses are labeled.

## 0. The resolved config, as it actually runs

Chain: `cgt_expr_v13_split` -> `v12_head` -> `v11_emb` -> `v9_mask` -> `_012` -> `_011` ->
`_010` -> `_008` -> `_006` -> `cgt_embed_005` -> `cgt_decoder_004` -> `cgt_decoder_003` ->
`default`. Resolved values, with the file that last sets each:

| knob | value | set in |
|---|---|---|
| optimizer | `torch.optim.AdamW` | `cgt_decoder_003` |
| lr | `3.0e-4`, CONSTANT | `cgt_expr_008` |
| weight_decay | `1.0e-8` | `cgt_expr_008` |
| betas / eps | `(0.9, 0.999)` / `1e-8` (torch defaults, never set) | nowhere |
| param groups | ONE. `opt_class(self.parameters(), **opt_cfg)` | `train_cgt_multitask.py:1804` |
| lr scheduler | `type: null` -> `return optimizer` (no schedule at all) | `cgt_expr_008`, `train_cgt_multitask.py:1815` |
| warmup | NONE (`warmup_steps: 0`, and the block is inert because `type` is null) | `cgt_expr_008` |
| grad clip | `clip_grad_norm: true`, `max_norm 10.0`, applied in `on_before_optimizer_step` | `cgt_decoder_003` |
| batch size | 32 | `cgt_expr_012` |
| accumulate_grad_batches | not passed to `L.Trainer` -> 1 | `train_cgt_multitask.py:3351` |
| precision | `bf16-mixed` | `cgt_decoder_004` |
| torch.compile | NONE anywhere in the script | grep |
| EMA / SWA / weight averaging | NONE anywhere (no `StochasticWeightAveraging`, no EMA callback) | grep |
| dropout | `0.1` everywhere (`model.dropout`, `perturbation_head.dropout`, preprocessor) | `cgt_expr_008` / `cgt_decoder_003` |
| stochastic depth / label smoothing / label noise | NONE | grep |
| graph KL regularizer | OFF (`graph_reg_lambda: 0.0`); replaced by the hard attention mask | `cgt_expr_010` |
| early stopping | OFF | `cgt_expr_012` |
| max_epochs | 6000 (v13 runs actually stopped at ~4,080 on the wall) | `cgt_expr_v13_split` |
| checkpoints | best-by-`val/expression/pearson_per_feature`, best-by-metric, last | `train_cgt_multitask.py:3190-3218` |

Scale facts, read off run `wq8y8nd5`'s W&B summary: `total_param_count` 6,671,629;
`n_train_supervised` 1,244; `trainer/global_step` 158,879 at epoch 4,073, i.e. **39
optimizer steps per epoch** = `ceil(1244/32)`. `perf/epoch_seconds` 67.95 at four runs per
A40 card. The "~3,700 examples" in the brief is the all-modality record count (4,074 in
`results/split_indices_manifest.json`); the expression dataloader iterates the
`require_modalities: [expression_log2_ratio]` subset, which is 1,244.

**One structural fact the config table hides.** `train_cgt_multitask.py:1261` samples ONE
`k` uniformly from `mask_schedule = [0, 10, 100, 1000]` per training batch. So only about
**25% of optimizer steps train the k=0 objective the leaderboard reports**; the rest reveal
10, 100 or 1,000 true labels. Section 1 tests whether that matters.

---

## 1. Steps to the val-Pearson peak, and what kind of slow this is

### 1a. The counts

Median `roll_max` argmax over the 23 non-collapsed v13 runs is **epoch 3,516**, range 875 to
4,093 (my pull, `scratchpad/v13_curves.csv`). At 39 steps per epoch:

- **137,124 optimizer steps to peak** (median), 3,516 passes over 1,244 examples,
  4.37M example presentations.
- Of those, about **34,300 steps** are k=0 steps.
- Wall cost: 67.95 s/epoch at four per A40, so 4,000 epochs is **3.1 days per run**.

For a 6.7M-parameter transformer on 1,244 examples, the conventional budget is tens to a few
hundred epochs. This run needs 3,516. It is 1 to 2 orders of magnitude slower than normal.

### 1b. But "still rising at 4,000" is largely an artifact of the statistic

The campaign scores `roll_max`, a RUNNING maximum, which is monotone by construction and
keeps absorbing upward noise. I recomputed the raw LEVEL (5-epoch centered mean AT the
epoch, not the running max) across the 19 v13 runs that reach every matched epoch:

| epoch | level (mean, n=19) | sd | `roll_max` (n=23) | traineval pf | `val pred_sd_ratio` |
|---|---|---|---|---|---|
| 250 | 0.0701 | 0.0220 | 0.1048 | 0.271 | 0.171 |
| 500 | 0.0849 | 0.0294 | 0.1109 | 0.398 | 0.263 |
| 1,000 | 0.1401 | 0.0276 | 0.1549 | 0.642 | 0.416 |
| 2,000 | 0.1496 | 0.0281 | 0.1676 | 0.733 | 0.464 |
| 3,000 | 0.1511 | 0.0284 | 0.1720 | 0.752 | 0.460 |
| 4,000 | 0.1541 | 0.0304 | 0.1752 | 0.776 | 0.464 |

Paired within run: level(4,000) - level(1,000) = **+0.0140, sd 0.0169, 13 of 19 positive,
t 3.6**. level(4,000) - level(2,000) = **+0.0045, sd 0.0125, t 1.6, not resolved**.

So: **91% of the epoch-4,000 level is present at epoch 1,000** (39,000 steps), and
everything after epoch 2,000 is inside the noise. The extra 0.014 is real but it costs 2.3
days per run. The `roll_max` statistic inflates the same interval to +0.020 because a
running max over a noisy curve rises even on a flat curve.

### 1c. Is it lr, gradient noise, or epochs? Measured answers.

**It is not a learning-rate-too-low problem.** Three independent measurements, none of them
previously assembled in one place:

1. The `_007` Halton sweep, 289 non-collapsed runs, lr log-uniform 4e-5 to 2.5e-3, median
   final epoch 67. Score is **monotone DECREASING in lr**, Spearman **-0.509, p 1.8e-20**.
   Binned `primary_roll_max`: lr 4e-5..1.3e-4 0.0615 (n 74), 1.3e-4..2.5e-4 0.0604 (54),
   2.5e-4..5e-4 0.0429 (52), 5e-4..1e-3 0.0271 (55), 1e-3..2.5e-3 0.0185 (54).
   (From `results/round_leaderboards.csv` joined to the v7 W&B configs. Caveat: 67 epochs is
   inside the dead zone of 1d, and the leaderboard `is_collapsed` test is the flawed
   whole-curve one the memory flags.)
2. The **wave-3 lr ladder** at 55 epochs, 2 seeds, config `cgt_expr_008`:
   L0 lr 3e-4 `roll_max` 0.1119 / 0.0200; L1 lr 1e-3 0.1021 / 0.0049;
   L2 lr 3e-3 **0.0045 / 0.0038** (dead); L3 lr 1e-3 + cosine 0.0128 / 0.0114 (dead).
   Raising lr kills the run.
3. The **wave-5 lr-DOWN arms** at 399 epochs, config `cgt_expr_011`, n=1 each:
   `P_lr1e4` (lr 1e-4) `roll_max` 0.1598 at epoch 398, still rising;
   `P_lr3e5` (lr 3e-5) 0.1551 at epoch 135; against `W_ref` at lr 3e-4 and 299 epochs,
   0.090 to 0.127 over 8 runs. Lowering lr scored HIGHER at a few hundred epochs.

`https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/6etb6y9h`

`https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/fl5fuv99`

`https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/6v78p223`

**It is not a clipping problem, and there is no instability for warmup to fix.**
`train/grad_norm_clip_frac` is `grad_norm / 10.0`, so >1 means clipped. Over all 4,073
epoch-means of `wq8y8nd5` the MAXIMUM clip_frac is **0.0160**; `bn37i9vs` 0.0207,
`8i75d8h1` 0.0173, `lp6guytz` 0.0153. **Zero clipped steps in any of the four runs I pulled.**
Epoch-0 grad norm is 0.153 to 0.173 and falls to ~0.030 by epoch 10, then sits at 0.032 to
0.053 for the remaining 4,000 epochs. The v16 smoke figure of clip_frac 0.47 at step 0 is a
single first step, grad norm 4.7, still a factor of 2 under the threshold. **Clipping at
max_norm 10.0 is inert and has been for every expression run.** It is not a lever, and it is
also not evidence of a healthy gradient scale, only that 10.0 is ~200x too loose to bind.

**It is partly a gradient-noise / shallow-direction problem, and the mechanism is visible.**
Two measured signatures:

- `train/grad_norm` does NOT decay as the fit improves. From epoch 100 to epoch 4,073 the
  eval-mode train Pearson goes 0.044 -> 0.773 while the epoch-mean grad norm goes
  0.032 -> 0.053, i.e. slightly UP. Under a signal-dominated gradient it would shrink toward
  zero as the model approaches its train optimum.
  Mechanism (this part is a property of the loss, not a hypothesis): the objective is
  pinball, `rho_tau(u) = max(tau*u, (tau-1)*u)`, `torchcell/losses/distributional.py:361`.
  Its derivative in the prediction is `-tau` or `1-tau`, a SIGN, never small. So the
  per-example gradient magnitude has a floor independent of fit quality, and at the optimum
  the minibatch gradient is pure sign noise of O(1) per scored entry. Batch 32 averages 32
  strains x ~6,127 genes of that.
- The first ~100 epochs are spent at the constant predictor. `val/expression/pred_sd_ratio`
  is 0.0003 to 0.011 through epoch 100 and first exceeds 0.05 at a **median epoch 100 (range
  77 to 124)** across the 19 runs. Under pinball, the risk-minimizing output for a gene with
  no usable input signal IS the marginal quantile, so the model sits in the constant basin
  and has to climb out along a shallow direction. The sd ratio then rises to 0.42 by epoch
  1,000 and **saturates at 0.46**, so the final model is permanently under-dispersed by a
  factor of 2.2.

**Is it an epoch-count problem?** No, in the sense that mattered: the marginal value of
epochs 2,000 to 4,000 is +0.0045 (t 1.6). Yes, in the sense that the first 1,000 epochs are
genuinely needed, and ~100 of them are spent producing a constant.

**The masking-dilution hypothesis is already falsified by an existing run.** If the 4x
dilution of k=0 gradients were the cause of the slow rise, the `M_off` arm (no mask
schedule, 100% of steps on k=0) should peak much earlier. It does not: `M_off` `roll_max`
0.2008 at **epoch 3,503**, against `M_sched` 0.1824 at 3,715, `M_fine` 0.2382 at 9,697,
`M_coarse` 0.2091 at 9,691, `M_nomix` 0.2057 at 4,109, `M_hi` 0.1887 at 3,398, `M_lo` 0.1804
at 8,859 (one run each, seed 0, split 0; arm spread sd 0.0222). So masking neither helps nor
costs the k=0 score, and it does not explain the slow rise. This also directly answers the
user's "does masking need warmup": no measurable warmup cost.

`https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/tow1z48n`

`https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/8r5ewoaq`

`https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/wq8y8nd5`

`https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/bn37i9vs`

`https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/8i75d8h1`

**Verdict for (1).** The dominant term is not lr and not clipping. It is (a) a ~100-epoch
constant-predictor plateau imposed by a sign-gradient loss whose minimizer for a weak-signal
feature is the marginal quantile, and (b) a gradient whose noise floor does not fall as the
fit improves, so progress along the shallow escape direction accumulates like a random walk.
The epoch count past 2,000 buys almost nothing. Hypothesis (untested, and it is the cheapest
thing to test): a 50-epoch warmup to a HIGHER peak lr with a cosine decay would shorten the
plateau, because the only lr evidence against high lr comes from constant-lr runs where the
first step is taken at full lr.

---

## 2. The batch-64 "collapse" is inverted

**The brief has this backwards, and so does the one-line memory summary if read quickly.**
From `results/pearson_round_readout.json` and the W&B configs I pulled:

| arm | batch | lr | seeds | epochs reached | `roll_max` (epoch) | collapsed? |
|---|---|---|---|---|---|---|
| `Q_pearson` | **32** | 3e-4 | 0,1,2 | 4,298 / 4,291 / 3,322 | 0.1452 (215), 0.1328 (151), 0.1366 (242) | **YES**, at 3,901 / 3,780 / 3,087; metric floored from 584 / 2,203 / 680 |
| `Q_pearson_b64` | **64** | 3e-4 | 0,1 | 9,899 / 9,899 | 0.1942 (1,926), 0.1856 (2,419) | NO |
| `Q_pearson_mse` | 32 | 3e-4 | 0,1,2 | 4,298 / 9,899 / 3,322 | 0.1365, 0.1936, 0.0349 | 2 of 3 YES |

So it is **batch 32 under the pure-Pearson objective that collapses to constant outputs**;
batch 64 survived to 9,899 epochs and landed ON the incumbent arm band (0.194 / 0.186 vs
0.1965 +/- 0.0222, which is itself an arm spread, not a replicate spread).

`https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/7ylecrjz`

`https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/wb2xocf2`

`https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/zaul43n1`

**Is the batch-64 arm a real optimization effect?** It is not established, and it is
confounded four ways:

1. **lr was not scaled.** Both arms ran lr 3e-4 (`gh_expr_008_arm.sh:447` sets only
   `data_module.batch_size=64`). At batch 64 the per-example step is half. Since section 1c
   shows score is monotone decreasing in lr in this regime, "batch 64 is better" and "the
   effective lr is lower" are the same arm.
2. **Different number of steps AND different number of epochs.** b64 reached 9,899 epochs
   (197,980 steps); the b32 seeds reached 3,322 to 4,298 epochs (129,558 to 167,622 steps).
   Neither epochs nor steps are matched.
3. **Different placement.** b64 ran as its own stage (`stage-pearson_b64`, slurm 2378268 and
   2378269, one run per task at 18.7 s/epoch); the b32 Pearson runs were packed (2378262
   hosts both `kspeoljg` and `ppc2pyv5`, 30.7 s/epoch).
4. **n = 2 vs 3, one split seed, one objective.** The collapse is a property of the PEARSON
   loss, which is invariant to per-column affine maps and whose implementation drops constant
   columns as invalid, so a constant output is a fixed point that reads as loss ~0.56 rather
   than as failure. The incumbent pinball objective at batch 32 collapses in 1 of 24 v13 runs
   (`825on260`), not 3 of 3.

**Conclusion for (2):** there is no measured batch-size effect for the incumbent objective.
The only batch-size comparison in the corpus is inside a collapsing objective, with lr
unscaled, unmatched budgets, and unmatched GPU packing. Do not carry "batch 64 collapses
Pearson" forward; the sentence is false as written, and the true sentence ("pure-Pearson at
batch 32 collapsed on 3 of 3 seeds") is about the loss, not the batch.

---

## 3. Weight decay: v15 state, and the LayerNorm question

### 3a. v15 is partial and cannot be read yet

Only **4 of the planned 12 runs** are synced to `zhao-group/torchcell_019_expr_v15`, all on
split seed 1, all at **epoch 444 to 451 of 6,000** (7.5% of budget), runtime ~44,800 s each,
last heartbeat 2026-09-18T01:02Z. Partial values, stated as partial, `roll_max` of the
5-epoch centered mean:

| run | arm | wd | epoch | `roll_max` (epoch) | traineval pf |
|---|---|---|---|---|---|
| `dabvh5po` | `W_ref_s1_seed0` | 1e-8 | 444 | 0.0918 (403) | 0.355 |
| `mv5zd8bu` | `W_ref_s1_seed1` | 1e-8 | 444 | 0.1094 (374) | 0.351 |
| `223ce8px` | `W_wd1e2_s1_seed0` | 1e-2 | 451 | 0.1103 (328) | 0.349 |
| `ax2dedpj` | `W_wd1e1_s1_seed0` | 1e-1 | 447 | 0.0933 (363) | 0.361 |

`https://wandb.ai/zhao-group/torchcell_019_expr_v15/runs/dabvh5po`

`https://wandb.ai/zhao-group/torchcell_019_expr_v15/runs/223ce8px`

`https://wandb.ai/zhao-group/torchcell_019_expr_v15/runs/ax2dedpj`

**No conclusion.** One seed per decay arm, 7.5% of the budget, and epoch 450 sits in the
steepest part of the curve (level 0.070 at 250 to 0.140 at 1,000), where run-to-run spread is
0.022 to 0.029. Nothing here separates 1e-8, 1e-2 and 1e-1. Note the seed-0 reference (0.0918)
is BELOW both decay arms and the seed-1 reference (0.1094) is above one of them, which is the
spread, not a signal.

### 3b. Decay applies to LayerNorm and every bias, and that is a real confound for the 1e-1 arm

`train_cgt_multitask.py:1804` is `optimizer = opt_class(self.parameters(), **opt_cfg)`. One
param group, no exclusion list. So `weight_decay` hits **LayerNorm weights and biases, every
Linear bias, and the ReZero gate scalars** (`beta_attn`, `beta_ffn`,
`equivariant_cell_graph_transformer.py:628-633`) exactly as hard as it hits the weight
matrices. At 1e-8 that is irrelevant. At **1e-1** it is not: AdamW's decoupled decay shrinks a
LayerNorm gain toward 0 at a rate independent of its gradient, which attenuates every
activation and pushes ReZero gates shut. So `W_wd1e1` does not measure "strong L2 on the
weights"; it measures "strong L2 on the weights AND on the normalization gains AND on the
residual gates". Whatever v15 reports at 1e-1, that arm is not attributable.

### 3c. What decay is already measured to do

- `_007` Halton sweep, wd log-uniform 1e-9 to 6e-4, n=289: **Spearman +0.038, p 0.52**. Null.
- v10 factorial, 1e-8 vs 1e-4 at epochs <= 990, 16 runs a side: **+0.0046, t +0.5**, against a
  resolution of ~0.02. Null (`results/v10_grid_factorial.json`).
- Wave-6 single runs at ~4,300 epochs, config `cgt_expr_012`, split 0 seed 0:
  `V_wd1e4` `roll_max` **0.2083** at epoch 4,318 (still rising); `V_wd1e2` **0.2010** at 1,388.
  Both inside the 0.1965 +/- 0.0222 arm band.

`https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/0k9ae8co`

`https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/jyl2i2d0`

So decay from 1e-9 to 1e-2 is a measured null across three independent rounds and two budgets.
1e-1 is the only genuinely untested point, and it is the one confounded by 3b.

---

## 4. Warmup

**There is none.** `regression_task.lr_scheduler.type` is `null` in `cgt_expr_008` and
nothing in the v13 chain overrides it; `configure_optimizers` returns a bare AdamW at line
1806/1816. lr is 3e-4 from step 0 to step 158,879.

The arms exist but were never run to a readable budget. `gh_expr_008_arm.sh:83` defines
`S1_warmup`, and it ran twice: `etaxogte` for 121 epochs (`roll_max` 0.1330) and `3d8opzhr`
for 40 epochs (0.0134). `S2_rezero` 73 epochs, `S3_both` 76 epochs. Against the matched
`L0_lr3e4` reference at 55 epochs (0.1119 / 0.0200), nothing is resolvable.

`https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/etaxogte`

**Two traps in the existing warmup arm, both of which should be fixed before it is rerun.**

1. `configure_optimizers` returns `{"interval": "epoch"}` (line 1834). So
   `CosineAnnealingWarmupRestarts` steps ONCE PER EPOCH, and `warmup_steps` and
   `first_cycle_steps` are counted in EPOCHS, not optimizer steps. `S1_warmup`'s
   `warmup_steps=10` is 10 epochs = 390 steps, and `first_cycle_steps=140` means the lr
   RESTARTS every 140 epochs for the whole run. That arm is a cyclic-restart schedule with a
   short ramp, not a warmup, and it was never labeled as one.
2. `L3_lr1e3_cosine` set `warmup_steps=0` with `max_lr=1e-3`, i.e. full 1e-3 on step 0. It
   died on both seeds (0.0128, 0.0114 at 55 epochs), which is what a post-LN perturbation
   block without warmup does. That is the single strongest reason to believe warmup is worth
   one more round: **every high-lr arm ever run started at full lr**, so "high lr fails here"
   and "high lr without warmup fails here" are not separated anywhere in the corpus.

**Does warmup relate to the slow start?** Not to the ~100-epoch constant plateau in any way I
can demonstrate. Gradient norms at epoch 0 are 0.15 to 0.17 and fall smoothly; there is no
spike, no clipping, and no loss blowup for warmup to prevent. Hypothesis (untested): warmup
matters here only as the enabler of a higher peak lr, not as a fix for the plateau.

One more datum for the same family: **1 of 24 v13 runs never left the plateau at all.**
`825on260` (`V_ref_s0_seed1`, identical config to `wq8y8nd5` except the init seed) sits at
val Pearson ~0 for all 4,081 epochs, traineval Pearson -0.012, grad norm decaying to 0.008.
The v10 grid had the same failure once in 32 (`te8272kk`). That is a **base rate of ~3.6%
(2 of 56) of complete optimization failure at init**, which is an optimization pathology, not
a data property.

`https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/825on260`

---

## 5. Regularization on, and train 0.77 vs val 0.20

### 5a. What is on

Dropout 0.1, applied in the encoder attention weights, both residual branches, the FFN, the
input preprocessor, the perturbation head and the readout MLP. Weight decay 1e-8, which is
numerically nothing. Attention structurally masked to the nine graphs at layer 1 (a hard
prior, and a strong one). The mask-reveal schedule, which is a stochastic input augmentation.
That is the complete list. No stochastic depth, no label smoothing, no input or label noise,
no mixup, no EMA, no weight averaging.

### 5b. Dropout is measured, weakly, to help

Wave-5 `H1_nodrop` (dropout 0.0) vs `H1_ref` (0.1), same config `cgt_expr_011`, paired by
init seed at ~1,620 epochs: seed 0 0.1649 vs 0.1814 (-0.0165), seed 1 0.1452 vs 0.1997
(-0.0544). Both negative, n=2. Wave-6 at ~4,100 epochs, one seed each: `V_drop2` (0.2)
`roll_max` 0.1858, `V_drop3` (0.3) 0.1382 and then a hard collapse to -1.4e-10 by the end,
against the 0.1965 +/- 0.0222 band. So 0.1 is fine, 0.0 is worse by ~0.035 at n=2, 0.2 is a
null, 0.3 destabilizes.

`https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/l6a0cd91`

`https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/famj38u0`

`https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/tcifmwri`

### 5c. Train 0.77 vs val 0.20 is NOT a capacity problem, and the proteome says so by contrast

The decisive comparison is the shape of the two curves, not the size of the gap.

**Expression** (v13, 1,244 train strains, label reliability ceiling 0.775 from
`expression_ceiling_replicate.json` route_b): traineval Pearson 0.271 -> 0.398 -> 0.642 ->
0.733 -> 0.752 -> **0.776** at epochs 250 / 500 / 1,000 / 2,000 / 3,000 / 4,000, while the
val level goes 0.070 -> 0.085 -> 0.140 -> 0.150 -> 0.151 -> 0.154. **Validation never turns
down.** The train side has reached the label's own reliability ceiling, so the remaining
train residual is measurement noise and there is nothing more to memorize.

**Proteome** (v14, ~3,580 train strains, ceiling 0.417 by duplicate strains / 0.614 by the
HIS3 replicate route, `proteome_ceiling_replicate.json`): `uc0pm2pv` val 0.0665 (ep 50) ->
0.0949 (100) -> **0.1280 (200, the peak)** -> 0.0999 (400) -> 0.0843 (800), while traineval
climbs 0.055 -> 0.153 -> 0.352 -> 0.534 -> 0.600. **Validation peaks at epoch ~200 and then
falls by a third while train triples.** That is textbook overfitting. `b95dpt9u` is the same
shape, peak 0.0883 at 418, traineval 0.618 at 1,160.

`https://wandb.ai/zhao-group/torchcell_019_prot_v14/runs/uc0pm2pv`

`https://wandb.ai/zhao-group/torchcell_019_prot_v14/runs/b95dpt9u`

**Reading.** The two modalities land at a similar fraction of their own ceilings (expression
0.21/0.775 = 27%; proteome 0.128/0.417 = 31%, or 0.099/0.614 = 16% by the HIS3 route,
`v14_frac_of_ceiling` 0.161 in the JSON). But they get there by opposite dynamics. On
expression, regularization is inert (dropout null above 0.1, decay null over 7 decades,
capacity helps: v10 trunk effect -0.012 favoring the BIGGER trunk, sharpened to -0.0175 t
-4.0 with the dead run dropped), validation never declines, and train saturates at the label
ceiling. That is the signature of **a data problem**: 1,244 strains and one measurement each,
where the strain-to-phenotype map does not transfer to unseen deletions. On proteome it is
**an overfitting problem** and regularization / early stopping should bite there.

Corroboration for the data reading: the partition moves the expression score five times more
than any arm ever has (v13 partition means 0.193 / 0.164 / 0.134 / 0.137, between-partition
sd 0.028 vs within 0.010, `results/v13_split_readout.json` and the 2026-09-16 read), and the
90/10 arm (10% more training strains) buys +0.016 to +0.017, comparable to 3,000 extra epochs.

---

## 6. Would EMA or SWA recover the late-epoch gains without the +12% val loss?

**Short answer: no for the loss, plausibly a small yes for the metric, and the mechanism the
question assumes is wrong.**

What is actually measured. `val/loss` bottoms at epoch 1,137 / 243 / 591 / 314 in the four
v13 runs I pulled and is 10.4 to 14.7% above its minimum by the end; at the Pearson peak it
is 8.9 to 12.8% above. Project-wide the same pattern: median loss minimum at epoch 481,
median Pearson peak at 2,488, median final loss 11.4% above its minimum
(`results/loss_min_vs_pearson_peak.json`). At the end, `val/expression/nmse` is 1.032,
`calib/coverage_50` 0.320 against a nominal 0.50, `calib/coverage_80` 0.569 against 0.80,
`pit_ks` 0.160, `pred_sd_ratio` 0.464 (all from `wq8y8nd5`'s summary).

**The trade is not a calibration artifact that averaging can undo.** Write the aggregate
NMSE of a zero-mean standardized target as `1 - 2*r*s + s^2`, with `r` the correlation and
`s = pred_sd_ratio`. At `r = 0.2098`, `s = 0.4644` this gives **1.021**, which reproduces the
logged 1.032 to within the approximation. The MSE-optimal scale is `s* = r = 0.21`, giving
NMSE `1 - r^2 = 0.956`. So the model's point predictions are **over**-dispersed relative to
what minimizes squared error given how weakly they correlate, not under-dispersed, and the
intervals are simultaneously too narrow (coverage 0.32 < 0.50). The rising val loss is the
model buying rank information by sharpening in a direction that costs it on the proper
scoring rule. Shrinking the prediction (which is what weight averaging does in the noise
directions) moves `s` down toward `r`, which lowers NMSE but does not raise `r`.

**What EMA/SWA could plausibly buy, labeled as hypothesis.** Hypothesis (untested): weight
averaging recovers roughly the gap between the raw per-epoch metric and its smoothed value,
because both are removing the same iterate noise. On `wq8y8nd5` the raw max is 0.2216 at
epoch 3,783 and the 5-epoch smoothed max is 0.2097 at 3,916, a 0.012 gap; on `8i75d8h1`
0.2172 vs 0.2127, 0.0045. So the plausible EMA gain is **on the order of 0.005 to 0.012**,
which is the same size as the entire epoch 1,000 -> 4,000 gain and about a third of the
within-partition spread. It is worth having and it is not a mechanism change.

**Evidence that exists today: essentially none.** No run in any 019 project has used EMA or
SWA. The only adjacent fact is that the three saved checkpoints are best-by-loss,
best-by-metric and last, and the first two are ~1,300 to 3,000 epochs apart
(`train_cgt_multitask.py:3198-3204`), so naively averaging THOSE is not an SWA and I would
not expect it to work.

---

## 7. The cheapest decisive experiments

All overrides are plain dotted Hydra, no `=` inside any value, in the style
`gh_expr_008_arm.sh` already uses. Budget rationale throughout: the epoch-1,000 level is 91%
of the epoch-4,000 level, and 1,000 epochs is 18.9 h per run at four per A40 against 3.1 days
for 4,000. Every design below is paired within split seed and init seed, which is what made
v13 readable.

### E1 (highest value). Warmup x peak-lr, at a 1,000-epoch budget

The one combination never run. Every high-lr arm in the corpus started at full lr on step 0,
and every low-lr arm ran constant. 12 runs, 3 cards, ~1 day.

Arms, crossed with `data_module.split_seed` in {0, 1} and `seed` in {0, 1}:

```
# O_const3e4 (reference, the incumbent at the honest budget)
trainer.max_epochs=1000

# O_warm1e3  (50-epoch warmup, cosine to 1e-6 over the whole run, NO restarts)
regression_task.optimizer.lr=1e-3
regression_task.lr_scheduler.type=CosineAnnealingWarmupRestarts
regression_task.lr_scheduler.first_cycle_steps=1000
regression_task.lr_scheduler.cycle_mult=1.0
regression_task.lr_scheduler.max_lr=1e-3
regression_task.lr_scheduler.min_lr=1e-6
regression_task.lr_scheduler.warmup_steps=50
regression_task.lr_scheduler.gamma=1.0
trainer.max_epochs=1000

# O_warm3e3  (same, peak 3e-3 -- the lr that died instantly without warmup)
regression_task.optimizer.lr=3e-3
regression_task.lr_scheduler.type=CosineAnnealingWarmupRestarts
regression_task.lr_scheduler.first_cycle_steps=1000
regression_task.lr_scheduler.cycle_mult=1.0
regression_task.lr_scheduler.max_lr=3e-3
regression_task.lr_scheduler.min_lr=1e-6
regression_task.lr_scheduler.warmup_steps=50
regression_task.lr_scheduler.gamma=1.0
trainer.max_epochs=1000
```

`first_cycle_steps` and `warmup_steps` are in EPOCHS because the scheduler interval is
`epoch` (section 4). Setting `first_cycle_steps` equal to `max_epochs` makes it a one-cycle
schedule with no restart, which is the thing that was never tested.

**Reads.** Primary: the raw level at epoch 1,000 (not `roll_max`), paired. Secondary, and
this is the mechanism read: the first epoch at which `val/expression/pred_sd_ratio` exceeds
0.05, today a median of 100. If warmup plus high lr shortens that plateau, the level at 250
and 500 epochs moves first and the whole curve shifts left. **Decision rule:** if
`O_warm1e3` does not beat the reference by more than 0.02 at epoch 1,000 on at least 3 of 4
pairs, close the optimizer line and stop spending rounds on lr.

### E2. Batch x lr, properly crossed, on the INCUMBENT objective

Settles the confound in section 2 for the loss we actually use. 8 runs, 2 cards, ~1 day.
Epoch-matched, so every arm sees the same number of passes over the data; report steps too.

```
# B32_ref
trainer.max_epochs=1000
# B64_samelr        (isolates the batch, leaves per-example lr halved -- the v9 arm's condition)
data_module.batch_size=64 trainer.max_epochs=1000
# B64_lr6e4         (linear lr scaling, so the per-example step matches B32_ref)
data_module.batch_size=64 regression_task.optimizer.lr=6e-4 trainer.max_epochs=1000
# B64_lr4p2e4       (sqrt scaling, 3e-4*sqrt(2))
data_module.batch_size=64 regression_task.optimizer.lr=4.24e-4 trainer.max_epochs=1000
```

2 init seeds, split seed 0, all four arms co-resident on one card so packing is a block.
**Read:** level at 1,000, and `train/grad_norm` epoch means. If the gradient really is noise
dominated, doubling the batch should reduce the epoch-mean grad norm by ~sqrt(2) and move the
curve left at matched epochs; if it does not, the noise story in 1c is wrong and I want to
know that.

### E3. Decoupled weight decay, which also un-confounds v15

This one needs a three-line code change before it can run:
`train_cgt_multitask.py:1804`, split `self.parameters()` into two AdamW groups, putting every
parameter with `ndim < 2` plus every `LayerNorm` weight and the ReZero scalars
(`beta_attn`, `beta_ffn`) in a `weight_decay: 0.0` group. Standard practice, and without it
section 3b says the 1e-1 arm is uninterpretable. Then 8 runs at 1,000 epochs:

```
# D_ref
trainer.max_epochs=1000
# D_wd1e2_decoupled
regression_task.optimizer.weight_decay=1e-2 trainer.max_epochs=1000
# D_wd1e1_decoupled
regression_task.optimizer.weight_decay=1e-1 trainer.max_epochs=1000
# D_wd1e1_coupled   (today's behavior, as the control that attributes 3b)
regression_task.optimizer.weight_decay=1e-1 trainer.max_epochs=1000
```

Given that decay is a null across `_007` (n 289, p 0.52), v10 (n 32, t 0.5) and wave 6 (two
single runs inside the band), my honest expectation is another null, and the value of the
round is that the LayerNorm confound stops contaminating it. **Consider deferring E3 behind
E1 and E2**, and in the meantime let v15 finish on its own budget rather than adding to it.

### E4. Weight averaging, at near-zero training cost

`trainer.checkpoint.metric_save_top_k` is already a config key
(`train_cgt_multitask.py:3206`). Set it on the E1 and E2 rounds so the best 20 metric
checkpoints are kept, then average the weights offline and score each average through the
existing eval path, which costs one validation pass (minutes, no training):

```
# on the training round
trainer.checkpoint.metric_save_top_k=20 trainer.max_epochs=1000

# scoring an averaged checkpoint, no fit
trainer.eval_ckpt_path=/path/to/averaged.ckpt
```

**Read:** val and test Pearson of the average against the single best-metric checkpoint, and
`val/loss`, `nmse`, `coverage_50` of both. Prediction from section 6, stated as a hypothesis
so it is falsifiable: the average gains 0.005 to 0.012 in Pearson, lowers NMSE toward
`1 - r^2`, and makes `coverage_50` WORSE, not better, because shrinking the point prediction
does not widen the predicted interval. If coverage improves too, my reading of the
uncalibration is wrong.

A true EMA (a decay-0.999 shadow copy updated in `on_before_optimizer_step`) is maybe 15
lines and is the better long-term form, but the checkpoint-averaging version requires no code
at all and answers the same question first.

### E5 (do not run, it is already answered)

Do not spend another round on epochs past 2,000, on grad clipping, or on the mask schedule as
an explanation for the slow rise. Sections 1b, 1c and the `M_off` run settle all three, and
the 2,000 -> 4,000 interval costs 2.3 days per run for +0.0045 (t 1.6).

---

## Files and artifacts I read

- `experiments/019-simb-multimodal/conf/cgt_expr_v13_split.yaml` and the full defaults chain
  through `cgt_decoder_003.yaml`
- `experiments/019-simb-multimodal/scripts/train_cgt_multitask.py` lines 1261, 1800-1888,
  3132-3365
- `experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh` lines 83-85, 137-145, 245-251,
  296, 310, 360-362, 447
- `torchcell/losses/distributional.py` lines 355-395
- `experiments/019-simb-multimodal/results/`: `round_leaderboards.csv`,
  `pearson_round_readout.json`, `v10_grid_factorial.json`, `loss_min_vs_pearson_peak.json`,
  `split_indices_manifest.json`, `expression_ceiling_replicate.json`,
  `proteome_ceiling_replicate.json`, `lowrank_output_ceiling.json`
- W&B histories pulled fresh: 24 runs of `torchcell_019_expr_v13`, 4 of `_v15`, 16 of
  `_prot_v14`, config pulls on `_v7`, `_v8`, `_v9`
- Working files: `scratchpad/v13_curves.csv`, `scratchpad/v13_levels.csv`, `scratchpad/v7.csv`
