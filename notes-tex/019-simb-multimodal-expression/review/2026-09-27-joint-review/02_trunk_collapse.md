# Reviewer 2: the model, and the collapses

Slice: the CGT as configured for v13 to v18, and the mechanism of the failed runs. I take the
coordinator's `ADDENDUM_0207.md` as given (two modes, 7 failures in 114, J_expr's 117 steps
per epoch at ~11 labeled rows) and independently reproduced its mode assignment before
reading it. Everything below traces to a file I read or a W&B history I pulled; scripts are
in this directory (`r02_*.py`), pulled histories in `r02/hist/`, escape tables in
`r02/escape_*.json`.

## 1. The three headline numbers

**(a) The v16 joint-helps-expression result is carried entirely by launch failures.** On the
round's own pre-registered statistic (`results/v16_joint_expr_readout.json`, mean
`val/expression/pearson_per_feature` over epochs 299 to 499), joint minus expression-only is
**+0.0605, sd 0.0395, t 3.43, p 0.019 on 6 pairs**. Restricted to the two (split, seed) cells
where the expression-only run actually launched, it is **+0.0069, sd 0.0085, n 2, t 0.81,
p 0.57**. Four of the six pairs are joint-versus-a-constant-predictor. The contrast that the
PI would be shown is a comparison against a dead control, not a measured synergy.

**(b) The v17 call flips on two runs, and it flips BOTH arms.** On `window_mean` from
`results/v17_locality_readout.json`: `L_prop2` minus `L_ref` is +0.0122 (sd 0.0316, t 1.28,
p 0.226) on 12 pairs and **+0.0224 (sd 0.0190, t 3.73, p 0.004) on the 11 pairs excluding
`sylu3gsw`**; `L_self` is +0.0095 (t 1.43, p 0.181) on 12 and **+0.0137 (sd 0.0138, t 3.13,
p 0.011) on 11 excluding `tnv3dbe4`**. Each arm lost exactly one run. Dropping them post hoc
is indefensible; a pre-registered detector would have made the same exclusions legitimate.
This is the highest-value cheap fix available to the whole campaign.

**(c) Every candidate arm the campaign has run is partly a LAUNCH-TIME lever.** Median epoch
at which `val/expression/pred_sd_ratio` first exceeds 0.05 (`r02/escape_*.json`):

| round | reference | candidate | candidate |
|---|---|---|---|
| v13 | `V_ref` 112 (96 to 124) | `V_concat` 85.5 (77 to 105) | |
| v17 | `L_ref` 113 (101 to 119) | `L_self` 93.5 (78 to 314) | `L_prop2` 60 (19 to 114) |
| v18 | `Y_ref` 105.5 (95 to 139) | `Y_ctx` 61 (45 to 68) | `Y_k0` 64 (54 to 71) |

Every candidate launches earlier than its reference, and the four that score positive
(`V_concat` +0.021, `L_prop2` +0.012, `L_self` +0.010, `Y_ctx` +0.005) are ordered the same
way as their launch advantage. `Y_k0` launches early and still scores -0.0122 (t -2.44,
p 0.033, 4 of 12 positive), so launch time is not sufficient. Within-round Spearman between
launch epoch and window score is -0.35 (v13), -0.28 (v17), -0.06 (v18), none significant, so
**launch time is a marker, not a demonstrated mediator** (hypothesis, test in E2 below).

## 2. The model as configured (v16 resolved config, read with `--cfg job`)

**Where the perturbation enters, relative to graph attention.** Encoder first, perturbation
after. `CellGraphTransformer.forward` (lines 2849 to 3145) runs the six
`GraphRegularizedTransformerLayer`s on `X = [cls; gene_embs].unsqueeze(0)`, i.e. **batch 1 on
the wildtype graph, before any strain exists**, with the nine-graph hard head mask on layer 1
only (`attention_mask.enabled true`, `graph_reg_lambda 0.0`, so the KL term is off and
`total_graph_reg_loss` is a constant zero tensor). `H_genes` and `h_CLS` are therefore
identical for every strain in the dataset. The strain enters at step 5,
`EquivariantPerturbationTransform`: each of the N gene tokens cross-attends the `|S_b|`
perturbed tokens. At `|S_b| = 1` the softmax is over one key, so the attention weight is
exactly 1 for every query and `c_b = W_O(W_V h_p + b_V) + b_O` is query-independent
(`results/perturbation_selector_degeneracy.json`: `s1_weights_unique [1.0]`,
`s1_wq_wk_ablation_max_change 0.0`, 16,200 of 32,760 attention parameters dead). Then
`_apply_residual` with `residual: postln` gives `H_pert = LN(h_i + dropout(c_b))` followed by
`LN(x + FFN(x))`, one layer (`num_layers 1`, `ffn_mult 4`). So **no graph attention ever sees
the perturbation, and the only strain-dependent quantity reaching the readout is one 90-d
vector added identically to all 6,607 tokens, then LayerNormed.**

**How the two heads share the trunk.** Completely. Lines 3060 to 3130: `pg_in` is built once
(`= H_genes_pert` with `concat_context false`) and passed to `self.per_gene_head` and then to
`self.per_gene_aux_head` unchanged. The encoder, the perturbation transform, the
`ObservedLabelEncoder` and the 32-latent `PerceiverMixing` are shared; the two heads differ
only in their own `PerGeneHead` MLP (`Linear(90,90) -> ReLU -> Dropout -> Linear(90,19)`,
9,919 parameters each against 6.67M in the model). There is no task-specific adapter, no
gradient balancing, no uncertainty weighting.

**How the joint loss is weighted.** `MaskedMultitaskLoss.forward`: `total = graph_reg +
sum_h w_h * loss_h` with `head_weights.per_gene = per_gene_aux = 1.0` and no normalization by
the number of active heads, so the joint arm's trunk receives roughly twice the gradient
magnitude of either single-head arm. `J_joint05` halves the aux weight and was not run in the
18. The per-head loss is `DistHead.loss` with `mode quantile`, `masked_mean(pinball(...))`
over the selected entries, so each head's term is a per-entry mean and a head supervised on 11
rows and a head supervised on 31 rows contribute equal-magnitude terms.

**Per-gene standardization and NaN masking.** `standardize_per_feature_target: [per_gene,
per_gene_aux]` forces plain per-feature z-score (`head_norm_method`), fitted on the TRAIN
split only in `compute_per_feature_target_stats`, stored as buffers, applied to the decoded
target in `_extract_targets_and_masks` and inverted in `_cache_epoch_metric` for the raw-unit
metric. Consequence that matters here: **after z-scoring, the constant "predict each gene's
marginal quantile" solution has target mean 0 and sd 1 per gene, so it is the exact
strain-blind optimum and it is reachable from `h_i` alone.** NaN (unquantified proteins) enters
twice: `MaskedMultitaskLoss.forward` folds `torch.isfinite(target)` into the feature mask and
zeroes the target, and `per_feature_pearson` routes to `_per_feature_pearson_sparse`, dropping
features with fewer than 3 finite pairs. Both are correct.

**Optimizer.** AdamW, lr 3.0e-4 constant, weight decay 1e-8 applied to every parameter
including LayerNorm gains, `lr_scheduler.type: null` so **no warmup and no decay**;
`clip_grad_norm true` at `max_norm 10.0`; `precision: bf16-mixed`; batch 32.
`train/grad_norm_clip_frac` runs 0.002 to 0.018 in every history I pulled, so **clipping never
binds and is not a lever here**. The learning rate is not logged at all (no `lr` key in any of
the 87 history columns).

**Residual / normalization variant.** `postln` in every v13 to v18 run. `rezero` exists
(`beta_attn`, `beta_ffn` init 0) and the module docstring cites the exact failure it was built
for ("identical config, three seeds, 0.1527 / 0.0235 / 0.0661, with seed 1 stuck at ~0.015 for
all 83 epochs"). **It has never been measured.** The only `S2_rezero` / `S3_both` / `S1_warmup`
runs in W&B are four runs in `torchcell_019_expr_v8` from 2026-07-29 that reached epochs 40,
73, 76 and 121 (`3d8opzhr`, `etaxogte`, `p7ayltm8`, `3m8d668j`; final `pred_sd_ratio` 0.040,
0.105, 0.047, 0.077). Every one was cut off **inside the pre-launch plateau**, whose median
end is epoch 67 to 113 in this trunk, so the round carries no information about either lever.
"rezero did not help" is not a measured claim; it is not measured.

**Zero-gated and forced-on pathways, and the telemetry that does not exist.** In these configs
`observed_labels.gate_mode: "on"` and `post_perturbation_mixing.gate_mode: "on"`; both register
`gate` as a **buffer**, and `_log_gates` iterates `named_parameters`, so **no `gate/*` key is
logged by any v13 to v18 run** (confirmed: no `gate/` column in the histories). The one
comment in the file warning that an unlogged gate cost the project a round is therefore
inoperative for the arms that ran. Separately, `ObservedLabelEncoder.forward` computes
`H + gate * proj([v*m, m])` where `proj` is a biased `Linear(2, 90) -> ReLU -> Linear(90, 90)`.
With everything masked (all validation, and the k=0 training steps) the input is `[0, 0]` and
`proj([0,0])` is the **bias path, not zero**, so the docstring's "a 100%-masked forward is
still an identity" is false: it is an identity plus a learned constant vector added to every
token. Strain-independent, so harmless to the metric, but it is an extra unconstrained
90-vector in the residual stream and the claim in the code is wrong.

**Checkpoint selection.** Three `ModelCheckpoint`s plus a fourth when `save_loss_min`: best by
`trainer.checkpoint.monitor` (v16 `val/proteome/pearson_per_feature`, J_expr overridden to
`val/expression/pearson_per_feature`, mode max), best by `metric_monitor`, `save_last`, and
best by `val/loss`. `trainer.test(ckpt_path="best")` loads the FIRST one. This selection is
right in general (`results/loss_min_vs_pearson_peak.json`: for `quantile`, the val-loss
minimum is at median epoch 477 while the Pearson peak is at median 3,269) and **wrong for a
never-launched run**: it selects the largest noise spike of a zero-variance prediction.
Measured consequence: `hyasw3nx` (val Pearson identically ~0, roll5 max 0.0246) reports
`test/expression/pearson_per_feature 0.0770`, while `m13meldl`, the same failure mode, reports
0.0068. Those two numbers are the spread of one noise draw, and `run_test` produces them
without any flag.

## 3. The failure taxonomy, from the histories

Launch detector used throughout: first epoch where `val/<pheno>/pred_sd_ratio > 0.05`. That
series is the sd of the prediction across strains divided by the sd of the target, averaged
over features, logged since the metabolism arms (`_reduce_epoch_pearson`, line 1580). It is the
right instrument and it is already logged in every run.

**Mode A, never launched.** `pred_sd_ratio` never leaves ~0; `nmse` pinned at 1.007 to 1.011
(exactly "predict each gene's mean"); `val/loss` pinned at the pinball floor; `traineval`
Pearson and `traineval` sd_ratio both ~0, so the run does not fit its own training set either;
grad norm decays monotonically.

| run | arm | max sd_ratio | val_pf final 20 ep | nmse final | traineval pf | grad norm ep 0 -> end |
|---|---|---|---|---|---|---|
| `hyasw3nx` | v16 J_expr s0 seed0 | 0.0193 | -0.0007 | 1.0078 | -0.0228 | 0.115 -> 0.020 |
| `c1n1jmp3` | v16 J_expr s1 seed1 | 0.0021 | +0.0001 | 1.0094 | +0.0089 | |
| `o3zk5y0i` | v16 J_expr s2 seed1 | 0.0008 | -0.0011 | 1.0114 | -0.0257 | |
| `m13meldl` | v16 J_expr s1 seed0 | 0.0379 | -0.0076 | 1.0087 | -0.0119 | |
| `825on260` | v13 V_ref s0 seed1 | 0.0010 | +0.0005 | 1.0095 | +0.0024 | 0.168 -> 0.011 |

https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/hyasw3nx

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/825on260

Two facts make this an absorbing state rather than a slow start. `825on260` ran **5,706
epochs** (about 222,000 optimizer steps) at `pred_sd_ratio` 0.0000 and `val/loss` 0.2525
throughout. And the v16 continuation resumed `hyasw3nx` from its `last.ckpt` as a new run
(`zwiulzhf`, `wandb.resumed_from hyasw3nx`, same weights and Adam moments): at epoch 555 it is
still at `val/expression/pearson_per_feature -5.8e-10` and `traineval -0.0257`. **A checkpoint
restart does not rescue mode A. Any restart rule must redraw the initialization.**

https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/zwiulzhf

Not co-residency: `hyasw3nx` shared compute-0-2 with `k7rzebp0` (J_joint) and `rh9wgkoo`
(J_ref), both of which launched at epochs 144 and 61. Not the seed axis alone either: split 1
failed at both seeds, splits 0 and 2 failed at one of two.

**Mode B, de-launch after a long healthy run.** `p204pwqb` (v18 `Y_k0` split 3 seed 0) launched
at epoch 63, climbed to `val_pf` 0.135 at epoch 1085 with `pred_sd_ratio` 0.484, then fell into
exactly the mode-A state: by epoch 1175 `sd_ratio` 0.0001, `nmse` 1.0076, `val/loss` 0.2411,
and at epochs 1197 to 1199 it is climbing back out (`sd_ratio` 0.012, 0.093, 0.174). Its
`window_mean` on the round statistic is 0.103, not far below the `Y_k0` arm mean of 0.1265,
because the v18 window is epochs 921 to 1121 and the excursion sits mostly after it. **Correction
to the addendum: `p204pwqb` did not end at 0.036; that is the last-20-rows value, and the
pre-registered statistic is 0.103.** The finding that survives is the important one: a run that
has trained for 1,000 epochs can fall back into the same zero-spread state, so a launch
detector has to run for the whole run.

https://wandb.ai/zhao-group/torchcell_019_expr_v18/runs/p204pwqb

**Mode C, launched but never consolidated.** `sylu3gsw` (v17 `L_prop2` split 3 seed 2) launched
at epoch 61, peaked at `val_pf` 0.066 at epoch 185, then held 0.01 to 0.03 for 1,000 epochs
while `pred_sd_ratio` **oscillated by 5x between adjacent epochs** (0.078, 0.457, 0.107, 0.355
on consecutive validation epochs near 1,180) and `nmse` with it (1.004 to 1.222). Its train fit
reached `traineval` 0.250 against 0.665 and 0.703 for its two card mates `uz3xe13h` and
`otr4u6db`, and its `val/loss` minimum (0.2316) is the worst of the three. So it is not overfit
and not mean-collapsed: it is a run whose prediction SCALE never settled. The code path that
distinguishes it from its mates is the one the arm turns on: `L_prop2` sets
`perturbation_propagation.enabled=true hops=2 gate_mode=on`, which forces 19 reachability
features per gene token through a gate that cannot close. `L_prop2` also has by far the widest
spread of the six arms in v17 and v18 (window sd 0.0441 against 0.0268 for `L_ref`). n = 1,
so this is a hypothesis, not a result.

https://wandb.ai/zhao-group/torchcell_019_expr_v17/runs/sylu3gsw

**Mode D, late launch, wrongly called a collapse.** `tnv3dbe4` (v17 `L_self` split 1 seed 0)
sat at `sd_ratio` 0.020 from epoch 50 to 300, launched at epoch 314, and rose monotonically to
`val_pf` 0.115 with `traineval` 0.592, still rising at epoch 1,199. Its card mates launched at
79 and 118. Its low window score is a truncation artifact of a fixed scoring window, not a
failure of the model.

https://wandb.ai/zhao-group/torchcell_019_expr_v17/runs/tnv3dbe4

## 4. Why a head fails to launch

**The strain-blind solution is an exact stationary point of the restricted problem, and it is
reachable from a strain-invariant input.** The readout is one shared MLP applied to
`LN(h_i + c_b)`. Set `c_b`'s contribution to zero and the head is a function of `h_i` alone,
which is identical for every strain. Under per-gene z-scoring the pinball-optimal
strain-independent prediction for gene g is gene g's marginal quantile vector, the same for all
strains. The measured signature is exactly that state: `nmse` 1.007 to 1.011 (the per-gene mean
predictor is `nmse` 1.0 by construction), `val/loss` at 0.2410 to 0.2580 against 0.2307 for
`hzca60i6`, the launched run on the same partition and card group, and a grad norm that decays by 10x. The gap between the
plateau and the escaped solution is **3% of the loss** (`hyasw3nx` 0.2392 versus `hzca60i6`
0.2310), so the objective barely prefers the model that works.

**The pinball derivative carries no magnitude information.** `pinball` (distributional.py:361)
returns `max(tau*u, (tau-1)*u)`, so `d/dq` is exactly `-tau` above the knot and `1-tau` below:
bounded by 0.95, independent of `|residual|`. A reporter 8 sd from its mean (the deleted gene's
own value, measured true mean -2.42) contributes the same +/-0.5 as a pure-noise reporter. At
the strain-blind fixed point, averaged over the K=19 grid `linspace(0.05, 0.95)`, the per-entry
gradient noise sd is `sqrt(mean tau(1-tau)) = sqrt(0.175) = 0.4183`, while the per-entry launch
signal is `mean phi(Phi^-1(tau)) * r / sqrt(1-r^2) = 0.2957 * r / sqrt(1-r^2)`, i.e. 0.060 at
the r = 0.2 this task actually achieves. So the useful component is **14% of the noise per
scored entry**, and the only independent replicates of the strain direction in a batch are the
rows.

**Label dilution costs a factor 1.7 in per-step launch SNR, and it is the one axis that
separates J_expr from every arm that worked.** Simulation (`r02_pinball_snr.py`, planted rank-1
strain signal, F = 1500 genes, 200 trials, gradient of the loss along the true strain direction
at the strain-blind fixed point; this is a simulation of the readout geometry, not a measurement
of the real model): per-step SNR for pinball at r = 0.2 is 4.06 at n = 32 labeled rows and
2.38 at n = 11 (`= 4.057 / sqrt(32/11)`); the scaling is exactly `sqrt(n)` across 11, 31, 32,
64, 128. MSE gives 4.08 at n = 32 and 2.18 at n = 11, so **the SNR argument does not single out
pinball**; what singles out pinball is the flat, magnitude-blind gradient and the exactness of
the strain-blind stationary point. The measured launch distributions line up with the row count:

| arm | labeled rows per batch of 32 | steps/epoch | launched | median launch epoch |
|---|---|---|---|---|
| v13 / v17 / v18 expression | 32 | 40 | 95 of 96 | 67 to 113 |
| v14 / v16 J_ref proteome | ~31 | 112 | 22 of 22 | 35 to 62 |
| v16 J_joint proteome head | ~31 | 117 | 6 of 6 | 95 to 144 |
| v16 J_joint expression (aux) | ~11 | 117 | 6 of 6 | (launched) |
| **v16 J_expr** | **~11** | **117** | **2 of 6** | **138, 195** |

Across 95 launched expression runs on the undiluted store the launch epoch never exceeded 314
and its 90th percentile is at most 115 in every round, so the 500-epoch v16 budget was not the
problem. The J_expr arm's two launches are at 138 and 195, both past every other round's p90,
and four never launch. **Hypothesis (untested): the mechanism is the row count, and the joint
arm's expression head survives the same 11 rows because the proteome loss on ~31 rows moves the
shared trunk out of the strain-blind region first.** The two-run launched-only contrast
(+0.0069) is consistent with the aux head then getting the same score as the single head, but
n = 2 resolves nothing.

**The reveal schedule is not implicated, and may protect.** `_masked_step` samples ONE k per
training batch from `[0, 10, 100, 1000]`, so only 25% of steps are unconditioned, and at k > 0
the head is handed 10 to 1,000 true values of its own label, which the Perceiver then spreads.
That is a competing descent direction that needs no strain conditioning, and the never-launched
runs do take it: `hyasw3nx` reaches `val/expression/pearson_per_feature@k3 = 0.2396` while its
k0 branch is identically 0. But the arm that removes the schedule is the arm that fails: v18's
`Y_k0` (mask off) is the only arm in the campaign that clears its own paired test, at
**-0.0122 (t -2.44, p 0.033, 4 of 12 positive)**, three of the four lowest window means in the
round are its runs, and the one mode-B de-launch in v18 is in it. So the objective's ~25% dilution of the
unconditioned gradient is real but its net effect at matched epochs is protective, not harmful.

**Not implicated, on the evidence:** gradient clipping (`clip_frac` 0.002 to 0.018, never
binds); the lr scheduler (there is none, in every arm); the ReZero and zero-gated pathways as
such (`residual: postln` in every run, and every gate in these configs is a forced-on buffer,
so nothing was gated shut); co-residency (launched and never-launched runs shared cards);
DDP (`devices: 1`, `strategy` forced to `auto`); the NaN masking (correct in both the loss and
the metric); the split partition (both seeds failed on v16 split 1, one of two on splits 0 and
2). `bf16-mixed` is a candidate I cannot rule out: the never-launched state has `pred_sd_ratio`
of order 1e-4, and I did not check whether the head's across-strain spread falls below bf16
resolution relative to `||h_i||`. That is measurable from a checkpoint (E1).

## 5. What this does to the PI's question

A provable joint statement requires the single-head controls to be alive. Right now:

- the expression side of v16 is joint-versus-dead on 4 of 6 pairs, and +0.007 on n = 2 where
  both launched;
- the proteome side is -0.0076 (t -1.41, p 0.218, 6 pairs), with all 12 runs launched, so that
  half is a clean measured null at an MDE of about 0.019 at 6 pairs;
- the joint arm's proteome head launches at epoch 95 to 144 against 44 to 61 for
  proteome-only (6 of 6 in each arm), so joint training demonstrably SLOWS the proteome head, which is the most
  likely explanation of the -0.008 at a 500-epoch budget (hypothesis; the 1,200-epoch
  continuation is the test and is at epoch ~605 of 1,200, partial).

## Proposed experiments

Ranked by evidence value per GPU-day inside 14 days. E1 and E2 are CPU-only and must precede
any launch. The pre-registered detector in E1 is a prerequisite for every GPU round below,
because without it the v17 situation recurs and no exclusion is defensible.

**E1. Pre-registered launch detector, restart rule, and a checkpoint autopsy. CPU, 0 GPU-days,
half a day of work.** Declare, before any run: a run is NOT LAUNCHED at epoch e if
`max(val/<pheno>/pred_sd_ratio)` over epochs 0..e is below 0.05; DE-LAUNCHED if it launched and
then holds `pred_sd_ratio < 0.05` for 25 consecutive validation epochs. Rule: a run not
launched by epoch **200** (the 95th percentile of the 95 launched expression runs is 115, the
maximum is 314, so 200 is deliberately loose and 314 is the hard bound) is killed and
**relaunched from a new init seed**, recorded as a relaunch, and the failed segment is excluded
from the paired statistic; a de-launched run is scored on its window as logged and flagged. A
resumed checkpoint is NOT an acceptable relaunch (`zwiulzhf` proves it does not escape). Ship
this as a `LaunchGuard` Lightning callback (the detector needs only
`trainer.callback_metrics`) plus a killed-run and not-launched guard in `v13_split_readout.py`,
and add `lr` and `gate/*` for buffer gates to the logged keys. Autopsy, same day, on the
existing `last.ckpt` of `hyasw3nx`, `825on260`, `hzca60i6` and `rh9wgkoo`: report
`sd_b(c_b) / ||h_i||` across a validation batch, the norms of `W_V` and `W_O`, and the
across-strain sd of the head output in fp32 versus bf16. Decision: if the never-launched
checkpoints have `c_b` spread at bf16 resolution, precision is the cause and E4 changes to a
precision arm. Value: makes every later call defensible; retroactively legitimizes the v17
11-pair reading.

**E2. Re-score every existing round at MATCHED POST-LAUNCH epoch. CPU, 0 GPU-days.** For each
run define `epoch - launch_epoch` and recompute each round's paired statistic over
post-launch epochs 800 to 1,000 (v17, v18), 300 to 500 (v16), 5,400 to 5,600 (v13). This
separates "the arm reaches a higher ceiling" from "the arm launches sooner", which no round has
done, and it is the direct test of the mediation hypothesis in section 1(c). Decision rule:
an arm whose advantage disappears at matched post-launch epoch is a speed lever and must be
reported as one; an arm that keeps it is a ceiling lever. Value: reinterprets four rounds and
about 120 GPU-days already spent, for free.

**E3. The clean joint contrast, three arms on an identical undiluted store.** Arms: `K_prot`
(proteome only), `K_expr` (expression only), `K_joint` (both), all three on the
`fig3_proteome` store with `require_modalities: [protein_abundance, expression_log2_ratio]`, so
every batch row carries BOTH labels and the ~1,099 both-labeled train strains are the instance
set for all three arms. 4 split seeds x 3 init seeds = **12 pairs per contrast**, 1,200 epochs,
3 arms co-resident per card, 12 tasks. `gpu` at `--array=..%3` (A40, walltime unlimited), about
30 h per task from v17's measured rate, so **about 15 card-days, 5 calendar days at 3 cards**.
Pre-registered statistic: mean `val/<pheno>/pearson_per_feature` over epochs 1,000 to 1,200,
paired t on the 12 (split, seed) differences, per head, with the E1 relaunch rule live.
Decision: **the PI can say joint training helps provably if `K_joint` minus `K_expr` on the
expression head AND `K_joint` minus `K_prot` on the proteome head are both above +0.02 with
Bonferroni-corrected CIs excluding 0, at least 3 of 4 partitions positive on each, and the
test-side sign agreeing on each.** At the pooled paired sd of 0.0167 from the addendum, 12
pairs give MDE 0.015, so +0.02 is detectable. This is the only design in the list that can
produce the statement the PI asked for. It also removes the row-count confound entirely, which
is what makes a null here interpretable.

**E4. Launch-hardening ladder on the diluted regime, 4 arms, one round.** The point is to make
the failure impossible so that dilution is no longer a hidden covariate, and to finally measure
the two levers the July stability arms never reached. Arms, all on the v16 J_expr configuration
(11 labeled rows per batch, the regime that fails 4 of 6, giving the round real power on a rare
event): `H_ref` (as is), `H_warmup` (`lr_scheduler.type=CosineAnnealingWarmupRestarts
warmup_steps=50 first_cycle_steps=1200 max_lr=3e-4 min_lr=1e-6`, i.e. warmup only, no peak
change), `H_rezero` (`model.perturbation_head.residual=rezero`), `H_both`. 3 split seeds x 4
init seeds = 12 runs per arm, 500 epochs, 4 per card, 12 tasks: **about 12 card-days, 4 calendar
days on `gpu` %3**. Pre-registered statistic: **launch fraction by epoch 200** (binomial, so
12 runs per arm resolves 0.33 versus 0.92 at p < 0.01 by Fisher), with window mean as a
secondary. Evidence for it: `postln` divides out the magnitude of `h_i + c_b`, which is where
`<h_i, c_b>` lives, and the module's own docstring records a 3-seed 0.1527 / 0.0235 / 0.0661
spread under it; the four historical `S*` runs died at epochs 40 to 121 and carry no
information. Decision: adopt whichever arm launches 11 of 12 or better and does not lose more
than 0.01 of window mean; then re-run E3's reference with it.

**E5. Two cheap objective changes against the flat-gradient diagnosis. 12 card-days.** Arms on
the undiluted v18 reference: `O_ref`, `O_huber` (`dist: point` with a Huber/`l1` elementwise
loss, i.e. a residual-scaled gradient without MSE's mean-collapse optimum), `O_pinball_mse`
(pinball plus a small masked-MSE anchor, weight 0.1, the anchor form `pearson_mse` already
implements for a different loss). 4 splits x 3 seeds x 3 arms, 1,200 epochs, 3 per card, 12
tasks, about 30 h each. Statistic and decision as E3's template. Evidence: the pinball
derivative is bounded and magnitude-blind (section 4), the pinball median de-weights the
deleted gene's own cell 8.9x against MSE (expression-fit review), and `dist: point` collapsed
6 of 6 in v9 for the opposite reason, so an anchored or robust loss is the untested middle.
Rank it below E3 and E4 because it can only change the level, not the validity of the joint
claim. Do NOT run it before E1, or a collapse will again be indistinguishable from a null.

**Explicitly not proposed.** Gradient clipping changes or an EMA of the weights: clipping never
binds (`clip_frac` <= 0.018) and an EMA of a run parked at the per-gene median produces the
per-gene median. Removing the reveal schedule to buy the 24% wall: v18 measured it at
-0.0122 (t -2.44, p 0.033) at matched epochs and its one de-launch is in that arm. Selecting a
different checkpoint: the monitor is already the metric, and the problem with the current
`run_test` is that it scores a never-launched run at all, which E1 fixes with a flag.
