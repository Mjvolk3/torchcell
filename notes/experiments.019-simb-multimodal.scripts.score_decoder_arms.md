---
id: z8uohsyjj7df5ad7mf3pbq6
title: Score_decoder_arms
desc: ''
updated: 1785531951607
created: 1785531951607
---

## 2026.07.31 - Make an Arm Comparison Survive Its Own Noise, and Retract the Best Result We Had

This scorer exists because the _008 arm sweep is too noisy to read by eye and three separate pooling bugs each silently voided a whole round of numbers before anyone noticed. Unpaired comparison is dead on arrival on this task -- the across-seed sd of an **identical** config is 0.0652, 41% of the baseline's own score, giving a one-seed unpaired MDE of 0.181, larger than the entire signal we are chasing -- so the seed has to be treated as a shared nuisance and every reported number is a **within-seed delta against a reference arm co-launched in the same wave**. The immediate payoff was a retraction: the arm on record as the best result anywhere, `A2_prop_sparse` at +0.0231, was a cross-wave comparison; scored against a valid in-wave reference it is +0.0030 with a 95% CI of +/-0.0321, i.e. never a win.

**The retraction, from the committed artifact.** `experiments/019-simb-multimodal/results/decoder_arms_torchcell_019_expr_v8.csv` (68 rows, written by this script, committed in `23ae858b`), wave 3, paired against `A1_ref` on the four shared seeds:

| seed | `A1_ref` smoothed | `A2_prop_sparse` smoothed | delta |
| --- | --- | --- | --- |
| 0 | 0.1576 | 0.1305 | -0.0271 |
| 1 | 0.0310 | 0.0443 | +0.0132 |
| 2 | 0.0723 | 0.0880 | +0.0158 |
| 42 | 0.0436 | 0.0537 | +0.0100 |

mean **+0.0030**, sd **0.0202**, t(3) 95% CI **+/-0.0321** -> `CI CONTAINS 0`. The sd is pandas `.std()`, i.e. **ddof=1** (sample sd), and the CI is built from *this arm's own* per-seed deltas -- not from the pooled paired sd of 0.0039, which is heteroscedastic across arms (0.0055 vs 0.00084, F(2,2)=42.9, p~0.023), was measured only on arms that are exact identities at init, and was measured pre-edit.

- **Three pooling bugs, each of which silently produced numbers rather than an error.**
  1. *Arm derived from config, not declared.* The old version inferred the arm from `multitask.free_gene_dim` + `model.perturbation_propagation` alone, so every arm that moved neither axis -- C0, D1, D2, E0, F1, G1, H0, GEARS, S1/S2/S3 -- fell through to the literal string `"A0_baseline"`. The reported baseline was a blend of **eleven different architectures**, and its "across-seed spread" was really an across-*arm* spread. Now the arm is read from the run TAG that `gh_expr_008_arm.sh` writes from the same `case` statement that sets the overrides; an untagged run **raises**.
  2. *A missing wave boundary.* `wave4a` (cgt_expr_010, hard masking frozen) was created at `01:00:50`, after the wave3 boundary, so without its own boundary those runs landed in the `wave3` bucket -- and wave4a is where `R_ref` was introduced, so `R_ref` became the in-wave paired reference for every wave3 arm. Each wave3 delta was then measured against a different config, a different graph-prior mechanism, and a reference cancelled at 77-81 epochs versus arms that ran 300.
  3. *POOL = HOST, not `gpu_type`.* IGB's cabbi partition reports the identical device string `"NVIDIA RTX 6000 Ada Ge"` as GilaHyper, so keying on the model merged GilaHyper's 2-runs-per-GPU packing with cabbi's 1-run-per-GPU probes and produced a duplicate `(arm, seed)` that crashed the paired report. The host fixes hardware *and* packing regime together. `budget` (`round(epochs/50)*50`) completes the key, because wave 5 ran the same arm and seed at 400 and at 300 epochs with its own `W_ref` at each.
- **The phantom noise floor this table exists to kill.** The 0.0032 "within-config noise floor" quoted earlier in the round is not a noise floor: both of its replicates straddle the `2026-07-28T22:26:23` source boundary *and* differ by 55-76 epochs, so it measured the source edit. Same class of error as the `A2_prop_sparse` headline.
- **Scored statistic is stated, not implied.** `smoothed` = max over epochs of a centered 5-epoch rolling mean; `raw_peak` (the max) is retained **only** so the selection bias stays visible, since it is an upward-biased order statistic whose bias grows with the number of epochs run -- an arm that merely trains longer otherwise "wins". Consecutive-epoch swings of ~+/-0.05 on run `kmd40o2h` are what forces the smoothing.
- **Shape-match, don't whitelist.** The first fix for bug (1) hardcoded allowed arm initials `A..W` and omitted `R`, so every `R_ref` run raised "no arm tag" -- the same class of bug via a different hardcoded list. The pattern is now `^[A-Z][A-Za-z0-9]*_`, which accepts every arm tag (`A1_ref`, `N_sink0`, `R_ref`, `B_film`) and rejects every lowercase axis/config tag (`mech-baseline`, `cgt_expr_010`) without enumerating either.
- **Runs shorter than the window are skipped loudly.** `min_periods=window` leaves the whole smoothed series NaN, so `idxmax()` raised `cannot convert float NaN to integer` and took the entire report down when six crashed wave-4a runs (2-4 epochs each) entered the project.
- **The train-side series nobody had read.** `history(keys=[...])` with a key that is not co-logged with the others silently returns **zero rows**, which is why `train/expression/*` had never been inspected; reading one key at a time via `scan_history` shows the model does not fit its own training set -- wave-3 `A1_ref` peaks at `train_pf_best` 0.2600 / 0.2684 / 0.2733 / 0.2671 across seeds 0/1/2/42 while val `smoothed` is 0.0310-0.1576.
- **Caveat on the retraction pairing itself.** `A2_prop_sparse` ran 50-51 epochs against `A1_ref`'s 242-250, so that pairing is budget-mismatched; it predates `budget` entering the key, and under the current `(wave, pool, budget, arm)` key the two arms land at budget 50 and 250 and no longer pair at all. The verdict stands either way -- there is no valid pairing in which `A2_prop_sparse` is a win. The committed CSV likewise predates the `pool`/`budget` columns; it covers waves 1-4b only.
