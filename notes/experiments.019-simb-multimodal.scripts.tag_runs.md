---
id: 3y2qnhw90fcj7kf1tjistie
title: Tag_runs
desc: ''
updated: 1785532002348
created: 1785532002348
---

## 2026.07.31 - Make the run list show the constraints a wave is selected on, not just arm names

`A2_prop_sparse` was scheduled into a wave despite failing the organism-transferability requirement: nothing in the W&B run list showed that it routes the perturbation over curated yeast adjacencies, so the constraint was not visible at review time. This script fixes that class of error by declaring the properties a wave is *selected* on as W&B tags -- tags being the one axis the run-list UI filters on, unlike config, which is queryable but not scannable -- and backfilling them onto runs that predate the convention. It is the bookkeeping layer under the scorer: `score_decoder_arms.py` reads its arm label off the tag rather than re-deriving it from config keys, and `wave4b_convergence.py` selects a whole cohort with `--stage-tag`.

- **Eight independent axes, one tag per axis**, so any axis can be filtered alone:

  | axis | example | what it decides |
  |---|---|---|
  | config | `cgt_expr_008` | which config file |
  | arm | `A4_nullsink` | the exact arm; the SCORER derives the arm from this tag |
  | seed | `seed0` | the seed |
  | wave | `wave3` | source-version cohort -- runs are NEVER pooled across waves |
  | stage | `stage-arch` / `stage-ladder` | architecture change vs optimizer ladder |
  | pack | `pack2` | runs co-resident per GPU -- part of the comparison conditions |
  | mech | `mech-nullsink` | hypothesis class, so sibling mechanisms group together |
  | xfer | `xfer-yes` / `xfer-no` | does the mechanism survive without curated yeast graphs |

- **`xfer` is the axis that would have caught A2.** Graph-routing arms (`A2_prop_sparse`, `A3_prop_h1`, `A4_prop_h2`, `A6_prop_sparse`) are `xfer-no`: they need `regulatory_interaction` / `tflink` / `physical_interaction` adjacencies that do not exist for an arbitrary organism, so they are diagnostics regardless of how well they score. The null sink, extra FFN depth, and the reference are `xfer-yes` -- perturbation set plus a per-gene representation only. In the scored table (`experiments/019-simb-multimodal/results/decoder_arms_torchcell_019_expr_v8.csv`: 67 arm runs, 28 arms, 5 waves) 10 rows carry `xfer-no` arms, now excludable with one filter instead of by recalling which arm names mean graph routing.
- **Backfill is for the pre-convention runs only.** `ARM_META` declares mechanism class + transferability for 14 arms; 10 of those appear in the scored table, covering 31 of its 67 rows. From wave 4 on, the same vocabulary is emitted at LAUNCH by `gh_expr_008_arm.sh` (`N_sink_mm` sets `ARM_TAGS=(mech-nullsink xfer-yes stage-wave4b)`), so new runs never need retrofitting. `wave4b_convergence.py --stage-tag stage-wave4b` resolved 8 runs off exactly that declared tag -> `results/wave_convergence_stage-wave4b.csv`.
- **`ARM_NOTE` carries what a tag cannot: intent.** One sentence per arm into `run.notes` (free text in the W&B run table), stating the expected DIFFERENCE rather than the plumbing -- e.g. `A5_sham` is "identical code path and parameter, but the sink is frozen shut (bias -20, p_null ~ 2e-9), so its paired delta must be ~0 -- if it is not, the paired estimator does not transfer to arms that change the forward pass at step 0."
- **The write is deliberately conservative.** Tags are a set union with whatever is already on the run; `run.notes` is written only when the field is empty, so a hand-written note is never overwritten; a run with no recognized arm tag (grid sweeps, probes) returns early untouched. Dry run by default, `--apply` to write, against `zhao-group/torchcell_019_expr_v8`.

  ```python
  arm = next((t for t in run.tags if t in ARM_META), None)
  if arm is None:
      return sorted(tags)  # not an arm run (grid sweep, probe, ...) -- leave alone
  ```

- **Known divergence to respect before re-running.** This script's `WAVE_BOUNDARIES` stops at `wave3`, while `score_decoder_arms.py` has since grown `wave4a` (2026-07-30T01:00:50), `wave4b` (01:38:57) and `wave5` (05:07:59); the scorer's table is the authority for wave bucketing. Re-running the backfill unchanged would stamp `wave3` -- and `pack2` -- onto wave-4/5 runs. That exact collapse already voided results once: per the comment at `score_decoder_arms.py:82-89`, a missing `wave4a` boundary put the wave-4a runs in the `wave3` bucket, so `R_ref` became the in-wave paired reference for every wave3 arm and each delta was measured against a different config whose reference was cancelled at 77-81 epochs versus arms that ran 300.
