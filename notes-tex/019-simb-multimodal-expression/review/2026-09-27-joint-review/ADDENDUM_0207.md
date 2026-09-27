# Addendum to the brief, 2026-09-27 02:07. Read before resuming. Do not redo these.

Established by the coordinator while you were paused (sources: W&B histories via the API,
run configs, `experiments/019-simb-multimodal/results/*.json`; census at
`joint_review/launch_census.json` in this directory):

1. **Collapse has two modes.** NEVER LAUNCHED: `val/<head>/pred_sd_ratio` stays at 0 from
   epoch 5 (the head predicts each gene's median; val loss pinned at the pinball floor 0.2525
   to 0.258; grad norm decaying): `825on260` (v13 V_ref), `hyasw3nx`, `c1n1jmp3`, `o3zk5y0i`
   (v16 J_expr). DECAYED after launching (train Pearson still rising, val falling):
   `m13meldl` (v16 J_expr, peak 0.051), `sylu3gsw` (v17 L_prop2, train 0.22 while val fell
   to 0.009), `p204pwqb` (v18 Y_k0 split 3 seed 0, 0.135 to 0.036). `tnv3dbe4` is a LATE
   LAUNCH (spread 0.02 until epoch 250, ended 0.115), not a collapse. 7 failures in 114
   first-segment runs; 0 of the 12 joint or proteome-only heads in v16.
2. **The v16 expression-only arm (J_expr) is a confounded control.** Its config equals v17
   L_ref except the store (`fig3_proteome`, `require_modalities []`) and an inactive aux head.
   It trains on the same 1,244 expression strains but iterates the whole union of 3,726
   genotypes: 117 optimizer steps per epoch (global_step 58,500 at epoch 500) against 40 in
   v17, so each batch of 32 holds about 11 labeled rows and the rest are masked out of the
   loss. Failure 4 of 6 against 3 of 108 expression-only runs elsewhere (Fisher p = 8e-5).
   The joint arm's expression head is `per_gene_aux` (no reveal schedule, z-scored per gene),
   sees the same ~11 labeled rows per step plus the proteome loss on ~31 rows, and launched 6
   of 6 (p = 0.061 against J_expr). J_ref uses `require_modalities [protein_abundance]`
   (3,581 train rows, 112 steps/epoch). Strains with BOTH labels in the train split: about
   1,099. Note also that J_expr's head is the MASKED `per_gene` head while the joint's
   expression head is unmasked, a second difference between the two expression heads.
   The confound is documented as intended in `conf/cgt_expr_v16_joint.yaml` ("differ in
   nothing but the active heads"); its consequence was not anticipated.
3. **Power (clean pairs, fixed window):** paired-difference sd 0.012 to 0.023 across ten
   contrasts, pooled 0.0167; reference seed sd within partition 0.006 to 0.014;
   between-partition sd 0.013 to 0.035. At sd 0.0167 and alpha 0.05 two-sided, 80% power:
   +0.02 needs 8 pairs, +0.01 needs 24; MDE 0.015 at 12 pairs, 0.012 at 18, 0.010 at 24,
   0.008 at 36.
4. **Continuation state:** IGB 2410399 tasks 0 and 1 at epoch ~605 (split 0), tasks 2 to 5
   pending under the %3 throttle behind other users.

Take these as given, cite the addendum, and spend your remaining effort on your own slice
and on the `## Proposed experiments` section. Keep your report under about 400 lines.
