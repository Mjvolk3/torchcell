---
id: 1whc5tbi52rhsj5mv068aoz
title: 'Handoff 2026.07.26 — CGT-Metabolism Track A'
desc: 'Session handoff: Track A pigment/metabolome transfer -- what was built, the honest scientific state, verification checklist, and the traps'
updated: 1785094150992
created: 1785094150992
---

## 2026.07.26 - Handoff: CGT-Metabolism Track A

**Read first:** [[plan.cgt-metabolism.2026.07.25]] (the plan + all dated run logs).
Math/derivations: [[scratch.2026.07.25.172340-adding-metabolism-explainer]] (Q1-Q15).
Dataset substrate + defects: [[scratch.2026.07.25.010919-adding-metabolism]].

### Where the work lives

ALL of it is on branch `plan/cgt-metabolism`, in the WORKTREE -- not the primary checkout:

`/home/michaelvolk/Documents/projects/torchcell.worktrees/plan/cgt-metabolism`

| artifact | path (relative to that worktree) |
| --- | --- |
| model class | `torchcell/models/cell_graph_transformer_metabolism.py` |
| aggregator | `torchcell/data/genotype_aggregate.py` (`DeletionKeyedGenotypeAggregator`) |
| config | `experiments/019-simb-multimodal/conf/gh_pigment_base.yaml` |
| SLURM (GH) | `experiments/019-simb-multimodal/scripts/gh_pigment_transfer.slurm` |
| SLURM (Delta, unsubmitted) | `experiments/019-simb-multimodal/scripts/delta_pigment_transfer.slurm` |
| driver | `experiments/019-simb-multimodal/scripts/run_pigment_conditions.py` |
| results | `experiments/019-simb-multimodal/results/pigment_transfer_runs*.json` |
| SLURM logs | `/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/019-pigment_<jobid>.out` |
| dataset LMDB | `/scratch/projects/torchcell-scratch/data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer` |

Commits: `6cf09ec4` (aggregator) - `e8648ddb` (Track A) - `fe6fe8c1` (pert pool) -
`e702cb7e` (main() + correction) - `73c956a7` (debugger import fix). **Not landed to main.**

### What was built

1. **`DeletionKeyedGenotypeAggregator`** - keys on the DELETION gene-set only, treating the
   constant heterologous cassette as reference-strain background. THE GATE: pigment/metabolome
   co-location goes 0 -> 4,432 (betaxanthin) and 0 -> 4,221 (beta-carotene). Without it the
   transfer experiment has literally zero data.
2. **`fig6_pigment_transfer` build** - 4,930 genotypes over exactly three datasets: Cachera
   betaxanthin (scalar), Ozaydin beta-carotene (ordinal -5..+5), Mulleder 19-AA metabolome.
   4,023 carry all three modalities.
3. **`CellGraphTransformerMetabolism`** - subclasses the equivariant CGT (inherits encoder +
   PERT operator rather than transcribing them), adds three SEPARATE heads. Units are mutually
   incomparable so they must not share a head. 7/7 tests pass incl. an atol=0 encoder parity test.
4. **Harness defect fixes** in `train_cgt_multitask.py`: real label routing
   (`metabolite_level` / `visual_score`), uncapped phenotype scan, explicit `scalar_heads`,
   scalar-head standardization. Also a `cell.py` fix: deletion-keyed groups span 3 perturbation
   counts, which had produced OVERLAPPING train/val/test indices.
5. **Noise ceilings** - betaxanthin r = 0.914 (n up to 44). beta-carotene rank agreement 0.54,
   and only 0.075 on the paper's own independent re-screen. Use SPEARMAN for beta-carotene.

### The scientific state -- honest version

**The model trains end-to-end on SLURM but does not learn either target.** Val loss parks at
~0.97 = the variance of the z-scored target; pearson wanders within noise (345 val genotypes,
SE ~ 0.054) against a 0.914 ceiling.

12-run learnable-embedding sweep (3 seeds x 4 conditions), archived as `*_learnable_baseline.json`:
`delta_betaxanthin = +0.042 +/- 0.064` vs `delta_beta_carotene = +0.010 +/- 0.019`. Predicted
direction, **NOT statistically supported** -- one seed carries it, and in that seed the
betaxanthin-alone arm never converged. The auxiliary `mulleder19` head itself peaks at r <= 0.025,
so there is essentially no signal to transfer.

Two distinct failure modes were observed and are now understood:

- **learnable embeddings** -> memorization (train r 0.86 / val r 0.02). A held-out gene's
  embedding never received a gradient, so generalization is impossible by construction. This is
  documented independently in `conf/cgt_embed_005.yaml`.
- **ESM2 content features, learnable table off** -> **mean collapse**: val pearson EXACTLY
  0.00000 with loss 0.970. That is what MSE pays for on a z-scored target.

### A claim I made and then DISPROVED -- do not re-propagate it

I hypothesized the scalar head attenuated the perturbation ~6,607x by averaging all gene tokens.
**Measured on a real batch, it is ~2.5x, not 6,607x** -- the equivariant perturbation operator
cross-attends, so it has already spread the perturbation before pooling. Run:

```bash
cd /home/michaelvolk/Documents/projects/torchcell.worktrees/plan/cgt-metabolism
PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
  torchcell/models/cell_graph_transformer_metabolism.py
```

```
h_CLS (reference cell)            0.000e+00   <- across-batch std
mean over ALL 6607 gene tokens    1.309e-02
mean over PERTURBED genes only    3.217e-02
```

**What DOES survive:** `h_CLS` carries EXACTLY zero per-sample variance -- it is the encoding of
the unperturbed reference graph, broadcast across the batch -- while occupying a full
`hidden_dim` of the head's input width. `perturbed_gene_pool` is a modest, justified improvement,
NOT the cure for the nulls.

### VERIFY THIS FIRST in a fresh session

The point of the next session is to check the model is what we intended:

1. Run the `main()` above. Confirm `torchcell package:` and `this model file:` share the WORKTREE
   root. If they don't, you are running main's library against the branch's file.
2. `pytest tests/torchcell/models/test_cell_graph_transformer_metabolism.py` (expect 7 passed).
3. Confirm the three heads exist with the right widths and are all SUPERVISED (per-head val
   losses finite and non-zero in every condition) -- a silently-unsupervised head reads as a
   clean null.
4. Check job **1333** (submitted 2026-07-26 13:58, GPU 0) in `results/pigment_transfer_runs.json`.
   **CORRECTION (verified from the job banner): it runs `A1` ONLY at 60 epochs, not all four
   conditions** -- so it produces NO Delta. It is the betaxanthin-alone diagnostic. Verified items
   1-3 all pass: package + model file both resolve to the worktree root, 7/7 tests, and the three
   heads (3,137 / 3,137 / 3,731 params) are genuinely supervised -- `val_loss_last` is ~1.10 in the
   one-head arms vs ~2.47 in the two-head arms, so `mulleder19` carries real gradient.
5. **`metric_taken_at: "peak"` is a biased estimator and every Delta is read through it.**
   `BestMetricTracker` (`train_cgt_multitask.py:1367`) keeps the running MAX over epochs. Measured
   on the non-learning trajectories: val-pearson has mean ~0 and sd 0.027-0.071 across epochs, and
   the running max over only ~20 epochs is **+0.083 / +0.106 / +0.092** (jobs 1331 / 1332 / 1333).
   The 12-run baseline's "peaks" (0.064-0.128) sit entirely inside that band, so
   `Delta = peak(joint) - peak(alone)` differences two maxima-of-noise: per-arm bias ~2.5x the
   claimed effect (+0.042), and differencing predicts a scatter (~0.05) matching the observed SD
   (0.064). The tracker is NOT wrong in general -- `:1349` justifies it for the 019 decoder sweep
   where runs genuinely peak then MSE-collapse. It only becomes a noise-maximizer on a null.
   **Fix: report the primary metric at the epoch selected by best val LOSS; keep peak as a labelled
   diagnostic.**

### Traps that already cost jobs

- `sbatch --export=VAR="a b"` -- the shell strips quotes and the tail becomes a separate sbatch
  arg. Killed job 1327.
- `run_pigment_conditions.py` reads **`PIGMENT_OVERRIDES` as a COMMA-separated env var** and does
  NOT read `sys.argv`. Job 1329 ran clean and silently ignored its overrides; the only tell was a
  byte-identical `total_param_count`.
- The driver **resumes from `results/pigment_transfer_runs_partial.json`** and skips completed
  (seed, condition) pairs. Job 1330 did no training at all and re-emitted old numbers. Archive the
  checkpoint before a config change.
- Debugging a worktree file imports the PRIMARY checkout's `torchcell` (see
  [[worktree-build-needs-pythonpath]]). Fixed in this model file via a `__main__` bootstrap.

### Open decisions / next steps

- **The real blocker: `mulleder19` must learn something** before the transfer contrast can decide
  anything. Right now there is no signal to transfer.
- Mean collapse is what point-MSE rewards -> try the distributional heads from the decoder note.
- **beta-carotene: filter/condition on `flag_petite`** (already emitted by `ozaydin2013.py`) --
  low colour is partly mitochondrial dysfunction, not carotenoid flux.
- Delta prepared, NOT submitted. Account is **`bbtp-delta-gpu`**, bare env python, no
  `conda activate`. Two prerequisites in the slurm header (push/checkout branch there; make the
  driver honour `PIGMENT_BASE_CONFIG`).
- Media port still blocked on two user decisions: fixed `0.165` vs scaled 5%, and whether SM maps
  to `setup_ynb_media`. See [[plan.cgt-metabolism.2026.07.25]].
- Merzbacher 2025 is the betaxanthin baseline: 3-class, **69.8% vs a 67.2% majority rate**, on 811
  metabolic genes, split unreleased.
