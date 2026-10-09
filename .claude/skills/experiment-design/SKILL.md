---
name: experiment-design
description: Checklist for designing or reviewing a torchcell training campaign so every run can later be placed on a scaling curve (N, D, C), including FLOP accounting, the data unit, splits, seeds and the replicate floor. Use when writing a new experiment config or launcher, planning a round or sweep, reviewing a campaign's readout, or when the words scaling, compute, FLOPs, model size, data size, or "how much more data/model" come up.
---

# Experiment design: the scaling checklist

Every campaign is a point on three axes, parameters N, data D and compute C, whether or
not it was designed as a scaling study. The Genesis questions (does more data or a bigger
model pay, what does a budget buy) can only be answered later if the bookkeeping was
done at run time. This checklist is what to confirm before launching and what to read
back when reviewing. The reference is `notes-tex/modeling/scaling-laws/`.

## Before launch

1. **Compute is logged, measured, not computed.** The trainer's callback list includes
   `torchcell.trainers.compute_accounting.ComputeAccounting`. It writes
   `compute/params_total`, `compute/params_non_embedding`, `compute/flops_per_step`,
   `compute/flops_per_record`, `compute/flops_per_epoch`, `compute/flops_cumulative`,
   `compute/records_per_epoch`, `compute/records_seen`, `compute/gpu_hours_cumulative`.
   FLOPs come from `torch.utils.flop_counter.FlopCounterMode` around one real training
   step; 6ND is for token transformers and does not apply to a gene-token model with
   graph penalties. The count covers the matmul family only, so it is a lower bound that
   is comparable across sizes of one architecture.
2. **N is stated as non-embedding parameters**, with the gene table or composite
   preprocessor reported separately (`compute/params_embedding`). A size ladder scales
   width and depth together in a fixed ratio; heads must divide the width (9 heads here).
3. **D is stated in the unit the loss averages over**, and the unit is held fixed across
   every run of a sweep. On 025 the unit is the record; on the 030 per-entry path it is the
   entry row (4.86M for S3, not 1.12M). `ComputeAccounting(unit=...)` records the choice.
   Repeated epochs are a hyperparameter, not more D: a D sweep is a sweep over DISTINCT
   training records.
4. **Subsample by the unit that leaks.** Nested D subsets are drawn by gene or by query
   pair, never by record, against ONE fixed evaluation set. On the trigenic builds that is
   the query-pair-disjoint split (`subset_definitions_030_q.py`, `subset_s3q_fractions.py`).
5. **The evaluation set is fixed and its leak audit is written down**: which training
   records could carry a held-out unit, how many, and that they were removed
   (`subset_definitions_030_q_summary.json` is the model).
6. **Loss is logged, not only the downstream metric.** `val/point_loss` (normalized MSE)
   is what a power law is fit to; Pearson plateaus and is the readout, not the fit target.
7. **Seeds where they are cheap**: three at the two smallest sizes of any size ladder,
   so the bootstrap has a noise floor.
8. **Learning rate suits each size** (tuned per size, or at least checked at the
   smallest), and the epoch budget is sized from a measured learning curve, not guessed.
9. **The replicate floor E is computed before the sweep**, in the loss's units, from the
   re-measured records of the evaluation set (`triple_noise_ceiling_030.py` has the
   method). A fitted E below it is a wrong fit; one far above says the model family is
   the limit.
10. **The run is placed in a saved W&B view** by the campaign's labeler
    (`wandb-curate` skill) so the compute keys appear beside the curves.

## When reviewing

- Report every arm with its N (non-embedding), D (distinct records in the loss unit),
  epochs, FLOPs per epoch and cumulative, GPU hours, seeds, and the loss window beside
  the Pearson window. A result with no compute column cannot enter a scaling figure.
- Ask the three questions: is the loss still falling with D at the largest set (then a
  new screen beats a larger model); with N at the largest size (then capacity binds);
  and where does the fitted E sit against the replicate floor.
- Hold out the largest run when fitting; fit on the rest; say whether it predicted.

## Where the pieces live

- Callback: `torchcell/trainers/compute_accounting.py` (test
  `tests/torchcell/trainers/test_compute_accounting.py`).
- Forms, fitting code, synthetic exercise: `experiments/041-scaling-laws/scripts/scaling_law_forms.py`.
- Replicate floor on the trigenic score: `experiments/030-solid-growth-multi/scripts/triple_noise_ceiling_030.py`.
- Design for the trigenic build: `notes-tex/modeling/scaling-laws/sections/5-torchcell.tex`.
