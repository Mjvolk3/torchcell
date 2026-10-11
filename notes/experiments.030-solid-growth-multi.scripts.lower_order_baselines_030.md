---
id: a596oanaehhv8zcr5zp2r8q
title: Lower_order_baselines_030
desc: ''
updated: 1791710712418
created: 1791710712418
---

## 2026.10.11 - Baselines that see the lower-order terms

**Why.** The joint model on the S3 pool trains on every single, every closure double and
every training triple, with a fitness head beside the interaction head. For a held-out
triple, every term of the published identity tau = f_ijk - f_ij f_k - eps_ik - eps_jk
except f_ijk is a training record, so the model knows the triple's null expectation and
only f_ijk is unknown. The gene-identity ladder of 010 and 025 (one-hot ridge over the
three genes, trained on triples alone) never sees those terms, so it cannot separate
"learned biology" from "learned the arithmetic the closure gives it". These rungs can.
None of them touches the held-out triple's own fitness or interaction, which is the
same information boundary the trainer has: splits are by record, and a 030 record holds
all of a genotype's entries.

**Rungs** (fit on the arm's training triples, scored on its pinned val and test):

| rung | prediction | parameters |
|---|---|---|
| N0 screen mean | training mean of tau in the triple's screen | 2 |
| N1 mult., query double | -(eps_ik + eps_jk), i.e. f_ijk = f_ij f_k | 0 |
| N2 mult., singles | f_i f_j f_k - f_ij f_k - eps_ik - eps_jk | 0 |
| N1s strain-matched | N1 with the control scores matched on the triple's own array strain | 0 |
| B1 additive gene ridge | the 010 rung, one-hot over the three genes | genes |
| L1 ridge, lower-order | ridge over 13 terms, 4 products, a screen indicator, missing indicators | ~35 |
| L2 boosted trees, lower-order | HistGradientBoosting over the same raw terms, missing handled natively | trees |
| L3 ridge, genes + lower-order | B1's one-hot beside L1's design, one alpha | genes + 35 |
| F1 ridge, lower-order to f_ijk | the fitness side: how much of f_ijk is arithmetic | ~35 |

Where an arithmetic term is missing the rung falls back to N0 for that row, so every
rung is scored on identical rows, and the covered fraction is reported beside it.

**Terms.** Resolved under the trainer's `LabelPolicy` (`kuzmin-first`, promoted to the
triple's own year) from `closure/entries.parquet` and `closure/triple_roles.parquet`:
singles by `select`, f_ij by `select_double` (the double-mutant query strain's own
fitness where the build carries it), the array-screen doubles without query-strain
entries, the digenic scores by `select`, and the strain-matched control terms exactly
as `closure_recompute_030.py` resolves them. The target tau and the screen of every
pinned val/test triple are asserted equal to `results/pinned_eval_table_030.parquet`.

**Arm Q masks held-out pairs.** `subset_definitions_030_q.py` removes every double whose
pair is a held-out query pair from the training pool, so on Q the baselines set the
terms of any held-out pair to missing (f_ij, the array double, eps_ij for the query
pair; the secondary-pair terms where a secondary pair is itself held out). A Q baseline
never sees a held-out triple's query double, as the model never did. Arm R masks
nothing: doubles never leave its training pool.

**Readouts.** Pooled Pearson and Spearman, within-screen Pearson (both series centered
by screen), per-screen Pearson, rmse, and a 2,000-resample paired bootstrap of test
Pearson gains between rungs. Predictions per triple are written so the transformer's
own val/test predictions can be paired against them later.

**Transformer rows.** `--wandb` reads the rank-0 runs tagged with each arm's config and
takes the mean validation interaction Pearson over the last 10 synced epochs and the
max over epochs. The server holds only part of the R curve (epochs 0 to 26 of
`xy3xnpau`, 100 to 163 of the hand-synced continuation, 164 of `u28pgzde`), so the row
records the synced-epoch count and the last epoch; it is a validation number of a model
with no test evaluation.

Launcher: [[experiments.030-solid-growth-multi.scripts.gh_lower_order_baselines_030]].
