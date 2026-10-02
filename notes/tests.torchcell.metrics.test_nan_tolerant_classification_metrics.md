---
id: d4d7f19kypulll4ml441e2s
title: Test_nan_tolerant_classification_metrics
desc: ''
updated: 1790916519396
created: 1790916519396
---

## 2026.10.01 - Exact values and findings for the NaN-tolerant classification metrics

`tests/torchcell/metrics/test_nan_tolerant_classification_metrics.py` pins `torchcell/metrics/nan_tolerant_classification_metrics.py` (0 percent to 97 percent line coverage in Phase 19 of [[test-campaign.2026.09.25]]). Callers: `fit_int_hetero_gnn_pool_binary_classification` (task "binary" when bins == 2, else "multiclass", never a `num_classes`; `(B, bins)` logits, 1-D float class-index targets with NaN) and `fit_int_cell_gin_diffpool_dense_binary` (adds AUROC).

Contract read from `_prepare_inputs`: only the target is masked (a 2-D target drops a row when any column is NaN); a (B, 2) target is one-hot and column 1 is the class; a wider target is returned as 2-D int64; predictions must be 2-D (`argmax(dim=1)`, no threshold), and 1-D probabilities raise `IndexError`. A NaN logit row is kept and argmax assigns it class 0.

Binary fixture: targets 1 1 1 0 0 0 1 0 nan nan against predicted 1 1 0 1 0 0 1 1 1 0, giving TP 3, FP 2, FN 1, TN 2 on eight valid rows: accuracy 0.625, precision 0.6, recall 0.75, F1 2/3, each equal to the torchmetrics binary metric on the masked rows, and unchanged when split across two updates. AUROC fixture: seven valid scores, 10 of 12 positive-negative pairs ordered, 0.8333333, equal to sklearn `roc_auc_score`; two NaN-target rows with extreme scores are dropped.

Findings (each test docstring starts "Finding:"):

- `num_classes` cannot be passed (lines 117 to 128, 234 to 243, 287 to 297): it is read from `kwargs` after `Metric.__init__` already rejected it (`ValueError: Unexpected keyword arguments: num_classes`), so the multiclass path always counts 2 classes. On a 3-class fixture class 2 is ignored: F1 0.8333 (torchmetrics macro 0.7778), precision 1.0 (0.8333), recall 0.75 (0.8333). The trainer's bins > 2 metrics are affected.
- Macro averages include an absent class as 0 (lines 151 to 157): two rows of class 0 give 0.5 where torchmetrics gives 1.0.
- Binary F1, precision and recall return NaN when `tp.sum() == 0` over both classes (lines 148, 263, 316), so an all-wrong batch is NaN where torchmetrics returns 0.0.
- AUROC ties (lines 202 to 215): the trapezoid runs over individually sorted samples, so two rows with identical scores give 1.0 or 0.0 by row order; sklearn gives 0.5.

Left uncovered: `_track_device`'s device move (line 27, needs a second device) and the three `if num_classes is None` raises (lines 124, 241, 294), unreachable because `num_classes` can never be supplied.

## 2026.10.02 - Audit 1 corrections, new pins, and reach

- AUROC ties: the 1.0 / 0.0 outcome relies on the order `torch.argsort(descending=True)` gives tied scores, which is not a stable sort and carries no contract for ties; it is deterministic on CPU for these sizes. Added the four-row case: logits [0, 1], [0, 1], [0, 2], [0, -1], targets [1, 0, 1, 0] give 1.0 (sklearn 0.875) and [0, 1, 0, 1] give 0.0 (sklearn 0.125).
- `test_base_forces_ddp_kwargs_and_outputs_float32_cpu` now also asserts the value 0.6.
- Finding (caller boundary): the binning transform writes an all-NaN row for an unmeasured label (`regression_to_classification.py` lines 260 to 262 one-hot, 241 to 242 soft), and the categorical/soft path of `fit_int_hetero_gnn_pool_binary_classification` (lines 291 to 293) takes `argmax(dim=1)`, which returns 0 for an all-NaN row. The metric receives class 0 and the NaN mask never fires: values [0.1, nan, 0.9] scored against predictions [0, 1, 1] give accuracy 2/3 over 3 rows instead of 1.0 over 2.
- The ordinal path (trainer lines 281 to 289) writes NaN back, and the mask fires: targets [1, nan, 2] count 2 rows.
- Finding: AUROC accumulates `cumsum` of the raw `.long()` target values, so an ordinal target 2 counts as two positives (lines 203 to 209); scores [3, 0, -1] with targets [1, 0, 2] give 1/3, sklearn on binarized targets 0.5. NaN targets are masked.

Reach (from audit 1, verified here): five trainers import the metric modules (`fit_int_cell_diffpool_dense_regression`, `fit_int_cell_sagpool_regression`, `fit_int_hetero_gnn_pool_binary_classification`, `fit_int_cell_gin_diffpool_dense_binary`, `fit_int_gat_diffpool_inception_regression`). Classification metrics are used by `fit_int_hetero_gnn_pool_binary_classification` (Accuracy, F1, Precision, Recall; multiclass whenever `num_bins` is 32 in `hetero_gnn_pool.yaml` and its sweep configs) and `fit_int_cell_gin_diffpool_dense_binary` (Accuracy, F1, AUROC). `fit_int_gat_diffpool_inception_regression` cannot be imported at all (ImportError on its Pearson/Spearman import, see [[tests.torchcell.metrics.test_nan_tolerant_metrics]]). Every affected logged metric belongs to experiment 003 runs from 2024-11 to 2025-01; none is in the manuscript or notes-tex.
