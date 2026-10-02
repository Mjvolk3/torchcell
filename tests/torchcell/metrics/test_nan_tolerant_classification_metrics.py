# tests/torchcell/metrics/test_nan_tolerant_classification_metrics.py
# [[tests.torchcell.metrics.test_nan_tolerant_classification_metrics]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/metrics/test_nan_tolerant_classification_metrics.py
"""Exact values for the NaN-tolerant classification metrics.

Callers: ``fit_int_hetero_gnn_pool_binary_classification`` builds Accuracy / F1 /
Precision / Recall with ``task = "binary" if bins == 2 else "multiclass"`` (never a
``num_classes``) and feeds ``(B, bins)`` logits with 1-D float class-index targets that are
NaN for unmeasured rows; ``fit_int_cell_gin_diffpool_dense_binary`` adds AUROC on the same
``(B, 2)`` logits.

NaN contract, read from ``_prepare_inputs``: ONLY the target is masked (a 2-D target drops
a row when any column is NaN); a NaN prediction is kept. Predictions must be 2-D logits or
probabilities (``argmax(dim=1)``); there is no threshold and no 1-D probability path.

Binary fixture (ten rows, logits [0, 1] predict class 1, [1, 0] predict class 0)::

    target     1 1 1 0 0 0 1 0 nan nan
    predicted  1 1 0 1 0 0 1 1  1   0

The two NaN-target rows are dropped. Positive class counts on the eight valid rows:
TP 3 (rows 0, 1, 6), FN 1 (row 2), FP 2 (rows 3, 7), TN 2 (rows 4, 5). Accuracy
(3 + 2) / 8 = 0.625, precision 3 / 5 = 0.6, recall 3 / 4 = 0.75, F1 2 * 0.6 * 0.75 / 1.35
= 2/3.

Multiclass fixture (four rows, three classes)::

    target     0 1 2 1
    predicted  0 1 2 2

Accuracy 3/4. Per class (tp, fp, fn): class 0 (1, 0, 0), class 1 (1, 0, 1), class 2
(1, 1, 0). The full three-class macro values are precision (1 + 1 + 0.5) / 3 = 0.8333,
recall (1 + 0.5 + 1) / 3 = 0.8333, F1 (1 + 2/3 + 2/3) / 3 = 0.7778 (torchmetrics agrees).

AUROC fixture: positive-class score sigmoid(z) from logits [0, z]. Positives z = {2, 0.5,
0, 3}, negatives z = {-1, 1.5, -0.5}; ordered pairs with positive > negative: 3 + 2 + 2 + 3
= 10 of 12, AUROC 10/12 = 0.8333333 (sklearn ``roc_auc_score`` agrees).
"""

import math
import re

import pytest
import torch
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from torchcell.metrics.nan_tolerant_classification_metrics import (
    NaNTolerantAccuracy,
    NaNTolerantAUROC,
    NaNTolerantF1Score,
    NaNTolerantPrecision,
    NaNTolerantRecall,
)
from torchcell.transforms.regression_to_classification import EqualWidthStrategy

NAN = float("nan")
COMPUTE_BEFORE_UPDATE = "ignore:The ``compute`` method of metric"


def _logits_for(classes: list[int], num_classes: int = 2) -> torch.Tensor:
    """One-hot logits whose argmax is ``classes``."""
    return torch.nn.functional.one_hot(torch.tensor(classes), num_classes).float()


BIN_TARGET = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, NAN, NAN])
BIN_PRED = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
BIN_LOGITS = _logits_for(BIN_PRED)
VALID_PRED = torch.tensor(BIN_PRED[:8])
VALID_TARGET = torch.tensor([1, 1, 1, 0, 0, 0, 1, 0])

MC_LOGITS = _logits_for([0, 1, 2, 2], num_classes=3) * 3.0
MC_TARGET = torch.tensor([0.0, 1.0, 2.0, 1.0])


# ------------------------------------------------------------------ _prepare_inputs


def test_prepare_inputs_masks_nan_targets_only_and_casts_to_long() -> None:
    """1-D target [1, nan, 0] with preds [[nan, 0], [0, 1], [2, 3]]: row 1 (NaN target)
    is dropped, row 0 (NaN prediction) is KEPT; targets come back as int64 [1, 0].
    """
    m = NaNTolerantAccuracy()
    preds = torch.tensor([[NAN, 0.0], [0.0, 1.0], [2.0, 3.0]])
    vp, vt = m._prepare_inputs(preds, torch.tensor([1.0, NAN, 0.0]))
    torch.testing.assert_close(vp, preds[[0, 2]], equal_nan=True, rtol=0, atol=0)
    assert vt.tolist() == [1, 0] and vt.dtype == torch.int64


def test_prepare_inputs_one_hot_target_uses_second_column() -> None:
    """A (B, 2) target is read as one-hot: the class is column 1. Rows [0, 1], [nan, 0],
    [1, 0]: row 1 has a NaN in a column and is dropped; classes [1, 0].
    """
    m = NaNTolerantAccuracy()
    vp, vt = m._prepare_inputs(
        torch.arange(6.0).view(3, 2), torch.tensor([[0.0, 1.0], [NAN, 0.0], [1.0, 0.0]])
    )
    assert vp.tolist() == [[0.0, 1.0], [4.0, 5.0]]
    assert vt.tolist() == [1, 0]


def test_prepare_inputs_wider_target_is_returned_as_2d_long() -> None:
    """A (B, 3) target is not converted to indices: the row with a NaN is dropped and the
    remaining row is returned as the 2-D int64 [[0, 1, 0]] (no multi-label support).
    """
    m = NaNTolerantAccuracy()
    vp, vt = m._prepare_inputs(
        torch.zeros(2, 3), torch.tensor([[0.0, 1.0, 0.0], [1.0, NAN, 0.0]])
    )
    assert vt.tolist() == [[0, 1, 0]] and vt.dtype == torch.int64
    assert vp.tolist() == [[0.0, 0.0, 0.0]]


def test_prepare_inputs_empty_and_all_nan_targets_return_empty_float() -> None:
    """Zero-element or all-NaN targets return two empty float32 tensors."""
    m = NaNTolerantAccuracy()
    for target in (torch.zeros(0), torch.tensor([NAN, NAN])):
        vp, vt = m._prepare_inputs(torch.zeros(target.numel(), 2), target)
        assert vp.shape == torch.Size([0]) and vt.shape == torch.Size([0])
        assert vp.dtype == torch.float32 and vt.dtype == torch.float32


# ------------------------------------------------------------------------ binary


def test_binary_counts_match_hand_confusion_matrix_and_torchmetrics() -> None:
    """Binary fixture (TP 3, FP 2, FN 1, TN 2 on eight valid rows): accuracy 0.625,
    precision 0.6, recall 0.75, F1 2/3, each equal to the torchmetrics binary metric on the
    hand-masked rows. If the NaN rows were kept, precision would be 3 / 6.
    """
    acc, f1 = NaNTolerantAccuracy(), NaNTolerantF1Score()
    prec, rec = NaNTolerantPrecision(), NaNTolerantRecall()
    for metric in (acc, f1, prec, rec):
        metric.update(BIN_LOGITS, BIN_TARGET)
    assert acc.compute().item() == 0.625
    assert acc.correct.item() == 5 and acc.total.item() == 8
    assert f1.tp.tolist() == [2.0, 3.0]
    assert f1.fp.tolist() == [1.0, 2.0]
    assert f1.fn.tolist() == [2.0, 1.0]
    assert prec.compute().item() == pytest.approx(0.6, abs=1e-7)
    assert rec.compute().item() == pytest.approx(0.75, abs=1e-7)
    assert f1.compute().item() == pytest.approx(2 / 3, abs=1e-7)
    for ours, theirs in (
        (acc, BinaryAccuracy()),
        (prec, BinaryPrecision()),
        (rec, BinaryRecall()),
        (f1, BinaryF1Score()),
    ):
        assert ours.compute().item() == pytest.approx(
            theirs(VALID_PRED, VALID_TARGET).item(), abs=1e-7
        )


def test_binary_accumulation_over_batches_equals_single_pass() -> None:
    """Splitting the fixture into rows 0-4 and 5-9 (the second carrying both NaN rows)
    gives the same counts and values as one update: F1 2/3, accuracy 0.625.
    """
    acc, f1 = NaNTolerantAccuracy(), NaNTolerantF1Score()
    for lo, hi in ((0, 5), (5, 10)):
        acc.update(BIN_LOGITS[lo:hi], BIN_TARGET[lo:hi])
        f1.update(BIN_LOGITS[lo:hi], BIN_TARGET[lo:hi])
    assert acc.total.item() == 8 and acc.compute().item() == 0.625
    assert f1.compute().item() == pytest.approx(2 / 3, abs=1e-7)


def test_nan_prediction_is_not_masked_and_counts_as_class_zero() -> None:
    """A NaN logit row survives masking; softmax makes it [nan, nan] and argmax returns
    index 0, so with target 1 it is a wrong prediction. preds [[nan, 0], [0, 1]],
    target [1, 1]: correct 1 of total 2 -> 0.5 (1.0 if NaN predictions were masked).
    """
    acc = NaNTolerantAccuracy()
    acc.update(torch.tensor([[NAN, 0.0], [0.0, 1.0]]), torch.tensor([1.0, 1.0]))
    assert acc.correct.item() == 1 and acc.total.item() == 2
    assert acc.compute().item() == 0.5


def test_one_dimensional_probabilities_raise_index_error() -> None:
    """There is no threshold path: 1-D probabilities hit ``argmax(dim=1)`` and raise."""
    acc = NaNTolerantAccuracy()
    with pytest.raises(
        IndexError,
        match=re.escape(
            "Dimension out of range (expected to be in range of [-1, 0], but got 1)"
        ),
    ):
        acc.update(torch.tensor([0.9, 0.1]), torch.tensor([1.0, 0.0]))


def test_binary_all_predictions_wrong_returns_nan_not_zero_finding() -> None:
    """Finding: compute returns NaN whenever ``tp.sum() == 0`` (summed over BOTH classes),
    so a batch where every prediction is wrong gives NaN for F1, precision and recall,
    while torchmetrics ``BinaryF1Score`` gives 0.0 for the same data. preds [[1, 0],
    [0, 1]] (classes 0, 1), targets [1, 0]: TP 0, FP 1, FN 1, TN 0
    (nan_tolerant_classification_metrics.py:148, :263, :316).
    Pinned until compute returns 0 when there are samples but no true positives.
    """
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([1.0, 0.0])
    for cls in (NaNTolerantF1Score, NaNTolerantPrecision, NaNTolerantRecall):
        m = cls()
        m.update(logits, target)
        assert math.isnan(m.compute().item())
    assert BinaryF1Score()(torch.tensor([0, 1]), torch.tensor([1, 0])).item() == 0.0


def test_binary_precision_zero_when_no_positive_predicted() -> None:
    """All predictions class 0 with targets [0, 1]: tp = [1, 0], so compute proceeds and
    precision[1] = 0 / (0 + 0 + 1e-10) = 0.0; recall[1] = 0 / 1 = 0.0; F1 0.0.
    """
    logits = _logits_for([0, 0])
    target = torch.tensor([0.0, 1.0])
    for cls in (NaNTolerantF1Score, NaNTolerantPrecision, NaNTolerantRecall):
        m = cls()
        m.update(logits, target)
        assert m.compute().item() == 0.0


@pytest.mark.filterwarnings(COMPUTE_BEFORE_UPDATE)
def test_empty_compute_and_reset_return_nan() -> None:
    """No valid row: accuracy, F1, precision, recall and AUROC compute to NaN (shape ()).
    After real updates, ``reset`` zeroes tp/fp/fn and correct/total.
    """
    for cls in (
        NaNTolerantAccuracy,
        NaNTolerantF1Score,
        NaNTolerantPrecision,
        NaNTolerantRecall,
        NaNTolerantAUROC,
    ):
        m = cls()
        m.update(torch.zeros(2, 2), torch.tensor([NAN, NAN]))
        out = m.compute()
        assert out.shape == torch.Size([]) and math.isnan(out.item())
    acc, f1 = NaNTolerantAccuracy(), NaNTolerantF1Score()
    acc.update(BIN_LOGITS, BIN_TARGET)
    f1.update(BIN_LOGITS, BIN_TARGET)
    acc.reset()
    f1.reset()
    assert acc.correct.item() == 0 and acc.total.item() == 0
    assert f1.tp.tolist() == [0.0, 0.0] and f1.fn.tolist() == [0.0, 0.0]
    assert math.isnan(acc.compute().item()) and math.isnan(f1.compute().item())


def test_base_forces_ddp_kwargs_and_outputs_float32_cpu() -> None:
    """Caller DDP kwargs are overwritten (compute_on_cpu False, sync_on_compute False,
    dist_sync_on_step True); the binary-fixture precision is 3 / 5 = 0.6, float32 on CPU.
    """
    m = NaNTolerantPrecision(
        compute_on_cpu=True, sync_on_compute=True, dist_sync_on_step=False
    )
    assert (m.compute_on_cpu, m.sync_on_compute, m.dist_sync_on_step) == (
        False,
        False,
        True,
    )
    m.update(BIN_LOGITS, BIN_TARGET)
    out = m.compute()
    assert out.dtype == torch.float32 and out.device == torch.device("cpu")
    assert out.item() == pytest.approx(0.6, abs=1e-7)


# -------------------------------------------------------------------- multiclass


def test_multiclass_accuracy_three_classes() -> None:
    """Multiclass fixture: predicted [0, 1, 2, 2] vs target [0, 1, 2, 1] -> 3/4."""
    acc = NaNTolerantAccuracy(task="multiclass")
    acc.update(MC_LOGITS, MC_TARGET)
    assert acc.compute().item() == 0.75


def test_multiclass_num_classes_cannot_be_passed_finding() -> None:
    """Finding: ``num_classes`` is read from ``kwargs`` only AFTER ``kwargs`` were handed
    to ``Metric.__init__``, which rejects unknown keys, so it can never be passed and the
    multiclass path always counts exactly 2 classes
    (nan_tolerant_classification_metrics.py:117-128, :234-243, :287-297). The trainer's
    ``bins > 2`` multiclass metrics are therefore 2-class.
    Pinned until num_classes is a named constructor argument.
    """
    for cls in (NaNTolerantF1Score, NaNTolerantPrecision, NaNTolerantRecall):
        with pytest.raises(
            ValueError, match=re.escape("Unexpected keyword arguments: `num_classes`")
        ):
            cls(task="multiclass", num_classes=3)
        assert cls(task="multiclass").tp.tolist() == [0.0, 0.0]


def test_multiclass_macro_drops_classes_beyond_two_finding() -> None:
    """Finding (consequence of the fixed 2-class state): on the three-class fixture class
    2 is never counted. F1 = mean(1, 2/3) = 0.8333 (torchmetrics macro 0.7778), precision
    mean(1, 1) = 1.0 (torchmetrics 0.8333), recall mean(1, 0.5) = 0.75 (torchmetrics
    0.8333). Pinned until num_classes is honored.
    """
    target_long = MC_TARGET.long()
    cases = (
        (NaNTolerantF1Score, 5 / 6, MulticlassF1Score(3, average="macro"), 7 / 9),
        (NaNTolerantPrecision, 1.0, MulticlassPrecision(3, average="macro"), 5 / 6),
        (NaNTolerantRecall, 0.75, MulticlassRecall(3, average="macro"), 5 / 6),
    )
    for cls, ours_expected, theirs, theirs_expected in cases:
        m = cls(task="multiclass")
        m.update(MC_LOGITS, MC_TARGET)
        assert m.compute().item() == pytest.approx(ours_expected, abs=1e-6)
        assert theirs(MC_LOGITS, target_long).item() == pytest.approx(
            theirs_expected, abs=1e-6
        )


def test_multiclass_macro_counts_absent_class_as_zero_finding() -> None:
    """Finding: a class with no support and no predictions enters the macro mean as 0,
    where torchmetrics excludes it. Two rows, both predicted and labeled class 0:
    tp = [2, 0], fp = fn = [0, 0]; F1 / precision / recall = mean(1, 0) = 0.5 versus
    torchmetrics ``MulticlassF1Score(2, average="macro")`` 1.0
    (nan_tolerant_classification_metrics.py:151-157). Pinned until absent classes are
    excluded from the macro average.
    """
    logits = _logits_for([0, 0])
    target = torch.tensor([0.0, 0.0])
    for cls in (NaNTolerantF1Score, NaNTolerantPrecision, NaNTolerantRecall):
        m = cls(task="multiclass")
        m.update(logits, target)
        assert m.compute().item() == pytest.approx(0.5, abs=1e-9)
    assert MulticlassF1Score(2, average="macro")(logits, target.long()).item() == 1.0


# ------------------------------------------------------------------------- AUROC


def _auroc_logits(z: list[float]) -> torch.Tensor:
    """Logits [0, z]: softmax column 1 is sigmoid(z)."""
    zt = torch.tensor(z)
    return torch.stack([torch.zeros_like(zt), zt], dim=1)


AUC_Z = [2.0, -1.0, 0.5, 1.5, -0.5, 0.0, 3.0, 9.0, -9.0]
AUC_T = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, NAN, NAN])


def test_auroc_matches_sklearn_across_batches_with_nan_targets() -> None:
    """AUROC fixture split into rows 0-3 and 4-8 (NaN targets in the second batch, whose
    extreme scores +-9 would change the value if kept): 10/12 = 0.8333333, equal to
    sklearn ``roc_auc_score`` on the seven valid rows. Buffers hold 4 and 3 scores.
    """
    m = NaNTolerantAUROC()
    logits = _auroc_logits(AUC_Z)
    m.update(logits[:4], AUC_T[:4])
    m.update(logits[4:], AUC_T[4:])
    probs = torch.sigmoid(torch.tensor(AUC_Z[:7])).numpy()
    expected = roc_auc_score(AUC_T[:7].numpy(), probs)
    assert expected == pytest.approx(10 / 12, abs=1e-12)
    assert m.compute().item() == pytest.approx(expected, abs=1e-6)
    assert [p.numel() for p in m.preds] == [4, 3]


def test_auroc_tied_scores_depend_on_row_order_finding() -> None:
    """Finding: the trapezoid runs over individually sorted samples, not over distinct
    thresholds, so tied scores of opposite class are credited by their sort order
    (nan_tolerant_classification_metrics.py:202-215). Two rows with identical logits
    [0, 0]: targets [1, 0] -> 1.0, targets [0, 1] -> 0.0; sklearn gives 0.5 for both.
    Four rows with logits [0, 1], [0, 1], [0, 2], [0, -1] (rows 0 and 1 tied): targets
    [1, 0, 1, 0] -> 1.0 (sklearn 0.875, the tie counted 1/2), targets [0, 1, 0, 1] -> 0.0
    (sklearn 0.125).

    The exact 1.0 / 0.0 values rely on the order ``torch.argsort(descending=True)`` gives
    tied scores. That sort is not stable and carries no ordering contract for ties; it is
    deterministic on CPU for these sizes, which is all this test observes. A change of
    sort implementation could move the values without any change in this module.
    Pinned until ties are grouped (or torchmetrics' binary_auroc is used).
    """
    logits = torch.zeros(2, 2)
    first, second = NaNTolerantAUROC(), NaNTolerantAUROC()
    first.update(logits, torch.tensor([1.0, 0.0]))
    second.update(logits, torch.tensor([0.0, 1.0]))
    assert first.compute().item() == 1.0
    assert second.compute().item() == 0.0
    assert roc_auc_score([1, 0], [0.5, 0.5]) == 0.5
    four = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0], [0.0, -1.0]])
    scores = torch.softmax(four, dim=-1)[:, 1].numpy()
    for target, ours_expected, sklearn_expected in (
        ([1.0, 0.0, 1.0, 0.0], 1.0, 0.875),
        ([0.0, 1.0, 0.0, 1.0], 0.0, 0.125),
    ):
        m = NaNTolerantAUROC()
        m.update(four, torch.tensor(target))
        assert m.compute().item() == ours_expected
        assert roc_auc_score(target, scores) == sklearn_expected


def test_auroc_single_class_returns_nan_and_rejects_multiclass() -> None:
    """Only positives -> NaN (AUROC undefined); ``task='multiclass'`` raises."""
    m = NaNTolerantAUROC()
    m.update(_auroc_logits([1.0, -1.0]), torch.tensor([1.0, 1.0]))
    assert math.isnan(m.compute().item())
    with pytest.raises(
        ValueError,
        match=re.escape("AUROC currently only supports binary classification"),
    ):
        NaNTolerantAUROC(task="multiclass")  # type: ignore[arg-type, unused-ignore]


@pytest.mark.filterwarnings(COMPUTE_BEFORE_UPDATE)
def test_auroc_reset_empties_buffers() -> None:
    """``reset`` empties both buffers; compute is NaN until the next update."""
    m = NaNTolerantAUROC()
    m.update(_auroc_logits(AUC_Z), AUC_T)
    assert m.compute().item() == pytest.approx(10 / 12, abs=1e-6)
    m.reset()
    assert m.preds == [] and m.targets == []
    assert math.isnan(m.compute().item())


# ------------------------------------------------- targets as the trainers build them


def test_categorical_target_from_argmax_of_nan_row_counts_as_class_zero_finding() -> (
    None
):
    """Finding (caller boundary): the binning transform writes an all-NaN row for an
    unmeasured label (regression_to_classification.py:260-262, one-hot; :241-242, soft),
    and the categorical/soft path of ``fit_int_hetero_gnn_pool_binary_classification``
    (lines 291-293) turns it into a target with ``argmax(dim=1)``, which returns 0 for an
    all-NaN row. The metric receives an int64 class 0, so ``_prepare_inputs``' NaN mask
    never fires and the unmeasured row is scored.

    values [0.1, nan, 0.9], edges [0, 0.5, 1] -> one-hot [[1, 0], [nan, nan], [0, 1]] ->
    targets [0, 0, 1]. Predicted classes [0, 1, 1]: accuracy 2 / 3 over 3 rows; with the
    unmeasured row masked it would be 2 / 2 = 1.0.
    Pinned until the trainer restores NaN for all-NaN rows before calling the metrics
    (as its ordinal path does).
    """
    values = torch.tensor([0.1, NAN, 0.9])
    onehot = EqualWidthStrategy().compute_onehot_labels(
        values, torch.tensor([0.0, 0.5, 1.0])
    )
    assert onehot[0].tolist() == [1.0, 0.0] and onehot[2].tolist() == [0.0, 1.0]
    assert torch.isnan(onehot[1]).all().item()
    targets = torch.argmax(onehot, dim=1)
    assert targets.tolist() == [0, 0, 1]
    acc = NaNTolerantAccuracy()
    acc.update(_logits_for([0, 1, 1]), targets)
    assert acc.total.item() == 3
    assert acc.compute().item() == pytest.approx(2 / 3, abs=1e-7)


def test_ordinal_float_targets_with_nan_are_masked() -> None:
    """The ordinal path (trainer lines 281-289) counts ``fitness > 0.5`` per row as a
    float class index and writes NaN back for rows with any NaN, so the mask does fire.
    Ordinal rows [[1, 0], [nan, nan], [1, 1]] -> targets [1, nan, 2]. Three-class logits
    predicting [1, 0, 2]: multiclass accuracy correct 2 of total 2 = 1.0.
    """
    fitness = torch.tensor([[1.0, 0.0], [NAN, NAN], [1.0, 1.0]])
    targets = torch.sum(fitness > 0.5, dim=1).float()
    targets[torch.isnan(fitness).any(dim=1)] = NAN
    assert targets[0].item() == 1.0 and targets[2].item() == 2.0
    assert math.isnan(targets[1].item())
    acc = NaNTolerantAccuracy(task="multiclass")
    acc.update(_logits_for([1, 0, 2], num_classes=3), targets)
    assert acc.correct.item() == 2 and acc.total.item() == 2
    assert acc.compute().item() == 1.0


def test_auroc_ordinal_targets_weight_higher_classes_finding() -> None:
    """Finding: AUROC casts targets with ``.long()`` and accumulates ``cumsum`` of the
    raw target values, so an ordinal target 2 counts as two positives
    (nan_tolerant_classification_metrics.py:203-209), while ``.bool()`` (line 198) treats
    it as one positive class. NaN targets are masked. Scores z = [3, 0, -1, 5], targets
    [1, 0, 2, nan]: row 3 dropped; descending order targets 1, 0, 2 -> tps [1, 1, 3],
    fps [0, 1, 1] -> tpr [1/3, 1/3, 1], fpr [0, 1, 1] -> area (1/3 + 1/3) / 2 * 1 = 1/3.
    sklearn on the binarized targets (t > 0) [1, 0, 1] gives 0.5.
    Pinned until AUROC rejects or binarizes non-binary targets.
    """
    m = NaNTolerantAUROC()
    m.update(
        torch.tensor([[0.0, 3.0], [0.0, 0.0], [0.0, -1.0], [0.0, 5.0]]),
        torch.tensor([1.0, 0.0, 2.0, NAN]),
    )
    assert m.targets[0].tolist() == [1, 0, 2]
    assert m.compute().item() == pytest.approx(1 / 3, abs=1e-7)
    assert roc_auc_score([1, 0, 1], [3.0, 0.0, -1.0]) == 0.5
