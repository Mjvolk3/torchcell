# tests/torchcell/metrics/test_nan_tolerant_metrics.py
# [[tests.torchcell.metrics.test_nan_tolerant_metrics]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/metrics/test_nan_tolerant_metrics.py
"""Exact values for the NaN-tolerant regression and correlation metrics.

Callers (``fit_int_cell_diffpool_dense_regression``, ``fit_int_hetero_gnn_pool_binary_
classification``, ``fit_int_cell_sagpool_regression``) construct every metric with the
default ``num_outputs=1`` and feed 1-D ``(B,)`` predictions and targets, the targets
carrying NaN for unmeasured labels. Those single-output paths are pinned against
independent oracles (torchmetrics' non-NaN metric or scipy on the hand-masked subset).

NaN contract, read from the source: every regression metric here drops an element when
EITHER the prediction or the target is NaN (``~isnan(preds) & ~isnan(target)``).

Shared multi-output fixture (three rows, two columns)::

    P2 = [[1, 2], [3, nan], [5, 6]]
    T2 = [[0, 0], [nan, 0], [1, 1]]

Column 0 keeps rows 0 and 2 (row 1 target NaN); column 1 keeps rows 0 and 2 (row 1
prediction NaN). Per-column squared errors: col0 1 + 16 = 17, col1 4 + 25 = 29; absolute
errors: col0 1 + 4 = 5, col1 2 + 5 = 7. The honest per-column means are therefore
MSE [8.5, 14.5] and MAE [2.5, 3.5]. The code instead boolean-indexes the 2-D tensor (which
flattens it), sums all four valid elements (17 + 29 = 46, 5 + 7 = 12) and divides by the
ROW count 3, broadcasting one number to both columns: MSE [46/3, 46/3], MAE [4, 4].

Shared accumulation fixture (single output, four update calls)::

    batch 1: p = [1, 2, 3],   t = [1.5, nan, 2]  -> valid (1, 1.5), (3, 2)
    batch 2: p = [nan, 5],    t = [1, nan]       -> no valid element
    batch 3: p = [0, 2, 10],  t = [1, 0, 7]      -> all three valid
    batch 4: p = [6],         t = [4]            -> one valid row

Squared errors 0.25, 1, 1, 4, 9, 4 sum to 19.25 over 6 valid rows: MSE 19.25 / 6 =
3.2083333; RMSE sqrt(3.2083333) = 1.7911822. Absolute errors 0.5, 1, 1, 2, 3, 2 sum to 9.5:
MAE 9.5 / 6 = 1.5833333. The mean of per-batch MSEs would be (0.625 + 14/3 + 4) / 3 =
3.0972222, a different number, so the tests distinguish pooled accumulation from it.
"""

import importlib
import math
import re

import numpy as np
import pytest
import scipy.stats
import torch
from torchmetrics.metric import Metric
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
)

from torchcell.metrics.nan_tolerant_metrics import (
    NaNTolerantMAE,
    NaNTolerantMetricBase,
    NaNTolerantMSE,
    NaNTolerantPearsonCorrCoef,
    NaNTolerantR2Score,
    NaNTolerantRMSE,
    NaNTolerantSpearmanCorrCoef,
    _final_aggregation,
    _handle_nan_mask,
    _nan_tolerant_error_compute,
    _nan_tolerant_mae_update,
    _nan_tolerant_mse_update,
    _nan_tolerant_pearson_update,
)

NAN = float("nan")
P2 = torch.tensor([[1.0, 2.0], [3.0, NAN], [5.0, 6.0]])
T2 = torch.tensor([[0.0, 0.0], [NAN, 0.0], [1.0, 1.0]])

BATCHES: list[tuple[torch.Tensor, torch.Tensor]] = [
    (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, NAN, 2.0])),
    (torch.tensor([NAN, 5.0]), torch.tensor([1.0, NAN])),
    (torch.tensor([0.0, 2.0, 10.0]), torch.tensor([1.0, 0.0, 7.0])),
    (torch.tensor([6.0]), torch.tensor([4.0])),
]
VALID_P = torch.tensor([1.0, 3.0, 0.0, 2.0, 10.0, 6.0])
VALID_T = torch.tensor([1.5, 2.0, 1.0, 0.0, 7.0, 4.0])

COMPUTE_BEFORE_UPDATE = "ignore:The ``compute`` method of metric"
SPEARMAN_BUFFER_WARNING = "ignore:Metric `SpearmanCorrcoef` will save all targets"


# --------------------------------------------------------------------------- helpers


def test_handle_nan_mask_drops_elements_nan_in_either_tensor() -> None:
    """Mask is True only where both tensors are finite-or-inf (not NaN).

    preds [1, nan, 3, 4], target [5, 6, nan, 8]: index 1 has a NaN prediction, index 2 a
    NaN target, so mask = [T, F, F, T], valid preds [1, 4], valid targets [5, 8].
    """
    mask, vp, vt = _handle_nan_mask(
        torch.tensor([1.0, NAN, 3.0, 4.0]), torch.tensor([5.0, 6.0, NAN, 8.0])
    )
    assert mask.tolist() == [True, False, False, True]
    assert vp.tolist() == [1.0, 4.0]
    assert vt.tolist() == [5.0, 8.0]


def test_mse_and_mae_update_single_output_sum_and_valid_count() -> None:
    """Single output: sums over valid elements and counts them.

    Batch 1 of the accumulation fixture: squared errors 0.25 + 1 = 1.25, absolute errors
    0.5 + 1 = 1.5, two valid elements.
    """
    p, t = BATCHES[0]
    sse, n = _nan_tolerant_mse_update(p, t)
    sae, n2 = _nan_tolerant_mae_update(p, t)
    assert sse.item() == 1.25
    assert sae.item() == 1.5
    assert n.item() == 2 and n2.item() == 2
    assert n.dtype == torch.int64


def test_mse_and_mae_update_all_nan_returns_zero_vector_and_zero_count() -> None:
    """No valid element: the zero vector of length ``num_outputs`` and a zero count."""
    p, t = BATCHES[1]
    sse, n = _nan_tolerant_mse_update(p, t, num_outputs=3)
    sae, n2 = _nan_tolerant_mae_update(p, t, num_outputs=3)
    assert sse.tolist() == [0.0, 0.0, 0.0]
    assert sae.tolist() == [0.0, 0.0, 0.0]
    assert n.item() == 0 and n2.item() == 0


def test_mse_and_mae_update_multi_output_pool_columns_and_divide_by_rows() -> None:
    """Finding: with ``num_outputs > 1`` the per-column sums are not per column.

    ``valid_preds = preds[valid_mask]`` flattens the 2-D input, so ``sum(dim=0)`` is the
    scalar total over all valid elements of every column (SSE 17 + 29 = 46, SAE 5 + 7 =
    12; see the module docstring), and the count is ``valid_mask.size(0)``, the row count
    3 including the rows that are NaN in a column. The honest per-column values are SSE
    [17, 29] and SAE [5, 7] with counts [2, 2]
    (torchcell/metrics/nan_tolerant_metrics.py:25-27 and :47-49).
    Pinned until the multi-output branch masks and counts per column.
    """
    sse, n = _nan_tolerant_mse_update(P2, T2, num_outputs=2)
    sae, n2 = _nan_tolerant_mae_update(P2, T2, num_outputs=2)
    assert sse.shape == torch.Size([]) and sse.item() == 46.0
    assert sae.shape == torch.Size([]) and sae.item() == 12.0
    assert n.item() == 3 and n2.item() == 3
    # The honest oracle, per column on the hand-masked rows {0, 2}:
    honest = ((P2[[0, 2]] - T2[[0, 2]]) ** 2).sum(dim=0)
    assert honest.tolist() == [17.0, 29.0]


def test_error_compute_divides_and_returns_nan_for_zero_count() -> None:
    """[6, 8] / 2 = [3, 4]; a zero count gives NaN of the same shape as the sum."""
    out = _nan_tolerant_error_compute(torch.tensor([6.0, 8.0]), torch.tensor(2))
    assert out.tolist() == [3.0, 4.0]
    empty = _nan_tolerant_error_compute(torch.zeros(3), torch.tensor(0))
    assert empty.shape == torch.Size([3])
    assert torch.isnan(empty).all().item()


# ------------------------------------------------------------- MSE / RMSE / MAE


def _accumulate(metric: Metric) -> torch.Tensor:
    for p, t in BATCHES:
        metric.update(p, t)
    out = metric.compute()
    assert isinstance(out, torch.Tensor)
    return out


def test_mse_accumulates_pooled_over_batches_with_unequal_valid_counts() -> None:
    """MSE over four batches = 19.25 / 6 = 3.2083333 (module docstring), equal to
    torchmetrics ``MeanSquaredError`` on the six hand-masked rows and NOT the mean of
    per-batch MSEs 3.0972222. State after accumulation: SSE 19.25, total 6.
    """
    m = NaNTolerantMSE()
    out = _accumulate(m)
    oracle = MeanSquaredError()(VALID_P, VALID_T)
    torch.testing.assert_close(out, oracle, rtol=0, atol=1e-6)
    torch.testing.assert_close(out, torch.tensor(19.25 / 6), rtol=0, atol=1e-6)
    assert abs(out.item() - 3.0972222) > 0.1
    assert m.sum_squared_error.tolist() == [19.25]
    assert m.total.item() == 6
    assert out.dtype == torch.float32 and out.shape == torch.Size([])


def test_mse_squared_false_is_rmse_and_matches_torchmetrics() -> None:
    """``squared=False`` returns sqrt(19.25 / 6) = 1.7911822, equal to
    ``MeanSquaredError(squared=False)`` on the masked rows and to ``NaNTolerantRMSE``.
    """
    rmse_flag = _accumulate(NaNTolerantMSE(squared=False))
    rmse_cls = _accumulate(NaNTolerantRMSE())
    oracle = MeanSquaredError(squared=False)(VALID_P, VALID_T)
    torch.testing.assert_close(rmse_flag, oracle, rtol=0, atol=1e-6)
    torch.testing.assert_close(rmse_cls, oracle, rtol=0, atol=1e-6)
    assert abs(rmse_flag.item() - math.sqrt(19.25 / 6)) < 1e-6


def test_mae_accumulates_pooled_over_batches() -> None:
    """MAE = 9.5 / 6 = 1.5833333, equal to ``MeanAbsoluteError`` on the masked rows."""
    m = NaNTolerantMAE()
    out = _accumulate(m)
    torch.testing.assert_close(
        out, MeanAbsoluteError()(VALID_P, VALID_T), rtol=0, atol=1e-6
    )
    assert abs(out.item() - 9.5 / 6) < 1e-6
    assert m.sum_abs_error.tolist() == [9.5] and m.total.item() == 6


def test_mse_mae_rmse_agree_with_torchmetrics_on_nan_free_random_input() -> None:
    """On NaN-free input (seed 0, 50 rows) each class equals its torchmetrics twin."""
    g = torch.Generator().manual_seed(0)
    p, t = torch.randn(50, generator=g), torch.randn(50, generator=g)
    for ours, theirs in [
        (NaNTolerantMSE(), MeanSquaredError()),
        (NaNTolerantMSE(squared=False), MeanSquaredError(squared=False)),
        (NaNTolerantRMSE(), MeanSquaredError(squared=False)),
        (NaNTolerantMAE(), MeanAbsoluteError()),
    ]:
        ours.update(p, t)
        torch.testing.assert_close(ours.compute(), theirs(p, t), rtol=1e-6, atol=1e-6)


def test_mse_multi_output_class_broadcasts_pooled_value_finding() -> None:
    """Finding: ``NaNTolerantMSE/RMSE/MAE(num_outputs=2)`` return one pooled number per
    column divided by the row count. On P2/T2: MSE [46/3, 46/3] = [15.3333, 15.3333],
    RMSE [3.9158, 3.9158], MAE [4, 4]; the honest per-column values are MSE [8.5, 14.5]
    and MAE [2.5, 3.5] (module docstring). Source: nan_tolerant_metrics.py:25-27, :47-49.
    Pinned until the multi-output update masks and counts per column.
    """
    mse = NaNTolerantMSE(num_outputs=2)
    mse.update(P2, T2)
    rmse = NaNTolerantRMSE(num_outputs=2)
    rmse.update(P2, T2)
    mae = NaNTolerantMAE(num_outputs=2)
    mae.update(P2, T2)
    torch.testing.assert_close(
        mse.compute(), torch.tensor([46 / 3, 46 / 3]), rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        rmse.compute(), torch.tensor([46 / 3, 46 / 3]).sqrt(), rtol=0, atol=1e-5
    )
    assert mae.compute().tolist() == [4.0, 4.0]


def test_mse_single_output_with_2d_input_pools_all_valid_elements() -> None:
    """``num_outputs=1`` on a 2-D input pools every valid element and divides by the
    valid-element count: (17 + 29) / 4 = 11.5 on P2/T2.
    """
    m = NaNTolerantMSE()
    m.update(P2, T2)
    assert m.compute().item() == 11.5
    assert m.total.item() == 4


@pytest.mark.filterwarnings(COMPUTE_BEFORE_UPDATE)
def test_error_metrics_return_nan_without_valid_rows() -> None:
    """Never updated, or updated only with all-NaN rows: compute is NaN, not 0.

    The all-NaN update (batch 2) adds SSE 0 and count 0, so ``total == 0`` and
    ``NaNTolerantMSE.compute`` takes its empty branch; RMSE and MAE return the NaN of
    ``_nan_tolerant_error_compute``.
    """
    for cls in (NaNTolerantMSE, NaNTolerantRMSE, NaNTolerantMAE):
        fresh = cls()
        assert math.isnan(fresh.compute().item())
        m = cls()
        m.update(*BATCHES[1])
        out = m.compute()
        assert math.isnan(out.item()) and out.shape == torch.Size([])


@pytest.mark.filterwarnings(COMPUTE_BEFORE_UPDATE)
def test_mse_reset_restores_initial_state_and_fresh_behavior() -> None:
    """After updates then ``reset``: SSE [0], total 0, compute NaN, and a new update gives
    the same value a fresh metric gives (batch 4: (6 - 4)^2 / 1 = 4).
    """
    m = NaNTolerantMSE()
    _accumulate(m)
    m.reset()
    assert m.sum_squared_error.tolist() == [0.0] and m.total.item() == 0
    assert math.isnan(m.compute().item())
    m.update(*BATCHES[3])
    assert m.compute().item() == 4.0


def test_float64_input_is_accumulated_in_float32_state() -> None:
    """States are registered as float32 zeros, and ``+=`` keeps the state dtype, so a
    float64 batch is reduced to a float32 result. p = [0.1, 0.2], t = [0, 0]:
    (0.01 + 0.04) / 2 = 0.025 in float32.
    """
    m = NaNTolerantMSE()
    m.update(
        torch.tensor([0.1, 0.2], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    out = m.compute()
    assert out.dtype == torch.float32
    assert out.device == torch.device("cpu")
    torch.testing.assert_close(out, torch.tensor(0.025), rtol=0, atol=1e-8)


def test_metric_base_forces_ddp_kwargs() -> None:
    """``NaNTolerantMetricBase`` overwrites the caller's DDP kwargs: compute_on_cpu False,
    sync_on_compute False, dist_sync_on_step True, and registers a [0.] device buffer.
    """
    m = NaNTolerantMSE(
        compute_on_cpu=True, sync_on_compute=True, dist_sync_on_step=False
    )
    assert isinstance(m, NaNTolerantMetricBase)
    assert m.compute_on_cpu is False
    assert m.sync_on_compute is False
    assert m.dist_sync_on_step is True
    assert m._device_buffer.tolist() == [0.0]
    filled = m._create_tensor_on_device(2.5, 2, 3)
    assert filled.shape == torch.Size([2, 3]) and (filled == 2.5).all().item()


# ------------------------------------------------------------------------- Pearson

PEARSON_BATCHES: list[tuple[torch.Tensor, torch.Tensor]] = [
    (torch.tensor([0.5, 1.0, NAN, 2.0]), torch.tensor([1.0, NAN, 3.0, 2.5])),
    (torch.tensor([NAN, 4.0]), torch.tensor([1.0, NAN])),
    (torch.tensor([3.0, -1.0, 0.0]), torch.tensor([2.0, -0.5, 1.5])),
    (torch.tensor([NAN, 5.0, NAN]), torch.tensor([7.0, 3.5, NAN])),
]
PEARSON_VALID_P = [0.5, 2.0, 3.0, -1.0, 0.0, 5.0]
PEARSON_VALID_T = [1.0, 2.5, 2.0, -0.5, 1.5, 3.5]


def test_pearson_streaming_matches_scipy_on_concatenated_valid_rows() -> None:
    """Four batches: batch 1 keeps rows 0 and 3 (row 1 target NaN, row 2 pred NaN);
    batch 2 has no valid row; batch 3 keeps all three; batch 4 keeps one row. The
    accumulated correlation equals ``scipy.stats.pearsonr`` on the six concatenated
    valid pairs. Sufficient statistics: sum_x = 0.5 + 2 + 3 - 1 + 0 + 5 = 9.5,
    sum_y = 1 + 2.5 + 2 - 0.5 + 1.5 + 3.5 = 10, n = 6.
    """
    m = NaNTolerantPearsonCorrCoef()
    for p, t in PEARSON_BATCHES:
        m.update(p, t)
    expected = scipy.stats.pearsonr(PEARSON_VALID_P, PEARSON_VALID_T)[0]
    out = m.compute()
    assert abs(out.item() - expected) < 1e-6
    assert m.sum_x.tolist() == [9.5] and m.sum_y.tolist() == [10.0]
    assert m.n_samples.tolist() == [6.0]
    torch.testing.assert_close(
        out,
        PearsonCorrCoef()(torch.tensor(PEARSON_VALID_P), torch.tensor(PEARSON_VALID_T)),
        rtol=0,
        atol=1e-6,
    )


def test_pearson_forward_returns_batch_value_and_compute_global() -> None:
    """``forward`` (how the trainers call it) returns the batch-only correlation and still
    accumulates: batch 3 alone equals pearsonr([3, -1, 0], [2, -0.5, 1.5]); compute after
    the four forwards equals the global scipy value.
    """
    m = NaNTolerantPearsonCorrCoef()
    batch_values = [m(p, t) for p, t in PEARSON_BATCHES]
    expected_b3 = scipy.stats.pearsonr([3.0, -1.0, 0.0], [2.0, -0.5, 1.5])[0]
    assert abs(batch_values[2].item() - expected_b3) < 1e-6
    assert math.isnan(batch_values[1].item())
    expected = scipy.stats.pearsonr(PEARSON_VALID_P, PEARSON_VALID_T)[0]
    assert abs(m.compute().item() - expected) < 1e-6


@pytest.mark.filterwarnings(COMPUTE_BEFORE_UPDATE)
def test_pearson_degenerate_cases_return_nan() -> None:
    """No update -> NaN of shape () (the (1,) tensor is squeezed by torchmetrics); one
    valid row -> var 0 -> NaN; an exactly representable constant target [2, 2, 2] ->
    var_y = 12/3 - 2^2 = 0 exactly -> NaN.
    """
    assert math.isnan(NaNTolerantPearsonCorrCoef().compute().item())
    one = NaNTolerantPearsonCorrCoef()
    one.update(torch.tensor([1.0, NAN]), torch.tensor([2.0, 3.0]))
    assert math.isnan(one.compute().item())
    const = NaNTolerantPearsonCorrCoef()
    const.update(torch.tensor([0.0, 1.0, 2.0]), torch.full((3,), 2.0))
    assert math.isnan(const.compute().item())


def test_pearson_constant_target_with_float_residue_returns_zero_finding() -> None:
    """Finding: the raw-moment variance ``sum_y2/n - mean_y^2`` leaves a positive float32
    residue for a constant target that is not exactly representable, so the zero-variance
    guard ``denom > 0`` passes and a constant column scores 0.0 instead of NaN.
    y = [1.3, 1.3, 1.3], x = [0, 1, 2]: var_y = 1.1920929e-07 (one float32 ulp), cov 0,
    result 0.0; scipy.stats.pearsonr returns nan with a ConstantInputWarning.
    (torchcell/metrics/nan_tolerant_metrics.py:358-368.)
    Pinned until compute uses centered (Welford) statistics or an explicit tolerance.
    """
    m = NaNTolerantPearsonCorrCoef()
    m.update(torch.tensor([0.0, 1.0, 2.0]), torch.full((3,), 1.3))
    var_y = (m.sum_y2 / 3 - (m.sum_y / 3) ** 2).item()
    assert var_y == pytest.approx(1.1920928955078125e-07, abs=0)
    assert m.compute().item() == 0.0


def test_pearson_constant_predictions_residue_sign_decides_nan_or_value_finding() -> (
    None
):
    """Finding: constant PREDICTIONS (a model that predicts the mean) leave a float32
    residue in ``var_x = sum_x2/n - mean_x^2`` whose sign depends on the constant and n
    (nan_tolerant_metrics.py:359-368), against y = [0, 1, ..., n-1]:

    - x = 1.3 repeated 7 times: var_x = -4.77e-07 < 0, so sqrt(var_x * var_y) is NaN,
      ``denom > 0`` is False, result NaN (the honest answer).
    - x = 1.3 repeated 3 times: var_x = +1.19e-07 > 0, cov 0, result 0.0.
    - x = 0.3 repeated 10 times: var_x = +2.24e-08 > 0 and the covariance residue is
      not zero either, so a small nonzero value (0.000555 observed) is reported.

    scipy returns nan for every case. Pinned until compute uses centered statistics or
    an explicit tolerance.
    """
    neg = NaNTolerantPearsonCorrCoef()
    neg.update(torch.full((7,), 1.3), torch.arange(7.0))
    assert (neg.sum_x2 / 7 - (neg.sum_x / 7) ** 2).item() < 0
    assert math.isnan(neg.compute().item())
    pos = NaNTolerantPearsonCorrCoef()
    pos.update(torch.full((3,), 1.3), torch.arange(3.0))
    assert (pos.sum_x2 / 3 - (pos.sum_x / 3) ** 2).item() > 0
    assert pos.compute().item() == 0.0
    small = NaNTolerantPearsonCorrCoef()
    small.update(torch.full((10,), 0.3), torch.arange(10.0))
    value = small.compute().item()
    assert 0.0 < value < 1e-3


def test_pearson_fitness_scale_stream_error_against_scipy() -> None:
    """Effect size of the raw-moment accumulation at fitness scale (reproducible probe).

    Generation: ``g = torch.Generator().manual_seed(0)``; y = 1 + 0.01 * randn(20000, g);
    x = y + 0.01 * randn(20000, g); streamed in consecutive batches of 64. Observed:
    NaNTolerantPearsonCorrCoef 0.7097331, scipy.stats.pearsonr (float64) 0.7092065,
    torchmetrics PearsonCorrCoef (centered, same float32 stream) 0.7092068. The raw-moment
    error is about 5.3e-4; the centered update's is about 3e-7.
    """
    g = torch.Generator().manual_seed(0)
    y = 1 + 0.01 * torch.randn(20000, generator=g)
    x = y + 0.01 * torch.randn(20000, generator=g)
    ours, centered = NaNTolerantPearsonCorrCoef(), PearsonCorrCoef()
    for i in range(0, 20000, 64):
        ours.update(x[i : i + 64], y[i : i + 64])
        centered.update(x[i : i + 64], y[i : i + 64])
    exact = scipy.stats.pearsonr(x.double().numpy(), y.double().numpy())[0]
    assert exact == pytest.approx(0.7092065121846524, abs=1e-9)
    assert ours.compute().item() == pytest.approx(0.7097331, abs=2e-6)
    assert centered.compute().item() == pytest.approx(0.7092068, abs=2e-6)
    assert abs(ours.compute().item() - exact) > 4e-4


def test_infinite_values_are_not_masked() -> None:
    """Only NaN is masked; +-inf passes the mask. MSE on p = [inf, 1], t = [0, 1] counts
    both rows (total 2) and returns inf. Pearson with one inf prediction counts 3 samples
    and returns NaN (inf - inf in the moments).
    """
    mse = NaNTolerantMSE()
    mse.update(torch.tensor([math.inf, 1.0]), torch.tensor([0.0, 1.0]))
    assert mse.total.item() == 2
    assert mse.compute().item() == math.inf
    pear = NaNTolerantPearsonCorrCoef()
    pear.update(torch.tensor([math.inf, 1.0, 2.0]), torch.tensor([0.0, 1.0, 2.0]))
    assert pear.n_samples.tolist() == [3.0]
    assert math.isnan(pear.compute().item())


def test_pearson_n_samples_is_float32_and_saturates_at_2_pow_24() -> None:
    """``n_samples`` is registered as float32 zeros, so counts are exact only up to
    2^24 = 16777216: adding 1 to a count of 2^24 leaves 16777216.0. Not reachable by the
    003 validation sets (far smaller), recorded as a precision limit.
    """
    m = NaNTolerantPearsonCorrCoef()
    assert m.n_samples.dtype == torch.float32
    m.n_samples += 2**24
    m.n_samples += torch.tensor(1)
    assert m.n_samples.item() == 16777216.0


def test_pearson_large_offset_cancellation_flips_sign_finding() -> None:
    """Finding: raw float32 moments cancel catastrophically under an offset. x = [0, 1, 2]
    + 11000, y = [0, 2, 1] + 11000 has Pearson 0.5 (offset-invariant; scipy gives 0.5);
    the metric returns -1.0. torchmetrics ``PearsonCorrCoef`` (centered updates) gives
    0.5. (torchcell/metrics/nan_tolerant_metrics.py:340-360.)
    Pinned until the update accumulates centered statistics.
    """
    x = torch.tensor([0.0, 1.0, 2.0]) + 11000.0
    y = torch.tensor([0.0, 2.0, 1.0]) + 11000.0
    m = NaNTolerantPearsonCorrCoef()
    m.update(x, y)
    assert m.compute().item() == -1.0
    assert scipy.stats.pearsonr(x.double().numpy(), y.double().numpy())[0] == (
        pytest.approx(0.5, abs=1e-12)
    )
    assert PearsonCorrCoef()(x, y).item() == pytest.approx(0.5, abs=1e-6)


def test_pearson_multi_output_pools_columns_and_compute_raises_finding() -> None:
    """Finding: ``num_outputs=2`` registers (2,) states but the update flattens the mask,
    so both columns receive the same pooled sums (valid elements of P2: x = 1, 5, 2, 6 ->
    sum_x 14, n 4 for each column), and ``compute`` then fails on ``if self.n_samples ==
    0`` with a two-element tensor (nan_tolerant_metrics.py:336-350).
    Pinned until update masks per column and compute uses ``(n_samples == 0).all()``.
    """
    m = NaNTolerantPearsonCorrCoef(num_outputs=2)
    m.update(P2, T2)
    assert m.sum_x.tolist() == [14.0, 14.0]
    assert m.n_samples.tolist() == [4.0, 4.0]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Boolean value of Tensor with more than one value is ambiguous"
        ),
    ):
        m.compute()


@pytest.mark.filterwarnings(COMPUTE_BEFORE_UPDATE)
def test_pearson_reset_clears_sums() -> None:
    """``reset`` zeroes all six sums; a following single batch equals a fresh metric."""
    m = NaNTolerantPearsonCorrCoef()
    for p, t in PEARSON_BATCHES:
        m.update(p, t)
    m.reset()
    for name in ("sum_x", "sum_y", "sum_xy", "sum_x2", "sum_y2", "n_samples"):
        assert getattr(m, name).tolist() == [0.0]
    assert math.isnan(m.compute().item())
    m.update(*PEARSON_BATCHES[2])
    expected = scipy.stats.pearsonr([3.0, -1.0, 0.0], [2.0, -0.5, 1.5])[0]
    assert abs(m.compute().item() - expected) < 1e-6


# -------------------------------------------------- streaming helpers (no callers)


def _stats(x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    """(mean_x, mean_y, M2_x, M2_y, C_xy, n): the parallel-variance state of a shard."""
    mx, my = x.mean(), y.mean()
    return [
        mx,
        my,
        ((x - mx) ** 2).sum(),
        ((y - my) ** 2).sum(),
        ((x - mx) * (y - my)).sum(),
        torch.tensor(float(x.numel()), dtype=x.dtype),
    ]


X1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
Y1 = torch.tensor([2.0, 1.0, 4.0], dtype=torch.float64)
X2 = torch.tensor([10.0, 12.0], dtype=torch.float64)
Y2 = torch.tensor([7.0, 9.0], dtype=torch.float64)
X3 = torch.tensor([-4.0], dtype=torch.float64)
Y3 = torch.tensor([0.5], dtype=torch.float64)


def test_final_aggregation_two_devices_equals_concatenated_statistics() -> None:
    """Shards (X1, Y1) n=3 and (X2, Y2) n=2. Concatenation x = [1, 2, 3, 10, 12],
    y = [2, 1, 4, 7, 9]: mean_x 5.6, mean_y 4.6, M2_x 101.2, M2_y 45.2, C_xy 65.2, n 5.
    (Per-shard: M2_x 2 + 2, cross term n1 n2 / n (m1 - m2)^2 = 6/5 * 9^2 = 97.2.)
    """
    states = [torch.stack(pair) for pair in zip(_stats(X1, Y1), _stats(X2, Y2))]
    out = _final_aggregation(*states)
    expected = [5.6, 4.6, 101.2, 45.2, 65.2, 5.0]
    for got, want in zip(out, expected):
        assert got.item() == pytest.approx(want, abs=1e-10)


def test_final_aggregation_three_unequal_devices_matches_numpy() -> None:
    """Three shards n = 3, 2, 1: the result equals numpy's statistics of the
    concatenation (mean, sum of squared deviations, co-moment, count 6).
    """
    states = [
        torch.stack(t) for t in zip(_stats(X1, Y1), _stats(X2, Y2), _stats(X3, Y3))
    ]
    x = np.concatenate([X1.numpy(), X2.numpy(), X3.numpy()])
    y = np.concatenate([Y1.numpy(), Y2.numpy(), Y3.numpy()])
    expected = [
        x.mean(),
        y.mean(),
        ((x - x.mean()) ** 2).sum(),
        ((y - y.mean()) ** 2).sum(),
        ((x - x.mean()) * (y - y.mean())).sum(),
        6.0,
    ]
    out = _final_aggregation(*states)
    for got, want in zip(out, expected):
        assert got.item() == pytest.approx(want, abs=1e-10)
    corr = out[4] / torch.sqrt(out[2] * out[3])
    assert corr.item() == pytest.approx(scipy.stats.pearsonr(x, y)[0], abs=1e-12)


def test_final_aggregation_single_device_returns_its_state() -> None:
    """One device: the six tensors' only rows are returned unchanged."""
    states = [s.unsqueeze(0) for s in _stats(X1, Y1)]
    out = _final_aggregation(*states)
    assert [o.item() for o in out] == [s[0].item() for s in states]


def test_final_aggregation_mutates_caller_state_finding() -> None:
    """Finding: ``vx1 += ...`` / ``vx2 += ...`` act on views of the caller's tensors, so
    aggregating overwrites ``vars_x`` (and ``vars_y``, ``corrs_xy``) in place. Shards
    M2_x [2, 2] become [2 + n1 d1^2, 2 + n2 d2^2] = [2 + 3 * 3.6^2, 2 + 2 * 5.4^2] =
    [40.88, 60.32] (nan_tolerant_metrics.py:202-204). A second aggregation of the same
    state then double counts. Pinned until the function works on copies.
    """
    states = [torch.stack(pair) for pair in zip(_stats(X1, Y1), _stats(X2, Y2))]
    _final_aggregation(*states)
    assert states[2][0].item() == pytest.approx(40.88, abs=1e-10)
    assert states[2][1].item() == pytest.approx(60.32, abs=1e-10)
    again = _final_aggregation(*states)
    assert again[2].item() == pytest.approx(101.2 + 97.2, abs=1e-10)


def test_pearson_update_helper_first_batch_exact_second_understates_finding() -> None:
    """Finding: ``_nan_tolerant_pearson_update`` adds sum((x_b - new_mean)^2), which omits
    the prior shard's shift n_a (m_a - new_mean)^2 (nan_tolerant_metrics.py:261-263).
    First batch (X1, Y1) from zero state is exact: means 2 and 7/3, M2_x 2, M2_y 42/9 =
    4.6667, C 2, n 3. After (X2, Y2): means 5.6 and 4.6 are exact, but M2_x is 2 + 19.36 +
    40.96 = 62.32 instead of 101.2 (missing 3 * 3.6^2 = 38.88), M2_y 29.7867 instead of
    45.2, C 40.72 instead of 65.2. The helper has no caller in torchcell or experiments.
    Pinned until the update uses the Welford old-mean/new-mean product.
    """
    z = torch.zeros(1, dtype=torch.float64)
    s1 = _nan_tolerant_pearson_update(X1, Y1, z, z, z, z, z, z.clone(), num_outputs=1)
    for got, want in zip(s1, [2.0, 7 / 3, 2.0, 42 / 9, 2.0, 3.0]):
        assert got.item() == pytest.approx(want, abs=1e-12)
    s2 = _nan_tolerant_pearson_update(X2, Y2, *s1, num_outputs=1)
    for got, want in zip(s2, [5.6, 4.6, 62.32, 29.786666666666665, 40.72, 5.0]):
        assert got.item() == pytest.approx(want, abs=1e-10)


def test_pearson_update_helper_all_nan_returns_inputs_unchanged() -> None:
    """A batch with no valid pair returns the six state tensors themselves."""
    a, b, c, d, e, n = (torch.tensor([float(i)]) for i in range(6))
    state = (a, b, c, d, e, n)
    out = _nan_tolerant_pearson_update(
        torch.tensor([NAN, 1.0]), torch.tensor([2.0, NAN]), a, b, c, d, e, n, 1
    )
    assert all(a is b for a, b in zip(out, state))


# ------------------------------------------------------------------------ Spearman


def test_spearman_warns_about_buffer_and_matches_scipy_without_ties() -> None:
    """Construction emits the buffer warning. Over the four Pearson batches (no tied
    values among the six valid pairs) the result equals ``scipy.stats.spearmanr`` on the
    concatenation; ``num_samples`` is 6 and the buffers hold 2, 3, 1 valid elements.
    """
    with pytest.warns(
        UserWarning, match=re.escape("Metric `SpearmanCorrcoef` will save")
    ):
        m = NaNTolerantSpearmanCorrCoef()
    for p, t in PEARSON_BATCHES:
        m.update(p, t)
    expected = scipy.stats.spearmanr(PEARSON_VALID_P, PEARSON_VALID_T)[0]
    out = m.compute()
    assert out.item() == pytest.approx(expected, abs=1e-6)
    assert m.num_samples.item() == 6
    assert [b.numel() for b in m.preds] == [2, 3, 1]


@pytest.mark.filterwarnings(SPEARMAN_BUFFER_WARNING)
def test_spearman_ties_ranked_by_order_not_averaged_finding() -> None:
    """Finding: ranks come from ``argsort(argsort(x))``, which gives tied values distinct
    consecutive ranks in row order instead of their average (nan_tolerant_metrics.py:
    434-435), so the score of tied data depends on row order.

    - x = [1, 1, 2, 3], y = [1, 2, 3, 4]: ranks x [0, 1, 2, 3], y [0, 1, 2, 3] -> 1.0.
    - Same x, y = [2, 1, 3, 4]: ranks y [1, 0, 2, 3], one swapped pair, sum d^2 = 2,
      1 - 6 * 2 / (4 * 15) = 0.8. scipy averages the x tie to [1.5, 1.5, 3, 4] and gives
      0.9486833 for BOTH orders.
    - A constant target gets ranks [0, 1, ..., n-1] in row order, so the score is the rank
      correlation of the predictions with ROW POSITION, not a fixed value: preds
      [1, 2, 3] against [5, 5, 5] -> 1.0, preds [3, 2, 1] -> -1.0; scipy returns nan
      for both.

    Pinned until ranks use average tie handling.
    """
    m = NaNTolerantSpearmanCorrCoef()
    m.update(torch.tensor([1.0, 1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert m.compute().item() == pytest.approx(1.0, abs=1e-6)
    assert scipy.stats.spearmanr([1, 1, 2, 3], [1, 2, 3, 4])[0] == pytest.approx(
        0.9486832980505138, abs=1e-12
    )
    swapped = NaNTolerantSpearmanCorrCoef()
    swapped.update(
        torch.tensor([1.0, 1.0, 2.0, 3.0]), torch.tensor([2.0, 1.0, 3.0, 4.0])
    )
    assert swapped.compute().item() == pytest.approx(0.8, abs=1e-6)
    assert scipy.stats.spearmanr([1, 1, 2, 3], [2, 1, 3, 4])[0] == pytest.approx(
        0.9486832980505138, abs=1e-12
    )
    const = NaNTolerantSpearmanCorrCoef()
    const.update(torch.tensor([1.0, 2.0, 3.0]), torch.full((3,), 5.0))
    assert const.compute().item() == pytest.approx(1.0, abs=1e-6)
    const_reversed = NaNTolerantSpearmanCorrCoef()
    const_reversed.update(torch.tensor([3.0, 2.0, 1.0]), torch.full((3,), 5.0))
    assert const_reversed.compute().item() == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.filterwarnings(SPEARMAN_BUFFER_WARNING)
def test_spearman_empty_and_all_nan_return_scalar_nan() -> None:
    """No valid sample: compute returns a 0-d NaN tensor (shape ()), buffers stay empty."""
    m = NaNTolerantSpearmanCorrCoef()
    m.update(torch.tensor([NAN, 1.0]), torch.tensor([1.0, NAN]))
    assert m.preds == [] and m.num_samples.item() == 0
    out = m.compute()
    assert out.shape == torch.Size([]) and math.isnan(out.item())


@pytest.mark.filterwarnings(SPEARMAN_BUFFER_WARNING)
def test_spearman_num_outputs_validation_never_rejects_finding() -> None:
    """Finding: the guard ``not isinstance(num_outputs, int) and num_outputs < 1`` can
    only fire for a non-int below 1, so ``num_outputs=0`` is accepted
    (nan_tolerant_metrics.py:396). Pinned until the guard uses ``or``.
    """
    assert NaNTolerantSpearmanCorrCoef(num_outputs=0).num_outputs == 0
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected argument `num_outputs` to be an int larger than 0, but got 0.5"
        ),
    ):
        NaNTolerantSpearmanCorrCoef(num_outputs=0.5)  # type: ignore[arg-type, unused-ignore]


@pytest.mark.filterwarnings(SPEARMAN_BUFFER_WARNING)
def test_spearman_multi_output_pools_columns_then_raises_finding() -> None:
    """Finding: ``num_outputs=2`` flattens both columns into one rank vector and hands it
    to a two-output Pearson, whose compute raises on the (2,) ``n_samples`` check (see
    the Pearson multi-output finding). Pinned with that fix.
    """
    m = NaNTolerantSpearmanCorrCoef(num_outputs=2)
    m.update(P2, T2)
    assert m.preds[0].tolist() == [1.0, 2.0, 5.0, 6.0]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Boolean value of Tensor with more than one value is ambiguous"
        ),
    ):
        m.compute()


@pytest.mark.filterwarnings(SPEARMAN_BUFFER_WARNING)
@pytest.mark.filterwarnings(COMPUTE_BEFORE_UPDATE)
def test_spearman_reset_empties_buffers() -> None:
    """``reset`` empties both buffers and zeroes the counter; compute is NaN again."""
    m = NaNTolerantSpearmanCorrCoef()
    m.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 3.0, 2.0]))
    assert m.compute().item() == pytest.approx(0.5, abs=1e-6)
    m.reset()
    assert m.preds == [] and m.target == [] and m.num_samples.item() == 0
    assert math.isnan(m.compute().item())


# ----------------------------------------------------------------------------- R2


def test_r2_single_batch_matches_torchmetrics_on_masked_rows() -> None:
    """One batch: R2 over valid rows equals ``R2Score`` on them. Batch 3 plus a NaN row:
    p = [0, 2, 10, nan], t = [1, 0, 7, 3]; valid t = [1, 0, 7], mean 8/3; SSE 1 + 4 + 9 =
    14; SST (5/3)^2 + (8/3)^2 + (13/3)^2 = 258/9 = 28.6667; R2 = 1 - 14 / 28.6667 =
    0.5116279.
    """
    m = NaNTolerantR2Score()
    m.update(torch.tensor([0.0, 2.0, 10.0, NAN]), torch.tensor([1.0, 0.0, 7.0, 3.0]))
    out = m.compute()
    assert out.item() == pytest.approx(1 - 14 / (258 / 9), abs=1e-6)
    oracle = R2Score()(torch.tensor([0.0, 2.0, 10.0]), torch.tensor([1.0, 0.0, 7.0]))
    torch.testing.assert_close(out, oracle, rtol=0, atol=1e-6)
    assert m.total.item() == 3


def test_r2_accumulation_uses_per_batch_target_mean_finding() -> None:
    """Finding: each update centers the targets on THAT batch's mean, so the accumulated
    total sum of squares is the within-batch sum, not the sum around the global mean
    (nan_tolerant_metrics.py:543-554). Batches (p [1, 2], t [1.5, 2.5]) and
    (p [10, 11], t [9, 12]): SSE 0.5 + 2 = 2.5; within-batch SST 0.5 + 4.5 = 5 -> R2 0.5.
    On the concatenation t = [1.5, 2.5, 9, 12], mean 6.25, SST 77.25 -> R2 = 1 - 2.5 /
    77.25 = 0.9676375 (torchmetrics ``R2Score`` agrees).
    Pinned until the update accumulates sum(t) and sum(t^2) and compute centers globally.
    """
    m = NaNTolerantR2Score()
    m.update(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 2.5]))
    m.update(torch.tensor([10.0, 11.0]), torch.tensor([9.0, 12.0]))
    assert m.compute().item() == 0.5
    oracle = R2Score()(
        torch.tensor([1.0, 2.0, 10.0, 11.0]), torch.tensor([1.5, 2.5, 9.0, 12.0])
    )
    assert oracle.item() == pytest.approx(1 - 2.5 / 77.25, abs=1e-6)


def test_r2_multi_output_pools_columns_finding() -> None:
    """Finding: with ``num_outputs=2`` the masked 2-D input is flattened, so both columns
    get the pooled R2. P2/T2 valid elements: p [1, 2, 5, 6], t [0, 0, 1, 1], mean 0.5,
    SSE 46, SST 1 -> R2 -45 in both columns. Per column (rows 0, 2): col0 SSE 17, SST 0.5
    -> -33; col1 SSE 29, SST 0.5 -> -57 (nan_tolerant_metrics.py:539-551).
    Pinned until the update masks per column.
    """
    m = NaNTolerantR2Score(num_outputs=2)
    m.update(P2, T2)
    assert m.compute().tolist() == [-45.0, -45.0]
    assert m.sum_squared_error.tolist() == [46.0, 46.0]


@pytest.mark.filterwarnings(COMPUTE_BEFORE_UPDATE)
def test_r2_degenerate_cases() -> None:
    """No update, or only an all-NaN batch -> NaN; constant target [3, 3] -> SST 0 ->
    NaN (not -inf); one valid row -> SST 0 -> NaN; reset restores SSE [0], SST [0],
    total 0.
    """
    assert math.isnan(NaNTolerantR2Score().compute().item())
    all_nan = NaNTolerantR2Score()
    all_nan.update(torch.tensor([NAN, 1.0]), torch.tensor([2.0, NAN]))
    assert all_nan.total.item() == 0
    assert math.isnan(all_nan.compute().item())
    const = NaNTolerantR2Score()
    const.update(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 3.0]))
    assert math.isnan(const.compute().item())
    assert const.sum_squared_error.tolist() == [5.0]
    one = NaNTolerantR2Score()
    one.update(torch.tensor([1.0, NAN]), torch.tensor([3.0, 4.0]))
    assert math.isnan(one.compute().item())
    one.reset()
    assert one.sum_squared_error.tolist() == [0.0]
    assert one.sum_squared_deviation.tolist() == [0.0]
    assert one.total.item() == 0


def test_gat_diffpool_inception_trainer_cannot_import_its_metrics() -> None:
    """Finding (reach): ``fit_int_gat_diffpool_inception_regression`` imports
    ``NaNTolerantPearsonCorrCoef`` and ``NaNTolerantSpearmanCorrCoef`` from
    ``torchcell.losses.multi_dim_nan_tolerant``, where they do not exist (they live in
    ``torchcell.metrics.nan_tolerant_metrics``), so the trainer module cannot be imported
    (fit_int_gat_diffpool_inception_regression.py:25-29; also listed as a known failure
    in tests/torchcell/test_import_all.py). Pinned until the import points at
    ``torchcell.metrics.nan_tolerant_metrics``.
    """
    with pytest.raises(
        ImportError,
        match=re.escape(
            "cannot import name 'NaNTolerantPearsonCorrCoef' from "
            "'torchcell.losses.multi_dim_nan_tolerant'"
        ),
    ):
        importlib.import_module(
            "torchcell.trainers.fit_int_gat_diffpool_inception_regression"
        )
