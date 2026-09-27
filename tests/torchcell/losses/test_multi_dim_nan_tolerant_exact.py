# tests/torchcell/losses/test_multi_dim_nan_tolerant_exact.py
# [[tests.torchcell.losses.test_multi_dim_nan_tolerant_exact]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_multi_dim_nan_tolerant_exact.py
"""Closed-form values for every loss in ``torchcell.losses.multi_dim_nan_tolerant``.

Shared regression fixture (three rows, two dimensions)::

    Y_PRED = [[1, 2], [3, 4], [5, 6]]
    Y_TRUE = [[0, nan], [1, 5], [nan, 3]]

Dimension 0 keeps rows 0 and 1 with residuals 1 and 2; dimension 1 keeps rows 1 and 2
with residuals 1 and 3. The per-dimension denominators are the valid counts, 2 and 2:

    L1        [ (1 + 2) / 2,           (1 + 3) / 2 ]           = [1.5, 2.0]
    Huber(1)  [ (0.5 + 1.5) / 2,       (0.5 + 2.5) / 2 ]       = [1.0, 1.5]
    Huber(2)  [ (0.5 + 2) / 2,         (0.5 + 4) / 2 ]         = [1.25, 2.25]
    log-cosh  [ (0.4337808 + 1.3250027) / 2, (0.4337808 + 2.3093275) / 2 ]
                                                             = [0.8793918, 1.3715547]
    MSE       [ (1 + 4) / 2,           (1 + 9) / 2 ]           = [2.5, 5.0]

Shared SupCR fixture: embeddings e0 = (1, 0), e1 = (0, 1), e2 = (1, 1) and labels
(0, 1, 3) at temperature 1. Cosine similarities: s01 = 0, s02 = s12 = 1/sqrt(2) = c,
diagonal 1. For anchor i and positive j the denominator sums exp(s_ik) over every k
whose label distance from i is at least the distance of j:

    anchor 0: j=1 (d=1): denominator 1 + e^c, numerator 1   -> log(1 + e^c)
              j=2 (d=3): denominator e^c, numerator e^c     -> 0
    anchor 1: j=0 (d=1): denominator 1 + e^c, numerator 1   -> log(1 + e^c)
              j=2 (d=2): denominator e^c                    -> 0
    anchor 2: j=0 (d=3): denominator e^c                    -> 0
              j=1 (d=2): denominator 2 e^c, numerator e^c   -> log 2

    loss = (2 log(1 + e^c) + log 2) / 6 = 0.4848379660

The independent reference used for random inputs is ``_naive_supcr``, an O(N^3)
transcription of that per-anchor formula with no sort or cumulative sum.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from torchcell.losses.multi_dim_nan_tolerant import (
    CategoricalEntropyRegLoss,
    CombinedCELoss,
    CombinedOrdinalCELoss,
    CombinedRegressionLoss,
    MonotonicParameter,
    MseCategoricalEntropyRegLoss,
    MultiDimNaNTolerantCELoss,
    MultiDimNaNTolerantOrdinalCELoss,
    NaNTolerantHuberLoss,
    NaNTolerantL1Loss,
    NaNTolerantLogCoshLoss,
    NaNTolerantMSELoss,
    NaNTolerantQuantileLoss,
    SupCR,
    WeightedDistLoss,
    WeightedMSELoss,
    WeightedSupCRCell,
    fast_soft_sort,
    isotonic_l2_pav,
)

NAN = float("nan")
Y_PRED = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
Y_TRUE = torch.tensor([[0.0, NAN], [1.0, 5.0], [NAN, 3.0]])
MASK = torch.tensor([[True, False], [True, True], [False, True]])
Y_TRUE_DIM1_NAN = torch.tensor([[0.0, NAN], [1.0, NAN], [NAN, NAN]])

EMB = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
LABELS = torch.tensor([0.0, 1.0, 3.0])
EXP_C = math.exp(1 / math.sqrt(2))
SUPCR_HAND = (2 * math.log(1 + EXP_C) + math.log(2)) / 6

LOG_COSH_1 = math.log(math.cosh(1.0))
LOG_COSH_2 = math.log(math.cosh(2.0))
LOG_COSH_3 = math.log(math.cosh(3.0))
LOG2 = math.log(2.0)


def _naive_supcr(emb: torch.Tensor, labels: torch.Tensor, temp: float) -> float:
    """SupCR by the per-anchor formula: the denominator sums over k != i with d_ik >= d_ij."""
    valid = ~torch.isnan(labels)
    emb, labels = emb[valid].double(), labels[valid].double()
    m = len(labels)
    if m < 2:
        return 0.0
    normed = emb / emb.norm(dim=1, keepdim=True)
    sims = (normed @ normed.T) / temp
    dist = (labels[None, :] - labels[:, None]).abs()
    total = 0.0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            denominator = sum(
                math.exp(sims[i, k].item())
                for k in range(m)
                if k != i and dist[i, k] >= dist[i, j]
            )
            total += -math.log(math.exp(sims[i, j].item()) / denominator)
    return total / (m * (m - 1))


# --------------------------------------------------------------------------- SupCR


def test_reversed_cumsum_is_the_suffix_sum() -> None:
    """[1, 2, 3] -> [6, 5, 3]."""
    out = SupCR()._reversed_cumsum(torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(out, torch.tensor([6.0, 5.0, 3.0]))


def test_similarity_is_cosine_over_temperature() -> None:
    """Orthogonal unit vectors at temperature 0.5 give 1/0.5 = 2 on the diagonal, 0 off it."""
    sims = SupCR(temperature=0.5).compute_similarity(EMB[:2])
    torch.testing.assert_close(sims, torch.tensor([[2.0, 0.0], [0.0, 2.0]]))


def test_dimension_loss_matches_the_hand_derivation() -> None:
    """(2 log(1 + e^(1/sqrt 2)) + log 2) / 6 = 0.4848379660 on the module fixture."""
    loss = SupCR(temperature=1.0).compute_dimension_loss(EMB, LABELS)
    assert loss.item() == pytest.approx(SUPCR_HAND, abs=1e-6)
    assert loss.item() == pytest.approx(0.4848379660, abs=1e-6)


def test_nan_labels_are_dropped_before_the_pairwise_sums() -> None:
    """A fourth sample with a NaN label leaves the three-sample value unchanged."""
    emb = torch.cat([EMB, torch.tensor([[5.0, -2.0]])])
    labels = torch.cat([LABELS, torch.tensor([NAN])])
    loss = SupCR(temperature=1.0).compute_dimension_loss(emb, labels)
    assert loss.item() == pytest.approx(SUPCR_HAND, abs=1e-6)


def test_fewer_than_two_valid_labels_returns_zero() -> None:
    """One valid label has no pairs, so the dimension loss is exactly 0."""
    loss = SupCR().compute_dimension_loss(EMB, torch.tensor([0.0, NAN, NAN]))
    assert loss.item() == 0.0


def test_tied_labels_exclude_the_anchor_from_the_denominator() -> None:
    """For d_ij = 0 the suffix sum covers the tied k but not k = i itself.

    Labels (0, 0, 1): anchors 0 and 1 each have a positive at distance 0, and the
    denominator is 1 + e^c (the tied positive plus the far sample; the anchor's own
    exp(1/T) = e is excluded), giving log(1 + e^c) for that pair and 0 for the far
    pair; anchor 2 contributes log 2 twice. Loss = (2 log(1 + e^c) + 2 log 2) / 6 =
    0.6003624961. Before the fix the anchor's own term entered the tied denominators
    and the value was (2 log(1 + e + e^c) + 2 log 2) / 6 = 0.8139067324.
    """
    loss = SupCR(temperature=1.0).compute_dimension_loss(
        EMB, torch.tensor([0.0, 0.0, 1.0])
    )
    expected = (2 * math.log(1 + EXP_C) + 2 * LOG2) / 6
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert loss.item() == pytest.approx(0.6003624961, abs=1e-6)
    assert loss.item() == pytest.approx(
        _naive_supcr(EMB, torch.tensor([0.0, 0.0, 1.0]), 1.0), abs=1e-6
    )


def test_forward_stacks_one_loss_per_label_column() -> None:
    """Column 0 is the hand value; column 1 with labels (0, 3, 1) matches the naive sum."""
    labels = torch.stack([LABELS, torch.tensor([0.0, 3.0, 1.0])], dim=1)
    losses = SupCR(temperature=1.0)(EMB, labels)
    assert losses.shape == (2,)
    assert losses[0].item() == pytest.approx(SUPCR_HAND, abs=1e-6)
    assert losses[1].item() == pytest.approx(
        _naive_supcr(EMB, labels[:, 1], 1.0), abs=1e-6
    )
    assert losses[1].item() == pytest.approx(0.2491357056, abs=1e-6)


def test_weighted_supcr_cell_normalizes_weights_and_combines() -> None:
    """Weights [1, 3] -> [0.25, 0.75]; total = 0.25 * 0.4848380 + 0.75 * 0.2491357 = 0.3080613."""
    cell = WeightedSupCRCell(temperature=1.0, weights=torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(cell.weights, torch.tensor([0.25, 0.75]))
    labels = torch.stack([LABELS, torch.tensor([0.0, 3.0, 1.0])], dim=1)
    total, dims = cell(EMB, labels)
    dim1 = _naive_supcr(EMB, labels[:, 1], 1.0)
    torch.testing.assert_close(
        dims, torch.tensor([SUPCR_HAND, dim1]), atol=1e-6, rtol=0
    )
    assert total.item() == pytest.approx(0.25 * SUPCR_HAND + 0.75 * dim1, abs=1e-6)
    assert total.item() == pytest.approx(0.3080612707, abs=1e-6)


def test_weighted_supcr_cell_default_weights_are_half_half() -> None:
    """``weights=None`` registers ones(2) / 2 = [0.5, 0.5]."""
    torch.testing.assert_close(WeightedSupCRCell().weights, torch.tensor([0.5, 0.5]))


def test_weighted_supcr_cell_renormalizes_over_valid_dimensions() -> None:
    """An all-NaN column loses its weight, so the total equals the surviving dimension."""
    cell = WeightedSupCRCell(temperature=1.0, weights=torch.tensor([1.0, 3.0]))
    labels = torch.stack([LABELS, torch.full((3,), NAN)], dim=1)
    total, dims = cell(EMB, labels)
    torch.testing.assert_close(dims, torch.tensor([SUPCR_HAND, 0.0]), atol=1e-6, rtol=0)
    assert total.item() == pytest.approx(SUPCR_HAND, abs=1e-6)


def test_weighted_supcr_cell_all_nan_returns_zeros() -> None:
    """Every label NaN -> (0.0, [0.0, 0.0])."""
    total, dims = WeightedSupCRCell()(EMB, torch.full((3, 2), NAN))
    assert total.item() == 0.0
    torch.testing.assert_close(dims, torch.zeros(2))


# ------------------------------------------------------- element-wise regression losses


def test_l1_means_over_the_valid_count_per_dimension() -> None:
    """[(1 + 2) / 2, (1 + 3) / 2] = [1.5, 2.0] with the NaN mask returned."""
    dims, mask = NaNTolerantL1Loss()(Y_PRED, Y_TRUE)
    torch.testing.assert_close(dims, torch.tensor([1.5, 2.0]))
    assert torch.equal(mask, MASK)


def test_l1_all_nan_dimension_is_zero() -> None:
    """Column 1 entirely NaN: the sum is 0 over a denominator clamped to 1."""
    dims, mask = NaNTolerantL1Loss()(Y_PRED, Y_TRUE_DIM1_NAN)
    torch.testing.assert_close(dims, torch.tensor([1.5, 0.0]))
    assert torch.equal(mask[:, 1], torch.tensor([False, False, False]))


def test_l1_rejects_mismatched_shapes() -> None:
    """The shape assertion carries its message."""
    with pytest.raises(
        AssertionError, match="Predictions and targets must have the same shape"
    ):
        NaNTolerantL1Loss()(Y_PRED, Y_TRUE[:2])


def test_huber_switches_from_quadratic_to_linear_at_delta() -> None:
    """Delta 1: [(0.5 + 1.5) / 2, (0.5 + 2.5) / 2]; delta 2: [(0.5 + 2) / 2, (0.5 + 4) / 2]."""
    dims1, mask = NaNTolerantHuberLoss(delta=1.0)(Y_PRED, Y_TRUE)
    dims2, _ = NaNTolerantHuberLoss(delta=2.0)(Y_PRED, Y_TRUE)
    torch.testing.assert_close(dims1, torch.tensor([1.0, 1.5]))
    torch.testing.assert_close(dims2, torch.tensor([1.25, 2.25]))
    assert torch.equal(mask, MASK)


def test_logcosh_means_over_the_valid_count() -> None:
    """[(logcosh 1 + logcosh 2) / 2, (logcosh 1 + logcosh 3) / 2] = [0.8793918, 1.3715547]."""
    dims, mask = NaNTolerantLogCoshLoss()(Y_PRED, Y_TRUE)
    expected = torch.tensor(
        [(LOG_COSH_1 + LOG_COSH_2) / 2, (LOG_COSH_1 + LOG_COSH_3) / 2]
    )
    torch.testing.assert_close(dims, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        dims, torch.tensor([0.8793917889, 1.3715546675]), atol=1e-6, rtol=0
    )
    assert torch.equal(mask, MASK)


def test_logcosh_all_nan_dimension_is_zero() -> None:
    """Masked residuals are 0 and log(cosh(1e-12)) underflows to exactly 0 in float32."""
    dims, _ = NaNTolerantLogCoshLoss()(Y_PRED, Y_TRUE_DIM1_NAN)
    assert dims[1].item() == 0.0
    assert dims[0].item() == pytest.approx(0.8793917889, abs=1e-6)


def test_mse_means_over_the_valid_count() -> None:
    """[(1 + 4) / 2, (1 + 9) / 2] = [2.5, 5.0]."""
    dims, mask = NaNTolerantMSELoss()(Y_PRED, Y_TRUE)
    torch.testing.assert_close(dims, torch.tensor([2.5, 5.0]))
    assert torch.equal(mask, MASK)


def test_weighted_mse_with_weights_is_the_normalized_weighted_mean() -> None:
    """[1, 3] -> [0.25, 0.75]; 0.25 * 2.5 + 0.75 * 5 = 4.375."""
    total, dims = WeightedMSELoss(weights=torch.tensor([1.0, 3.0]))(Y_PRED, Y_TRUE)
    torch.testing.assert_close(dims, torch.tensor([2.5, 5.0]))
    assert total.item() == pytest.approx(4.375, abs=1e-6)


def test_weighted_mse_without_weights_is_the_plain_mean() -> None:
    """``weights`` stays None and the total is (2.5 + 5) / 2 = 3.75."""
    loss = WeightedMSELoss()
    assert loss.weights is None
    total, _ = loss(Y_PRED, Y_TRUE)
    assert total.item() == pytest.approx(3.75, abs=1e-6)


def test_weighted_mse_drops_an_all_nan_dimension_from_the_weights() -> None:
    """Column 1 NaN: weights become [0.25, 0] and the total is 2.5 * 0.25 / 0.25 = 2.5."""
    total, dims = WeightedMSELoss(weights=torch.tensor([1.0, 3.0]))(
        Y_PRED, Y_TRUE_DIM1_NAN
    )
    torch.testing.assert_close(dims, torch.tensor([2.5, 0.0]))
    assert total.item() == pytest.approx(2.5, abs=1e-6)


def test_weighted_mse_all_nan_returns_zeros() -> None:
    """Every target NaN -> (0.0, [0.0, 0.0])."""
    total, dims = WeightedMSELoss()(Y_PRED, torch.full((3, 2), NAN))
    assert total.item() == 0.0
    torch.testing.assert_close(dims, torch.zeros(2))


def test_quantile_loss_sums_the_pinball_means() -> None:
    """Residuals target - pred = (2, -1); the NaN row is dropped.

    q=0.1: max(0.2, -1.8) = 0.2 and max(-0.1, 0.9) = 0.9, mean 0.55
    q=0.5: 1.0 and 0.5, mean 0.75
    q=0.9: 1.8 and max(-0.9, 0.1) = 0.1, mean 0.95
    sum over quantiles = 2.25
    """
    loss = NaNTolerantQuantileLoss([0.1, 0.5, 0.9])
    torch.testing.assert_close(loss.quantiles, torch.tensor([0.1, 0.5, 0.9]))
    out = loss(torch.tensor([0.0, 1.0, 5.0]), torch.tensor([2.0, 0.0, NAN]))
    assert out.item() == pytest.approx(2.25, abs=1e-6)


def test_quantile_loss_all_nan_is_zero() -> None:
    """No valid target -> 0.0."""
    out = NaNTolerantQuantileLoss([0.5])(torch.tensor([1.0]), torch.tensor([NAN]))
    assert out.item() == 0.0


# ------------------------------------------------------------ isotonic + soft sort


def test_isotonic_sorted_input_is_the_identity() -> None:
    """[3, 2, 1] is already non-increasing."""
    torch.testing.assert_close(
        isotonic_l2_pav(torch.tensor([3.0, 2.0, 1.0])), torch.tensor([3.0, 2.0, 1.0])
    )


def test_isotonic_pools_a_violating_pair_to_its_mean() -> None:
    """[3, 1, 2]: 1 < 2 violates, pooled to 1.5 -> [3, 1.5, 1.5]."""
    torch.testing.assert_close(
        isotonic_l2_pav(torch.tensor([3.0, 1.0, 2.0])), torch.tensor([3.0, 1.5, 1.5])
    )


def test_isotonic_pools_by_weighted_mean() -> None:
    """Weights [1, 1, 3]: (1 * 1 + 2 * 3) / 4 = 1.75 -> [3, 1.75, 1.75]."""
    out = isotonic_l2_pav(torch.tensor([3.0, 1.0, 2.0]), torch.tensor([1.0, 1.0, 3.0]))
    torch.testing.assert_close(out, torch.tensor([3.0, 1.75, 1.75]))


def test_isotonic_merges_backwards_after_a_pool() -> None:
    """[1, 2, 3]: (1, 2) -> 1.5, then 1.5 < 3 -> (1.5 * 2 + 3) / 3 = 2 -> [2, 2, 2]."""
    torch.testing.assert_close(
        isotonic_l2_pav(torch.tensor([1.0, 2.0, 3.0])), torch.tensor([2.0, 2.0, 2.0])
    )


def test_fast_soft_sort_forward_at_strength_one() -> None:
    """Values (0.5, 1, 4), w = (3, 2, 1): w - sorted = (-1, 1, 0.5) pools to 1/6 each.

    soft = w - 1/6 = (17/6, 11/6, 5/6); the sum 5.5 equals the sum of the inputs.
    """
    out = fast_soft_sort(torch.tensor([0.5, 1.0, 4.0], dtype=torch.float64), 1.0)
    torch.testing.assert_close(
        out, torch.tensor([17 / 6, 11 / 6, 5 / 6], dtype=torch.float64)
    )
    assert out.sum().item() == pytest.approx(5.5, abs=1e-12)


def test_fast_soft_sort_strength_scales_the_weights() -> None:
    """Strength 2: w = (1.5, 1, 0.5), w - sorted = (-2.5, 0, 0) pools to -5/6.

    soft = w + 5/6 = (7/3, 11/6, 4/3).
    """
    out = fast_soft_sort(torch.tensor([0.5, 1.0, 4.0], dtype=torch.float64), 2.0)
    torch.testing.assert_close(
        out, torch.tensor([7 / 3, 11 / 6, 4 / 3], dtype=torch.float64)
    )


def test_fast_soft_sort_is_the_hard_sort_when_gaps_fit_the_weights() -> None:
    """Values (1, 1.5, 3): w - sorted = (0, 0.5, 0) pools the first two to 0.25.

    soft = (2.75, 1.75, 1.0), so the two-block partition {0, 1}, {2} is exercised.
    """
    out = fast_soft_sort(torch.tensor([1.0, 1.5, 3.0], dtype=torch.float64), 1.0)
    torch.testing.assert_close(
        out, torch.tensor([2.75, 1.75, 1.0], dtype=torch.float64)
    )


def _central_difference(values: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    """D <grad_out, soft_sort(values)> / d values by central differences (piecewise linear)."""
    eps = 1e-3
    out = torch.zeros_like(values)
    for i in range(len(values)):
        plus, minus = values.clone(), values.clone()
        plus[i] += eps
        minus[i] -= eps
        delta = fast_soft_sort(plus, 1.0) - fast_soft_sort(minus, 1.0)
        out[i] = (delta * grad_out).sum() / (2 * eps)
    return out


def test_fast_soft_sort_backward_is_the_block_average() -> None:
    """``FastSoftSort.backward`` is the Jacobian-vector product of the soft sort.

    soft = w - v with v = PAV(w - s), so d soft / d s = +P (block averaging of the
    upstream gradient, unsorted). For values (1, 1.5, 3) and grad_output (1, 2, 3):
    blocks {0, 1} average to 1.5 and {2} keeps 3; unsorting by the permutation
    (2, 1, 0) gives (3, 1.5, 1.5). The central difference agrees and
    ``torch.autograd.gradcheck`` passes. Before the fix the code returned the negation.
    """
    values = torch.tensor([1.0, 1.5, 3.0], dtype=torch.float64, requires_grad=True)
    grad_out = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    fast_soft_sort(values, 1.0).backward(grad_out)
    assert values.grad is not None
    torch.testing.assert_close(
        values.grad, torch.tensor([3.0, 1.5, 1.5], dtype=torch.float64)
    )
    numeric = _central_difference(values.detach(), grad_out)
    torch.testing.assert_close(values.grad, numeric, atol=1e-8, rtol=0)
    fresh = torch.tensor([1.0, 1.5, 3.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: fast_soft_sort(v, 1.0), (fresh,))


def test_fast_soft_sort_backward_single_block_sums_to_one_each() -> None:
    """D sum(soft) / d values is +1 per element (one block, mean of ones); before the fix -1."""
    values = torch.tensor([0.5, 1.0, 4.0], dtype=torch.float64, requires_grad=True)
    fast_soft_sort(values, 1.0).sum().backward()
    assert values.grad is not None
    torch.testing.assert_close(values.grad, torch.full((3,), 1.0, dtype=torch.float64))


# ---------------------------------------------------------------- WeightedDistLoss


def test_dist_loss_selects_the_reduction_module() -> None:
    """L2 -> MSELoss(reduction="none"), L1 -> L1Loss(reduction="none"), else ValueError."""
    l2 = WeightedDistLoss(loss_fn="L2")
    l1 = WeightedDistLoss(loss_fn="L1")
    assert type(l2.loss_fn) is nn.MSELoss and l2.loss_fn.reduction == "none"
    assert type(l1.loss_fn) is nn.L1Loss and l1.loss_fn.reduction == "none"
    torch.testing.assert_close(l2.weights, torch.tensor([1.0]))
    with pytest.raises(ValueError, match="loss_fn must be 'L1' or 'L2'"):
        WeightedDistLoss(loss_fn="huber")


def test_label_distribution_is_uniform_when_no_label_is_in_range() -> None:
    """Labels (5, 6) outside [0, 2] -> x = (0, 1, 2) with density 1/3 each."""
    density, x = WeightedDistLoss()._get_label_distribution(
        torch.tensor([5.0, 6.0]), 0.0, 2.0, step=1.0
    )
    np.testing.assert_array_equal(x, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(density, np.full(3, 1 / 3))


def test_label_distribution_is_the_normalized_gaussian_kde() -> None:
    """Labels (0, 2) with bw_method 0.5: variance (ddof 1) 2, kernel variance 0.5.

    Unnormalized density at x = (0, 1, 2) is (1 + e^-4, 2 e^-1, e^-4 + 1); divided by
    its sum 2.7723902 this is (0.3673060, 0.2653879, 0.3673060). The NaN is dropped.
    """
    density, x = WeightedDistLoss(bandwidth=0.5)._get_label_distribution(
        torch.tensor([0.0, 2.0, NAN]), 0.0, 2.0, step=1.0
    )
    e4, e1 = math.exp(-4), math.exp(-1)
    total = 2 * (1 + e4) + 2 * e1
    expected = np.array([(1 + e4) / total, 2 * e1 / total, (1 + e4) / total])
    np.testing.assert_array_equal(x, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(density, expected, atol=1e-9)
    np.testing.assert_allclose(density, [0.3673060, 0.2653879, 0.3673060], atol=1e-7)


def test_batch_label_distribution_rounds_the_scaled_density() -> None:
    """Density (0.25, 0.5, 0.25) x 8 = (2, 4, 2) sums to 8, so no residual is spread."""
    out = WeightedDistLoss()._get_batch_label_distribution(
        np.array([0.25, 0.5, 0.25]), 8
    )
    np.testing.assert_array_equal(out, np.array([2.0, 4.0, 2.0]))


def test_batch_label_distribution_spreads_the_rounding_residual() -> None:
    """Density (0.3, 0.4, 0.3) x 5 = (1.5, 2, 1.5) rounds half-to-even to (2, 2, 2).

    The residual 5 - 6 = -1 is within range_res = int(0.5 * 3) = 1, so one count is
    taken from the argmax bin (index 0): (1, 2, 2).
    """
    out = WeightedDistLoss()._get_batch_label_distribution(np.array([0.3, 0.4, 0.3]), 5)
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 2.0]))
    assert out.sum() == 5


def test_batch_label_distribution_else_branch_flips_the_sign_of_a_deficit() -> None:
    """Finding: the ``else`` branch is unreachable for a normalized density and, when
    reached, corrects a negative residual in the wrong direction.

    Per-bin rounding error is at most 0.5, so |res_sum| <= n / 2 = range_res whenever
    the density sums to 1. An unnormalized density (0.5, 0.5, 0.5) x 8 = (4, 4, 4) gives
    res_sum = 8 - 12 = -4 > range_res = 1: iters = -4 // 1 = -4 already carries the
    sign, and ``iters * np.sign(res_sum)`` = +4 is ADDED to the argmax bin: (8, 4, 4),
    sum 16, further from 8. The positive case (0.25, 0.25, 0.25) x 8 = (2, 2, 2),
    res_sum +2, adds +2 correctly: (4, 2, 2), sum 8.
    """
    loss = WeightedDistLoss()
    deficit = loss._get_batch_label_distribution(np.array([0.5, 0.5, 0.5]), 8)
    surplus = loss._get_batch_label_distribution(np.array([0.25, 0.25, 0.25]), 8)
    np.testing.assert_array_equal(deficit, np.array([8.0, 4.0, 4.0]))
    np.testing.assert_array_equal(surplus, np.array([4.0, 2.0, 2.0]))


def test_theoretical_labels_repeat_each_grid_value_by_its_count() -> None:
    """Counts (2, 4, 2) from min_label 0, step 1 -> (0, 0, 1, 1, 1, 1, 2, 2); step 0.5 from -1 halves the grid."""
    loss = WeightedDistLoss()
    density = np.array([0.25, 0.5, 0.25])
    np.testing.assert_array_equal(
        loss._get_batch_theoretical_labels(density, 8, 0.0, step=1.0),
        np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]),
    )
    np.testing.assert_array_equal(
        loss._get_batch_theoretical_labels(density, 8, -1.0, step=0.5),
        np.array([-1.0, -1.0, -0.5, -0.5, -0.5, -0.5, 0.0, 0.0]),
    )


def test_dist_loss_forward_pairs_ascending_predictions_with_ascending_labels() -> None:
    """The soft-sorted predictions are paired with the theoretical labels smallest first.

    Targets (0, 0, 2, 2) give the KDE density (0.4089717, 0.1820566, 0.4089717) on
    x = (0, 1, 2); x 4 = (1.636, 0.728, 1.636) -> counts (2, 1, 2), residual -1 taken
    from bin 0 -> (1, 1, 2) -> labels (0, 1, 2, 2). Predictions equal to the targets
    soft-sort (strength 0.1, gaps <= 10) to exactly (0, 0, 2, 2) after the flip, so
    the MSE against (0, 1, 2, 2) is (0 + 1 + 0 + 0) / 4 = 0.25, and L1 is 1 / 4 = 0.25.
    Before the fix the descending sort (2, 2, 0, 0) gave 3.25 and 1.75.
    """
    y = torch.tensor([[0.0], [0.0], [2.0], [2.0]])
    total, dims = WeightedDistLoss(bandwidth=0.5)(y, y)
    assert dims.shape == (1,)
    assert dims[0].item() == pytest.approx(0.25, abs=1e-6)
    assert total.item() == pytest.approx(0.25, abs=1e-6)
    total_l1, _ = WeightedDistLoss(bandwidth=0.5, loss_fn="L1")(y, y)
    assert total_l1.item() == pytest.approx(0.25, abs=1e-6)


def test_dist_loss_default_weight_is_uniform_and_sums_to_one() -> None:
    """The default ones(1) weight repeats to (0.5, 0.5), so two dimensions average.

    Two identical columns give (0.25 + 0.25) / 2 = 0.25. An all-NaN column contributes
    0 and keeps its weight (the weights are not renormalized over valid dimensions, as
    they are in ``WeightedSupCRCell``), so the total is 0.5 * 0.25 = 0.125.
    """
    y = torch.tensor([[0.0], [0.0], [2.0], [2.0]])
    loss = WeightedDistLoss(bandwidth=0.5)
    total, dims = loss(torch.cat([y, y], 1), torch.cat([y, y], 1))
    torch.testing.assert_close(loss.weights, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(dims, torch.tensor([0.25, 0.25]), atol=1e-6, rtol=0)
    assert total.item() == pytest.approx(0.25, abs=1e-6)
    total_nan, dims_nan = WeightedDistLoss(bandwidth=0.5)(
        torch.cat([y, y], 1), torch.cat([y, torch.full((4, 1), NAN)], 1)
    )
    torch.testing.assert_close(dims_nan, torch.tensor([0.25, 0.0]), atol=1e-6, rtol=0)
    assert total_nan.item() == pytest.approx(0.125, abs=1e-6)


def test_dist_loss_rejects_a_weight_vector_of_the_wrong_length() -> None:
    """Two weights against three output columns."""
    loss = WeightedDistLoss(weights=torch.tensor([1.0, 1.0]))
    with pytest.raises(
        ValueError, match="Weight dimensions 2 don't match output dimensions 3"
    ):
        loss(torch.zeros(4, 3), torch.zeros(4, 3))


# ------------------------------------------------------------ CombinedRegressionLoss


@pytest.mark.parametrize(
    ("loss_type", "per_dim"),
    [
        ("mse", [2.5, 5.0]),
        ("l1", [1.5, 2.0]),
        ("huber", [1.0, 1.5]),
        ("logcosh", [(LOG_COSH_1 + LOG_COSH_2) / 2, (LOG_COSH_1 + LOG_COSH_3) / 2]),
    ],
)
def test_combined_regression_total_is_the_weighted_mean(
    loss_type: str, per_dim: list[float]
) -> None:
    """Weights [1, 3] -> [0.25, 0.75]; the total is 0.25 * dim0 + 0.75 * dim1.

    For MSE that is 0.25 * 2.5 + 0.75 * 5.0 = 4.375. Before the fix the [D, 1] loss
    stack broadcast against the [D] weights to [D, D] and the weights canceled, so the
    total was the plain sum 7.5.
    """
    loss = CombinedRegressionLoss(loss_type=loss_type, weights=torch.tensor([1.0, 3.0]))
    total, dims = loss(Y_PRED, Y_TRUE)
    expected = torch.tensor(per_dim)
    torch.testing.assert_close(dims, expected, atol=1e-6, rtol=0)
    assert total.item() == pytest.approx(
        float(0.25 * expected[0] + 0.75 * expected[1]), abs=1e-6
    )


def test_combined_regression_quantile_builds_the_grid_and_averages_dimensions() -> None:
    """Spacing 0.25 -> quantiles (0.25, 0.5, 0.75); default weights [0.5, 0.5].

    Dimension 0 residuals (-1, -2): q=0.25 mean 1.125, q=0.5 mean 0.75, q=0.75 mean
    0.375, sum 2.25. Dimension 1 residuals (1, -3): 1.25 + 1.0 + 0.75 = 3.0. The total
    is the weighted mean (2.25 + 3.0) / 2 = 2.625; before the fix the plain sum 5.25.
    """
    loss = CombinedRegressionLoss(loss_type="quantile", quantile_spacing=0.25)
    assert isinstance(loss.loss_fn, NaNTolerantQuantileLoss)
    torch.testing.assert_close(loss.loss_fn.quantiles, torch.tensor([0.25, 0.5, 0.75]))
    torch.testing.assert_close(loss.weights, torch.tensor([0.5, 0.5]))
    total, dims = loss(Y_PRED, Y_TRUE)
    torch.testing.assert_close(dims, torch.tensor([2.25, 3.0]), atol=1e-6, rtol=0)
    assert total.item() == pytest.approx(2.625, abs=1e-6)


def test_combined_regression_all_nan_dimension_is_zero() -> None:
    """Column 1 NaN -> dims [2.5, 0.0], total 2.5."""
    total, dims = CombinedRegressionLoss(weights=torch.tensor([1.0, 3.0]))(
        Y_PRED, Y_TRUE_DIM1_NAN
    )
    torch.testing.assert_close(dims, torch.tensor([2.5, 0.0]))
    assert total.item() == pytest.approx(2.5, abs=1e-6)


def test_combined_regression_constructor_errors() -> None:
    """Unknown loss type and a quantile loss without spacing both raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported loss type: foo"):
        CombinedRegressionLoss(loss_type="foo")
    with pytest.raises(
        ValueError, match="quantile_spacing must be provided for quantile loss"
    ):
        CombinedRegressionLoss(loss_type="quantile")


# --------------------------------------------------------------- cross entropy


CE_LOGITS = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 2.0, 2.0]])
CE_TARGETS = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, NAN, NAN]])
CE_TASK0 = (LOG2 + math.log(1 + math.exp(-1))) / 2
CE_TASK1 = LOG2


def test_multidim_ce_means_over_valid_samples_per_task() -> None:
    """Row 0 has uniform logits, so each task costs log 2 = 0.6931472; row 1 task 0 has
    logits (1, 0) against class 0, costing log(1 + e^-1) = 0.3132617; row 1 task 1 is NaN.

    dims = [(0.6931472 + 0.3132617) / 2, 0.6931472 / 1] = [0.5032044, 0.6931472]
    """
    dims, mask = MultiDimNaNTolerantCELoss(num_classes=2, num_tasks=2)(
        CE_LOGITS, CE_TARGETS
    )
    torch.testing.assert_close(
        dims, torch.tensor([CE_TASK0, CE_TASK1]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        dims, torch.tensor([0.5032044, 0.6931472]), atol=1e-6, rtol=0
    )
    assert torch.equal(mask, torch.tensor([[True, True], [True, False]]))


def test_combined_ce_returns_weight_scaled_task_losses() -> None:
    """Finding: the second return is ``weights * dim_losses``, not the raw task losses.

    Weights [1, 3] -> [0.25, 0.75]: dims = [0.1258011, 0.5198604], total = their sum
    over a weight sum of 1 = 0.6456615.
    """
    total, dims = CombinedCELoss(
        num_classes=2, num_tasks=2, weights=torch.tensor([1.0, 3.0])
    )(CE_LOGITS, CE_TARGETS)
    expected = torch.tensor([0.25 * CE_TASK0, 0.75 * CE_TASK1])
    torch.testing.assert_close(dims, expected, atol=1e-6, rtol=0)
    assert total.item() == pytest.approx(0.25 * CE_TASK0 + 0.75 * CE_TASK1, abs=1e-6)
    assert total.item() == pytest.approx(0.6456615, abs=1e-6)


def test_combined_ce_drops_an_all_nan_task_from_the_weights() -> None:
    """Task 1 NaN everywhere: weights [0.25, 0] and the total is task 0 alone, 0.5032044."""
    targets = CE_TARGETS.clone()
    targets[:, 2:] = NAN
    total, dims = CombinedCELoss(
        num_classes=2, num_tasks=2, weights=torch.tensor([1.0, 3.0])
    )(CE_LOGITS, targets)
    torch.testing.assert_close(
        dims, torch.tensor([0.25 * CE_TASK0, 0.0]), atol=1e-6, rtol=0
    )
    assert total.item() == pytest.approx(CE_TASK0, abs=1e-6)


# ------------------------------------------------------------------- ordinal


def test_monotonic_parameter_data_is_the_cumulative_softplus() -> None:
    """Raw (0, 0): softplus(0) = log 2, cumsum -> (log 2, 2 log 2); the raw tensor is unchanged."""
    param = MonotonicParameter(torch.zeros(2))
    torch.testing.assert_close(param.data, torch.tensor([LOG2, 2 * LOG2]))
    torch.testing.assert_close(param.detach(), torch.zeros(2))
    assert param.requires_grad is True
    assert (
        MonotonicParameter(torch.zeros(2), requires_grad=False).requires_grad is False
    )


def test_monotonic_parameter_data_is_read_only() -> None:
    """The property has no setter; the message is CPython's 3.11+ wording (env: 3.13)."""
    param = MonotonicParameter(torch.zeros(2))
    with pytest.raises(
        AttributeError,
        match="property 'data' of 'MonotonicParameter' object has no setter",
    ):
        param.data = torch.ones(2)  # type: ignore[misc]


def _zero_raw_thresholds(loss: MultiDimNaNTolerantOrdinalCELoss) -> None:
    with torch.no_grad():
        loss.raw_thresholds.zero_()


def test_ordinal_thresholds_are_cumulative_softplus_per_task() -> None:
    """Raw zeros for 3 classes -> (log 2, 2 log 2) on each of the 2 task rows."""
    loss = MultiDimNaNTolerantOrdinalCELoss(num_classes=3, num_tasks=2)
    assert loss.raw_thresholds.shape == (2, 2)
    _zero_raw_thresholds(loss)
    torch.testing.assert_close(
        loss.thresholds, torch.tensor([[LOG2, 2 * LOG2], [LOG2, 2 * LOG2]])
    )


ORD_LOGITS = torch.tensor([[LOG2, LOG2 + 1.0], [0.0, LOG2]])
ORD_TARGETS = torch.tensor([[0.0, 1.0, 1.0, 0.0], [NAN, NAN, 0.0, 1.0]])
ORD_TASK0 = LOG2
ORD_TASK1 = (math.log(1 + math.e) + LOG2) / 2


def test_ordinal_loss_one_threshold_case() -> None:
    """Two classes, one threshold log 2 per task (raw zeros).

    Row 0 task 0: logit log 2 -> p = sigmoid(0) = 0.5, target class 1 -> binary 1,
    BCE = log 2. Row 0 task 1: logit log 2 + 1 -> p = sigmoid(1), target class 0 ->
    binary 0, BCE = -log(1 - sigmoid(1)) = log(1 + e) = 1.3132617. Row 1 task 0 is NaN;
    row 1 task 1: logit log 2, class 1 -> log 2.
    dims = [log 2, (1.3132617 + 0.6931472) / 2] = [0.6931472, 1.0032044].
    """
    loss = MultiDimNaNTolerantOrdinalCELoss(num_classes=2, num_tasks=2)
    _zero_raw_thresholds(loss)
    dims, valid = loss(ORD_LOGITS, ORD_TARGETS)
    torch.testing.assert_close(
        dims, torch.tensor([ORD_TASK0, ORD_TASK1]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        dims, torch.tensor([0.6931472, 1.0032044]), atol=1e-6, rtol=0
    )
    assert torch.equal(valid, torch.tensor([True, True]))


def test_ordinal_loss_all_nan_task_is_zero_and_invalid() -> None:
    """Task 0 NaN in every row -> dims [0, 1.0032044], validity [False, True]."""
    loss = MultiDimNaNTolerantOrdinalCELoss(num_classes=2, num_tasks=2)
    _zero_raw_thresholds(loss)
    targets = ORD_TARGETS.clone()
    targets[:, :2] = NAN
    dims, valid = loss(ORD_LOGITS, targets)
    torch.testing.assert_close(dims, torch.tensor([0.0, ORD_TASK1]), atol=1e-6, rtol=0)
    assert torch.equal(valid, torch.tensor([False, True]))


def test_combined_ordinal_returns_weight_scaled_task_losses() -> None:
    """Finding (as CombinedCELoss): dims = weights * task losses.

    [1, 3] -> [0.25, 0.75]: dims = [0.1732868, 0.7524034], total 0.9256901; with task
    0 NaN the weights become [0.25, 0] and the total is task 1 alone, 1.0032044.
    """
    loss = CombinedOrdinalCELoss(
        num_classes=2, num_tasks=2, weights=torch.tensor([1.0, 3.0])
    )
    _zero_raw_thresholds(loss.loss_fn)
    total, dims = loss(ORD_LOGITS, ORD_TARGETS)
    torch.testing.assert_close(
        dims, torch.tensor([0.25 * ORD_TASK0, 0.75 * ORD_TASK1]), atol=1e-6, rtol=0
    )
    assert total.item() == pytest.approx(0.25 * ORD_TASK0 + 0.75 * ORD_TASK1, abs=1e-6)
    assert total.item() == pytest.approx(0.9256901, abs=1e-6)
    targets = ORD_TARGETS.clone()
    targets[:, :2] = NAN
    total_nan, dims_nan = loss(ORD_LOGITS, targets)
    torch.testing.assert_close(
        dims_nan, torch.tensor([0.0, 0.75 * ORD_TASK1]), atol=1e-6, rtol=0
    )
    assert total_nan.item() == pytest.approx(ORD_TASK1, abs=1e-6)


# ---------------------------------------------------------- entropy regularization


SOFT_TARGETS = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
UNIT_FEATURES = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
HALF_LOG3 = 0.5 * math.log(3)


def test_pairwise_distances_are_squared_euclidean() -> None:
    """(0,0), (3,4), (1,0): 25, 1, 20 off the diagonal."""
    out = CategoricalEntropyRegLoss(num_classes=2).compute_pairwise_distances(
        torch.tensor([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    )
    torch.testing.assert_close(
        out, torch.tensor([[0.0, 25.0, 1.0], [25.0, 0.0, 20.0], [1.0, 20.0, 0.0]])
    )


def test_target_distances_are_symmetric_kl() -> None:
    """P = (0.75, 0.25), q = (0.25, 0.75): KL(p||q) = 0.75 log 3 - 0.25 log 3 = 0.5 log 3
    = 0.5493061 in both directions, so the symmetric average is 0.5 log 3.
    """
    out = CategoricalEntropyRegLoss(num_classes=2).compute_target_distances(
        SOFT_TARGETS
    )
    torch.testing.assert_close(
        out, torch.tensor([[0.0, HALF_LOG3], [HALF_LOG3, 0.0]]), atol=1e-6, rtol=0
    )


def test_centers_are_probability_weighted_feature_means() -> None:
    """Class 0 weights (0.75, 0.25) -> center (0.75, 0.25); class 1 -> (0.25, 0.75)."""
    centers = CategoricalEntropyRegLoss(num_classes=2).compute_centers(
        UNIT_FEATURES, SOFT_TARGETS
    )
    assert list(centers.keys()) == [0]
    assert sorted(centers[0].keys()) == [0, 1]
    torch.testing.assert_close(centers[0][0], torch.tensor([0.75, 0.25]))
    torch.testing.assert_close(centers[0][1], torch.tensor([0.25, 0.75]))


def test_entropy_reg_forward_diversity_and_tightness() -> None:
    """Unit features have squared distance 2; target distance 0.5 log 3 both ways.

    diversity = -(2 * 0.5 log 3 * 2) / (2 * 1) = -log 3 = -1.0986123.
    tightness per sample: 0.75 * |(1,0) - (0.75,0.25)|^2 + 0.25 * |(1,0) - (0.25,0.75)|^2
    = 0.75 * 0.125 + 0.25 * 1.125 = 0.375 (the other sample is symmetric), mean 0.375.
    total = 0.1 * (-1.0986123) + 0.1 * 0.375 = -0.0723612.
    """
    total, diversity, tightness = CategoricalEntropyRegLoss(
        lambda_d=0.1, lambda_t=0.1, num_classes=2
    )(UNIT_FEATURES, SOFT_TARGETS, torch.tensor([True, True]))
    assert diversity.item() == pytest.approx(-math.log(3), abs=1e-6)
    assert tightness.item() == pytest.approx(0.375, abs=1e-6)
    assert total.item() == pytest.approx(-0.1 * math.log(3) + 0.0375, abs=1e-6)
    assert total.item() == pytest.approx(-0.0723612, abs=1e-6)


def test_entropy_reg_single_valid_sample_is_zero() -> None:
    """One valid sample: no pairs (diversity 0) and each center is the sample itself (tightness 0)."""
    total, diversity, tightness = CategoricalEntropyRegLoss(num_classes=2)(
        UNIT_FEATURES, SOFT_TARGETS, torch.tensor([True, False])
    )
    assert (total.item(), diversity.item(), tightness.item()) == (0.0, 0.0, 0.0)


def test_mse_entropy_reg_dim_losses_are_a_batch_by_task_matrix() -> None:
    """Finding: ``weights * valid_mask`` uses the [batch, tasks] NaN mask, so
    ``dim_losses`` is [batch, tasks] and the MSE total is weighted by valid COUNTS.

    pred [[1, 2], [3, 4]], target [[0, nan], [1, 5]]: MSE dims [(1 + 4) / 2, 1] =
    [2.5, 1.0]; default weights [0.5, 0.5] times the mask [[1, 0], [1, 1]] give
    [[0.5, 0], [0.5, 0.5]] (sum 1.5); dim_losses = [[1.25, 0], [1.25, 0.5]] (sum 3.0);
    mse_total = 3.0 / 1.5 = 2.0. The documented per-dimension weighted mean would be
    0.5 * 2.5 + 0.5 * 1.0 = 1.75. Row 0 has a NaN so only row 1 reaches the entropy
    term, which is 0 for a single sample; total = 2.0.
    """
    loss = MseCategoricalEntropyRegLoss(num_classes=2, num_tasks=2)
    total, parts = loss(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[0.0, NAN], [1.0, 5.0]]),
        torch.zeros(2, 2),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        UNIT_FEATURES,
    )
    assert sorted(parts) == [
        "dim_losses",
        "diversity_loss",
        "entropy_total",
        "mse_total",
        "tightness_loss",
        "total_loss",
    ]
    torch.testing.assert_close(
        parts["dim_losses"], torch.tensor([[1.25, 0.0], [1.25, 0.5]])
    )
    assert parts["mse_total"].item() == pytest.approx(2.0, abs=1e-6)
    assert parts["entropy_total"].item() == 0.0
    assert parts["diversity_loss"].item() == 0.0
    assert parts["tightness_loss"].item() == 0.0
    assert total.item() == pytest.approx(2.0, abs=1e-6)
    assert parts["total_loss"] is total
