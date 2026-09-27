# tests/torchcell/losses/test_mle_wasserstein_buffers.py
# [[tests.torchcell.losses.test_mle_wasserstein_buffers]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_mle_wasserstein_buffers.py
"""Ring buffers and the composite ``MleWassSupCR`` on CPU with exact values.

Ring buffer of size 4: writing rows (1, 2, 3) then (4, 5) fills slots 0..2, then slot 3
takes 4 and the wrap puts 5 in slot 0, leaving (5, 2, 3, 4) with pointer (3 + 2) mod 4
= 1, total min(5, 4) = 4, and the full flag set.

Wasserstein values use the translation identity from ``test_mle_wasserstein.py``:
debiased Sinkhorn at p = 2 on a cloud shifted by 1 is 1^2 / 2 = 0.5 per dimension,
whatever the cloud, so mixing buffer rows into the batch cannot change it.

SupCR values come from ``_naive_supcr``, the per-anchor formula with no sort or
cumulative sum, evaluated on the exact row set the buffered loss concatenates (the
batch followed by the buffer, which after the first call is the same batch again).

The composite with ``use_buffer=False`` on predictions = targets + 1 (16 rows, 2
dims) and 4-dim embeddings: mse 1.0 per dimension, Wasserstein 0.5 per dimension,
SupCR at the constructor temperature 0.1; total = 1.0 * 1.0 + 0.1 * 0.5 + 0.001 * S.
"""

import math

import pytest
import torch

from torchcell.losses.mle_wasserstein import (
    BufferedWeightedSupCRCell,
    BufferedWeightedWassersteinLoss,
    MleWassSupCR,
)

NAN = float("nan")
EMB = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
LABELS = torch.tensor([[0.0], [1.0], [3.0]])


def _naive_supcr(emb: torch.Tensor, labels: torch.Tensor, temp: float) -> float:
    """SupCR by the per-anchor formula, with the code's tie rule (k = i when d_ij = 0)."""
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
                if dist[i, k] >= dist[i, j]
            )
            total += -math.log(math.exp(sims[i, j].item()) / denominator)
    return total / (m * (m - 1))


def _clouds(n: int, shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    targets = torch.randn(n, 2)
    return targets + shift, targets


# --------------------------------------------------- BufferedWeightedWassersteinLoss


def test_wasserstein_buffer_fills_then_wraps() -> None:
    """(1, 2, 3) then (4, 5) into 4 slots -> (5, 2, 3, 4), pointer 1, total 4, full."""
    loss = BufferedWeightedWassersteinLoss(buffer_size=4, min_samples=2)
    assert loss.pred_buffer.shape == (4, 1) and loss.target_buffer.shape == (4, 1)
    loss.update_buffer(
        torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([[10.0], [20.0], [30.0]])
    )
    torch.testing.assert_close(
        loss.pred_buffer[:, 0], torch.tensor([1.0, 2.0, 3.0, 0.0])
    )
    assert int(loss.buffer_ptr) == 3 and int(loss.total_samples) == 3
    assert bool(loss.buffer_full) is False
    preds, targets = loss.get_buffer_samples()
    torch.testing.assert_close(preds[:, 0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(targets[:, 0], torch.tensor([10.0, 20.0, 30.0]))

    loss.update_buffer(torch.tensor([[4.0], [5.0]]), torch.tensor([[40.0], [50.0]]))
    torch.testing.assert_close(
        loss.pred_buffer[:, 0], torch.tensor([5.0, 2.0, 3.0, 4.0])
    )
    torch.testing.assert_close(
        loss.target_buffer[:, 0], torch.tensor([50.0, 20.0, 30.0, 40.0])
    )
    assert int(loss.buffer_ptr) == 1 and int(loss.total_samples) == 4
    assert bool(loss.buffer_full) is True
    preds, targets = loss.get_buffer_samples()
    torch.testing.assert_close(preds[:, 0], torch.tensor([5.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(targets[:, 0], torch.tensor([50.0, 20.0, 30.0, 40.0]))


def test_wasserstein_buffer_defaults_to_one_column_and_rejects_two() -> None:
    """Finding: ``weights=None`` sizes the buffer at one column, so 2-dim targets fail.

    The composite passes ``weights=None`` by default, so its buffered path raises on
    the very first update with two-dimensional targets; ``weights`` of length 2 sizes
    the buffer correctly.
    """
    loss = BufferedWeightedWassersteinLoss(buffer_size=16, min_samples=8)
    with pytest.raises(
        RuntimeError, match=r"Target sizes: \[4, 1\].  Tensor sizes: \[4, 2\]"
    ):
        loss.update_buffer(torch.zeros(4, 2), torch.zeros(4, 2))
    sized = BufferedWeightedWassersteinLoss(
        buffer_size=16, min_samples=8, weights=torch.tensor([1.0, 1.0])
    )
    assert sized.pred_buffer.shape == (16, 2)


def test_wasserstein_forward_below_min_samples_returns_zeros_of_length_two() -> None:
    """Finding: the early return is ``zeros(2)`` regardless of the buffer's one column.

    The row is still written to the buffer (total 1).
    """
    loss = BufferedWeightedWassersteinLoss(buffer_size=4, min_samples=2)
    total, dims = loss(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    assert total.item() == 0.0
    torch.testing.assert_close(dims, torch.zeros(2))
    assert int(loss.total_samples) == 1


def test_wasserstein_forward_with_full_buffer_weight_uses_the_whole_buffer() -> None:
    """8 rows shifted by 1 concatenated with the same 8 buffered rows: 0.5."""
    torch.manual_seed(0)
    targets = torch.randn(8, 1)
    loss = BufferedWeightedWassersteinLoss(buffer_size=16, min_samples=8)
    total, dims = loss(targets + 1.0, targets)
    assert int(loss.total_samples) == 8
    assert dims.shape == (1,)
    assert dims[0].item() == pytest.approx(0.5, abs=1e-5)
    assert total.item() == pytest.approx(0.5, abs=1e-5)


def test_wasserstein_forward_with_partial_buffer_weight_samples_the_buffer() -> None:
    """buffer_weight 0.5 draws int(8 * 0.5 / 0.5) = 8 buffer rows; the shift stays 1 -> 0.5."""
    torch.manual_seed(0)
    targets = torch.randn(8, 1)
    loss = BufferedWeightedWassersteinLoss(buffer_size=16, min_samples=8)
    total, dims = loss(targets + 1.0, targets, buffer_weight=0.5)
    assert dims[0].item() == pytest.approx(0.5, abs=1e-5)
    assert total.item() == pytest.approx(0.5, abs=1e-5)


def test_wasserstein_forward_with_no_buffer_draw_falls_back_to_the_batch() -> None:
    """One row at buffer_weight 0.4: int(1 * 0.4 / 0.6) = 0 buffer rows, a single-row
    dimension has fewer than 2 samples, so the Wasserstein term is 0.
    """
    loss = BufferedWeightedWassersteinLoss(buffer_size=4, min_samples=1)
    total, dims = loss(torch.tensor([[1.0]]), torch.tensor([[0.0]]), buffer_weight=0.4)
    assert total.item() == 0.0
    torch.testing.assert_close(dims, torch.zeros(1))


def test_wasserstein_forward_prefers_the_gathered_tensors() -> None:
    """With ``all_*`` given, the buffer takes the 8 gathered rows, not the 4 local ones."""
    torch.manual_seed(0)
    targets = torch.randn(8, 1)
    loss = BufferedWeightedWassersteinLoss(buffer_size=16, min_samples=8)
    total, dims = loss(targets[:4] + 1.0, targets[:4], targets + 1.0, targets)
    assert int(loss.total_samples) == 8
    torch.testing.assert_close(loss.pred_buffer[:8], targets + 1.0)
    assert total.item() == pytest.approx(0.5, abs=1e-5)


# ------------------------------------------------------- BufferedWeightedSupCRCell


def test_supcr_buffer_wraps_embeddings_and_labels_together() -> None:
    """3 rows then 2 into 4 slots: slot 3 takes the first new row, slot 0 the second."""
    loss = BufferedWeightedSupCRCell(buffer_size=4, embedding_dim=2, min_samples=2)
    assert loss.embedding_buffer.shape == (4, 2) and loss.label_buffer.shape == (4, 1)
    loss.update_buffer(EMB, LABELS)
    loss.update_buffer(EMB[:2] * 10, LABELS[:2] * 10)
    torch.testing.assert_close(
        loss.embedding_buffer,
        torch.tensor([[0.0, 10.0], [0.0, 1.0], [1.0, 1.0], [10.0, 0.0]]),
    )
    torch.testing.assert_close(
        loss.label_buffer[:, 0], torch.tensor([10.0, 1.0, 3.0, 0.0])
    )
    assert int(loss.buffer_ptr) == 1 and int(loss.total_samples) == 4
    assert bool(loss.buffer_full) is True


def test_supcr_forward_below_min_samples_returns_zeros_of_length_two() -> None:
    """3 rows against min_samples 4 -> (0.0, zeros(2))."""
    loss = BufferedWeightedSupCRCell(buffer_size=4, embedding_dim=2, min_samples=4)
    total, dims = loss(EMB, LABELS)
    assert total.item() == 0.0
    torch.testing.assert_close(dims, torch.zeros(2))


def test_supcr_forward_halves_the_loss_at_full_buffer_weight() -> None:
    """Finding: the scale is 1 - w + 0.5 w, so the default buffer_weight 1.0 halves the loss.

    The batch is concatenated with the buffer that now holds the same 3 rows, so the
    SupCR is evaluated on 6 rows with duplicated labels: 1.2417051 by the naive formula
    (ties include the anchor's self term). Returned: 0.5 * 1.2417051 = 0.6208526; the
    per-dimension value is the unscaled 1.2417051. The default weights buffer is
    ones(2) / 2 and broadcasts over the single label column.
    """
    loss = BufferedWeightedSupCRCell(
        buffer_size=4, embedding_dim=2, temperature=1.0, min_samples=2
    )
    torch.testing.assert_close(loss.base_supcr.weights, torch.tensor([0.5, 0.5]))
    total, dims = loss(EMB, LABELS)
    reference = _naive_supcr(
        torch.cat([EMB, EMB]), torch.cat([LABELS, LABELS])[:, 0], 1.0
    )
    assert reference == pytest.approx(1.2417051, abs=1e-6)
    assert dims.shape == (1,)
    assert dims[0].item() == pytest.approx(reference, abs=1e-6)
    assert total.item() == pytest.approx(0.5 * reference, abs=1e-6)


def test_supcr_forward_applies_the_temperature_override_and_zero_buffer_weight() -> (
    None
):
    """Temperature 0.5 is written into the inner SupCR; buffer_weight 0 leaves the scale 1."""
    loss = BufferedWeightedSupCRCell(
        buffer_size=4, embedding_dim=2, temperature=1.0, min_samples=2
    )
    total, dims = loss(EMB, LABELS, buffer_weight=0.0, temperature=0.5)
    assert loss.base_supcr.supcr.temperature == 0.5
    reference = _naive_supcr(
        torch.cat([EMB, EMB]), torch.cat([LABELS, LABELS])[:, 0], 0.5
    )
    assert reference == pytest.approx(1.3407291, abs=1e-6)
    assert total.item() == pytest.approx(reference, abs=1e-6)
    assert dims[0].item() == pytest.approx(reference, abs=1e-6)


# ------------------------------------------------------------------ MleWassSupCR


def test_composite_resolves_schedule_defaults_from_max_epochs() -> None:
    """Warmup = int(1000 * 0.1) = 100, stable = int(1000 * 0.5) = 500; counters start at 0."""
    loss = MleWassSupCR(use_buffer=False)
    assert loss.adaptive_weighting.warmup_epochs == 100
    assert loss.adaptive_weighting.stable_epoch == 500
    assert int(loss.forward_count) == 0 and int(loss.current_epoch) == 0
    explicit = MleWassSupCR(use_buffer=False, warmup_epochs=7, stable_epoch=9)
    assert (
        explicit.adaptive_weighting.warmup_epochs,
        explicit.adaptive_weighting.stable_epoch,
    ) == (7, 9)
    with pytest.raises(AttributeError, match="has no attribute 'adaptive_weighting'"):
        MleWassSupCR(use_buffer=False, use_adaptive_weighting=False).adaptive_weighting


def test_gather_is_the_identity_without_a_process_group() -> None:
    """No ``torch.distributed`` init -> the same tensor object comes back."""
    tensor = torch.zeros(3, 2)
    assert MleWassSupCR(use_buffer=False).gather_across_gpus(tensor) is tensor


def test_composite_without_buffer_combines_the_three_terms() -> None:
    """Finding: the scheduled temperature is logged but not applied without a buffer.

    mse 1.0 (shift 1 in both dimensions), Wasserstein 0.5 per dimension (mean 0.5),
    SupCR S = mean over the two label columns of the naive value at the CONSTRUCTOR
    temperature 0.1 (5.9496104), not at the scheduled 1.0 (1.9593947) that the dict
    reports. total = 1.0 + 0.05 + 0.001 * S = 1.0559496; the normalized entries are
    each weighted term over the total, and the unweighted ones over 1 + 0.5 + S.
    """
    predictions, targets = _clouds(16, 1.0)
    torch.manual_seed(1)
    z = torch.randn(16, 4)
    loss = MleWassSupCR(use_buffer=False, supcr_temperature=0.1)
    total, parts = loss(predictions, targets, z, epoch=0)

    assert sorted(parts) == [
        "mse_dim_losses",
        "mse_loss",
        "norm_unweighted_dist",
        "norm_unweighted_mse",
        "norm_unweighted_supcr",
        "norm_unweighted_wasserstein",
        "norm_weighted_mse",
        "norm_weighted_supcr",
        "norm_weighted_wasserstein",
        "supcr_dim_losses",
        "supcr_loss",
        "temperature",
        "total_loss",
        "total_weighted",
        "wasserstein_dim_losses",
        "wasserstein_loss",
        "weighted_mse",
        "weighted_supcr",
        "weighted_wasserstein",
    ]
    s0 = _naive_supcr(z, targets[:, 0], 0.1)
    s1 = _naive_supcr(z, targets[:, 1], 0.1)
    s_mean = 0.5 * (s0 + s1)
    s_scheduled = 0.5 * (
        _naive_supcr(z, targets[:, 0], 1.0) + _naive_supcr(z, targets[:, 1], 1.0)
    )
    assert parts["temperature"] == 1.0
    assert parts["mse_loss"] == 1.0 and parts["weighted_mse"] == 1.0
    torch.testing.assert_close(parts["mse_dim_losses"], torch.tensor([1.0, 1.0]))
    assert parts["wasserstein_loss"] == pytest.approx(0.5, abs=1e-5)
    assert parts["weighted_wasserstein"] == pytest.approx(0.05, abs=1e-6)
    torch.testing.assert_close(
        parts["wasserstein_dim_losses"], torch.tensor([0.5, 0.5]), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        parts["supcr_dim_losses"], torch.tensor([s0, s1]), atol=0, rtol=1e-5
    )
    assert parts["supcr_loss"] == pytest.approx(s_mean, rel=1e-5)
    assert parts["supcr_loss"] != pytest.approx(s_scheduled, rel=1e-2)
    assert parts["weighted_supcr"] == pytest.approx(0.001 * s_mean, rel=1e-5)
    expected_total = 1.0 + 0.05 + 0.001 * s_mean
    assert total.item() == pytest.approx(expected_total, rel=1e-6)
    assert (
        parts["total_weighted"] == total.item() and parts["total_loss"] == total.item()
    )
    assert parts["norm_weighted_mse"] == pytest.approx(1.0 / expected_total, rel=1e-6)
    assert parts["norm_weighted_wasserstein"] == pytest.approx(
        0.05 / expected_total, rel=1e-5
    )
    assert parts["norm_weighted_supcr"] == pytest.approx(
        0.001 * s_mean / expected_total, rel=1e-5
    )
    unweighted = 1.0 + 0.5 + s_mean
    assert parts["norm_unweighted_mse"] == pytest.approx(1.0 / unweighted, rel=1e-5)
    assert parts["norm_unweighted_wasserstein"] == pytest.approx(
        0.5 / unweighted, rel=1e-5
    )
    assert parts["norm_unweighted_dist"] == parts["norm_unweighted_wasserstein"]
    assert parts["norm_unweighted_supcr"] == pytest.approx(
        s_mean / unweighted, rel=1e-5
    )
    assert int(loss.forward_count) == 1 and int(loss.current_epoch) == 0


def test_composite_with_all_lambdas_zero_reports_eleven_zero_entries() -> None:
    """No term computed, so no normalized entries and no schedule keys appear."""
    predictions, targets = _clouds(16, 1.0)
    loss = MleWassSupCR(
        lambda_mse=0.0,
        lambda_wasserstein=0.0,
        lambda_supcr=0.0,
        use_buffer=False,
        use_adaptive_weighting=False,
        use_temp_scheduling=False,
    )
    total, parts = loss(predictions, targets, torch.zeros(16, 4))
    assert total.item() == 0.0
    assert sorted(parts) == [
        "mse_dim_losses",
        "mse_loss",
        "supcr_dim_losses",
        "supcr_loss",
        "total_loss",
        "total_weighted",
        "wasserstein_dim_losses",
        "wasserstein_loss",
        "weighted_mse",
        "weighted_supcr",
        "weighted_wasserstein",
    ]
    assert (parts["mse_loss"], parts["wasserstein_loss"], parts["supcr_loss"]) == (
        0.0,
        0.0,
        0.0,
    )
    assert (
        parts["weighted_mse"],
        parts["weighted_wasserstein"],
        parts["weighted_supcr"],
    ) == (0.0, 0.0, 0.0)
    for key in ("mse_dim_losses", "wasserstein_dim_losses", "supcr_dim_losses"):
        torch.testing.assert_close(parts[key], torch.zeros(2))
    assert parts["total_weighted"] == 0.0 and parts["total_loss"] == 0.0


def test_composite_buffered_path_at_epoch_zero() -> None:
    """Epoch 0: buffer_weight 0.1 and temperature 1.0 are both logged and applied.

    Weights [1, 1]: the MSE weights normalize to [0.5, 0.5] (mse 1.0); the Wasserstein
    weights stay [1, 1] so its total is the SUM 0.5 + 0.5 = 1.0 (the batch plus
    int(16 * 0.1 / 0.9) = 1 buffered row is still a shift by 1). SupCR runs at the
    scheduled temperature 1.0 on the batch concatenated with its own buffered copy
    (32 rows) and is scaled by 1 - 0.1 + 0.05 = 0.95: 0.95 * 2.6530887 = 2.5204343.
    total = 1.0 + 0.1 * 1.0 + 0.001 * 2.5204343 = 1.1025205.
    """
    predictions, targets = _clouds(16, 1.0)
    torch.manual_seed(1)
    z = torch.randn(16, 4)
    loss = MleWassSupCR(
        use_buffer=True,
        buffer_size=64,
        min_samples_for_wasserstein=16,
        min_samples_for_supcr=16,
        embedding_dim=4,
        weights=torch.tensor([1.0, 1.0]),
        warmup_epochs=100,
        stable_epoch=500,
        supcr_temperature=0.1,
    )
    total, parts = loss(predictions, targets, z, epoch=0)
    doubled_z = torch.cat([z, z])
    doubled_targets = torch.cat([targets, targets])
    s_ref = 0.5 * (
        _naive_supcr(doubled_z, doubled_targets[:, 0], 1.0)
        + _naive_supcr(doubled_z, doubled_targets[:, 1], 1.0)
    )
    assert parts["buffer_weight"] == pytest.approx(0.1, abs=1e-12)
    assert parts["temperature"] == 1.0
    assert parts["mse_loss"] == 1.0
    assert parts["wasserstein_loss"] == pytest.approx(1.0, abs=1e-5)
    torch.testing.assert_close(
        parts["wasserstein_dim_losses"], torch.tensor([0.5, 0.5]), atol=1e-5, rtol=0
    )
    assert parts["supcr_loss"] == pytest.approx(0.95 * s_ref, rel=1e-5)
    assert total.item() == pytest.approx(1.0 + 0.1 + 0.001 * 0.95 * s_ref, rel=1e-6)
    assert isinstance(loss.wasserstein_loss, BufferedWeightedWassersteinLoss)
    assert isinstance(loss.supcr_loss, BufferedWeightedSupCRCell)
    assert int(loss.wasserstein_loss.total_samples) == 16
    assert int(loss.supcr_loss.total_samples) == 16
    assert loss.supcr_loss.base_supcr.supcr.temperature == 1.0


def test_composite_buffered_default_weights_fail_on_two_dimensional_targets() -> None:
    """Finding: ``MleWassSupCR(use_buffer=True)`` with the default ``weights=None`` sizes
    its Wasserstein buffer at one column and raises on the first 2-dim forward.
    """
    predictions, targets = _clouds(16, 1.0)
    loss = MleWassSupCR(use_buffer=True)
    with pytest.raises(
        RuntimeError, match=r"Target sizes: \[16, 1\].  Tensor sizes: \[16, 2\]"
    ):
        loss(predictions, targets, torch.zeros(16, 128))
