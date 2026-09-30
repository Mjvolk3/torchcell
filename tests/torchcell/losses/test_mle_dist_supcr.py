# tests/torchcell/losses/test_mle_dist_supcr.py
# [[tests.torchcell.losses.test_mle_dist_supcr]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_mle_dist_supcr.py
"""Tests for the MLE distance SupCR loss and its weighting components.

2026.09.30 (Phase 14): every term in closed form on one hand-built three-sample case,
targets t = [0, 1, 2] (one dimension), predictions p = [2, 0.5, 1], embeddings
z = [(1, 0), (0, 1), (-1, 0)]; each number was checked with a two-line Python run.

* MSE (``WeightedMSELoss``, pairwise): ((2 - 0)^2 + (0.5 - 1)^2 + (1 - 2)^2) / 3
  = 5.25 / 3 = 1.75.
* Distribution term (``WeightedDistLoss``, bandwidth 0.5): the KDE of [0, 1, 2] with
  kernel sd 0.5 * std(ddof 1) = 0.5, evaluated at x = [0, 1, 2], is proportional to
  [1 + e^-2 + e^-8, 1 + 2 e^-2, 1 + e^-2 + e^-8]; times the batch size 3 that is
  [0.962, 1.076, 0.962], whose forward and backward cumulative sums first reach 1 at the
  middle bin, so the batch label distribution is [0, 2, 0] plus the residual 1 on the
  maximum bin: every theoretical label is 1. The soft sort equals the hard sort when
  every adjacent gap is at most 1 / 0.1 = 10, so the term is
  ((0.5 - 1)^2 + (1 - 1)^2 + (2 - 1)^2) / 3 = 1.25 / 3 = 0.41667. It ignores the
  pairing, which is what separates it from the MSE. On the batch doubled to six the
  same arithmetic gives the labels [0, 0, 1, 1, 2, 2] and (2 * 0.25) / 6 = 0.08333.
* SupCR at temperature T on z (cosines s01 = 0, s02 = -1, s12 = 0): anchors 0 and 2
  each contribute log(1 + e^(-1/T)) (the far positive has only itself in the
  denominator, a zero), anchor 1 contributes log 2 twice (a tie at distance 1), so
  the mean over the six ordered pairs is (log(1 + e^(-1/T)) + log 2) / 3: 0.3354696 at
  T = 1 and 0.2733584 at T = 0.5.
* Total with lambdas (1, 0.1, 0.001) at T = 1: 1.75 + 0.041667 + 0.00033547 =
  1.7920021. The gradient of the total on predictions t + b is
  2b * 1 + 0.1 * 2b = 2.2b (the SupCR term does not see b): +1.1 at b = 0.5 and
  -1.1 at b = -0.5.
* Schedules: buffer weight 0.1 + 0.2 e / w in warmup, 0.3 + 0.6 / (1 + e^(-10(q - 0.5)))
  in transition (q the progress), 0.9 after; temperature init (final / init)^(e / E)
  exponential and final + (init - final)(1 + cos(pi e / E)) / 2 cosine.
"""

import math
from typing import Any

import pytest
import torch

from torchcell.losses import mle_dist_supcr as mds
from torchcell.losses.mle_dist_supcr import (
    AdaptiveWeighting,
    BufferedWeightedDistLoss,
    BufferedWeightedSupCRCell,
    MleDistSupCR,
    TemperatureScheduler,
)
from torchcell.losses.multi_dim_nan_tolerant import WeightedDistLoss, WeightedSupCRCell

TARGETS = torch.tensor([[0.0], [1.0], [2.0]])
PREDICTIONS = torch.tensor([[2.0], [0.5], [1.0]])
EMBEDDINGS = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
MSE = 5.25 / 3
DIST = 1.25 / 3
DIST_DOUBLED = 0.5 / 6


def _supcr(temperature: float) -> float:
    return (math.log(1 + math.exp(-1 / temperature)) + math.log(2)) / 3


def _plain(**overrides: Any) -> MleDistSupCR:
    """Unbuffered, no gathering, no schedules, SupCR at T = 1, bandwidth 0.5."""
    config: dict[str, Any] = {
        "lambda_mse": 1.0,
        "lambda_dist": 0.1,
        "lambda_supcr": 0.001,
        "dist_bandwidth": 0.5,
        "supcr_temperature": 1.0,
        "use_buffer": False,
        "use_ddp_gather": False,
        "use_adaptive_weighting": False,
        "use_temp_scheduling": False,
    }
    config.update(overrides)
    return MleDistSupCR(**config)


def test_adaptive_weighting() -> None:
    """Warmup w = 100, stable 500: 0.1 + 0.2 e / 100 below 100; the sigmoid
    0.3 + 0.6 / (1 + exp(-10 (q - 0.5))) with q = (e - 100) / 400 up to 500; 0.9 after.
    """
    aw = AdaptiveWeighting(warmup_epochs=100, stable_epoch=500)
    assert aw.get_buffer_weight(0) == pytest.approx(0.1, abs=1e-15)
    assert aw.get_buffer_weight(50) == pytest.approx(0.2, abs=1e-15)
    # epoch 100 is already the transition: q = 0, 0.3 + 0.6 / (1 + e^5)
    assert aw.get_buffer_weight(100) == pytest.approx(
        0.3 + 0.6 / (1 + math.exp(5)), abs=1e-15
    )
    assert aw.get_buffer_weight(200) == pytest.approx(
        0.3 + 0.6 / (1 + math.exp(2.5)), abs=1e-15
    )
    assert aw.get_buffer_weight(300) == pytest.approx(0.6, abs=1e-15)
    assert aw.get_buffer_weight(400) == pytest.approx(
        0.3 + 0.6 / (1 + math.exp(-2.5)), abs=1e-15
    )
    assert aw.get_buffer_weight(500) == 0.9
    assert aw.get_buffer_weight(1000) == 0.9


def test_temperature_scheduler() -> None:
    """Exponential 1 * 0.1^(e / 1000); cosine 0.1 + 0.45 (1 + cos(pi e / 1000))."""
    ts_exp = TemperatureScheduler(init_temp=1.0, final_temp=0.1, schedule="exponential")
    assert ts_exp.get_temperature(0, 1000) == 1.0
    assert ts_exp.get_temperature(500, 1000) == pytest.approx(math.sqrt(0.1), abs=1e-15)
    assert ts_exp.get_temperature(1000, 1000) == pytest.approx(0.1, abs=1e-15)

    ts_cos = TemperatureScheduler(init_temp=1.0, final_temp=0.1, schedule="cosine")
    assert ts_cos.get_temperature(0, 1000) == pytest.approx(1.0, abs=1e-15)
    assert ts_cos.get_temperature(250, 1000) == pytest.approx(
        0.1 + 0.45 * (1 + math.sqrt(0.5)), abs=1e-15
    )
    assert ts_cos.get_temperature(500, 1000) == pytest.approx(0.55, abs=1e-15)
    assert ts_cos.get_temperature(1000, 1000) == pytest.approx(0.1, abs=1e-15)


def test_an_unknown_temperature_schedule_holds_the_initial_temperature() -> None:
    """Finding: an unrecognized schedule name is silently a constant schedule.

    ``TemperatureScheduler.get_temperature`` (mle_dist_supcr.py:71-72) falls through to
    ``return self.init_temp`` for any name but "exponential" and "cosine", so a typo
    such as "linear" trains at the initial temperature throughout. Pinned until an
    unknown schedule raises.
    """
    ts = TemperatureScheduler(init_temp=0.7, final_temp=0.1, schedule="linear")
    assert [ts.get_temperature(e, 100) for e in (0, 50, 100)] == [0.7, 0.7, 0.7]


def test_mle_dist_supcr_basic() -> None:
    """Every component and ratio of the three-sample case (module docstring)."""
    total, parts = _plain()(PREDICTIONS, TARGETS, EMBEDDINGS)
    supcr = _supcr(1.0)
    expected_total = MSE + 0.1 * DIST + 0.001 * supcr
    assert total.item() == pytest.approx(expected_total, rel=1e-6)
    assert parts["mse_loss"] == pytest.approx(MSE, rel=1e-6)
    assert parts["dist_loss"] == pytest.approx(DIST, rel=1e-6)
    assert parts["supcr_loss"] == pytest.approx(supcr, rel=1e-6)
    assert parts["weighted_mse"] == pytest.approx(MSE, rel=1e-6)
    assert parts["weighted_dist"] == pytest.approx(0.1 * DIST, rel=1e-6)
    assert parts["weighted_supcr"] == pytest.approx(0.001 * supcr, rel=1e-6)
    assert parts["total_loss"] == parts["total_weighted"] == total.item()
    assert parts["norm_weighted_mse"] == pytest.approx(MSE / expected_total, rel=1e-6)
    assert parts["norm_weighted_dist"] == pytest.approx(
        0.1 * DIST / expected_total, rel=1e-6
    )
    unweighted = MSE + DIST + supcr
    assert parts["norm_unweighted_supcr"] == pytest.approx(supcr / unweighted, rel=1e-6)
    for key in ("mse_dim_losses", "dist_dim_losses", "supcr_dim_losses"):
        assert parts[key].shape == (1,)
    assert torch.allclose(
        parts["dist_dim_losses"], torch.tensor([DIST]), rtol=1e-6, atol=0
    )
    # no schedule keys when the schedules are off
    assert "buffer_weight" not in parts
    assert "temperature" not in parts


def test_the_gradient_sign_follows_the_prediction_offset() -> None:
    """On predictions t + b the total is b^2 + 0.1 b^2 + const (the distribution term
    is the mean of (t + b - 1)^2 against the all-ones labels, and mean(t) = 1), so the
    gradient is 2.2 b: +1.1 at b = 0.5, -1.1 at b = -0.5.
    """
    for offset, expected in ((0.5, 1.1), (-0.5, -1.1)):
        b = torch.tensor(offset, requires_grad=True)
        total, _ = _plain()(TARGETS + b, TARGETS, EMBEDDINGS)
        total.backward()
        assert b.grad is not None
        assert b.grad.item() == pytest.approx(expected, abs=1e-6)


def test_switched_off_terms_report_zero_and_two_placeholder_dims() -> None:
    """Finding: a switched-off term logs ``torch.zeros(2)`` whatever the width.

    The three lambda-zero branches (mle_dist_supcr.py:562-594, 624-634) hard-code
    ``torch.zeros(2)  # Assuming 2 dimensions``, so a one-dimensional run logs two
    per-dimension zeros. With all three off the total is 0 and neither the
    ``norm_weighted_*`` nor the ``norm_unweighted_*`` ratios are written. Pinned until
    the placeholder takes the target width.
    """
    total, parts = _plain(lambda_mse=0.0, lambda_dist=0.0, lambda_supcr=0.0)(
        PREDICTIONS, TARGETS, EMBEDDINGS
    )
    assert total.item() == 0.0
    assert set(parts) == {
        "mse_loss",
        "mse_dim_losses",
        "weighted_mse",
        "dist_loss",
        "dist_dim_losses",
        "weighted_dist",
        "supcr_loss",
        "supcr_dim_losses",
        "weighted_supcr",
        "total_weighted",
        "total_loss",
    }
    for key in ("mse", "dist", "supcr"):
        assert parts[f"{key}_loss"] == 0.0
        assert parts[f"weighted_{key}"] == 0.0
        assert parts[f"{key}_dim_losses"].tolist() == [0.0, 0.0]


def test_distribution_term_alone_has_all_three_ratios() -> None:
    """With only the distribution term on, it carries the whole normalized weight."""
    _, parts = _plain(lambda_mse=0.0, lambda_supcr=0.0)(
        PREDICTIONS, TARGETS, EMBEDDINGS
    )
    assert parts["dist_loss"] == pytest.approx(DIST, rel=1e-6)
    assert parts["norm_weighted_dist"] == pytest.approx(1.0, abs=1e-12)
    assert parts["norm_weighted_mse"] == 0.0
    assert parts["norm_unweighted_dist"] == pytest.approx(1.0, abs=1e-12)
    assert parts["norm_unweighted_supcr"] == 0.0


def test_the_buffered_dist_loss_counts_the_current_batch_twice() -> None:
    """Finding: at buffer weight 1 the current batch is also read back from the buffer.

    ``BufferedWeightedDistLoss.forward`` writes the batch into the buffer
    (mle_dist_supcr.py:185) and then concatenates the batch with every buffered row
    (220-221), so the first batch is scored doubled: the six-sample labels
    [0, 0, 1, 1, 2, 2] give 0.08333, not the 0.41667 of the batch alone. Pinned until the
    buffer is read before it is written.
    """
    buffered = BufferedWeightedDistLoss(buffer_size=4, bandwidth=0.5, min_samples=3)
    loss, dims = buffered(PREDICTIONS, TARGETS)
    assert loss.item() == pytest.approx(DIST_DOUBLED, rel=1e-6)
    assert dims.tolist() == pytest.approx([DIST_DOUBLED], rel=1e-6)
    base, _ = WeightedDistLoss(bandwidth=0.5)(PREDICTIONS, TARGETS)
    assert base.item() == pytest.approx(DIST, rel=1e-6)


def test_buffered_dist_loss_waits_for_min_samples_then_wraps_around() -> None:
    """Buffer size 4, min 4: three samples return (0, zeros(2)); three more wrap, the
    first new row landing in slot 3 and the next two in slots 0 and 1, pointer
    (3 + 3) mod 4 = 2, total min(6, 4) = 4, and the buffer is full.
    """
    buffered = BufferedWeightedDistLoss(buffer_size=4, bandwidth=0.5, min_samples=4)
    loss, dims = buffered(PREDICTIONS, TARGETS)
    assert loss.item() == 0.0
    assert dims.tolist() == [0.0, 0.0]
    assert (int(buffered.buffer_ptr), int(buffered.total_samples)) == (3, 3)
    assert not bool(buffered.buffer_full)

    buffered.update_buffer(
        torch.tensor([[10.0], [11.0], [12.0]]), torch.tensor([[20.0], [21.0], [22.0]])
    )
    assert buffered.pred_buffer.flatten().tolist() == [11.0, 12.0, 1.0, 10.0]
    assert buffered.target_buffer.flatten().tolist() == [21.0, 22.0, 2.0, 20.0]
    assert (int(buffered.buffer_ptr), int(buffered.total_samples)) == (2, 4)
    assert bool(buffered.buffer_full)
    preds, targets = buffered.get_buffer_samples()
    assert preds.flatten().tolist() == [11.0, 12.0, 1.0, 10.0]
    assert targets.flatten().tolist() == [21.0, 22.0, 2.0, 20.0]


def test_buffered_dist_loss_subsamples_the_buffer_below_weight_one() -> None:
    """Buffer weight 0.5 on a batch of 3 draws int(3 * 0.5 / 0.5) = 3 buffered rows by
    ``torch.randperm``; under a fixed seed the loss equals the base loss on the batch
    followed by those three rows.
    """
    buffered = BufferedWeightedDistLoss(buffer_size=8, bandwidth=0.5, min_samples=3)
    buffered.update_buffer(PREDICTIONS + 3.0, TARGETS + 3.0)
    torch.manual_seed(0)
    loss, _ = buffered(PREDICTIONS, TARGETS, buffer_weight=0.5)

    rows_p, rows_t = buffered.get_buffer_samples()
    assert rows_p.flatten().tolist() == [5.0, 3.5, 4.0, 2.0, 0.5, 1.0]
    torch.manual_seed(0)
    index = torch.randperm(6)[:3]
    expected, _ = WeightedDistLoss(bandwidth=0.5)(
        torch.cat([PREDICTIONS, rows_p[index]]), torch.cat([TARGETS, rows_t[index]])
    )
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_buffered_dist_loss_fills_its_buffer_from_the_gathered_batch() -> None:
    """With gathered tensors the buffer takes all six gathered rows, not the local
    three, and the loss is the base loss on the gathered batch doubled.
    """
    gathered_p = torch.cat([PREDICTIONS, PREDICTIONS + 3.0])
    gathered_t = torch.cat([TARGETS, TARGETS + 3.0])
    buffered = BufferedWeightedDistLoss(buffer_size=8, bandwidth=0.5, min_samples=6)
    loss, _ = buffered(PREDICTIONS, TARGETS, gathered_p, gathered_t)
    assert int(buffered.total_samples) == 6
    assert buffered.pred_buffer[:6].flatten().tolist() == [2.0, 0.5, 1.0, 5.0, 3.5, 4.0]
    expected, _ = WeightedDistLoss(bandwidth=0.5)(
        torch.cat([gathered_p, gathered_p]), torch.cat([gathered_t, gathered_t])
    )
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_buffered_supcr_scales_the_loss_by_one_minus_half_the_buffer_weight() -> None:
    """Finding: the "buffer influence" factor scales the WHOLE loss.

    ``BufferedWeightedSupCRCell.forward`` (mle_dist_supcr.py:373-375) multiplies the
    SupCR on [batch; buffer] by 1 - w + 0.5 w, so at w = 1 even the current-batch
    anchors are halved and at w = 0.3 the factor is 0.85. The temperature argument
    overwrites the inner SupCR's (0.1 to 1.0) and persists; without it the stored
    temperature is used. Pinned until only the buffer rows are down-weighted.
    """
    base_doubled, dims_doubled = WeightedSupCRCell(temperature=1.0)(
        torch.cat([EMBEDDINGS, EMBEDDINGS]), torch.cat([TARGETS, TARGETS])
    )
    cell = BufferedWeightedSupCRCell(
        buffer_size=4, embedding_dim=2, temperature=0.1, min_samples=3
    )
    loss, dims = cell(EMBEDDINGS, TARGETS, temperature=1.0)
    assert cell.base_supcr.supcr.temperature == 1.0
    assert loss.item() == pytest.approx(0.5 * base_doubled.item(), rel=1e-6)
    assert torch.allclose(dims, dims_doubled, rtol=1e-6, atol=0)

    fresh = BufferedWeightedSupCRCell(
        buffer_size=4, embedding_dim=2, temperature=1.0, min_samples=3
    )
    weighted, _ = fresh(EMBEDDINGS, TARGETS, buffer_weight=0.3)
    assert fresh.base_supcr.supcr.temperature == 1.0
    assert weighted.item() == pytest.approx(0.85 * base_doubled.item(), rel=1e-6)


def test_buffered_supcr_waits_wraps_and_takes_the_gathered_batch() -> None:
    """Below min_samples: (0, zeros(2)). Gathered embeddings fill the buffer instead of
    the local ones; a second push of 3 into a buffer of 4 holding 3 wraps like the
    distribution buffer (slots 3, 0, 1; pointer 2; full).
    """
    cell = BufferedWeightedSupCRCell(
        buffer_size=4, embedding_dim=2, temperature=1.0, min_samples=4
    )
    gathered = EMBEDDINGS * 2.0
    loss, dims = cell(EMBEDDINGS, TARGETS, gathered, TARGETS + 5.0)
    assert loss.item() == 0.0
    assert dims.tolist() == [0.0, 0.0]
    assert cell.embedding_buffer[:3].tolist() == gathered.tolist()
    assert cell.label_buffer[:3].flatten().tolist() == [5.0, 6.0, 7.0]

    cell.update_buffer(EMBEDDINGS, TARGETS)
    assert cell.label_buffer.flatten().tolist() == [1.0, 2.0, 7.0, 0.0]
    assert cell.embedding_buffer.tolist() == [
        [0.0, 1.0],
        [-1.0, 0.0],
        [-2.0, 0.0],
        [1.0, 0.0],
    ]
    assert (int(cell.buffer_ptr), int(cell.total_samples)) == (2, 4)
    assert bool(cell.buffer_full)
    embeddings, labels = cell.get_buffer_samples()
    assert labels.flatten().tolist() == [1.0, 2.0, 7.0, 0.0]
    assert embeddings.shape == (4, 2)


def test_mle_dist_supcr_with_buffer() -> None:
    """Buffered composite at the default schedules with max_epochs 100: warmup
    int(100 * 0.1) = 10, stable int(100 * 0.5) = 50, so epoch 5 gives buffer weight
    0.1 + 0.2 * 5 / 10 = 0.2 and temperature 1 * 0.1^(5 / 100). Below the 64-sample
    minimum both buffered terms are exactly 0 and the MSE alone is the total.
    """
    loss = MleDistSupCR(use_ddp_gather=False, embedding_dim=2, max_epochs=100)
    total, parts = loss(PREDICTIONS, TARGETS, EMBEDDINGS, epoch=5)
    assert parts["buffer_weight"] == pytest.approx(0.2, abs=1e-15)
    assert parts["temperature"] == pytest.approx(0.1**0.05, abs=1e-12)
    assert parts["dist_loss"] == 0.0
    assert parts["supcr_loss"] == 0.0
    assert total.item() == pytest.approx(MSE, rel=1e-6)
    assert parts["norm_weighted_mse"] == pytest.approx(1.0, abs=1e-12)
    assert int(loss.current_epoch) == 5
    assert int(loss.forward_count) == 1
    # without an epoch argument the stored epoch is kept
    _, again = loss(PREDICTIONS, TARGETS, EMBEDDINGS)
    assert again["buffer_weight"] == pytest.approx(0.2, abs=1e-15)
    assert int(loss.forward_count) == 2


def test_mle_dist_supcr_adaptive_features() -> None:
    """Explicit warmup 10 and stable 50 at epochs 0, 30 and 100: buffer weights 0.1,
    0.3 + 0.6 / (1 + e^0) = 0.6 and 0.9; temperatures 1, 0.1^0.03 and 0.1^0.1.
    """
    loss = MleDistSupCR(
        use_ddp_gather=False,
        warmup_epochs=10,
        stable_epoch=50,
        embedding_dim=2,
        max_epochs=1000,
    )
    parts = [
        loss(PREDICTIONS, TARGETS, EMBEDDINGS, epoch=epoch)[1] for epoch in (0, 30, 100)
    ]
    assert [p["buffer_weight"] for p in parts] == pytest.approx(
        [0.1, 0.6, 0.9], abs=1e-15
    )
    assert [p["temperature"] for p in parts] == pytest.approx(
        [1.0, 0.1**0.03, 0.1**0.1], abs=1e-12
    )


class _GatherRecorder:
    """Stands in for ``torch.distributed`` with world size 2; rank 1 holds x + 3."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def is_initialized(self) -> bool:
        return True

    def get_world_size(self) -> int:
        return 2

    def all_gather(self, out: list[torch.Tensor], tensor: torch.Tensor) -> None:
        self.calls.append(tuple(tensor.shape))
        out[0].copy_(tensor)
        out[1].copy_(tensor + 3.0)


def test_gathering_feeds_the_buffers_while_the_mse_stays_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """World size 2, gather interval 2: the first forward all-gathers predictions,
    targets and embeddings (three calls), the MSE is still the local 1.75, and the
    distribution term is the base loss on the six gathered rows doubled. The second
    forward (count 1, 1 mod 2 = 1) does not gather and buffers the three local rows:
    total min(6 + 3, 8) = 8, pointer (6 + 3) mod 8 = 1.
    """
    recorder = _GatherRecorder()
    monkeypatch.setattr(mds, "dist", recorder)
    loss = _plain(
        use_buffer=True,
        use_ddp_gather=True,
        gather_interval=2,
        buffer_size=8,
        min_samples_for_dist=6,
        min_samples_for_supcr=6,
        embedding_dim=2,
    )
    _, parts = loss(PREDICTIONS, TARGETS, EMBEDDINGS)
    assert recorder.calls == [(3, 1), (3, 1), (3, 2)]
    assert parts["mse_loss"] == pytest.approx(MSE, rel=1e-6)
    gathered_p = torch.cat([PREDICTIONS, PREDICTIONS + 3.0])
    gathered_t = torch.cat([TARGETS, TARGETS + 3.0])
    expected, _ = WeightedDistLoss(bandwidth=0.5)(
        torch.cat([gathered_p, gathered_p]), torch.cat([gathered_t, gathered_t])
    )
    assert parts["dist_loss"] == pytest.approx(expected.item(), rel=1e-6)
    assert isinstance(loss.dist_loss, BufferedWeightedDistLoss)
    assert int(loss.dist_loss.total_samples) == 6

    loss(PREDICTIONS, TARGETS, EMBEDDINGS)
    assert len(recorder.calls) == 3
    assert (int(loss.dist_loss.total_samples), int(loss.dist_loss.buffer_ptr)) == (8, 1)


def test_gather_is_the_identity_on_one_process() -> None:
    """Without an initialized process group the tensor comes back unchanged."""
    x = torch.tensor([[1.0, 2.0]])
    assert _plain().gather_across_gpus(x) is x  # line 488 returns the tensor itself


def test_unbuffered_supcr_runs_at_the_scheduled_temperature() -> None:
    """Without a buffer the scheduled temperature is applied, not only logged.

    The exponential schedule gives exactly ``init_temperature`` = 1.0 at epoch 0, so the
    inner SupCR's temperature is overwritten from the constructor's 0.1 to 1.0 and the
    reported SupCR value equals ``WeightedSupCRCell(temperature=1.0)`` on the same
    inputs. Before the fix the value was the one at 0.1.
    """
    torch.manual_seed(0)
    targets = torch.randn(16, 2)
    predictions = targets + 1.0
    z_p = torch.randn(16, 4)
    loss = MleDistSupCR(
        lambda_dist=0.0,
        supcr_temperature=0.1,
        use_buffer=False,
        use_ddp_gather=False,
        use_adaptive_weighting=False,
        use_temp_scheduling=True,
        init_temperature=1.0,
        final_temperature=0.1,
        max_epochs=100,
    )
    assert isinstance(loss.supcr_loss, WeightedSupCRCell)
    assert loss.supcr_loss.supcr.temperature == 0.1
    _, parts = loss(predictions, targets, z_p, epoch=0)
    assert parts["temperature"] == 1.0
    assert loss.supcr_loss.supcr.temperature == 1.0
    at_one, _ = WeightedSupCRCell(temperature=1.0)(z_p, targets)
    at_tenth, _ = WeightedSupCRCell(temperature=0.1)(z_p, targets)
    assert parts["supcr_loss"] == pytest.approx(at_one.item(), rel=1e-6)
    assert parts["supcr_loss"] != pytest.approx(at_tenth.item(), rel=1e-2)


def test_a_small_buffer_weight_scores_the_batch_alone() -> None:
    """At buffer weight 0.1 a batch of 3 asks for int(3 * 0.1 / 0.9) = 0 buffered rows,
    so the loss is the batch's own closed form 1.25 / 3, not the doubled 0.5 / 6.
    """
    buffered = BufferedWeightedDistLoss(buffer_size=8, bandwidth=0.5, min_samples=3)
    loss, _ = buffered(PREDICTIONS, TARGETS, buffer_weight=0.1)
    assert loss.item() == pytest.approx(DIST, rel=1e-6)


def test_an_empty_buffer_takes_the_width_of_the_first_batch() -> None:
    """With ``weights=None`` both buffers start one column wide; the first two-column
    batch reallocates them to two columns before writing, and the rows land intact.
    """
    two = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dist_buffer = BufferedWeightedDistLoss(buffer_size=4, min_samples=8)
    assert dist_buffer.pred_buffer.shape == (4, 1)
    dist_buffer.update_buffer(two, two + 10.0)
    assert dist_buffer.pred_buffer.shape == (4, 2)
    assert dist_buffer.pred_buffer[:2].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert dist_buffer.target_buffer[:2].tolist() == [[11.0, 12.0], [13.0, 14.0]]

    supcr_buffer = BufferedWeightedSupCRCell(buffer_size=4, embedding_dim=2)
    assert supcr_buffer.label_buffer.shape == (4, 1)
    supcr_buffer.update_buffer(two, two + 10.0)
    assert supcr_buffer.label_buffer.shape == (4, 2)
    assert supcr_buffer.label_buffer[:2].tolist() == [[11.0, 12.0], [13.0, 14.0]]
