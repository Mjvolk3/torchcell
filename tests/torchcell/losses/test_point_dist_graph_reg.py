# tests/torchcell/losses/test_point_dist_graph_reg.py
# [[tests.torchcell.losses.test_point_dist_graph_reg]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_point_dist_graph_reg.py
"""``PointDistGraphReg`` composition on a three-sample batch with the distribution term off.

Predictions [1, 2, 3] against targets [1, 1, 1]: log-cosh mean is 0.5862611926 (see
``test_logcosh.py``), MSE is (0 + 1 + 4) / 3 = 5/3. With ``lambda_point`` 2.0 and a graph
regularization value 0.4 at ``lambda`` 0.5, the total is 2 * 0.5862611926 + 0.5 * 0.4 =
1.3725223852. The distribution loss stays off (``lambda`` 0) so the value is a hand
computation; its buffered variants are exercised in ``test_mle_dist_supcr.py``.

2026.09.30 (Phase 12). The loss takes the graph regularization as a precomputed scalar
under ``representations["graph_reg_loss"]``; no graph enters the module, so the graph term
is pinned as that scalar (or a function of one parameter, for the gradient). Added:

- Defaults: ``PointDistGraphReg()`` is log-cosh at lambda 1, a buffered dist loss at
  lambda 0.1 (bandwidth 0.5, buffer 256, ``min_samples`` 64), graph lambda 1, DDP gather on
  at interval 1. On the three-sample batch the buffered dist loss has 3 < 64 samples, so it
  returns exactly 0.0 and its buffer holds [1, 2, 3] at pointer 3.
- The distribution term's wiring, with the constructed component's ``forward`` replaced by
  a recorder returning 0.25: total = 2 * 0.5862611926 + 0.5 * 0.25 = 1.2975223852; the
  buffered call receives (predictions, targets, gathered predictions, gathered targets,
  ``buffer_weight=1.0``), the unbuffered call only (predictions, targets); with
  ``gather_interval`` 2 the gathered arguments are passed on calls 0 and 2 and are None on
  call 1.
- ``gather_across_gpus`` under a faked two-rank group concatenates rank 0 then rank 1.
- The unbuffered real ``WeightedDistLoss`` enters the total as ``lambda_dist`` times its
  own value (a composition identity; the KDE value itself is ``test_multi_dim_nan_tolerant``'s).
- Wasserstein construction: buffered (``min_samples`` 224, buffer 256, blur 0.05, p 2,
  scaling 0.9) and unbuffered (the flat blur/p/scaling carried through), the ImportError
  when geomloss is flagged unavailable, and a Finding: flat ``min_samples_for_dist`` never
  reaches the Wasserstein component (224 is hard-coded on the flat path).
- Masking: the mse branch drops a NaN target (predictions [1, 2, 3] against [1, NaN, 1]:
  (0 + 4) / 2 = 2.0); the log-cosh branch does not (Finding: NaN total, and the normalized
  keys are absent because NaN > 0 is False).
- Weighting: two-dimensional mse with weights [3, 1] normalized to [0.75, 0.25] over
  per-dimension losses 5/3 and 4/3 gives 1.25 + 1/3 = 1.5833333.
- A zero total (predictions equal targets, no graph term) omits every ``norm_*`` key; the
  returned key set is pinned exactly in both cases.
- ``lambda_graph_reg`` 0 ignores a present graph term; ``distribution_loss`` of type None
  builds no distribution component.
- Gradient sign on one parameter w with predictions w * [1, 2, 3], targets [2, 4, 6] and
  graph term w ** 2 at lambda 0.5: dL/dw at w = 1 is
  2 * (tanh(-1) + 2 tanh(-2) + 3 tanh(-3)) / 3 + 0.5 * 2 = -2.7832090514, negative, so a
  gradient step moves w toward its target 2.
"""

import math
from typing import Any

import pytest
import torch
import torch.distributed as torch_dist

import torchcell.losses.point_dist_graph_reg as pdgr
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.mle_dist_supcr import BufferedWeightedDistLoss
from torchcell.losses.mle_wasserstein import (
    BufferedWeightedWassersteinLoss,
    WeightedWassersteinLoss,
)
from torchcell.losses.multi_dim_nan_tolerant import WeightedDistLoss, WeightedMSELoss
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg

PREDICTIONS = torch.tensor([[1.0], [2.0], [3.0]])
TARGETS = torch.tensor([[1.0], [1.0], [1.0]])
LOGCOSH_MEAN = 0.5862611926
MSE_MEAN = 5.0 / 3.0


def _loss(point_type: str = "logcosh", lambda_point: float = 2.0) -> PointDistGraphReg:
    return PointDistGraphReg(
        point_estimator={"type": point_type, "lambda": lambda_point},
        distribution_loss={"type": "dist", "lambda": 0.0},
        graph_regularization={"lambda": 0.5},
        ddp={"use_ddp_gather": False},
    )


def test_logcosh_point_plus_graph_regularization() -> None:
    """Total = 2 * 0.5862611926 + 0.5 * 0.4 = 1.3725223852, every part reported."""
    total, parts = _loss()(PREDICTIONS, TARGETS, {"graph_reg_loss": torch.tensor(0.4)})
    assert total.item() == pytest.approx(1.3725223852, abs=1e-6)
    assert parts["point_loss"] == pytest.approx(LOGCOSH_MEAN, abs=1e-6)
    assert parts["weighted_point"] == pytest.approx(2 * LOGCOSH_MEAN, abs=1e-6)
    assert parts["dist_loss"] == 0.0 and parts["weighted_dist"] == 0.0
    assert parts["graph_reg_loss"] == pytest.approx(0.4, abs=1e-6)
    assert parts["weighted_graph_reg"] == pytest.approx(0.2, abs=1e-6)
    assert parts["total_loss"] == pytest.approx(1.3725223852, abs=1e-6)
    # normalized shares: weighted point / total, and the unweighted point / (point + reg)
    assert parts["norm_weighted_point"] == pytest.approx(
        2 * LOGCOSH_MEAN / 1.3725223852, abs=1e-6
    )
    assert parts["norm_unweighted_point"] == pytest.approx(
        LOGCOSH_MEAN / (LOGCOSH_MEAN + 0.4), abs=1e-6
    )
    assert parts["norm_weighted_graph_reg"] == pytest.approx(
        0.2 / 1.3725223852, abs=1e-6
    )


def test_mse_point_estimator() -> None:
    """The mse branch uses WeightedMSELoss: (0 + 1 + 4) / 3 = 5/3, times lambda 2."""
    total, parts = _loss("mse")(PREDICTIONS, TARGETS, {})
    assert parts["point_loss"] == pytest.approx(MSE_MEAN, abs=1e-6)
    assert total.item() == pytest.approx(2 * MSE_MEAN, abs=1e-6)
    assert parts["graph_reg_loss"] == 0.0


def test_missing_graph_reg_key_adds_nothing() -> None:
    """Without "graph_reg_loss" in representations the total is the weighted point loss."""
    total, parts = _loss()(PREDICTIONS, TARGETS, {})
    assert total.item() == pytest.approx(2 * LOGCOSH_MEAN, abs=1e-6)
    assert parts["weighted_graph_reg"] == 0.0


def test_forward_count_increments_per_call() -> None:
    """The registered counter tracks calls (used for DDP gather intervals)."""
    loss = _loss()
    assert loss.forward_count.item() == 0
    loss(PREDICTIONS, TARGETS, {})
    loss(PREDICTIONS, TARGETS, {})
    assert loss.forward_count.item() == 2


def test_gradient_reaches_the_predictions() -> None:
    """D total / d p = lambda_point * tanh(p - t) / 3 for the log-cosh branch."""
    predictions = PREDICTIONS.clone().requires_grad_(True)
    total, _ = _loss()(predictions, TARGETS, {})
    total.backward()
    assert predictions.grad is not None
    expected = 2 * torch.tensor([[0.0], [0.7615941560], [0.9640275801]]) / 3
    torch.testing.assert_close(predictions.grad, expected, atol=1e-6, rtol=0)


def test_flat_keyword_configuration_matches_the_nested_form() -> None:
    """The deprecated flat kwargs configure the same components."""
    flat = PointDistGraphReg(
        point_loss_type="logcosh",
        lambda_point=2.0,
        lambda_dist=0.0,
        lambda_graph_reg=0.5,
        use_ddp_gather=False,
    )
    total, _ = flat(PREDICTIONS, TARGETS, {"graph_reg_loss": torch.tensor(0.4)})
    assert total.item() == pytest.approx(1.3725223852, abs=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"point_estimator": {"type": "huber"}}, "Unknown point_loss_type: huber"),
        (
            {
                "distribution_loss": {"type": "kl", "lambda": 0.1},
                "ddp": {"use_ddp_gather": False},
            },
            "Unknown dist_loss_type: kl",
        ),
    ],
)
def test_unknown_component_types_raise(kwargs: dict[str, object], message: str) -> None:
    """A misspelled component type fails at construction, naming the value."""
    with pytest.raises(ValueError, match=message):
        PointDistGraphReg(**kwargs)  # type: ignore[arg-type]


FULL_KEYS = {
    "point_loss",
    "weighted_point",
    "dist_loss",
    "weighted_dist",
    "graph_reg_loss",
    "weighted_graph_reg",
    "norm_weighted_point",
    "norm_weighted_dist",
    "norm_weighted_graph_reg",
    "total_loss",
    "total_weighted",
    "norm_unweighted_point",
    "norm_unweighted_dist",
    "norm_unweighted_graph_reg",
}
UNNORMALIZED_KEYS = {
    "point_loss",
    "weighted_point",
    "dist_loss",
    "weighted_dist",
    "graph_reg_loss",
    "weighted_graph_reg",
    "total_loss",
    "total_weighted",
}


class _Recorder:
    """Stands in for a distribution component's ``forward``; returns 0.25 per call."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((args, kwargs))
        return torch.tensor(0.25), torch.zeros(1)


def test_default_configuration() -> None:
    """No arguments: log-cosh at 1, buffered dist at 0.1 (0.5 / 256 / 64), graph 1, gather on."""
    loss = PointDistGraphReg()
    assert isinstance(loss.point_loss, LogCoshLoss)
    assert isinstance(loss.dist_loss, BufferedWeightedDistLoss)
    assert (loss.lambda_point, loss.lambda_dist, loss.lambda_graph_reg) == (
        1.0,
        0.1,
        1.0,
    )
    assert (loss.point_loss_type, loss.dist_loss_type) == ("logcosh", "dist")
    assert (loss.use_buffer, loss.use_ddp_gather, loss.gather_interval) == (
        True,
        True,
        1,
    )
    assert loss.dist_loss.buffer_size == 256
    assert loss.dist_loss.min_samples == 64
    assert loss.dist_loss.base_dist_loss.bandwidth == 0.5


def test_buffered_dist_below_min_samples_is_zero_and_fills_the_buffer() -> None:
    """3 samples < min_samples 64: dist term exactly 0.0; the buffer holds [1, 2, 3]."""
    loss = PointDistGraphReg()
    total, parts = loss(PREDICTIONS, TARGETS, {})
    assert isinstance(loss.dist_loss, BufferedWeightedDistLoss)
    assert parts["dist_loss"] == 0.0 and parts["weighted_dist"] == 0.0
    assert total.item() == pytest.approx(LOGCOSH_MEAN, abs=1e-6)
    assert int(loss.dist_loss.buffer_ptr) == 3
    assert int(loss.dist_loss.total_samples) == 3
    torch.testing.assert_close(
        loss.dist_loss.pred_buffer[:4], torch.tensor([[1.0], [2.0], [3.0], [0.0]])
    )
    torch.testing.assert_close(
        loss.dist_loss.target_buffer[:4], torch.tensor([[1.0], [1.0], [1.0], [0.0]])
    )
    assert set(parts) == FULL_KEYS
    assert parts["norm_weighted_point"] == 1.0
    assert parts["norm_unweighted_dist"] == 0.0


def test_buffered_dist_term_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recorder returns 0.25: total 2 * 0.5862611926 + 0.5 * 0.25; gathered args passed."""
    loss = PointDistGraphReg(
        point_estimator={"type": "logcosh", "lambda": 2.0},
        distribution_loss={"type": "dist", "lambda": 0.5},
    )
    recorder = _Recorder()
    monkeypatch.setattr(loss.dist_loss, "forward", recorder)
    total, parts = loss(PREDICTIONS, TARGETS, {})
    assert total.item() == pytest.approx(2 * LOGCOSH_MEAN + 0.125, abs=1e-6)
    assert parts["dist_loss"] == 0.25
    assert parts["weighted_dist"] == 0.125
    assert parts["norm_weighted_dist"] == pytest.approx(
        0.125 / (2 * LOGCOSH_MEAN + 0.125), abs=1e-6
    )
    assert parts["norm_unweighted_dist"] == pytest.approx(
        0.25 / (LOGCOSH_MEAN + 0.25), abs=1e-6
    )
    (args, kwargs) = recorder.calls[0]
    assert len(recorder.calls) == 1
    assert kwargs == {"buffer_weight": 1.0}
    assert len(args) == 4
    # single process: gather returns its input, so the gathered args are the batch itself
    assert args[0] is PREDICTIONS and args[1] is TARGETS
    assert args[2] is PREDICTIONS and args[3] is TARGETS


def test_gather_interval_skips_off_interval_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gather_interval 2: gathered args on forward 0 and 2, None on forward 1."""
    loss = PointDistGraphReg(
        distribution_loss={"type": "dist", "lambda": 0.5},
        ddp={"use_ddp_gather": True, "gather_interval": 2},
    )
    recorder = _Recorder()
    monkeypatch.setattr(loss.dist_loss, "forward", recorder)
    totals = [loss(PREDICTIONS, TARGETS, {})[0].item() for _ in range(3)]
    gathered = [(c[0][2] is not None, c[0][3] is not None) for c in recorder.calls]
    assert gathered == [(True, True), (False, False), (True, True)]
    # the gather schedule does not change the value: 0.5862611926 + 0.5 * 0.25 each time
    assert totals == pytest.approx([LOGCOSH_MEAN + 0.125] * 3, abs=1e-6)
    assert loss.forward_count.item() == 3


def test_gather_disabled_passes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """use_ddp_gather False: the buffered component never receives gathered tensors."""
    loss = PointDistGraphReg(
        distribution_loss={"type": "dist", "lambda": 0.5}, ddp={"use_ddp_gather": False}
    )
    recorder = _Recorder()
    monkeypatch.setattr(loss.dist_loss, "forward", recorder)
    total, _ = loss(PREDICTIONS, TARGETS, {})
    assert recorder.calls[0][0][2:] == (None, None)
    assert total.item() == pytest.approx(LOGCOSH_MEAN + 0.125, abs=1e-6)


def test_unbuffered_dist_term_gets_only_the_local_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """use_buffer False builds WeightedDistLoss and calls it with (predictions, targets)."""
    loss = PointDistGraphReg(
        point_estimator={"type": "logcosh", "lambda": 2.0},
        distribution_loss={"type": "dist", "lambda": 0.5, "dist_bandwidth": 0.7},
        buffer={"use_buffer": False},
    )
    assert isinstance(loss.dist_loss, WeightedDistLoss)
    assert loss.dist_loss.bandwidth == 0.7
    recorder = _Recorder()
    monkeypatch.setattr(loss.dist_loss, "forward", recorder)
    total, _ = loss(PREDICTIONS, TARGETS, {})
    assert total.item() == pytest.approx(2 * LOGCOSH_MEAN + 0.125, abs=1e-6)
    assert len(recorder.calls[0][0]) == 2 and recorder.calls[0][1] == {}


def test_real_unbuffered_dist_term_enters_as_lambda_times_its_value() -> None:
    """Composition identity on targets [0, 1, 3]: total = 2 * logcosh + 0.5 * dist."""
    loss = PointDistGraphReg(
        point_estimator={"type": "logcosh", "lambda": 2.0},
        distribution_loss={"type": "dist", "lambda": 0.5},
        buffer={"use_buffer": False},
        ddp={"use_ddp_gather": False},
    )
    # spread targets: a constant target column makes the KDE covariance singular
    targets = torch.tensor([[0.0], [1.0], [3.0]])
    point = sum(math.log(math.cosh(r)) for r in (1.0, 1.0, 0.0)) / 3
    reference, _ = WeightedDistLoss(bandwidth=0.5)(PREDICTIONS, targets)
    total, parts = loss(PREDICTIONS, targets, {})
    assert parts["point_loss"] == pytest.approx(point, abs=1e-6)
    assert parts["dist_loss"] == pytest.approx(reference.item(), abs=1e-6)
    assert total.item() == pytest.approx(2 * point + 0.5 * reference.item(), abs=1e-6)


def test_gather_across_two_fake_ranks_concatenates_in_rank_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A faked two-rank group: rank 0 gives the tensor, rank 1 gives it plus 10."""

    def fake_all_gather(out: list[torch.Tensor], tensor: torch.Tensor) -> None:
        out[0].copy_(tensor)
        out[1].copy_(tensor + 10)

    monkeypatch.setattr(torch_dist, "is_initialized", lambda: True)
    monkeypatch.setattr(torch_dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch_dist, "all_gather", fake_all_gather)
    gathered = PointDistGraphReg().gather_across_gpus(PREDICTIONS)
    torch.testing.assert_close(
        gathered, torch.tensor([[1.0], [2.0], [3.0], [11.0], [12.0], [13.0]])
    )


def test_gather_is_identity_for_a_single_rank_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialized with world size 1 (line 231): the tensor comes back equal and
    ``all_gather`` is never called. The uninitialized half of the guard is pinned by the
    buffered wiring test, whose third recorded argument is the predictions object.
    """
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("all_gather must not be called for one rank")

    monkeypatch.setattr(dist, "all_gather", refuse)
    out = PointDistGraphReg().gather_across_gpus(PREDICTIONS)
    assert torch.equal(out, PREDICTIONS)


def test_buffered_wasserstein_configuration() -> None:
    """Nested defaults: buffered, min_samples 224, buffer 256, blur 0.05, p 2, scaling 0.9."""
    loss = PointDistGraphReg(distribution_loss={"type": "wasserstein"})
    assert isinstance(loss.dist_loss, BufferedWeightedWassersteinLoss)
    assert loss.dist_loss.min_samples == 224
    assert loss.dist_loss.buffer_size == 256
    base = loss.dist_loss.base_wasserstein_loss
    assert (base.blur, base.p, base.scaling) == (0.05, 2, 0.9)
    nested = PointDistGraphReg(
        distribution_loss={"type": "wasserstein", "min_samples_for_wasserstein": 16}
    )
    assert isinstance(nested.dist_loss, BufferedWeightedWassersteinLoss)
    assert nested.dist_loss.min_samples == 16


def test_unbuffered_wasserstein_takes_the_flat_parameters() -> None:
    """Flat kwargs: blur 0.1, p 1, scaling 0.5 reach WeightedWassersteinLoss."""
    loss = PointDistGraphReg(
        dist_loss_type="wasserstein",
        use_buffer=False,
        wasserstein_blur=0.1,
        wasserstein_p=1,
        wasserstein_scaling=0.5,
    )
    assert isinstance(loss.dist_loss, WeightedWassersteinLoss)
    assert (loss.dist_loss.blur, loss.dist_loss.p, loss.dist_loss.scaling) == (
        0.1,
        1,
        0.5,
    )


def test_flat_min_samples_does_not_reach_wasserstein() -> None:
    """Finding: the flat path hard-codes 224 (``point_dist_graph_reg.py:125``), so
    ``min_samples_for_dist=8`` is silently ignored for Wasserstein, and the flat
    signature has no ``min_samples_for_wasserstein``. Pinned until the flat path forwards it.
    """
    loss = PointDistGraphReg(dist_loss_type="wasserstein", min_samples_for_dist=8)
    assert isinstance(loss.dist_loss, BufferedWeightedWassersteinLoss)
    assert loss.dist_loss.min_samples == 224


def test_wasserstein_without_geomloss_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Geomloss unavailable: construction raises ImportError with the install hint."""
    monkeypatch.setattr(pdgr, "WASSERSTEIN_AVAILABLE", False)
    with pytest.raises(
        ImportError,
        match=r"^Wasserstein loss requires geomloss\. Install with: pip install geomloss$",
    ):
        PointDistGraphReg(distribution_loss={"type": "wasserstein"})


def test_dist_type_none_builds_no_distribution_component() -> None:
    """Type None with a positive lambda: no component, dist terms reported as 0.0."""
    loss = PointDistGraphReg(distribution_loss={"type": None, "lambda": 0.5})
    assert loss.dist_loss is None
    _, parts = loss(PREDICTIONS, TARGETS, {})
    assert (parts["dist_loss"], parts["weighted_dist"]) == (0.0, 0.0)


def test_mse_masks_a_nan_target() -> None:
    """[1, 2, 3] against [1, NaN, 1]: the NaN row is dropped, (0 + 4) / 2 = 2.0."""
    loss = _loss("mse", lambda_point=1.0)
    targets = torch.tensor([[1.0], [float("nan")], [1.0]])
    total, parts = loss(PREDICTIONS, targets, {})
    assert total.item() == 2.0
    assert parts["point_loss"] == 2.0
    assert set(parts) == FULL_KEYS


def test_logcosh_propagates_a_nan_target() -> None:
    """Finding: the default log-cosh point estimator is not NaN tolerant (unlike mse);
    one NaN target makes the total NaN, and the ``norm_*`` keys vanish because
    ``NaN > 0`` is False (``point_dist_graph_reg.py:333,350``). Pinned until log-cosh masks.
    """
    targets = torch.tensor([[1.0], [float("nan")], [1.0]])
    total, parts = _loss()(PREDICTIONS, targets, {})
    assert math.isnan(total.item())
    assert math.isnan(parts["point_loss"])
    assert set(parts) == UNNORMALIZED_KEYS


def test_mse_dimension_weights() -> None:
    """Weights [3, 1] -> [0.75, 0.25] over dim losses 5/3 and 4/3: 1.25 + 1/3."""
    loss = PointDistGraphReg(
        point_estimator={"type": "mse", "lambda": 1.0},
        distribution_loss={"type": "dist", "lambda": 0.0},
        weights=torch.tensor([3.0, 1.0]),
    )
    assert isinstance(loss.point_loss, WeightedMSELoss)
    predictions = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    total, _ = loss(predictions, targets, {})
    assert total.item() == pytest.approx(1.25 + 1.0 / 3.0, abs=1e-6)


def test_zero_total_omits_every_normalized_key() -> None:
    """Predictions equal targets, no graph term: total 0.0 and no ``norm_*`` keys."""
    total, parts = _loss()(TARGETS, TARGETS, {})
    assert total.item() == 0.0
    assert set(parts) == UNNORMALIZED_KEYS
    assert parts["total_loss"] == 0.0 and parts["total_weighted"] == 0.0


def test_zero_graph_lambda_ignores_a_present_graph_term() -> None:
    """lambda_graph_reg 0: a supplied graph_reg_loss of 5.0 contributes nothing."""
    loss = PointDistGraphReg(
        distribution_loss={"type": "dist", "lambda": 0.0},
        graph_regularization={"lambda": 0.0},
    )
    total, parts = loss(PREDICTIONS, TARGETS, {"graph_reg_loss": torch.tensor(5.0)})
    assert total.item() == pytest.approx(LOGCOSH_MEAN, abs=1e-6)
    assert (parts["graph_reg_loss"], parts["weighted_graph_reg"]) == (0.0, 0.0)


def test_gradient_sign_on_one_parameter() -> None:
    """W = 1, predictions w * [1, 2, 3], targets [2, 4, 6], graph term w ** 2 at 0.5:
    dL/dw = 2 * (tanh(-1) + 2 tanh(-2) + 3 tanh(-3)) / 3 + 1 = -2.7832090514 < 0.
    """
    w = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([[2.0], [4.0], [6.0]])
    total, _ = _loss()(w * x, targets, {"graph_reg_loss": w**2})
    total.backward()
    assert w.grad is not None
    assert w.grad.item() == pytest.approx(-2.7832090514, abs=1e-6)
    assert w.grad.item() < 0
