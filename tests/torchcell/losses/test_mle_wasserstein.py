# tests/torchcell/losses/test_mle_wasserstein.py
# [[tests.torchcell.losses.test_mle_wasserstein]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_mle_wasserstein.py
"""The two schedulers by closed form, and ``WeightedWassersteinLoss`` on translated clouds.

``AdaptiveWeighting(100, 500)``: 0.1 + 0.2 * epoch/100 during warmup; 0.3 + 0.6 /
(1 + exp(-10 (progress - 0.5))) until the stable epoch (progress 0 gives
0.3 + 0.6 / (1 + e^5) = 0.3040157, progress 0.5 gives 0.6); 0.9 after.
``TemperatureScheduler(1.0, 0.1)``: exponential 1.0 * 0.1^(epoch/max); cosine
0.1 + 0.45 (1 + cos(pi epoch/max)).

The Wasserstein loss wraps geomloss's debiased Sinkhorn divergence with p = 2, whose
ground cost is |x - y|^2 / 2. Translating a cloud by s moves every unit of mass by s, so
the divergence is exactly s^2 / 2: 0 for identical clouds, 0.5 at s = 1, 2.0 at s = 2,
independent of the cloud. That is the hand value the tests pin, along with the
all-NaN-dimension rule and the weighted sum.
"""

import math

import pytest
import torch

from torchcell.losses.mle_wasserstein import (
    AdaptiveWeighting,
    TemperatureScheduler,
    WeightedWassersteinLoss,
)


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        (0, 0.1),
        (50, 0.2),
        (100, 0.3 + 0.6 / (1 + math.exp(5))),
        (300, 0.6),
        (500, 0.9),
        (1000, 0.9),
    ],
)
def test_buffer_weight_follows_ramp_sigmoid_plateau(
    epoch: int, expected: float
) -> None:
    """Linear ramp to 0.3, sigmoid to 0.9, then flat."""
    assert AdaptiveWeighting(warmup_epochs=100, stable_epoch=500).get_buffer_weight(
        epoch
    ) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("schedule", "epoch", "expected"),
    [
        ("exponential", 0, 1.0),
        ("exponential", 500, 0.1**0.5),
        ("exponential", 1000, 0.1),
        ("cosine", 0, 1.0),
        ("cosine", 500, 0.55),
        ("cosine", 1000, 0.1),
        ("constant", 700, 1.0),
    ],
)
def test_temperature_schedules(schedule: str, epoch: int, expected: float) -> None:
    """Exponential decays geometrically, cosine anneals, anything else holds init_temp."""
    scheduler = TemperatureScheduler(init_temp=1.0, final_temp=0.1, schedule=schedule)
    assert scheduler.get_temperature(epoch, max_epochs=1000) == pytest.approx(
        expected, abs=1e-9
    )


def _clouds(shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    targets = torch.randn(16, 2)
    return targets + shift, targets


@pytest.mark.parametrize(("shift", "expected"), [(0.0, 0.0), (1.0, 0.5), (2.0, 2.0)])
def test_translated_cloud_costs_half_the_squared_shift(
    shift: float, expected: float
) -> None:
    """Debiased Sinkhorn at p = 2 on a translation by s is s^2 / 2 in every dimension."""
    total, per_dim = WeightedWassersteinLoss()(*_clouds(shift))
    assert per_dim.shape == (2,)
    torch.testing.assert_close(per_dim, torch.full((2,), expected), atol=1e-5, rtol=0)
    assert total.item() == pytest.approx(expected, abs=1e-5)


def test_an_all_nan_dimension_contributes_zero() -> None:
    """A target column that is entirely NaN scores 0, so the total halves to 0.25."""
    predictions, targets = _clouds(1.0)
    targets[:, 1] = float("nan")
    total, per_dim = WeightedWassersteinLoss()(predictions, targets)
    assert per_dim[1].item() == 0.0
    assert per_dim[0].item() == pytest.approx(0.5, abs=1e-5)
    assert total.item() == pytest.approx(0.25, abs=1e-5)


def test_weights_turn_the_mean_into_a_weighted_sum() -> None:
    """Weights [1, 3] on two 0.5 dimensions give 1 * 0.5 + 3 * 0.5 = 2.0, not the mean 0.5."""
    weighted = WeightedWassersteinLoss(weights=torch.tensor([1.0, 3.0]))
    total, per_dim = weighted(*_clouds(1.0))
    torch.testing.assert_close(per_dim, torch.tensor([0.5, 0.5]), atol=1e-5, rtol=0)
    assert total.item() == pytest.approx(2.0, abs=1e-5)
