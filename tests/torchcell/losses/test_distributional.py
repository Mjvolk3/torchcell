"""Tests for the distributional decoder heads + proper-scoring-rule losses."""

import math

import pytest
import torch

from torchcell.losses.distributional import (
    dist_param_dim,
    gaussian_crps,
    make_dist_head,
    pinball,
)


@pytest.mark.parametrize(
    "mu,sigma,y", [(0.0, 1.0, 0.0), (0.5, 2.0, -1.0), (-1.3, 0.4, 2.0), (3.0, 1.5, 3.2)]
)
def test_gaussian_crps_matches_monte_carlo(mu: float, sigma: float, y: float) -> None:
    """Closed-form Gaussian CRPS matches the Monte-Carlo energy form E|X-y| - 0.5 E|X-X'|."""
    torch.manual_seed(0)
    n = 1_000_000
    x = mu + sigma * torch.randn(n)
    xp = mu + sigma * torch.randn(n)
    mc = (x - y).abs().mean() - 0.5 * (x - xp).abs().mean()
    cf = gaussian_crps(torch.tensor(mu), torch.tensor(sigma), torch.tensor(y))
    assert cf.item() == pytest.approx(mc.item(), abs=3e-3)


def test_gaussian_crps_minimized_at_true_params() -> None:
    """CRPS is a PROPER scoring rule: for samples from N(0,1), the expected CRPS is lower at
    the true (mu=0, sigma=1) than at a biased mean or an over-/under-dispersed sigma.
    """
    torch.manual_seed(0)
    y = torch.randn(200_000)
    true = gaussian_crps(torch.zeros_like(y), torch.ones_like(y), y).mean()
    biased_mu = gaussian_crps(torch.full_like(y, 0.5), torch.ones_like(y), y).mean()
    over = gaussian_crps(torch.zeros_like(y), torch.full_like(y, 2.0), y).mean()
    under = gaussian_crps(torch.zeros_like(y), torch.full_like(y, 0.5), y).mean()
    assert true < biased_mu
    assert true < over
    assert true < under


def test_pinball_median_equals_half_mae() -> None:
    """Pinball at tau=0.5 reduces to 0.5 * |error|."""
    taus = torch.tensor([0.5])
    q = torch.zeros(5, 1)
    y = torch.tensor([1.0, -2.0, 3.0, -0.5, 0.0])
    pb = pinball(q, y, taus).squeeze(-1)
    assert torch.allclose(pb, 0.5 * y.abs())


def test_pinball_asymmetry() -> None:
    """For tau > 0.5, under-prediction (y > q) is penalized more than over-prediction."""
    tau = torch.tensor([0.9])
    under = pinball(torch.zeros(1, 1), torch.tensor([1.0]), tau)  # y - q = +1
    over = pinball(torch.zeros(1, 1), torch.tensor([-1.0]), tau)  # y - q = -1
    assert under.item() == pytest.approx(0.9)
    assert over.item() == pytest.approx(0.1)


@pytest.mark.parametrize(
    "dist,expected_p", [("point", 1), ("crps", 2), ("quantile", 19)]
)
def test_dist_param_dim_and_head_width(dist: str, expected_p: int) -> None:
    """dist_param_dim and DistHead.param_dim agree with the head Linear-width multiplier."""
    assert dist_param_dim(dist) == expected_p
    assert make_dist_head(dist).param_dim == expected_p


def test_point_selects_mean_and_median() -> None:
    """`.point()` is identity (point), mu (gaussian), median quantile (quantile)."""
    dh_point = make_dist_head("point")
    params = torch.randn(2, 3)
    assert torch.equal(dh_point.point(params), params)

    dh_gauss = make_dist_head("crps")
    params = torch.randn(2, 3, 2)
    assert torch.equal(dh_gauss.point(params), params[..., 0])

    dh_quant = make_dist_head("quantile")
    assert dh_quant.median_index == 9
    assert dh_quant.taus[dh_quant.median_index].item() == pytest.approx(0.5)
    params = torch.randn(2, 3, 19)
    assert torch.equal(dh_quant.point(params), params[..., dh_quant.median_index])


@pytest.mark.parametrize("dist", ["point", "crps", "quantile"])
def test_loss_masking_and_gradient(dist: str) -> None:
    """Loss respects the row mask, returns a connected zero when no row is supervised, and
    backprops for every mode.
    """
    torch.manual_seed(0)
    b, f = 4, 6
    dh = make_dist_head(dist)
    p = dist_param_dim(dist)
    params = (torch.randn(b, f) if p == 1 else torch.randn(b, f, p)).requires_grad_(
        True
    )
    target = torch.randn(b, f)

    # Empty mask -> exact zero, still connected (grad path exists).
    empty = dh.loss(params, target, torch.zeros(b, dtype=torch.bool))
    assert empty.item() == 0.0
    empty.backward(retain_graph=True)

    # Partial mask == loss on the selected rows only.
    mask = torch.tensor([True, False, True, True])
    masked = dh.loss(params, target, mask)
    direct = dh.loss(params[mask], target[mask], None)
    assert masked.item() == pytest.approx(direct.item())
    assert masked.item() > 0.0
    params.grad = None
    masked.backward()
    assert params.grad is not None and torch.isfinite(params.grad).all()


def test_crps_no_sigma_collapse() -> None:
    """Unlike Gaussian NLL, CRPS does not diverge to -inf as sigma -> 0 (finite, >= 0)."""
    y = torch.tensor([1.0])
    for s in (1e-4, 1e-2, 1.0, 10.0):
        c = gaussian_crps(torch.zeros(1), torch.full((1,), s), y)
        assert torch.isfinite(c).all()
        assert c.item() >= 0.0


def test_dist_head_to_device_cpu() -> None:
    """Taus buffer moves with the module (state uniform across modes)."""
    dh = make_dist_head("quantile").to("cpu")
    assert dh.taus.numel() == 19
    dh_point = make_dist_head("point")
    assert dh_point.taus.numel() == 0
    assert dh_point.param_dim == 1


def test_gaussian_crps_constant_at_zero() -> None:
    """At z=0 (y==mu), CRPS = sigma*(2*phi(0) - 1/sqrt(pi)) = sigma*(sqrt(2)-1)/sqrt(pi)."""
    sigma = 1.7
    expected = sigma * (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    c = gaussian_crps(torch.zeros(1), torch.full((1,), sigma), torch.zeros(1))
    assert c.item() == pytest.approx(expected, abs=1e-6)
