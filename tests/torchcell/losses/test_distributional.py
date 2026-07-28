"""Tests for the distributional decoder heads + proper-scoring-rule losses."""

import math

import pytest
import torch

from torchcell.losses.distributional import (
    DIST_TO_MODE,
    DistHead,
    coverage,
    dist_param_dim,
    energy_score,
    gaussian_crps,
    gaussian_nll,
    laplace_crps,
    make_dist_head,
    pinball,
    pit_ks,
    pit_values,
)

# Every `dist` config value, and the F the `energy` head needs to size its global V.
ALL_DISTS = ["point", "crps", "quantile", "laplace_crps", "nll", "energy"]
PROBABILISTIC_DISTS = ["crps", "quantile", "laplace_crps", "nll", "energy"]
NUM_FEATURES = 6


def build_head(dist: str, num_features: int = NUM_FEATURES) -> DistHead:
    """make_dist_head with the extra arg `energy` needs (ignored by every other mode)."""
    return make_dist_head(dist, num_features=num_features)


def ks_statistic(pit: torch.Tensor) -> float:
    """KS distance to Uniform(0, 1) -- the library function, as a float."""
    return float(pit_ks(pit).item())


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
    "dist,expected_p",
    [
        ("point", 1),
        ("crps", 2),
        ("quantile", 19),
        ("laplace_crps", 2),
        ("nll", 2),
        ("energy", 2),
    ],
)
def test_dist_param_dim_and_head_width(dist: str, expected_p: int) -> None:
    """dist_param_dim and DistHead.param_dim agree with the head Linear-width multiplier."""
    assert dist_param_dim(dist) == expected_p
    assert build_head(dist).param_dim == expected_p


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


@pytest.mark.parametrize("dist", ["laplace_crps", "nll", "energy"])
def test_point_is_mu_for_new_parametric_modes(dist: str) -> None:
    """`.point()` is mu for laplace (the MEDIAN), nll_gaussian and energy (the MEAN)."""
    params = torch.randn(2, NUM_FEATURES, 2)
    assert torch.equal(build_head(dist).point(params), params[..., 0])


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_loss_masking_and_gradient(dist: str) -> None:
    """Loss respects the row mask, returns a connected zero when no row is supervised, and
    backprops for every mode.
    """
    torch.manual_seed(0)
    b, f = 4, NUM_FEATURES
    dh = build_head(dist, f)
    p = dist_param_dim(dist)
    params = (torch.randn(b, f) if p == 1 else torch.randn(b, f, p)).requires_grad_(
        True
    )
    target = torch.randn(b, f)

    # Empty mask -> exact zero, still connected (grad path exists).
    empty = dh.loss(params, target, torch.zeros(b, dtype=torch.bool))
    assert empty.item() == 0.0
    empty.backward(retain_graph=True)

    # Partial mask == loss on the selected rows only. `energy` draws samples, so the RNG is
    # pinned to the same state for both calls to make the two paths comparable.
    mask = torch.tensor([True, False, True, True])
    torch.manual_seed(1)
    masked = dh.loss(params, target, mask)
    torch.manual_seed(1)
    direct = dh.loss(params[mask], target[mask], None)
    assert masked.item() == pytest.approx(direct.item())
    # nll_gaussian is a log-density, so it may be negative; every other mode is a distance.
    assert torch.isfinite(masked).all()
    assert masked.item() != 0.0
    if dist != "nll":
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


# ---------------------------------------------------------------------------
# laplace_crps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mu,b,y", [(0.0, 1.0, 0.0), (0.5, 2.0, -1.0), (-1.3, 0.4, 2.0), (3.0, 1.5, 3.2)]
)
def test_laplace_crps_matches_monte_carlo(mu: float, b: float, y: float) -> None:
    """Closed-form Laplace CRPS matches the energy form E|X-y| - 0.5 E|X-X'|, X ~ Laplace."""
    torch.manual_seed(0)
    n = 2_000_000
    dist = torch.distributions.Laplace(torch.tensor(mu), torch.tensor(b))
    x = dist.sample((n,))
    xp = dist.sample((n,))
    mc = (x - y).abs().mean() - 0.5 * (x - xp).abs().mean()
    cf = laplace_crps(torch.tensor(mu), torch.tensor(b), torch.tensor(y))
    assert cf.item() == pytest.approx(mc.item(), abs=5e-3)


def test_laplace_crps_at_the_median() -> None:
    """At y == mu the closed form collapses to b*(1 - 3/4) = b/4."""
    b = 1.7
    c = laplace_crps(torch.zeros(1), torch.full((1,), b), torch.zeros(1))
    assert c.item() == pytest.approx(0.25 * b, abs=1e-6)


def test_laplace_crps_no_scale_collapse() -> None:
    """Laplace CRPS stays finite and non-negative as b -> 0 (no 1/b^2 term)."""
    y = torch.tensor([1.0])
    for b in (1e-4, 1e-2, 1.0, 10.0):
        c = laplace_crps(torch.zeros(1), torch.full((1,), b), y)
        assert torch.isfinite(c).all()
        assert c.item() >= 0.0


# ---------------------------------------------------------------------------
# gaussian_nll
# ---------------------------------------------------------------------------


def test_gaussian_nll_matches_torch_reference_up_to_the_constant() -> None:
    """Our NLL equals torch's, which also drops the 0.5*log(2*pi) constant."""
    torch.manual_seed(0)
    mu = torch.randn(64)
    sigma = torch.rand(64) + 0.2
    y = torch.randn(64)
    ours = gaussian_nll(mu, sigma, y)
    ref = torch.nn.functional.gaussian_nll_loss(
        mu, y, sigma * sigma, full=False, eps=0.0, reduction="none"
    )
    assert torch.allclose(ours, ref, atol=1e-6)


def test_gaussian_nll_diverges_as_sigma_shrinks() -> None:
    """The 1/sigma^2 term is real: NLL -> -inf as sigma -> 0 at a perfectly fit point.

    This is the contrast with `test_crps_no_sigma_collapse` and the reason `nll` is the
    negative control of the sweep rather than a candidate default.
    """
    y = torch.zeros(1)
    values = [
        gaussian_nll(torch.zeros(1), torch.full((1,), s), y).item()
        for s in (1.0, 1e-1, 1e-2, 1e-3)
    ]
    assert values == sorted(values, reverse=True)
    assert values[-1] < -6.0


def test_gaussian_nll_shrinks_the_mu_gradient_as_sigma_grows() -> None:
    """The mu-gradient d/dmu = -(y - mu)/sigma^2: inflating sigma silences a hard point.

    The mechanism behind sigma-collapse (Seitzer et al. 2022) -- verified directly rather
    than assumed, since `nll` is included precisely so the effect can be measured.
    """
    y = torch.tensor([3.0])
    grads = []
    for s in (0.5, 1.0, 4.0):
        mu = torch.zeros(1, requires_grad=True)
        gaussian_nll(mu, torch.full((1,), s), y).backward()
        assert mu.grad is not None
        grads.append(abs(mu.grad.item()))
    assert grads[0] > grads[1] > grads[2]
    # CRPS's gradient on mu is bounded by 1 and does NOT vanish with sigma.
    mu = torch.zeros(1, requires_grad=True)
    gaussian_crps(mu, torch.full((1,), 4.0), y).backward()
    assert mu.grad is not None
    assert abs(mu.grad.item()) > grads[2]


# ---------------------------------------------------------------------------
# energy_score -- the oracle, unbiasedness, propriety, joint modelling
# ---------------------------------------------------------------------------


def test_energy_score_reduces_to_gaussian_crps_at_one_feature() -> None:
    """ORACLE: at F=1 and k=0 the energy score IS CRPS, up to Monte-Carlo error.

    The energy score is the multivariate generalization of CRPS, so in one dimension the
    ||.||_2 collapses to |.| and the estimator must reproduce the closed form that
    `test_gaussian_crps_matches_monte_carlo` already pins to Monte Carlo. This is the
    single most important check on `energy_score`.
    """
    torch.manual_seed(0)
    b, m = 4, 4096
    mu = torch.randn(b, 1)
    sigma = torch.rand(b, 1) + 0.5
    y = torch.randn(b, 1)
    with torch.no_grad():
        es = energy_score(mu, sigma, None, y, num_samples=m)
    cf = gaussian_crps(mu, sigma, y).squeeze(-1)
    assert es.shape == (b,)
    # Tolerances are Monte-Carlo error, not slack: the worst per-row deviation seen over 12
    # seeds at m=4096 is ~0.05 and the worst batch-mean deviation ~0.011.
    assert torch.allclose(es, cf, atol=8e-2)
    assert es.mean().item() == pytest.approx(cf.mean().item(), abs=3e-2)


def test_energy_score_is_unbiased_in_num_samples() -> None:
    """m=2 and m=512 agree in MEAN; only the variance differs. m trades noise, not bias."""
    torch.manual_seed(1)
    mu = torch.zeros(20_000, 3)
    sigma = torch.full((20_000, 3), 1.5)
    y = torch.full((20_000, 3), 0.4)
    with torch.no_grad():
        few = energy_score(mu, sigma, None, y, num_samples=2)
        many = energy_score(mu[:400], sigma[:400], None, y[:400], num_samples=512)
    assert few.mean().item() == pytest.approx(many.mean().item(), abs=2e-2)
    # ... and the variance really is the thing that changes.
    assert few.std().item() > 5.0 * many.std().item()


def test_energy_score_pairwise_denominator_is_m_times_m_minus_one() -> None:
    """The m(m-1) denominator is load-bearing: an m^2 denominator biases the score.

    Keeping the m zero self-distances shrinks the subtracted term by (m-1)/m, inflating the
    score by E||X-X'||/(2m) -- a spread-proportional surcharge that would push the optimum
    toward an over-sharp predictive distribution, worst at small m.
    """
    torch.manual_seed(2)
    n, m = 20_000, 2
    mu = torch.zeros(n, 3)
    sigma = torch.full((n, 3), 1.5)
    y = torch.full((n, 3), 0.4)
    with torch.no_grad():
        unbiased = energy_score(mu, sigma, None, y, num_samples=m).mean()
        reference = energy_score(mu[:400], sigma[:400], None, y[:400], 512).mean()
        # Reproduce the m^2 variant by hand on the same estimator shape.
        x = (mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(m, n, 3)).permute(
            1, 0, 2
        )
        t1 = torch.linalg.vector_norm(x - y.unsqueeze(1), dim=-1).mean(dim=1)
        biased = (t1 - torch.cdist(x, x).sum(dim=(1, 2)) / (2.0 * m * m)).mean()
    assert unbiased.item() == pytest.approx(reference.item(), abs=2e-2)
    assert biased.item() > reference.item() + 0.5


def test_energy_score_gradients_flow_through_the_samples() -> None:
    """Reparameterized sampling: mu, sigma and V all receive finite, non-zero gradients."""
    torch.manual_seed(3)
    b, f, k = 8, 5, 3
    mu = torch.randn(b, f, requires_grad=True)
    raw_sigma = torch.rand(b, f, requires_grad=True)
    v = torch.randn(f, k, requires_grad=True)
    y = torch.randn(b, f)
    energy_score(mu, raw_sigma.abs() + 0.1, v, y, num_samples=16).mean().backward()
    for g in (mu.grad, raw_sigma.grad, v.grad):
        assert g is not None
        assert torch.isfinite(g).all()
        assert g.abs().sum().item() > 0.0


def test_energy_score_rejects_degenerate_inputs() -> None:
    """Degenerate input is rejected: m >= 2 (the pairwise term needs a pair), mu is [B, F]."""
    mu = torch.zeros(2, 3)
    sigma = torch.ones(2, 3)
    y = torch.zeros(2, 3)
    with pytest.raises(AssertionError):
        energy_score(mu, sigma, None, y, num_samples=1)
    with pytest.raises(AssertionError):
        energy_score(mu.unsqueeze(0), sigma.unsqueeze(0), None, y.unsqueeze(0), 4)
    with pytest.raises(AssertionError):
        energy_score(mu, sigma, torch.randn(7, 2), y, 4)


@pytest.mark.parametrize("k", [0, 4])
def test_energy_score_zero_rank_matches_none(k: int) -> None:
    """k=0 is the diagonal-only ablation, expressible as V=None or as a [F, 0] factor."""
    torch.manual_seed(4)
    mu = torch.randn(6, 5)
    sigma = torch.rand(6, 5) + 0.5
    y = torch.randn(6, 5)
    v = torch.zeros(5, k)
    torch.manual_seed(9)
    with torch.no_grad():
        with_v = energy_score(mu, sigma, v, y, num_samples=32)
    torch.manual_seed(9)
    with torch.no_grad():
        without = energy_score(mu, sigma, None, y, num_samples=32)
    # A zero-column V is skipped outright; an all-zero V of k>0 columns draws z but adds
    # exactly 0. Either way the eps stream is shared, so the scores agree to float noise.
    assert torch.allclose(with_v, without, atol=1e-6)


def test_energy_score_prefers_a_correlated_v_on_correlated_data() -> None:
    """JOINT MODELLING TEST. With correlated targets, matching the correlation with V beats
    a diagonal family with the SAME marginals -- so k>0 buys something k=0 cannot.

    Both arms have identical per-feature marginal variances; the only difference is whether
    the predictive family can represent the cross-feature correlation.
    """
    torch.manual_seed(5)
    n_feat, b, m = 10, 256, 128
    v_true = torch.ones(n_feat, 1)
    diag = 0.1
    y = torch.randn(b, 1) @ v_true.T + diag * torch.randn(b, n_feat)
    marginal = math.sqrt(diag**2 + 1.0)
    with torch.no_grad():
        correlated = energy_score(
            torch.zeros(b, n_feat), torch.full((b, n_feat), diag), v_true, y, m
        ).mean()
        independent = energy_score(
            torch.zeros(b, n_feat), torch.full((b, n_feat), marginal), None, y, m
        ).mean()
    assert correlated.item() < independent.item()


# ---------------------------------------------------------------------------
# Propriety, empirically: the minimizing scale is the true scale
# ---------------------------------------------------------------------------

_SCALE_GRID = [0.25, 0.5, 0.75, 1.0, 1.3, 1.7, 2.2, 3.0, 5.0]


def test_gaussian_crps_propriety_scan() -> None:
    """Scoring N(0, 1.3) data with candidate sigmas: the minimum sits at the true sigma."""
    torch.manual_seed(6)
    s_true = 1.3
    y = s_true * torch.randn(200_000)
    scores = [
        gaussian_crps(torch.zeros_like(y), torch.full_like(y, s), y).mean().item()
        for s in _SCALE_GRID
    ]
    best = _SCALE_GRID[int(torch.tensor(scores).argmin().item())]
    assert best == pytest.approx(s_true)
    assert scores[0] > scores[4] and scores[-1] > scores[4]


def test_laplace_crps_propriety_scan() -> None:
    """Scoring Laplace(0, 1.3) data with candidate b: the minimum sits at the true b.

    Family-matched on purpose -- propriety says the TRUE distribution minimizes the
    expected score, so the data must come from a Laplace for the optimum to be b_true.
    """
    torch.manual_seed(7)
    b_true = 1.3
    y = torch.distributions.Laplace(0.0, b_true).sample((200_000,))
    scores = [
        laplace_crps(torch.zeros_like(y), torch.full_like(y, s), y).mean().item()
        for s in _SCALE_GRID
    ]
    best = _SCALE_GRID[int(torch.tensor(scores).argmin().item())]
    assert best == pytest.approx(b_true)
    assert scores[0] > scores[4] and scores[-1] > scores[4]


def test_energy_score_propriety_scan() -> None:
    """Scoring N(0, 1.3 I) data with candidate sigmas: the minimum sits at the true sigma."""
    torch.manual_seed(8)
    s_true = 1.3
    n, f = 400, 4
    y = s_true * torch.randn(n, f)
    with torch.no_grad():
        scores = [
            energy_score(
                torch.zeros(n, f), torch.full((n, f), s), None, y, num_samples=256
            )
            .mean()
            .item()
            for s in _SCALE_GRID
        ]
    best = _SCALE_GRID[int(torch.tensor(scores).argmin().item())]
    assert best == pytest.approx(s_true)
    assert scores[0] > scores[4] and scores[-1] > scores[4]


# ---------------------------------------------------------------------------
# PIT + coverage
# ---------------------------------------------------------------------------


def test_pit_gaussian_uniform_when_calibrated() -> None:
    """Correctly specified -> PIT ~ Uniform(0,1): mean 0.5, flat CDF, coverage == alpha."""
    torch.manual_seed(10)
    n = 200_000
    y = torch.randn(n)
    pit = pit_values("gaussian", y, mu=torch.zeros(n), scale=torch.ones(n))
    assert pit.min().item() >= 0.0 and pit.max().item() <= 1.0
    assert pit.mean().item() == pytest.approx(0.5, abs=5e-3)
    assert ks_statistic(pit) < 0.01
    for alpha in (0.5, 0.8, 0.95):
        assert coverage(pit, alpha).item() == pytest.approx(alpha, abs=0.01)


def test_pit_u_shaped_when_intervals_too_narrow() -> None:
    """Sigma too SMALL -> mass piles in the tails (U-shape) and coverage falls below alpha."""
    torch.manual_seed(11)
    n = 200_000
    y = torch.randn(n)
    pit = pit_values("gaussian", y, mu=torch.zeros(n), scale=torch.full((n,), 0.5))
    tails = ((pit < 0.1) | (pit > 0.9)).float().mean().item()
    middle = ((pit > 0.4) & (pit < 0.6)).float().mean().item()
    assert tails > 0.4  # uniform would be 0.2
    assert middle < 0.15  # uniform would be 0.2
    assert ks_statistic(pit) > 0.1
    assert coverage(pit, 0.8).item() < 0.6


def test_pit_hump_shaped_when_intervals_too_wide() -> None:
    """Sigma too LARGE -> mass concentrates at 0.5 (n-shape), over-coverage.

    This is the sigma-collapse signature the `nll` arm is expected to show.
    """
    torch.manual_seed(12)
    n = 200_000
    y = torch.randn(n)
    pit = pit_values("gaussian", y, mu=torch.zeros(n), scale=torch.full((n,), 2.0))
    tails = ((pit < 0.1) | (pit > 0.9)).float().mean().item()
    middle = ((pit > 0.4) & (pit < 0.6)).float().mean().item()
    assert tails < 0.05
    assert middle > 0.3
    assert coverage(pit, 0.8).item() > 0.95


def test_pit_skewed_when_the_mean_is_biased() -> None:
    """A biased mu shifts PIT mass off-centre without changing its dispersion much."""
    torch.manual_seed(13)
    n = 200_000
    y = torch.randn(n)
    pit = pit_values("gaussian", y, mu=torch.full((n,), 1.0), scale=torch.ones(n))
    assert pit.mean().item() < 0.4
    assert (pit < 0.5).float().mean().item() > 0.7


def test_pit_laplace_uniform_when_calibrated() -> None:
    """The Laplace CDF branch is calibrated on Laplace data (and is monotone in y)."""
    torch.manual_seed(14)
    n = 200_000
    b = 0.7
    y = torch.distributions.Laplace(0.3, b).sample((n,))
    pit = pit_values("laplace", y, mu=torch.full((n,), 0.3), scale=torch.full((n,), b))
    assert pit.mean().item() == pytest.approx(0.5, abs=5e-3)
    assert ks_statistic(pit) < 0.01
    assert coverage(pit, 0.8).item() == pytest.approx(0.8, abs=0.01)
    # Sanity on the two branches meeting at y == mu.
    mid = pit_values("laplace", torch.zeros(1), mu=torch.zeros(1), scale=torch.ones(1))
    assert mid.item() == pytest.approx(0.5)


def test_pit_energy_uniform_when_calibrated() -> None:
    """The sample-based branch recovers uniform PIT from a correctly specified head."""
    torch.manual_seed(15)
    n, f, m = 4000, 4, 256
    y = torch.randn(n, f)
    dh = DistHead("energy", num_features=f, rank=0)
    raw = torch.zeros(n, f, 2)
    # softplus(raw)+floor == sigma; solve raw so that sigma == 1.
    raw[..., 1] = math.log(math.expm1(1.0 - dh.sigma_floor))
    pit = dh.pit(raw, y, num_samples=m)
    assert pit.shape == (n, f)
    assert pit.mean().item() == pytest.approx(0.5, abs=1e-2)
    assert ks_statistic(pit) < 0.02
    assert coverage(pit, 0.8).item() == pytest.approx(0.8, abs=0.02)


def test_pit_quantile_interpolates_the_knots() -> None:
    """The quantile branch interpolates tau through the knots and clamps outside them."""
    dh = make_dist_head("quantile")
    taus = dh.taus
    knots = torch.linspace(-2.0, 2.0, 19).expand(3, 2, 19).contiguous()
    # A target exactly on a knot returns that knot's tau.
    for j in (0, 5, 9, 18):
        y = knots[..., j]
        pit = pit_values("quantile", y, quantiles=knots, taus=taus)
        assert torch.allclose(pit, torch.full_like(pit, taus[j].item()), atol=1e-6)
    # Halfway between two knots returns the midpoint tau.
    y = 0.5 * (knots[..., 3] + knots[..., 4])
    pit = pit_values("quantile", y, quantiles=knots, taus=taus)
    expected = 0.5 * (taus[3] + taus[4])
    assert torch.allclose(pit, torch.full_like(pit, expected.item()), atol=1e-6)
    # Outside the knot range the grid carries no information: clamp to 0 / 1.
    below = pit_values("quantile", torch.full((3, 2), -9.0), quantiles=knots, taus=taus)
    above = pit_values("quantile", torch.full((3, 2), 9.0), quantiles=knots, taus=taus)
    assert torch.equal(below, torch.zeros_like(below))
    assert torch.equal(above, torch.ones_like(above))


def test_pit_quantile_repairs_crossing() -> None:
    """Unconstrained heads can emit non-monotone quantiles; the PIT sorts them first."""
    taus = torch.linspace(0.05, 0.95, 3)
    crossed = torch.tensor([[[1.0, -1.0, 0.0]]])
    ordered = torch.tensor([[[-1.0, 0.0, 1.0]]])
    y = torch.zeros(1, 1)
    assert torch.equal(
        pit_values("quantile", y, quantiles=crossed, taus=taus),
        pit_values("quantile", y, quantiles=ordered, taus=taus),
    )


def test_pit_rejects_point_mode() -> None:
    """`point` has no predictive distribution, so it has no PIT."""
    with pytest.raises(ValueError):
        pit_values("point", torch.zeros(3), mu=torch.zeros(3), scale=torch.ones(3))


def test_coverage_endpoints() -> None:
    """Coverage counts the central alpha band of PIT values."""
    pit = torch.linspace(0.0, 1.0, 1001)
    assert coverage(pit, 0.8).item() == pytest.approx(0.8, abs=2e-3)
    assert coverage(torch.full((100,), 0.5), 0.01).item() == 1.0
    assert coverage(torch.zeros(100), 0.5).item() == 0.0
    with pytest.raises(AssertionError):
        coverage(pit, 1.0)


@pytest.mark.parametrize("dist", PROBABILISTIC_DISTS)
def test_dist_head_pit_shapes_and_mask(dist: str) -> None:
    """Every probabilistic head returns PIT [B, F] in [0, 1] and honours the row mask."""
    torch.manual_seed(16)
    b, f = 7, NUM_FEATURES
    dh = build_head(dist, f)
    p = dist_param_dim(dist)
    params = torch.randn(b, f, p)
    target = torch.randn(b, f)
    pit = dh.pit(params, target)
    assert pit.shape == (b, f)
    assert pit.min().item() >= 0.0 and pit.max().item() <= 1.0
    mask = torch.zeros(b, dtype=torch.bool)
    mask[[0, 2, 5]] = True
    assert dh.pit(params, target, mask).shape == (3, f)


# ---------------------------------------------------------------------------
# DistHead wiring for the new modes
# ---------------------------------------------------------------------------


def test_dist_to_mode_covers_every_config_value() -> None:
    """DIST_TO_MODE, dist_param_dim and DistHead.VALID_MODES stay in lockstep."""
    assert set(DIST_TO_MODE) == set(ALL_DISTS)
    assert set(DIST_TO_MODE.values()) == set(DistHead.VALID_MODES)
    for dist in ALL_DISTS:
        assert dist_param_dim(dist) == build_head(dist).param_dim


def test_energy_head_owns_a_global_v() -> None:
    """V is a head-owned nn.Parameter [F, k], NOT per-row output, so param_dim stays 2."""
    f, k = NUM_FEATURES, 4
    dh = make_dist_head("energy", num_features=f, rank=k)
    v = dh.v
    assert v is not None
    assert v.shape == (f, k)
    assert "_v" in dict(dh.named_parameters())
    assert dh.param_dim == 2
    # It trains: a backward through the loss reaches V.
    params = torch.randn(5, f, 2, requires_grad=True)
    dh.loss(params, torch.randn(5, f)).backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert v.grad.abs().sum().item() > 0.0


def test_energy_head_rank_zero_is_the_diagonal_ablation() -> None:
    """rank=0 -> no V at all: the score still couples features, the family cannot."""
    dh = make_dist_head("energy", num_features=NUM_FEATURES, rank=0)
    assert dh.v is None
    assert dh.param_dim == 2
    assert list(dh.parameters()) == []
    params = torch.randn(5, NUM_FEATURES, 2, requires_grad=True)
    loss = dh.loss(params, torch.randn(5, NUM_FEATURES))
    loss.backward()
    assert params.grad is not None and torch.isfinite(params.grad).all()


def test_energy_head_requires_num_features() -> None:
    """The head cannot size V without F, so `energy` must be given num_features."""
    with pytest.raises(AssertionError):
        make_dist_head("energy")


def test_energy_head_samples_shape_and_covariance() -> None:
    """DistHead.samples draws [B, m, F] from N(mu, diag(sigma^2) + V V^T)."""
    torch.manual_seed(17)
    f, k, b, m = 3, 2, 4, 20_000
    dh = DistHead("energy", num_features=f, rank=k, v_init=1.0)
    params = torch.zeros(b, f, 2)
    params[..., 1] = math.log(math.expm1(0.5 - dh.sigma_floor))  # sigma == 0.5
    x = dh.samples(params, m)
    assert x.shape == (b, m, f)
    v = dh.v
    assert v is not None
    expected = torch.diag(torch.full((f,), 0.25)) + v @ v.T
    empirical = torch.cov(x[0].T)
    assert torch.allclose(empirical, expected, atol=0.05)


def test_new_heads_scale_is_strictly_positive() -> None:
    """Laplace `b` and gaussian `sigma` share one softplus+floor map, so both stay > 0."""
    for dist in ("crps", "laplace_crps", "nll", "energy"):
        dh = build_head(dist)
        raw = torch.full((3, NUM_FEATURES, 2), -50.0)
        scale = dh._sigma(raw[..., 1])
        assert (scale > 0.0).all()
        assert scale.min().item() == pytest.approx(dh.sigma_floor, abs=1e-6)


def test_laplace_and_nll_head_losses_match_the_free_functions() -> None:
    """The head applies softplus+floor and then the documented closed form -- nothing else."""
    torch.manual_seed(18)
    b, f = 6, NUM_FEATURES
    params = torch.randn(b, f, 2)
    target = torch.randn(b, f)
    for dist, fn in (("laplace_crps", laplace_crps), ("nll", gaussian_nll)):
        dh = build_head(dist)
        expected = fn(params[..., 0], dh._sigma(params[..., 1]), target).mean()
        assert dh.loss(params, target).item() == pytest.approx(expected.item())


# ---------------------------------------------------------------------------
# pit_ks -- the scalar calibration summary
# ---------------------------------------------------------------------------


def test_pit_ks_is_zero_on_an_exactly_uniform_sample() -> None:
    """The midpoint grid (i-0.5)/N is the best possible uniform sample: D = 1/(2N)."""
    n = 1000
    p = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    assert pit_ks(p).item() == pytest.approx(0.5 / n, abs=1e-9)


def test_pit_ks_is_two_sided() -> None:
    """A sample entirely BELOW uniform and one entirely ABOVE score the same distance.

    A one-sided max(i/N - p) would report ~0 for the all-ones case; the sup-norm must not.
    """
    n = 500
    lo = pit_ks(torch.zeros(n))
    hi = pit_ks(torch.ones(n))
    assert lo.item() == pytest.approx(1.0, abs=1.0 / n)
    assert hi.item() == pytest.approx(1.0, abs=1.0 / n)


def test_pit_ks_matches_scipy_kstest() -> None:
    """Oracle: the same statistic scipy computes for the uniform null."""
    scipy_stats = pytest.importorskip("scipy.stats")
    torch.manual_seed(21)
    for sample in (torch.rand(2000), torch.rand(2000) ** 2, torch.rand(500) * 0.6):
        expected = scipy_stats.kstest(sample.numpy(), "uniform").statistic
        assert pit_ks(sample).item() == pytest.approx(float(expected), abs=1e-6)


def test_pit_ks_grows_with_miscalibration() -> None:
    """Monotone in how far the predictive sigma is from the truth."""
    torch.manual_seed(22)
    y = torch.randn(4000)
    mu = torch.zeros_like(y)
    ds = [
        pit_ks(pit_values("gaussian", y, mu=mu, scale=torch.full_like(y, s))).item()
        for s in (1.0, 0.7, 0.4, 0.2)
    ]
    assert ds == sorted(ds), f"KS should increase as sigma shrinks below 1: {ds}"


def test_pit_ks_quantile_grid_has_a_structural_floor() -> None:
    """WHY pit_ks is within-head only: a PERFECT quantile head still scores D ~ 0.05.

    The tau in [0.05, 0.95] grid carries no information outside its knots, so a calibrated
    head puts ~5% of PIT mass exactly at 0 and ~5% exactly at 1. That is a property of the
    grid, and it swamps any honest cross-mode comparison against a parametric head.
    """
    torch.manual_seed(23)
    n = 4000
    y = torch.randn(n, 1)
    taus = torch.linspace(0.05, 0.95, 19)
    # EXACTLY calibrated knots: the true standard-normal quantiles, same for every row.
    knots = torch.distributions.Normal(0.0, 1.0).icdf(taus).expand(n, 1, 19)
    d_quantile = pit_ks(pit_values("quantile", y, quantiles=knots, taus=taus)).item()
    d_gaussian = pit_ks(
        pit_values("gaussian", y, mu=torch.zeros_like(y), scale=torch.ones_like(y))
    ).item()
    assert d_quantile == pytest.approx(0.05, abs=0.015)
    assert d_gaussian < 0.03
    assert d_quantile > d_gaussian
