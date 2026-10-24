# tests/torchcell/losses/test_hl_gauss.py
# [[tests.torchcell.losses.test_hl_gauss]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_hl_gauss.py
"""HL-Gauss loss and per-dataset standardization (``torchcell.losses.hl_gauss``)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchcell.losses.hl_gauss import (
    DatasetStandardization,
    HLGaussConfig,
    HLGaussLoss,
    StandardizationStats,
    fit_standardization,
)


def test_config_defaults_match_decision_o4() -> None:
    cfg = HLGaussConfig()
    assert cfg.num_bins == 32
    assert cfg.bin_width == pytest.approx(0.25)
    assert cfg.sigma == pytest.approx(0.1875)


def test_config_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="low < high"):
        HLGaussConfig(low=1.0, high=-1.0)


def test_soft_target_sums_to_one_and_peaks_at_target_bin() -> None:
    loss = HLGaussLoss()
    target = torch.tensor([0.1, -3.9, 7.0])  # interior, edge, beyond the range
    probs, measured = loss.soft_target(target)
    assert probs.shape == (3, 32)
    assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-6)
    assert measured.all()
    # 0.1 lies in bin 16 ([0, 0.25)); -3.9 in bin 0; 7.0 is all mass in the open last bin
    assert int(probs[0].argmax()) == 16
    assert int(probs[1].argmax()) == 0
    assert float(probs[2, -1]) == pytest.approx(1.0, abs=1e-6)


def test_soft_target_mass_matches_normal_cdf() -> None:
    loss = HLGaussLoss()
    probs, _ = loss.soft_target(torch.tensor([0.0]))
    sigma = 0.1875
    # bin 16 is [0, 0.25): Phi(0.25/sigma) - Phi(0)
    expected = 0.5 * math.erf(0.25 / (sigma * math.sqrt(2.0)))
    assert float(probs[0, 16]) == pytest.approx(expected, rel=1e-5)


def test_missing_entries_carry_no_value_and_no_gradient() -> None:
    loss = HLGaussLoss()
    logits = torch.randn(2, 5, 32, requires_grad=True)
    target = torch.tensor(
        [[0.0, float("nan"), 1.0, float("nan"), -2.0], [float("nan")] * 5]
    )
    value = loss(logits, target)
    value.backward()
    assert torch.isfinite(value)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # rows of missing entries get exactly zero gradient
    assert float(logits.grad[0, 1].abs().sum()) == 0.0
    assert float(logits.grad[1].abs().sum()) == 0.0
    assert float(logits.grad[0, 0].abs().sum()) > 0.0
    # the mean is over the 3 measured entries only
    ce, measured = loss.elementwise(logits.detach(), target)
    assert int(measured.sum()) == 3
    assert float(value.detach()) == pytest.approx(float(ce.sum() / 3), rel=1e-6)


def test_all_missing_returns_connected_zero() -> None:
    loss = HLGaussLoss()
    logits = torch.randn(4, 32, requires_grad=True)
    value = loss(logits, torch.full((4,), float("nan")))
    assert float(value) == 0.0
    value.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) == 0.0


def test_scalar_shape_and_shape_check() -> None:
    loss = HLGaussLoss()
    assert loss(torch.zeros(3, 32), torch.zeros(3)).shape == ()
    with pytest.raises(ValueError, match="logits must be"):
        loss(torch.zeros(3, 31), torch.zeros(3))


def test_expectation_entropy_and_accuracy_readouts() -> None:
    loss = HLGaussLoss()
    uniform = torch.zeros(1, 32)
    assert float(loss.expectation(uniform)[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(loss.entropy(uniform)[0]) == pytest.approx(math.log(32), rel=1e-6)
    peaked = torch.full((3, 32), -1e4)
    peaked[:, 20] = 0.0  # center of bin 20 = -4 + 20.5 * 0.25 = 1.125
    assert torch.allclose(loss.expectation(peaked), torch.full((3,), 1.125))
    assert float(loss.entropy(peaked).max()) < 1e-3
    target = torch.tensor([1.1, 1.4, float("nan")])  # bin 20, bin 21, missing
    hits, measured = loss.accuracy_within_one(peaked, target)
    assert hits.tolist() == [True, True, False]
    assert measured.tolist() == [True, True, False]


def test_minimizing_the_loss_recovers_the_target() -> None:
    torch.manual_seed(0)
    loss = HLGaussLoss()
    logits = torch.zeros(4, 32, requires_grad=True)
    target = torch.tensor([-1.3, 0.0, 0.7, 2.2])
    opt = torch.optim.Adam([logits], lr=0.2)
    for _ in range(300):
        opt.zero_grad()
        loss(logits, target).backward()
        opt.step()
    assert torch.allclose(loss.expectation(logits.detach()), target, atol=0.05)


def test_fit_standardization_pooled_and_centered() -> None:
    y = np.array([[1.0, 10.0, np.nan], [3.0, 14.0, np.nan], [5.0, np.nan, np.nan]])
    pooled = fit_standardization("d", y, center_per_gene=False, transform="raw")
    finite = y[np.isfinite(y)]
    assert pooled.mu == pytest.approx(finite.mean())
    assert pooled.sd == pytest.approx(finite.std())
    assert pooled.n_train_values == 5
    z = pooled.apply(y)
    assert np.isnan(z[:, 2]).all()
    assert np.allclose(pooled.invert(z)[np.isfinite(y)], finite)

    centered = fit_standardization("d", y, center_per_gene=True, transform="log2")
    assert centered.gene_center == [3.0, 12.0, None]
    assert centered.mu == pytest.approx(0.0)
    zc = centered.apply(y)
    # residuals after centering: [-2, 0, 2] and [-2, 2] -> sd over 5 values
    assert centered.sd == pytest.approx(np.std([-2.0, 0.0, 2.0, -2.0, 2.0]))
    assert np.isnan(zc[:, 2]).all()  # no training value: no center, no target
    assert zc[0, 0] == pytest.approx(-2.0 / centered.sd)


def test_fit_standardization_refuses_degenerate_rows() -> None:
    with pytest.raises(ValueError, match="finite training values"):
        fit_standardization("d", np.array([[np.nan, 1.0]]), False, "raw")
    with pytest.raises(ValueError, match="zero spread"):
        fit_standardization("d", np.array([[2.0, 2.0]]), False, "raw")
    with pytest.raises(ValueError, match=r"\[n, width\]"):
        fit_standardization("d", np.array([1.0, 2.0]), False, "raw")


def test_center_must_match_flag() -> None:
    with pytest.raises(ValueError, match="gene_center"):
        DatasetStandardization(
            dataset="d",
            transform="raw",
            center_per_gene=True,
            mu=0.0,
            sd=1.0,
            n_train_records=1,
            n_train_values=2,
        )


def test_stats_sha256_is_stable_and_content_sensitive() -> None:
    y = np.array([[1.0, 2.0], [3.0, 5.0]])
    a = StandardizationStats.fit("v1", "abc", [("d", y, False, "raw")])
    b = StandardizationStats.fit("v1", "abc", [("d", y.copy(), False, "raw")])
    c = StandardizationStats.fit("v1", "abc", [("d", y + 1.0, False, "raw")])
    assert a.sha256() == b.sha256()
    assert a.sha256() != c.sha256()
    assert a.get("d").mu == pytest.approx(2.75)
    with pytest.raises(KeyError):
        a.get("missing")
    assert StandardizationStats.model_validate_json(a.canonical_json()) == a
