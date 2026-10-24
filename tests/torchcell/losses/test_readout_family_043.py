# tests/torchcell/losses/test_readout_family_043.py
# [[tests.torchcell.losses.test_readout_family_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_readout_family_043.py
"""mle and binned_quantile of ``torchcell.losses.readout_family_043`` on synthetic batches."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchcell.losses.readout_family_043 import (
    MLE_LOGVAR_CLAMP,
    ReadoutLossFamily043,
    fit_quantile_bins,
)
from torchcell.models.cell_model_043 import CellModel043Config


def _config(loss: str, k: int = 4) -> CellModel043Config:
    return CellModel043Config.model_validate(
        {
            "hidden_dim": 16,
            "num_heads": 2,
            "num_layers": 2,
            "mask_layers": (),
            "n_parents": 1,
            "n_datasets": 2,
            "n_envs": 1,
            "exp_type_classes": ("A",),
            "exp_type_mode": "flat",
            "loss": loss,
            "hl_gauss": {"num_bins": k, "low": -4.0, "high": 4.0, "sigma_bins": 0.75},
        }
    )


def test_mle_is_the_gaussian_nll_with_clamped_log_variance() -> None:
    loss = ReadoutLossFamily043(_config("mle"))
    target = torch.tensor([[1.0, float("nan")], [0.0, 2.0]])
    pred = torch.zeros(2, 2, 2)
    pred[0, 0] = torch.tensor([0.0, 0.0])  # unit variance: 0.5 (log 2 pi + 1)
    pred[1, 0] = torch.tensor([0.0, -50.0])  # clamped to -7
    pred[1, 1] = torch.tensor([2.0, 50.0])  # clamped to +7, exact mean
    elem, measured = loss.elementwise(pred, target)
    assert measured.tolist() == [[True, False], [True, True]]
    assert float(elem[0, 0]) == pytest.approx(0.5 * (math.log(2 * math.pi) + 1.0))
    assert float(elem[0, 1]) == 0.0
    assert float(elem[1, 0]) == pytest.approx(
        0.5 * (math.log(2 * math.pi) - MLE_LOGVAR_CLAMP)
    )
    assert float(elem[1, 1]) == pytest.approx(
        0.5 * (math.log(2 * math.pi) + MLE_LOGVAR_CLAMP)
    )
    assert loss.point(pred)[1, 1] == 2.0
    p = pred.clone().requires_grad_(True)
    loss.elementwise(p, target)[0].sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_fit_quantile_bins_equal_mass_and_centers() -> None:
    b = fit_quantile_bins("g", np.arange(8, dtype=float), num_bins=4)
    assert b.edges == pytest.approx([0.0, 1.75, 3.5, 5.25, 7.0])
    assert b.centers == pytest.approx([0.5, 2.5, 4.5, 6.5])  # two values per bin
    assert b.n_train_values == 8
    with pytest.raises(ValueError, match="finite training values"):
        fit_quantile_bins("g", np.array([1.0, np.nan]), 4)


def test_binned_quantile_unit_position_soft_target_and_expectation() -> None:
    bins = [
        fit_quantile_bins("a", np.arange(8, dtype=float), 4),
        fit_quantile_bins("b", 100.0 + 10.0 * np.arange(8, dtype=float), 4),
    ]
    loss = ReadoutLossFamily043(_config("binned_quantile"), bins)
    # group a: 0 -> u 0, 3.5 -> u 0.5, 7 -> u 1; group b: same positions at its scale
    target = torch.tensor([[0.0, 3.5, 7.0], [100.0, 135.0, float("nan")]])
    groups = torch.tensor([0, 1])
    u = loss.unit_position(target, groups)
    assert u[0].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert u[1, :2].tolist() == pytest.approx([0.0, 0.5])
    assert bool(torch.isnan(u[1, 2]))
    pred = torch.randn(2, 3, 4, requires_grad=True)
    elem, measured = loss.elementwise(pred, target, groups)
    assert elem.shape == (2, 3) and float(elem[1, 2]) == 0.0
    elem.sum().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    # one-hot logits on bin 2 give that group's bin-2 center
    onehot = torch.full((2, 3, 4), -1e4)
    onehot[..., 2] = 0.0
    pt = loss.point(onehot, groups)
    assert pt[0].tolist() == pytest.approx([4.5] * 3)
    assert pt[1].tolist() == pytest.approx([145.0] * 3)
    # record-level (1-D targets) too
    rec = loss.point(onehot[:, 0], groups)
    assert rec.tolist() == pytest.approx([4.5, 145.0])
    with pytest.raises(ValueError, match="group of every row"):
        loss.elementwise(pred, target)


def test_legacy_kinds_are_the_parent_computation() -> None:
    loss = ReadoutLossFamily043(_config("mse"))
    target = torch.tensor([0.5, float("nan")])
    elem, _ = loss.elementwise(torch.zeros(2), target, torch.tensor([0, 1]))
    assert elem.tolist() == [0.25, 0.0]
    with pytest.raises(ValueError, match="exactly for binned_quantile"):
        ReadoutLossFamily043(_config("binned_quantile"))
