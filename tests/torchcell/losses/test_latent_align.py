"""Tests for the latent alignment loss (torchcell/losses/latent_align.py)."""

import pytest
import torch

from torchcell.losses.latent_align import (
    latent_align_loss,
    masked_pairwise_pearson,
    strain_latent,
)


def _loop_pearson(targets: torch.Tensor, min_overlap: int) -> torch.Tensor:
    """Reference: per pair, Pearson over the features finite in both rows."""
    n = targets.shape[0]
    out = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            both = torch.isfinite(targets[i]) & torch.isfinite(targets[j])
            if int(both.sum()) < min_overlap:
                continue
            x = targets[i][both].double()
            y = targets[j][both].double()
            out[i, j] = torch.corrcoef(torch.stack([x, y]))[0, 1]
    return out


def test_pairwise_pearson_matches_corrcoef_on_complete_data() -> None:
    torch.manual_seed(0)
    t = torch.randn(7, 40, dtype=torch.float64)
    r, valid = masked_pairwise_pearson(t, min_overlap=5)
    assert bool(valid.all())
    torch.testing.assert_close(r, torch.corrcoef(t), atol=1e-10, rtol=0)


def test_pairwise_pearson_uses_only_shared_finite_features() -> None:
    torch.manual_seed(1)
    t = torch.randn(6, 50, dtype=torch.float64)
    t[torch.rand(6, 50) < 0.3] = float("nan")
    t[5, 12:] = float(
        "nan"
    )  # row 5 keeps 12 features, under min_overlap with some rows
    r, valid = masked_pairwise_pearson(t, min_overlap=10)
    ref = _loop_pearson(t, min_overlap=10)
    torch.testing.assert_close(r, ref, atol=1e-9, rtol=0)
    both = torch.isfinite(t).double() @ torch.isfinite(t).double().T
    assert torch.equal(valid, both >= 10)


def test_loss_is_zero_when_latent_cosine_equals_label_correlation() -> None:
    torch.manual_seed(2)
    t = torch.randn(9, 30)
    # Pearson over features equals the cosine of the row-centered rows, so the centered
    # labels are a latent that the loss cannot improve on.
    latent = t - t.mean(dim=1, keepdim=True)
    loss, n_pairs = latent_align_loss(
        latent, t, torch.ones(9, dtype=torch.bool), center=False
    )
    assert n_pairs == 36
    assert loss.item() == pytest.approx(0.0, abs=1e-10)


def test_collapsed_latents_pay_the_spread_of_label_correlations() -> None:
    torch.manual_seed(3)
    t = torch.randn(8, 20)
    latent = torch.ones(8, 5).repeat(1, 1) * 0.7  # every strain the same vector
    loss, n_pairs = latent_align_loss(
        latent, t, torch.ones(8, dtype=torch.bool), center=False
    )
    r = torch.corrcoef(t)
    iu = torch.triu_indices(8, 8, offset=1)
    expected = ((1.0 - r[iu[0], iu[1]]) ** 2).mean()
    assert n_pairs == 28
    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)


def test_loss_respects_the_row_mask_and_backpropagates() -> None:
    torch.manual_seed(4)
    t = torch.randn(10, 25)
    latent = torch.randn(10, 6, requires_grad=True)
    mask = torch.zeros(10, dtype=torch.bool)
    mask[[0, 3, 4, 8]] = True
    loss, n_pairs = latent_align_loss(latent, t, mask)
    assert n_pairs == 6
    loss.backward()
    assert latent.grad is not None
    assert torch.all(latent.grad[~mask] == 0)
    assert torch.any(latent.grad[mask] != 0)


def test_loss_with_fewer_than_two_rows_or_no_valid_pair_counts_nothing() -> None:
    t = torch.full((4, 12), float("nan"))
    latent = torch.randn(4, 3)
    loss, n_pairs = latent_align_loss(latent, t, torch.ones(4, dtype=torch.bool))
    assert n_pairs == 0 and loss.item() == 0.0
    mask = torch.zeros(4, dtype=torch.bool)
    mask[1] = True
    loss, n_pairs = latent_align_loss(latent, torch.randn(4, 12), mask)
    assert n_pairs == 0 and loss.item() == 0.0


def test_strain_latent_sources() -> None:
    torch.manual_seed(5)
    B, N, d = 3, 5, 4
    H = torch.randn(B, N, d)
    cls = torch.randn(B, d)
    idx = torch.tensor([1, 4, 2, 0])
    idx_b = torch.tensor([0, 0, 1, 2])
    reps = {"h_CLS_pert": cls, "H_genes_pert": H}
    assert torch.equal(strain_latent(reps, idx, idx_b, "cls", "sum"), cls)
    zs = strain_latent(reps, idx, idx_b, "zs", "sum")
    torch.testing.assert_close(zs[0], H[0, 1] + H[0, 4])
    torch.testing.assert_close(zs[2], H[2, 0])
    both = strain_latent(reps, idx, idx_b, "cls_zs", "sum")
    assert both.shape == (B, 2 * d)
    torch.testing.assert_close(both[:, :d], cls)
    with pytest.raises(ValueError, match="perturb_cls"):
        strain_latent({"h_CLS_pert": None, "H_genes_pert": H}, idx, idx_b, "cls", "sum")
    with pytest.raises(ValueError, match="source"):
        strain_latent(reps, idx, idx_b, "pool", "sum")
