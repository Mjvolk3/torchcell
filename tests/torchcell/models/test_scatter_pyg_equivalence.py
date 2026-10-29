# tests/torchcell/models/test_scatter_pyg_equivalence.py
# [[tests.torchcell.models.test_scatter_pyg_equivalence]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_scatter_pyg_equivalence.py
"""``torch_geometric.utils.scatter`` is the same op as the ``torch_scatter`` calls it replaced.

Commit 9a166da10 swapped ``scatter_add`` / ``scatter_mean`` / ``scatter_max`` (all called with
``dim=0``) in ``DeepSet``, ``SimpleLinearModel`` and ``SelfAttentionDeepSet`` for the PyG
pure-torch ``scatter(..., dim=0, reduce=sum|mean|max)``, because the torch_scatter wheels do
not import on glibc 2.28 (IGB Biocluster) or on XPU (Aurora), and ``torchcell.models``
imports those three modules at package import, which ``train_043.py`` reaches.

For every reduce used at those call sites, on random float inputs with an index that leaves
some segments empty (``dim_size`` larger than the largest index), the forward values and the
gradient with respect to ``src`` agree to float tolerance. Empty segments are 0 under both.
The default ``dim`` differs (torch_scatter -1, PyG 0), so the case at ``dim=-1`` passes the
dim explicitly to both; the call sites pass ``dim=0``.
"""

import pytest
import torch
import torch_scatter
from torch_geometric.utils import scatter

REFERENCE = {
    "sum": lambda src, index, dim, dim_size: torch_scatter.scatter_add(
        src, index, dim=dim, dim_size=dim_size
    ),
    "mean": lambda src, index, dim, dim_size: torch_scatter.scatter_mean(
        src, index, dim=dim, dim_size=dim_size
    ),
    "max": lambda src, index, dim, dim_size: torch_scatter.scatter_max(
        src, index, dim=dim, dim_size=dim_size
    )[0],
}


def _inputs(seed: int, dim: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Random ``src`` (37 rows x 5 features) and an index over 0..9 that skips 4 and 7."""
    generator = torch.Generator().manual_seed(seed)
    src = torch.randn(37, 5, generator=generator, dtype=torch.float64)
    allowed = torch.tensor([0, 1, 2, 3, 5, 6, 8])
    index = allowed[torch.randint(0, len(allowed), (37,), generator=generator)]
    dim_size = 10
    if dim == -1:
        src = src.T.contiguous()
    return src, index, dim_size


@pytest.mark.parametrize("reduce", ["sum", "mean", "max"])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pyg_scatter_matches_torch_scatter(reduce: str, dim: int, seed: int) -> None:
    """Forward values and src gradients agree for every reduce, both dims, empty segments."""
    src, index, dim_size = _inputs(seed, dim)
    src_ref = src.clone().requires_grad_(True)
    src_pyg = src.clone().requires_grad_(True)

    out_ref = REFERENCE[reduce](src_ref, index, dim, dim_size)
    out_pyg = scatter(src_pyg, index, dim=dim, dim_size=dim_size, reduce=reduce)

    assert out_ref.shape == out_pyg.shape
    torch.testing.assert_close(out_pyg, out_ref, rtol=1e-12, atol=1e-12)
    empty = torch.tensor([4, 7, 9])
    assert torch.count_nonzero(out_pyg.index_select(dim, empty)) == 0

    weight = torch.randn(
        out_ref.shape, generator=torch.Generator().manual_seed(99), dtype=torch.float64
    )
    (out_ref * weight).sum().backward()
    (out_pyg * weight).sum().backward()
    assert src_ref.grad is not None and src_pyg.grad is not None
    torch.testing.assert_close(src_pyg.grad, src_ref.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("reduce", ["sum", "mean", "max"])
def test_default_dim_size_matches_call_sites(reduce: str) -> None:
    """Without ``dim_size`` (as the call sites run) both size the output by ``index.max() + 1``."""
    src, index, _ = _inputs(3, 0)
    out_ref = REFERENCE[reduce](src, index, 0, None)
    out_pyg = scatter(src, index, dim=0, reduce=reduce)
    assert out_pyg.shape == (int(index.max()) + 1, 5)
    torch.testing.assert_close(out_pyg, out_ref, rtol=1e-12, atol=1e-12)
