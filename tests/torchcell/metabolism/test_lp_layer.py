# tests/torchcell/metabolism/test_lp_layer.py
"""The growth LP layer on a toy network: the optimum, and its duals against finite
differences.
"""

from __future__ import annotations

import torch

from torchcell.metabolism.lp_layer import GrowthLPSolver, growth_lp


def toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """A -> B (r0), B -> C (r1), A uptake (r2, exchange in), C secretion = growth (r3).

    Metabolites A, B, C. Balance: r2 = r0 = r1 = r3, so growth is the tightest bound on
    the chain.
    """
    s = torch.tensor(
        [[-1.0, 0.0, 1.0, 0.0], [1.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0]],
        dtype=torch.float64,
    )
    lb = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    ub = torch.tensor([10.0, 3.0, 5.0, 10.0], dtype=torch.float64)
    return s, lb, ub, 3


def test_growth_is_the_tightest_bound_on_the_chain() -> None:
    s, lb, ub, obj = toy()
    solver = GrowthLPSolver(s, obj, n_workers=1)
    v = growth_lp(lb.unsqueeze(0), ub.unsqueeze(0), solver)
    assert abs(v[0, obj].item() - 3.0) < 1e-9
    assert torch.allclose(s @ v[0], torch.zeros(3, dtype=torch.float64), atol=1e-9)


def test_dual_is_the_finite_difference_on_the_binding_bound() -> None:
    s, lb, ub, obj = toy()
    solver = GrowthLPSolver(s, obj, n_workers=1)
    ub_p = ub.unsqueeze(0).clone().requires_grad_(True)
    lb_p = lb.unsqueeze(0).clone().requires_grad_(True)
    g = growth_lp(lb_p, ub_p, solver)[0, obj]
    g.backward()
    assert ub_p.grad is not None and lb_p.grad is not None
    # Only the binding bound (r1's upper bound of 3) prices growth.
    expected = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(ub_p.grad, expected, atol=1e-9)
    assert torch.allclose(lb_p.grad, torch.zeros(1, 4, dtype=torch.float64), atol=1e-9)
    eps = 1e-3
    ub2 = ub.clone()
    ub2[1] += eps
    g2 = growth_lp(lb.unsqueeze(0), ub2.unsqueeze(0), solver)[0, obj].item()
    assert abs((g2 - g.item()) / eps - 1.0) < 1e-6


def test_batch_rows_are_independent() -> None:
    s, lb, ub, obj = toy()
    solver = GrowthLPSolver(s, obj, n_workers=1)
    ub_b = torch.stack([ub, ub.clone()])
    ub_b[1, 2] = 1.0  # tighten uptake on the second row
    v = growth_lp(torch.stack([lb, lb]), ub_b, solver)
    assert abs(v[0, obj].item() - 3.0) < 1e-9
    assert abs(v[1, obj].item() - 1.0) < 1e-9
