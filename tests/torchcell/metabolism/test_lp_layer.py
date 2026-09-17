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


def forced_toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """The chain with a FORCED maintenance flux the ceilings cannot carry.

    r1 (B -> C) must run at 2 or more, but r0 (A -> B) is capped at 1, so the balance
    r0 = r1 has no solution: the hard program is infeasible.
    """
    s, lb, ub, obj = toy()
    lb = lb.clone()
    ub = ub.clone()
    lb[1] = 2.0
    ub[0] = 1.0
    return s, lb, ub, obj


def test_hard_program_is_infeasible_and_silent() -> None:
    s, lb, ub, obj = forced_toy()
    solver = GrowthLPSolver(s, obj, n_workers=1)
    ub_p = ub.unsqueeze(0).clone().requires_grad_(True)
    v = growth_lp(lb.unsqueeze(0), ub_p, solver)
    assert solver.last is not None and not bool(solver.last.feasible[0])
    assert torch.all(v == 0)
    v[0, obj].backward()
    assert ub_p.grad is not None and torch.all(ub_p.grad == 0)


def test_elastic_program_is_feasible_with_the_slack_priced() -> None:
    s, lb, ub, obj = forced_toy()
    solver = GrowthLPSolver(s, obj, n_workers=1)
    m = 1.5
    ub_p = ub.unsqueeze(0).clone().requires_grad_(True)
    v = growth_lp(lb.unsqueeze(0), ub_p, solver, slack_penalty=m)
    assert solver.last is not None and bool(solver.last.feasible[0])
    # The chain runs at the ceiling of r0 (1), one unit of the forced 2 is unmet, and
    # the value is that growth minus M per unit of slack.
    assert abs(solver.last.growth[0] - 1.0) < 1e-9
    assert abs(solver.last.slack[0] - 1.0) < 1e-9
    assert abs(v[0, obj].item() - (1.0 - m)) < 1e-9
    balance = s @ torch.from_numpy(solver.last.v[0])
    assert torch.allclose(balance, torch.zeros(3, dtype=torch.float64), atol=1e-9)
    # Raising the ceiling of r0 carries one more unit of flux AND relieves one unit of
    # slack, so the value rises by 1 + M per unit: the dual of the binding ceiling.
    v[0, obj].backward()
    assert ub_p.grad is not None
    assert abs(ub_p.grad[0, 0].item() - (1.0 + m)) < 1e-9
    eps = 1e-3
    ub2 = ub.clone()
    ub2[0] += eps
    v2 = growth_lp(lb.unsqueeze(0), ub2.unsqueeze(0), solver, slack_penalty=m)
    assert abs((v2[0, obj].item() - v[0, obj].item()) / eps - (1.0 + m)) < 1e-6


def test_elastic_program_equals_the_hard_one_when_feasible() -> None:
    s, lb, ub, obj = toy()
    lb = lb.clone()
    lb[1] = 0.5  # a forced flux the ceilings can carry
    solver = GrowthLPSolver(s, obj, n_workers=1)
    hard = growth_lp(lb.unsqueeze(0), ub.unsqueeze(0), solver)
    elastic = growth_lp(lb.unsqueeze(0), ub.unsqueeze(0), solver, slack_penalty=1.0)
    assert solver.last is not None and abs(solver.last.slack[0]) < 1e-9
    assert abs(hard[0, obj].item() - elastic[0, obj].item()) < 1e-9
