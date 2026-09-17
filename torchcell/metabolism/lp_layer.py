# torchcell/metabolism/lp_layer.py
# [[torchcell.metabolism.lp_layer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/lp_layer

r"""A flux layer that solves: the growth optimum under learned bounds, with its duals.

Every projection-based parameterization in :mod:`flux_layer` returns a flux that is
*near* the polytope :math:`\{Sv=0,\ \ell\le v\le u\}`; none returns the point on it that
maximizes growth, which is what flux balance analysis computes and what ranks essential
genes on its own (028, rounds 2 to 10). This module returns that point.

.. math::

    z^\star(\ell, u) = \max\{c^\top v:\ Sv = 0,\ \ell \le v \le u\}

is a linear program per genotype in the batch, solved by HiGHS. Its gradient with respect
to the bounds is the envelope theorem: :math:`\partial z^\star/\partial u_j` is the dual
price of the upper bound on reaction :math:`j` (non-negative, nonzero only where the bound
is active) and :math:`\partial z^\star/\partial\ell_j` the negative of the lower bound's.
Those are what HiGHS returns as the bound marginals, so the backward pass costs nothing
beyond the forward solve. The argmax :math:`v^\star` is piecewise constant in the bounds
and carries no gradient of its own; only the objective column does.

The bounds are the layer's own: availability closes a deleted gene's reactions through
the GPR, and the enzyme-capacity ceiling from the abundance head tightens the box, so the
backbone's path to the prediction is *through the bounds of an exact solve*, the GECKO
construction with a learned enzyme pool.

Cost. One solve on yeast-GEM 9.0.2 (4,131 reactions split into 8,262 non-negative
variables, 2,806 balance rows) takes about 0.1 to 0.2 s in HiGHS; a batch of 32 is solved
in a forked process pool. Nothing here is organism-specific.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linprog

#: Module-level state shared with forked workers: the split-variable equality matrix and
#: the objective, set once per process by :func:`_init_pool`.
_A_EQ: sp.csr_matrix | None = None
_C: np.ndarray | None = None
_N_RXN: int = 0


def _init_pool(a_eq: sp.csr_matrix, c: np.ndarray, n_rxn: int) -> None:
    global _A_EQ, _C, _N_RXN
    _A_EQ, _C, _N_RXN = a_eq, c, n_rxn


def split_bounds(lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Bounds of the split :math:`v = p - n` that preserve the original box exactly.

    A reaction with :math:`\ell\ge0` gets :math:`p\in[\ell,u]` and :math:`n=0`; one with
    :math:`u\le0` gets the mirror; a reversible one gets :math:`p\in[0,u]`,
    :math:`n\in[0,-\ell]`. Returns ``(lower, upper)`` each of length ``2r``.
    """
    p_lo = np.maximum(lb, 0.0)
    p_hi = np.maximum(ub, 0.0)
    n_lo = np.maximum(-ub, 0.0)
    n_hi = np.maximum(-lb, 0.0)
    return np.concatenate([p_lo, n_lo]), np.concatenate([p_hi, n_hi])


def _solve_one(
    args: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]:
    """One growth LP: ``(v, growth, d_growth/d_lb, d_growth/d_ub, feasible)``."""
    lb, ub = args
    assert _A_EQ is not None and _C is not None
    r = _N_RXN
    # An inconsistent box (a capacity ceiling below a forced lower bound) is closed to
    # its lower bound rather than reported infeasible: the reaction is forced and capped
    # at once, and the cap wins.
    ub = np.maximum(ub, lb)
    lo, hi = split_bounds(lb, ub)
    res = linprog(
        -_C,
        A_eq=_A_EQ,
        b_eq=np.zeros(_A_EQ.shape[0]),
        bounds=np.stack([lo, hi], axis=1),
        method="highs",
    )
    if res.status != 0:
        return np.zeros(r), 0.0, np.zeros(r), np.zeros(r), False
    x = res.x
    v = x[:r] - x[r:]
    growth = float(-res.fun)
    # scipy minimizes f = -growth; a bound's marginal is df/d(bound), so d growth/d(bound)
    # is its negative. Chain through the split: p's bounds move with (lb, ub) where they
    # are positive, n's with (-ub, -lb) where those are positive.
    d_lo = -np.asarray(res.lower.marginals)
    d_hi = -np.asarray(res.upper.marginals)
    d_lb = d_lo[:r] * (lb > 0) - d_hi[r:] * (lb < 0)
    d_ub = d_hi[:r] * (ub > 0) - d_lo[r:] * (ub < 0)
    return v, growth, d_lb, d_ub, True


class GrowthLPSolver:
    """Holds the GEM's equality matrix and a process pool for batched solves."""

    def __init__(
        self, s: torch.Tensor, objective_index: int, n_workers: int | None = None
    ) -> None:
        """Split ``S`` into ``[S, -S]`` over ``(p, n)`` and point the objective at growth."""
        dense = (
            (s.to_dense() if s.is_sparse else s).detach().cpu().numpy().astype(float)
        )
        m, r = dense.shape
        s_csr = sp.csr_matrix(dense)
        self.a_eq = sp.hstack([s_csr, -s_csr]).tocsr()
        self.c = np.zeros(2 * r)
        self.c[objective_index] = 1.0
        self.c[r + objective_index] = -1.0
        self.n_rxn = r
        self.objective_index = objective_index
        # TC_LP_WORKERS sets the pool size from the job (a packed cell gets a share of its
        # job's CPUs); otherwise half the visible cores, at most 16.
        env_workers = os.environ.get("TC_LP_WORKERS")
        self.n_workers = (
            n_workers
            or (int(env_workers) if env_workers else None)
            or max(1, min(16, (os.cpu_count() or 2) // 2))
        )
        self._pool: Any = None

    def pool(self) -> Any:
        """The forked worker pool, created on first use so the parent may already hold CUDA."""
        if self._pool is None:
            ctx = mp.get_context("fork")
            self._pool = ctx.Pool(
                self.n_workers,
                initializer=_init_pool,
                initargs=(self.a_eq, self.c, self.n_rxn),
            )
        return self._pool

    def solve_batch(
        self, lb: np.ndarray, ub: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(v [B,r], growth [B], d_lb [B,r], d_ub [B,r], feasible [B])``."""
        jobs = [(lb[i], ub[i]) for i in range(lb.shape[0])]
        if self.n_workers == 1:
            _init_pool(self.a_eq, self.c, self.n_rxn)
            results = [_solve_one(j) for j in jobs]
        else:
            results = self.pool().map(_solve_one, jobs, chunksize=1)
        v = np.stack([x[0] for x in results])
        growth = np.array([x[1] for x in results])
        d_lb = np.stack([x[2] for x in results])
        d_ub = np.stack([x[3] for x in results])
        feasible = np.array([x[4] for x in results])
        return v, growth, d_lb, d_ub, feasible


class _GrowthLP(torch.autograd.Function):
    """``v = argmax`` per row; only the objective column carries a gradient."""

    @staticmethod
    def forward(
        ctx: Any, lb: torch.Tensor, ub: torch.Tensor, solver: GrowthLPSolver
    ) -> torch.Tensor:
        v, growth, d_lb, d_ub, feasible = solver.solve_batch(
            lb.detach().cpu().double().numpy(), ub.detach().cpu().double().numpy()
        )
        ctx.save_for_backward(
            torch.from_numpy(d_lb).to(lb), torch.from_numpy(d_ub).to(ub)
        )
        ctx.objective_index = solver.objective_index
        ctx.feasible_frac = float(feasible.mean())
        out = torch.from_numpy(v).to(lb)
        out[:, solver.objective_index] = torch.from_numpy(growth).to(lb)
        return out

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        d_lb, d_ub = ctx.saved_tensors
        g = grad_out[:, ctx.objective_index].unsqueeze(1)
        return g * d_lb, g * d_ub, None


def growth_lp(
    lb: torch.Tensor, ub: torch.Tensor, solver: GrowthLPSolver
) -> torch.Tensor:
    """The optimal flux per row of ``(lb, ub)``, differentiable in the bounds through growth."""
    return _GrowthLP.apply(lb, ub, solver)  # type: ignore[no-untyped-call,no-any-return]
