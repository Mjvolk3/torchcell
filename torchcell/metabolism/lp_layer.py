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

**The elastic program (028, round 18).** Under learned ceilings the program can become
infeasible: a forced flux (the ATP maintenance reaction's positive lower bound) needs
upstream reactions whose ceilings the backbone has closed. A hard program then returns
nothing and no gradient, and the state is absorbing (028, rounds 15b to 15d). With
``slack_penalty`` :math:`M>0` every forced bound is made elastic,

.. math::

    z^\star = \max\Big\{c^\top v - M\sum_{i\in F} s_i:\ Sv=0,\ \ell_i - s_i \le v_i,\
    0\le s_i\le \ell_i\ (i\in F),\ \ell\le v\le u\ \text{otherwise}\Big\},

where :math:`F` is the set of split variables with a positive lower bound. The zero flux
with :math:`s=\ell_F` is always feasible, so the program always solves; a genotype that
cannot meet its maintenance pays :math:`M` per unit of unmet flux and its value goes
negative, which is the order FBA's lethal call sits at the bottom of. The gradient is
still the envelope theorem, now with the dual of the elastic row standing in for the
forced bound's marginal, and it is nonzero exactly on the ceilings that keep the
maintenance flux from being met, which is the gradient a barrier would give without a
barrier. Nothing about the box or the balance is softened: :math:`Sv=0` and every
capacity ceiling hold exactly at the solution.

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
from pydantic import BaseModel, ConfigDict
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


class LPRow(BaseModel):
    """One genotype's solve: the flux, the value, its duals, and what the slack did."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    v: np.ndarray
    #: The objective value: growth, minus the slack penalty under the elastic program.
    value: float
    #: The flux through the objective column itself (equals ``value`` with no slack).
    growth: float
    d_lb: np.ndarray
    d_ub: np.ndarray
    feasible: bool
    #: Total unmet forced flux, mmol/gDW/h; zero under the hard program.
    slack: float


#: The program is solved in these units times mmol/gDW/h, so 1,000 puts it in
#: micromoles. HiGHS's presolve works to absolute tolerances near 1e-7, and the learned
#: enzyme ceilings sit at 1e-6 to 1e-4 mmol/gDW/h (028 round 15: median 0.084, a tenth
#: below 0.007, thousands near zero), which presolve rounds into an empty box and
#: reports as infeasible even for the elastic program that is feasible by construction
#: (measured 2026-09-17 at capacity scales 1 and 10: presolve infeasible on 8 of 8
#: genotypes, the same program in micromoles optimal on 8 of 8 at 0.06 s, and the
#: values identical to the no-presolve solve). A uniform scaling of the variables and
#: bounds leaves the objective's bound derivatives unchanged, so the duals need no
#: correction; the flux, the value and the slack are divided back.
UNIT_SCALE = 1000.0


def _solve_one(args: tuple[np.ndarray, np.ndarray, float]) -> LPRow:
    """One growth LP, hard (``slack_penalty == 0``) or elastic."""
    lb, ub, slack_penalty = args
    assert _A_EQ is not None and _C is not None
    r = _N_RXN
    # An inconsistent box (a capacity ceiling below a forced lower bound) is closed to
    # its lower bound rather than reported infeasible: the reaction is forced and capped
    # at once, and the cap wins.
    ub = np.maximum(ub, lb)
    lo, hi = split_bounds(lb * UNIT_SCALE, ub * UNIT_SCALE)
    n = 2 * r
    forced = np.flatnonzero(lo > 0) if slack_penalty > 0 else np.zeros(0, dtype=int)
    k = int(forced.size)
    if k:
        # x = (p, n, s): the forced variables' own lower bound moves to the elastic row
        # -x_i - s_i <= -lo_i, and the slack costs M in the objective.
        c = np.concatenate([-_C, np.full(k, slack_penalty)])
        bound_lo = np.concatenate([lo, np.zeros(k)])
        bound_lo[forced] = 0.0
        bound_hi = np.concatenate([hi, lo[forced]])
        a_eq = sp.hstack([_A_EQ, sp.csr_matrix((_A_EQ.shape[0], k))]).tocsr()
        rows = np.arange(k)
        a_ub = sp.csr_matrix(
            (
                -np.ones(2 * k),
                (np.concatenate([rows, rows]), np.concatenate([forced, n + rows])),
            ),
            shape=(k, n + k),
        )
        elastic_args: dict[str, Any] = {
            "A_ub": a_ub,
            "b_ub": -lo[forced],
            "A_eq": a_eq,
            "b_eq": np.zeros(a_eq.shape[0]),
            "bounds": np.stack([bound_lo, bound_hi], axis=1),
            "method": "highs",
        }
        res = linprog(c, **elastic_args)
        if res.status == 2:
            # THE ELASTIC PROGRAM IS FEASIBLE BY CONSTRUCTION (the zero flux with full
            # slack satisfies every row and bound), so "infeasible" here is HiGHS's
            # presolve, which misreads the box when thousands of ceilings sit at 1e-6
            # (measured 2026-09-17 at capacity scale 1 to 10: presolve says infeasible
            # on every genotype, the same program without presolve solves in 0.2 s).
            # The re-solve without presolve is the correction, and a second failure is
            # an error rather than a zero flux, because a zero here would put the
            # absorbing state back under a construction that was built to remove it.
            res = linprog(c, **elastic_args, options={"presolve": False})
            assert res.status == 0, (
                f"the elastic program is feasible by construction but HiGHS returned "
                f"status {res.status} without presolve: {res.message}"
            )
    else:
        res = linprog(
            -_C,
            A_eq=_A_EQ,
            b_eq=np.zeros(_A_EQ.shape[0]),
            bounds=np.stack([lo, hi], axis=1),
            method="highs",
        )
    if res.status != 0:
        return LPRow(
            v=np.zeros(r),
            value=0.0,
            growth=0.0,
            d_lb=np.zeros(r),
            d_ub=np.zeros(r),
            feasible=False,
            slack=0.0,
        )
    x = res.x / UNIT_SCALE
    v = x[:r] - x[r:n]
    # scipy minimizes f = -value; a bound's marginal is df/d(bound), so d value/d(bound)
    # is its negative. Chain through the split: p's bounds move with (lb, ub) where they
    # are positive, n's with (-ub, -lb) where those are positive. The unit scaling
    # multiplies the value and the bounds alike, so these derivatives are already in
    # the layer's units.
    d_lo = -np.asarray(res.lower.marginals)[:n]
    d_hi = -np.asarray(res.upper.marginals)[:n]
    if k:
        # The elastic row -x_i - s_i <= -lo_i: its marginal is df/d(-lo_i), so
        # d value/d lo_i = -(-marginal) = marginal, non-positive, as raising a forced
        # flux can only lower the value.
        d_lo[forced] = np.asarray(res.ineqlin.marginals)
    d_lb = d_lo[:r] * (lb > 0) - d_hi[r:] * (lb < 0)
    d_ub = d_hi[:r] * (ub > 0) - d_lo[r:] * (ub < 0)
    slack = float(x[n:].sum()) if k else 0.0
    return LPRow(
        v=v,
        value=float(-res.fun) / UNIT_SCALE,
        growth=float(v[np.flatnonzero(_C[:r])[0]]),
        d_lb=d_lb,
        d_ub=d_ub,
        feasible=True,
        slack=slack,
    )


class LPBatch(BaseModel):
    """The stacked rows of one batched solve, kept on the solver for the layer to log."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    v: np.ndarray
    value: np.ndarray
    growth: np.ndarray
    d_lb: np.ndarray
    d_ub: np.ndarray
    feasible: np.ndarray
    slack: np.ndarray


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
        #: The last batch solved, for the layer's per-epoch statistics.
        self.last: LPBatch | None = None

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
        self, lb: np.ndarray, ub: np.ndarray, slack_penalty: float = 0.0
    ) -> LPBatch:
        """Solve every row of ``(lb, ub)``; ``slack_penalty > 0`` selects the elastic program."""
        jobs = [(lb[i], ub[i], float(slack_penalty)) for i in range(lb.shape[0])]
        if self.n_workers == 1:
            _init_pool(self.a_eq, self.c, self.n_rxn)
            rows = [_solve_one(j) for j in jobs]
        else:
            rows = self.pool().map(_solve_one, jobs, chunksize=1)
        self.last = LPBatch(
            v=np.stack([x.v for x in rows]),
            value=np.array([x.value for x in rows]),
            growth=np.array([x.growth for x in rows]),
            d_lb=np.stack([x.d_lb for x in rows]),
            d_ub=np.stack([x.d_ub for x in rows]),
            feasible=np.array([x.feasible for x in rows]),
            slack=np.array([x.slack for x in rows]),
        )
        return self.last


class _GrowthLP(torch.autograd.Function):
    """``v = argmax`` per row; only the objective column carries a gradient."""

    @staticmethod
    def forward(
        ctx: Any,
        lb: torch.Tensor,
        ub: torch.Tensor,
        solver: GrowthLPSolver,
        slack_penalty: float,
    ) -> torch.Tensor:
        batch = solver.solve_batch(
            lb.detach().cpu().double().numpy(),
            ub.detach().cpu().double().numpy(),
            slack_penalty,
        )
        ctx.save_for_backward(
            torch.from_numpy(batch.d_lb).to(lb), torch.from_numpy(batch.d_ub).to(ub)
        )
        ctx.objective_index = solver.objective_index
        # A copy: `from_numpy` shares memory, and the column write below must not reach
        # the flux the solver keeps for the layer's statistics.
        out = torch.from_numpy(batch.v.copy()).to(lb)
        # The objective column carries the VALUE of the program, which is the growth
        # under the hard program and growth minus the slack penalty under the elastic
        # one; the duals saved above are the derivative of that value.
        out[:, solver.objective_index] = torch.from_numpy(batch.value).to(lb)
        return out

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        d_lb, d_ub = ctx.saved_tensors
        g = grad_out[:, ctx.objective_index].unsqueeze(1)
        return g * d_lb, g * d_ub, None, None


def growth_lp(
    lb: torch.Tensor,
    ub: torch.Tensor,
    solver: GrowthLPSolver,
    slack_penalty: float = 0.0,
) -> torch.Tensor:
    """The optimal flux per row of ``(lb, ub)``, differentiable in the bounds through growth."""
    return _GrowthLP.apply(lb, ub, solver, slack_penalty)  # type: ignore[no-untyped-call,no-any-return]
