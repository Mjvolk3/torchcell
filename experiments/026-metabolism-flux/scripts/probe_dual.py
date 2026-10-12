# experiments/026-metabolism-flux/scripts/probe_dual.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_dual
r"""The EXACT metric projection by dual ascent, and what it costs in float32.

THE PROGRAM. Projecting :math:`v_0` onto :math:`\{Sv=0\}\cap[lb,ub]` is a QP whose dual is
unconstrained and concave,

.. math::
    v(y)=\mathrm{clip}(v_0-S^\top y,\ lb,\ ub),\qquad
    g(y)=\tfrac12\lVert v-v_0\rVert^2+y^\top Sv,\qquad \nabla g(y)=Sv(y),

so maximizing :math:`g` over :math:`y\in\mathbb R^m` gives the metric projection in one
solve instead of the alternation's fixed number of sweeps. This measures whether L-BFGS on
that dual converges at genome scale in the precision the layer trains in.

WHAT IT REPORTS, AND HOW TO READ IT. The headline is ``abs_residual_max``, the largest
absolute row of :math:`Sv`. This probe is a measurement of the CONDITIONING of one
implementation, not a verdict on the route: S carries 213 dependent rows, so the dual is
degenerate, and restricting to the independent rows and moving to float64 is the obvious
next step and is not done here. A non-converging run is measured-and-badly-conditioned, not
measured-and-null.

Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_dual.py

Writes ``experiments/026-metabolism-flux/results/probe_dual.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import time

import torch
from dotenv import load_dotenv

load_dotenv()

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")

FLUX_SCALE = 100.0
TURNOVER_THRESHOLD = 1.0e-3
BALANCE_EPSILON = 1.0e-6
FLUX_ACTIVE = 1.0e-4
ITERATION_COUNTS: tuple[int, ...] = (50, 200, 500, 1000)


def main() -> None:
    torch.set_num_threads(8)
    started = time.time()
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(
        model, source.model_dir, with_thermo=False, with_independent_rows=True
    )
    s = gem.s.to_dense().float()
    rows = gem.independent_rows
    lb = gem.lb.clamp(-FLUX_SCALE, FLUX_SCALE)
    ub = gem.ub.clamp(-FLUX_SCALE, FLUX_SCALE)
    solution = model.optimize()
    assert solution.status == "optimal", f"the base FBA is {solution.status}."
    reference = torch.tensor(
        [solution.fluxes[r] for r in gem.rxn_ids], dtype=torch.float32
    )
    tight_lb, tight_ub = lb.clone(), ub.clone()
    tight_ub[reference < -1e-9] = torch.minimum(
        tight_ub[reference < -1e-9], torch.zeros(1)
    )
    tight_lb[reference > 1e-9] = torch.maximum(
        tight_lb[reference > 1e-9], torch.zeros(1)
    )
    batch = 2
    generator = torch.Generator().manual_seed(0)
    v0 = tight_lb + (tight_ub - tight_lb) * torch.rand(
        batch, s.shape[1], generator=generator
    )
    print(f"loaded in {time.time() - started:.1f}s; S {tuple(s.shape)}")

    def statistics(v: torch.Tensor, tag: str, seconds: float) -> dict[str, object]:
        residual = (s @ v.t()).t()[:, rows].abs()
        omega = 0.5 * ((s.abs() @ v.abs().t()).t()[:, rows])
        ratio = residual / (omega + BALANCE_EPSILON)
        active = omega > TURNOVER_THRESHOLD * FLUX_SCALE
        selected = ratio[active]
        out: dict[str, object] = {
            "tag": tag,
            "seconds": round(seconds, 2),
            "box_violation_frac": float(
                ((v > tight_ub + 1e-4) | (v < tight_lb - 1e-4)).float().mean()
            ),
            "abs_residual_max": float(residual.max()),
            "balance_median_all": float(ratio.median()),
            "active_rows": float(active.float().sum(1).mean()),
            "flux_active_frac": float((v.abs() > FLUX_ACTIVE).float().mean()),
            "balance_active_max": float(selected.max()),
            "balance_active_p95": float(torch.quantile(selected, 0.95)),
            "distance_from_v0": float((v - v0).norm(dim=1).mean()),
        }
        print(json.dumps(out))
        return out

    collected: list[dict[str, object]] = []
    for iterations in ITERATION_COUNTS:
        y = torch.zeros(batch, s.shape[0], requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [y],
            max_iter=iterations,
            history_size=20,
            tolerance_grad=1e-12,
            tolerance_change=1e-16,
            line_search_fn="strong_wolfe",
        )
        clock = time.time()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            v = torch.minimum(torch.maximum(v0 - (y @ s), tight_lb), tight_ub)
            dual = 0.5 * ((v - v0) ** 2).sum(1) + (y * (s @ v.t()).t()).sum(1)
            loss = (-dual).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            v = torch.minimum(torch.maximum(v0 - (y @ s), tight_lb), tight_ub)
        collected.append(
            statistics(v, f"dual L-BFGS {iterations} iterations", time.time() - clock)
        )
    collected.append(
        statistics(reference.unsqueeze(0).repeat(batch, 1), "reference FBA", 0.0)
    )

    payload = {
        "definitions": {
            "abs_residual_max": "the largest absolute row of S v over the independent rows.",
            "caveat": (
                "float32 and the FULL row set, which carries 213 dependent rows. A "
                "non-converging run here is a conditioning measurement, not evidence "
                "against the dual route."
            ),
        },
        "dtype": "float32",
        "batch": batch,
        "rows": collected,
    }
    out_path = osp.join(RESULTS_DIR, "probe_dual.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
