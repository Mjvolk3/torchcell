# experiments/026-metabolism-flux/scripts/probe_projrate.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_projrate
r"""Does K buy the honest balance rule, and does Dykstra's correction buy it?

THE SETUP, AND THE ONE THING THAT MAKES IT A FAIR TEST. The alternating projection is run
on the real S, the real bounds clipped to the layer's flux scale, and a directional
tightening taken from the signs of a REAL FBA flux, so the intersection provably contains a
working growth flux and a failure to reach it is a failure of the iteration rather than an
empty target. The iterate starts at a uniform random point of that box, which is what an
untrained head's proposal looks like, and ends on the clip, exactly as ``boxforce`` does.

WHAT IS READ. ``bal_act_max`` and ``bal_act_p95`` are the turnover-thresholded rule the
audit calls Genesis; ``active_rows`` is how many rows the rule is evaluated over, and it
is what distinguishes converging from emptying. A run whose active-row count falls while
the ratio stays pinned has not balanced the network, it has switched the network off.

``--method`` runs von Neumann's alternation, Dykstra's correction, or both. Dykstra's limit
is the NEAREST point of the intersection to the start, which is the property the wave-8
cells were queued for.

Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_projrate.py

Writes ``experiments/026-metabolism-flux/results/probe_projrate.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import time

import torch
from dotenv import load_dotenv

load_dotenv()

from torchcell.metabolism.constraints import (  # noqa: E402
    build_gem_tensors,
    null_space_basis,
)
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
NULL_BASIS_CACHE = osp.join(
    DATA_ROOT, "data/torchcell/yeast-GEM/null_space_basis_9_0_2.npy"
)

FLUX_SCALE = 100.0
TURNOVER_THRESHOLD = 1.0e-3
BALANCE_EPSILON = 1.0e-6
FLUX_ACTIVE = 1.0e-4
STEP_COUNTS: tuple[int, ...] = (20, 50, 100, 200, 500, 1000, 2000, 5000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="both", choices=["alternating", "dykstra", "both"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(
        model, source.model_dir, with_thermo=False, with_independent_rows=True
    )
    s = gem.s.to_dense().float()
    rows = gem.independent_rows
    basis = null_space_basis(gem.s, cache_path=NULL_BASIS_CACHE)
    print(f"null basis {tuple(basis.shape)}; loaded in {time.time() - started:.1f}s")

    lb = gem.lb.clamp(-FLUX_SCALE, FLUX_SCALE)
    ub = gem.ub.clamp(-FLUX_SCALE, FLUX_SCALE)
    solution = model.optimize()
    assert solution.status == "optimal", f"the base FBA is {solution.status}."
    reference = torch.tensor(
        [solution.fluxes[r] for r in gem.rxn_ids], dtype=torch.float32
    )
    # The directional tightening is read off the reference's OWN signs, so the intersection
    # of the tightened box with ker S provably contains the reference flux.
    tight_lb, tight_ub = lb.clone(), ub.clone()
    tight_ub[reference < -1e-9] = torch.minimum(
        tight_ub[reference < -1e-9], torch.zeros(1)
    )
    tight_lb[reference > 1e-9] = torch.maximum(
        tight_lb[reference > 1e-9], torch.zeros(1)
    )
    contains = bool(
        ((reference >= tight_lb - 1e-6) & (reference <= tight_ub + 1e-6)).all()
    )
    n_tightened = int(((tight_lb != lb) | (tight_ub != ub)).sum())
    print(f"tightened boxes {n_tightened}; contains the reference flux: {contains}")
    assert contains, "the tightened box lost its own reference flux; the test is void."

    def statistics(v: torch.Tensor) -> dict[str, float]:
        residual = (s @ v.t()).t()[:, rows].abs()
        omega = 0.5 * ((s.abs() @ v.abs().t()).t()[:, rows])
        ratio = residual / (omega + BALANCE_EPSILON)
        active = omega > TURNOVER_THRESHOLD * FLUX_SCALE
        out = {
            "box_violation_frac": float(
                ((v > tight_ub + 1e-4) | (v < tight_lb - 1e-4)).float().mean()
            ),
            "balance_median_all": float(ratio.median()),
            "active_rows": float(active.float().sum(1).mean()),
            "flux_active_frac": float((v.abs() > FLUX_ACTIVE).float().mean()),
        }
        selected = ratio[active]
        out["balance_active_max"] = float(selected.max())
        out["balance_active_p95"] = float(torch.quantile(selected, 0.95))
        return out

    def run(method: str) -> list[dict[str, object]]:
        generator = torch.Generator().manual_seed(0)
        v = tight_lb + (tight_ub - tight_lb) * torch.rand(
            args.batch, s.shape[1], generator=generator
        )
        p = torch.zeros_like(v)
        q = torch.zeros_like(v)
        collected: list[dict[str, object]] = []
        taken = 0
        clock = time.time()
        for steps in STEP_COUNTS:
            while taken < steps:
                if method == "alternating":
                    v = torch.minimum(torch.maximum(v, tight_lb), tight_ub)
                    v = (v @ basis) @ basis.t()
                else:
                    y = torch.minimum(torch.maximum(v + p, tight_lb), tight_ub)
                    p = v + p - y
                    w = ((y + q) @ basis) @ basis.t()
                    q = y + q - w
                    v = w
                taken += 1
            ended = torch.minimum(torch.maximum(v, tight_lb), tight_ub)
            row: dict[str, object] = {
                "method": method,
                "K": steps,
                "seconds": round(time.time() - clock, 2),
            }
            row.update(statistics(ended))
            collected.append(row)
            print(json.dumps(row))
        return collected

    methods = (
        ["alternating", "dykstra"] if args.method == "both" else [args.method]
    )
    out_rows: list[dict[str, object]] = []
    for method in methods:
        out_rows.extend(run(method))
    reference_row: dict[str, object] = {"method": "reference_FBA", "K": None}
    reference_row.update(statistics(reference.unsqueeze(0).repeat(args.batch, 1)))
    out_rows.append(reference_row)
    print(json.dumps(reference_row))

    payload = {
        "definitions": {
            "balance_active_max": (
                "max of |(Sv)_i| / (omega_i + 1e-6) over the independent rows whose "
                "turnover exceeds 1e-3 * flux_scale. The rule the audit calls Genesis."
            ),
            "active_rows": (
                "how many rows the rule is evaluated over. A falling count with a pinned "
                "ratio is the network emptying, not balancing."
            ),
        },
        "batch": args.batch,
        "threads": args.threads,
        "flux_scale": FLUX_SCALE,
        "n_tightened_boxes": n_tightened,
        "tightened_box_contains_reference": contains,
        "rows": out_rows,
    }
    out_path = osp.join(RESULTS_DIR, "probe_projrate.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
