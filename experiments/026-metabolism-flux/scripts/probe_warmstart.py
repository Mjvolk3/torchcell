# experiments/026-metabolism-flux/scripts/probe_warmstart.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_warmstart
r"""Is it the PROJECTION that fails the balance rule, or the PROPOSAL it is given?

THE CONTRAST. ``probe_projrate.py`` shows that K from 20 to 5,000 leaves the
turnover-thresholded balance ratio at 2.0 when the iteration starts at a random point of
the box. This script changes one thing: where it starts. The proposal is a real FBA flux
multiplied by ``1 + rel * N(0, 1)`` per reaction and clipped into the box, so ``rel`` is
the relative distance of the proposal from a feasible flux, and everything after it is the
layer's existing alternation ending on the clip.

WHAT IT SETTLES. If the rule is reachable at small ``rel`` and degrades monotonically as
``rel`` grows, then the projection is not the blocker and K is not the lever: the head's
proposal is. That is the measurement the warm-started residual head is built on, and the
``rel`` band a trained head lands in is what
``probe_trained_proposal_distance.py`` reports against this table.

Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_warmstart.py

Writes ``experiments/026-metabolism-flux/results/probe_warmstart.json``.
"""

from __future__ import annotations

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
#: Relative per-reaction noise on the reference flux. 0.0 is the reference itself.
RELATIVE_NOISE: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
#: Step counts of the alternation applied to each proposal. 100 is what the boxforce cells
#: run, so the middle row of this table is the existing head with a different start.
STEP_COUNTS: tuple[int, ...] = (20, 100)


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
    basis = null_space_basis(gem.s, cache_path=NULL_BASIS_CACHE)
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
    print(f"loaded in {time.time() - started:.1f}s; basis {tuple(basis.shape)}")

    def statistics(
        v: torch.Tensor, tag: str, proposal: torch.Tensor | None = None
    ) -> dict[str, object]:
        residual = (s @ v.t()).t()[:, rows].abs()
        omega = 0.5 * ((s.abs() @ v.abs().t()).t()[:, rows])
        ratio = residual / (omega + BALANCE_EPSILON)
        active = omega > TURNOVER_THRESHOLD * FLUX_SCALE
        selected = ratio[active]
        out: dict[str, object] = {
            "tag": tag,
            "active_rows": float(active.float().sum(1).mean()),
            "flux_active_frac": float((v.abs() > FLUX_ACTIVE).float().mean()),
            "balance_active_max": float(selected.max()),
            "balance_active_p95": float(torch.quantile(selected, 0.95)),
            "balance_median_all": float(ratio.median()),
            "box_violation_frac": float(
                ((v > tight_ub + 1e-4) | (v < tight_lb - 1e-4)).float().mean()
            ),
        }
        if proposal is not None:
            # THE STATISTIC A TRAINED HEAD IS MEASURED BY. `probe_trained_proposal_distance`
            # reports ||v_pre - P(v_pre)|| / ||v_pre|| off a checkpoint, and it can only be
            # placed in this table if the table carries the same number, so the projection
            # displacement is recorded here beside the `rel` that produced it.
            out["projection_displacement_rel"] = float(
                ((v - proposal).norm(dim=1) / proposal.norm(dim=1).clamp(min=1e-12)).mean()
            )
        print(json.dumps(out))
        return out

    def alternate(v0: torch.Tensor, steps: int) -> torch.Tensor:
        v = v0.clone()
        for k in range(steps):
            v = torch.minimum(torch.maximum(v, tight_lb), tight_ub)
            if k < steps - 1:
                v = (v @ basis) @ basis.t()
        return v

    generator = torch.Generator().manual_seed(1)
    collected = [statistics(reference.unsqueeze(0), "FBA flux itself")]
    for rel in RELATIVE_NOISE:
        noise = 1.0 + rel * torch.randn(2, s.shape[1], generator=generator)
        proposal = torch.minimum(
            torch.maximum(reference.unsqueeze(0) * noise, tight_lb), tight_ub
        )
        collected.append(statistics(proposal, f"proposal rel {rel}, no projection"))
        for steps in STEP_COUNTS:
            collected.append(
                statistics(
                    alternate(proposal, steps),
                    f"proposal rel {rel} -> alternating K={steps}, end on box",
                    proposal=proposal,
                )
            )
    # A uniform random point of the box, the control `probe_projrate.py` measures in full.
    uniform = tight_lb + (tight_ub - tight_lb) * torch.rand(
        2, s.shape[1], generator=generator
    )
    collected.append(
        statistics(
            alternate(uniform, 100), "uniform random in box -> K=100", proposal=uniform
        )
    )

    payload = {
        "definitions": {
            "rel": (
                "the proposal is the FBA flux times (1 + rel * N(0,1)) per reaction, "
                "clipped into the box, so rel is its relative distance from a feasible "
                "flux."
            )
        },
        "flux_scale": FLUX_SCALE,
        "rows": collected,
    }
    out_path = osp.join(RESULTS_DIR, "probe_warmstart.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
