# experiments/026-metabolism-flux/scripts/fva_reference.py
# [[experiments.026-metabolism-flux.scripts.fva_reference]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/fva_reference.py

r"""Flux variability analysis on the wild type, as the reference the sampler is scored against.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/fva_reference.py

WHY THIS IS THE RIGHT BASELINE
------------------------------
An amortized flux sampler makes a claim that needs a number attached: that conditioning on
phenotype data narrows the set of fluxes a reaction can carry. Flux variability analysis
answers the same question **from the constraints alone**, by minimizing and maximizing each
reaction subject to mass balance, the bounds, and a growth floor. So per reaction,

.. math::
    \text{information the data added}
    = \underbrace{\mathrm{width}_{\mathrm{FVA}}(j)}_{\text{constraints only}}
    - \underbrace{\mathrm{width}_{\text{model posterior}}(j)}_{\text{constraints}+\text{data}}

A narrower model interval means the phenotype measurements bought something; an equal one
means they did not. This is a **label-free** evaluation, which matters because it can be run
on double deletions where no production measurement exists at all.

The comparison is restricted to the reactions FVA licenses, meaning those whose FVA width is
already finite and small. A reaction that FVA leaves free over the full +/-1000 range is not
a reaction where narrowing means anything, it is one the constraints say nothing about.

*Caveat to state plainly:* classical sampling targets a UNIFORM distribution over the
feasible polytope, while the model targets whatever the data and priors induce. Interval
width is comparable between them; the distributions are not the same object.
"""

import json
import os
import os.path as osp

from cobra.flux_analysis import flux_variability_analysis
from dotenv import load_dotenv

from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

RESULTS_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "026-metabolism-flux", "results")

#: Growth floor, as a fraction of the unconstrained optimum. 0.9 is the convention, and it
#: is the reason a near-optimality assumption appears anywhere in this work: the widths that
#: make a flux interpretable were computed at 90 % of optimum, not because cells maximize
#: growth.
FRACTION_OF_OPTIMUM = 0.9


def main() -> None:
    """Run FVA on the wild type and write per-reaction intervals plus a width census."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = YeastGEM().model
    solution = model.optimize()
    print(f"wild-type growth: {solution.objective_value:.4f} 1/h", flush=True)

    fva = flux_variability_analysis(
        model, fraction_of_optimum=FRACTION_OF_OPTIMUM, loopless=False
    )
    fva["width"] = fva["maximum"] - fva["minimum"]
    fva.to_csv(osp.join(RESULTS_DIR, "fva_wildtype.csv"))

    width = fva["width"]
    census = {
        "fraction_of_optimum": FRACTION_OF_OPTIMUM,
        "wild_type_growth_per_h": float(solution.objective_value),
        "n_reactions": int(len(width)),
        "n_blocked_width_zero": int((width < 1e-9).sum()),
        "n_width_le_1": int((width <= 1.0).sum()),
        "n_width_le_10": int((width <= 10.0).sum()),
        "n_width_ge_1000": int((width >= 1000.0).sum()),
        "median_width": float(width.median()),
        "mean_width": float(width.mean()),
    }
    with open(osp.join(RESULTS_DIR, "fva_census.json"), "w") as f:
        json.dump(census, f, indent=2)
    print(json.dumps(census, indent=2))


if __name__ == "__main__":
    main()
