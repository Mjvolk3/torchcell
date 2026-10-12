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

WHICH MEDIUM, WHICH MODEL (2026-09-14)
--------------------------------------
The first version of this script ran ``YeastGEM().model`` with **no medium applied and no
cassette**, and every ``width_FVA - width_posterior`` number computed against it therefore
came from a medium and a model the 027 cells do not live in. The SBML's shipped bounds give
a wild-type growth of 0.085844 /h; YPD-approx with the Cachera betaxanthin cassette grows at
0.5352 /h, 6.2x higher.

So the census is now computed under named conditions rather than one unnamed default, and
the shipped default is kept as the CONTROL: the run asserts that the ``default`` condition
reproduces ``results/fva_census.json`` field for field, so any difference between the other
conditions and it is attributable to the flag and not to a drifted model. The pre-registered
read is the width census under (default, no cassette) / (YPD, no cassette) / (YPD, cassette)
side by side; the censuses differing by more than 5 % means every sampler-versus-FVA
comparison has to be restated against the YPD + cassette reference.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any

from cobra.flux_analysis import flux_variability_analysis
from dotenv import load_dotenv

from torchcell.metabolism.betaxanthin import build_betaxanthin_pathway
from torchcell.metabolism.media import FBA_MEDIA, media_to_bounds
from torchcell.metabolism.pathway import apply_pathway
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

RESULTS_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "026-metabolism-flux", "results")

#: Growth floor, as a fraction of the unconstrained optimum. 0.9 is the convention, and it
#: is the reason a near-optimality assumption appears anywhere in this work: the widths that
#: make a flux interpretable were computed at 90 % of optimum, not because cells maximize
#: growth.
FRACTION_OF_OPTIMUM = 0.9

#: name -> (medium key or None for the SBML's shipped bounds, cassette applied).
CONDITIONS: dict[str, tuple[str | None, bool]] = {
    "default": (None, False),
    "ypd": ("YPD", False),
    "ypd_cassette": ("YPD", True),
}
#: The census this script wrote before it had flags, kept as the control for `default`.
SHIPPED_CENSUS = osp.join(RESULTS_DIR, "fva_census.json")
#: Fields the control compares. `fraction_of_optimum` is a setting, not a measurement.
CONTROL_FIELDS = (
    "wild_type_growth_per_h",
    "n_reactions",
    "n_blocked_width_zero",
    "n_width_le_1",
    "n_width_le_10",
    "n_width_ge_1000",
    "median_width",
    "mean_width",
)


def build_model(media_key: str | None, cassette: bool) -> Any:
    """Yeast-GEM 9.0.2, optionally + the Cachera betaxanthin cassette, on one medium."""
    model = YeastGEM().model
    if cassette:
        model = apply_pathway(model, build_betaxanthin_pathway(model))
    if media_key is not None:
        bounds = media_to_bounds(FBA_MEDIA[media_key], model)
        unresolved = [
            r.component_name for r in bounds.resolutions if r.outcome != "resolved"
        ]
        if unresolved:
            raise ValueError(f"medium {media_key} left {unresolved} unresolved")
        bounds.apply(model)
    return model


def census(name: str, media_key: str | None, cassette: bool) -> dict[str, Any]:
    """FVA at the growth floor, plus the width census, for one condition."""
    model = build_model(media_key, cassette)
    solution = model.optimize()
    print(
        f"{name}: wild-type growth {solution.objective_value:.6f} 1/h "
        f"({len(model.reactions)} reactions)",
        flush=True,
    )
    fva = flux_variability_analysis(
        model, fraction_of_optimum=FRACTION_OF_OPTIMUM, loopless=False, processes=1
    )
    fva["width"] = fva["maximum"] - fva["minimum"]
    fva.to_csv(osp.join(RESULTS_DIR, f"fva_wildtype_{name}.csv"))
    width = fva["width"]
    return {
        "condition": name,
        "media_key": media_key,
        "media_name": None if media_key is None else FBA_MEDIA[media_key].name,
        "cassette": cassette,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--media",
        default=None,
        choices=sorted(FBA_MEDIA),
        help="run ONE medium instead of the three pre-registered conditions",
    )
    ap.add_argument(
        "--cassette",
        action="store_true",
        help="apply the Cachera betaxanthin cassette; only read with --media",
    )
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.media is not None:
        name = args.media.lower() + ("_cassette" if args.cassette else "")
        rows = {name: census(name, args.media, args.cassette)}
    else:
        rows = {
            name: census(name, media_key, cassette)
            for name, (media_key, cassette) in CONDITIONS.items()
        }

    # CONTROL. `default` must reproduce the census this script wrote before it had flags,
    # so a change in another condition is attributable to the flag and not to a drifted
    # model, a new cobra, or a new SBML.
    if "default" in rows and osp.exists(SHIPPED_CENSUS):
        with open(SHIPPED_CENSUS) as fh:
            shipped = json.load(fh)
        drift = {
            field: (shipped[field], rows["default"][field])
            for field in CONTROL_FIELDS
            if abs(float(shipped[field]) - float(rows["default"][field]))
            > 1e-6 * max(1.0, abs(float(shipped[field])))
        }
        if drift:
            raise ValueError(
                "the default condition no longer reproduces the shipped census "
                f"{SHIPPED_CENSUS}: {drift}. Every width comparison on record was made "
                "against that census, so this is a model drift and not a flag effect."
            )
        rows["default"]["reproduces_shipped_census"] = True
        print(f"control: default reproduces {SHIPPED_CENSUS} on all {len(CONTROL_FIELDS)} fields")

    out = osp.join(RESULTS_DIR, "fva_census_by_condition.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(json.dumps(rows, indent=2))

    if len(rows) > 1:
        base = rows["default"]["n_width_le_1"]
        print("\nlicensed set (FVA width <= 1), against the default condition:")
        for name, row in rows.items():
            print(
                f"  {name:<14s} {row['n_width_le_1']:>5d} of {row['n_reactions']:>5d} "
                f"({100.0 * (row['n_width_le_1'] - base) / base:+.1f} % vs default)"
            )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
