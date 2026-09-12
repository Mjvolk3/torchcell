# experiments/026-metabolism-flux/scripts/catpred_table_impact.py
# [[experiments.026-metabolism-flux.scripts.catpred_table_impact]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/catpred_table_impact.py

r"""What admitting CatPred would do to the tables the flux layer consumes.

The predicted tier is a median over the predictors the within-class study admits, so
admitting an eighth predictor changes a value on every unit whose median moves. This
resolves both parameters TWICE through the flux layer's own path,
:func:`~torchcell.metabolism.parameters.resolve_kcat_table` and
:func:`~torchcell.metabolism.parameters.resolve_km_table`, once against the shipped
plausibility file and once against the CatPred-inclusive sibling, and reports the
difference.

**The capacity bound is the quantity that matters.** The flux layer's enzyme constraint is
:math:`|v_j| \le k_{\mathrm{cat}} E`, saturation-free, so a unit's bound scales exactly
with its :math:`k_{cat}` and a change of more than a factor of two is
:math:`|\Delta \log_{10} k_{cat}| > 0.301`. That count is the headline here rather than a
mean shift, because a bound that doubles can change which reactions the model can carry
flux through and a bound that moves by five percent cannot.

**Nothing shipped is overwritten.** The per-predictor tables under
``<predictor>/verified/`` are append-only, CatPred's are new files with their own manifest,
and the resolved per-unit table is computed at run time from whichever plausibility file
the resolver is pointed at. Promotion is therefore a one-line change of that pointer, taken
when nothing is training against the old one, and this script measures what it would cost
before anybody takes it.

Writes ``results/catpred_table_impact.json`` and ``results/catpred_table_impact.parquet``.
Run from the worktree root after ``within_class_plausibility.py --suffix _with_catpred``::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/catpred_table_impact.py
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.metabolism.constraints import CatalyticUnits, build_gem_tensors
from torchcell.metabolism.kinetics_pipeline import sha256_of
from torchcell.metabolism.parameters import (
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    VERIFIED_PAIRS_FEATURE,
    ParameterTable,
    PredictorSelection,
    SubstrateMatch,
    load_predicted_unit_table,
    resolve_kcat_table,
    resolve_km_table,
)
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
KINETICS_DIR = osp.join(DATA_ROOT, KINETICS_MIRROR)
OED_MIRROR = osp.join(DATA_ROOT, "data", "enzyme_kinetics", "open_enzyme_database")
HOMOLOG_MIRROR = osp.join(OED_MIRROR, "all")
ORGANISM_MIRROR = osp.join(OED_MIRROR, "scerevisiae")

#: A capacity bound that moves by more than this in log10 has more than doubled or halved.
FACTOR_OF_TWO_DECADES = float(np.log10(2.0))


class SideBySide(BaseModel):
    """One parameter resolved both ways, and the difference."""

    model_config = ConfigDict(extra="forbid")

    parameter: str
    selected_shipped: list[str]
    selected_with_catpred: list[str]
    tier_counts_shipped: dict[str, int]
    tier_counts_with_catpred: dict[str, int]
    n_units: int
    n_units_changed: int
    median_abs_change_decades: float
    p90_abs_change_decades: float
    max_abs_change_decades: float
    n_units_over_factor_two: int
    frac_units_over_factor_two: float
    n_units_over_one_decade: int
    #: Signed median change among the units that moved, so a systematic shift is visible
    #: rather than hidden inside an absolute value.
    median_signed_change_decades: float


class ImpactReport(BaseModel):
    """The whole comparison, one object."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    plausibility_shipped: str
    plausibility_shipped_sha256: str
    plausibility_with_catpred: str
    plausibility_with_catpred_sha256: str
    factor_of_two_decades: float
    parameters: list[SideBySide]


def _resolve(
    units: CatalyticUnits, model_dir: str, plausibility: str, n_units: int
) -> dict[str, ParameterTable]:
    """Both parameter tables, through exactly the path the flux layer runs."""
    substrates = SubstrateMatch.from_verified_pairs(
        osp.join(KINETICS_DIR, VERIFIED_PAIRS_FEATURE)
    )
    predicted = {
        p: load_predicted_unit_table(p, n_units, KINETICS_DIR, plausibility)
        for p in ("k_cat", "K_M")
    }
    return {
        "k_cat": resolve_kcat_table(
            units,
            model_dir,
            ORGANISM_MIRROR,
            predicted=predicted["k_cat"],
            substrates=substrates,
            homolog_mirror_dir=HOMOLOG_MIRROR,
        ),
        "K_M": resolve_km_table(
            units,
            model_dir,
            ORGANISM_MIRROR,
            predicted=predicted["K_M"],
            substrates=substrates,
            homolog_mirror_dir=HOMOLOG_MIRROR,
        ),
    }


def main() -> None:
    """Resolve both ways, write the per-unit table and the summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="_with_catpred")
    args = parser.parse_args()

    shipped = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
    candidate = shipped.replace(".json", f"{args.suffix}.json")

    source = YeastGEM()
    units = build_gem_tensors(source.model, model_dir=source.model_dir).catalytic_units
    n_units = units.n_units

    before = _resolve(units, source.model_dir, shipped, n_units)
    after = _resolve(units, source.model_dir, candidate, n_units)

    rows: list[dict[str, object]] = []
    blocks: list[SideBySide] = []
    for parameter in ("k_cat", "K_M"):
        old = before[parameter].values.numpy().astype(float)
        new = after[parameter].values.numpy().astype(float)
        delta = np.log10(new) - np.log10(old)
        changed = np.abs(delta) > 1e-9
        rows.extend(
            {
                "unit_id": int(index),
                "parameter": parameter,
                "value_shipped": float(old[index]),
                "value_with_catpred": float(new[index]),
                "delta_decades": float(delta[index]),
                "provenance_shipped": str(before[parameter].provenance[index]),
                "provenance_with_catpred": str(after[parameter].provenance[index]),
            }
            for index in range(n_units)
        )
        moved = np.abs(delta[changed])
        blocks.append(
            SideBySide(
                parameter=parameter,
                selected_shipped=PredictorSelection.from_study(
                    parameter, shipped
                ).selected,
                selected_with_catpred=PredictorSelection.from_study(
                    parameter, candidate
                ).selected,
                tier_counts_shipped=before[parameter].tier_counts(),
                tier_counts_with_catpred=after[parameter].tier_counts(),
                n_units=int(n_units),
                n_units_changed=int(changed.sum()),
                median_abs_change_decades=(
                    float(np.median(moved)) if moved.size else 0.0
                ),
                p90_abs_change_decades=(
                    float(np.percentile(moved, 90)) if moved.size else 0.0
                ),
                max_abs_change_decades=float(moved.max()) if moved.size else 0.0,
                n_units_over_factor_two=int(
                    (np.abs(delta) > FACTOR_OF_TWO_DECADES).sum()
                ),
                frac_units_over_factor_two=float(
                    (np.abs(delta) > FACTOR_OF_TWO_DECADES).mean()
                ),
                n_units_over_one_decade=int((np.abs(delta) > 1.0).sum()),
                median_signed_change_decades=(
                    float(np.median(delta[changed])) if moved.size else 0.0
                ),
            )
        )

    report = ImpactReport(
        built_at=datetime.now(UTC).isoformat(),
        plausibility_shipped=shipped,
        plausibility_shipped_sha256=sha256_of(shipped),
        plausibility_with_catpred=candidate,
        plausibility_with_catpred_sha256=sha256_of(candidate),
        factor_of_two_decades=FACTOR_OF_TWO_DECADES,
        parameters=blocks,
    )
    out_json = osp.join(RESULTS, "catpred_table_impact.json")
    with open(out_json, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    pd.DataFrame(rows).to_parquet(
        osp.join(RESULTS, "catpred_table_impact.parquet"), index=False
    )

    for block in blocks:
        print(f"\n{block.parameter}")
        print(f"  admitted shipped      {block.selected_shipped}")
        print(f"  admitted with CatPred {block.selected_with_catpred}")
        print(f"  tiers shipped      {block.tier_counts_shipped}")
        print(f"  tiers with CatPred {block.tier_counts_with_catpred}")
        print(
            f"  {block.n_units_changed} of {block.n_units} units changed, median "
            f"{block.median_abs_change_decades:.3f} decades, p90 "
            f"{block.p90_abs_change_decades:.3f}, max "
            f"{block.max_abs_change_decades:.3f}"
        )
        print(
            f"  past a factor of two: {block.n_units_over_factor_two} "
            f"({block.frac_units_over_factor_two:.3%}); past one decade: "
            f"{block.n_units_over_one_decade}"
        )
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
