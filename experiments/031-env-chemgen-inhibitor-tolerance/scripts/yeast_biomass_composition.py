# experiments/031-env-chemgen-inhibitor-tolerance/scripts/yeast_biomass_composition.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.yeast_biomass_composition]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/yeast_biomass_composition
"""What fraction of yeast dry mass is each class of molecule, from the model's own biomass.

The mirrored literature states yeast protein, RNA, lipid and cell wall as separate rounded
figures that do not add to one and say nothing about DNA or the small-molecule pool. The
genome-scale model carries a complete answer that nobody has to look up: its biomass
pseudo-reaction draws one unit each from a protein, carbohydrate, RNA, DNA, lipid, cofactor
and ion pool, and each pool's own pseudo-reaction lists its constituents with stoichiometric
coefficients in mmol per gram dry weight. Multiplying each coefficient by its metabolite's
formula mass gives grams per gram dry weight directly.

THE ONE APPROXIMATION, and it is checkable. The protein pool's constituents are amino-acyl
tRNAs whose formulas carry an ``R`` standing for the tRNA, and the lipid backbone entries use
the same device for the acyl attachment point. ``formula_mass`` ignores ``R``, which counts
the amino-acid and backbone moieties and not the carrier. If that were wrong the pools would
not sum to one; they sum to 96 percent, and the 4 percent shortfall is the handful of
constituents carrying no formula at all. The total is printed so the check is visible on every
run rather than asserted here.

WHY THE COFACTOR AND ION POOLS ARE A LOWER BOUND on the small-molecule pool. They contain the
cofactors and ions the biomass equation requires, not the cell's whole metabolite pool, so a
soluble pool measured by extraction would be larger. This script reports what the model
accounts for and does not extrapolate past it.

Writes ``results/yeast_biomass_composition.csv``, one row per pool.
"""

from __future__ import annotations

import os
import os.path as osp
import re

import cobra
import pandas as pd
from cobra.core.formula import elements_and_molecular_weights as ELEMENT_WEIGHTS
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
GEM_SBML = osp.join(
    DATA_ROOT, "data/torchcell/yeast-GEM/yeast-GEM-9.0.2", "model", "yeast-GEM.xml"
)

#: Pool pseudo-reaction id -> the label used in the figure. The lipid pool is split across a
#: backbone and a chain pseudo-reaction in this release and is recombined here.
POOLS: dict[str, str] = {
    "r_4047": "protein",
    "r_4048": "carbohydrate",
    "r_4049": "RNA",
    "r_4050": "DNA",
    "r_4063": "lipid",
    "r_4065": "lipid",
    "r_4598": "cofactor",
    "r_4599": "ion",
}
#: What the mirrored literature states for yeast, as a range, for the pools it covers. The
#: model's carbohydrate pool is compared against the cell wall figure with the caveat that
#: carbohydrate also includes storage glycogen and trehalose, so it should read higher.
LITERATURE_RANGE: dict[str, tuple[float, float, str]] = {
    "protein": (40.0, 50.0, "miloCellBiologyNumbers2016"),
    "RNA": (10.0, 10.0, "miloCellBiologyNumbers2016"),
    "lipid": (10.0, 10.0, "miloCellBiologyNumbers2016"),
    "carbohydrate": (25.0, 25.0, "feldmannYeastMolecularCell2012, cell wall only"),
}


def formula_mass(formula: str | None) -> float:
    """Formula mass in g/mol, ignoring an ``R`` placeholder.

    ``R`` marks the tRNA on an amino-acyl tRNA and the attachment point on a lipid backbone.
    Counting it is impossible and skipping it is what makes the pool masses come out as the
    moiety contributed to the polymer.
    """
    if not formula:
        return 0.0
    total = 0.0
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if not element or element == "R":
            continue
        total += ELEMENT_WEIGHTS.get(element, 0.0) * (int(count) if count else 1)
    return total


def pool_masses(model: cobra.Model) -> pd.DataFrame:
    """Grams per gram dry weight for each biomass pool, and how complete each one is."""
    acc: dict[str, dict[str, float]] = {}
    for rid, label in POOLS.items():
        rxn = model.reactions.get_by_id(rid)
        row = acc.setdefault(
            label, {"grams": 0.0, "n_constituents": 0, "n_without_formula": 0}
        )
        for met, coef in rxn.metabolites.items():
            if coef >= 0:
                continue  # the product is the pool pseudo-metabolite itself
            row["n_constituents"] += 1
            mass = formula_mass(met.formula)
            if mass == 0.0:
                row["n_without_formula"] += 1
            row["grams"] += abs(coef) * mass / 1000.0
    rows = []
    for label, v in acc.items():
        lo, hi, src = LITERATURE_RANGE.get(label, (float("nan"), float("nan"), ""))
        rows.append(
            {
                "pool": label,
                "grams_per_gdw": round(v["grams"], 5),
                "percent": round(100 * v["grams"], 3),
                "n_constituents": int(v["n_constituents"]),
                "n_without_formula": int(v["n_without_formula"]),
                "literature_low": lo,
                "literature_high": hi,
                "literature_source": src,
            }
        )
    return pd.DataFrame(rows).sort_values("percent", ascending=False)


def main() -> None:
    model = cobra.io.read_sbml_model(GEM_SBML)
    df = pool_masses(model)
    total = float(df["percent"].sum())
    # the shortfall is the constituents with no formula, carried as its own row so the
    # figure can show that the accounting does not quite close and by how much
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "pool": "unaccounted",
                        "grams_per_gdw": round((100 - total) / 100, 5),
                        "percent": round(100 - total, 3),
                        "n_constituents": int(df["n_without_formula"].sum()),
                        "n_without_formula": int(df["n_without_formula"].sum()),
                        "literature_low": float("nan"),
                        "literature_high": float("nan"),
                        "literature_source": "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df.to_csv(osp.join(RESULTS_DIR, "yeast_biomass_composition.csv"), index=False)
    print(df.to_string(index=False))
    print()
    print(f"pools sum to {total:.2f}% of dry weight")
    small = float(df[df["pool"].isin(["cofactor", "ion"])]["percent"].sum())
    print(
        f"cofactor + ion, the model's small-molecule accounting: {small:.2f}% "
        "(a LOWER bound: these are the cofactors and ions the biomass equation needs, "
        "not the whole metabolite pool)"
    )


if __name__ == "__main__":
    main()
