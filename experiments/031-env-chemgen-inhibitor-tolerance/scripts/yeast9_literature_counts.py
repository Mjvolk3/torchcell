# experiments/031-env-chemgen-inhibitor-tolerance/scripts/yeast9_literature_counts.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.yeast9_literature_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/yeast9_literature_counts
"""Published counts for the yeast metabolome and cell composition, each with its quote.

Nothing here is measured. Every row carries the citation key of the mirrored paper it came
from and the verbatim sentence, so a figure that plots these can mark them as reported and a
reader can check them. The scripts that measure our own numbers write their own files; the
two are deliberately kept apart and drawn in different colors.

A quantity that the mirror does not state is recorded with ``value`` NA and a ``note`` saying
so, rather than being filled from a related organism. The small-molecule share of yeast dry
mass is the important case: the mirror states it for E. coli and never for yeast, and the two
must not be conflated.

Writes ``results/yeast9_literature_counts.csv`` and ``results/yeast_mass_composition.csv``.
"""

from __future__ import annotations

import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)

#: Metabolome-scope counts. ``quantity`` is the axis label; ``value`` the number as published.
METABOLOME: list[dict[str, object]] = [
    {
        "quantity": "YMDB, entries with a SMILES",
        "value": 16042,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "We first compared the 16,042 metabolites in YMDB with the metabolites "
        "in Yeast9 and identified 14,882 metabolites missing from Yeast9.",
    },
    {
        "quantity": "of those, missing from Yeast9",
        "value": 14882,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "14,882 metabolites with SMILES ... reports in the Yeast Metabolome "
        "Database (YMDB) are still missing from the widely used yeast GEM-Yeast9 model.",
    },
    {
        "quantity": "missing, lipids",
        "value": 14310,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "these missing metabolites comprise 14,310 lipids and 572 non-lipids",
    },
    {
        "quantity": "missing, non-lipids",
        "value": 572,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "these missing metabolites comprise 14,310 lipids and 572 non-lipids",
    },
    {
        "quantity": "Yeast-MetaTwin metabolites",
        "value": 16244,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "we reconstruct a yeast metabolic twin model, Yeast-MetaTwin, comprising "
        "16,244 metabolites, 1,976 metabolic genes and 59,865 reactions.",
    },
    {
        "quantity": "Yeast9 metabolite entries",
        "value": 2805,
        "source": "zhangYeast9ConsensusGenomescale2024",
        "quote": "Through the above improvements, Yeast9 contains 2805 metabolites, "
        "1162 genes, and 4130 reactions.",
    },
]

#: Coverage fractions, as published percentages of the YMDB yeast metabolome.
COVERAGE: list[dict[str, object]] = [
    {
        "model": "Yeast9",
        "scope": "whole metabolome",
        "percent": 7.0,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "This effort increased the yeast metabolome coverage from 7% to 92% "
        "(non-lipid from 54% to 75%).",
    },
    {
        "model": "Yeast9",
        "scope": "non-lipid",
        "percent": 54.0,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "This effort increased the yeast metabolome coverage from 7% to 92% "
        "(non-lipid from 54% to 75%).",
    },
    {
        "model": "Yeast-MetaTwin",
        "scope": "whole metabolome",
        "percent": 92.0,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "This effort increased the yeast metabolome coverage from 7% to 92% "
        "(non-lipid from 54% to 75%).",
    },
    {
        "model": "Yeast-MetaTwin",
        "scope": "non-lipid",
        "percent": 75.0,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "This effort increased the yeast metabolome coverage from 7% to 92% "
        "(non-lipid from 54% to 75%).",
    },
]

#: Dry-mass composition. The organism column is load-bearing: the soluble small-molecule
#: pool is stated for E. coli and NOT for yeast anywhere in the mirror, and presenting the
#: E. coli figure as a yeast figure would be wrong.
MASS: list[dict[str, object]] = [
    {
        "organism": "S. cerevisiae",
        "component": "protein",
        "percent_low": 40.0,
        "percent_high": 50.0,
        "source": "miloCellBiologyNumbers2016",
        "quote": "Similar efforts in budding yeast revealed that proteins constitute in "
        "the range of 40-50% of the cell dry mass, RNA ~10%, and lipid ~10%",
    },
    {
        "organism": "S. cerevisiae",
        "component": "RNA",
        "percent_low": 10.0,
        "percent_high": 10.0,
        "source": "miloCellBiologyNumbers2016",
        "quote": "proteins constitute in the range of 40-50% of the cell dry mass, "
        "RNA ~10%, and lipid ~10%",
    },
    {
        "organism": "S. cerevisiae",
        "component": "lipid",
        "percent_low": 10.0,
        "percent_high": 10.0,
        "source": "miloCellBiologyNumbers2016",
        "quote": "proteins constitute in the range of 40-50% of the cell dry mass, "
        "RNA ~10%, and lipid ~10%",
    },
    {
        "organism": "S. cerevisiae",
        "component": "cell wall",
        "percent_low": 25.0,
        "percent_high": 25.0,
        "source": "feldmannYeastMolecularCell2012",
        "quote": "The outer shell is a rigid structure about 100-200 nm thick and "
        "constituting about 25% of the total dry mass of the cell",
    },
    {
        "organism": "S. cerevisiae",
        "component": "small-molecule pool",
        "percent_low": float("nan"),
        "percent_high": float("nan"),
        "source": "",
        "quote": "NOT STATED FOR YEAST anywhere in the mirror.",
    },
    {
        "organism": "S. cerevisiae",
        "component": "DNA",
        "percent_low": float("nan"),
        "percent_high": float("nan"),
        "source": "",
        "quote": "NOT STATED FOR YEAST anywhere in the mirror.",
    },
    {
        "organism": "E. coli",
        "component": "small-molecule pool",
        "percent_low": 3.0,
        "percent_high": 3.9,
        "source": "miloCellBiologyNumbers2016, stephanopoulosMetabolicEngineeringPrinciples1998",
        "quote": "metabolites and cofactors pool 3 [percent of dry weight]; "
        "Soluble pool 3.9",
    },
]

#: Rule-based expansion, for scale.
RULES: list[dict[str, object]] = [
    {
        "quantity": "enzyme-associated reaction rules",
        "value": 21921,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "we first extracted 21,921 known enzyme-associated biochemical reaction "
        "rules and 213 spontaneous reaction rules from MetaNetX and MetaCyc",
    },
    {
        "quantity": "spontaneous reaction rules",
        "value": 213,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "21,921 known enzyme-associated biochemical reaction rules and 213 "
        "spontaneous reaction rules",
    },
    {
        "quantity": "reactions in the connected expansion",
        "value": 1092946,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "we extracted a connected yeast network with 1,092,946 reactions from "
        "this reaction pool by removing dead-end reactions",
    },
    {
        "quantity": "non-lipid metabolites still unconnected",
        "value": 267,
        "source": "wuSystematicallyExploringYeast2026",
        "quote": "we were unable to find enzyme-annotated reactions to connect 267 "
        "non-lipid metabolites (1.7% of the yeast metabolome) to the Yeast-MetaTwin model "
        "without introducing non-yeast-metabolome metabolites.",
    },
    {
        "quantity": "Yeast8 dead-end metabolites",
        "value": 464,
        "source": "chenGenomescaleModelingYeast2022",
        "quote": "Even in the latest version 8.5.0 of Yeast8 there are 464 out of 2742 "
        "metabolites participating in only one reaction or can only be consumed or "
        "produced, and those so-called dead-end metabolites indicate missing reactions",
    },
]


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(METABOLOME).to_csv(
        osp.join(RESULTS_DIR, "yeast9_literature_counts.csv"), index=False
    )
    pd.DataFrame(COVERAGE).to_csv(
        osp.join(RESULTS_DIR, "yeast9_literature_coverage.csv"), index=False
    )
    pd.DataFrame(MASS).to_csv(
        osp.join(RESULTS_DIR, "yeast_mass_composition.csv"), index=False
    )
    pd.DataFrame(RULES).to_csv(
        osp.join(RESULTS_DIR, "yeast9_literature_rules.csv"), index=False
    )
    print(
        f"wrote {len(METABOLOME)} metabolome counts, {len(COVERAGE)} coverage rows, "
        f"{len(MASS)} mass rows, {len(RULES)} rule rows"
    )
    missing = [
        r
        for r in MASS
        if r["organism"] == "S. cerevisiae" and pd.isna(r["percent_low"])
    ]
    for r in missing:
        print(f"  NOT STATED FOR YEAST: {r['component']}")


if __name__ == "__main__":
    main()
