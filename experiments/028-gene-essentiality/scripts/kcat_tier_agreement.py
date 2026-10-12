# experiments/028-gene-essentiality/scripts/kcat_tier_agreement.py
# [[experiments.028-gene-essentiality.scripts.kcat_tier_agreement]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/kcat_tier_agreement

r"""Do the kcat tiers agree with each other on the same catalytic units?

The shipped table fills each unit from the first tier that has a value: a yeast
measurement, then a cross-organism EC homolog measurement, then the median over the
admitted sequence predictors. A unit therefore carries ONE tier, and the table itself
never says whether the tiers would have agreed. This script evaluates every tier on every
unit it can reach, without the fall-through, and reports the pairwise agreement in
:math:`\log_{10}` decades: measured against homolog, measured against each predictor,
homolog against each predictor, and predictor against predictor. Spearman, Pearson in log
space, the median absolute difference in decades, the fraction within one decade, and the
number of units in each overlap.

Every value is per catalytic unit under the shipped conventions (substrate-matched
database rows, a complex takes the min over its subunits, the resolver's temperature
cascade). The predictors are the three the within-class study admits, read one at a time
through ``load_predicted_unit_table(restrict_to=...)``.

Writes ``results/kcat_tier_agreement.json``. Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/kcat_tier_agreement.py
"""

from __future__ import annotations

import os
import os.path as osp
from itertools import combinations

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import pearsonr, spearmanr

from torchcell.metabolism.constraints import build_gem_tensors
from torchcell.metabolism.enzyme_kinetics import KineticKind, load_mirrored_records
from torchcell.metabolism.parameters import (
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    VERIFIED_PAIRS_FEATURE,
    PredictorSelection,
    SubstrateMatch,
    _database_tiers,
    load_predicted_unit_table,
    uniprot_for_genes,
)
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
OED_SCEREVISIAE = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
)
OED_ALL = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all")
TARGET_TEMPERATURE_C = 30.0
YEAST = "Saccharomyces cerevisiae"
ONE_DECADE = 1.0


class Pair(BaseModel):
    """Agreement of two tiers on the units both reach."""

    a: str
    b: str
    n: int
    spearman: float
    pearson_log10: float
    median_abs_decades: float
    frac_within_one_decade: float
    #: Median of b minus a in decades: positive when b is systematically higher.
    median_signed_decades: float


class Tier(BaseModel):
    """One tier's coverage and center."""

    name: str
    n_units: int
    median_log10: float


class Report(BaseModel):
    """The agreement report."""

    n_units: int
    tiers: list[Tier]
    pairs: list[Pair]
    #: The pairs a reader wants first: each tier against the best-ranked predictor.
    admitted_predictors: list[str]
    best_predictor: str
    note: str


def tier_values(
    units, model_dir: str, organism, homolog, substrates
) -> dict[int, float]:
    """Unit -> kcat (1/s) for one database tier, evaluated on every unit it reaches."""
    gene_to_uniprot = uniprot_for_genes(model_dir, units.gene_ids)
    table = _database_tiers(
        units,
        gene_to_uniprot,
        organism,
        homolog,
        substrates,
        KineticKind.KCAT,
        TARGET_TEMPERATURE_C,
        "min",
    )
    return {u: float(v) for u, (v, _, _) in table.items()}


def pair(
    name_a: str, a: dict[int, float], name_b: str, b: dict[int, float]
) -> Pair | None:
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return None
    x = np.log10([a[u] for u in shared])
    y = np.log10([b[u] for u in shared])
    d = y - x
    return Pair(
        a=name_a,
        b=name_b,
        n=len(shared),
        spearman=float(spearmanr(x, y).statistic),
        pearson_log10=float(pearsonr(x, y).statistic),
        median_abs_decades=float(np.median(np.abs(d))),
        frac_within_one_decade=float(np.mean(np.abs(d) <= ONE_DECADE)),
        median_signed_decades=float(np.median(d)),
    )


def main() -> None:
    source = YeastGEM()
    gem = build_gem_tensors(source.model, source.model_dir)
    units = gem.catalytic_units
    kinetics_dir = osp.join(DATA_ROOT, KINETICS_MIRROR)
    substrates = SubstrateMatch.from_verified_pairs(
        osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
    )
    organism = load_mirrored_records(OED_SCEREVISIAE)
    # Cross-organism means cross-organism: the all-organism mirror carries the yeast rows
    # too, and left in they would let the EC key find the very assay the measured tier
    # holds and report an agreement that is the same row read twice.
    everyone = [r for r in load_mirrored_records(OED_ALL) if r.organism != YEAST]

    tiers: dict[str, dict[int, float]] = {}
    # Measured alone: no homolog rows, so a unit without an accession match is absent.
    tiers["measured"] = tier_values(units, source.model_dir, organism, [], substrates)
    # Homolog on EVERY unit: no organism rows, so nothing is taken by the accession key
    # first and the EC key is evaluated for all units, including the 97 measured ones.
    tiers["homolog"] = tier_values(units, source.model_dir, [], everyone, substrates)

    plausibility = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
    selection = PredictorSelection.from_study("k_cat", plausibility)
    for name in selection.selected:
        table = load_predicted_unit_table(
            "k_cat", units.n_units, kinetics_dir, plausibility, restrict_to=name
        )
        tiers[name] = {
            u: float(table.values[u])
            for u in range(units.n_units)
            if bool(table.mask[u])
        }
    median = load_predicted_unit_table(
        "k_cat", units.n_units, kinetics_dir, plausibility
    )
    tiers["predicted_median"] = {
        u: float(median.values[u]) for u in range(units.n_units) if bool(median.mask[u])
    }

    pairs = [
        p
        for name_a, name_b in combinations(tiers, 2)
        if (p := pair(name_a, tiers[name_a], name_b, tiers[name_b])) is not None
    ]
    report = Report(
        n_units=int(units.n_units),
        tiers=[
            Tier(
                name=k,
                n_units=len(v),
                median_log10=float(np.median(np.log10(list(v.values())))),
            )
            for k, v in tiers.items()
        ],
        pairs=pairs,
        admitted_predictors=selection.selected,
        best_predictor=selection.ranked[0][0],
        note=(
            "Each tier evaluated on every unit it reaches, without the table's fall-through; "
            "the homolog tier here excludes Saccharomyces cerevisiae rows; "
            "values in 1/s under the shipped conventions (substrate-matched rows, min over a "
            "complex's subunits, assay nearest 30 C). Decades are log10 differences."
        ),
    )
    for t in report.tiers:
        print(f"{t.name:18s} n={t.n_units:5d} median log10 {t.median_log10:6.2f}")
    for p in pairs:
        print(
            f"{p.a:18s} vs {p.b:18s} n={p.n:5d} rho={p.spearman:5.2f} r={p.pearson_log10:5.2f} "
            f"|d|={p.median_abs_decades:4.2f} within1={p.frac_within_one_decade:4.2f} "
            f"signed={p.median_signed_decades:+5.2f}"
        )
    out = osp.join(RESULTS_DIR, "kcat_tier_agreement.json")
    with open(out, "w") as fh:
        fh.write(report.model_dump_json(indent=2))
    print(out)


if __name__ == "__main__":
    main()
