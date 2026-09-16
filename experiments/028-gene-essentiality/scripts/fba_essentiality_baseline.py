# experiments/028-gene-essentiality/scripts/fba_essentiality_baseline.py
# [[experiments.028-gene-essentiality.scripts.fba_essentiality_baseline]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/fba_essentiality_baseline.py

r"""FBA single-gene-deletion essentiality on yeast-GEM 9.0.2, one row per medium.

The published FBA comparator (Merzbacher 2025) is cobrapy ``single_gene_deletion`` "with
glucose as the carbon source and the standard biomass reaction", one medium, scored on all
1,121 labeled genes (AUROC 0.69). The phenotype it is compared against was called on RICH
medium (YPD, the deletion collection), and the model this experiment trains is scored on
the same genes under no medium at all (the pooled arm has no GEM) or under YPD-approx (the
flux arm). This script puts FBA on every medium the ontology defines, so the medium row of
the comparison is measured rather than assumed:

- ``shipped``: the exchange bounds yeast-GEM ships with (minimal glucose, aerobic; the
  closest reproduction of the published comparator's condition),
- ``SM``: synthetic minimal (YNB vitamins, ammonium sulfate, glucose),
- ``SC``: synthetic complete (SM plus 20 amino acids, uracil, adenine),
- ``YPD``: the YPD approximation (YNB plus 20 amino acids; peptone is not modeled).

Per gene the readout is the knockout growth rate as a fraction of the wild type on that
medium, and the essentiality score is its negative. A gene absent from the GEM has no
reaction to remove and keeps wild-type growth (score 0), which is how the published
baseline treats it too; the count of such genes is reported per gene set.

Scored on three gene sets, each with a 2,000-draw gene bootstrap: all labeled genes
(1,121), the released held-out genes (223), and the held-out genes the 028 cells resolve
(195, read from a prediction dump so the set is the one the model was scored on). AUROC
is the headline; accuracy at three growth-ratio cutoffs is reported beside the majority
rate because the published accuracy (0.81) sits below the majority rate (0.861) and cannot
separate methods.

    python experiments/028-gene-essentiality/scripts/fba_essentiality_baseline.py
"""

from __future__ import annotations

import csv
import json
import os
import os.path as osp
import sys
import time

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import single_gene_deletion
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "scripts"))

from torchcell.metabolism.media import FBA_MEDIA, media_to_bounds  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
OUT_JSON = osp.join(RESULTS, "fba_essentiality_by_medium.json")
OUT_CSV = osp.join(RESULTS, "fba_essentiality_per_gene.csv")
FCL_SPLIT = osp.join(
    DATA_ROOT,
    "data/merzbacher2025_fcl/deletionprediction-main/data/yeast_essentiality_test_split.csv",
)
#: The dump whose pinned records define "the genes the 028 cells score" (195 of the 223).
OUR_GENES_DUMP = osp.join(
    DATA_ROOT, "test-predictions-028", "wave11_ess_pool_h12l1p1_e80_s2700.json"
)
#: The released column carries 1.0 for a NONESSENTIAL gene (a viability flag).
FCL_ESSENTIAL_VALUE = 0.0
MEDIA = ("shipped", "SM", "SC", "YPD")
CUTOFFS = (0.01, 0.05, 0.5)
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
#: The FBA solution is declared zero below this growth rate (1/h), the cobrapy convention.
GROWTH_FLOOR = 1e-6


class GeneSetScore(BaseModel):
    """AUROC with a gene bootstrap and cutoff accuracies on one gene set."""

    gene_set: str
    n: int
    n_essential: int
    n_in_gem: int
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    majority_rate: float
    accuracy_at_cutoff: dict[str, float]
    recall_essential_at_cutoff: dict[str, float]


class MediumRow(BaseModel):
    """One medium: the model state, the deletion scan, and its gene-set scores."""

    medium: str
    medium_name: str
    n_exchange_bounds: int
    wild_type_growth: float
    n_genes_deleted: int
    n_zero_growth: int
    seconds: float
    scores: list[GeneSetScore]


class Report(BaseModel):
    """The whole run: GEM and solver provenance plus one row per medium."""

    gem_version: str
    cobra_version: str
    biomass_reaction: str
    n_gem_genes: int
    label_source: str
    n_labeled: int
    n_labeled_in_gem: int
    gene_sets: dict[str, int]
    bootstrap: dict[str, int]
    rows: list[MediumRow]
    note: str = Field(
        default=(
            "score = -(knockout growth / wild-type growth) on the medium; genes absent "
            "from the GEM keep wild-type growth. AUROC with a gene bootstrap; accuracy at "
            "ratio cutoffs beside the majority rate."
        )
    )


def binary_auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUROC with mid-ranks on ties; nan when a class is absent."""
    pos = label == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(score: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(score)
    draws = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        value = binary_auroc(score[idx], label[idx])
        if not np.isnan(value):
            draws.append(value)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def read_labels() -> tuple[dict[str, int], set[str]]:
    labels: dict[str, int] = {}
    held_out: set[str] = set()
    with open(FCL_SPLIT) as handle:
        for row in csv.DictReader(handle):
            raw = str(row["essential"]).strip()
            if not raw:
                continue
            gene = str(row["knockout"]).strip()
            labels[gene] = int(float(raw) == FCL_ESSENTIAL_VALUE)
            if str(row["test"]).strip() == "1":
                held_out.add(gene)
    assert len(labels) == 1121 and len(held_out) == 223, (len(labels), len(held_out))
    assert sum(labels.values()) == 157
    return labels, held_out


def read_our_genes(labels: dict[str, int]) -> set[str]:
    with open(OUR_GENES_DUMP) as handle:
        dump = json.load(handle)
    genes = {
        rec["genes"][0]
        for rec in dump["predictions"]
        if rec["pinned"] and len(rec["genes"]) == 1
    }
    assert genes <= set(labels), sorted(genes - set(labels))[:5]
    assert len(genes) == 195, len(genes)
    return genes


def medium_model(key: str) -> tuple[cobra.Model, str, int]:
    model = YeastGEM().model
    if key == "shipped":
        return model, "yeast-GEM shipped exchange bounds", 0
    bounds = media_to_bounds(FBA_MEDIA[key], model)
    unresolved = bounds.unresolved_names
    assert not unresolved, f"medium {key} has unresolved components: {unresolved}"
    bounds.apply(model)
    return model, FBA_MEDIA[key].name, len(bounds.bounds)


def score_gene_set(
    name: str,
    genes: list[str],
    labels: dict[str, int],
    ratio: dict[str, float],
    gem_genes: set[str],
) -> GeneSetScore:
    label = np.array([labels[g] for g in genes], dtype=int)
    r = np.array([ratio.get(g, 1.0) for g in genes], dtype=float)
    score = -r
    lo, hi = bootstrap_ci(score, label)
    majority = max(label.mean(), 1 - label.mean())
    acc: dict[str, float] = {}
    rec: dict[str, float] = {}
    for c in CUTOFFS:
        call = (r < c).astype(int)
        acc[str(c)] = float((call == label).mean())
        rec[str(c)] = float(call[label == 1].mean())
    return GeneSetScore(
        gene_set=name,
        n=len(genes),
        n_essential=int(label.sum()),
        n_in_gem=sum(1 for g in genes if g in gem_genes),
        auroc=binary_auroc(score, label),
        auroc_ci_low=lo,
        auroc_ci_high=hi,
        majority_rate=float(majority),
        accuracy_at_cutoff=acc,
        recall_essential_at_cutoff=rec,
    )


def main() -> None:
    labels, held_out = read_labels()
    ours = read_our_genes(labels)
    gene_sets = {
        "all_labeled": sorted(labels),
        "held_out_223": sorted(held_out),
        "ours_195": sorted(ours),
    }
    base = YeastGEM().model
    gem_genes = {g.id for g in base.genes}
    labeled_in_gem = sorted(g for g in labels if g in gem_genes)
    biomass = str(base.objective.expression).split("*")[1].split(" ")[0]
    rows: list[MediumRow] = []
    per_gene: dict[str, dict[str, float]] = {}
    for key in MEDIA:
        model, medium_name, n_bounds = medium_model(key)
        t0 = time.time()
        wt = float(model.slim_optimize())
        assert wt > GROWTH_FLOOR, f"wild type does not grow on {key}: {wt}"
        deletions = single_gene_deletion(model, gene_list=labeled_in_gem, processes=4)
        ratio: dict[str, float] = {}
        n_zero = 0
        for ids, growth in zip(deletions["ids"], deletions["growth"], strict=True):
            (gene,) = tuple(ids)
            g = (
                0.0
                if (growth is None or np.isnan(growth) or growth < GROWTH_FLOOR)
                else float(growth)
            )
            # ROUNDED TO 1e-3 OF THE WILD TYPE before ranking. Two runs of the same
            # scan differed by 0.03 AUROC on the held-out genes (jobs 2069, 2070) because
            # ~900 of 1,121 deletions grow at 0.99 to 1.00 of the wild type in 26 to 37
            # distinct solver-noise values, and the rank statistic ordered that block by
            # noise. Rounding makes the block a tie, which is what it is.
            ratio[gene] = round(g / wt, 3)
            n_zero += int(g == 0.0)
            per_gene.setdefault(gene, {})[key] = ratio[gene]
        rows.append(
            MediumRow(
                medium=key,
                medium_name=medium_name,
                n_exchange_bounds=n_bounds,
                wild_type_growth=wt,
                n_genes_deleted=len(ratio),
                n_zero_growth=n_zero,
                seconds=time.time() - t0,
                scores=[
                    score_gene_set(name, genes, labels, ratio, gem_genes)
                    for name, genes in gene_sets.items()
                ],
            )
        )
        print(
            f"[{key}] WT {wt:.4f}/h, {len(ratio)} deletions, {n_zero} lethal; "
            + ", ".join(
                f"{s.gene_set} AUROC {s.auroc:.4f} [{s.auroc_ci_low:.3f}, {s.auroc_ci_high:.3f}]"
                for s in rows[-1].scores
            ),
            flush=True,
        )
    report = Report(
        gem_version=YeastGEM().version,
        cobra_version=cobra.__version__,
        biomass_reaction=biomass,
        n_gem_genes=len(gem_genes),
        label_source=FCL_SPLIT,
        n_labeled=len(labels),
        n_labeled_in_gem=len(labeled_in_gem),
        gene_sets={k: len(v) for k, v in gene_sets.items()},
        bootstrap={"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
        rows=rows,
    )
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    table = pd.DataFrame.from_dict(per_gene, orient="index").sort_index()
    table.index.name = "gene"
    table["essential"] = [labels[g] for g in table.index]
    table["held_out"] = [int(g in held_out) for g in table.index]
    table["ours_195"] = [int(g in ours) for g in table.index]
    table.to_csv(OUT_CSV)
    print(OUT_JSON)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
