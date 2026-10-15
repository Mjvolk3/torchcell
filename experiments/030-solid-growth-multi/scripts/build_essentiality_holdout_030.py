# experiments/030-solid-growth-multi/scripts/build_essentiality_holdout_030.py
# [[experiments.030-solid-growth-multi.scripts.build_essentiality_holdout_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/build_essentiality_holdout_030
"""A gene-level essentiality holdout over the single records of the 030 build.

Universe: the 5,694 single records, one per gene (asserted). Positives are genes with a
``GeneEssentialitySgdDataset`` entry (a converted fitness 0), negatives are genes whose
single carries only measured entries. Two held-out sets, disjoint, whose single records
leave the training pool (their doubles and triples stay):

(a) ``released``: every Merzbacher 2025 released TEST gene that resolves as a 030 single,
    labeled by the released file (the ``essential`` column is a viability flag and is
    inverted once, with the class counts asserted against the paper's Methods, as in
    028's ``read_fcl_essentiality_split``). 028 resolved 195 of the 223 in its build;
    the 030 count is reported, not assumed.
(b) ``matched``: 250 essential genes drawn from the essentials that ALSO carry a measured
    single (the 907 rule: label origin must not coincide with class) plus 250
    non-essential genes, greedy nearest-neighbor matched at seed 0 on (number of S3
    closure doubles containing the gene, number of triples containing the gene), all
    disjoint from (a). Acceptance: the coverage count (doubles + triples) as a score for
    essential vs non-essential on (b) must give a Mann-Whitney AUROC in [0.47, 0.53],
    so higher-order coverage cannot carry the label.

The PPI-degree confound of 028 (0.74 on its genome-wide set) is measured on both sets
with the SGD physical interaction graph (``$DATA_ROOT/data/sgd/genome/graph/
G_physical.pkl``, the graph 028's ``degree_affine_store.py`` used); a gene absent from
the graph has degree 0, and how many are absent is reported.

Reads ``closure/entries.parquet`` (singles, doubles, triples with their gene strings),
the released CSV under ``$DATA_ROOT/data/merzbacher2025_fcl/``, and script 2's pinned
val/test indices (the exclusion must not touch them; asserted).

Writes (``experiments/030-solid-growth-multi/results/``):

- ``essentiality_holdout_030.json.gz``: ``{"report": EssentialityHoldoutReport,
  "excluded_record_indices": [...], "released": {"genes", "labels", "record_indices"},
  "matched": {same}}``.
- ``essentiality_holdout_030_summary.json``: the report alone.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/build_essentiality_holdout_030.py
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import os.path as osp
import pickle
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from scipy import stats

SEED = 0
N_PER_CLASS = 250
ACCEPTANCE_BAND = (0.47, 0.53)
ESSENTIALITY_DATASET = "GeneEssentialitySgdDataset"
N_RELEASED_TEST = 223
N_RESOLVED_028 = 195
N_EXPECTED_SGD = 1_140
#: Merzbacher 2025's released yeast split, columns ``knockout``, ``essential``, ``test``.
FCL_ESSENTIALITY_SPLIT = (
    "data/merzbacher2025_fcl/deletionprediction-main/data/"
    "yeast_essentiality_test_split.csv"
)
#: The column named ``essential`` carries 1.0 for a NONESSENTIAL gene (028, verified
#: against the paper's Methods: "192 nonessential; 31 essential" held out).
FCL_ESSENTIAL_VALUE = 0.0
FCL_EXPECTED_COUNTS = {
    ("0", "nonessential"): 772,
    ("0", "essential"): 126,
    ("1", "nonessential"): 192,
    ("1", "essential"): 31,
}


class HoldoutSet(BaseModel):
    """Counts of one held-out gene set."""

    n_genes: int
    n_essential: int
    n_nonessential: int
    n_record_indices: int


class MatchingReport(BaseModel):
    """How the matched set was drawn and how well the covariates balance."""

    seed: int
    n_per_class: int
    covariates: list[str]
    pool_essential_with_measured: int
    pool_nonessential: int
    distance_mean: float
    distance_median: float
    distance_max: float
    n_exact_matches: int
    coverage_auroc_unmatched_pools: float = Field(
        description="coverage count as a score, essential-with-measured vs all negatives"
    )
    coverage_auroc_matched: float
    acceptance_band: tuple[float, float]
    accepted: bool
    doubles_median_essential: float
    doubles_median_nonessential: float
    triples_median_essential: float
    triples_median_nonessential: float


class PpiDegreeReport(BaseModel):
    """PPI degree as an essentiality score on both held-out sets."""

    source: str
    n_nodes: int
    n_edges: int
    auroc_released: float
    auroc_matched: float
    n_released_genes_absent_from_graph: int
    n_matched_genes_absent_from_graph: int


class EssentialityHoldoutReport(BaseModel):
    """Everything a reader of the holdout needs to know about how it was built."""

    entries_parquet: str
    n_single_records: int
    n_single_genes: int
    one_record_per_gene: bool
    n_essential_sgd: int
    n_measured: int
    n_essential_with_measured: int
    n_essential_only: int
    n_nonessential_measured_only: int
    released_split_path: str
    n_released_rows: int
    n_released_unlabeled_rows: int
    n_released_test_genes: int
    n_released_test_in_030_singles: int
    n_released_test_resolved_028: int
    released_test_absent_from_030: list[str]
    released_vs_030_label_disagreements: list[dict[str, Any]]
    released: HoldoutSet
    matched: HoldoutSet
    matching: MatchingReport
    ppi_degree: PpiDegreeReport | None
    n_excluded_record_indices: int
    excluded_all_order_1: bool
    excluded_disjoint_from_pinned_val_test: bool
    label_convention: str = (
        "released: the CSV column `essential` is a viability flag, inverted once, class "
        "counts asserted; matched: 1 = a GeneEssentialitySgdDataset entry under the "
        "single, 0 = measured entries only"
    )


def released_label(raw: str) -> int:
    """1 if essential else 0, from the released ``essential`` column value."""
    return int(float(raw) == FCL_ESSENTIAL_VALUE)


def read_released_split(path: str) -> tuple[dict[str, int], set[str], int, int]:
    """``(gene -> label, test genes, n rows, n unlabeled rows)``.

    Rows with an empty ``essential`` cell carry no label (38 of 1,159 in the released
    file, as 028's reader also counts); they are skipped and counted, and the class
    counts of the labeled rows must equal the paper's Methods sentence, which also
    proves every one of the 223 test genes is labeled. A file with other counts is a
    different experiment and is refused.
    """
    labels: dict[str, int] = {}
    test_genes: set[str] = set()
    counts: dict[tuple[str, str], int] = {}
    n_rows = 0
    n_unlabeled = 0
    with open(path) as handle:
        for row in csv.DictReader(handle):
            n_rows += 1
            gene = str(row["knockout"]).strip()
            raw = str(row["essential"]).strip()
            if not raw:
                n_unlabeled += 1
                continue
            essential = released_label(raw)
            labels[gene] = essential
            split = str(row["test"]).strip()
            if split == "1":
                test_genes.add(gene)
            key = (split, "essential" if essential else "nonessential")
            counts[key] = counts.get(key, 0) + 1
    assert counts == FCL_EXPECTED_COUNTS, (
        f"released split at {path} has class counts {counts}, not {FCL_EXPECTED_COUNTS}"
    )
    return labels, test_genes, n_rows, n_unlabeled


def auroc(positive: list[float], negative: list[float]) -> float:
    """Mann-Whitney AUROC of a score for positives against negatives."""
    u = stats.mannwhitneyu(positive, negative, alternative="two-sided").statistic
    return float(u) / (len(positive) * len(negative))


def greedy_match(
    cases: list[str],
    controls: list[str],
    covariate: dict[str, tuple[float, ...]],
    rng: np.random.Generator,
) -> list[tuple[str, str, float]]:
    """Each case takes its nearest unused control (Euclidean), ties broken at random.

    Cases are visited in the given order; a control is used once. Returns
    ``(case, control, distance)`` triples.
    """
    assert len(controls) >= len(cases), "fewer controls than cases"
    control_x = np.array([covariate[c] for c in controls], dtype=float)
    used = np.zeros(len(controls), dtype=bool)
    out: list[tuple[str, str, float]] = []
    for case in cases:
        d = np.sqrt(((control_x - np.array(covariate[case], dtype=float)) ** 2).sum(1))
        d[used] = np.inf
        best = np.flatnonzero(d == d.min())
        j = int(rng.choice(best))
        used[j] = True
        out.append((case, controls[j], float(d[j])))
    return out


def gene_coverage(gene_strings: pd.Series) -> Counter[str]:
    """How many of the given records (one gene string each) contain each gene."""
    counts: Counter[str] = Counter()
    for gs in gene_strings:
        counts.update(gs.split("|"))
    return counts


def physical_degree(path: str) -> tuple[dict[str, int], int, int]:
    """Degree per gene in the SGD physical interaction graph pickle."""
    with open(path, "rb") as f:
        wrapped = pickle.load(f)
    graph = wrapped.graph
    return dict(graph.degree()), graph.number_of_nodes(), graph.number_of_edges()


def main() -> None:
    """Build (a) and (b), run the acceptance check, write the holdout artifact."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    exp_030 = osp.join(data_root, "data/torchcell/experiments/030-solid-growth-multi")
    entries_path = osp.join(exp_030, "closure", "entries.parquet")
    results = osp.join(experiment_root, "030-solid-growth-multi", "results")
    os.makedirs(results, exist_ok=True)

    entries = pd.read_parquet(
        entries_path, columns=["idx", "order", "genes", "dataset"]
    )
    singles = entries[entries["order"] == 1]
    single_records = singles[["idx", "genes"]].drop_duplicates()
    one_per_gene = bool(
        single_records["idx"].is_unique and single_records["genes"].is_unique
    )
    assert one_per_gene, "a gene has two single records or a single has two genes"
    idx_of_gene = {
        g: int(i)
        for i, g in zip(single_records["idx"], single_records["genes"], strict=True)
    }
    essential = set(singles.loc[singles["dataset"] == ESSENTIALITY_DATASET, "genes"])
    measured = set(singles.loc[singles["dataset"] != ESSENTIALITY_DATASET, "genes"])
    assert len(essential) == N_EXPECTED_SGD, len(essential)
    both = essential & measured
    negatives = measured - essential
    label_030 = {g: int(g in essential) for g in idx_of_gene}

    doubles = entries.loc[entries["order"] == 2, ["idx", "genes"]].drop_duplicates()
    triples = entries.loc[entries["order"] == 3, ["idx", "genes"]].drop_duplicates()
    n_doubles = gene_coverage(doubles["genes"])
    n_triples = gene_coverage(triples["genes"])
    covariate: dict[str, tuple[float, ...]] = {
        str(g): (float(n_doubles.get(g, 0)), float(n_triples.get(g, 0)))
        for g in idx_of_gene
    }

    def coverage(g: str) -> float:
        return covariate[g][0] + covariate[g][1]

    # (a) released test genes present as 030 singles
    released_path = osp.join(data_root, FCL_ESSENTIALITY_SPLIT)
    released_labels, released_test, n_released_rows, n_unlabeled = read_released_split(
        released_path
    )
    assert len(released_test) == N_RELEASED_TEST, len(released_test)
    a_genes = sorted(g for g in released_test if g in idx_of_gene)
    a_absent = sorted(released_test - set(idx_of_gene))
    a_labels = {g: released_labels[g] for g in a_genes}
    disagreements = [
        {"gene": g, "released": a_labels[g], "build_030": label_030[g]}
        for g in a_genes
        if a_labels[g] != label_030[g]
    ]

    # (b) matched set, disjoint from (a)
    rng = np.random.default_rng(SEED)
    pos_pool = sorted(both - set(a_genes))
    neg_pool = sorted(negatives - set(a_genes))
    cases = [str(g) for g in rng.choice(pos_pool, N_PER_CLASS, replace=False)]
    pairs = greedy_match(cases, neg_pool, covariate, rng)
    controls = [c for _, c, _ in pairs]
    distances = np.array([d for _, _, d in pairs])
    b_genes = sorted(cases + controls)
    b_labels = {g: label_030[g] for g in b_genes}
    assert sum(b_labels.values()) == N_PER_CLASS and len(b_genes) == 2 * N_PER_CLASS
    assert not set(a_genes) & set(b_genes), "(a) and (b) overlap"

    cov_auroc = auroc([coverage(g) for g in cases], [coverage(g) for g in controls])
    accepted = ACCEPTANCE_BAND[0] <= cov_auroc <= ACCEPTANCE_BAND[1]
    matching = MatchingReport(
        seed=SEED,
        n_per_class=N_PER_CLASS,
        covariates=[
            "n_s3_closure_doubles_containing_gene",
            "n_triples_containing_gene",
        ],
        pool_essential_with_measured=len(pos_pool),
        pool_nonessential=len(neg_pool),
        distance_mean=float(distances.mean()),
        distance_median=float(np.median(distances)),
        distance_max=float(distances.max()),
        n_exact_matches=int((distances == 0).sum()),
        coverage_auroc_unmatched_pools=auroc(
            [coverage(g) for g in both], [coverage(g) for g in negatives]
        ),
        coverage_auroc_matched=cov_auroc,
        acceptance_band=ACCEPTANCE_BAND,
        accepted=accepted,
        doubles_median_essential=float(np.median([covariate[g][0] for g in cases])),
        doubles_median_nonessential=float(
            np.median([covariate[g][0] for g in controls])
        ),
        triples_median_essential=float(np.median([covariate[g][1] for g in cases])),
        triples_median_nonessential=float(
            np.median([covariate[g][1] for g in controls])
        ),
    )
    print(f"coverage AUROC on (b): {cov_auroc:.4f}; band {ACCEPTANCE_BAND}", flush=True)

    # PPI degree confound
    ppi_path = osp.join(data_root, "data/sgd/genome/graph/G_physical.pkl")
    degree, n_nodes, n_edges = physical_degree(ppi_path)

    def deg(g: str) -> float:
        return float(degree.get(g, 0))

    ppi = PpiDegreeReport(
        source=ppi_path,
        n_nodes=n_nodes,
        n_edges=n_edges,
        auroc_released=auroc(
            [deg(g) for g in a_genes if a_labels[g] == 1],
            [deg(g) for g in a_genes if a_labels[g] == 0],
        ),
        auroc_matched=auroc([deg(g) for g in cases], [deg(g) for g in controls]),
        n_released_genes_absent_from_graph=sum(g not in degree for g in a_genes),
        n_matched_genes_absent_from_graph=sum(g not in degree for g in b_genes),
    )

    # exclusion set and its invariants
    a_idx = sorted(idx_of_gene[g] for g in a_genes)
    b_idx = sorted(idx_of_gene[g] for g in b_genes)
    excluded = sorted(set(a_idx) | set(b_idx))
    assert len(excluded) == len(a_idx) + len(b_idx)
    with open(
        osp.join(
            exp_030, "001-multi-build", "processed", "perturbation_count_index.json"
        )
    ) as f:
        order_1 = set(json.load(f)["1"])
    all_order_1 = all(i in order_1 for i in excluded)
    with gzip.open(
        osp.join(results, "pinned_splits_from_010_seed_42.json.gz"), "rt"
    ) as f:
        pinned = json.load(f)["pinned"]
    pinned_eval = set(pinned["val"]) | set(pinned["test"])
    disjoint = not (set(excluded) & pinned_eval)

    report = EssentialityHoldoutReport(
        entries_parquet=entries_path,
        n_single_records=len(single_records),
        n_single_genes=len(idx_of_gene),
        one_record_per_gene=one_per_gene,
        n_essential_sgd=len(essential),
        n_measured=len(measured),
        n_essential_with_measured=len(both),
        n_essential_only=len(essential - measured),
        n_nonessential_measured_only=len(negatives),
        released_split_path=released_path,
        n_released_rows=n_released_rows,
        n_released_unlabeled_rows=n_unlabeled,
        n_released_test_genes=len(released_test),
        n_released_test_in_030_singles=len(a_genes),
        n_released_test_resolved_028=N_RESOLVED_028,
        released_test_absent_from_030=a_absent,
        released_vs_030_label_disagreements=disagreements,
        released=HoldoutSet(
            n_genes=len(a_genes),
            n_essential=sum(a_labels.values()),
            n_nonessential=len(a_genes) - sum(a_labels.values()),
            n_record_indices=len(a_idx),
        ),
        matched=HoldoutSet(
            n_genes=len(b_genes),
            n_essential=sum(b_labels.values()),
            n_nonessential=len(b_genes) - sum(b_labels.values()),
            n_record_indices=len(b_idx),
        ),
        matching=matching,
        ppi_degree=ppi,
        n_excluded_record_indices=len(excluded),
        excluded_all_order_1=all_order_1,
        excluded_disjoint_from_pinned_val_test=disjoint,
    )
    print(report.model_dump_json(indent=2))

    payload = {
        "report": report.model_dump(mode="json"),
        "excluded_record_indices": excluded,
        "released": {"genes": a_genes, "labels": a_labels, "record_indices": a_idx},
        "matched": {"genes": b_genes, "labels": b_labels, "record_indices": b_idx},
    }
    with gzip.open(osp.join(results, "essentiality_holdout_030.json.gz"), "wt") as f:
        json.dump(payload, f)
    with open(osp.join(results, "essentiality_holdout_030_summary.json"), "w") as f:
        f.write(report.model_dump_json(indent=2))

    assert accepted, f"coverage AUROC {cov_auroc:.4f} outside {ACCEPTANCE_BAND}"
    assert all_order_1, "an excluded index is not a single"
    assert disjoint, "an excluded index is a pinned val/test triple"
    print("finished: holdout written; acceptance passed")


if __name__ == "__main__":
    main()
