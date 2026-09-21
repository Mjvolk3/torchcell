# experiments/028-gene-essentiality/scripts/consensus_missed.py
# [[experiments.028-gene-essentiality.scripts.consensus_missed]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/consensus_missed

r"""The essential genes every predictor misses, and what they share.

``hard_genes.py`` and ``misclass_enrichment.py`` read one model at a time on the released
195 genes, of which 31 are essential. This script reads the ten repeated outer splits
instead, where each labeled gene is held out about twice, and intersects the predictors:

* the transformer with the single-experiment module (``w104``, nine seeds per split);
* the same transformer with no metabolic column (``w94`` and ``w121``, nine seeds);
* the graph convolution on the GO multi-hot (six seeds);
* the DCell-style hierarchy with a shared input weight (three seeds);
* FBA alone, scored as one minus the YPD knockout growth ratio.

A gene's misrank under one predictor is the fraction of the other class on the wrong side
of it inside its own test block (``misclass_enrichment.misrank``), averaged over the
seeds and over the splits that hold the gene out. The **consensus misrank** of a gene is
the MINIMUM of that over the four learned predictors: the best any of them does. A high
consensus misrank means every learned predictor ranks the gene on the wrong side; no
threshold is chosen for the enrichment. Terms are tested as in
``misclass_enrichment.py``: a one-sided Mann-Whitney U of the consensus misrank of the
class genes a term annotates against the rest of the class, Benjamini-Hochberg within the
class. Beside the GO biological-process terms three binary features are tested: an FBA
growth ratio of 1.0 on YPD, a physical degree under the class median, and a count of
propagated GO biological-process terms under the class median (a thinly annotated gene).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/consensus_missed.py
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
import os
import os.path as osp
import sys

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import mannwhitneyu, spearmanr

load_dotenv()
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from hard_genes import common_names  # noqa: E402
from misclass_enrichment import (  # noqa: E402
    MIN_GENES,
    TermRow,
    benjamini_hochberg,
    go_annotations,
    misrank,
    nonredundant,
    physical_degree,
)

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
PRED_DIR = os.getenv("TC_TEST_PRED_DIR", osp.join(DATA_ROOT, "test-predictions-028"))
N_SPLITS = 10
#: A gene counts as missed by one predictor, for the descriptive list only, above this.
MISSED = 0.5

#: label -> (transformer cell patterns, graph file, graph model row)
LEARNED: dict[str, tuple[tuple[str, ...], str | None, str | None]] = {
    "transformer, single-experiment module": (
        ("w104_resplit{r}ypdgpr_h12l1p1_e200_s*.json",),
        None,
        None,
    ),
    "transformer, no metabolism": (
        (
            "w94_resplit{r}pathpool_h12l1p1_e200_s*.json",
            "w121_resplit{r}pathpool_h12l1p1_e200_s*.json",
        ),
        None,
        None,
    ),
    "graph convolution on GO": (
        (),
        "graph_baselines_go400_s6_fcl_resplit{r}.json",
        "GCN",
    ),
    "DCell hierarchy, shared input": ((), "dcell_style_resplits.json", None),
}
FBA = "FBA growth ratio, YPD"


class GeneRow(BaseModel):
    """One labeled gene that some repeated split holds out."""

    gene: str
    name: str
    essential: int
    n_splits_held_out: int
    misrank: dict[str, float]
    consensus: float
    n_learned_missing: int
    fba_ratio_ypd: float
    n_go_terms: int
    physical_degree: int


class ClassReport(BaseModel):
    """One class: the enrichment of the consensus misrank and the genes all four miss."""

    gene_class: str
    n_genes: int
    mean_consensus: float
    n_missed_by_all_learned: int
    n_missed_by_all_learned_and_fba: int
    spearman_consensus_vs_go_terms: float
    spearman_consensus_vs_degree: float
    n_terms_tested: int
    n_terms_q_below_0_1: int
    top: list[TermRow]
    missed_by_all_learned: list[str]


class Report(BaseModel):
    predictors: list[str]
    cells_per_split: dict[str, list[int]]
    n_labeled: int
    n_held_out_at_least_once: int
    #: predictor -> mean misrank of the essential genes, and how many sit above MISSED.
    essential_mean_misrank: dict[str, float]
    essential_n_missed: dict[str, int]
    #: index k -> essential genes that exactly k of the learned predictors miss.
    essential_missed_by_k_learned: list[int]
    agreement_essential: dict[str, dict[str, float]]
    classes: list[ClassReport]
    genes: list[GeneRow]
    note: str


def split(r: int) -> dict[str, int]:
    """Gene -> label; the test block itself is read off the predictors' own dumps."""
    path = osp.join(RESULTS, "splits", f"ess_fcl_resplit{r}_val0.2_s0.json.gz")
    with gzip.open(path, "rt") as fh:
        return {g: int(v) for g, v in json.load(fh)["labels"].items()}


def transformer_block(patterns: tuple[str, ...], r: int) -> list[dict[str, float]]:
    cells = []
    for pat in patterns:
        for path in sorted(glob.glob(osp.join(RESULTS, pat.format(r=r)))):
            with open(path) as fh:
                dump_path = json.load(fh)["test"]["dump_path"]
            if not osp.isfile(dump_path):
                dump_path = osp.join(PRED_DIR, osp.basename(dump_path))
            with open(dump_path) as fh:
                preds = [
                    p for p in json.load(fh)["predictions"] if p.get("pinned", True)
                ]
            genes = [p["genes"][0] for p in preds]
            x = np.array([float(p["pred"]) for p in preds])
            y = np.array([int(round(float(p["target"]))) for p in preds])
            cells.append(misrank(genes, x, y))
    return cells


def graph_block(
    file: str, model: str | None, r: int, labels: dict[str, int]
) -> list[dict[str, float]]:
    with open(osp.join(RESULTS, file.format(r=r))) as fh:
        rows = json.load(fh)["rows"]
    if model is None:
        rows = [
            row
            for row in rows
            if row["model"] == "dcell_style_shared"
            and row["split_file"] == f"ess_fcl_resplit{r}_val0.2_s0.json.gz"
        ]
    else:
        rows = [
            row
            for row in rows
            if row["model"] == model and row["inputs"] == "GO (400 epochs)"
        ]
    assert len(rows) == 1, (file, r, len(rows))
    genes = rows[0]["test_genes"]
    y = np.array([labels[g] for g in genes])
    return [misrank(genes, np.array(z, dtype=float), y) for z in rows[0]["test_logits"]]


def main() -> None:
    with open(osp.join(RESULTS, "fba_essentiality_per_gene.csv")) as fh:
        ratio = {row["gene"]: float(row["YPD"]) for row in csv.DictReader(fh)}

    predictors = [*LEARNED, FBA]
    per_gene: dict[str, dict[str, list[float]]] = {}
    held: dict[str, int] = {}
    cells_per_split: dict[str, list[int]] = {p: [] for p in predictors}
    labels: dict[str, int] = {}
    for r in range(N_SPLITS):
        labels = split(r)
        blocks: dict[str, list[dict[str, float]]] = {}
        for name, (patterns, file, model) in LEARNED.items():
            blocks[name] = (
                transformer_block(patterns, r)
                if patterns
                else graph_block(str(file), model, r, labels)
            )
        test = sorted(blocks[next(iter(LEARNED))][0])
        for name, cells in blocks.items():
            assert cells and all(set(c) == set(test) for c in cells), (name, r)
        y = np.array([labels[g] for g in test])
        blocks[FBA] = [misrank(test, np.array([1.0 - ratio[g] for g in test]), y)]
        for name, cells in blocks.items():
            cells_per_split[name].append(len(cells))
            for g in test:
                per_gene.setdefault(g, {}).setdefault(name, []).append(
                    float(np.mean([c[g] for c in cells]))
                )
        for g in test:
            held[g] = held.get(g, 0) + 1

    genes = sorted(per_gene)
    term_genes, term_name = go_annotations(set(genes))
    n_terms = {g: sum(g in gs for gs in term_genes.values()) for g in genes}
    degree = physical_degree()
    names = common_names(genes)

    rows: list[GeneRow] = []
    for g in genes:
        mis = {p: float(np.mean(v)) for p, v in per_gene[g].items()}
        learned = [mis[p] for p in LEARNED]
        rows.append(
            GeneRow(
                gene=g,
                name=names[g],
                essential=labels[g],
                n_splits_held_out=held[g],
                misrank=mis,
                consensus=float(min(learned)),
                n_learned_missing=int(sum(v > MISSED for v in learned)),
                fba_ratio_ypd=ratio[g],
                n_go_terms=n_terms[g],
                physical_degree=int(degree.get(g, 0)),
            )
        )

    essential = [row for row in rows if row.essential == 1]
    agreement = {
        a: {
            b: float(
                spearmanr(
                    [row.misrank[a] for row in essential],
                    [row.misrank[b] for row in essential],
                ).statistic
            )
            for b in predictors
        }
        for a in predictors
    }

    classes: list[ClassReport] = []
    for cls, cls_name in ((1, "essential, ranked low"), (0, "nonessential, ranked high")):
        members = [row for row in rows if row.essential == cls]
        ids = {row.gene for row in members}
        values = {row.gene: row.consensus for row in members}
        features: dict[str, set[str]] = {t: gs & ids for t, gs in term_genes.items()}
        label = dict(term_name)
        med_degree = float(np.median([row.physical_degree for row in members]))
        med_terms = float(np.median([row.n_go_terms for row in members]))
        features["fba"] = {row.gene for row in members if row.fba_ratio_ypd >= 0.999}
        label["fba"] = "FBA knockout growth ratio 1.0"
        features["degree"] = {
            row.gene for row in members if row.physical_degree < med_degree
        }
        label["degree"] = "physical degree under the class median"
        features["thin"] = {row.gene for row in members if row.n_go_terms < med_terms}
        label["thin"] = "GO process terms under the class median"
        tested = {
            t: gs
            for t, gs in features.items()
            if MIN_GENES <= len(gs) <= len(members) / 2
            or (t in ("fba", "degree", "thin") and MIN_GENES <= len(gs) < len(members))
        }
        term_rows = []
        for t, gs in tested.items():
            inside = np.array([values[g] for g in gs])
            outside = np.array([values[g] for g in ids - gs])
            u = mannwhitneyu(inside, outside, alternative="greater")
            term_rows.append(
                TermRow(
                    term=t,
                    name=label[t],
                    n_annotated=len(gs),
                    n_class=len(members),
                    mean_misrank_annotated=float(inside.mean()),
                    mean_misrank_rest=float(outside.mean()),
                    effect=float(u.statistic / (len(inside) * len(outside))),
                    p=float(u.pvalue),
                    q=1.0,
                )
            )
        q = benjamini_hochberg(np.array([t.p for t in term_rows]))
        for t, qi in zip(term_rows, q, strict=True):
            t.q = float(qi)
        term_rows.sort(key=lambda t: (t.q, -t.effect))
        all_learned = sorted(
            (row for row in members if row.n_learned_missing == len(LEARNED)),
            key=lambda row: -row.consensus,
        )
        classes.append(
            ClassReport(
                gene_class=cls_name,
                n_genes=len(members),
                mean_consensus=float(np.mean(list(values.values()))),
                n_missed_by_all_learned=len(all_learned),
                n_missed_by_all_learned_and_fba=sum(
                    row.misrank[FBA] >= MISSED for row in all_learned
                ),
                spearman_consensus_vs_go_terms=float(
                    spearmanr(
                        [row.consensus for row in members],
                        [row.n_go_terms for row in members],
                    ).statistic
                ),
                spearman_consensus_vs_degree=float(
                    spearmanr(
                        [row.consensus for row in members],
                        [row.physical_degree for row in members],
                    ).statistic
                ),
                n_terms_tested=len(term_rows),
                n_terms_q_below_0_1=sum(t.q < 0.1 for t in term_rows),
                top=nonredundant(term_rows, features),
                missed_by_all_learned=[row.gene for row in all_learned],
            )
        )
        c = classes[-1]
        print(
            f"== {cls_name}: {c.n_genes} genes, mean consensus {c.mean_consensus:.3f}, "
            f"missed by all four learned {c.n_missed_by_all_learned} "
            f"(and by FBA {c.n_missed_by_all_learned_and_fba}); rho vs GO terms "
            f"{c.spearman_consensus_vs_go_terms:+.3f}, vs degree "
            f"{c.spearman_consensus_vs_degree:+.3f}; {c.n_terms_q_below_0_1} of "
            f"{c.n_terms_tested} terms q<0.1"
        )
        for t in c.top:
            print(
                f"   {t.name[:58]:58s} n {t.n_annotated:3d} in {t.mean_misrank_annotated:.3f} "
                f"out {t.mean_misrank_rest:.3f} effect {t.effect:.2f} q {t.q:.4f}"
            )
        for row in all_learned[:25]:
            print(
                f"      {row.gene:9s} {row.name:8s} consensus {row.consensus:.2f} "
                f"fba {row.fba_ratio_ypd:.2f} go {row.n_go_terms:3d} "
                f"deg {row.physical_degree:4d} held {row.n_splits_held_out}"
            )

    out = osp.join(RESULTS, "consensus_missed.json")
    with open(out, "w") as fh:
        json.dump(
            Report(
                predictors=predictors,
                cells_per_split=cells_per_split,
                n_labeled=len(labels),
                n_held_out_at_least_once=len(rows),
                essential_mean_misrank={
                    p: float(np.mean([row.misrank[p] for row in essential]))
                    for p in predictors
                },
                essential_n_missed={
                    p: int(sum(row.misrank[p] > MISSED for row in essential))
                    for p in predictors
                },
                essential_missed_by_k_learned=[
                    int(sum(row.n_learned_missing == k for row in essential))
                    for k in range(len(LEARNED) + 1)
                ],
                agreement_essential=agreement,
                classes=classes,
                genes=rows,
                note=(
                    "Consensus misrank is the minimum over the four learned predictors "
                    "of a gene's mean misrank over seeds and over the repeated splits "
                    f"that hold it out. 'Missed' in the gene lists means above {MISSED}. "
                    "Test blocks overlap across splits, so the q-values rank terms and "
                    "are not calibrated error rates."
                ),
            ).model_dump(),
            fh,
            indent=2,
        )
    print(out)


if __name__ == "__main__":
    main()
