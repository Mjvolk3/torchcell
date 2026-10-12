# experiments/028-gene-essentiality/scripts/misclass_enrichment.py
# [[experiments.028-gene-essentiality.scripts.misclass_enrichment]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/misclass_enrichment

"""Which annotations the misranked genes share, on each held-out set and in each class.

``hard_genes.py`` names the genes the best cells misrank on the released 195. This script
asks what those genes have in common, and asks it on both held-out sets the document
scores: the released 195 metabolic genes and the 4,674 genome-wide genes the released
split never labeled.

A gene's misrank is the fraction of the other class on the wrong side of it, averaged over
a model's cells (``hard_genes.misrank``). Within one class of one held-out set, a term is
tested by a one-sided Mann-Whitney U of the misrank of the genes it annotates against the
rest of the class, so every gene contributes and no misrank threshold is chosen. The effect
size is the U statistic as a probability: the chance that an annotated gene is misranked
worse than an unannotated one, 0.5 being no association. Terms are the propagated Gene
Ontology biological-process terms annotating at least ``MIN_GENES`` genes of the class and
at most half of it, plus two binary features the document argues from: an FBA knockout
growth ratio of 1.0 on YPD, and a degree in the physical graph under the class median.
Benjamini-Hochberg runs within each (set, model, class) family.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/misclass_enrichment.py
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
from scipy.stats import mannwhitneyu

load_dotenv()
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
PRED_DIR = os.getenv("TC_TEST_PRED_DIR", osp.join(DATA_ROOT, "test-predictions-028"))

#: A term must annotate this many genes of the class to be tested.
MIN_GENES = 5
#: Terms kept per family in the report, by q-value.
TOP = 12
#: Two terms sharing more than this Jaccard of their class genes count as one.
MAX_JACCARD = 0.5
#: What counts as a specific term, as a share of the class and as a floor.
SPECIFIC_FRAC = 0.05
SPECIFIC_MIN = 12


class Model(BaseModel):
    """One model on one held-out set: where its per-gene logits live."""

    label: str
    #: Result-file globs whose prediction dumps are pooled (transformer cells).
    cells: list[str] = []
    #: A graph-baselines result file and the row's ``inputs`` (cheap models).
    graph_file: str | None = None
    graph_inputs: str | None = None
    graph_name: str | None = None
    graph_model: str = "GCN"


class HeldOutSet(BaseModel):
    """One held-out gene set, its split file and the models read on it."""

    label: str
    split: str
    models: list[Model]


SETS: list[HeldOutSet] = [
    HeldOutSet(
        label="released 195 (metabolic genes)",
        split="ess_fcl_val0.2_s0.json.gz",
        models=[
            Model(
                label="transformer, layer-free solve",
                cells=["w49_pathfbapfbalogfree_h12l1p1_e200_s*.json"],
            ),
            Model(
                label="graph convolution on GO",
                graph_file="graph_baselines_go400_s6.json",
                graph_inputs="GO (400 epochs)",
            ),
        ],
    ),
    HeldOutSet(
        label="genome-wide 4,674 (no GEM reaction)",
        split="ess_genome_heldout_val0.2_s0.json.gz",
        models=[
            Model(
                label="transformer, layer-free solve",
                cells=["w81_genomepathlog_h12l1p1_e200_s*.json"],
            ),
            Model(
                label="graph convolution on ProtT5",
                graph_file="graph_baselines_genome_heldout.json",
                graph_inputs="ProtT5",
                graph_name="union of nine",
            ),
        ],
    ),
]


class TermRow(BaseModel):
    """One term tested within one class: who it annotates and how they are misranked."""

    term: str
    name: str
    n_annotated: int
    n_class: int
    mean_misrank_annotated: float
    mean_misrank_rest: float
    #: P(annotated gene misranked worse than an unannotated one); 0.5 is no association.
    effect: float
    p: float
    q: float


class Family(BaseModel):
    """One (held-out set, model, class) family of tests, corrected together."""

    held_out_set: str
    model: str
    gene_class: str
    n_genes: int
    n_cells: int
    class_mean_misrank: float
    n_terms_tested: int
    n_terms_q_below_0_1: int
    #: By q-value, with near-duplicate terms removed (propagation makes a parent and its
    #: child carry almost the same genes; Jaccard above ``MAX_JACCARD`` drops the later).
    top: list[TermRow]
    #: By q-value among the specific terms only: those annotating at most
    #: ``max(SPECIFIC_MIN, SPECIFIC_FRAC * class size)`` genes, since a propagated parent
    #: such as "metabolic process" outranks every child by sample size alone.
    top_specific: list[TermRow]
    #: The terms at q < 0.1 by effect size, the same redundancy filter applied.
    top_by_effect: list[TermRow]
    #: The two binary features the document argues from, whatever their q-value.
    named_features: list[TermRow]
    #: Every tested term -> [effect, q], so one model's top terms can be read under another.
    all_terms: dict[str, list[float]]
    hardest_genes: list[str]


class Report(BaseModel):
    """Every family."""

    min_genes: int
    families: list[Family]


def split_labels(split: str) -> dict[str, int]:
    with gzip.open(osp.join(RESULTS, "splits", split), "rt") as fh:
        return {g: int(v) for g, v in json.load(fh)["labels"].items()}


def misrank(genes: list[str], x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Per gene, the fraction of the other class on the wrong side of it."""
    pos, neg = x[y == 1], x[y == 0]
    out = {}
    for g, xi, yi in zip(genes, x, y, strict=True):
        other = neg if yi else pos
        wrong = other > xi if yi else other < xi
        out[g] = float(np.mean(wrong) + 0.5 * np.mean(other == xi))
    return out


def model_cells(model: Model, labels: dict[str, int]) -> list[dict[str, float]]:
    """One gene -> misrank map per cell of the model."""
    cells: list[dict[str, float]] = []
    for pat in model.cells:
        for path in sorted(glob.glob(osp.join(RESULTS, pat))):
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
    if model.graph_file:
        with open(osp.join(RESULTS, model.graph_file)) as fh:
            rows = json.load(fh)["rows"]
        rows = [
            r
            for r in rows
            if r["inputs"] == model.graph_inputs
            and r["model"] == model.graph_model
            and (model.graph_name is None or r["graph"] == model.graph_name)
        ]
        assert len(rows) == 1, [(r["graph"], r["inputs"]) for r in rows]
        genes = rows[0]["test_genes"]
        y = np.array([labels[g] for g in genes])
        for logits in rows[0]["test_logits"]:
            cells.append(misrank(genes, np.array(logits, dtype=float), y))
    assert cells, model
    return cells


def go_annotations(universe: set[str]) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Propagated biological-process term -> genes of the universe, and term -> name."""
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    dag = genome.go_dag
    term_genes: dict[str, set[str]] = {}
    for term, genes in genome.go_genes.items():
        if term not in dag or dag[term].is_obsolete:
            continue
        for t in {term} | set(dag[term].get_all_parents()):
            if dag[t].namespace == "biological_process":
                term_genes.setdefault(t, set()).update(set(genes) & universe)
    return term_genes, {t: dag[t].name for t in term_genes}


def physical_degree() -> dict[str, int]:
    """Gene -> degree in the physical interaction graph, as ``degree_by_class.py`` reads it."""
    from graph_baselines import gene_graphs

    _, graphs = gene_graphs()
    return {g: int(k) for g, k in graphs["physical"].degree()}


def fba_ratio_one() -> set[str]:
    with open(osp.join(RESULTS, "fba_essentiality_per_gene.csv")) as fh:
        return {r["gene"] for r in csv.DictReader(fh) if float(r["YPD"]) >= 0.999}


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def nonredundant(rows: list[TermRow], features: dict[str, set[str]]) -> list[TermRow]:
    """The first ``TOP`` rows in the order given, skipping near-duplicates of a kept row."""
    kept: list[TermRow] = []
    for r in rows:
        a = features[r.term]
        if all(
            len(a & features[k.term]) / len(a | features[k.term]) <= MAX_JACCARD
            for k in kept
        ):
            kept.append(r)
        if len(kept) == TOP:
            break
    return kept


def main() -> None:
    all_genes: set[str] = set()
    for s in SETS:
        all_genes |= set(split_labels(s.split))
    term_genes, term_name = go_annotations(all_genes)
    degree = physical_degree()
    ratio_one = fba_ratio_one()

    families: list[Family] = []
    for s in SETS:
        labels = split_labels(s.split)
        for model in s.models:
            cells = model_cells(model, labels)
            genes = sorted(cells[0])
            mis = {g: float(np.mean([c[g] for c in cells])) for g in genes}
            for cls, cls_name in (
                (1, "essential, ranked low"),
                (0, "nonessential, ranked high"),
            ):
                members = [g for g in genes if labels[g] == cls]
                values = np.array([mis[g] for g in members])
                features: dict[str, set[str]] = {
                    t: gs & set(members) for t, gs in term_genes.items()
                }
                names = dict(term_name)
                med = float(np.median([degree[g] for g in members]))
                features["degree"] = {g for g in members if degree[g] < med}
                names["degree"] = "physical degree under the class median"
                features["fba"] = set(members) & ratio_one
                names["fba"] = "FBA knockout growth ratio 1.0"
                rows: list[TermRow] = []
                for t, gs in features.items():
                    # A GO term over half the class is a statement about the class; the two
                    # named features are tested wherever both sides have MIN_GENES.
                    cap = (
                        len(members) - MIN_GENES
                        if t in ("degree", "fba")
                        else len(members) // 2
                    )
                    if not MIN_GENES <= len(gs) <= cap:
                        continue
                    mask = np.array([g in gs for g in members])
                    u, p = mannwhitneyu(
                        values[mask], values[~mask], alternative="greater"
                    )
                    rows.append(
                        TermRow(
                            term=t,
                            name=names[t],
                            n_annotated=int(mask.sum()),
                            n_class=len(members),
                            mean_misrank_annotated=float(values[mask].mean()),
                            mean_misrank_rest=float(values[~mask].mean()),
                            effect=float(u / (mask.sum() * (~mask).sum())),
                            p=float(p),
                            q=1.0,
                        )
                    )
                q = benjamini_hochberg(np.array([r.p for r in rows]))
                for r, qi in zip(rows, q, strict=True):
                    r.q = float(qi)
                rows.sort(key=lambda r: (r.q, -r.effect))
                hardest = [members[i] for i in np.argsort(-values)[:15]]
                families.append(
                    Family(
                        held_out_set=s.label,
                        model=model.label,
                        gene_class=cls_name,
                        n_genes=len(members),
                        n_cells=len(cells),
                        class_mean_misrank=float(values.mean()),
                        n_terms_tested=len(rows),
                        n_terms_q_below_0_1=sum(r.q < 0.1 for r in rows),
                        top=nonredundant(rows, features),
                        top_specific=nonredundant(
                            [
                                r
                                for r in rows
                                if r.term not in ("degree", "fba")
                                and r.n_annotated
                                <= max(SPECIFIC_MIN, SPECIFIC_FRAC * len(members))
                            ],
                            features,
                        ),
                        top_by_effect=nonredundant(
                            sorted(
                                (r for r in rows if r.q < 0.1), key=lambda r: -r.effect
                            ),
                            features,
                        ),
                        named_features=[r for r in rows if r.term in ("degree", "fba")],
                        all_terms={r.term: [r.effect, r.q] for r in rows},
                        hardest_genes=hardest,
                    )
                )
                print(
                    f"\n{s.label} | {model.label} | {cls_name}: {len(members)} genes, "
                    f"{len(rows)} terms, {families[-1].n_terms_q_below_0_1} at q < 0.1"
                )
                for r in families[-1].named_features + families[-1].top_specific[:5]:
                    print(
                        f"  q {r.q:.3g}  effect {r.effect:.2f}  n {r.n_annotated:4d}  {r.name}"
                    )
    out = osp.join(RESULTS, "misclass_enrichment.json")
    with open(out, "w") as fh:
        json.dump(
            Report(min_genes=MIN_GENES, families=families).model_dump(), fh, indent=2
        )
    print(out)


if __name__ == "__main__":
    main()
