# experiments/028-gene-essentiality/scripts/family_ensembles.py
# [[experiments.028-gene-essentiality.scripts.family_ensembles]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/family_ensembles.py

r"""Averaging the two model families that miss different genes.

``hard_genes.py`` measured that graph convolution on GO and the transformer cells fail
on different held-out genes (Spearman of the per-gene misrank 0.2 to 0.4, against 0.6
to 0.9 within the transformer family). This script asks what their average scores: for
each family the per-gene logits of its split-file cells are standardized per cell and
averaged over seeds, then families are averaged with equal weight, and the mean is
scored once on the 195 held-out genes with a 2,000-draw gene bootstrap. Nothing is
fitted: the weights are equal and the cells are the ones already trained, so the number
is a post hoc average of existing predictions, not a model selected on these genes.

Writes ``results/family_ensembles.json``.

    python experiments/028-gene-essentiality/scripts/family_ensembles.py
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
PRED_DIR = os.getenv("TC_TEST_PRED_DIR", osp.join(DATA_ROOT, "test-predictions-028"))

#: Family -> the transformer result-file globs whose dumps it pools.
TRANSFORMER_FAMILIES: dict[str, list[str]] = {
    "pathway + ProtT5 + affine (round 22, 3 seeds)": [
        "w38_protpathfba_h12l1p1_e200_s*.json"
    ],
    "flux head, pathway + ProtT5 (rounds 23 and 25, 6 seeds)": [
        "w39_pathlpelgateflux_h12l1p1_e200_s*.json"
    ],
    "flux head, ProtT5 (rounds 21 and 24, 6 seeds)": [
        "w37_lpelgateflux_h12l1p1_e200_s*.json"
    ],
    "layer-free log1p vector + affine, pathway + ProtT5 (round 33, 6 seeds)": [
        "w49_pathfbapfbalog_h12l1p1_e200_s*.json"
    ],
    "cached solve + flux head, pathway + ProtT5 (round 27, 6 seeds)": [
        "w43_pathlpoffgateflux_h12l1p1_e200_s*.json"
    ],
}
#: Family -> (graph-baselines file, row inputs) for the cheap models with saved logits.
GRAPH_FAMILIES: dict[str, tuple[str, str]] = {
    "GO convolution (400 epochs, 3 seeds)": (
        "graph_baselines_go400.json",
        "GO (400 epochs)",
    ),
    "GO convolution (400 epochs, 6 seeds)": (
        "graph_baselines_go400_s6.json",
        "GO (400 epochs)",
    ),
}
#: The averages scored, as lists of family names.
COMBINATIONS: list[list[str]] = [
    ["GO convolution (400 epochs, 3 seeds)"],
    ["GO convolution (400 epochs, 6 seeds)"],
    ["pathway + ProtT5 + affine (round 22, 3 seeds)"],
    ["flux head, pathway + ProtT5 (rounds 23 and 25, 6 seeds)"],
    ["cached solve + flux head, pathway + ProtT5 (round 27, 6 seeds)"],
    ["layer-free log1p vector + affine, pathway + ProtT5 (round 33, 6 seeds)"],
    [
        "GO convolution (400 epochs, 6 seeds)",
        "layer-free log1p vector + affine, pathway + ProtT5 (round 33, 6 seeds)",
    ],
    [
        "GO convolution (400 epochs, 6 seeds)",
        "cached solve + flux head, pathway + ProtT5 (round 27, 6 seeds)",
    ],
    [
        "GO convolution (400 epochs, 6 seeds)",
        "pathway + ProtT5 + affine (round 22, 3 seeds)",
        "cached solve + flux head, pathway + ProtT5 (round 27, 6 seeds)",
    ],
    [
        "GO convolution (400 epochs, 3 seeds)",
        "pathway + ProtT5 + affine (round 22, 3 seeds)",
    ],
    [
        "GO convolution (400 epochs, 3 seeds)",
        "flux head, pathway + ProtT5 (rounds 23 and 25, 6 seeds)",
    ],
    [
        "GO convolution (400 epochs, 3 seeds)",
        "flux head, ProtT5 (rounds 21 and 24, 6 seeds)",
    ],
    [
        "GO convolution (400 epochs, 3 seeds)",
        "pathway + ProtT5 + affine (round 22, 3 seeds)",
        "flux head, pathway + ProtT5 (rounds 23 and 25, 6 seeds)",
    ],
]


class Combination(BaseModel):
    """One average and its score."""

    families: list[str]
    n_cells: int
    auroc: float
    ci_low: float
    ci_high: float


class Report(BaseModel):
    """Every average scored."""

    n_genes: int
    n_essential: int
    families: dict[str, int]
    combinations: list[Combination]


def labels_of_split() -> dict[str, int]:
    with gzip.open(
        osp.join(RESULTS, "splits", "ess_fcl_val0.2_s0.json.gz"), "rt"
    ) as fh:
        split = json.load(fh)
    return {g: int(v) for g, v in split["labels"].items()}


def transformer_cells(globs: list[str]) -> list[dict[str, float]]:
    cells: list[dict[str, float]] = []
    for pat in globs:
        for f in sorted(glob.glob(osp.join(RESULTS, pat))):
            with open(f) as fh:
                result = json.load(fh)
            path = result["test"]["dump_path"]
            if not osp.isfile(path):
                path = osp.join(PRED_DIR, osp.basename(path))
            with open(path) as fh:
                dump = json.load(fh)
            cells.append(
                {
                    p["genes"][0]: float(p["pred"])
                    for p in dump["predictions"]
                    if p.get("pinned", True)
                }
            )
    return cells


def graph_cells(fname: str, inputs: str) -> list[dict[str, float]]:
    with open(osp.join(RESULTS, fname)) as fh:
        report = json.load(fh)
    cells: list[dict[str, float]] = []
    for row in report["rows"]:
        if row["inputs"] != inputs:
            continue
        assert row.get("test_genes"), "rerun graph_baselines.py with --save-logits"
        for logits in row["test_logits"]:
            cells.append(dict(zip(row["test_genes"], map(float, logits), strict=True)))
    return cells


def family_score(cells: list[dict[str, float]], genes: list[str]) -> np.ndarray:
    """Per-cell standardized logits averaged over the family's cells."""
    z = []
    for c in cells:
        x = np.array([c[g] for g in genes])
        z.append((x - x.mean()) / x.std())
    return np.mean(z, axis=0)


#: The fold-protocol families: transformer fold stems and the GO fold arm.
FOLD_TRANSFORMER_FAMILIES: dict[str, str] = {
    "flux head, pathway + ProtT5, five folds (round 25)": "w41_pathfluxcv{k}_h12l1p1_e200_s2700.json",
    "indicator + ProtT5 + affine, five folds (round 12)": "w23_chromprotcv{k}_h12l1p1_e200_s2700.json",
    "cached solve + flux head, pathway + ProtT5, five folds (round 27)": "w43_pathoffcv{k}_h12l1p1_e200_s2700.json",
    "layer-free log1p vector + affine, pathway + ProtT5, five folds (round 34)": "w50_pathlogfreecv{k}_h12l1p1_e200_s2700.json",
    "layer-free log1p, interior-point store, five folds (round 37)": "w53_pathlogipmcv{k}_h12l1p1_e200_s2700.json",
    "layer-free log1p, PDLP store, five folds (round 39)": "w55_pathlogpdlpcv{k}_h12l1p1_e200_s2700.json",
}
FOLD_GRAPH_FAMILY = ("GO convolution, five folds (400 epochs)", "GCN400")
FOLD_COMBINATIONS: list[list[str]] = [
    [FOLD_GRAPH_FAMILY[0]],
    ["indicator + ProtT5 + affine, five folds (round 12)"],
    ["flux head, pathway + ProtT5, five folds (round 25)"],
    ["cached solve + flux head, pathway + ProtT5, five folds (round 27)"],
    ["layer-free log1p vector + affine, pathway + ProtT5, five folds (round 34)"],
    [
        FOLD_GRAPH_FAMILY[0],
        "layer-free log1p vector + affine, pathway + ProtT5, five folds (round 34)",
    ],
    ["layer-free log1p, interior-point store, five folds (round 37)"],
    [
        FOLD_GRAPH_FAMILY[0],
        "layer-free log1p, interior-point store, five folds (round 37)",
    ],
    ["layer-free log1p, PDLP store, five folds (round 39)"],
    [FOLD_GRAPH_FAMILY[0], "layer-free log1p, PDLP store, five folds (round 39)"],
    [FOLD_GRAPH_FAMILY[0], "indicator + ProtT5 + affine, five folds (round 12)"],
    [FOLD_GRAPH_FAMILY[0], "flux head, pathway + ProtT5, five folds (round 25)"],
    [FOLD_GRAPH_FAMILY[0], "cached solve + flux head, pathway + ProtT5, five folds (round 27)"],
    [
        FOLD_GRAPH_FAMILY[0],
        "indicator + ProtT5 + affine, five folds (round 12)",
        "flux head, pathway + ProtT5, five folds (round 25)",
    ],
    [
        FOLD_GRAPH_FAMILY[0],
        "indicator + ProtT5 + affine, five folds (round 12)",
        "cached solve + flux head, pathway + ProtT5, five folds (round 27)",
    ],
]


def folds_main() -> None:
    """The same averages under the reported protocol: fold models, one per fold."""
    labels = labels_of_split()
    families: dict[str, list[dict[str, float]]] = {}
    for name, stem in FOLD_TRANSFORMER_FAMILIES.items():
        families[name] = transformer_cells([stem.format(k=k) for k in range(5)])
        assert len(families[name]) == 5, (name, len(families[name]))
    with open(osp.join(RESULTS, "graph_baselines_go_folds.json")) as fh:
        report = json.load(fh)
    arm = report["arms"][FOLD_GRAPH_FAMILY[1]]
    assert "fold_logits" in arm, (
        "rerun graph_baselines.py --arms go_folds --save-logits"
    )
    families[FOLD_GRAPH_FAMILY[0]] = [
        dict(zip(arm["genes"], map(float, logits), strict=True))
        for logits in arm["fold_logits"]
    ]
    genes = sorted(set.intersection(*(set(c) for cs in families.values() for c in cs)))
    y = np.array([labels[g] for g in genes])
    scores = {name: family_score(cells, genes) for name, cells in families.items()}
    rng = np.random.default_rng(0)
    combos: list[Combination] = []
    for names in FOLD_COMBINATIONS:
        s = np.mean([scores[n] for n in names], axis=0)
        auroc = float(roc_auc_score(y, s))
        draws = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) == 2:
                draws.append(float(roc_auc_score(y[idx], s[idx])))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        combos.append(
            Combination(
                families=names,
                n_cells=sum(len(families[n]) for n in names),
                auroc=auroc,
                ci_low=float(lo),
                ci_high=float(hi),
            )
        )
        print(f"{auroc:.4f} [{lo:.3f}, {hi:.3f}]  {' + '.join(names)}")
    out = osp.join(RESULTS, "family_ensembles_folds.json")
    with open(out, "w") as fh:
        fh.write(
            Report(
                n_genes=len(genes),
                n_essential=int(y.sum()),
                families={n: len(c) for n, c in families.items()},
                combinations=combos,
            ).model_dump_json(indent=1)
        )
    print(out)


def main() -> None:
    import sys

    if "--folds" in sys.argv:
        folds_main()
        return
    labels = labels_of_split()
    families: dict[str, list[dict[str, float]]] = {}
    for name, globs in TRANSFORMER_FAMILIES.items():
        families[name] = transformer_cells(globs)
    for name, (fname, inputs) in GRAPH_FAMILIES.items():
        families[name] = graph_cells(fname, inputs)
    genes = sorted(set.intersection(*(set(c) for cs in families.values() for c in cs)))
    y = np.array([labels[g] for g in genes])
    scores = {name: family_score(cells, genes) for name, cells in families.items()}
    rng = np.random.default_rng(0)
    combos: list[Combination] = []
    for names in COMBINATIONS:
        s = np.mean([scores[n] for n in names], axis=0)
        auroc = float(roc_auc_score(y, s))
        draws = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) == 2:
                draws.append(float(roc_auc_score(y[idx], s[idx])))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        combos.append(
            Combination(
                families=names,
                n_cells=sum(len(families[n]) for n in names),
                auroc=auroc,
                ci_low=float(lo),
                ci_high=float(hi),
            )
        )
        print(f"{auroc:.4f} [{lo:.3f}, {hi:.3f}]  {' + '.join(names)}")
    report = Report(
        n_genes=len(genes),
        n_essential=int(y.sum()),
        families={n: len(c) for n, c in families.items()},
        combinations=combos,
    )
    out = osp.join(RESULTS, "family_ensembles.json")
    with open(out, "w") as fh:
        fh.write(report.model_dump_json(indent=1))
    print(out)


if __name__ == "__main__":
    main()
