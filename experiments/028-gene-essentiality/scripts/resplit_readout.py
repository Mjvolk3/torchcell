# experiments/028-gene-essentiality/scripts/resplit_readout.py
# [[experiments.028-gene-essentiality.scripts.resplit_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/resplit_readout

"""Round 75: the three finalists on ten repeated outer splits, read as paired differences.

On the released 195 held-out genes (31 essential) the transformer with the layer-free
solve, graph convolution on GO and their equal-weight average have overlapping intervals,
and the average's 0.954 is the maximum of eleven post hoc choices. This read was fixed
before any cell of the round finished:

- ten test blocks, each a class-stratified fifth of the released labeled genes
  (``build_essentiality_splits.py --resplit R``), validation a stratified fifth of the rest;
- per split, three seeds of the round-33 layer-free cell (``w93_resplit{R}pathlogfree``)
  and three seeds of graph convolution on GO at 400 epochs
  (``graph_baselines.py --arms go400 --split ...``);
- a family's score is the mean over its seeds of the per-cell standardized logits, as
  ``family_ensembles.family_score``; the average is the mean of the two family scores
  with equal weight, nothing fitted;
- the statistics: the AUROC of each of the three per split, and across splits the mean,
  standard deviation and sign count of three paired differences: average minus layer-free,
  average minus GO convolution, layer-free minus GO convolution, plus average minus the
  better single model of that split.

A second read, fixed when its packs were submitted and before any of them finished
(``w94``): the layer-free cell's two controls on the same ten splits, three seeds each,
the FBA ratio affine alone (``protpathfba``) and no metabolism (``pathpool``). Statistic:
the per-split seed-mean held-out AUROC of each arm, and across splits the mean, standard
deviation and sign count of layer-free minus the ratio affine and of layer-free minus no
metabolism.

The test blocks overlap across splits (each gene is in about two of ten), so the ten
differences are not independent and no p-value is reported; the sign count and the spread
are the read.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/resplit_readout.py
"""

from __future__ import annotations

import argparse
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
N_SPLITS = 10
GO_INPUTS = "GO (400 epochs)"


class SplitRow(BaseModel):
    """One outer split: the three AUROCs on its test block."""

    resplit: int
    n_test: int
    n_test_essential: int
    n_cells_layer_free: int
    n_cells_go: int
    layer_free: float
    go_convolution: float
    average: float


class Difference(BaseModel):
    """One paired difference across the splits."""

    name: str
    per_split: list[float]
    mean: float
    sd: float
    n_positive: int
    n_splits: int


class Report(BaseModel):
    """The whole read."""

    splits: list[SplitRow]
    differences: list[Difference]
    #: Job 2684: the solve columns as node features of the graph family, against GO alone.
    graph_module: list[Difference]
    mean_auroc: dict[str, float]
    seed_mean_auroc_per_split: dict[str, list[float]]
    note: str


def standardized_mean(cells: list[np.ndarray]) -> np.ndarray:
    return np.mean([(x - x.mean()) / x.std() for x in cells], axis=0)


#: Set by ``--deep``: read both families at six seeds per split instead of three, to
#: check whether the round-75 margins are seed-limited. The three-seed read is the one
#: the document reports and is never overwritten; this writes its own file.
DEEP = False
#: Read the single-experiment form (w104, YPD ratio + vector + counts) as the transformer
#: family, at the same six seeds as --deep, so only the module's form changes.
PORTABLE = False
#: With --portable: read the graph family WITH the module as a zero-initialized affine on
#: its logit (job 2691, three seeds), so both families carry the single-experiment module.
GRAPH_AFFINE = False
AFFINE_INPUTS = "GO | affine: YPD ratio+vector+counts (400 epochs)"


def transformer_cells(resplit: int, genes: list[str]) -> list[np.ndarray]:
    cells = []
    waves = ("w93", "w97") if DEEP else ("w93",)
    patterns = [f"{w}_resplit{resplit}pathlogfree_h12l1p1_e200_s*.json" for w in waves]
    if PORTABLE:
        patterns = [f"w104_resplit{resplit}ypdgpr_h12l1p1_e200_s270[0-5].json"]
    for path in sorted(
        p for pat in patterns for p in glob.glob(osp.join(RESULTS, pat))
    ):
        with open(path) as fh:
            dump_path = json.load(fh)["test"]["dump_path"]
        if not osp.isfile(dump_path):
            dump_path = osp.join(PRED_DIR, osp.basename(dump_path))
        with open(dump_path) as fh:
            preds = [p for p in json.load(fh)["predictions"] if p.get("pinned", True)]
        by_gene = {p["genes"][0]: float(p["pred"]) for p in preds}
        assert set(by_gene) == set(genes), (resplit, len(by_gene), len(genes))
        cells.append(np.array([by_gene[g] for g in genes]))
    return cells


def go_cells(resplit: int, genes: list[str]) -> list[np.ndarray]:
    stem = "graph_baselines_go400_s6" if DEEP else "graph_baselines_go400"
    inputs = GO_INPUTS
    if GRAPH_AFFINE:
        stem, inputs = "graph_baselines_go_affine", AFFINE_INPUTS
    path = osp.join(RESULTS, f"{stem}_fcl_resplit{resplit}.json")
    with open(path) as fh:
        rows = [r for r in json.load(fh)["rows"] if r["inputs"] == inputs]
    assert len(rows) == 1
    index = {g: i for i, g in enumerate(rows[0]["test_genes"])}
    assert set(index) == set(genes), (resplit, len(index), len(genes))
    order = [index[g] for g in genes]
    return [np.array(logits, dtype=float)[order] for logits in rows[0]["test_logits"]]


def seed_mean_auroc(*patterns: str, expect: int = 3) -> float:
    values = []
    for pattern in patterns:
        for path in sorted(glob.glob(osp.join(RESULTS, pattern))):
            with open(path) as fh:
                values.append(float(json.load(fh)["test"]["pinned"]["auroc"]))
    assert len(values) == expect, (patterns, len(values))
    return float(np.mean(values))


def main() -> None:
    global DEEP, PORTABLE, GRAPH_AFFINE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deep",
        action="store_true",
        help=(
            "read both families at six seeds per split and write "
            "results/resplit_readout_s6.json, leaving the three-seed read in place"
        ),
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help=(
            "as --deep, with the single-experiment module form (w104) as the transformer "
            "family; writes results/resplit_readout_portable.json"
        ),
    )
    parser.add_argument(
        "--graph-affine",
        action="store_true",
        help=(
            "with --portable, the graph family reads the module as an affine on its logit "
            "(three seeds); writes results/resplit_readout_portable_affine.json"
        ),
    )
    args = parser.parse_args()
    PORTABLE = args.portable
    GRAPH_AFFINE = args.graph_affine
    assert not GRAPH_AFFINE or PORTABLE
    DEEP = args.deep or PORTABLE
    expect_cells = 6 if DEEP else 3
    rows: list[SplitRow] = []
    for r in range(N_SPLITS):
        with gzip.open(
            osp.join(RESULTS, "splits", f"ess_fcl_resplit{r}_val0.2_s0.json.gz"), "rt"
        ) as fh:
            split = json.load(fh)
        genes = sorted(split["genes"]["test"])
        y = np.array([int(split["labels"][g]) for g in genes])
        t_cells, g_cells = transformer_cells(r, genes), go_cells(r, genes)
        assert len(t_cells) == expect_cells and len(g_cells) == (
            3 if GRAPH_AFFINE else expect_cells
        ), (
            r,
            len(t_cells),
            len(g_cells),
        )
        t, g = standardized_mean(t_cells), standardized_mean(g_cells)
        rows.append(
            SplitRow(
                resplit=r,
                n_test=len(genes),
                n_test_essential=int(y.sum()),
                n_cells_layer_free=len(t_cells),
                n_cells_go=len(g_cells),
                layer_free=float(roc_auc_score(y, t)),
                go_convolution=float(roc_auc_score(y, g)),
                average=float(roc_auc_score(y, (t + g) / 2.0)),
            )
        )
        print(
            f"split {r}: layer-free {rows[-1].layer_free:.4f}  GO convolution "
            f"{rows[-1].go_convolution:.4f}  average {rows[-1].average:.4f}"
        )

    def difference(name: str, values: list[float]) -> Difference:
        v = np.array(values)
        return Difference(
            name=name,
            per_split=values,
            mean=float(v.mean()),
            sd=float(v.std(ddof=1)),
            n_positive=int((v > 0).sum()),
            n_splits=len(values),
        )

    diffs = [
        difference(
            "average minus layer-free", [s.average - s.layer_free for s in rows]
        ),
        difference(
            "average minus GO convolution", [s.average - s.go_convolution for s in rows]
        ),
        difference(
            "layer-free minus GO convolution",
            [s.layer_free - s.go_convolution for s in rows],
        ),
        difference(
            "average minus the better single model",
            [s.average - max(s.layer_free, s.go_convolution) for s in rows],
        ),
    ]
    arms = {
        "layer_free": "w93_resplit{r}pathlogfree_h12l1p1_e200_s*.json",
        "fba_ratio_affine": "w94_resplit{r}protpathfba_h12l1p1_e200_s*.json",
        "no_metabolism": "w94_resplit{r}pathpool_h12l1p1_e200_s*.json",
    }
    cell_means = {
        k: [seed_mean_auroc(pat.format(r=r)) for r in range(N_SPLITS)]
        for k, pat in arms.items()
    }
    # Round 78 adds seeds 2703 to 2705 of the vector and of the ratio control on every
    # split, under the w97 stems. At three seeds the vector's gain over the ratios alone
    # was +0.011 (sd 0.023, seven of ten), which does not separate from zero; the six-seed
    # means are the read fixed when those packs were submitted.
    six = {
        "layer_free": (
            "w93_resplit{r}pathlogfree_h12l1p1_e200_s*.json",
            "w97_resplit{r}pathlogfree_h12l1p1_e200_s*.json",
        ),
        "fba_ratio_affine": (
            "w94_resplit{r}protpathfba_h12l1p1_e200_s*.json",
            "w97_resplit{r}protpathfba_h12l1p1_e200_s*.json",
        ),
    }
    six_means: dict[str, list[float]] = {}
    for key, pats in six.items():
        try:
            six_means[key] = [
                seed_mean_auroc(*(p.format(r=r) for p in pats), expect=6)
                for r in range(N_SPLITS)
            ]
        except AssertionError as exc:
            print(f"six-seed means not ready for {key}: {exc}")
            six_means = {}
            break
    if six_means:
        cell_means.update({f"{k}_six_seeds": v for k, v in six_means.items()})
    for k, v in cell_means.items():
        print(f"seed-mean AUROC per split, {k}: " + " ".join(f"{x:.3f}" for x in v))
    diffs += [
        difference(
            "layer-free minus FBA ratio affine (seed means)",
            [
                a - b
                for a, b in zip(
                    cell_means["layer_free"],
                    cell_means["fba_ratio_affine"],
                    strict=True,
                )
            ],
        ),
        difference(
            "layer-free minus no metabolism (seed means)",
            [
                a - b
                for a, b in zip(
                    cell_means["layer_free"], cell_means["no_metabolism"], strict=True
                )
            ],
        ),
    ]
    if six_means:
        diffs.append(
            difference(
                "layer-free minus FBA ratio affine (six-seed means)",
                [
                    a - b
                    for a, b in zip(
                        six_means["layer_free"],
                        six_means["fba_ratio_affine"],
                        strict=True,
                    )
                ],
            )
        )
    for d in diffs:
        print(
            f"{d.name:42s} mean {d.mean:+.4f} sd {d.sd:.4f} positive {d.n_positive}/{d.n_splits}"
        )
    # Job 2684: the module's columns as node features of the graph family, seeds 2700 to
    # 2702, against graph convolution on GO alone at the same three seeds.
    def graph_mean(stem: str, model: str, inputs: str, r: int) -> float:
        with open(osp.join(RESULTS, f"{stem}_fcl_resplit{r}.json")) as fh:
            found = [
                x
                for x in json.load(fh)["rows"]
                if x["model"] == model and x["inputs"] == inputs
            ]
        assert len(found) == 1, (stem, model, inputs, r)
        return float(found[0]["mean"])

    go_alone = [
        graph_mean("graph_baselines_go400", "GCN", GO_INPUTS, r)
        for r in range(N_SPLITS)
    ]
    graph_module = [
        difference(
            f"{model}, {inputs}, minus graph convolution on GO",
            [
                graph_mean("graph_baselines_go_solve", model, inputs, r) - go_alone[r]
                for r in range(N_SPLITS)
            ],
        )
        for model, inputs in (
            ("GCN", "GO+solve log1p (400 epochs)"),
            ("GCN", "GO+FBA ratios+solve log1p (400 epochs)"),
            ("MLP", "GO+solve log1p"),
            ("GCN", "solve log1p (400 epochs)"),
        )
    ]
    # Job 2691 at three seeds and job 2693 at six, each against the GO-alone control rerun
    # in the same job.
    for stem, tag in (
        ("graph_baselines_go_affine", ""),
        ("graph_baselines_go_affine_s6", ", six seeds"),
    ):
        affine_control = [
            graph_mean(stem, "GCN", GO_INPUTS, r) for r in range(N_SPLITS)
        ]
        graph_module += [
            difference(
                f"GCN, {inputs}, minus graph convolution on GO{tag}",
                [
                    graph_mean(stem, "GCN", inputs, r) - affine_control[r]
                    for r in range(N_SPLITS)
                ],
            )
            for inputs in (
                "GO | affine: YPD ratio (400 epochs)",
                AFFINE_INPUTS,
                "GO | affine: four ratios+vector+counts (400 epochs)",
            )
        ]
    for d in graph_module:
        print(f"{d.name}: {d.mean:+.4f} sd {d.sd:.4f} positive {d.n_positive}/10")

    report = Report(
        splits=rows,
        differences=diffs,
        graph_module=graph_module,
        mean_auroc={
            k: float(np.mean([getattr(s, k) for s in rows]))
            for k in ("layer_free", "go_convolution", "average")
        },
        seed_mean_auroc_per_split=cell_means,
        note=(
            "Test blocks overlap across splits, so the differences are not independent; "
            "the sign count and spread are the read, and no p-value is reported."
        ),
    )
    out = osp.join(
        RESULTS,
        "resplit_readout_portable_affine.json"
        if GRAPH_AFFINE
        else "resplit_readout_portable.json"
        if PORTABLE
        else "resplit_readout_s6.json"
        if DEEP
        else "resplit_readout.json",
    )
    with open(out, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    print(out)


if __name__ == "__main__":
    main()
