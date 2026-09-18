# experiments/028-gene-essentiality/scripts/genome_heldout_readout.py
# [[experiments.028-gene-essentiality.scripts.genome_heldout_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/genome_heldout_readout
"""Round 30: the genome-wide labels as a larger held-out set.

Every model here is trained on the released labeled genes outside the released held-out
set (629 train, 158 val, the split ``ess_genome_heldout_val0.2_s0``) and read once on the
4,674 genes the released file never labeled: 974 essential by SGD annotation, 3,700
nonessential by a fitness record. Label origin coincides with class on that set, and
none of the 4,674 is a yeast-GEM gene (the released file labeled the GEM's genes), so
FBA's knockout ratio is 1.0 on every one of them and the solve has no call here.

Reads the transformer cells (``w46_*``) through their per-gene dumps and the cheap
models (``graph_baselines_*_genome_heldout.json``) through their saved logits, and
writes ``results/genome_heldout_readout.json`` for ``notes_tex_tables.py``.
"""

import glob
import gzip
import json
import os
import os.path as osp
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "scripts"))

from wave12_readout import PRED_DIR, binary_auroc, bootstrap_ci  # noqa: E402

RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SPLIT = osp.join(RESULTS, "splits", "ess_genome_heldout_val0.2_s0.json.gz")
FBA_CSV = osp.join(RESULTS, "fba_essentiality_per_gene.csv")

#: Transformer arms: label -> result-file glob.
TRANSFORMER_ARMS: dict[str, str] = {
    "pathway multi-hot + ProtT5 + FBA affine (round 22 recipe)": (
        "w46_genomeprotpathfba_h12l1p1_e200_s*.json"
    ),
    "cached solve + flux head, pathway + ProtT5 (round 27 form)": (
        "w46_genomepathlpoffgateflux_h12l1p1_e200_s*.json"
    ),
}
#: Cheap models: (result file, model, graph, row inputs) -> label.
BASE = "graph_baselines_genome_heldout.json"
EXTRA = "graph_baselines_extra_genome_heldout.json"
GO400 = "graph_baselines_go400_genome_heldout.json"
UNION = "union of nine"
GRAPH_ARMS: dict[tuple[str, str, str, str], str] = {
    (BASE, "MLP", "none", "ProtT5"): "MLP on ProtT5 alone",
    (EXTRA, "MLP", "none", "ProtT5+FBA ratios"): "MLP on ProtT5 + FBA ratios",
    (EXTRA, "MLP", "none", "ProtT5+chrom-pathway+FBA ratios"): "MLP on the three inputs",
    (BASE, "GCN", "physical", "ones"): "graph convolution, physical graph, all-ones input",
    (BASE, "GCN", "physical", "degree+clustering"): (
        "graph convolution, physical graph, degree and clustering"
    ),
    (BASE, "GCN", "physical", "ProtT5"): "graph convolution, physical graph, ProtT5",
    (BASE, "GCN", UNION, "ones"): "graph convolution, union of nine, all-ones input",
    (BASE, "GCN", UNION, "degree+clustering"): (
        "graph convolution, union of nine, degree and clustering"
    ),
    (BASE, "GCN", UNION, "ProtT5"): "graph convolution, union of nine, ProtT5",
    (BASE, "GATv2", "physical", "degree+clustering"): (
        "graph attention, physical graph, degree and clustering"
    ),
    (BASE, "GATv2", UNION, "degree+clustering"): (
        "graph attention, union of nine, degree and clustering"
    ),
    (EXTRA, "GCN", UNION, "ProtT5 (400 epochs)"): (
        "graph convolution on ProtT5 alone, 400 epochs"
    ),
    (EXTRA, "GCN", UNION, "ProtT5+FBA ratios"): "graph convolution on ProtT5 + FBA ratios",
    (EXTRA, "GCN", UNION, "ProtT5+chrom-pathway+FBA ratios (400 epochs)"): (
        "graph convolution on the three inputs, 400 epochs"
    ),
    (GO400, "GCN", UNION, "GO (400 epochs)"): "graph convolution on GO alone, 400 epochs",
    (GO400, "GCN", UNION, "GO+FBA ratios (400 epochs)"): (
        "graph convolution on GO + FBA ratios, 400 epochs"
    ),
    (GO400, "GCN", UNION, "ProtT5+GO+FBA ratios (400 epochs)"): (
        "graph convolution on ProtT5 + GO + FBA ratios, 400 epochs"
    ),
}


class ArmRead(BaseModel):
    """One arm on the genome held-out genes."""

    label: str
    kind: str
    n_cells: int
    cell_aurocs: list[float]
    mean: float
    sd: float
    pooled_auroc: float | None
    pooled_ci: tuple[float, float] | None
    n_genes: int
    select_epochs: list[int]
    wall_minutes: float | None


def split_labels() -> dict[str, int]:
    with gzip.open(SPLIT, "rt") as fh:
        split = json.load(fh)
    return {g: int(split["labels"][g]) for g in split["genes"]["test"]}


def pooled(cells: list[dict[str, float]], labels: dict[str, int]) -> tuple[float, tuple[float, float], int]:
    genes = sorted(set.intersection(*(set(c) for c in cells)))
    score = np.array([np.mean([c[g] for c in cells]) for g in genes])
    y = np.array([labels[g] for g in genes])
    return binary_auroc(score, y), bootstrap_ci(score, y), len(genes)


def dump_predictions(path: str, labels: dict[str, int]) -> tuple[dict[str, float], int]:
    """Every record of a transformer cell's test dump as ``gene -> logit``, and its seed.

    The trainer pins the released 195 genes, and none of them is in this split's test
    set, so the dump's ``pinned`` flag is False on every record and the read is over
    all of them; the gene set is asserted against the split's test genes.
    """
    with open(path) as fh:
        dump = json.load(fh)
    recs = dump["predictions"]
    assert not any(r["pinned"] for r in recs), path
    cell = {r["genes"][0]: float(r["pred"]) for r in recs}
    assert set(cell) == set(labels), (path, len(cell), len(labels))
    for r in recs:
        assert int(r["target"]) == labels[r["genes"][0]], (path, r["genes"][0])
    return cell, int(dump["cell"]["seed"])


def transformer_reads(labels: dict[str, int]) -> list[ArmRead]:
    """The transformer cells through their dumps; the trainer's summary step died on
    the empty pinned block (KeyError on ``accuracy``), after the dump was written.
    """
    reads: list[ArmRead] = []
    for label, pattern in TRANSFORMER_ARMS.items():
        paths = sorted(glob.glob(osp.join(PRED_DIR, pattern)))
        if not paths:
            continue
        aurocs, cells, seeds = [], [], []
        for path in paths:
            cell, seed = dump_predictions(path, labels)
            genes = sorted(cell)
            aurocs.append(
                binary_auroc(
                    np.array([cell[g] for g in genes]),
                    np.array([labels[g] for g in genes]),
                )
            )
            cells.append(cell)
            seeds.append(seed)
        auroc, ci, n = pooled(cells, labels)
        reads.append(
            ArmRead(
                label=label,
                kind="transformer",
                n_cells=len(paths),
                cell_aurocs=aurocs,
                mean=float(np.mean(aurocs)),
                sd=float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0,
                pooled_auroc=auroc,
                pooled_ci=ci,
                n_genes=n,
                select_epochs=[],
                wall_minutes=None,
            )
        )
        print(f"  {label}: seeds {seeds}")
    return reads


def graph_reads(labels: dict[str, int]) -> list[ArmRead]:
    reads: list[ArmRead] = []
    for (fname, model, graph, inputs), label in GRAPH_ARMS.items():
        src = osp.join(RESULTS, fname)
        if not osp.isfile(src):
            continue
        with open(src) as fh:
            rows = json.load(fh)["rows"]
        row = next(
            (
                r
                for r in rows
                if r["model"] == model and r["graph"] == graph and r["inputs"] == inputs
            ),
            None,
        )
        if row is None:
            continue
        cells = [
            dict(zip(row["test_genes"], map(float, logits), strict=True))
            for logits in row["test_logits"]
        ]
        auroc, ci, n = pooled(cells, labels)
        reads.append(
            ArmRead(
                label=label,
                kind=row["model"],
                n_cells=len(cells),
                cell_aurocs=[float(a) for a in row["test_auroc"]],
                mean=float(row["mean"]),
                sd=float(row["sd"]),
                pooled_auroc=auroc,
                pooled_ci=ci,
                n_genes=n,
                select_epochs=[int(e) for e in row["select_epoch"]],
                wall_minutes=float(row["seconds"]) / 60.0,
            )
        )
    return reads


def fba_read(labels: dict[str, int]) -> ArmRead:
    """FBA's knockout ratio on the test genes: 1.0 wherever the gene has no reaction."""
    import csv

    ratio = {g: 1.0 for g in labels}
    with open(FBA_CSV) as fh:
        for r in csv.DictReader(fh):
            if r["gene"] in ratio:
                ratio[r["gene"]] = float(r["YPD"])
    genes = sorted(labels)
    score = -np.array([ratio[g] for g in genes])
    y = np.array([labels[g] for g in genes])
    n_gem = sum(1 for g in genes if ratio[g] != 1.0)
    return ArmRead(
        label=f"FBA knockout ratio on YPD ({n_gem} of {len(genes)} genes carry a reaction)",
        kind="solver",
        n_cells=1,
        cell_aurocs=[binary_auroc(score, y)],
        mean=binary_auroc(score, y),
        sd=0.0,
        pooled_auroc=binary_auroc(score, y),
        pooled_ci=bootstrap_ci(score, y),
        n_genes=len(genes),
        select_epochs=[],
        wall_minutes=None,
    )


def main() -> None:
    labels = split_labels()
    reads = transformer_reads(labels) + graph_reads(labels) + [fba_read(labels)]
    report: dict[str, Any] = {
        "split": osp.basename(SPLIT),
        "n_test_genes": len(labels),
        "n_test_essential": int(sum(labels.values())),
        "arms": [r.model_dump() for r in reads],
    }
    out = osp.join(RESULTS, "genome_heldout_readout.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    for r in reads:
        ci = "" if r.pooled_ci is None else f" [{r.pooled_ci[0]:.3f}, {r.pooled_ci[1]:.3f}]"
        print(
            f"{r.mean:.4f} sd {r.sd:.3f} n={r.n_cells} pooled "
            f"{'--' if r.pooled_auroc is None else f'{r.pooled_auroc:.4f}'}{ci}  {r.label}"
        )
    print(out)


if __name__ == "__main__":
    main()
