# experiments/028-gene-essentiality/scripts/hard_genes.py
# [[experiments.028-gene-essentiality.scripts.hard_genes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/hard_genes.py

r"""Which held-out genes the best cells misrank, and whether they agree on them.

For every cell of the chosen arms the per-gene prediction dump gives a logit per
held-out gene. A gene's **misrank** in one cell is the fraction of the other class it
sits on the wrong side of: for an essential gene the fraction of nonessential genes
scoring above it, for a nonessential gene the fraction of essential genes scoring below
it; one minus the mean misrank over all genes of one class is that class's share of the
AUROC. Averaged over the cells of an arm the misrank says which genes an arm gets wrong
consistently rather than by seed; averaged over arms it says which genes every model
gets wrong. The script writes

- ``results/hard_genes.json``: per gene, per arm: mean and sd of the misrank over the
  arm's cells, the mean logit rank, and the gene's label, FBA growth ratios, pathway
  count and GO term count;
- the Spearman correlation of the per-gene misrank between arms (do the models fail on
  the same genes);
- a figure: **a**, per-gene misrank, genes sorted by their mean over arms, one marker
  per arm, essential and nonessential genes in two panels; **b**, the between-arm
  agreement as a heat map.

    python experiments/028-gene-essentiality/scripts/hard_genes.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "028-gene-essentiality")
PRED_DIR = os.getenv("TC_TEST_PRED_DIR", osp.join(DATA_ROOT, "test-predictions-028"))
FBA_CSV = osp.join(RESULTS, "fba_essentiality_per_gene.csv")

#: Arm label -> result-file globs whose dumps are pooled. The best cells of the
#: document: the reported recipe on the pathway input, its fold models, the flux head on
#: ProtT5 (six seeds) and on the pathway input, and the affine on ProtT5 as the floor.
ARMS: dict[str, list[str]] = {
    "pathway + ProtT5 + affine (round 22)": ["w38_protpathfba_h12l1p1_e200_s*.json"],
    "indicator + ProtT5 + affine, folds (round 12)": [
        "w23_chromprotcv[0-4]_h12l1p1_e200_s2700.json"
    ],
    "flux head, ProtT5 (rounds 21, 24)": ["w37_lpelgateflux_h12l1p1_e200_s*.json"],
    "flux head, pathway + ProtT5 (rounds 23, 25)": [
        "w39_pathlpelgateflux_h12l1p1_e200_s*.json"
    ],
    "cached solve + flux head, pathway + ProtT5 (round 27)": [
        "w43_pathlpoffgateflux_h12l1p1_e200_s*.json"
    ],
    "ProtT5 + affine (round 5)": ["w16_fba_h12l1p1_e200_s*.json"],
}

#: Arm label -> (graph-baselines result file, the row's ``inputs``): cheap models whose
#: rows carry per-gene logits (written with ``--save-logits``), one cell per seed.
GRAPH_ARMS: dict[str, tuple[str, str]] = {
    "GO convolution, 400 epochs": ("graph_baselines_go400.json", "GO (400 epochs)"),
    "GO convolution, 400 epochs, six seeds": (
        "graph_baselines_go400_s6.json",
        "GO (400 epochs)",
    ),
}


def load_graph_rows(
    fname: str, inputs: str
) -> list[tuple[dict[str, float], dict[str, int]]]:
    """Per-seed gene -> logit and gene -> label from a graph-baselines row, if saved."""
    path = osp.join(RESULTS, fname)
    if not osp.isfile(path):
        return []
    with open(path) as fh:
        report = json.load(fh)
    labels = split_labels()
    out: list[tuple[dict[str, float], dict[str, int]]] = []
    for row in report["rows"]:
        if row["inputs"] != inputs or not row.get("test_genes"):
            continue
        for logits in row["test_logits"]:
            lg = dict(zip(row["test_genes"], [float(v) for v in logits], strict=True))
            out.append((lg, {g: labels[g] for g in lg}))
    return out


def split_labels() -> dict[str, int]:
    """Gene -> label of the released split (the held-out genes' labels)."""
    import gzip

    with gzip.open(
        osp.join(RESULTS, "splits", "ess_fcl_val0.2_s0.json.gz"), "rt"
    ) as fh:
        split = json.load(fh)
    return {g: int(v) for g, v in split["labels"].items()}


class GeneRow(BaseModel):
    """One held-out gene's misrank per arm and what the gene is."""

    gene: str
    essential: int
    misrank: dict[str, float]
    misrank_sd: dict[str, float]
    n_cells: dict[str, int]
    mean_misrank: float
    fba_ratio_ypd: float | None
    n_pathways: int
    n_go_terms: int


class Report(BaseModel):
    """The whole analysis."""

    arms: dict[str, list[str]]
    n_genes: int
    n_essential: int
    genes: list[GeneRow]
    agreement: dict[str, dict[str, float]]
    hardest_essential: list[str]
    hardest_nonessential: list[str]
    figure: str


def load_dump(result_path: str) -> tuple[dict[str, float], dict[str, int]] | None:
    """Gene -> logit and gene -> label from the dump a result names, or None."""
    with open(result_path) as fh:
        result = json.load(fh)
    path = result.get("test", {}).get("dump_path")
    if not path:
        return None
    if not osp.isfile(path):
        path = osp.join(PRED_DIR, osp.basename(path))
        if not osp.isfile(path):
            return None
    with open(path) as fh:
        dump = json.load(fh)
    logits: dict[str, float] = {}
    labels: dict[str, int] = {}
    for p in dump["predictions"]:
        if not p.get("pinned", True):
            continue
        assert len(p["genes"]) == 1, p
        g = p["genes"][0]
        logits[g] = float(p["pred"])
        labels[g] = int(round(float(p["target"])))
    return logits, labels


def misrank(logits: dict[str, float], labels: dict[str, int]) -> dict[str, float]:
    """Per gene, the fraction of the other class on the wrong side of it."""
    genes = sorted(logits)
    ess = np.array([labels[g] for g in genes]) == 1
    x = np.array([logits[g] for g in genes])
    pos = x[ess]
    neg = x[~ess]
    out: dict[str, float] = {}
    for g, xi, e in zip(genes, x, ess):
        if e:
            out[g] = float(np.mean(neg > xi) + 0.5 * np.mean(neg == xi))
        else:
            out[g] = float(np.mean(pos < xi) + 0.5 * np.mean(pos == xi))
    return out


def gene_annotations() -> tuple[dict[str, int], dict[str, int]]:
    """Gene -> pathway count and gene -> propagated GO term count, from the built tensors."""
    import torch

    root = osp.join(DATA_ROOT, "data/scerevisiae")
    counts: list[dict[str, int]] = []
    for rel in (
        "sgd_gene_graph_hot/processed/pathways.pt",
        "go_term_hot/processed/go.pt",
    ):
        data, slices = torch.load(osp.join(root, rel), weights_only=False)
        emb = next(iter(data.embeddings.values()))
        ids = list(data.id)
        counts.append({g: int(emb[i].sum()) for i, g in enumerate(ids)})
    return counts[0], counts[1]


def fba_ratios() -> dict[str, float]:
    """Gene -> knockout growth ratio on YPD."""
    if not osp.isfile(FBA_CSV):
        return {}
    df = pd.read_csv(FBA_CSV)
    # Columns: gene, shipped, SM, SC, YPD (the knockout-to-wild-type growth ratio per
    # medium), essential, held_out, ours_195.
    return {str(g): float(r) for g, r in zip(df["gene"], df["YPD"])}


def figure(
    rows: list[GeneRow], arms: list[str], agreement: dict[str, dict[str, float]]
) -> str:
    apply_paper_style()
    fig = plt.figure(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(110)))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 2.2, 1.3], wspace=0.35)
    for k, (essential, title) in enumerate(((1, "essential"), (0, "nonessential"))):
        ax = fig.add_subplot(gs[0, k])
        sub = sorted(
            [r for r in rows if r.essential == essential], key=lambda r: -r.mean_misrank
        )
        x = np.arange(len(sub))
        for j, arm in enumerate(arms):
            y = [r.misrank[arm] for r in sub]
            ax.plot(x, y, "o", ms=2, color=PLOT_PALETTE[j], label=arm, alpha=0.8)
        ax.plot(x, [r.mean_misrank for r in sub], "-", color="black", lw=0.6)
        ax.set_xlabel(f"{title} held-out genes, sorted by mean misrank")
        ax.set_ylabel("misrank (fraction of the other class on the wrong side)")
        ax.set_ylim(0, 1)
        ax.set_title(chr(ord("a") + k), loc="left", fontweight="bold")
        if k == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    ax = fig.add_subplot(gs[0, 2])
    m = np.array([[agreement[a][b] for b in arms] for a in arms])
    # pcolormesh draws the cells as vector rectangles; imshow would rasterize them.
    im = ax.pcolormesh(
        np.arange(len(arms) + 1) - 0.5,
        np.arange(len(arms) + 1) - 0.5,
        m,
        vmin=0,
        vmax=1,
        cmap="Greys",
        edgecolors="white",
        linewidth=0.3,
    )
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.set_ylim(len(arms) - 0.5, -0.5)
    ax.set_aspect("equal")
    short = [a.split(" (")[0] for a in arms]
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(short, rotation=90)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(short)
    for i in range(len(arms)):
        for j in range(len(arms)):
            ax.text(
                j,
                i,
                f"{m[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if m[i, j] > 0.6 else "black",
            )
    ax.set_title("c", loc="left", fontweight="bold")
    cb = fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, label="Spearman of per-gene misrank"
    )
    # The colorbar's gradient is rasterized by default; the document wants vector.
    cb.solids.set_rasterized(False)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    base = osp.join(IMAGE_DIR, f"hard_genes_{timestamp()}")
    savefig_true_size_svg(fig, base + ".svg")
    fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    return base + ".svg"


def main() -> None:
    per_arm: dict[str, list[dict[str, float]]] = {}
    labels_all: dict[str, int] | None = None
    for arm, globs in ARMS.items():
        cells: list[dict[str, float]] = []
        for pat in globs:
            for f in sorted(glob.glob(osp.join(RESULTS, pat))):
                loaded = load_dump(f)
                if loaded is None:
                    continue
                logits, labels = loaded
                assert labels_all is None or labels == labels_all, f
                labels_all = labels
                cells.append(misrank(logits, labels))
        assert cells, f"no dumps for {arm}"
        per_arm[arm] = cells
    for arm, (fname, inputs) in GRAPH_ARMS.items():
        cells = []
        for logits, labels in load_graph_rows(fname, inputs):
            assert labels_all is None or labels == labels_all, arm
            labels_all = labels
            cells.append(misrank(logits, labels))
        if cells:
            per_arm[arm] = cells
    assert labels_all is not None
    genes = sorted(labels_all)
    n_path, n_go = gene_annotations()
    ratios = fba_ratios()
    rows: list[GeneRow] = []
    for g in genes:
        mr = {a: float(np.mean([c[g] for c in cells])) for a, cells in per_arm.items()}
        sd = {a: float(np.std([c[g] for c in cells])) for a, cells in per_arm.items()}
        rows.append(
            GeneRow(
                gene=g,
                essential=labels_all[g],
                misrank=mr,
                misrank_sd=sd,
                n_cells={a: len(c) for a, c in per_arm.items()},
                mean_misrank=float(np.mean(list(mr.values()))),
                fba_ratio_ypd=ratios.get(g),
                n_pathways=n_path.get(g, 0),
                n_go_terms=n_go.get(g, 0),
            )
        )
    arms = list(per_arm)
    agreement = {
        a: {
            b: float(
                spearmanr([r.misrank[a] for r in rows], [r.misrank[b] for r in rows])[0]
            )
            for b in arms
        }
        for a in arms
    }
    ess_sorted = sorted(
        [r for r in rows if r.essential == 1], key=lambda r: -r.mean_misrank
    )
    non_sorted = sorted(
        [r for r in rows if r.essential == 0], key=lambda r: -r.mean_misrank
    )
    fig_path = figure(rows, arms, agreement)
    report = Report(
        arms=ARMS,
        n_genes=len(rows),
        n_essential=sum(r.essential for r in rows),
        genes=rows,
        agreement=agreement,
        hardest_essential=[r.gene for r in ess_sorted[:15]],
        hardest_nonessential=[r.gene for r in non_sorted[:15]],
        figure=fig_path,
    )
    out = osp.join(RESULTS, "hard_genes.json")
    with open(out, "w") as fh:
        fh.write(report.model_dump_json(indent=1))
    print("arms:", {a: len(c) for a, c in per_arm.items()})
    print("hardest essential (mean misrank, pathways, GO terms, FBA ratio YPD):")
    for r in ess_sorted[:15]:
        print(
            f"  {r.gene} {r.mean_misrank:.2f} p{r.n_pathways} g{r.n_go_terms} fba {r.fba_ratio_ypd}"
        )
    print("hardest nonessential:")
    for r in non_sorted[:15]:
        print(
            f"  {r.gene} {r.mean_misrank:.2f} p{r.n_pathways} g{r.n_go_terms} fba {r.fba_ratio_ypd}"
        )
    print("agreement (Spearman of per-gene misrank):")
    for a in arms:
        print("  ", a, {b.split(" (")[0]: round(v, 2) for b, v in agreement[a].items()})
    print(fig_path)
    print(out)


if __name__ == "__main__":
    main()
