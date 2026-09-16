# experiments/028-gene-essentiality/scripts/wave12_readout.py
# [[experiments.028-gene-essentiality.scripts.wave12_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/wave12_readout.py

r"""Read one round of 028 cells: per-arm AUROC with gene bootstrap, curve stability, figure.

One row per finished result file, one summary per arm (mean, sd, n over seeds), and one
multipanel figure:

- **a**, validation AUROC per epoch, one thin line per seed and the seed mean per arm,
  which is where the wave-9 spikiness is read against the validation size;
- **b**, the median absolute epoch-to-epoch step of that curve per arm, against the
  number of labeled validation genes the arm read it on;
- **c**, pinned-test AUROC per cell (points) with the per-arm seed mean and the gene
  bootstrap interval of the pooled per-gene predictions, beside the comparators computed
  on the same 195 genes: FBA on each medium (``fba_essentiality_baseline.py``) and Flux
  Cone Learning (``fcl_comparison.py``);
- **d**, per-module gradient norm per epoch (seed mean) for the arms that logged it.

Every number comes from ``results/<stem>.json`` and the per-gene dumps those results
name; a cell whose dump is missing is reported without an interval, never skipped.

    python experiments/028-gene-essentiality/scripts/wave12_readout.py --round w12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from pydantic import BaseModel  # noqa: E402

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
FBA_JSON = osp.join(RESULTS, "fba_essentiality_by_medium.json")
FCL_JSON = osp.join(RESULTS, "fcl_comparison.json")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

#: Round -> (result globs, arm order for the figure). The arm of a result is the part of
#: its stem between the wave prefix and the architecture suffix; a round may pull a cell
#: from another wave in as its paired comparator.
ROUNDS: dict[str, tuple[list[str], list[str]]] = {
    "w12": (
        ["w12_*_e80_s*.json"],
        [
            "base",
            "fcl",
            "fcl_wd0",
            "fcl_wd1e-2",
            "fcl_wd1e-1",
            "fcl_b32",
            "gen",
            "fcl_lr3e-4",
            "fcl_lr3e-3",
            "fcl_posw",
            "fcl_noreg",
        ],
    ),
    # Wave 14: the two round-1 levers that moved three of three seeds (batch 32 and the
    # class weight) combined, the epoch budget raised to 200 since every fcl cell selected
    # its last smoothable epoch, and five-fold cross-validation on the 787 labeled
    # non-test genes whose fold models are averaged on the held-out genes, which is FCL's
    # own test protocol (`cv_ensemble_readout.py` scores the ensemble).
    "w14": (
        ["w14_*_s*.json"],
        ["b32posw", "b32posw200", "b32e200", "cv0", "cv1", "cv2", "cv3", "cv4"],
    ),
    # Wave 13: the prediction routed through the metabolic module, on the fcl split at
    # the base hyperparameters (batch 128, wd 1e-4, 80 epochs), paired by seed against the
    # w12 fcl pooled cell, which is the first arm of the round.
    "w13": (
        ["w12_fcl_h12l1p1_e80_s*.json", "w13_*_e80_s*.json"],
        [
            "fcl",
            "fluxbio",
            "fluxbio_s0",
            "fluxshuf_s0",
            "fluxctx",
            "fluxctx_s0",
            "fluxgate",
            "fluxgate_s0",
            "fluxoff",
            "fluxoffctx",
        ],
    ),
    # Wave 15 (round 4, jobs 2237-2245): the round-2 module readouts on top of the round-3
    # levers (batch 32, class weight, 200 epochs), paired by seed against the round-3
    # b32posw200 pooled cell, which is the first arm of the round.
    "w15": (
        ["w14_b32posw200_h12l1p1_e200_s*.json", "w15_*_e200_s*.json"],
        ["b32posw200", "gate_s0", "bio_s0", "shuf_s0"],
    ),
    # Wave 16 (round 5, packs): the pooled arm with FBA's four growth ratios added to the
    # context head (`--ess-readout fba`) on the round-3 levers, paired by seed against the
    # round-3 pooled cell, plus the same readout on the five cross-validation folds.
    "w16": (
        ["w14_b32posw200_h12l1p1_e200_s*.json", "w16_*_e200_s*.json"],
        ["b32posw200", "fba", "fbacv0", "fbacv1", "fbacv2", "fbacv3", "fbacv4"],
    ),
    # The 027 wave-11 width-12 and width-24 pooled cells, for a dry run of this script and
    # as the pre-move reference (same protocol as `base`).
    "wave11": (["wave11_ess_pool_h*_e80_s*.json"], ["ess_pool"]),
}
ARM_LABELS = {
    "base": "wave-11 protocol (75-gene val)",
    "fcl": "released labels, 158-gene val",
    "fcl_wd0": "weight decay 0",
    "fcl_wd1e-2": "weight decay 1e-2",
    "fcl_wd1e-1": "weight decay 1e-1",
    "fcl_b32": "batch 32",
    "gen": "genome labels (5,795 genes)",
    "fcl_lr3e-4": "lr 3e-4",
    "fcl_lr3e-3": "lr 3e-3",
    "fcl_posw": "class weight 772/126",
    "fcl_noreg": "graph regularization off",
    "b32posw": "batch 32 + class weight, 80 epochs",
    "b32posw200": "batch 32 + class weight, 200 epochs",
    "b32e200": "batch 32, 200 epochs",
    "cv0": "5-fold, fold 0",
    "cv1": "5-fold, fold 1",
    "cv2": "5-fold, fold 2",
    "cv3": "5-fold, fold 3",
    "cv4": "5-fold, fold 4",
    "fluxbio": "growth column only, constraints 0.1",
    "fluxbio_s0": "growth column only, constraints 0",
    "fluxctx": "context + growth column, constraints 0.1",
    "fluxctx_s0": "context + growth column, constraints 0",
    "fluxoff": "growth column only, stoichiometry only",
    "fluxoffctx": "context + growth column, stoichiometry only",
    "fluxgate": "gated growth column, constraints 0.1",
    "fluxgate_s0": "gated growth column, constraints 0",
    "fluxshuf_s0": "growth column only, kcat shuffled, constraints 0",
    "gate_s0": "gated growth column, constraints 0 (round-3 levers)",
    "bio_s0": "growth column only, constraints 0 (round-3 levers)",
    "shuf_s0": "growth column only, kcat shuffled (round-3 levers)",
    "fba": "context head + FBA growth ratios (round-3 levers)",
    "fbacv0": "context head + FBA ratios, fold 0",
    "fbacv1": "context head + FBA ratios, fold 1",
    "fbacv2": "context head + FBA ratios, fold 2",
    "fbacv3": "context head + FBA ratios, fold 3",
    "fbacv4": "context head + FBA ratios, fold 4",
}


class CellRow(BaseModel):
    """One finished cell."""

    stem: str
    arm: str
    seed: int
    hidden: int
    weight_decay: float
    batch_size: int
    split_file: str | None
    n_parameters: int
    n_val: int
    select_epoch: int
    select_val_auroc: float
    test_auroc_pinned: float
    test_auroc_ci_low: float | None
    test_auroc_ci_high: float | None
    test_accuracy_pinned: float
    test_recall_essential_pinned: float
    median_abs_step: float
    val_curve_range: float
    wall_time_min: float
    wandb_url: str | None


class ArmSummary(BaseModel):
    """Seed mean, sd and n per arm, plus the pooled-prediction bootstrap."""

    arm: str
    label: str
    n_seeds: int
    seeds: list[int]
    test_auroc_mean: float
    test_auroc_sd: float
    test_auroc_values: list[float]
    pooled_auroc: float | None
    pooled_ci_low: float | None
    pooled_ci_high: float | None
    median_abs_step_mean: float
    n_val: int
    select_epochs: list[int]


def binary_auroc(score: np.ndarray, label: np.ndarray) -> float:
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
        v = binary_auroc(score[idx], label[idx])
        if not np.isnan(v):
            draws.append(v)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def arm_of(stem: str) -> str:
    body = stem.split("_", 1)[1]
    return body.split("_h")[0]


def pinned_predictions(
    result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    path = result.get("test", {}).get("dump_path")
    if not path:
        return None
    if not osp.isfile(path):
        alt = osp.join(PRED_DIR, osp.basename(path))
        if not osp.isfile(alt):
            return None
        path = alt
    with open(path) as fh:
        dump = json.load(fh)
    # THE DUMP MUST BE THIS CELL'S. Until 2026-09-16 every width wrote the same dump name,
    # so a result can point at a file another cell overwrote (the wave-11 width-24 dumps
    # hold the width-12 cell). A dump whose cell disagrees is treated as missing.
    identity = (
        "arm",
        "head",
        "seed",
        "hidden",
        "layers",
        "preprocessor_layers",
        "epochs",
    )
    if any(dump["cell"].get(k) != result["cell"].get(k) for k in identity):
        return None
    recs = [r for r in dump["predictions"] if r["pinned"]]
    genes = [r["genes"][0] for r in recs]
    return (
        np.array([r["pred"] for r in recs], dtype=float),
        np.array([r["target"] for r in recs], dtype=int),
        genes,
    )


def load_round(
    round_key: str,
) -> tuple[list[CellRow], dict[str, list[dict[str, Any]]], dict[str, list[np.ndarray]]]:
    patterns, _ = ROUNDS[round_key]
    rows: list[CellRow] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    preds: dict[str, list[np.ndarray]] = {}
    paths = [p for pat in patterns for p in sorted(glob.glob(osp.join(RESULTS, pat)))]
    for path in paths:
        with open(path) as fh:
            result = json.load(fh)
        stem = osp.splitext(osp.basename(path))[0]
        cell = result["cell"]
        hist = result["history"]
        curve = np.array([h["val_pearson"] for h in hist], dtype=float)
        steps = np.abs(np.diff(curve))
        arm = arm_of(stem)
        pinned = result["test"]["pinned"]
        ci: tuple[float | None, float | None] = (None, None)
        pp = pinned_predictions(result)
        if pp is not None:
            ci = bootstrap_ci(pp[0], pp[1])
            preds.setdefault(arm, []).append(np.stack([pp[0], pp[1]]))
        rows.append(
            CellRow(
                stem=stem,
                arm=arm,
                seed=int(cell["seed"]),
                hidden=int(cell["hidden"]),
                weight_decay=float(cell["weight_decay"]),
                batch_size=int(cell["batch_size"]),
                split_file=cell.get("split_file"),
                n_parameters=int(result["n_parameters"]["total"]),
                n_val=int(hist[0]["n_val"]),
                select_epoch=int(result["select_epoch"]),
                select_val_auroc=float(result["select_val_pearson"]),
                test_auroc_pinned=float(pinned["auroc"]),
                test_auroc_ci_low=ci[0],
                test_auroc_ci_high=ci[1],
                test_accuracy_pinned=float(pinned["accuracy"]),
                test_recall_essential_pinned=float(pinned["recall_essential"]),
                median_abs_step=float(np.median(steps)),
                val_curve_range=float(curve.max() - curve.min()),
                wall_time_min=float(result["wall_time_s"]) / 60.0,
                wandb_url=result.get("wandb_url"),
            )
        )
        histories.setdefault(arm, []).append(
            {"seed": int(cell["seed"]), "history": hist}
        )
    return rows, histories, preds


def summarize(
    rows: list[CellRow], preds: dict[str, list[np.ndarray]], order: list[str]
) -> list[ArmSummary]:
    out: list[ArmSummary] = []
    arms = [a for a in order if any(r.arm == a for r in rows)]
    arms += sorted({r.arm for r in rows} - set(arms))
    for arm in arms:
        cells = sorted([r for r in rows if r.arm == arm], key=lambda r: r.seed)
        values = [r.test_auroc_pinned for r in cells]
        pooled: tuple[float | None, float | None, float | None] = (None, None, None)
        if arm in preds:
            # Per-gene predictions averaged over seeds (a logit mean), then one AUROC with
            # its gene bootstrap: the interval on the arm's prediction rather than on one
            # seed's.
            stack = np.stack(preds[arm])  # [n_seeds, 2, n_genes]
            score = stack[:, 0, :].mean(axis=0)
            label = stack[0, 1, :].astype(int)
            lo, hi = bootstrap_ci(score, label)
            pooled = (binary_auroc(score, label), lo, hi)
        out.append(
            ArmSummary(
                arm=arm,
                label=ARM_LABELS.get(arm, arm),
                n_seeds=len(cells),
                seeds=[r.seed for r in cells],
                test_auroc_mean=float(np.mean(values)),
                test_auroc_sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                test_auroc_values=values,
                pooled_auroc=pooled[0],
                pooled_ci_low=pooled[1],
                pooled_ci_high=pooled[2],
                median_abs_step_mean=float(np.mean([r.median_abs_step for r in cells])),
                n_val=cells[0].n_val,
                select_epochs=[r.select_epoch for r in cells],
            )
        )
    return out


def comparators() -> dict[str, tuple[float, float, float]]:
    """Name -> (AUROC, CI low, CI high) on the 195 genes."""
    out: dict[str, tuple[float, float, float]] = {}
    if osp.isfile(FBA_JSON):
        with open(FBA_JSON) as fh:
            fba = json.load(fh)
        for row in fba["rows"]:
            s = next(x for x in row["scores"] if x["gene_set"] == "ours_195")
            out[f"FBA {row['medium']}"] = (
                s["auroc"],
                s["auroc_ci_low"],
                s["auroc_ci_high"],
            )
    if osp.isfile(FCL_JSON):
        with open(FCL_JSON) as fh:
            fcl = json.load(fh)
        # `scored` rows: method x gene_set; ours_scored is the 195 genes every cell scores.
        for row in fcl["scored"]:
            if row["gene_set"] != "ours_scored" or row.get("seed") is not None:
                continue
            name = {"FCL": "FCL", "FBA": "FBA (released)"}.get(row["method"])
            if name is None:
                continue
            out[name] = (row["auroc"], row["ci_low"], row["ci_high"])
    return out


def style() -> None:
    apply_paper_style()
    plt.rcParams.update({"legend.fontsize": 6, "lines.linewidth": 0.8})


def figure(
    round_key: str,
    summaries: list[ArmSummary],
    histories: dict[str, list[dict[str, Any]]],
    rows: list[CellRow],
    comps: dict[str, tuple[float, float, float]],
) -> str:
    style()
    arms = [s.arm for s in summaries]
    color = {arm: PLOT_PALETTE[i % len(PLOT_PALETTE)] for i, arm in enumerate(arms)}
    width_in = mm_to_in(PANEL_WIDTHS_MM["full"])
    fig, axes = plt.subplots(2, 2, figsize=(width_in, mm_to_in(120)))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # a: validation curves
    for arm in arms:
        curves = []
        for h in histories.get(arm, []):
            c = np.array([e["val_pearson"] for e in h["history"]], dtype=float)
            curves.append(c)
            ax_a.plot(np.arange(len(c)), c, color=color[arm], alpha=0.25, linewidth=0.5)
        if curves:
            n = min(len(c) for c in curves)
            ax_a.plot(
                np.arange(n),
                np.mean([c[:n] for c in curves], axis=0),
                color=color[arm],
                label=ARM_LABELS.get(arm, arm),
            )
    ax_a.set_xlabel("epoch")
    ax_a.set_ylabel("validation AUROC")
    # The axis runs to 1.0 whatever the round reached, so the panels of every round read
    # on one scale; the floor drops to the lowest curve in tenths.
    floor = min(
        0.3,
        float(
            np.floor(
                10
                * np.nanmin(
                    [
                        np.nanmin([e["val_pearson"] for e in h["history"]])
                        for hs in histories.values()
                        for h in hs
                    ]
                )
            )
            / 10
        ),
    )
    ax_a.set_ylim(floor, 1.0)
    ax_a.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_a.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_a.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    # The arm legend is a FIGURE legend across the top, not an axes legend on a: with two
    # columns of long arm labels an axes legend ran over panel b.
    legend_handles, legend_labels = ax_a.get_legend_handles_labels()
    legend_ncol = 3
    legend_rows = int(np.ceil(len(legend_labels) / legend_ncol))

    # b: stability vs validation size
    for s in summaries:
        cells = [r for r in rows if r.arm == s.arm]
        ax_b.scatter(
            [r.n_val for r in cells],
            [r.median_abs_step for r in cells],
            color=color[s.arm],
            s=8,
            edgecolors="black",
            linewidths=0.3,
        )
    ax_b.set_xlabel("labeled validation genes")
    ax_b.set_ylabel("median |step| of validation AUROC")
    n_vals = [r.n_val for r in rows]
    # A log axis only when the round spans validation sizes (75 against 1,093 in round 1);
    # a round on one split file gets a linear axis with the one or two sizes it has.
    if max(n_vals) > 2 * min(n_vals):
        ax_b.set_xscale("log")
    else:
        ax_b.set_xticks(sorted(set(n_vals)))

    # c: test AUROC per arm with comparators
    x = 0
    ticks, labels = [], []
    for s in summaries:
        cells = [r for r in rows if r.arm == s.arm]
        ax_c.scatter(
            [x] * len(cells),
            [r.test_auroc_pinned for r in cells],
            color=color[s.arm],
            s=10,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )
        if (
            s.pooled_auroc is not None
            and s.pooled_ci_low is not None
            and s.pooled_ci_high is not None
        ):
            ax_c.errorbar(
                [x + 0.25],
                [s.pooled_auroc],
                yerr=[
                    [s.pooled_auroc - s.pooled_ci_low],
                    [s.pooled_ci_high - s.pooled_auroc],
                ],
                fmt="s",
                color=color[s.arm],
                markersize=3,
                capsize=1.5,
                elinewidth=0.6,
                markeredgecolor="black",
                markeredgewidth=0.3,
                zorder=3,
            )
        ticks.append(x)
        labels.append(s.arm)
        x += 1
    for name, (v, lo, hi) in comps.items():
        ax_c.errorbar(
            [x],
            [v],
            yerr=[[v - lo], [hi - v]],
            fmt="D",
            color=PLOT_PALETTE[5],
            markersize=3,
            capsize=1.5,
            elinewidth=0.6,
            markeredgecolor="black",
            markeredgewidth=0.3,
            zorder=3,
        )
        ticks.append(x)
        labels.append(name)
        x += 1
    ax_c.set_xticks(ticks)
    ax_c.set_xticklabels(labels, rotation=45, ha="right")
    ax_c.set_ylabel("held-out AUROC (195 genes)")
    ax_c.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_c.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_c.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax_c.set_ylim(0.4, 1.0)

    # d: gradient norms per module, seed mean, for arms that logged them
    keys = [
        "grad_norm_embedding_preprocessor_mean",
        "grad_norm_transformer_layers_mean",
        "grad_norm_essentiality_head_mean",
        "grad_norm_flux_layer_mean",
    ]
    styles = ["-", "--", ":", "-."]
    for arm in arms:
        hs = histories.get(arm, [])
        for key, ls in zip(keys, styles, strict=True):
            curves = [
                np.array([e.get(key, np.nan) for e in h["history"]], dtype=float)
                for h in hs
            ]
            if not curves or all(np.all(np.isnan(c)) for c in curves):
                continue
            n = min(len(c) for c in curves)
            ax_d.plot(
                np.arange(n),
                np.nanmean([c[:n] for c in curves], axis=0),
                color=color[arm],
                linestyle=ls,
                linewidth=0.7,
            )
    ax_d.set_xlabel("epoch")
    ax_d.set_ylabel("gradient norm per module (seed mean)")
    ax_d.set_yscale("log")
    from matplotlib.lines import Line2D

    ax_d.legend(
        [Line2D([0], [0], color="black", linestyle=ls) for ls in styles],
        ["preprocessor", "transformer", "essentiality head", "flux layer"],
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=2,
        frameon=False,
    )

    for ax, letter in zip(axes.flat, "abcd", strict=True):
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
        ax.text(
            -0.14,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
            ha="left",
        )
    # Fixed margins rather than tight_layout: the figure legend needs `legend_rows` lines
    # above the top row (about 3.2 mm each at 6 pt plus the row's own letter), and the
    # two columns sit at one gap whatever the length of b's axis label.
    height_mm = 120.0
    top = 1.0 - (legend_rows * 3.2 + 6.0) / height_mm
    fig.subplots_adjust(
        left=0.07, right=0.99, bottom=0.12, top=top, wspace=0.22, hspace=0.55
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.07, 1.0),
        ncol=legend_ncol,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=1.2,
    )
    os.makedirs(IMAGE_DIR, exist_ok=True)
    base = osp.join(IMAGE_DIR, f"{round_key}_readout_{timestamp()}")
    fig.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(fig, f"{base}.svg")
    plt.close(fig)
    return f"{base}.svg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="w12", choices=list(ROUNDS))
    args = parser.parse_args()
    rows, histories, preds = load_round(args.round)
    assert rows, f"no finished {args.round} result files under {RESULTS}"
    summaries = summarize(rows, preds, ROUNDS[args.round][1])
    comps = comparators()
    out = {
        "round": args.round,
        "n_cells": len(rows),
        "cells": [r.model_dump() for r in rows],
        "arms": [s.model_dump() for s in summaries],
        "comparators_on_195": {
            k: {"auroc": v[0], "ci_low": v[1], "ci_high": v[2]}
            for k, v in comps.items()
        },
        "bootstrap": {"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
    }
    path = osp.join(RESULTS, f"{args.round}_readout.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    table = pd.DataFrame([r.model_dump() for r in rows])
    table.to_csv(osp.join(RESULTS, f"{args.round}_readout.csv"), index=False)
    print(
        table[
            [
                "stem",
                "n_val",
                "select_epoch",
                "test_auroc_pinned",
                "test_auroc_ci_low",
                "test_auroc_ci_high",
                "median_abs_step",
                "wall_time_min",
            ]
        ].to_string(index=False)
    )
    for s in summaries:
        print(
            f"{s.arm:14s} n={s.n_seeds} mean {s.test_auroc_mean:.4f} sd {s.test_auroc_sd:.4f} "
            f"pooled {s.pooled_auroc if s.pooled_auroc is None else round(s.pooled_auroc, 4)} "
            f"[{s.pooled_ci_low}, {s.pooled_ci_high}] step {s.median_abs_step_mean:.4f} n_val {s.n_val} sel {s.select_epochs}"
        )
    for k, v in comps.items():
        print(f"{k:14s} {v[0]:.4f} [{v[1]:.3f}, {v[2]:.3f}]")
    print(figure(args.round, summaries, histories, rows, comps))
    print(path)


if __name__ == "__main__":
    main()
