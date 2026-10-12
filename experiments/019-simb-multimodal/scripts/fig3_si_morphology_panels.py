# experiments/019-simb-multimodal/scripts/fig3_si_morphology_panels.py
# [[experiments.019-simb-multimodal.scripts.fig3_si_morphology_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/fig3_si_morphology_panels
"""The data panels of the morphology SI figure (round v24), drawn from committed result files.

Every number is read from a results file; nothing is typed. Color is the input family an
arm reads (the metabolism and environment gate convention): taupe, the gene's own sequence
(genotype only); the fourth palette slot, the strain's measured phenotype as input
(transductive); gray, a permuted control. The learner is in the label. Scores are the
registered window, the mean of val/morphology/pearson_per_feature over epochs 1,000 to
1,199, over the 116 moving CalMorph features; a baseline is its validation score with the
embedding, rank and neighbor count selected on validation inside each split seed (an
upward-biased selection, stated in the caption).

  target_reliability   per-feature replicate reliability of the 278 CalMorph features as a
       histogram, the 0.5 threshold that defines the 116 moving features, and the ceiling
       (sqrt of reliability) of the moving set. morphology_feature_ceiling.csv,
       calmorph_moving_features.json, morph_v24_readout.json context.
  genotype_only        the inductive bars: ceiling; ridge and nearest neighbor on sequence
       embeddings (four split seeds, morphology_baselines_split_fig3_core_moving/); the CGT
       off the perturbed CLS on every morphology genotype (M_cls, twelve seeds) and on the
       proteome store's strains (MP_cls, twelve). Seed dots over the bars.
  measured_input       the transductive contrast: expression side, the strain's own
       expression as input (CMP_expr) against another strain's (CMP_exprperm); proteome
       side, the strain's own proteome (CM_prot) against another strain's (CM_protperm)
       and against genotype only on the same strains (MP_cls). Dashed, the ridge from the
       same measured modality on the moving features (expression_morphology_covariation,
       proteome_morphology_covariation); dotted, the proteome ridge with the growth axis
       removed.
  paired_difference    per split seed, own minus permuted (expression) and own minus
       genotype only (proteome), the figure's registered contrasts.
  fit_diagnostic       every run's training fit (traineval Pearson, last) against its
       window score, by arm: the genotype-only runs that never fit the training set, the
       two that escaped the plateau, and the conditioned runs that fit and overfit.
  ladder               the validation score at epochs 200 to 1,199, mean over seeds per arm.

Writes $ASSET_IMAGES_DIR/019-simb-multimodal/fig3_si_morph_<name>.{svg,png}.

    python experiments/019-simb-multimodal/scripts/fig3_si_morphology_panels.py
"""

from __future__ import annotations

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
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.utils import (  # noqa: E402
    MAX_HEIGHT_MM,
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-simb-multimodal")
PANEL_HEIGHT_MM = 48.0
SEQUENCE, SEQUENCE_FILL = PLOT_PALETTE[17], PLOT_PALETTE_FILL[17]
MEASURED, MEASURED_FILL = PLOT_PALETTE[3], PLOT_PALETTE_FILL[3]
CONTROL, CONTROL_FILL = PLOT_PALETTE[5], PLOT_PALETTE_FILL[5]
CEILING_FILL = "#EFEFEF"
LEGEND = {"frameon": True, "edgecolor": "black", "fancybox": False, "framealpha": 1.0}
SCORE_LABEL = "val Pearson per feature (116 moving)"
ARM_LABEL = {
    "M_cls": "CGT, genotype only, all strains",
    "MP_cls": "CGT, genotype only, proteome strains",
    "M_pool": "CGT, genotype only + gene pool",
    "CMP_expr": "own expression as input",
    "CMP_exprperm": "another strain's expression",
    "CM_expr": "own expression, CLS only (null by construction)",
    "CM_exprperm": "another strain's expression, CLS only",
    "CM_prot": "own proteome as input",
    "CM_protperm": "another strain's proteome",
}
ARM_FAMILY = {
    "M_cls": SEQUENCE,
    "MP_cls": SEQUENCE,
    "M_pool": SEQUENCE,
    "CMP_expr": MEASURED,
    "CM_prot": MEASURED,
    "CM_expr": MEASURED,
    "CMP_exprperm": CONTROL,
    "CM_protperm": CONTROL,
    "CM_exprperm": CONTROL,
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "svg.fonttype": "none",
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.5,
    }
)


def _load(name: str) -> dict[str, Any]:
    with open(osp.join(RESULTS, name)) as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _figure(width_key: str, ncols: int = 1, height_mm: float = PANEL_HEIGHT_MM, **kw: Any) -> tuple[Figure, Any]:
    assert height_mm <= MAX_HEIGHT_MM
    return plt.subplots(1, ncols, figsize=(mm_to_in(PANEL_WIDTHS_MM[width_key]), mm_to_in(height_mm)), **kw)


def _box(ax: Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")
    ax.set_axisbelow(True)


def _grid(ax: Axes, major: float = 0.2, minor: float = 0.1, axis: str = "y") -> None:
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(MultipleLocator(major))
    target.set_minor_locator(MultipleLocator(minor))
    ax.tick_params(axis=axis, which="minor", length=0)
    ax.grid(axis=axis, which="both", color="#CCCCCC", lw=0.4)


def _save(fig: Figure, name: str) -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"fig3_si_morph_{name}.svg"))
    fig.savefig(osp.join(IMG_DIR, f"fig3_si_morph_{name}.png"), dpi=300)
    plt.close(fig)
    print(f"wrote {osp.join(IMG_DIR, f'fig3_si_morph_{name}.svg')}")


def _runs(readout: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    return sorted([r for r in readout["runs"] if r["arm"] == arm and r.get("finished")], key=lambda r: r["split"])


def _bar(ax: Axes, x: float, values: list[float], color: str, fill: str | None = None, width: float = 0.7) -> None:
    """A mean bar with its s.d. and one dot per seed."""
    v = np.array(values, dtype=float)
    ax.bar(x, v.mean(), width=width, color=fill or color, edgecolor="black", linewidth=0.5, zorder=2)
    if len(v) > 1:
        ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="none", ecolor="black", elinewidth=0.5, capsize=1.5, zorder=3)
    rng = np.random.default_rng(0)
    ax.scatter(x + rng.uniform(-0.2, 0.2, len(v)), v, s=4, color="black", zorder=4, linewidths=0)


def _baselines() -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"ridge": [], "knn": []}
    for f in sorted(glob.glob(osp.join(RESULTS, "morphology_baselines_split_fig3_core_moving", "seed*.json"))):
        with open(f) as fh:
            b = json.load(fh)
        for key, name in (("B2_bilinear", "ridge"), ("B3_neighbor_average", "knn")):
            best = b[key]["best_embedding"]
            out[name].append(b[key]["by_embedding"][best]["selected_on_val"]["val_pearson_per_feature"])
    return out


def panel_target_reliability() -> None:
    df = pd.read_csv(osp.join(RESULTS, "morphology_feature_ceiling.csv"), index_col=0)
    moving = set(_load("calmorph_moving_features.json")["moving"])
    ctx = _load("morph_v24_readout.json")["context"]
    fig, ax = _figure("third")
    rel = df["reliability"].clip(0, 1)
    bins = np.linspace(0, 1, 21)
    ax.hist(rel[~df.index.isin(moving)], bins=bins, color=CONTROL_FILL, edgecolor="black", linewidth=0.5, label=f"quiet, {int((~df.index.isin(moving)).sum())} features")
    ax.hist(rel[df.index.isin(moving)], bins=bins, color=MEASURED_FILL, edgecolor="black", linewidth=0.5, label=f"moving, {len(moving)} features", bottom=None)
    ax.axvline(0.5, color="black", lw=0.6, ls="--")
    ax.set_xlabel("replicate reliability per CalMorph feature")
    ax.set_ylabel("features (n)")
    ax.set_ylim(0, 34)
    ax.text(0.98, 0.74, f"ceiling (sqrt reliability), mean:\nmoving set {ctx['ceiling_moving_mean']:.3f}, all 278 {ctx['ceiling_all_mean']:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ax.legend(loc="upper right", **LEGEND)
    _box(ax)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.95)
    _save(fig, "target_reliability")


def panel_genotype_only() -> None:
    ro = _load("morph_v24_readout.json")
    base = _baselines()
    ceiling = ro["context"]["ceiling_moving_mean"]
    fig, ax = _figure("third")
    xs = [0, 1, 2, 3, 4]
    ax.bar(0, ceiling, width=0.7, color=CEILING_FILL, edgecolor="black", linewidth=0.5, zorder=2)
    _bar(ax, 1, base["ridge"], SEQUENCE)
    _bar(ax, 2, base["knn"], SEQUENCE)
    _bar(ax, 3, [r["window_mean"] for r in _runs(ro, "M_cls")], SEQUENCE)
    _bar(ax, 4, [r["window_mean"] for r in _runs(ro, "MP_cls")], SEQUENCE)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            "replicate\nceiling",
            f"ridge on\nsequence\n(n = {len(base['ridge'])})",
            f"nearest\nneighbor\n(n = {len(base['knn'])})",
            f"CGT, all\nstrains\n(n = {len(_runs(ro, 'M_cls'))})",
            f"CGT, prot.\nstrains\n(n = {len(_runs(ro, 'MP_cls'))})",
        ]
    )
    ax.set_ylim(0, 1.0)
    _grid(ax)
    ax.set_ylabel(SCORE_LABEL)
    ax.set_title("genotype only (inductive)")
    _box(ax)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.3, top=0.9)
    _save(fig, "genotype_only")


def panel_measured_input() -> None:
    ro = _load("morph_v24_readout.json")
    em = _load("expression_morphology_covariation.json")["expression_to_morphology"]
    pm = _load("proteome_morphology_covariation.json")["proteome_to_morphology"]
    fig, axes = _figure("half", ncols=2, sharey=True)
    short = {"CMP_expr": "own", "CMP_exprperm": "another\nstrain's", "CM_prot": "own", "CM_protperm": "another\nstrain's", "MP_cls": "genotype\nonly"}
    groups = [
        (axes[0], "expression as input", [("CMP_expr", MEASURED), ("CMP_exprperm", CONTROL)], em["moving_features"]["median"], None),
        (axes[1], "proteome as input", [("CM_prot", MEASURED), ("CM_protperm", CONTROL), ("MP_cls", SEQUENCE)], pm["moving_features"]["median"], pm["growth_axis_removed_moving"]["median"]),
    ]
    for ax, title, arms, ridge, ridge_growth in groups:
        labels = []
        for i, (arm, color) in enumerate(arms):
            runs = _runs(ro, arm)
            _bar(ax, i, [r["window_mean"] for r in runs], color)
            labels.append(f"{short[arm]}\n(n = {len(runs)})")
        ax.axhline(ridge, color="black", lw=0.6, ls="--")
        if ridge_growth is not None:
            ax.axhline(ridge_growth, color="black", lw=0.6, ls=":")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylim(0, 0.4)
        _grid(ax, 0.1, 0.05)
        _box(ax)
    axes[0].set_ylabel(SCORE_LABEL)
    handles = [
        Patch(facecolor=MEASURED, edgecolor="black", label="own measurement as input (transductive)"),
        Patch(facecolor=CONTROL, edgecolor="black", label="another strain's measurement (control)"),
        Patch(facecolor=SEQUENCE, edgecolor="black", label="genotype only, same strains"),
        Line2D([0], [0], color="black", lw=0.6, ls="--", label="ridge from the measurement"),
        Line2D([0], [0], color="black", lw=0.6, ls=":", label="ridge, growth axis removed"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=2, columnspacing=0.8, handletextpad=0.4, **LEGEND)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.66, wspace=0.08)
    _save(fig, "measured_input")


def panel_paired_difference() -> None:
    ro = _load("morph_v24_readout.json")
    fig, axes = _figure("third", ncols=2, sharey=True)
    pairs = [
        (axes[0], "expression:\nown minus permuted", "CMP_expr", "CMP_exprperm", MEASURED),
        (axes[1], "proteome:\nown minus genotype only", "CM_prot", "MP_cls", MEASURED),
    ]
    for ax, title, a, b, color in pairs:
        ra = {r["split"]: r["window_mean"] for r in _runs(ro, a)}
        rb = {r["split"]: r["window_mean"] for r in _runs(ro, b)}
        splits = sorted(set(ra) & set(rb))
        diffs = np.array([ra[s] - rb[s] for s in splits])
        ax.bar(splits, diffs, width=0.7, color=color, edgecolor="black", linewidth=0.5, zorder=2)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("split seed")
        ax.text(0.03, 0.95, f"mean {diffs.mean():+.3f}, n = {len(splits)}\n{(diffs > 0).sum()} of {len(splits)} above 0", transform=ax.transAxes, ha="left", va="top", fontsize=6)
        ax.set_xticks(range(0, 12, 2))
        _grid(ax, 0.1, 0.05)
        _box(ax)
    axes[0].set_ylabel("difference, val Pearson per feature")
    axes[0].set_ylim(-0.05, 0.3)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.24, top=0.82, wspace=0.08)
    _save(fig, "paired_difference")


def panel_fit_diagnostic() -> None:
    ro = _load("morph_v24_readout.json")
    fig, ax = _figure("third", height_mm=64.0)
    markers = {"M_cls": "o", "MP_cls": "s", "M_pool": "D", "CMP_expr": "o", "CM_prot": "s", "CMP_exprperm": "o", "CM_protperm": "s"}
    for arm, m in markers.items():
        runs = _runs(ro, arm)
        if not runs:
            continue
        ax.scatter([r["traineval_pearson_last"] for r in runs], [r["window_mean"] for r in runs], s=9, marker=m, facecolor=ARM_FAMILY[arm], edgecolor="black", linewidths=0.4, label=f"{ARM_LABEL[arm]} (n = {len(runs)})", zorder=3)
    ax.plot([0, 0.8], [0, 0.8], color="#999999", lw=0.5, ls="--")
    ax.set_xlabel("training-set Pearson per feature, last epoch")
    ax.set_ylabel(SCORE_LABEL)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.35)
    _grid(ax, 0.1, 0.05)
    _grid(ax, 0.2, 0.1, axis="x")
    fig.legend(handles=ax.get_legend_handles_labels()[0], loc="upper left", bbox_to_anchor=(0.02, 1.0), ncol=1, handletextpad=0.4, labelspacing=0.25, **LEGEND)
    _box(ax)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.15, top=0.6)
    _save(fig, "fit_diagnostic")


def panel_ladder() -> None:
    ro = _load("morph_v24_readout.json")
    fig, ax = _figure("third", height_mm=64.0)
    styles = {"M_cls": ("-", SEQUENCE), "MP_cls": ("--", SEQUENCE), "CMP_expr": ("-", MEASURED), "CM_prot": ("--", MEASURED), "CMP_exprperm": ("-", CONTROL), "CM_protperm": ("--", CONTROL)}
    for arm, (ls, color) in styles.items():
        runs = _runs(ro, arm)
        if not runs:
            continue
        epochs = sorted(int(k) for k in runs[0]["ladder"])
        mean = [np.mean([r["ladder"][str(e)] for r in runs]) for e in epochs]
        ax.plot(epochs, mean, ls=ls, color=color, lw=0.9, marker="o", ms=2, label=f"{ARM_LABEL[arm]} (n = {len(runs)})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val Pearson per feature, mean over seeds")
    ax.set_ylim(0, 0.25)
    _grid(ax, 0.1, 0.05)
    fig.legend(handles=ax.get_legend_handles_labels()[0], loc="upper left", bbox_to_anchor=(0.02, 1.0), ncol=1, handletextpad=0.4, labelspacing=0.25, **LEGEND)
    _box(ax)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.15, top=0.62)
    _save(fig, "ladder")


def main() -> None:
    panel_target_reliability()
    panel_genotype_only()
    panel_measured_input()
    panel_paired_difference()
    panel_fit_diagnostic()
    panel_ladder()


if __name__ == "__main__":
    main()
