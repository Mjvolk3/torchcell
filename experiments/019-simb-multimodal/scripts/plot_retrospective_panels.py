# experiments/019-simb-multimodal/scripts/plot_retrospective_panels.py
# [[experiments.019-simb-multimodal.scripts.plot_retrospective_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/plot_retrospective_panels
"""Two panels the retrospective needs that no existing script draws.

PANEL A -- achieved against ceiling, every strand. The single most misread number in this
project is a bare Pearson $r$: 0.24 on expression and 0.37 on betaxanthin are 31 % and 41 %
of their respective ceilings, and 0.08 on morphology is 13 % of a ceiling that is higher
than either. Plotting the pair rather than the score is what makes those comparable. The
bar is the achieved score, the open bar behind it is the ceiling, and a strand with no
estimable ceiling (amino acid) gets no open bar rather than a guessed one.

PANEL B -- where the peak sits inside the run. The expression strand has never been trained
to a peak at any budget up to 10,000 epochs, while the metabolite strands peak in the first
half and then overfit. Both facts are invisible in a score table and obvious in one plot of
peak epoch against last epoch, so this panel draws the diagonal (peak = last, meaning the
run was cut off while still improving) and puts every strand's best run on it.

Both panels read ONLY committed artifacts: results/round_leaderboards.csv for the scores
and epochs, and the three ceiling JSONs. Nothing here recomputes a score.

Panels are written under STABLE names (no timestamp) because notes-tex/019-simb-multimodal/
converts them by name; the write time goes in the results JSON instead.

Run from repo root:
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/plot_retrospective_panels.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

EXPERIMENT = "019-simb-multimodal"

# (strand key, short label). Order is worst-realized to best, so the panel reads as a
# ranking rather than as the document's narrative order.
STRANDS = [
    ("expression_morphology_joint", "expr+morph\njoint"),
    ("morphology", "morphology"),
    ("beta_carotene", "beta-carotene"),
    ("amino_acid", "amino\nacid"),
    ("expression", "expression"),
    ("expression_masked", "expression\nmasked"),
    ("betaxanthin", "betaxanthin"),
    ("betaxanthin_amino_acid_joint", "betaxanthin\n+ metabolome"),
]


# (dx, dy, horizontal alignment) in points, per strand, for panel B's point labels.
LABEL_OFFSETS = {
    "expression_morphology_joint": (5, -9, "left"),
    "morphology": (-5, 4, "right"),
    "beta_carotene": (6, 2, "left"),
    "amino_acid": (6, 6, "left"),
    "expression": (-5, 4, "right"),
    "expression_masked": (0, 7, "center"),
    "betaxanthin": (6, -8, "left"),
    "betaxanthin_amino_acid_joint": (-6, 4, "right"),
}


def _rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )


def _box(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def load(results_dir: str) -> tuple[pd.DataFrame, dict[str, float | None]]:
    board = pd.read_csv(osp.join(results_dir, "round_leaderboards.csv"))
    with open(osp.join(results_dir, "expression_ceiling_replicate.json")) as fh:
        expression = float(json.load(fh)["primary_ceiling_mean_sqrt_r"]["ceiling"])
    with open(osp.join(results_dir, "morphology_noise_ceiling.json")) as fh:
        morphology = float(json.load(fh)["ceiling_mean_model_features"])
    with open(osp.join(results_dir, "pigment_noise_ceiling.json")) as fh:
        pigment = json.load(fh)
    ceilings = {
        "expression": expression,
        "expression_masked": expression,
        "morphology": morphology,
        # The joint arm scores a mean over two heads with different ceilings, so no single
        # ceiling applies and none is drawn.
        "expression_morphology_joint": None,
        "betaxanthin": float(pigment["betaxanthin"]["ceiling_pearson"]),
        "betaxanthin_amino_acid_joint": float(pigment["betaxanthin"]["ceiling_pearson"]),
        # Mulleder has one replicate per strain and no released SE.
        "amino_acid": None,
        "beta_carotene": float(pigment["beta_carotene"]["ceiling_spearman"]),
    }
    return board, ceilings


def best_rows(board: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for key, _label in STRANDS:
        group = board[board["strand"] == key]
        live = group[~group["is_collapsed"].fillna(False).astype(bool)]
        out[key] = live.loc[live["primary_roll_max"].idxmax()]
    return out


def panel_ceiling(best: dict, ceilings: dict, png: str, svg: str) -> list[dict]:
    """Horizontal bars, because eight strand names do not fit on a vertical axis.

    At a column width of 88.5 mm, eight two-line category labels collide on an x axis; the
    same labels sit comfortably down a y axis. The ordering is worst-realized at the bottom
    to best at the top, so the bar length and the reading order agree.
    """
    _rc()
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(56.0)),
        constrained_layout=True,
    )
    recorded = []
    for i, (key, label) in enumerate(STRANDS):
        score = float(best[key]["primary_roll_max"])
        ceiling = ceilings[key]
        if ceiling is not None:
            # The ceiling is drawn as an open bar BEHIND the score, so the gap is the
            # headroom and is read directly rather than computed by the viewer.
            ax.barh(
                i,
                ceiling,
                height=0.72,
                facecolor="none",
                edgecolor="black",
                linewidth=0.5,
                linestyle=(0, (2, 1.5)),
                zorder=1,
            )
        ax.barh(
            i,
            score,
            height=0.72,
            color=PLOT_PALETTE[i % 6],
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )
        annotation = f"{score:.3f}" + ("" if ceiling is None else f"  ({100 * score / ceiling:.0f}%)")
        ax.text(score + 0.012, i, annotation, ha="left", va="center", fontsize=5, zorder=3)
        recorded.append(
            {"strand": key, "score": score, "ceiling": ceiling, "run_id": str(best[key]["run_id"])}
        )
    ax.set_yticks(np.arange(len(STRANDS)))
    ax.set_yticklabels([label.replace("\n", " ") for _key, label in STRANDS], fontsize=5)
    ax.set_xlabel("validation correlation (dashed outline = measured ceiling)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(STRANDS) - 0.4)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="x", which="both", linewidth=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)
    _box(ax)
    fig.savefig(png, dpi=300)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)
    return recorded


def panel_convergence(best: dict, png: str, svg: str) -> list[dict]:
    _rc()
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(52.0)),
        constrained_layout=True,
    )
    recorded = []
    for i, (key, label) in enumerate(STRANDS):
        row = best[key]
        last = float(row["epochs"])
        peak = float(row["primary_epoch_at_roll_max"])
        ax.scatter(
            last,
            peak / last,
            s=18,
            color=PLOT_PALETTE[i % 6],
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        # Label placement is an explicit table rather than an automatic rule. Several
        # strands land on the same epoch budget (999) at similar peak fractions, so no
        # single offset separates them, and an eight-point panel does not justify a
        # collision solver.
        dx, dy, ha = LABEL_OFFSETS[key]
        ax.annotate(
            label.replace("\n", " "),
            (last, peak / last),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            fontsize=5,
        )
        recorded.append(
            {"strand": key, "last_epoch": last, "peak_epoch": peak, "peak_fraction": peak / last}
        )
    # Above this line the run was still improving when it stopped; below it, the run had
    # peaked and was overfitting for the rest of its budget. Both guide labels are in AXES
    # coordinates: in data coordinates on a log x axis starting at 40 they land off canvas.
    ax.axhline(0.9, color="black", linewidth=0.5, linestyle=(0, (3, 2)))
    ax.text(
        0.02,
        0.92,
        "cut off while still improving",
        transform=ax.transAxes,
        fontsize=5,
        va="bottom",
        color="#444444",
    )
    ax.axhline(0.5, color="#999999", linewidth=0.5, linestyle=(0, (1, 2)))
    ax.text(
        0.98,
        0.52,
        "peaked in the first half, then overfit",
        transform=ax.transAxes,
        ha="right",
        fontsize=5,
        va="bottom",
        color="#444444",
    )
    ax.set_xscale("log")
    ax.set_xlim(40, 20000)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("epochs the best run reached")
    ax.set_ylabel("peak epoch as a fraction of the run")
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="both", which="major", linewidth=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)
    _box(ax)
    fig.savefig(png, dpi=300)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)
    return recorded


def main() -> None:
    load_dotenv()
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    images_dir = os.environ["ASSET_IMAGES_DIR"]
    results_dir = osp.join(experiment_root, EXPERIMENT, "results")
    out_dir = osp.join(images_dir, EXPERIMENT)
    os.makedirs(out_dir, exist_ok=True)

    board, ceilings = load(results_dir)
    best = best_rows(board)

    ceiling_png = osp.join(out_dir, "retrospective_achieved_vs_ceiling.png")
    ceiling_svg = osp.join(out_dir, "retrospective_achieved_vs_ceiling.svg")
    conv_png = osp.join(out_dir, "retrospective_peak_position.png")
    conv_svg = osp.join(out_dir, "retrospective_peak_position.svg")

    payload = {
        "achieved_vs_ceiling": panel_ceiling(best, ceilings, ceiling_png, ceiling_svg),
        "peak_position": panel_convergence(best, conv_png, conv_svg),
        "written_at": timestamp(),
        "figures": [ceiling_svg, conv_svg],
    }
    with open(osp.join(results_dir, "retrospective_panels.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload["achieved_vs_ceiling"], indent=2))
    print(f"-> {ceiling_svg}\n-> {conv_svg}")


if __name__ == "__main__":
    main()
