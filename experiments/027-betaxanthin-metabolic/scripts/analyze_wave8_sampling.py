# experiments/027-betaxanthin-metabolic/scripts/analyze_wave8_sampling.py
# [[experiments.027-betaxanthin-metabolic.wave8-sampling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/analyze_wave8_sampling.py

r"""Score the wave-8 K-draw contrast, paired by seed, and say what the pairing can resolve.

THE CONTRAST HAS THREE COLUMNS and only one of them is the treatment.

``K = 16, sampled``
    the treatment. `flux_draws=16`, the latent draw on.
``K = 1, sampled``
    the control the treatment is PAIRED against. Same recipe, same seed, the latent draw
    on, one draw. The only difference from the treatment is the draw count.
``K = 1, deterministic``
    the wave-7 cells at the same seeds, latent draw OFF. Not the pair: it measures what
    turning the draw on costs by itself, which is a second question.

WHY THE PAIRING IS THE WHOLE POINT. 019 measured a between-seed sd near 0.04 on this
phenotype against a within-seed across-arm sd of 0.0058, so the nuisance axis is roughly
seven times the signal axis. An unpaired comparison of three cells against three cells is
reading the seed draw. The statistic reported here is therefore the mean of the per-seed
DIFFERENCE, its sd across seeds, and n; with n = 3 that sd is itself barely estimated and
the note says so rather than dressing it as a test.

SCORE: pinned Merzbacher test Spearman on the 629 held-out genes at the epoch selected by
the centered 5-epoch moving average of validation Pearson. Never a max over epochs, which
is an upward-biased order statistic whose bias grows with the epoch budget.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/analyze_wave8_sampling.py

Writes ``results/wave8_sampling_summary.json`` and a figure under
``$ASSET_IMAGES_DIR/027-betaxanthin-metabolic/``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import PLOT_PALETTE, mm_to_in, savefig_true_size_svg  # noqa: E402

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
RESULTS = osp.join(EXP_DIR, "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "027-betaxanthin-metabolic")

SEEDS = (2703, 2704, 2705)

#: column -> the result-file template it is read from. Two rounds, and they are NOT
#: comparable to each other: the 80-epoch cells all selected within a few epochs of the end
#: of their budget with the training loss still falling, so they are the SHORT-BUDGET read
#: and the 200-epoch cells are the definitive one.
COLUMNS: dict[str, str] = {
    "k16_sampled": "wave8_samp_k16_s{seed}.json",
    "k1_sampled": "wave8_samp_k1_s{seed}.json",
    "k1_deterministic": "wave6_proj_flux_projection_s{seed}.json",
    "k128_e200": "wave8_samp_k128_e200_s{seed}.json",
    "k1_e200": "wave8_samp_k1_e200_s{seed}.json",
}

#: Which columns belong to which round, so a figure never puts an 80-epoch cell beside a
#: 200-epoch one as though the two were the same experiment.
ROUNDS: dict[str, list[str]] = {
    "80 epochs (short budget)": ["k16_sampled", "k1_sampled", "k1_deterministic"],
    "200 epochs": ["k128_e200", "k1_e200"],
}

LABELS: dict[str, str] = {
    "k16_sampled": "K = 16\nsampled",
    "k1_sampled": "K = 1\nsampled",
    "k1_deterministic": "K = 1\ndeterministic",
    "k128_e200": "K = 128\n200 epochs",
    "k1_e200": "K = 1\n200 epochs",
}


class CellScore(BaseModel):
    """One cell: what it ran and the one number it is scored on."""

    model_config = ConfigDict(extra="forbid")

    column: str
    seed: int
    result_file: str
    present: bool
    select_epoch: int | None = None
    select_rule: str | None = None
    n_epochs: int | None = Field(
        default=None,
        description=(
            "Epochs the cell ran. Reported beside `select_epoch` because a selection at the "
            "end of the budget means the curve had not stopped improving."
        ),
    )
    test_spearman_pinned: float | None = None
    flux_draws: int | None = None
    latent_stochastic: bool | None = None
    effective_kl_weight: float | None = None
    latent_sigma_first: float | None = None
    latent_sigma_last: float | None = None
    latent_sigma_shrink: float | None = Field(
        default=None,
        description="First epoch's median sigma divided by the last epoch's.",
    )
    draw_pred_sd_first: float | None = None
    draw_pred_sd_last: float | None = None
    val_pred_sd_ratio_max: float | None = None
    wandb_url: str | None = None


class PairedContrast(BaseModel):
    """One paired contrast, reported as the difference statistic and nothing more."""

    model_config = ConfigDict(extra="forbid")

    treatment: str
    control: str
    seeds: list[int]
    differences: list[float]
    n: int
    mean_difference: float
    sd_difference: float | None = Field(
        default=None,
        description=(
            "Sample sd of the per-seed difference. None at n < 2. At n = 3 it is itself "
            "barely estimated, so it bounds the reading rather than supporting a test."
        ),
    )


def read_cell(column: str, seed: int) -> CellScore:
    """Read one cell's score, or record that it has not landed."""
    name = COLUMNS[column].format(seed=seed)
    path = osp.join(RESULTS, name)
    if not osp.isfile(path):
        return CellScore(column=column, seed=seed, result_file=name, present=False)
    with open(path) as handle:
        result: dict[str, Any] = json.load(handle)
    history = result.get("history") or []
    sampling = result["flux_meta"].get("sampling", {})
    sigma = [
        row["latent_sigma_p50"] for row in history if "latent_sigma_p50" in row
    ] or None
    spread = [
        row["flux_draw_pred_sd"] for row in history if "flux_draw_pred_sd" in row
    ] or None
    return CellScore(
        column=column,
        seed=seed,
        result_file=name,
        present=True,
        select_epoch=int(result["select_epoch"]),
        select_rule=str(result["select_rule"]),
        n_epochs=len(history) or None,
        test_spearman_pinned=float(result["test"]["pinned"]["spearman"]),
        flux_draws=sampling.get("flux_draws"),
        latent_stochastic=sampling.get("latent_stochastic"),
        effective_kl_weight=sampling.get("effective_kl_weight"),
        latent_sigma_first=sigma[0] if sigma else None,
        latent_sigma_last=sigma[-1] if sigma else None,
        latent_sigma_shrink=(sigma[0] / sigma[-1]) if sigma else None,
        draw_pred_sd_first=spread[0] if spread else None,
        draw_pred_sd_last=spread[-1] if spread else None,
        val_pred_sd_ratio_max=(
            max(row["val_pred_sd_ratio"] for row in history) if history else None
        ),
        wandb_url=result.get("wandb_url"),
    )


def contrast(
    scores: dict[tuple[str, int], CellScore], treatment: str, control: str
) -> PairedContrast:
    """The per-seed difference between two columns, over the seeds where both landed."""
    seeds: list[int] = []
    differences: list[float] = []
    for seed in SEEDS:
        a, b = scores[(treatment, seed)], scores[(control, seed)]
        if a.test_spearman_pinned is None or b.test_spearman_pinned is None:
            continue
        seeds.append(seed)
        differences.append(a.test_spearman_pinned - b.test_spearman_pinned)
    array = np.array(differences, dtype=float)
    return PairedContrast(
        treatment=treatment,
        control=control,
        seeds=seeds,
        differences=differences,
        n=len(differences),
        mean_difference=float(array.mean()) if array.size else float("nan"),
        sd_difference=float(array.std(ddof=1)) if array.size > 1 else None,
    )


def make_figure(
    scores: dict[tuple[str, int], CellScore], contrasts: list[PairedContrast]
) -> str:
    """Two panels: the paired scores themselves, and the paired differences.

    Panel a keeps every cell visible and JOINS the cells of one seed with a line, because
    the line is the comparison: three unpaired clouds at a between-seed sd of 0.04 would
    show nothing. Panel b is the quantity the contrast is actually read on, the per-seed
    difference, with zero marked.
    """
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
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(179.0), mm_to_in(62.0)), constrained_layout=True
    )
    # POSITIONS ARE GROUPED BY ROUND, with a gap between them, and a seed's line never
    # crosses the gap. An 80-epoch cell and a 200-epoch cell are different experiments, and
    # a line joining them would read as a trend where there is only a budget change.
    order: list[str] = []
    positions_list: list[float] = []
    cursor = 0.0
    round_of: dict[str, str] = {}
    for label, columns in ROUNDS.items():
        for column in columns:
            order.append(column)
            positions_list.append(cursor)
            round_of[column] = label
            cursor += 1.0
        cursor += 0.6
    positions = np.array(positions_list, dtype=float)
    place = dict(zip(order, positions))

    ax = axes[0]
    for i, seed in enumerate(SEEDS):
        for label, columns in ROUNDS.items():
            drawn = [
                (place[c], scores[(c, seed)].test_spearman_pinned)
                for c in columns
                if scores[(c, seed)].test_spearman_pinned is not None
            ]
            if not drawn:
                continue
            ax.plot(
                [p for p, _ in drawn],
                [v for _, v in drawn],
                marker="o",
                markersize=3,
                linewidth=0.8,
                color=PLOT_PALETTE[i],
                label=f"seed {seed}" if label == next(iter(ROUNDS)) else None,
            )
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[c] for c in order])
    ax.set_xlim(positions[0] - 0.4, positions[-1] + 0.4)
    ax.set_ylabel("pinned test Spearman")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_title("a  every cell, joined within a seed", loc="left", fontsize=6)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.08),
        ncol=len(SEEDS),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
    )

    ax = axes[1]
    labels = [f"{c.treatment}\nminus {c.control}" for c in contrasts]
    spots = np.arange(len(contrasts), dtype=float)
    for i, item in enumerate(contrasts):
        ax.scatter(
            [spots[i]] * len(item.differences),
            item.differences,
            s=10,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        if item.n:
            ax.plot(
                [spots[i] - 0.22, spots[i] + 0.22],
                [item.mean_difference] * 2,
                color=PLOT_PALETTE[i],
                linewidth=1.2,
                zorder=2,
            )
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks(spots)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(contrasts) - 0.5)
    ax.set_ylabel("paired difference in test Spearman")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_title("b  the per-seed difference, with its mean", loc="left", fontsize=6)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(width=0.5)

    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, f"wave8_sampling_paired_{timestamp()}")
    savefig_true_size_svg(fig, f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=300)
    plt.close(fig)
    return f"{stem}.svg"


def main() -> None:
    """Read every cell, form the two contrasts, write the JSON and the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="wave8_sampling_summary.json")
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip the figure, for a partial night where not every cell has landed.",
    )
    args = parser.parse_args()

    scores = {
        (column, seed): read_cell(column, seed) for column in COLUMNS for seed in SEEDS
    }
    contrasts = [
        contrast(scores, "k16_sampled", "k1_sampled"),
        contrast(scores, "k1_sampled", "k1_deterministic"),
        contrast(scores, "k128_e200", "k1_e200"),
    ]
    for row in sorted(scores.values(), key=lambda s: (s.column, s.seed)):
        if not row.present:
            print(f"{row.column:18s} s{row.seed}  NOT LANDED")
            continue
        # THE SELECTED EPOCH IS PRINTED WITH THE BUDGET IT CAME FROM. A cell that selects
        # at the end of its budget has not finished improving, and its score is where it
        # happened to be rather than what it converges to.
        budget = row.n_epochs or 0
        unfinished = (
            budget > 0
            and row.select_epoch is not None
            and row.select_epoch >= budget - 5
        )
        tail = " AT THE END OF ITS BUDGET" if unfinished else ""
        print(
            f"{row.column:18s} s{row.seed}  ep {row.select_epoch:3d}/{budget:3d}  "
            f"test {row.test_spearman_pinned:+.4f}{tail}"
        )
    for item in contrasts:
        sd = "n/a" if item.sd_difference is None else f"{item.sd_difference:.4f}"
        print(
            f"{item.treatment} minus {item.control}: mean "
            f"{item.mean_difference:+.4f}, sd {sd}, n {item.n}"
        )

    figure = None if args.no_figure else make_figure(scores, contrasts)
    payload = {
        "generated_at": timestamp(),
        "score": (
            "pinned Merzbacher test Spearman on 629 genes at the validation-selected epoch "
            "(centered 5-epoch moving average of validation Pearson)"
        ),
        "figure": figure,
        "cells": [row.model_dump() for row in scores.values()],
        "contrasts": [item.model_dump() for item in contrasts],
    }
    out_path = osp.join(RESULTS, args.out)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path}")
    if figure:
        print(f"wrote {figure}")


if __name__ == "__main__":
    main()
