# experiments/043-expression-proteome-morphology/scripts/figure_043_common.py
# [[experiments.043-expression-proteome-morphology.scripts.figure_043_common]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/figure_043_common
"""Shared layout of the 043 figure sheets (the two multi-panel figures the owner asked
for on 2026-10-27: one for the setup and the pretraining, one for the preliminary
evaluations).

The look is the 040 thesis sheet (``figure_thesis_panels.py`` on the 040 branch):
a 179 mm wide sheet of lettered panels, Arial 6 pt, bold lowercase letters on their
own line above each row, each letter followed by a status mark that says whether the
panel reads from a result on file (``ready``), from a partial result (``partial``), is
drawn from an idea only (``todo``), or is a boxed placeholder with the panel's intent
written inside it (``placeholder``). Panel modules expose ``PANELS``, a dict from
panel key to a function ``draw(ax, R) -> status``; ``figure_043_sheets.py`` composes
them. ``Reads`` loads every result file once and lazily.

Every number drawn comes from a file under ``results/`` or a W&B history cached
there; nothing is typed in.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import subprocess
from functools import cached_property
from typing import Any

import matplotlib
import numpy as np
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
SLUG = "043-expression-proteome-morphology"
R043 = osp.join(EXPERIMENT_ROOT, SLUG, "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, SLUG)
ENTITY, PROJECT = "zhao-group", "torchcell_043-expression-proteome-morphology"
WANDB_RUN = f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/"

RC = {
    "font.family": "Arial",
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "black",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "lines.linewidth": 0.8,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderpad": 0.2,
    "legend.labelspacing": 0.25,
    "legend.columnspacing": 1.0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.dpi": 72,
    "savefig.dpi": 72,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
}


def apply_rc() -> None:
    """The sheet style. Importing ``torchcell.models.cell_model_043`` resets matplotlib's
    rc, so ``new_sheet`` applies this again after every import.
    """
    plt.rcParams.update(RC)


apply_rc()

AMBER, BRICK, PURPLE, SAND, BLUE, GRAY = PLOT_PALETTE[:6]
AMBER_FILL, BRICK_FILL, PURPLE_FILL, SAND_FILL = PLOT_PALETTE_FILL[:4]

#: the twelve validation heads in the order the baselines figure draws them, with the
#: label a reader sees and the modality the dataset measures
DATASETS: list[tuple[str, str, str]] = [
    ("kemmeren", "Kemmeren 2014", "mRNA"),
    ("messner", "Messner 2023", "protein"),
    ("zelezniak", "Zelezniak 2018", "protein"),
    ("caudal", "Caudal 2024", "mRNA"),
    ("muenzner", "Muenzner 2024 protein", "protein"),
    ("turnover", "Muenzner 2024 turnover", "protein"),
    ("boocock", "Boocock 2025 pooled", "mRNA"),
    ("nguessan", "N'Guessan 2025", "mRNA"),
    ("vaishnav", "Vaishnav 2022", "promoter"),
    ("hu", "Hu 2007", "mRNA"),
    ("nadal", "Nadal-Ribelles 2025", "mRNA"),
    ("doubles", "Sameith 2015 doubles", "mRNA"),
]
DATASET_LABEL = {k: v for k, v, _ in DATASETS}
MODALITY_COLOR = {
    "mRNA": AMBER,
    "protein": BRICK,
    "promoter": PURPLE,
    "morphology": SAND,
}

#: the loss families and the arms the sheets contrast, in the palette's hue order
LOSS_COLOR = {"hlgauss": AMBER, "pinball": BRICK, "mse": PURPLE}
LOSS_LABEL = {"hlgauss": "HL-Gauss", "pinball": "pinball", "mse": "MSE"}
ARM_COLOR = {
    "bs16": AMBER,  # recipe v2: ontology, HL-Gauss, batch 16
    "pinball": BRICK,
    "pinball_injop": PURPLE,
    "inject_op": SAND,
    "routeB": BLUE,
    "baseline": GRAY,
}

STATUS_SYMBOL = {"ready": "●", "partial": "◐", "todo": "○", "placeholder": "□"}
STATUS_WORD = {
    "ready": "from a result on file",
    "partial": "partial result",
    "todo": "idea only",
    "placeholder": "placeholder",
}
LETTER_GAP_PT = 2.0
WIDE_TITLES: list[str] = []


class Reads:
    """Every result file the sheets read, loaded once each."""

    def json(self, name: str) -> Any:
        """One result file under ``results/``."""
        with open(osp.join(R043, name)) as fh:
            return json.load(fh)

    @cached_property
    def baselines(self) -> dict[str, Any]:
        """``baselines_043.json``."""
        return self.json("baselines_043.json")

    @cached_property
    def diagnostics(self) -> dict[str, Any]:
        """``diagnostics_043.json``."""
        return self.json("diagnostics_043.json")

    @cached_property
    def census(self) -> dict[str, Any]:
        """``build_043_store_census_v2.json``."""
        return self.json("build_043_store_census_v2.json")

    @cached_property
    def shapes(self) -> dict[str, Any]:
        """``data_shapes_043.json``."""
        return self.json("data_shapes_043.json")

    @cached_property
    def split(self) -> dict[str, Any]:
        """``global_split_v2_summary.json``."""
        return self.json("global_split_v2_summary.json")

    @cached_property
    def ontology(self) -> dict[str, Any]:
        """``ontology_annotation_043_v2.json``."""
        return self.json("ontology_annotation_043_v2.json")

    @cached_property
    def model_counts(self) -> dict[str, Any]:
        """``model_math_043_counts.json``."""
        return self.json("model_math_043_counts.json")

    @cached_property
    def ceilings(self) -> dict[str, Any]:
        """``within_dataset_ceilings.json``."""
        return self.json("within_dataset_ceilings.json")

    @cached_property
    def morphology_probe(self) -> dict[str, Any]:
        """``morphology_probe_043.json``."""
        return self.json("morphology_probe_043.json")

    def window_means(self, round_key: str) -> dict[str, Any]:
        """The window-means file of one round."""
        return self.json(f"{round_key}_window_means.json")


def new_sheet(height_mm: float, width_key: str = "full") -> Any:
    """A sheet at a standard Nature width and the given height, constrained layout."""
    apply_rc()
    fig = plt.figure(
        figsize=(mm_to_in(PANEL_WIDTHS_MM[width_key]), mm_to_in(height_mm)),
        layout="constrained",
    )
    # the top band (22 pt: a title line, a legend row and the letter) and the row gaps
    # hold the panel letters and their marks
    top = 1.0 - 22.0 / (72.0 * mm_to_in(height_mm))
    fig.get_layout_engine().set(
        w_pad=2 / 72, h_pad=2 / 72, hspace=0.06, wspace=0.04, rect=(0.0, 0.02, 1.0, top)
    )
    return fig


def rows(
    fig: Any,
    n_rows: int,
    n_cols: int | list[int],
    widths: list[list[float] | None] | None = None,
    heights: list[float] | None = None,
) -> list[Any]:
    """One subfigure per row, so a long tick label cannot set another row's margin.

    ``n_cols`` is one count for every row or one per row; ``widths`` gives a row's
    column width ratios (``None`` for equal columns); ``heights`` the row ratios.
    Returns the axes in reading order.
    """
    subs = fig.subfigures(n_rows, 1, height_ratios=heights)
    subs = [subs] if n_rows == 1 else list(subs)
    cols = [n_cols] * n_rows if isinstance(n_cols, int) else n_cols
    ratios = widths or [None] * n_rows
    out: list[Any] = []
    for sf, n, ratio in zip(subs, cols, ratios, strict=True):
        axes = sf.subplots(1, n, gridspec_kw={"width_ratios": ratio} if ratio else None)
        out.extend(list(np.atleast_1d(axes)))
    return out


def title(ax: Any, text: str, legend_rows: int = 0) -> None:
    """Panel title, one line, flush left, padded to clear a legend of ``legend_rows``
    rows above the axes. A title wider than its panel is reported at save time.
    """
    pad = 4.0 + 7.4 * legend_rows
    fig = ax.figure.get_figure(root=True)
    t = ax.set_title(text, pad=pad, loc="left")
    fig.draw_without_rendering()
    renderer = fig._get_renderer()
    scale = 72.0 / fig.dpi
    width = ax.get_window_extent().width * scale
    have = t.get_window_extent(renderer).width * scale
    if have > width:
        WIDE_TITLES.append(f"{text!r} is {have:.0f} pt wide, the panel {width:.0f}")
        print(f"  WIDE TITLE {WIDE_TITLES[-1]}", flush=True)


def fit_title(ax: Any, texts: list[str], legend_rows: int = 0) -> None:
    """``title`` with the first of ``texts`` that fits the panel's width (the long form
    on a wide panel, the short form when the panel is drawn at a third or a quarter).
    """
    fig = ax.figure.get_figure(root=True)
    fig.draw_without_rendering()
    renderer = fig._get_renderer()
    width = ax.get_window_extent().width
    for t in texts:
        probe = ax.text(0, 0, t, fontsize=6, transform=None)
        w = probe.get_window_extent(renderer).width
        probe.remove()
        if w <= 0.85 * width:
            title(ax, t, legend_rows)
            return
    title(ax, texts[-1], legend_rows)


def letters_status(fig: Any, axes: list[Any], statuses: list[str]) -> None:
    """Bold lowercase panel letters on a line of their own above each row, each
    followed by its status mark (the 040 white-cross rule: the letter sits above
    everything of its panel, at the panel's outer left; one row shares one baseline).
    """
    rows_: list[list[Any]] = []
    for ax in axes:
        if rows_ and rows_[-1][0].figure is ax.figure:
            rows_[-1].append(ax)
        else:
            rows_.append([ax])
    fig.draw_without_rendering()
    renderer = fig._get_renderer()
    inv = fig.transFigure.inverted()
    letters = iter("abcdefghijklmnop")
    status = iter(statuses)
    gap_px = LETTER_GAP_PT * fig.dpi / 72.0
    for row in rows_:
        boxes = [ax.get_tightbbox(renderer, for_layout_only=False) for ax in row]
        y_row = max(b.y1 for b in boxes) + gap_px
        for ax, box_ in zip(row, boxes, strict=True):
            ch = next(letters)
            st = next(status)
            x0, y0 = inv.transform((box_.x0, y_row))
            t = fig.text(
                x0, y0, ch, fontsize=8, fontweight="bold", ha="left", va="bottom"
            )
            t.set_in_layout(False)
            wide = 0.004 if ch in "mw" else 0.0
            g = fig.text(
                x0 + 0.013 + wide,
                y0 + 0.0015,
                STATUS_SYMBOL[st],
                fontsize=5,
                color=GRAY,
                ha="left",
                va="bottom",
                family="DejaVu Sans",
            )
            g.set_in_layout(False)


def status_key(fig: Any) -> None:
    """One line at the sheet's foot explaining the status marks."""
    parts = [f"{STATUS_SYMBOL[k]} {STATUS_WORD[k]}" for k in STATUS_SYMBOL]
    t = fig.text(
        0.0,
        -0.004,
        "    ".join(parts),
        fontsize=5,
        color=GRAY,
        ha="left",
        va="top",
        family="DejaVu Sans",
    )
    t.set_in_layout(False)


def placeholder(ax: Any, intent: str, head: str = "") -> str:
    """A boxed panel with its intent written inside, the paper's Fig 3 card style."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linestyle((0, (2, 2)))
        s.set_linewidth(0.6)
        s.set_color(GRAY)
    ax.set_facecolor("#FAFAFA")
    if head:
        ax.text(
            0.03,
            0.93,
            head,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            fontweight="bold",
        )
    ax.text(
        0.03,
        0.80 if head else 0.93,
        intent,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color="#333333",
        wrap=True,
        linespacing=1.3,
    )
    return "placeholder"


def hgrid(ax: Any, step: float, label_every: int = 2) -> None:
    """Gridlines every ``step``, labelled every ``label_every`` steps, minor marks hidden."""
    ax.yaxis.set_major_locator(MultipleLocator(step * label_every))
    ax.yaxis.set_minor_locator(MultipleLocator(step))
    ax.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.4, zorder=0)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_axisbelow(True)


def box(ax: Any) -> None:
    """Full black border, 0.5 pt, all four spines."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("black")


def save(fig: Any, name: str) -> dict[str, str]:
    """Write the timestamped PNG and true-size SVG of one sheet, and its bare-named PDF."""
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = timestamp()
    png = osp.join(OUT_DIR, f"{name}_{stamp}.png")
    svg = osp.join(OUT_DIR, f"{name}_{stamp}.svg")
    pdf = osp.join(OUT_DIR, f"{name}.pdf")
    savefig_true_size_svg(fig, svg)
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    subprocess.run(
        ["pdftoppm", "-r", "300", "-png", "-singlefile", pdf, png[: -len(".png")]],
        check=True,
    )
    assert not WIDE_TITLES, f"{name}: titles wider than their panel: {WIDE_TITLES}"
    return {"name": f"{name}_{stamp}", "png": png, "svg": svg, "pdf": pdf}


def demo(panels: dict[str, Any], name: str, height_mm: float = 60.0) -> dict[str, str]:
    """Draw every panel of a module on its own half-width sheet row, for checking."""
    R = Reads()
    keys = list(panels)
    n = len(keys)
    fig = new_sheet(height_mm * ((n + 1) // 2))
    axes = rows(fig, (n + 1) // 2, 2)
    statuses = []
    for ax, k in zip(axes, keys, strict=False):
        statuses.append(panels[k](ax, R))
    for ax in axes[n:]:
        ax.axis("off")
    letters_status(fig, axes[:n], statuses)
    status_key(fig)
    return save(fig, name)
