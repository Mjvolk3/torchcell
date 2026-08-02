# experiments/database/scripts/plot_supported_datasets_signal.py
# [[experiments.database.scripts.plot_supported_datasets_signal]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/plot_supported_datasets_signal
"""VIEW off the raw-data JSON: scatter of Instances (dataset length) vs Signal
(gzip bits), one point per built dataset, labeled and colored by phenotype
category. Reads only the JSON produced by build_supported_datasets_table.py.

The JSON stores the gzip size in BYTES; this view -- like the table view -- reports
BITS, so the two artifacts of the report carry the same unit.

Repo figure standard (CLAUDE.md "Figure & Plotting Standards"): the ordered,
green-free ``PLOT_PALETTE`` taken in category order, a strict panel width from
``PANEL_WIDTHS_MM``, Arial 6 pt with ``svg.fonttype: none``, a fully boxed axis, and
a true-size SVG saved next to the PNG. Eight phenotype categories take the first
eight palette entries, so the two *dark* repeats (bronze in the orange slot, maroon
in the red slot) land next to their primaries; per the standard we disambiguate with
SHAPE rather than by inventing colors, so each category also gets its own marker.

Run from the repo root (defaults to the newest pre-build snapshot):
  python experiments/database/scripts/plot_supported_datasets_signal.py
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from adjustText import adjust_text  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from torchcell.paper.tables import DatasetSignalRecord  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
BITS_PER_BYTE = 8

INK = "#000000"
GRID = "#4A4A4A"
PANEL_W_MM = PANEL_WIDTHS_MM["full"]  # 179 mm -- 49 labeled points need the width
PANEL_H_MM = 120.0  # loose (<= MAX_HEIGHT_MM); five decades of y read comfortably
# One marker per category, so the dark repeats (palette 7/8) never rest on color
# alone. Index order matches the JSON's ``sections`` order.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]


def newest_json() -> Path:
    """The most recent pre-build snapshot's JSON."""
    cands = sorted(RESULTS.glob("pre-build/*/supported_datasets.json"))
    if not cands:
        raise FileNotFoundError("No pre-build snapshot; run build_supported_datasets_table.py first")
    return cands[-1]


def _apply_rc() -> None:
    """Repo type + rule standards (Arial 6 pt, editable SVG text, 0.5 pt lines)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Liberation Sans is the metric-compatible Arial substitute on Linux.
            "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.titlesize": 7,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            "savefig.bbox": None,  # torchcell.mplstyle sets 'tight'; would break true-size
        }
    )


def _box(ax) -> None:
    """Full black border on all four spines + a light grid behind the marks."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=INK, width=0.5, length=2)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.25, color=GRID)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.12, color=GRID)
    ax.set_axisbelow(True)


def main() -> None:
    """Draw the instances-vs-signal scatter and save PNG + true-size SVG."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=None, help="supported_datasets.json (default: newest)")
    args = ap.parse_args()
    load_dotenv()

    data_path = args.data or newest_json()
    payload = json.loads(data_path.read_text())
    records = [DatasetSignalRecord(**d) for d in payload["datasets"]
               if d["built"] and d["signal_bytes"] > 0]

    _apply_rc()
    sections = payload["sections"]
    color = {s: PLOT_PALETTE[i % len(PLOT_PALETTE)] for i, s in enumerate(sections)}
    marker = {s: MARKERS[i % len(MARKERS)] for i, s in enumerate(sections)}

    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_W_MM), mm_to_in(PANEL_H_MM)))
    for s in sections:
        pts = [r for r in records if r.section == s]
        if not pts:
            continue
        ax.scatter([r.instances for r in pts],
                   [r.signal_bytes * BITS_PER_BYTE for r in pts],
                   s=18, color=color[s], marker=marker[s], edgecolor=INK,
                   linewidth=0.3, label=s, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Instances (dataset length)")
    ax.set_ylabel("Signal (gzip, bits)")
    ax.set_title(f"Training signal vs scale — {payload['status']} ({payload['date']})",
                 loc="left", color=INK)
    # Pad the limits by a fraction of a decade so the 6 pt labels have somewhere to
    # go: adjustText moves text but cannot push it outside the axes, and a label that
    # runs off the panel edge is the failure mode on a fixed 179 mm width.
    xs = [r.instances for r in records]
    ys = [r.signal_bytes * BITS_PER_BYTE for r in records]
    ax.set_xlim(min(xs) / 4, max(xs) * 20)  # extra room right: the longest labels sit there
    ax.set_ylim(min(ys) / 6, max(ys) * 4)
    _box(ax)
    # Legend INSIDE the axes (the points ride the diagonal, so the lower right is
    # empty), which keeps the panel at exactly 179 mm.
    leg = ax.legend(title="Phenotype category", loc="lower right", framealpha=0.95,
                    edgecolor=INK, handletextpad=0.4, borderpad=0.4, labelspacing=0.35)
    leg.get_frame().set_linewidth(0.5)
    leg.get_title().set_fontsize(6)

    # De-collide the 6 pt labels with thin leader lines.
    texts = [ax.text(r.instances, r.signal_bytes * BITS_PER_BYTE, r.name,
                     color=INK, zorder=4) for r in records]
    adjust_text(texts, ax=ax, expand=(1.15, 1.4),
                arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.3})

    out_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], "database")
    os.makedirs(out_dir, exist_ok=True)
    stem = osp.join(out_dir, "supported-datasets-instances-vs-signal")
    fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
    savefig_true_size_svg(fig, f"{stem}.svg", facecolor="white")
    plt.close(fig)
    print(f"Wrote {stem}.png + .svg  ({len(records)} datasets)")


if __name__ == "__main__":
    main()
