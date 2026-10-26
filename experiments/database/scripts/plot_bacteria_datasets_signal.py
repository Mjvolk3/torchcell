# experiments/database/scripts/plot_bacteria_datasets_signal.py
# [[experiments.database.scripts.plot_bacteria_datasets_signal]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/plot_bacteria_datasets_signal
"""VIEW off the bacteria JSON: (a) Instances vs Signal, one point per dataset, colored
by section with the host as marker shape; (b) the Signal controls as ratios to Signal
against Signal, one series per control. Reads only the JSON written by
build_bacteria_datasets_table.py.

Repo figure standard: ``PLOT_PALETTE`` in order, a strict ``PANEL_WIDTHS_MM`` width,
Arial 6 pt with ``svg.fonttype: none``, boxed axes, true-size SVG beside the PNG.

Run from the repo root (defaults to the newest pre-build snapshot):
  python experiments/database/scripts/plot_bacteria_datasets_signal.py
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
from build_bacteria_datasets_table import DatasetProbeRecord  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
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
PANEL_W_MM = PANEL_WIDTHS_MM["full"]
PANEL_H_MM = 105.0
HOST_MARKER = {"ecoli": "o", "pputida": "s", "scerevisiae": "^"}
HOST_LABEL = {
    "ecoli": "E. coli",
    "pputida": "P. putida",
    "scerevisiae": "S. cerevisiae",
}
CONTROLS = [
    ("structure_bytes", "structure only"),
    ("shuffled_bytes", "shuffled within path"),
    ("noise_bytes", "digit noise"),
    ("binary_bytes", "1-bit in place"),
    ("binary_packed_bytes", "1-bit packed"),
]


def _apply_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            # Organism names are set in Arial italic through mathtext.
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "savefig.bbox": None,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )


def italic(name: str) -> str:
    """An organism name in Arial italic (mathtext; spaces escaped)."""
    return r"$\mathit{" + name.replace(" ", r"\ ") + "}$"


def _box(ax: Axes) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("#000000")


def newest_json() -> Path:
    cands = sorted(RESULTS.glob("pre-build/*/ecoli-pputida_datasets.json"))
    if not cands:
        raise FileNotFoundError(
            "No pre-build snapshot; run build_bacteria_datasets_table.py first"
        )
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--no-timestamp", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    out_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], "database")
    os.makedirs(out_dir, exist_ok=True)
    payload = json.loads((args.data or newest_json()).read_text())
    recs = [DatasetProbeRecord(**d) for d in payload["datasets"]]
    recs = [r for r in recs if r.probed]
    sections = payload["sections"]
    color = {s: PLOT_PALETTE[i % len(PLOT_PALETTE)] for i, s in enumerate(sections)}

    _apply_rc()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_W_MM), mm_to_in(PANEL_H_MM))
    )
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.27, top=0.97, wspace=0.22)

    for r in recs:
        ax_a.scatter(
            r.instances,
            r.signal_bytes * BITS_PER_BYTE,
            s=9,
            marker=HOST_MARKER[r.organism],
            facecolor=color[r.section],
            edgecolor="#000000",
            linewidth=0.3,
            zorder=3,
        )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Instances (records)")
    ax_a.set_ylabel("Signal (gzip, bits)")
    ax_a.grid(True, which="major", color="#4A4A4A", linewidth=0.3, alpha=0.4)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=color[s],
            markeredgecolor="#000000",
            markeredgewidth=0.3,
            markersize=3.5,
            label=s,
        )
        for s in sections
    ]
    handles += [
        Line2D(
            [],
            [],
            marker=HOST_MARKER[h],
            linestyle="",
            markerfacecolor="#FFFFFF",
            markeredgecolor="#000000",
            markeredgewidth=0.5,
            markersize=3.5,
            label=italic(HOST_LABEL[h]),
        )
        for h in payload["organisms"]
    ]
    ax_a.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.09, -0.17),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        labelspacing=0.25,
        columnspacing=0.8,
    )
    ax_a.set_title("a", loc="left", fontsize=8, fontweight="bold")

    for i, (field, label) in enumerate(CONTROLS):
        c = PLOT_PALETTE[i]
        for r in recs:
            ax_b.scatter(
                r.signal_bytes * BITS_PER_BYTE,
                getattr(r, field) / r.signal_bytes,
                s=7,
                marker=HOST_MARKER[r.organism],
                facecolor=c,
                edgecolor="#000000",
                linewidth=0.3,
                zorder=3,
            )
    ax_b.axhline(1.0, color="#000000", linewidth=0.5, linestyle="--", zorder=2)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Signal (gzip, bits)")
    ax_b.set_ylabel("Control / Signal")
    ax_b.grid(True, which="major", color="#4A4A4A", linewidth=0.3, alpha=0.4)
    ax_b.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor=PLOT_PALETTE[i],
                markeredgecolor="#000000",
                markeredgewidth=0.3,
                markersize=3.5,
                label=label,
            )
            for i, (_, label) in enumerate(CONTROLS)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        labelspacing=0.25,
        columnspacing=0.8,
    )
    ax_b.set_title("b", loc="left", fontsize=8, fontweight="bold")
    _box(ax_a)
    _box(ax_b)

    label = "-".join(payload["organisms"])
    name = {"ecoli-pputida": "bacteria", "scerevisiae": "yeast"}.get(label, label)
    stem = f"{name}_datasets_signal" + ("" if args.no_timestamp else f"_{timestamp()}")
    png = osp.join(out_dir, stem + ".png")
    svg = osp.join(out_dir, stem + ".svg")
    pdf = osp.join(out_dir, stem + ".pdf")
    fig.savefig(png, dpi=300)
    savefig_true_size_svg(fig, svg)
    fig.savefig(pdf)
    print(f"Wrote {png}\nWrote {svg}\nWrote {pdf}")


if __name__ == "__main__":
    main()
