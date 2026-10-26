# experiments/database/scripts/plot_organism_signal_controls.py
# [[experiments.database.scripts.plot_organism_signal_controls]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/plot_organism_signal_controls
"""Signal (gzip) and its five controls, summed over every probed dataset of each
organism, as grouped bars: (a) total bits per organism, log axis; (b) each control
total as a ratio to that organism's Signal total.

Reads only ``dataset_probe_cache.json``, the per-dataset rows that
build_bacteria_datasets_table.py measured over every LMDB the adapter map serves, so
yeast, E. coli and P. putida are summed from one consistent probe variant. Also writes
the totals as CSV and prints them as a Markdown table for the note.

Repo figure standard: ``PLOT_PALETTE`` in order, a strict ``PANEL_WIDTHS_MM`` width,
Arial 6 pt with ``svg.fonttype: none``, boxed axes, true-size SVG beside the PNG.

Run from the repo root:
  python experiments/database/scripts/plot_organism_signal_controls.py
"""

from __future__ import annotations

import argparse
import csv
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from build_bacteria_datasets_table import (  # noqa: E402
    CACHE_PATH,
    PROBE_VARIANT,
    RESULTS,
    ProbeCache,
)
from dotenv import load_dotenv  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

BITS_PER_BYTE = 8
PANEL_W_MM = PANEL_WIDTHS_MM["full"]
PANEL_H_MM = 70.0
ORGANISMS = [
    ("scerevisiae", "S. cerevisiae"),
    ("ecoli", "E. coli"),
    ("pputida", "P. putida"),
]
# Signal first, then the controls in the order the tables list them.
SERIES = [
    ("signal_bytes", "Signal"),
    ("structure_bytes", "structure only"),
    ("shuffled_bytes", "shuffled within path"),
    ("noise_bytes", "digit noise"),
    ("binary_bytes", "1-bit in place"),
    ("binary_packed_bytes", "1-bit packed"),
]


class OrganismTotals(BaseModel):
    """Every probed dataset of one organism, summed."""

    organism: str
    label: str
    datasets: int
    instances: int
    bytes_by_series: dict[str, int]

    def bits(self, field: str) -> int:
        """Total of one series in bits."""
        return self.bytes_by_series[field] * BITS_PER_BYTE

    def ratio(self, field: str) -> float:
        """Total of one series over the Signal total."""
        return self.bytes_by_series[field] / self.bytes_by_series["signal_bytes"]


def organism_totals(cache: ProbeCache) -> list[OrganismTotals]:
    """Sum the cache rows per organism; every row must be probed under PROBE_VARIANT."""
    stale = [k for k, e in cache.entries.items() if e.variant != PROBE_VARIANT]
    if stale:
        raise ValueError(f"cache rows from another probe variant: {stale}")
    unprobed = [k for k, e in cache.entries.items() if not e.record.probed]
    if unprobed:
        raise ValueError(f"unprobed cache rows would undercount the sums: {unprobed}")
    known = {o for o, _ in ORGANISMS}
    unknown = {e.record.organism for e in cache.entries.values()} - known
    if unknown:
        raise ValueError(f"cache holds organisms this figure does not draw: {unknown}")
    out = []
    for org, label in ORGANISMS:
        recs = [e.record for e in cache.entries.values() if e.record.organism == org]
        out.append(
            OrganismTotals(
                organism=org,
                label=label,
                datasets=len(recs),
                instances=sum(r.instances for r in recs),
                bytes_by_series={
                    f: sum(getattr(r, f) for r in recs) for f, _ in SERIES
                },
            )
        )
    return out


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


def _grouped_bars(
    ax: Axes, totals: list[OrganismTotals], values: list[list[float]]
) -> np.ndarray:
    """One group per organism, one bar per series; returns the group centers."""
    x = np.arange(len(totals))
    width = 0.8 / len(SERIES)
    for i in range(len(SERIES)):
        ax.bar(
            x + (i - (len(SERIES) - 1) / 2) * width,
            [v[i] for v in values],
            width=width,
            color=PLOT_PALETTE[i],
            edgecolor="#000000",
            linewidth=0.4,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{italic(t.label)}\n{t.datasets} datasets\n"
            f"{t.instances / 1e6:.2f} M instances"
            for t in totals
        ]
    )
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="major", color="#4A4A4A", linewidth=0.3, alpha=0.4)
    return x


def write_table(totals: list[OrganismTotals]) -> str:
    """Write the totals CSV under results/ and return the Markdown table."""
    path = RESULTS / "organism_signal_controls.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["organism", "datasets", "instances"]
            + [f"{f.removesuffix('_bytes')}_bits" for f, _ in SERIES]
            + [f"{f.removesuffix('_bytes')}_over_signal" for f, _ in SERIES[1:]]
        )
        for t in totals:
            w.writerow(
                [t.organism, t.datasets, t.instances]
                + [t.bits(f) for f, _ in SERIES]
                + [f"{t.ratio(f):.4f}" for f, _ in SERIES[1:]]
            )
    print(f"Wrote {path}")
    head = "| Series | " + " | ".join(t.label for t in totals) + " |"
    rule = "| :-- |" + " --: |" * len(totals)
    rows = [
        "| Datasets | " + " | ".join(f"{t.datasets}" for t in totals) + " |",
        "| Instances | " + " | ".join(f"{t.instances:,}" for t in totals) + " |",
    ]
    for f, name in SERIES:
        rows.append(
            f"| {name}, total bits | "
            + " | ".join(f"{t.bits(f):.2e}" for t in totals)
            + " |"
        )
    for f, name in SERIES[1:]:
        rows.append(
            f"| {name} / Signal | "
            + " | ".join(f"{t.ratio(f):.3f}" for t in totals)
            + " |"
        )
    return "\n".join([head, rule, *rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-timestamp", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    out_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], "database")
    os.makedirs(out_dir, exist_ok=True)
    totals = organism_totals(ProbeCache.load(CACHE_PATH))
    print(write_table(totals))

    _apply_rc()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_W_MM), mm_to_in(PANEL_H_MM))
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.27, top=0.95, wspace=0.18)

    _grouped_bars(ax_a, totals, [[t.bits(f) for f, _ in SERIES] for t in totals])
    ax_a.set_ylabel("Total over datasets (gzip, bits)")
    ax_a.set_title("a", loc="left", fontsize=8, fontweight="bold")

    _grouped_bars(ax_b, totals, [[t.ratio(f) for f, _ in SERIES] for t in totals])
    ax_b.axhline(1.0, color="#000000", linewidth=0.5, linestyle="--", zorder=4)
    ax_b.set_ylabel("Total / Signal total")
    ax_b.set_title("b", loc="left", fontsize=8, fontweight="bold")

    fig.legend(
        handles=[
            Patch(
                facecolor=PLOT_PALETTE[i], edgecolor="#000000", linewidth=0.4, label=n
            )
            for i, (_, n) in enumerate(SERIES)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(SERIES),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    _box(ax_a)
    _box(ax_b)

    stem = "organism_signal_controls" + ("" if args.no_timestamp else f"_{timestamp()}")
    png = osp.join(out_dir, stem + ".png")
    svg = osp.join(out_dir, stem + ".svg")
    pdf = osp.join(out_dir, stem + ".pdf")
    fig.savefig(png, dpi=300)
    savefig_true_size_svg(fig, svg)
    fig.savefig(pdf)
    print(f"Wrote {png}\nWrote {svg}\nWrote {pdf}")


if __name__ == "__main__":
    main()
