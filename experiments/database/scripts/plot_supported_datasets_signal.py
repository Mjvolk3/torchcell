# experiments/database/scripts/plot_supported_datasets_signal.py
# [[experiments.database.scripts.plot_supported_datasets_signal]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/plot_supported_datasets_signal
"""VIEW off the raw-data JSON: scatter of Instances (dataset length) vs Signal
(gzip bytes), one point per built dataset, labeled and colored by phenotype
category. Reads only the JSON produced by build_supported_datasets_table.py.

Follows the project plot guidelines: torchcell.mplstyle color palette, output to
$ASSET_IMAGES_DIR with a timestamp. Point labels use Liberation Sans size 6 (the
metric-compatible Arial substitute available on Linux).

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
from torchcell.timestamp import timestamp  # noqa: E402

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
MPLSTYLE = REPO / "torchcell" / "torchcell.mplstyle"
LABEL_FONT = "Liberation Sans"  # Arial-metric substitute (Arial is unavailable on Linux)


def newest_json() -> Path:
    """The most recent pre-build snapshot's JSON."""
    cands = sorted(RESULTS.glob("pre-build/*/supported_datasets.json"))
    if not cands:
        raise FileNotFoundError("No pre-build snapshot; run build_supported_datasets_table.py first")
    return cands[-1]


def main() -> None:
    """Draw the instances-vs-signal scatter and save it to ASSET_IMAGES_DIR."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=None, help="supported_datasets.json (default: newest)")
    args = ap.parse_args()
    load_dotenv()

    data_path = args.data or newest_json()
    payload = json.loads(data_path.read_text())
    records = [DatasetSignalRecord(**d) for d in payload["datasets"] if d["built"]]

    plt.style.use(str(MPLSTYLE))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sections = payload["sections"]
    color = {s: palette[i % len(palette)] for i, s in enumerate(sections)}

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for s in sections:
        pts = [r for r in records if r.section == s]
        if not pts:
            continue
        ax.scatter([r.instances for r in pts], [r.signal_bytes for r in pts],
                   s=44, color=color[s], edgecolor="white", linewidth=0.6, label=s, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Instances (dataset length)")
    ax.set_ylabel("Signal (gzip, bytes)")
    ax.set_title(f"Training signal vs scale — {payload['status']} ({payload['date']})")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    # Legend outside the axes so it never covers points.
    ax.legend(fontsize=7, title="Phenotype category", title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.95, borderaxespad=0)

    # De-collide the size-6 labels with thin leader lines.
    texts = [ax.text(r.instances, r.signal_bytes, r.name, fontsize=6,
                     fontfamily=LABEL_FONT, color="0.12", zorder=4) for r in records]
    adjust_text(texts, ax=ax, expand=(1.15, 1.4),
                arrowprops={"arrowstyle": "-", "color": "0.55", "linewidth": 0.4})
    fig.tight_layout()

    out_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], "database")
    os.makedirs(out_dir, exist_ok=True)
    out = osp.join(out_dir, f"supported-datasets-instances-vs-signal_{timestamp()}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}  ({len(records)} datasets)")


if __name__ == "__main__":
    main()
