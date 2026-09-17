# experiments/028-gene-essentiality/scripts/grad_flow_readout.py
# [[experiments.028-gene-essentiality.scripts.grad_flow_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/grad_flow_readout

r"""Where the gradient goes on the module arms, across rounds.

Every cell logs a per-module gradient norm each epoch (preprocessor, transformer,
essentiality head, flux layer, the readout on top of the layer, the gate). The round
figures plot them per round; this one reads them across the module arms of rounds 2 to
13 side by side, which is the diagnosis the rounds share:

- **a**, the flux layer's share of the summed per-module gradient norm per epoch, seed
  mean. A forced arm's whole gradient passes through the layer; a gated arm's passes
  mostly through the MLP head; a warm-started arm with alpha frozen passes almost none,
  because the flux is 99 percent a fixed reference.
- **b**, the preprocessor's gradient norm per epoch, seed mean: whether the backbone is
  trained at all through the path the arm leaves it.
- **c**, the gate's value per epoch on the gated arms, seed mean.

    python experiments/028-gene-essentiality/scripts/grad_flow_readout.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from wave12_readout import ARM_LABELS, arm_of  # noqa: E402

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "028-gene-essentiality")

#: (glob, arm) in figure order: the pooled reference, the forced arms, the gated arms.
ARMS: list[tuple[str, str]] = [
    ("w14_b32posw200_h12l1p1_e200_s*.json", "b32posw200"),
    ("w15_bio_s0_h12l1p1_e200_s*.json", "bio_s0"),
    ("w17_biofba_h12l1p1_e200_s*.json", "biofba"),
    ("w18_biobox_h12l1p1_e200_s*.json", "biobox"),
    ("w21_warmbio_h12l1p1_e200_s*.json", "warmbio"),
    ("w24_warmoffbio_h12l1p1_e200_s*.json", "warmoffbio"),
    ("w15_gate_s0_h12l1p1_e200_s*.json", "gate_s0"),
    ("w17_gatefba_h12l1p1_e200_s*.json", "gatefba"),
    ("w21_warmgate_h12l1p1_e200_s*.json", "warmgate"),
    ("w21_warmgatealpha_h12l1p1_e200_s*.json", "warmgatealpha"),
    ("w24_warmoffgate_h12l1p1_e200_s*.json", "warmoffgate"),
]
MODULES = (
    "grad_norm_embedding_preprocessor_mean",
    "grad_norm_transformer_layers_mean",
    "grad_norm_essentiality_head_mean",
    "grad_norm_flux_layer_mean",
    "grad_norm_readout_mean",
    "grad_norm_gate_mean",
    "grad_norm_fba_readout_mean",
)
HUES = [PLOT_PALETTE[i] for i in (0, 1, 2, 4, 5, 3)]


def series(history: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) if r.get(key) is not None else np.nan for r in history])


def seed_mean(rows: list[np.ndarray]) -> np.ndarray:
    n = min(len(r) for r in rows)
    return np.nanmean(np.stack([r[:n] for r in rows]), axis=0)


def main() -> None:
    apply_paper_style()
    arms: list[dict] = []
    for pattern, arm in ARMS:
        files = sorted(glob.glob(osp.join(RESULTS, pattern)))
        files = [f for f in files if arm_of(osp.basename(f)[:-5]) == arm]
        if not files:
            continue
        share, pre, gate = [], [], []
        for f in files:
            with open(f) as fh:
                h = json.load(fh)["history"]
            parts = np.stack([series(h, k) for k in MODULES if k in h[-1]])
            total = np.nansum(parts, axis=0)
            flux = series(h, "grad_norm_flux_layer_mean")
            share.append(flux / total)
            pre.append(series(h, "grad_norm_embedding_preprocessor_mean"))
            gate.append(series(h, "ess_gate"))
        arms.append(
            {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "n": len(files),
                "share": seed_mean(share),
                "preprocessor": seed_mean(pre),
                "gate": seed_mean(gate),
            }
        )

    width = mm_to_in(PANEL_WIDTHS_MM["full"])
    fig, axes = plt.subplots(1, 3, figsize=(width, mm_to_in(62)))
    ax_a, ax_b, ax_c = axes
    for i, a in enumerate(arms):
        color = HUES[i % len(HUES)]
        # Solid for the pooled reference and the forced arms, dashed for the gated arms.
        style = "--" if "gate" in a["arm"] else "-"
        x = np.arange(len(a["preprocessor"]))
        if np.isfinite(a["share"]).any():
            ax_a.plot(x, a["share"], style, color=color, lw=0.9, label=a["label"])
        ax_b.plot(x, a["preprocessor"], style, color=color, lw=0.9, label=a["label"])
        if np.isfinite(a["gate"]).any():
            ax_c.plot(x, a["gate"], style, color=color, lw=0.9, label=a["label"])
    ax_a.set_yscale("log")
    ax_a.set_ylabel("flux layer share of gradient norm")
    ax_b.set_yscale("log")
    ax_b.set_ylabel("preprocessor gradient norm")
    ax_c.set_ylabel("gate value")
    ax_c.axhline(0, color="0.5", lw=0.5)
    for ax, letter in zip(axes, "abc", strict=True):
        ax.set_xlabel("epoch")
        ax.grid(axis="y", which="major", linewidth=0.3, color="0.85")
        ax.text(-0.28, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
    handles, labels = ax_b.get_legend_handles_labels()
    ncol = 2
    rows = int(np.ceil(len(labels) / ncol))
    fig.subplots_adjust(
        left=0.08, right=0.99, bottom=0.17, top=1.0 - (rows * 3.2 + 5.0) / 62, wspace=0.45
    )
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.08, 1.0),
        ncol=ncol,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=1.2,
    )
    os.makedirs(IMAGE_DIR, exist_ok=True)
    base = osp.join(IMAGE_DIR, f"grad_flow_{timestamp()}")
    fig.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(fig, f"{base}.svg")
    out = osp.join(RESULTS, "grad_flow_readout.json")
    with open(out, "w") as fh:
        json.dump(
            {
                "figure": f"{base}.svg",
                "arms": [
                    {
                        "arm": a["arm"],
                        "label": a["label"],
                        "n_seeds": a["n"],
                        "flux_share_first_last": [
                            float(np.nan_to_num(a["share"][0])),
                            float(np.nan_to_num(a["share"][-1])),
                        ],
                        "preprocessor_first_last": [
                            float(a["preprocessor"][0]),
                            float(a["preprocessor"][-1]),
                        ],
                        "gate_last": float(np.nan_to_num(a["gate"][-1])),
                    }
                    for a in arms
                ],
            },
            fh,
            indent=2,
        )
    for a in arms:
        print(
            f"{a['arm']:14s} n={a['n']} flux share {np.nan_to_num(a['share'][0]):.3f} -> "
            f"{np.nan_to_num(a['share'][-1]):.3f}  preprocessor {a['preprocessor'][0]:.2g} -> "
            f"{a['preprocessor'][-1]:.2g}  gate {np.nan_to_num(a['gate'][-1]):+.2f}"
        )
    print(f"{base}.svg\n{out}")


if __name__ == "__main__":
    main()
