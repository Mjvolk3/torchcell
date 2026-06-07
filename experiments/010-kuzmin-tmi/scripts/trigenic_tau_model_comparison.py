# experiments/010-kuzmin-tmi/scripts/trigenic_tau_model_comparison.py
# [[experiments.010-kuzmin-tmi.scripts.trigenic_tau_model_comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/trigenic_tau_model_comparison

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

import torchcell
from torchcell.timestamp import timestamp

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)

# --- Data from [[experiments.010-kuzmin-tmi.scripts.trigenic_tau_model_comparison]] ---
# TorchCell (Ours): reported as mean ± SE across 3 replicates
torchcell_vals = np.array([0.454 + 0.006, 0.454, 0.454 - 0.006])
# Dango Repro Best: 3 replicates
dango_vals = np.array([0.36759, 0.36708, 0.36637])
# DCell: 3 replicates
dcell_vals = np.array([0.17321017384529114, 0.1550033837556839, 0.14192065596580505])
# GEM: deterministic (single value)
gem_vals = np.array([0.0006])

models = ["Yeast9", "DCell", "DANGO", "TorchCell"]
means = [
    gem_vals.mean(),
    dcell_vals.mean(),
    dango_vals.mean(),
    torchcell_vals.mean(),
]
sems = [
    0.0,  # deterministic
    dcell_vals.std(ddof=1) / np.sqrt(len(dcell_vals)),
    dango_vals.std(ddof=1) / np.sqrt(len(dango_vals)),
    0.006,  # reported SE
]

# --- Colors from draw.io diagram (fill / outline) ---
# Yeast9=yellow, DCell=purple, DANGO=orange, TorchCell=red
fill_colors = ["#FFF2CC", "#E1D5E7", "#FFE6CC", "#F8CECC"]
edge_colors = ["#D6B656", "#9673A6", "#D79B00", "#B85450"]

fig, ax = plt.subplots(figsize=(5, 5))

ax.yaxis.grid(True, color="#D0D0D0", linewidth=1.0, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

bars = ax.bar(
    models,
    means,
    yerr=sems,
    capsize=5,
    color=fill_colors,
    edgecolor=edge_colors,
    linewidth=2.5,
    width=0.6,
    error_kw={"linewidth": 1.5, "capthick": 1.5, "color": "#333333"},
    zorder=3,
)

ax.set_ylabel(r"Pearson $r$ (Trigenic $\boldsymbol{\tau}$)")
ax.set_ylim(0, 0.55)
ax.set_yticks(np.arange(0, 0.55, 0.1))

# Add value labels above bars
for bar, mean, sem in zip(bars, means, sems):
    label_y = mean + sem + 0.012
    if mean < 0.01:
        label = f"{mean:.4f}"
    else:
        label = f"{mean:.3f}"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        label_y,
        label,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_edgecolor("black")

plt.tight_layout()

save_path = osp.join(
    ASSET_IMAGES_DIR, f"010-kuzmin-tmi/trigenic_tau_model_comparison_{timestamp()}.png"
)
os.makedirs(osp.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
print(f"Saved: {save_path}")
plt.close()
