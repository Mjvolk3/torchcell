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
assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set in the environment"

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)

# --- Replicate Pearson r values (Trigenic tau) for each model ---
# Every model is now grounded in 3 REAL replicate Pearson values (except the
# deterministic FBA model, which has a single value by nature). All error bars
# are the SAME statistic: the standard error of the mean (SEM = sample-SD/sqrt(n),
# ddof=1). This resolves the WS14 error-bar-provenance blocker so the four whiskers
# are apples-to-apples.
#
# TorchCell / CGT: 3 replicate runs tagged `inf_1` on wandb, Pearson r read from
#   the wandb scatter plots. Source of truth:
#   [[experiments.010-kuzmin-tmi.performance-diff-010-009]] (010 "All" models).
#   (Historical note: the abstract's earlier "0.454 +/- 0.006" reported the
#   POPULATION SD, np.std(ddof=0), of these same three values -- NOT a SEM. The
#   +/- 0.006 is replaced here by the SEM = 0.004, matching DANGO/DCell.)
torchcell_vals = np.array([0.462, 0.452, 0.447])
# DANGO (repro best): 3 replicate Pearson values.
dango_vals = np.array([0.36759, 0.36708, 0.36637])
# DCell: 3 replicate Pearson values.
dcell_vals = np.array([0.17321017384529114, 0.1550033837556839, 0.14192065596580505])
# Yeast9 (GEM/FBA): deterministic modeling -> single value, no error by nature.
gem_vals = np.array([0.0006])


def sem(a: np.ndarray) -> float:
    """Standard error of the mean; 0 for a single deterministic value."""
    if len(a) < 2:
        return 0.0
    return float(a.std(ddof=1) / np.sqrt(len(a)))


models = ["Yeast9", "DCell", "DANGO", "TorchCell"]
means = [
    gem_vals.mean(),
    dcell_vals.mean(),
    dango_vals.mean(),
    torchcell_vals.mean(),
]
sems = [
    sem(gem_vals),  # 0.0 (deterministic, single value)
    sem(dcell_vals),  # SEM
    sem(dango_vals),  # SEM
    sem(torchcell_vals),  # SEM (from real replicates; was pop-SD 0.006)
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
for bar, mean, sem_val in zip(bars, means, sems):
    label_y = mean + sem_val + 0.012
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

# Report the numbers driving the figure (all SEM, all apples-to-apples).
for model, mean, sem_val in zip(models, means, sems):
    print(f"{model:>10}: r = {mean:.4f} +/- {sem_val:.4f} (SEM)")

out_dir = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
os.makedirs(out_dir, exist_ok=True)
base = osp.join(out_dir, f"trigenic_tau_model_comparison_{timestamp()}")
plt.savefig(f"{base}.png", dpi=300)
plt.savefig(f"{base}.svg")
print(f"Saved: {base}.png")
print(f"Saved: {base}.svg")
plt.close()
