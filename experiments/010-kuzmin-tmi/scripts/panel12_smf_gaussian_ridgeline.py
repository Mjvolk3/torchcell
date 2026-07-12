# experiments/010-kuzmin-tmi/scripts/panel12_smf_gaussian_ridgeline.py
# [[experiments.010-kuzmin-tmi.scripts.panel12_smf_gaussian_ridgeline]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/panel12_smf_gaussian_ridgeline
"""Ridgeline plot of the 12-gene panel Costanzo2016 single-mutant-fitness Gaussians.

Each gene's SMF is drawn as a Gaussian N(mean, std) on a SHARED fitness axis,
stacked and ordered by mean fitness so both the center (mu) and the width (sigma)
are directly comparable gene-to-gene. Common names (SGD R64) annotate each ridge.
"""
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.timestamp import timestamp

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

# Common names from SGD R64-4-1 GFF; "" where no standard name exists.
COMMON = {
    "YBR203W": "COS111", "YDR057W": "YOS9", "YER079W": "", "YGL087C": "MMS2",
    "YIL174W": "", "YJR060W": "CBF1", "YKL033W-A": "", "YLL012W": "YEH1",
    "YLR104W": "LCL2", "YLR312C-B": "SPH1", "YPL046C": "ELC1", "YPL081W": "RPS9A",
}

src = osp.join(
    EXPERIMENT_ROOT, "010-kuzmin-tmi", "results", "inference_3",
    "singles_table_panel12_k200_queried.csv",
)
df = pd.read_csv(src)[["gene", "SmfCostanzo2016_fitness", "SmfCostanzo2016_std"]]
df = df.rename(columns={"SmfCostanzo2016_fitness": "mu", "SmfCostanzo2016_std": "sigma"})
df = df.sort_values("mu").reset_index(drop=True)  # ascending -> lowest fitness at bottom

x = np.linspace(0.2, 1.4, 800)
overlap = 1.8          # vertical exaggeration: how far a ridge may climb into the one above
row_h = 1.0
mean_sigma = df["sigma"].mean()

fig, ax = plt.subplots(figsize=(9, 8))
cmap = plt.cm.viridis
for i, r in df.iterrows():
    y = np.exp(-0.5 * ((x - r.mu) / r.sigma) ** 2) / (r.sigma * np.sqrt(2 * np.pi))
    y = y / y.max() * overlap * row_h        # normalize peak height so widths are comparable
    base = i * row_h
    color = cmap(0.15 + 0.7 * (r.sigma - df.sigma.min()) / (df.sigma.max() - df.sigma.min()))
    ax.fill_between(x, base, base + y, color=color, alpha=0.85, zorder=df.shape[0] - i)
    ax.plot(x, base + y, color="black", lw=0.8, zorder=df.shape[0] - i)
    label = f"{r.gene}" + (f" / {COMMON[r.gene]}" if COMMON[r.gene] else "")
    ax.text(0.21, base + 0.05, label, va="bottom", ha="left", fontsize=9, zorder=100)
    ax.text(0.21, base + 0.05, "", fontsize=9)
    # per-gene mu +/- sigma annotation at right
    ax.text(1.39, base + 0.05, rf"${r.mu:.3f}\pm{r.sigma:.3f}$",
            va="bottom", ha="right", fontsize=8, color="0.25", zorder=100)

ax.axvline(1.0, color="0.4", ls=":", lw=1.2, zorder=0)   # wild-type fitness
ax.text(1.0, df.shape[0] * row_h + 0.1, "WT = 1.0", ha="center", fontsize=9, color="0.4")
ax.set_yticks([])
ax.set_xlabel("Single-mutant fitness (Costanzo2016)")
ax.set_xlim(0.2, 1.4)
ax.set_title(
    "Panel-12 SMF Gaussians (Costanzo2016), ordered by mean fitness\n"
    rf"colored by $\sigma$; mean $\sigma$ = {mean_sigma:.3f}",
    fontsize=12,
)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.tight_layout()

out = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi",
               f"panel12_smf_gaussian_ridgeline_{timestamp()}.png")
os.makedirs(osp.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved: {out}")
