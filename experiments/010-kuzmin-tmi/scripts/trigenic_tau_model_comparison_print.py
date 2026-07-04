# experiments/010-kuzmin-tmi/scripts/trigenic_tau_model_comparison_print.py
# [[experiments.010-kuzmin-tmi.scripts.trigenic_tau_model_comparison_print]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/trigenic_tau_model_comparison_print
#
# Print-sized version of trigenic_tau_model_comparison.py for the paper figure:
#   - all text 8 pt Arial
#   - figure width = 60 mm (Nature sub-column); height chosen to stay compact
# Data is identical (inline) to the source script -- no model run needed.

import os
import os.path as osp

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from dotenv import load_dotenv

from torchcell.timestamp import timestamp

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

# --- Print styling: 8 pt Arial everywhere ---
MM = 1.0 / 25.4  # mm -> inch
FS_AXIS = 8   # y-axis title
FS_TICK = 7   # tick numbers + x category labels
FS_NUM = 7    # bold value labels above the bars
mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": FS_TICK,
        "axes.labelsize": FS_AXIS,
        "axes.titlesize": FS_AXIS,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
        # Math (italic r, bold tau) rendered in Arial to match the text.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,  # embed real TrueType (editable/selectable text)
        "ps.fonttype": 42,
    }
)

# --- Data (same as trigenic_tau_model_comparison.py) ---
torchcell_vals = np.array([0.454 + 0.006, 0.454, 0.454 - 0.006])
dango_vals = np.array([0.36759, 0.36708, 0.36637])
dcell_vals = np.array([0.17321017384529114, 0.1550033837556839, 0.14192065596580505])
gem_vals = np.array([0.0006])

models = ["Yeast9", "DCell", "DANGO", "CGT"]
means = [gem_vals.mean(), dcell_vals.mean(), dango_vals.mean(), torchcell_vals.mean()]
sems = [
    0.0,
    dcell_vals.std(ddof=1) / np.sqrt(len(dcell_vals)),
    dango_vals.std(ddof=1) / np.sqrt(len(dango_vals)),
    0.006,
]

# --- Colors from draw.io diagram (fill / outline) ---
fill_colors = ["#FFF2CC", "#E1D5E7", "#FFE6CC", "#F8CECC"]
edge_colors = ["#D6B656", "#9673A6", "#D79B00", "#B85450"]

# 55 mm wide (< 60 mm, leaves margin); height scaled to keep the aspect ratio.
WIDTH_MM = 55
HEIGHT_MM = WIDTH_MM * 55 / 60  # preserve the previous 60x55 proportions
fig, ax = plt.subplots(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))

ax.yaxis.grid(True, color="#D0D0D0", linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

bars = ax.bar(
    models,
    means,
    yerr=sems,
    capsize=2.0,
    color=fill_colors,
    edgecolor=edge_colors,
    linewidth=1.0,
    width=0.6,
    error_kw={"linewidth": 0.8, "capthick": 0.8, "color": "#333333"},
    zorder=3,
)

ax.set_ylabel(r"Pearson $r$ (Trigenic $\boldsymbol{\tau}$)")
ax.set_ylim(0, 0.55)
ax.set_yticks(np.arange(0, 0.55, 0.1))

for bar, mean, sem in zip(bars, means, sems):
    label_y = mean + sem + 0.012
    label = f"{mean:.4f}" if mean < 0.01 else f"{mean:.3f}"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        label_y,
        label,
        ha="center",
        va="bottom",
        fontsize=FS_NUM,
        fontweight="bold",
    )

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_edgecolor("black")

ax.tick_params(width=0.8, length=3)

plt.tight_layout(pad=0.3)

os.makedirs(osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi"), exist_ok=True)
stem = osp.join(
    ASSET_IMAGES_DIR,
    f"010-kuzmin-tmi/trigenic_tau_model_comparison_print_60mm_{timestamp()}",
)
fig.savefig(stem + ".pdf")  # vector for the paper
fig.savefig(stem + ".svg")  # vector, imports at true size in draw.io
fig.savefig(stem + ".png", dpi=600)  # high-res raster for quick review


def rescale_svg_for_drawio(path):
    """draw.io reads the SVG root width/height NUMBER as canvas units at
    100 units/inch, but matplotlib writes them in points (72/inch). Multiply
    by 100/72 (keeping viewBox) so the figure imports at its true mm size."""
    import re

    txt = open(path).read()
    m = re.search(r'<svg[^>]*>', txt)
    head = m.group(0)

    def bump(attr, s):
        mm = re.search(rf'{attr}="([0-9.]+)pt"', s)
        val = float(mm.group(1)) * 100.0 / 72.0
        return s.replace(mm.group(0), f'{attr}="{val:.2f}pt"')

    new_head = bump("height", bump("width", head))
    open(path, "w").write(txt.replace(head, new_head))


rescale_svg_for_drawio(stem + ".svg")

# Report the true rendered width in mm as a sanity check.
w_in, h_in = fig.get_size_inches()
print(f"Saved: {stem}.pdf / .png")
print(f"Figure size: {w_in*25.4:.2f} mm x {h_in*25.4:.2f} mm")
plt.close()
