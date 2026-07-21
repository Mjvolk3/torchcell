# experiments/019-echo-crispr-array/scripts/run_sgatools_clone.py
# [[experiments.019-echo-crispr-array.scripts.run_sgatools_clone]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/run_sgatools_clone
"""Demonstrate the SGAtools-style pipeline (torchcell.sga) on the real example
inputs, and produce the SGAtools-style outputs (normalized per-colony table,
per-strain scores, plate heatmaps, histogram, per-strain fitness plot).

The two provided example files are DIFFERENT plates (a 96-well gitter DAT and a
384-well ECHO picklist), so they do not join. This runner therefore does three
honest things:

  A. REAL normalization + heatmap + histogram on the 96-well DAT (no layout ->
     scoring skipped, exactly as SGAtools behaves without an array-layout file).
  B. REAL layout decode of the 384-well picklist -> layout map + figure.
  C. SIMULATED-size scored demo on the REAL 384 picklist layout, to show the
     scoring + per-strain fitness view end to end. Colony sizes here are
     simulated (clearly labelled); replace with a gitter DAT of THIS plate to
     get real scores.

Run from repo root:
    /Users/michaelvolk/miniconda3/bin/python \
        experiments/019-echo-crispr-array/scripts/run_sgatools_clone.py
"""

from __future__ import annotations

import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.sga import (
    NormalizationConfig,
    merge_layout,
    normalize_plate,
    read_echo_picklist,
    read_gitter_dat,
    score_plate,
    score_table,
)
from torchcell.sga.viz import (
    layout_heatmap,
    plate_heatmap,
    strain_fitness_plot,
    value_histogram,
)
from torchcell.timestamp import timestamp

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATA_DIR = osp.join(EXP_DIR, "data")
RESULTS_DIR = osp.join(EXP_DIR, "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

DAT_PATH = osp.join(DATA_DIR, "OD1_5-10nl-2.PNG.dat")
PICKLIST_PATH = osp.join(DATA_DIR, "ECHO_picklist_Plate1_OD1p0_10-25nL.csv")


def save(fig, name: str) -> None:
    path = osp.join(IMG_DIR, f"{name}_{timestamp()}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  wrote {path}")


def part_a_real_normalization() -> None:
    print("\n[A] REAL normalization on 96-well gitter DAT (no layout -> no scoring)")
    dat = read_gitter_dat(DAT_PATH)
    df = normalize_plate(merge_layout(dat, None))
    out = osp.join(RESULTS_DIR, "A_dat_normalized.csv")
    df.to_csv(out, index=False)
    print(
        f"  {len(df)} colonies | {int(df['is_missing'].sum())} missing | "
        f"{int(df['is_flagged'].sum())} flagged | wrote {out}"
    )
    save(
        plate_heatmap(df, "size", "OD1_5-10nl-2  raw colony size", cmap="viridis"),
        "A_heatmap_raw",
    )
    save(
        plate_heatmap(df, "norm", "OD1_5-10nl-2  normalized (1 = plate avg)"),
        "A_heatmap_norm",
    )
    save(value_histogram(df, "norm", "OD1_5-10nl-2  normalized size"), "A_hist_norm")


def part_b_real_layout() -> pd.DataFrame:
    print("\n[B] REAL layout decode of 384-well ECHO picklist")
    layout = read_echo_picklist(PICKLIST_PATH)
    reps = layout.groupby("strain").size().sort_values(ascending=False)
    out = osp.join(RESULTS_DIR, "B_picklist_layout.csv")
    layout.to_csv(out, index=False)
    print(
        f"  {len(layout)} wells | {layout['strain'].nunique()} strains | "
        f"{reps.min()}-{reps.max()} reps each | wrote {out}"
    )
    save(
        layout_heatmap(layout, "Plate1_OD1.0  strain layout (decoded picklist)"),
        "B_layout_map",
    )
    return layout


def part_c_simulated_score(layout: pd.DataFrame) -> None:
    print(
        "\n[C] SIMULATED-size scored demo on the REAL 384 layout "
        "(sizes are simulated; replace with a real DAT of this plate)"
    )
    cfg = NormalizationConfig()
    rng = np.random.default_rng(19)
    # ground-truth per-strain fitness multipliers (WT=1); a few sick, one fit.
    strains = sorted(layout["strain"].unique())
    truth = {s: 1.0 for s in strains}
    truth.update({"MMS2": 0.55, "ELC1": 0.7, "YOS9": 0.82, "SPH1": 1.18})
    truth[cfg.blank_name] = 0.0
    base = 250.0
    sizes = []
    for _, r in layout.iterrows():
        s = r["strain"]
        # spatial gradient (edge colonies larger) + multiplicative noise + escapers
        edge = 1.0 + 0.15 * (min(r["row"] - 1, 16 - r["row"]) < 2)
        val = base * truth[s] * edge * rng.lognormal(0, 0.12)
        if s != cfg.blank_name and rng.random() < 0.05:  # 5% CRISPR escaper -> WT-sized
            val = base * edge * rng.lognormal(0, 0.12)
        if s == cfg.blank_name:
            val = max(0.0, rng.normal(2, 2))  # ~no growth
        sizes.append(val)
    sim = layout.copy()
    sim["size"] = sizes
    sim["circularity"] = 0.98
    sim["flags"] = ""

    df = normalize_plate(sim, cfg)
    report = score_plate(df, cfg, plate_id="Plate1_OD1.0_SIMULATED")

    df.to_csv(osp.join(RESULTS_DIR, "C_sim_colonies_normalized.csv"), index=False)
    tbl = score_table(report).sort_values("relative_fitness")
    tbl.to_csv(osp.join(RESULTS_DIR, "C_sim_strain_scores.csv"), index=False)
    print(
        f"  WT median norm={report.wt_median_norm:.3f} | "
        f"blank median norm={report.blank_median_norm:.3f}"
    )
    print(
        tbl[
            ["strain", "n_used", "relative_fitness", "pvalue", "n_jackknife"]
        ].to_string(index=False)
    )
    save(
        plate_heatmap(df, "norm", "SIMULATED normalized (1 = plate avg)"),
        "C_sim_heatmap_norm",
    )
    save(strain_fitness_plot(report), "C_sim_strain_fitness")


def main() -> None:
    part_a_real_normalization()
    layout = part_b_real_layout()
    part_c_simulated_score(layout)
    print(f"\nDone. CSVs in results/, figures in {IMG_DIR}")


if __name__ == "__main__":
    main()
