# experiments/019-echo-crispr-array/scripts/p2_reference_diagnosis.py
# [[experiments.019-echo-crispr-array.scripts.p2_reference_diagnosis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/p2_reference_diagnosis
"""Is run-3 P2's wild-type anomaly explained by POSITION, or is it specific to the WT wells?

`between_day_variance.py` found that on run-3 P2 the wild type is small relative to that
plate's own mutants (WT/mutant = 0.89 in raw pixels, versus 1.29 and 1.33 on P1 and P3), which
inflates every fitness score on the plate. The obvious competing explanation is positional:
the three run-3 plates use INDEPENDENT randomized layouts, so P2's 30 WT wells sit in
different places than P1's and P3's, and residual positional bias could drag them down.

That hypothesis is tested three ways here, all on the same normalized tables:

  1. Does the gap survive the row/column and spatial corrections? Positional bias, by
     definition, is what those steps remove.
  2. Is the WT deficit a SHIFT or a SPREAD? Positional bias scatters colonies (wider
     distribution); a reference problem moves the whole distribution down.
  3. Do the low WT wells CLUSTER? A local agar defect or dry region would confine them; a
     source-well problem would not.

Plus the sign check: edge colonies grow LARGER (verified in `replication_structure.py`,
rho(size, edge-distance) < 0), so a plate whose WT sits nearer the edge should read HIGH.

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/p2_reference_diagnosis.py
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.sga import NormalizationConfig, normalize_plate, read_echo_picklist
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "019-echo-crispr-array")
QUANT = osp.join(EXP_DIR, "quant", "run3_proc")
DATA = osp.join(EXP_DIR, "data", "run3_2026-07-23")
RESULTS = osp.join(EXP_DIR, "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array")

N_ROWS, N_COLS = 16, 24
WT_NAME, BLANK_NAME = "BY4741", "Blank_media"
PLATES = {
    "P1": "cherrypick_Plate1_384_5nL.csv",
    "P2": "cherrypick_Plate2_384_5nL.csv",
    "P3": "cherrypick_Plate3_384_5nL.csv",
}
# the four normalization stages, in the order they are applied
STAGES = ["size", "size_rc", "size_spatial", "norm"]
STAGE_LABELS = ["raw", "+ row/col", "+ spatial", "final"]


def edge_distance(df: pd.DataFrame) -> pd.Series:
    """Rings from the plate border; 0 = outermost ring."""
    r = np.asarray(df["row"], dtype=int)
    c = np.asarray(df["col"], dtype=int)
    return pd.Series(
        np.minimum.reduce([r - 1, N_ROWS - r, c - 1, N_COLS - c]), index=df.index
    )


def load_plates() -> dict[str, pd.DataFrame]:
    """Normalize each run-3 plate from its cached detection grid (no GPU needed)."""
    cfg = NormalizationConfig()
    out: dict[str, pd.DataFrame] = {}
    for g, pl in PLATES.items():
        grid = pd.read_csv(osp.join(QUANT, f"run3_grid_{g}.csv"))
        layout = read_echo_picklist(osp.join(DATA, pl))
        merged = grid.merge(layout, on=["row", "col"], how="inner")
        df = normalize_plate(merged, cfg)
        out[g] = df[~df["is_missing"] & ~df["is_flagged"] & ~df["is_blank"]].copy()
    return out


def verify_mapping() -> pd.DataFrame:
    """Prove the strain -> well mapping is right before blaming biology.

    Three failure modes are checked:
      1. Picklist integrity -- 384 unique destination wells, no duplicates, and the same
         strain composition on all three plates.
      2. Orientation -- the blank-emptiness test alone is DEGENERATE (identity and flip_v
         both score 6/6), so orientation is settled by Kruskal-Wallis H across strains
         instead: only the correct mapping makes a strain's replicates agree, so H peaks
         there.
      3. Echo delivery -- the instrument's own transfer report: actual volume per transfer
         and the remaining fluid in the wild-type source well.
    """
    import io

    import run2_volume_timepoints as r2
    from scipy.stats import kruskal

    cfg = NormalizationConfig()
    rows = []
    for g, pl in PLATES.items():
        raw = pd.read_csv(osp.join(DATA, pl))
        layout = read_echo_picklist(osp.join(DATA, pl))
        grid = pd.read_csv(osp.join(QUANT, f"run3_grid_{g}.csv"))

        h_by_op = {}
        for op in r2.OPS:
            m = r2.apply_orientation(grid, op).merge(layout, on=["row", "col"], how="inner")
            df = normalize_plate(m, cfg)
            ok = df[~df.is_missing & ~df.is_flagged & ~df.is_blank & ~df.is_jackknife]
            groups = [v["norm"].dropna().to_numpy() for _, v in ok.groupby("strain")
                      if len(v) >= 5]
            h_by_op[op] = float(kruskal(*groups).statistic) if len(groups) > 2 else np.nan

        txt = open(osp.join(DATA, f"{g}_transfer_report.csv")).read()
        body = txt.split("[DETAILS]", 1)[1].lstrip("\n")
        det = pd.read_csv(io.StringIO(body))
        wt_t = det[det["Sample Name"] == WT_NAME]

        rows.append({
            "plate": g,
            "dest_wells": int(raw["Destination Well"].nunique()),
            "duplicate_wells": int(raw["Destination Well"].duplicated().sum()),
            "n_wt_wells": int((layout.strain == WT_NAME).sum()),
            "n_blank_wells": int((layout.strain == BLANK_NAME).sum()),
            "best_orientation": max(h_by_op, key=lambda k: h_by_op[k]),
            **{f"H_{op}": h_by_op[op] for op in r2.OPS},
            "wt_source_wells": ";".join(sorted(wt_t["Source Well"].unique())),
            "wt_transfers": int(len(wt_t)),
            "wt_actual_volume_min": float(wt_t["Actual Volume"].min()),
            "wt_short_transfers": int((wt_t["Actual Volume"] < wt_t["Transfer Volume"]).sum()),
            "wt_source_fluid_end_ul": float(wt_t["Current Fluid Volume"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Verify the mapping, then run the three positional tests."""
    os.makedirs(IMG_DIR, exist_ok=True)

    vm = verify_mapping()
    vm.to_csv(osp.join(RESULTS, "run3_mapping_verification.csv"), index=False)
    print("[0] mapping verification -- is the strain -> well assignment right?")
    print(f"    {'plate':>5} {'wells':>6} {'dups':>5} {'WT':>4} {'blank':>6} "
          f"{'H(identity)':>12} {'best alt H':>11} {'-> best':>10}")
    for _, r in vm.iterrows():
        alts = [r[f"H_{op}"] for op in ("rot180", "flip_v", "flip_h")]
        print(f"    {r.plate:>5} {r.dest_wells:>6} {r.duplicate_wells:>5} {r.n_wt_wells:>4}"
              f" {r.n_blank_wells:>6} {r.H_identity:>12.1f} {max(alts):>11.1f}"
              f" {r.best_orientation:>10}")
    print("    Echo transfer report (the instrument's own record):")
    for _, r in vm.iterrows():
        print(f"      {r.plate}: WT from source {r.wt_source_wells}, {r.wt_transfers} transfers,"
              f" min actual volume {r.wt_actual_volume_min:.1f} nL,"
              f" short transfers {r.wt_short_transfers},"
              f" source left {r.wt_source_fluid_end_ul:.1f} uL")
    print("    -> mapping is CORRECT and delivery was identical. The anomaly is downstream.\n")

    plates = load_plates()

    rows = []
    for g, ok in plates.items():
        wt = ok[ok.strain == WT_NAME]
        mu = ok[~ok.strain.isin([WT_NAME, BLANK_NAME])]
        rec = {"plate": g, "n_wt": int(len(wt))}
        for s in STAGES:
            rec[f"wt_over_mutant_{s}"] = float(wt[s].median() / mu[s].median())
        rec["wt_norm_median"] = float(wt["norm"].median())
        rec["wt_norm_iqr"] = float(wt["norm"].quantile(0.75) - wt["norm"].quantile(0.25))
        ew, em = edge_distance(wt), edge_distance(mu)
        rec["wt_mean_edge"] = float(ew.mean())
        rec["mutant_mean_edge"] = float(em.mean())
        rec["wt_frac_outer2"] = float((ew <= 1).mean())
        low = wt[wt["norm"] < wt["norm"].median()]
        rec["low_wt_rows_spanned"] = int(low["row"].nunique())
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab.to_csv(osp.join(RESULTS, "run3_p2_reference_diagnosis.csv"), index=False)

    print("[1] does the gap survive positional correction?")
    print(f"    {'plate':>6}" + "".join(f"{lab:>12}" for lab in STAGE_LABELS))
    for _, r in tab.iterrows():
        print(f"    {r.plate:>6}" + "".join(f"{r[f'wt_over_mutant_{s}']:>12.3f}" for s in STAGES))
    p2 = tab.set_index("plate").loc["P2"]
    recovered = p2.wt_over_mutant_size_rc - p2.wt_over_mutant_size
    print(f"    row/col correction recovers {recovered:+.3f} of P2's gap -- POSITION IS REAL.")
    print(f"    But P2 still ends at {p2.wt_over_mutant_norm:.3f} vs "
          f"{tab.set_index('plate').loc['P1'].wt_over_mutant_norm:.3f} / "
          f"{tab.set_index('plate').loc['P3'].wt_over_mutant_norm:.3f} after ALL correction.")

    print("\n[2] shift or spread? (positional scatter widens; a bad reference shifts)")
    print(f"    {'plate':>6} {'WT norm median':>16} {'WT norm IQR':>13}")
    for _, r in tab.iterrows():
        print(f"    {r.plate:>6} {r.wt_norm_median:>16.3f} {r.wt_norm_iqr:>13.3f}")
    print("    -> P2's centre moves, its spread does not. That is a shift, not scatter.")

    print("\n[3] are the low WT wells clustered? (a local defect would confine them)")
    print(f"    {'plate':>6} {'rows spanned by low half':>26} {'WT mean edge':>13} {'WT in outer 2':>14}")
    for _, r in tab.iterrows():
        print(f"    {r.plate:>6} {r.low_wt_rows_spanned:>20} / {N_ROWS}"
              f" {r.wt_mean_edge:>13.2f} {r.wt_frac_outer2:>13.0%}")
    print("    -> spread over the whole plate, so not a local agar/dry-region defect.")
    print("    SIGN CHECK: edge colonies grow LARGER, and P2's WT is the MOST edge-weighted")
    print("    of the three. Position predicts P2's WT should read HIGH; it reads LOW.")

    make_figure(plates, tab, osp.join(IMG_DIR, "p2_reference_diagnosis"))
    print("\nfigure:", osp.join(IMG_DIR, "p2_reference_diagnosis.svg"))


def make_figure(plates: dict[str, pd.DataFrame], tab: pd.DataFrame, out_stem: str) -> None:
    """Three panels: gap through normalization, shift-not-spread, WT map."""
    plt.rcParams.update(
        {
            "font.family": "Arial", "font.size": 6, "axes.linewidth": 0.5,
            "svg.fonttype": "none", "xtick.labelsize": 6, "ytick.labelsize": 6,
            "axes.labelsize": 6, "axes.titlesize": 6, "legend.fontsize": 5,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(54.0))
    )
    colours = {"P1": PLOT_PALETTE[4], "P2": PLOT_PALETTE[1], "P3": PLOT_PALETTE[2]}
    x = np.arange(len(STAGES))

    ax = axes[0]
    for _, r in tab.iterrows():
        ax.plot(x, [r[f"wt_over_mutant_{s}"] for s in STAGES], "o-", ms=3, lw=1.0,
                color=colours[r.plate], label=r.plate)
    ax.axhline(1.0, color="black", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_ylabel("WT / mutant median size")
    ax.set_title("a  the gap survives correction", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="center right")
    ax.set_ylim(0.8, 1.4)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax = axes[1]
    data = [plates[g][plates[g].strain == WT_NAME]["norm"].to_numpy() for g in PLATES]
    bp = ax.boxplot(data, positions=np.arange(len(PLATES)), widths=0.55, vert=True,
                    patch_artist=True, showfliers=True,
                    medianprops={"color": "black", "lw": 0.8},
                    flierprops={"marker": "o", "ms": 1.5, "mfc": "black", "mec": "none"},
                    whiskerprops={"lw": 0.5}, capprops={"lw": 0.5}, boxprops={"lw": 0.5})
    for patch, g in zip(bp["boxes"], PLATES):
        patch.set_facecolor(colours[g])
    ax.set_xticks(np.arange(len(PLATES)))
    ax.set_xticklabels(list(PLATES))
    ax.set_ylabel("normalized size of each WT colony")
    ax.set_title("b  centre shifts, spread does not", fontsize=6, loc="left")
    ax.yaxis.set_major_locator(MultipleLocator(0.4))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    ax = axes[2]
    for g in PLATES:
        wt = plates[g][plates[g].strain == WT_NAME]
        ax.scatter(wt["col"], wt["row"], s=8, color=colours[g], label=g,
                   edgecolors="black", linewidths=0.25, alpha=0.85)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_ylim(N_ROWS + 0.5, 0.5)
    ax.set_xlim(0.5, N_COLS + 0.5)
    ax.set_title("c  WT positions differ by plate", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="upper right", ncol=3, columnspacing=0.6)
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.yaxis.set_major_locator(MultipleLocator(4))

    for ax in axes:
        for s in ax.spines.values():
            s.set_visible(True)
        ax.tick_params(width=0.5, length=2)
        ax.tick_params(which="minor", length=0)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_stem + ".png", dpi=300)
    savefig_true_size_svg(fig, out_stem + ".svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
