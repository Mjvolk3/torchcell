# experiments/019-echo-crispr-array/scripts/build_reference_smf.py
# [[experiments.019-echo-crispr-array.scripts.build_reference_smf]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/build_reference_smf
"""Assemble the published single-mutant-fitness (SMF) reference for the wet-lab
12-gene panel, for benchmarking our CRISPR fitness assay.

Canonical source (per the exp-010 provenance chain): the queried singles table
    experiments/010-kuzmin-tmi/results/inference_3/singles_table_panel12_k200_queried.csv
which carries, per ORF, Costanzo 2016 and Kuzmin 2018/2020 single-mutant fitness
+ std (queried from the Costanzo2016 / Kuzmin SMF LMDBs). Costanzo has std for
all it covers; Kuzmin SMF has no std.

Panel mismatch: the 010 inference panel and the wet-lab plate share 10 of 12
ORFs. The wet-lab plate swapped YIL174W and YLR104W (in the reference) for
LCL1/YPL056C and SPH1/YLR313C (in the plate). So LCL1 and SPH1 get no reference
here; the other 10 get Costanzo (and Kuzmin where present).
"""

from __future__ import annotations

import os.path as osp

import numpy as np
import pandas as pd

EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
REPO = osp.dirname(osp.dirname(EXP_DIR))
SRC = osp.join(
    REPO,
    "experiments/010-kuzmin-tmi/results/inference_3/singles_table_panel12_k200_queried.csv",
)
# Extra queried singles (same query mechanism) that cover ORFs the panel-12 table
# omitted -- e.g. SPH1/YLR313C. Merged in to fill gaps.
EXTRA = osp.join(
    REPO,
    "experiments/010-kuzmin-tmi/results/inference_3/YLR313C_investigation_singles_queried.csv",
)
OUT = osp.join(EXP_DIR, "results", "reference_smf_12panel.csv")

# wet-lab panel strain (picklist Sample Name) -> (systematic ORF, common name)
PANEL = {
    "YEH1": ("YLL012W", "YEH1"),
    "YER079W": ("YER079W", ""),
    "YOS9": ("YDR057W", "YOS9"),
    "MMS2": ("YGL087C", "MMS2"),
    "YPL081W": ("YPL081W", "RPS9A"),
    "ELC1": ("YPL046C", "ELC1"),
    "YKL033W-A": ("YKL033W-A", ""),
    "YLR312C-B": ("YLR312C-B", ""),
    "LCL1": ("YPL056C", "LCL1"),  # not in reference panel
    "SPH1": ("YLR313C", "SPH1"),  # not in reference panel
    "YJR060W": ("YJR060W", "CBF1"),
    "COS111": ("YBR203W", "COS111"),
}


def _lookup(src, orf):
    """Return (costanzo_fit, costanzo_std, kuzmin_fit, kuzmin_std) for an ORF row
    keyed by 'gene', or NaNs if absent."""
    if orf not in src.index:
        return (np.nan,) * 4
    r = src.loc[orf]
    return (
        float(r["SmfCostanzo2016_fitness"]),
        float(r["SmfCostanzo2016_std"]),
        float(r["SmfKuzmin2018_fitness"]),
        float(r["SmfKuzmin2018_std"]),
    )


# LCL1/YPL056C is absent from both queried tables above; sourced from the exp-010
# validation-panel SMF table (origin/main, validation_panel_smf_costanzo_kuzmin.csv;
# Costanzo strain YPL056C_sn3389). Overrides the gap so the panel is complete (12/12).
COSTANZO_OVERRIDE = {"YPL056C": (0.9802, 0.0815)}


def main():
    src = pd.read_csv(SRC).set_index("gene")
    extra = pd.read_csv(EXTRA).set_index("gene")  # fills ORFs the panel table omits
    rows = []
    for strain, (orf, common) in PANEL.items():
        cf, cs, kf, ks = _lookup(src, orf)
        if np.isnan(cf):  # gap in panel table -> try the extra queried singles
            cf, cs, kf2, ks2 = _lookup(extra, orf)
            if np.isnan(kf):
                kf, ks = kf2, ks2
        if np.isnan(cf) and orf in COSTANZO_OVERRIDE:  # validation-panel fallback
            cf, cs = COSTANZO_OVERRIDE[orf]
        rows.append(
            {
                "strain": strain,
                "orf": orf,
                "common_name": common,
                "costanzo_smf": cf,
                "costanzo_sd": cs,
                "kuzmin_smf": kf,
                "kuzmin_sd": ks,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    missing = df[df["costanzo_smf"].isna()]["strain"].tolist()
    print(f"wrote {OUT}")
    print(
        f"Costanzo SMF for {int(df['costanzo_smf'].notna().sum())}/12, "
        f"Kuzmin for {int(df['kuzmin_smf'].notna().sum())}/12. "
        f"Still missing: {missing or 'none'}"
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
