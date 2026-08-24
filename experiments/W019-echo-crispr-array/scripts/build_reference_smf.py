# experiments/W019-echo-crispr-array/scripts/build_reference_smf.py
# [[experiments.W019-echo-crispr-array.scripts.build_reference_smf]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/build_reference_smf
"""Assemble the published single-mutant-fitness (SMF) reference for the wet-lab
12-gene panel, for benchmarking our CRISPR fitness assay.

Canonical source (per the exp-010 provenance chain): the queried singles table
    experiments/010-kuzmin-tmi/results/inference_3/singles_table_panel12_k200_queried.csv
which carries, per ORF, Costanzo 2016 and Kuzmin 2018/2020 single-mutant fitness
+ std (queried from the Costanzo2016 / Kuzmin SMF LMDBs). Costanzo has std for
all it covers; Kuzmin SMF has no std.

Panel mismatch: the 010 inference panel and the run-4 wet-lab plate share 11 of 12
ORFs. The plate swapped YIL174W (in the reference) for SPH1/YLR313C. SPH1 is not in
the panel-12 queried table, so it is filled from the YLR313C investigation table,
which was queried the same way. Every one of the 12 therefore gets a Costanzo value.

Corrected 2026.08.24: this panel previously carried LCL1/YPL056C in the s7 slot,
which is the EARLIER plate design. Run 4 built LCL2/YLR104W there, so the reference
had no row for the strain actually on the plate and the build list reported YLR104W
as a single with no published SMF. Costanzo 2016 does have YLR104W
(1.0322 +/- 0.0453, strains YLR104W_dma3202 / YLR104W_sn3294, n_samples 17), and it
is already in the panel-12 queried table, so the fix is a panel-membership fix and
needs no new source. YPL056C is REPLACED, not kept: it is not on the run-4 plate at
all, and run4_measured_summary.py keys this file by the plate's ORFs.
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
    "LCL2": ("YLR104W", "LCL2"),  # run-4 s7; was LCL1/YPL056C before 2026.08.24
    "SPH1": ("YLR313C", "SPH1"),  # not in the panel-12 table; filled from EXTRA
    "YJR060W": ("YJR060W", "CBF1"),
    "COS111": ("YBR203W", "COS111"),
}


def _lookup(src, orf):
    """Return (costanzo_fit, costanzo_std, kuzmin_fit, kuzmin_std) for an ORF row
    keyed by 'gene', or NaNs if absent.
    """
    if orf not in src.index:
        return (np.nan,) * 4
    r = src.loc[orf]
    return (
        float(r["SmfCostanzo2016_fitness"]),
        float(r["SmfCostanzo2016_std"]),
        float(r["SmfKuzmin2018_fitness"]),
        float(r["SmfKuzmin2018_std"]),
    )


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
        rows.append(
            {
                "strain": strain,
                "orf": orf,
                "common_name": common,
                "costanzo_smf": cf,
                "costanzo_se": cs,
                "kuzmin_smf": kf,
                "kuzmin_se": ks,
            }
        )
    df = pd.DataFrame(rows)
    missing = df[df["costanzo_smf"].isna()]["strain"].tolist()
    # Every strain on the run-4 plate has a Costanzo SMF. If that stops being true the
    # panel and the plate have diverged again, which is the defect this assert exists to
    # catch; fix the panel rather than reintroducing a hardcoded value.
    assert not missing, f"no Costanzo SMF for {missing}; panel and plate have diverged"
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(
        f"Costanzo SMF for {int(df['costanzo_smf'].notna().sum())}/12, "
        f"Kuzmin for {int(df['kuzmin_smf'].notna().sum())}/12. "
        f"Still missing: {missing or 'none'}"
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
