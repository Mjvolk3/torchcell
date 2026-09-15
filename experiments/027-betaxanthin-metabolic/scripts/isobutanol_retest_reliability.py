# experiments/027-betaxanthin-metabolic/scripts/isobutanol_retest_reliability.py
# [[experiments.027-betaxanthin-metabolic.scripts.isobutanol_retest_reliability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/isobutanol_retest_reliability
"""How reliable is the Montano Lopez 2024 isobutanol-biosensor target, from the screen's own repeats.

The released supplementary tables carry three independent views of the noise:

1. **Genes on more than one row of the first genome-wide screen (Table S2).** 190 ORFs appear
   on two to five rows. These are not selected on their value, so their first-vs-last
   agreement is the genome-wide single-measurement retest, and it is the number that bounds
   the pinned test Spearman (scored over the whole screen, not over hits).
2. **First screen vs the validated re-screen (Table S3).** The re-screen measured only strong
   hits (FC >= 2 up, FC <= 0.5 down) in triplicate. Selecting on extremes inflates Pearson,
   because the up and down blocks sit a decade apart, so the within-block correlations are
   reported beside the pooled one; the pooled 0.915 quoted in the plan is the inflated read.
3. **The triplicate STD of Table S3**, converted to log10 by the delta method
   (``STD / (FC ln 10)``), is the culture-to-culture noise of ONE re-screen measurement.

Every correlation is on log10 fold change, the training target's scale.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/isobutanol_retest_reliability.py

Writes ``results/isobutanol_retest_reliability.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RAW_XLSX = osp.join(
    DATA_ROOT, "data", "torchcell", "isobutanol_screen_lopez2024", "raw", "supplementary_tables.xlsx"
)
OUT = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results", "isobutanol_retest_reliability.json")


class Agreement(BaseModel):
    n: int
    pearson_log10: float
    spearman_log10: float


class WithinScreenRetest(BaseModel):
    n_rows: int
    n_orfs: int
    n_orfs_on_two_or_more_rows: int
    pairs: Agreement = Field(description="First vs last row of each repeated ORF, log10 FC.")
    pairs_within_one_decade: Agreement = Field(description="The same pairs, dropping those whose rows disagree by more than 1 decade.")
    sd_single_measurement_log10: float = Field(description="sd of the paired difference over sqrt(2).")
    sd_single_measurement_log10_robust: float = Field(description="1.4826 * MAD of the paired difference over sqrt(2).")
    sd_across_rows_log10: float = Field(description="sd of log10 FC over every Table S2 row.")
    sd_across_rows_log10_robust: float = Field(description="1.4826 * MAD of log10 FC over every Table S2 row.")
    reliability_estimate: float = Field(description="1 - (sd_single / sd_across)^2, floored at 0.")
    reliability_estimate_robust: float
    pearson_ceiling: float = Field(description="sqrt(reliability): best Pearson a perfect predictor reaches against one measurement.")
    pearson_ceiling_robust: float


class CrossScreen(BaseModel):
    n_rescreened_orfs: int
    n_in_first_screen: int
    pooled: Agreement = Field(description="S2 gene mean vs S3 average over both hit blocks; inflated by selection on extremes.")
    up_block: Agreement
    down_block: Agreement
    direction_agreement_fraction: float = Field(description="Fraction whose S2 gene mean is on the same side of FC 1 as S3.")


class TriplicateNoise(BaseModel):
    n: int
    median_cv: float = Field(description="Median STD / FC over re-screened strains, raw scale.")
    median_sd_log10: float = Field(description="Median STD / (FC ln 10): one re-screen measurement's sd on log10.")
    up_block_median_sd_log10: float
    down_block_median_sd_log10: float


class IsobutanolRetestReport(BaseModel):
    source_xlsx: str
    within_first_screen: WithinScreenRetest
    first_vs_rescreen: CrossScreen
    rescreen_triplicate: TriplicateNoise


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _agree(x: np.ndarray, y: np.ndarray) -> Agreement:
    return Agreement(
        n=int(len(x)),
        pearson_log10=float(np.corrcoef(x, y)[0, 1]),
        spearman_log10=float(pd.Series(x).corr(pd.Series(y), method="spearman")),
    )


def _mad_sd(v: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def main() -> None:
    s2 = pd.read_excel(RAW_XLSX, sheet_name="Table S2", header=1)
    s2 = s2.rename(columns={"Gene-knockout strain": "orf", "Fold change": "fc"})
    s2["orf"] = _norm(s2["orf"])
    s2 = s2.dropna(subset=["fc"])
    s2 = s2[s2["fc"] > 0].reset_index(drop=True)
    s2["l"] = np.log10(s2["fc"].astype(float))

    counts = s2.groupby("orf").size()
    repeated = counts[counts >= 2].index
    first = s2[s2["orf"].isin(repeated)].groupby("orf")["l"].first()
    last = s2[s2["orf"].isin(repeated)].groupby("orf")["l"].last()
    diff = (first - last).to_numpy()
    sd_single = float(np.std(diff) / np.sqrt(2))
    sd_single_r = _mad_sd(diff) / np.sqrt(2)
    sd_all = float(np.std(s2["l"]))
    sd_all_r = _mad_sd(s2["l"].to_numpy())
    rel = max(0.0, 1.0 - (sd_single / sd_all) ** 2)
    rel_r = max(0.0, 1.0 - (sd_single_r / sd_all_r) ** 2)
    within = WithinScreenRetest(
        n_rows=int(len(s2)),
        n_orfs=int(len(counts)),
        n_orfs_on_two_or_more_rows=int(len(repeated)),
        pairs=_agree(first.to_numpy(), last.to_numpy()),
        pairs_within_one_decade=_agree(first.to_numpy()[np.abs(diff) <= 1.0], last.to_numpy()[np.abs(diff) <= 1.0]),
        sd_single_measurement_log10=sd_single,
        sd_single_measurement_log10_robust=float(sd_single_r),
        sd_across_rows_log10=sd_all,
        sd_across_rows_log10_robust=sd_all_r,
        reliability_estimate=float(rel),
        reliability_estimate_robust=float(rel_r),
        pearson_ceiling=float(np.sqrt(rel)),
        pearson_ceiling_robust=float(np.sqrt(rel_r)),
    )

    s3 = pd.read_excel(RAW_XLSX, sheet_name="Table S3", header=2)
    blocks = []
    for suffix, label in (("", "up"), (".1", "down")):
        b = s3[[f"Gene knockout-strain{suffix}", f"FC (average){suffix}", f"STD{suffix}"]].dropna()
        b.columns = ["orf", "fc", "std"]
        b["orf"] = _norm(b["orf"])
        b["block"] = label
        blocks.append(b)
    s3l = pd.concat(blocks, ignore_index=True)
    s3l["l"] = np.log10(s3l["fc"].astype(float))
    s3l["sd_log10"] = s3l["std"].astype(float) / (s3l["fc"].astype(float) * np.log(10))

    s2_mean = s2.groupby("orf")["l"].mean()
    joined = s3l[s3l["orf"].isin(s2_mean.index)].copy()
    joined["l2"] = joined["orf"].map(s2_mean)
    up = joined[joined["block"] == "up"]
    down = joined[joined["block"] == "down"]
    cross = CrossScreen(
        n_rescreened_orfs=int(len(s3l)),
        n_in_first_screen=int(len(joined)),
        pooled=_agree(joined["l2"].to_numpy(), joined["l"].to_numpy()),
        up_block=_agree(up["l2"].to_numpy(), up["l"].to_numpy()),
        down_block=_agree(down["l2"].to_numpy(), down["l"].to_numpy()),
        direction_agreement_fraction=float(np.mean(np.sign(joined["l2"]) == np.sign(joined["l"]))),
    )
    trip = TriplicateNoise(
        n=int(len(s3l)),
        median_cv=float(np.median(s3l["std"] / s3l["fc"])),
        median_sd_log10=float(np.median(s3l["sd_log10"])),
        up_block_median_sd_log10=float(np.median(s3l.loc[s3l["block"] == "up", "sd_log10"])),
        down_block_median_sd_log10=float(np.median(s3l.loc[s3l["block"] == "down", "sd_log10"])),
    )

    report = IsobutanolRetestReport(
        source_xlsx=RAW_XLSX, within_first_screen=within, first_vs_rescreen=cross, rescreen_triplicate=trip
    )
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    print(json.dumps(report.model_dump(), indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
