# experiments/027-betaxanthin-metabolic/scripts/mulleder_retest_reliability.py
# [[experiments.027-betaxanthin-metabolic.scripts.mulleder_retest_reliability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/mulleder_retest_reliability
"""How reliable is the Mulleder 2016 amino-acid target, from the paper's own repeats.

The genome-wide screen is one culture per strain, so the released table carries no per-strain
replicate. Two things in the raw sheet of Table S3 still bound the noise:

1. **Strains measured in two batches.** ``data_raw`` holds 5,290 rows for 4,678 released
   strains; 194 ORFs appear in two batches. Their two measurements give a test-retest
   correlation per amino acid, which is the reliability of a single measurement on the
   log10 scale and therefore the ceiling on any model's correlation with the measured value
   (a model cannot correlate with a measurement better than the measurement correlates with
   itself, up to the square root of the reliability).
2. **The pooled quality-control sample**, injected 237 times across the run (identifier
   contains ``QC``). Its relative standard deviation per amino acid is the analytical
   variation the paper reports in Figure S1D, and it separates instrument noise from the
   biological plus culture-to-culture part of the retest disagreement.

Batch handling is deliberately simple and stated: each amino acid is scaled per batch to
the global median (the paper's MIPHENO-style batch mean scaling, on the median) before the
log10; the paper additionally applies PQN dilution correction, which this script does not,
so the retest numbers here are a LOWER bound on the released table's reliability. The
repeated strains are whichever the authors chose to re-run, not a random sample, which is
a second reason to read them as a bound.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/mulleder_retest_reliability.py

Writes ``results/mulleder_retest_reliability.json``.
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
RAW_XLS = osp.join(
    DATA_ROOT, "data", "torchcell", "amino_acid_mulleder2016", "raw", "Table_S3_Complete_Dataset.xls"
)
OUT = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results", "mulleder_retest_reliability.json")
META = {"index", "identifier", "gene", "ORF", "batch"}


class AminoAcidReliability(BaseModel):
    amino_acid: str
    n_pairs: int
    pearson_retest_log10: float = Field(description="Pearson between the two batch measurements, log10 mM, batch-median scaled.")
    spearman_retest_log10: float
    sd_single_measurement_log10: float = Field(description="sd of the paired difference over sqrt(2).")
    sd_across_strains_log10: float = Field(description="sd over every strain row after batch scaling.")
    reliability_estimate: float = Field(description="1 - (sd_single / sd_across)^2, floored at 0.")
    pearson_ceiling: float = Field(description="sqrt of the reliability: the maximum Pearson a perfect predictor of the true value reaches against one measurement.")
    qc_rsd_percent: float = Field(description="Relative sd of the 237 pooled QC injections, raw scale, the analytical variation.")
    qc_sd_log10: float


class RetestReport(BaseModel):
    source_xls: str
    n_raw_rows: int
    n_released_strains: int
    n_orfs_measured_twice: int
    n_qc_injections: int
    batches: list[str]
    batch_scaling: str
    per_amino_acid: list[AminoAcidReliability]
    mean_pearson_retest: float
    mean_spearman_retest: float
    mean_reliability: float
    mean_pearson_ceiling: float


def main() -> None:
    raw = pd.read_excel(RAW_XLS, sheet_name="data_raw")
    released = pd.read_excel(RAW_XLS, sheet_name="intracellular_concentration_mM")
    amino_acids = [c for c in raw.columns if c not in META]
    is_qc = raw["identifier"].astype(str).str.contains("QC", case=False)
    strains = raw.loc[~is_qc].copy()
    qc = raw.loc[is_qc].copy()

    scaled = strains.copy()
    for a in amino_acids:
        scaled[a] = pd.to_numeric(scaled[a], errors="coerce")
        global_median = np.nanmedian(scaled[a])
        scaled[a] = scaled.groupby("batch")[a].transform(lambda s: s / np.nanmedian(s) * global_median)
        scaled[a] = np.log10(scaled[a])

    counts = scaled.groupby("ORF").size()
    twice = counts[counts >= 2].index
    pairs = []
    for _, grp in scaled[scaled["ORF"].isin(twice)].groupby("ORF"):
        grp = grp.sort_values("batch")
        pairs.append((grp.iloc[0], grp.iloc[-1]))

    rows: list[AminoAcidReliability] = []
    for a in amino_acids:
        x = np.array([p[0][a] for p in pairs], dtype=float)
        y = np.array([p[1][a] for p in pairs], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        rho = float(pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman"))
        sd_single = float(np.std(x[m] - y[m]) / np.sqrt(2))
        sd_all = float(np.nanstd(scaled[a]))
        rel = float(max(0.0, 1.0 - (sd_single / sd_all) ** 2))
        q = pd.to_numeric(qc[a], errors="coerce").dropna()
        rows.append(
            AminoAcidReliability(
                amino_acid=a,
                n_pairs=int(m.sum()),
                pearson_retest_log10=r,
                spearman_retest_log10=rho,
                sd_single_measurement_log10=sd_single,
                sd_across_strains_log10=sd_all,
                reliability_estimate=rel,
                pearson_ceiling=float(np.sqrt(rel)),
                qc_rsd_percent=float(100.0 * q.std() / q.mean()),
                qc_sd_log10=float(np.log10(q).std()),
            )
        )

    report = RetestReport(
        source_xls=RAW_XLS,
        n_raw_rows=int(len(raw)),
        n_released_strains=int(len(released)),
        n_orfs_measured_twice=int(len(twice)),
        n_qc_injections=int(is_qc.sum()),
        batches=sorted(str(b) for b in raw["batch"].dropna().unique()),
        batch_scaling="per-batch median scaled to the global median per amino acid, then log10; no PQN",
        per_amino_acid=rows,
        mean_pearson_retest=float(np.mean([r.pearson_retest_log10 for r in rows])),
        mean_spearman_retest=float(np.mean([r.spearman_retest_log10 for r in rows])),
        mean_reliability=float(np.mean([r.reliability_estimate for r in rows])),
        mean_pearson_ceiling=float(np.mean([r.pearson_ceiling for r in rows])),
    )
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    df = pd.DataFrame([r.model_dump() for r in rows])
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    print(
        f"pairs {len(pairs)} | mean retest pearson {report.mean_pearson_retest:.3f} "
        f"spearman {report.mean_spearman_retest:.3f} | mean reliability {report.mean_reliability:.2f} "
        f"| mean pearson ceiling {report.mean_pearson_ceiling:.2f} | QC injections {report.n_qc_injections}"
    )
    print(OUT)


if __name__ == "__main__":
    main()
