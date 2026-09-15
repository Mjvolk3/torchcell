# experiments/027-betaxanthin-metabolic/scripts/target_reliability_table.py
# [[experiments.027-betaxanthin-metabolic.scripts.target_reliability_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/target_reliability_table
"""Reproducibility ceiling for every metabolically relevant target the module trains on.

A model cannot correlate with a measurement better than the measurement correlates with
itself. Writing ``rho`` for the reliability of the stored value, that bound is ``sqrt(rho)``,
and it is the number this script tabulates per target from each source's OWN repeats.

Three sources, three kinds of repeat, one estimator each:

- **Betaxanthin (Cachera 2023).** The release gives per-gene mean, sd and colony count over
  screen replicates 1, 2, 4 and 6, so the variance decomposes directly: the within-gene mean
  square is one colony's variance, and subtracting its contribution from the variance of the
  reported means leaves the between-gene variance. Reliability of the STORED value (the gene
  mean over ``n`` colonies) is ``between / var(means)``; reliability of ONE colony is
  ``between / (between + within)``. This assumes colonies of a gene are independent draws
  around that gene's true level, which a shared plate or position effect would violate by
  inflating the estimate.
- **Amino acids (Mulleder 2016).** 194 ORFs measured in two batches, read by
  ``mulleder_retest_reliability.py``.
- **Isobutanol (Montano Lopez 2024).** 190 ORFs on two or more rows of the genome-wide
  screen, read by ``isobutanol_retest_reliability.py``.

The two retest sources report a correlation between two measurements, which IS the
reliability of one measurement under classical test theory. The betaxanthin source reports
no pair, so its correlation columns are empty and its reliability comes from the variance
split. Run the other two scripts first; this one reads their result JSON.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/target_reliability_table.py

Writes ``results/target_reliability.json`` and
``notes-tex/027-metabolic-flux-module/tables/target_reliability.tex``.
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
REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
RESULTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
CACHERA_CSV = osp.join(DATA_ROOT, "data", "torchcell", "betaxanthin_cachera2023", "raw", "GA1_2_4_6.csv")
MULLEDER_JSON = osp.join(RESULTS, "mulleder_retest_reliability.json")
ISOBUTANOL_JSON = osp.join(RESULTS, "isobutanol_retest_reliability.json")
OUT_JSON = osp.join(RESULTS, "target_reliability.json")
OUT_TEX = osp.join(REPO_ROOT, "notes-tex", "027-metabolic-flux-module", "tables", "target_reliability.tex")
CACHERA_BASE = "corrected_mean_intensity.24"


class TargetReliability(BaseModel):
    target: str
    source: str = Field(description="Dataset class the 027 build loads this target from.")
    repeat_unit: str = Field(description="What one repeat is in the released data.")
    n_repeated: int = Field(description="Number of items carrying a repeat.")
    retest_pearson: float | None = Field(description="Correlation between two measurements; None where the source releases no pair.")
    retest_spearman: float | None
    reliability_stored_value: float = Field(description="Reliability of the value the module trains on.")
    pearson_ceiling_stored_value: float
    reliability_single_measurement: float
    pearson_ceiling_single_measurement: float
    scale: str
    caveat: str


class ReliabilityReport(BaseModel):
    sources: dict[str, str]
    targets: list[TargetReliability]


def betaxanthin() -> TargetReliability:
    d = pd.read_csv(CACHERA_CSV)
    m = pd.to_numeric(d[f"{CACHERA_BASE}_mean"], errors="coerce")
    s = pd.to_numeric(d[f"{CACHERA_BASE}_std"], errors="coerce")
    n = pd.to_numeric(d[f"{CACHERA_BASE}_count"], errors="coerce")
    keep = m.notna() & s.notna() & n.notna() & (n >= 2)
    mm, ss, nn = m[keep].to_numpy(), s[keep].to_numpy(), n[keep].to_numpy()
    within = float(np.mean(ss**2))
    var_means = float(np.var(mm))
    between = max(0.0, var_means - float(np.mean(ss**2 / nn)))
    rel_stored = between / var_means
    rel_single = between / (between + within)
    return TargetReliability(
        target="Betaxanthin",
        source="BetaxanthinCachera2023Dataset",
        repeat_unit=f"colonies per gene across screen replicates 1, 2, 4 and 6 (median {int(np.median(nn))})",
        n_repeated=int(keep.sum()),
        retest_pearson=None,
        retest_spearman=None,
        reliability_stored_value=rel_stored,
        pearson_ceiling_stored_value=float(np.sqrt(rel_stored)),
        reliability_single_measurement=rel_single,
        pearson_ceiling_single_measurement=float(np.sqrt(rel_single)),
        scale="corrected mean colony intensity at 24 h",
        caveat="Colonies are assumed independent within a gene; a shared plate or position effect inflates this.",
    )


def amino_acids() -> TargetReliability:
    d = json.load(open(MULLEDER_JSON))
    rel = float(d["mean_reliability"])
    return TargetReliability(
        target="Amino acids (19)",
        source="AminoAcidMulleder2016Dataset",
        repeat_unit="ORFs measured in two batches, averaged over the 19 amino acids",
        n_repeated=int(d["n_orfs_measured_twice"]),
        retest_pearson=float(d["mean_pearson_retest"]),
        retest_spearman=float(d["mean_spearman_retest"]),
        reliability_stored_value=rel,
        pearson_ceiling_stored_value=float(d["mean_pearson_ceiling"]),
        reliability_single_measurement=rel,
        pearson_ceiling_single_measurement=float(d["mean_pearson_ceiling"]),
        scale="log10 intracellular mM",
        caveat="One culture per strain, so the stored value IS a single measurement; batch median scaling only, no PQN, so this is a lower bound.",
    )


def isobutanol() -> TargetReliability:
    d = json.load(open(ISOBUTANOL_JSON))
    w = d["within_first_screen"]
    rel = float(w["reliability_estimate"])
    return TargetReliability(
        target="Isobutanol biosensor",
        source="IsobutanolScreenLopez2024Dataset",
        repeat_unit="ORFs appearing on two or more rows of the genome-wide screen",
        n_repeated=int(w["n_orfs_on_two_or_more_rows"]),
        retest_pearson=float(w["pairs"]["pearson_log10"]),
        retest_spearman=float(w["pairs"]["spearman_log10"]),
        reliability_stored_value=rel,
        pearson_ceiling_stored_value=float(w["pearson_ceiling"]),
        reliability_single_measurement=rel,
        pearson_ceiling_single_measurement=float(w["pearson_ceiling"]),
        scale="log10 fold change of biosensor fluorescence",
        caveat=(
            "Heavy tailed: 4 of 190 pairs disagree by more than a decade, and dropping them raises the "
            f"retest Pearson to {d['within_first_screen']['pairs_within_one_decade']['pearson_log10']:.2f}; "
            f"the robust reliability is {w['reliability_estimate_robust']:.2f} (ceiling "
            f"{w['pearson_ceiling_robust']:.2f})."
        ),
    )


def _fmt(v: float | None) -> str:
    return "--" if v is None else f"{v:.2f}"


def write_tex(report: ReliabilityReport) -> None:
    lines = [
        "% notes-tex/027-metabolic-flux-module/tables/target_reliability.tex",
        "%% SOURCE: generated by experiments/027-betaxanthin-metabolic/scripts/target_reliability_table.py",
        "%% from experiments/027-betaxanthin-metabolic/results/target_reliability.json, which that",
        "%% script computes from the Cachera GA1_2_4_6.csv release and the two retest result files",
        "%% mulleder_retest_reliability.json and isobutanol_retest_reliability.json.",
        "% Do not edit by hand.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Reproducibility ceiling of each metabolically relevant target, from the",
        r"  source's own repeats. Reliability is the fraction of the variance in the stored value",
        r"  that is not measurement noise, and its square root is the highest Pearson correlation",
        r"  any predictor of the true level can reach against that stored value. Retest columns",
        r"  are the correlation between two measurements of the same strain; the betaxanthin",
        r"  release carries no such pair, so its reliability comes from the within-gene and",
        r"  between-gene variance split instead. The single-measurement column differs from the",
        r"  stored-value column only for betaxanthin, whose stored value averages many colonies.}",
        r"  \label{tab:target-reliability}",
        r"  \footnotesize",
        r"  \begin{tabular}{@{}lrrrrr@{}}",
        r"    \toprule",
        r"    Target & Repeats & Retest $r$ & Retest $\rho$ & Reliability & Ceiling \\",
        r"    \midrule",
    ]
    for t in report.targets:
        lines.append(
            f"    {t.target} & {t.n_repeated} & {_fmt(t.retest_pearson)} & {_fmt(t.retest_spearman)} & "
            f"{t.reliability_stored_value:.2f} & {t.pearson_ceiling_stored_value:.2f} \\\\"
        )
    lines += [
        r"    \midrule",
        r"    \multicolumn{6}{@{}l@{}}{\emph{One measurement, not the stored average}} \\",
    ]
    for t in report.targets:
        lines.append(
            f"    {t.target} & {t.n_repeated} & -- & -- & "
            f"{t.reliability_single_measurement:.2f} & {t.pearson_ceiling_single_measurement:.2f} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    os.makedirs(osp.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    report = ReliabilityReport(
        sources={
            "betaxanthin": CACHERA_CSV,
            "amino_acids": MULLEDER_JSON,
            "isobutanol": ISOBUTANOL_JSON,
        },
        targets=[betaxanthin(), amino_acids(), isobutanol()],
    )
    with open(OUT_JSON, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    write_tex(report)
    for t in report.targets:
        print(
            f"{t.target:22s} n={t.n_repeated:5d} rel(stored)={t.reliability_stored_value:.3f} "
            f"ceiling={t.pearson_ceiling_stored_value:.3f} rel(single)={t.reliability_single_measurement:.3f} "
            f"ceiling={t.pearson_ceiling_single_measurement:.3f}"
        )
    print(OUT_JSON)
    print(OUT_TEX)


if __name__ == "__main__":
    main()
