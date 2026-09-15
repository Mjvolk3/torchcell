# experiments/027-betaxanthin-metabolic/scripts/target_reliability_table.py
# [[experiments.027-betaxanthin-metabolic.scripts.target_reliability_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/target_reliability_table
"""Reproducibility ceiling for every metabolically relevant target in the 027 build.

A model cannot correlate with a measurement better than the measurement correlates with
itself. Writing ``rho`` for the reliability of the stored value, that bound is ``sqrt(rho)``,
and it is the number this script tabulates per target from each source's OWN repeats.

Two heads read a metabolite target (betaxanthin, and the 19 amino acids as an auxiliary),
one reads isobutanol, and three further metabolite datasets enter the build as genotype
context that no head decodes (carotenoid, the Zelezniak metabolome, the da Silveira lipids).
All six are tabulated, because a target with no ceiling estimate is a target no result can be
read against, and because the context sets are the candidates for the next head.

Estimators, one per kind of repeat the source releases:

- **Variance components** where the release ships a per-item mean, standard deviation and
  count (betaxanthin). The within-item variance is pooled with ``n - 1`` weights, the
  between-item variance is the variance of the reported means less the mean of
  ``s^2 / n``, and the reliability of the stored mean is ``between / var(means)``. Cachera's
  own trial-to-trial correlation gives an independent cross-check through Spearman-Brown.
- **One-way ICC** where the release ships repeated raw measurements of the same item
  (carotenoid across batches, the Zelezniak metabolome across replicates within a protocol).
  ``ICC = (MSB - MSW) / (MSB + (n0 - 1) MSW)`` with ``n0`` the usual unbalanced-group size.
- **Retest correlation** where exactly two measurements exist per item (amino acids,
  isobutanol), computed by the two scripts named in RETEST_SOURCES and read from their JSON.
- **Dispersion back-solve** where the release ships a strictly standardized mean difference
  rather than a replicate (the da Silveira lipids): ``sigma = |log2 ratio / SSMD|`` recovers
  the within-mutant standard deviation, and the reliability follows as
  ``1 - median(sigma)^2 / var(log2 ratio)``, the median because the ratio explodes wherever
  SSMD approaches zero. This rests on the released SSMD definition, so it
  is reported as an estimate of the same kind but a weaker one.

Run the two retest scripts first; this one reads their result JSON. Run from the worktree
root::

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
RAW = osp.join(DATA_ROOT, "data", "torchcell")
CACHERA_CSV = osp.join(RAW, "betaxanthin_cachera2023", "raw", "GA1_2_4_6.csv")
OZAYDIN_XLSX = osp.join(RAW, "carotenoid_ozaydin2013", "raw", "1-s2.0-S109671761200081X-mmc1.xlsx")
ZELEZNIAK_TSV = osp.join(RAW, "metabolite_zelezniak2018", "raw", "metabolites_dataset.data_prep.tsv")
DASILVEIRA_XLSX = osp.join(RAW, "metabolite_dasilveira2014", "raw", "TableS4_complete_dataset_all_lipids.xlsx")
MULLEDER_JSON = osp.join(RESULTS, "mulleder_retest_reliability.json")
ISOBUTANOL_JSON = osp.join(RESULTS, "isobutanol_retest_reliability.json")
OUT_JSON = osp.join(RESULTS, "target_reliability.json")
OUT_TEX = osp.join(REPO_ROOT, "notes-tex", "027-metabolic-flux-module", "tables", "target_reliability.tex")
CACHERA_BASE = "corrected_mean_intensity.24"
#: Cachera's reported trial-to-trial correlation of the yellowness score, paper.md line 139:
#: "The clear linear correlation between the two trials (PCC = 0.765) demonstrates the
#: reproducibility of CRI-SPA (Figure 4A)." Four screens are combined in the release.
CACHERA_TRIAL_PCC = 0.765
CACHERA_N_SCREENS = 4
#: The control row of the CRI-SPA release, which carries no systematic gene name.
CACHERA_CONTROL_GENE = "0"


class TargetReliability(BaseModel):
    target: str
    source: str = Field(description="Dataset class the 027 build loads this target from.")
    read_by_a_head: bool = Field(description="Whether a trained head decodes this target, as opposed to it entering the build as genotype context.")
    estimator: str
    repeat_unit: str
    n_repeated: int
    n_features: int = Field(description="Number of measured features; 1 for a single-value target, more where the reliability is a median over features.")
    reliability_stored_value: float = Field(description="Reliability of the value the module trains on.")
    pearson_ceiling_stored_value: float
    reliability_single_measurement: float
    pearson_ceiling_single_measurement: float
    scale: str
    caveat: str


class ReliabilityReport(BaseModel):
    sources: dict[str, str]
    cachera_spearman_brown_cross_check: dict[str, float]
    targets: list[TargetReliability]


def icc_one_way(groups: list[np.ndarray]) -> tuple[float, int, float]:
    """One-way ICC(1,1) over unbalanced groups, with the group count and mean group size."""
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 2]
    k = len(groups)
    sizes = np.array([len(g) for g in groups], dtype=float)
    total = float(sizes.sum())
    grand = float(np.concatenate(groups).mean())
    ms_between = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1))
    ms_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups) / (total - k))
    n0 = (total - (sizes**2).sum() / total) / (k - 1)
    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
    return float(max(0.0, min(1.0, icc))), k, float(n0)


def spearman_brown(r1: float, k: int) -> float:
    return k * r1 / (1.0 + (k - 1) * r1)


def betaxanthin() -> TargetReliability:
    d = pd.read_csv(CACHERA_CSV)
    d = d[d["gene"].astype(str).str.strip() != CACHERA_CONTROL_GENE]
    m = pd.to_numeric(d[f"{CACHERA_BASE}_mean"], errors="coerce")
    s = pd.to_numeric(d[f"{CACHERA_BASE}_std"], errors="coerce")
    n = pd.to_numeric(d[f"{CACHERA_BASE}_count"], errors="coerce")
    # The paper keeps a gene only with three or more colonies: "mean yellow intensity and
    # size of gene-deletion strains with three or more colonies was used as the final yield"
    # (paper.md line 68).
    keep = m.notna() & s.notna() & n.notna() & (n >= 3)
    mm, ss, nn = m[keep].to_numpy(), s[keep].to_numpy(), n[keep].to_numpy()
    within = float(np.sum((nn - 1) * ss**2) / np.sum(nn - 1))
    var_means = float(np.var(mm))
    between = max(0.0, var_means - float(np.mean(ss**2 / nn)))
    rel_stored = between / var_means
    rel_single = between / (between + within)
    return TargetReliability(
        target="Betaxanthin",
        source="BetaxanthinCachera2023Dataset",
        read_by_a_head=True,
        estimator="variance components",
        repeat_unit=f"colonies per gene over screens 1, 2, 4 and 6 (median {int(np.median(nn))})",
        n_repeated=int(keep.sum()),
        n_features=1,
        reliability_stored_value=rel_stored,
        pearson_ceiling_stored_value=float(np.sqrt(rel_stored)),
        reliability_single_measurement=rel_single,
        pearson_ceiling_single_measurement=float(np.sqrt(rel_single)),
        scale="corrected mean colony intensity at 24 h",
        caveat="Colonies are assumed independent within a gene; a shared plate or position effect inflates this.",
    )


def carotenoid() -> TargetReliability:
    d = pd.read_excel(OZAYDIN_XLSX, sheet_name="Color scores of all deletions")
    d = d[["ORF name", "Batch", "Color"]].copy()
    d["Color"] = pd.to_numeric(d["Color"], errors="coerce")
    d = d.dropna(subset=["ORF name", "Color"])
    groups = [g["Color"].to_numpy(dtype=float) for _, g in d.groupby("ORF name") if len(g) >= 2]
    icc, k, n0 = icc_one_way(groups)
    return TargetReliability(
        target="Carotenoid",
        source="CarotenoidOzaydin2013Dataset",
        read_by_a_head=False,
        estimator="one-way ICC",
        repeat_unit=f"strains scored in more than one batch (mean {n0:.1f} scores)",
        n_repeated=k,
        n_features=1,
        reliability_stored_value=icc,
        pearson_ceiling_stored_value=float(np.sqrt(icc)),
        reliability_single_measurement=icc,
        pearson_ceiling_single_measurement=float(np.sqrt(icc)),
        scale="visual color score, 0 to 5",
        caveat="A coarse 6-level visual score, and the published second round of transformations re-screened only strains scoring 2.5 to 5, so it cannot extend this estimate.",
    )


def zelezniak_metabolome() -> TargetReliability:
    d = pd.read_csv(ZELEZNIAK_TSV, sep="\t")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d[d["value"] > 0]
    d["log10"] = np.log10(d["value"])
    per_feature: list[float] = []
    n_groups = 0
    for (_protocol, _metabolite), block in d[d["dataset"].isin([2, 3])].groupby(["dataset", "metabolite_id"]):
        groups = [g["log10"].to_numpy() for _, g in block.groupby("genotype") if len(g) >= 2]
        if len(groups) >= 5:
            icc, k, _ = icc_one_way(groups)
            per_feature.append(icc)
            n_groups = max(n_groups, k)
    rel = float(np.median(per_feature))
    return TargetReliability(
        target="Metabolome (Zelezniak)",
        source="MetaboliteZelezniak2018Dataset",
        read_by_a_head=False,
        estimator="one-way ICC, median over metabolites",
        repeat_unit="independent cultures of the same strain within protocol 2 or 3",
        n_repeated=n_groups,
        n_features=len(per_feature),
        reliability_stored_value=rel,
        pearson_ceiling_stored_value=float(np.sqrt(rel)),
        reliability_single_measurement=rel,
        pearson_ceiling_single_measurement=float(np.sqrt(rel)),
        scale="log10 abundance",
        caveat="Protocol 1, which covers all 96 genotypes, carries no replicate at all; protocols 2 and 3 re-grew only strains that varied in protocol 1, so these ICCs are measured on a deliberately variable subset.",
    )


def dasilveira_lipids() -> TargetReliability:
    log2 = pd.read_excel(DASILVEIRA_XLSX, sheet_name="Log2")
    ssmd = pd.read_excel(DASILVEIRA_XLSX, sheet_name="SSMD")
    features = [c for c in log2.columns if c not in {"Systematic Name", "Standard Name"}]
    per_feature: list[float] = []
    for c in features:
        if c not in ssmd.columns:
            continue
        x = pd.to_numeric(log2[c], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(ssmd[c], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(v) & (np.abs(v) > 0)
        if ok.sum() < 30:
            continue
        # sigma = |log2 / SSMD| explodes wherever SSMD approaches zero, which is a division
        # artifact and not a wide-spread mutant, so the within-mutant variance is taken from
        # the MEDIAN recovered sigma rather than its mean.
        sigma2 = float(np.median(np.abs(x[ok] / v[ok])) ** 2)
        var_all = float(np.var(x[ok]))
        if var_all > 0:
            per_feature.append(max(0.0, 1.0 - sigma2 / var_all))
    rel = float(np.median(per_feature))
    return TargetReliability(
        target="Lipids (da Silveira)",
        source="MetaboliteDaSilveira2014Dataset",
        read_by_a_head=False,
        estimator="dispersion back-solve from the released SSMD",
        repeat_unit="biological duplicates with up to six technical replicates, summarized per mutant",
        n_repeated=int(len(log2)),
        n_features=len(per_feature),
        reliability_stored_value=rel,
        pearson_ceiling_stored_value=float(np.sqrt(rel)),
        reliability_single_measurement=rel,
        pearson_ceiling_single_measurement=float(np.sqrt(rel)),
        scale="log2 ratio to wild type",
        caveat="No replicate is released, so the within-mutant spread is recovered from the SSMD column and inherits its definition; the weakest estimate in this table.",
    )


def amino_acids() -> TargetReliability:
    d = json.load(open(MULLEDER_JSON))
    rel = float(d["mean_reliability"])
    return TargetReliability(
        target="Amino acids",
        source="AminoAcidMulleder2016Dataset",
        read_by_a_head=True,
        estimator="retest correlation",
        repeat_unit="ORFs measured in two batches",
        n_repeated=int(d["n_orfs_measured_twice"]),
        n_features=len(d["per_amino_acid"]),
        reliability_stored_value=rel,
        pearson_ceiling_stored_value=float(d["mean_pearson_ceiling"]),
        reliability_single_measurement=rel,
        pearson_ceiling_single_measurement=float(d["mean_pearson_ceiling"]),
        scale="log10 intracellular mM",
        caveat="One culture per strain, so the stored value IS a single measurement; batch median scaling only, no dilution correction, so this is a lower bound.",
    )


def isobutanol() -> TargetReliability:
    d = json.load(open(ISOBUTANOL_JSON))
    w = d["within_first_screen"]
    rel = float(w["reliability_estimate"])
    return TargetReliability(
        target="Isobutanol biosensor",
        source="IsobutanolScreenLopez2024Dataset",
        read_by_a_head=True,
        estimator="retest correlation",
        repeat_unit="ORFs on two or more rows of the genome-wide screen",
        n_repeated=int(w["n_orfs_on_two_or_more_rows"]),
        n_features=1,
        reliability_stored_value=rel,
        pearson_ceiling_stored_value=float(w["pearson_ceiling"]),
        reliability_single_measurement=rel,
        pearson_ceiling_single_measurement=float(w["pearson_ceiling"]),
        scale="log10 fold change of biosensor fluorescence",
        caveat=(
            "Heavy tailed: 4 of 190 pairs disagree by more than a decade, and dropping them raises the "
            f"retest Pearson to {w['pairs_within_one_decade']['pearson_log10']:.2f}; the robust "
            f"reliability is {w['reliability_estimate_robust']:.2f} (ceiling {w['pearson_ceiling_robust']:.2f})."
        ),
    )


def write_tex(report: ReliabilityReport) -> None:
    lines = [
        "% notes-tex/027-metabolic-flux-module/tables/target_reliability.tex",
        "%% SOURCE: generated by experiments/027-betaxanthin-metabolic/scripts/target_reliability_table.py",
        "%% from experiments/027-betaxanthin-metabolic/results/target_reliability.json, which that",
        "%% script computes from each dataset's raw release and the two retest result files",
        "%% mulleder_retest_reliability.json and isobutanol_retest_reliability.json.",
        "% Do not edit by hand.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Reproducibility ceiling of every metabolically relevant target in the 027",
        r"  build, each from its own source's repeats. Reliability is the fraction of the variance",
        r"  in the stored value that is not measurement noise, and its square root is the highest",
        r"  Pearson correlation any predictor of the true level can reach against that stored",
        r"  value. The estimator differs by what each release ships: a variance split where it",
        r"  gives a mean, a standard deviation and a count; a one-way intraclass correlation where",
        r"  it gives repeated raw measurements; a retest correlation where it gives exactly two;",
        r"  and a back-solve from the released dispersion where it gives no replicate at all.",
        r"  Features is the number of measured quantities the reliability is a median over. The",
        r"  lower block enters the build as genotype context, with no head decoding it.}",
        r"  \label{tab:target-reliability}",
        r"  \footnotesize",
        r"  \begin{tabular}{@{}llrrrr@{}}",
        r"    \toprule",
        r"    Target & Estimator & Repeats & Features & Reliability & Ceiling \\",
        r"    \midrule",
        r"    \multicolumn{6}{@{}l@{}}{\emph{Read by a trained head}} \\",
    ]
    for t in report.targets:
        if t.read_by_a_head:
            lines.append(
                f"    {t.target} & {t.estimator} & {t.n_repeated} & {t.n_features} & "
                f"{t.reliability_stored_value:.2f} & {t.pearson_ceiling_stored_value:.2f} \\\\"
            )
    lines += [r"    \midrule", r"    \multicolumn{6}{@{}l@{}}{\emph{Present as genotype context only}} \\"]
    for t in report.targets:
        if not t.read_by_a_head:
            lines.append(
                f"    {t.target} & {t.estimator} & {t.n_repeated} & {t.n_features} & "
                f"{t.reliability_stored_value:.2f} & {t.pearson_ceiling_stored_value:.2f} \\\\"
            )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    os.makedirs(osp.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    combined = spearman_brown(CACHERA_TRIAL_PCC, CACHERA_N_SCREENS)
    report = ReliabilityReport(
        sources={
            "betaxanthin": CACHERA_CSV,
            "carotenoid": OZAYDIN_XLSX,
            "metabolome": ZELEZNIAK_TSV,
            "lipids": DASILVEIRA_XLSX,
            "amino_acids": MULLEDER_JSON,
            "isobutanol": ISOBUTANOL_JSON,
        },
        cachera_spearman_brown_cross_check={
            "reported_trial_pcc": CACHERA_TRIAL_PCC,
            "n_screens_combined": float(CACHERA_N_SCREENS),
            "implied_reliability_of_the_combined_score": combined,
            "implied_pearson_ceiling": float(np.sqrt(combined)),
        },
        targets=[
            betaxanthin(),
            amino_acids(),
            isobutanol(),
            carotenoid(),
            zelezniak_metabolome(),
            dasilveira_lipids(),
        ],
    )
    with open(OUT_JSON, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    write_tex(report)
    for t in report.targets:
        flag = "head" if t.read_by_a_head else "ctx "
        print(
            f"{flag} {t.target:24s} n={t.n_repeated:5d} feat={t.n_features:3d} "
            f"rel={t.reliability_stored_value:.3f} ceiling={t.pearson_ceiling_stored_value:.3f}  [{t.estimator}]"
        )
    print(
        f"Cachera cross-check: trial PCC {CACHERA_TRIAL_PCC} over {CACHERA_N_SCREENS} screens implies "
        f"reliability {combined:.3f}, ceiling {np.sqrt(combined):.3f}"
    )
    print(OUT_JSON)
    print(OUT_TEX)


if __name__ == "__main__":
    main()
