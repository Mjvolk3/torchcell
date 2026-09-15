# experiments/019-simb-multimodal/scripts/expression_ceiling_all.py
# [[experiments.019-simb-multimodal.scripts.expression_ceiling_all]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/expression_ceiling_all
"""The predictive ceiling of every gene-level readout panel, from its own reproducibility.

The metric the strand ranks on is the per-gene Pearson across held-out strains between a
prediction and the measured value. Write one measurement of gene g as x = s + e, signal
variance sigma_s^2 and noise variance sigma_e^2; a perfect predictor of s scores

    ceiling_g = corr(s, s + e) = sqrt(sigma_s^2 / (sigma_s^2 + sigma_e^2)) = sqrt(rel_g),

so the ceiling is the square root of the reliability, averaged over genes the way the
metric averages. Two estimators of rel_g are used, the same two as
expression_ceiling_replicate.py for Kemmeren:

  test-retest     rel_g = r_g, the Pearson across strains between two independent
                  measurements of the same strains (needs paired profiles);
  decomposition   rel_g = 1 - sigma_e,g^2 / sigma_total,g^2, noise variance from replicate
                  measurements of one genotype, total variance across the whole panel
                  (needs replicate noise but no paired panel).

Per panel:

  Kemmeren 2014 and Sameith 2015    read from results/expression_ceiling_replicate.json:
                                    the 82 deletions both studies profiled are the
                                    test-retest pair (route B there is the decomposition).
  Caudal 2024 natural isolates      the mirrored replicate table (60 samples, 29 isolates
                                    re-cultured independently) gives both estimators on
                                    log2(TPM + 1); total variance from the 943-isolate LMDB.
  Messner 2023 knockout proteome    test-retest from the 146 ORFs present as strains of
                                    different origin (results/proteome_messner_replicates
                                    .json of 028); decomposition from the 388 HIS3
                                    replicates whose per-protein SE the converted
                                    fig3_proteome records carry on the log2 scale.
  Zelezniak 2018 kinase proteome    decomposition from the per-strain SE over 3 biological
                                    replicates that the LMDB stores (the target is the
                                    3-replicate mean, so its noise variance is se^2).
  Nadal-Ribelles 2025 single cell   read from results/nadal_replication_coverage.json of
                                    028: same-genotype pseudobulk profiles across batches
                                    against the null of different genotypes.

The paper-reported reproducibility statement of each panel is carried verbatim with its
citation key and paper.md line, so the table says what the authors claimed and what the
data shows.

Writes results/expression_ceiling_all.json and the document table
notes-tex/019-simb-multimodal-expression/tables/expression_ceiling_all.tex.

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/expression_ceiling_all.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from torchcell.utils.paths import experiment_results_dir, experiment_root  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
CAUDAL_REPLICATES = osp.join(
    DATA_ROOT,
    "torchcell-library/caudalPantranscriptomeRevealsLarge2024/data/"
    "replicate_data_tpm_22042023.tab",
)
CAUDAL_LMDB = osp.join(
    DATA_ROOT, "data/torchcell/caudal_pantranscriptome2024/processed/lmdb"
)
ZELEZNIAK_LMDB = osp.join(
    DATA_ROOT, "data/torchcell/proteome_zelezniak2018/processed/lmdb"
)
FIG3_PROTEOME_LMDB = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/019-simb-multimodal/fig3_proteome/processed/lmdb",
)
MIN_PAIRS = 3  # a per-gene correlation over fewer points is not a number

# Verbatim statements, with the mirrored paper and the line of its paper.md.
REPORTED: dict[str, dict[str, str]] = {
    "kemmeren2014": {
        "key": "kemmerenLargeScaleGeneticPerturbations2014",
        "line": "130",
        "quote": (
            "Each mutant strain (Table S1) was grown twice, from two independently "
            "inoculated cultures. Cultures were harvested early during exponential "
            "growth in SC medium with 2% glucose. Each culture was expression-profiled "
            "in technical replicate to yield four measurements for each profiling mutant."
        ),
        "summary": "four measurements per mutant (two cultures, dye swap); no replicate correlation reported",
    },
    "sameith2015": {
        "key": "sameithHighresolutionGeneExpression2015",
        "line": "119",
        "quote": (
            "Single mutants differing from WT as well as all double mutants were "
            "profiled another two times from an independently inoculated culture. The "
            "reported FC is then the average of four replicate mutant expression "
            "profiles versus the average of all WTs."
        ),
        "summary": "four replicate profiles per mutant; no replicate correlation reported for expression",
    },
    "caudal2024": {
        "key": "caudalPantranscriptomeRevealsLarge2024",
        "line": "25",
        "quote": (
            "We performed independent culture replicates for 29 samples (Supplementary "
            "Fig. 1b). The data are highly reproducible, with an average correlation of "
            "0.94 between replicates (Supplementary Fig. 1c) and robust to different "
            "sample batches (Supplementary Fig. 1d)."
        ),
        "summary": "29 isolates re-cultured; average replicate correlation 0.94 (per sample, across genes)",
    },
    "messner2023": {
        "key": "messnerProteomicLandscapeGenomewide2023",
        "line": "579",
        "quote": (
            "Strains were not measured in replicates. However, for 145 ORFs, more than "
            "one strain exists in the library (these strains have different origins). "
            "141 gene deletions are duplicated and 4 triplicated."
        ),
        "summary": "one proteome per strain; median protein CV 8.1% technical, 11.3% across 388 WT replicates, 16.2% across knockouts (paper.md line 81)",
    },
    "zelezniak2018": {
        "key": "zelezniakMachineLearningPredicts2018",
        "line": "78",
        "quote": (
            "The coefficient of variation (CV) of enzymes at whole-process technical and "
            "biological levels. Cyan dots indicate CVs of a standardized proteome digest "
            "(quality control [QC] sample) that was used to monitor instrument "
            "performance over a 4-month acquisition period."
        ),
        "summary": "three biological replicates per kinase deletion against a 12-replicate wild type; CVs shown as a figure, no number in the text",
    },
    "nadal2025": {
        "key": "nadal-ribellesSinglecellResolvedGenotypephenotype2025",
        "line": "27",
        "quote": (
            "Despite substantial methodological differences, our dataset showed a "
            "consistent correlation between the number of differentially expressed "
            "genes per genotype in both studies (Supplementary Fig. 1i)."
        ),
        "summary": "no genotype-level replicate correlation reported; validation is the count of differentially expressed genes against Kemmeren",
    },
}


def _ceiling_from_rel(rel: np.ndarray) -> dict[str, float]:
    rel = rel[np.isfinite(rel)]
    c = np.sqrt(np.clip(rel, 0.0, 1.0))
    return {
        "n_genes": int(len(rel)),
        "mean_rel": float(rel.mean()),
        "median_rel": float(np.median(rel)),
        "ceiling_mean_sqrt": float(c.mean()),
        "ceiling_median": float(np.median(c)),
        "ceiling_p25": float(np.percentile(c, 25)),
        "ceiling_p75": float(np.percentile(c, 75)),
        "frac_rel_below_0": float((rel < 0).mean()),
    }


def _paired_pearson_per_gene(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column Pearson between two [n, G] matrices over their finite pairs."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(axis=0)
    x0 = np.where(ok, x, 0.0)
    y0 = np.where(ok, y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = x0.sum(axis=0) / np.maximum(n, 1)
        ym = y0.sum(axis=0) / np.maximum(n, 1)
        xc = np.where(ok, x0 - xm, 0.0)
        yc = np.where(ok, y0 - ym, 0.0)
        r = (xc * yc).sum(axis=0) / (
            np.linalg.norm(xc, axis=0) * np.linalg.norm(yc, axis=0)
        )
    r[n < MIN_PAIRS] = np.nan
    return r


def caudal() -> dict[str, Any]:
    # Comma-separated despite the .tab suffix (the header holds no tab).
    df = pd.read_csv(CAUDAL_REPLICATES, sep=",", low_memory=False)
    df["log2tpm"] = np.log2(df["tpm"].astype(float) + 1.0)
    mat = df.pivot_table(
        index="SampleID", columns="ORF", values="log2tpm", aggfunc="first"
    )
    strain_of = (
        df[["SampleID", "Standardized.name"]].drop_duplicates().set_index("SampleID")
    )["Standardized.name"]
    pairs: list[tuple[str, str]] = []
    for strain, ids in strain_of.groupby(strain_of).groups.items():
        ids = sorted(ids)
        if len(ids) >= 2:
            pairs.append((ids[0], ids[1]))
    a = mat.loc[[p[0] for p in pairs]].to_numpy(float)
    b = mat.loc[[p[1] for p in pairs]].to_numpy(float)
    genes = list(mat.columns)
    # per sample: the number the paper reports (correlation across genes)
    per_sample = [
        float(np.corrcoef(a[i], b[i])[0, 1])
        for i in range(len(pairs))
        if np.isfinite(a[i]).all() and np.isfinite(b[i]).all()
    ]
    # test-retest per gene across the paired isolates
    r_gene = _paired_pearson_per_gene(a, b)
    # decomposition: noise from the pairs, total across the 943 built isolates
    noise_var = np.nanvar(a - b, axis=0, ddof=1) / 2.0
    env = lmdb.open(CAUDAL_LMDB, readonly=True, lock=False)
    rows: list[dict[str, float]] = []
    with env.begin() as txn:
        for _, v in txn.cursor():
            rec = pickle.loads(v)
            rows.append(rec["experiment"]["phenotype"]["expression_tpm"])
    env.close()
    gene_idx = {g: i for i, g in enumerate(genes)}
    total = np.full((len(rows), len(genes)), np.nan)
    for i, row in enumerate(rows):
        for g, val in row.items():
            j = gene_idx.get(g)
            if j is not None and val is not None:
                total[i, j] = np.log2(float(val) + 1.0)
    n_iso = np.isfinite(total).sum(axis=0)
    total_var = np.nanvar(total, axis=0, ddof=1)
    rel_dec = 1.0 - noise_var / total_var
    keep = (n_iso >= 100) & np.isfinite(noise_var) & (total_var > 1e-8)
    return {
        "n_isolates_in_lmdb": int(len(rows)),
        "n_replicate_samples": int(mat.shape[0]),
        "n_replicate_pairs": int(len(pairs)),
        "n_genes_in_replicate_table": int(len(genes)),
        "per_sample_r_mean": float(np.mean(per_sample)),
        "per_sample_r_median": float(np.median(per_sample)),
        "test_retest": _ceiling_from_rel(r_gene[keep]),
        "decomposition": _ceiling_from_rel(rel_dec[keep]),
    }


def messner_decomposition() -> dict[str, Any]:
    env = lmdb.open(FIG3_PROTEOME_LMDB, readonly=True, lock=False)
    vals: list[dict[str, float]] = []
    ref_se: dict[str, float] | None = None
    ref_n: dict[str, int] | None = None
    with env.begin() as txn:
        for _, v in txn.cursor():
            recs = json.loads(v.decode())
            if isinstance(recs, dict):
                recs = [recs]
            for r in recs:
                ph = r["experiment"]["phenotype"]
                if ph["label_name"] != "protein_abundance":
                    continue
                vals.append(ph["protein_abundance"])
                if ref_se is None:
                    pr = r["experiment_reference"]["phenotype_reference"]
                    ref_se = pr["protein_abundance_se"]
                    ref_n = pr["n_replicates"]
    env.close()
    assert ref_se is not None and ref_n is not None
    prots = sorted(ref_se)
    m = np.array([[row[p] for p in prots] for row in vals], dtype=float)
    # the SE is on the log2 scale (delta method at build time); the noise variance of ONE
    # knockout measurement is the wild-type replicate variance sd^2 = n * se^2
    sd2 = np.array([ref_n[p] * ref_se[p] ** 2 for p in prots])
    total_var = np.nanvar(m, axis=0, ddof=1)
    rel = 1.0 - sd2 / total_var
    out = _ceiling_from_rel(rel)
    out["n_strains"] = int(m.shape[0])
    out["wt_replicates"] = int(ref_n[prots[0]])
    # the paper's CVs as a cross-check: 1 - (CV_wt / CV_ko)^2
    cv_rel = 1.0 - (0.113 / 0.162) ** 2
    out["paper_cv_check"] = {
        "cv_wt": 0.113,
        "cv_ko": 0.162,
        "rel": cv_rel,
        "ceiling": float(np.sqrt(cv_rel)),
    }
    return out


def zelezniak() -> dict[str, Any]:
    env = lmdb.open(ZELEZNIAK_LMDB, readonly=True, lock=False)
    means: list[dict[str, float]] = []
    ses: list[dict[str, float]] = []
    n_rep: set[int] = set()
    with env.begin() as txn:
        for _, v in txn.cursor():
            rec = pickle.loads(v)
            ph = rec["experiment"]["phenotype"]
            means.append(ph["protein_abundance"])
            ses.append(ph["protein_abundance_se"])
            n_rep.update(ph["n_replicates"].values())
    env.close()
    prots = sorted(set.intersection(*[set(d) for d in means]))
    m = np.array([[row[p] for p in prots] for row in means], dtype=float)
    se = np.array([[row[p] for p in prots] for row in ses], dtype=float)
    # the stored target is the replicate MEAN, whose noise variance is se^2
    noise = np.nanmean(se**2, axis=0)
    total_var = np.nanvar(m, axis=0, ddof=1)
    rel = 1.0 - noise / total_var
    out = _ceiling_from_rel(rel)
    out["n_strains"] = int(m.shape[0])
    out["n_replicates_per_strain"] = sorted(n_rep)
    return out


def main() -> None:
    res_dir = experiment_results_dir("019-simb-multimodal", __file__)
    with open(osp.join(res_dir, "expression_ceiling_replicate.json")) as f:
        kem = json.load(f)
    res_028 = osp.join(experiment_root(__file__), "028-knockout-expression", "results")
    with open(osp.join(res_028, "proteome_messner_replicates.json")) as f:
        mes_rep = json.load(f)
    with open(osp.join(res_028, "nadal_replication_coverage.json")) as f:
        nadal = json.load(f)

    out: dict[str, Any] = {
        "generated_by": "experiments/019-simb-multimodal/scripts/expression_ceiling_all.py",
        "definition": "ceiling_g = sqrt(rel_g); test-retest rel_g = r_g between two independent measurements across strains; decomposition rel_g = 1 - noise_var_g / total_var_g; panel ceiling = mean_g sqrt(rel_g) over genes with rel clipped to [0, 1]",
        "reported": REPORTED,
        "kemmeren2014": {
            "source": "results/expression_ceiling_replicate.json",
            "n_paired_deletions": kem["primary_ceiling_mean_sqrt_r"]["n_paired_strains"],
            "n_genes": kem["route_b_cross_study"]["n_genes"],
            "test_retest_r_per_gene_mean": kem["cross_study_test_retest"]["per_feature_mean_r"],
            "test_retest_r_per_strain_mean": kem["cross_study_test_retest"]["per_instance_mean_r"],
            "ceiling_test_retest": kem["primary_ceiling_mean_sqrt_r"]["ceiling"],
            "ceiling_decomposition": kem["route_b_cross_study"]["mean_ceiling"],
        },
        "sameith2015": {
            "source": "the same 82-deletion pair as kemmeren2014 (results/expression_ceiling_replicate.json)",
            "ceiling_test_retest": kem["primary_ceiling_mean_sqrt_r"]["ceiling"],
        },
        "caudal2024": caudal(),
        "messner2023": {
            "test_retest_duplicate_origin_strains": {
                "source": "028 results/proteome_messner_replicates.json",
                "n_duplicated_orfs": mes_rep["n_duplicated_orfs"],
                "n_proteins": mes_rep["n_proteins_scored"],
                "r_per_protein_mean": mes_rep["per_protein_mean_r"],
                "r_per_protein_median": mes_rep["per_protein_median_r"],
                "r_per_strain_median": mes_rep["per_strain_median_r"],
                "ceiling": mes_rep["ceiling_mean_sqrt_r"],
            },
            "decomposition_wt_replicates": messner_decomposition(),
        },
        "zelezniak2018": zelezniak(),
        "nadal2025": {
            "source": "028 results/nadal_replication_coverage.json",
            "cross_batch_same_genotype_median_r": nadal["cross_batch"][
                "same_genotype_median_r_batch_ref"
            ],
            "cross_batch_null_median_r": nadal["cross_batch"]["null_median_r_batch_ref"],
            "split_half_median_r": nadal["split_half"]["median_r"],
            "split_half_null_median_r": nadal["split_half"]["null_median_r"],
            "n_cross_batch_pairs": nadal["cross_batch"]["n_pairs"],
            "ceiling_sqrt_cross_batch_r": float(
                np.sqrt(max(nadal["cross_batch"]["same_genotype_median_r_batch_ref"], 0))
            ),
            "ceiling_sqrt_null": float(
                np.sqrt(max(nadal["cross_batch"]["null_median_r_batch_ref"], 0))
            ),
        },
    }
    dst = osp.join(res_dir, "expression_ceiling_all.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    for k in ("kemmeren2014", "caudal2024", "messner2023", "zelezniak2018", "nadal2025"):
        print(k, json.dumps({kk: vv for kk, vv in out[k].items() if kk != "source"})[:400])
    write_table(out)


def _f(x: float) -> str:
    return f"{x:.2f}"


def write_table(out: dict[str, Any]) -> None:
    kem = out["kemmeren2014"]
    cau = out["caudal2024"]
    mes_tr = out["messner2023"]["test_retest_duplicate_origin_strains"]
    mes_dc = out["messner2023"]["decomposition_wt_replicates"]
    zel = out["zelezniak2018"]
    nad = out["nadal2025"]
    rows = [
        (
            "Kemmeren 2014 (mRNA, 1,484 deletions)",
            "2 cultures $\\times$ dye swap",
            "no $r$ reported",
            f"{_f(kem['test_retest_r_per_gene_mean'])} ({kem['n_paired_deletions']} deletions shared with Sameith)",
            _f(kem["ceiling_test_retest"]),
            _f(kem["ceiling_decomposition"]),
        ),
        (
            "Sameith 2015 (mRNA, 82 + 72 deletions)",
            "4 profiles per mutant",
            "no $r$ reported",
            f"{_f(kem['test_retest_r_per_gene_mean'])} (the same pair)",
            _f(kem["ceiling_test_retest"]),
            "--",
        ),
        (
            f"Caudal 2024 (mRNA, {cau['n_isolates_in_lmdb']} isolates)",
            f"{cau['n_replicate_pairs']} isolates re-cultured",
            f"0.94 per sample (ours {_f(cau['per_sample_r_mean'])})",
            f"{_f(cau['test_retest']['mean_rel'])} ({cau['n_replicate_pairs']} pairs)",
            _f(cau["test_retest"]["ceiling_mean_sqrt"]),
            _f(cau["decomposition"]["ceiling_mean_sqrt"]),
        ),
        (
            f"Messner 2023 (protein, {mes_dc['n_strains']:,} deletions)",
            f"none; {mes_dc['wt_replicates']} WT replicates",
            "CV 11.3\\% WT, 16.2\\% KO",
            f"{_f(mes_tr['r_per_protein_mean'])} ({mes_tr['n_duplicated_orfs']} duplicated-origin ORFs)",
            _f(mes_tr["ceiling"]),
            f"{_f(mes_dc['ceiling_mean_sqrt'])} (CV check {_f(mes_dc['paper_cv_check']['ceiling'])})",
        ),
        (
            f"Zelezniak 2018 (protein, {zel['n_strains']} kinase deletions)",
            f"{min(zel['n_replicates_per_strain'])} to {max(zel['n_replicates_per_strain'])} cultures per strain",
            "CVs as a figure",
            "--",
            "--",
            _f(zel["ceiling_mean_sqrt"]),
        ),
        (
            "Nadal-Ribelles 2025 (single cell, 2,243 deletions)",
            "batches, no replicate design",
            "no $r$ reported",
            f"{_f(nad['cross_batch_same_genotype_median_r'])} across batches (null {_f(nad['cross_batch_null_median_r'])})",
            f"{_f(nad['ceiling_sqrt_cross_batch_r'])} (null {_f(nad['ceiling_sqrt_null'])})",
            "--",
        ),
    ]
    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/expression_ceiling_all.py",
        "%% from results/expression_ceiling_all.json. Do not edit by hand.",
        "%% SOURCE: results/expression_ceiling_all.json via expression_ceiling_all.py",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[The ceiling of every panel]{The per-gene Pearson a perfect predictor "
        "could reach on each panel, from that panel's own reproducibility. Test-retest "
        "$r$ is the per-gene Pearson across strains between two independent measurements "
        "of the same strains, averaged over genes; the ceiling is the mean over genes of "
        "$\\sqrt{r_g}$ (test-retest) or of $\\sqrt{1 - \\sigma_{e,g}^2 / \\sigma_{g}^2}$ "
        "(decomposition, noise variance from replicates of one genotype, total variance "
        "across the panel), reliabilities clipped to $[0, 1]$. Messner's test-retest "
        "pairs are strains of different origin carrying the same deletion, not replicate "
        "cultures, so that column is a lower bound; its decomposition takes the "
        "single-measurement noise to be the wild-type replicate variance. Nadal-Ribelles' "
        "same-genotype agreement across batches is read against the median of different "
        "genotypes. \\src{experiments/019-simb-multimodal/scripts/expression_ceiling_all.py}}",
        "  \\label{tab:ceiling-all}",
        "  \\setlength{\\tabcolsep}{3.5pt}",
        "  \\begin{tabular}{@{}p{3.6cm}p{2.3cm}p{2.4cm}p{3.6cm}p{2.0cm}p{2.4cm}@{}}",
        "    \\toprule",
        "    panel & replicate design & reported & test-retest $r$ (ours) & ceiling, test-retest & ceiling, decomposition \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    dst = osp.join(
        experiment_root(__file__),
        "..",
        "notes-tex",
        "019-simb-multimodal-expression",
        "tables",
        "expression_ceiling_all.tex",
    )
    dst = osp.abspath(dst)
    with open(dst, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
