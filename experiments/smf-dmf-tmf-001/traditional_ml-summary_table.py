# experiments/smf-dmf-tmf-001/traditional_ml-summary_table.py
# [[experiments.smf-dmf-tmf-001.traditional_ml-summary_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/smf-dmf-tmf-001/traditional_ml-summary_table
"""Extract best-per-embedding value ± CV-std into a single reconstructable table.

The 5-fold CV std is computed only inside the plotting scripts and written into the
`deduplicated_combined_df_spearman_1e{3,4}_add_cv.csv` files (fold_val_*_mean / _std).
This script pulls those out (plus test/val point estimates) into ONE tidy table per
study, so the final Supplementary tables can be reconstructed and people can be pointed
directly to a value+std table.

Selection rule: per (dataset, model, size, embedding) keep the single config with the
highest val_spearman (validation-selected, no test leakage); report its test/val and its
CV mean±std for every metric. CV std does not exist at 1e5 (no folds were run there).

Reads existing result CSVs only -- no model refit. Outputs:
  experiments/{DS}/results/traditional_ml_summary_with_std.csv   (long, machine-readable)
and prints markdown (value±std) for the Supplementary note.
"""
import os
import os.path as osp
import numpy as np
import pandas as pd

DATASETS = {"fitness": "smf-dmf-tmf-001", "interaction": "002-dmi-tmi"}
MODELS = {"random_forest": "RF", "svr": "SVR", "elastic_net": "EN"}
SIZES = ["1e3", "1e4", "1e5"]
METRICS = ["spearman", "pearson", "r2", "mse"]
HIGHER_BETTER = {"spearman", "pearson", "r2"}
FO = [
    "random_1", "random_10", "codon_frequency", "random_100",
    "normalized_chrom_pathways", "calm", "fudt_upstream", "fudt_downstream",
    "random_1000", "prot_T5_all", "prot_T5_no_dubious",
    "esm2_t33_650M_UR50D_all", "esm2_t33_650M_UR50D_no_dubious",
    "nt_window_5979", "nt_window_three_prime_300", "nt_window_five_prime_1003",
    "one_hot_gene",
]
SHORT = {
    "esm2_t33_650M_UR50D_all": "esm2_all", "esm2_t33_650M_UR50D_no_dubious": "esm2_no_dub",
    "prot_T5_no_dubious": "prot_T5_no_dub", "nt_window_three_prime_300": "nt_3prime_300",
    "nt_window_five_prime_1003": "nt_5prime_1003", "nt_window_5979": "nt_5979",
    "normalized_chrom_pathways": "chrom_pathways", "one_hot_gene": "one_hot (6607)",
    "random_1": "random_1 (1)", "random_10": "random_10 (10)",
    "random_100": "random_100 (100)", "random_1000": "random_1000 (1000)",
}
disp = lambda e: SHORT.get(e, e)


def _source_csv(ds_dir, model, size):
    """Prefer the add_cv dedup (has fold std) at 1e3/1e4; plain dedup at 1e5."""
    base = f"experiments/{ds_dir}/results/{model}"
    add_cv = osp.join(base, f"deduplicated_combined_df_spearman_{size}_add_cv.csv")
    plain = osp.join(base, f"deduplicated_combined_df_spearman_{size}.csv")
    if osp.exists(add_cv):
        return add_cv
    if osp.exists(plain):
        return plain
    return None


def extract():
    rows = []
    for dname, ddir in DATASETS.items():
        for model in MODELS:
            for size in SIZES:
                path = _source_csv(ddir, model, size)
                if path is None:
                    continue
                df = pd.read_csv(path)
                if "cell_dataset.node_embeddings" not in df.columns:
                    continue
                df["emb"] = (
                    df["cell_dataset.node_embeddings"].astype(str)
                    .str.replace(r"[\[\]']", "", regex=True)
                )
                if "val_spearman" not in df.columns:
                    continue
                for emb, g in df.groupby("emb"):
                    gv = g.dropna(subset=["val_spearman"])
                    if gv.empty:
                        continue
                    r = gv.loc[gv["val_spearman"].idxmax()]
                    rec = {
                        "dataset": dname, "model": model, "size": size, "embedding": emb,
                        "is_pert": r.get("cell_dataset.is_pert"),
                        "aggregation": r.get("cell_dataset.aggregation"),
                        "num_params": r.get("num_params"),
                    }
                    for mt in METRICS:
                        rec[f"test_{mt}"] = r.get(f"test_{mt}", np.nan)
                        rec[f"val_{mt}"] = r.get(f"val_{mt}", np.nan)
                        rec[f"cv_{mt}_mean"] = r.get(f"fold_val_{mt}_mean", np.nan)
                        rec[f"cv_{mt}_std"] = r.get(f"fold_val_{mt}_std", np.nan)
                    rows.append(rec)
    return pd.DataFrame(rows)


def save_per_study(summary):
    for dname, ddir in DATASETS.items():
        out = osp.join("experiments", ddir, "results", "traditional_ml_summary_with_std.csv")
        sub = summary[summary["dataset"] == dname].copy()
        sub.to_csv(out, index=False)
        print(f"wrote {out}  ({len(sub)} rows)")


def _cell(mean_test, cvstd):
    if pd.isna(mean_test):
        return "—"
    s = f"{mean_test:.3f}"
    if pd.notna(cvstd):
        s += f"±{cvstd:.3f}"
    return s


def md_by_model(summary, metric="spearman"):
    """value(test) ± CV-std across sizes, one table per dataset·model."""
    blocks = []
    for dname in DATASETS:
        for model in MODELS:
            sub = summary[(summary.dataset == dname) & (summary.model == model)]
            if sub.empty:
                continue
            lines = [
                f"**{dname} · {MODELS[model]}** — test-{metric} (±CV std where available)\n",
                "| embedding (dim) | $10^3$ | $10^4$ | $10^5$ |", "|---|--:|--:|--:|",
            ]
            for e in FO:
                cells = []
                for s in SIZES:
                    r = sub[(sub["size"] == s) & (sub.embedding == e)]
                    if r.empty:
                        cells.append("—")
                    else:
                        cells.append(_cell(r[f"test_{metric}"].iloc[0], r[f"cv_{metric}_std"].iloc[0]))
                if set(cells) == {"—"}:
                    continue
                lines.append(f"| {disp(e)} | " + " | ".join(cells) + " |")
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def md_pearson_headline(summary):
    """Pearson for the ceiling argument: fitness ~0.9 vs interaction ~0.4."""
    keep = ["one_hot_gene", "random_1000", "nt_window_three_prime_300", "esm2_t33_650M_UR50D_all"]
    lines = ["**Pearson headline** (test-pearson, best-val config)\n",
             "| dataset·model | embedding | $10^3$ | $10^4$ | $10^5$ |", "|---|---|--:|--:|--:|"]
    for dname in DATASETS:
        for model in ["random_forest", "svr"]:
            for e in keep:
                cells = []
                for s in SIZES:
                    r = summary[(summary.dataset == dname) & (summary.model == model)
                                & (summary["size"] == s) & (summary.embedding == e)]
                    cells.append(_cell(r["test_pearson"].iloc[0], r["cv_pearson_std"].iloc[0]) if not r.empty else "—")
                if set(cells) == {"—"}:
                    continue
                lines.append(f"| {dname}·{MODELS[model]} | {disp(e)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    summary = extract()
    save_per_study(summary)
    print("\n================= BY MODEL ACROSS SCALE (Spearman ± CV std) =================\n")
    print(md_by_model(summary, "spearman"))
    print("\n================= PEARSON HEADLINE =================\n")
    print(md_pearson_headline(summary))


if __name__ == "__main__":
    main()


def latex_paper_table(summary):
    """Emit the paper's SI results table: test-Spearman at n=1e5, RF+SVR × {fitness,
    interaction} (Elastic-Net dropped for the paper). Bold = best per column."""
    GROUPS = [
        ("Random projection", ["random_1", "random_10", "random_100", "random_1000"]),
        ("Hand-crafted", ["codon_frequency", "normalized_chrom_pathways"]),
        ("Biological (pretrained)", ["calm", "fudt_upstream", "fudt_downstream",
            "prot_T5_all", "prot_T5_no_dubious", "esm2_t33_650M_UR50D_all",
            "esm2_t33_650M_UR50D_no_dubious", "nt_window_5979",
            "nt_window_three_prime_300", "nt_window_five_prime_1003"]),
        ("Identity", ["one_hot_gene"]),
    ]
    LBL = {"random_1": r"random ($d{=}1$)", "random_10": r"random ($d{=}10$)",
        "random_100": r"random ($d{=}100$)", "random_1000": r"random ($d{=}1000$)",
        "codon_frequency": "codon frequency", "normalized_chrom_pathways": "chromosome pathways",
        "calm": "CaLM", "fudt_upstream": "FUDT upstream", "fudt_downstream": "FUDT downstream",
        "prot_T5_all": "ProtT5", "prot_T5_no_dubious": "ProtT5 (no dubious)",
        "esm2_t33_650M_UR50D_all": "ESM2-650M", "esm2_t33_650M_UR50D_no_dubious": "ESM2-650M (no dubious)",
        "nt_window_5979": "Nucleotide Tr.\\ (5979)", "nt_window_three_prime_300": "Nucleotide Tr.\\ (3$'$ 300)",
        "nt_window_five_prime_1003": "Nucleotide Tr.\\ (5$'$ 1003)", "one_hot_gene": r"\textbf{one-hot} ($d{=}6607$)"}
    cols = [("fitness", "random_forest"), ("fitness", "svr"),
            ("interaction", "random_forest"), ("interaction", "svr")]
    val = {}
    for (ds, mdl) in cols:
        for e in [g for _, gs in GROUPS for g in gs]:
            r = summary[(summary.dataset == ds) & (summary.model == mdl)
                        & (summary["size"] == "1e5") & (summary.embedding == e)]
            val[(ds, mdl, e)] = (float(r.test_spearman.iloc[0])
                                 if len(r) and r.test_spearman.notna().iloc[0] else None)
    best = {c: max((v for e, v in [((c, k), val[(c[0], c[1], k)]) for k in [g for _, gs in GROUPS for g in gs]] if v is not None), default=None) for c in cols}
    def fmt(ds, mdl, e):
        v = val[(ds, mdl, e)]
        if v is None: return "--"
        s = f"{v:.3f}"
        return r"\textbf{%s}" % s if abs(v - best[(ds, mdl)]) < 1e-9 else s
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering\footnotesize")
    L.append(r"\caption{Classical-ML gene-representation benchmark. Test Spearman $\rho$ at "
             r"$n=10^5$ training instances, best validation-selected configuration per encoding "
             r"(random forest, RF; support-vector regression, SVR; Elastic-Net omitted). "
             r"\textbf{Bold} marks the best encoding per column. Fitness is predicted well from "
             r"gene identity alone (one-hot) and biological content adds nothing; gene "
             r"interactions are $\sim$encoding-invariant near $\rho\approx0.47$. Cross-validated "
             r"means$\pm$s.d.\ at $n=10^{3},10^{4}$ and all metrics are in "
             r"\texttt{traditional\_ml\_summary\_with\_std.csv}.}")
    L.append(r"\label{tab:classical-ml-results}")
    L.append(r"\begin{tabular}{@{}l r r r r@{}}")
    L.append(r"\toprule")
    L.append(r" & \multicolumn{2}{c}{\textbf{Fitness}} & \multicolumn{2}{c}{\textbf{Interaction}}\\")
    L.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    L.append(r"\textbf{Gene encoding (dim)} & RF & SVR & RF & SVR\\")
    L.append(r"\midrule")
    for i, (gname, genes) in enumerate(GROUPS):
        if i: L.append(r"\addlinespace[2pt]")
        L.append(r"\multicolumn{5}{@{}l}{\emph{%s}}\\" % gname)
        for e in genes:
            L.append("%s & %s & %s & %s & %s\\\\" % (LBL[e],
                fmt("fitness","random_forest",e), fmt("fitness","svr",e),
                fmt("interaction","random_forest",e), fmt("interaction","svr",e)))
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    return "\n".join(L)


if __name__ == "__main__" and "--latex" in __import__("sys").argv:
    print(latex_paper_table(extract()))


def latex_paper_table_full(summary):
    """Expanded Table S2: test Spearman at n=1e3/1e4/1e5 for RF and SVR, both datasets as
    row blocks, cells = value(test) with 5-fold CV s.d. at 1e3/1e4 (single split at 1e5)."""
    GROUPS = [
        ("Random projection", ["random_1", "random_10", "random_100", "random_1000"]),
        ("Hand-crafted", ["codon_frequency", "normalized_chrom_pathways"]),
        ("Biological", ["calm", "fudt_upstream", "fudt_downstream", "prot_T5_all",
            "prot_T5_no_dubious", "esm2_t33_650M_UR50D_all", "esm2_t33_650M_UR50D_no_dubious",
            "nt_window_5979", "nt_window_three_prime_300", "nt_window_five_prime_1003"]),
        ("Identity", ["one_hot_gene"]),
    ]
    LBL = {"random_1": r"random ($d{=}1$)", "random_10": r"random ($d{=}10$)",
        "random_100": r"random ($d{=}100$)", "random_1000": r"random ($d{=}1000$)",
        "codon_frequency": "codon freq.", "normalized_chrom_pathways": "chrom. pathways",
        "calm": "CaLM", "fudt_upstream": "FUDT up", "fudt_downstream": "FUDT down",
        "prot_T5_all": "ProtT5", "prot_T5_no_dubious": "ProtT5 (nd)",
        "esm2_t33_650M_UR50D_all": "ESM2", "esm2_t33_650M_UR50D_no_dubious": "ESM2 (nd)",
        "nt_window_5979": "NT 5979", "nt_window_three_prime_300": "NT 3$'$",
        "nt_window_five_prime_1003": "NT 5$'$", "one_hot_gene": r"\textbf{one-hot}"}

    def cell(ds, mdl, size, e):
        r = summary[(summary.dataset == ds) & (summary.model == mdl)
                    & (summary["size"] == size) & (summary.embedding == e)]
        if r.empty or pd.isna(r.test_spearman.iloc[0]):
            return "--"
        t = r.test_spearman.iloc[0]
        sd = r.cv_spearman_std.iloc[0]
        return f"{t:.3f}\\,$\\pm$\\,{sd:.3f}" if pd.notna(sd) else f"{t:.3f}"

    L = [r"\begin{table*}[t]", r"\centering\scriptsize",
         r"\caption{Classical-ML gene-representation benchmark (expanded). Test Spearman "
         r"$\rho$ at $n=10^{3},10^{4},10^{5}$ for random forest (RF) and support-vector "
         r"regression (SVR), each the validation-selected best configuration per encoding; "
         r"$\pm$ is the 5-fold cross-validation s.d.\ ($n=10^{3},10^{4}$ only; $n=10^5$ is a "
         r"single held-out split). Fitness is predicted from gene identity alone (one-hot) "
         r"with biological content adding nothing; gene interactions stay near "
         r"$\rho\approx0.47$ for every encoding. Regenerated by "
         r"\texttt{traditional\_ml-summary\_table.py}; full metrics in "
         r"\texttt{traditional\_ml\_summary\_with\_std.csv}.}",
         r"\label{tab:classical-ml-results}",
         r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{\textbf{RF}} & \multicolumn{3}{c}{\textbf{SVR}}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"\textbf{Encoding} & $10^3$ & $10^4$ & $10^5$ & $10^3$ & $10^4$ & $10^5$\\"]
    for ds, dname in [("fitness", "Fitness"), ("interaction", "Interaction")]:
        L.append(r"\midrule")
        L.append(r"\multicolumn{7}{@{}l}{\textbf{%s}}\\" % dname)
        for gname, genes in GROUPS:
            L.append(r"\addlinespace[1pt]\multicolumn{7}{@{}l}{\emph{%s}}\\" % gname)
            for e in genes:
                cells = [cell(ds, m, s, e) for m in ("random_forest", "svr")
                         for s in ("1e3", "1e4", "1e5")]
                L.append("%s & %s\\\\" % (LBL[e], " & ".join(cells)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


if __name__ == "__main__" and "--latex-full" in __import__("sys").argv:
    print(latex_paper_table_full(extract()))


def latex_bench_table(summary, metric="spearman"):
    """Table S2/S3: test {metric} at n=1e3/1e4/1e5 for RF and SVR, fitness + interaction
    row blocks. Bold = best per column within each dataset block (max for spearman/pearson/r2,
    min for mse). +- = 5-fold CV s.d. (n=1e3/1e4 only).

    Width: the tabular must fit the sn-jnl submission \\textwidth (372pt); a tabular cannot
    wrap, so column width = widest cell. MSE is therefore reported in units of 1e-3 (every
    legitimate value is 0.003-0.029, so the leading "0.00" carries no information), and the
    two SVR fits that diverge numerically are shown in scientific notation with a dagger
    instead of ~40 characters of digits. Divergence is a property of the selection rule:
    configs are chosen by val_spearman, which is scale-invariant, so a well-ranked but
    badly scale-calibrated SVR fit can still carry an enormous squared error.
    """
    higher = metric != "mse"
    # MSE is displayed as value x 1e3; 1 decimal there keeps the same significant digits
    # that 4 decimals gave on the raw value (0.0256 -> 25.6).
    scale = 1000.0 if metric == "mse" else 1.0
    dec = 1 if metric == "mse" else 3
    DIVERGED = 1.0  # raw MSE above this is a blown-up fit, not a result on the same scale
    GROUPS = [
        ("Random projection", ["random_1", "random_10", "random_100", "random_1000"]),
        ("Hand-crafted", ["codon_frequency", "normalized_chrom_pathways"]),
        ("Biological", ["calm", "fudt_upstream", "fudt_downstream", "prot_T5_all",
            "prot_T5_no_dubious", "esm2_t33_650M_UR50D_all", "esm2_t33_650M_UR50D_no_dubious",
            "nt_window_5979", "nt_window_three_prime_300", "nt_window_five_prime_1003"]),
        ("Identity", ["one_hot_gene"]),
    ]
    LBL = {"random_1": r"random ($d{=}1$)", "random_10": r"random ($d{=}10$)",
        "random_100": r"random ($d{=}100$)", "random_1000": r"random ($d{=}1000$)",
        "codon_frequency": "codon freq.", "normalized_chrom_pathways": "chrom. pathways",
        "calm": "CaLM", "fudt_upstream": "FUDT up", "fudt_downstream": "FUDT down",
        "prot_T5_all": "ProtT5", "prot_T5_no_dubious": "ProtT5 (nd)",
        "esm2_t33_650M_UR50D_all": "ESM2", "esm2_t33_650M_UR50D_no_dubious": "ESM2 (nd)",
        "nt_window_5979": "NT 5979", "nt_window_three_prime_300": "NT 3$'$",
        "nt_window_five_prime_1003": "NT 5$'$", "one_hot_gene": r"one-hot"}
    genes = [g for _, gs in GROUPS for g in gs]
    cols = [(m, s) for m in ("random_forest", "svr") for s in ("1e3", "1e4", "1e5")]

    def tval(ds, m, s, e):
        r = summary[(summary.dataset == ds) & (summary.model == m)
                    & (summary["size"] == s) & (summary.embedding == e)]
        v = r[f"test_{metric}"].iloc[0] if len(r) else np.nan
        sd = r[f"cv_{metric}_std"].iloc[0] if len(r) else np.nan
        return (v, sd)

    label_metric = {"spearman": "Spearman $\\rho$", "pearson": "Pearson $r$",
                    "mse": r"MSE ($\times 10^{-3}$)"}[metric]
    tlabel = {"spearman": "results", "pearson": "pearson", "mse": "mse"}[metric]
    # Only the MSE table has diverged fits to explain; keep the Spearman caption clean.
    dagger_note = (r" $^\dag$SVR fit diverged: configurations are selected by validation "
                   r"Spearman, which is scale-invariant, so a well-ranked fit can still be "
                   r"badly scale-calibrated and carry an enormous squared error."
                   if metric == "mse" else "")
    L = [r"\begin{table*}[t]", r"\centering\scriptsize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Classical-ML gene-representation benchmark --- test %s at "
         r"$n=10^{3},10^{4},10^{5}$ for random forest (RF) and support-vector regression "
         r"(SVR), validation-selected best configuration per encoding. \textbf{Bold} = best "
         r"encoding per column within each dataset; $\pm$ is the 5-fold CV s.d.\ "
         r"($n=10^{3},10^{4}$; $n=10^5$ single split).%s Regenerated by "
         r"\texttt{traditional\_ml-summary\_table.py}.}" % (label_metric, dagger_note),
         r"\label{tab:classical-ml-%s}" % tlabel,
         r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{\textbf{RF}} & \multicolumn{3}{c}{\textbf{SVR}}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"\textbf{Encoding} & $10^3$ & $10^4$ & $10^5$ & $10^3$ & $10^4$ & $10^5$\\"]
    for ds, dname in [("fitness", "Fitness"), ("interaction", "Interaction")]:
        best = {}
        for (m, s) in cols:
            vals = [tval(ds, m, s, e)[0] for e in genes]
            vals = [v for v in vals if pd.notna(v)]
            best[(m, s)] = (max(vals) if higher else min(vals)) if vals else None
        L.append(r"\midrule")
        L.append(r"\multicolumn{7}{@{}l}{\textbf{%s}}\\" % dname)
        for gname, gg in GROUPS:
            L.append(r"\addlinespace[1pt]\multicolumn{7}{@{}l}{\emph{%s}}\\" % gname)
            for e in gg:
                cells = []
                for (m, s) in cols:
                    v, sd = tval(ds, m, s, e)
                    if pd.isna(v):
                        cells.append("--"); continue
                    if metric == "mse" and abs(v) >= DIVERGED:
                        # Scientific notation in the table's own 1e-3 units; no s.d. (the
                        # s.d. is as blown up as the mean and would re-widen the column).
                        x = v * scale
                        ex = int(np.floor(np.log10(abs(x))))
                        cells.append(r"$%.1f{\times}10^{%d}$\,$^\dag$"
                                     % (x / 10.0 ** ex, ex))
                        continue
                    txt = f"{v * scale:.{dec}f}"
                    if pd.notna(sd):
                        txt += f"\\,$\\pm$\\,{sd * scale:.{dec}f}"
                    if best[(m, s)] is not None and abs(v - best[(m, s)]) < 1e-9:
                        txt = r"\textbf{%s}" % txt
                    cells.append(txt)
                L.append("%s & %s\\\\" % (LBL[e], " & ".join(cells)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


if __name__ == "__main__" and "--latex-bench" in __import__("sys").argv:
    s = extract()
    print("%%%% ===== SPEARMAN =====")
    print(latex_bench_table(s, "spearman"))
    print("\n%%%% ===== MSE =====")
    print(latex_bench_table(s, "mse"))


def write_paper_tables(summary):
    """Write the benchmark tables as .tex into the paper for \\input (regenerable).

    Order here fixes their numbering in the SI: Spearman, then MSE, then Pearson. (Nature style --
    Supplementary Tables are numbered from 1 with NO "S" prefix; "Table S1" is an explicitly
    incorrect form. See paper/nature-biotech/preamble.tex.) Pearson is the metric the classical-ML
    Supplementary Note quotes in prose (one-hot fitness r ~ 0.87-0.90; interactions plateau at
    r ~ 0.4), so the table lets those claims be checked directly.

    Every table emits a `%% SOURCE:` line naming this script, so that
        grep '%% SOURCE:' paper/nature-biotech/sections/*.tex
    prints the provenance of every table in the manuscript.
    """
    d = "paper/nature-biotech/sections"
    for metric, name in [("spearman", "tab-classical-ml-spearman"),
                         ("mse", "tab-classical-ml-mse"),
                         ("pearson", "tab-classical-ml-pearson")]:
        path = osp.join(d, f"{name}.tex")
        with open(path, "w") as f:
            f.write("%% SOURCE: experiments/smf-dmf-tmf-001/traditional_ml-summary_table.py\n")
            f.write("%% AUTO-GENERATED -- do not hand-edit; edits are lost on the next run.\n")
            f.write("%% Regenerate: python experiments/smf-dmf-tmf-001/"
                    "traditional_ml-summary_table.py --write-tables\n")
            f.write("%% Reads: experiments/{smf-dmf-tmf-001,002-dmi-tmi}/results/"
                    "traditional_ml_summary_with_std.csv\n")
            f.write(latex_bench_table(summary, metric) + "\n")
        print("wrote", path)


if __name__ == "__main__" and "--write-tables" in __import__("sys").argv:
    write_paper_tables(extract())
