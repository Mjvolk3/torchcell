# experiments/032-amino-acid-module/scripts/aa_cvae_tables.py
# [[experiments.032-amino-acid-module.scripts.aa_cvae_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/032-amino-acid-module/scripts/aa_cvae_tables

r"""The conditional VAE's grid as the document's tables, seed means from the result file.

Reads ``results/aa_conditional_vae.json`` (``aa_conditional_vae.py``), averages every
cell over its three seeds, and writes ``tables/aa_cvae.tex`` (features by grid cell:
the posterior route, the prior route, the betaxanthin head on each latent, calibration,
KL) and ``tables/aa_cvae_baselines.tex`` (the plain MLP per feature set), plus
``results/aa_cvae_summary.json`` with the same means and their standard deviations for
the verifier. Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/032-amino-acid-module/scripts/aa_cvae_tables.py
"""

from __future__ import annotations

import collections
import json
import os
import os.path as osp
import sys

import numpy as np
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "scripts"))
from notes_tex_tables import fit_columns  # noqa: E402

STUDY = osp.join(EXPERIMENT_ROOT, "032-amino-acid-module")
RESULTS = osp.join(STUDY, "results")
TABLES = osp.join(osp.dirname(EXPERIMENT_ROOT), "notes-tex", "032-amino-acid-module", "tables")
SCRIPT = "experiments/032-amino-acid-module/scripts/aa_cvae_tables.py"
FEATURE_LABEL = {
    "go": "GO",
    "gonbr": "GO neighbor mean",
    "t5": "ProtT5",
    "go+gonbr+t5": "all three",
}
CELL_KEYS = (
    "posterior_mean_spearman",
    "posterior_r2",
    "prior_mean_spearman",
    "prior_r2",
    "prior_bx_spearman",
    "posterior_bx_spearman",
    "calibration_90",
    "kl_per_strain",
    "select_epoch",
)
BASE_KEYS = ("mlp_profile_mean_spearman", "mlp_profile_r2", "mlp_bx_spearman", "select_epoch")


def summarize(rows: list[dict], keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}
    return out


def main() -> None:
    d = json.load(open(osp.join(RESULTS, "aa_conditional_vae.json")))
    cells = collections.defaultdict(list)
    for r in d["cells"]:
        cells[(r["features"], r["latent"], r["beta"], r["lambda_bx"])].append(r)
    bases = collections.defaultdict(list)
    for r in d["baselines"]:
        bases[r["features"]].append(r)
    summary = {
        "generated_by": SCRIPT,
        "source": "results/aa_conditional_vae.json",
        "n_fit": d["n_fit"],
        "n_val": d["n_val"],
        "n_pinned": d["n_pinned"],
        "n_pinned_with_betaxanthin": d["n_pinned_with_betaxanthin"],
        "n_seeds": len(d["seeds"]),
        "epochs": d["epochs"],
        "seconds": d["seconds"],
        "cells": {f"{f}|{k}|{b}|{l}": summarize(rows, CELL_KEYS) for (f, k, b, l), rows in cells.items()},
        "baselines": {f: summarize(rows, BASE_KEYS) for f, rows in bases.items()},
    }
    best = max(summary["cells"].items(), key=lambda kv: kv[1]["posterior_mean_spearman"]["mean"])
    summary["best_posterior_cell"] = best[0]
    prior = [v["prior_mean_spearman"]["mean"] for v in summary["cells"].values()]
    summary["prior_range"] = {"min": float(min(prior)), "max": float(max(prior))}
    cal = [v["calibration_90"]["mean"] for v in summary["cells"].values()]
    summary["calibration_range"] = {"min": float(min(cal)), "max": float(max(cal))}
    bxp = max(summary["cells"].items(), key=lambda kv: kv[1]["prior_bx_spearman"]["mean"])
    summary["best_prior_bx_cell"] = bxp[0]
    with open(osp.join(RESULTS, "aa_cvae_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    rows = []
    for (f, k, b, l), rs in cells.items():
        m = summary["cells"][f"{f}|{k}|{b}|{l}"]
        rows.append(
            f"    {FEATURE_LABEL[f]} & {k} & {b:g} & {l:g} & "
            f"{m['posterior_mean_spearman']['mean']:.3f} & {m['posterior_r2']['mean']:.2f} & "
            f"{m['prior_mean_spearman']['mean']:.3f} & {m['prior_r2']['mean']:.2f} & "
            f"{m['prior_bx_spearman']['mean']:.3f} & {m['posterior_bx_spearman']['mean']:.3f} & "
            f"{m['calibration_90']['mean']:.2f} & {m['kl_per_strain']['mean']:.1f} \\\\"
        )
    base_rows = [
        f"    {FEATURE_LABEL[f]} & {summary['baselines'][f]['mlp_profile_mean_spearman']['mean']:.3f} "
        f"({summary['baselines'][f]['mlp_profile_mean_spearman']['sd']:.3f}) & "
        f"{summary['baselines'][f]['mlp_profile_r2']['mean']:.2f} & "
        f"{summary['baselines'][f]['mlp_bx_spearman']['mean']:.3f} "
        f"({summary['baselines'][f]['mlp_bx_spearman']['sd']:.3f}) \\\\"
        for f in bases
    ]
    tex = (
        "\\begin{table}[H]\n  \\centering\n  \\footnotesize\n"
        "  \\caption[The conditional VAE on the profile: posterior route against prior route]{"
        "The conditional VAE (\\texttt{aa\\_conditional\\_vae.py}) on the "
        f"{d['n_strains']:,} profiled strains, {d['n_fit']:,} fitted, {d['n_val']} validation, "
        f"{d['n_pinned']} pinned held out of every fit, means over three seeds. Posterior: "
        "the pinned profile decoded from the latent that saw the strain's measured profile; "
        "prior: decoded from the latent the genotype features alone imply; both as the mean "
        "per-amino-acid Spearman and the pooled $R^2$ of the standardized $\\log(1+\\text{mM})$ "
        "profile. Betaxanthin: Spearman of the linear head on the prior and the posterior "
        f"latent against the Cachera score on the {d['n_pinned_with_betaxanthin']} pinned strains that "
        "carry it. Calibration: the fraction of true values inside the central 90 percent of "
        "64 prior samples (0.90 when calibrated). KL: the mean per-strain divergence from the "
        "posterior to the conditional prior. Epoch and cell selected on the validation "
        "prior-route mean Spearman. Written by \\texttt{aa\\_cvae\\_tables.py}.}\n"
        "  \\label{tab:aa-cvae}\n"
        "  \\begin{tabular}{@{}lrrrrrrrrrrr@{}}\n    \\hline\n"
        "    features & $k$ & $\\beta$ & $\\lambda$ & post.\\ $\\rho$ & post.\\ $R^2$ & prior $\\rho$ & "
        "prior $R^2$ & bx prior & bx post. & cal.\\ 90 & KL \\\\\n    \\hline\n"
        + "\n".join(rows) + "\n    \\hline\n  \\end{tabular}\n\\end{table}\n"
    )
    tex_b = (
        "\\begin{table}[H]\n  \\centering\n  \\footnotesize\n"
        "  \\caption[The plain MLP from the same genotype features, no latent]{A two-layer "
        "MLP of width 128 from the same standardized genotype features to the profile and, "
        "separately, to the betaxanthin score, on the same strains and split, epoch selected "
        "on the validation profile Spearman; means over three seeds with the standard "
        "deviation. Written by \\texttt{aa\\_cvae\\_tables.py}.}\n"
        "  \\label{tab:aa-cvae-baselines}\n"
        "  \\begin{tabular}{@{}lrrr@{}}\n    \\hline\n"
        "    features & profile $\\rho$ (sd) & profile $R^2$ & betaxanthin $\\rho$ (sd) \\\\\n"
        "    \\hline\n" + "\n".join(base_rows) + "\n    \\hline\n  \\end{tabular}\n\\end{table}\n"
    )
    os.makedirs(TABLES, exist_ok=True)
    for name, body in (("aa_cvae.tex", tex), ("aa_cvae_baselines.tex", tex_b)):
        with open(osp.join(TABLES, name), "w") as fh:
            fh.write(f"%% notes-tex/032-amino-acid-module/tables/{name}\n%% GENERATED by {SCRIPT}\n%% SOURCE: results/aa_conditional_vae.json\n" + fit_columns(body))
        print(osp.join(TABLES, name))
    print("best posterior", best[0], round(best[1]["posterior_mean_spearman"]["mean"], 3), round(best[1]["posterior_r2"]["mean"], 3))
    print("prior range", summary["prior_range"], "calibration range", summary["calibration_range"], "best prior bx", bxp[0], round(bxp[1]["prior_bx_spearman"]["mean"], 3), round(bxp[1]["prior_bx_spearman"]["sd"], 3))


if __name__ == "__main__":
    main()
