# experiments/019-simb-multimodal/scripts/expression_baselines_split_table.py
# [[experiments.019-simb-multimodal.scripts.expression_baselines_split_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/expression_baselines_split_table
"""LaTeX tables for the expression document from the baseline result files.

Reads, and never recomputes:
  results/expression_baselines_split/seed<k>.json and seed0_fold90.json
      (expression_baselines_split.py: B0-B3 on the CellDataModule partitions)
  results/knn_embedding_probe.json
      (knn_embedding_probe.py: the parameter-free neighbor probe per embedding)

Writes:
  notes-tex/019-simb-multimodal-expression/tables/expression_baselines_split.tex
  notes-tex/019-simb-multimodal-expression/tables/knn_embedding_probe.tex
  results/expression_baselines_split/summary.json   (the across-split means and sds)

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/expression_baselines_split_table.py
"""

from __future__ import annotations

import json
import os.path as osp
from pathlib import Path

import numpy as np

from torchcell.utils.paths import experiment_results_dir

REPO = Path(__file__).resolve().parents[3]
TABLES = REPO / "notes-tex" / "019-simb-multimodal-expression" / "tables"
SPLIT_SEEDS = [0, 1, 2, 3]
EMB_ORDER = ["prot_T5_all", "calm", "species_lm_5p_3p", "normalized_chrom_pathways"]
EMB_NAME = {
    "prot_T5_all": "ProtT5",
    "calm": "CaLM",
    "species_lm_5p_3p": "species LM",
    "normalized_chrom_pathways": "chromatin pathways",
}
KNN_ARMS = [
    "prot_T5_all",
    "esm2_t33_650M_UR50D_all",
    "calm",
    "codon_frequency",
    "normalized_chrom_pathways",
    "species_lm_5p_3p",
    "nt_5prime_3prime",
    "random_10",
    "random_100",
    "random_1000",
]
KNN_NAME = {
    "prot_T5_all": "ProtT5 (1,024)",
    "esm2_t33_650M_UR50D_all": "ESM2 650M (1,280)",
    "calm": "CaLM (768)",
    "codon_frequency": "codon frequency (64)",
    "normalized_chrom_pathways": "chromatin pathways (197)",
    "species_lm_5p_3p": "species LM, both flanks (1,536)",
    "nt_5prime_3prime": "nucleotide transformer, both flanks (5,120)",
    "random_10": "random (10)",
    "random_100": "random (100)",
    "random_1000": "random (1,000)",
}


def _f(x: float | None) -> str:
    return "--" if x is None else f"${x:+.3f}$"


def _load(results_dir: str) -> dict[str, dict]:
    out = {}
    for seed in SPLIT_SEEDS:
        with open(
            osp.join(results_dir, "expression_baselines_split", f"seed{seed}.json")
        ) as f:
            out[f"seed{seed}"] = json.load(f)
    with open(
        osp.join(results_dir, "expression_baselines_split", "seed0_fold90.json")
    ) as f:
        out["seed0_fold90"] = json.load(f)
    return out


def _cell(res: dict, base: str, emb: str, key: str) -> float | None:
    by = res[base]["by_embedding"]
    if emb not in by:
        return None
    return by[emb]["selected_on_val"].get(key)


def split_table(res: dict[str, dict]) -> tuple[str, dict]:
    rows = []
    summary: dict = {}
    for base, label in (
        ("B2_bilinear", "B2 bilinear ridge"),
        ("B3_neighbor_average", "B3 neighbor mean"),
    ):
        for emb in EMB_ORDER:
            vals = [
                _cell(res[f"seed{s}"], base, emb, "val_pearson_per_feature")
                for s in SPLIT_SEEDS
            ]
            tests = [
                _cell(res[f"seed{s}"], base, emb, "test_pearson_per_feature")
                for s in SPLIT_SEEDS
            ]
            v90 = _cell(res["seed0_fold90"], base, emb, "val_pearson_per_feature")
            va, ta = np.array(vals, float), np.array(tests, float)
            summary[f"{base}/{emb}"] = {
                "val_by_split": vals,
                "test_by_split": tests,
                "val_mean": float(va.mean()),
                "val_sd": float(va.std(ddof=1)),
                "test_mean": float(ta.mean()),
                "test_sd": float(ta.std(ddof=1)),
                "val_split0_fold90": v90,
                "fold90_minus_split0_val": float(v90 - vals[0]),
            }
            val_cells = " & ".join(_f(v) for v in vals)
            test_cells = " & ".join(_f(t) for t in tests)
            rows.append(
                f"    {label} & {EMB_NAME[emb]} & val & {val_cells} & "
                f"${va.mean():.3f} \\pm {va.std(ddof=1):.3f}$ & {_f(v90)} \\\\"
            )
            rows.append(
                f"     &  & test & {test_cells} & "
                f"${ta.mean():.3f} \\pm {ta.std(ddof=1):.3f}$ & -- \\\\"
            )
        rows.append("    \\midrule")
    rows.pop()
    n_val = [res[f"seed{s}"]["split"]["n_val"] for s in SPLIT_SEEDS]
    n_test = [res[f"seed{s}"]["split"]["n_test"] for s in SPLIT_SEEDS]
    n_train = res["seed0"]["split"]["n_train"]
    n_train90 = res["seed0_fold90"]["split"]["n_train"]
    b1 = {s: res[f"seed{s}"]["B1_no_change_val"]["nmse"] for s in SPLIT_SEEDS}
    tex = "\n".join(
        [
            "%% GENERATED by experiments/019-simb-multimodal/scripts/expression_baselines_split_table.py",
            "%% from results/expression_baselines_split/seed<k>.json and seed0_fold90.json",
            "%% (expression_baselines_split.py). Do not edit by hand.",
            "%% SOURCE: results/expression_baselines_split/*.json via expression_baselines_split_table.py",
            "\\begin{table}[htbp]",
            "  \\centering",
            "  \\footnotesize",
            "  \\caption[Linear and neighbor baselines on the model's partitions]{"
            "\\texttt{B2} and \\texttt{B3} on the four \\file{CellDataModule} partitions the v13 round "
            "trains on, \\file{pearson_per_feature} on val and on test, where rank, ridge and $k$ are chosen "
            "on val and test is read at the chosen cell. Each partition has "
            f"{n_train} training strains and {n_val[0]}, {n_val[1]}, {n_val[2]}, {n_val[3]} val "
            f"({n_test[0]}, {n_test[1]}, {n_test[2]}, {n_test[3]} test) labeled expression strains. "
            "The last column is split 0 with the test records folded into train "
            f"({n_train90} training strains, the same val set), the linear analog of the round's 90/10 arm. "
            "\\texttt{B0} is $0$ and \\texttt{B1} is $0$ on every partition by construction; \\texttt{B1}'s "
            f"val \\file{{nmse}} is {b1[0]:.3f}, {b1[1]:.3f}, {b1[2]:.3f}, {b1[3]:.3f} on splits 0 to 3. "
            "\\src{experiments/019-simb-multimodal/scripts/expression_baselines_split.py}}",
            "  \\label{tab:baselines-split}",
            "  \\setlength{\\tabcolsep}{3.5pt}",
            "  \\begin{tabular}{@{}lllrrrrrr@{}}",
            "    \\toprule",
            "     &  &  & \\multicolumn{4}{c}{split seed} & mean $\\pm$ sd & 90/10 \\\\",
            "    \\cmidrule(lr){4-7}",
            "    baseline & embedding & read & 0 & 1 & 2 & 3 &  & val \\\\",
            "    \\midrule",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return tex, summary


def knn_table(results_dir: str) -> str:
    with open(osp.join(results_dir, "knn_embedding_probe.json")) as f:
        d = json.load(f)
    rows = []
    for arm in KNN_ARMS:
        e = d["arms"][arm]["modalities"]["expression"]
        by_k = {int(k): v for k, v in e["pearson_per_feature_by_k"].items()}
        k_best = max(by_k, key=lambda k: by_k[k])
        rows.append(
            f"    {KNN_NAME[arm]} & {k_best} & ${by_k[k_best]:+.3f}$ & "
            f"${by_k[1]:+.3f}$ & ${by_k[10]:+.3f}$ & ${by_k[25]:+.3f}$ \\\\"
        )
    n_val = d["arms"]["prot_T5_all"]["modalities"]["expression"]["n_val"]
    return "\n".join(
        [
            "%% GENERATED by experiments/019-simb-multimodal/scripts/expression_baselines_split_table.py",
            "%% from results/knn_embedding_probe.json (knn_embedding_probe.py). Do not edit by hand.",
            "%% SOURCE: results/knn_embedding_probe.json via expression_baselines_split_table.py",
            "\\begin{table}[htbp]",
            "  \\centering",
            "  \\footnotesize",
            "  \\caption[The kNN embedding probe on expression]{"
            "The parameter-free neighbor probe on expression: a held-out deletion's profile predicted as "
            "the similarity-weighted mean of the profiles of its $k$ nearest other deleted genes in the "
            "embedding space named, \\file{pearson_per_feature} on the "
            f"{n_val} single-deletion validation strains of split seed 0, best $k$ over "
            "$\\{1, 3, 5, 10, 25, 50\\}$ and three fixed $k$. Dimension in parentheses. The random rows "
            "are embeddings of matched dimension drawn once, and set the floor. One-hot is undefined "
            "(every cosine similarity is zero) and is omitted. "
            "\\src{experiments/019-simb-multimodal/scripts/knn_embedding_probe.py}}",
            "  \\label{tab:knn-probe}",
            "  \\begin{tabular}{@{}lrrrrr@{}}",
            "    \\toprule",
            "    embedding (dim) & best $k$ & at best $k$ & $k{=}1$ & $k{=}10$ & $k{=}25$ \\\\",
            "    \\midrule",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def main() -> None:
    results_dir = experiment_results_dir("019-simb-multimodal", __file__)
    res = _load(results_dir)
    tex, summary = split_table(res)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "expression_baselines_split.tex").write_text(tex)
    (TABLES / "knn_embedding_probe.tex").write_text(knn_table(results_dir))
    summary_path = osp.join(results_dir, "expression_baselines_split", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "generated_by": "experiments/019-simb-multimodal/scripts/expression_baselines_split_table.py",
                "split_seeds": SPLIT_SEEDS,
                "by_baseline_embedding": summary,
            },
            f,
            indent=1,
        )
    for key, v in summary.items():
        print(
            f"{key:<42} val {v['val_mean']:.3f} +/- {v['val_sd']:.3f}   "
            f"test {v['test_mean']:.3f} +/- {v['test_sd']:.3f}   90/10 val {v['val_split0_fold90']:+.3f} "
            f"({v['fold90_minus_split0_val']:+.3f})"
        )
    print(f"wrote {TABLES / 'expression_baselines_split.tex'}")
    print(f"wrote {TABLES / 'knn_embedding_probe.tex'}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
