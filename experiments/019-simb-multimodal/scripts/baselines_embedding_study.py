# experiments/019-simb-multimodal/scripts/baselines_embedding_study.py
# [[experiments.019-simb-multimodal.scripts.baselines_embedding_study]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/baselines_embedding_study
"""Which representation of the deleted gene carries its protein and mRNA response?

Reads, and never recomputes, the full embedding study written by
expression_baselines_split.py --embedding-set full on the four split partitions:
  results/baselines_split_fig3_proteome_full/seed<k>.json   (protein_abundance)
  results/expression_baselines_split_full/seed<k>.json      (expression_log2_ratio)

For every embedding, B2 (bilinear ridge from the deleted gene's embedding, rank and
ridge selected on val) and B3 (mean profile of the k nearest deleted genes in embedding
space, k selected on val), the across-seed mean and sd of the val and test per-gene
Pearson.

Writes:
  notes-tex/019-simb-multimodal-expression/tables/baselines_embedding_study.tex
  results/baselines_embedding_study.json
  $ASSET_IMAGES_DIR/019-simb-multimodal/baselines_embedding_study.{svg,png}

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/baselines_embedding_study.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
TABLES = REPO / "notes-tex" / "019-simb-multimodal-expression" / "tables"
SEEDS = [0, 1, 2, 3]
PANELS = {
    "proteome": ("baselines_split_fig3_proteome_full", "Messner proteome"),
    "expression": ("expression_baselines_split_full", "Kemmeren expression"),
}
FAMILY = {
    "prot_T5_all": "protein LM",
    "prot_T5_no_dubious": "protein LM",
    "esm2_650M_all": "protein LM",
    "esm2_650M_no_dubious": "protein LM",
    "calm": "coding sequence",
    "codon_frequency": "coding sequence",
    "species_lm_five_prime": "regulatory DNA",
    "species_lm_three_prime": "regulatory DNA",
    "species_lm_5p_3p": "regulatory DNA",
    "nt_window_5979": "regulatory DNA",
    "nt_window_5979_max": "regulatory DNA",
    "nt_window_five_prime_1003": "regulatory DNA",
    "nt_window_three_prime_300": "regulatory DNA",
    "nt_window_five_prime_5979": "regulatory DNA",
    "nt_window_three_prime_5979": "regulatory DNA",
    "nt_5prime_3prime": "regulatory DNA",
    "normalized_chrom_pathways": "graph",
    "random_1024": "control",
    "random_100": "control",
    "random_10": "control",
    "prot_T5+calm": "composite",
    "prot_T5+esm2": "composite",
    "prot_T5+species_lm_5p_3p": "composite",
    "esm2+calm": "composite",
    "esm2+species_lm_5p_3p": "composite",
    "model_stack": "composite",
    "model_stack+esm2": "composite",
    "all_sequence": "composite",
}
NAME = {
    "prot_T5_all": "ProtT5",
    "prot_T5_no_dubious": "ProtT5, no dubious",
    "esm2_650M_all": "ESM2 650M",
    "esm2_650M_no_dubious": "ESM2 650M, no dubious",
    "calm": "CaLM",
    "codon_frequency": "codon frequency",
    "species_lm_five_prime": "species LM, 5' flank",
    "species_lm_three_prime": "species LM, 3' flank",
    "species_lm_5p_3p": "species LM, both flanks",
    "nt_window_5979": "NT, 5,979 window (mean)",
    "nt_window_5979_max": "NT, 5,979 window (max)",
    "nt_window_five_prime_1003": "NT, 5' 1,003",
    "nt_window_three_prime_300": "NT, 3' 300",
    "nt_window_five_prime_5979": "NT, 5' 5,979",
    "nt_window_three_prime_5979": "NT, 3' 5,979",
    "nt_5prime_3prime": "NT, 5' 1,003 + 3' 300",
    "normalized_chrom_pathways": "chromatin pathways",
    "random_1024": "random (1,024)",
    "random_100": "random (100)",
    "random_10": "random (10)",
    "prot_T5+calm": "ProtT5 + CaLM",
    "prot_T5+esm2": "ProtT5 + ESM2",
    "prot_T5+species_lm_5p_3p": "ProtT5 + species LM",
    "esm2+calm": "ESM2 + CaLM",
    "esm2+species_lm_5p_3p": "ESM2 + species LM",
    "model_stack": "model stack (species LM, CaLM, ProtT5)",
    "model_stack+esm2": "model stack + ESM2",
    "all_sequence": "all sequence (+ NT window)",
}
FAMILY_COLOR = {
    "protein LM": PLOT_PALETTE[0],
    "coding sequence": PLOT_PALETTE[1],
    "regulatory DNA": PLOT_PALETTE[2],
    "graph": PLOT_PALETTE[3],
    "composite": PLOT_PALETTE[4],
    "control": PLOT_PALETTE[5],
}


def _load(results_dir: str, sub: str) -> dict[int, dict[str, Any]]:
    out = {}
    for s in SEEDS:
        p = osp.join(results_dir, sub, f"seed{s}.json")
        if osp.exists(p):
            with open(p) as f:
                out[s] = json.load(f)
    return out


def _stat(
    res: dict[int, dict[str, Any]], base: str, emb: str, key: str
) -> tuple[float, float, int]:
    vals = []
    for r in res.values():
        by = r[base]["by_embedding"]
        if emb in by:
            vals.append(float(by[emb]["selected_on_val"][key]))
    if not vals:
        return float("nan"), float("nan"), 0
    a = np.array(vals)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)


def _dim(res: dict[int, dict[str, Any]], emb: str) -> int | None:
    for r in res.values():
        by = r["B2_bilinear"]["by_embedding"]
        if emb in by:
            return int(by[emb]["emb_dim"])
    return None


def main() -> None:
    load_dotenv()
    results_dir = experiment_results_dir("019-simb-multimodal", __file__)
    images = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-simb-multimodal")
    os.makedirs(images, exist_ok=True)
    data = {k: _load(results_dir, sub) for k, (sub, _) in PANELS.items()}
    for k, v in data.items():
        print(f"{k}: seeds {sorted(v)}")
    embs = [
        e
        for e in FAMILY
        if any(
            e in r["B2_bilinear"]["by_embedding"]
            for v in data.values()
            for r in v.values()
        )
    ]

    summary: dict[str, Any] = {
        "generated_by": "experiments/019-simb-multimodal/scripts/baselines_embedding_study.py",
        "seeds": {k: sorted(v) for k, v in data.items()},
        "embeddings": {},
    }
    for e in embs:
        summary["embeddings"][e] = {
            "family": FAMILY[e],
            "dim": _dim(data["proteome"], e) or _dim(data["expression"], e),
        }
        for k in PANELS:
            for base, short in (("B2_bilinear", "B2"), ("B3_neighbor_average", "B3")):
                for key, kk in (
                    ("val_pearson_per_feature", "val"),
                    ("test_pearson_per_feature", "test"),
                ):
                    m, sd, n = _stat(data[k], base, e, key)
                    summary["embeddings"][e][f"{k}_{short}_{kk}"] = {
                        "mean": m,
                        "sd": sd,
                        "n": n,
                    }
    with open(osp.join(results_dir, "baselines_embedding_study.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # order by the proteome B3 val mean, the strongest baseline on the panel that asked
    def key(e: str) -> float:
        v = summary["embeddings"][e]["proteome_B3_val"]["mean"]
        return -v if np.isfinite(v) else 1.0

    order = sorted(embs, key=key)

    # ---- table ---------------------------------------------------------------------
    def cell(e: str, k: str, short: str) -> str:
        v = summary["embeddings"][e][f"{k}_{short}_val"]
        t = summary["embeddings"][e][f"{k}_{short}_test"]
        if not np.isfinite(v["mean"]):
            return "-- & --"
        return f"${v['mean']:.3f} \\pm {v['sd']:.3f}$ & ${t['mean']:.3f}$"

    rows = []
    for e in order:
        d = summary["embeddings"][e]["dim"]
        rows.append(
            f"    {NAME[e]} & {FAMILY[e]} & {d:,} & "
            + " & ".join(
                cell(e, k, b) for k in ("proteome", "expression") for b in ("B2", "B3")
            )
            + " \\\\"
        )
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / "baselines_embedding_study.tex", "w") as f:
        f.write(
            "%% GENERATED by experiments/019-simb-multimodal/scripts/baselines_embedding_study.py\n"
            "%% from results/baselines_split_fig3_proteome_full/ and results/expression_baselines_split_full/.\n"
            "%% Do not edit by hand.\n"
            "%% SOURCE: results/baselines_embedding_study.json via baselines_embedding_study.py\n"
            "\\begin{table}[htbp]\n  \\centering\n  \\footnotesize\n"
            "  \\caption[Every gene representation as the baseline's perturbation input]{Every "
            "gene representation the builder serves, as the perturbation input of the two "
            "embedding baselines on the four split partitions of the proteome and of the "
            "expression build. B2 is the bilinear ridge from the deleted gene's embedding "
            "(rank and ridge selected on val), B3 the mean profile of the $k$ nearest deleted "
            "genes in embedding space ($k$ selected on val). Val is the mean $\\pm$ sd over "
            "the four seeds of the per-gene Pearson; test is the mean at the val-selected "
            "cell. Ordered by the proteome's B3 val mean. "
            "\\src{experiments/019-simb-multimodal/scripts/baselines_embedding_study.py}}\n"
            "  \\label{tab:baselines-embedding-study}\n"
            "  \\setlength{\\tabcolsep}{2.5pt}\n"
            "  \\begin{tabular}{@{}p{3.5cm}p{1.6cm}r rr rr rr rr@{}}\n"
            "    \\toprule\n"
            "    & & & \\multicolumn{4}{c}{proteome} & \\multicolumn{4}{c}{expression} \\\\\n"
            "    \\cmidrule(lr){4-7} \\cmidrule(lr){8-11}\n"
            "    representation & family & dim & B2 val & test & B3 val & test & B2 val & test & B3 val & test \\\\\n"
            "    \\midrule\n"
        )
        f.write("\n".join(rows) + "\n")
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")

    # ---- figure ----------------------------------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "axes.titlesize": 6,
            "legend.fontsize": 6,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(95)), sharey=True
    )
    y = np.arange(len(order))
    for ax, (k, (_, title)), letter in zip(axes, PANELS.items(), "ab"):
        for i, e in enumerate(order):
            c = FAMILY_COLOR[FAMILY[e]]
            for base, marker, face in (
                ("B3", "o", c),
                ("B2", "s", PLOT_PALETTE_FILL[PLOT_PALETTE.index(c)]),
            ):
                v = summary["embeddings"][e][f"{k}_{base}_val"]
                if not np.isfinite(v["mean"]):
                    continue
                ax.errorbar(
                    v["mean"],
                    i,
                    xerr=v["sd"],
                    fmt=marker,
                    ms=3.2,
                    mfc=face,
                    mec="black",
                    mew=0.4,
                    ecolor="black",
                    elinewidth=0.5,
                    capsize=1.5,
                )
        ax.axvline(0, color="black", lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([NAME[e] for e in order], fontsize=5)
        ax.set_xlabel("val Pearson per gene, mean over four partitions (bar = sd)")
        ax.set_title(f"{title}: circle B3 neighbor mean, square B2 ridge")
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.grid(axis="x", lw=0.3, color="0.85")
        ax.set_axisbelow(True)
        panel_label(ax, letter)
    axes[0].invert_yaxis()  # shared y: invert once, best row at the top
    handles = [
        plt.Line2D(
            [], [], marker="o", ls="none", mfc=c, mec="black", mew=0.4, ms=4, label=f
        )
        for f, c in FAMILY_COLOR.items()
    ]
    axes[1].legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
    )
    fig.subplots_adjust(left=0.24, right=0.99, top=0.9, bottom=0.1, wspace=0.08)
    for stem in (
        osp.join(images, f"baselines_embedding_study_{timestamp()}"),
        osp.join(images, "baselines_embedding_study"),
    ):
        fig.savefig(stem + ".png", dpi=300)
        savefig_true_size_svg(fig, stem + ".svg")
    print("wrote", TABLES / "baselines_embedding_study.tex")


if __name__ == "__main__":
    main()
