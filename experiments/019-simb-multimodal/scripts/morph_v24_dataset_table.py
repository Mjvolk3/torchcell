# experiments/019-simb-multimodal/scripts/morph_v24_dataset_table.py
# [[experiments.019-simb-multimodal.scripts.morph_v24_dataset_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/morph_v24_dataset_table
"""The morphology round's data, as two tables: the strain sets and the target.

Reads the two built stores under ``$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/``
(``fig3_core``: Kemmeren 2014, Sameith 2015, Ohya 2005 and the single- and double-mutant
fitness panels; ``fig3_morph_proteome``: the three expression panels, Messner 2023 and Ohya
2005), their ``processed/phenotype_label_index.json`` (which genotype records carry which
label) and their ``data_module_cache/index_seed_0.json`` (the split seed 0 partition), and
reproduces the training script's ``require_modalities`` filter by intersecting each split
with the label sets, so the per-arm train, validation and test counts are the ones the
runs printed. The target block comes from ``results/morphology_noise_ceiling.json``,
``results/morphology_feature_ceiling.csv`` and ``results/calmorph_moving_features.json``;
the reference numbers from the two covariation files and the sequence baselines on
fig3_core.

    python experiments/019-simb-multimodal/scripts/morph_v24_dataset_table.py

Writes results/morph_v24_dataset.json and notes-tex/figure-3-gate/tables/morph_v24_dataset.tex.
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import statistics as st
from typing import Any

import pandas as pd
from dotenv import load_dotenv

HERE = osp.dirname(osp.abspath(__file__))
RESULTS = osp.join(osp.dirname(HERE), "results")
REPO = osp.abspath(osp.join(HERE, "..", "..", ".."))
TABLES = osp.join(REPO, "notes-tex", "figure-3-gate", "tables")
load_dotenv(osp.join(REPO, ".env"))
DATA_ROOT = os.environ["DATA_ROOT"]
STORES = {
    "fig3_core": "Kemmeren 2014, Sameith 2015, Ohya 2005, the Costanzo 2016 and Kuzmin 2018/2020 fitness panels",
    "fig3_morph_proteome": "Kemmeren 2014, Sameith 2015, Messner 2023, Ohya 2005",
}
LABEL = {"calmorph": "morphology", "expression_log2_ratio": "expression", "protein_abundance": "proteome"}
# Arm -> (store, required labels), as gh_expr_008_arm.sh sets them.
ARMS = {
    "M_cls, M_pool": ("fig3_core", ["calmorph"]),
    "CM_expr, CM_exprperm, CMP_expr, CMP_exprperm": ("fig3_core", ["expression_log2_ratio", "calmorph"]),
    "MP_cls, CM_prot, CM_protperm": ("fig3_morph_proteome", ["protein_abundance", "calmorph"]),
}


def store_facts(tag: str) -> dict[str, Any]:
    root = osp.join(DATA_ROOT, "data/torchcell/experiments/019-simb-multimodal", tag)
    li = {k: set(v) for k, v in json.load(open(osp.join(root, "processed", "phenotype_label_index.json"))).items()}
    idx = json.load(open(osp.join(root, "data_module_cache", "index_seed_0.json")))
    every = set().union(*li.values())
    out: dict[str, Any] = {
        "genotypes": len(every),
        "by_label": {LABEL.get(k, k): len(v) for k, v in li.items()},
        "morphology_and_expression": len(li["calmorph"] & li.get("expression_log2_ratio", set())),
        "morphology_and_proteome": len(li["calmorph"] & li.get("protein_abundance", set())) if "protein_abundance" in li else None,
        "all_three": len(li["calmorph"] & li.get("expression_log2_ratio", set()) & li.get("protein_abundance", set())) if "protein_abundance" in li else None,
        "split_seed_0": {k: len(v) for k, v in idx.items()},
        "arms": {},
    }
    for arm, (store, req) in ARMS.items():
        if store != tag:
            continue
        allowed = set.intersection(*(li[r] for r in req))
        out["arms"][arm] = {k: len(set(v) & allowed) for k, v in idx.items()}
    return out


def main() -> None:
    stores = {t: store_facts(t) for t in STORES}
    noise = json.load(open(osp.join(RESULTS, "morphology_noise_ceiling.json")))
    ceil = pd.read_csv(osp.join(RESULTS, "morphology_feature_ceiling.csv")).rename(columns={"Unnamed: 0": "feature"})
    moving = json.load(open(osp.join(RESULTS, "calmorph_moving_features.json")))
    mv = ceil[ceil["feature"].isin(moving["moving"])]
    prot = json.load(open(osp.join(RESULTS, "proteome_morphology_covariation.json")))
    expr = json.load(open(osp.join(RESULTS, "expression_morphology_covariation.json")))
    base: dict[str, float] = {}
    for B in ("B2_bilinear", "B3_neighbor_average"):
        vals = []
        for f in sorted(glob.glob(osp.join(RESULTS, "morphology_baselines_split_fig3_core", "seed*.json"))):
            j = json.load(open(f))
            vals.append(j[B]["by_embedding"][j[B]["best_embedding"]]["selected_on_val"]["val_pearson_per_feature"])
        base[B] = st.mean(vals)
    target = {
        "source_mutants": noise["n_mutants"],
        "wt_replicates": noise["n_wt_replicates"],
        "base_features": noise["n_base_features"],
        "model_features": noise["n_model_features"],
        "moving_features": moving["n_moving"],
        "reliability_threshold": moving["reliability_threshold"],
        "ceiling_all_mean": float(noise["ceiling_mean_model_features"]),
        "ceiling_moving_mean": float(mv["ceiling"].mean()),
        "ceiling_moving_min": float(mv["ceiling"].min()),
        "ridge_measured_proteome_moving": prot["proteome_to_morphology"]["moving_features"]["median"],
        "ridge_measured_expression_moving": expr["expression_to_morphology"]["moving_features"]["median"],
        "sequence_baselines_all278": base,
        "normalization": "Yeo-Johnson per feature, fit on the training rows",
    }
    with open(osp.join(RESULTS, "morph_v24_dataset.json"), "w") as f:
        json.dump({"generated_by": "experiments/019-simb-multimodal/scripts/morph_v24_dataset_table.py", "stores": stores, "target": target}, f, indent=1)

    # Markdown to the terminal.
    print("| store | sources | genotypes | morphology | expression | proteome | morph+expr | morph+prot | all three | seed-0 train/val/test |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for t, s in stores.items():
        b, sp = s["by_label"], s["split_seed_0"]
        print(f"| {t} | {STORES[t]} | {s['genotypes']:,} | {b.get('morphology', 0):,} | {b.get('expression', 0):,} | {b.get('proteome', 0) or '--'} | {s['morphology_and_expression']:,} | {s['morphology_and_proteome'] or '--'} | {s['all_three'] or '--'} | {sp['train']:,} / {sp['val']} / {sp['test']} |")
    print()
    print("| arms | store | required labels | train | val | test |")
    print("|---|---|---|---|---|---|")
    for t, s in stores.items():
        for arm, c in s["arms"].items():
            print(f"| {arm} | {t} | {', '.join(LABEL[r] for r in ARMS[arm][1])} | {c['train']:,} | {c['val']} | {c['test']} |")
    print()
    print(f"target: Ohya 2005 CalMorph, {target['source_mutants']:,} mutants and {target['wt_replicates']} wild-type replicates; {target['base_features']} base features, {target['model_features']} after the degenerate drop, {target['moving_features']} moving (reliability >= {target['reliability_threshold']}); ceiling mean sqrt reliability {target['ceiling_all_mean']:.3f} (all) / {target['ceiling_moving_mean']:.3f} (moving, min {target['ceiling_moving_min']:.3f}); ridge from measured proteome {target['ridge_measured_proteome_moving']:.3f}, from measured expression {target['ridge_measured_expression_moving']:.3f} (moving, median per feature, out of fold); sequence baselines on all 278: bilinear {base['B2_bilinear']:.3f}, neighbor average {base['B3_neighbor_average']:.3f}")

    L = lambda s: s.replace("_", r"\_")
    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/morph_v24_dataset_table.py. Do not edit by hand.",
        "%% SOURCE: the two stores' processed/phenotype_label_index.json and data_module_cache/index_seed_0.json;",
        "%% results/morphology_noise_ceiling.json, morphology_feature_ceiling.csv, calmorph_moving_features.json,",
        "%% proteome_morphology_covariation.json, expression_morphology_covariation.json, morphology_baselines_split_fig3_core/",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \caption[The morphology round's strain sets]{The morphology round's strain sets. Genotype records per store"
        r" and per label, the overlaps, the split seed 0 partition, and per arm the rows left after the training"
        r" script's \texttt{require\_modalities} filter (the counts the runs print). A genotype is one deletion strain"
        r" after deduplication and aggregation; a strain measured by two panels is one record carrying both labels."
        r" \src{experiments/019-simb-multimodal/scripts/morph_v24_dataset_table.py}}",
        r"  \label{tab:morph-strains}",
        r"  \begin{tabular}{@{}lp{32mm}rrrrrrr@{}}",
        r"    \toprule",
        r"    store & sources & genotypes & morph. & expr. & prot. & morph.\ + expr. & morph.\ + prot. & all three \\",
        r"    \midrule",
    ]
    for t, s in stores.items():
        b = s["by_label"]
        lines.append(f"    \\texttt{{{L(t)}}} & {STORES[t]} & {s['genotypes']:,} & {b.get('morphology', 0):,} & {b.get('expression', 0):,} & {b.get('proteome') or '--'} & {s['morphology_and_expression']:,} & {s['morphology_and_proteome'] or '--'} & {s['all_three'] or '--'} \\\\")
    lines += [r"    \midrule", r"    \multicolumn{9}{@{}l}{\emph{Per arm, split seed 0, after the label filter (train / validation / test rows)}}\\"]
    for t, s in stores.items():
        for arm, c in s["arms"].items():
            lines.append(f"    \\multicolumn{{9}}{{@{{}}l}}{{\\texttt{{{L(arm)}}} on \\texttt{{{L(t)}}}, {', '.join(LABEL[r] for r in ARMS[arm][1])}: {c['train']:,} / {c['val']} / {c['test']}}} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", "",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \caption[The morphology target]{The morphology target and its references. CalMorph (Ohya 2005) gives one"
        r" population-averaged profile per deletion and 122 wild-type replicates; a feature's reliability is one minus"
        r" the replicate variance over the variance across mutants, its ceiling the square root, the best across-strain"
        r" Pearson a perfect predictor of the signal can reach against one noisy measurement. The round predicts the"
        r" moving features only. The ridge rows are the out-of-fold median Pearson per feature from the measured other"
        r" modality on the strains both panels measured; the sequence rows are the best key selected on validation,"
        r" four split seeds, all 278 features. \src{experiments/019-simb-multimodal/scripts/morph_v24_dataset_table.py}}",
        r"  \label{tab:morph-target}",
        r"  \begin{tabular}{@{}lr@{}}",
        r"    \toprule",
        r"    quantity & value \\",
        r"    \midrule",
        f"    deletion strains measured (Ohya 2005) & {target['source_mutants']:,} \\\\",
        f"    wild-type replicates & {target['wt_replicates']} \\\\",
        f"    base CalMorph features & {target['base_features']} \\\\",
        f"    after dropping the three degenerate & {target['model_features']} \\\\",
        f"    moving features, reliability $\\geq$ {target['reliability_threshold']} (the target) & {target['moving_features']} \\\\",
        f"    target transform & {target['normalization']} \\\\",
        f"    ceiling, mean over all {target['model_features']} features & {target['ceiling_all_mean']:.3f} \\\\",
        f"    ceiling, mean over the moving features (minimum) & {target['ceiling_moving_mean']:.3f} ({target['ceiling_moving_min']:.3f}) \\\\",
        f"    ridge from the measured proteome, moving features & {target['ridge_measured_proteome_moving']:.3f} \\\\",
        f"    ridge from measured expression, moving features & {target['ridge_measured_expression_moving']:.3f} \\\\",
        f"    sequence bilinear ridge, all 278 features & {base['B2_bilinear']:.3f} \\\\",
        f"    sequence neighbor average, all 278 features & {base['B3_neighbor_average']:.3f} \\\\",
        r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    with open(osp.join(TABLES, "morph_v24_dataset.tex"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
