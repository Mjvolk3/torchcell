# experiments/019-simb-multimodal/scripts/variance_stratified_pearson.py
# [[experiments.019-simb-multimodal.scripts.variance_stratified_pearson]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/variance_stratified_pearson
"""Per-feature Pearson on the genes that move: the validation score by target-variance stratum.

`val/<pheno>/pearson_per_feature` averages one Pearson per gene over every gene the panel
measures, 6,169 transcripts or 1,850 proteins. Most of those barely change across
deletions, so their correlations are noise around zero and dilute the average, and the
headline number does not say whether the model predicts the large swings. This script
re-scores the same validation predictions on strata of genes ranked by how much they
move across TRAINING strains (per-gene sd of the log2 ratio over the train split, so the
ranking never touches the scored strains).

Predictions come from the best-validation checkpoints scored by
`gh_eval_ckpt_predictions.slurm` (train_cgt_multitask.py `trainer.eval_ckpt_path`), which
writes $DATA_ROOT/val-predictions/<run group>.json with per-record point predictions in
raw units. The linear (B2, bilinear ridge on ProtT5) and neighbor (B3, kNN on ProtT5)
baselines are refit here on the same partition so every stratum carries the same three
readers, and for the proteome the duplicate-strain ceiling per protein
(`proteome_ceiling_replicate.py` route D) is stratified the same way.

Run from the repo root once the dumps exist:
    python experiments/019-simb-multimodal/scripts/variance_stratified_pearson.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

load_dotenv()
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from expression_baselines import (  # type: ignore[import-not-found]  # noqa: E402
    EMBEDDINGS,
    _bilinear,
    _embedding_matrix,
)
from expression_baselines_split import (  # type: ignore[import-not-found]  # noqa: E402
    _load_records,
    _load_split,
    _pert_matrix,
)

from torchcell.datasets.scerevisiae.messner2023 import (  # noqa: E402
    build_uniprot_to_orf_map,
)
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

# The checkpoints scored by gh_eval_ckpt_predictions.slurm: W&B run id, arm, split seed,
# the run group the dump is named after, and the partition's dataset.
RUNS: list[dict[str, Any]] = [
    {"round": "v13", "run": "wq8y8nd5", "arm": "V_ref_s0", "seed": 0, "split_seed": 0,
     "group": "compute-0-2-2397312_3fcfb279cc58c21f926585737901556cf40cded9e6c3fc039c1b34abbc9b7248"},
    {"round": "v13", "run": "bn37i9vs", "arm": "V_concat_s0", "seed": 0, "split_seed": 0,
     "group": "compute-0-2-2397312_ecd690dcabb6286e9a2f646c3232e14167f234c7778433803e0420024f402c61"},
    {"round": "v13", "run": "lp6guytz", "arm": "V_ref_s2", "seed": 0, "split_seed": 2,
     "group": "compute-0-1-2397316_f4f3eb97310e9f0ded0ac8f43fac8569a5fa44a5d023576aeac414c734472433"},
    {"round": "v14", "run": "uc0pm2pv", "arm": "P_ref_s0", "seed": 0, "split_seed": 0,
     "group": "compute-3-3-2400351_45b093533290bd592f8cfc6d59a1f07556b3c4aa16eb56882545a02c020b8845"},
    {"round": "v14", "run": "bc8bngdr", "arm": "P_concat_s0", "seed": 0, "split_seed": 0,
     "group": "compute-3-3-2400351_190566f3daabdfe4dc73d7309269139c074951adb6e07764d3234aa0b20a9247"},
    {"round": "v14", "run": "25ok3tce", "arm": "P_ref_s2", "seed": 1, "split_seed": 2,
     "group": "compute-3-3-2401108_e4609a5c2e63232a6035fab136b7875f37697ad44dfd33241d871e14adddd99b"},
]
ROUND = {
    "v13": {"tag": "fig3_core", "label": "expression_log2_ratio", "pheno": "expression",
            "baselines": "expression_baselines_split", "name": "expression (Kemmeren)"},
    "v14": {"tag": "fig3_proteome", "label": "protein_abundance", "pheno": "proteome",
            "baselines": "baselines_split_fig3_proteome", "name": "proteome (Messner)"},
}
# Strata by percentile of the per-gene TRAIN sd: (lower, upper] in percent.
STRATA: list[tuple[str, float, float]] = [
    ("0-50", 0, 50), ("50-80", 50, 80), ("80-90", 80, 90), ("90-95", 90, 95),
    ("95-99", 95, 99), ("99-100", 99, 100),
]
TOPS: list[tuple[str, float]] = [("all", 0), ("top 50%", 50), ("top 20%", 80), ("top 10%", 90),
                                 ("top 5%", 95), ("top 1%", 99)]
RAW_DIR = "data/torchcell/proteome_messner2023/raw"
EMB = "prot_T5_all"


def per_gene_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """One Pearson per column over its finite pairs; NaN where fewer than 3 or constant."""
    ok = np.isfinite(pred) & np.isfinite(true)
    n = ok.sum(axis=0)
    p0 = np.where(ok, pred, 0.0)
    t0 = np.where(ok, true, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pm = p0.sum(0) / np.maximum(n, 1)
        tm = t0.sum(0) / np.maximum(n, 1)
        p = np.where(ok, p0 - pm, 0.0)
        t = np.where(ok, t0 - tm, 0.0)
        num = (p * t).sum(0)
        den = np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0)
        r = num / den
    r[(den <= 1e-8) | (n < 3)] = np.nan
    return np.asarray(r)


def _strata_means(r: np.ndarray, pct: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, lo, hi in STRATA:
        m = (pct > lo) & (pct <= hi) if lo > 0 else (pct <= hi)
        out[name] = float(np.nanmean(r[m]))
    for name, lo in TOPS:
        m = pct > lo if lo > 0 else np.ones_like(pct, dtype=bool)
        out[name] = float(np.nanmean(r[m]))
    return out


def _load_dump(path: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    with open(path) as fh:
        d = json.load(fh)
    head = "per_gene"
    dump_keys = list(d["head_keys"][head])
    if set(dump_keys) != set(keys):
        raise ValueError(
            f"{path}: dump names {len(dump_keys)} genes, the LMDB {len(keys)};"
            f" {len(set(dump_keys) ^ set(keys))} differ"
        )
    col = np.array([dump_keys.index(k) for k in keys])
    recs = [r for r in d["predictions"][head] if r["target"] is not None]
    pred = np.array([r["pred"] for r in recs], dtype=np.float64)[:, col]
    tgt = np.array([r["target"] for r in recs], dtype=np.float64)[:, col]
    return pred, tgt, [int(r["record_index"]) for r in recs]


def _proteome_reliability(data_root: str, keys: list[str]) -> np.ndarray:
    """Route D of proteome_ceiling_replicate.py: per-protein r between duplicate strains."""
    matrix = pd.read_csv(osp.join(data_root, RAW_DIR, "yeast5k_noimpute_wide.csv"))
    matrix = matrix.set_index("Protein.Group")
    up2orf = build_uniprot_to_orf_map()
    matrix.index = matrix.index.astype(str).map(up2orf)
    meta = pd.read_csv(osp.join(data_root, RAW_DIR, "yeast5k_metadata.csv")).set_index("Filename")
    cols = [c for c in matrix.columns if c in meta.index]
    kind = meta.loc[cols, "sampletype"]
    wt = matrix[[c for c, k in zip(cols, kind) if k == "HIS3"]]
    ko = matrix[[c for c, k in zip(cols, kind) if k == "ko"]]
    log_ko = np.log2(ko.to_numpy(dtype=float)) - np.nanmean(np.log2(wt.to_numpy(dtype=float)), 1)[:, None]
    orf = meta.loc[ko.columns, "ORF"].astype(str).to_numpy()
    first: dict[str, int] = {}
    pairs: list[tuple[int, int]] = []
    for j, o in enumerate(orf):
        if o in first:
            pairs.append((first[o], j))
        else:
            first[o] = j
    a = log_ko[:, [p[0] for p in pairs]]
    b = log_ko[:, [p[1] for p in pairs]]
    r = per_gene_pearson(a.T, b.T)
    by_orf = dict(zip(matrix.index.astype(str), r))
    return np.array([by_orf.get(k, np.nan) for k in keys])


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    img_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-simb-multimodal")
    os.makedirs(img_dir, exist_ok=True)
    results_dir = experiment_results_dir("019-simb-multimodal", __file__)

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )
    emb = _embedding_matrix(EMBEDDINGS[EMB], data_root, genome, graph)
    dim = len(next(iter(emb.values())))

    out: dict[str, Any] = {
        "generated_by": "experiments/019-simb-multimodal/scripts/variance_stratified_pearson.py",
        "stratum_variable": "per-gene sd of the log2 ratio over TRAIN strains, percentile",
        "strata": [s[0] for s in STRATA],
        "tops": [t[0] for t in TOPS],
        "entries": [],
    }
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in RUNS:
        rd = ROUND[spec["round"]]
        base = osp.join(data_root, "data/torchcell/experiments/019-simb-multimodal", rd["tag"])
        key = (spec["round"], spec["split_seed"])
        if key not in cache:
            split = _load_split(osp.join(base, "data_module_cache"), spec["split_seed"], rd["label"])
            perts_tr, y_tr, keys = _load_records(base, split["train"], rd["label"])
            perts_va, y_va, keys_va = _load_records(base, split["val"], rd["label"])
            assert keys == keys_va
            sd_tr = np.nanstd(y_tr, axis=0, ddof=1)
            pct = 100.0 * (np.argsort(np.argsort(sd_tr)) + 0.5) / len(sd_tr)
            mu = np.nanmean(y_tr, axis=0, keepdims=True)
            r_tr = np.nan_to_num(y_tr - mu, nan=0.0)
            p_tr, ok_tr = _pert_matrix(perts_tr, emb, dim)
            p_va, ok_va = _pert_matrix(perts_va, emb, dim)
            pm = p_tr[ok_tr].mean(0, keepdims=True)
            ps = p_tr[ok_tr].std(0, keepdims=True) + 1e-8
            p_tr = (p_tr - pm) / ps
            p_va = (p_va - pm) / ps
            # Baseline cells as selected on validation by expression_baselines_split.py.
            with open(osp.join(results_dir, rd["baselines"], f"seed{spec['split_seed']}.json")) as fh:
                bj = json.load(fh)
            b2 = bj["B2_bilinear"]["by_embedding"][EMB]["selected_on_val"]
            b3 = bj["B3_neighbor_average"]["by_embedding"][EMB]["selected_on_val"]
            r_hat2 = _bilinear(
                r_tr[ok_tr], np.nan_to_num(y_va - mu, nan=0.0), p_tr[ok_tr], p_va,
                int(b2["k_gene"]), float(b2["ridge"]),
            )
            e_tr = p_tr[ok_tr] / (np.linalg.norm(p_tr[ok_tr], axis=1, keepdims=True) + 1e-12)
            e_va = p_va / (np.linalg.norm(p_va, axis=1, keepdims=True) + 1e-12)
            nn = np.argsort(-(e_va @ e_tr.T), axis=1)[:, : int(b3["k"])]
            with np.errstate(invalid="ignore"):
                r_hat3 = np.nan_to_num(np.nanmean((y_tr - mu)[ok_tr][nn], axis=1), nan=0.0)
            rel = _proteome_reliability(data_root, keys) if spec["round"] == "v14" else None
            cache[key] = {
                "split": split, "keys": keys, "y_va": y_va, "pct": pct, "sd_tr": sd_tr,
                "b2": per_gene_pearson(r_hat2[ok_va] + mu, y_va[ok_va]),
                "b3": per_gene_pearson(r_hat3[ok_va] + mu, y_va[ok_va]),
                "b2_cell": b2, "b3_cell": b3, "rel": rel, "ok_va": ok_va,
            }
        c = cache[key]
        dump = osp.join(data_root, "val-predictions", f"{spec['group']}.json")
        pred, tgt, rec_idx = _load_dump(dump, c["keys"])
        # The dump's targets must be the LMDB's validation rows, in the split's order.
        if rec_idx != list(c["split"]["val"]):
            raise ValueError(f"{dump}: record order differs from the split's val indices")
        if not np.allclose(np.nan_to_num(tgt), np.nan_to_num(c["y_va"]), atol=1e-3):
            raise ValueError(f"{dump}: targets differ from the LMDB validation rows")
        r_model = per_gene_pearson(pred, tgt)
        entry = {
            **spec,
            "n_val": int(len(rec_idx)),
            "n_genes": int(len(c["keys"])),
            "all_genes_pearson": float(np.nanmean(r_model)),
            "model": _strata_means(r_model, c["pct"]),
            "B2": _strata_means(c["b2"], c["pct"]),
            "B3": _strata_means(c["b3"], c["pct"]),
            "sd_train_at_percentile": {
                str(p): float(np.percentile(c["sd_tr"], p)) for p in (50, 80, 90, 95, 99)
            },
        }
        if c["rel"] is not None:
            entry["ceiling"] = _strata_means(np.sqrt(np.clip(c["rel"], 0, 1)), c["pct"])
        out["entries"].append(entry)
        print(f"\n{spec['round']} {spec['arm']} seed {spec['seed']} ({spec['run']}): "
              f"all-gene val Pearson {entry['all_genes_pearson']:.4f}")
        print(f"  {'stratum':10s} {'model':>8s} {'B2':>8s} {'B3':>8s}" + ("  ceiling" if "ceiling" in entry else ""))
        for name, *_ in STRATA + [(t[0],) for t in TOPS]:
            line = f"  {name:10s} {entry['model'][name]:8.4f} {entry['B2'][name]:8.4f} {entry['B3'][name]:8.4f}"
            if "ceiling" in entry:
                line += f"  {entry['ceiling'][name]:7.3f}"
            print(line)

    with open(osp.join(results_dir, "variance_stratified_pearson.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none", "axes.linewidth": 0.5}
    )
    fig, axes = plt.subplots(1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["wide"]), mm_to_in(55)))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.22, wspace=0.3)
    x = np.arange(len(STRATA))
    for ax, rnd in zip(axes, ("v13", "v14")):
        ents = [e for e in out["entries"] if e["round"] == rnd]
        for i, e in enumerate(ents):
            ax.plot(x, [e["model"][s[0]] for s in STRATA], marker="o", markersize=2.5,
                    linewidth=0.8, color=PLOT_PALETTE[i], label=f"{e['arm']} seed {e['seed']}")
        e0 = ents[0]
        ax.plot(x, [e0["B2"][s[0]] for s in STRATA], marker="s", markersize=2.5, linewidth=0.8,
                linestyle="--", color=PLOT_PALETTE[4], label="B2 ridge ProtT5 (split 0)")
        ax.plot(x, [e0["B3"][s[0]] for s in STRATA], marker="^", markersize=2.5, linewidth=0.8,
                linestyle="--", color=PLOT_PALETTE[5], label="B3 kNN ProtT5 (split 0)")
        if "ceiling" in e0:
            ax.plot(x, [e0["ceiling"][s[0]] for s in STRATA], marker="D", markersize=2.5,
                    linewidth=0.8, linestyle=":", color="black", label="duplicate-strain ceiling")
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in STRATA])
        ax.set_xlabel("gene stratum, percentile of train sd")
        ax.set_ylabel("per-gene Pearson, validation")
        ax.set_title(ROUND[rnd]["name"], fontsize=6)
        ax.set_ylim(-0.1, 0.7 if rnd == "v14" else 0.5)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.4)
        ax.tick_params(which="minor", length=0)
        ax.legend(frameon=True, edgecolor="black", fancybox=False, fontsize=5, loc="upper left")
    for ax, letter in zip(axes, "ab"):
        panel_label(ax, letter)
    stem = "variance_stratified_pearson"
    for name in (f"{stem}_{timestamp()}", stem):
        fig.savefig(osp.join(img_dir, f"{name}.png"), dpi=300)
        savefig_true_size_svg(fig, osp.join(img_dir, f"{name}.svg"))
    print("wrote", osp.join(results_dir, "variance_stratified_pearson.json"))


if __name__ == "__main__":
    main()
