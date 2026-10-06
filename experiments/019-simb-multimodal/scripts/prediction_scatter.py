# experiments/019-simb-multimodal/scripts/prediction_scatter.py
# [[experiments.019-simb-multimodal.scripts.prediction_scatter]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/prediction_scatter
"""What do the expression predictions look like?

The per-gene Pearson that scores every round is a summary of 6,127 correlations; this
figure shows the predictions themselves, for the best expression checkpoint on file (v13
V_ref, split seed 0, run wq8y8nd5, validation Pearson per gene 0.225 at its best epoch)
beside the two sequence baselines of the same partition, the bilinear ridge (B2) and the
nearest-neighbor average (B3), both on ProtT5 with the cells `expression_baselines_split.py`
selected on validation. Everything is the 155 validation strains x 6,127 graph genes of
split seed 0 on fig3_core.

Panels. (a) model prediction against the measured log2 ratio, every strain-gene pair;
(b) the same with the per-gene train mean removed from both axes, which is what the per-
gene Pearson sees; (c) B3 on the centered axes. (d) per-gene Pearson of each predictor
against the gene's train sd, with the running median; (e) per-gene spread ratio sd(pred)
/ sd(target) against the same axis, the line at the Pearson value being the ratio that
minimizes squared error. (f) and (g) one strain each: measured and predicted log2 ratio
over all genes, genes ordered by the measured value, the strongest-response validation
strain and a median one. (h) and (i) one gene each across the 155 strains: the gene with
the highest model Pearson, and the gene at the median.

    python experiments/019-simb-multimodal/scripts/prediction_scatter.py

Writes results/prediction_scatter.json (every number printed on the figure) and the
figure to $ASSET_IMAGES_DIR/019-simb-multimodal/prediction_scatter_<timestamp>.{svg,png}.
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
from dotenv import load_dotenv

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
from variance_stratified_pearson import (  # type: ignore[import-not-found]  # noqa: E402
    EMB,
    ROUND,
    RUNS,
    _load_dump,
    per_gene_pearson,
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

ARM = "V_ref_s0"
COLORS = {"model": PLOT_PALETTE[0], "B2": PLOT_PALETTE[1], "B3": PLOT_PALETTE[2]}
NAMES = {"model": "CGT (v13 reference)", "B2": "bilinear ridge", "B3": "neighbor average"}


def running_median(x: np.ndarray, y: np.ndarray, bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    edges = np.linspace(0, len(xs), bins + 1).astype(int)
    cx = np.array([xs[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    cy = np.array([np.median(ys[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    return cx, cy


def box(ax: Any) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    img_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-simb-multimodal")
    os.makedirs(img_dir, exist_ok=True)
    results_dir = experiment_results_dir("019-simb-multimodal", __file__)
    spec = next(r for r in RUNS if r["arm"] == ARM)
    rd = ROUND[spec["round"]]

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
    gene_set = set(genome.gene_set)

    base = osp.join(data_root, "data/torchcell/experiments/019-simb-multimodal", rd["tag"])
    split = _load_split(osp.join(base, "data_module_cache"), spec["split_seed"], rd["label"])
    perts_tr, y_tr, keys = _load_records(base, split["train"], rd["label"])
    perts_va, y_va, keys_va = _load_records(base, split["val"], rd["label"])
    assert keys == keys_va
    in_graph = np.array([k in gene_set for k in keys])
    keys = [k for k, ok in zip(keys, in_graph) if ok]
    y_tr, y_va = y_tr[:, in_graph], y_va[:, in_graph]
    mu = np.nanmean(y_tr, axis=0, keepdims=True)
    sd_tr = np.nanstd(y_tr, axis=0, ddof=1)

    # Model predictions, rows in the dump's own order, matched to the validation records.
    dump = osp.join(data_root, "val-predictions", f"{spec['group']}.json")
    pred_m, tgt_m, rec_idx = _load_dump(dump, keys, gene_set)
    pos = {r: i for i, r in enumerate(split["val"])}
    rows = np.array([pos[r] for r in rec_idx])
    y_va, perts_va = y_va[rows], [perts_va[i] for i in rows]
    assert np.allclose(np.nan_to_num(tgt_m), np.nan_to_num(y_va), atol=1e-5)

    # Baselines on the same strains, the cells selected on validation.
    with open(osp.join(results_dir, rd["baselines"], f"seed{spec['split_seed']}.json")) as fh:
        bj = json.load(fh)
    b2 = bj["B2_bilinear"]["by_embedding"][EMB]["selected_on_val"]
    b3 = bj["B3_neighbor_average"]["by_embedding"][EMB]["selected_on_val"]
    p_tr, ok_tr = _pert_matrix(perts_tr, emb, dim)
    p_va, ok_va = _pert_matrix(perts_va, emb, dim)
    pm, ps = p_tr[ok_tr].mean(0, keepdims=True), p_tr[ok_tr].std(0, keepdims=True) + 1e-8
    p_tr, p_va = (p_tr - pm) / ps, (p_va - pm) / ps
    r_tr = np.nan_to_num(y_tr - mu, nan=0.0)
    pred_b2 = _bilinear(r_tr[ok_tr], np.nan_to_num(y_va - mu, nan=0.0), p_tr[ok_tr], p_va, int(b2["k_gene"]), float(b2["ridge"])) + mu
    e_tr = p_tr[ok_tr] / (np.linalg.norm(p_tr[ok_tr], axis=1, keepdims=True) + 1e-12)
    e_va = p_va / (np.linalg.norm(p_va, axis=1, keepdims=True) + 1e-12)
    nn = np.argsort(-(e_va @ e_tr.T), axis=1)[:, : int(b3["k"])]
    with np.errstate(invalid="ignore"):
        pred_b3 = np.nan_to_num(np.nanmean((y_tr - mu)[ok_tr][nn], axis=1), nan=0.0) + mu
    # Strains whose deleted gene has no embedding get the per-gene mean from both baselines.
    pred_b2[~ok_va], pred_b3[~ok_va] = mu, mu

    preds = {"model": pred_m, "B2": pred_b2, "B3": pred_b3}
    ok = np.isfinite(y_va)
    out: dict[str, Any] = {
        "generated_by": "experiments/019-simb-multimodal/scripts/prediction_scatter.py",
        "checkpoint": {k: spec[k] for k in ("round", "run", "arm", "seed", "split_seed", "group")},
        "baselines": {"B2": b2, "B3": b3, "embedding": EMB, "source": f"results/{rd['baselines']}/seed{spec['split_seed']}.json"},
        "n_val_strains": int(y_va.shape[0]),
        "n_genes": int(y_va.shape[1]),
        "n_val_strains_without_embedding": int((~ok_va).sum()),
        "predictors": {},
    }
    per_gene: dict[str, np.ndarray] = {}
    ratio: dict[str, np.ndarray] = {}
    for name, p in preds.items():
        r = per_gene_pearson(p, y_va)
        per_gene[name] = r
        dev_p = p - np.nanmean(p, axis=0, keepdims=True)
        dev_t = y_va - np.nanmean(y_va, axis=0, keepdims=True)
        ratio[name] = np.nanstd(np.where(ok, dev_p, np.nan), axis=0) / np.nanstd(dev_t, axis=0)
        out["predictors"][name] = {
            "pearson_per_gene_mean": float(np.nanmean(r)),
            "pearson_per_gene_median": float(np.nanmedian(r)),
            "pooled_pearson_raw": float(np.corrcoef(p[ok], y_va[ok])[0, 1]),
            "pooled_pearson_centered": float(np.corrcoef(dev_p[ok], dev_t[ok])[0, 1]),
            "spread_ratio_median": float(np.nanmedian(ratio[name])),
            "sd_pred_pooled": float(np.std(p[ok])),
            "sd_target_pooled": float(np.std(y_va[ok])),
        }

    # Example strains and genes.
    strain_sd = np.nanstd(y_va, axis=1)
    i_strong = int(np.argmax(strain_sd))
    i_median = int(np.argsort(strain_sd)[len(strain_sd) // 2])
    r_m = per_gene["model"]
    j_best = int(np.nanargmax(r_m))
    j_median = int(np.argsort(np.nan_to_num(r_m, nan=-9))[np.isfinite(r_m).sum() // 2])
    out["examples"] = {
        "strain_strong": {"deleted": perts_va[i_strong], "sd_over_genes": float(strain_sd[i_strong])},
        "strain_median": {"deleted": perts_va[i_median], "sd_over_genes": float(strain_sd[i_median])},
        "gene_best": {"gene": keys[j_best], "pearson": {n: float(per_gene[n][j_best]) for n in preds}, "train_sd": float(sd_tr[j_best])},
        "gene_median": {"gene": keys[j_median], "pearson": {n: float(per_gene[n][j_median]) for n in preds}, "train_sd": float(sd_tr[j_median])},
    }
    with open(osp.join(results_dir, "prediction_scatter.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for name, s in out["predictors"].items():
        print(f"{name:<6} " + "  ".join(f"{k} {v:.3f}" for k, v in s.items()))
    print(json.dumps(out["examples"], indent=1))

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(165)))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.05, wspace=0.32, hspace=0.42)
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f), (ax_g, ax_h, ax_i) = axes
    ident = dict(color="black", lw=0.5, ls="--")

    # (a) raw pooled.
    lim = (-2.0, 2.0)
    ax_a.hexbin(y_va[ok], pred_m[ok], gridsize=70, extent=(*lim, *lim), cmap="Greys", bins="log", linewidths=0.1)
    ax_a.plot(lim, lim, **ident)
    s = out["predictors"]["model"]
    ax_a.set(xlim=lim, ylim=lim, xlabel="measured log2 ratio", ylabel="predicted log2 ratio")
    ax_a.set_title(f"{NAMES['model']}, all pairs; pooled r {s['pooled_pearson_raw']:.2f}", fontsize=6)

    # (b), (c) centered pooled.
    clim = (-1.0, 1.0)
    for ax, name in ((ax_b, "model"), (ax_c, "B3")):
        p = preds[name]
        dev_p = (p - np.nanmean(p, axis=0, keepdims=True))[ok]
        dev_t = (y_va - np.nanmean(y_va, axis=0, keepdims=True))[ok]
        ax.hexbin(dev_t, dev_p, gridsize=70, extent=(*clim, *clim), cmap="Greys", bins="log", linewidths=0.1)
        ax.plot(clim, clim, **ident)
        slope = float(np.polyfit(dev_t, dev_p, 1)[0])
        xx = np.array(clim)
        ax.plot(xx, slope * xx, color=COLORS[name], lw=0.8)
        st = out["predictors"][name]
        ax.set(xlim=clim, ylim=clim, xlabel="measured, gene-centered", ylabel="predicted, gene-centered")
        ax.set_title(f"{NAMES[name]}; r {st['pooled_pearson_centered']:.2f}, slope {slope:.2f}", fontsize=6)
        out["predictors"][name]["centered_slope"] = slope

    # (d) per-gene Pearson vs train sd.
    for name in ("model", "B2", "B3"):
        r = per_gene[name]
        good = np.isfinite(r)
        ax_d.scatter(sd_tr[good], r[good], s=0.6, color=COLORS[name], alpha=0.25, linewidths=0, rasterized=True)
        cx, cy = running_median(sd_tr[good], r[good])
        ax_d.plot(cx, cy, color=COLORS[name], lw=1.0, label=f"{NAMES[name]} (mean {np.nanmean(r):.3f})")
    ax_d.axhline(0, color="black", lw=0.5)
    ax_d.set(xscale="log", xlabel="gene train sd of log2 ratio", ylabel="per-gene Pearson (155 strains)", ylim=(-0.6, 1.0))
    ax_d.legend(frameon=False, loc="upper left")

    # (e) per-gene spread ratio vs train sd.
    for name in ("model", "B2", "B3"):
        q = ratio[name]
        good = np.isfinite(q)
        ax_e.scatter(sd_tr[good], q[good], s=0.6, color=COLORS[name], alpha=0.25, linewidths=0, rasterized=True)
        cx, cy = running_median(sd_tr[good], q[good])
        ax_e.plot(cx, cy, color=COLORS[name], lw=1.0, label=f"{NAMES[name]} (median {np.nanmedian(q):.2f})")
    ax_e.axhline(1.0, color="black", lw=0.5)
    ax_e.axhline(out["predictors"]["model"]["pearson_per_gene_mean"], color=COLORS["model"], lw=0.5, ls=":")
    ax_e.set(xscale="log", yscale="log", xlabel="gene train sd of log2 ratio", ylabel="sd(pred) / sd(measured), per gene", ylim=(0.01, 10))
    ax_e.legend(frameon=False, loc="upper left")

    # (f) strains: pooled hexbin of B2 for symmetry with (c) is less useful than a second strain; use
    # (f) strongest strain, (g) median strain.
    for ax, i, tag in ((ax_f, i_strong, "strongest response"), (ax_g, i_median, "median response")):
        o = np.argsort(np.nan_to_num(y_va[i], nan=0.0))
        x = np.arange(len(o))
        ax.scatter(x, y_va[i][o], s=0.5, color="black", linewidths=0, label="measured", rasterized=True)
        for name in ("model", "B3"):
            ax.scatter(x, preds[name][i][o], s=0.5, color=COLORS[name], alpha=0.6, linewidths=0, label=NAMES[name], rasterized=True)
        deleted = ",".join(perts_va[i])
        ax.set(xlabel="genes, ordered by measured value", ylabel="log2 ratio", ylim=(-2.5, 2.5))
        ax.set_title(f"strain {deleted}Δ, {tag}; r {per_gene_pearson(preds['model'][i:i + 1].T, y_va[i:i + 1].T)[0]:.2f}", fontsize=6)
        ax.legend(frameon=False, loc="upper left", markerscale=6)

    # (h), (i) genes across strains.
    for ax, j, tag in ((ax_h, j_best, "highest model Pearson"), (ax_i, j_median, "median model Pearson")):
        for name in ("model", "B3"):
            ax.scatter(y_va[:, j], preds[name][:, j], s=4, color=COLORS[name], alpha=0.8, linewidths=0, label=f"{NAMES[name]} r {per_gene[name][j]:.2f}")
        lo = float(np.nanmin(np.concatenate([y_va[:, j], preds["model"][:, j]]))) - 0.05
        hi = float(np.nanmax(np.concatenate([y_va[:, j], preds["model"][:, j]]))) + 0.05
        ax.plot([lo, hi], [lo, hi], **ident)
        ax.set(xlabel="measured log2 ratio", ylabel="predicted log2 ratio", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_title(f"gene {keys[j]}, {tag}; train sd {sd_tr[j]:.2f}", fontsize=6)
        ax.legend(frameon=False, loc="upper left")

    for ax, letter in zip(axes.ravel(), "abcdefghi"):
        box(ax)
        panel_label(ax, letter)

    with open(osp.join(results_dir, "prediction_scatter.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    stem = osp.join(img_dir, f"prediction_scatter_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    print("wrote", stem + ".svg")


if __name__ == "__main__":
    main()
