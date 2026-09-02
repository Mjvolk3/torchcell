# experiments/010-kuzmin-tmi/scripts/inference_space_predictor_agreement.py
# [[experiments.010-kuzmin-tmi.scripts.inference_space_predictor_agreement]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_space_predictor_agreement

"""How much do the 010 predictors agree with each other on UNMEASURED triples?

``additive_vs_transformer_inference_space.py`` found that the additive per-gene
ridge and the transformer are essentially uncorrelated over the 010 design
space, even though on measured held-out data they score within 0.05 Pearson of
each other. That leaves two readings, and they have opposite consequences:

  (a) the transformer has learned genuine non-additive structure that the
      additive model cannot express, or
  (b) the transformer's extrapolation onto never-measured triples is not
      pinned down by the training data at all, so its design-space ranking is
      arbitrary in the directions no held-out measurement constrains.

The three 010 checkpoints (M01, M02, M03) are the discriminator. They are
independent training runs of the SAME architecture on the SAME split, and all
three scored the same 4,370,595-triple inference_1 space. Under (a) they should
agree strongly with one another and disagree with the additive model. Under (b)
they should disagree with one another about as much as with the additive model.

This script joins the three checkpoints' predictions on the gene triple, adds
the additive per-gene ridge prediction for the same triples, and reports the
full correlation matrix plus top-K selection overlaps between every pair.
"""

import glob
import os
import os.path as osp
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr

from torchcell.utils import PANEL_WIDTHS_MM, mm_to_in, savefig_true_size_svg

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
INFERENCE_1 = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/010-kuzmin-tmi/inference_1/inferred",
)
COEF_NPZ = osp.join(RESULTS_DIR, "additive_baseline_B1_coefficients.npz")

CHECKPOINTS = {"M01": "lzs9pcj3", "M02": "yv4r30bi", "M03": "c7671wgj"}
TOP_K = [100, 1_000, 10_000, 100_000]


def main() -> None:
    frames = {}
    for tag, run in CHECKPOINTS.items():
        matches = glob.glob(osp.join(INFERENCE_1, f"*-{run}-best-pearson-*.parquet"))
        assert len(matches) == 1, f"expected one parquet for {run}, got {matches}"
        df = pd.read_parquet(matches[0], columns=["gene1", "gene2", "gene3", "prediction"])
        frames[tag] = df.rename(columns={"prediction": tag})
        print(f"{tag} ({run}): {len(df)} rows from {osp.basename(matches[0])}")

    merged = frames["M01"]
    for tag in ("M02", "M03"):
        merged = merged.merge(frames[tag], on=["gene1", "gene2", "gene3"], how="inner")
    print(f"triples scored by all three checkpoints: {len(merged)}")

    coef = np.load(COEF_NPZ, allow_pickle=True)
    gene_to_col = {g: j for j, g in enumerate(coef["gene_names"])}
    beta, intercept = coef["beta"], float(coef["intercept"])

    in_vocab = np.ones(len(merged), dtype=bool)
    additive = np.full(len(merged), intercept)
    for col in ("gene1", "gene2", "gene3"):
        idx = merged[col].map(gene_to_col)
        known = idx.notna().to_numpy()
        in_vocab &= known
        contrib = np.zeros(len(merged))
        contrib[known] = beta[idx[known].to_numpy().astype(np.int64)]
        additive += contrib
    merged["additive"] = additive
    print(f"triples with all three genes in the additive fit: {int(in_vocab.sum())}")
    merged = merged[in_vocab].reset_index(drop=True)

    names = ["additive", "M01", "M02", "M03"]
    values = {k: merged[k].to_numpy().astype(np.float64) for k in names}
    for k in names:
        print(f"{k:>9}: mean {values[k].mean(): .6f}  sd {values[k].std():.6f}")

    rows = []
    print("\npairwise agreement on the unmeasured design space")
    for a, b in combinations(names, 2):
        r = float(np.corrcoef(values[a], values[b])[0, 1])
        rho = float(spearmanr(values[a], values[b])[0])
        print(f"  {a:>9} vs {b:<9} Pearson {r: .4f}  Spearman {rho: .4f}")
        rows.append({"pair": f"{a}|{b}", "quantity": "pearson", "value": r})
        rows.append({"pair": f"{a}|{b}", "quantity": "spearman", "value": rho})

    print("\ntop-K selection overlap (most positive predicted interaction)")
    order = {k: np.argsort(-values[k]) for k in names}
    order_bot = {k: np.argsort(values[k]) for k in names}
    for k in TOP_K:
        for a, b in combinations(names, 2):
            top = np.intersect1d(order[a][:k], order[b][:k]).size / k
            bot = np.intersect1d(order_bot[a][:k], order_bot[b][:k]).size / k
            print(f"  K={k:>7} {a:>9} vs {b:<9} top {top:.4f}  bottom {bot:.4f}")
            rows.append({"pair": f"{a}|{b}", "quantity": f"top_{k}_overlap", "value": top})
            rows.append(
                {"pair": f"{a}|{b}", "quantity": f"bottom_{k}_overlap", "value": bot}
            )

    out = osp.join(RESULTS_DIR, "inference_space_predictor_agreement.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    plot(values, names)


def plot(values: dict[str, np.ndarray], names: list[str]) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )
    n = len(names)
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(88.5)),
        sharex=True,
        sharey=True,
    )
    rng = np.random.default_rng(0)
    pick = rng.choice(values[names[0]].size, size=min(200_000, values[names[0]].size), replace=False)
    label = {"additive": "Additive", "M01": "CGT M01", "M02": "CGT M02", "M03": "CGT M03"}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            ax = axes[i, j]
            if i == j:
                ax.hist(values[a][pick], bins=80, color="0.6", linewidth=0)
            else:
                ax.hexbin(
                    values[b][pick],
                    values[a][pick],
                    gridsize=55,
                    bins="log",
                    cmap="magma_r",
                    linewidths=0,
                )
                r = np.corrcoef(values[a], values[b])[0, 1]
                ax.set_title(f"r = {r:.3f}", fontsize=5, pad=1.5)
            if i == n - 1:
                ax.set_xlabel(label[b])
            if j == 0:
                ax.set_ylabel(label[a])
            for spine in ax.spines.values():
                spine.set_visible(True)
    fig.tight_layout()

    stem = osp.join(
        ASSET_IMAGES_DIR, "010-kuzmin-tmi", "inference_space_predictor_agreement"
    )
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()
