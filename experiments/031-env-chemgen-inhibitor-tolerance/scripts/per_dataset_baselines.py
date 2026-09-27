# experiments/031-env-chemgen-inhibitor-tolerance/scripts/per_dataset_baselines.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.per_dataset_baselines]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/per_dataset_baselines
"""Ridge and k-nearest-neighbor baselines on every dataset, against its reliability ceiling.

``baseline_ceilings.py`` answered this for Vanacloig alone. A pooled model has to be judged
against what each SOURCE allows on its own, because a dataset whose simple baseline is
already near its ceiling has no headroom for pooling to recover, and a dataset with no
served uncertainty has no ceiling to be judged against at all.

THE TASK, per dataset. Build the gene-by-compound response matrix, hold out whole compounds,
and predict the held-out column from the molecule embedding. Only a molecular feature can
help, because the held-out compound was never dosed.

SPLIT SIZE FOLLOWS THE PANEL. Leave-one-compound-out for a panel of at most ``LOCO_MAX``
compounds, which is the exhaustive answer; ``N_FOLDS``-fold grouped by compound otherwise,
because Wildenhain's 5,170 compounds would be 5,170 fits per encoder. The split actually
used is written into every row, so the two are never silently compared as though equal.

TWO TARGETS. The raw response, and the response with each gene's TRAINING-compound mean
subtracted. The gene main effect dominates every one of these matrices, so a model can score
well on the raw target while carrying no compound-specific information. The centered target
removes it and is the one that matters for predicting an unseen inhibitor. Centering is refit
inside each fold.

THE CEILING IS PER DATASET AND PER TARGET, and it exists only where the source serves an
uncertainty. Vanacloig serves one on every record, the two Hillenmeyer arms on a third and a
quarter of theirs, Wildenhain on 4 percent, and Hoepfner on none. Where too few records carry
one the ceiling is NA and the score is reported alone, which is an honest gap rather than an
estimate.

THE NULL. ``gene_mean`` predicts each gene's training-compound mean and carries no molecular
information. On the centered target its prediction is identically zero, so the null there is
``random_neighbor``: the similarity-weighted mean of k RANDOMLY chosen training compounds,
averaged over ``N_RANDOM_DRAWS`` draws. An encoder that does not beat it has contributed
nothing.

Writes ``results/per_dataset_baselines.csv`` (one row per dataset x fold x features x model x
target), ``results/per_dataset_baselines_summary.csv`` (medians over folds), and the figure
``encoder_comparison.{svg,png}`` into ``ASSET_IMAGES_DIR/031-env-chemgen-inhibitor-tolerance/``.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from glob import glob
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numpy.typing import NDArray
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV

from torchcell.molecule.similarity import cosine_matrix, tanimoto_matrix
from torchcell.utils import PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
EMB_DIR = osp.join(RESULTS_DIR, "embeddings")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "031-env-chemgen-inhibitor-tolerance")

NAMES = [
    "vanacloig2022",
    "hillenmeyer2008_hom",
    "hillenmeyer2008_het",
    "hoepfner2014",
    "wildenhain2015",
]
LABEL = {
    "vanacloig2022": "Vanacloig",
    "hillenmeyer2008_hom": "Hillenmeyer HOM",
    "hillenmeyer2008_het": "Hillenmeyer HET",
    "hoepfner2014": "Hoepfner",
    "wildenhain2015": "Wildenhain",
}
COLOR = dict(zip(NAMES, PLOT_PALETTE[:5], strict=True))
HOST_GENES = {"YBL005W", "YDR011W", "YGL013C"}
FINGERPRINTS = {"ecfp4_count", "ecfp4_bit", "fcfp4_count", "maccs"}
ALPHAS = np.logspace(-1, 6, 22)
KNN_KS = (1, 5)
N_RANDOM_DRAWS = 10
LOCO_MAX = 60
N_FOLDS = 10
#: Below this many records carrying a served standard error, a per-fold ceiling is not
#: estimated at all rather than estimated from a biased subset.
MIN_SE_FOR_CEILING = 200
RNG = np.random.default_rng(0)


def queried_gene(genotype: str) -> str:
    """The one perturbed gene of a genotype that is not part of the sensitizing host."""
    genes = [g for g in str(genotype).split("|") if g not in HOST_GENES]
    return genes[0] if len(genes) == 1 else ""


def load_matrix(name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Gene-by-compound response and standard error, indexed by InChIKey."""
    df = pd.read_parquet(
        osp.join(RESULTS_DIR, f"records_{name}.parquet"),
        columns=["gene", "inchikey", "response", "response_se", "n_small_molecules"],
    )
    df = df[df["n_small_molecules"] == 1]
    df["qgene"] = df["gene"].map(queried_gene)
    df = df[(df["qgene"] != "") & (df["inchikey"].astype(str).str.len() > 0)]
    resp = df.pivot_table(
        index="qgene", columns="inchikey", values="response", aggfunc="mean"
    )
    se = df.pivot_table(
        index="qgene", columns="inchikey", values="response_se", aggfunc="mean"
    ).reindex(index=resp.index, columns=resp.columns)
    return resp, se, list(resp.columns)


def load_molecule_embeddings(keys: list[str]) -> dict[str, NDArray[np.float64]]:
    """Per encoder, the compound-by-feature matrix aligned to the response columns."""
    out: dict[str, NDArray[np.float64]] = {}
    for path in sorted(glob(osp.join(EMB_DIR, "*.npz"))):
        enc = osp.basename(path)[:-4]
        z = np.load(path, allow_pickle=True)
        idx = {str(k): i for i, k in enumerate(z["inchikey"])}
        if not all(k in idx for k in keys):
            continue  # an encoder that cannot represent every compound of this dataset
        A = np.asarray(z["X"], dtype=np.float64)[[idx[k] for k in keys]]
        # the RDKit descriptor block has undefined entries for some structures
        if np.isnan(A).any():
            with np.errstate(invalid="ignore"):
                med = np.nanmedian(A, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            A = np.where(np.isnan(A), med, A)
        out[enc] = A
    return out


def ceiling(target: NDArray[np.float64], se: NDArray[np.float64]) -> float:
    """Sqrt of 1 - mean(SE^2)/Var(target): the limit on r against the noise-free value."""
    ok = np.isfinite(target) & np.isfinite(se)
    if ok.sum() < MIN_SE_FOR_CEILING:
        return float("nan")
    var = float(np.var(target[ok], ddof=1))
    if var <= 0:
        return float("nan")
    rel = 1.0 - float(np.mean(se[ok] ** 2)) / var
    return float(np.sqrt(rel)) if rel > 0 else 0.0


def similarity(
    enc: str, X: NDArray[np.float64], ia: NDArray[np.int_], ib: NDArray[np.int_]
) -> NDArray[np.float64]:
    if enc in FINGERPRINTS:
        return tanimoto_matrix(X[ia], X[ib])
    return cosine_matrix(X[ia], X[ib])


def knn_predict(
    sims: NDArray[np.float64], train_profiles: NDArray[np.float64], k: int
) -> NDArray[np.float64]:
    """Similarity-weighted mean of the k most similar training profiles."""
    order = np.argsort(-sims)[:k]
    w = np.clip(sims[order], 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    masked = np.ma.masked_invalid(train_profiles[order])
    return np.ma.average(masked, axis=0, weights=w).filled(np.nan)


def ridge_predict(
    Xtr: NDArray[np.float64], Ytr: NDArray[np.float64], Xte: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Multi-output ridge with the penalty chosen by efficient leave-one-out on train."""
    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(Ytr, axis=0)
    overall = float(np.nanmean(Ytr))
    col_mean = np.where(np.isfinite(col_mean), col_mean, overall)
    Y = np.where(np.isnan(Ytr), col_mean, Ytr)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    model = RidgeCV(alphas=ALPHAS)
    model.fit((Xtr - mu) / sd, Y)
    return np.asarray(model.predict((Xte - mu) / sd), dtype=np.float64)


def score(pred: NDArray[np.float64], obs: NDArray[np.float64]) -> tuple[float, float]:
    ok = np.isfinite(pred) & np.isfinite(obs)
    if ok.sum() < 20 or np.std(pred[ok]) == 0:
        return float("nan"), float("nan")
    return (
        float(spearmanr(pred[ok], obs[ok])[0]),
        float(pearsonr(pred[ok], obs[ok])[0]),
    )


def folds_for(n_compounds: int) -> tuple[list[NDArray[np.int_]], str]:
    """Held-out compound index blocks, and the name of the scheme that produced them."""
    order = RNG.permutation(n_compounds)
    if n_compounds <= LOCO_MAX:
        return [np.array([i]) for i in range(n_compounds)], "leave_one_compound_out"
    return list(np.array_split(order, N_FOLDS)), f"{N_FOLDS}fold_compound_grouped"


def run_dataset(name: str) -> pd.DataFrame:
    """Every fold, encoder, model and target for one dataset."""
    resp, se, keys = load_matrix(name)
    mol = load_molecule_embeddings(keys)
    R, S = resp.to_numpy(), se.to_numpy()
    blocks, scheme = folds_for(len(keys))
    frac_se = float(np.isfinite(S).mean())
    print(
        f"{name}: {R.shape[0]:,} genes x {R.shape[1]:,} compounds, "
        f"{len(mol)} encoders, {len(blocks)} folds ({scheme}), "
        f"SE on {100 * frac_se:.0f}% of cells"
    )
    rows: list[dict[str, Any]] = []
    for f, te in enumerate(blocks):
        tr = np.setdiff1d(np.arange(len(keys)), te)
        with np.errstate(invalid="ignore"):
            gene_mean = np.nanmean(R[:, tr], axis=1)
        for target in ("raw", "centered"):
            obs = R[:, te] if target == "raw" else R[:, te] - gene_mean[:, None]
            tgt_tr = R[:, tr] if target == "raw" else R[:, tr] - gene_mean[:, None]
            cap = ceiling(obs.ravel(), S[:, te].ravel())
            base = dict(
                dataset=name,
                fold=f,
                scheme=scheme,
                n_train_compounds=len(tr),
                n_test_compounds=len(te),
                target=target,
                ceiling=cap,
            )
            if target == "raw":
                pred = np.repeat(gene_mean[:, None], len(te), axis=1)
                rho, r = score(pred.ravel(), obs.ravel())
                rows.append(
                    {
                        **base,
                        "features": "none",
                        "model": "gene_mean",
                        "spearman": rho,
                        "pearson": r,
                    }
                )
            else:
                draws = []
                for _ in range(N_RANDOM_DRAWS):
                    pick = RNG.choice(len(tr), size=min(5, len(tr)), replace=False)
                    with np.errstate(invalid="ignore"):
                        pred = np.repeat(
                            np.nanmean(tgt_tr[:, pick], axis=1)[:, None],
                            len(te),
                            axis=1,
                        )
                    draws.append(score(pred.ravel(), obs.ravel()))
                rows.append(
                    {
                        **base,
                        "features": "none",
                        "model": "random_neighbor",
                        "spearman": float(np.nanmedian([d[0] for d in draws])),
                        "pearson": float(np.nanmedian([d[1] for d in draws])),
                    }
                )
            for enc, X in mol.items():
                pred = ridge_predict(X[tr], tgt_tr.T, X[te]).T
                rho, r = score(pred.ravel(), obs.ravel())
                rows.append(
                    {
                        **base,
                        "features": enc,
                        "model": "ridge",
                        "spearman": rho,
                        "pearson": r,
                    }
                )
                sims = similarity(enc, X, te, tr)
                for k in KNN_KS:
                    pred = np.column_stack(
                        [knn_predict(sims[i], tgt_tr.T, k) for i in range(len(te))]
                    )
                    rho, r = score(pred.ravel(), obs.ravel())
                    rows.append(
                        {
                            **base,
                            "features": enc,
                            "model": f"knn{k}",
                            "spearman": rho,
                            "pearson": r,
                        }
                    )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame) -> None:
    """Encoder comparison: per dataset, per encoder, against the null and the ceiling."""
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )
    cent = summary[(summary["target"] == "centered") & (summary["model"] == "ridge")]
    encoders = sorted(cent["features"].unique())
    present = [n for n in NAMES if n in set(cent["dataset"])]

    fig, axes = plt.subplots(
        2, 2, figsize=(mm_to_in(179.0), mm_to_in(110.0)), constrained_layout=True
    )

    # (a) every encoder, every dataset, ridge on the centered target
    ax = axes[0, 0]
    width = 0.8 / max(len(present), 1)
    x = np.arange(len(encoders))
    for i, name in enumerate(present):
        sub = cent[cent["dataset"] == name].set_index("features")
        vals = [float(sub["spearman_median"].get(e, np.nan)) for e in encoders]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width=width,
            color=COLOR[name],
            edgecolor="black",
            lw=0.3,
            label=LABEL[name],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_", " ") for e in encoders], rotation=45, ha="right")
    ax.set_ylabel("Spearman, compound cold, centered")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(frameon=False, fontsize=4.5, ncol=2)
    ax.set_title("a  molecule features by encoder and dataset", loc="left", fontsize=6)

    # (b) best model against the no-feature null, per dataset
    ax = axes[0, 1]
    rows = []
    for name in present:
        d = summary[(summary["dataset"] == name) & (summary["target"] == "centered")]
        best = d[d["model"] == "ridge"]["spearman_median"].max()
        null = d[d["model"] == "random_neighbor"]["spearman_median"].median()
        rows.append((name, null, best))
    y = np.arange(len(rows))
    ax.barh(
        y - 0.2,
        [r[1] for r in rows],
        height=0.4,
        color="#666666",
        edgecolor="black",
        lw=0.3,
        label="random neighbor (null)",
    )
    ax.barh(
        y + 0.2,
        [r[2] for r in rows],
        height=0.4,
        color=[COLOR[r[0]] for r in rows],
        edgecolor="black",
        lw=0.3,
        label="best ridge",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[r[0]] for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Spearman, centered target")
    ax.legend(frameon=False, fontsize=4.5, loc="lower right")
    ax.set_title("b  every dataset beats its null", loc="left", fontsize=6)

    # (c) ridge against kNN, one point per dataset and encoder
    ax = axes[1, 0]
    piv = (
        summary[summary["target"] == "centered"]
        .pivot_table(
            index=["dataset", "features"], columns="model", values="spearman_median"
        )
        .reset_index()
    )
    piv = piv[piv["features"] != "none"]
    knn_col = "knn5" if "knn5" in piv.columns else "knn1"
    for name in present:
        sub = piv[piv["dataset"] == name]
        ax.scatter(
            sub[knn_col],
            sub["ridge"],
            s=8,
            color=COLOR[name],
            edgecolor="black",
            lw=0.3,
            label=LABEL[name],
            zorder=3,
        )
    lo = float(np.nanmin(piv[[knn_col, "ridge"]].to_numpy()))
    hi = float(np.nanmax(piv[[knn_col, "ridge"]].to_numpy()))
    ax.plot([lo, hi], [lo, hi], color="#666666", lw=0.5, ls="--")
    ax.set_xlabel(f"{knn_col} Spearman")
    ax.set_ylabel("ridge Spearman")
    ax.legend(frameon=False, fontsize=4.5, loc="upper left")
    ax.set_title("c  a linear map beats neighbor transfer", loc="left", fontsize=6)

    # (d) fraction of the ceiling reached, where a ceiling exists
    ax = axes[1, 1]
    have = summary[
        (summary["target"] == "centered")
        & (summary["model"] == "ridge")
        & summary["ceiling_median"].notna()
    ]
    rows = []
    for name in present:
        sub = have[have["dataset"] == name]
        if sub.empty:
            rows.append((name, np.nan, np.nan))
            continue
        best = sub.loc[sub["spearman_median"].idxmax()]
        rows.append(
            (name, float(best["spearman_median"]), float(best["ceiling_median"]))
        )
    y = np.arange(len(rows))
    for i, (name, best, cap) in enumerate(rows):
        if not np.isfinite(cap):
            ax.text(
                0.02,
                i,
                "no served uncertainty",
                fontsize=5,
                va="center",
                color="#A24A46",
            )
            continue
        ax.barh(i, cap, color="#F5F5F5", edgecolor="#666666", lw=0.4)
        ax.barh(i, best, color=COLOR[name], edgecolor="black", lw=0.4)
        ax.text(
            cap + 0.01,
            i,
            f"{100 * best / cap:.0f}% of ceiling",
            fontsize=4.5,
            va="center",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[r[0]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Spearman (bar) against the reliability ceiling (outline)")
    ax.set_title("d  headroom, where a ceiling exists", loc="left", fontsize=6)

    for ax in axes.ravel():
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    # the raster goes FIRST: savefig_true_size_svg rescales the figure for draw.io
    fig.savefig(osp.join(IMAGE_DIR, "encoder_comparison.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMAGE_DIR, "encoder_comparison.svg"))
    plt.close(fig)
    print(f"  wrote {osp.join(IMAGE_DIR, 'encoder_comparison.svg')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=NAMES, choices=NAMES)
    args = ap.parse_args()

    all_rows = pd.concat([run_dataset(n) for n in args.datasets], ignore_index=True)
    all_rows.to_csv(osp.join(RESULTS_DIR, "per_dataset_baselines.csv"), index=False)
    summary = (
        all_rows.groupby(["dataset", "target", "features", "model", "scheme"])
        .agg(
            folds=("spearman", "size"),
            spearman_median=("spearman", "median"),
            pearson_median=("pearson", "median"),
            ceiling_median=("ceiling", "median"),
        )
        .reset_index()
    )
    summary["frac_of_ceiling"] = summary["spearman_median"] / summary["ceiling_median"]
    summary.to_csv(
        osp.join(RESULTS_DIR, "per_dataset_baselines_summary.csv"), index=False
    )
    print()
    best = (
        summary[(summary["target"] == "centered") & (summary["model"] == "ridge")]
        .sort_values("spearman_median", ascending=False)
        .groupby("dataset")
        .head(1)
    )
    print(
        best[
            [
                "dataset",
                "features",
                "folds",
                "spearman_median",
                "ceiling_median",
                "frac_of_ceiling",
            ]
        ].to_string(index=False)
    )
    make_figure(summary)


if __name__ == "__main__":
    main()
