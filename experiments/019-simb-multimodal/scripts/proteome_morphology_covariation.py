# experiments/019-simb-multimodal/scripts/proteome_morphology_covariation.py
# [[experiments.019-simb-multimodal.scripts.proteome_morphology_covariation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/proteome_morphology_covariation
"""Does the Messner 2023 knockout proteome carry information about CalMorph morphology of
the same deletions, and on which features?

Asked 2026-09-27 before launching the joint rounds: the Figure 3 story is that the labels
correlate and that the correlation can be learned to improve prediction of one label from
another. The proteome-against-expression version of this question read 0.036 per strain
(the replicate floor) and 0.22 by ridge from the observed modality. This script asks the
same of proteome against morphology, on the deletions both panels measured.

Four reads, all out of fold (5 folds over strains, ridge with the penalty chosen on an
inner split), all against a strain-permuted null:
  1. observed proteome -> each CalMorph feature: held-out Pearson per feature, summarized
     over the features the model uses (results/morphology_feature_ceiling.csv) and split
     by replicate reliability, so "the features that move under deletion" (reliability
     above 0.5, i.e. knockout variance at least twice the wild-type replicate variance)
     are read on their own;
  2. observed morphology -> each protein: held-out Pearson per protein;
  3. response magnitude: does a deletion that moves the proteome move morphology?
     Spearman across strains between the L2 norm of the per-feature z-scored proteome
     response and the same for morphology;
  4. the first proteome principal component (the slow-growth axis in every knockout panel)
     projected out of both sides before read 1, so a shared growth axis is not mistaken
     for feature-level coupling.

Inputs: the proteome LMDB (data/torchcell/proteome_messner2023, log2 of strain over the
HIS3 reference as in 028's proteome_expression_covariation.py), the CalMorph mutant table
mt4718data.tsv and the wild-type replicate table wt122data.tsv from the Ohya 2005 mirror,
and results/morphology_feature_ceiling.csv for the model features and their reliability.

Writes results/proteome_morphology_covariation.json and
$ASSET_IMAGES_DIR/019-simb-multimodal/proteome_morphology_covariation.{svg,png}.

    python experiments/019-simb-multimodal/scripts/proteome_morphology_covariation.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from typing import Any

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
LMDB_PROTEOME = osp.join(
    DATA_ROOT, "data/torchcell/proteome_messner2023/processed/lmdb"
)
MORPH_DIR = osp.join(
    DATA_ROOT, "torchcell-library/ohyaHighdimensionalLargescalePhenotyping2005a/data"
)
RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
CEILING_CSV = osp.join(RESULTS, "morphology_feature_ceiling.csv")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-simb-multimodal")
FOLDS = 5
LAMBDAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
RELIABLE = 0.5
SEED = 0
ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]

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


def _load_lmdb(path: str) -> list[dict[str, Any]]:
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        for _, value in txn.cursor():
            records.append(pickle.loads(value))
    env.close()
    return records


def _messner(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Deletion x protein log2 ratio over the HIS3 reference, duplicate strains of one
    ORF averaged (the same construction as 028's proteome_expression_covariation.py).
    """
    ref = records[0]["reference"]["phenotype_reference"]["protein_abundance"]
    ref_log = pd.Series({g: np.log2(v) for g, v in ref.items() if v > 0})
    rows: dict[str, list[pd.Series]] = {}
    for rec in records:
        perts = rec["experiment"]["genotype"]["perturbations"]
        if len(perts) != 1:
            continue
        orf = perts[0]["systematic_gene_name"]
        ab = rec["experiment"]["phenotype"]["protein_abundance"]
        s = pd.Series({g: np.log2(v) for g, v in ab.items() if v > 0}) - ref_log
        rows.setdefault(orf, []).append(s.dropna())
    return pd.DataFrame(
        {o: pd.concat(v, axis=1).mean(axis=1) for o, v in rows.items()}
    ).T


def _morphology() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mt = pd.read_csv(osp.join(MORPH_DIR, "mt4718data.tsv"), sep="\t", index_col=0)
    wt = pd.read_csv(osp.join(MORPH_DIR, "wt122data.tsv"), sep="\t", index_col=0)
    ceiling = pd.read_csv(CEILING_CSV, index_col=0)
    # the CalMorph tables key strains by lowercase ORF; the proteome by systematic name
    mt.index = [str(i).upper() for i in mt.index]
    feats = [f for f in ceiling.index if f in mt.columns]
    return mt[feats].astype(float), wt[feats].astype(float), ceiling.loc[feats]


def _zscore(m: pd.DataFrame) -> pd.DataFrame:
    return (m - m.mean()) / m.std(ddof=0)


def _ridge_oof(
    X: np.ndarray, Y: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Out-of-fold ridge predictions of every column of Y from X; the penalty is chosen
    on an inner split of each training fold by mean held-out Pearson over columns.
    """
    n = X.shape[0]
    order = rng.permutation(n)
    folds = np.array_split(order, FOLDS)
    pred = np.full_like(Y, np.nan, dtype=float)
    chosen: list[float] = []
    for k in range(FOLDS):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(FOLDS) if j != k])
        inner_te = tr[: len(tr) // 5]
        inner_tr = tr[len(tr) // 5 :]
        best, best_lam = -np.inf, LAMBDAS[0]
        for lam in LAMBDAS:
            p = _ridge_fit_predict(X[inner_tr], Y[inner_tr], X[inner_te], lam)
            score = np.nanmean(_col_pearson(p, Y[inner_te]))
            if score > best:
                best, best_lam = score, lam
        chosen.append(best_lam)
        pred[te] = _ridge_fit_predict(X[tr], Y[tr], X[te], best_lam)
    return pred, float(np.median(chosen))


def _ridge_fit_predict(
    Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, lam: float
) -> np.ndarray:
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    # dual form when features outnumber rows
    if Xc.shape[1] > Xc.shape[0]:
        K = Xc @ Xc.T
        alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), Yc)
        return (Xte - mx) @ (Xc.T @ alpha) + my
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return (Xte - mx) @ W + my


def _col_pearson(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Pc = P - P.mean(0)
    Yc = Y - Y.mean(0)
    num = (Pc * Yc).sum(0)
    den = np.sqrt((Pc**2).sum(0) * (Yc**2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return r


def _project_out_pc1(M: np.ndarray) -> np.ndarray:
    Mc = M - M.mean(0)
    u, s, vt = np.linalg.svd(Mc, full_matrices=False)
    return Mc - np.outer(u[:, 0] * s[0], vt[0])


def main() -> None:
    rng = np.random.default_rng(SEED)
    P = _messner(_load_lmdb(LMDB_PROTEOME))
    mt, wt, ceiling = _morphology()
    strains = sorted(set(P.index) & set(mt.index))
    P = P.loc[strains]
    M = mt.loc[strains]
    # proteins measured in at least 90 percent of the shared strains; the rest imputed at
    # the protein mean for the ridge only
    keep = P.notna().mean() >= 0.9
    P = P.loc[:, keep]
    Pz = _zscore(P).fillna(0.0)
    Mz = _zscore(M)
    reliability = ceiling["reliability"].reindex(M.columns)
    moving = reliability >= RELIABLE

    X, Y = Pz.to_numpy(), Mz.to_numpy()
    # 1. proteome -> morphology
    pred, lam1 = _ridge_oof(X, Y, rng)
    r_pm = pd.Series(_col_pearson(pred, Y), index=M.columns)
    perm = rng.permutation(X.shape[0])
    pred_null, _ = _ridge_oof(X[perm], Y, rng)
    r_pm_null = pd.Series(_col_pearson(pred_null, Y), index=M.columns)
    # 4. growth axis removed from both sides
    Xg, Yg = _project_out_pc1(X), _project_out_pc1(Y)
    pred_g, _ = _ridge_oof(Xg, Yg, rng)
    r_pm_g = pd.Series(_col_pearson(pred_g, Yg), index=M.columns)
    # 2. morphology -> proteome
    pred_mp, lam2 = _ridge_oof(Y, X, rng)
    r_mp = pd.Series(_col_pearson(pred_mp, X), index=P.columns)
    pred_mp_null, _ = _ridge_oof(Y[perm], X, rng)
    r_mp_null = pd.Series(_col_pearson(pred_mp_null, X), index=P.columns)
    # 3. response magnitude
    mag_p = pd.Series(np.linalg.norm(Pz.to_numpy(), axis=1), index=strains)
    mag_m = pd.Series(np.linalg.norm(Mz.fillna(0.0).to_numpy(), axis=1), index=strains)
    rho_mag = spearmanr(mag_p, mag_m)
    # the first PC of each side against the other, the growth-axis check
    pc1_p = np.linalg.svd(X - X.mean(0), full_matrices=False)[0][:, 0]
    pc1_m = np.linalg.svd(np.nan_to_num(Y - np.nanmean(Y, 0)), full_matrices=False)[0][
        :, 0
    ]
    r_pc1 = float(abs(np.corrcoef(pc1_p, pc1_m)[0, 1]))

    def summ(r: pd.Series, mask: pd.Series | None = None) -> dict[str, float]:
        s = r if mask is None else r[mask.reindex(r.index).fillna(False)]
        s = s.dropna()
        return {
            "n": int(len(s)),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max()),
            "frac_above_0.3": float((s > 0.3).mean()),
        }

    top = r_pm.sort_values(ascending=False).head(15)
    out = {
        "generated_by": "experiments/019-simb-multimodal/scripts/proteome_morphology_covariation.py",
        "n_shared_strains": len(strains),
        "n_proteins": int(P.shape[1]),
        "n_morph_features": int(M.shape[1]),
        "n_moving_features": int(moving.sum()),
        "reliability_threshold": RELIABLE,
        "ridge_lambda_median": {"proteome_to_morph": lam1, "morph_to_proteome": lam2},
        "proteome_to_morphology": {
            "all_features": summ(r_pm),
            "moving_features": summ(r_pm, moving),
            "quiet_features": summ(r_pm, ~moving),
            "null_all": summ(r_pm_null),
            "growth_axis_removed_all": summ(r_pm_g),
            "growth_axis_removed_moving": summ(r_pm_g, moving),
            "top_features": [
                {
                    "feature": f,
                    "description": str(ceiling.loc[f, "description"]),
                    "r": float(v),
                    "reliability": float(reliability[f]),
                    "r_growth_removed": float(r_pm_g[f]),
                }
                for f, v in top.items()
            ],
        },
        "morphology_to_proteome": {"all_proteins": summ(r_mp), "null": summ(r_mp_null)},
        "response_magnitude": {
            "spearman": float(rho_mag.correlation),
            "p": float(rho_mag.pvalue),
            "n": len(strains),
        },
        "pc1_agreement_abs_r": r_pc1,
        "context": {
            "proteome_to_expression_ridge_per_feature": 0.226,
            "expression_to_proteome_ridge_per_feature": 0.218,
            "source": "review/2026-09-27-joint-review/01_data_ceilings.md",
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "proteome_morphology_covariation.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # figure: a per-feature r by reliability; b distributions; c response magnitude
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(55))
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.88, wspace=0.35)
    a, b, c = axes
    a.scatter(reliability, r_pm, s=6, color=ORANGE, label="proteome -> feature")
    a.scatter(reliability, r_pm_g, s=6, color=PURPLE, label="growth axis removed")
    a.axvline(RELIABLE, color="black", lw=0.5, ls="--")
    a.axhline(0, color="black", lw=0.5)
    a.set_xlabel("CalMorph feature reliability (1 - WT var / KO var)")
    a.set_ylabel("held-out Pearson from the observed proteome")
    a.set_title(f"per feature, {len(strains)} shared deletions")
    a.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False)
    bins = np.linspace(-0.2, 0.8, 41)
    b.hist(
        r_pm_null.dropna(),
        bins=bins,
        color=GRAY,
        alpha=0.6,
        label="strain-permuted null",
    )
    b.hist(
        r_pm[~moving].dropna(),
        bins=bins,
        color=YELLOW,
        alpha=0.7,
        label="quiet features",
    )
    b.hist(
        r_pm[moving].dropna(),
        bins=bins,
        color=ORANGE,
        alpha=0.7,
        label="moving features",
    )
    b.hist(
        r_mp.dropna(),
        bins=bins,
        histtype="step",
        color=RED,
        lw=0.8,
        label="morphology -> protein",
    )
    b.set_xlabel("held-out Pearson")
    b.set_ylabel("count")
    b.set_title("proteome -> morphology and the reverse")
    b.legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False)
    c.scatter(mag_p, mag_m, s=4, color=BLUE, alpha=0.5)
    c.set_xlabel("proteome response, L2 of z-scores")
    c.set_ylabel("morphology response, L2 of z-scores")
    c.set_title(f"response magnitude, Spearman {rho_mag.correlation:.2f}")
    for ax in (a, b, c):
        for s_ in ax.spines.values():
            s_.set_visible(True)
    a.yaxis.set_major_locator(MultipleLocator(0.2))
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)
    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "proteome_morphology_covariation.svg"))
    fig.savefig(osp.join(IMG_DIR, "proteome_morphology_covariation.png"), dpi=300)
    savefig_true_size_svg(
        fig, osp.join(IMG_DIR, f"proteome_morphology_covariation_{timestamp()}.svg")
    )
    plt.close(fig)
    print(
        json.dumps(
            {k: v for k, v in out.items() if k != "proteome_to_morphology"}, indent=1
        )
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in out["proteome_to_morphology"].items()
                if k != "top_features"
            },
            indent=1,
        )
    )
    for t in out["proteome_to_morphology"]["top_features"][:10]:
        print(
            f"  {t['feature']:10s} r {t['r']:.3f} (growth removed {t['r_growth_removed']:.3f}) rel {t['reliability']:.2f}  {t['description']}"
        )


if __name__ == "__main__":
    main()
