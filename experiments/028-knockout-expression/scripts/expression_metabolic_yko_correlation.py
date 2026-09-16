# experiments/028-knockout-expression/scripts/expression_metabolic_yko_correlation.py
# [[experiments.028-knockout-expression.scripts.expression_metabolic_yko_correlation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/expression_metabolic_yko_correlation
"""Does a deletion's expression or protein profile say anything about its metabolic
phenotype?

The knockout collection is the bridge: Kemmeren 2014 (mRNA, 1,484 deletions) and
Messner 2023 (protein, 4,699 deletions) profile the same single-gene deletions that the
metabolic screens score, so a deletion's molecular profile can be set beside its
metabolite readout. Three reads per (molecular source x metabolic panel), all on the
deletions the two share:

  univariate      per metabolite, the Pearson between the metabolite and every gene
                  across shared deletions; the largest |r| against the largest |r| under
                  a strain permutation (the null for a max over thousands of genes is
                  far above zero), plus the number of genes past BH-FDR 5%.
  multivariate    ridge from the whole profile to the metabolite, out-of-fold Pearson
                  over 5 folds, against the same statistic on permuted strains. This is
                  the "is there information anywhere in the profile" read.
  structure       Mantel-style: the Spearman between deletion x deletion similarity in
                  profile space (Pearson over genes) and deletion x deletion distance in
                  metabolite space (Euclidean over z-scored metabolites), with a strain
                  permutation null. Says whether deletions with alike profiles have alike
                  metabolic phenotypes without picking a metabolite.

Metabolic panels: Mulleder 2016 (19 amino acids, mM, log2), Cooper 2010 (16 CE-LIF
peaks, ratio to plate mean, log2, zeros dropped), Cachera 2023 (betaxanthin, corrected
fluorescence, as stored), Ozaydin 2013 (beta-carotene colony color score -5..5, as
stored), Zelezniak 2018 (central carbon metabolites, log2, kinase deletions only),
da Silveira 2014 (lipids, log2). The panel's knockout is the one `kanmx_deletion`
perturbation of the record; pathway cassettes (Cachera, Ozaydin) are the strain
background and are ignored. Messner proteins are kept when measured in >= 95% of the
shared deletions and mean-imputed otherwise.

Run from the repo root:
    python experiments/028-knockout-expression/scripts/expression_metabolic_yko_correlation.py
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from scipy.stats import t as t_dist
from sklearn.model_selection import KFold  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from cross_study_ko_expression import LMDB, _load_lmdb, _profiles  # noqa: E402
from cross_study_structure import _matrix, _palette_cmap  # noqa: E402
from proteome_expression_covariation import LMDB_PROTEOME, _messner  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

PANELS: dict[str, dict[str, Any]] = {
    "mulleder": {
        "lmdb": "data/torchcell/amino_acid_mulleder2016/processed/lmdb",
        "label": "Mulleder 2016 amino acids",
        "key": "metabolite_level",
        "transform": "log2",
    },
    "cooper": {
        "lmdb": "data/torchcell/amino_acid_cooper2010/processed/lmdb",
        "label": "Cooper 2010 amino acids",
        "key": "metabolite_level",
        "transform": "log2_drop_zero",
    },
    "cachera": {
        "lmdb": "data/torchcell/betaxanthin_cachera2023/processed/lmdb",
        "label": "Cachera 2023 betaxanthin",
        "key": "metabolite_level",
        "transform": "none",
    },
    "ozaydin": {
        "lmdb": "data/torchcell/carotenoid_ozaydin2013/processed/lmdb",
        "label": "Ozaydin 2013 beta-carotene",
        "key": "visual_score",
        "transform": "none",
    },
    "zelezniak": {
        "lmdb": "data/torchcell/metabolite_zelezniak2018/processed/lmdb",
        "label": "Zelezniak 2018 metabolome",
        "key": "metabolite_level",
        "transform": "log2",
    },
    "dasilveira": {
        "lmdb": "data/torchcell/metabolite_dasilveira2014/processed/lmdb",
        "label": "da Silveira 2014 lipids",
        "key": "metabolite_level",
        "transform": "log2_drop_zero",
    },
}
SOURCES = {"kemmeren": "Kemmeren mRNA", "messner": "Messner protein"}
MIN_SHARED = 40  # deletions a (source, panel, metabolite) needs
MIN_GENE_COVER = (
    0.95  # fraction of shared deletions a Messner protein must be measured in
)
N_PERM_UNI = 200
N_PERM_RIDGE = 20
N_PERM_MANTEL = 99
FOLDS = 5
ALPHAS = (1e1, 1e2, 1e3, 1e4)
SEED = 0


AA3 = {
    "alanine": "Ala",
    "arginine": "Arg",
    "asparagine": "Asn",
    "aspartate": "Asp",
    "citrulline": "Cit",
    "glutamate": "Glu",
    "glutamine": "Gln",
    "glycine": "Gly",
    "histidine": "His",
    "isoleucine": "Ile",
    "leucine": "Leu",
    "lysine": "Lys",
    "methionine": "Met",
    "ornithine": "Orn",
    "phenylalanine": "Phe",
    "proline": "Pro",
    "serine": "Ser",
    "threonine": "Thr",
    "tryptophan": "Trp",
    "tyrosine": "Tyr",
    "valine": "Val",
}


def _short(name: str) -> str:
    """Three-letter amino-acid codes for the tick labels; Cooper's co-eluting peaks
    keep their '+'.
    """
    parts = [AA3.get(p, p.replace("_", " ")) for p in name.split("+")]
    return "+".join(parts)


def _deletion(rec: dict[str, Any]) -> str:
    dels = [
        p["systematic_gene_name"]
        for p in rec["experiment"]["genotype"]["perturbations"]
        if "deletion" in str(p.get("perturbation_type", ""))
    ]
    if len(dels) != 1:
        raise ValueError(f"expected one deletion, got {dels}")
    return str(dels[0])


def _panel_matrix(records: list[dict[str, Any]], spec: dict[str, Any]) -> pd.DataFrame:
    """Deletion x metabolite, transformed; duplicate deletions averaged."""
    rows: dict[str, list[pd.Series]] = {}
    for rec in records:
        ph = rec["experiment"]["phenotype"]
        val = ph[spec["key"]]
        s = pd.Series(val if isinstance(val, dict) else {spec["key"]: val}, dtype=float)
        if spec["transform"] == "log2":
            s = pd.Series(np.log2(s.to_numpy()), index=s.index)
        elif spec["transform"] == "log2_drop_zero":
            s = pd.Series(np.log2(s.where(s > 0).to_numpy()), index=s.index)
        rows.setdefault(_deletion(rec), []).append(s)
    return pd.DataFrame(
        {d: pd.concat(v, axis=1).mean(axis=1) for d, v in rows.items()}
    ).T


def _shared(X: pd.DataFrame, Y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both to shared deletions; keep well-covered genes, mean-impute the rest,
    drop genes without variance.
    """
    strains = sorted(set(X.index) & set(Y.index))
    A = X.loc[strains]
    A = A.loc[:, A.notna().mean() >= MIN_GENE_COVER]
    A = A.fillna(A.mean())
    A = A.loc[:, A.std() > 0]
    return A, Y.loc[strains]


def _corr_matrix(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson of every column of A with y (both finite, y 1-d)."""
    Az = (A - A.mean(0)) / A.std(0)
    yz = (y - y.mean()) / y.std()
    return np.asarray(Az.T @ yz / len(y))


def _bh(p: np.ndarray, q: float = 0.05) -> int:
    p = np.sort(p)
    k = np.arange(1, len(p) + 1)
    ok = p <= q * k / len(p)
    return int(k[ok].max()) if ok.any() else 0


def _univariate(
    A: pd.DataFrame, y: pd.Series, rng: np.random.Generator
) -> dict[str, Any]:
    m = y.notna().to_numpy()
    Am = A.to_numpy()[m]
    ym = y.to_numpy()[m]
    n = int(m.sum())
    r = _corr_matrix(Am, ym)
    tt = r * np.sqrt((n - 2) / np.clip(1 - r**2, 1e-12, None))
    p = 2 * t_dist.sf(np.abs(tt), n - 2)
    null = np.array(
        [np.abs(_corr_matrix(Am, rng.permutation(ym))).max() for _ in range(N_PERM_UNI)]
    )
    order = np.argsort(-np.abs(r))[:5]
    return {
        "n": n,
        "max_abs_r": float(np.abs(r).max()),
        "null_max_abs_r_p95": float(np.quantile(null, 0.95)),
        "null_max_abs_r_median": float(np.median(null)),
        "p_perm": float((null >= np.abs(r).max()).mean()),
        "n_bh_fdr05": _bh(p),
        "top_genes": [(str(A.columns[i]), float(r[i])) for i in order],
    }


class _RidgeSVD:
    """Ridge on a fixed split, factored once so any target and any alpha is a
    matrix-vector product: w = V diag(s / (s^2 + alpha)) U^T y on the standardized
    training block.
    """

    def __init__(self, Atr: np.ndarray, Ate: np.ndarray) -> None:
        mu, sd = Atr.mean(0), Atr.std(0)
        U, s, Vt = np.linalg.svd((Atr - mu) / sd, full_matrices=False)
        self.U, self.s = U, s
        self.P_te = ((Ate - mu) / sd) @ Vt.T

    def predict(self, ytr: np.ndarray, alpha: float) -> np.ndarray:
        c = self.U.T @ (ytr - ytr.mean())
        return np.asarray(self.P_te @ (self.s / (self.s**2 + alpha) * c) + ytr.mean())


class _OOF:
    """Out-of-fold ridge predictions with the alpha chosen on an inner split of each
    training fold; folds fixed by seed so permuted targets reuse the factorization.
    """

    def __init__(self, A: np.ndarray, seed: int) -> None:
        self.folds = []
        for tr, te in KFold(FOLDS, shuffle=True, random_state=seed).split(A):
            outer = _RidgeSVD(A[tr], A[te])
            inner = [
                (itr, ite, _RidgeSVD(A[tr][itr], A[tr][ite]))
                for itr, ite in KFold(3, shuffle=True, random_state=seed + 1).split(tr)
            ]
            self.folds.append((tr, te, outer, inner))

    def pearson(self, y: np.ndarray) -> float:
        pred = np.zeros_like(y)
        for tr, te, outer, inner in self.folds:
            ytr = y[tr]
            best, best_r = ALPHAS[0], -np.inf
            for a in ALPHAS:
                rr = [
                    np.corrcoef(m.predict(ytr[itr], a), ytr[ite])[0, 1]
                    for itr, ite, m in inner
                ]
                if np.nanmean(rr) > best_r:
                    best, best_r = a, float(np.nanmean(rr))
            pred[te] = outer.predict(ytr, best)
        return float(np.corrcoef(pred, y)[0, 1])


def _multivariate(
    A: pd.DataFrame, y: pd.Series, rng: np.random.Generator
) -> dict[str, Any]:
    m = y.notna().to_numpy()
    Am = A.to_numpy()[m]
    ym = y.to_numpy()[m]
    oof = _OOF(Am, SEED)
    r = oof.pearson(ym)
    null = np.array([oof.pearson(rng.permutation(ym)) for _ in range(N_PERM_RIDGE)])
    return {
        "n": int(m.sum()),
        "oof_pearson": r,
        "null_oof_p95": float(np.quantile(null, 0.95)),
        "null_oof_mean": float(null.mean()),
        "null_oof_sd": float(null.std()),
        "p_perm": float((null >= r).mean()),
    }


def _mantel(
    A: pd.DataFrame,
    Y: pd.DataFrame,
    rng: np.random.Generator,
    n_perm: int = N_PERM_MANTEL,
) -> dict[str, Any]:
    """Profile similarity against metabolite distance over deletion pairs."""
    keep = Y.notna().all(axis=1)
    Am = A.loc[keep].to_numpy()
    Ym = Y.loc[keep].to_numpy()
    Ym = (Ym - Ym.mean(0)) / Ym.std(0)
    Az = (Am - Am.mean(1, keepdims=True)) / Am.std(1, keepdims=True)
    sim = (Az @ Az.T) / Am.shape[1]
    iu = np.triu_indices(len(Am), k=1)
    s = sim[iu]
    d = pdist(Ym, "euclidean")
    rs = rankdata(s)
    rs = (rs - rs.mean()) / rs.std()

    def rho_of(dd: np.ndarray) -> float:
        rd = rankdata(dd)
        return float(np.mean(rs * (rd - rd.mean()) / rd.std()))

    rho = rho_of(d)
    null_a = np.array(
        [
            rho_of(pdist(Ym[rng.permutation(len(Am))], "euclidean"))
            for _ in range(n_perm)
        ]
    )
    return {
        "n_deletions": int(keep.sum()),
        "n_pairs": int(len(s)),
        "spearman": rho,
        "null_p95_abs": float(np.quantile(np.abs(null_a), 0.95))
        if n_perm
        else float("nan"),
        "p_perm": float((np.abs(null_a) >= abs(rho)).mean())
        if n_perm
        else float("nan"),
        "_sim": s,
        "_dist": d,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw the figure from the saved JSON; the Mantel scatter of panel f is "
        "recomputed without its permutation null",
    )
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    images = osp.join(os.environ["ASSET_IMAGES_DIR"], "028-knockout-expression")
    os.makedirs(images, exist_ok=True)
    results_dir = experiment_results_dir("028-knockout-expression", __file__)
    results_path = osp.join(results_dir, "expression_metabolic_yko_correlation.json")
    rng = np.random.default_rng(SEED)

    kem, _, _ = _profiles(_load_lmdb(osp.join(data_root, LMDB["kemmeren"])))
    sources = {
        "kemmeren": _matrix(kem, sorted(kem)),
        "messner": _messner(_load_lmdb(osp.join(data_root, LMDB_PROTEOME)))[0],
    }
    panels = {
        k: _panel_matrix(_load_lmdb(osp.join(data_root, v["lmdb"])), v)
        for k, v in PANELS.items()
    }
    for k, v in sources.items():
        print(f"source {k}: {v.shape}")
    for k, v in panels.items():
        print(f"panel {k}: {v.shape}")

    out: dict[str, Any] = {
        "generated_by": "experiments/028-knockout-expression/scripts/expression_metabolic_yko_correlation.py",
        "settings": {
            "min_shared": MIN_SHARED,
            "min_gene_cover": MIN_GENE_COVER,
            "n_perm_univariate": N_PERM_UNI,
            "n_perm_ridge": N_PERM_RIDGE,
            "n_perm_mantel": N_PERM_MANTEL,
            "folds": FOLDS,
            "alphas": list(ALPHAS),
        },
        "shapes": {
            **{f"source_{k}": list(v.shape) for k, v in sources.items()},
            **{f"panel_{k}": list(v.shape) for k, v in panels.items()},
        },
        "reads": {},
    }
    mantel_keep: dict[str, dict[str, Any]] = {}
    if args.plot_only:
        with open(results_path) as f:
            out = json.load(f)
        for key in ("kemmeren__mulleder", "messner__mulleder"):
            sk, pk = key.split("__")
            A, Ym = _shared(sources[sk], panels[pk])
            man = _mantel(A, Ym, rng, n_perm=0)
            man.update(out["reads"][key]["mantel"])
            mantel_keep[key] = man
    for sk, X in sources.items():
        for pk, Y in panels.items():
            if args.plot_only:
                break
            A, Ym = _shared(X, Y)
            key = f"{sk}__{pk}"
            rec: dict[str, Any] = {
                "n_shared_deletions": int(len(A)),
                "n_genes": int(A.shape[1]),
                "metabolites": {},
            }
            print(f"\n{key}: {len(A)} shared deletions, {A.shape[1]} genes")
            if len(A) < MIN_SHARED:
                rec["skipped"] = f"fewer than {MIN_SHARED} shared deletions"
                out["reads"][key] = rec
                continue
            for met in Ym.columns:
                y = Ym[met]
                if y.notna().sum() < MIN_SHARED or y.std() == 0:
                    continue
                uni = _univariate(A, y, rng)
                mul = _multivariate(A, y, rng)
                rec["metabolites"][met] = {"univariate": uni, "multivariate": mul}
                print(
                    f"  {met:28s} n={uni['n']:5d} max|r|={uni['max_abs_r']:.3f} "
                    f"(null95 {uni['null_max_abs_r_p95']:.3f}, FDR hits {uni['n_bh_fdr05']:4d}) "
                    f"ridge oof r={mul['oof_pearson']:+.3f} (null95 {mul['null_oof_p95']:+.3f})"
                )
            if Ym.shape[1] >= 3 and Ym.notna().all(axis=1).sum() >= MIN_SHARED:
                man = _mantel(A, Ym, rng)
                rec["mantel"] = {k: v for k, v in man.items() if not k.startswith("_")}
                mantel_keep[key] = man
                print(
                    f"  mantel: Spearman {man['spearman']:+.3f} over {man['n_pairs']:,} pairs, "
                    f"null95 |rho| {man['null_p95_abs']:.3f}, p {man['p_perm']:.3f}"
                )
            out["reads"][key] = rec

    if not args.plot_only:
        with open(results_path, "w") as f:
            json.dump(out, f, indent=1)

    # ------------------------------------------------------------------ figure
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
    legend_kw = dict(
        frameon=True, edgecolor="black", fancybox=False, framealpha=1.0, markerscale=2.5
    )
    col = {"kemmeren": PLOT_PALETTE[3], "messner": PLOT_PALETTE[1]}
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(110))
    )

    def bars(ax: Any, pk: str, title: str) -> None:
        mets = sorted(
            {
                m
                for sk in sources
                for m in out["reads"].get(f"{sk}__{pk}", {}).get("metabolites", {})
            }
        )
        if not mets:
            ax.set_title(f"{title}: no shared deletions")
            return

        # order by the Messner read when present, else Kemmeren
        def score(m: str) -> float:
            for sk in ("messner", "kemmeren"):
                v = out["reads"].get(f"{sk}__{pk}", {}).get("metabolites", {}).get(m)
                if v:
                    return float(v["multivariate"]["oof_pearson"])
            return 0.0

        mets.sort(key=score, reverse=True)
        x = np.arange(len(mets))
        w = 0.38
        for i, sk in enumerate(("kemmeren", "messner")):
            reads = out["reads"].get(f"{sk}__{pk}", {}).get("metabolites", {})
            vals = [
                reads[m]["multivariate"]["oof_pearson"] if m in reads else np.nan
                for m in mets
            ]
            null = [
                reads[m]["multivariate"]["null_oof_p95"] if m in reads else np.nan
                for m in mets
            ]
            n = max((v["multivariate"]["n"] for v in reads.values()), default=0)
            ax.bar(
                x + (i - 0.5) * w,
                vals,
                w,
                facecolor=col[sk],
                edgecolor="black",
                lw=0.4,
                label=f"{SOURCES[sk]} (n = {n:,})",
            )
            ax.plot(
                x + (i - 0.5) * w,
                null,
                ls="none",
                marker="_",
                color="black",
                ms=5,
                mew=0.8,
            )
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in mets], rotation=90, fontsize=5)
        ax.set_ylabel("out-of-fold Pearson, ridge from full profile")
        ax.set_title(title)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(axis="y", which="both", lw=0.3, color="0.85")
        ax.tick_params(which="minor", length=0)
        ax.set_axisbelow(True)
        # headroom so the framed legend sits above every bar (white-cross rule)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.05), hi + 0.55 * (hi - min(lo, -0.05)))
        ax.legend(loc="upper right", **legend_kw)

    bars(axes[0, 0], "mulleder", "amino acids, Mulleder (tick = permutation 95th)")
    bars(axes[0, 1], "cooper", "amino acids, Cooper")

    # c. the one-number pigment screens. Zelezniak and da Silveira share under 125
    # deletions with either source and stay in the JSON only.
    ax = axes[0, 2]
    small = []
    for pk, lab in (("cachera", "betaxanthin"), ("ozaydin", "beta-carotene")):
        for sk in sources:
            reads = out["reads"].get(f"{sk}__{pk}", {}).get("metabolites", {})
            for v in reads.values():
                small.append((lab, sk, v["multivariate"]))
    if small:
        labels = sorted({s[0] for s in small})
        x = np.arange(len(labels))
        w = 0.38
        for i, sk in enumerate(("kemmeren", "messner")):
            vals = [
                next(
                    (s[2]["oof_pearson"] for s in small if s[0] == lab and s[1] == sk),
                    np.nan,
                )
                for lab in labels
            ]
            null = [
                next(
                    (s[2]["null_oof_p95"] for s in small if s[0] == lab and s[1] == sk),
                    np.nan,
                )
                for lab in labels
            ]
            ax.bar(
                x + (i - 0.5) * w,
                vals,
                w,
                facecolor=col[sk],
                edgecolor="black",
                lw=0.4,
                label=SOURCES[sk]
                + " (n = "
                + ", ".join(
                    f"{s[2]['n']:,}"
                    for lab in labels
                    for s in small
                    if s[0] == lab and s[1] == sk
                )
                + ")",
            )
            ax.plot(
                x + (i - 0.5) * w,
                null,
                ls="none",
                marker="_",
                color="black",
                ms=5,
                mew=0.8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("out-of-fold Pearson, ridge from full profile")
        ax.set_title("pigment screens, one number per deletion")
        ax.set_ylim(-0.05, 0.75)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(axis="y", which="both", lw=0.3, color="0.85")
        ax.tick_params(which="minor", length=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", **legend_kw)

    # d. univariate: observed max |r| against its permutation null, every read
    ax = axes[1, 0]
    for sk in sources:
        xs, ys = [], []
        for pk in ("mulleder", "cooper", "cachera", "ozaydin"):
            for m, v in (
                out["reads"].get(f"{sk}__{pk}", {}).get("metabolites", {}).items()
            ):
                xs.append(v["univariate"]["null_max_abs_r_p95"])
                ys.append(v["univariate"]["max_abs_r"])
        ax.scatter(
            xs, ys, s=6, color=col[sk], edgecolor="black", lw=0.2, label=SOURCES[sk]
        )
    lim = ax.get_xlim(), ax.get_ylim()
    hi = max(lim[0][1], lim[1][1])
    ax.plot([0, hi], [0, hi], color="black", lw=0.5, ls="--")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi * 1.35)
    ax.set_xlabel("permutation null: 95th percentile of max |r| over genes")
    ax.set_ylabel("observed max |r| over genes")
    ax.set_title("best single gene per metabolite, four large panels")
    ax.legend(loc="upper left", **legend_kw)

    # e. the strongest single pair
    ax = axes[1, 1]
    best_pair = None
    for sk, X in sources.items():
        for pk in ("mulleder", "cooper", "cachera", "ozaydin"):
            for m, v in (
                out["reads"].get(f"{sk}__{pk}", {}).get("metabolites", {}).items()
            ):
                gap = (
                    v["univariate"]["max_abs_r"] - v["univariate"]["null_max_abs_r_p95"]
                )
                if best_pair is None or gap > best_pair[0]:
                    best_pair = (gap, sk, pk, m, v["univariate"]["top_genes"][0])
    if best_pair is not None:
        _, sk, pk, m, (gene, r) = best_pair
        A, Ym = _shared(sources[sk], panels[pk])
        y = Ym[m]
        ok = y.notna()
        ax.scatter(
            A.loc[ok, gene], y[ok], s=4, color=col[sk], edgecolor="black", lw=0.15
        )
        ax.set_xlabel(f"{SOURCES[sk]}: {gene} log2 ratio")
        ax.set_ylabel(f"{PANELS[pk]['label'].split()[0]}: {m}")
        ax.set_title(f"strongest pair, r = {r:+.2f} (n = {int(ok.sum()):,})")
        out["best_pair"] = {
            "source": sk,
            "panel": pk,
            "metabolite": m,
            "gene": gene,
            "r": r,
        }

    # f. structure: the mRNA read, which is the stronger of the two
    ax = axes[1, 2]
    mk = next(
        (k for k in ("kemmeren__mulleder", "messner__mulleder") if k in mantel_keep),
        None,
    )
    if mk is not None:
        man = mantel_keep[mk]
        sk = mk.split("__")[0]
        ax.hexbin(
            man["_sim"],
            man["_dist"],
            gridsize=45,
            bins="log",
            cmap=_palette_cmap(col[sk]),
            linewidths=0,
        )
        ax.set_xlabel(f"{SOURCES[sk]}: deletion x deletion profile Pearson")
        ax.set_ylabel("Mulleder amino-acid distance (z-scored)")
        ax.set_title(
            f"alike profiles, alike amino acids (Spearman {man['spearman']:+.2f})"
        )
        ax.set_xlabel(f"{SOURCES[sk]}: deletion x deletion Pearson")

    for ax, letter in zip(axes.ravel(), "abcdef"):
        panel_label(ax, letter)
    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.95, bottom=0.12, wspace=0.35, hspace=0.75
    )
    stem = osp.join(images, f"expression_metabolic_yko_correlation_{timestamp()}")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", stem + ".svg")


if __name__ == "__main__":
    main()
