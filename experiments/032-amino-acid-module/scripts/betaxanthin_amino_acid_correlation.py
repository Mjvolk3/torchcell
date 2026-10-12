# experiments/032-amino-acid-module/scripts/betaxanthin_amino_acid_correlation.py
# [[experiments.032-amino-acid-module.scripts.betaxanthin_amino_acid_correlation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/032-amino-acid-module/scripts/betaxanthin_amino_acid_correlation

r"""Does a deletion's betaxanthin level correlate with its amino-acid pool? Cachera 2023
against Mulleder 2016 and against Cooper 2010, strain by strain.

The three screens profile the same deletion collection, so every gene both screens hold
gives one paired observation. The question is whether the betaxanthin endpoint and the
amino-acid endpoint share any strain-level signal, which is what a joint model of the two
would learn from. Three reads, each on the strains both screens measured:

* per amino acid (19 Mulleder concentrations, 17 Cooper peaks): the Spearman of the
  betaxanthin score against the amino-acid value, with a strain bootstrap (1,000 draws)
  for the interval, and the overlap of the top 5 percent of strains on the two axes
  against the count expected by chance;
* the pool as a whole: ridge regression from the standardized log amino-acid profile to
  the betaxanthin score, out of fold over five strain folds, scored by the out-of-fold
  Spearman (the alpha chosen on the training folds by an inner split), so a signal that
  is spread thinly over the pool is not missed by the one-at-a-time read, beside the
  same ridge from the deleted gene's propagated GO multi-hot (the betaxanthin study's
  best genotype-side input) and from GO with the pool appended, on the same strains and
  folds, so the pool's contribution beyond the gene's identity is read directly; and
* the ceiling every read is bounded by: the geometric mean of the two endpoints'
  reliabilities (betaxanthin's stored value from the 027 variance-components read,
  Mulleder's per-amino-acid retest, Cooper's duplicate-strain agreement), which is the
  correlation two perfectly related latent quantities would show through this noise.

Cachera's score is the corrected 24 h colony fluorescence, population-centered, so
negative values are ordinary; Mulleder's values are mM on the prototroph on minimal
medium; Cooper's are linear ratios to the plate mean on the auxotroph in complete medium.
Missing values are left out pair by pair; for the ridge a missing amino acid is filled
with the column median over the training fold and the filled fraction is recorded.

Writes ``results/betaxanthin_amino_acid_correlation.json`` and a two-panel figure to
``ASSET_IMAGES_DIR/032-amino-acid-module/`` (Spearman per amino acid with the bootstrap
interval, the ceiling drawn beside each bar).

Run from the repo root::

    python experiments/032-amino-acid-module/scripts/betaxanthin_amino_acid_correlation.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import hypergeom, spearmanr

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
IMAGES = osp.join(os.environ["ASSET_IMAGES_DIR"], "032-amino-acid-module")
RESULTS = osp.join(EXPERIMENT_ROOT, "032-amino-acid-module", "results")
CACHERA = osp.join(DATA_ROOT, "data/torchcell/betaxanthin_cachera2023")
MULLEDER = osp.join(DATA_ROOT, "data/torchcell/amino_acid_mulleder2016")
COOPER = osp.join(DATA_ROOT, "data/torchcell/amino_acid_cooper2010")
RELIABILITY = osp.join(
    EXPERIMENT_ROOT, "027-betaxanthin-metabolic/results/target_reliability.json"
)
MULLEDER_RETEST = osp.join(
    EXPERIMENT_ROOT,
    "027-betaxanthin-metabolic/results/mulleder_retest_reliability.json",
)
CONCORDANCE = osp.join(RESULTS, "amino_acid_dataset_concordance.json")
#: The propagated GO multi-hot over every annotated gene (target_go_affine_store.py).
GO_STORE = osp.join(
    DATA_ROOT, "data/torchcell/reference_flux/ess_solve_encodings_YPD_bx_go.npz"
)
MULLEDER_KEYS = [
    "alanine",
    "arginine",
    "asparagine",
    "aspartate",
    "glutamate",
    "glutamine",
    "glycine",
    "histidine",
    "isoleucine",
    "leucine",
    "lysine",
    "methionine",
    "phenylalanine",
    "proline",
    "serine",
    "threonine",
    "tryptophan",
    "tyrosine",
    "valine",
]
COOPER_KEYS = [
    "lysine_related_peak1",
    "arginine",
    "gshbiotin",
    "n_acetylornithine",
    "leucine+isoleucine+citrulline",
    "glutamine+valine",
    "methionine+proline",
    "threonine",
    "alanine",
    "serine",
    "asparagine+tyrosine",
    "glycine",
    "lysine_a",
    "ornithine",
    "lysine_b",
    "glutamate",
    "aspartate",
]
TOP_FRAC = 0.05
N_BOOT = 1000
N_FOLDS = 5
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


AA3 = {
    "alanine": "Ala",
    "arginine": "Arg",
    "asparagine": "Asn",
    "aspartate": "Asp",
    "glutamate": "Glu",
    "glutamine": "Gln",
    "glycine": "Gly",
    "histidine": "His",
    "isoleucine": "Ile",
    "leucine": "Leu",
    "lysine": "Lys",
    "methionine": "Met",
    "phenylalanine": "Phe",
    "proline": "Pro",
    "serine": "Ser",
    "threonine": "Thr",
    "tryptophan": "Trp",
    "tyrosine": "Tyr",
    "valine": "Val",
    "citrulline": "Cit",
    "lysine_a": "Lys a",
    "lysine_b": "Lys b",
    "lysine_related_peak1": "Lys-related",
    "gshbiotin": "GSH-biotin",
    "n_acetylornithine": "N-acetyl-Orn",
    "ornithine": "Orn",
}


def short(name: str) -> str:
    """Three-letter codes for the tick labels; composite peaks keep their plus."""
    return "+".join(AA3.get(part, part) for part in name.split("+"))


class FeatureStat(BaseModel):
    """The betaxanthin score against one amino-acid quantity over shared strains."""

    screen: str
    key: str
    n: int
    spearman: float
    spearman_p: float
    ci_low: float
    ci_high: float
    top5_overlap: int
    top5_expected: float
    top5_enrichment: float
    top5_hypergeom_p: float
    #: The amino acid's own reliability (retest or duplicate agreement) and the ceiling
    #: a correlation with the betaxanthin score could reach through both noises.
    reliability: float | None
    ceiling: float | None


class PoolFit(BaseModel):
    """Ridge to the betaxanthin score, out of fold, from the amino-acid profile
    (``pool``), from the deleted gene's GO multi-hot (``go``), or from both
    (``go+pool``), on the same strains and the same folds.
    """

    screen: str
    inputs: str
    n: int
    #: Strains with a GO row in the store (the rest read an all-zero row).
    n_with_go: int
    n_features: int
    filled_fraction: float
    alpha_by_fold: list[float]
    spearman_oof: float
    pearson_oof: float
    spearman_by_fold: list[float]
    permuted_spearman_oof: list[float]


class Report(BaseModel):
    """The whole read: manifests, shared counts, per-quantity statistics, pool fits."""

    cachera_manifest: dict[str, Any]
    mulleder_manifest: dict[str, Any]
    cooper_manifest: dict[str, Any]
    n_cachera: int
    n_shared_mulleder: int
    n_shared_cooper: int
    betaxanthin_reliability: float
    features: list[FeatureStat]
    pool: list[PoolFit]
    figure: str
    note: str


def manifest(root: str) -> dict[str, Any]:
    with open(osp.join(root, "preprocess", "build_manifest.json")) as fh:
        m = json.load(fh)
    return {k: v for k, v in m.items() if not isinstance(v, (dict, list))}


def feature_stat(
    screen: str,
    key: str,
    bx: np.ndarray,
    aa: np.ndarray,
    reliability: float | None,
    bx_reliability: float,
    rng: np.random.Generator,
) -> FeatureStat:
    ok = np.isfinite(bx) & np.isfinite(aa)
    x, y = bx[ok], aa[ok]
    rho, p = spearmanr(x, y)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, x.size, x.size)
        draws[b] = spearmanr(x[idx], y[idx]).correlation
    k = max(1, int(round(TOP_FRAC * x.size)))
    top_x = set(np.argsort(-x)[:k].tolist())
    top_y = set(np.argsort(-y)[:k].tolist())
    overlap = len(top_x & top_y)
    expected = k * k / x.size
    return FeatureStat(
        screen=screen,
        key=key,
        n=int(x.size),
        spearman=float(rho),
        spearman_p=float(p),
        ci_low=float(np.percentile(draws, 2.5)),
        ci_high=float(np.percentile(draws, 97.5)),
        top5_overlap=overlap,
        top5_expected=float(expected),
        top5_enrichment=float(overlap / expected),
        top5_hypergeom_p=float(hypergeom.sf(overlap - 1, x.size, k, k)),
        reliability=reliability,
        ceiling=None
        if reliability is None
        else float(np.sqrt(max(reliability, 0.0) * bx_reliability)),
    )


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge on centered inputs; returns (weights, intercept)."""
    mu = x.mean(axis=0)
    xc = x - mu
    ym = y.mean()
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ (y - ym))
    return w, float(ym - mu @ w)


def pool_fit(
    screen: str,
    inputs: str,
    bx: np.ndarray,
    profile: np.ndarray | None,
    go: np.ndarray | None,
    folds: list[np.ndarray],
    rng: np.random.Generator,
) -> PoolFit:
    """Five strain folds (shared by every fit on one screen); the alpha of each fold
    chosen on an inner split of its training strains; the profile log-scaled,
    median-filled and standardized on the training fold only; the GO multi-hot
    standardized on the training fold only; the two concatenated when both are given,
    the GO block scaled to the pool block's total variance so one penalty serves both.
    """
    n = bx.size
    assert np.isfinite(bx).all()
    n_features = (0 if profile is None else profile.shape[1]) + (
        0 if go is None else go.shape[1]
    )
    pred = np.full(n, np.nan)
    alphas, by_fold = [], []
    filled = 0
    for f in range(N_FOLDS):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(N_FOLDS) if g != f])
        blocks = []
        if profile is not None:
            lp = np.where(
                profile > 0, np.log10(np.where(profile > 0, profile, 1.0)), np.nan
            )
            med = np.nanmedian(lp[train], axis=0)
            filled += int(np.isnan(lp[test]).sum())
            blocks.append(np.where(np.isnan(lp), med, lp))
        if go is not None:
            blocks.append(go.astype(float))
        raw = np.concatenate(blocks, axis=1)
        mu, sd = raw[train].mean(axis=0), raw[train].std(axis=0)
        sd[sd == 0] = 1.0
        z = (raw - mu) / sd
        if profile is not None and go is not None:
            # One penalty over two blocks of very different width would let the 2,362
            # GO columns swamp the 19 pool columns, so the GO block is scaled to carry
            # the same total variance as the pool block.
            z[:, profile.shape[1] :] *= np.sqrt(profile.shape[1] / go.shape[1])
        # The inner split: the last fifth of the training strains selects alpha.
        cut = int(0.8 * train.size)
        inner_tr, inner_va = train[:cut], train[cut:]
        best = max(
            ALPHAS,
            key=lambda a: (
                spearmanr(
                    z[inner_va] @ ridge_fit(z[inner_tr], bx[inner_tr], a)[0],
                    bx[inner_va],
                ).correlation
            ),
        )
        w, b0 = ridge_fit(z[train], bx[train], best)
        pred[test] = z[test] @ w + b0
        alphas.append(best)
        by_fold.append(float(spearmanr(pred[test], bx[test]).correlation))
    permuted = []
    for _ in range(20):
        perm = rng.permutation(n)
        permuted.append(float(spearmanr(pred, bx[perm]).correlation))
    return PoolFit(
        screen=screen,
        inputs=inputs,
        n=int(n),
        n_with_go=0 if go is None else int((go.sum(axis=1) > 0).sum()),
        n_features=int(n_features),
        filled_fraction=float(filled / (n * n_features)),
        alpha_by_fold=alphas,
        spearman_oof=float(spearmanr(pred, bx).correlation),
        pearson_oof=float(np.corrcoef(pred, bx)[0, 1]),
        spearman_by_fold=by_fold,
        permuted_spearman_oof=permuted,
    )


def main() -> None:
    rng = np.random.default_rng(0)
    bx = pd.read_csv(osp.join(CACHERA, "preprocess", "data.csv")).set_index("orf")
    assert not bx.index.duplicated().any()
    mul = pd.read_csv(osp.join(MULLEDER, "preprocess", "data.csv")).set_index("orf")
    coo = pd.read_csv(osp.join(COOPER, "preprocess", "data.csv"))
    assert not coo["perturbed_gene_name"].duplicated().any()
    coo = coo.set_index("perturbed_gene_name")
    with open(RELIABILITY) as fh:
        rel = json.load(fh)
    bx_rel = float(
        next(t for t in rel["targets"] if t["target"] == "Betaxanthin")[
            "reliability_stored_value"
        ]
    )
    with open(MULLEDER_RETEST) as fh:
        mul_rel = {
            row["amino_acid"]: float(row["spearman_retest_log10"])
            for row in json.load(fh)["per_amino_acid"]
        }
    with open(CONCORDANCE) as fh:
        coo_rel = {
            r["key"]: float(r["spearman"])
            for r in json.load(fh)["cooper_duplicate_retest"]
        }

    shared_m = sorted(set(bx.index) & set(mul.index))
    shared_c = sorted(set(bx.index) & set(coo.index))
    print(
        f"Cachera {len(bx)} strains; shared with Mulleder {len(shared_m)}, "
        f"with Cooper {len(shared_c)}; betaxanthin reliability {bx_rel:.3f}"
    )
    features: list[FeatureStat] = []
    y_m = bx.loc[shared_m, "level"].to_numpy(float)
    m = mul.loc[shared_m, MULLEDER_KEYS].to_numpy(float)
    for j, key in enumerate(MULLEDER_KEYS):
        features.append(
            feature_stat("mulleder2016", key, y_m, m[:, j], mul_rel[key], bx_rel, rng)
        )
    y_c = bx.loc[shared_c, "level"].to_numpy(float)
    c = coo.loc[shared_c, COOPER_KEYS].to_numpy(float)
    for j, key in enumerate(COOPER_KEYS):
        features.append(
            feature_stat("cooper2010", key, y_c, c[:, j], coo_rel.get(key), bx_rel, rng)
        )
    for s in features:
        ceiling = "" if s.ceiling is None else f" ceiling {s.ceiling:.2f}"
        print(
            f"  {s.screen:<12s} {s.key:<30s} n {s.n:4d} rho {s.spearman:+.3f} "
            f"[{s.ci_low:+.3f}, {s.ci_high:+.3f}] p {s.spearman_p:.1e} top5 "
            f"{s.top5_overlap}/{s.top5_expected:.1f} (x{s.top5_enrichment:.1f}){ceiling}"
        )
    # The deleted gene's GO multi-hot, the betaxanthin study's best genotype-side input
    # (the affine store of round 150); a strain outside the store reads an all-zero row.
    go_store = np.load(GO_STORE)
    go_rows = {str(g): i for i, g in enumerate(go_store["genes"])}
    go_all = np.asarray(go_store["v_deletion"], dtype=np.float32)

    def go_matrix(strains: list[str]) -> np.ndarray:
        out = np.zeros((len(strains), go_all.shape[1]), dtype=np.float32)
        for i, s in enumerate(strains):
            if s in go_rows:
                out[i] = go_all[go_rows[s]]
        return out

    pool = []
    for screen, y, prof, strains in (
        ("mulleder2016", y_m, m, shared_m),
        ("cooper2010", y_c, c, shared_c),
    ):
        assert np.isfinite(y).all()
        folds = np.array_split(rng.permutation(y.size), N_FOLDS)
        g = go_matrix(strains)
        print(
            f"  {screen}: {int((g.sum(axis=1) > 0).sum())} of {len(strains)} strains "
            "carry a GO row"
        )
        pool.append(pool_fit(screen, "pool", y, prof, None, folds, rng))
        pool.append(pool_fit(screen, "go", y, None, g, folds, rng))
        pool.append(pool_fit(screen, "go+pool", y, prof, g, folds, rng))
    for p in pool:
        print(
            f"  {p.inputs:<8s} {p.screen:<12s} n {p.n} out-of-fold Spearman {p.spearman_oof:+.3f} "
            f"(folds {[round(v, 3) for v in p.spearman_by_fold]}; permuted "
            f"{np.mean(p.permuted_spearman_oof):+.3f} sd {np.std(p.permuted_spearman_oof):.3f}; "
            f"filled {p.filled_fraction:.3f})"
        )

    # ---- figure: one panel per screen, the Spearman with its interval and ceiling ----
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    fig_w = PANEL_WIDTHS_MM["full"]
    fig_h = 60.0
    fig = plt.figure(figsize=(mm_to_in(fig_w), mm_to_in(fig_h)))
    panels = [
        ("a", "mulleder2016", MULLEDER_KEYS, PLOT_PALETTE[0], 0.085, 0.47),
        ("b", "cooper2010", COOPER_KEYS, PLOT_PALETTE[1], 0.605, 0.385),
    ]
    for letter, screen, keys, color, left, width in panels:
        ax = fig.add_axes([left, 0.40, width, 0.48])
        rows = [s for s in features if s.screen == screen]
        xs = np.arange(len(rows))
        vals = np.array([s.spearman for s in rows])
        lo = vals - np.array([s.ci_low for s in rows])
        hi = np.array([s.ci_high for s in rows]) - vals
        ax.bar(
            xs,
            vals,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            width=0.7,
            label="Spearman, strains both screens hold",
        )
        ax.errorbar(
            xs,
            vals,
            yerr=[lo, hi],
            fmt="none",
            ecolor="black",
            elinewidth=0.5,
            capsize=1.2,
            capthick=0.5,
        )
        ceilings = [np.nan if s.ceiling is None else s.ceiling for s in rows]
        ax.plot(
            xs,
            ceilings,
            linestyle="none",
            marker="o",
            markersize=2.6,
            markerfacecolor=PLOT_PALETTE[2],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="ceiling from the two reliabilities",
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([short(k) for k in keys], rotation=90)
        if letter == "a":
            ax.set_ylabel("Spearman with betaxanthin")
        else:
            ax.set_yticklabels([])
        ax.set_ylim(-0.2, 0.6)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax.grid(axis="y", which="both", linewidth=0.3, color="#CCCCCC")
        ax.tick_params(which="minor", length=0)
        ax.tick_params(length=2, width=0.5)
        ax.set_xlim(-0.7, len(rows) - 0.3)
        if letter == "a":
            ax.legend(
                loc="lower left",
                bbox_to_anchor=(0, 1.03),
                ncol=2,
                frameon=False,
                fontsize=6,
                handlelength=1.2,
                columnspacing=1.0,
            )
        ax.text(
            -0.12 if letter == "a" else -0.05,
            1.03,
            letter,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
    os.makedirs(IMAGES, exist_ok=True)
    stamp = timestamp()
    svg = osp.join(IMAGES, f"betaxanthin_amino_acid_correlation_{stamp}.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(svg.replace(".svg", ".png"), dpi=300)
    plt.close(fig)

    report = Report(
        cachera_manifest=manifest(CACHERA),
        mulleder_manifest=manifest(MULLEDER),
        cooper_manifest=manifest(COOPER),
        n_cachera=int(len(bx)),
        n_shared_mulleder=len(shared_m),
        n_shared_cooper=len(shared_c),
        betaxanthin_reliability=bx_rel,
        features=features,
        pool=pool,
        figure=svg,
        note=(
            "Betaxanthin: Cachera 2023 corrected 24 h colony fluorescence per gene "
            "(population-centered score). Amino acids: Mulleder 2016 mM (prototroph, "
            "minimal medium) and Cooper 2010 ratio to the plate mean (auxotroph, complete "
            "medium). Strains matched on the systematic ORF. The ceiling is "
            "sqrt(reliability_betaxanthin * reliability_amino_acid) with the betaxanthin "
            "reliability the stored value of the 027 variance-components read, Mulleder's "
            "the log10 retest Spearman over 194 strains, Cooper's the duplicate-strain "
            "Spearman. The pool fit is ridge over the log10 profile, standardized and "
            "median-filled on the training fold, five strain folds, alpha chosen on an "
            "inner split, scored out of fold; the permuted reference shuffles the target. "
            "The go and go+pool fits use the same folds with the deleted gene's propagated "
            "GO multi-hot (ess_solve_encodings_YPD_bx_go.npz), standardized on the training "
            "fold, alone and beside the pool."
        ),
    )
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "betaxanthin_amino_acid_correlation.json")
    with open(out, "w") as fh:
        fh.write(report.model_dump_json(indent=2))
    print(out)
    print(svg)


if __name__ == "__main__":
    main()
