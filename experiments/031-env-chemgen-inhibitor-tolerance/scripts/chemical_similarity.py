# experiments/031-env-chemgen-inhibitor-tolerance/scripts/chemical_similarity.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.chemical_similarity]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/chemical_similarity
"""Do Vanacloig's inhibitors have chemical neighbors in Hillenmeyer, and does chemical
similarity predict response similarity?

Exact InChIKey overlap between Vanacloig 2022 and Hillenmeyer 2008 is two compounds per
arm. This script asks the softer question for every molecular encoder in
``results/embeddings/`` (written by ``embed_compounds.py``):

1. **Nearest neighbor.** For each Vanacloig compound, the most similar Hillenmeyer
   compound under that encoder (Tanimoto for count/bit fingerprints, cosine for dense
   embeddings), the similarity, and whether the neighbor is an exact match.
2. **Chemistry vs response.** Over every (Vanacloig condition, partner condition) pair
   that has both a chemical similarity and a cross-dataset response Spearman
   (``cross_spearman_<partner>.csv`` from ``cross_dataset_similarity.py``), the Spearman
   between the two. If chemically similar compounds produce similar per-gene response
   profiles, a model can use chemistry to transfer across datasets even without exact
   matches. The top-decile pairs by chemical similarity are also compared with the rest.

Writes ``results/chemical_similarity_summary.csv`` (one row per encoder x partner),
``results/nearest_neighbors_<encoder>_<partner>.csv``, and one figure per partner.
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

from torchcell.molecule.similarity import cosine_matrix, tanimoto_matrix  # noqa: E402
from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
EMB_DIR = osp.join(RESULTS_DIR, "embeddings")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "031-env-chemgen-inhibitor-tolerance")
STABLE_NAMES = "--stable" in sys.argv
INK = "#000000"
ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
FINGERPRINTS = {"ecfp4_count", "ecfp4_bit", "fcfp4_count", "maccs"}
PARTNERS = {"hom": "hillenmeyer2008_hom", "het": "hillenmeyer2008_het"}


def _apply_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            "savefig.bbox": None,
        }
    )


def _box(ax: Axes) -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=INK, width=0.5, length=2)


def _save(fig: plt.Figure, name: str) -> str:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    stem = osp.join(IMAGES_DIR, name if STABLE_NAMES else f"{name}_{timestamp()}")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    return stem + ".svg"


# ------------------------------------------------------------------------ data
def compounds_of(name: str) -> pd.DataFrame:
    """Distinct (inchikey, compound name) of a flattened dataset, one compound per row."""
    df = pd.read_parquet(
        osp.join(RESULTS_DIR, f"records_{name}.parquet"),
        columns=["inchikey", "compound", "n_small_molecules"],
    )
    df = df[
        df["n_small_molecules"] == 1
    ]  # a single dosed compound is a single condition
    out = df.drop_duplicates("inchikey")[["inchikey", "compound"]]
    return out[out["inchikey"] != ""].reset_index(drop=True)


def load_embedding(encoder: str) -> tuple[dict[str, int], np.ndarray]:
    z = np.load(osp.join(EMB_DIR, f"{encoder}.npz"), allow_pickle=True)
    keys = [str(k) for k in z["inchikey"]]
    X = np.asarray(z["X"], dtype=np.float64)
    if (
        encoder == "rdkit_2d"
    ):  # descriptors: impute NaN by column median, then standardize
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), med, X)
        sd = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return {k: i for i, k in enumerate(keys)}, X


def similarity(
    encoder: str, X: np.ndarray, ia: np.ndarray, ib: np.ndarray
) -> np.ndarray:
    if encoder in FINGERPRINTS:
        return tanimoto_matrix(X[ia], X[ib])
    return cosine_matrix(X[ia], X[ib])


# -------------------------------------------------------------------- analysis
def run(
    encoder: str, partner: str, V: pd.DataFrame, H: pd.DataFrame, cross: pd.DataFrame
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray, np.ndarray]:
    idx, X = load_embedding(encoder)
    v = V[V["inchikey"].isin(idx)].reset_index(drop=True)
    h = H[H["inchikey"].isin(idx)].reset_index(drop=True)
    S = similarity(
        encoder, X, v["inchikey"].map(idx).to_numpy(), h["inchikey"].map(idx).to_numpy()
    )
    nn = S.argmax(axis=1)
    neighbors = pd.DataFrame(
        {
            "vanacloig": v["compound"],
            "vanacloig_inchikey": v["inchikey"],
            "neighbor": h.loc[nn, "compound"].to_numpy(),
            "neighbor_inchikey": h.loc[nn, "inchikey"].to_numpy(),
            "similarity": S[np.arange(len(v)), nn],
            "exact": (v["inchikey"].to_numpy() == h.loc[nn, "inchikey"].to_numpy()),
        }
    ).sort_values("similarity", ascending=False)
    # chemistry vs response over the pairs present in the cross matrix
    chem, resp = [], []
    for i, vc in enumerate(v["compound"]):
        if vc not in cross.index:
            continue
        for j, hc in enumerate(h["compound"]):
            if hc in cross.columns and np.isfinite(cross.loc[vc, hc]):
                chem.append(S[i, j])
                resp.append(cross.loc[vc, hc])
    chem_a, resp_a = np.asarray(chem), np.asarray(resp)
    rho, p = spearmanr(chem_a, resp_a)
    q90 = np.quantile(chem_a, 0.9)
    top, rest = resp_a[chem_a >= q90], resp_a[chem_a < q90]
    mw = mannwhitneyu(top, rest, alternative="greater").pvalue
    summary = {
        "encoder": encoder,
        "partner": partner,
        "n_vanacloig": len(v),
        "n_partner": len(h),
        "nn_similarity_median": float(np.median(neighbors["similarity"])),
        "n_exact": int(neighbors["exact"].sum()),
        "n_vanacloig_with_neighbor_above_0.5": int(
            (neighbors["similarity"] >= 0.5).sum()
        ),
        "n_pairs": len(chem_a),
        "chem_vs_response_spearman": rho,
        "chem_vs_response_p": p,
        "top_decile_response_rho": float(np.median(top)),
        "rest_response_rho": float(np.median(rest)),
        "top_vs_rest_mannwhitney_p": mw,
    }
    return summary, neighbors, chem_a, resp_a


def fig_partner(
    partner: str, per_encoder: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]
) -> str:
    encoders = list(per_encoder)
    n = len(encoders)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(40 * nrow + 8)),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.93, bottom=0.10, wspace=0.3, hspace=0.55
    )
    letters = "abcdefghijklmnop"
    for k, enc in enumerate(encoders):
        ax = axes[k // ncol, k % ncol]
        chem, resp, nb = per_encoder[enc]
        ax.scatter(chem, resp, s=1.5, color=GRAY, alpha=0.4, linewidths=0)
        q90 = np.quantile(chem, 0.9)
        sel = chem >= q90
        ax.scatter(
            chem[sel],
            resp[sel],
            s=2.5,
            color=RED,
            linewidths=0,
            label="top decile by chemistry",
        )
        r = spearmanr(chem, resp)[0]
        ax.set_title(
            f"{enc}: rho {r:+.3f}, NN median {np.median(nb['similarity']):.2f}"
        )
        ax.set_xlabel("chemical similarity (Tanimoto / cosine)")
        ax.set_ylabel(f"response Spearman vs {partner.upper()}")
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.grid(True, linewidth=0.3, alpha=0.3)
        _box(ax)
        panel_label(ax, letters[k])
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    path = _save(fig, f"chemistry_vs_response_{partner}")
    plt.close(fig)
    return path


def main() -> None:
    _apply_rc()
    encoders = sorted(osp.basename(p)[:-4] for p in glob(osp.join(EMB_DIR, "*.npz")))
    V = compounds_of("vanacloig2022")
    rows, paths = [], []
    for partner, pname in PARTNERS.items():
        H = compounds_of(pname)
        cross = pd.read_csv(
            osp.join(RESULTS_DIR, f"cross_spearman_{partner}.csv"), index_col=0
        )
        per_encoder = {}
        for enc in encoders:
            summary, neighbors, chem, resp = run(enc, partner, V, H, cross)
            rows.append(summary)
            neighbors.to_csv(
                osp.join(RESULTS_DIR, f"nearest_neighbors_{enc}_{partner}.csv"),
                index=False,
            )
            per_encoder[enc] = (chem, resp, neighbors)
            print(
                f"{partner} {enc:16s} pairs {summary['n_pairs']:5d}  chem-vs-response rho {summary['chem_vs_response_spearman']:+.3f} (p {summary['chem_vs_response_p']:.1e})  NN median {summary['nn_similarity_median']:.2f}  top-decile vs rest {summary['top_decile_response_rho']:+.3f} / {summary['rest_response_rho']:+.3f}"
            )
        paths.append(fig_partner(partner, per_encoder))
    pd.DataFrame(rows).to_csv(
        osp.join(RESULTS_DIR, "chemical_similarity_summary.csv"), index=False
    )
    print("\n".join(paths))


if __name__ == "__main__":
    main()
