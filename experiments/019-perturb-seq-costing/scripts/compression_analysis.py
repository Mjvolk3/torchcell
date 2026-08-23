# experiments/019-perturb-seq-costing/scripts/compression_analysis.py
# [[experiments.019-perturb-seq-costing.scripts.compression_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/compression_analysis
"""Is the yeast perturbation-response matrix compressible, and by how much?

Compressed Perturb-seq (Yao et al. 2024) needs a number of composite samples of
order ``(q + r) log n`` rather than ``n``, where n is the library size, r is the
rank of the perturbation-by-gene effect matrix, and q is the number of non-zeros
per module. Sec. 3.4 states that as the method's premise. Nothing in this
document has ever checked it, and n, q and r have been treated as unplannable.

Two of the three are measurable from data already in the mirror, and this script
measures them. Kemmeren et al.'s 1,484 single-deletion expression profiles ARE a
perturbation-by-gene effect matrix for yeast -- the same object compressed
Perturb-seq would be recovering -- so its singular value spectrum is r and the
concentration of its loadings is q, for our organism rather than for THP1 cells.

It also tests the assumption the whole scheme rests on. Guide-pooling "requires
the non-trivial assumption that the effect sizes of guides tend to combine
additively in log expression space" (Sec. 3.4). Sameith et al.'s double-deletion
compendium is that assumption's direct test in yeast: for every double whose two
singles were also profiled BY THE SAME LAB ON THE SAME PLATFORM, the additive
prediction is the sum of the singles, and the residual is what compressed
recovery would have to absorb.

WHAT THIS CANNOT DO. It says nothing about the OTHER q in this document -- the
per-guide detection probability of Sec. 6.1, an unrelated quantity that shares a
letter. The naming collision is real, is flagged in the glossary, and is worth
keeping in mind while reading: an expression compendium cannot measure how often
a guide barcode is seen in a read, because there are no guides in it.

    python experiments/019-perturb-seq-costing/scripts/compression_analysis.py

Outputs to experiments/019-perturb-seq-costing/results/:
    compression_spectrum.csv      singular values and cumulative variance
    compression_sparsity.csv      genes carrying the loading mass, per component
    compression_additivity.csv    per-double observed vs additive prediction
    compression_summary.json      r, q and the sample-complexity numbers
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import lmdb
import pickle

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
RESULTS_DIR = osp.join(
    os.environ["EXPERIMENT_ROOT"], "019-perturb-seq-costing", "results"
)

# Variance fractions at which the rank is reported. No single one of these is
# "the" rank -- a spectrum that decays smoothly has no knee to find, and quoting
# one cut-off as if it did is the same error Sec. 4.4 corrects for the responder
# threshold. So the whole curve is emitted and three points are named.
VARIANCE_CUTS = (0.5, 0.8, 0.9, 0.95)

# Fractions of a component's squared loading mass at which its support is
# reported. Yao's q is "non-zeros per module", which presumes an exactly sparse
# factorization -- FR-Perturb gets one from sparse PCA. An SVD component has no
# exact zeros and is dense by construction, so any support measured from it is
# an UPPER bound on what a sparse factorization of the same matrix would need,
# and reporting a single mass fraction would disguise that as a measurement.
# The whole curve is emitted instead, and the row sparsity below is the
# assumption-free companion.
SUPPORT_MASSES = (0.5, 0.75, 0.9)

# The responder threshold of Sec. 4.4, reused here so the two analyses cannot
# drift apart. Row sparsity -- how many genes one perturbation moves -- is the
# most direct sparsity in this matrix and needs no basis at all, which is why it
# is the number carried into the sample-complexity estimate.
RESPONDER_FOLD = 1.25


class LmdbRecords:
    """Minimal reader over a built torchcell dataset LMDB.

    Same contract as effect_size_analysis.py: keys are the record index as ASCII
    bytes, values are pickled dicts of plain builtins. Duplicated rather than
    imported because that module runs a full pass over three datasets on import
    of its main(), and this one needs the matrices rather than the summaries.
    """

    def __init__(self, root: str) -> None:
        self.path = osp.join(root, "processed", "lmdb")
        self.env = lmdb.open(self.path, readonly=True, lock=False, subdir=True)
        with self.env.begin() as tx:
            self.n = tx.stat()["entries"]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        with self.env.begin() as tx:
            raw = tx.get(str(i).encode())
        if raw is None:
            raise KeyError(f"no record {i} in {self.path}")
        return pickle.loads(raw)


def require(path: str, label: str) -> str:
    if not osp.isdir(osp.join(path, "processed", "lmdb")):
        raise SystemExit(
            f"{label} LMDB not found at {path}. Build it first; this script has "
            "no fallback and will not fabricate a matrix."
        )
    return path


def load_profiles(dataset, name: str) -> tuple[list[tuple[str, ...]], list[dict]]:
    """Perturbed gene tuple and gene-keyed response dict, per strain.

    Keys are KEPT, unlike in effect_size_analysis.py, which drops them the moment
    it has the values. A matrix needs a common column space and the additivity
    test needs to look up a specific single by the gene it deleted, so identity
    has to survive the read.
    """
    perts, profiles = [], []
    for i in tqdm(range(len(dataset)), desc=name):
        d = dataset[i]
        genes = tuple(
            p["systematic_gene_name"]
            for p in d["experiment"]["genotype"]["perturbations"]
        )
        ratios = d["experiment"]["phenotype"]["expression_log2_ratio"]
        if not ratios:
            continue
        perts.append(genes)
        profiles.append(ratios)
    return perts, profiles


def to_matrix(profiles: list[dict], columns: list[str]) -> np.ndarray:
    """Strains x genes, on a fixed column order, with missing entries at zero.

    Zero-filling is the right imputation HERE and would not be elsewhere: the
    entries are log ratios against wild type, so zero is "no change", which is
    the correct prior for a gene the array did not measure in that strain. The
    alternative -- dropping any gene missing anywhere -- would discard columns
    for the convenience of the linear algebra.
    """
    idx = {g: j for j, g in enumerate(columns)}
    m = np.zeros((len(profiles), len(columns)), dtype=np.float32)
    for i, prof in enumerate(profiles):
        for g, v in prof.items():
            j = idx.get(g)
            if j is not None and v is not None and not np.isnan(v):
                m[i, j] = v
    return m


def spectrum(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Singular values and cumulative fraction of squared Frobenius norm.

    NOT centered. The matrix is already a set of differences against wild type,
    so the origin is meaningful and subtracting a column mean would remove the
    average response to being deleted at all -- which is real signal, not an
    offset. This is the same reason the responder threshold is a magnitude cut
    rather than a deviation from a per-gene mean.
    """
    sv = np.linalg.svd(mat, compute_uv=False)
    energy = sv ** 2
    return sv, np.cumsum(energy) / energy.sum()


def effective_rank(sv: np.ndarray) -> float:
    """Participation ratio of the squared spectrum.

    A cut-off-free companion to the variance-cut ranks: it is
    ``(sum s^2)^2 / sum s^4``, which equals k exactly when k singular values are
    equal and the rest are zero, and degrades smoothly in between. Reported
    because every variance cut is a choice and this one is not.
    """
    e = sv.astype(np.float64) ** 2
    return float(e.sum() ** 2 / (e ** 2).sum())


def support_size(vec: np.ndarray, mass: float) -> int:
    """How many entries carry ``mass`` of the squared norm of ``vec``."""
    e = np.sort(vec.astype(np.float64) ** 2)[::-1]
    c = np.cumsum(e) / e.sum()
    return int(np.searchsorted(c, mass) + 1)


def samples_needed(q: float, r: float, n: int) -> float:
    """Yao's stated scaling, O((q + r) log n), with the constant set to 1.

    The constant is unknown and is not invented: the paper states the ORDER, and
    a compressed-sensing bound's constant depends on the measurement ensemble and
    the recovery guarantee wanted. So every absolute number this produces is a
    shape, not a budget, and the figure says so. What survives the unknown
    constant is the RATIO to n, which is the whole claim.
    """
    return (q + r) * np.log(n)


def additivity(
    singles: dict[str, np.ndarray], doubles: list[tuple[tuple[str, ...], np.ndarray]]
) -> pd.DataFrame:
    """Observed double-deletion response against the sum of its own singles.

    The comparison is WITHIN Sameith -- same lab, platform and noise floor -- for
    the reason Sec. 4.4 already gives: pairing Kemmeren singles with Sameith
    doubles would confound the perturbation count with the study. That costs
    sample size, because only the doubles whose two singles were also profiled
    here can be tested, and it is worth it.
    """
    rows = []
    for genes, obs in doubles:
        if len(genes) != 2 or any(g not in singles for g in genes):
            continue
        pred = singles[genes[0]] + singles[genes[1]]
        # Restrict to genes that actually moved under the prediction. A
        # correlation over all 6,000 genes is dominated by the ~5,700 that sit
        # at zero in both vectors and would report near-perfect additivity for
        # any pair of sparse vectors, which is a statement about sparsity rather
        # than about additivity.
        active = (np.abs(pred) > np.log2(1.25)) | (np.abs(obs) > np.log2(1.25))
        if active.sum() < 10:
            continue
        p, o = pred[active], obs[active]
        denom = np.linalg.norm(p)
        rows.append({
            "genes": "|".join(genes),
            "n_active": int(active.sum()),
            "pearson_r": float(np.corrcoef(p, o)[0, 1]),
            # Relative residual: how much of the additive prediction's magnitude
            # is left unexplained. This is the quantity compressed recovery has
            # to absorb, and it is more interpretable than a correlation because
            # it is on the scale of the effect itself.
            "relative_residual": float(np.linalg.norm(o - p) / denom) if denom else np.nan,
            # Slope of observed on predicted through the origin. Below 1 means
            # the double is QUIETER than additive, which is the buffering
            # direction most yeast genetic interactions take.
            "slope": float(p @ o / (p @ p)) if denom else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    kem = require(
        osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"), "Kemmeren 2014"
    )
    sm = require(
        osp.join(DATA_ROOT, "data/torchcell/sm_microarray_sameith2015"),
        "Sameith 2015 singles",
    )
    dm = require(
        osp.join(DATA_ROOT, "data/torchcell/dm_microarray_sameith2015"),
        "Sameith 2015 doubles",
    )

    kem_perts, kem_prof = load_profiles(LmdbRecords(kem), "Kemmeren singles")
    sm_perts, sm_prof = load_profiles(LmdbRecords(sm), "Sameith singles")
    dm_perts, dm_prof = load_profiles(LmdbRecords(dm), "Sameith doubles")

    # One column space for everything, so a Sameith double and a Sameith single
    # are directly subtractable.
    columns = sorted({g for prof in kem_prof + sm_prof + dm_prof for g in prof})
    print(f"\ncommon gene space: {len(columns)} genes")

    E = to_matrix(kem_prof, columns)
    print(f"Kemmeren effect matrix: {E.shape[0]} strains x {E.shape[1]} genes")

    # --- r: rank of the effect matrix ---------------------------------------
    sv, cum = spectrum(E)
    ranks = {f"rank_{int(c*100)}pct": int(np.searchsorted(cum, c) + 1)
             for c in VARIANCE_CUTS}
    eff_rank = effective_rank(sv)
    pd.DataFrame({
        "component": np.arange(1, sv.size + 1),
        "singular_value": sv,
        "cumulative_variance": cum,
    }).to_csv(osp.join(RESULTS_DIR, "compression_spectrum.csv"), index=False)

    # --- q: non-zeros per module --------------------------------------------
    # Right singular vectors are the gene-space modules. Only the leading ones
    # matter: past the rank the components are noise and their support is the
    # whole genome by construction.
    _, _, vt = np.linalg.svd(E, full_matrices=False)
    top = max(ranks["rank_90pct"], 1)
    sup = {m: [support_size(vt[i], m) for i in range(top)] for m in SUPPORT_MASSES}
    # Row sparsity: genes moved by ONE perturbation, at the Sec. 4.4 threshold.
    # No basis, no factorization, no choice of mass fraction -- and it is the
    # quantity a reader can check against Table 8 directly.
    row_nnz = (np.abs(E) > np.log2(RESPONDER_FOLD)).sum(axis=1)
    pd.DataFrame({
        "component": np.arange(1, top + 1),
        **{f"genes_for_{int(m*100)}pct_mass": sup[m] for m in SUPPORT_MASSES},
    }).to_csv(osp.join(RESULTS_DIR, "compression_sparsity.csv"), index=False)

    # --- the additivity assumption, tested in yeast -------------------------
    sm_singles = {
        g[0]: to_matrix([p], columns)[0]
        for g, p in zip(sm_perts, sm_prof) if len(g) == 1
    }
    dm_pairs = [(g, to_matrix([p], columns)[0]) for g, p in zip(dm_perts, dm_prof)]
    add = additivity(sm_singles, dm_pairs)
    add.to_csv(osp.join(RESULTS_DIR, "compression_additivity.csv"), index=False)

    # q for the sample-complexity estimate is the ROW sparsity, not the SVD
    # support. The SVD support is dense (see SUPPORT_MASSES) and using it would
    # charge compressed sensing for a basis it does not use.
    q_med = float(np.median(row_nnz))
    r_used = ranks["rank_90pct"]
    summary = {
        "n_strains": int(E.shape[0]),
        "n_genes": int(E.shape[1]),
        **ranks,
        "effective_rank_participation_ratio": eff_rank,
        "responder_fold": RESPONDER_FOLD,
        "q_median_genes_moved_per_perturbation": q_med,
        "q_iqr": [float(np.percentile(row_nnz, 25)),
                  float(np.percentile(row_nnz, 75))],
        "svd_support_median_by_mass": {
            str(m): float(np.median(sup[m])) for m in SUPPORT_MASSES
        },
        "additivity_n_testable_doubles": int(len(add)),
        "additivity_median_pearson_r": (
            float(add.pearson_r.median()) if len(add) else None),
        "additivity_median_relative_residual": (
            float(add.relative_residual.median()) if len(add) else None),
        "additivity_median_slope": (
            float(add.slope.median()) if len(add) else None),
        "samples_needed_genome_scale": {
            str(n): samples_needed(q_med, r_used, n) for n in (200, 1000, 6000)
        },
    }
    with open(osp.join(RESULTS_DIR, "compression_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== r: how many components carry the response ===")
    for c in VARIANCE_CUTS:
        print(f"  {int(c*100):>3}% of variance: {ranks[f'rank_{int(c*100)}pct']:>4} components")
    print(f"  participation ratio     : {eff_rank:.1f}")
    print(f"\n=== q: genes moved by one perturbation at {RESPONDER_FOLD}x ===")
    print(f"  median {q_med:.0f}, IQR {summary['q_iqr'][0]:.0f}-{summary['q_iqr'][1]:.0f}"
          f"  of {E.shape[1]} genes")
    print(f"\n=== SVD module support, an UPPER bound (components are dense) ===")
    for m in SUPPORT_MASSES:
        print(f"  {int(m*100):>3}% of loading mass: median {np.median(sup[m]):>5.0f} genes")
    print("\n=== additivity of doubles, within Sameith ===")
    if len(add):
        print(f"  {len(add)} testable doubles")
        print(f"  median Pearson r vs additive prediction : {add.pearson_r.median():.3f}")
        print(f"  median relative residual               : {add.relative_residual.median():.3f}")
        print(f"  median slope (observed on predicted)   : {add.slope.median():.3f}")
    else:
        print("  none testable: no double has both of its singles in this compendium")
    print("\n=== sample complexity, (q+r) log n with unit constant ===")
    for n, m in summary["samples_needed_genome_scale"].items():
        print(f"  n = {int(n):>5} targets -> {m:>8.0f} composite samples "
              f"({m/int(n):.2f} x n)")
    print(f"\nwrote 4 files to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
