# experiments/017-hoepfner-background-mutations/scripts/hoepfner_compute_risk_metrics.py
# [[experiments.017-hoepfner-background-mutations.scripts.hoepfner_compute_risk_metrics]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/017-hoepfner-background-mutations/scripts/hoepfner_compute_risk_metrics
"""Phase 1 (heavy extraction) for the Hoepfner 2014 background-mutation risk audit.

Question this experiment answers: Hoepfner et al. 2014 showed that a subset of the
HETEROZYGOUS (HIP) yeast deletion strains carry undocumented BACKGROUND MUTATIONS
(chr XI aneuploidy, a WHI2/YOR043w premature stop, a chr V 12 kb amplification) that were
co-inherited through construction-lab batches and produce promiscuous compound
hypersensitivity NOT caused by the annotated single-gene perturbation. torchcell stores
these strains as clean single-locus perturbations, so any record for an affected strain
carries a genotype that is WRONG at the sequence level. This script quantifies, from our
own sha256-pinned raw score matrices, what fraction of the built dataset is at risk.

The paper's detector logic (paper.md lines 170-194), reproduced here from our data:
  promiscuous hypersensitivity  +  chromosomal ADJACENCY of the deleted ORFs
  (adjacency == same construction lab == co-inherited secondary mutation)
Promiscuity alone is not enough (some genes are genuinely multidrug-sensitive, e.g. the
PDR efflux network); the discriminator is a RUN of neighbouring-ORF deletion strains all
sharing broad hypersensitivity, which has no biological cause other than a batch artifact.

This phase does the expensive one-time parse of the two ~600/500 MB pinned matrices and
caches lightweight per-strain metrics + a score sample. Phase 2
(`hoepfner_plot_risk.py`) runs the detectors + tiering + plots off these caches so
thresholds can be tuned without re-reading 1 GB.

Raw sources (sha256-pinned; identical files the LMDB loader consumes):
  $DATA_ROOT/data/torchcell/env_chemgen_hoepfner2014/raw/HIP_scores.txt  (HIP_scores sha256 dbc5041d...)
  $DATA_ROOT/data/torchcell/env_chemgen_hoepfner2014/raw/HOP_scores.txt  (HOP_scores sha256 99b386a8...)
SGD R64 coordinates (same FASTAs the loader uses to resolve the ORF universe):
  $DATA_ROOT/data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/{orf_coding_all,rna_coding}_*.fasta
"""

import json
import os
import os.path as osp
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "017-hoepfner-background-mutations", "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_DIR = osp.join(DATA_ROOT, "data/torchcell/env_chemgen_hoepfner2014/raw")
GENOME_DIR = osp.join(
    DATA_ROOT, "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830"
)
SGD_FASTAS = (
    osp.join(GENOME_DIR, "orf_coding_all_R64-4-1_20230830.fasta"),
    osp.join(GENOME_DIR, "rna_coding_R64-4-1_20230830.fasta"),
)

# Score-matrix header: '(Ad.|MADL) scores for Exp. <cmb>_<conc>_<HIP|HOP>_<study>' with an
# optional trailing ' z-score' (the companion gene-wise column we EXCLUDE — we score on the
# atomic per-experiment sensitivity column, exactly as the LMDB loader does).
_COL_RE = re.compile(
    r"^(?P<prefix>Ad\.|MADL) scores for Exp\. "
    r"(?P<cmb>\d+)_(?P<conc>[\d.]+)_(?P<assay>HIP|HOP)_(?P<study>\S+?)(?P<z> z-score)?$"
)

# FASTA header carries coordinates: '>YOR043W WHI2 SGDID:... Chr XV from 410870-412330, ...'
_COORD_RE = re.compile(r"Chr (\S+) from (\d+)-(\d+)")
_ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16,
}

# Sensitivity score is a robust experiment-wise z (negative = hypersensitive). Multiple
# thresholds are carried so Phase 2 can show threshold-robustness; -4 is the primary
# "strong hypersensitive" cut (roughly a 1-in-1e4 tail under a clean strain).
HYPER_THRESHOLDS = (-3.0, -4.0, -5.0)
RESIST_THRESHOLDS = (3.0, 4.0, 5.0)
WHI2_ORF = "YOR043W"
MIN_WHI2_OVERLAP = 100  # min shared non-NaN experiments to compute a whi2 profile corr
SAMPLE_N = 2_000_000    # per-assay raw-score sample size for the global histogram


def load_sgd_coords() -> dict[str, dict]:
    """ORF -> {gene, chrom, chrom_idx, start, end, pos} from the SGD R64 FASTAs.

    The systematic-name key set is the SAME universe the loader uses to keep/drop rows, so
    our strain set reproduces the LMDB's exactly.
    """
    coords: dict[str, dict] = {}
    for path in SGD_FASTAS:
        with open(path) as handle:
            for line in handle:
                if not line.startswith(">"):
                    continue
                head = line[1:].rstrip("\n")
                tok = head.split()
                orf = tok[0]
                gene = tok[1] if len(tok) > 1 and tok[1] != orf else orf
                m = _COORD_RE.search(head)
                if m is None:
                    coords[orf] = dict(
                        gene=gene, chrom=None, chrom_idx=np.nan,
                        start=np.nan, end=np.nan, pos=np.nan,
                    )
                    continue
                chrom, a, b = m.group(1), int(m.group(2)), int(m.group(3))
                coords[orf] = dict(
                    gene=gene, chrom=chrom, chrom_idx=_ROMAN.get(chrom, np.nan),
                    start=a, end=b, pos=(a + b) / 2.0,
                )
    return coords


def sensitivity_positions(header: list[str]) -> list[int]:
    """File-column positions of the sensitivity (non-z-score) columns, in file order."""
    positions = []
    for i, raw in enumerate(header):
        name = raw.strip().strip('"')
        m = _COL_RE.match(name)
        if m is not None and m.group("z") is None:
            positions.append(i)
    return positions


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r over already-aligned finite vectors."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else np.nan


def load_matrix(path: str, assay: str, sgd: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Load one assay's sensitivity matrix restricted to the SGD ORF universe.

    Returns (M [strains x experiments] float with NaN for empty, orfs [strains], n_exp_cols).
    """
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
    sens_pos = sensitivity_positions(header)
    print(f"[{assay}] {len(sens_pos)} sensitivity experiments (columns)")
    usecols = [0] + sens_pos
    df = pd.read_csv(
        path, sep="\t", header=0, usecols=usecols, quotechar='"',
        na_values=[""], engine="c",
    )
    df.columns = ["orf"] + [f"e{i}" for i in range(len(sens_pos))]
    df["orf"] = df["orf"].astype(str).str.strip().str.strip('"')
    df = df.set_index("orf")
    keep = df.index.isin(sgd)                       # match the loader's R64 drop
    df = df[keep]
    orfs = df.index.to_numpy()
    M = df.to_numpy(dtype=float)
    return M, orfs, len(sens_pos)


def strain_metrics(M: np.ndarray, orfs: np.ndarray, sgd: dict, assay: str) -> pd.DataFrame:
    """Per-strain risk metrics from the assay matrix."""
    n_exp = np.sum(~np.isnan(M), axis=1)
    with np.errstate(invalid="ignore", all="ignore"):
        mean_sens = np.nanmean(M, axis=1)
        median_sens = np.nanmedian(M, axis=1)
        std_sens = np.nanstd(M, axis=1)
        min_sens = np.nanmin(np.where(np.isnan(M), np.inf, M), axis=1)
    rows = dict(
        orf=orfs,
        gene=[sgd.get(o, {}).get("gene", o) for o in orfs],
        chrom=[sgd.get(o, {}).get("chrom") for o in orfs],
        chrom_idx=[sgd.get(o, {}).get("chrom_idx", np.nan) for o in orfs],
        pos=[sgd.get(o, {}).get("pos", np.nan) for o in orfs],
        n_exp=n_exp,
        mean_sens=mean_sens,
        median_sens=median_sens,
        std_sens=std_sens,
        min_sens=np.where(np.isfinite(min_sens), min_sens, np.nan),
    )
    denom = np.where(n_exp > 0, n_exp, np.nan)
    for t in HYPER_THRESHOLDS:
        rows[f"frac_hyper_{abs(int(t))}"] = np.sum(M <= t, axis=1) / denom
    for t in RESIST_THRESHOLDS:
        rows[f"frac_resist_{int(t)}"] = np.sum(M >= t, axis=1) / denom
    df = pd.DataFrame(rows)
    df["assay"] = assay
    return df


def add_whi2_corr(df: pd.DataFrame, M: np.ndarray, orfs: np.ndarray) -> pd.DataFrame:
    """Correlate every strain's profile with the whi2/YOR043w strain (cluster-3 marker)."""
    idx = np.where(orfs == WHI2_ORF)[0]
    if len(idx) == 0:
        df["whi2_corr"] = np.nan
        df["whi2_overlap"] = 0
        return df
    W = M[idx[0]]
    wmask = ~np.isnan(W)
    corr = np.full(len(orfs), np.nan)
    overlap = np.zeros(len(orfs), dtype=int)
    for s in range(len(orfs)):
        v = M[s]
        m = wmask & ~np.isnan(v)
        k = int(m.sum())
        overlap[s] = k
        if k >= MIN_WHI2_OVERLAP:
            corr[s] = pearson(v[m], W[m])
    df = df.copy()
    df["whi2_corr"] = corr
    df["whi2_overlap"] = overlap
    return df


def chrom_offsets(sgd: dict) -> dict[int, float]:
    """Cumulative bp offset per chromosome (extent = max gene end on that chrom)."""
    extent: dict[int, float] = {}
    for rec in sgd.values():
        ci = rec.get("chrom_idx")
        if isinstance(ci, float) and np.isnan(ci):
            continue
        extent[int(ci)] = max(extent.get(int(ci), 0.0), float(rec.get("end") or 0.0))
    offsets, run = {}, 0.0
    for ci in range(1, 17):
        offsets[ci] = run
        run += extent.get(ci, 0.0)
    return offsets, extent


def main() -> None:
    sgd = load_sgd_coords()
    offsets, extent = chrom_offsets(sgd)
    print(f"SGD universe: {len(sgd)} systematic names")

    summary: dict = {"thresholds": {"hyper": HYPER_THRESHOLDS, "resist": RESIST_THRESHOLDS}}
    frames = []
    samples = {}
    total_records = 0

    for assay, fname in (("HIP", "HIP_scores.txt"), ("HOP", "HOP_scores.txt")):
        path = osp.join(RAW_DIR, fname)
        M, orfs, n_cols = load_matrix(path, assay, sgd)
        df = strain_metrics(M, orfs, sgd, assay)
        df = add_whi2_corr(df, M, orfs)
        df["cum_pos"] = df.apply(
            lambda r: (offsets.get(int(r["chrom_idx"]), np.nan) + r["pos"])
            if not pd.isna(r["chrom_idx"]) else np.nan,
            axis=1,
        )
        # drop strains with zero measured experiments (mito/blank rows -> 0 LMDB records)
        df = df[df["n_exp"] > 0].reset_index(drop=True)

        flat = M[~np.isnan(M)]
        rng = np.random.default_rng(0)
        samples[assay] = (
            flat if flat.size <= SAMPLE_N
            else rng.choice(flat, size=SAMPLE_N, replace=False)
        )

        records = int(df["n_exp"].sum())
        total_records += records
        base = {
            f"neg_rate_{abs(int(t))}": float((flat <= t).mean()) for t in HYPER_THRESHOLDS
        }
        summary[assay] = dict(
            n_experiments=int(n_cols),
            n_strains=int(len(df)),
            n_records=records,
            n_cells_total=int(flat.size),
            score_mean=float(flat.mean()),
            score_std=float(flat.std()),
            score_p01=float(np.percentile(flat, 1)),
            score_p50=float(np.percentile(flat, 50)),
            score_p99=float(np.percentile(flat, 99)),
            baseline_neg_rate=base,
        )
        out = osp.join(RESULTS_DIR, f"{assay.lower()}_strain_metrics.csv")
        df.to_csv(out, index=False)
        print(f"[{assay}] strains={len(df)} records={records:,} -> {out}")
        frames.append(df)
        del M

    summary["total_records"] = total_records
    summary["lmdb_expected_records"] = 29_996_238
    summary["reconciliation_match"] = bool(total_records == 29_996_238)
    summary["chrom_offsets"] = {str(k): v for k, v in offsets.items()}
    summary["chrom_extent"] = {str(k): v for k, v in extent.items()}
    summary["paper_characterized_hip_strains"] = 157

    with open(osp.join(RESULTS_DIR, "summary_stats.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(
        osp.join(RESULTS_DIR, "score_samples.npz"),
        HIP=samples["HIP"].astype(np.float32),
        HOP=samples["HOP"].astype(np.float32),
    )
    pd.concat(frames, ignore_index=True).to_csv(
        osp.join(RESULTS_DIR, "all_strain_metrics.csv"), index=False
    )

    print("\n=== reconciliation ===")
    print(f"parsed total records = {total_records:,}")
    print(f"LMDB expected        = 29,996,238")
    print(f"match                = {summary['reconciliation_match']}")
    print(f"\nwrote summary_stats.json, {{hip,hop,all}}_strain_metrics.csv, score_samples.npz")


if __name__ == "__main__":
    main()
