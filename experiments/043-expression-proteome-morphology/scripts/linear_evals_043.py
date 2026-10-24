# experiments/043-expression-proteome-morphology/scripts/linear_evals_043.py
# [[experiments.043-expression-proteome-morphology.scripts.linear_evals_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/linear_evals_043
"""The 043 linear-evaluation ladder over frozen inputs, on the fold file, CPU only.

Implements the primer's "Linear evaluations first" and "The evaluation protocol"
(notes-tex/multimodal/043-datasets-primer/sections/output-design.tex) on
``conf/splits/global_split_v1.json`` (``GlobalSplit043``; the run refuses a store whose
raw index hash differs from the file's). Store generation v2 (``STORE_043_VERSION=v2``,
``store_version_043``) reads ``raw_043_v2`` and ``global_split_v2.json`` and writes
``results/linear_evals_043_v2.json``, ``results/linear_evals_043_v2/`` and the cache
``linear_evals_cache_v2``.

Stages (``--stage``):

* ``extract``: reads the records each part needs by raw index through
  ``dataset_043.Raw043Dataset`` (process pool) and caches label matrices under
  ``$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/linear_evals_cache``;
  also the Costanzo 2016 single-mutant fitness from the ``smf_costanzo2016`` dev store.
* ``deletion`` (split A), ``doubles`` (split B), ``isolates`` (split C), ``segregants``
  (split D), ``vaishnav`` (split E), ``morphology`` (the Ohya 2005 probe): one JSON and
  CSVs per part under ``results/linear_evals_043/``.
* ``report``: merges the parts into ``results/linear_evals_043.json``, writes the summary
  CSVs and the figures.

Targets (primer table ``tab:output-scales``): Kemmeren 2014 log2 ratio as released; Hu 2007
X score as released; Nadal-Ribelles 2025 log2 FC with ``|x| > 20`` set missing (the
sentinel rule, confirmed by the owner 2026-10-21), scored per environment; Messner 2023
log2(strain / record reference); Zelezniak 2018 strain minus record reference; Caudal 2024
log2(TPM + 1); Muenzner 2024 protein and turnover log2 (non-positive values missing);
Boocock 2025 pooled and N'Guessan 2025 log2(CPM + 1), CPM from the record's
``total_umis``, released zeros (omitted keys) as 0 counts; Vaishnav 2022 the released
level. Records of one unit (genotype, environment) are averaged. A target column is kept
when finite in at least half the unit rows of the dataset (a coverage rule on the target,
fixed before any fold is read). Jackson 2020 is not scored: its 11 TF deletions put about
one gene in each gene fold.

Inputs: the frozen S288C tokens of the deleted gene, the four blocks
(FUDT upstream 768, FUDT downstream 768, ProtT5-XL 1,024, CaLM 768) each LayerNorm-
standardized per gene (no affine), alone and concatenated (the 3,328-wide composite).

Rungs: N0, the training-mean profile; ridge, multi-output and closed form (kernel form,
one eigendecomposition per fit), inputs and targets z-scored on the training rows,
missing training targets set to the training mean (z = 0), penalty
``lambda = (trace K / n) * 10^{-3..3}`` (13 values) chosen on the inner fold (the
validation fold ``(k + 1) mod n_folds`` of test fold k) by the median per-gene Pearson,
then refit on every non-test fold; kNN, the nanmean profile of the k in {5, 10, 20}
nearest training units by cosine on the token. Isolates: GBLUP (kernel ridge with the
kinship ``Z Z^T / p`` over the standardized per-gene variant, loss-of-function and
absence indicators, 3 x 6,607 columns; the records carry per-gene calls with an
off-graph allele pointer, not variant positions, so identity by state is over gene-level
calls) and the ridge on the 0/1 loss-of-function plus absence vector. Segregants: ridge on
the parent-of-origin vector (+1 non-reference parent, -1 reference parent, 0 unassigned or
posterior < 0.9, at each gene's midpoint, the ``dataset_043`` block rule); Boocock's
79,123 de novo cells are not read in this pass. Vaishnav: ridge on k-mer counts (k 1 to
6, 5,460 columns) and on the one-hot 80-mer (320 columns), each with a library indicator.

Statistics (primer ``tab:output-protocol``): median per-gene Pearson across held-out
units (primary); per-unit Pearson across genes, centered (the training-mean profile
subtracted from prediction and measurement) and raw; Spearman beside each. A constant
prediction (N0 per gene; N0 and the per-gene-mean rung after centering) carries no
ordering and is scored 0, so a paired difference against N0 on those statistics is the
model's statistic itself. Intervals: 95% paired bootstrap, 1,000 resamples of held-out
units, seed 0. Every fold is fitted; fold 0 is the reported one, the fold mean is beside
it.

    python experiments/043-expression-proteome-morphology/scripts/linear_evals_043.py --stage extract
    python experiments/043-expression-proteome-morphology/scripts/linear_evals_043.py --stage deletion --dataset kemmeren
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import pickle
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool
from typing import Any

import lmdb
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from dataset_043 import RAW_INDEX_NAME, Dataset043Config, Raw043Dataset  # noqa: E402
from global_split_043 import GlobalSplit043, Role, load_split  # noqa: E402
from make_global_split_043 import isolate_key  # noqa: E402
from store_version_043 import active as active_store  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"
EMBED_FILES = {
    "fudt_up": ("fudt_embedding/processed/species_upstream.pt", "species_upstream"),
    "fudt_down": (
        "fudt_embedding/processed/species_downstream.pt",
        "species_downstream",
    ),
    "prott5": (
        "protT5_embedding/processed/prot_t5_xl_uniref50_all.pt",
        "prot_t5_xl_uniref50_all",
    ),
    "calm": ("calm_embedding/processed/calm.pt", "calm"),
}
BLOCKS = list(EMBED_FILES)
EMBEDDINGS = ["composite", *BLOCKS]
KS = (5, 10, 20)
NADAL_SENTINEL = 20.0
LOG2_15 = float(np.log2(1.5))
SEGREGANT_POSTERIOR = 0.9
# The 019 / cross_dataset_correlation morphology construction (Table morph).
M_FOLDS, M_LAMBDAS, M_SEED, M_ZCLIP, M_MIN_COVER = (
    5,
    [1.0, 10.0, 100.0, 1e3, 1e4],
    0,
    8.0,
    0.5,
)
MOVING_PATH = (
    "/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/"
    "experiments/019-simb-multimodal/results/calmorph_moving_features.json"
)

DELETION = {
    "kemmeren": "MicroarrayKemmeren2014Dataset",
    "hu": "MicroarrayScoreHu2007Dataset",
    "nadal": "NadalRibellesPerturbSeq2025Dataset",
    "messner": "ProteomeMessner2023Dataset",
    "zelezniak": "ProteomeZelezniak2018Dataset",
}
LOW_AGREEMENT = {"hu", "nadal_ypd", "nadal_nacl", "jackson"}
OTHER = {
    "sameith_dm": "DmMicroarraySameith2015Dataset",
    "ohya": "ScmdOhya2005Dataset",
    "caudal": "CaudalPanTranscriptome2024Dataset",
    "muenzner": "ProteomeMuenzner2024Dataset",
    "turnover": "ProteinTurnoverMuenzner2024Dataset",
    "boocock": "Boocock2025Dataset",
    "nguessan": "Nguessan2025Dataset",
    "vaishnav": "PooledPromoterExpressionVaishnav2022Dataset",
}
ALL = {**DELETION, **OTHER}


class LinearEvalConfig(BaseModel):
    """Run configuration."""

    data_root: str
    experiment_dir: str
    split_path: str
    ridge_log10_grid: list[float] = Field(
        default_factory=lambda: [x / 2 for x in range(-6, 7)],
        description="lambda = (trace K / n) * 10^g",
    )
    knn_k: list[int] = Field(default_factory=lambda: list(KS))
    n_boot: int = 1000
    boot_seed: int = 0
    min_target_cover: float = 0.5
    workers: int = 16

    @property
    def cache(self) -> str:
        """Directory under DATA_ROOT holding the extracted label and token caches."""
        return osp.join(
            self.data_root,
            "data/torchcell/experiments",
            EXPERIMENT,
            active_store().linear_evals_cache,
        )

    @property
    def results(self) -> str:
        """The experiment's committed results directory."""
        return osp.join(self.experiment_dir, "results")

    @property
    def part_dir(self) -> str:
        """Per-dataset result files of the ladder, one directory under results."""
        return osp.join(self.results, "linear_evals_043" + active_store().suffix)


# ============================================================================ extract

_DS: Raw043Dataset | None = None


def _segregant_markers(ds: Raw043Dataset, g: dict[str, Any]) -> np.ndarray:
    """+1 non-reference parent, -1 reference parent, 0 unassigned / low posterior."""
    import bisect

    parents = (g["parent_1"], g["parent_2"])
    ref = [i for i, p in enumerate(parents, start=1) if p["peter_strain_id"] is None]
    reference_parent = ref[0] if ref else 1
    by_chrom: dict[str, tuple[list[int], list[dict[str, Any]]]] = {}
    for block in g["blocks"]:
        starts, blocks = by_chrom.setdefault(block["chromosome"], ([], []))
        starts.append(block["start"])
        blocks.append(block)
    out = np.zeros(ds.n_genes, dtype=np.int8)
    for gi, gene in enumerate(ds.genes):
        chrom, start, end = ds.coords[gene]
        if chrom not in by_chrom:
            continue
        mid = (start + end) // 2
        starts, blocks = by_chrom[chrom]
        k = bisect.bisect_right(starts, mid) - 1
        if (
            k < 0
            or mid > blocks[k]["end"]
            or blocks[k]["posterior"] < SEGREGANT_POSTERIOR
        ):
            continue
        out[gi] = -1 if blocks[k]["parent"] == reference_parent else 1
    return out


def _extract_one(item: tuple[str, int]) -> dict[str, Any] | None:
    """One record -> the compact row the ladder needs (pool worker)."""
    assert _DS is not None
    ds = _DS
    name, raw_index = item
    if name == ALL["boocock"]:
        with ds._txn() as txn:
            raw = txn.get(f"data_{raw_index}".encode())
        if b"SM (synthetic minimal" not in raw:  # de novo cells: not read in this pass
            return None
        record = json.loads(raw)
    else:
        record = ds.record(raw_index)
    ex = record["experiment"]
    if ex["dataset_name"] != name:
        raise ValueError(f"data_{raw_index} is {ex['dataset_name']}, index says {name}")
    ph = ex["phenotype"]
    label = ph["label_name"]
    g = ex["genotype"]
    env_key = hashlib.sha256(
        json.dumps(ex["environment"], sort_keys=True).encode()
    ).hexdigest()[:12]
    row: dict[str, Any] = {
        "dataset": name,
        "raw_index": raw_index,
        "env_key": env_key,
        "media": ex["environment"]["media"]["name"],
        "env_pert": json.dumps(ex["environment"].get("perturbations") or [])[:400],
    }
    if name == ALL["vaishnav"]:
        (p,) = g["perturbations"]
        row["mer"] = p["promoter_sequence"][
            p["variable_region_start"] - 1 : p["variable_region_end"]
        ]
        row["library"] = str(ph["library_id"])
        row["y"] = float("nan") if ph[label] is None else float(ph[label])
        return row
    if name == ALL["ohya"]:
        value = ph[label]
        row["features"] = sorted(value)
        row["y"] = np.array(
            [np.nan if value[k] is None else float(value[k]) for k in row["features"]],
            dtype=np.float32,
        )
    else:
        y, _ = ds._label_vector(name, ph[label])
        row["y"] = y.numpy().astype(np.float32)
    if name in (ALL["messner"], ALL["zelezniak"]):
        yr, _ = ds._label_vector(
            name, record["experiment_reference"]["phenotype_reference"][label]
        )
        row["y_ref"] = yr.numpy().astype(np.float32)
    if "perturbations" not in g:  # segregant
        row["key"] = f"{g['cross']}:{g['segregant_id']}"
        row["markers"] = _segregant_markers(ds, g)
        row["total_umis"] = float(ph["total_umis"])
        return row
    if name in (ALL["caudal"], ALL["muenzner"], ALL["turnover"]):
        row["key"] = isolate_key(record)
        gi, pt, _, _, _ = ds._mark_perturbations(g["perturbations"], is_isolate=True)
        row["marked"] = (np.array(gi, dtype=np.int32), np.array(pt, dtype=np.int8))
        return row
    genes = []
    for p in g["perturbations"]:
        idx = ds._gene_idx(p["systematic_gene_name"])
        genes.append(p["systematic_gene_name"] if idx is None else ds.genes[idx])
    row["genes"] = sorted(genes)
    return row


def extract_smf(data_root: str) -> pd.Series:
    """Costanzo 2016 single-mutant fitness per ORF: mean over the gene's single-deletion
    records (KanMX and NatMX, every temperature), the 019 morphology screen's source
    (``smf_costanzo2016``, ``morphology_input_screen.py``).
    """
    env = lmdb.open(
        osp.join(data_root, "data/torchcell/smf_costanzo2016/processed/lmdb"),
        readonly=True,
        lock=False,
    )
    vals: dict[str, list[float]] = defaultdict(list)
    with env.begin() as txn:
        for _, v in txn.cursor():
            e = pickle.loads(v)["experiment"]
            perts = e["genotype"]["perturbations"]
            if len(perts) != 1:
                continue
            pt = perts[0]["perturbation_type"]
            if not pt.endswith("deletion") or "heterozygous" in pt:
                continue
            f = e["phenotype"]["fitness"]
            if f is not None and np.isfinite(f):
                vals[perts[0]["systematic_gene_name"]].append(float(f))
    env.close()
    return pd.Series({g: float(np.mean(v)) for g, v in vals.items()}, name="smf")


def extract(cfg: LinearEvalConfig) -> dict[str, Any]:
    """Read every needed record once; pickle one list of rows per dataset."""
    global _DS
    ds = Raw043Dataset(
        Dataset043Config(data_root=cfg.data_root, datasets=list(ALL.values()))
    )
    _DS = ds
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    t0 = time.time()
    with Pool(cfg.workers) as pool:
        for n, row in enumerate(
            pool.imap(_extract_one, ds.items, chunksize=128), start=1
        ):
            if row is not None:
                rows[row["dataset"]].append(row)
            if n % 20000 == 0:
                print(
                    f"{n} / {len(ds.items)} records, {time.time() - t0:.0f} s",
                    flush=True,
                )
    os.makedirs(cfg.cache, exist_ok=True)
    counts = {}
    for name, rs in rows.items():
        with open(osp.join(cfg.cache, f"{name}.pkl"), "wb") as f:
            pickle.dump(rs, f, protocol=5)
        counts[name] = len(rs)
    smf = extract_smf(cfg.data_root)
    smf.to_pickle(osp.join(cfg.cache, "smf_costanzo2016.pkl"))
    with open(osp.join(cfg.cache, "genes.json"), "w") as f:
        json.dump(ds.genes, f)
    meta = {
        "records": counts,
        "smf_genes": len(smf),
        "seconds": time.time() - t0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(osp.join(cfg.cache, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    return meta


def load_rows(cfg: LinearEvalConfig, short: str) -> list[dict[str, Any]]:
    with open(osp.join(cfg.cache, f"{ALL[short]}.pkl"), "rb") as f:
        return pickle.load(f)


# ============================================================================ inputs


def load_tokens(cfg: LinearEvalConfig, genes: list[str]) -> dict[str, np.ndarray]:
    """Embedding name -> (6,607, d) array in gene-index order, each block LayerNormed."""
    out = {}
    for name, (path, key) in EMBED_FILES.items():
        data, _ = torch.load(
            osp.join(cfg.data_root, "data/scerevisiae", path),
            weights_only=False,
            map_location="cpu",
        )
        ids = list(data.id)
        if sorted(ids) != genes:
            raise ValueError(f"{name}: embedding ids differ from the store gene set")
        pos = {g: i for i, g in enumerate(ids)}
        E = data.embeddings[key].numpy().astype(np.float64)[[pos[g] for g in genes]]
        E = (E - E.mean(1, keepdims=True)) / E.std(1, keepdims=True)
        out[name] = E
    out["composite"] = np.concatenate([out[b] for b in BLOCKS], axis=1)
    return out


# ============================================================================ statistics


def _corr_cols(P: np.ndarray, Y: np.ndarray, min_n: int = 3) -> np.ndarray:
    """Pearson per column over finite pairs; constant prediction -> 0; too few or
    constant measurement -> NaN.
    """
    ok = np.isfinite(P) & np.isfinite(Y)
    n = ok.sum(0)
    p0, y0 = np.where(ok, P, 0.0), np.where(ok, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pm, ym = p0.sum(0) / n, y0.sum(0) / n
        pc, yc = np.where(ok, P - pm, 0.0), np.where(ok, Y - ym, 0.0)
        sp, sy = np.sqrt((pc**2).sum(0)), np.sqrt((yc**2).sum(0))
        r = (pc * yc).sum(0) / (sp * sy)
    r = np.where(sp <= 1e-12 * np.maximum(1.0, np.abs(pm)), 0.0, r)
    r = np.where((n < min_n) | (sy <= 1e-12), np.nan, r)
    return r


def _rank_cols(P: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ranks per column over the jointly finite entries."""
    ok = np.isfinite(P) & np.isfinite(Y)
    rp = pd.DataFrame(np.where(ok, P, np.nan)).rank(axis=0).to_numpy()
    ry = pd.DataFrame(np.where(ok, Y, np.nan)).rank(axis=0).to_numpy()
    return rp, ry


def unit_stats(P: np.ndarray, Y: np.ndarray, mu: np.ndarray) -> dict[str, np.ndarray]:
    """Per-gene (columns) and per-unit (rows) correlations of one held-out block."""
    out = {"gene_pearson": _corr_cols(P, Y)}
    out["gene_spearman"] = _corr_cols(*_rank_cols(P, Y))
    Pc, Yc = (P - mu).T, (Y - mu).T
    out["unit_centered_pearson"] = _corr_cols(Pc, Yc)
    out["unit_centered_spearman"] = _corr_cols(*_rank_cols(Pc, Yc))
    out["unit_raw_pearson"] = _corr_cols(P.T, Y.T)
    return out


def summarize(stats: dict[str, np.ndarray]) -> dict[str, float | int]:
    s: dict[str, float | int] = {}
    for k, v in stats.items():
        f = v[np.isfinite(v)]
        s[f"median_{k}"] = float(np.median(f)) if len(f) else float("nan")
        s[f"mean_{k}"] = float(np.mean(f)) if len(f) else float("nan")
        s[f"n_{k}"] = int(len(f))
    return s


def paired_boot(
    Pa: np.ndarray,
    Pb: np.ndarray | None,
    Y: np.ndarray,
    mu: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, list[float]]:
    """95% intervals over resampled held-out units for the medians of model a, and of
    a - b when b is given: per-gene Pearson (recomputed on the resampled units) and the
    centered per-unit Pearson.
    """
    rng = np.random.default_rng(seed)
    m = Y.shape[0]
    ua = _corr_cols((Pa - mu).T, (Y - mu).T)
    ub = None if Pb is None else _corr_cols((Pb - mu).T, (Y - mu).T)
    ga, gd, ca, cd = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        ra = np.nanmedian(_corr_cols(Pa[idx], Y[idx]))
        ga.append(ra)
        sa = np.nanmedian(ua[idx])
        ca.append(sa)
        if Pb is not None:
            assert ub is not None
            gd.append(ra - np.nanmedian(_corr_cols(Pb[idx], Y[idx])))
            cd.append(sa - np.nanmedian(ub[idx]))
    q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731
    out = {"ci_median_gene_pearson": q(ga), "ci_median_unit_centered_pearson": q(ca)}
    if Pb is not None:
        out["ci_diff_median_gene_pearson"] = q(gd)
        out["ci_diff_median_unit_centered_pearson"] = q(cd)
    return out


# ============================================================================ fitting


def _zfit(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.nanmean(A, 0)
    s = np.nanstd(A, 0)
    s = np.where(s > 0, s, 1.0)
    return m, s


def kernel_ridge_path(
    Ktr: np.ndarray, Yz: np.ndarray, Kte: np.ndarray, lams: list[float]
) -> list[np.ndarray]:
    """Predictions for each lambda, one eigendecomposition (Ktr centered beforehand)."""
    s, U = np.linalg.eigh(Ktr)
    UtY = U.T @ Yz
    KteU = Kte @ U
    return [KteU @ (UtY / (s + lam)[:, None]) for lam in lams]


def _center_kernel(Ktr: np.ndarray, Kte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center a linear kernel on the training rows (feature centering in kernel form)."""
    n = Ktr.shape[0]
    one = np.ones((n, n)) / n
    cm = Ktr.mean(0)
    Ktr_c = Ktr - one @ Ktr - Ktr @ one + one @ Ktr @ one
    Kte_c = Kte - Kte.mean(1, keepdims=True) - cm[None, :] + Ktr.mean()
    return Ktr_c, Kte_c


def ridge_fold(
    X: np.ndarray | None,
    K: np.ndarray | None,
    Y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    grid: list[float],
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Inner-fold penalty on (tr minus va -> va), refit on tr (which contains va), predict te.

    X is z-scored on the fitting rows (linear kernel); or K is a precomputed kernel.
    Targets z-scored per column on the fitting rows, missing set to 0 in z.
    """

    def fit_predict(
        fit: np.ndarray, out: np.ndarray, rel: list[float]
    ) -> list[np.ndarray]:
        ym, ys = _zfit(Y[fit])
        Yz = np.nan_to_num((Y[fit] - ym) / ys)
        if X is not None:
            xm, xs = _zfit(X[fit])
            Xf, Xo = (X[fit] - xm) / xs, (X[out] - xm) / xs
            Kf, Ko = Xf @ Xf.T, Xo @ Xf.T
        else:
            assert K is not None
            Kf, Ko = _center_kernel(K[np.ix_(fit, fit)], K[np.ix_(out, fit)])
        scale = np.trace(Kf) / len(fit)
        preds = kernel_ridge_path(Kf, Yz, Ko, [scale * 10**g for g in rel])
        return [p * ys + ym for p in preds]

    inner = np.setdiff1d(tr, va)
    preds = fit_predict(inner, va, grid)
    scores = {}
    for g, p in zip(grid, preds, strict=True):
        scores[str(g)] = float(np.nanmedian(_corr_cols(p, Y[va])))
    best = max(grid, key=lambda g: (np.nan_to_num(scores[str(g)], nan=-9), -g))
    return fit_predict(tr, te, [best])[0], best, scores


def knn_fold(
    E: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray, ks: list[int]
) -> dict[int, np.ndarray]:
    """Nanmean profile of the k nearest training units by cosine on E."""
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    sim = En[te] @ En[tr].T
    order = np.argsort(-sim, axis=1)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in ks:
            nb = tr[order[:, :k]]
            out[k] = np.nanmean(Y[nb], axis=1)
    return out


# ============================================================================ panel ladder


def panel_ladder(
    cfg: LinearEvalConfig,
    name: str,
    Y: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
    inputs: dict[str, tuple[str, np.ndarray]],
    knn_inputs: dict[str, np.ndarray],
    unit_ids: list[str],
    col_ids: list[str],
    eval_folds: list[int],
    boot_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    """Every rung on every fold of one panel; fold 0 with bootstrap intervals.

    ``inputs``: model name -> ("x", X) for a linear kernel on z-scored X or ("k", K) for a
    precomputed kernel. ``knn_inputs``: embedding name -> token matrix.
    """
    grid = cfg.ridge_log10_grid
    per_fold: dict[int, dict[str, Any]] = {}
    fold0_preds: dict[str, np.ndarray] = {}
    fold0_stats: dict[str, dict[str, np.ndarray]] = {}
    for k in eval_folds:
        v = (k + 1) % n_folds
        te = np.where(folds == k)[0]
        tr = np.where(folds != k)[0]
        va = np.where(folds == v)[0]
        if len(te) < 3:
            continue
        mu = np.nanmean(Y[tr], 0)
        preds: dict[str, np.ndarray] = {"N0": np.tile(mu, (len(te), 1))}
        chosen = {}
        for mname, (kind, A) in inputs.items():
            p, best, _ = ridge_fold(
                A if kind == "x" else None,
                A if kind == "k" else None,
                Y,
                tr,
                va,
                te,
                grid,
            )
            preds[mname] = p
            chosen[mname] = best
        for ename, E in knn_inputs.items():
            for kk, p in knn_fold(E, Y, tr, te, cfg.knn_k).items():
                preds[f"knn{kk}_{ename}"] = p
        fold_rows = {}
        for mname, p in preds.items():
            st = unit_stats(p, Y[te], mu)
            fold_rows[mname] = summarize(st)
            if k == 0:
                fold0_preds[mname] = p
                fold0_stats[mname] = st
        per_fold[k] = {
            "n_test": int(len(te)),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "log10_lambda_rel": chosen,
            "models": fold_rows,
        }
        print(
            f"{name} fold {k}: n_test {len(te)} | "
            + " ".join(
                f"{m} {r['median_gene_pearson']:.3f}/{r['median_unit_centered_pearson']:.3f}"
                for m, r in fold_rows.items()
                if not m.startswith("knn") or m.startswith("knn10")
            ),
            flush=True,
        )
    result: dict[str, Any] = {
        "name": name,
        "n_units": int(len(Y)),
        "n_columns": int(Y.shape[1]),
        "folds": per_fold,
    }
    models = list(per_fold[min(per_fold)]["models"]) if per_fold else []
    result["fold_mean"] = {
        m: {
            s: float(np.nanmean([per_fold[k]["models"][m][s] for k in per_fold]))
            for s in (
                "median_gene_pearson",
                "median_unit_centered_pearson",
                "median_unit_raw_pearson",
                "median_gene_spearman",
                "median_unit_centered_spearman",
            )
        }
        for m in models
    }
    result["n_folds_scored"] = len(per_fold)
    if 0 in per_fold:
        te = np.where(folds == 0)[0]
        mu = np.nanmean(Y[folds != 0], 0)
        boots = {}
        for m in models:
            if m == "N0" or (m.startswith("knn") and not m.startswith("knn10_")):
                continue
            boots[f"{m}_vs_N0"] = paired_boot(
                fold0_preds[m], None, Y[te], mu, cfg.n_boot, cfg.boot_seed
            )
        for a, b in boot_pairs:
            if a in fold0_preds and b in fold0_preds:
                boots[f"{a}_vs_{b}"] = paired_boot(
                    fold0_preds[a], fold0_preds[b], Y[te], mu, cfg.n_boot, cfg.boot_seed
                )
        result["fold0_bootstrap"] = boots
        os.makedirs(cfg.part_dir, exist_ok=True)
        rows = []
        for m, st in fold0_stats.items():
            for i, u in enumerate(np.array(unit_ids)[te]):
                rows.append(
                    {
                        "model": m,
                        "unit": u,
                        "unit_centered_pearson": st["unit_centered_pearson"][i],
                        "unit_raw_pearson": st["unit_raw_pearson"][i],
                        "unit_centered_spearman": st["unit_centered_spearman"][i],
                    }
                )
        pd.DataFrame(rows).to_csv(
            osp.join(cfg.part_dir, f"{name}_fold0_per_unit.csv"), index=False
        )
        grow = []
        for m, st in fold0_stats.items():
            grow.append(
                pd.DataFrame(
                    {
                        "model": m,
                        "column": col_ids,
                        "gene_pearson": st["gene_pearson"],
                        "gene_spearman": st["gene_spearman"],
                    }
                )
            )
        pd.concat(grow).to_csv(
            osp.join(cfg.part_dir, f"{name}_fold0_per_gene.csv"), index=False
        )
    return result


def summary_rows(result: dict[str, Any]) -> pd.DataFrame:
    """One row per model: fold-0 statistics, intervals, fold mean."""
    rows = []
    folds = {
        int(k): v for k, v in result["folds"].items()
    }  # str keys after a JSON round trip
    if 0 not in folds:
        return pd.DataFrame()
    f0 = folds[0]
    for m, r in f0["models"].items():
        b = result.get("fold0_bootstrap", {}).get(f"{m}_vs_N0", {})
        rows.append(
            {
                "panel": result["name"],
                "model": m,
                "n_test_units": f0["n_test"],
                "fold0_median_per_gene_pearson": r["median_gene_pearson"],
                "fold0_median_per_gene_pearson_ci95": b.get("ci_median_gene_pearson"),
                "fold0_median_per_gene_spearman": r["median_gene_spearman"],
                "fold0_median_centered_per_unit_pearson": r[
                    "median_unit_centered_pearson"
                ],
                "fold0_median_centered_per_unit_pearson_ci95": b.get(
                    "ci_median_unit_centered_pearson"
                ),
                "fold0_median_centered_per_unit_spearman": r[
                    "median_unit_centered_spearman"
                ],
                "fold0_median_raw_per_unit_pearson": r["median_unit_raw_pearson"],
                "foldmean_median_per_gene_pearson": result["fold_mean"][m][
                    "median_gene_pearson"
                ],
                "foldmean_median_centered_per_unit_pearson": result["fold_mean"][m][
                    "median_unit_centered_pearson"
                ],
            }
        )
    return pd.DataFrame(rows)


def _cover(Y: np.ndarray, frac: float) -> np.ndarray:
    return np.isfinite(Y).mean(0) >= frac


def _write(cfg: LinearEvalConfig, part: str, payload: dict[str, Any]) -> None:
    os.makedirs(cfg.part_dir, exist_ok=True)
    payload["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    with open(osp.join(cfg.part_dir, f"{part}.json"), "w") as f:
        json.dump(payload, f, indent=1, default=float)


# ============================================================================ split A


def deletion_matrices(
    cfg: LinearEvalConfig, short: str, genes: list[str]
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Panel name -> (unit x 6,607 target matrix, unit gene list), averaged per unit."""
    rows = load_rows(cfg, short)
    groups: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    envs = sorted({r["env_key"] for r in rows})
    for r in rows:
        if len(r["genes"]) != 1:
            raise ValueError(f"{short}: record with {len(r['genes'])} deleted genes")
        y = r["y"].astype(np.float64)
        if short == "messner":
            with np.errstate(invalid="ignore", divide="ignore"):
                y = np.where(
                    (y > 0) & (r["y_ref"] > 0), np.log2(y) - np.log2(r["y_ref"]), np.nan
                )
        elif short == "zelezniak":
            y = y - r["y_ref"]
        elif short == "nadal":
            y = np.where(np.abs(y) > NADAL_SENTINEL, np.nan, y)
        panel = f"nadal_env{envs.index(r['env_key'])}" if short == "nadal" else short
        groups[panel][r["genes"][0]].append(y)
    out = {}
    for panel, by_gene in groups.items():
        units = sorted(by_gene)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            M = np.stack([np.nanmean(np.stack(by_gene[u]), 0) for u in units])
        out[panel] = (M, units)
    return out


def nadal_env_names(cfg: LinearEvalConfig) -> dict[str, dict[str, str]]:
    """nadal_env<i> -> its environment key and environmental perturbations as stored."""
    rows = load_rows(cfg, "nadal")
    envs = sorted({r["env_key"] for r in rows})
    pert = {r["env_key"]: r["env_pert"] for r in rows}
    return {
        f"nadal_env{i}": {"env_key": e, "env_perturbations": pert[e]}
        for i, e in enumerate(envs)
    }


def run_deletion(
    cfg: LinearEvalConfig, split: GlobalSplit043, short: str
) -> dict[str, Any]:
    genes = json.load(open(osp.join(cfg.cache, "genes.json")))
    gidx = {g: i for i, g in enumerate(genes)}
    tokens = load_tokens(cfg, genes)
    out: dict[str, Any] = {"dataset": ALL[short], "panels": {}}
    for panel, (M, units) in deletion_matrices(cfg, short, genes).items():
        keep_u = [i for i, u in enumerate(units) if u in gidx]
        n_outside = len(units) - len(keep_u)
        M = M[keep_u]
        units = [units[i] for i in keep_u]
        cols = _cover(M, cfg.min_target_cover)
        Y = M[:, cols]
        col_ids = [genes[i] for i in np.where(cols)[0]]
        folds = np.array([split.gene_folds[u] for u in units])
        rows_tok = [gidx[u] for u in units]
        inputs = {f"ridge_{e}": ("x", tokens[e][rows_tok]) for e in EMBEDDINGS}
        knn = {e: tokens[e][rows_tok] for e in EMBEDDINGS}
        pairs = [("ridge_composite", f"ridge_{b}") for b in BLOCKS] + [
            ("ridge_composite", "knn10_composite")
        ]
        res = panel_ladder(
            cfg,
            panel,
            Y,
            folds,
            split.n_gene_folds,
            inputs,
            knn,
            units,
            col_ids,
            list(range(split.n_gene_folds)),
            pairs,
        )
        res["n_units_outside_index"] = n_outside
        res["n_columns_dropped_by_cover"] = int((~cols).sum())
        out["panels"][panel] = res
        summary_rows(res).to_csv(
            osp.join(cfg.part_dir, f"{panel}_summary.csv"), index=False
        )
    if short == "nadal":
        out["nadal_envs"] = nadal_env_names(cfg)
    _write(cfg, f"deletion_{short}", out)
    return out


# ============================================================================ split B


def run_doubles(cfg: LinearEvalConfig, split: GlobalSplit043) -> dict[str, Any]:
    """The 72 Sameith doubles against rungs that see only their components."""
    genes = json.load(open(osp.join(cfg.cache, "genes.json")))
    gidx = {g: i for i, g in enumerate(genes)}
    kem = load_rows(cfg, "kemmeren")
    dm = load_rows(cfg, "sameith_dm")
    K = {r["genes"][0]: r["y"].astype(np.float64) for r in kem}
    cols = np.isfinite(np.stack(list(K.values()))).all(0) & np.isfinite(
        np.stack([r["y"] for r in dm])
    ).all(0)
    roles = {g: split.deletion_role([g]) for g in K}
    # A Kemmeren gene with no index position (store v2 admits 2: YLL016W, YNCB0010W) has
    # no frozen token, so it cannot enter the token ridge; it is counted, as the split-A
    # panels count n_units_outside_index.
    outside = sorted(g for g in K if g not in gidx)
    train_genes = sorted(g for g, r in roles.items() if r == Role.TRAIN and g in gidx)
    val_genes = sorted(g for g, r in roles.items() if r == Role.VAL and g in gidx)
    mu = np.mean(np.stack([K[g] for g in train_genes]), 0)[cols]
    tokens = load_tokens(cfg, genes)
    # single-fit ridge f on the Kemmeren training genes, penalty on the validation fold
    fit_genes = train_genes + val_genes
    Yk = np.stack([K[g][cols] for g in fit_genes])
    tr = np.arange(len(fit_genes))
    va = np.arange(len(train_genes), len(fit_genes))
    pair_genes = []
    for r in dm:
        a, b = r["genes"]
        for g in (a, b):
            if g not in K:
                raise ValueError(f"double gene {g} has no Kemmeren single")
            if roles[g] != Role.TRAIN:
                raise ValueError(f"double gene {g} is {roles[g]}, not training")
        pair_genes += [a, b]
    uniq = sorted(set(pair_genes))
    f_of: dict[str, dict[str, np.ndarray]] = {}
    lam_of = {}
    for e in EMBEDDINGS:
        X = np.concatenate(
            [
                tokens[e][[gidx[g] for g in fit_genes]],
                tokens[e][[gidx[g] for g in uniq]],
            ]
        )
        Yall = np.concatenate([Yk, np.full((len(uniq), Yk.shape[1]), np.nan)])
        te = np.arange(len(fit_genes), len(fit_genes) + len(uniq))
        p, best, _ = ridge_fold(X, None, Yall, tr, va, te, cfg.ridge_log10_grid)
        f_of[e] = {g: p[i] for i, g in enumerate(uniq)}
        lam_of[e] = best
    Ymeas, rungs, strong_weak, ids = [], defaultdict(list), [], []
    for r in dm:
        a, b = r["genes"]
        ma, mb = K[a][cols], K[b][cols]
        s, w = (ma, mb) if np.linalg.norm(ma) >= np.linalg.norm(mb) else (mb, ma)
        strong_weak.append((s, w))
        Ymeas.append(r["y"].astype(np.float64)[cols])
        ids.append(f"{a}|{b}")
        rungs["per_gene_mean"].append(mu)
        rungs["stronger_single"].append(s)
        rungs["additive_Mx_plus_My"].append(ma + mb)
        for e in EMBEDDINGS:
            rungs[f"inductive_additive_ridge_{e}"].append(f_of[e][a] + f_of[e][b])
    Y = np.stack(Ymeas)
    # fitted a M_strong + b M_weak + c, leave one double out, pooled over genes
    fitted, coefs = [], []
    for d in range(len(dm)):
        others = [j for j in range(len(dm)) if j != d]
        A = np.concatenate(
            [
                np.stack([strong_weak[j][0], strong_weak[j][1], np.ones(len(mu))], 1)
                for j in others
            ]
        )
        t = np.concatenate([Y[j] for j in others])
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        coefs.append(coef)
        s, w = strong_weak[d]
        fitted.append(coef[0] * s + coef[1] * w + coef[2])
    rungs["fitted_a_Mx_b_My_c_loo"] = fitted
    P = {k: np.stack(v) for k, v in rungs.items()}
    additive = P["additive_Mx_plus_My"]
    restrict = np.abs(Y - additive) > LOG2_15
    res: dict[str, Any] = {
        "n_doubles": len(dm),
        "n_genes": int(cols.sum()),
        "n_kemmeren_train_genes_for_mean": len(train_genes),
        "kemmeren_genes_outside_index": outside,
        "ridge_log10_lambda_rel": lam_of,
        "fitted_coef_mean": [float(x) for x in np.mean(coefs, 0)],
        "fitted_coef_sd": [float(x) for x in np.std(coefs, 0)],
        "restricted_genes_per_double_median": float(np.median(restrict.sum(1))),
        "rungs": {},
    }
    rows = []
    for name, p in P.items():
        st_all = unit_stats(p, Y, mu)
        Pm, Ym = np.where(restrict, p, np.nan), np.where(restrict, Y, np.nan)
        st_r = {
            "unit_centered_pearson": _corr_cols((Pm - mu).T, (Ym - mu).T),
            "unit_centered_spearman": _corr_cols(*_rank_cols((Pm - mu).T, (Ym - mu).T)),
        }
        pc, yc = (p - mu).ravel(), (Y - mu).ravel()
        slope_all = float(np.polyfit(pc, yc, 1)[0]) if pc.std() > 0 else float("nan")
        top = np.abs(pc) >= np.quantile(np.abs(pc), 0.9)
        slope_top = (
            float(np.polyfit(pc[top], yc[top], 1)[0]) if pc.std() > 0 else float("nan")
        )
        res["rungs"][name] = {
            **{
                f"all_{k}": v
                for k, v in summarize(
                    {
                        k: st_all[k]
                        for k in (
                            "unit_centered_pearson",
                            "unit_centered_spearman",
                            "unit_raw_pearson",
                        )
                    }
                ).items()
            },
            **{f"restricted_{k}": v for k, v in summarize(st_r).items()},
            "calibration_slope_all_entries": slope_all,
            "calibration_slope_predicted_top_decile": slope_top,
        }
        for i, u in enumerate(ids):
            rows.append(
                {
                    "rung": name,
                    "double": u,
                    "centered_pearson_all_genes": st_all["unit_centered_pearson"][i],
                    "raw_pearson_all_genes": st_all["unit_raw_pearson"][i],
                    "centered_pearson_restricted_genes": st_r["unit_centered_pearson"][
                        i
                    ],
                    "n_restricted_genes": int(restrict[i].sum()),
                }
            )
    pd.DataFrame(rows).to_csv(
        osp.join(cfg.part_dir, "doubles_per_double.csv"), index=False
    )
    # paired bootstrap over doubles of the centered per-double Pearson
    per = pd.DataFrame(rows)
    rng = np.random.default_rng(cfg.boot_seed)
    idx = [rng.integers(0, len(dm), len(dm)) for _ in range(cfg.n_boot)]
    boots = {}
    for ref in ("fitted_a_Mx_b_My_c_loo", "additive_Mx_plus_My"):
        for name in P:
            if name == ref:
                continue
            for col in (
                "centered_pearson_all_genes",
                "centered_pearson_restricted_genes",
            ):
                a = per[per.rung == name][col].to_numpy()
                b = per[per.rung == ref][col].to_numpy()
                d = [np.nanmedian(a[i]) - np.nanmedian(b[i]) for i in idx]
                boots[f"{name}_vs_{ref}|{col}"] = {
                    "diff_of_medians": float(np.nanmedian(a) - np.nanmedian(b)),
                    "ci95": [
                        float(np.nanpercentile(d, 2.5)),
                        float(np.nanpercentile(d, 97.5)),
                    ],
                    "n_doubles_up": int(np.sum(a > b)),
                }
    res["bootstrap"] = boots
    pd.DataFrame(
        [
            {"rung": k, **{kk: vv for kk, vv in v.items()}}
            for k, v in res["rungs"].items()
        ]
    ).rename(
        columns={
            "all_median_unit_centered_pearson": "median_centered_per_double_pearson_all_genes",
            "restricted_median_unit_centered_pearson": "median_centered_per_double_pearson_restricted_genes",
            "all_median_unit_raw_pearson": "median_raw_per_double_pearson_all_genes",
            "all_median_unit_centered_spearman": "median_centered_per_double_spearman_all_genes",
        }
    ).to_csv(osp.join(cfg.part_dir, "doubles_summary.csv"), index=False)
    _write(cfg, "doubles", res)
    print(
        json.dumps(
            {
                k: (
                    v["all_median_unit_centered_pearson"],
                    v["restricted_median_unit_centered_pearson"],
                )
                for k, v in res["rungs"].items()
            },
            indent=1,
        )
    )
    return res


# ============================================================================ split C


def run_isolates(cfg: LinearEvalConfig, split: GlobalSplit043) -> dict[str, Any]:
    genes = json.load(open(osp.join(cfg.cache, "genes.json")))
    G = len(genes)
    out: dict[str, Any] = {
        "panels": {},
        "note": (
            "GBLUP kinship from per-gene call indicators (variant, loss of function, absence; "
            "the records carry no variant positions); indicator ridge on the 0/1 LoF + absence vector"
        ),
    }
    for short in ("caudal", "muenzner", "turnover"):
        rows = load_rows(cfg, short)
        units = [r["key"] for r in rows]
        if len(set(units)) != len(units):
            raise ValueError(f"{short}: duplicate isolate keys")
        Y = np.stack([r["y"] for r in rows]).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            Y = (
                np.log2(Y + 1.0)
                if short == "caudal"
                else np.where(Y > 0, np.log2(Y), np.nan)
            )
        Z = np.zeros((len(rows), 3 * G), dtype=np.float64)
        for i, r in enumerate(rows):
            gi, pt = r["marked"]
            for t, off in ((1, 0), (2, G), (3, 2 * G)):  # VARIANT, LOF, ABSENT
                Z[i, off + gi[pt == t]] = 1.0
        lofabs = np.clip(Z[:, G : 2 * G] + Z[:, 2 * G :], 0, 1)
        lofabs = lofabs[:, lofabs.std(0) > 0]
        Zs = Z[:, Z.std(0) > 0]
        Zs = (Zs - Zs.mean(0)) / Zs.std(0)
        Kin = Zs @ Zs.T / Zs.shape[1]
        cols = _cover(Y, cfg.min_target_cover)
        Yc = Y[:, cols]
        col_ids = [genes[i] for i in np.where(cols)[0]]
        folds = np.array([split.isolate_folds[u] for u in units])
        inputs = {
            "gblup_kinship": ("k", Kin),
            "ridge_lof_absence_indicator": ("x", lofabs),
        }
        res = panel_ladder(
            cfg,
            short,
            Yc,
            folds,
            split.n_isolate_folds,
            inputs,
            {},
            units,
            col_ids,
            list(range(split.n_isolate_folds)),
            [("gblup_kinship", "ridge_lof_absence_indicator")],
        )
        res["n_kinship_columns"] = int(Zs.shape[1])
        res["n_indicator_columns"] = int(lofabs.shape[1])
        out["panels"][short] = res
        summary_rows(res).to_csv(
            osp.join(cfg.part_dir, f"isolates_{short}_summary.csv"), index=False
        )
    _write(cfg, "isolates", out)
    return out


# ============================================================================ split D


def run_segregants(cfg: LinearEvalConfig, split: GlobalSplit043) -> dict[str, Any]:
    genes = json.load(open(osp.join(cfg.cache, "genes.json")))
    out: dict[str, Any] = {
        "panels": {},
        "not_run": "Boocock 2025 de novo cells (79,123 records) not read in this pass",
    }
    for short in ("boocock", "nguessan"):
        rows = load_rows(cfg, short)
        units = [r["key"] for r in rows]
        C = np.nan_to_num(np.stack([r["y"] for r in rows]).astype(np.float64))
        tot = np.array([r["total_umis"] for r in rows])
        Y = np.log2(C / tot[:, None] * 1e6 + 1.0)
        cols = (C > 0).mean(0) >= cfg.min_target_cover
        Yc = Y[:, cols]
        col_ids = [genes[i] for i in np.where(cols)[0]]
        Mk = np.stack([r["markers"] for r in rows]).astype(np.float64)
        Mk = Mk[:, Mk.std(0) > 0]
        folds = np.array([split.segregant_folds[f"{ALL[short]}:{u}"] for u in units])
        res = panel_ladder(
            cfg,
            short if short != "boocock" else "boocock_pooled",
            Yc,
            folds,
            split.n_segregant_folds,
            {"ridge_parent_of_origin": ("x", Mk)},
            {},
            units,
            col_ids,
            list(range(split.n_segregant_folds)),
            [],
        )
        res["n_marker_columns"] = int(Mk.shape[1])
        res["gene_rule"] = (
            "genes with a nonzero count in at least half the panel's records"
        )
        out["panels"][res["name"]] = res
        summary_rows(res).to_csv(
            osp.join(cfg.part_dir, f"segregants_{res['name']}_summary.csv"), index=False
        )
    _write(cfg, "segregants", out)
    return out


# ============================================================================ split E


def kmer_matrix(mers: list[str], kmax: int = 6) -> np.ndarray:
    idx = {c: i for i, c in enumerate("ACGT")}
    blocks = []
    for k in range(1, kmax + 1):
        M = np.zeros((len(mers), 4**k), dtype=np.float32)
        for r, s in enumerate(mers):
            codes = [idx[c] for c in s]
            for j in range(len(s) - k + 1):
                v = 0
                for c in codes[j : j + k]:
                    v = v * 4 + c
                M[r, v] += 1
        blocks.append(M)
    return np.concatenate(blocks, 1)


def onehot_matrix(mers: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate("ACGT")}
    M = np.zeros((len(mers), 80 * 4), dtype=np.float32)
    for r, s in enumerate(mers):
        for j, c in enumerate(s):
            M[r, 4 * j + idx[c]] = 1
    return M


def primal_ridge_fold(
    X: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    grid: list[float],
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Scalar-target ridge in the primal (n >> p), penalty on the inner fold."""

    def fit(
        fit_rows: np.ndarray, out_rows: np.ndarray, rel: list[float]
    ) -> list[np.ndarray]:
        xm, xs = _zfit(X[fit_rows])
        ym, ys = y[fit_rows].mean(), y[fit_rows].std()
        Xf = (X[fit_rows] - xm) / xs
        A = (Xf.T @ Xf).astype(np.float64)
        b = Xf.T.astype(np.float64) @ ((y[fit_rows] - ym) / ys)
        s, U = np.linalg.eigh(A)
        Utb = U.T @ b
        scale = np.trace(A) / len(fit_rows)
        Xo = ((X[out_rows] - xm) / xs).astype(np.float64)
        XoU = Xo @ U
        return [XoU @ (Utb / (s + scale * 10**g)) * ys + ym for g in rel]

    inner = np.setdiff1d(tr, va)
    scores = {
        str(g): float(np.corrcoef(p, y[va])[0, 1])
        for g, p in zip(grid, fit(inner, va, grid), strict=True)
    }
    best = max(grid, key=lambda g: (scores[str(g)], -g))
    return fit(tr, te, [best])[0], best, scores


def run_vaishnav(cfg: LinearEvalConfig, split: GlobalSplit043) -> dict[str, Any]:
    from scipy.stats import spearmanr

    rows_all = load_rows(cfg, "vaishnav")
    rows = [r for r in rows_all if np.isfinite(r["y"])]
    n_level_missing = len(rows_all) - len(rows)
    mers = [r["mer"] for r in rows]
    y = np.array([r["y"] for r in rows], dtype=np.float64)
    libs = sorted({r["library"] for r in rows})
    L = np.array(
        [[1.0 if r["library"] == lb else 0.0 for lb in libs[1:]] for r in rows],
        dtype=np.float32,
    )
    folds = np.array([split.promoter_folds[m] for m in mers])
    feats = {
        "kmer_1_6_ridge": np.concatenate([kmer_matrix(mers), L], 1),
        "onehot_80mer_ridge": np.concatenate([onehot_matrix(mers), L], 1),
    }
    out: dict[str, Any] = {
        "n_records": len(rows),
        "n_records_level_missing": n_level_missing,
        "n_distinct_80mers": len(set(mers)),
        "libraries": {lb: int(sum(r["library"] == lb for r in rows)) for lb in libs},
        "models": {},
    }
    for name, X in feats.items():
        X = X[:, X.std(0) > 0]
        per_fold = {}
        for k in range(split.n_promoter_folds):
            te, tr, va = (
                np.where(folds == k)[0],
                np.where(folds != k)[0],
                np.where(folds == (k + 1) % split.n_promoter_folds)[0],
            )
            p, best, _ = primal_ridge_fold(X, y, tr, va, te, cfg.ridge_log10_grid)
            per_fold[k] = {
                "n_test": int(len(te)),
                "pearson": float(np.corrcoef(p, y[te])[0, 1]),
                "spearman": float(spearmanr(p, y[te]).statistic),
                "log10_lambda_rel": best,
                "per_library_pearson": {
                    lb: float(
                        np.corrcoef(
                            p[np.array([rows[i]["library"] == lb for i in te])],
                            y[te][np.array([rows[i]["library"] == lb for i in te])],
                        )[0, 1]
                    )
                    for lb in libs
                },
            }
            if k == 0:
                rng = np.random.default_rng(cfg.boot_seed)
                bs = []
                for _ in range(cfg.n_boot):
                    i = rng.integers(0, len(te), len(te))
                    bs.append(np.corrcoef(p[i], y[te][i])[0, 1])
                per_fold[k]["pearson_ci95"] = [
                    float(np.percentile(bs, 2.5)),
                    float(np.percentile(bs, 97.5)),
                ]
            print(
                f"vaishnav {name} fold {k}: pearson {per_fold[k]['pearson']:.3f}",
                flush=True,
            )
        out["models"][name] = {
            "n_features": int(X.shape[1]),
            "folds": per_fold,
            "fold_mean_pearson": float(
                np.mean([v["pearson"] for v in per_fold.values()])
            ),
            "fold_mean_spearman": float(
                np.mean([v["spearman"] for v in per_fold.values()])
            ),
        }
    out["N0_note"] = (
        "N0 (the training mean) is constant across promoters; its Pearson is 0 by construction"
    )
    pd.DataFrame(
        [
            {
                "model": m,
                "fold0_pearson_across_promoters": v["folds"][0]["pearson"],
                "fold0_pearson_ci95": v["folds"][0]["pearson_ci95"],
                "fold0_spearman_across_promoters": v["folds"][0]["spearman"],
                "foldmean_pearson_across_promoters": v["fold_mean_pearson"],
                "foldmean_spearman_across_promoters": v["fold_mean_spearman"],
            }
            for m, v in out["models"].items()
        ]
    ).to_csv(osp.join(cfg.part_dir, "vaishnav_summary.csv"), index=False)
    _write(cfg, "vaishnav", out)
    return out


# ============================================================================ morphology probe


def _ridge_primal(
    Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, lam: float
) -> np.ndarray:
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    if Xc.shape[1] > Xc.shape[0]:
        Kx = Xc @ Xc.T
        return (Xte - mx) @ (
            Xc.T @ np.linalg.solve(Kx + lam * np.eye(Kx.shape[0]), Yc)
        ) + my
    return (Xte - mx) @ np.linalg.solve(
        Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc
    ) + my


def _colr(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Pc, Yc = P - P.mean(0), Y - Y.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Pc * Yc).sum(0) / np.sqrt((Pc**2).sum(0) * (Yc**2).sum(0))


def morph_oof(
    X: np.ndarray, Y: np.ndarray, f: np.ndarray | None, residualize: bool
) -> tuple[np.ndarray, np.ndarray]:
    """5-fold out-of-fold ridge (the Table morph construction). With ``residualize`` the
    fitness ``f`` is regressed out of every target on the training folds (applied to the
    held-out fold with the training coefficients) and appended to the inputs. Returns
    (predictions, targets) in the space scored (raw or residual).
    """
    rng = np.random.default_rng(M_SEED)
    n = X.shape[0]
    fl = np.array_split(rng.permutation(n), M_FOLDS)
    pred = np.full_like(Y, np.nan)
    targ = Y.copy() if not residualize else np.full_like(Y, np.nan)
    for k in range(M_FOLDS):
        te = fl[k]
        tr = np.concatenate([fl[j] for j in range(M_FOLDS) if j != k])
        Xk, Yk = X, Y
        if residualize:
            assert f is not None
            A = np.stack([np.ones(len(tr)), f[tr]], 1)
            B, *_ = np.linalg.lstsq(A, Y[tr], rcond=None)
            Yk = Y - np.stack([np.ones(n), f], 1) @ B
            fz = (f - f[tr].mean()) / f[tr].std()
            Xk = np.concatenate([X, fz[:, None]], 1)
            targ[te] = Yk[te]
        itel, itr = tr[: len(tr) // 5], tr[len(tr) // 5 :]
        best, bl = -np.inf, M_LAMBDAS[0]
        for lam in M_LAMBDAS:
            s = np.nanmean(
                _colr(_ridge_primal(Xk[itr], Yk[itr], Xk[itel], lam), Yk[itel])
            )
            if s > best:
                best, bl = s, lam
        pred[te] = _ridge_primal(Xk[tr], Yk[tr], Xk[te], bl)
    return pred, targ


def _mz(df: pd.DataFrame) -> pd.DataFrame:
    return ((df - df.mean()) / df.std(ddof=0)).clip(-M_ZCLIP, M_ZCLIP)


def run_morphology(cfg: LinearEvalConfig, split: GlobalSplit043) -> dict[str, Any]:
    genes = json.load(open(osp.join(cfg.cache, "genes.json")))
    gidx = {g: i for i, g in enumerate(genes)}
    with open(MOVING_PATH, "rb") as fh:
        raw = fh.read()
    moving = json.loads(raw)["moving"]
    rows = load_rows(cfg, "ohya")
    feats = rows[0]["features"]
    by_g: dict[str, list[np.ndarray]] = defaultdict(list)
    n_multi = 0
    for r in rows:
        if len(r["genes"]) != 1:
            n_multi += 1
            continue
        by_g[r["genes"][0]].append(r["y"].astype(np.float64))
    ohya = pd.DataFrame(
        {g: np.nanmean(np.stack(v), 0) for g, v in by_g.items()}, index=feats
    ).T
    target = ohya[[m for m in moving if m in ohya.columns]]
    target = target.loc[:, target.std(ddof=0) > 0]
    smf = pd.read_pickle(osp.join(cfg.cache, "smf_costanzo2016.pkl"))
    tokens = load_tokens(cfg, genes)
    # measured-input bars, by genotype
    measured: dict[str, pd.DataFrame] = {}
    for short in ("kemmeren", "messner", "hu"):
        M, units = deletion_matrices(cfg, short, genes)[short]
        measured[short] = pd.DataFrame(M, index=units)
    inputs: dict[str, pd.DataFrame] = {}
    in_index = [g for g in target.index if g in gidx]
    for e in EMBEDDINGS:
        inputs[f"token_{e}"] = pd.DataFrame(
            tokens[e][[gidx[g] for g in in_index]], index=in_index
        )
    for short, M in measured.items():
        inputs[f"measured_{short}"] = M
    fold0 = {g for g, f in split.gene_folds.items() if f == split.test_fold}
    res: dict[str, Any] = {
        "n_ohya_genotypes": int(len(target)),
        "n_ohya_multi_gene_records_skipped": n_multi,
        "n_moving_features": int(target.shape[1]),
        "n_smf_genes": int(len(smf)),
        "moving_features_source": {
            "path": MOVING_PATH,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "smf_source": "smf_costanzo2016 dev store, mean over the gene's single-deletion records",
        "inputs": {},
    }
    oof_store: dict[str, dict[str, tuple[np.ndarray, np.ndarray, list[str]]]] = (
        defaultdict(dict)
    )

    def score(
        name: str, X: pd.DataFrame, rows_: list[str], mode: str
    ) -> dict[str, Any]:
        Xs = X.reindex(rows_)
        Xs = Xs.loc[:, Xs.notna().mean() >= M_MIN_COVER]
        Xs = Xs.loc[:, Xs.std(ddof=0) > 0]
        Xa = _mz(Xs).fillna(0.0).to_numpy()
        Ya = _mz(target.loc[rows_]).fillna(0.0).to_numpy()
        f = smf.reindex(rows_).to_numpy() if mode != "raw_all" else None
        p, t = morph_oof(Xa, Ya, f, residualize=(mode == "residualized"))
        r = _colr(p, t)
        oof_store[name][mode] = (p, t, rows_)
        sub = np.array([g in fold0 for g in rows_])
        out = {
            "n_genotypes": len(rows_),
            "n_input_features": int(Xa.shape[1]),
            "median_r_over_116": float(np.nanmedian(r)),
            "mean_r_over_116": float(np.nanmean(r)),
        }
        if sub.sum() >= 10:
            out["fold0_gene_subgroup_n"] = int(sub.sum())
            out["fold0_gene_subgroup_median_r"] = float(
                np.nanmedian(_colr(p[sub], t[sub]))
            )
        return out

    with_smf = [g for g in target.index if g in smf.index]
    for name, X in inputs.items():
        shared = sorted(set(X.index) & set(target.index))
        shared_smf = sorted(set(shared) & set(with_smf))
        res["inputs"][name] = {
            "raw_all": score(name, X, shared, "raw_all"),
            "raw_smf_subset": score(name, X, shared_smf, "raw_smf_subset"),
            "residualized": score(name, X, shared_smf, "residualized"),
        }
        print(
            "morphology",
            name,
            json.dumps(
                {k: v["median_r_over_116"] for k, v in res["inputs"][name].items()}
            ),
            flush=True,
        )
    fit_only = pd.DataFrame({"smf": smf.reindex(with_smf)})
    res["inputs"]["fitness_only"] = {
        "raw_smf_subset": score("fitness_only", fit_only, with_smf, "raw_smf_subset")
    }
    # paired bootstrap over genotypes: token composite vs measured Kemmeren on their shared genotypes
    rng = np.random.default_rng(cfg.boot_seed)
    boots = {}
    for meas in ("measured_kemmeren", "measured_messner", "measured_hu"):
        for mode in ("raw_smf_subset", "residualized"):
            pa, ta, ra = oof_store["token_composite"][mode]
            pb, tb, rb = oof_store[meas][mode]
            common = sorted(set(ra) & set(rb))
            pa_pos = {g: i for i, g in enumerate(ra)}
            pb_pos = {g: i for i, g in enumerate(rb)}
            ia = [pa_pos[g] for g in common]
            ib = [pb_pos[g] for g in common]
            ds_ = []
            base = float(
                np.nanmedian(_colr(pa[ia], ta[ia]))
                - np.nanmedian(_colr(pb[ib], tb[ib]))
            )
            for _ in range(cfg.n_boot):
                j = rng.integers(0, len(common), len(common))
                ds_.append(
                    np.nanmedian(_colr(pa[ia][j], ta[ia][j]))
                    - np.nanmedian(_colr(pb[ib][j], tb[ib][j]))
                )
            boots[f"token_composite_vs_{meas}|{mode}"] = {
                "n_common_genotypes": len(common),
                "diff_median_r": base,
                "ci95": [
                    float(np.nanpercentile(ds_, 2.5)),
                    float(np.nanpercentile(ds_, 97.5)),
                ],
                "note": "each side's out-of-fold predictions from its own fit; scored on the common genotypes",
            }
    res["bootstrap"] = boots
    table = []
    for name, v in res["inputs"].items():
        table.append(
            {
                "input": name,
                "raw_all_n_genotypes": v.get("raw_all", {}).get("n_genotypes"),
                "raw_all_median_oof_pearson_over_116": v.get("raw_all", {}).get(
                    "median_r_over_116"
                ),
                "raw_smf_subset_n_genotypes": v["raw_smf_subset"]["n_genotypes"],
                "raw_smf_subset_median_oof_pearson_over_116": v["raw_smf_subset"][
                    "median_r_over_116"
                ],
                "residualized_median_oof_pearson_over_116": v.get(
                    "residualized", {}
                ).get("median_r_over_116"),
                "raw_all_fold0_gene_subgroup_median_oof_pearson": v.get(
                    "raw_all", {}
                ).get("fold0_gene_subgroup_median_r"),
                "raw_all_fold0_gene_subgroup_n": v.get("raw_all", {}).get(
                    "fold0_gene_subgroup_n"
                ),
            }
        )
    pd.DataFrame(table).to_csv(
        osp.join(cfg.part_dir, "morphology_summary.csv"), index=False
    )
    _write(cfg, "morphology", res)
    return res


# ============================================================================ report

PANEL_LABEL = {
    "kemmeren": "Kemmeren 2014",
    "messner": "Messner 2023",
    "zelezniak": "Zelezniak 2018",
    "hu": "Hu 2007",
    "nadal_env0": "Nadal-Ribelles YPD",
    "nadal_env1": "Nadal-Ribelles\nYPD + 0.4 M NaCl",
}
EMB_LABEL = {
    "composite": "composite",
    "fudt_up": "FUDT up",
    "fudt_down": "FUDT down",
    "prott5": "ProtT5",
    "calm": "CaLM",
}


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )


def _grid(ax: Any, lo: float, hi: float) -> None:
    from matplotlib.ticker import MultipleLocator

    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="#DDDDDD", zorder=0)
    ax.axhline(0, color="black", linewidth=0.5, zorder=1)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)


def _save(fig: Any, title: str) -> str:
    import matplotlib.pyplot as plt

    from torchcell.timestamp import timestamp
    from torchcell.utils.utils import savefig_true_size_svg

    d = osp.join(os.environ["ASSET_IMAGES_DIR"], EXPERIMENT)
    os.makedirs(d, exist_ok=True)
    base = osp.join(d, f"{title}{active_store().suffix}_{timestamp()}")
    fig.savefig(base + ".png", dpi=300)
    savefig_true_size_svg(fig, base + ".svg")
    plt.close(fig)
    return base + ".svg"


def _yerr(v: float, ci: list[float] | None) -> list[list[float]]:
    if not ci:
        return [[0.0], [0.0]]
    return [[max(0.0, v - ci[0])], [max(0.0, ci[1] - v)]]


def report(cfg: LinearEvalConfig, split: GlobalSplit043) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from torchcell.utils.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in

    parts = {}
    for fn in sorted(os.listdir(cfg.part_dir)):
        if fn.endswith(".json"):
            parts[fn[:-5]] = json.load(open(osp.join(cfg.part_dir, fn)))
    merged = {
        "generated_by": f"experiments/{EXPERIMENT}/scripts/linear_evals_043.py",
        "split_version": split.version,
        "split_store_sha256": split.store_sha256,
        "config": cfg.model_dump(),
        "headline_reference_excludes": sorted(LOW_AGREEMENT),
        "parts": parts,
        "report_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(
        osp.join(cfg.results, active_store().result_name("linear_evals_043")), "w"
    ) as f:
        json.dump(merged, f, indent=1, default=float)

    # ---- split A summary table
    frames = []
    for key, part in parts.items():
        if key.startswith("deletion_"):
            for panel, res in part["panels"].items():
                df = summary_rows(res)
                df["reference_set"] = (
                    "low_agreement_reported_separately"
                    if (panel in LOW_AGREEMENT or panel.startswith("nadal"))
                    else "headline"
                )
                frames.append(df)
    split_a = pd.concat(frames) if frames else pd.DataFrame()
    split_a.to_csv(
        osp.join(
            cfg.results,
            active_store().result_name("linear_evals_043_split_a_summary", "csv"),
        ),
        index=False,
    )
    _style()
    figs = {}
    panels = [p for p in PANEL_LABEL if not split_a.empty and p in set(split_a.panel)]
    if panels:
        fig, axes = plt.subplots(
            2,
            len(panels),
            figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(95)),
            squeeze=False,
        )
        groups = ["ridge", "knn5", "knn10", "knn20"]
        width = 0.16
        for j, panel in enumerate(panels):
            df = split_a[split_a.panel == panel].set_index("model")
            for i, (stat, lab) in enumerate(
                (
                    ("fold0_median_per_gene_pearson", "Median per-gene Pearson r"),
                    (
                        "fold0_median_centered_per_unit_pearson",
                        "Median centered\nper-strain Pearson r",
                    ),
                )
            ):
                ax = axes[i, j]
                for e_i, e in enumerate(EMBEDDINGS):
                    for g_i, g in enumerate(groups):
                        m = f"{g}_{e}"
                        if m not in df.index:
                            continue
                        v = float(df.loc[m, stat])
                        ci = df.loc[m, stat + "_ci95"]
                        ci = ci if isinstance(ci, list) else None
                        ax.bar(
                            g_i + (e_i - 2) * width,
                            v,
                            width,
                            color=PLOT_PALETTE[e_i],
                            edgecolor="black",
                            linewidth=0.3,
                            zorder=2,
                            label=EMB_LABEL[e]
                            if (g_i == 0 and i == 0 and j == 0)
                            else None,
                            yerr=_yerr(v, ci),
                            error_kw={
                                "elinewidth": 0.4,
                                "capsize": 0.8,
                                "ecolor": "black",
                            },
                        )
                ax.set_xticks(range(len(groups)))
                ax.set_xticklabels(
                    ["ridge", "kNN 5", "kNN 10", "kNN 20"], rotation=45, ha="right"
                )
                _grid(ax, -0.2, 0.4)
                if j == 0:
                    ax.set_ylabel(lab)
                if i == 0:
                    n = int(df.iloc[0]["n_test_units"])
                    ax.set_title(f"{PANEL_LABEL[panel]}\n(fold 0, n={n} genes)")
        fig.legend(loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        figs["split_a_ladder"] = _save(fig, "linear_evals_split_a_ladder")

    if "doubles" in parts:
        d = parts["doubles"]["rungs"]
        names = list(d)
        fig, ax = plt.subplots(
            figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(62))
        )
        x = np.arange(len(names))
        for k, (col, lab) in enumerate(
            (
                ("all_median_unit_centered_pearson", "all genes"),
                (
                    "restricted_median_unit_centered_pearson",
                    "genes with |M_xy - (M_x + M_y)| > log2 1.5",
                ),
            )
        ):
            ax.bar(
                x + (k - 0.5) * 0.38,
                [d[n][col] for n in names],
                0.38,
                color=PLOT_PALETTE[k],
                edgecolor="black",
                linewidth=0.3,
                zorder=2,
                label=lab,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                n.replace("inductive_additive_ridge_", "f(x_a)+f(x_b) ").replace(
                    "_", " "
                )
                for n in names
            ],
            rotation=60,
            ha="right",
        )
        ax.set_ylabel("Median centered per-double Pearson r\n(72 Sameith doubles)")
        _grid(ax, -0.2, 1.0)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        figs["doubles_ladder"] = _save(fig, "linear_evals_doubles_ladder")

    if "morphology" in parts:
        mo = parts["morphology"]["inputs"]
        names = list(mo)
        fig, ax = plt.subplots(
            figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(60))
        )
        x = np.arange(len(names))
        for k, (mode, lab) in enumerate(
            (("raw_smf_subset", "raw"), ("residualized", "growth-residualized"))
        ):
            vals = [mo[n].get(mode, {}).get("median_r_over_116", np.nan) for n in names]
            ax.bar(
                x + (k - 0.5) * 0.38,
                vals,
                0.38,
                color=PLOT_PALETTE[k],
                edgecolor="black",
                linewidth=0.3,
                zorder=2,
                label=lab,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"{n.replace('token_', 'token ').replace('measured_', 'measured ').replace('_', ' ')}\n(n={mo[n]['raw_smf_subset']['n_genotypes']})"
                for n in names
            ],
            rotation=60,
            ha="right",
        )
        ax.set_ylabel("Median out-of-fold Pearson r\nover 116 moving CalMorph features")
        _grid(ax, -0.2, 0.6)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        figs["morphology_probe"] = _save(fig, "linear_evals_morphology_probe")

    other = []
    for key, part in parts.items():
        if key in ("isolates", "segregants"):
            for panel, res in part["panels"].items():
                df = summary_rows(res)
                other.append(df)
    other_df = pd.concat(other) if other else pd.DataFrame()
    other_df.to_csv(
        osp.join(
            cfg.results,
            active_store().result_name(
                "linear_evals_043_isolates_segregants_summary", "csv"
            ),
        ),
        index=False,
    )
    if not other_df.empty:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(55)),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        ax = axes[0]
        rows = other_df[other_df.model != "N0"].reset_index(drop=True)
        cols = {m: PLOT_PALETTE[i] for i, m in enumerate(sorted(set(rows.model)))}
        for i, r in rows.iterrows():
            ci = r["fold0_median_per_gene_pearson_ci95"]
            v = r["fold0_median_per_gene_pearson"]
            ax.bar(
                i,
                v,
                0.8,
                color=cols[r.model],
                edgecolor="black",
                linewidth=0.3,
                zorder=2,
                yerr=_yerr(v, ci if isinstance(ci, list) else None),
                error_kw={"elinewidth": 0.4, "capsize": 0.8, "ecolor": "black"},
            )
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(
            [f"{r.panel}\n{r.model.replace('_', ' ')}" for r in rows.itertuples()],
            rotation=60,
            ha="right",
        )
        ax.set_ylabel("Median per-gene Pearson r\n(fold 0, held-out units)")
        _grid(ax, -0.2, 0.8)
        ax = axes[1]
        if "vaishnav" in parts:
            vm = parts["vaishnav"]["models"]
            for i, (m, v) in enumerate(vm.items()):
                p0 = (
                    v["folds"]["0"]["pearson"]
                    if "0" in v["folds"]
                    else v["folds"][0]["pearson"]
                )
                ci = (
                    v["folds"]["0"]["pearson_ci95"]
                    if "0" in v["folds"]
                    else v["folds"][0]["pearson_ci95"]
                )
                ax.bar(
                    i,
                    p0,
                    0.8,
                    color=PLOT_PALETTE[i],
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=2,
                    yerr=_yerr(p0, ci),
                    error_kw={"elinewidth": 0.4, "capsize": 0.8, "ecolor": "black"},
                )
            ax.set_xticks(range(len(vm)))
            ax.set_xticklabels(
                ["k-mer ridge", "one-hot ridge"], rotation=60, ha="right"
            )
            ax.set_ylabel("Pearson r across held-out\npromoters (Vaishnav, fold 0)")
        _grid(ax, -0.2, 1.0)
        fig.tight_layout()
        figs["isolates_segregants_promoters"] = _save(
            fig, "linear_evals_isolates_segregants_promoters"
        )
    merged["figures"] = figs
    with open(
        osp.join(cfg.results, active_store().result_name("linear_evals_043")), "w"
    ) as f:
        json.dump(merged, f, indent=1, default=float)
    print(json.dumps(figs, indent=1))
    return merged


# ============================================================================ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "extract",
            "deletion",
            "doubles",
            "isolates",
            "segregants",
            "vaishnav",
            "morphology",
            "report",
        ],
    )
    parser.add_argument("--dataset", choices=sorted(DELETION), default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    load_dotenv()
    exp_dir = osp.join(os.environ["EXPERIMENT_ROOT"], EXPERIMENT)
    cfg = LinearEvalConfig(
        data_root=os.environ["DATA_ROOT"],
        experiment_dir=exp_dir,
        split_path=osp.join(
            exp_dir, "conf", "splits", f"{active_store().split_version}.json"
        ),
        workers=args.workers,
        n_boot=args.n_boot,
    )
    split = load_split(cfg.split_path)
    split.check_store(
        osp.join(
            cfg.data_root,
            "data/torchcell/experiments",
            EXPERIMENT,
            active_store().raw_tag,
            "raw",
            RAW_INDEX_NAME,
        )
    )
    os.makedirs(cfg.part_dir, exist_ok=True)
    t0 = time.time()
    if args.stage == "extract":
        extract(cfg)
    elif args.stage == "deletion":
        assert args.dataset is not None
        run_deletion(cfg, split, args.dataset)
    elif args.stage == "doubles":
        run_doubles(cfg, split)
    elif args.stage == "isolates":
        run_isolates(cfg, split)
    elif args.stage == "segregants":
        run_segregants(cfg, split)
    elif args.stage == "vaishnav":
        run_vaishnav(cfg, split)
    elif args.stage == "morphology":
        run_morphology(cfg, split)
    elif args.stage == "report":
        report(cfg, split)
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
