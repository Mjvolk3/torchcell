# experiments/043-expression-proteome-morphology/scripts/cross_dataset_correlation.py
# [[experiments.043-expression-proteome-morphology.scripts.cross_dataset_correlation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/cross_dataset_correlation
"""Correlations between datasets of the same readout type in the 043 raw store.

Reads the RAW LMDB of the 043 store (one JSON record per served experiment; the processed
samples mix datasets under shared labels) and measures how well datasets that read the
same kind of quantity agree: expression against expression, protein against protein,
mRNA against protein, and gene-level matrices against CalMorph morphology.

Extraction (``--stage extract``) keeps, per record of the 13 gene- or isolate-keyed
datasets (Vaishnav is skipped; Boocock and N'Guessan contribute only their genotype
identifiers), the genotype key, the environment key and the label values with gene ids
made comparable:

* genotype key: the sorted perturbed systematic gene names joined by ``|``; an isolate by
  its strain name (``natural_isolate.isolate``); a segregant by ``cross:segregant_id``.
* environment key: sha256 of the environment JSON (12 hex digits).
* Messner 2023 UniProt accessions -> systematic ORF with
  ``torchcell.datasets.scerevisiae.messner2023.build_uniprot_to_orf_map`` (SGD GFF);
  Nadal-Ribelles 2025 standard names -> systematic ORF by inverting
  ``build_orf_to_gene_name_map``. A released id with no mapping is dropped from the
  comparison (counted in the output), never guessed.

The record-level matrices are cached as pickled DataFrames under
``$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/cross_dataset_cache``.

Quantities compared (``QUANTITY`` below, the table in the note): Kemmeren and Sameith log2
ratio to wild type; Hu 2007 error-model X score; Nadal-Ribelles pseudobulk log2 fold change;
Jackson 2020 UMI counts, turned here into log2((CPM + 1) / (CPM_HO + 1)) against the HO
deletion in the same environment; Caudal TPM as log2(TPM + 1) centered per gene over the
isolates; Messner and Muenzner MaxLFQ abundance (linear) as log2, Messner relative to its
record's reference (log2 ratio), Muenzner centered per protein over the isolates;
Zelezniak (already a log signal) minus its reference; Muenzner turnover as log2 of the
degradation rate, centered per protein.

Statistics (``--stage analyze``), per pair:

* (A) per shared genotype, Pearson and Spearman across the shared genes (at least
  ``MIN_GENES`` finite pairs): median, IQR, n genotypes;
* (B) per shared gene, Pearson across the shared genotypes (at least ``MIN_GENOTYPES``):
  median, IQR, n genes, fraction above 0.3.

The morphology link is a ridge from each gene-level matrix to the CalMorph vector on the
genotypes shared with Ohya 2005: the design of
``experiments/019-simb-multimodal/scripts/morphology_input_screen.py`` (5-fold out-of-fold
ridge, penalty from {1, 10, 100, 1e3, 1e4} on an inner split, inputs and targets z-scored
and clipped at 8, input features kept when measured in half the genotypes, one row
permutation null, seed 0), on all CalMorph features (the 116 moving-feature list is not on
main), plus the 116-feature list when ``--moving-features`` names that file.

Writes ``results/cross_dataset_correlation.json``,
``results/cross_dataset_correlation/<pair>_per_genotype.csv`` and ``..._per_gene.csv``, and
three figures to ``$ASSET_IMAGES_DIR/043-expression-proteome-morphology/``.

    python experiments/043-expression-proteome-morphology/scripts/cross_dataset_correlation.py --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import time
from collections import defaultdict
from itertools import combinations
from typing import Any

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from torchcell.datasets.scerevisiae.messner2023 import (  # noqa: E402
    build_orf_to_gene_name_map,
    build_uniprot_to_orf_map,
)
from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils.paths import experiment_results_dir  # noqa: E402
from torchcell.utils.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

EXPERIMENT = "043-expression-proteome-morphology"
MIN_GENES = 20
MIN_GENOTYPES = 5
FOLDS, LAMBDAS, SEED, ZCLIP, MIN_COVER = 5, [1.0, 10.0, 100.0, 1e3, 1e4], 0, 8.0, 0.5
HO = "YDL227C"

KEEP = {
    "MicroarrayKemmeren2014Dataset": "kemmeren",
    "SmMicroarraySameith2015Dataset": "sameith_sm",
    "MicroarrayScoreHu2007Dataset": "hu",
    "TfDeletionPseudobulkJackson2020Dataset": "jackson",
    "NadalRibellesPerturbSeq2025Dataset": "nadal",
    "ProteomeMessner2023Dataset": "messner",
    "ProteomeZelezniak2018Dataset": "zelezniak",
    "ScmdOhya2005Dataset": "ohya",
    "CaudalPanTranscriptome2024Dataset": "caudal",
    "ProteomeMuenzner2024Dataset": "muenzner",
    "ProteinTurnoverMuenzner2024Dataset": "turnover",
}
SEGREGANT = {"Boocock2025Dataset": "boocock", "Nguessan2025Dataset": "nguessan"}

QUANTITY: dict[str, dict[str, str]] = {
    "kemmeren": {
        "label": "expression_log2_ratio",
        "released": "two-channel microarray log2 ratio, deletion over wild type",
        "scale": "log2",
        "used": "as released",
    },
    "sameith_sm": {
        "label": "expression_log2_ratio",
        "released": "two-channel microarray log2 ratio, deletion over wild type",
        "scale": "log2",
        "used": "as released",
    },
    "hu": {
        "label": "expression_score",
        "released": "error-model X score (weighted duplicate average), deletion vs BY4741",
        "scale": "signed score, log-ratio based",
        "used": "as released",
    },
    "nadal": {
        "label": "expression_log2_ratio",
        "released": "pseudobulk scRNA-seq log2 fold change vs wild type",
        "scale": "log2",
        "used": "as released; standard names mapped to ORFs",
    },
    "jackson": {
        "label": "umi_count",
        "released": "pseudobulk UMI counts per gene (zeros omitted)",
        "scale": "linear counts",
        "used": "log2((CPM+1)/(CPM_HO+1)) vs the HO deletion in the same environment",
    },
    "messner": {
        "label": "protein_abundance",
        "released": "SWATH-MS MaxLFQ batch-corrected quantity",
        "scale": "linear",
        "used": "log2(strain / record reference); UniProt mapped to ORFs",
    },
    "zelezniak": {
        "label": "protein_abundance",
        "released": "SWATH-MS label-free log signal (SVA)",
        "scale": "log",
        "used": "strain minus record reference (difference of logs)",
    },
    "muenzner": {
        "label": "protein_abundance",
        "released": "SWATH-MS MaxLFQ precursor batch-corrected quantity",
        "scale": "linear",
        "used": "log2, centered per protein over the isolates",
    },
    "turnover": {
        "label": "degradation_rate",
        "released": "dynamic SILAC degradation rate, per hour",
        "scale": "linear",
        "used": "log2, centered per protein over the isolates",
    },
    "caudal": {
        "label": "expression_tpm",
        "released": "RNA-seq TPM",
        "scale": "linear",
        "used": "log2(TPM+1), centered per gene over the isolates",
    },
    "ohya": {
        "label": "calmorph",
        "released": "CalMorph feature values",
        "scale": "mixed per feature",
        "used": "z-scored per feature over the shared genotypes, clipped at 8",
    },
}


def _env_key(environment: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()[
        :12
    ]


def _genotype_key(genotype: dict[str, Any]) -> str:
    if "perturbations" not in genotype:
        return f"{genotype['cross']}:{genotype['segregant_id']}"
    isolate = genotype.get("natural_isolate")
    if isolate is not None:
        return f"isolate:{isolate['isolate']}"
    return "|".join(
        sorted(p["systematic_gene_name"] for p in genotype["perturbations"])
    )


# ---------------------------------------------------------------------------- extract


def extract(raw_path: str, cache: str, data_root: str, stride: int) -> dict[str, Any]:
    """One pass over the raw LMDB; record-level matrices to the cache.

    ``stride`` > 1 keeps every stride-th record (a smoke test of the pipeline only).
    """
    up2orf = build_uniprot_to_orf_map(data_root)
    name2orf: dict[str, str] = {}
    for orf, name in build_orf_to_gene_name_map(data_root).items():
        name2orf.setdefault(name.upper(), orf)
        name2orf.setdefault(orf.upper(), orf)
    rows: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    vals: dict[str, list[dict[str, float]]] = defaultdict(list)
    extra: dict[str, list[float]] = defaultdict(list)
    envs: dict[str, dict[str, Any]] = {}
    unmapped: dict[str, set[str]] = defaultdict(set)
    segregants: dict[str, dict[str, Any]] = {
        n: {"ids": set(), "crosses": set(), "parents": set()}
        for n in SEGREGANT.values()
    }
    n_seen = 0
    t0 = time.time()
    env = lmdb.open(raw_path, readonly=True, lock=False, readahead=True)
    with env.begin() as txn:
        source = (
            (raw for _, raw in txn.cursor())
            if stride == 1
            else (
                txn.get(f"data_{i}".encode())
                for i in range(0, txn.stat()["entries"], stride)
            )
        )
        for raw in source:
            n_seen += 1
            head = raw[:200]
            if b"PooledPromoterExpressionVaishnav2022Dataset" in head:
                continue
            record = json.loads(raw)
            ex = record["experiment"]
            dname = ex["dataset_name"]
            if dname in SEGREGANT:
                g = ex["genotype"]
                s = segregants[SEGREGANT[dname]]
                s["ids"].add(f"{g['cross']}:{g['segregant_id']}")
                s["crosses"].add(g["cross"])
                for parent in (g["parent_1"], g["parent_2"]):
                    s["parents"].add((parent["name"], parent["assembly_ref"]["sha256"]))
                continue
            if dname not in KEEP:  # Sameith Dm (double deletions): no pair uses it
                continue
            short = KEEP[dname]
            ph = ex["phenotype"]
            label = ph["label_name"]
            values = ph[label]
            ekey = _env_key(ex["environment"])
            envs.setdefault(ekey, ex["environment"])
            if short == "messner":
                ref = record["experiment_reference"]["phenotype_reference"][label]
                mapped: dict[str, float] = {}
                for acc, x in values.items():
                    orf = up2orf.get(acc)
                    r = ref.get(acc)
                    if orf is None:
                        unmapped[short].add(acc)
                        continue
                    if x and r and x > 0 and r > 0:
                        mapped[orf] = float(np.log2(x) - np.log2(r))
                values = mapped
            elif short == "zelezniak":
                ref = record["experiment_reference"]["phenotype_reference"][label]
                values = {
                    k: float(x - ref[k])
                    for k, x in values.items()
                    if x is not None and ref.get(k) is not None
                }
            elif short == "nadal":
                mapped = {}
                for name, x in values.items():
                    orf = name2orf.get(name.upper())
                    if orf is None:
                        unmapped[short].add(name)
                        continue
                    mapped[orf] = float(x)
                values = mapped
            elif short == "jackson":
                extra[short].append(float(ph["total_umis"]))
            rows[short].append((_genotype_key(ex["genotype"]), ekey, len(rows[short])))
            vals[short].append(
                {k: float(x) for k, x in values.items() if x is not None}
            )
            if n_seen % 20000 == 0:
                print(f"{n_seen} records, {time.time() - t0:.0f} s", flush=True)
    env.close()
    os.makedirs(cache, exist_ok=True)
    counts = {}
    for short in rows:
        frame = pd.DataFrame.from_records(vals[short]).astype(np.float32)
        frame.index = pd.MultiIndex.from_tuples(
            rows[short], names=["genotype", "environment", "record"]
        )
        frame.to_pickle(osp.join(cache, f"{short}.pkl"))
        counts[short] = {"records": len(frame), "genes": frame.shape[1]}
    if extra["jackson"]:
        pd.Series(extra["jackson"]).to_pickle(osp.join(cache, "jackson_total_umis.pkl"))
    meta = {
        "records_seen": n_seen,
        "seconds": time.time() - t0,
        "counts": counts,
        "unmapped_ids": {k: len(v) for k, v in unmapped.items()},
        "environments": {
            k: {
                "media": v["media"]["name"],
                "temperature": v.get("temperature"),
                "fields": {
                    f: v[f]
                    for f in (
                        "aerobicity",
                        "duration_hours",
                        "duration_generations",
                        "culture_format",
                    )
                    if f in v
                },
                "perturbations": v.get("perturbations"),
            }
            for k, v in envs.items()
        },
        "segregants": {
            n: {
                "n_ids": len(s["ids"]),
                "crosses": sorted(s["crosses"]),
                "parents": sorted(list(p) for p in s["parents"]),
            }
            for n, s in segregants.items()
        },
        "segregant_ids_shared": len(
            segregants["boocock"]["ids"] & segregants["nguessan"]["ids"]
        ),
        "segregant_crosses_shared": sorted(
            segregants["boocock"]["crosses"] & segregants["nguessan"]["crosses"]
        ),
    }
    with open(osp.join(cache, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


# ---------------------------------------------------------------------------- stats


def _q(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "median": float(med), "q1": float(q1), "q3": float(q3)}


def per_genotype(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Pearson and Spearman per shared row across the shared columns."""
    rows = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    out = []
    for key in rows:
        x = a.loc[key, cols].to_numpy(dtype=float)
        y = b.loc[key, cols].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < MIN_GENES or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            continue
        out.append(
            {
                "genotype": key,
                "n_genes": int(ok.sum()),
                "pearson": float(np.corrcoef(x[ok], y[ok])[0, 1]),
                "spearman": float(spearmanr(x[ok], y[ok]).statistic),
            }
        )
    return pd.DataFrame(out, columns=["genotype", "n_genes", "pearson", "spearman"])


def per_gene(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Pearson per shared column across the shared rows."""
    rows = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    x = a.loc[rows, cols].to_numpy(dtype=float)
    y = b.loc[rows, cols].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(0)
    xm = np.where(ok, x, 0.0)
    ym = np.where(ok, y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = xm.sum(0) / n
        my = ym.sum(0) / n
        xc = np.where(ok, x - mx, 0.0)
        yc = np.where(ok, y - my, 0.0)
        r = (xc * yc).sum(0) / np.sqrt((xc**2).sum(0) * (yc**2).sum(0))
        sd_b = np.sqrt((yc**2).sum(0) / n)
    frame = pd.DataFrame(
        {"gene": cols, "n_genotypes": n, "pearson": r, "sd_second": sd_b}
    )
    return frame[(frame.n_genotypes >= MIN_GENOTYPES) & np.isfinite(frame.pearson)]


def summarize(
    name: str, a: pd.DataFrame, b: pd.DataFrame, out_dir: str
) -> dict[str, Any]:
    """Both statistics for one pair; per-genotype and per-gene CSVs."""
    g = per_genotype(a, b)
    p = per_gene(a, b)
    g.to_csv(osp.join(out_dir, f"{name}_per_genotype.csv"), index=False)
    p.to_csv(osp.join(out_dir, f"{name}_per_gene.csv"), index=False)
    return {
        "n_shared_genotypes": len(a.index.intersection(b.index)),
        "n_shared_genes": len(a.columns.intersection(b.columns)),
        "A_per_genotype_pearson": _q(g.pearson.to_numpy()),
        "A_per_genotype_spearman": _q(g.spearman.to_numpy()),
        "B_per_gene_pearson": _q(p.pearson.to_numpy()),
        "B_fraction_genes_above_0.3": float((p.pearson > 0.3).mean())
        if len(p)
        else None,
    }


# ---------------------------------------------------------------------------- matrices


def by_genotype(frame: pd.DataFrame, environment: str | None = None) -> pd.DataFrame:
    """Mean over records per genotype (optionally in one environment)."""
    if environment is not None:
        frame = frame[frame.index.get_level_values("environment") == environment]
    return frame.groupby(level="genotype").mean()


def centered_log2(frame: pd.DataFrame, pseudo: float) -> pd.DataFrame:
    """log2(x + pseudo) per genotype, centered per column over the genotypes."""
    m = np.log2(by_genotype(frame) + pseudo)
    return m - m.mean()


def jackson_ratios(
    frame: pd.DataFrame, totals: pd.Series, min_umi: int
) -> dict[str, pd.DataFrame]:
    """Per environment: genotype x gene log2((CPM+1)/(CPM_HO+1)); genes with >= min_umi."""
    counts = frame.fillna(0.0)
    counts["__total"] = totals.to_numpy()
    summed = counts.groupby(level=["genotype", "environment"]).sum()
    total = summed.pop("__total")
    out = {}
    for envk in summed.index.get_level_values("environment").unique():
        cell = summed.xs(envk, level="environment")
        tot = total.xs(envk, level="environment")
        if HO not in cell.index:
            continue
        cpm = cell.div(tot, axis=0) * 1e6
        ratio = np.log2(cpm + 1.0) - np.log2(cpm.loc[HO] + 1.0)
        keep = cell.ge(0).all() & (cell.sum() >= min_umi)
        ratio = ratio.loc[:, keep].drop(index=HO)
        out[envk] = ratio
    return out


# ---------------------------------------------------------------------------- morphology


def _ridge(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, lam: float) -> np.ndarray:
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    if Xc.shape[1] > Xc.shape[0]:
        K = Xc @ Xc.T
        a = np.linalg.solve(K + lam * np.eye(K.shape[0]), Yc)
        return (Xte - mx) @ (Xc.T @ a) + my
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    return (Xte - mx) @ np.linalg.solve(A, Xc.T @ Yc) + my


def _colr(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Pc, Yc = P - P.mean(0), Y - Y.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Pc * Yc).sum(0) / np.sqrt((Pc**2).sum(0) * (Yc**2).sum(0))


def _oof(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    folds = np.array_split(rng.permutation(n), FOLDS)
    pred = np.full_like(Y, np.nan, dtype=float)
    for k in range(FOLDS):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(FOLDS) if j != k])
        itel, itr = tr[: len(tr) // 5], tr[len(tr) // 5 :]
        best, bl = -np.inf, LAMBDAS[0]
        for lam in LAMBDAS:
            s = np.nanmean(_colr(_ridge(X[itr], Y[itr], X[itel], lam), Y[itel]))
            if s > best:
                best, bl = s, lam
        pred[te] = _ridge(X[tr], Y[tr], X[te], bl)
    return pred


def _z(df: pd.DataFrame) -> pd.DataFrame:
    return ((df - df.mean()) / df.std(ddof=0)).clip(-ZCLIP, ZCLIP)


def morphology_ridge(
    inputs: dict[str, pd.DataFrame], ohya: pd.DataFrame, features: list[str] | None
) -> dict[str, Any]:
    """Ridge from each input matrix to the CalMorph vector on the shared genotypes."""
    target = (
        ohya if features is None else ohya[[f for f in features if f in ohya.columns]]
    )
    target = target.loc[:, target.std(ddof=0) > 0]
    rows = {}
    for name, X in inputs.items():
        shared = sorted(set(X.index) & set(target.index))
        if len(shared) < 5 * FOLDS:
            rows[name] = {"n_genotypes": len(shared), "skipped": "fewer than 25 shared"}
            continue
        Xs = X.reindex(shared)
        Xs = Xs.loc[:, Xs.notna().mean() >= MIN_COVER]
        Xs = Xs.loc[:, Xs.std(ddof=0) > 0]
        Xa = _z(Xs).fillna(0.0).to_numpy()
        Y = _z(target.loc[shared]).fillna(0.0).to_numpy()
        rng = np.random.default_rng(SEED)
        r = _colr(_oof(Xa, Y, rng), Y)
        null = _colr(_oof(Xa[rng.permutation(len(shared))], Y, rng), Y)
        rows[name] = {
            "n_genotypes": len(shared),
            "n_input_features": int(Xa.shape[1]),
            "n_target_features": int(Y.shape[1]),
            "mean_r_over_features": float(np.nanmean(r)),
            "median_r_over_features": float(np.nanmedian(r)),
            "null_mean_r": float(np.nanmean(null)),
            "null_median_r": float(np.nanmedian(null)),
        }
        print("morphology", name, json.dumps(rows[name]), flush=True)
    return rows


# ---------------------------------------------------------------------------- analyze


def analyze(cache: str, out_dir: str, moving_path: str | None) -> dict[str, Any]:
    """Every pair; returns the JSON-ready result."""
    meta = json.load(open(osp.join(cache, "meta.json")))
    mats = {
        s: pd.read_pickle(osp.join(cache, f"{s}.pkl"))
        for s in KEEP.values()
        if osp.exists(osp.join(cache, f"{s}.pkl"))
    }
    envs = meta["environments"]
    pairs: dict[str, Any] = {}
    os.makedirs(out_dir, exist_ok=True)

    kem = by_genotype(mats["kemmeren"])
    pairs["01_kemmeren_x_sameith_sm"] = summarize(
        "01_kemmeren_x_sameith_sm", kem, by_genotype(mats["sameith_sm"]), out_dir
    )
    pairs["02_kemmeren_x_hu"] = summarize(
        "02_kemmeren_x_hu", kem, by_genotype(mats["hu"]), out_dir
    )

    nadal = mats["nadal"]
    nadal_envs = sorted(nadal.index.get_level_values("environment").unique())
    nadal_by_env = {e: by_genotype(nadal, e) for e in nadal_envs}
    for i, e in enumerate(nadal_envs, start=1):
        name = f"03_kemmeren_x_nadal_condition{i}"
        pairs[name] = summarize(name, kem, nadal_by_env[e], out_dir)
        pairs[name]["environment"] = e
    if len(nadal_envs) == 2:
        name = "05a_nadal_condition1_x_condition2"
        pairs[name] = summarize(
            name, nadal_by_env[nadal_envs[0]], nadal_by_env[nadal_envs[1]], out_dir
        )
        # Within-condition duplicates: the first two records of a (genotype, env) cell.
        dup_a, dup_b = [], []
        for (_g, _e), block in nadal.groupby(level=["genotype", "environment"]):
            if len(block) >= 2:
                dup_a.append(block.iloc[0].rename((_g, _e)))
                dup_b.append(block.iloc[1].rename((_g, _e)))
        name = "05b_nadal_within_condition_duplicates"
        if dup_a:
            da = pd.DataFrame(dup_a)
            db = pd.DataFrame(dup_b)
            da.index = [f"{g}@{e}" for g, e in da.index]
            db.index = da.index
            pairs[name] = summarize(name, da, db, out_dir)
        else:
            pairs[name] = {"n_shared_genotypes": 0, "note": "no duplicate cells"}

    totals = pd.read_pickle(osp.join(cache, "jackson_total_umis.pkl"))
    jr = jackson_ratios(mats["jackson"], totals, min_umi=10)
    ypd = [e for e in jr if envs[e]["media"].startswith("YPD")]
    for i, e in enumerate(sorted(ypd), start=1):
        name = f"04_kemmeren_x_jackson_ypd{i}"
        pairs[name] = summarize(name, kem, jr[e], out_dir)
        pairs[name]["environment"] = e
    # Jackson environment effect: per genotype, Pearson between every pair of environments.
    rows = []
    for e1, e2 in combinations(sorted(jr), 2):
        a, b = jr[e1], jr[e2]
        for g in a.index.intersection(b.index):
            cols = a.columns.intersection(b.columns)
            x, y = a.loc[g, cols].to_numpy(float), b.loc[g, cols].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= MIN_GENES:
                rows.append(
                    {
                        "genotype": g,
                        "env_1": e1,
                        "env_2": e2,
                        "n_genes": int(ok.sum()),
                        "pearson": float(np.corrcoef(x[ok], y[ok])[0, 1]),
                        "spearman": float(spearmanr(x[ok], y[ok]).statistic),
                    }
                )
    jframe = pd.DataFrame(rows)
    jframe.to_csv(
        osp.join(out_dir, "06_jackson_between_environments_per_genotype.csv"),
        index=False,
    )
    pairs["06_jackson_between_environments"] = {
        "n_environments": len(jr),
        "n_genotypes": int(jframe.genotype.nunique()) if len(jframe) else 0,
        "n_genotype_environment_pairs": len(jframe),
        "A_per_genotype_pearson": _q(jframe.pearson.to_numpy())
        if len(jframe)
        else {"n": 0},
        "A_per_genotype_spearman": _q(jframe.spearman.to_numpy())
        if len(jframe)
        else {"n": 0},
        "note": "log2 ratio to HO in each environment; genes with >= 10 UMIs summed",
    }

    messner = by_genotype(mats["messner"])
    pairs["07_messner_x_zelezniak"] = summarize(
        "07_messner_x_zelezniak", messner, by_genotype(mats["zelezniak"]), out_dir
    )
    pairs["08_kemmeren_x_messner"] = summarize(
        "08_kemmeren_x_messner", kem, messner, out_dir
    )

    caudal = centered_log2(mats["caudal"], 1.0)
    muen = centered_log2(mats["muenzner"].clip(lower=0), 0.0)
    turn = centered_log2(mats["turnover"], 0.0)
    for frame in (caudal, muen, turn):
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    pairs["09a_caudal_x_muenzner_proteome"] = summarize(
        "09a_caudal_x_muenzner_proteome", caudal, muen, out_dir
    )
    pairs["09b_muenzner_turnover_x_abundance"] = summarize(
        "09b_muenzner_turnover_x_abundance", turn, muen, out_dir
    )

    ohya = by_genotype(mats["ohya"])
    inputs = {
        "kemmeren": kem,
        "messner": messner,
        "sameith_sm": by_genotype(mats["sameith_sm"]),
        "hu": by_genotype(mats["hu"]),
    }
    morph = {"all_features": morphology_ridge(inputs, ohya, None)}
    if moving_path is not None:
        with open(moving_path, "rb") as f:
            raw = f.read()
        moving = json.loads(raw)["moving"]
        morph["moving_features"] = morphology_ridge(inputs, ohya, moving)
        morph["moving_features_source"] = {
            "path": moving_path,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "n": len(moving),
        }
    return {"pairs": pairs, "morphology": morph, "extract": meta}


# ---------------------------------------------------------------------------- figures


def _style() -> None:
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
        }
    )


def _save(fig: plt.Figure, image_dir: str, title: str) -> str:
    os.makedirs(image_dir, exist_ok=True)
    base = osp.join(image_dir, f"{title}_{timestamp()}")
    fig.savefig(base + ".png", dpi=300)
    savefig_true_size_svg(fig, base + ".svg")
    plt.close(fig)
    return base + ".svg"


SHORT = {
    "01_kemmeren_x_sameith_sm": "1 Kemmeren x Sameith Sm",
    "02_kemmeren_x_hu": "2 Kemmeren x Hu",
    "03_kemmeren_x_nadal_condition1": "3 Kemmeren x Nadal YPD",
    "03_kemmeren_x_nadal_condition2": "3 Kemmeren x Nadal salt",
    "04_kemmeren_x_jackson_ypd1": "4 Kemmeren x Jackson YPD 10 h",
    "04_kemmeren_x_jackson_ypd2": "4 Kemmeren x Jackson YPD 4 h a",
    "04_kemmeren_x_jackson_ypd3": "4 Kemmeren x Jackson YPD 4 h b",
    "05a_nadal_condition1_x_condition2": "5a Nadal YPD x salt",
    "05b_nadal_within_condition_duplicates": "5b Nadal duplicates",
    "06_jackson_between_environments": "6 Jackson env x env",
    "07_messner_x_zelezniak": "7 Messner x Zelezniak",
    "08_kemmeren_x_messner": "8 Kemmeren x Messner",
    "09a_caudal_x_muenzner_proteome": "9a Caudal x Muenzner",
    "09b_muenzner_turnover_x_abundance": "9b turnover x abundance",
}


def figures(result: dict[str, Any], out_dir: str, image_dir: str) -> dict[str, str]:
    """Per-genotype distributions, per-gene mRNA-protein scatter, morphology ridge."""
    _style()
    paths = {}
    names = [
        n
        for n in sorted(result["pairs"])
        if osp.exists(osp.join(out_dir, f"{n}_per_genotype.csv"))
    ]
    data, labels = [], []
    for n in names:
        frame = pd.read_csv(osp.join(out_dir, f"{n}_per_genotype.csv"))
        if len(frame):
            data.append(frame.pearson.to_numpy())
            labels.append(f"{SHORT.get(n, n)} (n={len(frame)})")
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(80)))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(PLOT_PALETTE[i % len(PLOT_PALETTE)])
        box.set_edgecolor("black")
        box.set_linewidth(0.5)
    for key in ("whiskers", "caps", "medians"):
        for line in bp[key]:
            line.set_color("black")
            line.set_linewidth(0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Per-genotype Pearson r\n(across shared genes)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(-1, 1.05)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="#DDDDDD")
    fig.tight_layout()
    paths["per_genotype"] = _save(fig, image_dir, "cross_dataset_per_genotype_pearson")

    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(45))
    )
    for ax, (n, title) in zip(
        axes,
        [
            ("08_kemmeren_x_messner", "Kemmeren mRNA x Messner protein"),
            ("09a_caudal_x_muenzner_proteome", "Caudal mRNA x Muenzner protein"),
        ],
        strict=True,
    ):
        frame = pd.read_csv(osp.join(out_dir, f"{n}_per_gene.csv"))
        ax.scatter(
            frame.sd_second, frame.pearson, s=1.5, color=PLOT_PALETTE[0], linewidths=0
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(0.3, color=PLOT_PALETTE[1], linewidth=0.5, linestyle="--")
        ax.set_title(f"{title}\n(n={len(frame)} genes)")
        ax.set_xlabel("Protein SD across genotypes (log2)")
        ax.set_ylabel("Per-gene Pearson r")
        ax.set_ylim(-1, 1)
    fig.tight_layout()
    paths["mrna_protein"] = _save(fig, image_dir, "cross_dataset_mrna_protein_per_gene")

    morph = result["morphology"]
    sets = [k for k in ("all_features", "moving_features") if k in morph]
    inputs = [
        k
        for k in morph["all_features"]
        if "mean_r_over_features" in morph["all_features"][k]
    ]
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(60)))
    width = 0.8 / max(1, 2 * len(sets))
    for j, s in enumerate(sets):
        for k, stat in enumerate(("mean_r_over_features", "null_mean_r")):
            xs = np.arange(len(inputs)) + (2 * j + k) * width
            ys = [morph[s][i].get(stat, np.nan) for i in inputs]
            color = PLOT_PALETTE[j] if k == 0 else PLOT_PALETTE[5]
            ax.bar(
                xs,
                ys,
                width,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=f"{s.replace('_', ' ')}: {'model' if k == 0 else 'row-permutation null'}",
            )
    ax.set_xticks(np.arange(len(inputs)) + width * (len(sets) * 2 - 1) / 2)
    ax.set_xticklabels(
        [f"{i}\n(n={morph['all_features'][i]['n_genotypes']})" for i in inputs]
    )
    ax.set_ylabel("Mean validation Pearson r\nper CalMorph feature")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.tight_layout()
    paths["morphology"] = _save(fig, image_dir, "cross_dataset_morphology_ridge")
    return paths


def headline(result: dict[str, Any]) -> str:
    """Markdown headline table."""
    lines = [
        "| pair | n shared genotypes | n shared genes | median per-genotype Pearson r (n) | median per-genotype Spearman rho | median per-gene Pearson r (n genes) | fraction of genes with r > 0.3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, p in sorted(result["pairs"].items()):
        a = p.get("A_per_genotype_pearson", {})
        s = p.get("A_per_genotype_spearman", {})
        b = p.get("B_per_gene_pearson", {})
        f = p.get("B_fraction_genes_above_0.3")
        lines.append(
            f"| {name} | {p.get('n_shared_genotypes', p.get('n_genotype_environment_pairs', ''))} | "
            f"{p.get('n_shared_genes', '')} | "
            + (f"{a['median']:.3f} ({a['n']})" if a.get("n") else "n/a")
            + " | "
            + (f"{s['median']:.3f}" if s.get("n") else "n/a")
            + " | "
            + (f"{b['median']:.3f} ({b['n']})" if b.get("n") else "n/a")
            + " | "
            + (f"{f:.3f}" if f is not None else "n/a")
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    """Extract, analyze, plot."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=["extract", "analyze", "all"], default="all")
    parser.add_argument("--moving-features", default=None)
    parser.add_argument(
        "--stride", type=int, default=1, help="smoke test: every k-th record"
    )
    parser.add_argument(
        "--tag", default="", help="suffix for the cache, results and figures"
    )
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    store = osp.join(data_root, "data/torchcell/experiments", EXPERIMENT)
    raw_path = osp.join(store, "raw_043", "raw", "lmdb")
    cache = osp.join(store, f"cross_dataset_cache{args.tag}")
    results = experiment_results_dir(EXPERIMENT, __file__)
    if args.tag:
        results = osp.join(results, f"smoke{args.tag}")
    out_dir = osp.join(results, "cross_dataset_correlation")
    image_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], EXPERIMENT + args.tag)
    if args.stage in ("extract", "all"):
        meta = extract(raw_path, cache, data_root, args.stride)
        print(
            json.dumps(
                {
                    k: meta[k]
                    for k in ("records_seen", "seconds", "counts", "unmapped_ids")
                },
                indent=1,
            )
        )
    if args.stage in ("analyze", "all"):
        result = analyze(cache, out_dir, args.moving_features)
        result["quantity"] = QUANTITY
        result["figures"] = figures(result, out_dir, image_dir)
        result["generated_by"] = (
            f"experiments/{EXPERIMENT}/scripts/cross_dataset_correlation.py"
        )
        result["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
        with open(osp.join(results, "cross_dataset_correlation.json"), "w") as f:
            json.dump(result, f, indent=1, default=str)
        print(headline(result))
        print(json.dumps(result["morphology"], indent=1))


if __name__ == "__main__":
    main()
