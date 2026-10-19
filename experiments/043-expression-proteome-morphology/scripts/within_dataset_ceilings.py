# experiments/043-expression-proteome-morphology/scripts/within_dataset_ceilings.py
# [[experiments.043-expression-proteome-morphology.scripts.within_dataset_ceilings]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/within_dataset_ceilings
"""Within-dataset reproducibility of the 043 expression datasets.

Each section measures how well a dataset agrees with ITSELF, which bounds what any
model can score against it. A cross-dataset agreement can only be read against these.

``jackson`` (Jackson 2020, dev LMDB ``tf_deletion_pseudobulk_jackson2020``, the same 784
records the raw store serves). A record is one construction (one of six independently
built barcoded strains per genotype) in one condition, its cells summed. Within one
condition, for every pair of constructions:

* level profile: Pearson over genes of log2(CPM + 1) of the two constructions;
* deletion-effect profile: Pearson over genes of log2((CPM + 1) / (CPM_HO + 1)), with
  CPM_HO the condition's stored reference (the six HO constructions SUMMED). The HO
  constructions themselves are left out of the effect comparison, since the summed
  reference contains them.

CPM uses the record's ``total_umis`` (every feature). Gene rule, floor in {10, 100,
1000}: primary (``condition_sum``), a gene enters every pair of a condition when its UMIs
summed over that condition's records reach the floor, so one gene set per condition and
every pair is scored; secondary (``both_constructions``), a gene enters a pair when it
has at least ``floor`` UMIs in both constructions, which leaves shallow pairs with fewer
than 20 genes unscored and so over-represents deep constructions. Same-genotype pairs
(15 per genotype per condition) are read against the null of different-genotype pairs in
the same condition, which share the same reference and the same flask.

``kemmeren`` (Kemmeren 2014, raw GEO SOFT pickles GSE42527 + GSE42526). A GEO title
``<gene>-del-<k>-<x>`` names culture k and array x of the dye pair. Per array the loader's
own functions give the (deletion, reference pool) signals
(``_collect_replicate_pairs_static``) and the per-culture profile is
``create_expression_experiment``'s mean of within-array log2(deletion / refpool), so the
normalization is the loader's. Mutants with two cultures give a test-retest pair
(culture 1 against the next culture number present): per-strain Pearson across genes
(null: culture 1 of one mutant against the second culture of another), per-gene Pearson
across the paired mutants, and the ceiling of
``experiments/019-simb-multimodal/scripts/expression_ceiling_all.py``:
ceiling_g = sqrt(clip(r_g, 0, 1)), averaged over genes with at least 3 pairs. The
decomposition route of the same script is also given: rel_g = 1 - noise_var_g /
total_var_g, noise variance = var(culture 1 - culture 2) / 2 over the paired mutants,
total variance over the first culture of every mutant.

``hu`` (Hu 2007). Reads the mirrored paper and Supplementary Methods and records the
sentences that settle whether replicate arrays are released; no computation is
possible when they are not, and the Kemmeren x Hu agreement of
``results/cross_dataset_correlation.json`` is carried as the available bound.

``boocock`` (Boocock 2025, dev LMDB ``boocock2025``). Per panel: records, distinct
genotypes (block content), cells, genes per record, UMIs per record, and the fraction
of marker weight whose posterior of the called parent is below 0.9. The pooled BY x RM
panel (393 segregants, cells summed per segregant) and the de novo BY x RM cells of
cross A are compared on a 5 kb grid of called parents (forward-filled from the
preceding block): each de novo cell's best concordance with any pooled segregant
decides whether a segregant is in both. For every pooled segregant the 20 de novo cells
nearest in genotype are summed, and the Pearson over genes of the two deviation
profiles (log2 CPM minus the own panel's log2 CPM) is read against 20 random cells.

``nguessan`` (N'Guessan 2025, dev LMDB ``nguessan2025``): segregants, cells per
segregant, genes and UMIs per record, the posterior fraction as above, and the same
nearest-genotype comparison against Boocock's pooled panel (both BY x RM, both pooled).

Writes ``results/within_dataset_ceilings.json``, per-dataset CSVs under
``results/within_dataset_ceilings/`` and two figures under
``$ASSET_IMAGES_DIR/043-expression-proteome-morphology/``.

    python experiments/043-expression-proteome-morphology/scripts/within_dataset_ceilings.py --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations, product
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from torchcell.datasets.scerevisiae.kemmeren2014 import (  # noqa: E402
    MicroarrayKemmeren2014Dataset,
)
from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils.paths import experiment_results_dir  # noqa: E402
from torchcell.utils.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from store_census import _iter_dev  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"
FLOORS = (10, 100, 1000)
GENE_RULES = ("condition_sum", "both_constructions")
MIN_GENES = 20
MIN_PAIRS = 3
GRID_BP = 5000
K_NEAREST = 20
DEV_FLOOR = 10
SEED = 0
HU_KEY = "huGeneticReconstructionFunctional2007"


def _q(x: Any) -> dict[str, float | int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {
        "n": int(len(x)),
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "mean": float(x.mean()),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_GENES or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------- Jackson


def jackson(data_root: str, out_dir: str) -> dict[str, Any]:
    """Construction-vs-construction agreement within each condition."""
    path = osp.join(
        data_root, "data/torchcell/tf_deletion_pseudobulk_jackson2020/processed/lmdb"
    )
    rows: list[dict[str, Any]] = []
    counts: list[dict[str, float]] = []
    reference: dict[str, tuple[dict[str, float], float]] = {}
    for record in _iter_dev(path):
        ex = record["experiment"]
        ph = ex["phenotype"]
        pert = ex["genotype"]["perturbations"][0]
        condition = next(iter(ph["library_cell_counts"]))
        rows.append(
            {
                "condition": condition,
                "genotype": pert["perturbed_gene_name"],
                "construction": pert["construction"]["strain_accession"],
                "total_umis": float(ph["total_umis"]),
                "n_cells": int(ph["n_cells"]),
            }
        )
        counts.append({k: float(v) for k, v in ph["umi_count"].items()})
        ref = record["reference"]["phenotype_reference"]
        reference.setdefault(
            condition,
            (
                {k: float(v) for k, v in ref["umi_count"].items()},
                float(ref["total_umis"]),
            ),
        )
    meta = pd.DataFrame(rows)
    mat = pd.DataFrame.from_records(counts).fillna(0.0)
    genes = list(mat.columns)
    umi = mat.to_numpy(dtype=float)
    cpm = umi / meta.total_umis.to_numpy()[:, None] * 1e6
    level = np.log2(cpm + 1.0)
    effect = np.full_like(level, np.nan)
    for condition, (ref_counts, ref_total) in reference.items():
        ref_vec = np.array([ref_counts.get(g, 0.0) for g in genes]) / ref_total * 1e6
        idx = np.flatnonzero((meta.condition == condition).to_numpy())
        effect[idx] = np.log2((cpm[idx] + 1.0) / (ref_vec[None, :] + 1.0))

    pair_rows = []
    for condition, sub in meta.groupby("condition"):
        idx = sub.index.to_numpy()
        summed = umi[idx].sum(0)
        for i, j in combinations(idx, 2):
            same = meta.genotype[i] == meta.genotype[j]
            ho = "HO" in (meta.genotype[i], meta.genotype[j])
            for floor, rule in product(FLOORS, GENE_RULES):
                ok = (
                    summed >= floor
                    if rule == "condition_sum"
                    else (umi[i] >= floor) & (umi[j] >= floor)
                )
                pair_rows.append(
                    {
                        "condition": condition,
                        "rule": rule,
                        "floor": floor,
                        "genotype_i": meta.genotype[i],
                        "genotype_j": meta.genotype[j],
                        "same_genotype": bool(same),
                        "involves_ho": bool(ho),
                        "n_genes": int(ok.sum()),
                        "level_r": _pearson(level[i, ok], level[j, ok]),
                        "effect_r": (
                            float("nan")
                            if ho
                            else _pearson(effect[i, ok], effect[j, ok])
                        ),
                    }
                )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(osp.join(out_dir, "jackson_construction_pairs.csv"), index=False)
    meta.to_csv(osp.join(out_dir, "jackson_records.csv"), index=False)

    def block(frame: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for stat in ("level_r", "effect_r"):
            same = frame[frame.same_genotype][stat]
            diff = frame[~frame.same_genotype][stat]
            out[stat] = {
                "same_genotype": _q(same),
                "different_genotype_null": _q(diff),
                "median_difference": float(np.nanmedian(same) - np.nanmedian(diff))
                if same.notna().any() and diff.notna().any()
                else None,
            }
        out["n_genes_per_pair"] = _q(frame.n_genes)
        return out

    per_condition: dict[str, Any] = {}
    for condition, sub in meta.groupby("condition"):
        per_condition[condition] = {
            "records": int(len(sub)),
            "genotypes": int(sub.genotype.nunique()),
            "constructions_per_genotype": _q(sub.groupby("genotype").size()),
            "umis_per_construction": _q(sub.total_umis),
            "cells_per_construction": _q(sub.n_cells),
            "floors": {
                str(f): block(
                    pairs[
                        (pairs.condition == condition)
                        & (pairs.floor == f)
                        & (pairs.rule == "condition_sum")
                    ]
                )
                for f in FLOORS
            },
        }
    return {
        "source": path,
        "records": int(len(meta)),
        "conditions": int(meta.condition.nunique()),
        "genotypes": int(meta.genotype.nunique()),
        "constructions": int(meta.construction.nunique()),
        "reference": "the condition's stored reference: the six HO constructions summed",
        "gene_rules": {
            "condition_sum": "primary: a gene enters every pair of a condition when its UMIs summed over that condition's records reach the floor (one gene set per condition, every pair scored)",
            "both_constructions": "secondary: a gene enters a pair when it has >= floor UMIs in both constructions (pairs with < 20 such genes are not scored, so deep constructions are over-represented)",
        },
        "umis_per_construction": _q(meta.total_umis),
        "cells_per_construction": _q(meta.n_cells),
        "pooled": {
            rule: {
                str(f): block(pairs[(pairs.floor == f) & (pairs.rule == rule)])
                for f in FLOORS
            }
            for rule in GENE_RULES
        },
        "per_condition": per_condition,
    }


# ---------------------------------------------------------------------------- Kemmeren

TITLE = re.compile(r"-del-(?:mata-)?(\d+)-?([a-d])$", re.IGNORECASE)


def kemmeren(data_root: str, out_dir: str) -> dict[str, Any]:
    """Own-culture test-retest from the raw GEO arrays, through the loader's functions."""
    raw_dir = osp.join(data_root, "data/torchcell/microarray_kemmeren2014/raw")
    ds = MicroarrayKemmeren2014Dataset
    resolver = SimpleNamespace(
        raw_dir=raw_dir,
        resolved_by_table_s1_gene=0,
        resolved_by_table_s1_orf=0,
        resolved_by_release_title_names=0,
        released_gene_names={},
    )
    systematic_to_strain, common_to_systematic = ds._load_mating_type_map(resolver)
    gsms: dict[str, Any] = {}
    series_of: dict[str, str] = {}
    probe_map: dict[str, str] = {}
    sha: dict[str, str] = {}
    for acc in ("GSE42527", "GSE42526"):
        pkl = osp.join(raw_dir, f"{acc}.pkl")
        with open(pkl, "rb") as f:
            raw = f.read()
        sha[f"{acc}.pkl"] = hashlib.sha256(raw).hexdigest()
        gse = pickle.loads(raw)
        del raw
        if not probe_map:
            probe_map = ds._extract_probe_to_gene_mapping(None, gse)
        for name, gsm in gse.gsms.items():
            gsms[name] = gsm
            series_of[name] = acc

    by_culture: dict[str, dict[int, list[Any]]] = defaultdict(lambda: defaultdict(list))
    unparsed: list[str] = []
    for name, gsm in gsms.items():
        title = gsm.metadata["title"][0]
        if "-del" not in title.lower():
            continue
        token = title.split("-del")[0].strip()
        if "]" in token:
            token = token.split("]")[-1].strip()
        orf = ds.resolve_gene_name_comprehensive(
            resolver, token.upper(), common_to_systematic, systematic_to_strain
        )
        m = TITLE.search(title.strip())
        if m is None:
            unparsed.append(title)
            continue
        by_culture[orf][int(m.group(1))].append(gsm)

    profiles: dict[tuple[str, int], dict[str, float]] = {}
    t0 = time.time()
    for k, (orf, cultures) in enumerate(sorted(by_culture.items())):
        for culture, arrays in cultures.items():
            pairs = ds._collect_replicate_pairs_static(arrays, probe_map)
            experiment, _, _ = ds.create_expression_experiment(
                "MicroarrayKemmeren2014Dataset",
                {
                    "systematic_gene_name": orf,
                    "perturbed_gene_name": resolver.released_gene_names[orf],
                    "num_replicates": len(arrays),
                    "strain": systematic_to_strain[orf],
                },
                pairs,
            )
            if experiment is not None:
                profiles[(orf, culture)] = dict(
                    experiment.phenotype.expression_log2_ratio
                )
        if k % 200 == 0:
            print(f"kemmeren {k}/{len(by_culture)} mutants, {time.time() - t0:.0f} s")

    frame = pd.DataFrame.from_dict(profiles, orient="index").astype(float)
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["orf", "culture"])
    mutants = sorted({o for o, _ in frame.index})
    paired = []
    for orf in mutants:
        cs = sorted(c for o, c in frame.index if o == orf)
        if len(cs) >= 2:
            paired.append((orf, cs[0], cs[1]))
    first = frame.loc[
        [(o, min(c for oo, c in frame.index if oo == o)) for o in mutants]
    ]
    a = frame.loc[[(o, c1) for o, c1, _ in paired]].to_numpy()
    b = frame.loc[[(o, c2) for o, _, c2 in paired]].to_numpy()
    genes = list(frame.columns)

    per_strain = []
    for i, (orf, c1, c2) in enumerate(paired):
        ok = np.isfinite(a[i]) & np.isfinite(b[i])
        per_strain.append(
            {
                "orf": orf,
                "culture_1": c1,
                "culture_2": c2,
                "series": series_of[by_culture[orf][c1][0].name],
                "n_genes": int(ok.sum()),
                "pearson": _pearson(a[i, ok], b[i, ok]),
            }
        )
    per_strain_df = pd.DataFrame(per_strain)
    rng = np.random.default_rng(SEED)
    null = []
    n = len(paired)
    for i in range(n):
        j = (i + 1 + rng.integers(n - 1)) % n
        ok = np.isfinite(a[i]) & np.isfinite(b[j])
        null.append(_pearson(a[i, ok], b[j, ok]))

    ok = np.isfinite(a) & np.isfinite(b)
    cnt = ok.sum(0)
    a0, b0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        am = a0.sum(0) / np.maximum(cnt, 1)
        bm = b0.sum(0) / np.maximum(cnt, 1)
        ac, bc = np.where(ok, a0 - am, 0.0), np.where(ok, b0 - bm, 0.0)
        r_gene = (ac * bc).sum(0) / np.sqrt((ac**2).sum(0) * (bc**2).sum(0))
    r_gene[cnt < MIN_PAIRS] = np.nan
    noise_var = np.nanvar(a - b, axis=0, ddof=1) / 2.0
    total = first.to_numpy()
    total_var = np.nanvar(total, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_dec = 1.0 - noise_var / total_var
    per_gene = pd.DataFrame(
        {
            "gene": genes,
            "n_pairs": cnt,
            "test_retest_r": r_gene,
            "noise_var": noise_var,
            "total_var_first_culture": total_var,
            "rel_decomposition": rel_dec,
        }
    )
    per_strain_df.to_csv(osp.join(out_dir, "kemmeren_per_strain.csv"), index=False)
    per_gene.to_csv(osp.join(out_dir, "kemmeren_per_gene.csv"), index=False)

    def ceiling(rel: np.ndarray) -> dict[str, float | int]:
        rel = rel[np.isfinite(rel)]
        c = np.sqrt(np.clip(rel, 0.0, 1.0))
        return {
            "n_genes": int(len(rel)),
            "mean_rel": float(rel.mean()),
            "median_rel": float(np.median(rel)),
            "ceiling_mean_sqrt": float(c.mean()),
            "ceiling_median": float(np.median(c)),
            "ceiling_p25": float(np.percentile(c, 25)),
            "ceiling_p75": float(np.percentile(c, 75)),
            "frac_rel_below_0": float((rel < 0).mean()),
        }

    keep = (cnt >= MIN_PAIRS) & (total_var > 1e-8)
    r_kept = r_gene[keep]
    sb = 2 * np.clip(r_kept, 0, 1) / (1 + np.clip(r_kept, 0, 1))
    culture_counts = Counter(len(c) for c in by_culture.values())
    series_counts = Counter(series_of[by_culture[o][c1][0].name] for o, c1, _ in paired)
    return {
        "source": {"raw_dir": raw_dir, "sha256": sha},
        "arrays_parsed": int(
            sum(len(v) for c in by_culture.values() for v in c.values())
        ),
        "arrays_unparsed_titles": unparsed,
        "mutants": len(by_culture),
        "mutants_by_number_of_cultures": {
            str(k): v for k, v in sorted(culture_counts.items())
        },
        "paired_mutants": len(paired),
        "paired_mutants_by_series": dict(series_counts),
        "series": {
            "GSE42527": "responsive mutants",
            "GSE42526": "non-responsive mutants",
        },
        "per_strain_pearson": _q(per_strain_df.pearson),
        "per_strain_null_different_mutants": _q(null),
        "per_gene_pearson": _q(r_kept),
        "ceiling_test_retest": ceiling(r_kept),
        "ceiling_decomposition": ceiling(rel_dec[keep]),
        "two_culture_mean_spearman_brown": {
            "note": "derived, not measured: rel2 = 2 r / (1 + r) per gene, the reliability of the mean of two exchangeable cultures",
            "ceiling_mean_sqrt": float(np.sqrt(sb).mean()),
        },
        "decomposition_total_variance": f"first culture of each of {len(mutants)} mutants",
        "definition": "ceiling_g = sqrt(clip(r_g, 0, 1)) over genes with >= 3 paired mutants (expression_ceiling_all.py); decomposition rel_g = 1 - var(c1 - c2) / 2 / total_var_g",
    }


# ---------------------------------------------------------------------------- Hu


def _find(path: str, needles: list[str]) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode()
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for needle in needles:
            if needle in line:
                start = line.index(needle)
                end = line.find(".", start + len(needle))
                out.append(
                    {
                        "line": lineno,
                        "quote": line[start : end + 1 if end > 0 else len(line)],
                    }
                )
    return out


def hu(data_root: str, results: str) -> dict[str, Any]:
    """Whether Hu 2007 releases replicate arrays; the available bound if not."""
    lib = osp.join(data_root, "torchcell-library", HU_KEY)
    paper = osp.join(lib, "paper.md")
    methods = osp.join(lib, "si", "si8.md")
    quotes = {
        "paper.md": _find(
            paper,
            [
                "Each experiment was independently carried out at least twice",
                "performed weighted duplicate averaging",
            ],
        ),
        "si/si8.md": _find(
            methods,
            [
                "The repeatability of each gene's measured expression change across multiple independent biologically repeated experiments"
            ],
        ),
    }
    shas = {}
    for name, p in (("paper.md", paper), ("si/si8.md", methods)):
        with open(p, "rb") as f:
            shas[name] = hashlib.sha256(f.read()).hexdigest()
    with open(osp.join(results, "cross_dataset_correlation.json")) as f:
        cross = json.load(f)
    pair = cross["pairs"]["02_kemmeren_x_hu"]
    return {
        "replicates_released": False,
        "reason": "GEO GSE4654 holds one sample per strain: the error-model X score after weighted duplicate averaging; the per-array data went to the Longhorn Array Database and www.iyerlab.org/tfnetwork, never archived (torchcell/datasets/scerevisiae/hu2007.py module docstring)",
        "quotes": quotes,
        "sha256": shas,
        "bound_kemmeren_x_hu": {
            "source": "results/cross_dataset_correlation.json pairs.02_kemmeren_x_hu",
            "n_shared_genotypes": pair["n_shared_genotypes"],
            "per_genotype_pearson": pair["A_per_genotype_pearson"],
            "per_gene_pearson": pair["B_per_gene_pearson"],
        },
    }


# ---------------------------------------------------------------------------- segregants


def _grid(chrom_len: dict[str, int]) -> tuple[dict[str, np.ndarray], int]:
    pos = {c: np.arange(0, L + 1, GRID_BP) for c, L in sorted(chrom_len.items())}
    return pos, int(sum(len(v) for v in pos.values()))


def _encode(blocks: list[dict[str, Any]], grid: dict[str, np.ndarray]) -> np.ndarray:
    """Called parent (-1 = parent 1, +1 = parent 2) on the grid, forward-filled."""
    by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for b in blocks:
        by_chrom[b["chromosome"]].append(
            (int(b["start"]), 1 if b["parent"] == 2 else -1)
        )
    out = []
    for chrom, pos in grid.items():
        starts = sorted(by_chrom.get(chrom, []))
        if not starts:
            out.append(np.zeros(len(pos), dtype=np.int8))
            continue
        s = np.array([x for x, _ in starts])
        v = np.array([p for _, p in starts], dtype=np.int8)
        k = np.clip(np.searchsorted(s, pos, side="right") - 1, 0, None)
        out.append(v[k])
    return np.concatenate(out)


def _uncertain_fraction(blocks: list[dict[str, Any]], cut: float) -> float:
    w = np.array([b["n_markers"] for b in blocks], dtype=float)
    p = np.array([b["posterior"] for b in blocks], dtype=float)
    return float(w[p < cut].sum() / w.sum()) if w.sum() else float("nan")


def _block_key(blocks: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [(b["chromosome"], b["start"], b["end"], b["parent"]) for b in blocks]
        ).encode()
    ).hexdigest()[:16]


def _log_cpm(counts: np.ndarray, total: float) -> np.ndarray:
    return np.log2(counts / total * 1e6 + 1.0)


def _deviation_agreement(
    pooled: np.ndarray,
    pooled_totals: np.ndarray,
    cells: np.ndarray,
    cell_totals: np.ndarray,
    concord: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Per pooled segregant: deviation r against its k nearest cells and k random ones."""
    panel_a = _log_cpm(pooled.sum(0), pooled_totals.sum())
    panel_b = _log_cpm(cells.sum(0), cell_totals.sum())
    rows = []
    for i in range(pooled.shape[0]):
        order = np.argsort(-concord[i])[:k]
        rand = rng.choice(cells.shape[0], size=k, replace=False)
        x_counts = pooled[i]
        x = _log_cpm(x_counts, pooled_totals[i]) - panel_a
        rec = {
            "segregant": i,
            "nearest_concordance_mean": float(concord[i, order].mean()),
            "random_concordance_mean": float(concord[i, rand].mean()),
        }
        for tag, idx in (("nearest", order), ("random", rand)):
            y_counts = cells[idx].sum(0)
            y = _log_cpm(y_counts, cell_totals[idx].sum()) - panel_b
            ok = (x_counts >= DEV_FLOOR) & (y_counts >= DEV_FLOOR)
            rec[f"{tag}_r"] = _pearson(x[ok], y[ok])
            rec[f"{tag}_n_genes"] = int(ok.sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def _paired(a: pd.Series, b: pd.Series) -> dict[str, float | int]:
    """Paired nearest-minus-random difference and its Wilcoxon signed-rank test."""
    ok = a.notna() & b.notna()
    d = (a[ok] - b[ok]).to_numpy()
    test = wilcoxon(d)
    return {
        "n": int(len(d)),
        "median_difference": float(np.median(d)),
        "fraction_positive": float((d > 0).mean()),
        "wilcoxon_p": float(test.pvalue),
    }


def segregants(data_root: str, out_dir: str) -> dict[str, Any]:
    """Boocock 2025 and N'Guessan 2025 internal checks."""
    boo_path = osp.join(data_root, "data/torchcell/boocock2025/processed/lmdb")
    ngu_path = osp.join(data_root, "data/torchcell/nguessan2025/processed/lmdb")
    rows: list[dict[str, Any]] = []
    keep_blocks: dict[int, list[dict[str, Any]]] = {}
    keep_counts: dict[int, dict[str, int]] = {}
    chrom_len: dict[str, int] = defaultdict(int)
    genes_detected: dict[str, set[str]] = defaultdict(set)
    t0 = time.time()
    for source, path in (("boocock", boo_path), ("nguessan", ngu_path)):
        for record in _iter_dev(path):
            ex = record["experiment"]
            g, ph = ex["genotype"], ex["phenotype"]
            pooled = g["call_method"].startswith(("whole-genome", "reference-panel"))
            panel = (
                "nguessan"
                if source == "nguessan"
                else ("Ap pooled" if pooled else f"{g['cross']} de novo")
            )
            idx = len(rows)
            rows.append(
                {
                    "panel": panel,
                    "segregant_id": g["segregant_id"],
                    "genotype_key": _block_key(g["blocks"]),
                    "n_blocks": len(g["blocks"]),
                    "uncertain_lt_0.9": _uncertain_fraction(g["blocks"], 0.9),
                    "uncertain_lt_0.99": _uncertain_fraction(g["blocks"], 0.99),
                    "n_cells": int(ph["n_cells"]),
                    "total_umis": float(ph["total_umis"]),
                    "genes_detected": len(ph["umi_count"]),
                    "n_features": int(ph["n_features"]),
                }
            )
            genes_detected[panel].update(ph["umi_count"])
            if panel in ("Ap pooled", "A de novo", "nguessan"):
                keep_blocks[idx] = g["blocks"]
                keep_counts[idx] = ph["umi_count"]
                for b in g["blocks"]:
                    chrom_len[b["chromosome"]] = max(
                        chrom_len[b["chromosome"]], int(b["end"])
                    )
            if idx % 20000 == 0:
                print(f"segregants {idx} records, {time.time() - t0:.0f} s", flush=True)
    meta = pd.DataFrame(rows)
    meta.to_csv(osp.join(out_dir, "segregant_records.csv"), index=False)

    panels: dict[str, Any] = {}
    for panel, sub in meta.groupby("panel"):
        panels[panel] = {
            "records": int(len(sub)),
            "distinct_genotypes": int(sub.genotype_key.nunique()),
            "cells": int(sub.n_cells.sum()),
            "cells_per_record": _q(sub.n_cells),
            "umis_per_record": _q(sub.total_umis),
            "genes_detected_per_record": _q(sub.genes_detected),
            "genes_detected_in_panel": len(genes_detected[panel]),
            "analysis_features": sorted(int(x) for x in sub.n_features.unique()),
            "blocks_per_genotype": _q(sub.n_blocks),
            "marker_fraction_posterior_lt_0.9": _q(sub["uncertain_lt_0.9"]),
            "marker_fraction_posterior_lt_0.99": _q(sub["uncertain_lt_0.99"]),
        }

    grid, n_grid = _grid(chrom_len)
    ids = {
        p: meta.index[meta.panel == p].to_numpy()
        for p in ("Ap pooled", "A de novo", "nguessan")
    }
    enc = {
        p: np.stack([_encode(keep_blocks[i], grid) for i in v]).astype(np.float32)
        for p, v in ids.items()
    }
    gene_list = sorted(
        set().union(*(set(keep_counts[i]) for v in ids.values() for i in v))
    )
    gidx = {g: j for j, g in enumerate(gene_list)}

    def dense(p: str) -> tuple[np.ndarray, np.ndarray]:
        m = np.zeros((len(ids[p]), len(gene_list)), dtype=np.float32)
        for r, i in enumerate(ids[p]):
            for gname, v in keep_counts[i].items():
                m[r, gidx[gname]] = v
        return m, meta.total_umis.to_numpy()[ids[p]]

    def concordance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        defined = (x != 0).astype(np.float32) @ (y != 0).astype(np.float32).T
        return 0.5 * (1.0 + (x @ y.T) / np.maximum(defined, 1.0))

    rng = np.random.default_rng(SEED)
    comparisons: dict[str, Any] = {}
    pooled_counts, pooled_tot = dense("Ap pooled")
    for name, other in (
        ("ap_pooled_x_a_de_novo", "A de novo"),
        ("ap_pooled_x_nguessan", "nguessan"),
    ):
        conc = concordance(enc["Ap pooled"], enc[other])
        best = conc.max(axis=1)
        best_other = conc.max(axis=0)
        cells, cell_tot = dense(other)
        k = K_NEAREST if other == "A de novo" else 1
        dev = _deviation_agreement(
            pooled_counts, pooled_tot, cells, cell_tot, conc, k, rng
        )
        dev.to_csv(osp.join(out_dir, f"{name}_deviation.csv"), index=False)
        pd.DataFrame({"pooled_best_concordance": best}).to_csv(
            osp.join(out_dir, f"{name}_best_concordance.csv"), index=False
        )
        ok = (pooled_counts.sum(0) >= DEV_FLOOR) & (cells.sum(0) >= DEV_FLOOR)
        comparisons[name] = {
            "grid_bp": GRID_BP,
            "grid_points": n_grid,
            "pooled_best_concordance": _q(best),
            "other_best_concordance": _q(best_other),
            "pooled_with_match_ge_0.95": int((best >= 0.95).sum()),
            "other_with_match_ge_0.95": int((best_other >= 0.95).sum()),
            "k_nearest": k,
            "deviation_r_nearest": _q(dev.nearest_r),
            "deviation_r_random": _q(dev.random_r),
            "deviation_paired": _paired(dev.nearest_r, dev.random_r),
            "nearest_concordance": _q(dev.nearest_concordance_mean),
            "random_concordance": _q(dev.random_concordance_mean),
            "panel_level_per_gene_log_cpm_r": _pearson(
                _log_cpm(pooled_counts.sum(0), pooled_tot.sum())[ok],
                _log_cpm(cells.sum(0), cell_tot.sum())[ok],
            ),
            "panel_level_genes": int(ok.sum()),
        }
    return {
        "source": {"boocock": boo_path, "nguessan": ngu_path},
        "panels": panels,
        "comparisons": comparisons,
        "definitions": {
            "concordance": f"fraction of {GRID_BP} bp grid points where the forward-filled called parents agree, over points defined in both",
            "deviation_r": f"Pearson over genes (>= {DEV_FLOOR} UMIs on both sides) of log2 CPM minus the own panel's log2 CPM",
            "posterior_fraction": "marker-weighted fraction of a genotype's blocks whose posterior of the called parent is below the cut",
        },
    }


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


def _boxes(ax: plt.Axes, data: list[np.ndarray], colors: list[str]) -> None:
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for box, color in zip(bp["boxes"], colors, strict=True):
        box.set_facecolor(color)
        box.set_edgecolor("black")
        box.set_linewidth(0.5)
    for key in ("whiskers", "caps", "medians"):
        for line in bp[key]:
            line.set_color("black")
            line.set_linewidth(0.5)


def _grid_y(ax: plt.Axes) -> None:
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)


def figures(out_dir: str, image_dir: str) -> dict[str, str]:
    """Ceilings figure (Jackson, Kemmeren) and segregant figure."""
    _style()
    paths: dict[str, str] = {}
    pairs = pd.read_csv(osp.join(out_dir, "jackson_construction_pairs.csv"))
    strain = pd.read_csv(osp.join(out_dir, "kemmeren_per_strain.csv"))
    gene = pd.read_csv(osp.join(out_dir, "kemmeren_per_gene.csv"))
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(52))
    )
    ax = axes[0]
    data, colors, labels = [], [], []
    for f in FLOORS:
        sub = pairs[
            (pairs.floor == f) & (pairs.rule == "condition_sum") & ~pairs.involves_ho
        ]
        for same, color in ((True, PLOT_PALETTE[0]), (False, PLOT_PALETTE[5])):
            x = sub[sub.same_genotype == same].effect_r.dropna().to_numpy()
            data.append(x)
            colors.append(color)
            labels.append(f"{'same' if same else 'diff.'}\n≥{f}")
    _boxes(ax, data, colors)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson r, deletion-effect profile\n(construction vs construction)")
    ax.set_xlabel("Genotype pair, UMI floor per gene")
    ax.set_title("Jackson 2020, 11 conditions pooled")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(-0.4, 1.0)
    _grid_y(ax)

    ax = axes[1]
    rng = np.random.default_rng(SEED)
    jit = rng.uniform(-0.15, 0.15, len(strain))
    ax.scatter(1 + jit, strain.pearson, s=2, color=PLOT_PALETTE[0], linewidths=0)
    ax.boxplot(
        [strain.pearson.dropna().to_numpy()],
        positions=[1],
        widths=0.5,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        boxprops={"linewidth": 0.5},
        whiskerprops={"linewidth": 0.5},
        capprops={"linewidth": 0.5},
    )
    ax.set_xticks([1])
    ax.set_xticklabels([f"culture 1 vs 2\n(n={strain.pearson.notna().sum()} mutants)"])
    ax.set_ylabel("Per-strain Pearson r across genes")
    ax.set_title("Kemmeren 2014, own cultures")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(-0.2, 1.0)
    _grid_y(ax)

    ax = axes[2]
    r = gene.test_retest_r.dropna().to_numpy()
    ax.hist(
        r,
        bins=np.linspace(-1, 1, 41),
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.3,
    )
    c = np.sqrt(np.clip(r, 0, 1)).mean()
    ax.axvline(
        np.median(r), color="black", linewidth=0.8, label=f"median r {np.median(r):.2f}"
    )
    ax.axvline(
        c,
        color=PLOT_PALETTE[1],
        linewidth=0.8,
        linestyle="--",
        label=f"ceiling {c:.2f}",
    )
    ax.set_xlabel("Per-gene Pearson r across paired mutants")
    ax.set_ylabel("Genes")
    ax.set_title(f"Kemmeren 2014, {len(r)} genes")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    paths["ceilings"] = _save(fig, image_dir, "within_dataset_ceilings")

    seg = pd.read_csv(osp.join(out_dir, "segregant_records.csv"))
    dev_a = pd.read_csv(osp.join(out_dir, "ap_pooled_x_a_de_novo_deviation.csv"))
    dev_n = pd.read_csv(osp.join(out_dir, "ap_pooled_x_nguessan_deviation.csv"))
    best_a = pd.read_csv(
        osp.join(out_dir, "ap_pooled_x_a_de_novo_best_concordance.csv")
    )
    best_n = pd.read_csv(osp.join(out_dir, "ap_pooled_x_nguessan_best_concordance.csv"))
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(50))
    )
    ax = axes[0]
    order = ["Ap pooled", "nguessan", "A de novo", "B de novo", "3004 de novo"]
    order = [p for p in order if p in set(seg.panel)]
    _boxes(
        ax,
        [np.log10(seg[seg.panel == p].total_umis.to_numpy()) for p in order],
        [PLOT_PALETTE[i] for i in range(len(order))],
    )
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(
        [p.replace("nguessan", "NGuessan").replace(" ", "\n", 1) for p in order]
    )
    ax.set_ylabel("log10 UMIs per record")
    ax.set_title("Depth per record")
    ax = axes[1]
    bins = np.linspace(0.4, 1.0, 31)
    ax.hist(
        best_a.pooled_best_concordance,
        bins=bins,
        histtype="step",
        color=PLOT_PALETTE[0],
        linewidth=1.0,
        label="vs A de novo cells",
    )
    ax.hist(
        best_n.pooled_best_concordance,
        bins=bins,
        histtype="step",
        color=PLOT_PALETTE[2],
        linewidth=1.0,
        label="vs NGuessan segregants",
    )
    ax.axvline(
        0.95, color="black", linewidth=0.5, linestyle="--", label="identity cut 0.95"
    )
    ax.set_xlabel("Best genotype concordance of a pooled segregant")
    ax.set_ylabel("Pooled segregants (of 393)")
    ax.set_title("Is any segregant in two panels?")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[2]
    _boxes(
        ax,
        [
            dev_a.nearest_r.dropna().to_numpy(),
            dev_a.random_r.dropna().to_numpy(),
            dev_n.nearest_r.dropna().to_numpy(),
            dev_n.random_r.dropna().to_numpy(),
        ],
        [PLOT_PALETTE[0], PLOT_PALETTE[5], PLOT_PALETTE[2], PLOT_PALETTE[5]],
    )
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(
        [
            f"A cells,\nnearest {K_NEAREST}",
            f"A cells,\nrandom {K_NEAREST}",
            "NGuessan,\nnearest 1",
            "NGuessan,\nrandom 1",
        ]
    )
    ax.set_ylabel("Pearson r of deviation profiles\n(pooled segregant vs matched)")
    ax.set_title("Genotype-driven deviation across panels")
    ax.axhline(0, color="black", linewidth=0.5)
    _grid_y(ax)
    fig.tight_layout()
    paths["segregants"] = _save(fig, image_dir, "within_dataset_segregants")
    return paths


# ---------------------------------------------------------------------------- main


def main() -> None:
    """Run the requested stages and merge into the result json."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--stage",
        choices=["jackson", "kemmeren", "hu", "segregants", "figures", "all"],
        default="all",
    )
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    results = experiment_results_dir(EXPERIMENT, __file__)
    out_dir = osp.join(results, "within_dataset_ceilings")
    os.makedirs(out_dir, exist_ok=True)
    dst = osp.join(results, "within_dataset_ceilings.json")
    result: dict[str, Any] = {}
    if osp.exists(dst):
        with open(dst) as f:
            result = json.load(f)
    stages = (
        ["jackson", "kemmeren", "hu", "segregants", "figures"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        t0 = time.time()
        if stage == "jackson":
            result["jackson2020"] = jackson(data_root, out_dir)
        elif stage == "kemmeren":
            result["kemmeren2014"] = kemmeren(data_root, out_dir)
        elif stage == "hu":
            result["hu2007"] = hu(data_root, results)
        elif stage == "segregants":
            result["segregants"] = segregants(data_root, out_dir)
        elif stage == "figures":
            result["figures"] = figures(
                out_dir, osp.join(os.environ["ASSET_IMAGES_DIR"], EXPERIMENT)
            )
        result.setdefault("runs", {})[stage] = {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "seconds": round(time.time() - t0, 1),
        }
        result["generated_by"] = (
            f"experiments/{EXPERIMENT}/scripts/within_dataset_ceilings.py"
        )
        with open(dst, "w") as f:
            json.dump(result, f, indent=1, default=str)
        print(f"stage {stage} done in {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
