# experiments/018-natural-isolate-genomics/scripts/expression_variability_analysis.py
# [[experiments.018-natural-isolate-genomics.scripts.expression_variability_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/expression_variability_analysis

"""Cross-modality expression variability: engineered KOs vs natural isolates.

The question this answers is the one issue #66 is really about: *where do the bits
come from* in the inputs the Cell Graph Transformer consumes. Genome-driven variation
(natural isolates) and perturbation-driven variation (deletion collections) both
produce a ~6,000-dimensional expression vector, but they arrive at it from wildly
different amounts of genotype information.

To compare them we need one currency. All four datasets are reduced to a **log2 ratio
against their own reference**:

* Kemmeren 2014 (1,484 single KOs) and Sameith 2015 (82 single / 72 double KOs) store
  ``expression_log2_ratio`` = log2(mutant / wild-type ref-pool). Already differential.
* Caudal 2024 (943 natural isolates) stores absolute ``expression_tpm`` plus a
  population-mean reference, so we form log2((TPM_iso + 1) / (TPM_pop + 1)).

Both are then "how far is this strain's transcriptome from its reference", per gene.

Gene universe is anchored on the S288C R64 reference ORF set torchcell models
(``genome.gene_set``, 6,607 ORFs), intersected with what each dataset measures.

TWO CAVEATS THAT ARE REPORTED, NOT BURIED:

1. *Sameith signs.* A deleted gene's own probe must go down. That oracle holds in
   97% of Kemmeren records but only ~84% (single) / ~72% (double) of Sameith's,
   consistent with a sign-orientation bug in the Sameith loader (it trusts the GEO
   ``VALUE`` column, whose orientation is inconsistent within GSE42536, rather than
   recomputing from the Cy5/Cy3 signal columns as Kemmeren's loader does). We verify
   the oracle here and report it. Sameith DE counts are therefore a LOWER BOUND and
   Kemmeren is the primary single-KO arm.
2. *Pseudocount.* Caudal's log2 ratio needs a floor for TPM=0. We use +1 and restrict
   to genes with population-mean TPM >= 1, and we sweep the choice to show it does not
   drive the result.
"""

import json
import os
import os.path as osp
import pickle
from dataclasses import dataclass, field

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "018-natural-isolate-genomics")
RESULTS_DIR = osp.join(EXP_DIR, "results")

# --------------------------------------------------------------------------
# Differential-expression threshold -- SOURCED, not guessed.
# See results/de_threshold_provenance.json (written by this script) and the
# dendron note. Kemmeren 2014's own criterion for a "significantly changed" gene.
# --------------------------------------------------------------------------
DE_THRESHOLD_FC = 1.7  # fold change
DE_THRESHOLD_LOG2 = float(np.log2(DE_THRESHOLD_FC))  # ~0.766
THRESHOLD_SWEEP = [0.5, DE_THRESHOLD_LOG2, 1.0, 1.5, 2.0]

CAUDAL_PSEUDOCOUNT = 1.0
CAUDAL_MIN_REF_TPM = 1.0

DATASETS = {
    "kemmeren2014_single_ko": "data/torchcell/microarray_kemmeren2014",
    "sameith2015_single_ko": "data/torchcell/sm_microarray_sameith2015",
    "sameith2015_double_ko": "data/torchcell/dm_microarray_sameith2015",
    "caudal2024_natural_isolate": "data/torchcell/caudal_pantranscriptome2024",
}


@dataclass
class Modality:
    """One dataset reduced to a strain x gene log2-ratio matrix."""

    name: str
    strains: list[str] = field(default_factory=list)
    genes: list[str] = field(default_factory=list)
    log2: np.ndarray | None = None  # (n_strain, n_gene), NaN where unmeasured
    perturbed: list[list[str]] = field(default_factory=list)


def _lmdb_records(subpath: str):
    d = osp.join(DATA_ROOT, subpath, "processed", "lmdb")
    env = lmdb.open(d, readonly=True, lock=False, max_readers=2048)
    with env.begin() as txn:
        cur = txn.cursor()
        for _, v in cur:
            yield pickle.loads(v)
    env.close()


def _load_ko_modality(name: str, subpath: str, gene_set: set[str]) -> Modality:
    """Kemmeren / Sameith: expression_log2_ratio is already differential."""
    rows: list[dict[str, float]] = []
    strains: list[str] = []
    perturbed: list[list[str]] = []
    for rec in _lmdb_records(subpath):
        exp = rec["experiment"]
        ph = exp["phenotype"]
        perts = [p["systematic_gene_name"] for p in exp["genotype"]["perturbations"]]
        rows.append(ph["expression_log2_ratio"])
        strains.append("+".join(sorted(perts)))
        perturbed.append(perts)

    measured = sorted(set().union(*(set(r) for r in rows)) & gene_set)
    idx = {g: i for i, g in enumerate(measured)}
    mat = np.full((len(rows), len(measured)), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        for g, v in r.items():
            j = idx.get(g)
            if j is not None:
                mat[i, j] = v
    return Modality(
        name=name, strains=strains, genes=measured, log2=mat, perturbed=perturbed
    )


def _load_caudal_modality(name: str, subpath: str, gene_set: set[str]) -> Modality:
    """Caudal: absolute TPM -> log2 ratio vs the population-mean reference."""
    tpm_rows: list[dict[str, float]] = []
    strains: list[str] = []
    ref: dict[str, float] | None = None
    absent: list[list[str]] = []
    for rec in _lmdb_records(subpath):
        exp = rec["experiment"]
        ph = exp["phenotype"]
        perts = exp["genotype"]["perturbations"]
        if ref is None:
            ref = rec["reference"]["phenotype_reference"]["expression_tpm"]
        tpm_rows.append(ph["expression_tpm"])
        strains.append(perts[0]["strain_id"] if perts else f"idx{len(strains)}")
        absent.append(
            [
                p["systematic_gene_name"]
                for p in perts
                if p["perturbation_type"] == "natural_gene_absence"
            ]
        )
    assert ref is not None

    # Reference ORFs that are expressed in the population mean.
    measured = sorted(g for g in ref if g in gene_set and ref[g] >= CAUDAL_MIN_REF_TPM)
    idx = {g: i for i, g in enumerate(measured)}
    refv = np.array([ref[g] for g in measured], dtype=np.float64)

    mat = np.full((len(tpm_rows), len(measured)), np.nan, dtype=np.float32)
    for i, r in enumerate(tpm_rows):
        v = np.full(len(measured), np.nan)
        for g, x in r.items():
            j = idx.get(g)
            if j is not None:
                v[j] = x
        mat[i] = np.log2((v + CAUDAL_PSEUDOCOUNT) / (refv + CAUDAL_PSEUDOCOUNT)).astype(
            np.float32
        )
    return Modality(
        name=name, strains=strains, genes=measured, log2=mat, perturbed=absent
    )


def _deleted_gene_oracle(m: Modality) -> dict:
    """A deleted gene's own probe must go DOWN. Sanity-checks sign convention."""
    gidx = {g: i for i, g in enumerate(m.genes)}
    vals: list[float] = []
    for i, perts in enumerate(m.perturbed):
        for g in perts:
            j = gidx.get(g)
            if j is not None and m.log2 is not None and not np.isnan(m.log2[i, j]):
                vals.append(float(m.log2[i, j]))
    if not vals:
        return {"n": 0}
    a = np.array(vals)
    return {
        "n": int(a.size),
        "median_log2_of_deleted_gene": float(np.median(a)),
        "frac_negative": float((a < 0).mean()),
    }


def _describe(m: Modality) -> dict:
    assert m.log2 is not None
    flat = m.log2[~np.isnan(m.log2)]
    q = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])
    de = {}
    for t in THRESHOLD_SWEEP:
        per_strain = np.nansum(np.abs(m.log2) >= t, axis=1)
        de[f"{t:.3f}"] = {
            "mean_de_genes_per_strain": float(per_strain.mean()),
            "median_de_genes_per_strain": float(np.median(per_strain)),
            "frac_of_all_values_de": float((np.abs(flat) >= t).mean()),
        }
    return {
        "n_strains": len(m.strains),
        "n_genes": len(m.genes),
        "n_values": int(flat.size),
        "mean": float(flat.mean()),
        "sd": float(flat.std()),
        "mad": float(np.median(np.abs(flat - np.median(flat)))),
        "iqr": float(q[4] - q[2]),
        "quantiles": {
            "p1": float(q[0]),
            "p5": float(q[1]),
            "p25": float(q[2]),
            "p50": float(q[3]),
            "p75": float(q[4]),
            "p95": float(q[5]),
            "p99": float(q[6]),
        },
        "de_by_threshold": de,
        "deleted_gene_oracle": _deleted_gene_oracle(m),
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    print("[1/4] loading reference ORF set ...", flush=True)
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    gene_set = set(genome.gene_set)
    print(f"      reference ORFs: {len(gene_set)}", flush=True)

    print("[2/4] loading modalities ...", flush=True)
    mods: dict[str, Modality] = {}
    for name, sub in DATASETS.items():
        if name.startswith("caudal"):
            m = _load_caudal_modality(name, sub, gene_set)
        else:
            m = _load_ko_modality(name, sub, gene_set)
        mods[name] = m
        print(
            f"      {name:32s} strains={len(m.strains):5d} genes={len(m.genes):5d}",
            flush=True,
        )

    # Shared gene universe across all four -- the fair comparison basis.
    shared = set(mods[next(iter(mods))].genes)
    for m in mods.values():
        shared &= set(m.genes)
    shared_sorted = sorted(shared)
    print(
        f"      shared reference ORFs across all modalities: {len(shared_sorted)}",
        flush=True,
    )

    print("[3/4] summarising distributions + DE counts ...", flush=True)
    summary: dict[str, dict] = {}
    per_strain_frames: list[pd.DataFrame] = []
    for name, m in mods.items():
        # restrict to the shared universe for the headline comparison
        keep = [i for i, g in enumerate(m.genes) if g in shared]
        assert m.log2 is not None
        sub = Modality(
            name=name,
            strains=m.strains,
            genes=[m.genes[i] for i in keep],
            log2=m.log2[:, keep],
            perturbed=m.perturbed,
        )
        summary[name] = _describe(sub)
        summary[name]["basis"] = "shared_reference_orfs"

        assert sub.log2 is not None
        de_counts = {
            f"n_de_{t:.3f}": np.nansum(np.abs(sub.log2) >= t, axis=1)
            for t in THRESHOLD_SWEEP
        }
        df = pd.DataFrame(
            {
                "dataset": name,
                "strain": sub.strains,
                "n_perturbed_genes": [len(p) for p in sub.perturbed],
                "mean_abs_log2": np.nanmean(np.abs(sub.log2), axis=1),
                "sd_log2": np.nanstd(sub.log2, axis=1),
                **de_counts,
            }
        )
        per_strain_frames.append(df)

        # per-gene variability across strains
        pg = pd.DataFrame(
            {
                "dataset": name,
                "gene": sub.genes,
                "sd_across_strains": np.nanstd(sub.log2, axis=0),
                "mean_abs_log2": np.nanmean(np.abs(sub.log2), axis=0),
                "n_strains_de": np.nansum(
                    np.abs(sub.log2) >= DE_THRESHOLD_LOG2, axis=0
                ),
            }
        )
        pg.to_parquet(osp.join(RESULTS_DIR, f"per_gene_variability_{name}.parquet"))

        # the raw log2 matrix, for plotting
        np.save(osp.join(RESULTS_DIR, f"log2_matrix_{name}.npy"), sub.log2)
        with open(osp.join(RESULTS_DIR, f"log2_axes_{name}.json"), "w") as fh:
            json.dump({"strains": sub.strains, "genes": sub.genes}, fh)

    per_strain = pd.concat(per_strain_frames, ignore_index=True)
    per_strain.to_parquet(
        osp.join(RESULTS_DIR, "per_strain_expression_summary.parquet")
    )

    print("[4/4] writing summary ...", flush=True)
    out = {
        "de_threshold": {
            "fold_change": DE_THRESHOLD_FC,
            "log2": DE_THRESHOLD_LOG2,
            "sweep": THRESHOLD_SWEEP,
        },
        "caudal_pseudocount": CAUDAL_PSEUDOCOUNT,
        "caudal_min_ref_tpm": CAUDAL_MIN_REF_TPM,
        "shared_reference_orfs": len(shared_sorted),
        "modalities": summary,
    }
    with open(osp.join(RESULTS_DIR, "expression_variability_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n=== EXPRESSION VARIABILITY (shared reference-ORF basis) ===")
    hdr = f"{'dataset':32s} {'strains':>7s} {'SD':>7s} {'IQR':>7s} {'DE@1.7x':>9s} {'oracle-':>8s}"
    print(hdr)
    print(f"{'':32s} {'':>7s} {'':>7s} {'':>7s} {'per strain':>9s} {'frac neg':>8s}")
    for name, s in summary.items():
        de = s["de_by_threshold"][f"{DE_THRESHOLD_LOG2:.3f}"][
            "mean_de_genes_per_strain"
        ]
        orc = s["deleted_gene_oracle"].get("frac_negative")
        orc_s = f"{orc:.2f}" if orc is not None else "  n/a"
        print(
            f"{name:32s} {s['n_strains']:7d} {s['sd']:7.3f} {s['iqr']:7.3f} "
            f"{de:9.1f} {orc_s:>8s}"
        )
    print(f"\nresults -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
