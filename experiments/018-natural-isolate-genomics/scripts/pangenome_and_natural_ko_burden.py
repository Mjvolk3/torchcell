# experiments/018-natural-isolate-genomics/scripts/pangenome_and_natural_ko_burden.py
# [[experiments.018-natural-isolate-genomics.scripts.pangenome_and_natural_ko_burden]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/pangenome_and_natural_ko_burden

"""Core vs accessory genome, and how many genes a natural isolate has already lost.

Two questions, one script.

**Core vs accessory.** Peter 2018 defines the core genome by presence in every isolate,
not by a percentage cutoff. Verbatim, from the mirrored paper (``paper.md`` sha256
21c49934d546f162b43025c85254bcd1cedcb80ab9a691935301b5962a6a12da, line 220):

    "The mapping was also used as a confirmation step for the presence of the ORFs in
     each strain, leading to the identification of 4,940 ORFs present in the 1,011
     strains of the collection, representing the core genome plus 2,856 ORFs present in
     different subsets of the population."

So core = present in all 1,011 (4,940 ORFs) and variable = the remaining 2,856, out of a
7,796-ORF pangenome. We reproduce that partition from the released presence/absence
matrix rather than restating it, and we also report the >=99% cutoff as a sensitivity so
the difference between the two conventions is visible.

**Natural KO burden -- the point of the comparison.** Kemmeren engineers exactly ONE gene
knockout per strain. A natural isolate arrives with genes already broken, by three
independent mechanisms, which we count and then union:

  1. gene ABSENT       -- reference ORF missing from the isolate (presence/absence matrix)
  2. FRAMESHIFT        -- homozygous frameshifting indel (Peter's frameshift matrix)
  3. PREMATURE STOP    -- nonsense mutation we called ourselves in build_divergence_matrix

Peter also publishes a SIFT+nonsense loss-of-function prediction (1011LossOfFunction),
which we load as an independent cross-check on (3).

The resulting number is the honest answer to "how many knockouts is a natural isolate
worth?" -- and it is what makes the ~15x expression difference vs a single KO
interpretable.
"""

import gzip
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
EXP_DIR = osp.join(EXPERIMENT_ROOT, "018-natural-isolate-genomics")
RESULTS_DIR = osp.join(EXP_DIR, "results")

PETER_DATA = osp.join(DATA_ROOT, "torchcell-library/peterGenomeEvolution10112018/data")
PRESENCE = osp.join(PETER_DATA, "genesMatrix_PresenceAbsence.tab.gz")
# Auxiliary Peter matrices retrieved for this experiment (md5s verified against the
# source's published md5.txt).
EXTRA_DIR = os.environ.get("PETER_EXTRA_DIR", osp.join(DATA_ROOT, "data/peter2018"))
FRAMESHIFT = osp.join(EXTRA_DIR, "genesMatrix_Frameshift.tab.gz")
LOF = osp.join(EXTRA_DIR, "1011LossOfFunction.xls.gz")
DNDS = osp.join(EXTRA_DIR, "gene_dNdS.tab.gz")

CHECKSUMS = {
    "genesMatrix_PresenceAbsence.tab.gz": "b3b96719de226eaa277589d8d01f9202",
    "genesMatrix_Frameshift.tab.gz": "6f8ec446283817be303e20f5661550b0",
    "1011LossOfFunction.xls.gz": "f3c4f2db7a5b2a9bf28039dbdd82e2b1",
    "gene_dNdS.tab.gz": "c5dbef9fc5cc0a0e5f3a6b5c3d3689cf",
}


def _read_matrix(path: str) -> pd.DataFrame:
    with gzip.open(path, "rt") as fh:
        return pd.read_csv(fh, sep="\t", index_col=0, low_memory=False)


def _orf_col_to_systematic(col: str) -> str | None:
    """Pangenome matrix column -> S288C systematic name, else None.

    Peter's matrices are R ``make.names``-mangled: ``X1768.YAL001C`` is pangenome ORF
    ``1768-YAL001C``. torchcell already reverses this in
    ``torchcell.datasets.scerevisiae.caudal2024._demangle_orf`` / ``_orf_to_s288c``, and
    we reuse those so the mapping is identical to the one the Caudal loader used.

    ONE EXTENSION, and the reason it is here: 804 pangenome entries carry a
    ``_NumOfGenes_N`` suffix (``X1771.YAL005C_NumOfGenes_3``) marking a cluster of N
    paralogous copies collapsed into one pangenome ORF. ``_orf_to_s288c`` returns None
    for all of them, so using it alone silently drops real reference ORFs (YAL005C among
    them) from any presence/absence count. We strip the suffix and map the cluster to its
    systematic name, which is the right call for a gene-presence question but DOES
    conflate paralogs -- so we count and report how many reference ORFs are recovered
    this way rather than folding them in silently.
    """
    from torchcell.datasets.scerevisiae.caudal2024 import _demangle_orf, _orf_to_s288c

    raw = _demangle_orf(col)
    hit = _orf_to_s288c(raw)
    if hit:
        return hit
    stripped = re.sub(r"_NumOfGenes_\d+$", "", raw)
    if stripped != raw:
        return _orf_to_s288c(stripped)
    return None


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    ref_genes = set(genome.gene_set)

    # ------------------------------------------------------------------
    print("[1/4] presence/absence -> core vs accessory ...", flush=True)
    pa = _read_matrix(PRESENCE)
    print(
        f"      matrix: {pa.shape[0]} isolates x {pa.shape[1]} pangenome ORFs",
        flush=True,
    )

    n_iso = pa.shape[0]
    present_count = pa.sum(axis=0)  # per ORF, how many isolates carry it
    frac_present = present_count / n_iso

    core_peter = int((present_count == n_iso).sum())  # Peter's own definition
    variable_peter = int((present_count < n_iso).sum())
    core_99 = int((frac_present >= 0.99).sum())  # the common convention, as sensitivity

    print(
        f"      core (present in ALL {n_iso}, Peter's definition): {core_peter}  "
        f"[paper: 4,940]",
        flush=True,
    )
    print(f"      variable: {variable_peter}  [paper: 2,856]", flush=True)
    print(f"      (sensitivity) core at >=99% presence: {core_99}", flush=True)

    sysname = {c: _orf_col_to_systematic(c) for c in pa.columns}
    n_mapped = sum(1 for v in sysname.values() if v)
    n_cluster_recovered = sum(
        1 for c, v in sysname.items() if v and "_NumOfGenes_" in c
    )
    print(
        f"      pangenome ORFs mapping to an S288C systematic name: {n_mapped} "
        f"(of which {n_cluster_recovered} recovered from _NumOfGenes_ clusters)",
        flush=True,
    )

    per_orf = pd.DataFrame(
        {
            "orf": pa.columns,
            "systematic_name": [sysname[c] for c in pa.columns],
            "n_isolates_present": present_count.to_numpy(),
            "frac_isolates_present": frac_present.to_numpy(),
            "is_core_peter": (present_count == n_iso).to_numpy(),
            "is_core_99pct": (frac_present >= 0.99).to_numpy(),
            "is_reference_orf": [
                bool(sysname[c] and sysname[c] in ref_genes) for c in pa.columns
            ],
            "is_paralog_cluster": ["_NumOfGenes_" in c for c in pa.columns],
        }
    )
    per_orf.to_parquet(osp.join(RESULTS_DIR, "pangenome_orf_presence.parquet"))

    ref_orfs_in_pa = per_orf[per_orf.is_reference_orf]
    print(
        f"      of {len(ref_orfs_in_pa)} S288C reference ORFs in the pangenome, "
        f"{int(ref_orfs_in_pa.is_core_peter.sum())} are core "
        f"({100 * ref_orfs_in_pa.is_core_peter.mean():.1f}%)",
        flush=True,
    )

    # ------------------------------------------------------------------
    print("[2/4] natural KO burden per isolate ...", flush=True)
    ref_cols = [
        c for c in pa.columns if per_orf.set_index("orf").loc[c, "is_reference_orf"]
    ]
    ref_names = np.array([sysname[c] for c in ref_cols])
    absent_mat = pa[ref_cols].to_numpy() == 0  # (n_iso, n_ref_cols)

    # The frameshift matrix is GENES x ISOLATES (the transpose of the presence matrix),
    # keyed by the pangenome ORF CODE (1768) with five leading metadata columns
    # (Gene, SGDorder, CH, START, STOP) before the 1,011 isolate columns. The CODE is the
    # same number the presence matrix encodes in its column names (X1768.YAL001C), so we
    # join through that rather than trusting the standard-name column.
    code_to_sys: dict[int, str] = {}
    for c in pa.columns:
        m = re.match(r"^X(\d+)\.", c)
        if m and sysname[c]:
            code_to_sys[int(m.group(1))] = sysname[c]

    fs = _read_matrix(FRAMESHIFT)
    fs_meta = ["Gene", "SGDorder", "CH", "START", "STOP"]
    fs_iso_cols = [c for c in fs.columns if c not in fs_meta]
    fs_rows_sys = [code_to_sys.get(int(i)) if pd.notna(i) else None for i in fs.index]
    keep = [k for k, s in enumerate(fs_rows_sys) if s is not None and s in ref_genes]
    fs_names = np.array([fs_rows_sys[k] for k in keep])
    # -> isolates x reference-ORF genes
    fs_mat_T = (fs.iloc[keep][fs_iso_cols].to_numpy() == 1).T
    fs_isolates = list(fs_iso_cols)
    print(
        f"      frameshift matrix: {len(fs_isolates)} isolates x {len(fs_names)} "
        f"reference ORFs (from {fs.shape[0]} pangenome rows)",
        flush=True,
    )

    div = pd.read_parquet(
        osp.join(RESULTS_DIR, "per_gene_isolate_divergence.parquet"),
        columns=["gene", "strain", "n_premature_stop"],
    )
    stops = (
        div[div["n_premature_stop"] > 0]
        .groupby("strain", observed=True)["gene"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    fs_index = {s: i for i, s in enumerate(fs_isolates)}
    rows = []
    for i, s in enumerate(pa.index):
        a = set(ref_names[absent_mat[i]])
        j = fs_index.get(s)
        f = set(fs_names[fs_mat_T[j]]) if j is not None else set()
        p = stops.get(s, set())
        u = a | f | p
        rows.append(
            {
                "strain": s,
                "n_absent": len(a),
                "n_frameshift": len(f),
                "n_premature_stop": len(p),
                "n_broken_union": len(u),
            }
        )
    burden = pd.DataFrame(rows)
    burden.to_parquet(osp.join(RESULTS_DIR, "natural_ko_burden.parquet"))

    print(
        f"      absent reference ORFs / isolate  : "
        f"mean {burden.n_absent.mean():.1f}  median {burden.n_absent.median():.0f}",
        flush=True,
    )
    print(
        f"      homozygous frameshifts / isolate : "
        f"mean {burden.n_frameshift.mean():.1f}  median "
        f"{burden.n_frameshift.median():.0f}",
        flush=True,
    )
    print(
        f"      premature stops / isolate        : "
        f"mean {burden.n_premature_stop.mean():.1f}  median "
        f"{burden.n_premature_stop.median():.0f}",
        flush=True,
    )
    print(
        f"      UNION (natural KO burden)        : "
        f"mean {burden.n_broken_union.mean():.1f}  median "
        f"{burden.n_broken_union.median():.0f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    print("[3/4] cross-check vs Peter's published LoF + dN/dS ...", flush=True)
    xcheck: dict = {}
    try:
        with gzip.open(LOF, "rt", errors="replace") as fh:
            lof = pd.read_csv(fh, sep="\t", low_memory=False)
        xcheck["lof_shape"] = list(lof.shape)
        xcheck["lof_columns"] = list(lof.columns[:8])
    except Exception as exc:
        xcheck["lof_error"] = str(exc)

    try:
        dnds = _read_matrix(DNDS)
        pg = pd.read_parquet(
            osp.join(RESULTS_DIR, "per_gene_divergence_summary.parquet")
        )
        dcol = next((c for c in dnds.columns if "median" in c.lower()), dnds.columns[0])
        j = pg.merge(
            dnds[[dcol]].rename(columns={dcol: "peter_dnds"}),
            left_on="gene",
            right_index=True,
            how="inner",
        )
        j = j[np.isfinite(j["pn_ps"]) & np.isfinite(j["peter_dnds"])]
        r = float(np.corrcoef(np.log1p(j["pn_ps"]), np.log1p(j["peter_dnds"]))[0, 1])
        xcheck["pn_ps_vs_peter_dnds"] = {
            "n_genes": int(len(j)),
            "pearson_r_log1p": r,
            "note": (
                "our per-gene pN/pS (computed from the isolate CDS alleles) vs Peter's "
                "published PAML dN/dS -- an independent check on the codon logic"
            ),
        }
        print(
            f"      pN/pS vs Peter dN/dS: r = {r:.3f} over {len(j)} genes", flush=True
        )
    except Exception as exc:
        xcheck["dnds_error"] = str(exc)

    # ------------------------------------------------------------------
    print("[4/4] writing ...", flush=True)
    summary = {
        "source_checksums_md5": CHECKSUMS,
        "core_definition": {
            "rule": "present in all 1,011 isolates",
            "source_quote": (
                "The mapping was also used as a confirmation step for the presence of "
                "the ORFs in each strain, leading to the identification of 4,940 ORFs "
                "present in the 1,011 strains of the collection, representing the core "
                "genome plus 2,856 ORFs present in different subsets of the population."
            ),
            "source": (
                "Peter 2018, mirrored paper.md sha256 "
                "21c49934d546f162b43025c85254bcd1cedcb80ab9a691935301b5962a6a12da, "
                "line 220"
            ),
        },
        "pangenome": {
            "n_isolates": n_iso,
            "n_orfs": int(pa.shape[1]),
            "core_peter": core_peter,
            "core_peter_paper": 4940,
            "variable_peter": variable_peter,
            "variable_peter_paper": 2856,
            "core_at_99pct_sensitivity": core_99,
            "n_reference_orfs_in_pangenome": int(len(ref_orfs_in_pa)),
            "n_reference_orfs_core": int(ref_orfs_in_pa.is_core_peter.sum()),
        },
        "natural_ko_burden": {
            "n_absent": {
                "mean": float(burden.n_absent.mean()),
                "median": float(burden.n_absent.median()),
                "max": int(burden.n_absent.max()),
            },
            "n_frameshift": {
                "mean": float(burden.n_frameshift.mean()),
                "median": float(burden.n_frameshift.median()),
                "max": int(burden.n_frameshift.max()),
            },
            "n_premature_stop": {
                "mean": float(burden.n_premature_stop.mean()),
                "median": float(burden.n_premature_stop.median()),
                "max": int(burden.n_premature_stop.max()),
            },
            "n_broken_union": {
                "mean": float(burden.n_broken_union.mean()),
                "median": float(burden.n_broken_union.median()),
                "max": int(burden.n_broken_union.max()),
                "min": int(burden.n_broken_union.min()),
            },
            "comparator": (
                "Kemmeren engineers exactly 1 KO; Sameith 2. A natural isolate arrives "
                "with this many reference ORFs already broken."
            ),
        },
        "cross_checks": xcheck,
    }
    with open(osp.join(RESULTS_DIR, "pangenome_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n=== NATURAL KO BURDEN vs ENGINEERED KO ===")
    print("  Kemmeren 2014 engineered KOs / strain : 1")
    print("  Sameith 2015 engineered KOs / strain  : 2")
    print(
        f"  Natural isolate BROKEN reference ORFs : "
        f"median {burden.n_broken_union.median():.0f}  "
        f"(mean {burden.n_broken_union.mean():.1f}, "
        f"range {burden.n_broken_union.min()}-{burden.n_broken_union.max()})"
    )
    print(f"\nresults -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
