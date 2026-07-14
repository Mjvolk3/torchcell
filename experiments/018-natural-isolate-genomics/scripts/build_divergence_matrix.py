# experiments/018-natural-isolate-genomics/scripts/build_divergence_matrix.py
# [[experiments.018-natural-isolate-genomics.scripts.build_divergence_matrix]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/build_divergence_matrix

"""Per-ORF nucleotide divergence of the 1,011 natural isolates vs S288C R64.

Consumes Peter 2018 ``allReferenceGenesWithSNPsAndIndelsInferred.tar.gz`` (6,015
per-gene FASTAs, each holding one sequence per isolate, SNPs+indels inferred onto
the S288C reference gene span) and diffs every isolate allele against the S288C
R64 genomic span served by ``SCerevisiaeGenome`` -- the exact reference ORF set
torchcell uses in modeling (``genome.gene_set``, 6,607 ORFs).

Divergence weighting follows Peter 2018's own published convention. Their
``filesDescription.txt`` defines the distance matrix as: "for each pair of
strains, the value is the percentage, based on SNPs, of non-identical bases.
Heterozygous differences were half-weighted compared to the homozygous
differences." We generalize that to arbitrary IUPAC ambiguity codes with

    w(code, ref) = 1 - [ref in alleles(code)] / |alleles(code)|

which returns 0/1 for an unambiguous base and exactly 1/2 for a two-allele
heterozygous site whose reference base is one of the two alleles -- i.e. it
reduces to Peter's half-weighting, and correctly charges a full difference to a
het site at which *neither* allele matches the reference.

Codon-level statistics (synonymous / non-synonymous / premature stop / codon
usage) are computed on INTRONLESS genes only, where the genomic span is the CDS.
Intron-containing genes are still scored for nucleotide divergence, but excluded
from codon analysis and counted in the manifest. Peter's gene set is entirely
nuclear (no Q0* mitochondrial genes), so the standard genetic code applies.
"""

import json
import multiprocessing as mp
import os
import os.path as osp
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

EXP_DIR = osp.join(EXPERIMENT_ROOT, "018-natural-isolate-genomics")
RESULTS_DIR = osp.join(EXP_DIR, "results")

# Peter 2018 gene FASTAs live compressed in the library mirror; we extract once to
# a documented DATA_ROOT location (8.6 GB) and reuse it on subsequent runs.
PETER_TARBALL = osp.join(
    DATA_ROOT,
    "torchcell-library/peterGenomeEvolution10112018/data",
    "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz",
)
PETER_TARBALL_SHA256 = (
    "b5400b89499fe84b1feada51abd7742c29838ae1f28c0cbd208b6622ca533f25"
)
GENES_DIR = os.environ.get(
    "PETER_GENES_DIR",
    osp.join(DATA_ROOT, "data/peter2018/reference_genes_with_snps_indels"),
)

N_WORKERS = int(os.environ.get("N_WORKERS", "64"))
# Smoke-test hook: score only the first N genes.
N_GENES_LIMIT = int(os.environ.get("N_GENES_LIMIT", "0"))


def _ensure_genes_dir() -> None:
    """Extract the pinned Peter tarball once, verifying its sha256 first."""
    import hashlib
    import tarfile

    if osp.isdir(GENES_DIR) and len(os.listdir(GENES_DIR)) >= 6000:
        return
    h = hashlib.sha256()
    with open(PETER_TARBALL, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != PETER_TARBALL_SHA256:
        raise RuntimeError(
            f"sha256 mismatch for {PETER_TARBALL}: got {h.hexdigest()}, "
            f"expected {PETER_TARBALL_SHA256}"
        )
    os.makedirs(GENES_DIR, exist_ok=True)
    print(f"      extracting {PETER_TARBALL} -> {GENES_DIR} ...", flush=True)
    with tarfile.open(PETER_TARBALL, "r:gz") as tf:
        tf.extractall(GENES_DIR, filter="data")


# --------------------------------------------------------------------------
# IUPAC weighting table (see module docstring)
# --------------------------------------------------------------------------
BASES = "ACGT"
IUPAC: dict[str, str] = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "S": "CG",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
}
# N and '-' are NO-CALLS, not "all four alleles": they are excluded from the
# denominator rather than charged as a difference.

# contains[code_byte, ref_base_idx] -> ref base is one of the code's alleles
LUT_CONTAINS = np.zeros((256, 4), dtype=np.float32)
# n_alleles[code_byte] -> ploidy-of-call; 0 marks a no-call (N, -, anything else)
LUT_NALLELES = np.zeros(256, dtype=np.float32)
# unambiguous[code_byte] -> code is a single definite base (for codon analysis)
LUT_UNAMBIG = np.zeros(256, dtype=bool)
# base_idx[code_byte] -> 0..3 for ACGT, else 0 (only read where unambiguous)
LUT_BASEIDX = np.zeros(256, dtype=np.int8)

for code, alleles in IUPAC.items():
    b = ord(code)
    LUT_NALLELES[b] = len(alleles)
    for a in alleles:
        LUT_CONTAINS[b, BASES.index(a)] = 1.0
    if len(alleles) == 1:
        LUT_UNAMBIG[b] = True
        LUT_BASEIDX[b] = BASES.index(alleles)

# Standard genetic code, indexed by 16*b0 + 4*b1 + b2 over ACGT
CODON_TABLE = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"
AA = np.frombuffer(CODON_TABLE.encode(), dtype=np.uint8)  # (64,)
STOP = ord("*")


@dataclass(frozen=True)
class GeneSpec:
    """Reference metadata for one ORF."""

    gene: str
    ref_seq: str
    intronless: bool
    orf_classification: str
    chromosome: int


def _read_fasta(path: str) -> tuple[list[str], list[str]]:
    heads: list[str] = []
    seqs: list[str] = []
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                    buf = []
                heads.append(line[1:].rstrip("\n"))
            else:
                buf.append(line.strip())
    if buf:
        seqs.append("".join(buf))
    return heads, seqs


def _strain_of(header: str) -> str:
    """Recover the 3-letter isolate code from a Peter gene-FASTA header.

    Headers come in two forms -- ``AEE_YAL001C_TFC3`` and ``SACE_YAU_YAL001C_TFC3``
    -- but both name a distinct isolate (all 1,011 codes are unique once the
    optional ``SACE_`` prefix is stripped).
    """
    tok = header.split("\t")[0]
    if tok.startswith("SACE_"):
        tok = tok[5:]
    return tok.split("_")[0]


def _score_gene(spec: GeneSpec) -> tuple[pd.DataFrame, np.ndarray, list[str]] | None:
    """Diff every isolate allele of one ORF against S288C.

    Returns (per-isolate rows, per-strain codon counts (n, 64), strain order).
    """
    path = osp.join(GENES_DIR, f"{spec.gene}.fasta")
    heads, seqs = _read_fasta(path)
    if not heads:
        return None

    strains = [_strain_of(h) for h in heads]
    ref = spec.ref_seq
    L = len(ref)

    ref_arr = np.frombuffer(ref.encode(), dtype=np.uint8)
    ref_ok = LUT_UNAMBIG[ref_arr]  # reference positions that are definite ACGT
    ref_idx = LUT_BASEIDX[ref_arr].astype(np.int64)

    # Split isolates by whether an indel changed the span length. Equal-length
    # alleles get an exact position-wise diff; length-changed alleles are flagged
    # and reported, never silently Hamming-compared against a shifted frame.
    eq_i = [i for i, s in enumerate(seqs) if len(s) == L]
    ne_i = [i for i, s in enumerate(seqs) if len(s) != L]

    rows: list[dict] = []
    codon_counts = np.zeros((len(seqs), 64), dtype=np.int64)

    if eq_i:
        packed = "".join(seqs[i] for i in eq_i).encode()
        arr = np.frombuffer(packed, dtype=np.uint8).reshape(len(eq_i), L)

        nall = LUT_NALLELES[arr]  # (n, L); 0 at no-calls
        valid = (nall > 0) & ref_ok[None, :]
        contains = LUT_CONTAINS[arr, ref_idx[None, :]]  # (n, L)
        # w = 1 - [ref in alleles(code)] / |alleles(code)|
        w = np.where(valid, 1.0 - contains / np.maximum(nall, 1.0), 0.0)

        n_valid = valid.sum(axis=1)
        w_diff = w.sum(axis=1, dtype=np.float64)
        # exact (unambiguous, differing) vs heterozygous contributions
        unamb = LUT_UNAMBIG[arr] & valid
        n_hom_diff = (unamb & (w > 0)).sum(axis=1)
        n_het = (valid & ~LUT_UNAMBIG[arr]).sum(axis=1)

        n_codon_diff = np.zeros(len(eq_i), dtype=np.int64)
        n_syn = np.zeros(len(eq_i), dtype=np.int64)
        n_nonsyn = np.zeros(len(eq_i), dtype=np.int64)
        n_stop = np.zeros(len(eq_i), dtype=np.int64)

        if spec.intronless and L % 3 == 0:
            nc = L // 3
            iso_b = LUT_BASEIDX[arr].astype(np.int64).reshape(len(eq_i), nc, 3)
            iso_clean = LUT_UNAMBIG[arr].reshape(len(eq_i), nc, 3).all(axis=2)
            ref_b = ref_idx.reshape(nc, 3)
            ref_clean = ref_ok.reshape(nc, 3).all(axis=1)

            iso_cod = iso_b[:, :, 0] * 16 + iso_b[:, :, 1] * 4 + iso_b[:, :, 2]
            ref_cod = ref_b[:, 0] * 16 + ref_b[:, 1] * 4 + ref_b[:, 2]

            clean = iso_clean & ref_clean[None, :]
            iso_aa = AA[iso_cod]
            ref_aa = AA[ref_cod]

            changed = clean & (iso_cod != ref_cod[None, :])
            same_aa = iso_aa == ref_aa[None, :]
            n_codon_diff = changed.sum(axis=1)
            n_syn = (changed & same_aa).sum(axis=1)
            n_nonsyn = (changed & ~same_aa).sum(axis=1)

            # premature stop: an internal codon that becomes a stop in the isolate
            internal = np.ones(nc, dtype=bool)
            internal[-1] = False
            gained_stop = (
                clean & (iso_aa == STOP) & (ref_aa[None, :] != STOP) & internal[None, :]
            )
            n_stop = gained_stop.sum(axis=1)

            # codon usage over clean isolate codons
            for k in range(len(eq_i)):
                sel = iso_cod[k][iso_clean[k]]
                codon_counts[eq_i[k]] = np.bincount(sel, minlength=64)

        for k, i in enumerate(eq_i):
            rows.append(
                {
                    "gene": spec.gene,
                    "strain": strains[i],
                    "len_iso": L,
                    "is_indel": False,
                    "n_valid": int(n_valid[k]),
                    "n_hom_diff": int(n_hom_diff[k]),
                    "n_het": int(n_het[k]),
                    "w_diff": float(w_diff[k]),
                    "n_codon_diff": int(n_codon_diff[k]),
                    "n_syn": int(n_syn[k]),
                    "n_nonsyn": int(n_nonsyn[k]),
                    "n_premature_stop": int(n_stop[k]),
                }
            )

    for i in ne_i:
        rows.append(
            {
                "gene": spec.gene,
                "strain": strains[i],
                "len_iso": len(seqs[i]),
                "is_indel": True,
                "n_valid": 0,
                "n_hom_diff": 0,
                "n_het": 0,
                "w_diff": float("nan"),
                "n_codon_diff": 0,
                "n_syn": 0,
                "n_nonsyn": 0,
                "n_premature_stop": 0,
            }
        )

    df = pd.DataFrame(rows)
    df["divergence"] = df["w_diff"] / df["n_valid"].replace(0, np.nan)
    return df, codon_counts, strains


def _worker(spec: GeneSpec):
    try:
        return _score_gene(spec)
    except Exception as exc:  # surface the gene, never swallow
        print(f"[FAIL] {spec.gene}: {exc}", file=sys.stderr, flush=True)
        raise


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _ensure_genes_dir()

    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    print("[1/5] loading S288C R64 reference genome ...", flush=True)
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    ref_gene_set = sorted(genome.gene_set)
    print(f"      torchcell reference ORF set: {len(ref_gene_set)}", flush=True)

    available = {
        f[: -len(".fasta")] for f in os.listdir(GENES_DIR) if f.endswith(".fasta")
    }
    shared = sorted(set(ref_gene_set) & available)
    peter_only = sorted(available - set(ref_gene_set))
    ref_only = sorted(set(ref_gene_set) - available)
    print(
        f"      Peter gene FASTAs: {len(available)} | shared with reference set: "
        f"{len(shared)} | reference-only: {len(ref_only)} | Peter-only: "
        f"{len(peter_only)}",
        flush=True,
    )

    print("[2/5] building gene specs (reference span + intron status) ...", flush=True)
    specs: list[GeneSpec] = []
    n_intron = 0
    for g in shared:
        gene = genome[g]
        n_cds = len(list(genome.db.children(g, featuretype="CDS")))
        intronless = n_cds == 1
        if not intronless:
            n_intron += 1
        specs.append(
            GeneSpec(
                gene=g,
                ref_seq=str(gene.seq),
                intronless=intronless,
                orf_classification=str(gene.orf_classification[0])
                if gene.orf_classification
                else "Unknown",
                chromosome=int(gene.chromosome),
            )
        )
    print(
        f"      {len(specs)} genes | intronless: {len(specs) - n_intron} | "
        f"intron-containing (excluded from codon stats): {n_intron}",
        flush=True,
    )
    if N_GENES_LIMIT:
        specs = specs[:N_GENES_LIMIT]
        print(f"      SMOKE TEST: limited to {len(specs)} genes", flush=True)

    print(f"[3/5] scoring divergence on {N_WORKERS} workers ...", flush=True)
    frames: list[pd.DataFrame] = []
    codon_total: dict[str, np.ndarray] = {}
    with mp.Pool(N_WORKERS) as pool:
        for k, out in enumerate(pool.imap_unordered(_worker, specs, chunksize=8)):
            if out is None:
                continue
            df, cc, strains = out
            frames.append(df)
            for i, s in enumerate(strains):
                if s not in codon_total:
                    codon_total[s] = np.zeros(64, dtype=np.int64)
                codon_total[s] += cc[i]
            if (k + 1) % 500 == 0:
                print(f"      {k + 1}/{len(specs)} genes", flush=True)

    print("[4/5] assembling tables ...", flush=True)
    div = pd.concat(frames, ignore_index=True)
    div["gene"] = div["gene"].astype("category")
    div["strain"] = div["strain"].astype("category")
    print(f"      per (gene, isolate) rows: {len(div):,}", flush=True)

    meta = pd.DataFrame(
        [
            {
                "gene": s.gene,
                "intronless": s.intronless,
                "orf_classification": s.orf_classification,
                "chromosome": s.chromosome,
                "ref_len": len(s.ref_seq),
            }
            for s in specs
        ]
    )

    ok = div[~div["is_indel"]]
    per_gene = (
        ok.groupby("gene", observed=True)
        .agg(
            n_isolates=("strain", "size"),
            mean_divergence=("divergence", "mean"),
            median_divergence=("divergence", "median"),
            max_divergence=("divergence", "max"),
            std_divergence=("divergence", "std"),
            total_syn=("n_syn", "sum"),
            total_nonsyn=("n_nonsyn", "sum"),
            n_isolates_with_premature_stop=(
                "n_premature_stop",
                lambda x: int((x > 0).sum()),
            ),
            mean_n_het=("n_het", "mean"),
        )
        .reset_index()
        .merge(meta, on="gene", how="left")
    )
    per_gene["pn_ps"] = per_gene["total_nonsyn"] / per_gene["total_syn"].replace(
        0, np.nan
    )

    per_strain = (
        ok.groupby("strain", observed=True)
        .agg(
            n_genes=("gene", "size"),
            mean_divergence=("divergence", "mean"),
            median_divergence=("divergence", "median"),
            total_w_diff=("w_diff", "sum"),
            total_valid=("n_valid", "sum"),
            total_syn=("n_syn", "sum"),
            total_nonsyn=("n_nonsyn", "sum"),
            total_het=("n_het", "sum"),
            n_genes_with_premature_stop=(
                "n_premature_stop",
                lambda x: int((x > 0).sum()),
            ),
        )
        .reset_index()
    )
    per_strain["genome_wide_divergence"] = (
        per_strain["total_w_diff"] / per_strain["total_valid"]
    )
    n_indel = (
        div.groupby("strain", observed=True)["is_indel"].sum().rename("n_indel_genes")
    )
    per_strain = per_strain.merge(n_indel, on="strain", how="left")

    codon_df = pd.DataFrame(
        {
            "strain": list(codon_total),
            **{
                f"codon_{i}": [codon_total[s][i] for s in codon_total]
                for i in range(64)
            },
        }
    )

    ref_codons = np.zeros(64, dtype=np.int64)
    for s in specs:
        if not s.intronless or len(s.ref_seq) % 3:
            continue
        b = LUT_BASEIDX[np.frombuffer(s.ref_seq.encode(), np.uint8)].astype(np.int64)
        u = LUT_UNAMBIG[np.frombuffer(s.ref_seq.encode(), np.uint8)]
        nc = len(s.ref_seq) // 3
        cod = b.reshape(nc, 3)
        clean = u.reshape(nc, 3).all(axis=1)
        idx = cod[:, 0] * 16 + cod[:, 1] * 4 + cod[:, 2]
        ref_codons += np.bincount(idx[clean], minlength=64)

    print("[5/5] writing results ...", flush=True)
    div.to_parquet(osp.join(RESULTS_DIR, "per_gene_isolate_divergence.parquet"))
    per_gene.to_parquet(osp.join(RESULTS_DIR, "per_gene_divergence_summary.parquet"))
    per_strain.to_parquet(
        osp.join(RESULTS_DIR, "per_strain_divergence_summary.parquet")
    )
    codon_df.to_parquet(osp.join(RESULTS_DIR, "codon_counts_per_strain.parquet"))
    np.save(osp.join(RESULTS_DIR, "reference_codon_counts.npy"), ref_codons)
    meta.to_parquet(osp.join(RESULTS_DIR, "gene_metadata.parquet"))

    manifest = {
        "source_tarball": "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz",
        "source_tarball_sha256": PETER_TARBALL_SHA256,
        "source_url": (
            "http://1002genomes.u-strasbg.fr/files/"
            "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz"
        ),
        "reference": "S288C R64-4-1_20230830 (SCerevisiaeGenome genomic span, strand-corrected)",
        "reference_orf_set_size": len(ref_gene_set),
        "peter_gene_fastas": len(available),
        "genes_scored": len(specs),
        "genes_reference_only_not_in_peter": len(ref_only),
        "genes_peter_only_not_in_reference": len(peter_only),
        "intron_containing_excluded_from_codon_stats": n_intron,
        "n_isolates": int(div["strain"].nunique()),
        "n_gene_isolate_pairs": int(len(div)),
        "n_indel_pairs_excluded_from_hamming": int(div["is_indel"].sum()),
        "het_weighting": (
            "w = 1 - [ref in alleles(code)] / |alleles(code)|; reduces to Peter 2018's "
            "published half-weighting of heterozygous differences"
        ),
        "het_weighting_source": (
            "Peter 2018 filesDescription.txt: 'for each pair of strains, the value is "
            "the percentage, based on SNPs, of non-identical bases. Heterozygous "
            "differences were half-weighted compared to the homozygous differences.'"
        ),
        "no_call_codes_excluded": ["N", "-"],
    }
    with open(osp.join(RESULTS_DIR, "divergence_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n=== SUMMARY ===")
    print(f"gene x isolate pairs        : {len(div):,}")
    print(f"isolates                    : {div['strain'].nunique()}")
    print(f"genes scored                : {len(specs):,}")
    print(f"indel pairs (len != ref)     : {int(div['is_indel'].sum()):,}")
    print(
        f"mean per-isolate divergence : "
        f"{per_strain['genome_wide_divergence'].mean() * 100:.4f}%"
    )
    print(
        f"median per-gene divergence  : "
        f"{per_gene['median_divergence'].median() * 100:.4f}%"
    )
    print(f"results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
