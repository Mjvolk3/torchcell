# experiments/018-natural-isolate-genomics/scripts/bit_accounting.py
# [[experiments.018-natural-isolate-genomics.scripts.bit_accounting]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/bit_accounting

"""L_C ledger: where do the bits in the CGT's inputs actually come from?

Issue #66 starts from an observation in the supported-datasets table: Caudal's
*perturbation* set gzips to ~95 MB, about the same as its *phenotype* (~94 MB), so a
natural isolate's genotype appears to carry as much information as its transcriptome.
This script asks whether that is real information or encoding overhead, and puts
genome-driven and perturbation-driven modalities on one ledger.

L_C(D) is a COMPRESSED LENGTH -- a computable upper bound on Kolmogorov complexity
(a gzip codelength), NOT an entropy. We reuse the paper's own compressor
(``torchcell.paper.tables.stream_gzip_signal``, zlib level 6, gzip container,
streamed over concatenated records) so every number here is directly comparable to
the ``Signal (gzip)`` column of the supported-datasets table.

For each modality we compress several encodings of the SAME underlying content:

  as_stored        -- exactly the bytes the LMDB record serializes to (this is what
                      the paper table counts). Carries JSON keys, URIs, sha256s.
  values_only      -- the measurements alone, fixed gene order, no keys. The
                      information, with schema overhead stripped.
  minimal_genotype -- the genotype in its smallest faithful encoding. For a KO that is
                      a gene index; for an isolate that is its variant list vs S288C.
  sequence_raw     -- an isolate's actual reference-ORF nucleotides (~9 Mb of ACGT).
  sequence_diff    -- the same isolate expressed as a diff against S288C.

The contrast that matters for the CGT: a Kemmeren genotype is ONE gene index out of
6,607 -- log2(6607) = 12.7 bits -- yet it produces a ~6,000-dimensional expression
response. A natural isolate's genotype is thousands of sequence edits. Same phenotype
dimensionality, genotype information differing by orders of magnitude. The ratio
phenotype-bits / genotype-bits is what tells you how much the model must *invent*
versus *read*.
"""

import json
import math
import os
import os.path as osp
import pickle
import zlib
from collections.abc import Iterator

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "018-natural-isolate-genomics")
RESULTS_DIR = osp.join(EXP_DIR, "results")

GENES_DIR = os.environ.get(
    "PETER_GENES_DIR",
    osp.join(DATA_ROOT, "data/peter2018/reference_genes_with_snps_indels"),
)

GZIP_LEVEL = 6  # identical to torchcell.paper.tables.stream_gzip_signal
REF_ORF_COUNT = 6607  # torchcell genome.gene_set


def stream_gzip(chunks: Iterator[bytes], level: int = GZIP_LEVEL) -> tuple[int, int]:
    """Bytes of the gzip stream over concatenated chunks. Mirrors the paper's Signal.

    Streaming (rather than gzipping each record alone) is what lets the compressor
    amortize a shared dictionary across records -- exactly how the supported-datasets
    table computes its numbers, and the reason repeated JSON keys are cheap.
    """
    comp = zlib.compressobj(level, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    total = 0
    n = 0
    for c in chunks:
        total += len(comp.compress(c))
        n += 1
    total += len(comp.flush())
    return n, total


def _lmdb_values(subpath: str) -> Iterator[dict]:
    import lmdb

    d = osp.join(DATA_ROOT, subpath, "processed", "lmdb")
    env = lmdb.open(d, readonly=True, lock=False, max_readers=2048)
    with env.begin() as txn:
        for _, v in txn.cursor():
            yield pickle.loads(v)
    env.close()


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str).encode()


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------
def phenotype_as_stored(rec: dict) -> bytes:
    return _json_bytes(rec["experiment"]["phenotype"])


def genotype_as_stored(rec: dict) -> bytes:
    return _json_bytes(rec["experiment"]["genotype"])


def _phenotype_values(rec: dict, field: str, genes: list[str]) -> bytes:
    d = rec["experiment"]["phenotype"][field]
    v = np.array([d.get(g, np.nan) for g in genes], dtype=np.float32)
    return v.tobytes()


def ko_minimal_genotype(rec: dict, gene_index: dict[str, int]) -> bytes:
    """A KO genotype in its smallest faithful form: the perturbed gene indices."""
    perts = rec["experiment"]["genotype"]["perturbations"]
    idx = sorted(
        gene_index[p["systematic_gene_name"]]
        for p in perts
        if p["systematic_gene_name"] in gene_index
    )
    return np.array(idx, dtype=np.uint16).tobytes()


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    print("[1/5] reference ORF set ...", flush=True)
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    ref_genes = sorted(genome.gene_set)
    gene_index = {g: i for i, g in enumerate(ref_genes)}

    rows: list[dict] = []

    def add(modality, encoding, n, nbytes, n_strains, n_genes, note=""):
        rows.append(
            {
                "modality": modality,
                "encoding": encoding,
                "n_records": n,
                "L_C_bytes": nbytes,
                "L_C_bits": nbytes * 8,
                "n_strains": n_strains,
                "n_genes": n_genes,
                "bits_per_strain": nbytes * 8 / max(n_strains, 1),
                "bits_per_strain_gene": nbytes * 8 / max(n_strains * n_genes, 1),
                "note": note,
            }
        )

    # ----------------------------------------------------------------------
    # 1. KO collections -- genotype vs phenotype
    # ----------------------------------------------------------------------
    print("[2/5] KO modalities (Kemmeren, Sameith) ...", flush=True)
    ko_sets = {
        "kemmeren2014_single_ko": "data/torchcell/microarray_kemmeren2014",
        "sameith2015_double_ko": "data/torchcell/dm_microarray_sameith2015",
    }
    for name, sub in ko_sets.items():
        recs = list(_lmdb_values(sub))
        n_strains = len(recs)
        genes = sorted(set(recs[0]["experiment"]["phenotype"]["expression_log2_ratio"]))
        n_genes = len(genes)

        n, b = stream_gzip(phenotype_as_stored(r) for r in recs)
        add(
            name,
            "phenotype_as_stored",
            n,
            b,
            n_strains,
            n_genes,
            "LMDB phenotype JSON = the paper's Signal column",
        )

        n, b = stream_gzip(
            _phenotype_values(r, "expression_log2_ratio", genes) for r in recs
        )
        add(
            name,
            "phenotype_values_only",
            n,
            b,
            n_strains,
            n_genes,
            "float32 log2 ratios, fixed gene order, no keys",
        )

        n, b = stream_gzip(genotype_as_stored(r) for r in recs)
        add(
            name,
            "genotype_as_stored",
            n,
            b,
            n_strains,
            n_genes,
            "LMDB genotype JSON (perturbation records)",
        )

        n, b = stream_gzip(ko_minimal_genotype(r, gene_index) for r in recs)
        add(
            name,
            "genotype_minimal",
            n,
            b,
            n_strains,
            n_genes,
            "perturbed gene indices as uint16",
        )

        # information-theoretic floor: choosing k genes out of the reference set
        k = len(recs[0]["experiment"]["genotype"]["perturbations"])
        floor_bits = math.log2(math.comb(REF_ORF_COUNT, k))
        add(
            name,
            "genotype_combinatorial_floor",
            n_strains,
            int(np.ceil(floor_bits * n_strains / 8)),
            n_strains,
            n_genes,
            f"log2(C({REF_ORF_COUNT},{k})) = {floor_bits:.1f} bits/strain",
        )

    # ----------------------------------------------------------------------
    # 2. Natural isolates -- Caudal phenotype + genotype
    # ----------------------------------------------------------------------
    print("[3/5] natural isolates (Caudal) ...", flush=True)
    caudal = list(_lmdb_values("data/torchcell/caudal_pantranscriptome2024"))
    n_strains = len(caudal)
    cgenes = sorted(
        set(caudal[0]["reference"]["phenotype_reference"]["expression_tpm"])
    )
    n_genes = len(cgenes)

    n, b = stream_gzip(phenotype_as_stored(r) for r in caudal)
    add(
        "caudal2024_natural_isolate",
        "phenotype_as_stored",
        n,
        b,
        n_strains,
        n_genes,
        "LMDB phenotype JSON = the paper's Signal column (94 MB)",
    )

    n, b = stream_gzip(_phenotype_values(r, "expression_tpm", cgenes) for r in caudal)
    add(
        "caudal2024_natural_isolate",
        "phenotype_values_only",
        n,
        b,
        n_strains,
        n_genes,
        "float32 TPM, fixed gene order, no keys",
    )

    n, b = stream_gzip(genotype_as_stored(r) for r in caudal)
    add(
        "caudal2024_natural_isolate",
        "genotype_as_stored",
        n,
        b,
        n_strains,
        n_genes,
        "LMDB genotype JSON = the ~95 MB perturbation figure from issue #66",
    )

    # ----------------------------------------------------------------------
    # 3. Natural isolates -- the actual SEQUENCE, raw vs diffed against S288C
    # ----------------------------------------------------------------------
    print("[4/5] isolate sequence: raw vs diff-vs-S288C ...", flush=True)
    div = pd.read_parquet(osp.join(RESULTS_DIR, "per_gene_isolate_divergence.parquet"))
    strains = sorted(div["strain"].astype(str).unique())
    n_iso = len(strains)

    # Reconstruct each isolate's concatenated reference-ORF sequence from the Peter
    # gene FASTAs. Genes are visited in a fixed order so the encoding is canonical.
    genes_avail = sorted(
        f[: -len(".fasta")] for f in os.listdir(GENES_DIR) if f.endswith(".fasta")
    )

    def _read_gene(g: str) -> dict[str, str]:
        out: dict[str, str] = {}
        name = None
        buf: list[str] = []
        with open(osp.join(GENES_DIR, f"{g}.fasta")) as fh:
            for line in fh:
                if line.startswith(">"):
                    if name:
                        out[name] = "".join(buf)
                    tok = line[1:].split("\t")[0]
                    if tok.startswith("SACE_"):
                        tok = tok[5:]
                    name = tok.split("_")[0]
                    buf = []
                else:
                    buf.append(line.strip())
        if name:
            out[name] = "".join(buf)
        return out

    # Build per-isolate sequence and per-isolate diff, one gene at a time to bound RAM.
    seq_parts: dict[str, list[str]] = {s: [] for s in strains}
    diff_parts: dict[str, list[str]] = {s: [] for s in strains}
    for gi, g in enumerate(genes_avail):
        alleles = _read_gene(g)
        try:
            ref = str(genome[g].seq)
        except Exception:
            continue
        for s in strains:
            a = alleles.get(s)
            if a is None:
                continue
            seq_parts[s].append(a)
            if len(a) == len(ref):
                d = [f"{gi}:{i}:{a[i]}" for i in range(len(a)) if a[i] != ref[i]]
                if d:
                    diff_parts[s].append(",".join(d))
            else:
                # length-changed allele: store it verbatim (indels are not a
                # substitution list); this is the honest cost of an indel
                diff_parts[s].append(f"{gi}:INDEL:{a}")
        if (gi + 1) % 1000 == 0:
            print(f"      {gi + 1}/{len(genes_avail)} genes", flush=True)

    n, b = stream_gzip(("".join(seq_parts[s])).encode() for s in strains)
    add(
        "caudal2024_natural_isolate",
        "sequence_raw_nt",
        n,
        b,
        n_iso,
        len(genes_avail),
        "concatenated reference-ORF nucleotides per isolate (~9 Mb ACGT each)",
    )

    n, b = stream_gzip((";".join(diff_parts[s])).encode() for s in strains)
    add(
        "caudal2024_natural_isolate",
        "sequence_diff_vs_S288C",
        n,
        b,
        n_iso,
        len(genes_avail),
        "isolate encoded as a variant list against S288C (conditional complexity)",
    )

    # 2-bit packing floor for the raw sequence (no compression, just the alphabet)
    total_nt = sum(len(x) for x in seq_parts[strains[0]])
    add(
        "caudal2024_natural_isolate",
        "sequence_2bit_floor",
        n_iso,
        int(np.ceil(total_nt * 2 / 8)) * n_iso,
        n_iso,
        len(genes_avail),
        f"{total_nt:,} nt/isolate at 2 bits/base, uncompressed",
    )

    print("[5/5] writing ledger ...", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(osp.join(RESULTS_DIR, "bit_ledger.parquet"))
    df.to_csv(osp.join(RESULTS_DIR, "bit_ledger.csv"), index=False)

    # Headline ratios
    def get(mod, enc, col="L_C_bits"):
        r = df[(df.modality == mod) & (df.encoding == enc)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    ratios = {}
    for mod in df.modality.unique():
        ph = get(mod, "phenotype_values_only")
        gt_stored = get(mod, "genotype_as_stored")
        gt_min = get(mod, "genotype_minimal")
        if math.isnan(gt_min):
            gt_min = get(mod, "sequence_diff_vs_S288C")
        ratios[mod] = {
            "phenotype_bits_values_only": ph,
            "genotype_bits_as_stored": gt_stored,
            "genotype_bits_minimal": gt_min,
            "phenotype_per_genotype_minimal": ph / gt_min if gt_min else None,
            "genotype_schema_overhead_frac": (
                1 - gt_min / gt_stored if gt_stored else None
            ),
        }
    with open(osp.join(RESULTS_DIR, "bit_ledger_ratios.json"), "w") as fh:
        json.dump(ratios, fh, indent=2)

    pd.set_option("display.width", 200)
    print("\n=== L_C LEDGER (gzip level 6, streamed) ===")
    show = df[
        ["modality", "encoding", "L_C_bytes", "bits_per_strain", "bits_per_strain_gene"]
    ].copy()
    show["L_C_MB"] = (show["L_C_bytes"] / 1e6).round(2)
    print(
        show[
            [
                "modality",
                "encoding",
                "L_C_MB",
                "bits_per_strain",
                "bits_per_strain_gene",
            ]
        ].to_string(index=False)
    )
    print(f"\nresults -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
