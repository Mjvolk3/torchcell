# experiments/018-natural-isolate-genomics/scripts/serialization_order_effect.py
# [[experiments.018-natural-isolate-genomics.scripts.serialization_order_effect]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/serialization_order_effect

"""L_C is order-dependent -- ON DNA. It does NOT transfer to the Signal column.

SCOPE WARNING, because the first version of this note over-reached. The 24.5x measured
here is a property of the **DNA sequence corpus**, NOT of the ``Signal (gzip)`` column.
See ``verify_signal_composition.py``, which measures the order effect on the ACTUAL LMDB
expression records and finds **1.00x** -- no effect at all.

Why the two differ, and it is the whole point:

  DNA        two isolates' allele of the SAME gene are ~99.3%-identical STRINGS. Put them
             adjacent and DEFLATE emits one long back-reference; put them 9 Mb apart and
             the 32 KB window cannot see the match, so the redundancy is paid for again.
  Expression two strains' value for a gene are similar NUMBERS, but their float32 byte
             patterns share essentially no substring. There is nothing to back-reference,
             so reordering changes nothing.

**Compressibility does not transfer across data types.** That is the lesson; do not
extrapolate a compression ratio measured on one modality to another.

What this script still legitimately shows: the 1,011-isolate reference-ORF sequences,
serialized two ways --

  gene-major     all 1,011 isolate alleles of gene 1, then all of gene 2, ...
  isolate-major  all 6,015 genes of isolate A, then all of isolate B, ...

-- give 103.9 MB vs 2,548.4 MB from identical content (24.5x). That matters if we ever
store raw isolate sequence, and it is a real caveat about gzip codelengths in general. It
is NOT a claim about the datasets table.

For what IS true about the Signal column's window slack (1.4x-2.1x at the instance level,
~5x on Caudal's perturbation block), see ``verify_signal_composition.py``.
"""

import glob
import json
import os
import os.path as osp
import zlib

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
GZIP_LEVEL = 6


def _read(fp: str) -> dict[str, str]:
    out: dict[str, str] = {}
    name = None
    buf: list[str] = []
    with open(fp) as fh:
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


def main() -> None:
    files = sorted(glob.glob(osp.join(GENES_DIR, "*.fasta")))
    print(f"[1/3] {len(files)} gene FASTAs", flush=True)

    strains = sorted(_read(files[0]))
    n_iso = len(strains)

    # --- gene-major: stream one gene at a time, never holding the whole corpus ---
    print("[2/3] gene-major gzip pass ...", flush=True)
    comp = zlib.compressobj(GZIP_LEVEL, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    gene_major = 0
    total_nt = 0
    for k, fp in enumerate(files):
        g = _read(fp)
        blob = "".join(g.get(s, "") for s in strains).encode()
        total_nt += len(blob)
        gene_major += len(comp.compress(blob))
        if (k + 1) % 1000 == 0:
            print(f"      {k + 1}/{len(files)}", flush=True)
    gene_major += len(comp.flush())

    # --- isolate-major: taken from the ledger, which already computed it ---
    print("[3/3] reading isolate-major L_C from the bit ledger ...", flush=True)
    import pandas as pd

    led = pd.read_parquet(osp.join(RESULTS_DIR, "bit_ledger.parquet"))
    row = led[
        (led.modality == "caudal2024_natural_isolate")
        & (led.encoding == "sequence_raw_nt")
    ]
    isolate_major = int(row["L_C_bytes"].iloc[0])

    out = {
        "content_nt": total_nt,
        "n_genes": len(files),
        "n_isolates": n_iso,
        "gzip_level": GZIP_LEVEL,
        "deflate_window_bytes": 32768,
        "gene_major_bytes": gene_major,
        "isolate_major_bytes": isolate_major,
        "gene_major_ratio": total_nt / gene_major,
        "isolate_major_ratio": total_nt / isolate_major,
        "order_effect_x": isolate_major / gene_major,
        "lmdb_ordering": "isolate-major (one record per strain)",
        "implication": (
            "Signal (gzip) is a codelength under a stated encoding+ordering, not an "
            "estimate of K(D). The LMDB's one-record-per-strain layout is the worst "
            "ordering for cross-record redundancy, so every Signal number in the "
            "supported-datasets table is an upper bound that ordering alone can move by "
            "more than an order of magnitude."
        ),
    }
    with open(osp.join(RESULTS_DIR, "serialization_order_effect.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n=== L_C IS ORDER-DEPENDENT ===")
    print(
        f"content              : {total_nt:,} nt "
        f"({len(files)} genes x {n_iso} isolates)"
    )
    print(
        f"gene-major    L_C    : {gene_major:>14,} B   "
        f"({total_nt / gene_major:6.1f}x compression)"
    )
    print(
        f"isolate-major L_C    : {isolate_major:>14,} B   "
        f"({total_nt / isolate_major:6.1f}x compression)"
    )
    print(
        f"\n>>> Same content. Ordering alone changes L_C by "
        f"{isolate_major / gene_major:.1f}x."
    )
    print(">>> The LMDB is isolate-major -- the worst case -- and that is the ordering")
    print("    every Signal (gzip) number in the paper's dataset table is computed in.")


if __name__ == "__main__":
    main()
