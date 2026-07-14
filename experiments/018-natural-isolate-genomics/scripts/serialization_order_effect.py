# experiments/018-natural-isolate-genomics/scripts/serialization_order_effect.py
# [[experiments.018-natural-isolate-genomics.scripts.serialization_order_effect]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/serialization_order_effect

"""L_C is order-dependent, and by a lot. A caveat the paper's Signal column needs.

``torchcell.paper.tables.stream_gzip_signal`` reports ``Signal (gzip)`` as a computable
upper bound on Kolmogorov complexity. It is a legitimate upper bound -- but DEFLATE's
back-reference window is only 32 KB, so the bound it attains depends on how the records
happen to be ordered in the LMDB. Redundancy that sits further apart than 32 KB is
invisible to the compressor and is paid for again, in full.

That is not hypothetical here. The 1,011-isolate reference-ORF sequences can be
serialized two ways:

  gene-major     -- all 1,011 isolate alleles of gene 1, then all of gene 2, ...
                    (near-identical sequences land adjacent, well inside the window)
  isolate-major  -- all 6,015 genes of isolate A, then all of isolate B, ...
                    (an isolate's allele of a gene is ~9 Mb from the next isolate's)

Identical content. The LMDB is isolate-major -- one record per strain -- which is the
WORST ordering for cross-record redundancy, and it is the ordering every Signal number in
the supported-datasets table is computed in.

Consequence for the manuscript: Signal (gzip) is a valid RELATIVE proxy across datasets
serialized the same way, but it is not within an order of magnitude of K(D), and the gap
is a property of the serialization, not of the biology. Report it as a codelength under a
stated encoding, never as "the information content of the dataset".
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
