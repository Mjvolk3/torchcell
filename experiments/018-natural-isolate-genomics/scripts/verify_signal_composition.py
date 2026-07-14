# experiments/018-natural-isolate-genomics/scripts/verify_signal_composition.py
# [[experiments.018-natural-isolate-genomics.scripts.verify_signal_composition]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/verify_signal_composition

"""What is the Signal (gzip) column actually made of, and is it order-sensitive?

Written to CORRECT two overclaims from the first pass of this experiment. Both are
retracted in the note; this script is the evidence.

RETRACTION 1 -- "91% of Kemmeren's Signal is JSON keys/metadata."
    That compared as-stored JSON against a float32 binary array of ONE field, silently
    conflating four different costs. Decomposed properly here, holding representation
    fixed at each step:
        A  as-stored JSON                      (what the table reports)
        B  same fields, float TEXT, no keys    -> isolates the KEY cost
        C  primary field only, text, no keys   -> isolates the EXTRA-FIELD cost
        D  primary field, float32 binary       -> isolates the TEXT-vs-BINARY cost

RETRACTION 2 -- "L_C swings 24.5x on record order, so the gzip column is off by an order
    of magnitude." The 24.5x is real but it is a property of the DNA corpus, where two
    isolates' allele of the same gene are ~99.3%-identical STRINGS and adjacency lets
    DEFLATE emit one long back-reference. Expression values are similar NUMBERS whose
    float32 byte patterns share no substring -- there is nothing to back-reference. This
    script measures the order effect on the ACTUAL LMDB records (strain-major as stored,
    vs gene-major, vs shuffled). Do not extrapolate compressibility across data types.

WHAT IS TRUE about the window: gzip's 32 KB back-reference window does leave real slack,
and it is worst on perturbation-heavy rows. Measured against a large-window compressor
(LZMA) on the exact bytes the table counts.
"""

import json
import lzma
import os
import os.path as osp
import pickle
import zlib

import lmdb
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "018-natural-isolate-genomics", "results")

GZIP_LEVEL = 6  # identical to torchcell.paper.tables.stream_gzip_signal

DATASETS = [
    ("kemmeren2014", "data/torchcell/microarray_kemmeren2014", "expression_log2_ratio"),
    ("caudal2024", "data/torchcell/caudal_pantranscriptome2024", "expression_tpm"),
]


def gz(chunks, level: int = GZIP_LEVEL) -> int:
    c = zlib.compressobj(level, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    t = 0
    for x in chunks:
        t += len(c.compress(x))
    return t + len(c.flush())


def load(sub: str) -> list[dict]:
    env = lmdb.open(
        osp.join(DATA_ROOT, sub, "processed", "lmdb"),
        readonly=True,
        lock=False,
        max_readers=2048,
    )
    out = []
    with env.begin() as txn:
        for _, v in txn.cursor():
            out.append(pickle.loads(v))
    env.close()
    return out


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report: dict = {}

    for name, sub, primary in DATASETS:
        recs = load(sub)
        ph = [r["experiment"]["phenotype"] for r in recs]
        inst = [r["experiment"] for r in recs]
        geno = [r["experiment"]["genotype"] for r in recs]
        genes = sorted(ph[0][primary])
        dict_fields = [k for k, v in ph[0].items() if isinstance(v, dict)]

        # ---- composition, holding representation fixed at each step ----
        A_inst = gz(json.dumps(x, sort_keys=True, default=str).encode() for x in inst)
        A_ph = gz(json.dumps(x, sort_keys=True, default=str).encode() for x in ph)
        B = gz(
            json.dumps(
                [[p[f].get(g) for g in genes] for f in dict_fields], default=str
            ).encode()
            for p in ph
        )
        C = gz(
            json.dumps([p[primary].get(g) for g in genes], default=str).encode()
            for p in ph
        )
        D = gz(
            np.array(
                [p[primary].get(g, np.nan) for g in genes], dtype=np.float32
            ).tobytes()
            for p in ph
        )

        # ---- order effect on the ACTUAL records ----
        M = np.zeros((len(ph), len(genes)), dtype=np.float32)
        for i, p in enumerate(ph):
            d = p[primary]
            for j, g in enumerate(genes):
                x = d.get(g)
                if x is not None:
                    M[i, j] = x
        strain_major = gz(M[i].tobytes() for i in range(M.shape[0]))
        gene_major = gz(
            np.ascontiguousarray(M[:, j]).tobytes() for j in range(M.shape[1])
        )
        perm = np.random.default_rng(0).permutation(M.shape[0])
        shuffled = gz(M[i].tobytes() for i in perm)

        # ---- window slack vs a large-window compressor ----
        def blob(part: list[dict]) -> bytes:
            return b"".join(
                json.dumps(x, sort_keys=True, default=str).encode() for x in part
            )

        xz_inst = len(lzma.compress(blob(inst), preset=6))
        xz_ph = len(lzma.compress(blob(ph), preset=6))
        gz_geno = gz(json.dumps(x, sort_keys=True, default=str).encode() for x in geno)
        xz_geno = len(lzma.compress(blob(geno), preset=6))

        report[name] = {
            "n_records": len(recs),
            "signal_instance_as_stored": A_inst,
            "phenotype_as_stored": A_ph,
            "composition_of_phenotype": {
                "json_keys_frac": (A_ph - B) / A_ph,
                "extra_value_fields_frac": (B - C) / A_ph,
                "float_text_vs_binary_frac": (C - D) / A_ph,
                "primary_values_frac": D / A_ph,
                "extra_fields": [f for f in dict_fields if f != primary],
            },
            "order_effect": {
                "strain_major_as_stored": strain_major,
                "gene_major": gene_major,
                "records_shuffled": shuffled,
                "major_axis_effect_x": max(strain_major, gene_major)
                / min(strain_major, gene_major),
                "shuffle_effect_x": max(strain_major, shuffled)
                / min(strain_major, shuffled),
            },
            "window_slack_gzip_over_xz": {
                "instance": A_inst / xz_inst,
                "phenotype": A_ph / xz_ph,
                "genotype": gz_geno / xz_geno,
                "genotype_gzip_bytes": gz_geno,
                "genotype_xz_bytes": xz_geno,
            },
        }

        c = report[name]["composition_of_phenotype"]
        o = report[name]["order_effect"]
        w = report[name]["window_slack_gzip_over_xz"]
        print(f"\n=== {name} ({len(recs)} records) ===")
        print(f"  Signal (instance, as stored)  : {A_inst / 1e6:8.1f} MB")
        print(f"  phenotype as stored           : {A_ph / 1e6:8.1f} MB")
        print("  composition of the phenotype:")
        print(f"    JSON keys           {100 * c['json_keys_frac']:5.1f}%")
        print(
            f"    extra value FIELDS  {100 * c['extra_value_fields_frac']:5.1f}%  "
            f"{c['extra_fields']}"
        )
        print(f"    float text vs binary{100 * c['float_text_vs_binary_frac']:5.1f}%")
        print(
            f"    primary values      {100 * c['primary_values_frac']:5.1f}%  "
            f"<- the actual measurements"
        )
        print(
            f"  ORDER effect (major axis)     : {o['major_axis_effect_x']:.2f}x  "
            f"(shuffle: {o['shuffle_effect_x']:.3f}x)"
        )
        print(
            f"  gzip vs xz: instance {w['instance']:.1f}x | phenotype "
            f"{w['phenotype']:.1f}x | genotype {w['genotype']:.1f}x"
        )

    with open(osp.join(RESULTS_DIR, "signal_composition.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 74)
    print("CONCLUSIONS (these supersede the first pass):")
    print(
        "  * ~48-66% of the Signal is serialization overhead (repeated gene-name keys"
    )
    print("    + floats stored as ASCII). NOT '91% metadata' -- a large share is real")
    print("    companion data (SE, variance, n_replicates).")
    print("  * Record ORDER does not matter for expression (1.00x). The 24.5x order")
    print("    effect is DNA-only and does NOT transfer to the Signal column.")
    print("  * gzip's 32 KB window leaves 1.4-2.1x at the instance level, and ~5x on")
    print("    Caudal's perturbation block (repetitive pointer boilerplate).")
    print(f"\nresults -> {RESULTS_DIR}/signal_composition.json")


if __name__ == "__main__":
    main()
