# experiments/010-kuzmin-tmi/scripts/inference_4_generate_triples.py
# [[experiments.010-kuzmin-tmi.scripts.inference_4_generate_triples]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_4_generate_triples
#
# Build the inference_4 triple space as an LMDB the InferenceDataset can read.
#
# THE SPACE. From the roster written by inference_4_gene_selection.py, keep a triple
# when it carries 1 to 2 metabolic genes AND 1 to 2 regulators, and when at least
# --min-supported of its genes were measured under at least --min-screens distinct
# Kuzmin query screens. Enumeration walks only the class multisets that can satisfy
# those rules, so the 135 million unconstrained triples are never materialized.
#
# WHY THE SCREEN GATE AND NOT PRESENCE. 841 of the 934 roster genes appear somewhere in
# the Kuzmin trigenic data, so requiring presence removes 0.08 percent of the space and
# changes nothing. 191 clear 50 distinct screens. That is the gate that separates a gene
# the data constrains from one seen in a single query, and single-screen genes are what
# produced inference_1's positive tail.
#
# RECORD CONSTRUCTION. InferenceDataset stores one JSON list per key holding an
# experiment and its reference. Calling pydantic per record would dominate the runtime
# at this scale, so the record is built ONCE through
# InferenceDataset.create_experiment_from_triple, validated there, and then reused as a
# template whose only per-triple edits are the three gene names. The template is dumped
# to the summary so the substitution is auditable.
#
# The stricter support tiers are SUBSETS of the looser ones, so a run at
# --min-supported 1 also answers 2 and 3 by filtering the index afterwards. Nothing
# needs re-running to tighten the gate.
#
# Run from repo root:
#   ~/miniconda3/envs/torchcell/bin/python \
#     experiments/010-kuzmin-tmi/scripts/inference_4_generate_triples.py --dry-run
#   ... then without --dry-run, or via the SLURM wrapper.
#
# Outputs:
#   $DATA_ROOT/data/torchcell/experiments/010-kuzmin-tmi/inference_4/processed/lmdb
#   $DATA_ROOT/.../inference_4/triple_index.parquet   idx -> genes + strata
#   results/inference_4/generation_summary.json

import argparse
import json
import os
import os.path as osp
import sys
import time
from itertools import combinations

import lmdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results", "inference_4")
OUT_ROOT = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_4")

WRITE_CHUNK = 200_000     # records per LMDB transaction
INDEX_CHUNK = 1_000_000   # rows per parquet row group


def load_roster(min_screens: int) -> pd.DataFrame:
    df = pd.read_csv(osp.join(RESULTS_DIR, "gene_candidates.csv"))
    df = df[df["keep"]].copy()
    df["supported"] = df["distinct_screens"] >= min_screens
    df["role"] = np.where(
        df["is_metabolic"] & df["is_regulator"], "B",
        np.where(df["is_metabolic"], "M", "R"),
    )
    return df.sort_values("gene").reset_index(drop=True)


def valid_class_multisets(min_supported: int):
    """Which (class, class, class) multisets satisfy composition and support."""
    from itertools import combinations_with_replacement

    names = ["MH", "ML", "RH", "RL", "BH", "BL"]
    out = []
    for combo in combinations_with_replacement(names, 3):
        n_met = sum(c[0] in "MB" for c in combo)
        n_reg = sum(c[0] in "RB" for c in combo)
        n_sup = sum(c[1] == "H" for c in combo)
        if 1 <= n_met <= 2 and 1 <= n_reg <= 2 and n_sup >= min_supported:
            out.append(combo)
    return out


def iter_triples(by_class: dict[str, list[str]], multisets):
    """Yield (gene1, gene2, gene3) for every valid class multiset, without repeats.

    A multiset with a repeated class draws a combination from that class so a gene is
    never paired with itself, and distinct classes are disjoint by construction, so no
    triple is emitted twice.
    """
    for combo in multisets:
        counts: dict[str, int] = {}
        for c in combo:
            counts[c] = counts.get(c, 0) + 1
        blocks = []
        for cname, k in counts.items():
            genes = by_class.get(cname, [])
            if len(genes) < k:
                blocks = None
                break
            blocks.append(list(combinations(genes, k)) if k > 1 else [(g,) for g in genes])
        if blocks is None:
            continue
        if len(blocks) == 1:
            for a in blocks[0]:
                yield a
        elif len(blocks) == 2:
            for a in blocks[0]:
                for b in blocks[1]:
                    yield a + b
        else:
            for a in blocks[0]:
                for b in blocks[1]:
                    for c in blocks[2]:
                        yield a + b + c


def build_template():
    """One pydantic-validated record, used as the substitution template."""
    from inference_dataset_1 import InferenceDataset

    exp = InferenceDataset.create_experiment_from_triple(
        ("YAL001C", "YAL002W", "YAL003W"), dataset_name="inference_4"
    )
    ref = InferenceDataset._create_default_reference(InferenceDataset, exp)
    rec = [{"experiment": exp.model_dump(), "experiment_reference": ref.model_dump()}]
    perts = rec[0]["experiment"]["genotype"]["perturbations"]
    assert len(perts) == 3, f"template has {len(perts)} perturbations"
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-screens", type=int, default=50)
    ap.add_argument("--min-supported", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--dry-run", action="store_true",
                    help="count the space and print one record, write nothing")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    roster = load_roster(args.min_screens)
    by_class: dict[str, list[str]] = {}
    for _, r in roster.iterrows():
        by_class.setdefault(r["role"] + ("H" if r["supported"] else "L"), []).append(r["gene"])
    print("class sizes:", {k: len(v) for k, v in sorted(by_class.items())})

    multisets = valid_class_multisets(args.min_supported)
    print(f"valid class multisets: {len(multisets)}")

    template = build_template()
    if args.dry_run:
        print("\ntemplate record (gene names are placeholders):")
        print(json.dumps(template, indent=2)[:2200])
        n = sum(1 for _ in iter_triples(by_class, multisets))
        print(f"\ntriples that would be written: {n:,}")
        print(f"estimated LMDB size at 3.9 kB/record: {n * 3.9 / 1e6:.1f} GB")
        return

    os.makedirs(OUT_ROOT, exist_ok=True)
    processed = osp.join(OUT_ROOT, "processed")
    os.makedirs(processed, exist_ok=True)
    lmdb_path = osp.join(processed, "lmdb")
    if osp.exists(lmdb_path):
        raise SystemExit(
            f"{lmdb_path} already exists. Move it aside before regenerating; this "
            "script never overwrites an existing inference space."
        )

    supported = set(roster.loc[roster["supported"], "gene"])
    metabolic = set(roster.loc[roster["is_metabolic"], "gene"])
    regulator = set(roster.loc[roster["is_regulator"], "gene"])

    env = lmdb.open(lmdb_path, map_size=int(4e12), readonly=False)
    index_schema = pa.schema([
        ("index", pa.int64()), ("gene1", pa.string()), ("gene2", pa.string()),
        ("gene3", pa.string()), ("n_metabolic", pa.int8()),
        ("n_regulator", pa.int8()), ("n_supported", pa.int8()),
    ])
    index_writer = pq.ParquetWriter(
        osp.join(OUT_ROOT, "triple_index.parquet"), index_schema, compression="zstd"
    )

    t0 = time.time()
    written = 0
    buf_rows: list[tuple] = []
    txn = env.begin(write=True)
    perts = template[0]["experiment"]["genotype"]["perturbations"]

    for triple in tqdm(iter_triples(by_class, multisets), desc="writing", unit="rec"):
        g1, g2, g3 = triple
        for p, g in zip(perts, triple):
            p["systematic_gene_name"] = g
            p["perturbed_gene_name"] = g
            # create_experiment_from_triple writes strain_id as f"{gene}_deletion";
            # the template substitution has to reproduce that shape, not just the name.
            p["strain_id"] = f"{g}_deletion"
        txn.put(str(written).encode("utf-8"),
                json.dumps(template).encode("utf-8"))
        buf_rows.append((
            written, g1, g2, g3,
            sum(g in metabolic for g in triple),
            sum(g in regulator for g in triple),
            sum(g in supported for g in triple),
        ))
        written += 1

        if written % WRITE_CHUNK == 0:
            txn.commit()
            txn = env.begin(write=True)
        if len(buf_rows) >= INDEX_CHUNK:
            index_writer.write_table(
                pa.Table.from_arrays(
                    [pa.array([r[i] for r in buf_rows], type=index_schema.field(i).type)
                     for i in range(7)],
                    schema=index_schema,
                )
            )
            buf_rows = []

    txn.commit()
    if buf_rows:
        index_writer.write_table(
            pa.Table.from_arrays(
                [pa.array([r[i] for r in buf_rows], type=index_schema.field(i).type)
                 for i in range(7)],
                schema=index_schema,
            )
        )
    index_writer.close()
    env.sync()
    env.close()

    elapsed = time.time() - t0
    size_gb = sum(
        osp.getsize(osp.join(lmdb_path, f)) for f in os.listdir(lmdb_path)
    ) / 1e9
    summary = {
        "min_screens": args.min_screens,
        "min_supported": args.min_supported,
        "n_roster_genes": int(len(roster)),
        "class_sizes": {k: len(v) for k, v in sorted(by_class.items())},
        "n_valid_class_multisets": len(multisets),
        "n_triples": written,
        "lmdb_path": lmdb_path,
        "lmdb_size_gb": round(size_gb, 2),
        "index_path": osp.join(OUT_ROOT, "triple_index.parquet"),
        "elapsed_s": round(elapsed, 1),
        "records_per_s": round(written / elapsed, 1),
        "template_record": template,
    }
    with open(osp.join(RESULTS_DIR, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {written:,} records, {size_gb:.1f} GB, in {elapsed / 60:.1f} min "
          f"({written / elapsed:,.0f} rec/s)")
    print(f"lmdb  {lmdb_path}")
    print(f"index {osp.join(OUT_ROOT, 'triple_index.parquet')}")


if __name__ == "__main__":
    main()
