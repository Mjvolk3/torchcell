# experiments/036-dataset-fixes-before-kg-build/scripts/caudal2024_per_replicate.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.caudal2024_per_replicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/caudal2024_per_replicate
"""Prove the per-culture records of ``CaudalPanTranscriptome2024Dataset`` against the release.

Tracking issue #1178. Reads, never writes a store:

- the converted store at ``--scratch-root`` (built by the branch's loader);
- the previous store at ``--previous-root`` (built by ``origin/main``'s loader);
- the pinned Datafile 1 zip, Datafile 2 and Caudal's Supplementary Table 1.

Reports the record count per isolate and per replicate position (the k-th culture of an
isolate in store order); for each isolate's Datafile 1 record, every key path that differs
from the previous store's record (expected: only ``replicate_id`` on the 26 replicated
isolates) and the max absolute TPM and count difference; for each other culture, that its
TPM equals the Datafile 2 cells exactly under the Datafile 1 gene id; the mean over each
(genotype, environment) group against the previous value, which is ONE culture, not a mean;
how close each Datafile 2 sample sits to the stored Datafile 1 culture (the Final_set
sample is that culture); the Final_set mapped reads against Supplementary Table 1; and
that the shared reference is unchanged.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/caudal2024_per_replicate.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/caudal2024_per_replicate.py \
        --previous-root <scratch>/baseline_origin_main/caudal_pantranscriptome2024 \
        --scratch-root <scratch>/caudal_pantranscriptome2024
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import pickle
from collections import Counter
from statistics import median
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datasets.scerevisiae.caudal2024 import (
    CITATION_KEY,
    REPLICATE_TAB_REL,
    SI_TABLES_XLSX,
    r_make_names,
)
from torchcell.sequence.natural_isolate import genotype_isolate

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"


def read_store(root: str) -> list[dict[str, Any]]:
    """Every record of ``<root>/processed/lmdb`` in index order, interned refs resolved."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(root, "processed", "interned")
    if osp.isdir(interned_dir):
        ienv = lmdb.open(interned_dir, readonly=True, lock=False)
        with ienv.begin() as txn:
            interned = {k.decode(): pickle.loads(v) for k, v in txn.cursor()}
        ienv.close()
    env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
        records = [
            resolve_interned(pickle.loads(txn.get(str(i).encode())), interned)
            for i in range(n)
        ]
    env.close()
    return records


def differing_paths(old: Any, new: Any, path: str = "") -> list[str]:
    """Key paths where ``new`` differs from ``old``; a key on one side only is named."""
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[str] = []
        for key in sorted(set(old) | set(new)):
            if key not in old:
                out.append(f"{path}/{key} (new key)")
            elif key not in new:
                out.append(f"{path}/{key} (removed key)")
            else:
                out += differing_paths(old[key], new[key], f"{path}/{key}")
        return out
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [
            p
            for i, (a, b) in enumerate(zip(old, new, strict=True))
            for p in differing_paths(a, b, f"{path}[{i}]")
        ]
    return [] if old == new else [path]


def max_abs_diff(a: dict[str, float], b: dict[str, float]) -> float:
    """Max |a - b| over the shared keys; the key sets must be equal."""
    assert set(a) == set(b), "key sets differ"
    return max((abs(a[k] - b[k]) for k in a), default=0.0)


def closeness(stored: dict[str, float], sample: dict[str, float]) -> dict[str, float]:
    """Median |dTPM| and Pearson r of log2(TPM + 1) over the shared genes."""
    genes = sorted(set(stored) & set(sample))
    x = np.log2(np.array([stored[g] for g in genes]) + 1)
    y = np.log2(np.array([sample[g] for g in genes]) + 1)
    return {
        "n_genes": len(genes),
        "median_abs_dtpm": round(median(abs(stored[g] - sample[g]) for g in genes), 4),
        "pearson_log2_tpm": round(float(np.corrcoef(x, y)[0, 1]), 5),
    }


def main() -> None:
    """Compare the two stores and the release; write the results JSON."""
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-root", required=True)
    parser.add_argument("--scratch-root", required=True)
    args = parser.parse_args()
    data_root = os.environ["DATA_ROOT"]

    previous = read_store(args.previous_root)
    scratch = read_store(args.scratch_root)
    prev_by_isolate = {
        genotype_isolate(r["experiment"]["genotype"]): r for r in previous
    }
    assert len(prev_by_isolate) == len(previous)

    by_isolate: dict[str, list[dict[str, Any]]] = {}
    for record in scratch:
        isolate = genotype_isolate(record["experiment"]["genotype"])
        assert isolate is not None
        by_isolate.setdefault(isolate, []).append(record)
    positions = Counter(
        k for recs in by_isolate.values() for k in range(1, len(recs) + 1)
    )

    # The release, read here with pandas and not through the loader's functions.
    d2 = pd.read_csv(osp.join(data_root, REPLICATE_TAB_REL), dtype=str)
    d2["tpm"] = d2["tpm"].astype(float)
    samples = d2.drop_duplicates("SampleID").set_index("SampleID")
    table_s1 = pd.read_excel(
        osp.join(data_root, "torchcell-library", CITATION_KEY, SI_TABLES_XLSX),
        sheet_name="Table S1",
        dtype=str,
    ).set_index("Standardized_name")

    # 1. The Datafile 1 record of every isolate against the previous store.
    paths: Counter[str] = Counter()
    max_tpm = 0.0
    max_count = 0.0
    replicated: dict[str, str] = {}
    for isolate, recs in by_isolate.items():
        first = recs[0]
        old = prev_by_isolate[isolate]
        for p in differing_paths(old, first):
            paths[p] += 1
        new_ph, old_ph = (
            first["experiment"]["phenotype"],
            old["experiment"]["phenotype"],
        )
        max_tpm = max(
            max_tpm, max_abs_diff(old_ph["expression_tpm"], new_ph["expression_tpm"])
        )
        max_count = max(
            max_count,
            max_abs_diff(old_ph["expression_count"], new_ph["expression_count"]),
        )
        if new_ph["replicate_id"] is not None:
            replicated[isolate] = new_ph["replicate_id"]

    # 2. Every other culture against Datafile 2, cell for cell.
    culture_checks: list[dict[str, Any]] = []
    for isolate, recs in sorted(by_isolate.items()):
        stored_df1 = recs[0]["experiment"]["phenotype"]["expression_tpm"]
        for record in recs[1:]:
            ph = record["experiment"]["phenotype"]
            sample = ph["replicate_id"]
            rows = d2[d2["SampleID"] == sample]
            released = dict(
                zip(rows["ORF"].map(r_make_names), rows["tpm"], strict=True)
            )
            # A stored gene id is Datafile 1's systematic_name (or the loader's served
            # id), and R make.names of it is Datafile 1's ORF spelling, which is
            # make.names of the Datafile 2 ORF.
            exact = sum(
                int(released.get(r_make_names(gene), math.nan) == value)
                for gene, value in ph["expression_tpm"].items()
            )
            culture_checks.append(
                {
                    "isolate": isolate,
                    "sample_id": sample,
                    "final_group": samples.loc[sample, "Final.Group"],
                    "n_genes": len(ph["expression_tpm"]),
                    "n_genes_equal_to_release": exact,
                    "n_released_rows": len(rows),
                    "n_mapped_reads": ph["n_mapped_reads"],
                    "n_mapped_reads_released": int(
                        samples.loc[sample, "Number.of.clean.mapped.reads"].replace(
                            ",", ""
                        )
                    ),
                    "expression_count_gap": [g["field"] for g in ph["provenance_gaps"]],
                    "vs_datafile_1_culture": closeness(
                        stored_df1, ph["expression_tpm"]
                    ),
                }
            )

    # 3. The Final_set sample against the stored Datafile 1 culture and Table S1.
    final_set: list[dict[str, Any]] = []
    for isolate, sample in sorted(replicated.items()):
        rows = d2[d2["SampleID"] == sample]
        stored = by_isolate[isolate][0]["experiment"]["phenotype"]["expression_tpm"]
        orf_to_gene = {r_make_names(gene): gene for gene in stored}
        assert len(orf_to_gene) == len(stored)
        sample_tpm = {
            orf_to_gene[o]: t
            for o, t in zip(rows["ORF"].map(r_make_names), rows["tpm"], strict=True)
            if o in orf_to_gene
        }
        row = table_s1.loc[isolate]
        final_set.append(
            {
                "isolate": isolate,
                "sample_id": sample,
                "final_group": samples.loc[sample, "Final.Group"],
                "mapped_reads_equal_table_s1": samples.loc[
                    sample, "Number.of.clean.mapped.reads"
                ]
                == row["Mapped_clean_reads"],
                "library_equal_table_s1": samples.loc[sample, "Library"]
                == row["Library"],
                "run_equal_table_s1": samples.loc[sample, "Sequencing.run"]
                == row["Run"],
                "vs_datafile_1_culture": closeness(stored, sample_tpm),
            }
        )

    # 4. Mean over each (genotype, environment) group against the previous value.
    mean_diffs: dict[str, float] = {}
    for isolate, recs in by_isolate.items():
        genes = Counter(
            g for r in recs for g in r["experiment"]["phenotype"]["expression_tpm"]
        )
        mean = {
            g: sum(
                r["experiment"]["phenotype"]["expression_tpm"].get(g, 0.0) for r in recs
            )
            / sum(
                1 for r in recs if g in r["experiment"]["phenotype"]["expression_tpm"]
            )
            for g in genes
        }
        old = prev_by_isolate[isolate]["experiment"]["phenotype"]["expression_tpm"]
        shared = set(mean) & set(old)
        mean_diffs[isolate] = max(abs(mean[g] - old[g]) for g in shared)
    multi = {i: d for i, d in mean_diffs.items() if len(by_isolate[i]) > 1}
    single = {i: d for i, d in mean_diffs.items() if len(by_isolate[i]) == 1}

    # 5. The shared reference.
    references = {json.dumps(r["reference"], sort_keys=True) for r in scratch}
    prev_references = {json.dumps(r["reference"], sort_keys=True) for r in previous}

    out = {
        "script": "experiments/036-dataset-fixes-before-kg-build/scripts/caudal2024_per_replicate.py",
        "previous_root": args.previous_root,
        "scratch_root": args.scratch_root,
        "n_records_previous": len(previous),
        "n_records_scratch": len(scratch),
        "n_isolates": len(by_isolate),
        "isolates_by_culture_count": dict(
            sorted(Counter(len(r) for r in by_isolate.values()).items())
        ),
        "records_by_replicate_position": dict(sorted(positions.items())),
        "datafile_1_records": {
            "key_paths_differing_from_previous_store": dict(paths),
            "max_abs_diff_tpm": max_tpm,
            "max_abs_diff_count": max_count,
            "n_with_replicate_id": len(replicated),
        },
        "culture_records": {
            "n": len(culture_checks),
            "all_tpm_equal_to_release": all(
                c["n_genes"] == c["n_genes_equal_to_release"] for c in culture_checks
            ),
            "all_mapped_reads_equal_to_release": all(
                c["n_mapped_reads"] == c["n_mapped_reads_released"]
                for c in culture_checks
            ),
            "n_released_rows_total": sum(c["n_released_rows"] for c in culture_checks),
            "n_genes_stored_total": sum(c["n_genes"] for c in culture_checks),
            "vs_datafile_1_culture_pearson_range": [
                min(
                    c["vs_datafile_1_culture"]["pearson_log2_tpm"]
                    for c in culture_checks
                ),
                max(
                    c["vs_datafile_1_culture"]["pearson_log2_tpm"]
                    for c in culture_checks
                ),
            ],
            "vs_datafile_1_culture_median_abs_dtpm_range": [
                min(
                    c["vs_datafile_1_culture"]["median_abs_dtpm"]
                    for c in culture_checks
                ),
                max(
                    c["vs_datafile_1_culture"]["median_abs_dtpm"]
                    for c in culture_checks
                ),
            ],
            "per_culture": culture_checks,
        },
        "final_set_is_the_datafile_1_culture": {
            "n": len(final_set),
            "all_mapped_reads_library_run_equal_table_s1": all(
                f["mapped_reads_equal_table_s1"]
                and f["library_equal_table_s1"]
                and f["run_equal_table_s1"]
                for f in final_set
            ),
            "pearson_range": [
                min(f["vs_datafile_1_culture"]["pearson_log2_tpm"] for f in final_set),
                max(f["vs_datafile_1_culture"]["pearson_log2_tpm"] for f in final_set),
            ],
            "median_abs_dtpm_range": [
                min(f["vs_datafile_1_culture"]["median_abs_dtpm"] for f in final_set),
                max(f["vs_datafile_1_culture"]["median_abs_dtpm"] for f in final_set),
            ],
            "per_isolate": final_set,
        },
        "mean_reproduction": {
            "statistic": "max over genes of |mean over an isolate's records - previous value|",
            "single_culture_isolates_max": max(single.values()),
            "replicated_isolates_max": max(multi.values()),
            "replicated_isolates_median": median(multi.values()),
            "note": "the previous value is the Datafile 1 culture alone, not a mean, so "
            "the replicated isolates' group mean differs by design; the Datafile 1 "
            "record itself reproduces it (datafile_1_records.max_abs_diff_tpm)",
        },
        "reference": {
            "n_distinct_scratch": len(references),
            "equal_to_previous": references == prev_references,
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = osp.join(RESULTS, "caudal2024_per_replicate.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, default=str)
        handle.write("\n")
    summary = {
        k: v
        for k, v in out.items()
        if k not in ("culture_records", "final_set_is_the_datafile_1_culture")
    }
    print(json.dumps(summary, indent=2, default=str))
    print(path)


if __name__ == "__main__":
    main()
