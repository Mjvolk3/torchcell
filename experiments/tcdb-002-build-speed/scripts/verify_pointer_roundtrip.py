# experiments/tcdb-002-build-speed/scripts/verify_pointer_roundtrip.py
# [[experiments.tcdb-002-build-speed.scripts.verify_pointer_roundtrip]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/verify_pointer_roundtrip
"""Prove a build's Experiment blobs resolve back to the records they hash.

The Experiment node id is the sha256 of the fully inlined ``experiment.model_dump()``
JSON. With interned-constant pointers (torchcell/datamodels/interned_constant.py) the
blob written to ``serialized_data`` no longer contains the environment (or a large
genotype); this script reads a BioCypher output directory, loads every
``InternedConstant`` row into a store, verifies each payload hashes to its id, then
resolves every ``Experiment`` row's pointers and recomputes the sha256 of the
resolved JSON. A row whose recomputed hash differs from its id is a broken build.

It also reports what the pointers bought: the Experiment CSV bytes on disk against
the bytes the same rows would have taken inlined.

    python experiments/tcdb-002-build-speed/scripts/verify_pointer_roundtrip.py \
        <biocypher-out dir> --job <slurm job id>

Writes ``results/<job>_pointer_roundtrip.csv`` (one row per label kind and total).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os.path as osp
from collections import Counter
from pathlib import Path

from torchcell.datamodels.interned_constant import (
    POINTER_KEY,
    collect_pointers,
    resolve_pointers,
    verified_constant,
)
from torchcell.knowledge_graphs.incremental_import import (
    _iter_rows,
    _unquote,
    discover_csv_groups,
)

RESULTS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("--job", required=True)
    args = parser.parse_args()

    groups = {g.label: g for g in discover_csv_groups(Path(args.out_dir))}
    constants_group = groups["InternedConstant"]
    experiment_group = groups["Experiment"]

    id_col = constants_group.columns.index(":ID")
    kind_col = constants_group.columns.index("kind")
    data_col = constants_group.columns.index("serialized_data")
    store: dict[str, object] = {}
    kinds: Counter[str] = Counter()
    constant_bytes: Counter[str] = Counter()
    for row in _iter_rows(constants_group.part_paths):
        # csv.reader already strips the quotes and un-doubles the embedded ones; a
        # second un-doubling corrupted a payload holding a real '' (a compound name,
        # job 2959) and reported the store as corrupt.
        ref = _unquote(row[id_col])
        payload = _unquote(row[data_col])
        store[ref] = verified_constant(ref, payload)
        kind = _unquote(row[kind_col])
        kinds[kind] += 1
        constant_bytes[kind] += len(payload)

    id_col = experiment_group.columns.index(":ID")
    data_col = experiment_group.columns.index("serialized_data")
    n_rows = 0
    n_bad = 0
    bad_sample: list[str] = []
    blob_bytes = 0
    inline_bytes = 0
    pointers_per_kind: Counter[str] = Counter()
    for row in _iter_rows(experiment_group.part_paths):
        n_rows += 1
        node_id = _unquote(row[id_col])
        blob = _unquote(row[data_col])
        blob_bytes += len(blob)
        data = json.loads(blob)
        refs: set[str] = set()
        collect_pointers(data, refs)
        for field in ("genotype", "environment", "phenotype"):
            value = data.get(field)
            if isinstance(value, dict) and POINTER_KEY in value:
                pointers_per_kind[field] += 1
        inline = json.dumps(resolve_pointers(data, store))
        inline_bytes += len(inline)
        if hashlib.sha256(inline.encode("utf-8")).hexdigest() != node_id:
            n_bad += 1
            if len(bad_sample) < 5:
                bad_sample.append(node_id)

    rows = [
        {
            "job": args.job,
            "label": "Experiment",
            "rows": n_rows,
            "rows_failing_roundtrip": n_bad,
            "blob_bytes": blob_bytes,
            "inline_bytes": inline_bytes,
            "pointers_environment": pointers_per_kind["environment"],
            "pointers_genotype": pointers_per_kind["genotype"],
        }
    ]
    for kind in sorted(kinds):
        rows.append(
            {
                "job": args.job,
                "label": f"InternedConstant:{kind}",
                "rows": kinds[kind],
                "rows_failing_roundtrip": 0,
                "blob_bytes": constant_bytes[kind],
                "inline_bytes": 0,
                "pointers_environment": 0,
                "pointers_genotype": 0,
            }
        )
    out = osp.join(RESULTS_DIR, f"{args.job}_pointer_roundtrip.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    total_constant = sum(constant_bytes.values())
    print(
        f"Experiment rows {n_rows:,}; failing round trip {n_bad:,}; "
        f"blob bytes {blob_bytes / 1e9:.3f} GB against {inline_bytes / 1e9:.3f} GB inlined; "
        f"constants {sum(kinds.values()):,} rows, {total_constant / 1e9:.3f} GB "
        f"({dict(kinds)})"
    )
    if bad_sample:
        print("first failing ids:", bad_sample)
    print(f"wrote {out}")
    if n_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
