# experiments/036-dataset-fixes-before-kg-build/scripts/single_cell_leaf_merge_audit.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.single_cell_leaf_merge_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/single_cell_leaf_merge_audit.py
"""Measure whether Boocock 2025's and Jackson 2020's single-cell UMI-count leaves can be
one class without losing a stored value (#1016), and check the merged stores against the
stores they replaced.

Two passes, each over the LMDB ``processed/`` directories of dev stores (read only):

``round-trip <processed>...``: the decision measurement, run on the stores built BEFORE
the merge. Every stored phenotype (every record and every distinct reference) is
validated by ``CandidateLeaf`` (the merged field set, its per-source fields nullable with
a typed gap) after setting each field the store does not carry to None with a gap; the
dump must return every stored field with an identical value and JSON type. It also
measures the invariants a merged ``total_umis`` must respect: how many records have a
total equal to their stored counts (gene plus marker) and how many exceed them.

``old-vs-new <old processed> <new dataset root>``: after the rebuild, pair the retired
store with its successor record by record (LMDB key order; both builds are
deterministic over the same raw bytes) and report every field of the old record that
the new record dropped or changed. A gap added to ``provenance_gaps`` and a new field are
reported as additions.

``cells <dataset root>``: on a rebuilt store, per distinct reference (a panel or a
condition, every cell of it): the cell count, the distinct bare barcodes, the distinct
(library, barcode) pairs and the reference phenotype's JSON size. A bare barcode that
repeats across libraries is why the merged leaf carries ``cell_libraries``.

Results: ``experiments/036-dataset-fixes-before-kg-build/results/single_cell_leaf_merge_audit.json``
(one entry per invocation, keyed by mode and store).

    PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/single_cell_leaf_merge_audit.py round-trip <processed>...
    PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/single_cell_leaf_merge_audit.py old-vs-new <old processed> <new root>
    PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/single_cell_leaf_merge_audit.py cells <new root>
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import pickle
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lmdb
from pydantic import Field, field_validator, model_validator
from sortedcontainers import SortedDict

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datamodels.pydant import ModelStrict
from torchcell.datamodels.schema import Phenotype, _require_value_or_gap
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

RESULTS = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "single_cell_leaf_merge_audit.json"
)
#: Fields only one of the two pre-merge leaves carried.
SOURCE_SPECIFIC = (
    "cell_cycle_stage_counts",
    "library_cell_counts",
    "marker_umi_count",
    "cell_barcodes",
    "cell_total_umis",
)


class CandidateLeaf(Phenotype, ModelStrict):
    """The merged field set as measured before the merge: both leaves' fields, the
    source-specific ones nullable with a typed gap, ``total_umis`` bounded below by the
    stored counts (Jackson's equality and Boocock's inequality are both admitted).
    """

    graph_level: str = "node"
    label_name: str = "umi_count"
    label_statistic_name: str | None = None
    umi_count: dict[str, int]
    total_umis: int
    n_features: int
    feature_universe_sha256: str
    n_cells: int
    cell_cycle_stage_counts: dict[str, int] | None
    library_cell_counts: dict[str, int] | None
    marker_umi_count: dict[str, int] | None
    cell_barcodes: list[str] | None
    cell_total_umis: list[int] | None
    measurement_type: str = Field(default="scrnaseq_umi_count")

    @field_validator("umi_count", "marker_umi_count", mode="before")
    @classmethod
    def _sorted(cls, v: Any) -> Any:
        return v if v is None else SortedDict(v)

    @model_validator(mode="after")
    def _check(self) -> CandidateLeaf:
        _require_value_or_gap(self, SOURCE_SPECIFIC)
        markers = self.marker_umi_count or {}
        if any(n < 1 for n in self.umi_count.values()):
            raise ValueError("a stored gene count is not positive")
        if set(markers) & set(self.umi_count):
            raise ValueError("a feature is keyed as both a gene and a marker")
        if self.total_umis < sum(self.umi_count.values()) + sum(markers.values()):
            raise ValueError("total_umis is below the stored counts")
        if self.n_features < len(self.umi_count):
            raise ValueError("n_features is below the keyed features")
        if not re.fullmatch(r"[0-9a-f]{64}", self.feature_universe_sha256):
            raise ValueError("feature_universe_sha256 is not 64 lowercase hex digits")
        for name in ("cell_cycle_stage_counts", "library_cell_counts"):
            counts = getattr(self, name)
            if counts is not None and sum(counts.values()) != self.n_cells:
                raise ValueError(f"{name} does not sum to n_cells")
        if self.cell_barcodes is not None:
            if len(set(self.cell_barcodes)) != self.n_cells:
                raise ValueError("cell_barcodes are not n_cells distinct barcodes")
            if self.cell_total_umis is None:
                raise ValueError("cell_barcodes without cell_total_umis")
            if sum(self.cell_total_umis) != self.total_umis:
                raise ValueError("cell_total_umis does not sum to total_umis")
        return self


def stream(processed: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """(LMDB key, record with interned sub-objects resolved), in key order."""
    ienv = lmdb.open(osp.join(processed, "interned"), readonly=True, lock=False)
    with ienv.begin() as txn:
        interned = {k.decode(): pickle.loads(v) for k, v in txn.cursor()}
    ienv.close()
    env = lmdb.open(osp.join(processed, "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        for key, value in txn.cursor():
            yield key.decode(), resolve_interned(pickle.loads(value), interned)
    env.close()


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=dict))


def _round_trip(stored: dict[str, Any], tally: Counter[str]) -> None:
    """Validate one stored phenotype through the candidate and compare every field."""
    payload = dict(stored)
    gapped = {g["field"] for g in stored["provenance_gaps"]}
    absent = [f for f in SOURCE_SPECIFIC if f not in stored]
    for name in absent:
        payload[name] = None
    payload["provenance_gaps"] = list(stored["provenance_gaps"]) + [
        ProvenanceGap(field=name, reason=ProvenanceGapReason.not_reported_by_primary)
        for name in absent
        if name not in gapped
    ]
    dumped = _json(CandidateLeaf.model_validate(payload).model_dump(mode="json"))
    original = _json(stored)
    for name, value in original.items():
        if name == "provenance_gaps":
            tally[
                "gaps_kept" if dumped[name][: len(value)] == value else "gaps_lost"
            ] += 1
        elif name not in dumped:
            tally[f"dropped:{name}"] += 1
        elif dumped[name] != value or type(dumped[name]) is not type(value):
            tally[f"changed:{name}"] += 1
    tally["round_tripped"] += 1


def round_trip(processed: str) -> dict[str, Any]:
    records: Counter[str] = Counter()
    references: Counter[str] = Counter()
    seen: set[int] = set()
    totals: Counter[str] = Counter()
    outside = 0
    measurement_types: Counter[str] = Counter()
    for _, rec in stream(processed):
        phenotype = rec["experiment"]["phenotype"]
        _round_trip(phenotype, records)
        measurement_types[phenotype["measurement_type"]] += 1
        stored = sum(phenotype["umi_count"].values()) + sum(
            (phenotype.get("marker_umi_count") or {}).values()
        )
        if phenotype["total_umis"] == stored:
            totals["equal_to_stored_counts"] += 1
        else:
            totals["above_stored_counts"] += 1
            outside += phenotype["total_umis"] - stored
        reference = rec["reference"]["phenotype_reference"]
        if id(reference) not in seen:
            seen.add(id(reference))
            _round_trip(reference, references)
    return {
        "records": dict(records),
        "references": dict(references),
        "n_references": len(seen),
        "record_totals": dict(totals),
        "umis_outside_stored_counts": outside,
        "measurement_types": dict(measurement_types),
    }


def _diff(
    old: Any, new: Any, path: str, problems: Counter[str], added: Counter[str]
) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for name, value in old.items():
            if name not in new:
                problems[f"dropped:{path}.{name}"] += 1
            else:
                _diff(value, new[name], f"{path}.{name}", problems, added)
        for name in set(new) - set(old):
            added[f"{path}.{name}"] += 1
        return
    if path.endswith(".provenance_gaps"):
        old_gaps, new_gaps = _json(old), _json(new)
        if any(g not in new_gaps for g in old_gaps):
            problems[f"dropped:{path}"] += 1
        for gap in new_gaps:
            if gap not in old_gaps:
                added[f"{path}[field={gap['field']}]"] += 1
        return
    if _json(old) != _json(new):
        problems[f"changed:{path}"] += 1


def old_vs_new(old_processed: str, new_root: str) -> dict[str, Any]:
    problems: Counter[str] = Counter()
    added: Counter[str] = Counter()
    seen: set[int] = set()
    n = 0
    pairs = zip(
        stream(old_processed), stream(osp.join(new_root, "processed")), strict=True
    )
    for (old_key, old), (new_key, new) in pairs:
        n += 1
        if old_key != new_key:
            problems["key_order"] += 1
        _diff(old["experiment"], new["experiment"], "experiment", problems, added)
        _diff(old["publication"], new["publication"], "publication", problems, added)
        if id(old["reference"]) not in seen:
            seen.add(id(old["reference"]))
            _diff(old["reference"], new["reference"], "reference", problems, added)
    return {
        "n_records": n,
        "n_references": len(seen),
        "dropped_or_changed": dict(problems),
        "added": dict(sorted(added.items())),
    }


def cells(root: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    seen: set[int] = set()
    for _, rec in stream(osp.join(root, "processed")):
        reference = rec["reference"]["phenotype_reference"]
        if id(reference) in seen:
            continue
        seen.add(id(reference))
        barcodes = reference["cell_barcodes"]
        libraries = reference["cell_libraries"]
        out[",".join(sorted(reference["library_cell_counts"]))] = {
            "n_cells": reference["n_cells"],
            "distinct_barcodes": len(set(barcodes)),
            "distinct_library_barcode_pairs": len(set(zip(libraries, barcodes))),
            "reference_phenotype_json_bytes": len(json.dumps(_json(reference))),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    rt = sub.add_parser("round-trip")
    rt.add_argument("processed", nargs="+")
    on = sub.add_parser("old-vs-new")
    on.add_argument("old_processed")
    on.add_argument("new_root")
    ce = sub.add_parser("cells")
    ce.add_argument("root")
    args = parser.parse_args()
    measured: dict[str, Any] = {}
    if args.mode == "round-trip":
        for processed in args.processed:
            measured[f"round-trip:{processed}"] = round_trip(processed)
    elif args.mode == "old-vs-new":
        key = f"old-vs-new:{args.old_processed}"
        measured[key] = old_vs_new(args.old_processed, args.new_root)
    else:
        measured[f"cells:{args.root}"] = cells(args.root)
    print(json.dumps(measured, indent=2))
    # Read the file only now: a long pass must not overwrite an entry another
    # invocation wrote while it ran.
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results.update(measured)
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
