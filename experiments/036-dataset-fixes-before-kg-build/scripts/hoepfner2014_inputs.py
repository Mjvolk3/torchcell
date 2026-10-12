# experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_inputs.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.hoepfner2014_inputs]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_inputs
"""Measure what the #506 Hoepfner 2014 input fixes change, without a 3.1M-record build.

Three modes (combinable):

- default (``--stream``): runs the loader's REAL record path
  (``_column_meta`` / ``_column_census`` / ``_iter_records``, the per-column schema
  validation included) over the full deposited ``HIP_scores.txt`` / ``HOP_scores.txt`` in
  the dev raw directory, with a dict-backed interned store and no LMDB. Every yielded
  record is unpickled and tallied. Reports, per arm (2026.10.12, every row stored as
  released, every non-excluded column stored, no detection threshold): records kept (vs
  the old dev drop ledger), rows with no score, rows whose token is not ORF-shaped, the
  records on SGD essential genes
  (``$DATA_ROOT/data/torchcell/gene_essentiality_sgd/preprocess/gene_set.json``), the
  excluded glucose-starvation columns and records, the columns / records dosed above
  200 uM (solvent gap) and of the pH agents, the columns and records whose compound
  carries the typed ``inchikey`` gap, the Table S5 construction coverage of HIP rows and
  records, the marker-locus records and their derived functional dose, the IC30-range
  columns, and the flag set against the 017 CSV. Writes
  ``results/hoepfner2014_inputs.json``.
- ``--slice-build``: copies every row of a fixed subset of deposited columns (the
  glucose-starvation, hydrochloric-acid, sodium-hydroxide, sodium-acetate and sorbitol
  columns plus the first ``--n-extra`` other encodable columns per arm), keeping EVERY
  sensitivity column's emptiness pattern,
  into a sliced raw directory under the scratch root, then runs the real ``process()``
  there. The sliced files carry their own sha256, so the slice build overrides the pinned
  raw hashes for that run only (a verification harness, never the loader). Writes
  ``results/hoepfner2014_inputs_slice.json``.
- ``--dev-lmdb``: after the slurm rebuild, reads the dev store
  (``$DATA_ROOT/data/torchcell/env_chemgen_hoepfner2014``) and tallies the same per-record
  quantities from it (experiment type, leaves, solvent gaps, construction coverage). Writes ``results/hoepfner2014_inputs_dev_lmdb.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_inputs.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_inputs.py \
        --slice-build --scratch-root <dir>
    python experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_inputs.py \
        --dev-lmdb
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import pickle
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import TypeAdapter

from torchcell.datamodels.schema import (
    ExperimentType,
    HeterozygousDeletionPerturbation,
    heterozygous_deletion_functional_copies,
)
from torchcell.datasets.scerevisiae import hoepfner2014 as m
from torchcell.literature.manifest import sha256_file

RESULTS = Path("experiments/036-dataset-fixes-before-kg-build/results")
SLUG = "data/torchcell/env_chemgen_hoepfner2014"
ESSENTIAL = "data/torchcell/gene_essentiality_sgd/preprocess/gene_set.json"
OLD_FLAG_CSV = Path(
    "experiments/017-hoepfner-background-mutations/results/table_s5_affected_strains.csv"
)
MARKER_LOCI = {
    "YOR202W": "HIS3",
    "YCL018W": "LEU2",
    "YEL021W": "URA3",
    "YLR303W": "MET17",
    "YBR115C": "LYS2",
}
SLICE_CMBS = ("4019", "4016", "4017", "4013", "4008")


class _DictTxn:
    """A dict-backed stand-in for the interned LMDB write transaction."""

    def __init__(self) -> None:
        self.store: dict[bytes, bytes] = {}

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def put(self, key: bytes, value: bytes) -> None:
        self.store[key] = value


def _essential(data_root: str) -> set[str]:
    with open(osp.join(data_root, ESSENTIAL)) as handle:
        return set(json.load(handle))


def _env_of(record: dict[str, Any], interned: dict[str, Any]) -> dict[str, Any]:
    env = record["experiment"]["environment"]
    if "$ref" in env:
        return pickle.loads(interned[env["$ref"].encode()])  # type: ignore[no-any-return]
    return env  # type: ignore[no-any-return]


def stream(data_root: str) -> dict[str, Any]:
    """Run the loader's record path over the full deposited matrices; tally records.

    No genome is read (2026.10.12): every row is stored under its released
    'Systematic Name', every sensitivity column of a non-excluded condition is stored
    (a compound with no structure identifier carries the typed ``inchikey`` gap), and
    there is no detection threshold.
    """
    raw_dir = osp.join(data_root, SLUG, "raw")
    dataset = m.EnvChemgenHoepfner2014Dataset.__new__(m.EnvChemgenHoepfner2014Dataset)
    dataset.name = "EnvChemgenHoepfner2014Dataset"
    table_s1 = osp.join(raw_dir, "Table_S1.xls")
    meta = m._load_compound_meta(table_s1)
    ic30 = m._load_ic30(table_s1)
    strains = m.load_table_s5_strains(Path.cwd())
    essential = _essential(data_root)
    background = m.hoepfner_background()
    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    itxn = _DictTxn()
    counts = m._BuildCounts()
    out: dict[str, Any] = {"by_assay": {}}
    deposited_ids: set[str] = set()
    gapped_ids: set[str] = set()
    for filename, assay in m._ASSAYS:
        with open(osp.join(raw_dir, filename)) as handle:
            header = handle.readline().rstrip("\n").split("\t")
        columns = dataset._column_meta(header, assay, meta, itxn)
        census = dataset._column_census(header, assay, meta)
        deposited_ids |= census.cmbs
        gapped_ids |= {col.cmb for col in columns if col.inchikey is None}
        refs = {study: {"$ref": study} for study in {col.study for col in columns}}
        reference_refs = {(assay, study): ref for study, ref in refs.items()}
        n_records = 0
        types: Counter[str] = Counter()
        leaves: Counter[str] = Counter()
        solvent_gap = 0
        ph_gap = 0
        gapped_records = 0
        essential_records = 0
        constructed: Counter[str] = Counter()
        marker: Counter[str] = Counter()
        marker_dose: dict[str, Any] = {}
        env_cache: dict[str, dict[str, Any]] = {}
        gapped_columns = {col.index for col in columns if col.inchikey is None}
        column_of_ref: dict[str, int] = {}
        for value in dataset._iter_records(
            osp.join(raw_dir, filename),
            assay,
            columns,
            reference_refs,
            {"$ref": "publication"},
            counts,
            strains,
            validate,
            census,
        ):
            record = pickle.loads(value)
            n_records += 1
            experiment = record["experiment"]
            types[experiment["experiment_type"]] += 1
            (pert,) = experiment["genotype"]["perturbations"]
            leaves[pert["perturbation_type"]] += 1
            ref = experiment["environment"].get("$ref", "")
            env = env_cache.get(ref)
            if env is None:
                env = _env_of(record, itxn.store)  # type: ignore[arg-type]
                env_cache[ref] = env
            small = env["perturbations"][0]
            if small["solvent"] is None:
                solvent_gap += 1
            if len(env["perturbations"]) > 1:
                ph_gap += 1
            if small["compound"]["inchikey"] is None:
                gapped_records += 1
            if pert["systematic_gene_name"] in essential:
                essential_records += 1
            if assay == "HIP":
                constructed[
                    "none"
                    if pert["construction"] is None
                    else "plate_and_well"
                    if pert["construction"]["plate"] is not None
                    else "lab_and_batch_only"
                ] += 1
            gene = pert["systematic_gene_name"]
            if gene in MARKER_LOCI:
                marker[MARKER_LOCI[gene]] += 1
                if assay == "HIP" and gene not in marker_dose:
                    leaf = HeterozygousDeletionPerturbation.model_validate(pert)
                    marker_dose[MARKER_LOCI[gene]] = (
                        heterozygous_deletion_functional_copies(background, leaf)
                    )
        del column_of_ref
        not_orf_shaped = counts.not_orf_shaped.get(assay, [])
        out["by_assay"][assay] = {
            "records_kept": n_records,
            "experiment_types": dict(types),
            "perturbation_types": dict(leaves),
            "rows_read": counts.rows[assay],
            "rows_without_any_score": counts.empty_rows[assay],
            "scored_sensitivity_cells": counts.source_cells[assay],
            "kept_columns": len(columns),
            "sensitivity_columns": len(census.sensitivity),
            "not_orf_shaped_rows": [row.model_dump() for row in not_orf_shaped],
            "glucose_starvation_columns": sum(
                1 for _, cmb in census.excluded if cmb == "4019"
            ),
            "glucose_starvation_records_excluded": counts.excluded[(assay, "4019")],
            "compound_identity_gap_columns": len(gapped_columns),
            "compound_identity_gap_cmb_ids": len(
                {col.cmb for col in columns if col.inchikey is None}
            ),
            "compound_identity_gap_records": gapped_records,
            "records_on_sgd_essential_genes": essential_records,
            "solvent_gap_columns": sum(
                1 for col in columns if col.conc > m.DMSO_CEILING_UM
            ),
            "solvent_gap_cmb_ids": sorted(
                {col.cmb for col in columns if col.conc > m.DMSO_CEILING_UM}
            ),
            "solvent_gap_records": solvent_gap,
            "ph_gap_columns": sum(
                1 for col in columns if col.cmb in m.PH_AGENT_CONDITIONS
            ),
            "ph_gap_records": ph_gap,
            "hip_construction_rows": dict(counts.construction)
            if assay == "HIP"
            else {},
            "hip_construction_records": dict(constructed),
            "marker_locus_records": dict(marker),
            "hip_marker_functional_copies": marker_dose,
            "positive_control_columns": len(census.positive_control),
            "kept_columns_with_ic30": sum(1 for col in columns if col.cmb in ic30),
            "kept_columns_outside_half_to_twice_ic30": sum(
                1
                for col in columns
                if col.cmb in ic30 and not 0.5 <= col.conc / ic30[col.cmb] <= 2.0
            ),
        }
    out["compound_identity_gap_cmb_ids_both_arms"] = len(gapped_ids)
    out["deposited_cmb_ids_both_arms"] = len(deposited_ids)
    flagged_now = set(counts.flagged)
    flagged_listed = {
        orf for orf, entries in strains.items() if any(e.flagged for e in entries)
    }
    with open(OLD_FLAG_CSV) as handle:
        old = {row["orf"] for row in csv.DictReader(handle)}
    out["flags"] = {
        "n_flagged_strains_with_records": len(flagged_now),
        "n_flagged_records": sum(counts.flagged.values()),
        "n_017_csv_strains": len(old),
        "flagged_now_not_in_017_csv": sorted(flagged_now - old),
        "in_017_csv_not_flagged_now": sorted(old - flagged_now),
        "n_flagged_in_table_s5": len(flagged_listed),
        "flagged_in_table_s5_without_kept_hip_records": sorted(
            flagged_listed - {orf.upper() for orf in flagged_now}
        ),
    }
    old_ledger = osp.join(data_root, SLUG, "dropped_records.json")
    with open(old_ledger) as handle:
        before = json.load(handle)["kept_by_assay"]
    out["dev_ledger_kept_by_assay_before"] = before
    out["records_kept_after"] = {
        assay: out["by_assay"][assay]["records_kept"] for assay in out["by_assay"]
    }
    out["records_kept_total_after"] = sum(out["records_kept_after"].values())
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "hoepfner2014_inputs.json", "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    return out


def slice_build(data_root: str, scratch_root: str, n_extra: int) -> dict[str, Any]:
    """Build a column slice of the real matrices through the real ``process()``."""
    import shutil

    raw_dir = osp.join(data_root, SLUG, "raw")
    root = osp.join(scratch_root, "env_chemgen_hoepfner2014_slice")
    if osp.exists(root):
        raise FileExistsError(f"{root} exists; choose a fresh --scratch-root")
    sliced_raw = osp.join(root, "raw")
    os.makedirs(sliced_raw)
    meta = m._load_compound_meta(osp.join(raw_dir, "Table_S1.xls"))
    pins: dict[str, dict[str, str]] = {}
    for filename, assay in m._ASSAYS:
        with open(osp.join(raw_dir, filename)) as handle:
            header = handle.readline().rstrip("\n").split("\t")
            keep = [0]
            extra = 0
            sensitivity: list[int] = []
            for index, raw in enumerate(header):
                match = m._COL_RE.match(raw.strip().strip('"'))
                if match is None or match.group("z") or match.group("assay") != assay:
                    continue
                sensitivity.append(index)
                cmb = match.group("cmb")
                if cmb in SLICE_CMBS:
                    keep.append(index)
                elif (meta.get(cmb) or {}).get("smiles") and extra < n_extra:
                    keep.append(index)
                    extra += 1
            # Every other sensitivity column is written as a placeholder column (CMB
            # id 0, no Table S1 SMILES) carrying the real cell, so the slice keeps the
            # full matrix's emptiness pattern.
            others = [i for i in sensitivity if i not in keep]
            with open(osp.join(sliced_raw, filename), "w") as out:
                new_header = [header[i] for i in keep] + [
                    f'"Ad. scores for Exp. 0_{n}_{assay}_9999"'
                    for n in range(len(others))
                ]
                out.write("\t".join(new_header) + "\n")
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    row = [parts[i] if i < len(parts) else '""' for i in keep]
                    row += [parts[i] if i < len(parts) else '""' for i in others]
                    out.write("\t".join(row) + "\n")
        pins[filename] = {
            "url": m._DRYAD_FILES[filename]["url"],
            "sha256": sha256_file(Path(osp.join(sliced_raw, filename))),
        }
    shutil.copy2(
        osp.join(raw_dir, "Table_S1.xls"), osp.join(sliced_raw, "Table_S1.xls")
    )
    pins["Table_S1.xls"] = m._DRYAD_FILES["Table_S1.xls"]
    m._DRYAD_FILES.clear()
    m._DRYAD_FILES.update(pins)
    dataset = m.EnvChemgenHoepfner2014Dataset(root=root)
    with open(osp.join(root, "dropped_records.json")) as handle:
        ledger = json.load(handle)
    types: Counter[str] = Counter()
    leaves: Counter[str] = Counter()
    solvent_none = 0
    ph = 0
    example: dict[str, Any] = {}
    for i in range(len(dataset)):
        experiment = dataset[i]["experiment"]
        types[experiment["experiment_type"]] += 1
        (pert,) = experiment["genotype"]["perturbations"]
        leaves[pert["perturbation_type"]] += 1
        perts = experiment["environment"]["perturbations"]
        solvent_none += perts[0]["solvent"] is None
        ph += len(perts) > 1
        if pert["perturbed_gene_name"] == "YBR075W" and "renamed" not in example:
            example["renamed"] = pert
        if pert["systematic_gene_name"] == "YBR115C" and "lys2_hip" not in example:
            if pert["perturbation_type"] == "heterozygous_deletion":
                example["lys2_hip"] = pert
    reference = dataset[0]["reference"]
    result = {
        "root": root,
        "n_records": len(dataset),
        "experiment_types": dict(types),
        "perturbation_types": dict(leaves),
        "records_with_solvent_gap": solvent_none,
        "records_with_ph_factor": ph,
        "ledger_kept_by_assay": ledger["kept_by_assay"],
        "ledger_not_orf_shaped_rows": {
            assay: len(rows) for assay, rows in ledger["not_orf_shaped"].items()
        },
        "ledger_excluded_conditions": ledger["excluded_conditions"],
        "reference_genome": reference["genome_reference"],
        "examples": example,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "hoepfner2014_inputs_slice.json", "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=str)
    return result


def dev_lmdb(data_root: str) -> dict[str, Any]:
    """Tally the per-record quantities from the rebuilt dev store."""
    from torchcell.verification.runners import stream_records

    types: Counter[str] = Counter()
    leaves: Counter[str] = Counter()
    solvent_none: Counter[str] = Counter()
    construction: Counter[str] = Counter()
    for record in stream_records(osp.join(data_root, SLUG)):
        experiment = record["experiment"]
        types[experiment["experiment_type"]] += 1
        (pert,) = experiment["genotype"]["perturbations"]
        leaf = pert["perturbation_type"]
        leaves[leaf] += 1
        if experiment["environment"]["perturbations"][0]["solvent"] is None:
            solvent_none[leaf] += 1
        if leaf == "heterozygous_deletion":
            construction["none" if pert["construction"] is None else "set"] += 1
    result = {
        "experiment_types": dict(types),
        "perturbation_types": dict(leaves),
        "solvent_gap_records": dict(solvent_none),
        "hip_construction_records": dict(construction),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "hoepfner2014_inputs_dev_lmdb.json", "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def main() -> None:
    """Parse the mode flags and run the requested measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--slice-build", action="store_true")
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--n-extra", type=int, default=4)
    parser.add_argument("--dev-lmdb", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    if not args.no_stream:
        print(json.dumps(stream(data_root), indent=2, sort_keys=True))
    if args.slice_build:
        if args.scratch_root is None:
            raise SystemExit("--slice-build needs --scratch-root")
        print(
            json.dumps(
                slice_build(data_root, args.scratch_root, args.n_extra),
                indent=2,
                sort_keys=True,
                default=str,
            )[:6000]
        )
    if args.dev_lmdb:
        print(json.dumps(dev_lmdb(data_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
