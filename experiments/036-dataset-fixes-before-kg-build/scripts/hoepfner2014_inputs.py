# experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_inputs.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.hoepfner2014_inputs]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_inputs
"""Measure what the #506 Hoepfner 2014 input fixes change, without a 3.1M-record build.

Three modes (combinable):

- default (``--stream``): runs the loader's REAL record path
  (``_column_meta`` / ``_column_census`` / ``_iter_records``, the per-column schema
  validation included) over the full deposited ``HIP_scores.txt`` / ``HOP_scores.txt`` in
  the dev raw directory, with a dict-backed interned store and no LMDB. Every yielded
  record is unpickled and tallied. Reports, per arm: records kept (vs the old dev drop
  ledger), rows the detection rule drops and how many of those are in the SGD essential
  gene set (``$DATA_ROOT/data/torchcell/gene_essentiality_sgd/preprocess/gene_set.json``),
  the excluded glucose-starvation columns and records, the columns / records dosed above
  200 uM (solvent gap) and of the pH agents, the RENAMED-ORF records now carrying a
  ``constructed_orf`` (and how many target an essential gene), the Table S5 construction
  coverage of HIP rows and records, the marker-locus records and their derived functional
  dose, the encodable-only filter, the IC30-range columns, and the flag set against the
  017 CSV. Writes ``results/hoepfner2014_inputs.json`` and
  ``results/hoepfner2014_inputs_detection_dropped.csv``.
- ``--slice-build``: copies every row of a fixed subset of deposited columns (the
  glucose-starvation, hydrochloric-acid, sodium-hydroxide, sodium-acetate and sorbitol
  columns plus the first ``--n-extra`` other encodable columns per arm), keeping EVERY
  sensitivity column's emptiness pattern so the detection rule sees the full denominator,
  into a sliced raw directory under the scratch root, then runs the real ``process()``
  there. The sliced files carry their own sha256, so the slice build overrides the pinned
  raw hashes for that run only (a verification harness, never the loader). Writes
  ``results/hoepfner2014_inputs_slice.json``.
- ``--dev-lmdb``: after the slurm rebuild, reads the dev store
  (``$DATA_ROOT/data/torchcell/env_chemgen_hoepfner2014``) and tallies the same per-record
  quantities from it (experiment type, leaves, solvent gaps, constructed ORFs, construction
  coverage). Writes ``results/hoepfner2014_inputs_dev_lmdb.json``.

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


def _genome() -> Any:
    """The read-only S288C genome whose resolver the loader uses."""
    from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

    data_root = os.environ["DATA_ROOT"]
    return SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )


def _essential(data_root: str) -> set[str]:
    with open(osp.join(data_root, ESSENTIAL)) as handle:
        return set(json.load(handle))


def _env_of(record: dict[str, Any], interned: dict[str, Any]) -> dict[str, Any]:
    env = record["experiment"]["environment"]
    if "$ref" in env:
        return pickle.loads(interned[env["$ref"].encode()])  # type: ignore[no-any-return]
    return env  # type: ignore[no-any-return]


def stream(data_root: str) -> dict[str, Any]:
    """Run the loader's record path over the full deposited matrices; tally records."""
    raw_dir = osp.join(data_root, SLUG, "raw")
    dataset = m.EnvChemgenHoepfner2014Dataset.__new__(m.EnvChemgenHoepfner2014Dataset)
    dataset.name = "EnvChemgenHoepfner2014Dataset"
    table_s1 = osp.join(raw_dir, "Table_S1.xls")
    meta = m._load_compound_meta(table_s1)
    ic30 = m._load_ic30(table_s1)
    strains = m.load_table_s5_strains(Path.cwd())
    sgd_genes = m._load_sgd_genes(data_root)
    essential = _essential(data_root)
    background = m.hoepfner_background()
    resolve = _genome().resolve_gene_name
    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    itxn = _DictTxn()
    counts = m._BuildCounts()
    out: dict[str, Any] = {"by_assay": {}}
    unencodable_ids: set[str] = set()
    deposited_ids: set[str] = set()
    detection_rows: list[dict[str, Any]] = []
    for filename, assay in m._ASSAYS:
        with open(osp.join(raw_dir, filename)) as handle:
            header = handle.readline().rstrip("\n").split("\t")
        columns, dropped = dataset._column_meta(header, assay, meta, itxn)
        census = dataset._column_census(header, assay, meta)
        unencodable_ids |= {cmb for _, cmb in census.unencodable}
        deposited_ids |= census.cmbs
        refs = {study: {"$ref": study} for study in {col.study for col in columns}}
        reference_refs = {(assay, study): ref for study, ref in refs.items()}
        n_records = 0
        types: Counter[str] = Counter()
        leaves: Counter[str] = Counter()
        solvent_gap = 0
        ph_gap = 0
        renamed_records = 0
        renamed_essential_records = 0
        constructed: Counter[str] = Counter()
        marker: Counter[str] = Counter()
        marker_dose: dict[str, Any] = {}
        env_cache: dict[str, dict[str, Any]] = {}
        for value in dataset._iter_records(
            osp.join(raw_dir, filename),
            assay,
            sgd_genes,
            columns,
            dropped,
            reference_refs,
            {"$ref": "publication"},
            counts,
            strains,
            validate,
            resolve,
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
            if pert["constructed_orf"] is not None:
                renamed_records += 1
                if pert["systematic_gene_name"] in essential:
                    renamed_essential_records += 1
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
        undetected = counts.undetected[assay]
        for row in undetected:
            detection_rows.append(
                {
                    "assay": assay,
                    "source_name": row.source_name,
                    "systematic_name": row.systematic_name,
                    "sgd_essential": row.systematic_name in essential,
                    "n_scored_columns": row.n_scored_columns,
                    "n_columns": row.n_columns,
                    "n_records": row.n_records,
                }
            )
        renamed = counts.renamed_orfs[assay]
        out["by_assay"][assay] = {
            "records_kept": n_records,
            "experiment_types": dict(types),
            "perturbation_types": dict(leaves),
            "kept_columns": len(columns),
            "detection_columns": len(census.detection),
            "detection_rows_kept": counts.detected_rows[assay],
            "detection_rows_dropped": len(undetected),
            "detection_rows_dropped_sgd_essential": sum(
                1 for row in undetected if row.systematic_name in essential
            ),
            "detection_records_dropped": sum(row.n_records for row in undetected),
            "detection_records_dropped_sgd_essential": sum(
                row.n_records for row in undetected if row.systematic_name in essential
            ),
            "rows_without_any_score": counts.empty_rows[assay],
            "orf_rule_rows_dropped": len(counts.dropped_orfs[assay]),
            "glucose_starvation_columns": sum(
                1 for _, cmb in census.excluded if cmb == "4019"
            ),
            "glucose_starvation_records_excluded": counts.excluded[(assay, "4019")],
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
            "renamed_rows": len(renamed),
            "renamed_targets_sgd_essential": sorted(
                {cur for cur in renamed.values() if cur in essential}
            ),
            "renamed_records": renamed_records,
            "renamed_records_sgd_essential": renamed_essential_records,
            "hip_construction_rows": dict(counts.construction)
            if assay == "HIP"
            else {},
            "hip_construction_records": dict(constructed),
            "marker_locus_records": dict(marker),
            "hip_marker_functional_copies": marker_dose,
            "unencodable_columns": len(census.unencodable),
            "unencodable_cmb_ids": len({cmb for _, cmb in census.unencodable}),
            "unencodable_records": counts.unencodable[assay],
            "positive_control_columns": len(census.positive_control),
            "kept_columns_with_ic30": sum(1 for col in columns if col.cmb in ic30),
            "kept_columns_outside_half_to_twice_ic30": sum(
                1
                for col in columns
                if col.cmb in ic30 and not 0.5 <= col.conc / ic30[col.cmb] <= 2.0
            ),
        }
    out["unencodable_cmb_ids_both_arms"] = len(unencodable_ids)
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
    with open(RESULTS / "hoepfner2014_inputs_detection_dropped.csv", "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detection_rows[0]))
        writer.writeheader()
        writer.writerows(detection_rows)
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
            # Every other sensitivity column is written as an unencodable placeholder
            # column (CMB id 0, no Table S1 SMILES) carrying the real cell, so the
            # detection rule sees the same denominator and numerator as the full build.
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
    dataset = m.EnvChemgenHoepfner2014Dataset(root=root, genome=_genome())
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
        "ledger_detection_dropped": {
            assay: len(value["dropped"]) for assay, value in ledger["detection"].items()
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
    constructed: Counter[str] = Counter()
    construction: Counter[str] = Counter()
    for record in stream_records(osp.join(data_root, SLUG)):
        experiment = record["experiment"]
        types[experiment["experiment_type"]] += 1
        (pert,) = experiment["genotype"]["perturbations"]
        leaf = pert["perturbation_type"]
        leaves[leaf] += 1
        if experiment["environment"]["perturbations"][0]["solvent"] is None:
            solvent_none[leaf] += 1
        if pert.get("constructed_orf") is not None:
            constructed[leaf] += 1
        if leaf == "heterozygous_deletion":
            construction["none" if pert["construction"] is None else "set"] += 1
    result = {
        "experiment_types": dict(types),
        "perturbation_types": dict(leaves),
        "solvent_gap_records": dict(solvent_none),
        "constructed_orf_records": dict(constructed),
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
