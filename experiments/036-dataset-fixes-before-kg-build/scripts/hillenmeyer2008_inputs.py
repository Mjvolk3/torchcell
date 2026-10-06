# experiments/036-dataset-fixes-before-kg-build/scripts/hillenmeyer2008_inputs.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.hillenmeyer2008_inputs]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/hillenmeyer2008_inputs
"""Measure what the #505 Hillenmeyer 2008 input fixes change, per arm (het, hom).

Reads the raw-mirror matrices and key files
(``$DATA_ROOT/torchcell-raw/hillenmeyerChemicalGenomicPortrait2008/data/``) through the
LOADER'S OWN parsing functions (``parse_columns``, ``read_matrix_rows``,
``count_records``), resolving every row id with the read-only S288C genome exactly as
``process()`` does. No full LMDB is built (2.7M + 1.1M records). Reports, per arm:

- the exact record count the new loader writes (the ``expected_count`` oracle) and the
  old dev-store count (``preprocess/dropped_records.json`` of the dev tree), i.e. the
  records that change class (every record: ``EngineeredCopyNumberPerturbation`` ->
  ``HeterozygousDeletionPerturbation``, ``KanMxDeletionPerturbation`` ->
  ``BarcodedKanMxDeletionPerturbation``);
- the current genes fed by two or more source ORFs (the old merged records), the source
  ORFs and the records each now gets;
- every header / key-file disagreement, its rule and its outcome;
- the minimal-media arrays and the medium each is now served on;
- the 0gen arrays dropped and the records they would have written;
- the BY marker loci (HIS3, LYS2, MET15): rows, records, and the functional dose
  ``heterozygous_deletion_functional_copies`` derives from the BY4743 background;
- the dropped strains (hom PDR5, the ``YDL227C:ctrl_*`` rows) and their records;
- the suspicious-batch counts behind the SOM's "647 strains", several ways.

Then a SLICED BUILD: a handful of arrays x rows go through ``iter_records`` and
``build_reference`` (the functions ``process()`` calls), each record is dumped as the
LMDB stores it, written to the scratch root, re-validated, and run through the
environment-response verifier and the adapter's node builders.

With ``--dev-lmdb`` it instead streams the rebuilt dev stores
(``$DATA_ROOT/data/torchcell/env_chemgen_hillenmeyer2008_{het,hom}``) and counts records
by perturbation type, constructed-ORF relation, marker locus and pre-culture source (run
it after the slurm rebuild).

Writes ``experiments/036-dataset-fixes-before-kg-build/results/hillenmeyer2008_inputs.json``
(or ``hillenmeyer2008_inputs_dev_lmdb.json``) and ``hillenmeyer2008_key_header_conflicts.csv``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/hillenmeyer2008_inputs.py \
        --scratch-root <scratch dir>
    python experiments/036-dataset-fixes-before-kg-build/scripts/hillenmeyer2008_inputs.py --dev-lmdb
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
from collections import Counter
from typing import Any

from dotenv import load_dotenv

from torchcell.datamodels.schema import (
    HeterozygousDeletionPerturbation,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    heterozygous_deletion_functional_copies,
)
from torchcell.datasets.scerevisiae import hillenmeyer2008 as h

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
#: Arrays of the sliced build: one per behavior the fix changes.
SLICE_ARRAYS: dict[str, list[str]] = {
    # benomyl / nocodazole 15gen (YPD; three arrays, so YPD stays the modal medium the
    # verifier's environment_perturbed rule takes as baseline), minimal medium 5gen and
    # -5gen (SD + supplement gap)
    "het": ["01_04_24_02", "01_04_24_03", "01_04_24_04", "03_06_05_04", "03_06_05_01"],
    # pH8 (header wins), minimal->SC (key wins), biotin 25 %, -5gen psoralen
    "hom": ["04_11_17_01", "03_06_06_04", "04_11_24_10", "04_02_24_13"],
}
#: Source ORFs of the sliced build: a merge pair, the three marker loci, HO (YDL227C)
#: and an ORF built in two batches (YBR020W, both arms).
SLICE_ORFS = {
    "YAR042W",
    "YAR044W",
    "YOR202W",
    "YBR115C",
    "YLR303W",
    "YDL227C",
    "YBR020W",
}
MARKER_NAMES = {"YOR202W": "HIS3", "YBR115C": "LYS2", "YLR303W": "MET15/MET17"}


def _genome(data_root: str) -> Any:
    """The read-only S288C genome the loader resolves names with."""
    from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

    return SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )


def _arm_inputs(
    arm: str, data_root: str, resolve_name: Any
) -> tuple[list[h.ColumnSpec], h.MatrixRows]:
    """Columns and rows of one arm, through the loader's own functions."""
    spec = h.MATRICES[arm]
    raw = h.raw_mirror_dir(data_root) / "data"
    with open(raw / spec.filename) as handle:
        header = handle.readline().rstrip("\n").split("\t")
    columns = h.parse_columns(
        header,
        h.read_control_set_map(raw / spec.keyfile),
        h.read_key_conditions(raw / spec.keyfile),
    )
    rows = h.read_matrix_rows(
        raw / spec.filename,
        len(header),
        h._load_sgd_genes(data_root),
        resolve_name,
        arm,
    )
    return columns, rows


def _old_count(arm: str, data_root: str) -> int | None:
    """Kept records of the current dev store, from its build-time drop report."""
    path = osp.join(
        data_root,
        f"data/torchcell/env_chemgen_hillenmeyer2008_{arm}/preprocess/dropped_records.json",
    )
    if not osp.exists(path):
        return None
    with open(path) as handle:
        return int(json.load(handle)["kept_records"])


def _suspicious(arm: str, data_root: str, rows: h.MatrixRows) -> dict[str, Any]:
    """Rows in the SOM's ten suspicious batches, counted several ways (for the 647)."""
    spec = h.MATRICES[arm]
    raw = h.raw_mirror_dir(data_root) / "data"
    batches = set(h.SUSPICIOUS_BATCHES)
    exact = slash = 0
    exact_measured = 0
    with open(raw / spec.filename) as handle:
        handle.readline()
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            batch = cells[0].strip().strip('"').partition(":")[2]
            if batch in batches:
                exact += 1
                if any(c.strip().upper() not in h._MISSING for c in cells[1:]):
                    exact_measured += 1
            if set(batch.split("/")) & batches:
                slash += 1
    kept = [r for r in rows.rows if r.batch in batches]
    return {
        "raw_rows_exact_batch": exact,
        "raw_rows_exact_batch_with_any_value": exact_measured,
        "raw_rows_incl_slash_joined_batches": slash,
        "kept_rows_exact_batch": len(kept),
        "kept_distinct_current_orfs_exact_batch": len({r.orf for r in kept}),
        "kept_distinct_source_orfs_exact_batch": len({r.source_orf for r in kept}),
        "som_count": 647,
    }


def measure_arm(arm: str, data_root: str, resolve_name: Any) -> dict[str, Any]:
    """Every count the note and the report quote for one arm."""
    columns, rows = _arm_inputs(arm, data_root, resolve_name)
    groups, dropped_groups = h.group_columns(columns)
    n_new = h.count_records(rows.rows, groups)
    background = h.hillenmeyer_background()

    sources: dict[str, set[str]] = {}
    for row in rows.rows:
        sources.setdefault(row.orf, set()).add(row.source_orf)
    merged = {
        gene: sorted(srcs) for gene, srcs in sorted(sources.items()) if len(srcs) >= 2
    }
    merged_records = {
        gene: {
            src: h.count_records(
                [r for r in rows.rows if r.source_orf == src and r.orf == gene], groups
            )
            for src in srcs
        }
        for gene, srcs in merged.items()
    }
    multi_batch = {
        src: n for src, n in Counter(r.source_orf for r in rows.rows).items() if n >= 2
    }

    conflicts = [
        {
            "arm": arm,
            "filename": c.filename,
            "header_condition": c.key_check.header_condition,
            "key_condition": c.key_check.key_condition,
            "rule": c.key_check.rule.value if c.key_check.rule else None,
            "served": (
                "dropped: " + str(c.drop_reason)
                if c.drop_reason is not None
                else (c.environment.media.name if c.environment else None)
            ),
            "records_affected": h.count_records(rows.rows, {c.filename: [c]}),
        }
        for c in columns
        if c.key_check.outcome is h.KeyHeaderOutcome.conflict
    ]
    minimal = [
        {
            "filename": c.filename,
            "header": c.header,
            "key_condition": c.key_check.key_condition,
            "old_medium": "SD",
            "new_medium": (
                c.environment.media.name if c.environment is not None else None
            ),
            "supplement_gap": (
                c.environment is not None
                and "auxotroph_supplements" in c.environment.gapped_fields()
            ),
            "drop_reason": c.drop_reason,
        }
        for c in columns
        if c.header.split(":")[1].strip().lower() == "minimal media"
    ]
    zero_gen = [
        {
            "filename": c.filename,
            "header": c.header,
            "records_not_written": h.count_records(rows.rows, {c.filename: [c]}),
        }
        for c in columns
        if c.drop_reason == h.DROP_ZERO_GENERATIONS
    ]

    markers: dict[str, Any] = {}
    for orf, name in MARKER_NAMES.items():
        marker_rows = [r for r in rows.rows if r.orf == orf]
        doses = []
        for row in marker_rows:
            pert = h.strain_perturbation(
                arm, row, "pool", None, h.marker_loci(background)
            )
            doses.append(
                heterozygous_deletion_functional_copies(background, pert)
                if isinstance(pert, HeterozygousDeletionPerturbation)
                else None
            )
        markers[orf] = {
            "gene": name,
            "rows": [r.row_id for r in marker_rows],
            "records": h.count_records(marker_rows, groups),
            "background_functional_copies": background.functional_copies(orf),
            "het_functional_copies_after_deletion": doses if arm == "het" else None,
            "old_encoding": (
                "copy_number 1 of reference_copy_number 2 (functional_dose 0.5)"
                if arm == "het"
                else "KanMxDeletionPerturbation of a present gene"
            ),
            "gapped_fields": sorted(
                h.strain_perturbation(
                    arm, marker_rows[0], "pool", None, h.marker_loci(background)
                ).gapped_fields()
            )
            if marker_rows
            else [],
        }

    dropped_strains: dict[str, Any] = {}
    for strain in rows.dropped_strains:
        bucket = dropped_strains.setdefault(strain.rule, {"rows": [], "records": 0})
        bucket["rows"].append(strain.row_id)
        bucket["records"] += h.count_records([strain], groups)

    return {
        "arrays": len(columns),
        "kept_arrays": sum(len(m) for m in groups.values()),
        "kept_environment_groups": len(groups),
        "dropped_arrays_by_rule": dict(
            Counter(str(c.drop_reason) for c in columns if c.drop_reason)
        ),
        "dropped_records_by_rule": {
            rule: sum(
                h.count_records(rows.rows, {k: m})
                for k, m in dropped_groups.items()
                if m[0].drop_reason == rule
            )
            for rule in sorted({str(m[0].drop_reason) for m in dropped_groups.values()})
        },
        "key_header_outcomes": dict(
            Counter(c.key_check.outcome.value for c in columns)
        ),
        "n_conflicts": len(conflicts),
        "conflicts": conflicts,
        "kept_rows": len(rows.rows),
        "kept_current_genes": len({r.orf for r in rows.rows}),
        "dropped_source_orfs": len(rows.dropped_genes),
        "new_record_count": n_new,
        "old_record_count_dev_store": _old_count(arm, data_root),
        "records_changing_class": n_new,
        "renamed_source_orfs": len(rows.constructed),
        "merged_relation_source_orfs": sum(
            1 for c in rows.constructed.values() if c.relation is not None
        ),
        "n_genes_fed_by_two_source_orfs": len(merged),
        "genes_fed_by_two_source_orfs": merged,
        "records_per_source_orf_of_those_genes": merged_records,
        "n_source_orfs_built_in_two_or_more_batches": len(multi_batch),
        "rows_of_those_source_orfs": sum(multi_batch.values()),
        "minimal_media_arrays": minimal,
        "zero_generation_arrays": zero_gen,
        "zero_generation_records_not_written": sum(
            z["records_not_written"] for z in zero_gen
        ),
        "marker_loci": markers,
        "dropped_strains": dropped_strains,
        "suspicious_batches": _suspicious(arm, data_root, rows),
    }


def sliced_build(
    arm: str, data_root: str, resolve_name: Any, scratch_root: str
) -> dict[str, Any]:
    """Run the record builders on SLICE_ARRAYS x SLICE_ORFS into the scratch root."""
    from torchcell.adapters.cell_adapter import CellAdapter
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset,
    )
    from torchcell.verification.report import Provenance

    columns, rows = _arm_inputs(arm, data_root, resolve_name)
    spec = h.MATRICES[arm]
    raw = h.raw_mirror_dir(data_root) / "data"
    sizes = h.read_control_set_sizes(raw / spec.controls_file)
    chosen = [c for c in columns if c.filename in SLICE_ARRAYS[arm]]
    groups, dropped = h.group_columns(chosen)
    sliced = h.MatrixRows(
        rows=[r for r in rows.rows if r.source_orf in SLICE_ORFS],
        dropped_strains=[],
        dropped_genes={},
        constructed=rows.constructed,
    )
    background = h.hillenmeyer_background()
    name = f"{arm.capitalize()}Hillenmeyer2008Dataset"
    references = {
        m[0].control_set: h.build_reference(
            name, spec, m[0].control_set, sizes[m[0].control_set], background
        )
        for m in groups.values()
    }
    records: list[dict[str, Any]] = []
    for experiment, control_set in h.iter_records(
        name, spec, sliced, groups, background
    ):
        records.append(
            {
                "experiment": experiment.model_dump(),
                "reference": references[control_set].model_dump(),
                "publication": {"doi": h.DOI},
            }
        )
    for record in records:  # round-trip: the stored dump re-validates to the class
        StrainEnvironmentResponseExperiment.model_validate(record["experiment"])
        StrainEnvironmentResponseExperimentReference.model_validate(record["reference"])
    out_dir = osp.join(scratch_root, f"env_chemgen_hillenmeyer2008_{arm}_slice")
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "records.json"), "w") as handle:
        json.dump(records, handle, indent=1, default=str)
    report = verify_environment_response_dataset(
        records,
        dataset_name=f"env_chemgen_hillenmeyer2008_{arm}_slice",
        provenance=Provenance(source_uri=str(raw / spec.filename)),
        expected_count=len(records),
        sgd_genes=h._load_sgd_genes(data_root),
        resolve_gene_name=resolve_name,
    )
    adapter = CellAdapter.__new__(CellAdapter)
    first = StrainEnvironmentResponseExperiment.model_validate(records[0]["experiment"])
    nodes = CellAdapter._perturbation_node.__wrapped__(  # type: ignore[attr-defined]
        adapter, {"experiment": first}, "perturbation (chunked)"
    )
    experiment_nodes = CellAdapter._experiment_node.__wrapped__(  # type: ignore[attr-defined]
        adapter, {"experiment": first}, "experiment (chunked)"
    )
    return {
        "arrays": [c.filename for c in chosen],
        "dropped_arrays": {
            c.filename: c.drop_reason for m in dropped.values() for c in m
        },
        "rows": [r.row_id for r in sliced.rows],
        "n_records": len(records),
        "records_path": osp.join(out_dir, "records.json"),
        "perturbation_types": dict(
            Counter(
                p["perturbation_type"]
                for r in records
                for p in r["experiment"]["genotype"]["perturbations"]
            )
        ),
        "media": dict(
            Counter(r["experiment"]["environment"]["media"]["name"] for r in records)
        ),
        "pre_culture_sources": dict(
            Counter(
                r["experiment"]["environment"]["pre_culture"]["source"] for r in records
            )
        ),
        "constructed_orfs": sorted(
            {
                json.dumps(p["constructed_orf"]["source_systematic_name"])
                + ":"
                + str(p["constructed_orf"]["relation"])
                for r in records
                for p in r["experiment"]["genotype"]["perturbations"]
                if p.get("constructed_orf")
            }
        ),
        "verifier": [
            {
                "level": x.level.value,
                "name": x.name,
                "passed": x.passed,
                "message": x.message,
            }
            for x in report.results
        ],
        "adapter_perturbation_node_types": [
            n.get_properties()["perturbation_type"] for n in nodes
        ],
        "adapter_experiment_node_labels": [n.get_label() for n in experiment_nodes],
    }


def dev_lmdb(data_root: str) -> dict[str, Any]:
    """Stream the rebuilt dev stores and count what the fix put on the records."""
    from torchcell.verification.runners import stream_records

    out: dict[str, Any] = {}
    for arm in ("het", "hom"):
        root = osp.join(data_root, f"data/torchcell/env_chemgen_hillenmeyer2008_{arm}")
        types: Counter[str] = Counter()
        relations: Counter[str] = Counter()
        pre: Counter[str] = Counter()
        media: Counter[str] = Counter()
        markers: Counter[str] = Counter()
        n = 0
        for record in stream_records(root):
            n += 1
            experiment = record["experiment"]
            for p in experiment["genotype"]["perturbations"]:
                types[p["perturbation_type"]] += 1
                constructed = p.get("constructed_orf")
                if constructed:
                    relations[str(constructed["relation"])] += 1
                if p["systematic_gene_name"] in MARKER_NAMES:
                    markers[p["systematic_gene_name"]] += 1
            environment = experiment["environment"]
            pre[str((environment.get("pre_culture") or {}).get("source"))] += 1
            media[environment["media"]["name"]] += 1
        out[arm] = {
            "n_records": n,
            "perturbation_types": dict(types),
            "constructed_orf_relations": dict(relations),
            "pre_culture_sources": dict(pre),
            "media": dict(media),
            "marker_locus_records": dict(markers),
        }
    return out


def main() -> None:
    """Measure both arms (or the rebuilt dev stores) and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-lmdb", action="store_true")
    parser.add_argument(
        "--scratch-root",
        help="directory the sliced build writes its records to (required without "
        "--dev-lmdb)",
    )
    args = parser.parse_args()
    if not args.dev_lmdb and args.scratch_root is None:
        parser.error("--scratch-root is required for the sliced build")
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    os.makedirs(RESULTS, exist_ok=True)
    if args.dev_lmdb:
        counts = dev_lmdb(data_root)
        with open(osp.join(RESULTS, "hillenmeyer2008_inputs_dev_lmdb.json"), "w") as f:
            json.dump(counts, f, indent=2)
        print(json.dumps(counts, indent=2))
        return
    resolve_name = _genome(data_root).resolve_gene_name
    result: dict[str, Any] = {"arms": {}, "slice": {}}
    for arm in ("het", "hom"):
        result["arms"][arm] = measure_arm(arm, data_root, resolve_name)
        result["slice"][arm] = sliced_build(
            arm, data_root, resolve_name, args.scratch_root
        )
    with open(osp.join(RESULTS, "hillenmeyer2008_inputs.json"), "w") as handle:
        json.dump(result, handle, indent=2, default=str)
    with open(
        osp.join(RESULTS, "hillenmeyer2008_key_header_conflicts.csv"), "w", newline=""
    ) as handle:
        rows = [c for arm in result["arms"].values() for c in arm["conflicts"]]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        arm: {
            k: v
            for k, v in data.items()
            if k
            in {
                "new_record_count",
                "old_record_count_dev_store",
                "n_conflicts",
                "n_genes_fed_by_two_source_orfs",
                "zero_generation_records_not_written",
                "dropped_strains",
                "suspicious_batches",
                "dropped_records_by_rule",
            }
        }
        for arm, data in result["arms"].items()
    }
    print(json.dumps(summary, indent=2))
    print(json.dumps(result["slice"], indent=2, default=str))


if __name__ == "__main__":
    main()
