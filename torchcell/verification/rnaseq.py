# torchcell/verification/rnaseq
# [[torchcell.verification.rnaseq]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/rnaseq
"""L0-L4 record-level verifier for RNA-seq expression datasets (roadmap WS10; Caudal2024).

For ``RNASeqExpressionPhenotype`` datasets (the Caudal natural-isolate pan-transcriptome),
where each isolate stores ABSOLUTE per-gene ``expression_tpm`` + ``expression_count`` on its
own genome. The schema validator already enforces non-empty, non-negative, key-matched
maps, so L0 subsumes those. This verifier adds:

1. L1 ``strain_uniqueness`` -- one record per isolate (each isolate's perturbations all
   carry one ``strain_id``; that id is unique across records).
2. L2 ``tpm_value_fidelity`` -- every TPM is finite and >= 0.
3. L2 ``count_value_fidelity`` -- every raw count is a non-negative integer.
4. L3 ``measurement_type_consistent`` -- one shared measurement_type (no cross-assay mix).
5. L3 ``reference_finite`` -- the shared population-mean reference TPMs are all finite.
6. L4 ``gene_containment`` (caller) -- the measured gene universe is contained in the S288C
   reference gene set.

The verifier operates purely on the pydantic/LMDB records -- no graph (Phase A).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from torchcell.verification.levels import l0_structural, l1_count, l2_value_fidelity
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

Record = dict[str, Any]


def _record_strain(experiment: dict[str, Any]) -> str | None:
    """Return the isolate id of a record (the shared ``strain_id`` of its perturbations)."""
    for pert in experiment["genotype"]["perturbations"]:
        strain = pert.get("strain_id")
        if strain is not None:
            return str(strain)
    return None


def _l1_strain_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: exactly one record per isolate (strain_id) and every record has a strain."""
    seen: dict[str, int] = {}
    n_missing = 0
    for rec in records:
        strain = _record_strain(rec["experiment"])
        if strain is None:
            n_missing += 1
            continue
        seen[strain] = seen.get(strain, 0) + 1
    dups = {s: n for s, n in seen.items() if n > 1}
    passed = not dups and n_missing == 0
    return LevelResult(
        level=Level.L1,
        name="strain_uniqueness",
        passed=passed,
        message=(
            f"{len(seen)} unique isolates, one record each"
            if passed
            else f"{len(dups)} isolates duplicated, {n_missing} records without a strain"
        ),
        details={
            "n_strains": len(seen),
            "n_duplicated": len(dups),
            "n_missing": n_missing,
        },
    )


def _l3_measurement_type_consistent(records: Sequence[Record]) -> LevelResult:
    """L3: all records share a single measurement_type (no silent cross-assay mixing)."""
    types = {rec["experiment"]["phenotype"]["measurement_type"] for rec in records}
    return LevelResult(
        level=Level.L3,
        name="measurement_type_consistent",
        passed=len(types) <= 1,
        message=(
            f"single measurement_type: {next(iter(types), None)!r}"
            if len(types) <= 1
            else f"{len(types)} distinct measurement_types mixed: {sorted(types)}"
        ),
        details={"measurement_types": sorted(types)},
    )


def _l3_reference_finite(records: Sequence[Record]) -> LevelResult:
    """L3: every reference TPM is finite (the population-mean baseline is well-defined)."""
    n = 0
    bad = 0
    for rec in records:
        levels = rec["reference"]["phenotype_reference"]["expression_tpm"]
        for v in levels.values():
            n += 1
            if not math.isfinite(float(v)):
                bad += 1
    holds = bad == 0
    return LevelResult(
        level=Level.L3,
        name="reference_finite",
        passed=holds,
        message=(
            f"reference TPM finite for all {n} values"
            if holds
            else f"{bad}/{n} reference TPM values non-finite"
        ),
        details={"n_values": n, "n_bad": bad},
    )


def verify_rnaseq_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for an RNA-seq expression dataset.

    L4 (gene containment vs the S288C reference gene set) is asserted by the caller.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_strain_uniqueness(records))

    tpm_values = [
        float(v)
        for rec in records
        for v in rec["experiment"]["phenotype"]["expression_tpm"].values()
    ]
    tpm_result = l2_value_fidelity(tpm_values, allow_nan=False, minimum=0.0)
    report.add(
        LevelResult(
            level=Level.L2,
            name="tpm_value_fidelity",
            passed=tpm_result.passed,
            message=tpm_result.message,
            details=tpm_result.details,
        )
    )

    count_bad = 0
    count_n = 0
    for rec in records:
        for v in rec["experiment"]["phenotype"]["expression_count"].values():
            count_n += 1
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                count_bad += 1
    report.add(
        LevelResult(
            level=Level.L2,
            name="count_value_fidelity",
            passed=count_bad == 0,
            message=(
                f"{count_n} counts are non-negative integers"
                if count_bad == 0
                else f"{count_bad}/{count_n} counts are not non-negative integers"
            ),
            details={"n_values": count_n, "n_bad": count_bad},
        )
    )

    report.add(_l3_measurement_type_consistent(records))
    report.add(_l3_reference_finite(records))
    return report


def rnaseq_gene_set(records: Sequence[Record]) -> set[str]:
    """Union of measured expression genes across records (the L4 containment key)."""
    genes: set[str] = set()
    for rec in records:
        genes.update(rec["experiment"]["phenotype"]["expression_tpm"].keys())
    return genes


__all__ = ["verify_rnaseq_dataset", "rnaseq_gene_set", "Record"]
