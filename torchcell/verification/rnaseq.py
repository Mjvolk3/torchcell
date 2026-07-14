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

import json
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


def _expr_map(phenotype: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return (per-gene expression dict, family kind) for either expression phenotype.

    ``RNASeqExpressionPhenotype`` (Caudal) stores absolute ``expression_tpm``;
    ``PseudobulkExpressionPhenotype`` (Nadal-Ribelles) stores per-gene log2 fold-change vs
    WT in ``expression_log2_ratio``. One verifier serves both families.
    """
    if "expression_log2_ratio" in phenotype:
        return phenotype["expression_log2_ratio"], "log2_ratio"
    return phenotype["expression_tpm"], "tpm"


def _env_identity(environment: dict[str, Any]) -> str:
    """Stable condition identity of an environment (media/temp/perturbations/duration)."""
    return json.dumps(environment, sort_keys=True, default=str)


def _record_strain(experiment: dict[str, Any]) -> str | None:
    """Return the strain id of a record (the shared ``strain_id`` of its perturbations)."""
    for pert in experiment["genotype"]["perturbations"]:
        strain = pert.get("strain_id")
        if strain is not None:
            return str(strain)
    return None


def _l1_strain_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: one record per (strain, environment) and every record carries a strain id.

    Keyed on (strain_id, environment) so a strain profiled in two conditions (Nadal-Ribelles
    control vs NaCl) is two legitimate records, while a genome-scale single-condition survey
    (Caudal, one environment) still reduces to one record per strain.
    """
    seen: dict[tuple[str, str], int] = {}
    n_missing = 0
    for rec in records:
        exp = rec["experiment"]
        strain = _record_strain(exp)
        if strain is None:
            n_missing += 1
            continue
        key = (strain, _env_identity(exp["environment"]))
        seen[key] = seen.get(key, 0) + 1
    dups = {k: n for k, n in seen.items() if n > 1}
    n_strains = len({k[0] for k in seen})
    passed = not dups and n_missing == 0
    return LevelResult(
        level=Level.L1,
        name="strain_uniqueness",
        passed=passed,
        message=(
            f"{len(seen)} unique (strain, condition) records over {n_strains} strains"
            if passed
            else f"{len(dups)} (strain, condition) duplicated, "
            f"{n_missing} records without a strain"
        ),
        details={
            "n_records": len(seen),
            "n_strains": n_strains,
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
    """L3: every reference expression value is finite (baseline is well-defined)."""
    n = 0
    bad = 0
    for rec in records:
        levels, _ = _expr_map(rec["reference"]["phenotype_reference"])
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
            f"reference expression finite for all {n} values"
            if holds
            else f"{bad}/{n} reference expression values non-finite"
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

    kind = _expr_map(records[0]["experiment"]["phenotype"])[1] if records else "tpm"
    expr_values = [
        float(v)
        for rec in records
        for v in _expr_map(rec["experiment"]["phenotype"])[0].values()
    ]
    if kind == "tpm":
        # Absolute TPM: finite and non-negative.
        fidelity = l2_value_fidelity(expr_values, allow_nan=False, minimum=0.0)
        fidelity_name = "tpm_value_fidelity"
    else:
        # Pseudobulk log2 fold-change vs WT: finite, negatives allowed (down-regulation).
        fidelity = l2_value_fidelity(expr_values, allow_nan=False)
        fidelity_name = "log2_ratio_value_fidelity"
    report.add(
        LevelResult(
            level=Level.L2,
            name=fidelity_name,
            passed=fidelity.passed,
            message=fidelity.message,
            details=fidelity.details,
        )
    )

    # Raw integer counts exist only for the absolute-TPM family.
    if kind == "tpm":
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
        expr, _ = _expr_map(rec["experiment"]["phenotype"])
        genes.update(expr.keys())
    return genes


__all__ = ["verify_rnaseq_dataset", "rnaseq_gene_set", "Record"]
