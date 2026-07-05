# torchcell/verification/metabolite
# [[torchcell.verification.metabolite]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/metabolite
"""L0-L4 record-level verifier for metabolite datasets (roadmap WS8; Cachera2023).

For `MetabolitePhenotype` datasets (e.g. the Cachera CRI-SPA betaxanthin screen). The
schema validator already enforces non-empty levels, matching level/replicate keys, and
non-negative SE, so L0 subsumes those. This verifier adds:

1. L1 `orf_uniqueness` -- one record per deleted ORF.
2. L2 `level_finiteness` -- measured levels are finite (guards inf/NaN).
3. L3 `reference_zero` -- the control level is 0 (population-centered baseline).
4. L3 `measurement_type_consistent` -- every record shares one measurement_type (levels
   from different assays must not be silently mixed).
5. L4 `gene_containment` (caller) -- the screened deletions overlap the yeast deletion
   collection used by the other datasets.
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


def _deleted_genes(experiment: dict[str, Any]) -> list[str]:
    """Systematic names deleted in this experiment's genotype."""
    return [
        p["systematic_gene_name"]
        for p in experiment["genotype"]["perturbations"]
        if p.get("systematic_gene_name") is not None
    ]


def _l1_orf_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: exactly one record per deleted ORF."""
    seen: dict[str, int] = {}
    for rec in records:
        for g in _deleted_genes(rec["experiment"]):
            seen[g] = seen.get(g, 0) + 1
    dups = {g: n for g, n in seen.items() if n > 1}
    return LevelResult(
        level=Level.L1,
        name="orf_uniqueness",
        passed=not dups,
        message=(
            f"{len(seen)} unique deleted ORFs, one record each"
            if not dups
            else f"{len(dups)} ORFs appear in multiple records"
        ),
        details={"n_orfs": len(seen), "n_duplicated": len(dups)},
    )


def _l3_reference_zero(records: Sequence[Record]) -> LevelResult:
    """L3: the reference (control) metabolite level is 0 (centered baseline)."""
    worst = 0.0
    n = 0
    for rec in records:
        levels = rec["reference"]["phenotype_reference"]["metabolite_level"]
        for v in levels.values():
            n += 1
            worst = max(worst, abs(float(v)))
    holds = worst == 0.0
    return LevelResult(
        level=Level.L3,
        name="reference_zero",
        passed=holds,
        message=(
            f"reference metabolite level == 0 for all {n} values"
            if holds
            else f"reference level not identically 0: max|v|={worst:.3g}"
        ),
        details={"n_values": n, "worst_abs": worst},
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


def verify_metabolite_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for a metabolite dataset.

    L4 (cross-source gene overlap with the deletion collection) is asserted by the
    caller across datasets.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_orf_uniqueness(records))

    levels = [
        float(v)
        for rec in records
        for v in rec["experiment"]["phenotype"]["metabolite_level"].values()
    ]
    report.add(l2_value_fidelity(levels, allow_nan=False))

    # SE may be NaN where a metabolite has a single replicate; never negative.
    se_values = [
        float(v)
        for rec in records
        for v in (
            rec["experiment"]["phenotype"].get("metabolite_level_se") or {}
        ).values()
        if not (isinstance(v, float) and math.isnan(v))
    ]
    se_result = l2_value_fidelity(se_values, allow_nan=False, minimum=0.0)
    report.add(
        LevelResult(
            level=Level.L2,
            name="se_nonnegative",
            passed=se_result.passed,
            message=se_result.message,
            details=se_result.details,
        )
    )

    report.add(_l3_reference_zero(records))
    report.add(_l3_measurement_type_consistent(records))
    return report


def metabolite_gene_set(records: Sequence[Record]) -> set[str]:
    """Union of deleted systematic gene names across records (the L4 overlap key)."""
    genes: set[str] = set()
    for rec in records:
        genes.update(_deleted_genes(rec["experiment"]))
    return genes


__all__ = ["verify_metabolite_dataset", "metabolite_gene_set", "Record"]
