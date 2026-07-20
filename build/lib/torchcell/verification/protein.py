# torchcell/verification/protein
# [[torchcell.verification.protein]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/protein
"""L0-L4 record-level verifier for protein-abundance datasets (WS9; Zelezniak2018).

For `ProteinAbundancePhenotype` datasets (e.g. Zelezniak SWATH-MS of kinase knockouts).
The schema validator already enforces non-empty abundances, matching abundance/replicate
keys, and non-negative SE, so L0 subsumes those. This verifier adds:

1. L1 `orf_uniqueness` -- one record per deleted (knocked-out) ORF.
2. L2 `value_fidelity` -- measured abundances are finite (guards inf/NaN).
3. L2 `se_nonnegative` -- released SE values are finite + non-negative.
4. L3 `reference_finite` -- the WT reference abundance is finite + key-matched to the
   experiment (absolute abundances, not a centered 0).
5. L3 `measurement_type_consistent` -- every record shares one measurement_type.
6. L4 `gene_containment` (caller) -- the screened knockouts overlap the deletion
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
    """Systematic names knocked out in this experiment's genotype."""
    return [
        p["systematic_gene_name"]
        for p in experiment["genotype"]["perturbations"]
        if p.get("systematic_gene_name") is not None
    ]


def _l1_orf_uniqueness(
    records: Sequence[Record], allow_duplicate_orfs: bool = False
) -> LevelResult:
    """L1: one record per knocked-out ORF.

    Some datasets intentionally hold multiple deletion strains for the same ORF (e.g.
    Messner 2023 has 145 ORFs with 2-3 strains "of different origins", each analysed
    independently). For those, ``allow_duplicate_orfs=True`` turns this into an
    informational check (duplicates reported, not failed) -- the per-instance
    uniqueness is guaranteed by the record count (L1 ``expected_count``).
    """
    seen: dict[str, int] = {}
    for rec in records:
        for g in _deleted_genes(rec["experiment"]):
            seen[g] = seen.get(g, 0) + 1
    dups = {g: n for g, n in seen.items() if n > 1}
    passed = allow_duplicate_orfs or not dups
    if not dups:
        message = f"{len(seen)} unique knocked-out ORFs, one record each"
    elif allow_duplicate_orfs:
        message = f"{len(seen)} ORFs, {len(dups)} with multiple strains (expected)"
    else:
        message = f"{len(dups)} ORFs appear in multiple records"
    return LevelResult(
        level=Level.L1,
        name="orf_uniqueness",
        passed=passed,
        message=message,
        details={"n_orfs": len(seen), "n_duplicated": len(dups)},
    )


def _l3_reference_finite(records: Sequence[Record]) -> LevelResult:
    """L3: the WT reference abundance is finite + key-matched for every protein."""
    n = 0
    bad = 0
    for rec in records:
        exp_keys = set(rec["experiment"]["phenotype"]["protein_abundance"])
        levels = rec["reference"]["phenotype_reference"]["protein_abundance"]
        if set(levels) != exp_keys:
            bad += 1
            continue
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
            f"reference abundance finite + key-matched for all {n} values"
            if holds
            else f"{bad} reference abundances non-finite or key-mismatched"
        ),
        details={"n_values": n, "n_bad": bad},
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


def verify_protein_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    allow_duplicate_orfs: bool = False,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for a protein-abundance dataset.

    L4 (cross-source gene overlap with the deletion collection) is asserted by the
    caller across datasets. ``allow_duplicate_orfs`` relaxes L1 uniqueness for
    datasets that intentionally hold multiple strains per ORF (e.g. Messner 2023).
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_orf_uniqueness(records, allow_duplicate_orfs=allow_duplicate_orfs))

    levels = [
        float(v)
        for rec in records
        for v in rec["experiment"]["phenotype"]["protein_abundance"].values()
    ]
    report.add(l2_value_fidelity(levels, allow_nan=False))

    se_values = [
        float(v)
        for rec in records
        for v in (
            rec["experiment"]["phenotype"].get("protein_abundance_se") or {}
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

    report.add(_l3_reference_finite(records))
    report.add(_l3_measurement_type_consistent(records))
    return report


def protein_gene_set(records: Sequence[Record]) -> set[str]:
    """Union of knocked-out systematic gene names across records (the L4 overlap key)."""
    genes: set[str] = set()
    for rec in records:
        genes.update(_deleted_genes(rec["experiment"]))
    return genes


__all__ = ["verify_protein_dataset", "protein_gene_set", "Record"]
