# torchcell/verification/environment_response
# [[torchcell.verification.environment_response]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/environment_response
"""L0-L4 record-level verifier for environment-response datasets (WS15).

For ``EnvironmentResponsePhenotype`` datasets: chemical-genomic / stress screens read
out as ``(deletion genotype x EnvironmentPerturbation -> response)`` records (e.g. the
Vanacloig 2022 anaerobic hydrolysate-toxin bar-seq screen). The schema validator already
enforces numeric-vs-categorical coherence and the uncertainty invariant (subsumed by L0);
this verifier adds:

1. L1 ``count`` -- exact record-count oracle.
2. L1 ``pair_uniqueness`` -- one record per (screened deletion x compound) pair.
3. L2 ``response_finiteness`` -- numeric responses are finite (SIGNED; negatives allowed).
4. L2 ``se_nonnegative`` -- reported SEs are non-negative.
5. L3 ``measurement_type_consistent`` -- one measurement_type across the dataset.
6. L3 ``reference_zero`` -- the reference (parent-strain) response is 0.
7. L3 ``environment_perturbed`` -- every experiment carries >= 1 environment perturbation.
8. L4 ``gene_containment`` (caller) -- screened deletions overlap the deletion collection.
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


def _screened_genes(
    experiment: dict[str, Any], background: frozenset[str]
) -> list[str]:
    """Systematic names deleted in this experiment's genotype, minus the constant
    drug-sensitized background (which is identical in every strain and is not a
    screened deletion, so it must not enter the L1/L4 deleted-ORF key).
    """
    return [
        p["systematic_gene_name"]
        for p in experiment["genotype"]["perturbations"]
        if p.get("systematic_gene_name") is not None
        and p["systematic_gene_name"] not in background
    ]


def _compound_name(experiment: dict[str, Any]) -> str | None:
    """The first environment perturbation's compound/agent name, if any."""
    perts = experiment["environment"].get("perturbations") or []
    if not perts:
        return None
    p = perts[0]
    name = p.get("compound_name") or p.get("agent") or p.get("stress_type")
    return None if name is None else str(name)


def _l1_pair_uniqueness(
    records: Sequence[Record], background: frozenset[str]
) -> LevelResult:
    """L1: exactly one record per (screened deletion ORF, compound) pair."""
    seen: dict[tuple[str, str | None], int] = {}
    for rec in records:
        exp = rec["experiment"]
        comp = _compound_name(exp)
        for g in _screened_genes(exp, background):
            key = (g, comp)
            seen[key] = seen.get(key, 0) + 1
    dups = {k: n for k, n in seen.items() if n > 1}
    return LevelResult(
        level=Level.L1,
        name="pair_uniqueness",
        passed=not dups,
        message=(
            f"{len(seen)} unique (ORF, compound) pairs, one record each"
            if not dups
            else f"{len(dups)} (ORF, compound) pairs appear in multiple records"
        ),
        details={"n_pairs": len(seen), "n_duplicated": len(dups)},
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


def _l3_reference_zero(records: Sequence[Record]) -> LevelResult:
    """L3: the reference (parent-strain) response is 0 -- log2(1)=0, the control baseline."""
    worst = 0.0
    n = 0
    for rec in records:
        v = rec["reference"]["phenotype_reference"]["environment_response"]
        if v is None:
            continue
        n += 1
        worst = max(worst, abs(float(v)))
    holds = worst == 0.0
    return LevelResult(
        level=Level.L3,
        name="reference_zero",
        passed=holds,
        message=(
            f"reference response == 0 for all {n} records"
            if holds
            else f"reference response not identically 0: max|v|={worst:.3g}"
        ),
        details={"n_values": n, "worst_abs": worst},
    )


def _l3_environment_perturbed(records: Sequence[Record]) -> LevelResult:
    """L3: every experiment carries at least one environment perturbation."""
    n_missing = sum(
        1
        for rec in records
        if not (rec["experiment"]["environment"].get("perturbations") or [])
    )
    return LevelResult(
        level=Level.L3,
        name="environment_perturbed",
        passed=n_missing == 0,
        message=(
            f"all {len(records)} experiments carry >= 1 environment perturbation"
            if n_missing == 0
            else f"{n_missing} experiments have no environment perturbation"
        ),
        details={"n_records": len(records), "n_missing": n_missing},
    )


def verify_environment_response_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    background_genes: frozenset[str] = frozenset(),
) -> VerificationReport:
    """Run the L0-L3 record-level gate for an environment-response dataset.

    ``background_genes`` are the systematic names of the constant drug-sensitized
    background (e.g. Vanacloig 3DeltaAlpha = PDR1/PDR3/SNQ2), excluded from the
    (ORF, compound) uniqueness and gene-set keys. L4 (cross-source gene overlap with the
    deletion collection) is asserted by the caller.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_pair_uniqueness(records, background_genes))

    responses = [
        float(rec["experiment"]["phenotype"]["environment_response"])
        for rec in records
        if rec["experiment"]["phenotype"]["environment_response"] is not None
    ]
    report.add(l2_value_fidelity(responses, allow_nan=False))

    se_values = [
        float(v)
        for rec in records
        if (v := rec["experiment"]["phenotype"].get("environment_response_se"))
        is not None
        and not (isinstance(v, float) and math.isnan(v))
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

    report.add(_l3_measurement_type_consistent(records))
    report.add(_l3_reference_zero(records))
    report.add(_l3_environment_perturbed(records))
    return report


def environment_response_gene_set(
    records: Sequence[Record], background_genes: frozenset[str] = frozenset()
) -> set[str]:
    """Union of screened (non-background) deleted gene names -- the L4 overlap key."""
    genes: set[str] = set()
    for rec in records:
        genes.update(_screened_genes(rec["experiment"], background_genes))
    return genes


__all__ = [
    "verify_environment_response_dataset",
    "environment_response_gene_set",
    "Record",
]
