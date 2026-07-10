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
import sys
from collections.abc import Callable, Iterable, Sequence
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


def verify_environment_response_dataset_streaming(
    records: Iterable[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    sgd_genes: set[str],
    background_genes: frozenset[str] = frozenset(),
    min_containment: float = 0.90,
) -> VerificationReport:
    """Single-pass, memory-bounded L0-L4 gate for LARGE environment-response datasets.

    Semantically identical to ``verify_environment_response_dataset`` (+ the caller's L4
    gene-containment), but consumes ``records`` as a stream so a 30M-record dataset (e.g.
    the Hoepfner HIP-HOP atlas) never has to be materialized in RAM. The dominant
    accumulator is the (ORF, compound) pair set; interning keeps it a few GB, not the
    ~450 GB a full materialization would cost.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    n_records = 0
    l0_failures: list[dict[str, Any]] = []
    pair_seen: set[tuple[str, str | None]] = set()
    n_pairs = 0
    n_pair_dups = 0
    n_responses = 0
    bad_responses: list[dict[str, Any]] = []
    n_se = 0
    bad_se: list[dict[str, Any]] = []
    measurement_types: set[str] = set()
    ref_worst = 0.0
    n_ref = 0
    n_env_missing = 0
    screened: set[str] = set()

    for i, rec in enumerate(records):
        exp = rec["experiment"]
        n_records += 1
        try:
            validate(exp)
        except (ValueError, TypeError) as err:
            l0_failures.append({"index": i, "error": str(err)[:500]})

        comp = _compound_name(exp)
        comp_key = None if comp is None else sys.intern(comp)
        for gene in _screened_genes(exp, background_genes):
            gene = sys.intern(gene)
            screened.add(gene)
            key = (gene, comp_key)
            if key in pair_seen:
                n_pair_dups += 1
            else:
                pair_seen.add(key)
                n_pairs += 1

        response = exp["phenotype"]["environment_response"]
        if response is not None:
            n_responses += 1
            if math.isnan(response) or math.isinf(response):
                bad_responses.append({"index": i, "value": repr(response)})
        se = exp["phenotype"].get("environment_response_se")
        if se is not None and not (isinstance(se, float) and math.isnan(se)):
            n_se += 1
            if se < 0.0:
                bad_se.append({"index": i, "value": se})
        measurement_types.add(exp["phenotype"]["measurement_type"])

        ref_val = rec["reference"]["phenotype_reference"]["environment_response"]
        if ref_val is not None:
            n_ref += 1
            ref_worst = max(ref_worst, abs(float(ref_val)))
        if not (exp["environment"].get("perturbations") or []):
            n_env_missing += 1

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(
        LevelResult(
            level=Level.L0,
            name="structural",
            passed=not l0_failures,
            message=(
                f"{n_records} records validated"
                if not l0_failures
                else f"{len(l0_failures)}/{n_records} records failed schema validation"
            ),
            details={
                "n_records": n_records,
                "n_failures": len(l0_failures),
                "failures": l0_failures[:10],
            },
        )
    )
    report.add(
        LevelResult(
            level=Level.L1,
            name="count",
            passed=n_records == expected_count,
            message=f"observed {n_records}, expected {expected_count}",
            details={"observed": n_records, "expected": expected_count},
        )
    )
    report.add(
        LevelResult(
            level=Level.L1,
            name="pair_uniqueness",
            passed=n_pair_dups == 0,
            message=(
                f"{n_pairs} unique (ORF, compound) pairs, one record each"
                if n_pair_dups == 0
                else f"{n_pair_dups} (ORF, compound) records duplicate an existing pair"
            ),
            details={"n_pairs": n_pairs, "n_duplicated": n_pair_dups},
        )
    )
    report.add(
        LevelResult(
            level=Level.L2,
            name="value_fidelity",
            passed=not bad_responses,
            message=(
                f"{n_responses} values checked"
                if not bad_responses
                else f"{len(bad_responses)}/{n_responses} values invalid"
            ),
            details={
                "n_values": n_responses,
                "n_bad": len(bad_responses),
                "bad": bad_responses[:20],
            },
        )
    )
    report.add(
        LevelResult(
            level=Level.L2,
            name="se_nonnegative",
            passed=not bad_se,
            message=(
                f"{n_se} values checked"
                if not bad_se
                else f"{len(bad_se)}/{n_se} values invalid"
            ),
            details={"n_values": n_se, "n_bad": len(bad_se), "bad": bad_se[:20]},
        )
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="measurement_type_consistent",
            passed=len(measurement_types) <= 1,
            message=(
                f"single measurement_type: {next(iter(measurement_types), None)!r}"
                if len(measurement_types) <= 1
                else f"{len(measurement_types)} distinct measurement_types mixed: "
                f"{sorted(measurement_types)}"
            ),
            details={"measurement_types": sorted(measurement_types)},
        )
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="reference_zero",
            passed=ref_worst == 0.0,
            message=(
                f"reference response == 0 for all {n_ref} records"
                if ref_worst == 0.0
                else f"reference response not identically 0: max|v|={ref_worst:.3g}"
            ),
            details={"n_values": n_ref, "worst_abs": ref_worst},
        )
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="environment_perturbed",
            passed=n_env_missing == 0,
            message=(
                f"all {n_records} experiments carry >= 1 environment perturbation"
                if n_env_missing == 0
                else f"{n_env_missing} experiments have no environment perturbation"
            ),
            details={"n_records": n_records, "n_missing": n_env_missing},
        )
    )
    overlap = len(screened & sgd_genes) / len(screened) if screened else 0.0
    report.add(
        LevelResult(
            level=Level.L4,
            name="gene_containment_sgd",
            passed=overlap >= min_containment,
            message=(
                f"{overlap:.3f} of {len(screened)} measured genes are S288C reference "
                f"genes (>= {min_containment})"
            ),
            details={
                "n_measured": len(screened),
                "n_in_sgd": len(screened & sgd_genes),
                "overlap": overlap,
                "missing_examples": sorted(screened - sgd_genes)[:20],
            },
        )
    )
    return report


__all__ = [
    "verify_environment_response_dataset",
    "verify_environment_response_dataset_streaming",
    "environment_response_gene_set",
    "Record",
]
