# torchcell/verification/fitness
# [[torchcell.verification.fitness]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/fitness
"""L0-L4 record-level verifier for single-mutant fitness datasets.

For ``FitnessPhenotype`` datasets: a deletion/allele strain's growth relative to wild-type
(``fitness``, with WT == 1.0). The schema validator already enforces the uncertainty
invariant (subsumed by L0); this verifier adds:

1. L1 ``count`` -- exact record-count oracle.
2. L1 ``pair_uniqueness`` -- one record per (screened STRAIN x environment) pair (the strain
   is the full genotype signature, so an allelic series is distinct, not duplicate).
3. L2 ``value_fidelity`` -- fitness values are finite and non-negative (WT == 1, sick < 1).
4. L2 ``se_nonnegative`` -- reported fitness SEs are non-negative.
5. L3 ``reference_one`` -- the reference (wild-type) fitness is 1.0 (the convention baseline).
6. L4 ``gene_containment`` (caller) -- screened deletions overlap the S288C gene universe.
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


def _genotype_signature(
    experiment: dict[str, Any],
) -> tuple[tuple[str | None, str | None, str | None, str | None], ...]:
    """Canonical STRAIN identity: the sorted set of perturbations, each keyed by
    ``(systematic_gene_name, perturbation_type, perturbed_gene_name, strain_id)``.

    ``strain_id`` (present on SGA perturbation variants, else None) distinguishes an allelic
    SERIES that shares gene + type -- e.g. Baryshnikova 2010 has 58 genes with more than one
    temperature-sensitive allele (YAL041W x4); without it they collide as duplicates. It is
    None for non-SGA perturbations, so adding it only ever REFINES the key -- pre-existing
    fitness datasets keep their (already-unique) signatures.
    """
    return tuple(
        sorted(
            (
                p.get("systematic_gene_name"),
                p.get("perturbation_type"),
                p.get("perturbed_gene_name"),
                p.get("strain_id"),
            )
            for p in experiment["genotype"]["perturbations"]
        )
    )


def _environment_signature(experiment: dict[str, Any]) -> tuple[Any, ...]:
    """Canonical environment identity: temperature, media, and duration scalars."""
    env = experiment["environment"]
    temp = (env.get("temperature") or {}).get("value")
    media = (env.get("media") or {}).get("name")
    return (temp, media, env.get("duration_hours"), env.get("duration_generations"))


def _l1_pair_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: exactly one record per (screened STRAIN, environment) pair."""
    seen: dict[tuple[Any, ...], int] = {}
    for rec in records:
        exp = rec["experiment"]
        key = (_genotype_signature(exp), _environment_signature(exp))
        seen[key] = seen.get(key, 0) + 1
    dups = {k: n for k, n in seen.items() if n > 1}
    return LevelResult(
        level=Level.L1,
        name="pair_uniqueness",
        passed=not dups,
        message=(
            f"{len(seen)} unique (strain, environment) records, one each"
            if not dups
            else f"{len(dups)} (strain, environment) pairs appear in multiple records"
        ),
        details={"n_pairs": len(seen), "n_duplicated": len(dups)},
    )


def _l3_reference_one(records: Sequence[Record]) -> LevelResult:
    """L3: the reference (wild-type) fitness is 1.0 -- the WT-normalized convention."""
    worst = 0.0
    n = 0
    for rec in records:
        v = rec["reference"]["phenotype_reference"]["fitness"]
        if v is None:
            continue
        n += 1
        worst = max(worst, abs(float(v) - 1.0))
    holds = worst == 0.0
    return LevelResult(
        level=Level.L3,
        name="reference_one",
        passed=holds,
        message=(
            f"reference fitness == 1.0 for all {n} records"
            if holds
            else f"reference fitness not identically 1.0: max|v-1|={worst:.3g}"
        ),
        details={"n_values": n, "worst_abs_dev": worst},
    )


def verify_fitness_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for a single-mutant fitness dataset.

    L4 (cross-source gene overlap with the S288C reference) is asserted by the caller via
    :func:`fitness_gene_set`.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_pair_uniqueness(records))

    fitness_values = [
        float(rec["experiment"]["phenotype"]["fitness"])
        for rec in records
        if rec["experiment"]["phenotype"]["fitness"] is not None
    ]
    report.add(l2_value_fidelity(fitness_values, allow_nan=False, minimum=0.0))

    se_values = [
        float(v)
        for rec in records
        if (v := rec["experiment"]["phenotype"].get("fitness_se")) is not None
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

    report.add(_l3_reference_one(records))
    return report


def fitness_gene_set(records: Sequence[Record]) -> set[str]:
    """Union of screened deleted gene names -- the L4 gene-containment key."""
    genes: set[str] = set()
    for rec in records:
        for p in rec["experiment"]["genotype"]["perturbations"]:
            name = p.get("systematic_gene_name")
            if name is not None:
                genes.add(name)
    return genes


__all__ = ["verify_fitness_dataset", "fitness_gene_set", "Record"]
