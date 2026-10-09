# torchcell/verification/protein_fold_change
# [[torchcell.verification.protein_fold_change]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/protein_fold_change
"""L0-L3 record-level verifier for protein FOLD-CHANGE datasets (issue #770).

The relative sibling of :mod:`torchcell.verification.protein`, and not a mode of it: an
absolute abundance and a ratio answer different questions, and the rules that matter
here do not exist there. A ``ProteinFoldChangePhenotype`` carries a scale, a named
denominator and a per-protein p-value, so this verifier asserts:

1. L0 ``structural`` -- every record validates against ``ExperimentType``.
2. L1 ``count`` -- the exact record-count oracle.
3. L1 ``contrast_uniqueness`` -- one record per ``(genotype, reference_basis)``
   contrast; two records of one contrast would be the same measurement stored twice.
4. L2 ``value_fidelity`` -- every stored fold change is finite.
5. L2 ``p_values_are_probabilities`` -- every stored p-value is in ``(0, 1]``. The
   schema admits ``[0, 1]``; a released p-value of exactly 0 is not a probability any
   test reports, so this is the tighter gate.
6. L3 ``reference_is_the_scales_neutral_value`` -- the stored denominator is the
   neutral value of the record's own scale for every stored key, which is what makes
   experiment over reference reproduce the released number.
7. L3 ``fold_change_scale_consistent`` / ``measurement_type_consistent`` -- one scale
   and one statistic across the dataset, since a linear ratio and a log2 ratio of the
   same contrast pool into nothing.
8. L4 is the caller's: the cross-source re-read of the released sheet and the
   assembly-containment rule the family runner adds.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from typing import Any

from torchcell.datamodels.schema import FoldChangeScale
from torchcell.verification.levels import (
    l0_structural,
    l1_count,
    l2_value_fidelity,
    l3_convention,
)
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

Record = dict[str, Any]


def _contrast(record: Record) -> tuple[Any, ...]:
    """The contrast one record measures: genotype x environment against a denominator.

    All three parts are keyed, because a fold change is a ratio of one
    ``genotype x environment`` cell to a named reference and any of the three can be
    what distinguishes two records. A genotype panel varies the perturbations at one
    environment (Carruthers 2025), and a wild-type panel varies the environment with no
    perturbation at all (Caglar 2017); keying on the genotype alone would collapse the
    second panel's records into one contrast. A record that repeats all three IS the
    same measurement stored twice, which is what the rule exists to catch.
    """
    perturbations = tuple(
        sorted(
            (
                str(perturbation["systematic_gene_name"]),
                str(perturbation["perturbation_type"]),
            )
            for perturbation in record["experiment"]["genotype"]["perturbations"]
        )
    )
    environment = json.dumps(record["experiment"]["environment"], sort_keys=True)
    return (
        perturbations,
        environment,
        str(record["experiment"]["phenotype"]["reference_basis"]),
    )


def _l1_contrast_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: one record per ``(genotype, reference_basis)`` contrast."""
    seen: dict[tuple[Any, ...], int] = {}
    for record in records:
        key = _contrast(record)
        seen[key] = seen.get(key, 0) + 1
    duplicated = {key: n for key, n in seen.items() if n > 1}
    holds = not duplicated
    return LevelResult(
        level=Level.L1,
        name="contrast_uniqueness",
        passed=holds,
        message=(
            f"{len(seen)} distinct contrasts, one record each"
            if holds
            else f"{len(duplicated)} contrasts appear in more than one record"
        ),
        details={"n_contrasts": len(seen), "n_duplicated": len(duplicated)},
    )


def _l2_p_values_are_probabilities(records: Sequence[Record]) -> LevelResult:
    """L2: every stored p-value is in ``(0, 1]``."""
    values = [
        float(value)
        for record in records
        for value in (
            record["experiment"]["phenotype"].get("protein_fold_change_p_value") or {}
        ).values()
    ]
    bad = [value for value in values if not 0.0 < value <= 1.0]
    holds = not bad
    return LevelResult(
        level=Level.L2,
        name="p_values_are_probabilities",
        passed=holds,
        message=(
            f"all {len(values)} p-values lie in (0, 1]"
            if holds
            else f"{len(bad)} p-values are not in (0, 1]"
        ),
        details={"n_values": len(values), "n_bad": len(bad)},
    )


def _l3_reference_is_neutral(records: Sequence[Record]) -> LevelResult:
    """L3: the denominator is the neutral value of the record's own scale, keyed alike."""
    n = 0
    bad = 0
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        scale = FoldChangeScale(phenotype["fold_change_scale"])
        reference = record["reference"]["phenotype_reference"]
        if set(reference["protein_fold_change"]) != set(
            phenotype["protein_fold_change"]
        ):
            bad += 1
            continue
        if FoldChangeScale(reference["fold_change_scale"]) is not scale:
            bad += 1
            continue
        for value in reference["protein_fold_change"].values():
            n += 1
            if float(value) != scale.neutral_value:
                bad += 1
    holds = bad == 0
    return l3_convention(
        "reference_is_the_scales_neutral_value",
        holds,
        detail=(
            f"all {n} reference values are their scale's neutral value and key-matched "
            "to the experiment"
            if holds
            else f"{bad} reference values are not the scale's neutral value or are "
            "key-mismatched"
        ),
    )


def _l3_single_valued(records: Sequence[Record], field: str, name: str) -> LevelResult:
    """L3: every record shares one value of ``field`` (no silent mixing)."""
    values = {str(record["experiment"]["phenotype"][field]) for record in records}
    return l3_convention(
        name,
        len(values) == 1,
        detail=(
            f"single {field}: {next(iter(values))!r}"
            if len(values) == 1
            else f"{len(values)} distinct {field}s mixed: {sorted(values)}"
        ),
    )


def verify_protein_fold_change_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for a protein fold-change dataset."""
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((record["experiment"] for record in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_contrast_uniqueness(records))

    fold_changes = [
        float(value)
        for record in records
        for value in record["experiment"]["phenotype"]["protein_fold_change"].values()
    ]
    report.add(l2_value_fidelity(fold_changes, allow_nan=False))
    report.add(_l2_p_values_are_probabilities(records))
    report.add(_l3_reference_is_neutral(records))
    report.add(
        _l3_single_valued(records, "fold_change_scale", "fold_change_scale_consistent")
    )
    report.add(
        _l3_single_valued(records, "measurement_type", "measurement_type_consistent")
    )
    return report


def protein_fold_change_locus_set(records: Sequence[Record]) -> set[str]:
    """The host loci a fold-change dataset names: its tested proteins and its
    perturbed genes.

    Both are identifiers the loader resolved against the record's own pinned assembly,
    so containment over their union is the whole L4 question for this family.
    """
    from torchcell.verification.runners import host_perturbed_gene_set

    measured = host_perturbed_gene_set(records)
    for record in records:
        measured |= set(record["experiment"]["phenotype"]["protein_fold_change"])
    return measured


def fold_change_p_value_round_trip(
    neg_log10_p_value: float, stored_p_value: float, *, tol: float = 1e-12
) -> bool:
    """Whether ``stored_p_value`` is exactly ``10**-neg_log10_p_value``.

    The released column is -log10(p) and the stored column is p, so the only honest
    check is the inverse of the conversion the loader applied.
    """
    if not 0.0 < stored_p_value <= 1.0 or not math.isfinite(neg_log10_p_value):
        return False
    return abs(-math.log10(stored_p_value) - neg_log10_p_value) <= tol


__all__ = [
    "Record",
    "fold_change_p_value_round_trip",
    "protein_fold_change_locus_set",
    "verify_protein_fold_change_dataset",
]
