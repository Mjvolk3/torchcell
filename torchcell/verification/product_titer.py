# torchcell/verification/product_titer
# [[torchcell.verification.product_titer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/product_titer
"""L0-L3 record-level verifier for product-titer datasets (the bioproduction rows).

The family verifier the three titer loaders were each hand-rolling: Carruthers 2025's
battery lived in its test file, de Siqueira 2025's ``titer_levels`` says "there is no
shared titer verifier yet", and Kang 2026's ``verify_build`` assembles the same levels
inline. This module is that verifier, and
:func:`torchcell.verification.runners.run_product_titer` adds the L4 gene-universe
containment on top of it (the universe belongs to the host a record's own
``genome_reference`` names, which is runner plumbing, not family plumbing).

What it checks, and why each rule is the same rule for every dataset of the family:

1. L0 ``structural`` -- every record validates against ``ExperimentType``.
2. L1 ``count`` -- the record count against the loader's own oracle.
3. L2 ``value_fidelity`` -- every stored titer is finite and non-negative.
4. L2 ``uncertainty_nonnegative`` -- every RELEASED uncertainty is finite and
   non-negative. Records that release none are not counted, which is why the rule is
   separate from the titer one.
5. L2 ``se_is_the_uncertainty_over_sqrt_n`` -- the derived ``titer_se`` is exactly
   ``titer_uncertainty / sqrt(n_samples)`` within the dataset's own tolerance, and a
   record that releases no uncertainty stores no ``titer_se``. One rule covers both
   designs: a dataset with per-strain SDs (Carruthers) is checked value by value, and a
   dataset whose uncertainty is a typed gap (Kang, de Siqueira) is checked for having
   derived nothing from nothing.
6. L3 ``titer_unit_is_the_pinned_unit`` -- the unit decision its note records.
7. L3 ``uncertainty_is_typed_or_gapped`` -- an uncertainty NUMBER and its TYPE are
   either both stored or both named in ``provenance_gaps``. A number without its type
   is unreadable and a silent ``None`` is indistinguishable from "not applicable".
8. L3 ``replicate_design_is_sourced_or_gapped`` -- the same rule for ``n_samples`` and
   ``sample_unit``.
9. L3 ``heterologous_pathway_gene_counts`` -- the per-record count of heterologous
   pathway genes is one the dataset declares (5 for the pIY670 chassis, 6/11/13 for
   Kang's integration variants), so a production strain that lost its pathway fails.
10. L3 ``product_is_the_declared_one`` -- the measured product is the one the dataset
    is named for, so two products are never silently pooled into one label.

A dataset's own cross-source joins (a released oracle table re-read from the raw
mirror) stay in the loader module that owns the reader and are added to this report by
that module's ``verify_build``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Sequence
from typing import Any

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


def _phenotypes(records: Sequence[Record]) -> list[dict[str, Any]]:
    """The stored phenotype of every record, in record order."""
    return [record["experiment"]["phenotype"] for record in records]


def _gapped_fields(phenotype: dict[str, Any]) -> set[str]:
    """The field names this phenotype's typed ``ProvenanceGap`` entries name."""
    return {str(gap["field"]) for gap in phenotype["provenance_gaps"]}


def _l2_se_identity(records: Sequence[Record], tol: float) -> LevelResult:
    """L2: ``titer_se == titer_uncertainty / sqrt(n_samples)``, and nothing from nothing.

    Both halves are one rule because they are one question: is the DERIVED standard
    error exactly what the released uncertainty and replicate count imply? A record
    that releases no uncertainty answers it by storing no ``titer_se``; deriving one
    anyway would be a fabricated precision.
    """
    derived: list[dict[str, Any]] = []
    n_pairs = 0
    n_without = 0
    worst = 0.0
    for index, phenotype in enumerate(_phenotypes(records)):
        uncertainty = phenotype["titer_uncertainty"]
        se = phenotype["titer_se"]
        if uncertainty is None:
            n_without += 1
            if se is not None:
                derived.append(
                    {"index": index, "reason": "titer_se without an uncertainty"}
                )
            continue
        n_pairs += 1
        expected = float(uncertainty) / math.sqrt(float(phenotype["n_samples"]))
        difference = abs(float(se) - expected)
        worst = max(worst, difference)
        if difference > tol:
            derived.append(
                {
                    "index": index,
                    "titer_se": se,
                    "expected": expected,
                    "diff": difference,
                }
            )
    passed = not derived
    return LevelResult(
        level=Level.L2,
        name="se_is_the_uncertainty_over_sqrt_n",
        passed=passed,
        message=(
            f"{n_pairs} pairs agree within {tol} "
            f"(titer_se == titer_uncertainty / sqrt(n_samples)); {n_without} records "
            "release no uncertainty and store no titer_se"
            if passed
            else f"{len(derived)} of {len(records)} records break the derivation"
        ),
        details={
            "tol": tol,
            "n_pairs": n_pairs,
            "n_without_uncertainty": n_without,
            "worst_abs_diff": worst,
            "worst": derived[:20],
        },
    )


def verify_product_titer_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    titer_unit: str,
    titer_unit_detail: str,
    se_tol: float,
    pathway_gene_counts: Collection[int],
    product_names: Collection[str],
) -> VerificationReport:
    """Run the family's L0-L3 gate over a built product-titer store.

    ``titer_unit_detail``, ``se_tol``, ``pathway_gene_counts`` and ``product_names``
    come from the dataset's own note: the unit decision it records, the tolerance it
    states for the derived standard error, the pathway composition of its chassis, and
    the product it titers. L4 is the caller's (the runner's host-aware containment, plus
    whatever cross-source oracle the loader module holds).
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    phenotypes = _phenotypes(records)

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((record["experiment"] for record in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(
        l2_value_fidelity(
            [float(phenotype["titer"]) for phenotype in phenotypes], minimum=0.0
        )
    )
    uncertainties = [
        float(phenotype["titer_uncertainty"])
        for phenotype in phenotypes
        if phenotype["titer_uncertainty"] is not None
    ]
    report.add(
        l2_value_fidelity(uncertainties, minimum=0.0).model_copy(
            update={"name": "uncertainty_nonnegative"}
        )
    )
    report.add(_l2_se_identity(records, se_tol))

    units = {str(phenotype["titer_unit"]) for phenotype in phenotypes}
    report.add(
        l3_convention(
            "titer_unit_is_the_pinned_unit",
            units == {titer_unit},
            detail=f"stored units {sorted(units)}; {titer_unit_detail}",
        )
    )
    report.add(
        l3_convention(
            "uncertainty_is_typed_or_gapped",
            all(
                (
                    phenotype["titer_uncertainty"] is not None
                    and phenotype["titer_uncertainty_type"] is not None
                )
                or (
                    phenotype["titer_uncertainty"] is None
                    and phenotype["titer_uncertainty_type"] is None
                    and {"titer_uncertainty", "titer_uncertainty_type"}
                    <= _gapped_fields(phenotype)
                )
                for phenotype in phenotypes
            ),
            detail=(
                "an uncertainty number and its type are both stored, or both named in "
                "provenance_gaps; a number without its type is unreadable"
            ),
        )
    )
    report.add(
        l3_convention(
            "replicate_design_is_sourced_or_gapped",
            all(
                (
                    phenotype["n_samples"] is not None
                    and phenotype["sample_unit"] is not None
                )
                or (
                    phenotype["n_samples"] is None
                    and phenotype["sample_unit"] is None
                    and {"n_samples", "sample_unit"} <= _gapped_fields(phenotype)
                )
                for phenotype in phenotypes
            ),
            detail=(
                "a replicate count and what one replicate IS are both stored, or both "
                "named in provenance_gaps"
            ),
        )
    )
    counts = {
        sum(
            1
            for perturbation in record["experiment"]["genotype"]["perturbations"]
            if perturbation["perturbation_type"] == "heterologous_pathway"
        )
        for record in records
    }
    report.add(
        l3_convention(
            "heterologous_pathway_gene_counts",
            counts <= set(pathway_gene_counts),
            detail=(
                f"per-record heterologous pathway gene counts {sorted(counts)}; the "
                f"dataset declares {sorted(pathway_gene_counts)}"
            ),
        )
    )
    products = {str(phenotype["product"]["name"]) for phenotype in phenotypes}
    report.add(
        l3_convention(
            "product_is_the_declared_one",
            products <= set(product_names),
            detail=(
                f"stored products {sorted(products)}; the dataset titers "
                f"{sorted(product_names)}"
            ),
        )
    )
    return report


__all__ = ["Record", "verify_product_titer_dataset"]
