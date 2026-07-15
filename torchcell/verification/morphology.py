# torchcell/verification/morphology
# [[torchcell.verification.morphology]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/morphology
"""L0-L4 record-level verifier for the Ohya2005 CalMorph morphology dataset (WS6).

Assembles the reusable checks in :mod:`torchcell.verification.levels` into a
:class:`VerificationReport` for the SCMD/Ohya2005 CalMorph deletion-mutant dataset.

The CalMorph schema (`CalMorphPhenotype`) already enforces, at instantiation, that
every `calmorph` key is one of the 281 `CALMORPH_LABELS`, every CV key is one of the
220 `CALMORPH_STATISTICS`, and that no value is NaN. L0 (structural) therefore already
guarantees those. This verifier adds the checks the schema does NOT encode:

1. L1 `calmorph_completeness` -- the schema accepts any *subset* of parameters; a
   correct build measures the FULL 281 base + 220 CV vocabulary in every record, so a
   record short of that has silently dropped parameters.
2. L2 finiteness -- the schema blocks NaN but not +/-inf.
3. L2 `cv_nonnegative` -- a coefficient of variation (SD/mean of a non-negative
   morphological measurement) is non-negative by definition; a negative signals a
   computation bug the schema would not catch.
4. L3 `vocabulary_parity` -- the label vocabulary is internally consistent
   (281 + 220 == 501 == `CALMORPH_PARAMETERS`, base and CV disjoint), i.e. the
   convention the records adhere to holds.

L4 (cross-source perturbed-gene overlap with the deletion expression datasets) is
asserted by the caller across datasets, not here.

NOTE on the WT: the 122 wildtype profiles are AGGREGATED into each record's reference
phenotype (a single mean-WT baseline, itself carrying the full 281+220 vocabulary),
not stored as 122 separate records. So the record-count oracle is the mutant count
(4718; all strains retained, ORF names reconciled to R64); the WT is verified as a populated reference, not counted from records.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from torchcell.datamodels.calmorph_labels import (
    CALMORPH_LABELS,
    CALMORPH_PARAMETERS,
    CALMORPH_STATISTICS,
)
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

# One LMDB entry: {"experiment": {...}, "reference": {...}, "publication": {...}}.
Record = dict[str, Any]

_BASE_LABELS = set(CALMORPH_LABELS)
_CV_LABELS = set(CALMORPH_STATISTICS)


def _l1_calmorph_completeness(records: Sequence[Record]) -> LevelResult:
    """L1: every record measures the full 281 base + 220 CV CalMorph vocabulary."""
    short: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        phenotype = rec["experiment"]["phenotype"]
        base = set(phenotype["calmorph"].keys())
        cv = set((phenotype.get("calmorph_coefficient_of_variation") or {}).keys())
        missing_base = _BASE_LABELS - base
        missing_cv = _CV_LABELS - cv
        if missing_base or missing_cv:
            short.append(
                {
                    "index": i,
                    "missing_base": len(missing_base),
                    "missing_cv": len(missing_cv),
                }
            )
    passed = not short
    return LevelResult(
        level=Level.L1,
        name="calmorph_completeness",
        passed=passed,
        message=(
            f"all {len(records)} records carry the full "
            f"{len(_BASE_LABELS)} base + {len(_CV_LABELS)} CV parameters"
            if passed
            else f"{len(short)}/{len(records)} records missing CalMorph parameters"
        ),
        details={
            "n_records": len(records),
            "n_base_expected": len(_BASE_LABELS),
            "n_cv_expected": len(_CV_LABELS),
            "n_short_records": len(short),
            "short": short[:20],
        },
    )


def _l1_reference_populated(records: Sequence[Record]) -> LevelResult:
    """L1: every record's WT reference phenotype carries the full vocabulary.

    The 122 WT profiles are aggregated into this reference; verifying it is populated
    is how the WT is accounted for (the 122 count is not record-recoverable).
    """
    bad: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        ref_ph = rec["reference"]["phenotype_reference"]
        n_base = len(ref_ph.get("calmorph") or {})
        n_cv = len(ref_ph.get("calmorph_coefficient_of_variation") or {})
        if n_base != len(_BASE_LABELS) or n_cv != len(_CV_LABELS):
            bad.append({"index": i, "n_base": n_base, "n_cv": n_cv})
    passed = not bad
    return LevelResult(
        level=Level.L1,
        name="reference_populated",
        passed=passed,
        message=(
            f"WT reference carries the full {len(_BASE_LABELS)}+{len(_CV_LABELS)} "
            f"vocabulary in all {len(records)} records"
            if passed
            else f"{len(bad)}/{len(records)} records have an under-populated WT reference"
        ),
        details={"n_bad": len(bad), "bad": bad[:20]},
    )


def _l3_vocabulary_parity() -> LevelResult:
    """L3: the CalMorph label vocabulary is internally consistent (281+220==501)."""
    union_ok = _BASE_LABELS | _CV_LABELS == set(CALMORPH_PARAMETERS)
    disjoint_ok = not (_BASE_LABELS & _CV_LABELS)
    counts_ok = (
        len(_BASE_LABELS) == 281
        and len(_CV_LABELS) == 220
        and len(CALMORPH_PARAMETERS) == 501
    )
    holds = union_ok and disjoint_ok and counts_ok
    return l3_convention(
        "vocabulary_parity",
        holds,
        detail=(
            f"CALMORPH_LABELS({len(_BASE_LABELS)}) + "
            f"CALMORPH_STATISTICS({len(_CV_LABELS)}) == "
            f"CALMORPH_PARAMETERS({len(CALMORPH_PARAMETERS)}), disjoint={disjoint_ok}"
        ),
    )


def verify_morphology_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for the Ohya2005 CalMorph dataset.

    Args:
        records: the per-LMDB-entry dicts (experiment/reference/publication).
        dataset_name: dataset identity for the report.
        provenance: where the records came from.
        expected_count: the mutant record-count oracle (Ohya2005 = 4718).

    Returns:
        A :class:`VerificationReport` carrying the L0-L3 results. L4 (cross-source
        perturbed-gene overlap) is asserted across datasets by the caller.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)

    # L0: every experiment record validates against the schema union (this already
    # enforces label-vocabulary membership + no-NaN via CalMorphPhenotype validators).
    report.add(l0_structural((rec["experiment"] for rec in records), validate))

    # L1: exact mutant count + full-vocabulary coverage + populated WT reference.
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_calmorph_completeness(records))
    report.add(_l1_reference_populated(records))

    # L2: finiteness of base measurements (schema blocks NaN, not inf).
    base_values = [
        float(v)
        for rec in records
        for v in rec["experiment"]["phenotype"]["calmorph"].values()
    ]
    report.add(l2_value_fidelity(base_values, allow_nan=False))

    # L2: CV values finite and non-negative (CV is non-negative by definition).
    cv_values = [
        float(v)
        for rec in records
        for v in (
            rec["experiment"]["phenotype"].get("calmorph_coefficient_of_variation")
            or {}
        ).values()
    ]
    cv_result = l2_value_fidelity(cv_values, allow_nan=False, minimum=0.0)
    report.add(
        LevelResult(
            level=Level.L2,
            name="cv_nonnegative",
            passed=cv_result.passed,
            message=cv_result.message,
            details=cv_result.details,
        )
    )

    # L3: the label vocabulary the records adhere to is internally consistent.
    report.add(_l3_vocabulary_parity())

    return report


def perturbed_gene_set(records: Sequence[Record]) -> set[str]:
    """Union of systematic gene names deleted across records (the L4 overlap key)."""
    genes: set[str] = set()
    for rec in records:
        for pert in rec["experiment"]["genotype"]["perturbations"]:
            name = pert.get("systematic_gene_name")
            if name is not None:
                genes.add(name)
    return genes


__all__ = ["verify_morphology_dataset", "perturbed_gene_set", "Record"]
