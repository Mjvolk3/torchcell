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
5. L1 `record_uniqueness` -- one record per (genotype, environment, replicate_id), so
   a strain stored twice is caught while the replicate records of one strain stay
   distinct by their ``replicate_id``.
6. L1 `replicate_records` -- the number of records carrying a ``replicate_id`` (the
   released wild-type samples) is the count the loader declares, and their ids are
   distinct.

L4 (cross-source perturbed-gene overlap with the deletion expression datasets) is
asserted by the caller across datasets, not here.

NOTE on the WT (per-replicate policy, #1178 / #1171; 2026.11.24): every released
wild-type sample is its own record carrying its ``replicate_id`` (Ohya 2005 122,
Ohnuki 2018 114, Ohnuki 2022 749), and each record's reference phenotype is a summary
(the per-feature mean over those samples, ``replicate_id`` None) carrying the full
281+220 vocabulary. So the record-count oracle is mutants + wild-type samples (Ohya
4718 + 122 = 4840), and ``replicate_records`` counts the wild-type part.
"""

from __future__ import annotations

import json
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


def _l1_calmorph_completeness(
    records: Sequence[Record], expected_incomplete: int = 0
) -> LevelResult:
    """L1: every record measures the full 281 base + 220 CV CalMorph vocabulary.

    ``expected_incomplete`` is the count the loader DECLARES (#1045): a released row
    with a blank cell is stored with that parameter key-absent (Ohnuki 2022's HRD1 row,
    two blank cells), so the rule passes on exactly the declared number of short
    records and an undeclared one still fails.
    """
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
    passed = len(short) == expected_incomplete
    declared = (
        f" ({expected_incomplete} declared by the loader)"
        if expected_incomplete
        else ""
    )
    return LevelResult(
        level=Level.L1,
        name="calmorph_completeness",
        passed=passed,
        message=(
            f"all {len(records)} records carry the full "
            f"{len(_BASE_LABELS)} base + {len(_CV_LABELS)} CV parameters"
            if not short
            else f"{len(short)}/{len(records)} records missing CalMorph parameters"
            f"{declared}"
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

    The reference summarizes the released WT samples (each of which is also a record,
    counted by ``replicate_records``); every record must carry it fully populated.
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


def _l1_record_uniqueness(records: Sequence[Record]) -> LevelResult:
    """L1: one record per (genotype, environment, replicate_id).

    The genotype and environment are their whole stored objects, so two strains or two
    conditions are never merged by a key that forgot a field. ``replicate_id`` joins the
    key (read with ``.get``, empty when unset), so the wild-type sample records of one
    strain stay distinct and a mutant stored once per strain keys as before.
    """
    seen: dict[tuple[str, str, str], int] = {}
    for rec in records:
        experiment = rec["experiment"]
        replicate_id = experiment["phenotype"].get("replicate_id")
        key = (
            json.dumps(experiment["genotype"], sort_keys=True),
            json.dumps(experiment["environment"], sort_keys=True),
            "" if replicate_id is None else str(replicate_id),
        )
        seen[key] = seen.get(key, 0) + 1
    duplicated = {key: n for key, n in seen.items() if n > 1}
    return LevelResult(
        level=Level.L1,
        name="record_uniqueness",
        passed=not duplicated,
        message=(
            f"{len(seen)} unique (genotype, environment, replicate_id) records, one each"
            if not duplicated
            else f"{len(duplicated)} (genotype, environment, replicate_id) keys appear "
            "in more than one record"
        ),
        details={
            "n_keys": len(seen),
            "n_duplicated": len(duplicated),
            "n_genotypes": len({genotype for genotype, _, _ in seen}),
        },
    )


def _l1_replicate_records(
    records: Sequence[Record], expected_replicate_records: int
) -> LevelResult:
    """L1: the records carrying a ``replicate_id`` are the declared count, all distinct.

    In the SCMD family these are the released wild-type samples; a mutant row is one
    culture per strain and carries none.
    """
    ids = [
        rec["experiment"]["phenotype"].get("replicate_id")
        for rec in records
        if rec["experiment"]["phenotype"].get("replicate_id") is not None
    ]
    distinct = len(set(ids)) == len(ids)
    passed = len(ids) == expected_replicate_records and distinct
    return LevelResult(
        level=Level.L1,
        name="replicate_records",
        passed=passed,
        message=(
            f"{len(ids)} records carry a replicate_id (expected "
            f"{expected_replicate_records}), "
            + ("all distinct" if distinct else f"{len(ids) - len(set(ids))} repeated")
        ),
        details={
            "observed": len(ids),
            "expected": expected_replicate_records,
            "n_distinct": len(set(ids)),
        },
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
    expected_replicate_records: int,
    expected_incomplete: int = 0,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for the Ohya2005 CalMorph dataset.

    Args:
        records: the per-LMDB-entry dicts (experiment/reference/publication).
        dataset_name: dataset identity for the report.
        provenance: where the records came from.
        expected_count: the record-count oracle, mutants plus released wild-type
            samples (Ohya2005 = 4718 + 122 = 4840).
        expected_replicate_records: how many records carry a ``replicate_id`` (the
            released wild-type samples; Ohya2005 = 122).
        expected_incomplete: how many records the loader declares short of the full
            CalMorph vocabulary (Ohnuki 2022's HRD1 row, two blank cells key-absent).

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

    # L1: exact record count + one record per key + the declared replicate records +
    # full-vocabulary coverage + populated WT reference.
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_record_uniqueness(records))
    report.add(_l1_replicate_records(records, expected_replicate_records))
    report.add(_l1_calmorph_completeness(records, expected_incomplete))
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
