# torchcell/verification/pooled_sort_seq
# [[torchcell.verification.pooled_sort_seq]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/pooled_sort_seq
# Test file: tests/torchcell/verification/test_pooled_sort_seq.py
"""L0-L4 record-level verifier for pooled sort-seq expression datasets (#991).

For ``PooledSortSeqExpressionPhenotype`` datasets: a massively parallel reporter assay
in which the whole library is measured at once, so the identity of a record is the
SEQUENCE rather than a well. The schema validator already rejects a non-finite estimate,
an unlabelled uncertainty, a non-positive read count and an empty identifier at
instantiation, so L0 subsumes those. What this verifier adds is about the SET of records
and the genotype beside each one, which no single record can check:

1. L1 ``sequence_uniqueness`` -- one record per ``sequence_id``. A pooled library has no
   well, so this is the identity rule that stands in its place; two records sharing it
   would be one measurement stored twice.
2. L1 ``count`` -- the record count against the caller's oracle.
3. L2 ``value_fidelity`` -- every estimate finite, and within the bin scale the caller
   states (a bin index cannot be negative, nor above the bin count).
4. L2 ``released_values`` (caller) -- the stored estimates equal the released file's own,
   which is the bit-for-bit check that the loader transformed nothing.
5. L3 ``units_declared`` -- every record names its ``expression_units``, its
   ``reporter_gene`` and its ``readout``, and the readout is a pooled one: a per-well or
   per-cell member here would mean the estimate is being compared with readings it is
   not comparable to.
6. L3 ``sequence_is_the_genotype`` -- each record's genotype carries exactly one
   promoter-bearing perturbation, its ``promoter_variant_id`` equals the phenotype's
   ``sequence_id``, and its ``promoter_sequence`` is unambiguous DNA whose variable-region
   span lies inside it. This is what makes the dataset a sequence-level genotype store
   rather than a table of numbers keyed by a label.
7. L3 ``uncertainty_is_typed_or_gapped`` -- a record with no stored uncertainty carries
   the typed ``ProvenanceGap`` that says so, so a missing dispersion is never a silent
   ``None``.
8. L4 ``read_count_coverage`` (caller) -- how many records carry the release's own read
   count, reported rather than required: a release that carries none is honest, and the
   number belongs in the report.

The verifier is STREAMING over a factory of records: a reporter library runs to tens of
thousands of records and nothing here needs them all in memory at once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from torchcell.verification.levels import l1_count, l2_value_fidelity
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

Record = dict[str, Any]

#: The ``ReporterReadout`` members a POOLED assay may carry. A per-well or per-cell
#: member would claim a sample unit a sort-seq estimate does not have.
POOLED_READOUTS: frozenset[str] = frozenset({"sort_seq_fluorescence"})

_DNA = frozenset("ACGT")


def _l1_sequence_uniqueness(sequence_ids: Iterable[str]) -> LevelResult:
    """L1: one record per sequence identifier (the pooled identity rule)."""
    seen: set[str] = set()
    duplicates: list[str] = []
    n = 0
    for sequence_id in sequence_ids:
        n += 1
        if sequence_id in seen:
            if len(duplicates) < 20:
                duplicates.append(sequence_id)
            continue
        seen.add(sequence_id)
    return LevelResult(
        level=Level.L1,
        name="sequence_uniqueness",
        passed=not duplicates,
        message=(
            f"{n} records over {len(seen)} distinct sequence identifiers"
            if not duplicates
            else f"{n - len(seen)} of {n} records repeat a sequence identifier"
        ),
        details={"n_records": n, "n_sequences": len(seen), "duplicates": duplicates},
    )


def _l3_units_declared(records: Iterable[Record]) -> LevelResult:
    """L3: every record names its units, reporter and a POOLED readout."""
    missing = 0
    not_pooled: dict[str, int] = {}
    readouts: dict[str, int] = {}
    n = 0
    for record in records:
        n += 1
        phenotype = record["experiment"]["phenotype"]
        readout = str(phenotype.get("readout") or "")
        readouts[readout] = readouts.get(readout, 0) + 1
        if not (
            str(phenotype.get("expression_units") or "").strip()
            and str(phenotype.get("reporter_gene") or "").strip()
            and readout
        ):
            missing += 1
        elif readout not in POOLED_READOUTS:
            not_pooled[readout] = not_pooled.get(readout, 0) + 1
    passed = missing == 0 and not not_pooled
    return LevelResult(
        level=Level.L3,
        name="units_declared",
        passed=passed,
        message=(
            f"all {n} records name units, reporter and a pooled readout"
            if passed
            else f"{missing} records leave one unset and "
            f"{sum(not_pooled.values())} name a readout that is not pooled"
        ),
        details={
            "n_records": n,
            "n_missing": missing,
            "readouts": readouts,
            "not_pooled": not_pooled,
        },
    )


def _l3_sequence_is_the_genotype(records: Iterable[Record]) -> LevelResult:
    """L3: the genotype carries the record's own sequence, with a span inside it."""
    problems: list[dict[str, Any]] = []
    n = 0
    lengths: dict[int, int] = {}
    for record in records:
        n += 1
        phenotype = record["experiment"]["phenotype"]
        perturbations = record["experiment"]["genotype"]["perturbations"]
        with_promoter = [p for p in perturbations if p.get("promoter_sequence")]
        if len(with_promoter) != 1:
            problems.append(
                {
                    "sequence_id": phenotype.get("sequence_id"),
                    "why": f"{len(with_promoter)} promoter-bearing perturbations",
                }
            )
            continue
        perturbation = with_promoter[0]
        sequence = str(perturbation["promoter_sequence"])
        lengths[len(sequence)] = lengths.get(len(sequence), 0) + 1
        start = perturbation.get("variable_region_start")
        end = perturbation.get("variable_region_end")
        why = None
        if perturbation.get("promoter_variant_id") != phenotype.get("sequence_id"):
            why = "the genotype's variant id is not the phenotype's sequence id"
        elif set(sequence) - _DNA:
            why = "the promoter sequence is not unambiguous ACGT"
        elif start is not None and not (1 <= int(start) <= int(end) <= len(sequence)):
            why = f"the variable span {start}-{end} is not inside the sequence"
        if why is not None and len(problems) < 20:
            problems.append({"sequence_id": phenotype.get("sequence_id"), "why": why})
    return LevelResult(
        level=Level.L3,
        name="sequence_is_the_genotype",
        passed=not problems,
        message=(
            f"all {n} records carry their own sequence on the genotype"
            if not problems
            else f"{len(problems)} of {n} records do not"
        ),
        details={"n_records": n, "sequence_lengths": lengths, "problems": problems},
    )


def _l3_uncertainty_is_typed_or_gapped(records: Iterable[Record]) -> LevelResult:
    """L3: an absent uncertainty carries the typed gap that explains it."""
    silent: list[str] = []
    n = 0
    n_typed = 0
    n_gapped = 0
    for record in records:
        n += 1
        phenotype = record["experiment"]["phenotype"]
        if phenotype.get("expression_level_uncertainty_type") is not None:
            n_typed += 1
            continue
        gapped = {
            str(gap.get("field")) for gap in phenotype.get("provenance_gaps") or []
        }
        if {
            "expression_level_uncertainty",
            "expression_level_uncertainty_type",
        } <= gapped:
            n_gapped += 1
            continue
        if len(silent) < 20:
            silent.append(str(phenotype.get("sequence_id")))
    return LevelResult(
        level=Level.L3,
        name="uncertainty_is_typed_or_gapped",
        passed=not silent,
        message=(
            f"{n_typed} records carry a typed uncertainty and {n_gapped} a typed gap"
            if not silent
            else f"{len(silent)} of {n} records leave the uncertainty a silent None"
        ),
        details={
            "n_records": n,
            "n_typed": n_typed,
            "n_gapped": n_gapped,
            "silent": silent,
        },
    )


def l4_read_count_coverage(records: Iterable[Record]) -> LevelResult:
    """L4: how many records carry the release's own read count (reported, not required)."""
    n = 0
    n_with = 0
    smallest: int | None = None
    for record in records:
        n += 1
        read_count = record["experiment"]["phenotype"].get("read_count")
        if read_count is None:
            continue
        n_with += 1
        value = int(read_count)
        smallest = value if smallest is None else min(smallest, value)
    return LevelResult(
        level=Level.L4,
        name="read_count_coverage",
        passed=True,
        message=(
            f"{n_with} of {n} records carry the release's read count"
            + (f" (smallest {smallest})" if smallest is not None else "")
        ),
        details={"n_records": n, "n_with_read_count": n_with, "smallest": smallest},
    )


def l2_released_values(
    records: Iterable[Record], released: dict[str, float], tolerance: float = 0.0
) -> LevelResult:
    """L2: every stored estimate equals the released file's own value for that sequence."""
    missing: list[str] = []
    drifted: list[dict[str, Any]] = []
    n = 0
    worst = 0.0
    for record in records:
        n += 1
        phenotype = record["experiment"]["phenotype"]
        sequence_id = str(phenotype["sequence_id"])
        if sequence_id not in released:
            if len(missing) < 20:
                missing.append(sequence_id)
            continue
        delta = abs(float(phenotype["expression_level"]) - released[sequence_id])
        worst = max(worst, delta)
        if delta > tolerance and len(drifted) < 20:
            drifted.append({"sequence_id": sequence_id, "delta": delta})
    passed = not missing and not drifted
    return LevelResult(
        level=Level.L2,
        name="released_values",
        passed=passed,
        message=(
            f"all {n} stored estimates equal the released value (max |delta| {worst})"
            if passed
            else f"{len(missing)} sequences are not in the release and "
            f"{len(drifted)} stored estimates differ from it"
        ),
        details={
            "n_records": n,
            "max_abs_delta": worst,
            "not_in_release": missing,
            "drifted": drifted,
        },
    )


def verify_pooled_sort_seq_dataset_streaming(
    records: Callable[[], Iterable[Record]],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    maximum: float,
    minimum: float = 0.0,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for a pooled sort-seq dataset.

    ``records`` is a FACTORY, called once per pass, so a store larger than memory is
    never materialized. ``minimum`` / ``maximum`` bound the bin scale the dataset's
    estimates live on. L2 ``released_values`` and L4 are added by the caller, which is
    the only side that knows the released file and the host's other datasets.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType
    from torchcell.verification.promoter_activity import l0_structural_streaming

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural_streaming(records(), validate))
    count = 0
    for _ in records():
        count += 1
    report.add(l1_count(count, expected_count))
    report.add(
        _l1_sequence_uniqueness(
            str(record["experiment"]["phenotype"]["sequence_id"])
            for record in records()
        )
    )
    report.add(
        l2_value_fidelity(
            (
                float(record["experiment"]["phenotype"]["expression_level"])
                for record in records()
            ),
            allow_nan=False,
            minimum=minimum,
            maximum=maximum,
        )
    )
    report.add(_l3_units_declared(records()))
    report.add(_l3_sequence_is_the_genotype(records()))
    report.add(_l3_uncertainty_is_typed_or_gapped(records()))
    return report
