# torchcell/verification/promoter_activity
# [[torchcell.verification.promoter_activity]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/promoter_activity
# Test file: tests/torchcell/verification/test_promoter_activity.py
"""L0-L4 record-level verifier for promoter-activity (reporter-fusion) datasets.

For ``PromoterActivityPhenotype`` datasets: a transcriptional-reporter library where
each strain carries one promoter driving a reporter, and the stored number is that
reporter's signal. The schema validator already rejects a non-finite signal, an
unlabelled uncertainty and an empty identifier at instantiation, so L0 subsumes those.
This verifier adds what the schema cannot encode, because each rule is about the SET of
records rather than one record:

1. L1 ``reading_uniqueness`` -- one record per (``well_id``, ``duration_hours``). A
   reporter library holds one promoter in many wells, so the promoter label is NOT an
   identity; the well and the read time are.
2. L1 ``count`` -- the record count against the caller's oracle.
3. L2 ``value_fidelity`` -- every signal finite and non-negative. An absolute reporter
   intensity cannot be negative; a dataset storing a background-subtracted signal passes
   ``minimum=None``.
4. L3 ``units_declared`` -- every record names its ``activity_units``, its
   ``reporter_gene`` and its ``readout``, so two screens' numbers are never compared
   without knowing what they are.
5. L3 ``promoter_gene_is_a_locus`` -- a non-null ``promoter_gene`` resolves to itself
   in the record's own assembly, so the join key to the gene node is real. A null is
   accepted and counted: a label that names no single gene is an honest absence.
6. L3 ``reference_is_the_control_reading`` -- the reference's phenotype is a reading of
   the SAME promoter at the SAME time as the record, which is what makes a stored
   record's ratio to its reference the source's own fold change.
7. L4 ``gene_containment`` (caller) -- the measured promoters' genes overlap the gene
   universe the other datasets of this host use.

The verifier is STREAMING over an iterable of records: a reporter library crossed with a
condition panel and a time course runs to tens of thousands of records, and nothing here
needs them all in memory at once.
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


def _l1_reading_uniqueness(keys: Iterable[tuple[str, float | None]]) -> LevelResult:
    """L1: one record per (well identifier, read time)."""
    seen: set[tuple[str, float | None]] = set()
    duplicates: list[str] = []
    n = 0
    for key in keys:
        n += 1
        if key in seen:
            if len(duplicates) < 20:
                duplicates.append(f"{key[0]} @ {key[1]}")
            continue
        seen.add(key)
    return LevelResult(
        level=Level.L1,
        name="reading_uniqueness",
        passed=not duplicates,
        message=(
            f"{n} records over {len(seen)} distinct (well, read time) keys"
            if not duplicates
            else f"{n - len(seen)} of {n} records repeat a (well, read time) key"
        ),
        details={"n_records": n, "n_keys": len(seen), "duplicates": duplicates},
    )


def _l3_units_declared(records: Iterable[Record]) -> LevelResult:
    """L3: every record names its units, its reporter and how it was read."""
    missing = 0
    n = 0
    readouts: dict[str, int] = {}
    for record in records:
        n += 1
        phenotype = record["experiment"]["phenotype"]
        readout = str(phenotype.get("readout") or "")
        readouts[readout] = readouts.get(readout, 0) + 1
        if not (
            str(phenotype.get("activity_units") or "").strip()
            and str(phenotype.get("reporter_gene") or "").strip()
            and readout
        ):
            missing += 1
    return LevelResult(
        level=Level.L3,
        name="units_declared",
        passed=missing == 0,
        message=(
            f"all {n} records name units, reporter and readout"
            if missing == 0
            else f"{missing} of {n} records leave one of them unset"
        ),
        details={"n_records": n, "n_missing": missing, "readouts": readouts},
    )


def _l3_promoter_gene_is_a_locus(
    genes: Iterable[str | None], resolve_gene_name: Callable[[str], Any]
) -> LevelResult:
    """L3: every non-null promoter gene resolves to itself in the pinned assembly."""
    unresolved: list[str] = []
    n_null = 0
    checked: set[str] = set()
    for gene in genes:
        if gene is None:
            n_null += 1
            continue
        if gene in checked:
            continue
        checked.add(gene)
        resolution = resolve_gene_name(gene)
        if resolution.systematic_name != gene:
            unresolved.append(gene)
    return LevelResult(
        level=Level.L3,
        name="promoter_gene_is_a_locus",
        passed=not unresolved,
        message=(
            f"{len(checked)} distinct promoter genes resolve to themselves; "
            f"{n_null} records carry no gene"
            if not unresolved
            else f"{len(unresolved)} of {len(checked)} promoter genes do not resolve "
            "to themselves"
        ),
        details={
            "n_genes": len(checked),
            "n_null_records": n_null,
            "not_a_locus": unresolved[:20],
        },
    )


def _l3_reference_is_the_control_reading(records: Iterable[Record]) -> LevelResult:
    """L3: the reference reads the same promoter at the same time as the record."""
    mismatched: list[dict[str, Any]] = []
    n = 0
    for record in records:
        n += 1
        experiment = record["experiment"]
        reference = record["reference"]
        own = experiment["phenotype"]
        control = reference["phenotype_reference"]
        same_promoter = own["promoter_name"] == control["promoter_name"]
        same_time = (
            experiment["environment"]["duration_hours"]
            == reference["environment_reference"]["duration_hours"]
        )
        if not (same_promoter and same_time):
            if len(mismatched) < 20:
                mismatched.append(
                    {
                        "index": n - 1,
                        "promoter": own["promoter_name"],
                        "reference_promoter": control["promoter_name"],
                        "hours": experiment["environment"]["duration_hours"],
                        "reference_hours": reference["environment_reference"][
                            "duration_hours"
                        ],
                    }
                )
    return LevelResult(
        level=Level.L3,
        name="reference_is_the_control_reading",
        passed=not mismatched,
        message=(
            f"all {n} references read the same promoter at the same time"
            if not mismatched
            else f"{len(mismatched)} of {n} references read another promoter or time"
        ),
        details={"n_records": n, "mismatched": mismatched},
    )


def verify_promoter_activity_dataset_streaming(
    records: Callable[[], Iterable[Record]],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    resolve_gene_name: Callable[[str], Any],
    minimum: float | None = 0.0,
) -> VerificationReport:
    """Run the L0-L3 record-level gate for a promoter-activity dataset.

    ``records`` is a FACTORY, called once per pass, so a store larger than memory is
    never materialized. L4 (gene-universe overlap with the host's other datasets) is
    asserted by the caller across datasets.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural_streaming(records(), validate))
    count = 0
    for _ in records():
        count += 1
    report.add(l1_count(count, expected_count))
    report.add(
        _l1_reading_uniqueness(
            (
                record["experiment"]["phenotype"]["well_id"],
                record["experiment"]["environment"]["duration_hours"],
            )
            for record in records()
        )
    )
    report.add(
        l2_value_fidelity(
            (
                record["experiment"]["phenotype"]["promoter_activity"]
                for record in records()
            ),
            allow_nan=False,
            minimum=minimum,
        )
    )
    report.add(_l3_units_declared(records()))
    report.add(
        _l3_promoter_gene_is_a_locus(
            (
                record["experiment"]["phenotype"]["promoter_gene"]
                for record in records()
            ),
            resolve_gene_name,
        )
    )
    report.add(_l3_reference_is_the_control_reading(records()))
    return report


def l0_structural_streaming(
    records: Iterable[Record], validator: Callable[[Any], object]
) -> LevelResult:
    """L0 over a stream: every record's experiment validates against the schema union.

    The shared ``l0_structural`` takes the experiments already projected out; this wraps
    it so a caller streams whole records instead of building a list of experiments.
    """
    from torchcell.verification.levels import l0_structural

    return l0_structural((record["experiment"] for record in records), validator)
