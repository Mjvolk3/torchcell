# tests/torchcell/verification/test_promoter_activity.py
# [[tests.torchcell.verification.test_promoter_activity]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_promoter_activity.py
"""``torchcell/verification/promoter_activity.py``: each L0-L3 row, passing and failing.

The records are built by hand from the real schema classes, so a row is exercised
against the same shape a loader writes. Each failing case changes exactly ONE thing, so
a row that stops discriminating shows up as a test that no longer fails.

The two rows worth the most here are ``reading_uniqueness`` and
``reference_is_the_control_reading``: a reporter library holds one promoter in many
wells, so the promoter label is not an identity, and the reference being the same
promoter at the same time is what makes a record's ratio to it the source's own fold
change.
"""

from __future__ import annotations

from typing import Any

from torchcell.datamodels.media import LB
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    Environment,
    Genotype,
    HeterologousPathwayPerturbation,
    PromoterActivityExperiment,
    PromoterActivityExperimentReference,
    PromoterActivityPhenotype,
    ReporterReadout,
    SampleUnit,
    Temperature,
)
from torchcell.verification.promoter_activity import (
    verify_promoter_activity_dataset_streaming,
)
from torchcell.verification.report import Level, Provenance, VerificationReport

PROVENANCE = Provenance(
    source_uri="data/release.xlsx",
    citation_key="someKey2022",
    sha256="0" * 64,
    method="one record per (well, hour)",
    page="sheet 1",
)
REFERENCE_GENOME = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="MG1655",
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
)


class _Resolution:
    """The one field ``promoter_gene_is_a_locus`` reads off a resolution."""

    def __init__(self, systematic_name: str | None) -> None:
        self.systematic_name = systematic_name


def _resolver(known: set[str]) -> Any:
    """Resolve a tag to itself when the assembly carries it, else to nothing."""

    def resolve(name: str) -> _Resolution:
        return _Resolution(name if name in known else None)

    return resolve


def _phenotype(
    value: float,
    *,
    promoter: str = "thrA",
    gene: str | None = "b0002",
    well: str = "Untreated|AZ01|A1",
    units: str = "GFP fluorescence in the reader's own units",
    reporter: str = "gfp",
) -> PromoterActivityPhenotype:
    return PromoterActivityPhenotype(
        promoter_activity=value,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        promoter_name=promoter,
        promoter_gene=gene,
        readout=ReporterReadout.plate_reader_fluorescence,
        reporter_gene=reporter,
        activity_units=units,
        well_id=well,
    )


def _record(
    value: float,
    hour: float,
    *,
    promoter: str = "thrA",
    gene: str | None = "b0002",
    well: str = "Untreated|AZ01|A1",
    control: float | None = None,
    control_promoter: str | None = None,
    control_hour: float | None = None,
    units: str = "GFP fluorescence in the reader's own units",
) -> dict[str, Any]:
    genotype = Genotype(
        perturbations=[
            HeterologousPathwayPerturbation(
                systematic_gene_name="gfp",
                perturbed_gene_name="gfp",
                gene_namespace="ecoli_k12_mg1655_bnumber",
                pathway_name="promoter-GFP reporter",
                source_organism="unreported",
                is_heterologous=True,
                localization="episomal_plasmid",
                promoter_name=promoter,
                copy_number=1.0,
            )
        ]
    )
    environment = Environment(
        media=LB, temperature=Temperature(value=37.0), duration_hours=hour
    )
    experiment = PromoterActivityExperiment(
        dataset_name="Synthetic",
        genotype=genotype,
        environment=environment,
        phenotype=_phenotype(
            value, promoter=promoter, gene=gene, well=well, units=units
        ),
    )
    reference = PromoterActivityExperimentReference(
        dataset_name="Synthetic",
        genome_reference=REFERENCE_GENOME,
        environment_reference=Environment(
            media=LB,
            temperature=Temperature(value=37.0),
            duration_hours=hour if control_hour is None else control_hour,
        ),
        phenotype_reference=_phenotype(
            value if control is None else control,
            promoter=promoter if control_promoter is None else control_promoter,
            gene=gene,
            well=f"Untreated|{well.split('|', 1)[1]}",
        ),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _run(
    records: list[dict[str, Any]], *, known: set[str] | None = None, **kwargs: Any
) -> VerificationReport:
    return verify_promoter_activity_dataset_streaming(
        lambda: iter(records),
        dataset_name="Synthetic",
        provenance=PROVENANCE,
        expected_count=kwargs.pop("expected_count", len(records)),
        resolve_gene_name=_resolver({"b0002"} if known is None else known),
        **kwargs,
    )


def _row(report: VerificationReport, name: str) -> Any:
    (row,) = [r for r in report.results if r.name == name]
    return row


# --------------------------------------------------------------------------- #
# A clean store
# --------------------------------------------------------------------------- #
def test_a_clean_store_passes_every_row_and_covers_l0_to_l3() -> None:
    records = [
        _record(12.0, 2.0, well="Untreated|AZ01|A1"),
        _record(13.0, 3.0, well="Untreated|AZ01|A1"),
        _record(14.0, 2.0, well="Ampicillin Treatment|AZ01|A1", control=12.0),
    ]
    report = _run(records)
    assert [r.name for r in report.results if not r.passed] == []
    assert report.passed is True
    assert {r.level for r in report.results} == {Level.L0, Level.L1, Level.L2, Level.L3}
    assert _row(report, "reading_uniqueness").details["n_keys"] == 3
    assert _row(report, "units_declared").details["readouts"] == {
        "plate_reader_fluorescence": 3
    }


def test_one_promoter_in_two_wells_is_two_readings_not_a_duplicate() -> None:
    """The library holds one promoter in many wells, so the label is not an identity."""
    records = [
        _record(12.0, 2.0, well="Untreated|AZ01|A1"),
        _record(19.0, 2.0, well="Untreated|AZ02|H4"),
    ]
    report = _run(records)
    assert _row(report, "reading_uniqueness").passed is True
    assert _row(report, "reading_uniqueness").details["n_keys"] == 2


# --------------------------------------------------------------------------- #
# Each row, failing on exactly one change
# --------------------------------------------------------------------------- #
def test_the_same_well_read_twice_at_one_hour_fails_uniqueness() -> None:
    records = [
        _record(12.0, 2.0, well="Untreated|AZ01|A1"),
        _record(12.5, 2.0, well="Untreated|AZ01|A1"),
    ]
    row = _row(_run(records), "reading_uniqueness")
    assert row.passed is False
    assert row.details["n_keys"] == 1
    assert row.details["duplicates"] == ["Untreated|AZ01|A1 @ 2.0"]


def test_a_count_that_is_not_the_oracle_fails_l1() -> None:
    row = _row(_run([_record(12.0, 2.0)], expected_count=9), "count")
    assert row.passed is False
    assert row.details == {"observed": 1, "expected": 9}


def test_a_negative_signal_fails_value_fidelity_by_default() -> None:
    """An absolute reporter intensity cannot be negative."""
    row = _row(_run([_record(-1.0, 2.0)]), "value_fidelity")
    assert row.passed is False
    assert row.details["bad"][0]["reason"] == "< 0.0"


def test_a_background_subtracted_store_passes_with_no_minimum() -> None:
    report = _run([_record(-1.0, 2.0)], minimum=None)
    assert _row(report, "value_fidelity").passed is True


def test_a_record_with_no_units_fails_units_declared() -> None:
    """The schema refuses empty units at construction, so this is a store that drifted."""
    record = _record(12.0, 2.0)
    record["experiment"]["phenotype"]["activity_units"] = "   "
    row = _row(_run([record]), "units_declared")
    assert row.passed is False
    assert row.details["n_missing"] == 1


def test_a_promoter_gene_the_assembly_does_not_carry_fails_its_row() -> None:
    row = _row(
        _run([_record(12.0, 2.0, gene="b9999")], known={"b0002"}),
        "promoter_gene_is_a_locus",
    )
    assert row.passed is False
    assert row.details["not_a_locus"] == ["b9999"]


def test_a_null_promoter_gene_is_counted_and_not_a_failure() -> None:
    """A label naming no single gene is an honest absence, not a broken join key."""
    row = _row(_run([_record(12.0, 2.0, gene=None)]), "promoter_gene_is_a_locus")
    assert row.passed is True
    assert row.details == {"n_genes": 0, "n_null_records": 1, "not_a_locus": []}


def test_a_reference_reading_another_promoter_fails_its_row() -> None:
    row = _row(
        _run([_record(12.0, 2.0, control_promoter="thrL")]),
        "reference_is_the_control_reading",
    )
    assert row.passed is False
    assert row.details["mismatched"][0]["reference_promoter"] == "thrL"


def test_a_reference_read_at_another_hour_fails_its_row() -> None:
    row = _row(
        _run([_record(12.0, 2.0, control_hour=9.0)]), "reference_is_the_control_reading"
    )
    assert row.passed is False
    assert row.details["mismatched"][0]["reference_hours"] == 9.0


def test_a_signal_the_schema_would_refuse_fails_l0() -> None:
    """A store that drifted past the schema is reported as data, not raised."""
    record = _record(12.0, 2.0)
    record["experiment"]["phenotype"]["readout"] = "telepathy"
    row = _row(_run([record]), "structural")
    assert row.passed is False
    assert row.details["n_failures"] == 1


def test_the_report_carries_the_callers_provenance_and_dataset_name() -> None:
    report = _run([_record(12.0, 2.0)])
    assert report.dataset_name == "Synthetic"
    assert report.provenance is PROVENANCE


def test_the_records_factory_is_called_once_per_pass_not_materialized() -> None:
    """A store larger than memory is streamed, so the factory is re-entered per row."""
    calls = 0
    records = [_record(12.0, 2.0)]

    def factory() -> Any:
        nonlocal calls
        calls += 1
        return iter(records)

    report = verify_promoter_activity_dataset_streaming(
        factory,
        dataset_name="Synthetic",
        provenance=PROVENANCE,
        expected_count=1,
        resolve_gene_name=_resolver({"b0002"}),
    )
    assert report.passed is True
    assert calls == 7
