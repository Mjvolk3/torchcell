# tests/torchcell/verification/test_genome_architecture.py
# [[tests.torchcell.verification.test_genome_architecture]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_genome_architecture.py
"""The genome-architecture L0-L4 gate on hand-built records.

Hermetic: the records are built here, the "release" is a dict, the pinned quotes are
one ``SourcedValue`` whose source is a file written into ``tmp_path``, so nothing reads
``$DATA_ROOT``. Each test drives ONE rule to a failure and asserts the message and the
count it reports, which is what makes the gate worth running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from torchcell.datamodels.genome_architecture_features import PUDDU2019_WGS_COVERAGE
from torchcell.datamodels.schema import (
    BarcodedKanMxDeletionPerturbation,
    CultureEnvironment,
    GenomeArchitectureExperiment,
    GenomeArchitectureExperimentReference,
    GenomeArchitecturePhenotype,
    Genotype,
    MatingType,
    Media,
    Publication,
    SampleUnit,
    StrainBackground,
    StrainReferenceGenome,
    Temperature,
    UncertaintyType,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.genome_architecture import (
    genome_architecture_gene_set,
    verify_genome_architecture_records,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

CHROMOSOMES = [f"chr{i:02d}" for i in range(1, 17)]
SYMBOLS = list(PUDDU2019_WGS_COVERAGE.by_symbol)
PROVENANCE = Provenance(source_uri="si/si3.xlsx", citation_key="test", sha256="0" * 64)


def _profile(**overrides: float) -> dict[str, float]:
    """A full 28-feature profile of an unperturbed diploid colony."""
    values: dict[str, float] = {symbol: 2.0 for symbol in CHROMOSOMES}
    values.update(
        {
            "rDNA": 125.0,
            "CUP1": 14.0,
            "mitochondria": 17.5,
            "2-micron": 49.0,
            "Ty1": 37.0,
            "Ty2": 11.5,
            "Ty3": 2.0,
            "Ty4": 3.0,
            "Ty5": 1.2,
            "Telomeres": 756.0,
            "ANtot": 0.0,
            "GCR": 0.0,
        }
    )
    values.update(overrides)
    return values


def _environment() -> CultureEnvironment:
    return CultureEnvironment(
        media=Media(
            name="YPAD (YPD + adenine), liquid",
            state="liquid",
            is_synthetic=False,
            base_medium="YPAD",
        ),
        temperature=Temperature(value=30.0),
    )


def _reference() -> GenomeArchitectureExperimentReference:
    return GenomeArchitectureExperimentReference(
        dataset_name="test",
        genome_reference=StrainReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4743",
            ploidy="diploid",
            background=StrainBackground(
                name="BY4743",
                mating_type=MatingType.a_alpha,
                ploidy="diploid",
                provenance=[
                    SourcedValue(value="BY4743", provenance=PROVENANCE, quote="BY4743")
                ],
            ),
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=_environment(),
        phenotype_reference=GenomeArchitecturePhenotype(
            assay="puddu2019_wgs_coverage",
            genome_architecture={"rDNA": 124.99, **{c: 2.0 for c in CHROMOSOMES}},
            genome_architecture_sd={"rDNA": 7.85},
            uncertainty_type=UncertaintyType.sample_sd,
            n_samples=8,
            sample_unit=SampleUnit.colony,
            provenance_gaps=[
                ProvenanceGap(
                    field="genome_architecture",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    keys=[s for s in SYMBOLS if s not in {"rDNA", *CHROMOSOMES}],
                    note="no wild-type statistic released",
                )
            ],
        ),
    )


def _record(
    *, sample_id: str, gene: str = "YAL001C", **overrides: float
) -> dict[str, Any]:
    experiment = GenomeArchitectureExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                BarcodedKanMxDeletionPerturbation(
                    systematic_gene_name=gene,
                    perturbed_gene_name=gene,
                    barcode="ACGT",
                    downtag_barcode="TGCA",
                )
            ]
        ),
        environment=_environment(),
        phenotype=GenomeArchitecturePhenotype(
            assay="puddu2019_wgs_coverage",
            genome_architecture=_profile(**overrides),
            n_samples=1,
            sample_unit=SampleUnit.colony,
            sample_id=sample_id,
            sample_accession=f"ERS{sample_id}",
        ),
    )
    return {
        "experiment": experiment.model_dump(),
        "reference": _reference().model_dump(),
        "publication": Publication(
            doi="10.1038/s41586-019-1549-9",
            doi_url="https://doi.org/10.1038/s41586-019-1549-9",
        ).model_dump(),
    }


def _released(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        str(rec["experiment"]["phenotype"]["sample_id"]): dict(
            rec["experiment"]["phenotype"]["genome_architecture"]
        )
        for rec in records
    }


def _quote(tmp_path: Path) -> SourcedValue:
    """A pinned quote whose source file is written here, so the audit can pass."""
    text = "rho0 colonies were defined as having less than one mtDNA copy."
    path = tmp_path / "test" / "paper.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    from torchcell.verification.report import sha256_file

    return SourcedValue(
        value=1.0,
        provenance=Provenance(
            source_uri="paper.md", citation_key="test", sha256=sha256_file(path)
        ),
        quote=text,
    )


def _run(
    records: list[dict[str, Any]],
    tmp_path: Path,
    *,
    released: dict[str, dict[str, float]] | None = None,
    expected_count: int | None = None,
    expected_rho0: int = 0,
    expected_aneuploid_fraction: float = 0.0,
    workbook_cells: frozenset[str] = frozenset({"Unique colony ID"}),
    workbook_values: list[SourcedValue] | None = None,
) -> Any:
    return verify_genome_architecture_records(
        records,
        dataset_name="test",
        provenance=PROVENANCE,
        expected_count=len(records) if expected_count is None else expected_count,
        required_features=frozenset(SYMBOLS),
        released=_released(records) if released is None else released,
        paper_values=[_quote(tmp_path)],
        workbook_values=workbook_values or [],
        library_root=tmp_path,
        workbook_cells=workbook_cells,
        rho0_feature="mitochondria",
        rho0_threshold=1.0,
        expected_rho0=expected_rho0,
        expected_aneuploid_fraction=expected_aneuploid_fraction,
    )


def _result(report: Any, name: str) -> Any:
    return next(r for r in report.results if r.name == name)


def test_a_faithful_two_colony_store_passes_every_level(tmp_path: Path) -> None:
    records = [_record(sample_id="SD1b"), _record(sample_id="SD1b2")]
    report = _run(records, tmp_path)
    assert [r.name for r in report.results if not r.passed] == []
    assert report.passed is True
    assert _result(report, "release_fidelity").details["n_compared"] == 56
    assert _result(report, "feature_fidelity").details["n_features"] == 28
    assert _result(report, "sample_uniqueness").details["n_samples"] == 2
    assert genome_architecture_gene_set(records) == {"YAL001C"}


def test_two_records_of_one_colony_fail_sample_uniqueness(tmp_path: Path) -> None:
    records = [_record(sample_id="SD1b"), _record(sample_id="SD1b")]
    report = _run(records, tmp_path)
    result = _result(report, "sample_uniqueness")
    assert result.passed is False
    assert result.details["n_duplicated"] == 1
    assert "more than one record" in result.message


def test_a_stored_value_that_differs_from_the_release_fails_l2(tmp_path: Path) -> None:
    records = [_record(sample_id="SD1b")]
    released = _released(records)
    released["SD1b"]["rDNA"] = 200.0
    report = _run(records, tmp_path, released=released)
    result = _result(report, "release_fidelity")
    assert result.passed is False
    assert result.details["n_disagreements"] == 1
    assert result.details["disagreements"][0] == {
        "index": 0,
        "sample_id": "SD1b",
        "feature": "rDNA",
        "stored": 125.0,
        "released": 200.0,
    }


def test_a_colony_missing_from_the_release_fails_l2(tmp_path: Path) -> None:
    records = [_record(sample_id="SD1b")]
    report = _run(records, tmp_path, released={})
    result = _result(report, "release_fidelity")
    assert result.passed is False
    assert result.details["n_samples_not_in_release"] == 1


def test_a_short_profile_fails_assay_coverage(tmp_path: Path) -> None:
    record = _record(sample_id="SD1b")
    record["experiment"]["phenotype"]["genome_architecture"].pop("Ty5")
    report = _run([record], tmp_path, released=_released([record]))
    result = _result(report, "assay_coverage")
    assert result.passed is False
    assert result.details["missing_required"] == [
        {"index": 0, "missing_required": ["Ty5"]}
    ]


def test_the_rho0_count_and_aneuploid_fraction_are_checked_against_the_paper(
    tmp_path: Path,
) -> None:
    records = [
        _record(sample_id="SD1b", mitochondria=0.05),
        _record(sample_id="SD1b2", chr01=1.75, ANtot=0.25),
    ]
    report = _run(records, tmp_path, expected_rho0=1, expected_aneuploid_fraction=0.5)
    result = _result(report, "paper_statistics")
    assert result.passed is True
    assert result.details["n_rho0_served"] == 1
    assert result.details["n_aneuploid_served"] == 1
    assert result.details["aneuploid_fraction"] == 0.5
    wrong = _run(records, tmp_path, expected_rho0=303, expected_aneuploid_fraction=0.52)
    assert _result(wrong, "paper_statistics").passed is False


def test_a_wild_type_rho0_colony_counts_toward_the_paper_total(tmp_path: Path) -> None:
    """The paper counts every released colony; the store serves only the knockouts."""
    records = [_record(sample_id="SD1b", mitochondria=0.05)]
    report = verify_genome_architecture_records(
        records,
        dataset_name="test",
        provenance=PROVENANCE,
        expected_count=1,
        required_features=frozenset(SYMBOLS),
        released=_released(records),
        paper_values=[_quote(tmp_path)],
        workbook_values=[],
        library_root=tmp_path,
        workbook_cells=frozenset(),
        rho0_feature="mitochondria",
        rho0_threshold=1.0,
        expected_rho0=2,
        expected_aneuploid_fraction=0.0,
        wild_type_rho0=1,
    )
    result = _result(report, "paper_statistics")
    assert (result.passed, result.details["n_rho0_total"]) == (True, 2)


def test_a_legend_cell_the_workbook_does_not_hold_fails_the_audit(
    tmp_path: Path,
) -> None:
    absent = SourcedValue(
        value="x", provenance=PROVENANCE, quote="Not a cell of this workbook"
    )
    report = _run([_record(sample_id="SD1b")], tmp_path, workbook_values=[absent])
    result = _result(report, "provenance_audit")
    assert result.passed is False
    assert result.details["n_audited"] == 2
    assert "is not in the sheet" in result.details["failed"][0]


def test_a_reference_with_no_dispersion_fails_the_reference_rule(
    tmp_path: Path,
) -> None:
    record = _record(sample_id="SD1b")
    record["reference"]["phenotype_reference"]["genome_architecture_sd"] = None
    record["reference"]["phenotype_reference"]["uncertainty_type"] = None
    report = _run([record], tmp_path, released=_released([record]))
    result = _result(report, "reference_profile")
    assert result.passed is False
    assert "the reference profile carries no dispersion" in result.details["problems"]


def test_a_dataset_mixing_two_assays_is_refused_outright(tmp_path: Path) -> None:
    records = [_record(sample_id="SD1b"), _record(sample_id="SD1b2")]
    records[1]["experiment"]["phenotype"]["assay"] = "another_assay"
    with pytest.raises(ValueError, match="carries one assay vocabulary"):
        _run(records, tmp_path)


def test_a_required_feature_outside_the_assay_is_refused_outright(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="are not features of assay"):
        verify_genome_architecture_records(
            [_record(sample_id="SD1b")],
            dataset_name="test",
            provenance=PROVENANCE,
            expected_count=1,
            required_features=frozenset({"chrXVII"}),
            released={},
            paper_values=[],
            workbook_values=[],
            library_root=tmp_path,
            workbook_cells=frozenset(),
            rho0_feature="mitochondria",
            rho0_threshold=1.0,
            expected_rho0=0,
            expected_aneuploid_fraction=0.0,
        )
