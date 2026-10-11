# tests/torchcell/datamodels/test_genome_architecture_features.py
# [[tests.torchcell.datamodels.test_genome_architecture_features]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_genome_architecture_features.py
"""The genome-architecture vocabulary and the phenotype that validates against it.

Hermetic: the assay is a module constant and the phenotype is pure pydantic, so nothing
here reads ``$DATA_ROOT``.
"""

from __future__ import annotations

import pytest

from torchcell.datamodels.genome_architecture_features import (
    GENOME_ARCHITECTURE_ASSAYS,
    PUDDU2019_WGS_COVERAGE,
    ArchitectureBasis,
    architecture_assay,
)
from torchcell.datamodels.schema import (
    GenomeArchitecturePhenotype,
    SampleUnit,
    UncertaintyType,
)

CHROMOSOMES = [f"chr{i:02d}" for i in range(1, 17)]


def _diploid_profile(**overrides: float) -> dict[str, float]:
    """An unperturbed diploid profile: 2.0 per chromosome, no deviation, no event."""
    profile: dict[str, float] = {symbol: 2.0 for symbol in CHROMOSOMES}
    profile.update({"ANtot": 0.0, "GCR": 0.0, "rDNA": 125.0, "mitochondria": 17.5})
    profile.update(overrides)
    return profile


def test_the_puddu_assay_declares_the_28_released_columns_with_their_bases() -> None:
    assay = architecture_assay("puddu2019_wgs_coverage")
    assert assay is PUDDU2019_WGS_COVERAGE
    assert GENOME_ARCHITECTURE_ASSAYS == {"puddu2019_wgs_coverage": assay}
    assert len(assay.features) == 28
    assert sorted(assay.by_symbol) == sorted(
        [
            "2-micron",
            "ANtot",
            "CUP1",
            "GCR",
            "Telomeres",
            "Ty1",
            "Ty2",
            "Ty3",
            "Ty4",
            "Ty5",
            "mitochondria",
            "rDNA",
            *CHROMOSOMES,
        ]
    )
    assert assay.symbols_with_basis(
        ArchitectureBasis.copies_per_haploid_genome
    ) == frozenset({"rDNA", "CUP1", "Ty1", "Ty2", "Ty3", "Ty4", "Ty5"})
    assert assay.symbols_with_basis(ArchitectureBasis.copies_per_cell) == frozenset(
        {"mitochondria", "2-micron"}
    )
    assert assay.symbols_with_basis(
        ArchitectureBasis.chromosome_copies_per_cell
    ) == frozenset(CHROMOSOMES)
    assert assay.symbols_with_basis(ArchitectureBasis.reads_per_million) == frozenset(
        {"Telomeres"}
    )
    assert assay.by_symbol["rDNA"].region == "chromosome XII:452000-459000"
    assert assay.reference_assembly == "S288c_R64-1-1_Ensembl"


def test_an_unregistered_assay_name_raises_with_the_registered_ones() -> None:
    with pytest.raises(ValueError, match="unknown genome-architecture assay"):
        architecture_assay("puddu2019")


def test_a_rearrangement_count_must_be_integral_but_a_copy_number_need_not_be() -> None:
    assay = PUDDU2019_WGS_COVERAGE
    assay.validate_value("GCR", 3.0)
    assay.validate_value("chr12", 2.237544922)
    with pytest.raises(ValueError, match="must be integral"):
        assay.validate_value("GCR", 1.5)
    with pytest.raises(ValueError, match="finite value >= 0.0"):
        assay.validate_value("rDNA", -1.0)
    with pytest.raises(ValueError, match="finite value >= 0.0"):
        assay.validate_value("Telomeres", float("nan"))


def test_a_phenotype_refuses_a_key_the_assay_does_not_declare() -> None:
    with pytest.raises(ValueError, match="not features of assay"):
        GenomeArchitecturePhenotype(
            assay="puddu2019_wgs_coverage",
            genome_architecture={"rDNA": 125.0, "chrXVII": 2.0},
        )


def test_a_phenotype_refuses_an_aggregate_that_is_not_its_own_chromosome_sum() -> None:
    with pytest.raises(ValueError, match="chromosome copy numbers deviate"):
        GenomeArchitecturePhenotype(
            assay="puddu2019_wgs_coverage",
            genome_architecture=_diploid_profile(ANtot=0.5),
        )
    kept = GenomeArchitecturePhenotype(
        assay="puddu2019_wgs_coverage",
        genome_architecture=_diploid_profile(chr01=1.75, ANtot=0.25),
    )
    assert kept.genome_architecture["ANtot"] == 0.25


def test_the_aggregate_identity_is_only_checked_with_every_chromosome_present() -> None:
    """A reference that states the published aggregate alone is not a derived sum."""
    partial = GenomeArchitecturePhenotype(
        assay="puddu2019_wgs_coverage",
        genome_architecture={"ANtot": 0.15, "chr01": 2.0},
    )
    assert partial.genome_architecture["ANtot"] == 0.15


def test_a_dispersion_needs_its_uncertainty_type_and_the_reverse() -> None:
    with pytest.raises(ValueError, match="needs uncertainty_type"):
        GenomeArchitecturePhenotype(
            assay="puddu2019_wgs_coverage",
            genome_architecture={"rDNA": 125.0},
            genome_architecture_sd={"rDNA": 7.85},
        )
    with pytest.raises(ValueError, match="names the kind of a dispersion"):
        GenomeArchitecturePhenotype(
            assay="puddu2019_wgs_coverage",
            genome_architecture={"rDNA": 125.0},
            uncertainty_type=UncertaintyType.sample_sd,
        )
    with pytest.raises(ValueError, match="non-negative dispersion"):
        GenomeArchitecturePhenotype(
            assay="puddu2019_wgs_coverage",
            genome_architecture={"rDNA": 125.0},
            genome_architecture_sd={"rDNA": -0.1},
            uncertainty_type=UncertaintyType.sample_sd,
        )
    ok = GenomeArchitecturePhenotype(
        assay="puddu2019_wgs_coverage",
        genome_architecture={"rDNA": 124.99},
        genome_architecture_sd={"rDNA": 7.85},
        uncertainty_type=UncertaintyType.sample_sd,
        n_samples=8,
        sample_unit=SampleUnit.colony,
    )
    assert ok.label_statistic_name == "genome_architecture_sd"
    assert ok.graph_level == "global"
    assert ok.label_name == "genome_architecture"


def test_an_empty_profile_is_refused_and_the_repr_stays_a_summary() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        GenomeArchitecturePhenotype(
            assay="puddu2019_wgs_coverage", genome_architecture={}
        )
    phenotype = GenomeArchitecturePhenotype(
        assay="puddu2019_wgs_coverage",
        genome_architecture=_diploid_profile(),
        sample_id="SD0863b",
        sample_accession="ERS838232",
    )
    assert repr(phenotype) == (
        "GenomeArchitecturePhenotype(assay='puddu2019_wgs_coverage', features=20, "
        "sample_id='SD0863b')"
    )
    assert phenotype.assay_vocabulary is PUDDU2019_WGS_COVERAGE
