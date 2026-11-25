# tests/torchcell/datasets/ecoli/test_rousset2018.py
# [[tests.torchcell.datasets.ecoli.test_rousset2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_rousset2018.py
"""Rousset 2018 loader: the four stored screens, tables, symbols, records, mirror.

The synthetic tests run everywhere. The identifier tests read the REAL
``EcoliK12MG1655Genome`` class over the synthetic MG1655 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` through a stubbed ``resolve``
with the network refused. Derived expectations for that assembly: ``thrL``, ``thrA``,
``thrW``, ``proB`` and ``proC`` are symbols of b0001, b0002, b0003, b0005 and b0006;
``yaaP`` and ``insZ`` are symbols of the pseudogenes b0004 and b0007; ``thrA1`` is a
synonym of b0002, so ``thrA`` and ``thrA1`` collide on one locus; ``PRO2`` matches
b0005 and b0006 case-insensitively, so it is ambiguous; ``nope`` is in no layer.

The data tests (``--data``) audit every module-level ``SourcedValue`` against the
sha256-pinned paper OCR of BOTH mirrors, read the three real tables from the raw mirror,
pin the measured retention arithmetic and identifier histogram on the deposited MG1655
set, and re-measure the growth-screen de-duplication against Cui 2018's own pinned
table.
"""

from __future__ import annotations

import hashlib
import os
import os.path as osp
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

import torchcell.datasets.ecoli.cui2018 as cui
import torchcell.datasets.ecoli.rousset2018 as r
from tests.torchcell.datasets._genome_injection_fakes import install_bacterial_fakes
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.media import LB, MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    ASSEMBLY_SET_ACCESSIONS,
    BACTERIAL_ASSEMBLY_SETS,
    AssayType,
    ComponentDefinition,
    ConcentrationUnit,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    MeasurementType,
    MediaComponentRole,
    PhagePerturbation,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainReferenceGenome,
)
from torchcell.datasets.bacteria_common import BacterialGenomeInjector
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.literature.manifest import Manifest, RetrievalMethod
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.report import Level
from torchcell.verification.sourced import (
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

# --------------------------------------------------------------------------- #
# Synthetic tables
# --------------------------------------------------------------------------- #
#: One row per (spacer, gene, guide orientation); the gene's orientation is fixed to
#: ``+``, so a ``-`` guide targets the coding strand and a ``+`` guide the template one.
_ROWS: tuple[tuple[str, str | None, str], ...] = (
    ("A" * 20, "thrL", "-"),
    ("C" * 20, "thrA", "-"),
    ("G" * 20, "thrW", "+"),
    ("T" * 20, None, "+"),
)


def _growth(rows: tuple[tuple[str, str | None, str], ...] = _ROWS) -> pd.DataFrame:
    """An S1-shaped table: ``coding`` is released and intergenic rows carry no gene."""
    return pd.DataFrame(
        {
            "target": [spacer for spacer, _, _ in rows],
            "position": list(range(100, 100 + len(rows))),
            "ori": [ori for _, _, ori in rows],
            "coding": [None if gene is None else ori == "-" for _, gene, ori in rows],
            "gene": [gene for _, gene, _ in rows],
            "essential": [gene == "thrA" for _, gene, _ in rows],
            "gene_left": [None if gene is None else 1 for _, gene, _ in rows],
            "gene_right": [None if gene is None else 9 for _, gene, _ in rows],
            "gene_ori": [None if gene is None else "+" for _, gene, _ in rows],
            "log2FC": [-1.5 - i for i in range(len(rows))],
            "padj": [1e-3 / (i + 1) for i in range(len(rows))],
            "gamma": [0.9 + i / 100 for i in range(len(rows))],
        }
    )


def _phage(genes: tuple[str, ...] = ("thrL", "thrA")) -> pd.DataFrame:
    """An S4-shaped table: every row is a coding-strand guide (``ori`` != ``gene_ori``)."""
    return pd.DataFrame(
        {
            "target": [chr(ord("A") + i) * 20 for i in range(len(genes))],
            "position": list(range(100, 100 + len(genes))),
            "ori": ["-"] * len(genes),
            "gene": list(genes),
            "essential": [gene == "thrA" for gene in genes],
            "gene_left": [1] * len(genes),
            "gene_right": [9] * len(genes),
            "gene_ori": ["+"] * len(genes),
            "log2FC_lambda": [0.4 - i for i in range(len(genes))],
            "log2FC_T4": [-1.1 + i for i in range(len(genes))],
            "log2FC_186": [0.2 + i for i in range(len(genes))],
        }
    )


def _transduction(genes: tuple[str, ...] = ("thrL", "thrA")) -> pd.DataFrame:
    """An S6-shaped table: ``pos`` and ``gene_right`` before ``gene_left``."""
    return pd.DataFrame(
        {
            "target": [chr(ord("A") + i) * 20 for i in range(len(genes))],
            "pos": list(range(100, 100 + len(genes))),
            "ori": ["-"] * len(genes),
            "gene": list(genes),
            "essential": [gene == "thrA" for gene in genes],
            "gene_right": [9] * len(genes),
            "gene_left": [1] * len(genes),
            "gene_ori": ["+"] * len(genes),
            "log2FC": [-0.3 - i for i in range(len(genes))],
        }
    )


def _write(tmp_path: Path, filename: str, frame: pd.DataFrame) -> Path:
    path = tmp_path / filename
    frame.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# The five stored screens
# --------------------------------------------------------------------------- #
def test_the_five_stored_screens_name_their_table_column_strain_and_phage() -> None:
    """#905: the growth screen (S1 Table, LC-E75) is stored beside the four phage ones."""
    assert [
        (c.screen_id, c.table, c.column, c.strain, c.phage) for c in r.CONDITIONS
    ] == [
        ("growth_LC-E75", r.GROWTH_TABLE, "log2FC", "LC-E75", None),
        ("phage_lambda", r.PHAGE_TABLE, "log2FC_lambda", "FR-E01", "lambda"),
        ("phage_T4", r.PHAGE_TABLE, "log2FC_T4", "FR-E01", "T4"),
        ("phage_186cIts", r.PHAGE_TABLE, "log2FC_186", "FR-E01", "186cIts"),
        ("lambda_transduction", r.TRANSDUCTION_TABLE, "log2FC", "FR-E01", "lambda"),
    ]
    assert r.TABLE_COLUMNS == {
        r.GROWTH_TABLE: ("log2FC", "padj"),
        r.PHAGE_TABLE: ("log2FC_lambda", "log2FC_T4", "log2FC_186"),
        r.TRANSDUCTION_TABLE: ("log2FC",),
    }


def test_only_the_transduction_arm_is_not_a_pooled_barcode_growth_assay() -> None:
    by_assay = {c.screen_id: c.assay_type for c in r.CONDITIONS}
    assert by_assay["lambda_transduction"] is AssayType.other
    assert {
        assay for screen, assay in by_assay.items() if screen != "lambda_transduction"
    } == {AssayType.pooled_competitive_growth_barcode}
    transduction = next(c for c in r.CONDITIONS if c.screen_id == "lambda_transduction")
    assert "packaged cosmid, not the surviving cell pool" in transduction.units


# --------------------------------------------------------------------------- #
# #760: the PAIR of loaders cannot store one measurement twice
#
# Each loader is correct alone; only the pair could be wrong, so these read BOTH
# modules. The de-duplication is structural: Rousset's screens do not include the
# growth screen, and the two loaders' screen vocabularies are disjoint, so no
# (spacer, screen_id) measurement key can exist in both stores. Measured on the built
# dev LMDBs by
# ``experiments/036-dataset-fixes-before-kg-build/scripts/rousset2018_cui2018_overlap_verification.py``:
# 0 shared experiment content ids and 0 shared (spacer, screen_id) keys over 68,436 +
# 141,542 records, on 16,979 spacers both libraries carry.
# --------------------------------------------------------------------------- #
def test_the_two_loaders_screen_vocabularies_are_disjoint() -> None:
    """Rousset's growth screen and Cui's LC-E75 screen are two releases of one screen's
    reads (#878); since #905 both are stored, each under its own publication and screen
    id, so a (spacer, screen_id) key names one release.
    """
    rousset_screens = {c.screen_id for c in r.CONDITIONS}
    cui_screens = {screen_id for screen_id, _ in cui.SCREENS}
    assert cui_screens == {"LC-E18", "LC-E75"}
    assert rousset_screens.isdisjoint(cui_screens)


def test_the_growth_screen_deferral_quote_names_cui_as_the_source() -> None:
    deferral = r.GROWTH_SCREEN_DEFERRAL
    assert deferral.provenance.sha256 == r.PAPER_MD_SHA256
    assert (
        "The data for the screen performed with strain LC-E75 grown in rich medium was "
        "obtained from our previous study [26]"
    ) in deferral.quote
    assert "Cui 2018" in str(deferral.note)
    assert r.GROWTH_SCREEN_REPLICATES.value == 3
    assert "in triplicates" in r.GROWTH_SCREEN_REPLICATES.quote


def test_the_growth_screen_runs_in_cui_2018s_culture_and_host() -> None:
    growth = r.CONDITIONS[0]
    assert r.environment(growth) == cui.screen_environment()
    assert r.BACKGROUNDS["LC-E75"] == cui.strain_background("LC-E75")


def test_a_growth_phenotype_carries_its_padj_and_the_deseq2_correction() -> None:
    growth = r.phenotype(-2.0, r.CONDITIONS[0], 0.004)
    assert growth.environment_response_p_value_adjusted == 0.004
    assert growth.p_value_adjustment_method == "benjamini_hochberg"
    assert growth.environment_response_p_value is None
    assert growth.n_samples == 3
    assert r.PADJ_CORRECTION.provenance.sha256 == r.DESEQ2_REFERENCE_SHA256
    assert "Benjamini and Hochberg" in r.PADJ_CORRECTION.quote
    phage = r.phenotype(-2.0, r.CONDITIONS[1])
    assert phage.environment_response_p_value_adjusted is None
    assert phage.p_value_adjustment_method is None


def test_a_growth_record_names_s1_as_its_identifier_source() -> None:
    stored = r.StoredGuide(
        spacer="A" * 20,
        reported_symbol="thrL",
        identifier_kind=r.GROWTH_IDENTIFIER_KIND,
    )
    (perturbation,) = r.genotype(stored).perturbations
    assert isinstance(perturbation, ReleasedIdentifierPerturbation)
    assert perturbation.identifier_kind == r.GROWTH_IDENTIFIER_KIND
    assert "S1" in r.GROWTH_IDENTIFIER_KIND
    (default,) = r.genotype(
        r.StoredGuide(spacer="A" * 20, reported_symbol="thrL")
    ).perturbations
    assert isinstance(default, ReleasedIdentifierPerturbation)
    assert default.identifier_kind == r.IDENTIFIER_KIND


def test_the_synthetic_build_stores_no_cui_screen_id(
    tmp_path: Path, mirrored: Path
) -> None:
    dataset = r.CrispriScreenRousset2018Dataset(root=str(tmp_path / "dataset"))
    stored = {
        dataset[index]["experiment"]["phenotype"]["screen_id"]
        for index in range(len(dataset))
    }
    assert stored == {c.screen_id for c in r.CONDITIONS}
    assert stored.isdisjoint({screen_id for screen_id, _ in cui.SCREENS})


# --------------------------------------------------------------------------- #
# Media, environments, phenotypes, genotypes
# --------------------------------------------------------------------------- #
def test_the_one_medium_derives_from_lb_and_asserts_no_lb_amounts() -> None:
    medium = r.ROUSSET2018_LB_MALTOSE_CACL2
    assert medium.base_medium == "LB"
    assert MEDIA_LIBRARY["LB"] is LB
    assert medium.state == "liquid"
    assert medium.is_synthetic is False
    lb_components = medium.components[:3]
    assert [c.compound.name for c in lb_components] == [
        "tryptone",
        "yeast extract",
        "sodium chloride",
    ]
    assert all(c.concentration is None for c in lb_components)
    assert [c.definition for c in medium.components[:2]] == [
        ComponentDefinition.intrinsically_undefined
    ] * 2
    assert not hasattr(r, "ROUSSET2018_LB")


def _dosed(medium: Any) -> list[tuple[str, Any, float | None, Any]]:
    """Every component of ``medium`` the paper gives an amount for."""
    return [
        (
            c.compound.name,
            c.role,
            c.concentration.value if c.concentration else None,
            c.concentration.unit if c.concentration else None,
        )
        for c in medium.components
        if c.concentration is not None
    ]


def test_the_phage_medium_adds_the_atc_maltose_and_calcium_the_paper_lists() -> None:
    extra = r.ROUSSET2018_LB_MALTOSE_CACL2.components[3:]
    assert _dosed(r.ROUSSET2018_LB_MALTOSE_CACL2) == [
        (
            "anhydrotetracycline",
            MediaComponentRole.other,
            1.0,
            ConcentrationUnit.micromolar,
        ),
        (
            "maltose",
            MediaComponentRole.carbon_source,
            0.2,
            ConcentrationUnit.percent_w_v,
        ),
        (
            "calcium chloride",
            MediaComponentRole.bulk_salt,
            5.0,
            ConcentrationUnit.millimolar,
        ),
    ]
    assert all(c.compound.inchikey is not None for c in extra)


def test_the_atc_inducer_is_a_medium_component_at_its_stated_dose() -> None:
    """ATc is in the medium, not on the environment axis.

    The paper puts it there ("LB containing 1 microM aTc, 0.2% Maltose and 5 mM CaCl2"),
    it is constant across the dataset rather than the varied condition, and leaving the
    environment axis to the phage alone is what lets the adapter conf enable `phage
    perturbation` without also enabling `environment perturbation`.
    """
    (atc,) = [
        c
        for c in r.ROUSSET2018_LB_MALTOSE_CACL2.components
        if c.compound.name == "anhydrotetracycline"
    ]
    assert atc.compound.inchikey == "KTTKGQINVKPHLY-DOCRCCHOSA-N"
    assert atc.provenance
    assert atc.concentration is not None
    assert (atc.concentration.value, atc.concentration.unit) == (
        1.0,
        ConcentrationUnit.micromolar,
    )


def test_a_phage_environment_carries_exactly_its_phage_at_moi_one() -> None:
    environment = r.environment(r.CONDITIONS[2])
    assert environment.media is r.ROUSSET2018_LB_MALTOSE_CACL2
    assert environment.temperature is not None
    assert environment.temperature.value == 37.0
    assert environment.duration_hours == 2.0
    assert environment.duration_generations is None
    assert environment.provenance_gaps == []
    (phage,) = environment.perturbations
    assert isinstance(phage, PhagePerturbation)
    assert phage.name == "T4"
    assert phage.multiplicity_of_infection == 1.0
    assert phage.host_of_propagation == "MG1655"
    assert (
        phage.family,
        phage.genome_type,
        phage.ncbi_taxid,
        phage.genome_accession,
        phage.titer_pfu_per_ml,
    ) == (None, None, None, None, None)


def test_the_two_lambda_screens_share_an_environment_and_split_on_screen_id() -> None:
    challenge = next(c for c in r.CONDITIONS if c.screen_id == "phage_lambda")
    transduction = next(c for c in r.CONDITIONS if c.screen_id == "lambda_transduction")
    assert r.environment(challenge) == r.environment(transduction)
    assert r.phenotype(0.1, challenge).screen_id == "phage_lambda"
    assert r.phenotype(0.1, transduction).screen_id == "lambda_transduction"


def test_each_screen_carries_exactly_one_environment_perturbation() -> None:
    """A phage on every phage screen; Cui's aTc on the growth screen (#905)."""
    for condition in r.CONDITIONS:
        (perturbation,) = r.environment(condition).perturbations
        if condition.phage is None:
            assert isinstance(perturbation, SmallMoleculePerturbation)
            assert perturbation.compound.name == "anhydrotetracycline"
        else:
            assert isinstance(perturbation, PhagePerturbation)


def test_every_stored_record_is_environment_perturbed() -> None:
    assert set(r.EXPECTED_SCREEN_CENSUS) == {c.screen_id for c in r.CONDITIONS}
    assert sum(r.EXPECTED_SCREEN_CENSUS.values()) == 23372 + 4 * 17220 == 92252
    for condition in r.CONDITIONS:
        assert len(r.environment(condition).perturbations) == 1


def test_a_phenotype_is_a_signed_log2_ratio_over_three_biological_replicates() -> None:
    phenotype = r.phenotype(-9.0, r.CONDITIONS[2])
    assert phenotype.measurement_type is MeasurementType.log2_ratio
    assert phenotype.assay_type is AssayType.pooled_competitive_growth_barcode
    assert phenotype.environment_response == -9.0
    assert phenotype.n_samples == 3
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    assert phenotype.environment_response_uncertainty is None
    assert phenotype.environment_response_se is None
    assert phenotype.category is None
    assert [(g.field, g.reason) for g in phenotype.provenance_gaps] == [
        (
            "environment_response_uncertainty",
            ProvenanceGapReason.not_reported_by_primary,
        )
    ]


def test_the_reference_response_is_zero_for_every_screen() -> None:
    for condition in r.CONDITIONS:
        reference = r.reference_phenotype(condition)
        assert reference.environment_response == 0.0
        assert reference.n_samples is None
        assert reference.gapped_fields() == {"n_samples"}
        assert reference.screen_id == condition.screen_id


def test_a_genotype_is_crispri_of_its_released_symbol_by_the_reported_spacer() -> None:
    stored = r.StoredGuide(spacer="ACCACGCACTCTGACCATCT", reported_symbol="sucA")
    (perturbation,) = r.genotype(stored).perturbations
    assert isinstance(perturbation, ReleasedIdentifierPerturbation)
    assert perturbation.perturbation_type == "released_identifier"
    assert perturbation.systematic_gene_name == "sucA"
    assert perturbation.perturbed_gene_name == "sucA"
    assert perturbation.gene_namespace == "ecoli_k12_mg1655_bnumber"
    assert perturbation.released_kind is ReleasedKind.crispr_interference
    assert perturbation.strain_label == "ACCACGCACTCTGACCATCT"
    assert perturbation.identifier_kind == r.IDENTIFIER_KIND
    assert perturbation.crispr is not None
    assert perturbation.crispr.effector == "dCas9"
    assert perturbation.crispr.guide_sequence == "ACCACGCACTCTGACCATCT"
    assert perturbation.crispr.n_guides == 1
    assert not hasattr(r, "resolve_symbols")
    assert not hasattr(r, "canonical_symbol")


def test_two_guides_on_one_gene_are_two_strains_and_one_gene() -> None:
    guides = [
        r.genotype(r.StoredGuide(spacer=spacer, reported_symbol="sucA")).perturbations[
            0
        ]
        for spacer in ("A" * 20, "C" * 20)
    ]
    assert all(isinstance(g, ReleasedIdentifierPerturbation) for g in guides)
    first, second = guides
    assert isinstance(first, ReleasedIdentifierPerturbation)
    assert isinstance(second, ReleasedIdentifierPerturbation)
    assert first.systematic_gene_name == second.systematic_gene_name
    assert first.crispr is not None and second.crispr is not None
    assert first.crispr.guide_sequence != second.crispr.guide_sequence


def test_the_two_host_strains_are_backgrounds_on_the_mg1655_assembly() -> None:
    """FR-E01 for the phage screens; LC-E75, Cui 2018's own record, for growth."""
    assert set(r.BACKGROUNDS) == {"FR-E01", "LC-E75"}
    for label, background in r.BACKGROUNDS.items():
        assert background.name == label
        assert background.reference_strain == "MG1655"
        assert background.assembly_set == "ecoli_K12_MG1655_ASM584v2"
        assert background.parents == ["MG1655"]
        assert background.alleles == []
        assert background.provenance
        assert background.gapped_fields() == set()
    assert "HK022" in str(r.FR_E01_BACKGROUND.construction)
    assert r.LC_E75_CASSETTE.value == "LC-E75"
    assert "Cui 2018" in str(r.LC_E75_CASSETTE.note)


# --------------------------------------------------------------------------- #
# Reading the three tables
# --------------------------------------------------------------------------- #
def test_read_table_keeps_the_release_as_given(tmp_path: Path) -> None:
    frame = r.read_table(_write(tmp_path, r.GROWTH_TABLE, _growth()), r.GROWTH_TABLE)
    assert list(frame.columns) == list(r.TABLE_HEADERS[r.GROWTH_TABLE])
    assert frame["gene"].isna().tolist() == [False, False, False, True]
    assert frame["log2FC"].tolist() == [-1.5, -2.5, -3.5, -4.5]


def test_read_table_refuses_a_changed_header(tmp_path: Path) -> None:
    frame = _growth().rename(columns={"gamma": "Gamma"})
    with pytest.raises(ValueError, match="header"):
        r.read_table(_write(tmp_path, r.GROWTH_TABLE, frame), r.GROWTH_TABLE)


def test_read_table_refuses_a_spacer_that_is_not_twenty_nucleotides(
    tmp_path: Path,
) -> None:
    frame = _growth()
    frame.loc[0, "target"] = "ACGT"
    with pytest.raises(ValueError, match="20 nt"):
        r.read_table(_write(tmp_path, r.GROWTH_TABLE, frame), r.GROWTH_TABLE)


def test_read_table_refuses_a_repeated_spacer(tmp_path: Path) -> None:
    frame = _growth()
    frame.loc[1, "target"] = "A" * 20
    with pytest.raises(ValueError, match="repeats spacers"):
        r.read_table(_write(tmp_path, r.GROWTH_TABLE, frame), r.GROWTH_TABLE)


def test_read_table_refuses_an_empty_consumed_value(tmp_path: Path) -> None:
    frame = _phage()
    frame.loc[1, "log2FC_T4"] = None
    with pytest.raises(ValueError, match="column log2FC_T4 has empty cells"):
        r.read_table(_write(tmp_path, r.PHAGE_TABLE, frame), r.PHAGE_TABLE)


def test_coding_strand_keeps_only_in_gene_coding_rows_of_the_growth_table(
    tmp_path: Path,
) -> None:
    frame = r.read_table(_write(tmp_path, r.GROWTH_TABLE, _growth()), r.GROWTH_TABLE)
    assert r.coding_strand(frame, r.GROWTH_TABLE).tolist() == [True, True, False, False]


def test_coding_strand_refuses_a_growth_row_whose_released_call_disagrees(
    tmp_path: Path,
) -> None:
    frame = _growth()
    frame.loc[0, "coding"] = False
    table = r.read_table(_write(tmp_path, r.GROWTH_TABLE, frame), r.GROWTH_TABLE)
    with pytest.raises(ValueError, match="disagrees with"):
        r.coding_strand(table, r.GROWTH_TABLE)


def test_coding_strand_requires_every_phage_table_row_to_be_coding_strand(
    tmp_path: Path,
) -> None:
    frame = r.read_table(_write(tmp_path, r.PHAGE_TABLE, _phage()), r.PHAGE_TABLE)
    assert r.coding_strand(frame, r.PHAGE_TABLE).tolist() == [True, True]
    template = _phage()
    template.loc[1, "ori"] = "+"
    table = r.read_table(_write(tmp_path, r.PHAGE_TABLE, template), r.PHAGE_TABLE)
    with pytest.raises(ValueError, match="1 rows target the template strand"):
        r.coding_strand(table, r.PHAGE_TABLE)


def test_the_transduction_table_header_keeps_the_release_spellings(
    tmp_path: Path,
) -> None:
    frame = r.read_table(
        _write(tmp_path, r.TRANSDUCTION_TABLE, _transduction()), r.TRANSDUCTION_TABLE
    )
    assert list(frame.columns)[1] == "pos"
    assert list(frame.columns)[5:7] == ["gene_right", "gene_left"]


# --------------------------------------------------------------------------- #
# Identifiers on the synthetic MG1655 assembly
# --------------------------------------------------------------------------- #
@pytest.fixture
def mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The synthetic MG1655 genome; the network refuses."""
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    return EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=True)


class _Locus:
    def __init__(self, symbol: str | None) -> None:
        self.symbol = symbol


class _Resolution:
    def __init__(self, systematic_name: str | None) -> None:
        self.systematic_name = systematic_name


class _SymbolGenome:
    """The two attributes ``canonical_symbol`` reads, with a symbol shared by two loci."""

    def __init__(self) -> None:
        self.genbank = SimpleNamespace(
            loci={"b0010": _Locus(None), "b0011": _Locus("dup"), "b0012": _Locus("dup")}
        )

    def resolve_gene_name(self, name: str) -> _Resolution:
        return _Resolution("b0012" if name == "dup" else None)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _fake_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Three stand-in files whose sha256s replace the pinned ones."""
    paths: dict[str, Path] = {}
    pins: dict[str, str] = {}
    for filename in r.TABLE_SHA256:
        payload = f"{filename} bytes".encode()
        path = tmp_path / filename
        path.write_bytes(payload)
        paths[filename] = path
        pins[filename] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(r, "TABLE_SHA256", pins)
    return paths


def test_deposit_is_idempotent_and_refuses_a_changed_mirror_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fake_tables(tmp_path, monkeypatch)
    data_root = str(tmp_path / "data_root")
    root = r.deposit_raw_mirror(table_paths=paths, data_root=data_root)
    r.deposit_raw_mirror(table_paths=paths, data_root=data_root)
    assert root == Path(data_root) / r.RAW_DIR_REL
    manifest = r.load_manifest(data_root)
    assert [record.path for record in manifest.files] == [
        r.table_rel(filename) for filename in r.TABLE_SHA256
    ]
    growth = manifest.files[0]
    assert growth.retrieval is not None
    assert growth.retrieval.method is RetrievalMethod.pmc_cloud
    assert growth.retrieval.params == {"key": f"PMC6242692.1/{r.GROWTH_TABLE}"}
    assert growth.retrieval.source_url == (
        f"https://pmc-oa-opendata.s3.amazonaws.com/PMC6242692.1/{r.GROWTH_TABLE}"
    )
    assert (
        r.manifest_sha256(manifest, r.table_rel(r.GROWTH_TABLE))
        == (r.TABLE_SHA256[r.GROWTH_TABLE])
    )
    (root / r.table_rel(r.PHAGE_TABLE)).write_bytes(b"changed upstream")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        r.deposit_raw_mirror(table_paths=paths, data_root=data_root)


def test_deposit_refuses_bytes_that_are_not_a_pinned_table(tmp_path: Path) -> None:
    paths = {filename: tmp_path / "wrong" for filename in r.TABLE_SHA256}
    (tmp_path / "wrong").write_bytes(b"some other file")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        r.deposit_raw_mirror(table_paths=paths, data_root=str(tmp_path))


def test_deposit_refuses_a_partial_set_of_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fake_tables(tmp_path, monkeypatch)
    del paths[r.PHAGE_TABLE]
    with pytest.raises(RuntimeError, match="no retrieved bytes given for"):
        r.deposit_raw_mirror(table_paths=paths, data_root=str(tmp_path / "dr"))


def test_manifest_sha256_refuses_a_path_the_manifest_does_not_record() -> None:
    manifest = Manifest(citation_key=r.CITATION_KEY)
    with pytest.raises(KeyError, match="is not in the raw-mirror manifest"):
        r.manifest_sha256(manifest, r.table_rel(r.GROWTH_TABLE))


def test_the_mirror_dir_and_the_table_urls_are_built_from_the_pinned_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_ROOT", "/dr")
    assert r.raw_mirror_dir() == Path("/dr") / r.RAW_DIR_REL
    assert r.table_url(r.TRANSDUCTION_TABLE).endswith(
        f"PMC6242692.1/{r.TRANSDUCTION_TABLE}"
    )


# --------------------------------------------------------------------------- #
# Registration and injection
# --------------------------------------------------------------------------- #
def test_the_loader_is_registered_and_receives_the_mg1655_genome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert dataset_registry["CrispriScreenRousset2018Dataset"] is (
        r.CrispriScreenRousset2018Dataset
    )
    log = install_bacterial_fakes(monkeypatch)
    kwargs = BacterialGenomeInjector("/dr").genome_kwargs(
        r.CrispriScreenRousset2018Dataset
    )
    # the loader declares no genome parameter, so the injector fills nothing
    assert kwargs == {}
    assert [name for name, _ in log] == []
    dataset = r.CrispriScreenRousset2018Dataset.__new__(
        r.CrispriScreenRousset2018Dataset
    )
    assert dataset.experiment_class is EnvironmentResponseExperiment
    assert dataset.reference_class is (EnvironmentResponseExperimentReference)
    assert dataset.raw_file_names == [
        r.GROWTH_TABLE,
        r.PHAGE_TABLE,
        r.TRANSDUCTION_TABLE,
        r.ESSENTIALITY_COUNTS,
        r.PHAGE_COUNTS,
    ]


# --------------------------------------------------------------------------- #
# Supplementary verifier row
# --------------------------------------------------------------------------- #
def _records(census: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"experiment": {"phenotype": {"screen_id": screen}}}
        for screen, count in census.items()
        for _ in range(count)
    ]


def test_the_screen_census_pins_the_per_screen_split() -> None:
    release = {
        "growth_LC-E75": 23372,
        "phage_lambda": 17220,
        "phage_T4": 17220,
        "phage_186cIts": 17220,
        "lambda_transduction": 17220,
    }
    passed = r.screen_census(_records(release))
    assert passed.passed
    assert passed.details["census"] == release
    assert sum(release.values()) == r.EXPECTED_RECORDS
    shifted = dict(release)
    shifted["phage_T4"] += 1
    shifted["phage_186cIts"] -= 1
    failed = r.screen_census(_records(shifted))
    assert not failed.passed
    assert "is not" in failed.message


def _test_record(**experiment_fields: Any) -> dict[str, Any]:
    blank = dict.fromkeys(r.RELEASED_TEST_FIELDS)
    return {
        "experiment": {"phenotype": {**blank, **experiment_fields}},
        "reference": {"phenotype_reference": dict(blank)},
    }


def test_released_test_where_released_passes_on_the_released_pattern() -> None:
    growth = _test_record(
        screen_id="growth_LC-E75",
        environment_response_p_value_adjusted=0.01,
        p_value_adjustment_method="benjamini_hochberg",
    )
    phage = _test_record(screen_id="phage_T4")
    result = r.released_test_where_released([growth, phage, phage])
    assert result.passed
    assert result.level == Level.L2
    assert result.details == {"n_records": 3, "n_growth": 1, "wrong": {}}


def test_released_test_where_released_counts_every_departure() -> None:
    bare_growth = _test_record(screen_id="growth_LC-E75")
    carrying_phage = _test_record(
        screen_id="phage_T4",
        environment_response_p_value_adjusted=0.01,
        p_value_adjustment_method="benjamini_hochberg",
    )
    wrong_method = _test_record(
        screen_id="growth_LC-E75",
        environment_response_p_value_adjusted=0.01,
        p_value_adjustment_method="bonferroni",
    )
    result = r.released_test_where_released([bare_growth, carrying_phage, wrong_method])
    assert not result.passed
    assert result.details["wrong"] == {
        "experiment.environment_response_p_value_adjusted": 2,
        "experiment.p_value_adjustment_method": 2,
        "experiment.p_value_adjustment_method_value": 2,
    }


def test_the_released_test_fields_exist_on_the_phenotype() -> None:
    fields = EnvironmentResponsePhenotype.model_fields
    assert set(r.RELEASED_TEST_FIELDS) <= set(fields)
    for condition in r.CONDITIONS:
        built = r.reference_phenotype(condition)
        assert {name: getattr(built, name) for name in r.RELEASED_TEST_FIELDS} == (
            dict.fromkeys(r.RELEASED_TEST_FIELDS)
        )


def test_only_the_growth_table_releases_a_test_column() -> None:
    assert [table for table, header in r.TABLE_HEADERS.items() if "padj" in header] == [
        r.GROWTH_TABLE
    ]


# --------------------------------------------------------------------------- #
# End to end on a synthetic release
# --------------------------------------------------------------------------- #
#: The synthetic S1 Table: one kept coding-strand guide per storable symbol, plus one
#: row for each drop rule. ``thrA`` and ``thrA1`` are the collision pair, ``PRO2`` the
#: ambiguous symbol, ``nope`` the retired one, the ``+``-strand ``thrL`` guide the
#: template-strand case and the geneless row the intergenic one.
_SYNTHETIC_GROWTH: tuple[tuple[str, str | None, str], ...] = (
    ("A" * 20, "thrL", "-"),
    ("C" * 20, "thrL", "+"),
    ("G" * 20, "thrA", "-"),
    ("T" * 20, "thrA1", "-"),
    ("AC" * 10, "thrW", "-"),
    ("AG" * 10, "PRO2", "-"),
    ("AT" * 10, "nope", "-"),
    ("CG" * 10, None, "+"),
    ("CT" * 10, "yaaP", "-"),
)
#: The synthetic phage/transduction library: two storable genes and the retired symbol.
_SYNTHETIC_PHAGE_GENES = ("thrL", "thrW", "nope")


def _pin(
    strain: str, *, background: Any = None, data_root: str | None = None
) -> ReferenceGenome:
    """``assembly_reference`` without the deposited assembly report."""
    assembly_set = BACTERIAL_ASSEMBLY_SETS[strain]
    if background is None:
        return ReferenceGenome(
            species="Escherichia coli",
            strain=strain,
            assembly_set=cast(Any, assembly_set),
            assembly_accession=ASSEMBLY_SET_ACCESSIONS[assembly_set][0],
        )
    return StrainReferenceGenome(
        species="Escherichia coli",
        strain=background.name,
        assembly_set=cast(Any, assembly_set),
        assembly_accession=ASSEMBLY_SET_ACCESSIONS[assembly_set][0],
        background=background,
    )


@pytest.fixture
def mirrored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mg1655: EcoliK12MG1655Genome
) -> Path:
    """A tmp ``DATA_ROOT`` whose raw mirror holds the three synthetic tables."""
    frames = {
        r.GROWTH_TABLE: _growth(_SYNTHETIC_GROWTH),
        r.PHAGE_TABLE: _phage(_SYNTHETIC_PHAGE_GENES),
        r.TRANSDUCTION_TABLE: _transduction(_SYNTHETIC_PHAGE_GENES),
    }
    paths = {
        filename: _write(tmp_path, filename, frame)
        for filename, frame in frames.items()
    }
    monkeypatch.setattr(
        r,
        "TABLE_SHA256",
        {
            filename: hashlib.sha256(path.read_bytes()).hexdigest()
            for filename, path in paths.items()
        },
    )
    data_root = tmp_path / "data_root"
    r.deposit_raw_mirror(table_paths=paths, data_root=str(data_root))
    _deposit_synthetic_counts(data_root, monkeypatch)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr(
        r, "bacterial_genome", lambda host, strain, data_root=None: mg1655
    )
    monkeypatch.setattr(r, "assembly_reference", _pin)
    return data_root


#: A library guide no stored record names, so the count tables hold more rows than the
#: records, as the real 92,026-guide tables do.
_UNSTORED_SPACER = "GT" * 10


def _count_frame(spacers: list[str], filename: str) -> pd.DataFrame:
    """A count table over ``spacers`` with the pinned header; one zero per table."""
    columns = list(r.COUNT_TABLE_HEADERS[filename])
    frame = pd.DataFrame(
        [[10 * (i + 1) + j for j in range(len(columns))] for i in range(len(spacers))],
        index=spacers,
        columns=columns,
    )
    frame.iloc[0, 0] = 0
    return frame


def _synthetic_counts() -> dict[str, pd.DataFrame]:
    """Both synthetic count tables: every synthetic S1 / S4 spacer plus one unstored."""
    growth = [spacer for spacer, _, _ in _SYNTHETIC_GROWTH] + [_UNSTORED_SPACER]
    phage = list(_phage(_SYNTHETIC_PHAGE_GENES)["target"]) + [_UNSTORED_SPACER]
    return {
        r.ESSENTIALITY_COUNTS: _count_frame(growth, r.ESSENTIALITY_COUNTS),
        r.PHAGE_COUNTS: _count_frame(phage, r.PHAGE_COUNTS),
    }


def _deposit_synthetic_counts(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    frames: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Write the synthetic count tables into the raw mirror and record them, re-pinned."""
    from torchcell.literature.manifest import ArtifactRecord

    root = data_root / r.RAW_DIR_REL
    pins: dict[str, str] = {}
    records = []
    for filename, frame in (frames or _synthetic_counts()).items():
        path = root / r.table_rel(filename)
        frame.to_csv(path, sep="\t")
        pins[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            ArtifactRecord(
                path=r.table_rel(filename),
                role="raw_data",
                bytes=path.stat().st_size,
                sha256=pins[filename],
            )
        )
    manifest = r.load_manifest(str(data_root))
    (root / "manifest.json").write_text(
        manifest.model_copy(
            update={"files": [*manifest.files, *records]}
        ).model_dump_json()
    )
    monkeypatch.setattr(r, "COUNT_TABLE_SHA256", pins)


def test_the_loader_builds_the_synthetic_release_end_to_end(
    tmp_path: Path, mirrored: Path
) -> None:
    root = tmp_path / "dataset"
    dataset = r.CrispriScreenRousset2018Dataset(root=str(root))
    # 7 coding-strand in-gene S1 guides (#905) + 3 released guides x 3 phage columns
    # + 3 transduction records. "nope" is on no locus of the synthetic annotation and
    # is stored as released.
    assert len(dataset) == 19
    assert sorted(dataset.gene_set) == [
        "PRO2",
        "nope",
        "thrA",
        "thrA1",
        "thrL",
        "thrW",
        "yaaP",
    ]
    references = dataset.experiment_reference_index
    assert references is not None
    assert len(references) == 5
    census: dict[str, int] = {}
    for index in range(len(dataset)):
        screen = dataset[index]["experiment"]["phenotype"]["screen_id"]
        census[screen] = census.get(screen, 0) + 1
    assert census == {
        "growth_LC-E75": 7,
        "phage_lambda": 3,
        "phage_T4": 3,
        "phage_186cIts": 3,
        "lambda_transduction": 3,
    }
    first = dataset[0]["experiment"]
    assert first["genotype"]["perturbations"][0]["systematic_gene_name"] == "thrL"
    assert first["genotype"]["perturbations"][0]["crispr"]["guide_sequence"] == "A" * 20
    assert first["phenotype"]["screen_id"] == "growth_LC-E75"
    # The S1 row's released padj (1e-3 / (0 + 1)) and its DESeq2 correction.
    assert first["phenotype"]["environment_response_p_value_adjusted"] == 1e-3
    assert first["phenotype"]["p_value_adjustment_method"] == "benjamini_hochberg"
    assert (
        first["genotype"]["perturbations"][0]["identifier_kind"]
        == r.GROWTH_IDENTIFIER_KIND
    )
    assert (root / "raw" / r.GROWTH_TABLE).is_file()
    assert not (root / "raw" / "41467_2018_4209_MOESM8_ESM.csv").exists()
    assert (root / "preprocess" / "build_manifest.json").is_file()


def test_the_build_accounts_for_every_dropped_synthetic_record(
    tmp_path: Path, mirrored: Path
) -> None:
    import json as _json

    root = tmp_path / "dataset"
    r.CrispriScreenRousset2018Dataset(root=str(root))
    drops = _json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert (
        drops["source_records"],
        drops["kept_records"],
        drops["dropped_records"],
    ) == (21, 19, 2)
    assert {
        rule["rule"]: (rule["n_records"], rule["items"]) for rule in drops["rules"]
    } == {
        "guide_targets_no_gene": (1, []),
        "guide_targets_the_template_strand": (1, []),
    }
    # No rule names another dataset: the S1 rows are not cross-referenced (#905).
    assert all(rule["served_by"] is None for rule in drops["rules"])
    report = _json.loads(
        (root / "preprocess" / "identifier_reconciliation.json").read_text()
    )
    assert (report["released_symbols"], report["stored_symbols"]) == (7, 7)


def test_the_build_accounts_for_every_released_replicate_count(
    tmp_path: Path, mirrored: Path
) -> None:
    """#1178: the count tables are read, and every column is a screen's replicate pair."""
    import json as _json

    root = tmp_path / "dataset"
    dataset = r.CrispriScreenRousset2018Dataset(root=str(root))
    # the counts change no record: one per (guide, screen), as before
    assert len(dataset) == 19
    assert (root / "raw" / r.ESSENTIALITY_COUNTS).is_file()
    assert (root / "raw" / r.PHAGE_COUNTS).is_file()
    release = r.ReplicateRelease.model_validate_json(
        (root / "preprocess" / "replicates_not_stored.json").read_text()
    )
    assert release.reason == r.READ_COUNT_NOT_STORED
    assert {
        t.count_table: (t.guides, t.released_cells) for t in release.count_tables
    } == {r.ESSENTIALITY_COUNTS: (10, 60), r.PHAGE_COUNTS: (4, 72)}
    (phage,) = [t for t in release.count_tables if t.count_table == r.PHAGE_COUNTS]
    assert phage.screens_by_column["Before_Lambda_186_transduction_2"] == [
        "phage_lambda",
        "phage_186cIts",
        "lambda_transduction",
    ]
    assert phage.screens_by_column["Before_T4_3"] == ["phage_T4"]
    accounts = {s.screen_id: s for s in release.screens}
    assert {s: (a.stored_guides, a.count_cells) for s, a in accounts.items()} == {
        "growth_LC-E75": (7, 42),
        "phage_lambda": (3, 18),
        "phage_T4": (3, 18),
        "phage_186cIts": (3, 18),
        "lambda_transduction": (3, 18),
    }
    assert sum(accounts["growth_LC-E75"].zero_count_cells.values()) == 1
    assert accounts["phage_186cIts"].zero_count_cells["After_186_1"] == 1
    assert [p.replicate for p in accounts["phage_T4"].pairs] == ["1", "2", "3"]
    raw = _json.loads((root / "preprocess" / "replicates_not_stored.json").read_text())
    assert raw["dataset"] == dataset.name


def test_the_build_refuses_a_stored_guide_the_count_table_lacks(
    tmp_path: Path, mirrored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _synthetic_counts()
    frames[r.PHAGE_COUNTS] = frames[r.PHAGE_COUNTS].drop(index="B" * 20)
    root = mirrored / r.RAW_DIR_REL
    manifest = r.load_manifest(str(mirrored))
    (root / "manifest.json").write_text(
        manifest.model_copy(
            update={
                "files": [
                    f
                    for f in manifest.files
                    if f.path
                    not in {r.table_rel(name) for name in r.COUNT_TABLE_SHA256}
                ]
            }
        ).model_dump_json()
    )
    _deposit_synthetic_counts(mirrored, monkeypatch, frames)
    with pytest.raises(RuntimeError, match="stored guides are not rows of"):
        r.CrispriScreenRousset2018Dataset(root=str(tmp_path / "dataset"))
    assert not (tmp_path / "dataset" / "processed" / "lmdb").exists()


def test_screen_samples_name_every_count_column_and_pair_by_suffix() -> None:
    named: dict[str, set[str]] = {name: set() for name in r.COUNT_TABLE_HEADERS}
    for condition in r.CONDITIONS:
        samples = r.SCREEN_SAMPLES[condition.screen_id]
        for pair in samples.pairs:
            assert pair.after.endswith(f"_{pair.replicate}")
            assert pair.before.endswith(f"_{pair.replicate}")
            named[samples.count_table] |= {pair.after, pair.before}
    assert {k: sorted(v) for k, v in named.items()} == {
        k: sorted(v) for k, v in r.COUNT_TABLE_HEADERS.items()
    }
    assert r.SCREEN_SAMPLES["phage_T4"].pairs[0].before == "Before_T4_1"
    assert (
        r.SCREEN_SAMPLES["phage_186cIts"].pairs[2].before
        == "Before_Lambda_186_transduction_3"
    )


def test_replicate_release_refuses_a_column_no_screen_reads() -> None:
    frames = _synthetic_counts()
    frames[r.ESSENTIALITY_COUNTS]["Extra_1"] = 1
    stored = {
        c.screen_id: list(frames[r.SCREEN_SAMPLES[c.screen_id].count_table].index[:1])
        for c in r.CONDITIONS
    }
    with pytest.raises(RuntimeError, match="no screen's replicate pairs name"):
        r.replicate_release("d", stored, frames)
    frames = _synthetic_counts()
    frames[r.PHAGE_COUNTS] = frames[r.PHAGE_COUNTS].drop(columns="Before_T4_2")
    with pytest.raises(RuntimeError, match="lacks the columns"):
        r.replicate_release("d", stored, frames)


def test_read_count_table_keeps_the_release_and_refuses_a_departure(
    tmp_path: Path,
) -> None:
    frame = _synthetic_counts()[r.ESSENTIALITY_COUNTS]
    path = tmp_path / "counts.txt"
    frame.to_csv(path, sep="\t")
    read = r.read_count_table(path, r.ESSENTIALITY_COUNTS)
    assert list(read.columns) == list(r.COUNT_TABLE_HEADERS[r.ESSENTIALITY_COUNTS])
    assert read.loc["A" * 20, "After_1"] == 0
    for broken, message in (
        (frame.rename(columns={"After_1": "after_1"}), "header"),
        (frame.rename(index={"A" * 20: "A" * 19}), "20 nt"),
        (frame.rename(index={"C" * 20: "A" * 20}), "repeats a spacer"),
        (frame.astype(float).assign(After_2=0.5), "integer"),
        (frame.assign(After_2=-1), "negative"),
    ):
        broken.to_csv(path, sep="\t")
        with pytest.raises(ValueError, match=message):
            r.read_count_table(path, r.ESSENTIALITY_COUNTS)
    frame.astype(float).assign(After_2=float("nan")).to_csv(path, sep="\t")
    with pytest.raises(ValueError, match="empty count"):
        r.read_count_table(path, r.ESSENTIALITY_COUNTS)


def test_the_repository_files_are_fetched_from_the_pinned_commit() -> None:
    url = r.gitlab_raw_url(r.REPOSITORY_FILE[r.PHAGE_COUNTS].path_in_repo)
    assert url == (
        "https://gitlab.pasteur.fr/api/v4/projects/dbikard%2FdCas9_genome_wide_screen"
        "/repository/files/Generating%20count%20tables%20from%20fastq%20files%2F"
        "Counts_table_phages.txt/raw?ref=712b3b94c5457c75bcdb1dbc4ad7b9d270241bcc"
    )
    retrieval = r.repository_retrieval(r.REPOSITORY_FILE[r.PHAGE_COUNTS])
    assert retrieval.method is RetrievalMethod.direct_url
    assert retrieval.retriever == "torchcell.literature.retrieve.direct_url"
    assert retrieval.params == {"url": url}
    assert retrieval.sha256 == r.COUNT_TABLE_SHA256[r.PHAGE_COUNTS]
    assert set(r.raw_sha256()) == {*r.TABLE_SHA256, *r.COUNT_TABLE_SHA256}


def test_deposit_repository_files_is_idempotent_and_refuses_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torchcell.literature.retrieve as retrieve

    paths = _fake_tables(tmp_path, monkeypatch)
    data_root = str(tmp_path / "data_root")
    r.deposit_raw_mirror(table_paths=paths, data_root=data_root)
    payloads = {
        f.path_in_repo: f"{f.filename} bytes".encode() for f in r.REPOSITORY_FILES
    }
    monkeypatch.setattr(
        r,
        "REPOSITORY_FILES",
        tuple(
            f.model_copy(
                update={"sha256": hashlib.sha256(payloads[f.path_in_repo]).hexdigest()}
            )
            for f in r.REPOSITORY_FILES
        ),
    )
    by_url = {r.gitlab_raw_url(p): payload for p, payload in payloads.items()}
    fetched: list[str] = []

    def _fetch(url: str) -> bytes:
        fetched.append(url)
        return by_url[url]

    monkeypatch.setattr(retrieve, "direct_url", _fetch)
    root = r.deposit_repository_files(data_root)
    r.deposit_repository_files(data_root)
    assert len(fetched) == 4
    manifest = r.load_manifest(data_root)
    added = {f.path: f for f in manifest.files[len(r.TABLE_SHA256) :]}
    assert sorted(added) == sorted(f.mirror_rel for f in r.REPOSITORY_FILES)
    counts = added[f"data/{r.PHAGE_COUNTS}"]
    assert counts.role == "raw_data"
    assert counts.retrieval is not None
    assert counts.retrieval.method is RetrievalMethod.direct_url
    assert added[f"ref/{r.PHAGE_NOTEBOOK}"].role == "si_text"
    assert len(manifest.si_expected) == 5 + 3
    (root / f"data/{r.ESSENTIALITY_COUNTS}").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        r.deposit_repository_files(data_root)
    (root / f"data/{r.ESSENTIALITY_COUNTS}").unlink()
    by_url[r.gitlab_raw_url(r.REPOSITORY_FILE[r.ESSENTIALITY_COUNTS].path_in_repo)] = (
        b"upstream drift"
    )
    with pytest.raises(RuntimeError, match="the pin is"):
        r.deposit_repository_files(data_root)


def test_the_two_screens_of_one_guide_are_two_records_on_one_strain(
    tmp_path: Path, mirrored: Path
) -> None:
    root = tmp_path / "dataset"
    dataset = r.CrispriScreenRousset2018Dataset(root=str(root))
    by_screen = {
        dataset[index]["experiment"]["phenotype"]["screen_id"]: dataset[index]
        for index in range(len(dataset))
        if dataset[index]["experiment"]["genotype"]["perturbations"][0]["crispr"][
            "guide_sequence"
        ]
        == "A" * 20
    }
    challenge = by_screen["phage_lambda"]["experiment"]
    transduction = by_screen["lambda_transduction"]["experiment"]
    assert challenge["genotype"] == transduction["genotype"]
    assert challenge["environment"] == transduction["environment"]
    assert challenge["phenotype"]["environment_response"] == 0.4
    assert transduction["phenotype"]["environment_response"] == -0.3
    assert set(by_screen) == {c.screen_id for c in r.CONDITIONS}
    for screen, record in by_screen.items():
        strain = "LC-E75" if screen == "growth_LC-E75" else "FR-E01"
        assert record["reference"]["genome_reference"]["strain"] == strain


def test_download_refuses_a_mirror_whose_file_is_gone(
    tmp_path: Path, mirrored: Path
) -> None:
    (mirrored / r.RAW_DIR_REL / r.table_rel(r.PHAGE_TABLE)).unlink()
    dataset = r.CrispriScreenRousset2018Dataset.__new__(
        r.CrispriScreenRousset2018Dataset
    )
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        dataset.download()


def test_the_build_refuses_a_raw_file_whose_sha256_drifted(
    tmp_path: Path, mirrored: Path
) -> None:
    (mirrored / r.RAW_DIR_REL / r.table_rel(r.TRANSDUCTION_TABLE)).write_text("drift\n")
    with pytest.raises(RuntimeError, match="sha256"):
        r.CrispriScreenRousset2018Dataset(root=str(tmp_path / "dataset"))


def test_preprocess_raw_is_a_passthrough_and_create_experiment_is_refused() -> None:
    dataset = r.CrispriScreenRousset2018Dataset.__new__(
        r.CrispriScreenRousset2018Dataset
    )
    sentinel = object()
    assert dataset.preprocess_raw(sentinel) is sentinel
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()


def test_verify_build_passes_on_the_synthetic_release(
    tmp_path: Path, mirrored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    r.CrispriScreenRousset2018Dataset(root=str(root))
    monkeypatch.setattr(
        r,
        "EXPECTED_SCREEN_CENSUS",
        {
            "growth_LC-E75": 7,
            "phage_lambda": 3,
            "phage_T4": 3,
            "phage_186cIts": 3,
            "lambda_transduction": 3,
        },
    )
    report = r.verify_build(str(root), data_root=str(mirrored), expected_count=19)
    verdicts = {result.name: result.passed for result in report.results}
    assert verdicts["structural"] and verdicts["count"]
    assert verdicts["pair_uniqueness"] is True
    assert verdicts["measurement_type_consistent"] is True
    assert verdicts["reference_zero"] is True
    assert verdicts["environment_perturbed"] is True
    assert verdicts["screen_census"] is True
    assert verdicts["released_test_where_released"] is True
    # "nope" is on no locus of the synthetic annotation: the genome-reference rows
    # report it and the verdict stands (#1069)
    assert verdicts["current_genome_genes"] is False
    assert all(
        result.finding_class.value == "report"
        for result in report.results
        if not result.passed
    )
    assert report.passed
    assert (root / "preprocess" / "verification_report.json").is_file()


def test_verify_build_refuses_a_genome_of_another_strain(
    tmp_path: Path, mirrored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    r.CrispriScreenRousset2018Dataset(root=str(root))
    monkeypatch.setattr(
        r, "bacterial_genome", lambda host, strain, data_root=None: object()
    )
    with pytest.raises(TypeError, match="expected the MG1655 genome"):
        r.verify_build(str(root), data_root=str(mirrored))


def test_main_builds_the_dataset_under_data_root_and_verifies_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json as _json

    data_root = tmp_path / "dr"
    preprocess = data_root / "data/torchcell/ecoli_crispri_rousset2018/preprocess"
    preprocess.mkdir(parents=True)
    (preprocess / "dropped_records.json").write_text(
        _json.dumps({"rules": [{"rule": "guide_targets_no_gene", "n_records": 1}]})
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    built: list[str] = []

    class _Dataset:
        def __init__(self, root: str) -> None:
            built.append(root)

        def __len__(self) -> int:
            return 8

        def __getitem__(self, index: int) -> str:
            return f"record {index}"

    monkeypatch.setattr(r, "CrispriScreenRousset2018Dataset", _Dataset)
    monkeypatch.setattr(
        r,
        "verify_build",
        lambda root, data_root=None: SimpleNamespace(summary=lambda: "PASS"),
    )
    r.main()
    out = capsys.readouterr().out
    assert built == [str(data_root / "data/torchcell/ecoli_crispri_rousset2018")]
    assert "len = 8" in out
    assert "record 0" in out
    assert "guide_targets_no_gene" in out
    assert "PASS" in out


# --------------------------------------------------------------------------- #
# Data tests: the two mirrors and the release
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    return os.environ["DATA_ROOT"]


@pytest.mark.data
def test_every_sourced_value_is_backed_by_its_verbatim_quote() -> None:
    root = osp.join(_data_root(), "torchcell-library")
    values = [v for v in vars(r).values() if isinstance(v, SourcedValue)]
    deseq2 = [v for v in values if v.provenance.sha256 == r.DESEQ2_REFERENCE_SHA256]
    assert deseq2 == [r.PADJ_CORRECTION]
    notebooks = {id(v) for v in r.REPOSITORY_SOURCED_VALUES}
    paper = [v for v in values if v is not r.PADJ_CORRECTION and id(v) not in notebooks]
    assert len(paper) == len(r.SOURCED_VALUES) == 27
    assert {v.provenance.citation_key for v in paper} == {r.CITATION_KEY}
    for value in paper:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"


@pytest.mark.data
def test_the_padj_correction_is_verbatim_in_the_deposited_deseq2_reference() -> None:
    """#905: the BH sentence, re-read from the hash-pinned raw-mirror bytes."""
    path = Path(_data_root()) / r.RAW_DIR_REL / r.DESEQ2_REFERENCE_REL
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == r.DESEQ2_REFERENCE_SHA256
    assert r.PADJ_CORRECTION.quote in payload.decode()
    manifest = r.load_manifest(_data_root())
    (record,) = [f for f in manifest.files if f.path == r.DESEQ2_REFERENCE_REL]
    assert record.retrieval is not None
    assert record.retrieval.method is RetrievalMethod.pmc_cloud
    assert record.retrieval.params == {"key": r.DESEQ2_REFERENCE_KEY}


@pytest.mark.data
def test_every_notebook_value_is_verbatim_in_the_deposited_notebook() -> None:
    """#1178: the replicate pairing quotes, re-read from the raw-mirror notebook bytes."""
    root = osp.join(_data_root(), "torchcell-raw")
    assert len(r.REPOSITORY_SOURCED_VALUES) == 4
    for value in r.REPOSITORY_SOURCED_VALUES:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"


@pytest.mark.data
def test_the_repository_files_are_deposited_with_their_retrieval() -> None:
    mirror = Path(_data_root()) / r.RAW_DIR_REL
    manifest = r.load_manifest(_data_root())
    by_path = {f.path: f for f in manifest.files}
    for record in r.REPOSITORY_FILES:
        payload = (mirror / record.mirror_rel).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record.sha256
        entry = by_path[record.mirror_rel]
        assert entry.sha256 == record.sha256
        assert entry.retrieval is not None
        assert entry.retrieval.method is RetrievalMethod.direct_url
        assert entry.retrieval.params == {"url": r.gitlab_raw_url(record.path_in_repo)}
        assert entry.retrieval.retrieved_at == r.GITLAB_RETRIEVED_AT
    assert manifest.provenance_complete


@pytest.mark.data
def test_the_count_tables_hold_every_stored_guide_in_three_replicate_pairs() -> None:
    """#1178: 92,026 guides per table; every stored guide is a row of its screen's table."""
    mirror = osp.join(_data_root(), r.RAW_DIR_REL)
    counts = {
        name: r.read_count_table(osp.join(mirror, r.table_rel(name)), name)
        for name in r.COUNT_TABLE_SHA256
    }
    assert counts[r.ESSENTIALITY_COUNTS].shape == (92026, 6)
    assert counts[r.PHAGE_COUNTS].shape == (92026, 18)
    assert list(counts[r.ESSENTIALITY_COUNTS].index) == list(
        counts[r.PHAGE_COUNTS].index
    )
    tables = {
        filename: r.read_table(osp.join(mirror, r.table_rel(filename)), filename)
        for filename in r.TABLE_SHA256
    }
    release = r.replicate_release(
        "ecoli_crispri_rousset2018",
        {
            c.screen_id: [
                str(s)
                for s in tables[c.table].loc[
                    r.coding_strand(tables[c.table], c.table), "target"
                ]
            ]
            for c in r.CONDITIONS
        },
        counts,
    )
    assert {s.screen_id: s.stored_guides for s in release.screens} == (
        r.EXPECTED_SCREEN_CENSUS
    )
    assert sum(s.count_cells for s in release.screens) == 6 * r.EXPECTED_RECORDS
    zeros = {s.screen_id: sum(s.zero_count_cells.values()) for s in release.screens}
    assert zeros == {
        "growth_LC-E75": 1428,
        "phage_lambda": 593,
        "phage_T4": 1,
        "phage_186cIts": 613,
        "lambda_transduction": 2609,
    }
    # The third pre-infection sample of lambda / 186 / transduction is the shallow one
    # (695,288 reads against 4.4-5.0 million for the other two).
    before = counts[r.PHAGE_COUNTS].filter(like="Before_Lambda_186_transduction").sum()
    assert before.to_dict() == {
        "Before_Lambda_186_transduction_1": 5038582,
        "Before_Lambda_186_transduction_2": 4373190,
        "Before_Lambda_186_transduction_3": 695288,
    }


@pytest.mark.data
def test_the_count_tables_are_the_counts_behind_the_released_log2fc() -> None:
    """The mean of the three paired log2 ratios tracks each released DESeq2 log2FC.

    Evidence, not a build step: on the control-guide normalization the authors state
    (``GROWTH_REPLICATE_PAIRING``), log2(After_i / Before_i) averaged over the three
    pairs agrees with S1 Table's log2FC to a median 0.0015 on the guides with no zero
    count. DESeq2's negative-binomial fit is not the plain mean, so the agreement is
    close rather than exact.
    """
    import numpy as np

    mirror = osp.join(_data_root(), r.RAW_DIR_REL)
    control = "TGAGACCAGTCTAGGTCTCG"
    tables = {
        filename: r.read_table(osp.join(mirror, r.table_rel(filename)), filename)
        for filename in r.TABLE_SHA256
    }
    counts = {
        name: r.read_count_table(osp.join(mirror, r.table_rel(name)), name)
        for name in r.COUNT_TABLE_SHA256
    }
    medians: dict[str, float] = {}
    finite: dict[str, int] = {}
    for condition in r.CONDITIONS:
        samples = r.SCREEN_SAMPLES[condition.screen_id]
        table = counts[samples.count_table]
        normalized = table / table.loc[control]
        released = tables[condition.table].set_index("target")[condition.column]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.stack(
                [
                    np.log2(
                        normalized.loc[released.index, p.after].to_numpy(dtype=float)
                        / normalized.loc[released.index, p.before].to_numpy(dtype=float)
                    )
                    for p in samples.pairs
                ],
                axis=1,
            )
        keep = np.isfinite(ratios).all(axis=1)
        finite[condition.screen_id] = int(keep.sum())
        difference = np.abs(ratios[keep].mean(axis=1) - released.to_numpy()[keep])
        medians[condition.screen_id] = round(float(np.median(difference)), 4)
    assert finite == {
        "growth_LC-E75": 58317,
        "phage_lambda": 16627,
        "phage_T4": 17219,
        "phage_186cIts": 16610,
        "lambda_transduction": 15693,
    }
    assert medians == {
        "growth_LC-E75": 0.0015,
        "phage_lambda": 0.0807,
        "phage_T4": 0.0122,
        "phage_186cIts": 0.0814,
        "lambda_transduction": 0.0822,
    }


@pytest.mark.data
def test_the_release_has_the_measured_shape_and_sign_distribution() -> None:
    mirror = osp.join(_data_root(), r.RAW_DIR_REL)
    tables = {
        filename: r.read_table(osp.join(mirror, r.table_rel(filename)), filename)
        for filename in r.TABLE_SHA256
    }
    assert tables[r.GROWTH_TABLE].shape == (59246, 12)
    assert tables[r.PHAGE_TABLE].shape == (17220, 11)
    assert tables[r.TRANSDUCTION_TABLE].shape == (17220, 9)
    assert set(tables[r.PHAGE_TABLE]["target"]) == set(
        tables[r.TRANSDUCTION_TABLE]["target"]
    )
    coding = {
        filename: r.coding_strand(frame, filename) for filename, frame in tables.items()
    }
    growth = tables[r.GROWTH_TABLE]
    assert int(coding[r.GROWTH_TABLE].sum()) == 23372
    assert int(growth["gene"].isna().sum()) == 5063
    assert int((~coding[r.GROWTH_TABLE] & growth["gene"].notna()).sum()) == 30811
    kept = growth.loc[coding[r.GROWTH_TABLE], "log2FC"]
    assert round(float((kept < 0).mean()), 4) == 0.9249
    assert round(float(kept.min()), 2) == -11.95


@pytest.mark.data
def test_s1_padj_is_numeric_on_every_row_with_no_unadjusted_p_value() -> None:
    mirror = osp.join(_data_root(), r.RAW_DIR_REL)
    growth = r.read_table(osp.join(mirror, r.table_rel(r.GROWTH_TABLE)), r.GROWTH_TABLE)
    assert int(growth["padj"].notna().sum()) == len(growth) == 59246
    assert 0.0 <= float(growth["padj"].min()) <= float(growth["padj"].max()) < 1.0
    assert not {"pvalue", "p_value", "pval"} & set(growth.columns)


@pytest.mark.data
def test_the_release_resolves_to_the_measured_identifier_counts() -> None:
    from torchcell.datasets.bacteria_common import bacterial_genome

    mirror = osp.join(_data_root(), r.RAW_DIR_REL)
    tables = {
        filename: r.read_table(osp.join(mirror, r.table_rel(filename)), filename)
        for filename in r.TABLE_SHA256
    }
    record_tables = {r.PHAGE_TABLE, r.TRANSDUCTION_TABLE}
    symbols = sorted(
        {
            str(name)
            for filename, frame in tables.items()
            if filename in record_tables
            for name in frame.loc[r.coding_strand(frame, filename), "gene"].dropna()
        }
    )
    assert len(symbols) == 3708
    genome = bacterial_genome("ecoli", "MG1655")
    assert isinstance(genome, EcoliK12MG1655Genome)
    assert len(set(symbols)) == 3708
    # every released symbol is a stored identifier; what the pinned annotation makes of
    # it is a genome-reference finding, measured here read-only
    statuses: dict[str, int] = {}
    for symbol in symbols:
        statuses[genome.resolve_gene_name(symbol).status.value] = (
            statuses.get(genome.resolve_gene_name(symbol).status.value, 0) + 1
        )
    assert sum(statuses.values()) == 3708
    assert statuses["retired"] == 28
    assert statuses["ambiguous"] == 1


@pytest.mark.data
def test_the_built_store_matches_the_retention_arithmetic() -> None:
    import json

    preprocess = osp.join(
        _data_root(), "data/torchcell/ecoli_crispri_rousset2018", "preprocess"
    )
    with open(osp.join(preprocess, "dropped_records.json")) as handle:
        log = json.load(handle)
    assert log["source_records"] == 128126
    assert log["kept_records"] == r.EXPECTED_RECORDS == 92252
    assert log["dropped_records"] == 35874
    assert {rule["rule"]: rule["n_records"] for rule in log["rules"]} == {
        "guide_targets_no_gene": 5063,
        "guide_targets_the_template_strand": 30811,
    }
    assert all(rule["served_by"] is None for rule in log["rules"])
    with open(osp.join(preprocess, "gene_set.json")) as handle:
        assert len(json.load(handle)) == r.EXPECTED_GENES


@pytest.mark.data
def test_the_stored_records_carry_the_measured_sign_distribution() -> None:
    """The per-screen sign census of the built store, the note's record-type evidence.

    ``EnvironmentResponsePhenotype`` is the record type because the value is signed and
    routinely negative; these are the fractions that claim rests on, measured over the
    stored records rather than over the released rows. Since #1069 every released
    symbol is stored, so the phage-derived counts equal the released S4 / S6 negative
    counts (re-measured 2026-11-24; the pins here predated #1069 and read 4,740, 14,359,
    7,670 and 15,253 over the 68,436 records stored then).
    """
    from torchcell.verification.runners import stream_records

    root = osp.join(_data_root(), "data/torchcell/ecoli_crispri_rousset2018")
    negative: Counter[str] = Counter()
    total: Counter[str] = Counter()
    minimum: dict[str, float] = {}
    for record in stream_records(root):
        phenotype = record["experiment"]["phenotype"]
        screen = str(phenotype["screen_id"])
        value = float(phenotype["environment_response"])
        total[screen] += 1
        negative[screen] += value < 0
        minimum[screen] = min(minimum.get(screen, value), value)
    assert dict(total) == r.EXPECTED_SCREEN_CENSUS
    growth_negative = negative.pop("growth_LC-E75")
    growth_total = total.pop("growth_LC-E75")
    minimum.pop("growth_LC-E75")
    assert growth_total == 23372
    assert round(growth_negative / growth_total, 4) == 0.9249
    assert dict(negative) == {
        "phage_lambda": 4775,
        "phage_T4": 14456,
        "phage_186cIts": 7717,
        "lambda_transduction": 15350,
    }
    assert {screen: round(value, 4) for screen, value in minimum.items()} == {
        "phage_lambda": -2.5409,
        "phage_T4": -2.5929,
        "phage_186cIts": -3.3441,
        "lambda_transduction": -10.9705,
    }
    assert {screen: round(negative[screen] / total[screen], 4) for screen in total} == {
        "phage_lambda": 0.2773,
        "phage_T4": 0.8395,
        "phage_186cIts": 0.4481,
        "lambda_transduction": 0.8914,
    }
