# tests/torchcell/datasets/ecoli/test_rousset2018.py
# [[tests.torchcell.datasets.ecoli.test_rousset2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_rousset2018.py
"""Rousset 2018 loader: the five screens, table reading, symbols, records, mirror.

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
and pin the measured retention arithmetic and identifier histogram on the deposited
MG1655 set.
"""

from __future__ import annotations

import hashlib
import os
import os.path as osp
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

import torchcell.datasets.ecoli.rousset2018 as r
from tests.torchcell.datasets._genome_injection_fakes import (
    FakeMG1655Genome,
    install_bacterial_fakes,
)
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
    AssemblyReferenceGenome,
    BacterialCrisprInterferencePerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    ComponentDefinition,
    ConcentrationUnit,
    MeasurementType,
    MediaComponentRole,
    PhagePerturbation,
    SampleUnit,
    SmallMoleculePerturbation,
)
from torchcell.datasets.bacteria_common import (
    BacterialGenomeInjector,
    LocusTagResolutionError,
)
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.literature.manifest import Manifest, RetrievalMethod
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
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
# The five screens
# --------------------------------------------------------------------------- #
def test_the_five_screens_name_their_table_column_strain_and_phage() -> None:
    assert [
        (c.screen_id, c.table, c.column, c.strain, c.phage) for c in r.CONDITIONS
    ] == [
        ("growth_17_generations", r.GROWTH_TABLE, "log2FC", "LC-E75", None),
        ("phage_lambda", r.PHAGE_TABLE, "log2FC_lambda", "FR-E01", "lambda"),
        ("phage_T4", r.PHAGE_TABLE, "log2FC_T4", "FR-E01", "T4"),
        ("phage_186cIts", r.PHAGE_TABLE, "log2FC_186", "FR-E01", "186cIts"),
        ("lambda_transduction", r.TRANSDUCTION_TABLE, "log2FC", "FR-E01", "lambda"),
    ]
    assert len({c.screen_id for c in r.CONDITIONS}) == 5
    assert r.TABLE_COLUMNS == {
        r.GROWTH_TABLE: ("log2FC",),
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
# Media, environments, phenotypes, genotypes
# --------------------------------------------------------------------------- #
def test_both_media_derive_from_lb_and_assert_no_lb_amounts() -> None:
    for medium in (r.ROUSSET2018_LB, r.ROUSSET2018_LB_MALTOSE_CACL2):
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
    assert [c.definition for c in r.ROUSSET2018_LB.components[:2]] == [
        ComponentDefinition.intrinsically_undefined
    ] * 2


def test_the_phage_medium_adds_maltose_and_calcium_chloride_at_their_stated_doses() -> (
    None
):
    extra = r.ROUSSET2018_LB_MALTOSE_CACL2.components[3:]
    assert [
        (
            c.compound.name,
            c.role,
            c.concentration.value if c.concentration else None,
            c.concentration.unit if c.concentration else None,
        )
        for c in extra
    ] == [
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


def test_the_growth_environment_doses_atc_at_one_nanomolar_and_gaps_temperature() -> (
    None
):
    environment = r.environment(r.CONDITIONS[0])
    assert environment.media is r.ROUSSET2018_LB
    assert environment.temperature is None
    assert environment.duration_generations == 17.0
    assert environment.duration_hours is None
    assert environment.aerobicity == "aerobic"
    (atc,) = environment.perturbations
    assert isinstance(atc, SmallMoleculePerturbation)
    assert atc.compound.name == "anhydrotetracycline"
    assert atc.compound.inchikey == "KTTKGQINVKPHLY-DOCRCCHOSA-N"
    assert (atc.concentration.value, atc.concentration.unit) == (
        1.0,
        ConcentrationUnit.nanomolar,
    )
    (gap,) = environment.provenance_gaps
    assert gap.field == "temperature"
    assert gap.reason is ProvenanceGapReason.not_reported_by_primary
    assert gap.looked_in is not None
    assert gap.looked_in.citation_key == r.CUI2018_KEY


def test_a_phage_environment_doses_atc_at_one_micromolar_plus_its_phage_at_moi_one() -> (
    None
):
    environment = r.environment(r.CONDITIONS[2])
    assert environment.media is r.ROUSSET2018_LB_MALTOSE_CACL2
    assert environment.temperature is not None
    assert environment.temperature.value == 37.0
    assert environment.duration_hours == 2.0
    assert environment.duration_generations is None
    assert environment.provenance_gaps == []
    atc, phage = environment.perturbations
    assert isinstance(atc, SmallMoleculePerturbation)
    assert (atc.concentration.value, atc.concentration.unit) == (
        1.0,
        ConcentrationUnit.micromolar,
    )
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


def test_every_screen_carries_an_environment_perturbation() -> None:
    assert all(r.environment(c).perturbations for c in r.CONDITIONS)


def test_a_phenotype_is_a_signed_log2_ratio_over_three_biological_replicates() -> None:
    phenotype = r.phenotype(-9.0, r.CONDITIONS[0])
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


def test_a_genotype_is_crispri_of_its_locus_by_the_reported_spacer() -> None:
    stored = r.StoredGuide(
        spacer="ACCACGCACTCTGACCATCT",
        reported_symbol="sucA",
        systematic_gene_name="b0726",
        perturbed_gene_name="sucA",
    )
    (perturbation,) = r.genotype(stored).perturbations
    assert isinstance(perturbation, BacterialCrisprInterferencePerturbation)
    assert perturbation.perturbation_type == "bacterial_crispr_interference"
    assert perturbation.systematic_gene_name == "b0726"
    assert perturbation.perturbed_gene_name == "sucA"
    assert perturbation.gene_namespace == "ecoli_k12_mg1655_bnumber"
    assert perturbation.expression_direction == "decreased"
    assert perturbation.state == "present"
    assert perturbation.crispr.effector == "dCas9"
    assert perturbation.crispr.guide_sequence == "ACCACGCACTCTGACCATCT"
    assert perturbation.crispr.n_guides == 1
    assert perturbation.identifier_mapping is not None
    assert perturbation.identifier_mapping.source_identifier == "sucA"
    assert perturbation.identifier_mapping.route == "gene_symbol"


def test_two_guides_on_one_gene_are_two_strains_and_one_gene() -> None:
    guides = [
        r.genotype(
            r.StoredGuide(
                spacer=spacer,
                reported_symbol="sucA",
                systematic_gene_name="b0726",
                perturbed_gene_name="sucA",
            )
        ).perturbations[0]
        for spacer in ("A" * 20, "C" * 20)
    ]
    assert all(isinstance(g, BacterialCrisprInterferencePerturbation) for g in guides)
    first, second = guides
    assert isinstance(first, BacterialCrisprInterferencePerturbation)
    assert isinstance(second, BacterialCrisprInterferencePerturbation)
    assert first.systematic_gene_name == second.systematic_gene_name
    assert first.crispr.guide_sequence != second.crispr.guide_sequence


def test_the_two_host_strains_are_backgrounds_on_the_one_mg1655_assembly() -> None:
    assert set(r.BACKGROUNDS) == {"LC-E75", "FR-E01"}
    for label, background in r.BACKGROUNDS.items():
        assert background.name == label
        assert background.reference_strain == "MG1655"
        assert background.assembly_set == "ecoli_K12_MG1655_ASM584v2"
        assert background.parents == ["MG1655"]
        assert background.alleles == []
        assert background.provenance
        assert background.gapped_fields() == set()
    assert "186 attB" in str(r.LC_E75_BACKGROUND.construction)
    assert "HK022" in str(r.FR_E01_BACKGROUND.construction)


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


def test_resolve_symbols_stores_b_numbers_and_names_every_unstorable_symbol(
    mg1655: EcoliK12MG1655Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 5 of the 7 synthetic symbols resolve (nope is retired, PRO2 ambiguous): 0.714,
    # under the release threshold of 0.95, which the next test exercises.
    monkeypatch.setattr(r, "MIN_RESOLVED_FRACTION", 0.7)
    resolution = r.resolve_symbols(
        ["thrL", "thrA", "thrA1", "thrW", "yaaP", "PRO2", "nope"],
        mg1655,
        label="synthetic",
    )
    assert resolution.stored == {"thrL": "b0001", "thrW": "b0003", "yaaP": "b0004"}
    assert resolution.retired == ("nope",)
    assert resolution.collided == ("thrA", "thrA1")
    assert resolution.ambiguous == ("PRO2",)
    assert resolution.unstorable == frozenset({"nope", "thrA", "thrA1", "PRO2"})
    assert resolution.report.ambiguous_kept == {"PRO2": ("b0005", "b0006")}


def test_resolve_symbols_stops_below_the_resolution_threshold(
    mg1655: EcoliK12MG1655Genome,
) -> None:
    with pytest.raises(LocusTagResolutionError, match=r"1 of 3 names \(0\.333\)"):
        r.resolve_symbols(["thrL", "nope", "PRO2"], mg1655, label="threshold")


def test_canonical_symbol_reads_the_annotation_not_the_release(
    mg1655: EcoliK12MG1655Genome,
) -> None:
    assert r.canonical_symbol(mg1655, "b0002") == "thrA"
    assert r.canonical_symbol(mg1655, "b0004") == "yaaP"


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


def test_canonical_symbol_falls_back_to_the_tag_when_the_symbol_resolves_elsewhere() -> (
    None
):
    genome: Any = _SymbolGenome()
    assert r.canonical_symbol(genome, "b0010") == "b0010"
    assert r.canonical_symbol(genome, "b0011") == "b0011"
    assert r.canonical_symbol(genome, "b0012") == "dup"


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
    assert set(kwargs) == {"ecoli_genome"}
    assert isinstance(kwargs["ecoli_genome"], FakeMG1655Genome)
    assert [name for name, _ in log] == ["FakeMG1655Genome"]
    dataset = r.CrispriScreenRousset2018Dataset.__new__(
        r.CrispriScreenRousset2018Dataset
    )
    assert dataset.experiment_class is BacterialEnvironmentResponseExperiment
    assert dataset.reference_class is (BacterialEnvironmentResponseExperimentReference)
    assert dataset.raw_file_names == [
        r.GROWTH_TABLE,
        r.PHAGE_TABLE,
        r.TRANSDUCTION_TABLE,
    ]


def test_the_loader_refuses_a_genome_of_another_strain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = r.CrispriScreenRousset2018Dataset.__new__(
        r.CrispriScreenRousset2018Dataset
    )
    dataset.ecoli_genome = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="needs the MG1655 genome"):
        dataset._genome()


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
        "growth_17_generations": 23209,
        "phage_lambda": 17100,
        "phage_T4": 17100,
        "phage_186cIts": 17100,
        "lambda_transduction": 17100,
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
) -> AssemblyReferenceGenome:
    """``assembly_reference`` without the deposited assembly report."""
    assembly_set = BACTERIAL_ASSEMBLY_SETS[strain]
    return AssemblyReferenceGenome(
        species="Escherichia coli",
        strain=strain if background is None else background.name,
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
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr(r, "MIN_RESOLVED_FRACTION", 0.5)
    monkeypatch.setattr(
        r, "bacterial_genome", lambda host, strain, data_root=None: mg1655
    )
    monkeypatch.setattr(r, "assembly_reference", _pin)
    return data_root


def test_the_loader_builds_the_synthetic_release_end_to_end(
    tmp_path: Path, mirrored: Path
) -> None:
    root = tmp_path / "dataset"
    dataset = r.CrispriScreenRousset2018Dataset(root=str(root))
    # 3 growth records + 2 guides x 3 phage columns + 2 transduction records
    assert len(dataset) == 11
    assert sorted(dataset.gene_set) == ["b0001", "b0003", "b0004"]
    references = dataset.experiment_reference_index
    assert references is not None
    assert len(references) == 5
    census: dict[str, int] = {}
    for index in range(len(dataset)):
        screen = dataset[index]["experiment"]["phenotype"]["screen_id"]
        census[screen] = census.get(screen, 0) + 1
    assert census == {
        "growth_17_generations": 3,
        "phage_lambda": 2,
        "phage_T4": 2,
        "phage_186cIts": 2,
        "lambda_transduction": 2,
    }
    first = dataset[0]["experiment"]
    assert first["genotype"]["perturbations"][0]["systematic_gene_name"] == "b0001"
    assert first["genotype"]["perturbations"][0]["crispr"]["guide_sequence"] == "A" * 20
    assert first["phenotype"]["environment_response"] == -1.5
    assert first["phenotype"]["screen_id"] == "growth_17_generations"
    assert (root / "raw" / r.GROWTH_TABLE).is_file()
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
    ) == (21, 11, 10)
    assert {
        rule["rule"]: (rule["n_records"], rule["items"]) for rule in drops["rules"]
    } == {
        "guide_targets_no_gene": (1, []),
        "guide_targets_the_template_strand": (1, []),
        "gene_symbol_is_not_in_the_mg1655_annotation": (5, ["nope"]),
        "gene_symbol_collides_with_another_symbol_on_one_mg1655_locus": (
            2,
            ["thrA", "thrA1"],
        ),
        "gene_symbol_is_ambiguous_in_mg1655": (1, ["PRO2"]),
    }
    report = _json.loads(
        (root / "preprocess" / "identifier_reconciliation.json").read_text()
    )
    assert (report["released_symbols"], report["stored_symbols"]) == (7, 3)
    assert report["stored_b_numbers"] == 3
    assert report["ambiguous"] == {"PRO2": ["b0005", "b0006"]}
    assert report["retired"] == ["nope"]
    assert report["collided"] == ["thrA", "thrA1"]


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
    growth = by_screen["growth_17_generations"]
    assert growth["reference"]["genome_reference"]["strain"] == "LC-E75"
    assert by_screen["phage_T4"]["reference"]["genome_reference"]["strain"] == "FR-E01"


def test_a_direct_run_opens_the_mg1655_genome_itself(
    tmp_path: Path, mirrored: Path, mg1655: EcoliK12MG1655Genome
) -> None:
    dataset = r.CrispriScreenRousset2018Dataset.__new__(
        r.CrispriScreenRousset2018Dataset
    )
    dataset.ecoli_genome = None
    assert dataset._genome() is mg1655
    assert dataset.ecoli_genome is mg1655


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
            "growth_17_generations": 3,
            "phage_lambda": 2,
            "phage_T4": 2,
            "phage_186cIts": 2,
            "lambda_transduction": 2,
        },
    )
    report = r.verify_build(str(root), data_root=str(mirrored), expected_count=11)
    verdicts = {result.name: result.passed for result in report.results}
    assert verdicts["structural"] and verdicts["count"]
    assert verdicts["pair_uniqueness"] is True
    assert verdicts["measurement_type_consistent"] is True
    assert verdicts["reference_zero"] is True
    assert verdicts["environment_perturbed"] is True
    assert verdicts["screen_census"] is True
    assert verdicts["current_genome_genes"] is True
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
            return 11

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
    assert "len = 11" in out
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
    assert len(values) == len(r.SOURCED_VALUES) == 28
    assert {v.provenance.citation_key for v in values} == {
        r.CITATION_KEY,
        r.CUI2018_KEY,
    }
    for value in values:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"


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
def test_the_release_resolves_to_the_measured_identifier_counts() -> None:
    from torchcell.datasets.bacteria_common import bacterial_genome

    mirror = osp.join(_data_root(), r.RAW_DIR_REL)
    tables = {
        filename: r.read_table(osp.join(mirror, r.table_rel(filename)), filename)
        for filename in r.TABLE_SHA256
    }
    symbols = sorted(
        {
            str(name)
            for filename, frame in tables.items()
            for name in frame.loc[r.coding_strand(frame, filename), "gene"].dropna()
        }
    )
    assert len(symbols) == 3944
    genome = bacterial_genome("ecoli", "MG1655")
    assert isinstance(genome, EcoliK12MG1655Genome)
    resolution = r.resolve_symbols(symbols, genome, label="release")
    assert len(resolution.stored) == r.EXPECTED_GENES == 3896
    assert len(set(resolution.stored.values())) == r.EXPECTED_GENES
    assert (
        len(resolution.retired),
        len(resolution.collided),
        resolution.ambiguous,
    ) == (31, 16, ("rffT",))
    assert {
        status.value: n for status, n in resolution.report.status_histogram.items()
    } == {
        "current": 0,
        "renamed": 3853,
        "non_gene_feature": 59,
        "retired": 31,
        "ambiguous": 1,
    }
    assert resolution.report.layer_histogram == {
        "locus tag": 0,
        "old locus tag": 0,
        "RefSeq locus tag": 0,
        "gene symbol": 3734,
        "gene synonym": 179,
        "not found": 31,
    }


@pytest.mark.data
def test_the_built_store_matches_the_retention_arithmetic() -> None:
    import json

    preprocess = osp.join(
        _data_root(), "data/torchcell/ecoli_crispri_rousset2018", "preprocess"
    )
    with open(osp.join(preprocess, "dropped_records.json")) as handle:
        log = json.load(handle)
    assert log["source_records"] == 128126
    assert log["kept_records"] == r.EXPECTED_RECORDS == 91609
    assert log["dropped_records"] == 36517
    assert {rule["rule"]: rule["n_records"] for rule in log["rules"]} == {
        "guide_targets_no_gene": 5063,
        "guide_targets_the_template_strand": 30811,
        "gene_symbol_is_not_in_the_mg1655_annotation": 307,
        "gene_symbol_collides_with_another_symbol_on_one_mg1655_locus": 321,
        "gene_symbol_is_ambiguous_in_mg1655": 15,
    }
    with open(osp.join(preprocess, "gene_set.json")) as handle:
        assert len(json.load(handle)) == r.EXPECTED_GENES
