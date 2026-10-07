# tests/torchcell/datasets/ecoli/test_tong2020.py
# [[tests.torchcell.datasets.ecoli.test_tong2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_tong2020.py
"""Tong 2020 loader: Table S1 reading, the two backgrounds, the ECK crosswalk, records.

The synthetic tests run everywhere. The identifier tests read the REAL
``EcoliK12MG1655Genome`` and ``EcoliK12BW25113Genome`` classes over the synthetic K-12
assemblies of ``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` through a
stubbed ``resolve`` with the network refused. Derived expectations for that pair: the
one-to-one ECK pairs are ECK0001 (b0001, BW25113_0001), ECK0002 (b0002, BW25113_0002),
ECK0003 (b0003, BW25113_4412, whose numerics disagree) and ECK0004 (the pseudogene b0004,
BW25113_0004); ECK0005 sits on two BW25113 loci, so b0005 has no partner; ``thrA1`` and
``Hs`` are synonyms of b0002; b0099 is in neither annotation.

The data tests (``--data``) audit every module-level ``SourcedValue`` against the
sha256-pinned paper OCR, read the real Table S1 from the raw mirror, and pin the measured
identifier histograms on the deposited K-12 sets.
"""

from __future__ import annotations

import hashlib
import io
import os
import os.path as osp
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import torchcell.datasets.ecoli.tong2020 as t
from tests.torchcell.datasets._genome_injection_fakes import (
    FakeBW25113Genome,
    install_bacterial_fakes,
)
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.media import MEDIA_LIBRARY, MOPS_MINIMAL
from torchcell.datamodels.schema import (
    BacterialFitnessExperiment,
    BacterialFitnessExperimentReference,
    EnvironmentPhysicalPerturbation,
    MediaComponentRole,
    PhysicalFactor,
    SampleUnit,
)
from torchcell.datasets.bacteria_common import (
    BacterialGenomeInjector,
    LocusTagResolutionError,
)
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import (
    BW25113_ASSEMBLY,
    MG1655_ASSEMBLY,
    EcoliK12BW25113Genome,
    EcoliK12MG1655Genome,
)
from torchcell.verification.sourced import (
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)


# --------------------------------------------------------------------------- #
# Table S1 bytes
# --------------------------------------------------------------------------- #
def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def test_leading_archive_is_the_bytes_through_the_first_end_of_central_directory() -> (
    None
):
    first = _zip({"a.txt": b"alpha"})
    second = _zip({"b.txt": b"beta"})
    assert t.leading_archive(first + b"trailing" + second) == first
    with zipfile.ZipFile(io.BytesIO(t.leading_archive(first + second))) as archive:
        assert archive.read("a.txt") == b"alpha"


def test_leading_archive_refuses_bytes_without_an_archive() -> None:
    with pytest.raises(ValueError, match="no zip end-of-central-directory"):
        t.leading_archive(b"not a zip")


def _workbook(endpoint: pd.DataFrame, comparison: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        endpoint.to_excel(writer, sheet_name=t.ENDPOINT_SHEET, index=False)
        comparison.to_excel(writer, sheet_name=t.KEIO_COMPARISON_SHEET, index=False)
    return buffer.getvalue()


def _endpoint(ids: list[str], genes: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({t.ID_COL: ids, t.GENE_COL: genes})
    for offset, carbon in enumerate(t.CARBON_SOURCE_COLUMNS):
        frame[carbon] = [0.5 + offset / 100 + row for row in range(len(ids))]
    return frame


def _comparison(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({t.KEIO_ID_COL: ids, "Gene names ": ids})


def _write_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: pd.DataFrame,
    comparison: pd.DataFrame,
) -> Path:
    """Write a workbook (plus a trailing junk archive) and pin its leading archive."""
    data = _workbook(endpoint, comparison)
    monkeypatch.setattr(t, "LEADING_ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())
    path = tmp_path / t.XLSX_FILENAME
    path.write_bytes(data + _zip({"stale.xml": b"<x/>"}))
    return path


def test_read_table_s1_reads_both_sheets_through_the_pinned_leading_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_table(
        tmp_path,
        monkeypatch,
        _endpoint(["b0001", "b0002", "b0006"], ["thrL", "thrA", "proC"]),
        _comparison(["b0001", "b0002", "b0002"]),
    )
    table = t.read_table_s1(path)
    assert table.endpoint[t.ID_COL].tolist() == ["b0001", "b0002", "b0006"]
    assert table.keio_ids == frozenset({"b0001", "b0002"})
    assert table.keio_rows == 3
    assert table.endpoint["Glucose"].tolist() == [0.78, 1.78, 2.78]


def test_read_table_s1_refuses_an_unpinned_leading_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_table(
        tmp_path, monkeypatch, _endpoint(["b0001"], ["thrL"]), _comparison(["b0001"])
    )
    monkeypatch.setattr(t, "LEADING_ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="leading xlsx archive sha256"):
        t.read_table_s1(path)


def test_read_table_s1_refuses_a_changed_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoint = _endpoint(["b0001"], ["thrL"]).rename(columns={"Glucose": "glucose"})
    path = _write_table(tmp_path, monkeypatch, endpoint, _comparison(["b0001"]))
    with pytest.raises(ValueError, match="header"):
        t.read_table_s1(path)


def test_read_table_s1_refuses_a_repeated_b_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_table(
        tmp_path,
        monkeypatch,
        _endpoint(["b0001", "b0001"], ["thrL", "thrL"]),
        _comparison(["b0001"]),
    )
    with pytest.raises(ValueError, match=r"repeats b-numbers \['b0001'\]"):
        t.read_table_s1(path)


def test_read_table_s1_refuses_a_keio_b_number_missing_from_s1a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_table(
        tmp_path,
        monkeypatch,
        _endpoint(["b0001"], ["thrL"]),
        _comparison(["b0001", "b0009"]),
    )
    with pytest.raises(ValueError, match=r"absent from S1A: \['b0009'\]"):
        t.read_table_s1(path)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def test_deposit_is_idempotent_and_refuses_a_changed_mirror_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "download.xlsx"
    source.write_bytes(b"table s1 bytes")
    digest = hashlib.sha256(b"table s1 bytes").hexdigest()
    monkeypatch.setattr(t, "XLSX_SHA256", digest)
    data_root = str(tmp_path / "data_root")
    root = t.deposit_raw_mirror(xlsx_path=source, data_root=data_root)
    t.deposit_raw_mirror(xlsx_path=source, data_root=data_root)
    manifest = t.load_manifest(data_root)
    assert root == Path(data_root) / t.RAW_DIR_REL
    assert t.manifest_sha256(manifest, t.XLSX_REL) == digest
    (record,) = manifest.files
    assert record.retrieval is not None
    assert (
        record.retrieval.retriever == "torchcell.literature.retrieve.pmc_cloud_object"
    )
    assert record.retrieval.params == {"key": "PMC7527729.1/mBio.02259-20-st001.xlsx"}
    assert record.retrieval.source_url == (
        "https://pmc-oa-opendata.s3.amazonaws.com/PMC7527729.1/mBio.02259-20-st001.xlsx"
    )
    (root / t.XLSX_REL).write_bytes(b"changed upstream")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        t.deposit_raw_mirror(xlsx_path=source, data_root=data_root)


def test_deposit_refuses_bytes_that_are_not_the_pinned_table(tmp_path: Path) -> None:
    source = tmp_path / "download.xlsx"
    source.write_bytes(b"some other file")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        t.deposit_raw_mirror(xlsx_path=source, data_root=str(tmp_path))
    assert not (tmp_path / t.RAW_DIR_REL).exists()


# --------------------------------------------------------------------------- #
# Identifiers on the synthetic K-12 pair
# --------------------------------------------------------------------------- #
@pytest.fixture
def k12(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome]:
    """Both synthetic K-12 genomes; the network refuses."""
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    files |= write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    return (
        EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=True),
        EcoliK12BW25113Genome(genome_root=str(tmp_path / "bw25113"), overwrite=True),
    )


def _ids(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({t.ID_COL: names})


def test_resolve_strains_assigns_backgrounds_and_crosswalks_keio_b_numbers(
    k12: tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mg1655, bw25113 = k12
    # 6 of the 7 synthetic Keio names resolve (b0099 is retired): 0.857, under the
    # release threshold of 0.95, which the next-but-one test exercises.
    monkeypatch.setattr(t, "MIN_RESOLVED_FRACTION", 0.8)
    endpoint = _ids(
        ["b0001", "b0003", "b0004", "b0002", "thrA1", "b0005", "b0099", "b0006"]
    )
    keio = frozenset({"b0001", "b0003", "b0004", "b0002", "thrA1", "b0005", "b0099"})
    strains = t.resolve_strains(endpoint, keio, mg1655, bw25113, label="synthetic")
    assert [
        (s.row, s.reported, s.collection, s.systematic_gene_name, s.perturbed_gene_name)
        for s in strains.kept
    ] == [
        (0, "b0001", "keio", "BW25113_0001", "thrL"),
        (1, "b0003", "keio", "BW25113_4412", "hokC"),
        (2, "b0004", "keio", "BW25113_0004", "yaaP"),
        # b0002 IS the locus thrA1 also reaches, so it keeps it; thrA1 is dropped
        (3, "b0002", "keio", "BW25113_0002", "thrA"),
        (7, "b0006", "srna", "b0006", "proC"),
    ]
    assert {s.gene_namespace for s in strains.kept if s.collection == "keio"} == {
        "ecoli_k12_bw25113_locus_tag"
    }
    assert [s.gene_namespace for s in strains.kept if s.collection == "srna"] == [
        "ecoli_k12_mg1655_bnumber"
    ]
    assert strains.dropped_fragment == ["thrA1"]
    assert strains.dropped_not_in_mg1655 == ["b0099"]
    assert strains.dropped_ambiguous == []
    assert strains.dropped_no_eck_partner == ["b0005"]
    assert [
        (u.reported, u.eck, u.bw25113, u.numerics_agree) for u in strains.crosswalk
    ] == [
        ("b0001", "ECK0001", "BW25113_0001", True),
        ("b0003", "ECK0003", "BW25113_4412", False),
        ("b0004", "ECK0004", "BW25113_0004", True),
        ("b0002", "ECK0002", "BW25113_0002", True),
    ]
    assert strains.crosswalk_pairs == 4
    bw = strains.keio_vs_bw25113
    assert bw.unique_names == 4
    assert bw.status_histogram[GeneNameStatus.CURRENT] == 3
    assert bw.status_histogram[GeneNameStatus.NON_GENE_FEATURE] == 1
    assert strains.keio_vs_mg1655.kept_on_collision == ("b0002", "thrA1")
    assert strains.srna_vs_mg1655.unique_names == 1


def test_a_collision_group_with_no_direct_member_is_dropped_whole(
    k12: tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome],
) -> None:
    mg1655, bw25113 = k12
    endpoint = _ids(["b0001", "Hs", "thrA1", "b0006"])
    strains = t.resolve_strains(
        endpoint, frozenset({"b0001", "Hs", "thrA1"}), mg1655, bw25113, label="frag"
    )
    assert [s.systematic_gene_name for s in strains.kept] == ["BW25113_0001", "b0006"]
    assert strains.dropped_fragment == ["Hs", "thrA1"]


def test_a_collection_below_the_resolution_threshold_stops_the_build(
    k12: tuple[EcoliK12MG1655Genome, EcoliK12BW25113Genome],
) -> None:
    mg1655, bw25113 = k12
    endpoint = _ids(["b0001", "b0097", "b0098", "b0099", "b0006"])
    keio = frozenset({"b0001", "b0097", "b0098", "b0099"})
    with pytest.raises(LocusTagResolutionError, match=r"1 of 4 names \(0\.250\)"):
        t.resolve_strains(endpoint, keio, mg1655, bw25113, label="threshold")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def test_the_screen_medium_is_solid_mops_minimal_with_agar_and_no_carbon() -> None:
    medium = t.TONG2020_MOPS_MINIMAL_AGAR
    assert medium.state == "solid"
    assert medium.base_medium == "MOPS_MINIMAL"
    assert MEDIA_LIBRARY[medium.base_medium] is MOPS_MINIMAL
    assert medium.components[:-1] == MOPS_MINIMAL.components
    agar = medium.components[-1]
    assert agar.compound.name == "agar"
    assert agar.role is MediaComponentRole.gelling_agent
    assert agar.concentration is None
    assert not any(
        c.role is MediaComponentRole.carbon_source for c in medium.components
    )


def test_the_carbon_source_is_a_physical_factor_whose_dose_is_a_typed_gap() -> None:
    environment = t.environment("Glucose")
    assert environment.media is t.TONG2020_MOPS_MINIMAL_AGAR
    assert environment.temperature is not None
    assert environment.temperature.value == 37.0
    assert environment.duration_hours == 24.0
    assert environment.aerobicity == "aerobic"
    (perturbation,) = environment.perturbations
    assert isinstance(perturbation, EnvironmentPhysicalPerturbation)
    assert perturbation.factor is PhysicalFactor.carbon_source
    assert perturbation.agent is not None
    assert perturbation.agent.name == "D-glucose"
    assert perturbation.agent.inchikey == "WQZGKKKJIJFFOK-GASJEMHNSA-N"
    assert perturbation.magnitude is None
    (gap,) = perturbation.provenance_gaps
    assert gap.field == "magnitude"
    assert gap.reason is ProvenanceGapReason.deferred_pending_source_review
    assert gap.resolve_with is not None
    assert gap.resolve_with.source_uri == "https://edbrownlab.shinyapps.io/CarPE/"


def test_every_carbon_source_column_is_one_distinct_environment() -> None:
    agents = [
        t.environment(carbon).perturbations[0].model_dump()["agent"]["name"]
        for carbon in t.CARBON_SOURCE_COLUMNS
    ]
    assert len(t.CARBON_SOURCE_COLUMNS) == 30
    assert len(set(agents)) == 30


def test_keio_phenotype_carries_two_technical_replicates_and_no_uncertainty() -> None:
    phenotype = t.phenotype(0.42, "keio")
    assert phenotype.fitness == 0.42
    assert phenotype.n_samples == 2
    assert phenotype.sample_unit is SampleUnit.technical_replicate
    assert phenotype.fitness_uncertainty is None
    assert phenotype.fitness_se is None
    assert [(g.field, g.reason) for g in phenotype.provenance_gaps] == [
        ("fitness_uncertainty", ProvenanceGapReason.not_reported_by_primary)
    ]


def test_library_phenotype_gaps_the_undescribed_replicate_design() -> None:
    phenotype = t.phenotype(1.1, "srna")
    assert phenotype.n_samples is None
    assert phenotype.sample_unit is None
    assert phenotype.gapped_fields() == {
        "fitness_uncertainty",
        "n_samples",
        "sample_unit",
    }


def test_a_zero_growth_cell_stays_zero() -> None:
    assert t.phenotype(0.0, "keio").fitness == 0.0


def test_the_reference_is_the_normalization_baseline_of_one() -> None:
    reference = t.reference_phenotype()
    assert reference.fitness == 1.0
    assert reference.gapped_fields() == {"n_samples"}


def test_genotype_names_the_collection_and_only_keio_names_a_cassette() -> None:
    keio = t.StrainRecord(
        row=0,
        reported="b0001",
        collection="keio",
        systematic_gene_name="BW25113_0001",
        perturbed_gene_name="thrL",
        gene_namespace="ecoli_k12_bw25113_locus_tag",
    )
    library = t.StrainRecord(
        row=1,
        reported="b4417",
        collection="srna",
        systematic_gene_name="b4417",
        perturbed_gene_name="rybB",
        gene_namespace="ecoli_k12_mg1655_bnumber",
    )
    (keio_deletion,) = t.genotype(keio).perturbations
    (library_deletion,) = t.genotype(library).perturbations
    assert keio_deletion.model_dump(
        include={"systematic_gene_name", "gene_namespace", "collection", "cassette"}
    ) == {
        "systematic_gene_name": "BW25113_0001",
        "gene_namespace": "ecoli_k12_bw25113_locus_tag",
        "collection": "Keio collection",
        "cassette": "kanamycin cassette",
    }
    assert library_deletion.model_dump(
        include={"systematic_gene_name", "gene_namespace", "collection", "cassette"}
    ) == {
        "systematic_gene_name": "b4417",
        "gene_namespace": "ecoli_k12_mg1655_bnumber",
        "collection": "sRNA and small protein deletion library",
        "cassette": None,
    }


def test_the_loader_is_registered_and_receives_the_bw25113_genome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert dataset_registry["CarbonSourceTong2020Dataset"] is (
        t.CarbonSourceTong2020Dataset
    )
    log = install_bacterial_fakes(monkeypatch)
    kwargs = BacterialGenomeInjector("/dr").genome_kwargs(t.CarbonSourceTong2020Dataset)
    assert set(kwargs) == {"ecoli_genome"}
    assert isinstance(kwargs["ecoli_genome"], FakeBW25113Genome)
    assert [name for name, _ in log] == ["FakeBW25113Genome"]
    dataset = t.CarbonSourceTong2020Dataset.__new__(t.CarbonSourceTong2020Dataset)
    assert dataset.experiment_class is BacterialFitnessExperiment
    assert dataset.reference_class is BacterialFitnessExperimentReference


# --------------------------------------------------------------------------- #
# Data tests: the mirrors, the deposited K-12 sets, the release
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    return os.environ["DATA_ROOT"]


@pytest.mark.data
def test_every_sourced_value_is_backed_by_its_verbatim_quote() -> None:
    root = osp.join(_data_root(), "torchcell-library")
    values = [v for v in vars(t).values() if isinstance(v, SourcedValue)]
    assert len(values) == len(t.SOURCED_VALUES) == 18
    for value in values:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"


@pytest.mark.data
def test_the_release_resolves_to_the_measured_strain_counts() -> None:
    from torchcell.datasets.bacteria_common import bacterial_genome

    table = t.read_table_s1(osp.join(_data_root(), t.RAW_DIR_REL, t.XLSX_REL))
    assert table.endpoint.shape == (3796, 32)
    assert (len(table.keio_ids), table.keio_rows) == (3725, 3727)
    assert int(table.endpoint[list(t.CARBON_SOURCE_COLUMNS)].isna().sum().sum()) == 0
    mg1655 = bacterial_genome("ecoli", "MG1655")
    bw25113 = bacterial_genome("ecoli", "BW25113")
    assert isinstance(mg1655, EcoliK12MG1655Genome)
    assert isinstance(bw25113, EcoliK12BW25113Genome)
    strains = t.resolve_strains(
        table.endpoint, table.keio_ids, mg1655, bw25113, label="release"
    )
    counts = pd.Series([s.collection for s in strains.kept]).value_counts().to_dict()
    assert counts == {"keio": 3644, "srna": 70}
    assert (
        len(strains.dropped_fragment),
        strains.dropped_not_in_mg1655,
        strains.dropped_ambiguous,
        len(strains.dropped_no_eck_partner),
    ) == (59, ["b0370", "b0510", "b3776", "b4223", "b4274", "b4590"], [], 17)
    keio = strains.keio_vs_mg1655
    assert {s.value: n for s, n in keio.status_histogram.items()} == {
        "current": 3581,
        "renamed": 2,
        "non_gene_feature": 137,
        "retired": 5,
        "ambiguous": 0,
    }
    bw = strains.keio_vs_bw25113
    assert {s.value: n for s, n in bw.status_histogram.items()} == {
        "current": 3536,
        "renamed": 0,
        "non_gene_feature": 108,
        "retired": 0,
        "ambiguous": 0,
    }
    assert sum(not use.numerics_agree for use in strains.crosswalk) == 8
    assert sum(t.EXPECTED_RECORDS.values()) == 30 * len(strains.kept) == 111420
