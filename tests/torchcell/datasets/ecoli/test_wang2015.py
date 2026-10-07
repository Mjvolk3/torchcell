# tests/torchcell/datasets/ecoli/test_wang2015.py
# [[tests.torchcell.datasets.ecoli.test_wang2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_wang2015.py
"""The Wang 2015 isoprenol-tolerance loader (``torchcell.datasets.ecoli.wang2015``).

Synthetic tests (run everywhere) feed the parser a hand-built ``pdftotext -layout`` text
of Tables S2 to S4 and build over the real ``EcoliK12BW25113Genome`` on the synthetic
BW25113 assembly of ``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` (JW4367 on
BW25113_0001 thrL, JW0001 on BW25113_0002 thrA, hokC on BW25113_4412), served through a
stubbed ``resolve`` with the network refused. ``pdftotext`` itself is never run there.
The synthetic strains, sorted by gene name:

    strain    Table S2 number  family  OD600 no / with   result
    BW25113   (parent)                 8.00 / 4.00       reference, ratio 0.5
    hokC      JW4367           OMP     8.00 / 6.00       BW25113_4412; JW4367 is thrL's
                                                         locus, a disagreement; log2 1.5
    thrA      JW0001           ABC     8.00 / 2.00       BW25113_0002, JW0001 stored; -1
    thrL      JJW4367          RND     5.00 / 2.50       BW25113_0001; malformed number,
                                                         a disagreement; 0
    yaaQ      JW9999           MFS     4.00 / 2.00       name not on the genome, dropped

Three of four names resolve (0.75 < 0.95), so the success path lowers
``MIN_RESOLVED_FRACTION`` to 0.5 and a separate test shows the default stops the build.

Data-gated tests (``@pytest.mark.data``) read the real raw mirror, the literature mirror
and the built dev-tree LMDB under ``$DATA_ROOT`` (they never build it): the pins, every
sourced quote, the real extraction against its pinned digest, the OCR of Table S3
against the text-layer parse, the Keio list in the Fuhrer 2017 raw mirror against the
five Table S2 disagreements, the acrA record by hand, and the L0-L4 verifier.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import torchcell.datasets.ecoli.wang2015 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError, RawSha256MismatchError
from torchcell.datamodels.media import YT_2X
from torchcell.datamodels.schema import (
    AssayType,
    AssemblyReferenceGenome,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Compound,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    MeasurementType,
    PhysicalFactor,
    SampleUnit,
    SmallMoleculePerturbation,
)
from torchcell.datasets.bacteria_common import LocusTagResolutionError
from torchcell.literature.manifest import RetrievalMethod, RetrievalRecord
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import BW25113_ASSEMBLY, EcoliK12BW25113Genome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

D = "Δ"
PM = "±"
TABLE_S2 = f"""Table S2. Mutant strains used in this study.

  Names        Keio No.      MDT               Names     Keio No.      MDT
                            families                                  families
BW{D}thrA        JW0001         ABC          BW{D}hokC       JW4367        OMP
BW{D}thrL        JJW4367        RND          BW{D}yaaQ       JW9999        MFS
BW{D}acrAB       This study      -


                                                3
"""
TABLE_S3 = f"""Table S3. Cell growth of the MDT null mutants in the absence and presence of isoprenol.

    Strains          No         0.5% (v/v)           Strains        No          0.5% (v/v)
                  isoprenol      isoprenol                       isoprenol       isoprenol
  BW25113        8.00 {PM} 0.10     4.00 {PM} 0.01         BW{D}hokC      8.00 {PM} 0.20   6.00 {PM} 0.10

  BW{D}thrA        8.00 {PM} 0.05     2.00 {PM} 0.04         BW{D}yaaQ      4.00 {PM} 0.02   2.00 {PM} 0.00
  BW{D}thrL        5.00 {PM} 0.30     2.50 {PM} 0.01
Note: The results are presented as means {PM} standard divisions.
"""
TABLE_S4 = f"""Table S4. Transcript profiles of targeted transporters in BW{D}acrA.
        acrA             3.16 {PM} 0.87                           1.59 {PM} 0.08
"""
TEXT = "header page\n" + TABLE_S2 + TABLE_S3 + TABLE_S4
SECTION_S2 = m.table_section(TEXT, m.TABLE_S2_MARKER, m.TABLE_S3_MARKER)
SECTION_S3 = m.table_section(TEXT, m.TABLE_S3_MARKER, m.TABLE_S4_MARKER)
SYNTHETIC_DIGEST = m.table_s3_digest(m.parse_table_s3(SECTION_S3))
SYNTHETIC_COUNTS = {"above_52.5": 1, "above_57.5": 1}
REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="BW25113",
    assembly_set="ecoli_K12_BW25113_ASM75055v1",
    assembly_accession="GCA_000750555.1",
)
LOG2_1_5 = math.log2(1.5)


def _row(strain: str, od0: float, od1: float) -> m.TableS3Row:
    return m.TableS3Row(
        strain=strain, od600_without=od0, sd_without=0.1, od600_with=od1, sd_with=0.1
    )


@pytest.fixture
def bw25113(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12BW25113Genome:
    """The real BW25113 class over the synthetic assembly; network refused."""
    files = write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "bw25113"
    root.mkdir()
    return EcoliK12BW25113Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def presence_only_pins(monkeypatch: pytest.MonkeyPatch) -> list[Mapping[str, str]]:
    """Replace the build-time byte check with a presence check that records the pins."""
    calls: list[Mapping[str, str]] = []

    def record(raw_dir: str, pins: Mapping[str, str]) -> None:
        missing = [f for f in pins if not osp.exists(osp.join(raw_dir, f))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")
        calls.append(dict(pins))

    monkeypatch.setattr(m, "verify_raw_files", record)
    return calls


@pytest.fixture
def synthetic_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stated counts, table size and digest of the synthetic tables."""
    monkeypatch.setattr(m, "TABLE_S3_MUTANTS", 4)
    monkeypatch.setattr(m, "TABLE_S3_SHA256", SYNTHETIC_DIGEST)
    monkeypatch.setattr(m, "REPORTED_COUNTS", SYNTHETIC_COUNTS)
    monkeypatch.setattr(m, "REPORTED_TOLERANT_COUNT", 0)
    monkeypatch.setattr(m, "TABLE_S2_MDT_COUNT", 3)


@pytest.fixture
def synthetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bw25113: EcoliK12BW25113Genome,
    presence_only_pins: list[Mapping[str, str]],
    synthetic_constants: None,
) -> Path:
    """A dataset root under ``tmp_path/data/torchcell`` whose raw/ holds a stand-in PDF
    and whose text layer is the synthetic one.
    """
    root = tmp_path / m.DATASET_ROOT_REL
    (root / "raw").mkdir(parents=True)
    (root / "raw" / m.SI_PDF).write_bytes(b"%PDF-1.4 synthetic")
    monkeypatch.setattr(m, "layout_text", lambda path: TEXT)
    monkeypatch.setattr(m, "pdftotext_version", lambda: "pdftotext version test")
    monkeypatch.setattr(m, "assembly_reference", lambda strain, **_: REFERENCE)
    monkeypatch.setattr(m, "IDENTITY_EVIDENCE", {"hokC": ("evidence_key",)})
    return root


# --------------------------------------------------------------------------- #
# Parsing the tables
# --------------------------------------------------------------------------- #
def test_table_section_requires_each_caption_once_and_in_order() -> None:
    assert SECTION_S2.startswith("Table S2.")
    assert "Table S3." not in SECTION_S2
    assert f"BW{D}acrA." not in SECTION_S3
    with pytest.raises(m.TableExtractionError, match="'Table S5.' occurs 0 times"):
        m.table_section(TEXT, m.TABLE_S2_MARKER, "Table S5.")
    with pytest.raises(m.TableExtractionError, match="'Table S2.' occurs 2 times"):
        m.table_section(TEXT + TABLE_S2, m.TABLE_S2_MARKER, m.TABLE_S3_MARKER)
    with pytest.raises(m.TableExtractionError, match="precedes"):
        m.table_section(TEXT, m.TABLE_S3_MARKER, m.TABLE_S2_MARKER)


def test_parse_table_s2_keeps_tokens_verbatim_two_entries_per_line() -> None:
    entries = m.parse_table_s2(SECTION_S2)
    assert [(e.gene_name, e.keio_token, e.mdt_family) for e in entries] == [
        ("thrA", "JW0001", "ABC"),
        ("hokC", "JW4367", "OMP"),
        ("thrL", "JJW4367", "RND"),
        ("yaaQ", "JW9999", "MFS"),
        ("acrAB", "This study", "-"),
    ]


def test_parse_table_s2_refuses_an_unparsed_label_and_a_repeat() -> None:
    with pytest.raises(m.TableExtractionError, match="6 strain labels but 5 parsed"):
        m.parse_table_s2(SECTION_S2 + f"BW{D}lonely\n")
    with pytest.raises(m.TableExtractionError, match="repeats a strain"):
        m.parse_table_s2(SECTION_S2 + f"BW{D}thrA  JW0001  ABC\n")


def test_parse_table_s3_reads_means_and_sds() -> None:
    rows = {row.strain: row for row in m.parse_table_s3(SECTION_S3)}
    assert set(rows) == {
        "BW25113",
        f"BW{D}hokC",
        f"BW{D}thrA",
        f"BW{D}yaaQ",
        f"BW{D}thrL",
    }
    assert rows[f"BW{D}hokC"] == m.TableS3Row(
        strain=f"BW{D}hokC",
        od600_without=8.0,
        sd_without=0.2,
        od600_with=6.0,
        sd_with=0.1,
    )
    assert rows[f"BW{D}thrA"].growth_ratio == 0.25
    assert rows[f"BW{D}thrA"].growth_inhibition_pct == 75.0
    assert rows[f"BW{D}thrA"].gene_name == "thrA"
    with pytest.raises(ValueError, match="'BW25113' is not a deletion strain label"):
        _ = rows["BW25113"].gene_name


def test_parse_table_s3_refuses_a_row_it_cannot_read() -> None:
    with pytest.raises(m.TableExtractionError, match="6 strain labels but 5 parsed"):
        m.parse_table_s3(SECTION_S3 + f"  BW{D}broken   8.00   4.00\n")


def test_table_s3_digest_ignores_the_layout_order() -> None:
    rows = m.parse_table_s3(SECTION_S3)
    assert m.table_s3_digest(list(reversed(rows))) == SYNTHETIC_DIGEST
    changed = [rows[0].model_copy(update={"sd_with": 0.02}), *rows[1:]]
    assert m.table_s3_digest(changed) != SYNTHETIC_DIGEST


def test_split_parent_sorts_and_refuses_bad_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "TABLE_S3_MUTANTS", 4)
    rows = m.parse_table_s3(SECTION_S3)
    parent, mutants = m.split_parent(rows)
    assert parent.strain == "BW25113"
    assert [row.gene_name for row in mutants] == ["hokC", "thrA", "thrL", "yaaQ"]
    with pytest.raises(m.TableExtractionError, match="has 0 BW25113 rows"):
        m.split_parent(mutants)
    with pytest.raises(m.TableExtractionError, match="has 2 BW25113 rows"):
        m.split_parent([*rows, parent])
    with pytest.raises(m.TableExtractionError, match="repeats a strain"):
        m.split_parent([*rows, mutants[0]])
    with pytest.raises(m.TableExtractionError, match="3 deletion rows, expected 4"):
        m.split_parent(rows[:-1])


def test_keio_entries_must_be_exactly_the_measured_strains() -> None:
    entries = m.parse_table_s2(SECTION_S2)
    mutants = [_row(f"BW{D}{g}", 8.0, 4.0) for g in ("hokC", "thrA", "thrL", "yaaQ")]
    keio = m.keio_entries_for(mutants, entries)
    assert sorted(keio) == ["hokC", "thrA", "thrL", "yaaQ"]
    with pytest.raises(
        m.StrainSetMismatchError,
        match=r"Table S3 only: \['zzzZ'\]; Table S2 Keio only: \['yaaQ'\]",
    ):
        m.keio_entries_for([*mutants[:3], _row(f"BW{D}zzzZ", 8.0, 4.0)], entries)


def test_reported_counts_are_recomputed_and_a_mismatch_stops(
    monkeypatch: pytest.MonkeyPatch, synthetic_constants: None
) -> None:
    entries = m.parse_table_s2(SECTION_S2)
    _, mutants = m.split_parent(m.parse_table_s3(SECTION_S3))
    keio = m.keio_entries_for(mutants, entries)
    checks = m.reported_count_checks(mutants, keio)
    assert [(c.stated, c.measured) for c in checks] == [(1, 1), (1, 1), (0, 0), (3, 3)]
    monkeypatch.setattr(m, "REPORTED_TOLERANT_COUNT", 2)
    with pytest.raises(
        m.ReportedCountMismatchError,
        match=r"MDT strains with growth inhibition below 47.5%: stated 2, measured 0",
    ):
        m.reported_count_checks(mutants, keio)


def test_the_real_constants_are_the_papers() -> None:
    assert m.REPORTED_COUNTS == {"above_52.5": 17, "above_57.5": 7}
    assert (m.REPORTED_TOLERANT_COUNT, m.TABLE_S2_MDT_COUNT) == (11, 45)
    assert m.TABLE_S3_MUTANTS == 46
    assert [f.name for f in m.RAW_FILES] == ["si1.pdf"]
    raw = m.RAW_FILES_BY_NAME["si1.pdf"]
    assert raw.sha256 == raw.retrieval.sha256 == m.SI_PDF_SHA256
    assert raw.retrieval.method == RetrievalMethod.pmc_cloud
    assert raw.mirror_relpath == "data/si1.pdf"


# --------------------------------------------------------------------------- #
# pdftotext invocation (subprocess faked; the real tool runs in the data tests)
# --------------------------------------------------------------------------- #
def test_layout_text_and_version_call_pdftotext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(args)
        if args[1] == "-v":
            return SimpleNamespace(stdout="", stderr="pdftotext version 9.9\nCopyright")
        return SimpleNamespace(stdout=TEXT.encode("utf-8"), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert m.layout_text("/x/si1.pdf") == TEXT
    assert m.pdftotext_version() == "pdftotext version 9.9"
    record = m.extraction_record("ab" * 32)
    assert (record.tool, record.version, record.params) == (
        "pdftotext",
        "pdftotext version 9.9",
        {"args": ["-layout", "-enc", "UTF-8"]},
    )
    assert record.input_sha256 == ["ab" * 32]
    assert calls[0] == ["pdftotext", "-layout", "-enc", "UTF-8", "/x/si1.pdf", "-"]


# --------------------------------------------------------------------------- #
# Records (pure)
# --------------------------------------------------------------------------- #
def test_log2_relative_tolerance_is_the_papers_ratio() -> None:
    parent = _row("BW25113", 8.30, 4.12)
    acra = _row(f"BW{D}acrA", 8.13, 6.26)
    assert m.log2_relative_tolerance(acra, parent) == pytest.approx(
        0.6333743039086592, abs=1e-15
    )
    assert m.log2_relative_tolerance(parent, parent) == 0.0


def test_response_phenotype_types_the_readout_and_gaps_the_uncertainty() -> None:
    phenotype = m.response_phenotype(-1.0)
    assert phenotype.environment_response == -1.0
    assert phenotype.measurement_type is MeasurementType.log2_ratio
    assert phenotype.assay_type is AssayType.liquid_od_growth
    assert (phenotype.n_samples, phenotype.sample_unit) == (
        2,
        SampleUnit.biological_replicate,
    )
    assert phenotype.environment_response_se is None
    assert phenotype.environment_response_uncertainty is None
    assert phenotype.gapped_fields() == {
        "environment_response_uncertainty",
        "environment_response_se",
    }
    assert phenotype.units == m.RESPONSE_UNITS


def test_environment_is_2yt_with_ph_and_isoprenol() -> None:
    env = m.environment()
    assert env.media == YT_2X
    assert env.temperature is not None and env.temperature.value == 30.0
    assert (env.aerobicity, env.duration_hours) == ("aerobic", 12.0)
    ph, isoprenol = env.perturbations
    assert isinstance(ph, EnvironmentPhysicalPerturbation)
    assert ph.factor is PhysicalFactor.ph
    assert ph.magnitude is not None and ph.magnitude.value == 7.0
    assert ph.gapped_fields() == {"agent"}
    assert isinstance(isoprenol, SmallMoleculePerturbation)
    assert isoprenol.concentration.value == 0.5
    assert isoprenol.concentration.unit is ConcentrationUnit.percent_v_v
    assert isoprenol.gapped_fields() == {"solvent"}
    assert isoprenol.compound.name == "isoprenol"


def test_isoprenol_goes_through_the_identity_layer_and_is_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compound = m.isoprenol_compound()
    assert compound.inchikey is None
    assert compound.gapped_fields() == {"inchikey"}

    def row(inchikey: str) -> Any:
        return lambda label: Compound(name="3-methylbut-3-en-1-ol", inchikey=inchikey)

    monkeypatch.setattr(m, "resolved_compound", row(m.ISOPRENOL_INCHIKEY))
    assert m.isoprenol_compound().inchikey == m.ISOPRENOL_INCHIKEY
    monkeypatch.setattr(m, "resolved_compound", row("AAAAAAAAAAAAAA-BBBBBBBBBB-N"))
    with pytest.raises(m.CompoundIdentityConflictError, match="AAAAAAAAAAAAAA"):
        m.isoprenol_compound()


def test_the_recorded_inchikey_is_the_rdkit_key_of_the_smiles() -> None:
    pytest.importorskip("rdkit")
    from torchcell.datamodels.compound_identity import inchikey_from_smiles

    assert inchikey_from_smiles(m.ISOPRENOL_SMILES) == m.ISOPRENOL_INCHIKEY


def _identity(accession: str | None) -> m.StrainIdentity:
    return m.StrainIdentity(
        row=_row(f"BW{D}thrA", 8.0, 2.0),
        locus_tag="BW25113_0002",
        symbol="thrA",
        keio_token="JW0001",
        mdt_family="ABC",
        keio_accession=accession,
    )


def test_deletion_genotype_stores_the_keio_number_only_when_it_agrees() -> None:
    (with_number,) = m.deletion_genotype(_identity("JW0001")).perturbations
    assert with_number.model_dump()["construction"]["strain_accession"] == "JW0001"
    dumped = with_number.model_dump()
    assert (
        dumped["systematic_gene_name"],
        dumped["gene_namespace"],
        dumped["collection"],
        dumped["cassette"],
    ) == (
        "BW25113_0002",
        "ecoli_k12_bw25113_locus_tag",
        "Keio collection",
        "kanamycin cassette flanked by FLP recognition target sites",
    )
    (without,) = m.deletion_genotype(_identity(None)).perturbations
    assert without.model_dump()["construction"] is None


def test_build_experiment_and_reference_round_trip() -> None:
    env = m.environment()
    parent = _row("BW25113", 8.0, 4.0)
    experiment = m.build_experiment("ds", _identity("JW0001"), parent, env)
    assert experiment.phenotype.environment_response == -1.0
    reference = m.build_reference("ds", REFERENCE, env)
    dumped = reference.model_dump()
    assert dumped["genome_reference"]["assembly_set"] == "ecoli_K12_BW25113_ASM75055v1"
    assert dumped["phenotype_reference"]["environment_response"] == 0.0
    assert dumped["environment_reference"] == env.model_dump()
    again = type(reference).model_validate(dumped)
    assert again.genome_reference == REFERENCE


# --------------------------------------------------------------------------- #
# Identifiers on the synthetic genome
# --------------------------------------------------------------------------- #
def _synthetic_inputs() -> tuple[list[m.TableS3Row], dict[str, m.TableS2Entry]]:
    _, mutants = m.split_parent(m.parse_table_s3(SECTION_S3))
    return mutants, m.keio_entries_for(mutants, m.parse_table_s2(SECTION_S2))


def test_resolve_strains_places_by_name_and_ledgers_the_numbers(
    monkeypatch: pytest.MonkeyPatch,
    bw25113: EcoliK12BW25113Genome,
    synthetic_constants: None,
) -> None:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.5)
    monkeypatch.setattr(m, "IDENTITY_EVIDENCE", {"hokC": ("evidence_key",)})
    mutants, keio = _synthetic_inputs()
    kept, rule, ledger = m.resolve_strains(bw25113, mutants, keio, label="t")
    assert [
        (k.row.gene_name, k.locus_tag, k.symbol, k.keio_accession) for k in kept
    ] == [
        ("hokC", "BW25113_4412", "hokC", None),
        ("thrA", "BW25113_0002", "thrA", "JW0001"),
        ("thrL", "BW25113_0001", "thrL", None),
    ]
    assert (rule.rule, rule.items) == (
        "strain_name_not_on_a_bw25113_locus",
        [f"BW{D}yaaQ (Table S2 JW9999): retired yaaQ"],
    )
    assert ledger.n_keio_agree == 1
    assert [d.model_dump() for d in ledger.keio_disagreements] == [
        {
            "gene_name": "hokC",
            "name_locus": "BW25113_4412",
            "keio_token": "JW4367",
            "keio_resolution": "renamed BW25113_0001",
            "keio_locus_symbol": "thrL",
            "evidence": ["evidence_key"],
        },
        {
            "gene_name": "thrL",
            "name_locus": "BW25113_0001",
            "keio_token": "JJW4367",
            "keio_resolution": "retired JJW4367",
            "keio_locus_symbol": None,
            "evidence": [],
        },
    ]
    assert ledger.names.status_histogram[GeneNameStatus.RENAMED] == 3
    assert ledger.keio_numbers.retired_kept == ("JJW4367", "JW9999")


def test_resolve_strains_stops_below_the_threshold(
    bw25113: EcoliK12BW25113Genome, synthetic_constants: None
) -> None:
    mutants, keio = _synthetic_inputs()
    with pytest.raises(LocusTagResolutionError, match=r"3 of 4 names \(0\.750\)"):
        m.resolve_strains(bw25113, mutants, keio, label="t")


# --------------------------------------------------------------------------- #
# The hermetic build
# --------------------------------------------------------------------------- #
def test_build_three_records_with_the_parent_reference_and_ledgers(
    synthetic: Path,
    monkeypatch: pytest.MonkeyPatch,
    bw25113: EcoliK12BW25113Genome,
    presence_only_pins: list[Mapping[str, str]],
) -> None:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.5)
    dataset = m.EnvChemgenWang2015Dataset(root=str(synthetic), ecoli_genome=bw25113)
    assert presence_only_pins == [m.DATA_SHA256]
    assert len(dataset) == 3
    assert dataset.experiment_class is BacterialEnvironmentResponseExperiment
    assert dataset.reference_class is BacterialEnvironmentResponseExperimentReference

    values = []
    for idx in range(3):
        experiment = dataset[idx]["experiment"]
        (perturbation,) = experiment["genotype"]["perturbations"]
        values.append(
            (
                perturbation["systematic_gene_name"],
                (perturbation["construction"] or {}).get("strain_accession"),
                experiment["phenotype"]["environment_response"],
            )
        )
    assert values[0][:2] == ("BW25113_4412", None)
    assert values[0][2] == pytest.approx(LOG2_1_5, abs=1e-15)
    assert values[1:] == [("BW25113_0002", "JW0001", -1.0), ("BW25113_0001", None, 0.0)]
    first = dataset[0]
    assert first["experiment"]["environment"] == m.environment().model_dump()
    assert first["reference"]["genome_reference"] == REFERENCE.model_dump()
    assert first["reference"]["phenotype_reference"]["environment_response"] == 0.0
    assert first["publication"] == m.PUBLICATION.model_dump()

    preprocess = synthetic / "preprocess"
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert (
        drops["table_rows"],
        drops["source_records"],
        drops["kept_records"],
        drops["dropped_records"],
    ) == (5, 4, 3, 1)
    ledger = json.loads((preprocess / "identifier_reconciliation.json").read_text())
    assert [d["gene_name"] for d in ledger["keio_disagreements"]] == ["hokC", "thrL"]
    extraction = json.loads((preprocess / "extraction.json").read_text())
    assert extraction["table_s3_sha256"] == SYNTHETIC_DIGEST
    assert extraction["pdftotext"] == "pdftotext version test"
    assert extraction["parent_growth_inhibition_pct"] == 50.0
    table = (preprocess / "table_s3.csv").read_text(encoding="utf-8").splitlines()
    assert table[0].startswith("record,strain,mdt_family,keio_token,keio_accession")
    assert table[1].startswith(",BW25113,")
    assert table[3] == (
        f"1,BW{D}thrA,ABC,JW0001,JW0001,BW25113_0002,thrA,8.0,0.05,2.0,0.04,75.0,-1.0"
    )


def test_build_refuses_a_digest_other_than_the_pin(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> None:
    monkeypatch.setattr(m, "TABLE_S3_SHA256", "0" * 64)
    with pytest.raises(m.TableExtractionError, match=f"sha256 {SYNTHETIC_DIGEST}"):
        m.EnvChemgenWang2015Dataset(root=str(synthetic), ecoli_genome=bw25113)


def test_build_stops_below_the_resolution_threshold(
    synthetic: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    with pytest.raises(LocusTagResolutionError, match=r"3 of 4 names \(0\.750\)"):
        m.EnvChemgenWang2015Dataset(root=str(synthetic), ecoli_genome=bw25113)


def test_build_refuses_a_genome_of_another_strain(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> None:
    monkeypatch.setattr(
        type(bw25113), "ASSEMBLY_SET", "ecoli_K12_MG1655_ASM584v2", raising=False
    )
    with pytest.raises(ValueError, match="needs the ecoli_K12_BW25113_ASM75055v1"):
        m.EnvChemgenWang2015Dataset(root=str(synthetic), ecoli_genome=bw25113)


def test_process_verifies_the_real_pin(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> None:
    """With the real check back, the stand-in PDF is refused by its pin."""
    from torchcell.data import verify_raw_files

    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    with pytest.raises(RawSha256MismatchError, match=f"expected {m.SI_PDF_SHA256}"):
        m.EnvChemgenWang2015Dataset(root=str(synthetic), ecoli_genome=bw25113)


def test_run_verification_passes_on_the_synthetic_build(
    synthetic: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bw25113: EcoliK12BW25113Genome,
) -> None:
    import torchcell.verification.runners as runners

    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.5)
    m.EnvChemgenWang2015Dataset(root=str(synthetic), ecoli_genome=bw25113)
    monkeypatch.setattr(runners, "_genome_for_reference", lambda ref, base: bw25113)
    monkeypatch.setattr(
        runners,
        "_gene_set_for_reference",
        lambda ref, base: {locus.tag for locus in BW25113_LOCI},
    )
    paper = tmp_path / "torchcell-library" / "key" / "paper.md"
    paper.parent.mkdir(parents=True)
    paper.write_text("the strain was grown at 30 C.", encoding="utf-8")
    sourced = SourcedValue(
        value=30.0,
        quote="grown at 30 C",
        provenance=Provenance(
            source_uri="paper.md",
            citation_key="key",
            sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
            method="synthetic",
        ),
    )
    monkeypatch.setattr(m, "SOURCED_VALUES", {"temperature_c": sourced})
    report = m.run_verification(str(tmp_path))
    assert report.passed, report.summary()
    names = {result.name for result in report.results}
    assert {
        "reference_zero",
        "environment_perturbed",
        "canonical_gene_names",
        "gene_containment_bw25113_locus_tags",
        "provenance_audit",
    } <= names
    assert (synthetic / "preprocess" / "verification_report.json").exists()


# --------------------------------------------------------------------------- #
# Raw mirror and CLI
# --------------------------------------------------------------------------- #
def _synthetic_raw_file(tmp_path: Path) -> tuple[m.RawFile, Path]:
    payload = b"%PDF-1.4 synthetic"
    path = tmp_path / "download" / "s.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    url = "https://example.org/s.pdf"
    raw = m.RawFile(
        name="s.pdf",
        sha256=sha,
        bytes=len(payload),
        description="s",
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=url,
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": "s.pdf"},
            sha256=sha,
            retrieved_at="2026-10-07",
        ),
    )
    return raw, path


@pytest.fixture
def one_raw_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[m.RawFile, Path]:
    raw, path = _synthetic_raw_file(tmp_path)
    monkeypatch.setattr(m, "RAW_FILES", (raw,))
    monkeypatch.setattr(m, "DATA_SHA256", {raw.name: raw.sha256})
    monkeypatch.setattr(m, "RAW_FILES_BY_NAME", {raw.name: raw})
    monkeypatch.setattr(m, "pdftotext_version", lambda: "pdftotext version test")
    return raw, path


def test_deposit_is_idempotent_and_refuses_a_differing_file(
    tmp_path: Path, one_raw_file: tuple[m.RawFile, Path]
) -> None:
    raw, path = one_raw_file
    data_root = str(tmp_path / "root")
    root = m.deposit_raw_mirror(sources={raw.name: path}, data_root=data_root)
    assert root == Path(data_root) / m.RAW_DIR_REL
    m.deposit_raw_mirror(sources={raw.name: path}, data_root=data_root)
    manifest = m.load_manifest(data_root)
    (record,) = manifest.files
    assert (record.path, record.sha256, record.role) == (
        "data/s.pdf",
        raw.sha256,
        "raw_data",
    )
    assert record.processing is not None
    assert record.processing.tool == "pdftotext"
    assert manifest.si_expected == list(m.NOT_MIRRORED)
    assert m.manifest_sha256(manifest, "data/s.pdf") == raw.sha256
    with pytest.raises(KeyError, match="data/t.pdf is not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "data/t.pdf")

    (root / "data" / "s.pdf").write_bytes(b"edited")
    with pytest.raises(RuntimeError, match="exists with a different sha256; refusing"):
        m.deposit_raw_mirror(sources={raw.name: path}, data_root=data_root)
    path.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="s.pdf sha256 mismatch"):
        m.deposit_raw_mirror(sources={raw.name: path}, data_root=data_root)
    with pytest.raises(KeyError, match="no source given for"):
        m.deposit_raw_mirror(sources={}, data_root=data_root)


def test_download_links_the_mirror_and_checks_the_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    one_raw_file: tuple[m.RawFile, Path],
) -> None:
    raw, path = one_raw_file
    data_root = tmp_path / "root"
    m.deposit_raw_mirror(sources={raw.name: path}, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    dataset = object.__new__(m.EnvChemgenWang2015Dataset)
    monkeypatch.setattr(
        m.EnvChemgenWang2015Dataset, "raw_dir", str(tmp_path / "raw"), raising=False
    )
    dataset.download()
    linked = tmp_path / "raw" / "s.pdf"
    assert linked.is_symlink()
    assert os.readlink(linked) == str(data_root / m.RAW_DIR_REL / "data" / "s.pdf")
    assert m.library_dir(str(data_root)) == data_root / m.LIBRARY_DIR_REL

    manifest_path = data_root / m.RAW_DIR_REL / "manifest.json"
    manifest_path.write_text(manifest_path.read_text().replace(raw.sha256, "0" * 64))
    with pytest.raises(ManifestPinMismatchError, match="data/s.pdf"):
        dataset.download()
    (data_root / m.RAW_DIR_REL / "data" / "s.pdf").unlink()
    manifest_path.write_text(manifest_path.read_text().replace("0" * 64, raw.sha256))
    (tmp_path / "raw" / "s.pdf").unlink()
    with pytest.raises(RuntimeError, match="required raw artifact missing from mirror"):
        dataset.download()


def test_retrieve_runs_the_recorded_retriever_and_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    one_raw_file: tuple[m.RawFile, Path],
) -> None:
    raw, _ = one_raw_file
    calls: list[RetrievalRecord] = []
    payloads = [b"%PDF-1.4 synthetic", b"wrong"]

    def fake(record: RetrievalRecord) -> bytes:
        calls.append(record)
        return payloads[len(calls) - 1]

    monkeypatch.setattr(m, "run_retriever", fake)
    out = m.retrieve_raw_files(tmp_path / "fetched")
    assert out == {"s.pdf": tmp_path / "fetched" / "s.pdf"}
    assert [c.params for c in calls] == [{"key": "s.pdf"}]
    (tmp_path / "fetched" / "s.pdf").unlink()
    with pytest.raises(RawSha256MismatchError):
        m.retrieve_raw_files(tmp_path / "fetched")
    assert not (tmp_path / "fetched" / "s.pdf").exists()


def test_main_dispatches_the_three_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        m, "retrieve_raw_files", lambda dest: seen.setdefault("retrieved", Path(dest))
    )

    def fake_deposit(*, sources: Mapping[str, Path], data_root: str) -> Path:
        seen["sources"] = dict(sources)
        seen["data_root"] = data_root
        return tmp_path / "mirror"

    monkeypatch.setattr(m, "deposit_raw_mirror", fake_deposit)
    download = str(tmp_path / "dl")
    assert m.main(["deposit", "--download-dir", download, "--retrieve"]) == 0
    assert seen["retrieved"] == Path(download)
    assert seen["sources"] == {"si1.pdf": Path(download) / "si1.pdf"}
    assert seen["data_root"] == str(tmp_path)

    class FakeDataset:
        def __init__(self, root: str) -> None:
            seen["root"] = root

        def __len__(self) -> int:
            return 46

    monkeypatch.setattr(m, "EnvChemgenWang2015Dataset", FakeDataset)
    assert m.main(["build"]) == 0
    assert seen["root"] == osp.join(str(tmp_path), m.DATASET_ROOT_REL)

    for passed, code in ((True, 0), (False, 1)):
        report = SimpleNamespace(passed=passed, summary=lambda: "summary")
        monkeypatch.setattr(m, "run_verification", lambda root, r=report: r)
        assert m.main(["verify"]) == code
    assert "len = 46" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Real data (``--data``): the mirrors and the built dev-tree LMDB, never rebuilt
# --------------------------------------------------------------------------- #
def _real_data_root() -> str:
    return os.environ["DATA_ROOT"]


def _built_root() -> str:
    return osp.join(_real_data_root(), m.DATASET_ROOT_REL)


def _real_pdf() -> Path:
    return m.raw_mirror_dir(_real_data_root()) / "data" / m.SI_PDF


@pytest.mark.data
def test_real_mirror_matches_the_module_pins() -> None:
    from torchcell.data.experiment_dataset import file_sha256

    manifest = m.load_manifest(_real_data_root())
    for raw in m.RAW_FILES:
        assert m.manifest_sha256(manifest, raw.mirror_relpath) == raw.sha256
        path = m.raw_mirror_dir(_real_data_root()) / raw.mirror_relpath
        assert path.stat().st_size == raw.bytes
        assert file_sha256(path) == raw.sha256
    library_copy = m.library_dir(_real_data_root()) / "si" / m.SI_PDF
    assert file_sha256(library_copy) == m.SI_PDF_SHA256


@pytest.mark.data
@pytest.mark.parametrize("key", sorted(m.SOURCED_VALUES))
def test_real_sourced_values_are_verbatim(key: str) -> None:
    result = audit_sourced_value(
        m.SOURCED_VALUES[key], Path(_real_data_root()) / "torchcell-library"
    )
    assert result.passed, result.message


def _real_tables() -> tuple[list[m.TableS2Entry], list[m.TableS3Row]]:
    text = m.layout_text(_real_pdf())
    return (
        m.parse_table_s2(m.table_section(text, m.TABLE_S2_MARKER, m.TABLE_S3_MARKER)),
        m.parse_table_s3(m.table_section(text, m.TABLE_S3_MARKER, m.TABLE_S4_MARKER)),
    )


@pytest.mark.data
def test_real_extraction_matches_the_pinned_digest_and_the_papers_counts() -> None:
    entries, rows = _real_tables()
    assert m.table_s3_digest(rows) == m.TABLE_S3_SHA256
    assert (len(entries), len(rows)) == (48, 47)
    _, mutants = m.split_parent(rows)
    checks = m.reported_count_checks(mutants, m.keio_entries_for(mutants, entries))
    assert [c.measured for c in checks] == [17, 7, 11, 45]
    tokens = {e.gene_name: e.keio_token for e in entries}
    assert (tokens["acrA"], tokens["acrB"], tokens["mdtD"]) == (
        "JW0451",
        "JJW0452",
        "JW2077",
    )


#: One Table S3 row of the MinerU OCR (``si/si1.md``); the OCR writes two of the 47 deltas
#: as U+2206 (increment), which this test folds onto U+0394 before comparing.
_OCR_ROW = re.compile(
    r"<tr><td>(BW25113|BW[Δ∆]\w+)</td><td>(\d+\.\d+) ± (\d+\.\d+)</td>"
    r"<td>(\d+\.\d+) ± (\d+\.\d+)</td></tr>"
)


@pytest.mark.data
def test_real_text_layer_agrees_with_the_independent_ocr() -> None:
    """Two extractions of Table S3, the PDF text layer and MinerU's OCR of the page
    images, agree on all 47 rows and 188 numbers.
    """
    ocr_path = m.library_dir(_real_data_root()) / m.SI1_MD
    ocr = ocr_path.read_text(encoding="utf-8")
    section = ocr[ocr.index("Table S3.") : ocr.index("Table S4.")]
    from_ocr = {
        strain.replace("∆", D): tuple(float(v) for v in values)
        for strain, *values in _OCR_ROW.findall(section)
    }
    _, rows = _real_tables()
    from_pdf = {
        row.strain: (row.od600_without, row.sd_without, row.od600_with, row.sd_with)
        for row in rows
    }
    assert len(from_pdf) == 47
    assert from_ocr == from_pdf


@pytest.mark.data
def test_real_keio_list_confirms_the_five_table_s2_disagreements() -> None:
    """The Keio strain list Fuhrer 2017 publishes (Table EV1A, in its raw mirror) agrees
    with the BW25113 annotation: Table S2's number for each of the five strains is
    another gene's strain (or no strain), and the strain's own gene has another number.
    """
    import torchcell.datasets.ecoli.fuhrer2017 as fuhrer

    path = fuhrer.raw_mirror_dir(_real_data_root()) / "data" / fuhrer.TABLE_EV1
    keio = fuhrer.read_table_ev1a(path)
    gene_of = {e.jw_id: e.gene_name for e in keio}
    number_of = {e.gene_name: e.jw_id for e in keio}
    ledger = json.loads(
        Path(_built_root(), "preprocess", "identifier_reconciliation.json").read_text()
    )
    observed = {
        d["gene_name"]: (d["keio_token"], gene_of.get(d["keio_token"]))
        for d in ledger["keio_disagreements"]
    }
    assert observed == {
        "acrA": ("JW0451", "acrB"),
        "acrB": ("JJW0452", None),
        "emrK": ("JW2364", "emrY"),
        "emrY": ("JW2365", "emrK"),
        "mdtD": ("JW2077", "gatB"),
    }
    assert {g: number_of[g] for g in observed} == {
        "acrA": "JW0452",
        "acrB": "JW0451",
        "emrK": "JW2365",
        "emrY": "JW2364",
        "mdtD": "JW2062",
    }


@pytest.mark.data
def test_real_build_counts_and_the_acra_record() -> None:
    """46 records, none dropped; acrA by hand from Table S3:
    log2((6.26 / 8.13) / (4.12 / 8.30)).
    """
    from torchcell.verification.runners import load_records

    drops = json.loads(
        Path(_built_root(), "preprocess", "dropped_records.json").read_text()
    )
    assert (drops["source_records"], drops["kept_records"]) == (46, 46)
    ledger = json.loads(
        Path(_built_root(), "preprocess", "identifier_reconciliation.json").read_text()
    )
    assert ledger["n_keio_agree"] == 41
    assert ledger["names"]["layer_histogram"]["gene symbol"] == 44
    assert ledger["names"]["layer_histogram"]["gene synonym"] == 2
    records = load_records(_built_root())
    assert len(records) == 46
    by_gene = {
        r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"]: r
        for r in records
    }
    acra = by_gene["acrA"]["experiment"]
    (perturbation,) = acra["genotype"]["perturbations"]
    assert perturbation["systematic_gene_name"] == "BW25113_0463"
    assert perturbation["construction"] is None
    assert acra["phenotype"]["environment_response"] == pytest.approx(
        math.log2((6.26 / 8.13) / (4.12 / 8.30)), abs=1e-15
    )
    accessions = {
        gene: (
            r["experiment"]["genotype"]["perturbations"][0]["construction"] or {}
        ).get("strain_accession")
        for gene, r in by_gene.items()
    }
    assert sorted(g for g, a in accessions.items() if a is None) == [
        "acrA",
        "acrB",
        "emrK",
        "emrY",
        "mdtD",
    ]
    assert accessions["tolC"] == "JW5503"
    assert accessions["mdfA"] == "JW0826"
    responses = [r["experiment"]["phenotype"]["environment_response"] for r in records]
    assert sum(v < 0 for v in responses) == 25


@pytest.mark.data
def test_real_build_passes_l0_to_l4() -> None:
    report = m.run_verification(_real_data_root())
    assert report.passed, report.summary()
