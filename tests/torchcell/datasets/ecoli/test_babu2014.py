# tests/torchcell/datasets/ecoli/test_babu2014.py
# [[tests.torchcell.datasets.ecoli.test_babu2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_babu2014.py
"""The Babu 2014 eSGA interaction loader (``torchcell.datasets.ecoli.babu2014``).

Synthetic tests (run everywhere) build Table S1, Table S2 and Butland's array roster in
``tmp_path`` over the synthetic MG1655 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py``, served through a stubbed
``resolve`` with the network refused, with ``b0010`` added as a ``/gene_synonym`` of
``b0003`` so the annotation-remap case exists. ``verify_raw_files`` is replaced with a
presence check (synthetic bytes cannot carry the real pins; the pins are asserted by the
refusal test and the data-gated tests), and the four release-count constants are
monkeypatched to the synthetic release's own numbers.

The ten synthetic Table S2 rows, and what each one exercises:

    donor   recipient  score   outcome
    b0001   b0003      -5.0    kept, This Study, aggravating
    b0001   b0004      +4.0    kept, pseudogene recipient
    b0002   b0003      -6.0    dropped: essential donor, a hypomorph
    b0001   b0006      -7.0    dropped: SPA-tagged recipient, a hypomorph
    b0001   b0010      +3.5    dropped: b0010 is a synonym of b0003 (issue #753)
    b0001   b0099      -4.0    dropped: on no locus of the annotation
    b0005   b0007      +5.5    kept, Butland et al. screen set
    b0005   b0004      -3.9    dropped: contradictory duplicate
    b0005   b0004      +4.1    dropped: contradictory duplicate
    b0003   b0001      +4.5    kept, the reciprocal of row 1 with the opposite sign

Four records survive. Data-gated tests (``--data``) read the real raw mirrors and the
built dev-tree LMDB under ``$DATA_ROOT`` (they never build it): the manifest pins of both
citation keys, the provenance audit of every sourced value, the measured counts and sign
split, hand-checked cells read off ``si23.xls``, and the recorded ledgers.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pytest

import torchcell.datasets.ecoli.babu2014 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError, check_manifest_pin
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    BacterialGeneInteractionExperiment,
    BacterialGeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    MediaComponentRole,
    SampleUnit,
)
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.literature.manifest import RetrievalMethod
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.sourced import ProvenanceGapReason, audit_sourced_value

REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="Hfr Cavalli x Keio (K-12 BW25113) eSGA conjugant",
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
    background=m.chassis_background(),
)

#: ``(Bnum, Gene, Essentiality, screen set)`` of the synthetic Table S1.
SYNTHETIC_DONORS: tuple[tuple[str, str, str, str], ...] = (
    ("b0001", "thrL", "non-essential", m.SCREEN_THIS_STUDY),
    ("b0002", "thrA", "essential", m.SCREEN_THIS_STUDY),
    ("b0003", "thrW", "non-essential", m.SCREEN_THIS_STUDY),
    ("b0005", "proB", "non-essential", m.SCREEN_BUTLAND),
)
#: ``(label, gene, b-number)`` of the synthetic Butland array roster.
SYNTHETIC_ARRAY: tuple[tuple[str, str, str], ...] = (
    (m.KEIO_NON_ESSENTIAL, "thrL", "b0001"),
    (m.KEIO_NON_ESSENTIAL, "thrW", "b0003"),
    (m.KEIO_NON_ESSENTIAL, "yaaP", "b0004"),
    (m.SPA_TAG_ESSENTIAL, "proC", "b0006"),
    (m.KEIO_NON_ESSENTIAL, "insZ", "b0007"),
)
#: ``(donor bnum, donor gene, recipient bnum, recipient gene, score)`` of Table S2.
SYNTHETIC_PAIRS: tuple[tuple[str, str, str, str, float], ...] = (
    ("b0001", "thrL", "b0003", "thrW", -5.0),
    ("b0001", "thrL", "b0004", "yaaP", 4.0),
    ("b0002", "thrA", "b0003", "thrW", -6.0),
    ("b0001", "thrL", "b0006", "proC", -7.0),
    ("b0001", "thrL", "b0010", "newG", 3.5),
    ("b0001", "thrL", "b0099", "ghostG", -4.0),
    ("b0005", "proB", "b0007", "insZ", 5.5),
    ("b0005", "proB", "b0004", "yaaP", -3.9),
    ("b0005", "proB", "b0004", "yaaP", 4.1),
    ("b0003", "thrW", "b0001", "thrL", 4.5),
)
#: The four pairs the retention rules keep, as ``(donor, recipient, score, screen)``.
KEPT_PAIRS: tuple[tuple[str, str, float, str], ...] = (
    ("b0001", "b0003", -5.0, m.SCREEN_THIS_STUDY),
    ("b0001", "b0004", 4.0, m.SCREEN_THIS_STUDY),
    ("b0005", "b0007", 5.5, m.SCREEN_BUTLAND),
    ("b0003", "b0001", 4.5, m.SCREEN_THIS_STUDY),
)


def _write_sheet(
    path: Path,
    spec: m.SheetSpec,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    title: str | None = None,
    extra_sheet: bool = False,
) -> None:
    """Write one sheet with its title cell, filler rows, header and data."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = spec.sheet_name
    sheet.append([spec.title if title is None else title])
    for _ in range(spec.header_row - 1):
        sheet.append(["a note"])
    sheet.append(list(columns))
    for row in rows:
        sheet.append(list(row))
    if extra_sheet:
        workbook.create_sheet("second")
    workbook.save(path)


def _write_table_s1(path: Path) -> None:
    """Table S1 with its two read columns, a filler column and the footnote row."""
    columns = [
        "Bnum",
        "Gene",
        "JW number",
        "Essentiality",
        "Donors screened in this and previous study",
    ]
    rows: list[list[Any]] = [
        [b, gene, f"JW{b[1:]}", essentiality, screen]
        for b, gene, essentiality, screen in SYNTHETIC_DONORS
    ]
    rows.append(["*see Supplementary Methods for details", None, None, None, None])
    _write_sheet(path, m.TABLE_S1_SPEC, columns, rows)


def _write_table_s2(path: Path, *, pairs: Sequence[Any] | None = None) -> None:
    """Table S2 as the release writes it: ``<bnum>__<gene>`` ids and the score."""
    chosen = SYNTHETIC_PAIRS if pairs is None else pairs
    rows = [
        [f"{donor}__{donor_gene}", f"{recipient}__{recipient_gene}", score]
        for donor, donor_gene, recipient, recipient_gene, score in chosen
    ]
    _write_sheet(path, m.TABLE_S2_SPEC, list(m.TABLE_S2_SPEC.columns), rows)


def _write_butland_roster(path: Path) -> None:
    """Butland Supplementary Table 1, the recipient array roster."""
    _write_sheet(
        path,
        m.BUTLAND_TABLE_S1_SPEC,
        list(m.BUTLAND_TABLE_S1_SPEC.columns),
        [list(row) for row in SYNTHETIC_ARRAY],
    )


def _write_raw(raw: Path) -> None:
    """All three consumed sheets, under the names the loader links from the mirrors."""
    raw.mkdir(parents=True, exist_ok=True)
    _write_table_s1(raw / m.TABLE_S1)
    _write_table_s2(raw / m.TABLE_S2)
    _write_butland_roster(raw / m.BUTLAND_TABLE_S1)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The real MG1655 class over the synthetic assembly, b0010 a synonym of b0003."""
    loci = [
        locus.model_copy(update={"synonyms": (*locus.synonyms, "b0010")})
        if locus.tag == "b0003"
        else locus
        for locus in MG1655_LOCI
    ]
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, loci, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "mg1655"
    root.mkdir()
    return EcoliK12MG1655Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def synthetic_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the four asserted release counts at the synthetic release's numbers."""
    monkeypatch.setattr(m, "N_DONORS", len(SYNTHETIC_DONORS))
    monkeypatch.setattr(m, "N_RELEASED_PAIRS", len(SYNTHETIC_PAIRS))
    monkeypatch.setattr(m, "SIGN_SPLIT", (5, 5))
    monkeypatch.setattr(m, "N_HYPOMORPHIC_RECIPIENTS", 1)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A directory holding all three synthetic sheets."""
    raw = tmp_path / "sheets"
    _write_raw(raw)
    return raw


@pytest.fixture
def presence_only_pins(monkeypatch: pytest.MonkeyPatch) -> list[Mapping[str, str]]:
    """Replace the build-time byte check with a presence check that records the pins."""
    calls: list[Mapping[str, str]] = []

    def record(raw: str, pins: Mapping[str, str]) -> None:
        missing = [f for f in pins if not osp.exists(osp.join(raw, f))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")
        calls.append(dict(pins))

    monkeypatch.setattr(m, "verify_raw_files", record)
    return calls


@pytest.fixture
def synthetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mg1655: EcoliK12MG1655Genome,
    synthetic_counts: None,
    presence_only_pins: list[Mapping[str, str]],
) -> Path:
    """A dataset root whose ``raw/`` holds the three synthetic sheets."""
    root = tmp_path / m.DATASET_ROOT_REL
    _write_raw(root / "raw")
    monkeypatch.setattr(m, "reference_genome", lambda *a, **k: REFERENCE)
    return root


@pytest.fixture
def built(
    synthetic: Path, mg1655: EcoliK12MG1655Genome
) -> m.GeneInteractionBabu2014Dataset:
    return m.GeneInteractionBabu2014Dataset(root=str(synthetic), ecoli_genome=mg1655)


# --------------------------------------------------------------------------- #
# The pinned release
# --------------------------------------------------------------------------- #
def test_the_three_consumed_files_are_pinned_with_their_own_retrieval() -> None:
    assert [raw.name for raw in m.all_raw_files()] == [
        "si22.xls",
        "si23.xls",
        "si2.xls",
    ]
    babu = {raw.name: raw for raw in m.RAW_FILES}
    assert set(babu) == {m.TABLE_S1, m.TABLE_S2}
    for raw in m.RAW_FILES:
        assert raw.retrieval.method is RetrievalMethod.pmc_cloud
        assert raw.retrieval.source_url is not None
        assert m.PMCID in raw.retrieval.source_url
        assert raw.retrieval.sha256 == raw.sha256
    assert m.BUTLAND_RAW_FILE.retrieval.method is RetrievalMethod.springer_esm
    assert "41592_2008_BFnmeth1239_MOESM306_ESM.xls" in str(
        m.BUTLAND_RAW_FILE.retrieval.source_url
    )
    assert m.DATA_SHA256 == {
        raw.name: raw.sha256 for raw in (*m.RAW_FILES, m.BUTLAND_RAW_FILE)
    }


def test_the_deferral_file_lives_under_its_own_citation_key(tmp_path: Path) -> None:
    """Butland's roster belongs to Butland's mirror, never copied into Babu's."""
    assert m.raw_mirror_dir(str(tmp_path)).name == m.CITATION_KEY
    assert m.butland_mirror_dir(str(tmp_path)).name == m.BUTLAND_KEY
    assert m.CITATION_KEY != m.BUTLAND_KEY


def test_every_sourced_value_is_bound_to_a_pinned_artifact() -> None:
    allowed = {
        (m.CITATION_KEY, m.PAPER_MD): m.PAPER_MD_SHA256,
        (m.CITATION_KEY, m.PROTOCOL_S2_MD): m.PROTOCOL_S2_MD_SHA256,
        (m.CITATION_KEY, m.PROTOCOL_S14_MD): m.PROTOCOL_S14_MD_SHA256,
        (m.CITATION_KEY, m.PROTOCOL_S16_MD): m.PROTOCOL_S16_MD_SHA256,
        (m.BUTLAND_KEY, m.BUTLAND_PAPER_MD): m.BUTLAND_PAPER_MD_SHA256,
        (m.BUTLAND_KEY, m.BUTLAND_METHODS_MD): m.BUTLAND_METHODS_MD_SHA256,
    }
    for name, value in m.SOURCED_VALUES.items():
        key = (str(value.provenance.citation_key), value.provenance.source_uri)
        assert key in allowed, name
        assert value.provenance.sha256 == allowed[key], name
        assert value.quote.strip()


def test_the_score_definition_follows_the_deferral_to_butland() -> None:
    """Protocol S2 defers the S score to Butland 2008, and that is where it is quoted."""
    assert m.SOURCED_VALUES["score_deferral"].value == m.BUTLAND_KEY
    assert m.SOURCED_VALUES["score_deferral"].provenance.citation_key == m.CITATION_KEY
    assert m.SOURCED_VALUES["score_definition"].provenance.citation_key == m.BUTLAND_KEY
    assert "Collins 2006" in str(m.SOURCED_VALUES["score_definition"].note)


def test_the_eight_colony_design_is_recorded_beside_the_build_and_on_the_record() -> (
    None
):
    """The ledger carries the design WITH its provenance; #793 put it on the record too.

    Both halves matter: the graph property is what a query can read, and this file is
    what makes it auditable (the verbatim Protocol S2 quote, the citation key, the
    source path and its sha256), which a float property cannot carry.
    """
    structure = m.replicate_structure()
    assert (structure.n_samples, structure.sample_unit) == (8, "colony")
    assert structure.replicate_screens * structure.colonies_per_screen == 8
    assert structure.citation_key == m.CITATION_KEY
    assert structure.source_uri == m.PROTOCOL_S2_MD
    assert "eight replicate measurements" in structure.quote
    assert "n_samples" in GeneInteractionPhenotype.model_fields
    assert m.phenotype(-1.0, m.SCREEN_THIS_STUDY).n_samples == structure.n_samples


def test_the_released_counts_are_the_sourced_ones() -> None:
    assert m.N_DONORS == 163
    assert m.N_RELEASED_PAIRS == 42705
    assert m.SIGN_SPLIT == (25239, 17466)
    assert m.N_HYPOMORPHIC_RECIPIENTS == 149
    assert m.EXPECTED_RECORDS == 38579


# --------------------------------------------------------------------------- #
# Reading the sheets
# --------------------------------------------------------------------------- #
def test_table_s2_splits_the_ids_and_checks_the_sign_split(
    raw_dir: Path, synthetic_counts: None
) -> None:
    frame = m.read_table_s2(raw_dir / m.TABLE_S2)
    assert list(frame.columns) == [
        "donor_id",
        "donor_gene",
        "recipient_id",
        "recipient_gene",
        "score",
    ]
    assert len(frame) == len(SYNTHETIC_PAIRS)
    assert frame.iloc[0].tolist() == ["b0001", "thrL", "b0003", "thrW", -5.0]


def test_table_s2_refuses_a_sign_split_the_paper_does_not_state(
    tmp_path: Path, synthetic_counts: None
) -> None:
    path = tmp_path / m.TABLE_S2
    _write_table_s2(path, pairs=SYNTHETIC_PAIRS[:-1])
    with pytest.raises(m.ReleaseContentError, match="rows with sign split"):
        m.read_table_s2(path)


def test_table_s2_refuses_a_shifted_title(tmp_path: Path) -> None:
    path = tmp_path / m.TABLE_S2
    _write_sheet(
        path,
        m.TABLE_S2_SPEC,
        list(m.TABLE_S2_SPEC.columns),
        [["b0001__thrL", "b0003__thrW", -5.0]],
        title="Supplementary Table S2 | something else",
    )
    with pytest.raises(m.SheetFormatError, match="title"):
        m.read_table_s2(path)


def test_donors_drop_the_footnote_row_and_carry_both_read_columns(
    raw_dir: Path, synthetic_counts: None
) -> None:
    donors = m.read_donors(raw_dir / m.TABLE_S1)
    assert set(donors) == {b for b, _, _, _ in SYNTHETIC_DONORS}
    assert donors["b0002"].is_hypomorph is True
    assert donors["b0001"].is_hypomorph is False
    assert donors["b0005"].screen_id == m.SCREEN_BUTLAND
    assert donors["b0001"].screen_id == m.SCREEN_THIS_STUDY


def test_donors_refuse_a_count_the_paper_does_not_state(raw_dir: Path) -> None:
    with pytest.raises(m.ReleaseContentError, match="donors, the paper states"):
        m.read_donors(raw_dir / m.TABLE_S1)


def test_the_hypomorph_list_comes_from_butlands_roster(
    raw_dir: Path, synthetic_counts: None
) -> None:
    spa = m.read_hypomorphic_recipients(raw_dir / m.BUTLAND_TABLE_S1)
    assert spa == frozenset({"b0006"})


def test_the_hypomorph_list_refuses_a_roster_of_another_size(raw_dir: Path) -> None:
    with pytest.raises(m.ReleaseContentError, match="SPA-tag essential genes"):
        m.read_hypomorphic_recipients(raw_dir / m.BUTLAND_TABLE_S1)


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
def test_the_genotype_is_two_leaves_distinguished_by_cassette() -> None:
    genotype = m.pair_genotype("b0002", "thrA", "b0001", "thrL")
    assert genotype.systematic_gene_names == ["b0001", "b0002"]
    by_tag = {
        p.systematic_gene_name: p
        for p in genotype.perturbations
        if isinstance(p, BacterialDeletionPerturbation)
    }
    assert len(by_tag) == 2
    assert by_tag["b0002"].cassette == m.DONOR_CASSETTE
    assert by_tag["b0002"].collection == m.DONOR_COLLECTION
    assert by_tag["b0001"].cassette == m.RECIPIENT_CASSETTE
    assert by_tag["b0001"].collection == m.RECIPIENT_COLLECTION
    assert {p.gene_namespace for p in by_tag.values()} == {m.MG1655_NAMESPACE}
    assert {p.identifier_mapping for p in by_tag.values()} == {None}


def test_a_reciprocal_pair_is_two_distinct_genotypes() -> None:
    """Swapping donor and recipient swaps the markers, so the strains differ."""
    forward = m.pair_genotype("b0001", "thrL", "b0003", "thrW")
    reverse = m.pair_genotype("b0003", "thrW", "b0001", "thrL")
    assert forward.systematic_gene_names == reverse.systematic_gene_names
    assert forward != reverse


def test_the_phenotype_is_a_signed_score_with_a_gapped_p_value() -> None:
    phenotype = m.phenotype(-4.43348, m.SCREEN_THIS_STUDY)
    assert phenotype.gene_interaction == -4.43348
    assert phenotype.gene_interaction_p_value is None
    assert phenotype.screen_id == m.SCREEN_THIS_STUDY
    assert phenotype.gapped_fields() == {"gene_interaction_p_value"}
    assert (
        phenotype.provenance_gaps[0].reason
        is ProvenanceGapReason.not_reported_by_primary
    )
    assert m.reference_phenotype(m.SCREEN_BUTLAND).gene_interaction == 0.0


def test_the_phenotype_carries_protocol_s2s_eight_colonies(tmp_path: Path) -> None:
    """#793: the sourced replicate design is ON the record, and it is the quote's own
    number rather than a literal retyped in the phenotype builder.
    """
    phenotype = m.phenotype(-4.43348, m.SCREEN_THIS_STUDY)
    assert phenotype.n_samples == 8
    assert phenotype.sample_unit is SampleUnit.colony
    assert phenotype.n_samples == int(m.SOURCED_VALUES["n_samples"].value)
    # no per-pair dispersion is released, which is a different absence from the p-value
    assert phenotype.gene_interaction_uncertainty is None
    assert phenotype.gene_interaction_uncertainty_type is None
    assert phenotype.gapped_fields() == {"gene_interaction_p_value"}
    # the reference is 0 by construction, not a measured cell, so it carries no design
    reference = m.reference_phenotype(m.SCREEN_THIS_STUDY)
    assert reference.n_samples is None
    assert reference.sample_unit is None
    # the json beside the build keeps the same two numbers with their provenance
    structure = m.replicate_structure()
    assert (structure.n_samples, structure.sample_unit) == (8, "colony")
    assert structure.replicate_screens * structure.colonies_per_screen == 8


def test_the_chassis_background_pins_mg1655_and_names_both_parents() -> None:
    background = m.chassis_background()
    assert background.reference_strain == "MG1655"
    assert background.assembly_set == m.MG1655_ASSEMBLY_SET
    assert background.parents == ["Hfr Cavalli", "Keio collection (K-12 BW25113)"]
    assert background.alleles == []
    assert background.genotype_statement is None
    assert background.gapped_fields() == {"genotype_statement"}
    assert background.is_fully_sourced is False


def test_the_medium_is_lb_with_both_drugs_and_no_asserted_amount() -> None:
    medium = m.SELECTION_MEDIUM
    assert medium.state == "solid"
    assert medium.base_medium in MEDIA_LIBRARY
    roles = [component.role for component in medium.components]
    assert roles.count(MediaComponentRole.selection_agent) == 2
    assert MediaComponentRole.gelling_agent in roles
    assert all(component.concentration is None for component in medium.components)
    names = {component.compound.name for component in medium.components}
    assert {"kanamycin", "chloramphenicol", "agar"} <= names


def test_the_environment_is_one_condition() -> None:
    environment = m.SCREEN_ENVIRONMENT
    assert environment.perturbations == []
    assert environment.temperature is not None
    assert environment.temperature.value == 32.0
    assert environment.duration_hours == 36.0
    assert environment.aerobicity == "aerobic"


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
def test_the_dataset_is_registered_under_its_class_name() -> None:
    assert dataset_registry["GeneInteractionBabu2014Dataset"] is (
        m.GeneInteractionBabu2014Dataset
    )
    assert m.GeneInteractionBabu2014Dataset.REFERENCE_STRAIN == m.REFERENCE_STRAIN_NAME


def test_the_build_keeps_only_the_four_typable_pairs(
    built: m.GeneInteractionBabu2014Dataset,
) -> None:
    assert len(built) == len(KEPT_PAIRS)
    items = [built[i] for i in range(len(built))]
    observed: set[tuple[str, str, float, str]] = set()
    for item in items:
        experiment = BacterialGeneInteractionExperiment.model_validate(
            item["experiment"]
        )
        BacterialGeneInteractionExperimentReference.model_validate(item["reference"])
        genotype = experiment.genotype
        assert isinstance(genotype, Genotype)
        by_cassette = {
            p.cassette: p.systematic_gene_name
            for p in genotype.perturbations
            if isinstance(p, BacterialDeletionPerturbation)
        }
        observed.add(
            (
                str(by_cassette[m.DONOR_CASSETTE]),
                str(by_cassette[m.RECIPIENT_CASSETTE]),
                experiment.phenotype.gene_interaction,
                str(experiment.phenotype.screen_id),
            )
        )
    assert observed == set(KEPT_PAIRS)
    built.close_lmdb()


def test_the_build_ledgers_account_for_every_dropped_row(
    built: m.GeneInteractionBabu2014Dataset,
) -> None:
    built.close_lmdb()
    preprocess = Path(built.preprocess_dir)
    ledger = json.loads((preprocess / "dropped_records.json").read_text())
    assert ledger["source_records"] == len(SYNTHETIC_PAIRS)
    assert ledger["kept_records"] == len(KEPT_PAIRS)
    assert ledger["dropped_records"] == len(SYNTHETIC_PAIRS) - len(KEPT_PAIRS)
    counts = {rule["rule"]: rule["n_records"] for rule in ledger["rules"]}
    assert counts == {
        m.RULE_HYPOMORPH: 2,
        m.RULE_NOT_A_TAG: 1,
        m.RULE_REMAPPED: 1,
        m.RULE_DUPLICATE: 2,
    }
    items = {rule["rule"]: rule["items"] for rule in ledger["rules"]}
    assert items[m.RULE_HYPOMORPH] == ["b0002", "b0006"]
    assert items[m.RULE_NOT_A_TAG] == ["b0099"]
    assert items[m.RULE_REMAPPED] == ["b0010 -> b0003"]
    assert ledger["n_aggravating"] == 1
    assert ledger["n_alleviating"] == 3
    assert ledger["screen_id_counts"] == {m.SCREEN_THIS_STUDY: 3, m.SCREEN_BUTLAND: 1}


def test_the_build_records_the_reciprocal_pair_and_its_sign_disagreement(
    built: m.GeneInteractionBabu2014Dataset,
) -> None:
    built.close_lmdb()
    reciprocal = json.loads(
        (Path(built.preprocess_dir) / "reciprocal_pairs.json").read_text()
    )
    assert reciprocal["n_pairs"] == 1
    assert reciprocal["n_records"] == 2
    assert reciprocal["n_sign_disagreements"] == 1
    assert reciprocal["pairs"] == ["b0001 <-> b0003"]


def test_the_build_writes_the_identifier_and_not_loaded_ledgers(
    built: m.GeneInteractionBabu2014Dataset,
) -> None:
    built.close_lmdb()
    preprocess = Path(built.preprocess_dir)
    identifiers = json.loads(
        (preprocess / "identifier_reconciliation.json").read_text()
    )
    assert identifiers["assembly_set"] == "ecoli_K12_MG1655_ASM584v2"
    assert identifiers["gene_namespace"] == m.MG1655_NAMESPACE
    assert identifiers["unique_names"] == 9
    not_loaded = json.loads((preprocess / "files_not_loaded.json").read_text())
    assert not_loaded == list(m.NOT_LOADED)
    assert (preprocess / "replicate_structure.json").exists()
    assert (preprocess / "build_manifest.json").exists()


def test_the_build_refuses_a_pair_whose_donor_is_not_in_table_s1(
    tmp_path: Path,
    mg1655: EcoliK12MG1655Genome,
    synthetic_counts: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "N_RELEASED_PAIRS", 1)
    monkeypatch.setattr(m, "SIGN_SPLIT", (1, 0))
    path = tmp_path / m.TABLE_S2
    _write_table_s2(path, pairs=(("b0006", "proC", "b0003", "thrW", -5.0),))
    frame = m.read_table_s2(path)
    donors = {
        b: m.DonorSpec(
            b_number=b, gene=gene, is_hypomorph=False, screen_id=m.SCREEN_THIS_STUDY
        )
        for b, gene, _, _ in SYNTHETIC_DONORS
    }
    with pytest.raises(m.ReleaseContentError, match="are not in Table S1"):
        m.retain(
            frame,
            dataset_name="t",
            genome=mg1655,
            donors=donors,
            hypomorphic_recipients=frozenset(),
        )


# --------------------------------------------------------------------------- #
# Verification over the synthetic build
# --------------------------------------------------------------------------- #
def test_the_l0_to_l4_gate_passes_on_the_synthetic_build(
    built: m.GeneInteractionBabu2014Dataset, mg1655: EcoliK12MG1655Genome
) -> None:
    built.close_lmdb()
    from torchcell.verification.runners import load_records

    records = load_records(built.root)
    released = m.released_scores(osp.join(built.root, "raw", m.TABLE_S2))
    report = m.verify_records(
        records,
        released=released,
        universe=set(mg1655.genbank.loci),
        expected_count=len(KEPT_PAIRS),
    )
    failures = [result.name for result in report.results if not result.passed]
    assert failures == []
    assert report.passed


def test_released_scores_excludes_the_contradictory_duplicate(
    raw_dir: Path, synthetic_counts: None
) -> None:
    released = m.released_scores(raw_dir / m.TABLE_S2)
    assert ("b0005", "b0004") not in released
    assert released[("b0001", "b0003")] == -5.0


# --------------------------------------------------------------------------- #
# Data-gated: the real mirrors and the built dev-tree LMDB
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    root = os.environ.get("DATA_ROOT")
    if root is None or not osp.isdir(osp.join(root, "torchcell-raw")):
        pytest.skip("no DATA_ROOT with a raw mirror")
    return root


@pytest.mark.data
def test_both_raw_mirrors_hold_exactly_the_consumed_files() -> None:
    root = _data_root()
    for key, files in (
        (m.CITATION_KEY, m.RAW_FILES),
        (m.BUTLAND_KEY, (m.BUTLAND_RAW_FILE,)),
    ):
        manifest = m.load_manifest_of(key, root)
        assert manifest.citation_key == key
        assert [record.path for record in manifest.files] == [
            raw.mirror_relpath for raw in files
        ]
        for raw in files:
            assert m.manifest_sha256(manifest, raw.mirror_relpath) == raw.sha256


@pytest.mark.data
def test_a_manifest_pin_mismatch_is_refused() -> None:
    root = _data_root()
    manifest = m.load_manifest_of(m.CITATION_KEY, root)
    with pytest.raises(ManifestPinMismatchError):
        check_manifest_pin(
            m.RAW_FILES[0].mirror_relpath,
            m.manifest_sha256(manifest, m.RAW_FILES[0].mirror_relpath),
            "0" * 64,
        )


@pytest.mark.data
def test_every_sourced_value_audits_against_the_literature_mirror() -> None:
    root = _data_root()
    library = Path(root) / "torchcell-library"
    if not library.is_dir():
        pytest.skip("no literature mirror")
    for name, value in m.SOURCED_VALUES.items():
        result = audit_sourced_value(value, library)
        assert result.passed, f"{name}: {result.message}"


@pytest.mark.data
def test_the_real_table_s2_is_the_released_screen() -> None:
    root = _data_root()
    path = m.raw_mirror_dir(root) / f"data/{m.TABLE_S2}"
    if not path.exists():
        pytest.skip("raw mirror not deposited")
    frame = m.read_table_s2(path)
    assert len(frame) == m.N_RELEASED_PAIRS
    assert frame["donor_id"].nunique() == m.N_DONORS
    assert frame["recipient_id"].nunique() == 3876
    assert (int((frame.score < 0).sum()), int((frame.score > 0).sum())) == m.SIGN_SPLIT
    scores = {
        (row.donor_id, row.recipient_id): row.score
        for row in frame.itertuples(index=False)
    }
    assert scores[("b0002", "b0015")] == -4.43348
    assert scores[("b0002", "b0146")] == 4.26527
    assert scores[("b4485", "b4375")] == -3.91471
    assert frame.score.min() == -30.924
    assert frame.score.max() == 22.6433


@pytest.mark.data
def test_the_real_butland_roster_holds_the_149_hypomorphs() -> None:
    root = _data_root()
    path = m.butland_mirror_dir(root) / f"data/{m.BUTLAND_TABLE_S1}"
    if not path.exists():
        pytest.skip("Butland raw mirror not deposited")
    spa = m.read_hypomorphic_recipients(path)
    assert len(spa) == 149
    assert {"b3988", "b4043", "b4052"} <= spa


@pytest.mark.data
def test_the_dev_store_ledgers_state_the_measured_build() -> None:
    root = _data_root()
    preprocess = Path(root) / m.DATASET_ROOT_REL / "preprocess"
    if not (preprocess / "dropped_records.json").exists():
        pytest.skip("dev store not built")
    ledger = json.loads((preprocess / "dropped_records.json").read_text())
    assert ledger["source_records"] == m.N_RELEASED_PAIRS
    assert ledger["kept_records"] == m.EXPECTED_RECORDS
    assert {rule["rule"]: rule["n_records"] for rule in ledger["rules"]} == {
        m.RULE_HYPOMORPH: 3420,
        m.RULE_NOT_A_TAG: 183,
        m.RULE_REMAPPED: 519,
        m.RULE_DUPLICATE: 4,
    }
    assert (ledger["n_aggravating"], ledger["n_alleviating"]) == (22732, 15847)
    assert ledger["kept_donors"] == 155
    assert ledger["kept_recipients"] == 3658
    assert ledger["screen_id_counts"] == {
        m.SCREEN_THIS_STUDY: 37852,
        m.SCREEN_BUTLAND: 727,
    }
    report = json.loads((preprocess / "verification_report.json").read_text())
    assert all(result["passed"] for result in report["results"])


@pytest.mark.data
def test_the_dev_store_records_the_reciprocal_census() -> None:
    root = _data_root()
    path = Path(root) / m.DATASET_ROOT_REL / "preprocess" / "reciprocal_pairs.json"
    if not path.exists():
        pytest.skip("dev store not built")
    reciprocal = json.loads(path.read_text())
    assert reciprocal["n_pairs"] == 97
    assert reciprocal["n_records"] == 194
    assert reciprocal["n_sign_disagreements"] == 24


@pytest.mark.data
def test_the_real_donor_catalog_splits_the_two_screen_sets() -> None:
    root = _data_root()
    path = m.raw_mirror_dir(root) / f"data/{m.TABLE_S1}"
    if not path.exists():
        pytest.skip("raw mirror not deposited")
    donors = m.read_donors(path)
    assert len(donors) == 163
    screens = pd.Series([donor.screen_id for donor in donors.values()]).value_counts()
    assert screens[m.SCREEN_THIS_STUDY] == 124
    assert screens[m.SCREEN_BUTLAND] == 39
    hypomorphs = sorted(b for b, donor in donors.items() if donor.is_hypomorph)
    assert hypomorphs == ["b0156", "b0635", "b3021", "b3064", "b3146", "b3189", "b3461"]


# --------------------------------------------------------------------------- #
# The two raw mirrors
# --------------------------------------------------------------------------- #
def test_deposit_writes_both_manifests_each_under_its_own_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources: dict[str, str | Path] = {}
    staged = tmp_path / "src"
    staged.mkdir()
    for raw in m.all_raw_files():
        path = staged / raw.name
        path.write_bytes(raw.name.encode())
        sources[raw.name] = path
    monkeypatch.setattr(m, "_sha256", lambda path: m.DATA_SHA256[Path(path).name])
    root = str(tmp_path / "root")
    babu_root, butland_root = m.deposit_raw_mirror(sources=sources, data_root=root)
    babu = m.load_manifest_of(m.CITATION_KEY, root)
    butland = m.load_manifest_of(m.BUTLAND_KEY, root)
    assert babu.citation_key == m.CITATION_KEY
    assert babu.doi == m.PAPER_DOI
    assert [record.path for record in babu.files] == [
        raw.mirror_relpath for raw in m.RAW_FILES
    ]
    assert list(babu.si_expected) == list(m.NOT_LOADED)
    assert butland.citation_key == m.BUTLAND_KEY
    assert butland.doi == m.BUTLAND_DOI
    assert [record.path for record in butland.files] == [
        m.BUTLAND_RAW_FILE.mirror_relpath
    ]
    assert (babu_root / m.RAW_FILES[0].mirror_relpath).exists()
    assert (butland_root / m.BUTLAND_RAW_FILE.mirror_relpath).exists()
    assert m.manifest_sha256(babu, f"data/{m.TABLE_S2}") == m.DATA_SHA256[m.TABLE_S2]
    # Idempotent: the same bytes deposit again without raising.
    m.deposit_raw_mirror(sources=sources, data_root=root)


def test_deposit_refuses_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match=m.BUTLAND_TABLE_S1):
        m.deposit_raw_mirror(sources={}, data_root=str(tmp_path))


def test_deposit_refuses_a_hash_mismatch(tmp_path: Path) -> None:
    sources: dict[str, str | Path] = {}
    staged = tmp_path / "src"
    staged.mkdir()
    for raw in m.all_raw_files():
        path = staged / raw.name
        path.write_bytes(b"not the released bytes")
        sources[raw.name] = path
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        m.deposit_raw_mirror(sources=sources, data_root=str(tmp_path / "root"))


def test_deposit_refuses_to_overwrite_a_mirror_file_of_another_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    existing = root / m.RAW_DIR_REL / m.RAW_FILES[0].mirror_relpath
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"an older release")
    sources: dict[str, str | Path] = {}
    staged = tmp_path / "src"
    staged.mkdir()
    for raw in m.all_raw_files():
        path = staged / raw.name
        path.write_bytes(raw.name.encode())
        sources[raw.name] = path
    hashes = dict(m.DATA_SHA256)

    def fake_sha256(path: str | Path) -> str:
        name = Path(path).name
        if Path(path) == existing:
            return "0" * 64
        return hashes[name]

    monkeypatch.setattr(m, "_sha256", fake_sha256)
    with pytest.raises(RuntimeError, match="different sha256"):
        m.deposit_raw_mirror(sources=sources, data_root=str(root))


def test_manifest_sha256_refuses_an_unknown_path() -> None:
    from torchcell.literature.manifest import Manifest

    manifest = Manifest(citation_key=m.CITATION_KEY, doi=m.PAPER_DOI, title="t")
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, f"data/{m.TABLE_S2}")


def test_retrieve_raw_files_runs_the_recorded_retriever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: list[str] = []

    def fake(record: Any) -> bytes:
        served.append(str(record.source_url))
        return b"bytes"

    monkeypatch.setattr(m, "run_retriever", fake)
    monkeypatch.setattr(m, "write_verified", lambda data, path, sha, url: None)
    out = m.retrieve_raw_files(tmp_path, names=[m.BUTLAND_TABLE_S1])
    assert list(out) == [m.BUTLAND_TABLE_S1]
    assert served == [str(m.BUTLAND_RAW_FILE.retrieval.source_url)]


def test_the_mirror_directories_hang_off_data_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_ROOT", "/tmp/root")
    assert m.raw_mirror_dir() == Path(f"/tmp/root/torchcell-raw/{m.CITATION_KEY}")
    assert m.butland_mirror_dir() == Path(f"/tmp/root/torchcell-raw/{m.BUTLAND_KEY}")
    assert m.library_dir(m.BUTLAND_KEY) == Path(
        f"/tmp/root/torchcell-library/{m.BUTLAND_KEY}"
    )


# --------------------------------------------------------------------------- #
# download() and the CLI
# --------------------------------------------------------------------------- #
@pytest.fixture
def tmp_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``DATA_ROOT`` whose two raw mirrors hold the synthetic sheets, pinned to the
    real sha256 so ``download`` and the manifest checks run unmodified.
    """
    data_root = tmp_path / "root"
    staged = tmp_path / "staged"
    _write_raw(staged)
    sources: dict[str, str | Path] = {
        raw.name: staged / raw.name for raw in m.all_raw_files()
    }
    monkeypatch.setattr(m, "_sha256", lambda path: m.DATA_SHA256[Path(path).name])
    m.deposit_raw_mirror(sources=sources, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    return data_root


class _Downloader(m.GeneInteractionBabu2014Dataset):
    """``download()`` alone, with a raw directory the test chooses.

    ``raw_dir`` is a read-only property on the base class, so the only way to exercise
    the two-mirror link step without a full build is to override it.
    """

    def __init__(self, raw: Path) -> None:
        self._raw = raw

    @property
    def raw_dir(self) -> str:
        return str(self._raw)


def test_download_links_both_mirrors_into_raw(
    tmp_mirror: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        m,
        "link_verified",
        lambda src, dest, sha: Path(dest).write_bytes(Path(src).read_bytes()),
    )
    dataset = _Downloader(tmp_path / "linked")
    dataset.download()
    assert sorted(p.name for p in Path(dataset.raw_dir).iterdir()) == sorted(
        m.DATA_SHA256
    )


def test_download_refuses_a_file_missing_from_the_deferral_mirror(
    tmp_mirror: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "link_verified", lambda src, dest, sha: None)
    (tmp_mirror / m.BUTLAND_RAW_DIR_REL / m.BUTLAND_RAW_FILE.mirror_relpath).unlink()
    dataset = _Downloader(tmp_path / "linked")
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        dataset.download()
    assert not Path(dataset.raw_dir).exists()


def test_download_refuses_a_manifest_pin_that_drifts(
    tmp_mirror: Path, tmp_path: Path
) -> None:
    path = tmp_mirror / m.RAW_DIR_REL / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    dataset = _Downloader(tmp_path / "linked")
    with pytest.raises(ManifestPinMismatchError):
        dataset.download()


def test_the_genome_injection_refuses_another_assembly(
    synthetic: Path, mg1655: EcoliK12MG1655Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = m.GeneInteractionBabu2014Dataset(root=str(synthetic), ecoli_genome=mg1655)
    dataset.close_lmdb()
    object.__setattr__(dataset, "ecoli_genome", mg1655)
    monkeypatch.setattr(type(mg1655), "ASSEMBLY_SET", "ecoli_K12_BW25113_ASM75055v1")
    with pytest.raises(ValueError, match="needs the ecoli_K12_MG1655_ASM584v2 genome"):
        dataset._genome()


def test_the_dataset_declares_its_schema_classes_and_raw_files(
    synthetic: Path, mg1655: EcoliK12MG1655Genome
) -> None:
    dataset = m.GeneInteractionBabu2014Dataset(root=str(synthetic), ecoli_genome=mg1655)
    assert dataset.experiment_class is BacterialGeneInteractionExperiment
    assert dataset.reference_class is BacterialGeneInteractionExperimentReference
    assert dataset.raw_file_names == [raw.name for raw in m.all_raw_files()]
    frame = pd.DataFrame({"a": [1]})
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()
    dataset.close_lmdb()


def test_cli_deposits_from_the_library_or_a_rerun_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = tmp_path / "root"
    for key in (m.CITATION_KEY, m.BUTLAND_KEY):
        si = data_root / "torchcell-library" / key / "si"
        si.mkdir(parents=True)
        _write_raw(si)
    monkeypatch.setattr(m, "_sha256", lambda path: m.DATA_SHA256[Path(path).name])
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    assert m.main(["deposit"]) == 0
    printed = capsys.readouterr().out.split()
    assert printed == [
        str(data_root / m.RAW_DIR_REL),
        str(data_root / m.BUTLAND_RAW_DIR_REL),
    ]

    retrieved: list[str] = []

    def fake_retrieve(dest: str) -> dict[str, Path]:
        retrieved.append(dest)
        return {
            raw.name: data_root / "torchcell-library" / m.CITATION_KEY / "si" / raw.name
            for raw in m.all_raw_files()
        }

    monkeypatch.setattr(m, "retrieve_raw_files", fake_retrieve)
    assert m.main(["deposit", "--retrieve-into", str(tmp_path / "fetched")]) == 0
    assert retrieved == [str(tmp_path / "fetched")]


def test_cli_build_and_verify(
    built: m.GeneInteractionBabu2014Dataset,
    mg1655: EcoliK12MG1655Genome,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``build`` re-loads the store the ``built`` fixture wrote; ``verify`` runs the
    gate over it with the synthetic genome and no library audit.
    """
    built.close_lmdb()
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert m.main(["build"]) == 0
    assert capsys.readouterr().out.strip().endswith(f"len = {len(KEPT_PAIRS)}")

    monkeypatch.setattr(m, "SOURCED_VALUES", {})
    monkeypatch.setattr(
        m, "_gene_universe", lambda records, base: set(mg1655.genbank.loci)
    )
    assert m.main(["verify"]) == 0
    assert "gene_interaction_babu2014: PASS" in capsys.readouterr().out
