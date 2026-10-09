# tests/torchcell/datasets/ecoli/test_niu2019.py
# [[tests.torchcell.datasets.ecoli.test_niu2019]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_niu2019.py
"""Niu 2019 loader: the SI docx reader, the guide spacers, retention, the records.

The synthetic tests run everywhere. They write a four-table ``.docx`` the way the
loader reads one and resolve its gene labels against the REAL
``EcoliK12BW25113Genome`` over the synthetic BW25113 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py``, with the network refused.
Derived expectations for that assembly and that fixture: ``thrA``, ``hokC``, ``yaaX``,
``proB`` and ``thrL`` resolve, ``nosuchgene`` resolves to nothing, and ``sufBCDS`` is
the operon label the real release also carries. Of the fixture's 13 released numeric
cells, 4 become records (one activation target, two interference targets, and the
two-gene interference combination strain) and 9 are accounted for by the four rules.

The data tests (``--data``) audit every module-level ``SourcedValue`` against the
sha256-pinned paper OCR, pin the raw mirror's recorded digest, and read the real store:
51 records, L0 to L4 PASS.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

import pytest

import torchcell.datasets.ecoli.niu2019 as n
from tests.torchcell.datasets._genome_injection_fakes import (
    FakeBW25113Genome,
    install_bacterial_fakes,
)
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.schema import (
    ASSEMBLY_SET_ACCESSIONS,
    BACTERIAL_ASSEMBLY_SETS,
    AssayType,
    AssemblyReferenceGenome,
    BacterialCrisprActivationPerturbation,
    BacterialCrisprInterferencePerturbation,
    ConcentrationUnit,
    MeasurementType,
    SampleUnit,
    SmallMoleculePerturbation,
)
from torchcell.datasets.bacteria_common import BacterialGenomeInjector
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.sequence.genome.ecoli.k12 import (
    BW25113_ASSEMBLY,
    EcoliK12BW25113Genome,
    EcoliK12StrainName,
)
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

DATA_ROOT = os.environ.get("DATA_ROOT", "")
_MIRROR = Path(DATA_ROOT, n.RAW_DIR_REL) if DATA_ROOT else None
_LIBRARY = Path(DATA_ROOT, n.LIBRARY_DIR_REL) if DATA_ROOT else None
_STORE = Path(DATA_ROOT, n.DATASET_ROOT_REL) if DATA_ROOT else None

requires_mirror = pytest.mark.skipif(
    _MIRROR is None or not (_MIRROR / n.si_rel()).exists(),
    reason="the Niu 2019 raw mirror is not deposited under $DATA_ROOT",
)
requires_library = pytest.mark.skipif(
    _LIBRARY is None or not (_LIBRARY / n.PAPER_MD).exists(),
    reason="the Niu 2019 OCR mirror is not present under $DATA_ROOT",
)
requires_store = pytest.mark.skipif(
    _STORE is None or not (_STORE / "processed" / "lmdb").exists(),
    reason="the Niu 2019 dev-tree LMDB is not built",
)


# --------------------------------------------------------------------------- #
# A synthetic SI .docx, written the way the loader reads one
# --------------------------------------------------------------------------- #
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _spacer(base: str, length: int) -> str:
    """One synthetic guide spacer, as the release flanks it."""
    return n.OLIGO_PREFIX + base * length + n.OLIGO_SUFFIX.lower()


#: Suppl. Table 1 as the fixture releases it: three guide oligos over two genes, one of
#: them twice with DIFFERENT spacers (the release's own ``marA`` / ``sspA`` shape), one
#: 22-nt spacer (its own ``N20-prpR`` shape), and one non-N20 cloning primer.
TABLE_1_ROWS: tuple[tuple[str, ...], ...] = (
    n.TABLE_1_HEADER,
    ("N20-thrA", _spacer("A", 20), "CRISPRa"),
    ("N20-yaaX", _spacer("C", 22), "CRISPRi"),
    ("N20-proB", _spacer("G", 20), "CRISPRi"),
    ("N20-proB", _spacer("T", 20), "CRISPRi"),
    ("thrA-F", "ATGCATGCATGC", "amplification of thrA"),
)
#: Suppl. Table 2 is never consumed, so the fixture carries a one-row stub of its shape.
TABLE_2_ROWS: tuple[tuple[str, ...], ...] = (
    ("Gene", "Protein", "Mutation site", "Frequence"),
    ("Carbohydrate metabolism",),
    ("b0002 (thrA)", "aspartokinase I", "337 C>T (P113S)", "1.00"),
)
#: Suppl. Table 3: one stored target, one growth-censored row, the operon label and an
#: unresolvable label. A category heading is a merged row carrying ONE filled cell.
TABLE_3_ROWS: tuple[tuple[str, ...], ...] = (
    n.TARGET_TABLE_HEADER,
    ("Carbohydrate metabolism", ""),
    ("thrA", "aspartokinase I", "1.24 ± 0.01", "1.06 ± 0.02"),
    ("hokC", "toxin HokC", "-", "1.19 ± 0.01"),
    ("Stress response", ""),
    ("sufBCDS", "SufBCD Fe-S cluster assembly protein", "1.48 ± 0.02", "1.32 ± 0.01"),
    ("nosuchgene", "a protein of no locus", "1.10 ± 0.01", "-"),
)
#: Suppl. Table 4: two stored targets (one of them with two released spacers, so its
#: guide is None) and one fully censored row.
TABLE_4_ROWS: tuple[tuple[str, ...], ...] = (
    n.TARGET_TABLE_HEADER,
    ("Lipid metabolism", ""),
    ("yaaX", "protein YaaX", "1.07 ± 0.01", "1.06 ± 0.02"),
    ("proB", "glutamate 5-kinase", "1.09 ± 0.01", "-"),
    ("thrL", "thr operon leader peptide", "-", "-"),
)
#: The fixture's combination strains: the activation one carries the operon label and is
#: dropped whole, the interference one is two single genes and is stored.
COMBINATION_LABELS = {
    "crispr_activation": "sufBCDS-thrA",
    "crispr_interference": "yaaX-proB",
}
COMBINATION_GROWTH = {
    "crispr_activation": (3.55, 0.03),
    "crispr_interference": (1.26, 0.02),
}


def _paragraph(text: str) -> str:
    return f'<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'


def _row(cells: tuple[str, ...]) -> str:
    return (
        "<w:tr>" + "".join(f"<w:tc>{_paragraph(c)}</w:tc>" for c in cells) + "</w:tr>"
    )


def _table(title: str, rows: tuple[tuple[str, ...], ...]) -> str:
    return _paragraph(title) + "<w:tbl>" + "".join(_row(r) for r in rows) + "</w:tbl>"


def write_si_docx(
    path: Path,
    *,
    table_1: tuple[tuple[str, ...], ...] = TABLE_1_ROWS,
    table_3: tuple[tuple[str, ...], ...] = TABLE_3_ROWS,
    table_4: tuple[tuple[str, ...], ...] = TABLE_4_ROWS,
    titles: tuple[str, ...] = n.SI_TABLE_TITLES,
    footnote: str = n.DASH_FOOTNOTE,
    extra_tables: str = "",
) -> Path:
    """Write a minimal ``.docx`` carrying the SI's four tables in document order."""
    body = (
        _table(titles[0], table_1)
        + _table(titles[1], TABLE_2_ROWS)
        + _table(titles[2], table_3)
        + _table(titles[3], table_4)
        + extra_tables
        + _paragraph(footnote)
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W}"><w:body>{body}</w:body></w:document>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
    return path


@pytest.fixture
def si(tmp_path: Path) -> Path:
    """The synthetic SI document."""
    return write_si_docx(tmp_path / "source" / n.SI_FILENAME)


# --------------------------------------------------------------------------- #
# Reading the SI
# --------------------------------------------------------------------------- #
def test_read_si_returns_the_four_tables_in_document_order(si: Path) -> None:
    tables = n.read_si(si)
    assert len(tables.tables) == 4
    assert tuple(tables.tables[0][0]) == n.TABLE_1_HEADER
    assert tuple(tables.tables[2][0]) == n.TARGET_TABLE_HEADER
    assert tuple(tables.tables[3][0]) == n.TARGET_TABLE_HEADER
    assert tables.tables[3][2][0] == "yaaX"
    assert n.DASH_FOOTNOTE in tables.paragraphs


def test_read_si_refuses_a_fifth_table(tmp_path: Path) -> None:
    path = write_si_docx(
        tmp_path / "five.docx", extra_tables="<w:tbl><w:tr><w:tc></w:tc></w:tr></w:tbl>"
    )
    with pytest.raises(n.TableFormatError, match="5 tables, expected 4"):
        n.read_si(path)


def test_read_si_refuses_a_renamed_table_title(tmp_path: Path) -> None:
    titles = (
        *n.SI_TABLE_TITLES[:2],
        "Suppl. Table 3 Something else",
        n.SI_TABLE_TITLES[3],
    )
    path = write_si_docx(tmp_path / "retitled.docx", titles=titles)
    with pytest.raises(n.TableFormatError, match="Suppl. Table 3 Effect of CRISPR"):
        n.read_si(path)


def test_read_si_refuses_an_si_without_the_dash_footnote(tmp_path: Path) -> None:
    path = write_si_docx(tmp_path / "nofoot.docx", footnote="no footnote here")
    with pytest.raises(n.TableFormatError, match="dash footnote is not present"):
        n.read_si(path)


def test_read_si_refuses_a_document_with_no_body(tmp_path: Path) -> None:
    path = tmp_path / "nobody.docx"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml", f'<w:document xmlns:w="{_W}"></w:document>'
        )
    with pytest.raises(n.TableFormatError, match="has no w:body"):
        n.read_si(path)


# --------------------------------------------------------------------------- #
# Guide spacers
# --------------------------------------------------------------------------- #
def test_guide_spacers_take_the_middle_of_each_oligo_at_its_released_length(
    si: Path,
) -> None:
    spacers = n.read_guide_spacers(n.read_si(si).tables[0])
    assert spacers.n_oligo_rows == 4
    assert spacers.by_gene == {"thra": "A" * 20, "yaax": "C" * 22}
    assert spacers.ambiguous == {"prob": ["G" * 20, "T" * 20]}
    assert spacers.length_histogram == {20: 3, 22: 1}
    # a case difference in one symbol is not a second gene
    assert spacers.spacer_of("ThrA") == "A" * 20
    # two different spacers and no rule for choosing is not a spacer
    assert spacers.spacer_of("proB") is None
    assert spacers.spacer_of("opgB") is None


def test_guide_spacers_refuse_a_renamed_header(si: Path) -> None:
    rows = [["Primer", "Sequence", "Purpose"], *n.read_si(si).tables[0][1:]]
    with pytest.raises(n.TableFormatError, match="Suppl. Table 1 header is"):
        n.read_guide_spacers(rows)


def test_guide_spacers_refuse_an_n20_row_that_is_not_an_oligo_of_the_design() -> None:
    rows = [list(n.TABLE_1_HEADER), ["N20-thrA", "ACGTACGTACGT", "CRISPRa"]]
    with pytest.raises(n.ReleaseContentError, match="is not an N20 oligo"):
        n.read_guide_spacers(rows)


def test_guide_spacers_refuse_a_non_acgt_spacer() -> None:
    oligo = n.OLIGO_PREFIX + "ACGTNNNNACGTACGTACGT" + n.OLIGO_SUFFIX
    rows = [list(n.TABLE_1_HEADER), ["N20-thrA", oligo, "CRISPRa"]]
    with pytest.raises(n.ReleaseContentError, match="non-ACGT spacer"):
        n.read_guide_spacers(rows)


# --------------------------------------------------------------------------- #
# The two target tables
# --------------------------------------------------------------------------- #
def test_target_rows_parse_both_value_cells_and_mark_the_dashes(si: Path) -> None:
    rows = n.read_target_table(n.read_si(si).tables[2], "crispr_activation")
    assert [(r.label, r.category, r.growth_ratio, r.pinene_ratio) for r in rows] == [
        ("thrA", "Carbohydrate metabolism", 1.24, 1.06),
        ("hokC", "Carbohydrate metabolism", None, 1.19),
        ("sufBCDS", "Stress response", 1.48, 1.32),
        ("nosuchgene", "Stress response", 1.10, None),
    ]
    assert [r.growth_is_censored for r in rows] == [False, True, False, False]
    assert [r.pinene_is_censored for r in rows] == [False, False, False, True]
    assert rows[0].growth_sd == 0.01
    assert rows[0].growth_cell == "1.24 ± 0.01"


def test_target_rows_refuse_a_renamed_header(si: Path) -> None:
    rows = [["Target", *n.TARGET_TABLE_HEADER[1:]], *n.read_si(si).tables[2][1:]]
    with pytest.raises(n.TableFormatError, match="crispr_activation table header is"):
        n.read_target_table(rows, "crispr_activation")


def test_target_rows_refuse_a_value_cell_that_is_neither_a_ratio_nor_a_dash() -> None:
    rows = [
        list(n.TARGET_TABLE_HEADER),
        ["thrA", "aspartokinase I", "about 1.2", "1.06 ± 0.02"],
    ]
    with pytest.raises(n.ReleaseContentError, match="growth cell 'about 1.2'"):
        n.read_target_table(rows, "crispr_activation")


# --------------------------------------------------------------------------- #
# The synthetic BW25113 annotation
# --------------------------------------------------------------------------- #
@pytest.fixture
def bw25113(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12BW25113Genome:
    """The synthetic BW25113 genome; the network refuses."""
    files = write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    return EcoliK12BW25113Genome(genome_root=str(tmp_path / "bw25113"), overwrite=True)


def test_canonical_symbol_is_the_annotations_own_spelling(
    bw25113: EcoliK12BW25113Genome,
) -> None:
    assert n.canonical_symbol(bw25113, "BW25113_0002") == "thrA"
    # hokC's numerics disagree with its b-number, and the symbol still resolves back
    assert n.canonical_symbol(bw25113, "BW25113_4412") == "hokC"


def test_a_released_label_with_no_locus_resolves_to_nothing(
    bw25113: EcoliK12BW25113Genome,
) -> None:
    assert n._resolve(bw25113, "nosuchgene") is None
    assert n._resolve(bw25113, "thrA") == ("BW25113_0002", "thrA")


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
@pytest.fixture
def retention(
    si: Path, bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> n.Retention:
    """Retention over the synthetic release, with the fixture's arm sizes."""
    _install_fixture_sizes(monkeypatch)
    tables = n.read_si(si)
    return n.retain(
        {
            "crispr_activation": n.read_target_table(
                tables.tables[2], "crispr_activation"
            ),
            "crispr_interference": n.read_target_table(
                tables.tables[3], "crispr_interference"
            ),
        },
        n.read_guide_spacers(tables.tables[0]),
        genome=bw25113,
        dataset_name="synthetic",
    )


def _install_fixture_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture's arm sizes and combination strains, in place of the release's."""
    monkeypatch.setattr(n, "N_ACTIVATION_TARGETS", 4)
    monkeypatch.setattr(n, "N_INTERFERENCE_TARGETS", 3)
    monkeypatch.setattr(n, "COMBINATION_LABELS", COMBINATION_LABELS)
    monkeypatch.setattr(n, "COMBINATION_GROWTH", COMBINATION_GROWTH)


def test_retention_stores_the_resolvable_single_gene_growth_rows(
    retention: n.Retention,
) -> None:
    assert [(arm, label) for arm, label, _ in retention.strains] == [
        ("crispr_activation", "thrA"),
        ("crispr_interference", "yaaX"),
        ("crispr_interference", "proB"),
        ("crispr_interference", "yaaX-proB"),
    ]
    assert retention.ratios == {
        "crispr_activation/thrA": 1.24,
        "crispr_interference/yaaX": 1.07,
        "crispr_interference/proB": 1.09,
        "crispr_interference/yaaX-proB": 1.26,
    }
    targets = dict(((arm, label), t) for arm, label, t in retention.strains)
    assert [t.locus_tag for t in targets[("crispr_activation", "thrA")]] == [
        "BW25113_0002"
    ]
    # the combination strain is ONE record with one leaf per gene, in released order
    combination = targets[("crispr_interference", "yaaX-proB")]
    assert [(t.reported_symbol, t.locus_tag) for t in combination] == [
        ("yaaX", "BW25113_0008"),
        ("proB", "BW25113_0005"),
    ]
    assert [t.guide_sequence for t in combination] == ["C" * 22, None]


def test_retention_accounts_for_every_released_cell_it_does_not_store(
    retention: n.Retention,
) -> None:
    log = retention.drop_log
    assert (log.released_numeric_cells, log.stored_records, log.dropped_cells) == (
        13,
        4,
        9,
    )
    assert log.censored_cells == 5
    assert {rule.rule: (rule.n_cells, rule.items) for rule in log.rules} == {
        n.RULE_PINENE_RATIO: (6, []),
        n.RULE_MULTI_GENE: (1, ["crispr_activation/sufBCDS"]),
        n.RULE_COMBINATION_OPERON: (1, ["crispr_activation/sufBCDS-thrA"]),
        n.RULE_LABEL_NOT_IN_ANNOTATION: (1, ["crispr_activation/nosuchgene"]),
    }
    assert sum(rule.n_cells for rule in log.rules) == log.dropped_cells


def test_retention_records_why_a_stored_strain_carries_no_spacer(
    retention: n.Retention,
) -> None:
    assert retention.spacer_absences == {
        "crispr_interference/proB": (
            "the release gives this gene two DIFFERENT N20 spacers and no rule for "
            "choosing between them"
        ),
        "crispr_interference/yaaX-proB:proB": (
            "the release gives this gene two DIFFERENT N20 spacers and no rule for "
            "choosing between them"
        ),
    }


def test_retention_refuses_an_arm_whose_row_count_is_not_the_papers(
    si: Path, bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fixture_sizes(monkeypatch)
    monkeypatch.setattr(n, "N_ACTIVATION_TARGETS", 57)
    tables = n.read_si(si)
    with pytest.raises(n.ReleaseContentError, match="4 data rows, the paper states 57"):
        n.retain(
            {
                "crispr_activation": n.read_target_table(
                    tables.tables[2], "crispr_activation"
                ),
                "crispr_interference": n.read_target_table(
                    tables.tables[3], "crispr_interference"
                ),
            },
            n.read_guide_spacers(tables.tables[0]),
            genome=bw25113,
            dataset_name="synthetic",
        )


# --------------------------------------------------------------------------- #
# Genotype, phenotype, environment
# --------------------------------------------------------------------------- #
def _target(symbol: str = "thrA", tag: str = "BW25113_0002") -> n.StoredTarget:
    return n.StoredTarget(
        reported_symbol=symbol, locus_tag=tag, gene_name=symbol, guide_sequence="A" * 20
    )


def test_each_arm_takes_its_own_leaf_with_the_shared_effector() -> None:
    activation = n.perturbation(_target(), "crispr_activation")
    assert isinstance(activation, BacterialCrisprActivationPerturbation)
    assert activation.expression_direction == "increased"
    assert activation.systematic_gene_name == "BW25113_0002"
    assert activation.perturbed_gene_name == "thrA"
    assert activation.gene_namespace == "ecoli_k12_bw25113_locus_tag"
    assert activation.crispr.effector == "dCas9*-MCPSoxS"
    assert activation.crispr.guide_sequence == "A" * 20
    assert activation.identifier_mapping is not None
    assert activation.identifier_mapping.source_identifier == "thrA"
    assert activation.identifier_mapping.route == "gene_symbol"

    interference = n.perturbation(_target(), "crispr_interference")
    assert isinstance(interference, BacterialCrisprInterferencePerturbation)
    assert interference.expression_direction == "decreased"


def test_a_label_that_is_already_the_locus_tag_derives_no_mapping() -> None:
    leaf = n.perturbation(
        _target(symbol="BW25113_0002", tag="BW25113_0002"), "crispr_activation"
    )
    assert leaf.identifier_mapping is None


def test_a_genotype_holds_one_leaf_per_targeted_gene() -> None:
    genotype = n.genotype(
        [_target(), _target(symbol="yaaX", tag="BW25113_0008")], "crispr_activation"
    )
    assert [p.systematic_gene_name for p in genotype.perturbations] == [
        "BW25113_0002",
        "BW25113_0008",
    ]


def test_a_genotype_refuses_a_strain_that_targets_nothing() -> None:
    with pytest.raises(ValueError, match="targets at least one gene"):
        n.genotype([], "crispr_activation")


def test_the_stored_phenotype_is_the_log2_of_the_released_ratio() -> None:
    phenotype = n.phenotype(1.24, "crispr_activation")
    assert phenotype.measurement_type is MeasurementType.log2_ratio
    assert phenotype.assay_type is AssayType.liquid_od_growth
    assert phenotype.environment_response == pytest.approx(math.log2(1.24))
    assert phenotype.n_samples == 3
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    assert phenotype.screen_id == "crispr_activation"
    # the released SD is of the RATIO, so neither uncertainty field is filled
    assert phenotype.environment_response_uncertainty is None
    assert phenotype.environment_response_se is None
    assert {gap.field for gap in phenotype.provenance_gaps} == {
        "environment_response_uncertainty",
        "environment_response_se",
    }


def test_the_reference_phenotype_is_the_no_guide_control_at_zero() -> None:
    reference = n.reference_phenotype("crispr_interference")
    assert reference.environment_response == 0.0
    assert reference.n_samples is None
    assert [gap.field for gap in reference.provenance_gaps] == ["n_samples"]


def test_the_environment_is_the_one_tolerance_culture() -> None:
    culture = n.environment()
    assert culture.temperature is not None
    assert culture.temperature.value == 37.0
    assert culture.duration_hours == 12.0
    assert culture.media.base_medium == "LB"
    assert len(culture.perturbations) == 1
    challenge = culture.perturbations[0]
    assert isinstance(challenge, SmallMoleculePerturbation)
    compound = challenge.compound
    assert challenge.concentration is not None
    assert challenge.concentration.value == 0.5
    assert challenge.concentration.unit is ConcentrationUnit.percent
    # pinene has no curated compound row, so its structure identifiers are a typed gap
    assert compound.pubchem_cid is None
    assert compound.inchikey is None
    assert [gap.field for gap in compound.provenance_gaps] == ["inchikey"]


def test_the_inducer_is_a_medium_component_and_not_an_environment_perturbation() -> (
    None
):
    culture = n.environment()
    atc = [
        component
        for component in culture.media.components
        if component.compound.name == "anhydrotetracycline"
    ]
    assert len(atc) == 1
    assert atc[0].concentration is not None
    assert atc[0].concentration.value == 200.0
    assert atc[0].concentration.unit is ConcentrationUnit.nanomolar
    challenge = culture.perturbations[0]
    assert isinstance(challenge, SmallMoleculePerturbation)
    assert challenge.compound.name != "anhydrotetracycline"


def test_the_designed_parent_is_a_background_and_not_a_perturbation() -> None:
    background = n.NIU2019_BACKGROUND
    assert background.name == "BW25113(PT5-dxs)"
    assert background.alleles == []
    assert background.construction is None
    assert [gap.field for gap in background.provenance_gaps] == ["construction"]


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
def test_deposit_writes_the_file_and_a_manifest_that_records_its_retrieval(
    si: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(n, "SI_SHA256", hashlib.sha256(si.read_bytes()).hexdigest())
    root = n.deposit_raw_mirror(si_path=si, data_root=str(tmp_path / "data_root"))
    assert (root / n.si_rel()).read_bytes() == si.read_bytes()
    manifest = n.load_manifest(str(tmp_path / "data_root"))
    assert manifest.citation_key == n.CITATION_KEY
    assert manifest.doi == n.DOI
    assert [record.path for record in manifest.files] == [n.si_rel()]
    retrieval = manifest.files[0].retrieval
    assert retrieval is not None
    assert retrieval.method.value == "pmc_cloud"
    assert retrieval.params == {"key": n.SI_BUCKET_KEY}
    assert n.manifest_sha256(manifest, n.si_rel()) == n.SI_SHA256
    # idempotent by sha256
    assert (
        n.deposit_raw_mirror(si_path=si, data_root=str(tmp_path / "data_root")) == root
    )


def test_deposit_refuses_a_source_whose_sha256_is_not_the_pinned_one(
    si: Path, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        n.deposit_raw_mirror(si_path=si, data_root=str(tmp_path / "data_root"))


def test_deposit_refuses_to_overwrite_a_mirror_file_with_other_bytes(
    si: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data_root"
    dest = n.raw_mirror_dir(str(data_root)) / n.si_rel()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"not the released file")
    monkeypatch.setattr(n, "SI_SHA256", hashlib.sha256(si.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        n.deposit_raw_mirror(si_path=si, data_root=str(data_root))


def test_the_manifest_lookup_refuses_a_path_it_does_not_record(
    si: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(n, "SI_SHA256", hashlib.sha256(si.read_bytes()).hexdigest())
    n.deposit_raw_mirror(si_path=si, data_root=str(tmp_path / "data_root"))
    manifest = n.load_manifest(str(tmp_path / "data_root"))
    with pytest.raises(KeyError, match="data/other.docx"):
        n.manifest_sha256(manifest, "data/other.docx")


def test_the_recorded_retrieval_names_the_pmc_article_datasets_object() -> None:
    assert n.si_url().endswith(f"/{n.PMCID}.1/{n.SI_FILENAME}")
    assert n.si_retrieval().sha256 == n.SI_SHA256


# --------------------------------------------------------------------------- #
# The loader end to end, hermetic
# --------------------------------------------------------------------------- #
def _pin(strain: EcoliK12StrainName, background: Any = None) -> AssemblyReferenceGenome:
    assembly_set = BACTERIAL_ASSEMBLY_SETS[strain]
    return AssemblyReferenceGenome(
        species="Escherichia coli",
        strain=strain,
        assembly_set=cast("Any", assembly_set),
        assembly_accession=ASSEMBLY_SET_ACCESSIONS[assembly_set][0],
    )


@pytest.fixture
def mirrored(
    si: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bw25113: EcoliK12BW25113Genome,
) -> Path:
    """A tmp ``DATA_ROOT`` whose raw mirror holds the synthetic SI; returns it."""
    monkeypatch.setattr(n, "SI_SHA256", hashlib.sha256(si.read_bytes()).hexdigest())
    data_root = tmp_path / "data_root"
    n.deposit_raw_mirror(si_path=si, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _install_fixture_sizes(monkeypatch)
    monkeypatch.setattr(
        n, "bacterial_genome", lambda host, strain, data_root=None: bw25113
    )
    monkeypatch.setattr(n, "assembly_reference", _pin)
    return data_root


def test_the_loader_builds_the_synthetic_release_end_to_end(
    tmp_path: Path, mirrored: Path
) -> None:
    root = tmp_path / "dataset"
    dataset = n.CrisprPineneToleranceNiu2019Dataset(root=str(root))
    assert len(dataset) == 4
    assert sorted(dataset.gene_set) == ["BW25113_0002", "BW25113_0005", "BW25113_0008"]
    references = dataset.experiment_reference_index
    assert references is not None
    # one reference per arm, because screen_id names the arm
    assert len(references) == 2

    first = dataset[0]["experiment"]
    assert first["phenotype"]["measurement_type"] == "log2_ratio"
    assert first["phenotype"]["environment_response"] == pytest.approx(math.log2(1.24))
    assert first["phenotype"]["screen_id"] == "crispr_activation"
    assert [p["perturbation_type"] for p in first["genotype"]["perturbations"]] == [
        "bacterial_crispr_activation"
    ]
    assert dataset[0]["reference"]["phenotype_reference"]["environment_response"] == 0.0
    assert dataset[0]["publication"]["doi"] == n.DOI

    # the combination strain is the one record with more than one leaf
    sizes = sorted(
        len(dataset[i]["experiment"]["genotype"]["perturbations"])
        for i in range(len(dataset))
    )
    assert sizes == [1, 1, 1, 2]

    drops = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert (drops["released_numeric_cells"], drops["stored_records"]) == (13, 4)
    assert {rule["rule"]: rule["n_cells"] for rule in drops["rules"]} == {
        n.RULE_PINENE_RATIO: 6,
        n.RULE_MULTI_GENE: 1,
        n.RULE_COMBINATION_OPERON: 1,
        n.RULE_LABEL_NOT_IN_ANNOTATION: 1,
    }
    released = json.loads(
        (root / "preprocess" / "released_growth_ratios.json").read_text()
    )
    assert released["arms"]["crispr_activation"][0]["growth_cell"] == "1.24 ± 0.01"
    assert (
        released["combination_strains"]["crispr_interference"]["growth_ratio"] == 1.26
    )
    spacers = json.loads((root / "preprocess" / "guide_spacers.json").read_text())
    assert spacers["n_oligo_rows"] == 4
    assert spacers["genes_with_two_different_spacers"] == {"prob": ["G" * 20, "T" * 20]}
    not_loaded = json.loads((root / "preprocess" / "not_loaded.json").read_text())
    variants = [item for item in not_loaded if item.get("n_rows") == 374]
    assert len(variants) == 1
    assert variants[0]["n_writable_on_the_variant_leaves"] == 373
    assert (root / "raw" / n.SI_FILENAME).is_file()


def test_the_synthetic_build_passes_l0_to_l4(
    tmp_path: Path, mirrored: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    root = tmp_path / "dataset"
    n.CrisprPineneToleranceNiu2019Dataset(root=str(root))
    report = n.verify_build(str(root), genome=bw25113, expected_count=4)
    assert report.passed, report.summary()
    assert (root / "preprocess" / "verification_report.json").is_file()


def test_the_loader_is_registered_and_receives_the_bw25113_genome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        dataset_registry["CrisprPineneToleranceNiu2019Dataset"]
        is n.CrisprPineneToleranceNiu2019Dataset
    )
    assert n.CrisprPineneToleranceNiu2019Dataset.REFERENCE_STRAIN == "BW25113"
    install_bacterial_fakes(monkeypatch)
    injector = BacterialGenomeInjector(data_root="/nowhere")
    kwargs = injector.genome_kwargs(n.CrisprPineneToleranceNiu2019Dataset)
    assert set(kwargs) == {"ecoli_genome"}
    assert isinstance(kwargs["ecoli_genome"], FakeBW25113Genome)


def test_the_loader_refuses_a_genome_of_the_wrong_strain() -> None:
    dataset = n.CrisprPineneToleranceNiu2019Dataset.__new__(
        n.CrisprPineneToleranceNiu2019Dataset
    )
    dataset.ecoli_genome = cast("Any", object())
    with pytest.raises(AttributeError):
        dataset._genome()


def test_create_experiment_is_not_the_entry_point() -> None:
    dataset = n.CrisprPineneToleranceNiu2019Dataset.__new__(
        n.CrisprPineneToleranceNiu2019Dataset
    )
    with pytest.raises(NotImplementedError, match="builds records in process"):
        dataset.create_experiment()


def test_the_module_cli_names_its_three_commands() -> None:
    with pytest.raises(SystemExit):
        n.main([])


# --------------------------------------------------------------------------- #
# The pinned release (--data)
# --------------------------------------------------------------------------- #
@pytest.mark.data
@requires_library
def test_every_sourced_value_is_verbatim_in_the_pinned_ocr() -> None:
    assert _LIBRARY is not None
    root = osp.join(DATA_ROOT, "torchcell-library")
    values = [v for v in vars(n).values() if isinstance(v, SourcedValue)]
    assert len(values) == len(n.SOURCED_VALUES) == 11
    for value in values:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"


@pytest.mark.data
@requires_mirror
def test_the_raw_mirror_holds_the_one_pinned_supplementary_file() -> None:
    assert _MIRROR is not None
    manifest = n.load_manifest(DATA_ROOT)
    assert n.manifest_sha256(manifest, n.si_rel()) == n.SI_SHA256
    assert n._sha256(_MIRROR / n.si_rel()) == n.SI_SHA256
    assert manifest.provenance_complete is True


@pytest.mark.data
@requires_mirror
def test_the_pinned_si_holds_the_release_the_paper_states() -> None:
    assert _MIRROR is not None
    tables = n.read_si(_MIRROR / n.si_rel())
    activation = n.read_target_table(tables.tables[2], "crispr_activation")
    interference = n.read_target_table(tables.tables[3], "crispr_interference")
    assert (len(activation), len(interference)) == (57, 20)
    spacers = n.read_guide_spacers(tables.tables[0])
    assert spacers.n_oligo_rows == 91
    assert spacers.length_histogram == {20: 90, 22: 1}
    assert sorted(spacers.ambiguous) == ["mara", "sspa"]
    numeric = {
        "crispr_activation/growth": sum(
            1 for r in activation if r.growth_ratio is not None
        ),
        "crispr_activation/pinene": sum(
            1 for r in activation if r.pinene_ratio is not None
        ),
        "crispr_interference/growth": sum(
            1 for r in interference if r.growth_ratio is not None
        ),
        "crispr_interference/pinene": sum(
            1 for r in interference if r.pinene_ratio is not None
        ),
    }
    assert numeric == {
        "crispr_activation/growth": 43,
        "crispr_activation/pinene": 32,
        "crispr_interference/growth": 9,
        "crispr_interference/pinene": 6,
    }
    assert sorted({r.label for r in activation} & n.MULTI_GENE_LABELS) == [
        "flgFGH",
        "sufBCDS",
    ]


@pytest.mark.data
@requires_store
def test_the_built_store_holds_the_measured_records() -> None:
    assert _STORE is not None
    records = list(n.stored_records(str(_STORE)))
    assert len(records) == n.EXPECTED_RECORDS == 51
    arms = {r["experiment"]["phenotype"]["screen_id"] for r in records}
    assert arms == {"crispr_activation", "crispr_interference"}
    leaves = [
        p["perturbation_type"]
        for r in records
        for p in r["experiment"]["genotype"]["perturbations"]
    ]
    assert leaves.count("bacterial_crispr_activation") == 41
    assert leaves.count("bacterial_crispr_interference") == 15
    assert {r["experiment"]["phenotype"]["measurement_type"] for r in records} == {
        "log2_ratio"
    }
    drops = json.loads(
        (Path(_STORE) / "preprocess" / "dropped_records.json").read_text()
    )
    assert drops["released_numeric_cells"] == n.RELEASED_NUMERIC_CELLS == 94
    assert drops["stored_records"] == 51
    assert drops["censored_cells"] == 64


@pytest.mark.data
@requires_store
def test_the_built_store_passes_l0_to_l4() -> None:
    assert _STORE is not None
    report = n.verify_build(str(_STORE), data_root=DATA_ROOT)
    assert report.passed, report.summary()
    assert osp.isfile(osp.join(str(_STORE), "preprocess", "verification_report.json"))
