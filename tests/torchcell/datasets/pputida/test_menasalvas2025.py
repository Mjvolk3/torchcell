# tests/torchcell/datasets/pputida/test_menasalvas2025.py
# [[tests.torchcell.datasets.pputida.test_menasalvas2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_menasalvas2025.py
"""Tests for the Menasalvas 2025 biosensor-coupled CRISPRi selection loader.

Everything here runs with NO network and NO ``$DATA_ROOT``: the Supplementary
Information markdown is synthesized to the shape the pinned OCR has, the raw mirror is
deposited under ``tmp_path`` by the module's own ``deposit_raw_mirror`` with its pins
monkeypatched to the synthetic digests, and the KT2440 annotation is the real genome
class over a synthetic assembly. A separate block, skipped without the mirror, pins the
numbers measured on the REAL bytes: 58 records, 60 distinct targets and 28/30 rows.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
from pathlib import Path
from typing import Any

import pytest

import torchcell.datasets.pputida.menasalvas2025 as mv
from torchcell.datamodels.media import M9_NREL_MOPS_MENASALVAS2025
from torchcell.datamodels.schema import (
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    SmallMoleculePerturbation,
)
from torchcell.datasets.bacteria_common import reconcile_locus_tags

# --------------------------------------------------------------------------- #
# Synthetic Supplementary Information markdown
# --------------------------------------------------------------------------- #
#: Locus tags the synthetic assembly annotates, with the symbol it gives each one.
LOCUS_SPECS: tuple[tuple[str, str | None], ...] = (
    ("PP_1815", "pyrF"),
    ("PP_2428", "sotB"),
    ("PP_3400", "alkB"),
    ("PP_4373", "fleQ"),
    ("PP_5003", "phaA"),
    ("PP_5004", "phaB"),
    ("PP_5005", "phaC-II"),
    ("PP_5007", None),
    ("PP_1656", "relA"),
    ("PP_2710", None),
    ("PP_4485", "hisQ"),
    ("PP_2074", None),
)
#: The ``Gene`` cells of the synthetic round-1 table, one per expected row.
ROUND1_CELLS: tuple[str, ...] = (
    "phaAZC-II / PP_5003-PP_5005",
    "PP_5007",
    "alkB /PP_3400",
    "sotB / PP_2428",
    "fleQ / PP_4373",
)
#: The ``Gene`` cells of the synthetic round-2 table; no tag may repeat round 1.
ROUND2_CELLS: tuple[str, ...] = ("relA PP_1656", "PP_2710", "hisQ |PP_4485", "PP_2074")

ASSEMBLY_REPORT = """# Assembly name:  ASM756v2
# Organism name:  Pseudomonas putida KT2440 (g-proteobacteria)
# Infraspecific name:  strain=KT2440
# Taxid:          160488
# GenBank assembly accession: GCA_000007565.2
# RefSeq assembly accession: GCF_000007565.2
# RefSeq assembly and GenBank assemblies identical: yes
#
## Assembly-Units:
AE015451.2\tassembled-molecule\tna\tChromosome\tAE015451.2\t=\tNC_002947.4
"""
ASSEMBLY_REPORT_MEMBER = "GCA_000007565.2_ASM756v2_assembly_report.txt"


def _table(number: int, title: str, cells: tuple[str, ...]) -> str:
    """One caption line plus the single-line ``<table>`` the OCR emits for it."""
    header = "".join(
        f'<td colspan="1" rowspan="1">{name}</td>' for name in mv.TABLE_HEADER
    )
    rows = [f"<tr>{header}</tr>"]
    for index, cell in enumerate(cells):
        body = "".join(
            f'<td colspan="1" rowspan="1">{value}</td>'
            for value in (cell, f"synthetic function {index}", "enzyme")
        )
        rows.append(f"<tr>{body}</tr>")
    body_html = "".join(rows)
    return f"Supplementary Table {number}. {title}\n\n<table>{body_html}</table>\n"


def synthetic_markdown(
    round1: tuple[str, ...] = ROUND1_CELLS,
    round2: tuple[str, ...] = ROUND2_CELLS,
    *,
    header: tuple[str, ...] | None = None,
) -> str:
    """The two tables, wrapped in enough surrounding prose to exercise the scanner."""
    text = "# Supplementary Materials for\n\nsome prose\n\n"
    text += _table(
        1, "Lower pedF-RBS-pyrF Isoprenol Threshold: Enriched gRNAs.", round1
    )
    text += "\nmore prose\n\n"
    text += _table(
        2,
        "Second Round pedF-RBS-pyrF with Higher Isoprenol Threshold: Enriched gRNAs.",
        round2,
    )
    text += "\n# Supplementary Table 4. Strains Used in This Study.\n\n<table><tr>"
    text += '<td colspan="1" rowspan="1">TEAM-862</td></tr></table>\n'
    if header is not None:
        text = text.replace(
            "".join(
                f'<td colspan="1" rowspan="1">{name}</td>' for name in mv.TABLE_HEADER
            ),
            "".join(f'<td colspan="1" rowspan="1">{name}</td>' for name in header),
            1,
        )
    return text


# --------------------------------------------------------------------------- #
# Synthetic: the Gene-cell parser
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("cell", "tags", "symbol"),
    [
        ("PP_5007", ("PP_5007",), None),
        ("sotB / PP_2428", ("PP_2428",), "sotB"),
        ("cmpX PP_2087", ("PP_2087",), "cmpX"),
        ("hisQ |PP_4485", ("PP_4485",), "hisQ"),
        ("relA PP_1656", ("PP_1656",), "relA"),
        ("alkB /PP_3400", ("PP_3400",), "alkB"),
        ("phaAZC-II / PP_5003-PP_5005", ("PP_5003", "PP_5004", "PP_5005"), "phaAZC-II"),
    ],
)
def test_gene_cells_parse_to_tags_and_the_sources_own_symbol(
    cell: str, tags: tuple[str, ...], symbol: str | None
) -> None:
    """Every spelling the released tables use resolves to its tags and its symbol."""
    assert mv.parse_gene_cell(cell) == (tags, symbol)


def test_an_operon_range_expands_by_integer_enumeration() -> None:
    """``PP_5003-PP_5005`` is three loci, not the two endpoints the regex would find."""
    tags, _ = mv.parse_gene_cell("phaAZC-II / PP_5003-PP_5005")
    assert tags == ("PP_5003", "PP_5004", "PP_5005")
    assert len(tags) == 3


def test_a_gene_cell_with_no_locus_tag_is_refused() -> None:
    """A cell naming only a symbol cannot be keyed to a locus, so the build stops."""
    with pytest.raises(RuntimeError, match="names no PP_ locus tag"):
        mv.parse_gene_cell("phaF")


def test_a_descending_locus_range_is_refused() -> None:
    """``PP_5005-PP_5003`` is not an ascending operon span."""
    with pytest.raises(RuntimeError, match="does not ascend"):
        mv.parse_gene_cell("x / PP_5005-PP_5003")


def test_a_gene_cell_repeating_a_tag_is_refused() -> None:
    """A tag named twice would silently collapse, so it is a refusal instead."""
    with pytest.raises(RuntimeError, match="repeats a locus tag"):
        mv.parse_gene_cell("PP_5003-PP_5005 PP_5004")


# --------------------------------------------------------------------------- #
# Synthetic: the table readers
# --------------------------------------------------------------------------- #
def test_read_table_returns_the_three_released_columns() -> None:
    """The reader keeps ``Gene``, ``Function`` and ``FunctionalCategory`` verbatim."""
    rows = mv.read_table(synthetic_markdown(), 1)
    assert len(rows) == len(ROUND1_CELLS)
    assert rows[0] == (ROUND1_CELLS[0], "synthetic function 0", "enzyme")


def test_read_table_picks_the_table_its_caption_names() -> None:
    """Table 2's rows are Table 2's, although three tables share the document."""
    rows = mv.read_table(synthetic_markdown(), 2)
    assert [row[0] for row in rows] == list(ROUND2_CELLS)


def test_a_changed_table_header_is_refused() -> None:
    """A renamed column means the export moved; parsing on would mis-key the data."""
    markdown = synthetic_markdown(header=("Locus", "Function", "FunctionalCategory"))
    with pytest.raises(RuntimeError, match="header is"):
        mv.read_table(markdown, 1)


def test_a_missing_table_is_refused() -> None:
    """A table the loader needs and the markdown lacks stops the build."""
    with pytest.raises(RuntimeError, match="Supplementary Table 9 is not in"):
        mv.read_table(synthetic_markdown(), 9)


def test_a_ragged_table_row_is_refused() -> None:
    """A row with a dropped cell would shift the columns, so it is a refusal."""
    markdown = synthetic_markdown().replace(
        '<td colspan="1" rowspan="1">enzyme</td></tr>', "</tr>", 1
    )
    with pytest.raises(RuntimeError, match="row has 2 cells"):
        mv.read_table(markdown, 1)


def test_read_selected_targets_checks_the_per_round_row_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A round whose row count is not the Methods' stated one is a refusal."""
    path = tmp_path / "si1.md"
    path.write_text(synthetic_markdown(), encoding="utf-8")
    monkeypatch.setattr(mv, "EXPECTED_ROWS", {1: 28, 2: 30})
    with pytest.raises(RuntimeError, match="the Methods state 28"):
        mv.read_selected_targets(str(path))


def test_read_selected_targets_refuses_a_target_shared_by_the_two_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The paper states the rounds do not overlap, so an overlap is a refusal."""
    path = tmp_path / "si1.md"
    overlapping = ROUND2_CELLS[:-1] + ("sotB / PP_2428",)
    path.write_text(synthetic_markdown(round2=overlapping), encoding="utf-8")
    monkeypatch.setattr(
        mv, "EXPECTED_ROWS", {1: len(ROUND1_CELLS), 2: len(overlapping)}
    )
    with pytest.raises(RuntimeError, match="the two rounds share"):
        mv.read_selected_targets(str(path))


def test_read_selected_targets_tags_each_row_with_its_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round membership is what keeps the two screens apart, so it is stored."""
    path = tmp_path / "si1.md"
    path.write_text(synthetic_markdown(), encoding="utf-8")
    monkeypatch.setattr(
        mv, "EXPECTED_ROWS", {1: len(ROUND1_CELLS), 2: len(ROUND2_CELLS)}
    )
    targets = mv.read_selected_targets(str(path))
    assert len(targets) == len(ROUND1_CELLS) + len(ROUND2_CELLS)
    assert [t.round_number for t in targets] == [1] * len(ROUND1_CELLS) + [2] * len(
        ROUND2_CELLS
    )
    assert targets[0].locus_tags == ("PP_5003", "PP_5004", "PP_5005")


# --------------------------------------------------------------------------- #
# Synthetic: the genotype
# --------------------------------------------------------------------------- #
def test_the_host_background_is_exactly_the_declared_seven_perturbations() -> None:
    """Every record carries the pyrF lesion, the five pathway genes and the reporter."""
    host = mv.host_perturbations()
    assert len(host) == 7
    assert {p.systematic_gene_name for p in host} == set(mv.HOST_BACKGROUND_NAMES)
    kinds = {p.perturbation_type for p in host}
    assert kinds == {"bacterial_deletion", "heterologous_pathway"}


def test_the_host_background_refuses_to_drift_from_its_declared_name_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verifier's ``background_genes`` must stay the set the genotype builds."""
    monkeypatch.setattr(mv, "HOST_BACKGROUND_NAMES", frozenset({"PP_1815"}))
    with pytest.raises(RuntimeError, match="the host background is"):
        mv.host_perturbations()


def test_the_pyrf_deletion_is_the_strain_tables_own_locus() -> None:
    """``PP_1815`` comes from "Pp KT2440 ΔPP_1815/pyrF", not from a symbol lookup."""
    deletion = mv.pyrf_deletion()
    assert deletion.systematic_gene_name == "PP_1815"
    assert deletion.perturbed_gene_name == "pyrF"
    assert deletion.gene_namespace == "pputida_kt2440_locus_tag"
    assert mv.PYRF_DELETION.quote == "Pp KT2440 ΔPP_1815/pyrF"


def test_the_five_pathway_genes_are_integrated_with_an_unreported_origin() -> None:
    """The pathway is chromosomal here, and its source organisms are not released."""
    pathway = mv.pathway_perturbations()
    assert [p.systematic_gene_name for p in pathway] == [
        "mvaS",
        "mvaE",
        "MKmm",
        "PMDHKQ",
        "aphA",
    ]
    assert {p.localization for p in pathway} == {"chromosomal_integration"}
    assert {p.integration_locus for p in pathway} == {
        "PP_5322intergenic",
        "PP_0871intergenic",
    }
    assert {p.promoter_name for p in pathway} == {"Pcv", "Ptrc"}
    assert {p.source_organism for p in pathway} == {mv.SOURCE_ORGANISM_UNREPORTED}
    assert {p.pathway_name for p in pathway} == {"isoprenol pathway"}


def test_a_pathway_token_absent_from_the_quoted_plasmid_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored identifier must be a substring of the quote that justifies it."""
    monkeypatch.setattr(
        mv,
        "PATHWAY_GENES",
        ({"token": "nudB", "promoter": "Ptrc", "locus": "PP_0871intergenic"},),
    )
    with pytest.raises(RuntimeError, match="is not in the quoted pIY670 description"):
        mv.pathway_perturbations()


def test_the_reporter_is_the_teredinibacter_pyrf_homolog_on_a_plasmid() -> None:
    """The growth-coupled reporter is a heterologous addition, not the native gene."""
    reporter = mv.reporter_perturbation()
    assert reporter.systematic_gene_name == "TERTU_1389"
    assert reporter.source_organism == "Teredinibacter turnerae T7901"
    assert reporter.localization == "episomal_plasmid"
    assert reporter.promoter_name == "PpedF"


def test_a_knockdown_carries_three_designed_guides_and_no_spacer() -> None:
    """The library designs three guides per gene and released no enriched spacer."""
    knockdown = mv.crispri_perturbation("PP_2428", "sotB")
    assert knockdown.perturbation_type == "bacterial_crispr_interference"
    assert knockdown.crispr is not None
    assert knockdown.crispr.effector == "dCpf1/dCas12a"
    assert knockdown.crispr.n_guides == 3
    assert knockdown.crispr.guide_sequence is None
    assert knockdown.expression_direction == "decreased"


# --------------------------------------------------------------------------- #
# Synthetic: the environment and the phenotype
# --------------------------------------------------------------------------- #
def _compound_names(environment: Any) -> set[str]:
    """Names of the compounds one environment's small-molecule edits carry."""
    return {
        edit.compound.name
        for edit in environment.perturbations
        if isinstance(edit, SmallMoleculePerturbation)
    }


def test_the_selection_environment_is_induced_and_the_control_is_not() -> None:
    """Crystal violet is what makes the selection arm isoprenol-producing."""
    induced = mv._environment(induced=True)
    control = mv._environment(induced=False)
    names = _compound_names(induced)
    assert "crystal violet" in names
    control_names = _compound_names(control)
    assert "crystal violet" not in control_names
    assert "kanamycin" in control_names
    assert induced.duration_hours == 24.0
    assert induced.media.name == M9_NREL_MOPS_MENASALVAS2025.name


def test_the_selection_temperature_is_a_typed_absence() -> None:
    """The paper never states this plate's temperature, so none is carried over."""
    induced = mv._environment(induced=True)
    assert induced.temperature is None
    assert induced.gapped_fields() == {"temperature"}
    assert mv.CONJUGATION_TEMPERATURE.value == 30.0


def test_the_environment_refuses_a_medium_that_is_not_an_m9_derivative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The served medium must still be the M9 the Methods name."""
    swapped = M9_NREL_MOPS_MENASALVAS2025.model_copy(
        update={"base_medium": "LB", "name": "not M9"}
    )
    monkeypatch.setattr(mv, "M9_NREL_MOPS_MENASALVAS2025", swapped)
    with pytest.raises(RuntimeError, match="is not an M9 derivative"):
        mv._environment(induced=True)


def test_the_record_phenotype_is_a_categorical_biosensor_call_not_a_titer() -> None:
    """The released readout is growth under the biosensor, with no number at all."""
    phenotype = mv.selection_phenotype(1, is_reference=False)
    assert phenotype.measurement_type == "categorical"
    assert phenotype.assay_type == "biosensor_readout"
    assert phenotype.category == "enhanced"
    assert phenotype.category_label == "enriched"
    assert phenotype.environment_response is None
    assert phenotype.n_samples == 4
    assert phenotype.sample_unit == "biological_replicate"
    assert phenotype.screen_id == "round-1-lower-isoprenol-threshold"


def test_the_phenotype_declares_the_score_and_the_dispersion_as_typed_absences() -> (
    None
):
    """No per-guide read count and no replicate dispersion were released."""
    phenotype = mv.selection_phenotype(2, is_reference=False)
    assert phenotype.gapped_fields() == {
        "environment_response",
        "environment_response_uncertainty",
        "environment_response_uncertainty_type",
    }
    assert phenotype.environment_response_se is None


def test_the_reference_phenotype_is_the_uninduced_baseline() -> None:
    """The control arm omits the inducer, so no clone can be enriched there."""
    reference = mv.selection_phenotype(1, is_reference=True)
    assert reference.category == "no_change"
    assert reference.category_label == "no crystal violet"


def test_each_round_keeps_its_own_screen_id() -> None:
    """``screen_id`` is what keeps two independently thresholded screens apart."""
    ids = {
        mv.selection_phenotype(n, is_reference=False).screen_id
        for n, _ in mv.ROUND_TABLES
    }
    assert ids == set(mv.SCREEN_IDS.values())
    assert len(ids) == 2


# --------------------------------------------------------------------------- #
# Synthetic: retention bookkeeping
# --------------------------------------------------------------------------- #
def _accounting(**overrides: Any) -> mv.BuildAccounting:
    """A balanced accounting record, overridable field by field."""
    fields: dict[str, Any] = {
        "dataset": "D",
        "source_rows": 58,
        "candidate_records": 58,
        "kept_records": 58,
        "dropped_records": 0,
        "distinct_targets": 60,
    }
    fields.update(overrides)
    return mv.BuildAccounting(**fields)


def test_balanced_accounting_passes_its_own_check() -> None:
    """Nothing is dropped, and the arithmetic says so."""
    accounting = _accounting()
    accounting.check()
    assert accounting.kept_records == accounting.candidate_records == 58
    assert accounting.dropped_records == 0


def test_accounting_refuses_arithmetic_that_does_not_balance() -> None:
    """A record that is neither kept nor dropped is a silent loss."""
    with pytest.raises(RuntimeError, match="!= 58 candidates"):
        _accounting(kept_records=57).check()


def test_accounting_refuses_a_drop_with_no_declared_rule() -> None:
    """This dataset declares no drop rule, so a drop is a build error."""
    with pytest.raises(RuntimeError, match="records dropped, but this dataset"):
        _accounting(kept_records=57, dropped_records=1).check()


# --------------------------------------------------------------------------- #
# Synthetic: the raw mirror
# --------------------------------------------------------------------------- #
def _sha256_bytes(path: Path) -> str:
    """sha256 of a small file read whole."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A raw mirror under ``tmp_path`` written by the module's own deposit function."""
    staging = tmp_path / "staging"
    staging.mkdir()
    pdf = staging / mv.SI_PDF_FILENAME
    pdf.write_bytes(b"%PDF-1.7 synthetic supplementary information\n")
    markdown = staging / mv.SI1_MD_FILENAME
    markdown.write_text(synthetic_markdown(), encoding="utf-8")
    monkeypatch.setattr(mv, "SI_PDF_SHA256", _sha256_bytes(pdf))
    monkeypatch.setattr(mv, "SI1_MD_SHA256", _sha256_bytes(markdown))
    monkeypatch.setattr(
        mv, "EXPECTED_ROWS", {1: len(ROUND1_CELLS), 2: len(ROUND2_CELLS)}
    )
    monkeypatch.setattr(mv, "EXPECTED_RECORDS", len(ROUND1_CELLS) + len(ROUND2_CELLS))
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    root = mv.deposit_raw_mirror(
        si_pdf_path=pdf, si_md_path=markdown, data_root=str(data_root)
    )
    assert root == mv.raw_mirror_dir(str(data_root))
    return data_root


def test_the_deposit_records_a_rerunnable_retrieval_and_an_ocr_processing_step(
    synthetic_mirror: Path,
) -> None:
    """The PDF carries its PMC retrieval; the markdown carries the MinerU run."""
    manifest = mv.load_manifest(str(synthetic_mirror))
    by_path = {record.path: record for record in manifest.files}
    pdf = by_path[mv.SI_PDF_REL]
    assert pdf.retrieval is not None
    assert pdf.retrieval.method == "pmc_cloud"
    assert pdf.retrieval.retriever == ("torchcell.literature.retrieve.pmc_cloud_object")
    assert pdf.retrieval.params == {"key": f"{mv.PMC_PREFIX}/{mv.SI_PDF_FILENAME}"}
    assert pdf.original_filename == mv.SI_PDF_FILENAME
    ocr = by_path[mv.SI1_MD_REL]
    assert ocr.retrieval is None
    assert ocr.processing is not None
    assert ocr.processing.tool == "mineru"
    assert ocr.processing.version == mv.MINERU_VERSION
    assert ocr.processing.input_sha256 == [mv.SI_PDF_SHA256]


def test_the_manifest_enumerates_every_deposit_and_states_the_titer_gap(
    synthetic_mirror: Path,
) -> None:
    """Dryad, Zenodo, PRIDE and the BioProject are listed; none holds a titer."""
    manifest = mv.load_manifest(str(synthetic_mirror))
    assert any(mv.DRYAD_DOI in source for source in manifest.si_data_sources)
    assert any(mv.ZENODO_DOI in source for source in manifest.si_data_sources)
    assert any(mv.PRIDE_ACCESSION in source for source in manifest.si_data_sources)
    assert any(mv.BIOPROJECT_ACCESSION in source for source in manifest.si_data_sources)
    assert any(
        "NO PER-STRAIN ISOPRENOL TITER IS RELEASED" in entry
        for entry in manifest.si_expected
    )
    assert any(
        "NO PER-GUIDE ENRICHMENT MATRIX IS RELEASED" in entry
        for entry in manifest.si_expected
    )


def test_manifest_sha256_reads_a_pin_and_refuses_an_unknown_path(
    synthetic_mirror: Path,
) -> None:
    """A path the mirror does not hold is a KeyError, never a silent None."""
    manifest = mv.load_manifest(str(synthetic_mirror))
    assert mv.manifest_sha256(manifest, mv.SI1_MD_REL) == mv.SI1_MD_SHA256
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        mv.manifest_sha256(manifest, "si/absent.md")


def test_redepositing_the_same_bytes_is_idempotent(synthetic_mirror: Path) -> None:
    """A second deposit of matching bytes rewrites the manifest and nothing else."""
    root = mv.raw_mirror_dir(str(synthetic_mirror))
    before = (root / mv.SI1_MD_REL).read_bytes()
    mv.deposit_raw_mirror(
        si_pdf_path=root / mv.SI_PDF_REL,
        si_md_path=root / mv.SI1_MD_REL,
        data_root=str(synthetic_mirror),
    )
    assert (root / mv.SI1_MD_REL).read_bytes() == before


def test_a_mirror_file_with_a_different_digest_is_never_overwritten(
    synthetic_mirror: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream drift is detected, not followed."""
    root = mv.raw_mirror_dir(str(synthetic_mirror))
    (root / mv.SI1_MD_REL).write_text("drifted", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        mv.deposit_raw_mirror(
            si_pdf_path=root / mv.SI_PDF_REL,
            si_md_path=tmp_path / "staging" / mv.SI1_MD_FILENAME,
            data_root=str(synthetic_mirror),
        )


def test_the_isoprenol_inchikey_is_recorded_and_agrees_with_the_table() -> None:
    """No record names a product here, but the recorded key is the table's own.

    The curation step this constant was recorded for has happened (the table's isoprenol
    row, PubChem CID 12988), so ``resolved_compound`` no longer gaps ``inchikey``; the
    constant is kept as the cross-check that the row is the molecule this paper means.
    """
    assert mv.ISOPRENOL_INCHIKEY == "CPJRRXSHAYUTGL-UHFFFAOYSA-N"
    from torchcell.datamodels.compound_identity import resolved_compound

    isoprenol = resolved_compound("isoprenol")
    assert isoprenol.inchikey == mv.ISOPRENOL_INCHIKEY
    assert isoprenol.gapped_fields() == set()


def test_raw_mirror_dir_falls_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Called with no argument, the mirror path comes from ``DATA_ROOT``."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert mv.raw_mirror_dir() == tmp_path / mv.RAW_DIR_REL


# --------------------------------------------------------------------------- #
# Hermetic end to end: the synthetic assembly, the mirror, the built store
# --------------------------------------------------------------------------- #
def _synthetic_loci() -> list[Any]:
    """One ``SyntheticLocus`` per :data:`LOCUS_SPECS` entry, laid end to end."""
    from tests.torchcell.sequence.genome._bacterial_fixtures import SyntheticLocus

    loci = []
    cursor = 1
    for index, (tag, symbol) in enumerate(LOCUS_SPECS):
        start, end = cursor, cursor + 11
        cursor = end + 3
        loci.append(
            SyntheticLocus(
                tag=tag,
                parts=((start, end),),
                strand="+" if index % 2 == 0 else "-",
                symbol=symbol,
                product=f"synthetic product {tag}",
                protein_id=f"AAN{index:05d}.1",
                protein="MKV",
            )
        )
    return loci


@pytest.fixture
def synthetic_kt2440(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real KT2440 class over a synthetic assembly, with the network refused."""
    import tests.torchcell.sequence.genome._bacterial_fixtures as fixtures
    from torchcell.sequence.genome.pputida.kt2440 import (
        KT2440_ASSEMBLY,
        PPutidaKT2440Genome,
    )

    monkeypatch.setattr(fixtures, "SEQUENCE", fixtures.SEQUENCE * 6)
    files = fixtures.write_assembly(
        tmp_path / "tier",
        KT2440_ASSEMBLY,
        _synthetic_loci(),
        [fixtures.gaf_row("pyrF", "pyrF|PP_1815", "GO:0000001")],
    )
    fixtures.forbid_network(monkeypatch)
    fixtures.serve_tier(monkeypatch, files)
    report = tmp_path / "tier" / ASSEMBLY_REPORT_MEMBER
    report.write_text(ASSEMBLY_REPORT)

    def serve_report(assembly_set: str, filename: str, **_: Any) -> str:
        if filename != ASSEMBLY_REPORT_MEMBER:
            raise FileNotFoundError(f"{assembly_set}/{filename} is not in the fixture")
        return str(report)

    import torchcell.datasets.bacteria_common as bacteria_common

    monkeypatch.setattr(bacteria_common, "resolve", serve_report)
    root = tmp_path / "kt2440"
    root.mkdir()
    return PPutidaKT2440Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def built(synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path) -> Any:
    """The loader built over the synthetic mirror and the synthetic annotation."""
    return mv.IsoprenolSelectionMenasalvas2025Dataset(
        root=str(tmp_path / "build"), pputida_genome=synthetic_kt2440
    )


def test_the_build_writes_one_record_per_selected_target(built: Any) -> None:
    """Record count equals the parsed rows, and the two references are the rounds'."""
    assert len(built) == len(ROUND1_CELLS) + len(ROUND2_CELLS)
    assert built.experiment_class is BacterialEnvironmentResponseExperiment
    assert built.reference_class is BacterialEnvironmentResponseExperimentReference
    assert built.raw_file_names == [mv.SI1_MD_FILENAME]


def test_each_built_record_carries_the_host_background_plus_its_knockdowns(
    built: Any,
) -> None:
    """A record is the constant host plus one guide's targets, and nothing else."""
    item = built[0]
    perturbations = item["experiment"]["genotype"]["perturbations"]
    knockdowns = [
        p
        for p in perturbations
        if p["perturbation_type"] == "bacterial_crispr_interference"
    ]
    others = {
        p["systematic_gene_name"]
        for p in perturbations
        if p["perturbation_type"] != "bacterial_crispr_interference"
    }
    assert others == set(mv.HOST_BACKGROUND_NAMES)
    assert len(knockdowns) == 3
    assert [p["systematic_gene_name"] for p in knockdowns] == [
        "PP_5003",
        "PP_5004",
        "PP_5005",
    ]


def test_the_stored_gene_name_is_the_annotations_not_the_tables(built: Any) -> None:
    """The source writes ``phaAZC-II``; the assembly annotates ``PP_5004`` as phaB."""
    perturbations = built[0]["experiment"]["genotype"]["perturbations"]
    by_tag = {
        p["systematic_gene_name"]: p["perturbed_gene_name"]
        for p in perturbations
        if p["perturbation_type"] == "bacterial_crispr_interference"
    }
    assert by_tag["PP_5004"] == "phaB"
    assert by_tag["PP_5005"] == "phaC-II"


def test_an_unannotated_locus_keeps_its_tag_as_the_common_name(built: Any) -> None:
    """``PP_5007`` has no symbol in the assembly, so the tag is its common name."""
    names: dict[str, str] = {}
    for index in range(len(built)):
        for p in built[index]["experiment"]["genotype"]["perturbations"]:
            if p["perturbation_type"] == "bacterial_crispr_interference":
                names[p["systematic_gene_name"]] = p["perturbed_gene_name"]
    assert names["PP_5007"] == "PP_5007"
    assert names["PP_2428"] == "sotB"


def test_the_build_records_its_own_retention_and_its_unasserted_chassis(
    built: Any, tmp_path: Path
) -> None:
    """The accounting states the arithmetic and the chassis the records do not carry."""
    accounting = json.loads(
        Path(built.preprocess_dir, "build_accounting.json").read_text()
    )
    assert accounting["kept_records"] == len(built)
    assert accounting["dropped_records"] == 0
    assert accounting["distinct_targets"] == 11
    assert accounting["reconciliation"]["unique_names"] == 11
    assert accounting["reconciliation"]["status_histogram"]["current"] == 11
    assert accounting["reconciliation"]["outside_namespace"] == []
    joined = " ".join(accounting["notes"])
    assert "no ProductTiterPhenotype is written" in joined
    assert "PJ23119-PP_1697" in joined


def test_the_build_keeps_the_tables_verbatim_cells_beside_the_record(
    built: Any,
) -> None:
    """``preprocess/selected_targets.csv`` keeps the source's own spelling and function."""
    import pandas as pd

    frame = pd.read_csv(osp.join(built.preprocess_dir, "selected_targets.csv"))
    assert len(frame) == len(built)
    assert set(frame["round"]) == {1, 2}
    row = frame.loc[frame["gene_cell"] == "phaAZC-II / PP_5003-PP_5005"].iloc[0]
    assert row["source_symbol"] == "phaAZC-II"
    assert row["locus_tags"] == "PP_5003;PP_5004;PP_5005"
    assert row["annotation_symbols"] == "phaA;phaB;phaC-II"


def test_a_target_outside_the_pinned_assembly_stops_the_build(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tag the annotation does not carry is a refusal, never a dropped record."""
    from torchcell.datasets.bacteria_common import LocusTagResolutionError

    root = mv.raw_mirror_dir(str(synthetic_mirror))
    markdown = root / mv.SI1_MD_REL
    markdown.write_text(
        synthetic_markdown(round1=ROUND1_CELLS[:-1] + ("PP_9999",)), encoding="utf-8"
    )
    digest = _sha256_bytes(markdown)
    monkeypatch.setattr(mv, "SI1_MD_SHA256", digest)
    manifest = mv.load_manifest(str(synthetic_mirror))
    for record in manifest.files:
        if record.path == mv.SI1_MD_REL:
            record.sha256 = digest
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    with pytest.raises(LocusTagResolutionError):
        mv.IsoprenolSelectionMenasalvas2025Dataset(
            root=str(tmp_path / "refused"), pputida_genome=synthetic_kt2440
        )


def test_a_target_in_another_hosts_namespace_stops_the_build(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reconciled name outside the KT2440 namespace is a refusal, not a record.

    No ``PP_`` tag can reach this branch through the parser, so the reconciliation is
    stubbed: the guard exists for a future annotation whose resolution crosses hosts.
    """
    real = reconcile_locus_tags

    def crossing(genome: Any, names: Any, **kwargs: Any) -> Any:
        stored, report = real(genome, names, **kwargs)
        return stored, report.model_copy(update={"outside_namespace": ["b0002"]})

    monkeypatch.setattr(mv, "reconcile_locus_tags", crossing)
    with pytest.raises(RuntimeError, match="selected targets outside"):
        mv.IsoprenolSelectionMenasalvas2025Dataset(
            root=str(tmp_path / "refused"), pputida_genome=synthetic_kt2440
        )


def test_a_missing_mirror_file_stops_the_build(
    synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path
) -> None:
    """A mirror the manifest describes but does not hold is a refusal."""
    (mv.raw_mirror_dir(str(synthetic_mirror)) / mv.SI1_MD_REL).unlink()
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        mv.IsoprenolSelectionMenasalvas2025Dataset(
            root=str(tmp_path / "refused"), pputida_genome=synthetic_kt2440
        )


def test_a_parsed_target_count_that_is_not_the_stated_one_stops_the_build(
    built: Any, synthetic_kt2440: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record total is checked against the Methods, not taken from the file."""
    monkeypatch.setattr(mv, "EXPECTED_RECORDS", 99)
    with pytest.raises(RuntimeError, match="not the stated 99"):
        mv.IsoprenolSelectionMenasalvas2025Dataset(
            root=str(tmp_path / "refused"), pputida_genome=synthetic_kt2440
        )


def test_the_genome_is_opened_from_the_tier_when_none_is_injected(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loader built with no genome asks ``bacterial_genome`` for one."""
    calls: list[tuple[str, str]] = []

    def fake(host: str, strain: str, *args: Any, **kwargs: Any) -> Any:
        calls.append((host, strain))
        return synthetic_kt2440

    monkeypatch.setattr(mv, "bacterial_genome", fake)
    dataset = mv.IsoprenolSelectionMenasalvas2025Dataset(root=str(tmp_path / "tier"))
    assert len(dataset) == len(ROUND1_CELLS) + len(ROUND2_CELLS)
    assert calls == [("pputida", "KT2440")]


def test_the_unimplemented_interfaces_behave_as_the_family_does(built: Any) -> None:
    """``preprocess_raw`` is a pass-through and ``create_experiment`` is not used."""
    assert built.preprocess_raw("x") == "x"
    with pytest.raises(NotImplementedError):
        built.create_experiment()


# --------------------------------------------------------------------------- #
# Hermetic: the verification rows
# --------------------------------------------------------------------------- #
def test_verify_build_passes_on_the_hermetic_store(
    built: Any, synthetic_kt2440: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The L0-L4 gate is green on a synthetic build, and the report is written."""
    monkeypatch.setattr(
        mv, "bacterial_genome", lambda *args, **kwargs: synthetic_kt2440
    )
    report = mv.verify_build(built.root)
    assert report.passed, report.summary()
    names = {result.name for result in report.results}
    assert "stored_targets_are_loci_of_the_pinned_assembly" in names
    assert "assembly_pin_is_kt2440_genbank_with_no_asserted_background" in names
    assert "every_genotype_carries_the_stated_host_background" in names
    assert Path(built.preprocess_dir, "verification_report.json").is_file()


def test_the_supplementary_rows_fail_when_the_records_do_not_hold(
    built: Any, synthetic_kt2440: Any
) -> None:
    """Each supplementary rule is reachable in the failing direction."""
    from torchcell.verification.runners import load_records

    records = load_records(built.root)
    assert mv.stored_tags_are_loci(records, synthetic_kt2440).passed
    assert mv.assembly_pin(records).passed
    assert mv.host_background_present(records).passed

    broken = [
        {
            "experiment": {
                "genotype": {
                    "perturbations": [
                        {
                            "systematic_gene_name": "PP_9999",
                            "perturbation_type": "bacterial_crispr_interference",
                        }
                    ]
                }
            },
            "reference": {
                "genome_reference": {
                    "assembly_set": "ecoli_K12_MG1655_ASM584v2",
                    "assembly_accession": "GCA_000005845.2",
                    "background": None,
                }
            },
        }
    ]
    assert not mv.stored_tags_are_loci(broken, synthetic_kt2440).passed
    assert not mv.assembly_pin(broken).passed
    assert not mv.host_background_present(broken).passed


def test_main_builds_verifies_and_prints(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The module's own runner builds, prints the accounting and prints the report."""
    monkeypatch.setattr(
        mv, "bacterial_genome", lambda *args, **kwargs: synthetic_kt2440
    )
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    mv.main()
    out = capsys.readouterr().out
    assert f"len = {len(ROUND1_CELLS) + len(ROUND2_CELLS)}" in out
    assert "distinct_targets" in out
    assert "[PASS] L0 structural" in out


# --------------------------------------------------------------------------- #
# Data-gated: the numbers measured on the REAL pinned bytes
# --------------------------------------------------------------------------- #
LIBRARY_SI = osp.join(
    "/scratch/projects/torchcell-scratch/torchcell-library",
    mv.CITATION_KEY,
    "si/si1.md",
)
requires_real_si = pytest.mark.skipif(
    not osp.isfile(LIBRARY_SI), reason="requires the Menasalvas 2025 library mirror"
)


@requires_real_si
@pytest.mark.data
def test_the_real_tables_hold_the_stated_28_and_30_targets() -> None:
    """Measured on the pinned OCR: 58 rows over 60 distinct loci, rounds disjoint."""
    targets = mv.read_selected_targets(LIBRARY_SI)
    assert len(targets) == mv.EXPECTED_RECORDS == 58
    assert sum(1 for t in targets if t.round_number == 1) == 28
    assert sum(1 for t in targets if t.round_number == 2) == 30
    tags = [tag for target in targets for tag in target.locus_tags]
    assert len(tags) == 60
    assert len(set(tags)) == 60
    multi = [t for t in targets if len(t.locus_tags) > 1]
    assert [t.gene_cell for t in multi] == ["phaAZC-II / PP_5003-PP_5005"]
