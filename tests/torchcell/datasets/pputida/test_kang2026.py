# tests/torchcell/datasets/pputida/test_kang2026.py
# [[tests.torchcell.datasets.pputida.test_kang2026]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_kang2026.py
"""The Kang 2026 P. putida isoprenyl acetate titer loader.

Synthetic tests run everywhere: they exercise the standard-library ``.docx`` table
reader, the Δ86kb span reconciliation (against a stub genome placed at the real
coordinates, so the measured 63 / 1 / 4 / 6 arithmetic is pinned rather than
remembered), the titer and environment builders with the typed gaps this paper forces,
every cross-source oracle and its refusal, and a full end-to-end build over a synthetic
KT2440 assembly, a synthetic SI document and a synthetic OCR mirror -- no network and no
``$DATA_ROOT``.

The synthetic SI document reproduces the released Table S4 and Table S9 numbers exactly,
because the oracles those tables feed are joins onto OTHER statements of the same
measurement (the Results prose, the released off-gas fraction); a fixture with invented
numbers would exercise the plumbing while retiring the check.

The ``@pytest.mark.data`` tests read the real ``$DATA_ROOT``: they pin the raw mirror's
recorded digest against the module constant, re-read every sourced quote out of the two
pinned mirrors, type the chassis against the deposited KT2440 annotation, and run L0 to
L4 over the built LMDB. They are skipped unless the mirrors, the KT2440 tier cache and
the built store are all present.

Derived expectations for the pinned bytes: 19 records (7 Table 1 rows, 5 Table S4 rows,
7 Table S9 sampled times); 13 locus tags, all current; a Δ86kb coordinate span of 91,743
bp containing 67 loci entirely, of which 63 are also in the stated ``PP_4023-PP_4092``
range, with ``PP_4023`` truncated, four span-only loci and six tag numbers this assembly
does not annotate.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import pandas as pd
import pytest

import torchcell.datasets.pputida.kang2026 as kang
from torchcell.datamodels.media import (
    M9_MOPS_KANG2026,
    M9_NREL_HIGH_N_KANG2026,
    M9_NREL_KANG2026,
)
from torchcell.datamodels.schema import (
    AlleleEdit,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    PhysicalFactor,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductYieldUnit,
    SampleUnit,
    SmallMoleculePerturbation,
)
from torchcell.literature.manifest import ROLE_SI_DATA, Manifest, RetrievalMethod

DATA_ROOT = os.environ.get("DATA_ROOT", "")
_MIRROR = Path(DATA_ROOT, "torchcell-raw", kang.CITATION_KEY) if DATA_ROOT else None
_LIBRARY = (
    Path(DATA_ROOT, "torchcell-library", kang.CITATION_KEY) if DATA_ROOT else None
)
_STORE = (
    Path(DATA_ROOT, "data/torchcell/isoprenyl_acetate_titer_kang2026")
    if DATA_ROOT
    else None
)
_TIER = Path(DATA_ROOT, "data/pputida/kt2440/genome") if DATA_ROOT else None

requires_mirror = pytest.mark.skipif(
    _MIRROR is None or not (_MIRROR / kang.SI_DOCX_REL).exists(),
    reason="the Kang 2026 raw mirror is not deposited under $DATA_ROOT",
)
requires_library = pytest.mark.skipif(
    _LIBRARY is None or not (_LIBRARY / kang.PAPER_MD).exists(),
    reason="the Kang 2026 OCR mirror is not present under $DATA_ROOT",
)
requires_genome = pytest.mark.skipif(
    _TIER is None or not _TIER.exists(),
    reason="the KT2440 genome tier cache is not present under $DATA_ROOT",
)
requires_store = pytest.mark.skipif(
    _STORE is None or not (_STORE / "processed" / "lmdb").exists(),
    reason="the Kang 2026 dev-tree LMDB is not built",
)


# --------------------------------------------------------------------------- #
# A synthetic .docx, written the way the loader reads one
# --------------------------------------------------------------------------- #
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

#: Table S4 as released: the enzyme, its origin, its identifier and its tube titer.
TABLE_S4_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("ATF1", "S. cerevisiae", "NCBI: NP_015022.3", "405"),
    ("ATF2", "S. cerevisiae", "NCBI: NP_011693.1", "157"),
    ("SAAT", "Fragaria × ananassa (strawberry)", "GenBank: AAG13130.1", "22"),
    ("AAT", "Rosa hybrida", "GenBank: AY850287.1", "119"),
    ("CAT", "E. coli", "UniProtKB: P00484", "15"),
)
#: Table S9 as released: time, glucose, xylose, total sugar, aqueous isoprenol, then the
#: three isoprenyl acetate phases and the off-gas fraction.
TABLE_S9_ROWS: tuple[tuple[str, ...], ...] = (
    ("21.4", "3.8", "4.6", "8.4", "190.7", "19.4", "79.0", "0", "0"),
    ("45.4", "0.6", "2.3", "2.9", "197.0", "132.0", "890.2", "31", "2.9"),
    ("69.4", "2.2", "3.2", "5.4", "175.0", "179.2", "1172.8", "93.1", "6.4"),
    ("93.4", "2.1", "3.2", "5.2", "231.0", "248.5", "1127.8", "307.4", "18.3"),
    ("117.4", "2.7", "3.4", "6.1", "234.7", "185.6", "923", "497.6", "31.0"),
    ("141.4", "3.4", "4.0", "7.4", "307.8", "345.8", "948.4", "615.2", "32.2"),
    ("165.4", "1.0", "4.1", "5.1", "254.2", "85.7", "604.8", "744.6", "51.9"),
)


def _paragraph(text: str) -> str:
    return f'<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'


def _row(cells: tuple[str, ...]) -> str:
    return (
        "<w:tr>" + "".join(f"<w:tc>{_paragraph(c)}</w:tc>" for c in cells) + "</w:tr>"
    )


def _table(caption: str, rows: tuple[tuple[str, ...], ...]) -> str:
    return _paragraph(caption) + "<w:tbl>" + "".join(_row(r) for r in rows) + "</w:tbl>"


def write_si_docx(
    path: Path,
    *,
    s4_rows: tuple[tuple[str, str, str, str], ...] = TABLE_S4_ROWS,
    s9_rows: tuple[tuple[str, ...], ...] = TABLE_S9_ROWS,
    s9_header: tuple[str, ...] = kang.SI_TABLE9_COLUMNS,
    s4_header: tuple[str, ...] = (
        "Name",
        "Biological origin",
        "Notes",
        "Acquisition source",
        "Source and identifier",
        kang.SI_TABLE4_COLUMN,
    ),
    extra_tables: str = "",
) -> Path:
    """Write a minimal ``.docx`` carrying Table S4 and Table S9."""
    s4 = _table(
        "Table S4. Comparison of alcohol acyltransferases (AATs) for isoprenyl acetate "
        "production.",
        (
            s4_header,
            *(
                (name, origin, "Wild-type", "source", ident, titer)
                for name, origin, ident, titer in s4_rows
            ),
        ),
    )
    s9 = _table(
        "Table S9. Time-course analysis of sugar consumption and isoprenyl acetate "
        "partitioning in fed-batch fermentation",
        (s9_header, *s9_rows),
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W}"><w:body>{s4}{s9}{extra_tables}</w:body></w:document>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"/>',
        )
        archive.writestr("word/document.xml", document)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# The .docx reader
# --------------------------------------------------------------------------- #
def test_the_reader_returns_each_table_with_the_caption_above_it(
    tmp_path: Path,
) -> None:
    """A ``.docx`` body is paragraphs and tables in order; the caption is the paragraph."""
    tables = kang.read_si_tables(write_si_docx(tmp_path / "si.docx"))
    assert [table.caption.split(".")[0] for table in tables] == ["Table S4", "Table S9"]
    assert tables[0].header()[-1] == kang.SI_TABLE4_COLUMN
    assert tables[1].rows[1][0] == "21.4"


def test_the_reader_normalizes_whitespace_so_a_quote_can_be_matched(
    tmp_path: Path,
) -> None:
    """Cell text is the paragraphs joined by single spaces, which is what quotes match."""
    path = tmp_path / "si.docx"
    write_si_docx(
        path,
        extra_tables=_table(
            "Table S2. Strains and plasmids used in this study.",
            (("Name", "Description"), ("PIPA", "P. putida   KT2440\tΔphaABC")),
        ),
    )
    table = kang.si_table(path, 2)
    assert table.rows[1][1] == "P. putida KT2440 ΔphaABC"


def test_a_table_number_the_document_does_not_carry_is_refused(tmp_path: Path) -> None:
    """One table per caption prefix, or the reader refuses rather than guessing."""
    path = write_si_docx(tmp_path / "si.docx")
    with pytest.raises(RuntimeError, match="holds 0 tables captioned 'Table S7.'"):
        kang.si_table(path, 7)


def test_a_duplicated_table_caption_is_refused(tmp_path: Path) -> None:
    """Two tables captioned the same way mean the document changed shape."""
    path = tmp_path / "si.docx"
    write_si_docx(
        path,
        extra_tables=_table(
            "Table S4. Comparison of alcohol acyltransferases (AATs) again.",
            (("Name",), ("ATF1",)),
        ),
    )
    with pytest.raises(RuntimeError, match="holds 2 tables captioned"):
        kang.si_table(path, 4)


def test_a_column_the_table_does_not_carry_is_refused(tmp_path: Path) -> None:
    """A column is found by its exact header text, never by position."""
    table = kang.si_table(write_si_docx(tmp_path / "si.docx"), 4)
    with pytest.raises(RuntimeError, match="has no column 'Yield'"):
        table.column("Yield")


def test_a_table_with_no_rows_has_no_header(tmp_path: Path) -> None:
    """An empty table is a refusal, not an empty header."""
    table = kang.SiTable(caption="Table S4. empty", rows=[])
    with pytest.raises(RuntimeError, match="has no rows"):
        table.header()


# --------------------------------------------------------------------------- #
# The Δ86kb span: three statements, measured
# --------------------------------------------------------------------------- #
def _stub_genome(loci: dict[str, tuple[int, int]]) -> Any:
    """A genome stub exposing only what :func:`resolve_span` reads."""
    return SimpleNamespace(
        genbank=SimpleNamespace(loci=sorted(loci)), __getitem__=None, **{}
    )


class _StubGenome:
    """A genome stub exposing only ``genbank.loci`` and coordinate lookup."""

    def __init__(self, loci: dict[str, tuple[int, int]]) -> None:
        self.genbank = SimpleNamespace(loci=sorted(loci))
        self._loci = loci

    def __getitem__(self, tag: str) -> Any:
        start, end = self._loci[tag]
        return SimpleNamespace(start=start, end=end)


def test_the_span_report_reproduces_the_measured_three_way_disagreement() -> None:
    """The intersection is typed; everything the statements disagree on is reported."""
    genome = _StubGenome(
        {
            # straddles the left edge: named by the range, truncated by the span
            "PP_4023": (kang.SPAN_START - 605, kang.SPAN_START + 2_223),
            # fully inside and in the range
            "PP_4024": (kang.SPAN_START + 3_000, kang.SPAN_START + 4_000),
            "PP_4092": (kang.SPAN_END - 404, kang.SPAN_END - 45),
            # fully inside, out of tag order: span-only
            "PP_5652": (kang.SPAN_START + 82_062, kang.SPAN_START + 82_547),
            # outside the span and outside the range
            "PP_5351": (6_099_177, 6_100_000),
        }
    )
    stub: Any = genome
    report = kang.resolve_span(stub)
    assert report.span_length == 91_743
    assert report.full_deletions == ["PP_4024", "PP_4092"]
    assert report.partial_deletions == ["PP_4023"]
    assert report.span_only == ["PP_5652"]
    assert "PP_5351" not in report.inside_span
    # every tag number of the stated range that this stub does not annotate
    assert len(report.unannotated_tag_numbers) == (
        kang.SPAN_TAG_LAST - kang.SPAN_TAG_FIRST + 1 - 3
    )
    findings = {row["item"]: row["finding"] for row in report.disagreement_rows()}
    assert "PP_5652" in findings
    assert "truncates the gene" in findings["PP_4023"]
    assert findings["PP_4026"].startswith("inside the stated locus-tag range")


def test_the_stated_label_disagrees_with_the_stated_coordinates() -> None:
    """``Δ86kb`` against 91,743 bp: the arithmetic is pinned, not narrated."""
    assert kang.SPAN_END - kang.SPAN_START + 1 == 91_743
    assert "86kb" in kang.SPAN_DESIGNATION


def test_a_non_numeric_locus_tag_is_not_read_as_a_tag_number() -> None:
    """``PP_mr45`` carries no tag number, so the range cannot claim it."""
    assert kang._tag_number("PP_4023") == 4023
    assert kang._tag_number("PP_mr45") is None


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
def test_the_titer_is_stored_in_the_numerically_identical_microgram_unit() -> None:
    """``ConcentrationUnit`` has no mg/L, and 1 mg/L is exactly 1 ug/mL."""
    phenotype = kang.titer_phenotype(414.0, n_samples=3)
    assert phenotype.titer == 414.0
    assert phenotype.titer_unit is ConcentrationUnit.ug_per_ml
    assert "mg_per_l" not in {member.name for member in ConcentrationUnit}


def test_every_uncertainty_is_a_typed_gap_because_no_sd_was_released() -> None:
    """The design is sourced; the number is not, so neither half is half-filled."""
    phenotype = kang.titer_phenotype(414.0, n_samples=3)
    assert phenotype.titer_uncertainty is None
    assert phenotype.titer_uncertainty_type is None
    assert phenotype.titer_se is None
    assert {"titer_uncertainty", "titer_uncertainty_type"} <= phenotype.gapped_fields()
    assert phenotype.n_samples == 3
    assert phenotype.sample_unit is SampleUnit.biological_replicate


def test_the_replicate_design_is_sourced_as_biological_triplicates() -> None:
    """Three biological replicates, and the sample-SD type, both carry a quote."""
    assert kang.N_REPLICATES.value == 3
    assert "biological triplicates" in kang.N_REPLICATES.quote
    assert kang.UNCERTAINTY_TYPE_STATED.value == "sample_sd"
    assert "standard deviation" in kang.UNCERTAINTY_TYPE_STATED.quote


def test_a_fed_batch_record_gaps_the_replicate_design_it_never_had() -> None:
    """One bioreactor run states no replicate count, so n_samples is a typed absence."""
    phenotype = kang.titer_phenotype(1909.4, n_samples=None)
    assert phenotype.n_samples is None
    assert phenotype.sample_unit is None
    assert {"n_samples", "sample_unit"} <= phenotype.gapped_fields()


def test_the_released_yield_is_carried_only_where_it_was_stated() -> None:
    """0.067 g/g is stated once, for the run's maximum titer."""
    with_yield = kang.titer_phenotype(1909.4, n_samples=None, product_yield=0.067)
    assert with_yield.product_yield == 0.067
    assert with_yield.product_yield_unit is ProductYieldUnit.g_per_g_substrate
    assert "product_yield" not in with_yield.gapped_fields()
    without = kang.titer_phenotype(1435.1, n_samples=None)
    assert {"product_yield", "product_yield_unit"} <= without.gapped_fields()


def test_the_product_carries_the_canonical_name_and_an_inchikey_gap() -> None:
    """No compound-identity row exists, so the key stays in the module as a constant."""
    product = kang.isoprenyl_acetate()
    assert product.name == kang.PRODUCT_NAME
    assert product.inchikey is None
    assert "inchikey" in product.gapped_fields()
    assert kang.ISOPRENYL_ACETATE_INCHIKEY == "OCUAPVNNQFAQSM-UHFFFAOYSA-N"


def test_the_quantification_method_is_the_sourced_gc_fid() -> None:
    """One stored titer is a GC-FID measurement, and the Methods say so."""
    phenotype = kang.titer_phenotype(414.0, n_samples=3)
    assert phenotype.quantification_method == "GC-FID"
    assert "GC-FID" in kang.QUANTIFICATION.quote


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def test_a_mixed_sugar_condition_carries_one_carbon_factor_per_sugar() -> None:
    """The Kang media are carbon-source free, so each sugar is a physical factor."""
    env = kang.environment(kang.CONDITIONS["flask_m9hn_glcxyl_1010"])
    assert env.media is M9_NREL_HIGH_N_KANG2026
    carbon = [
        pert
        for pert in env.perturbations
        if isinstance(pert, EnvironmentPhysicalPerturbation)
        and pert.factor is PhysicalFactor.carbon_source
    ]
    magnitudes = [pert.magnitude for pert in carbon]
    agents = [pert.agent for pert in carbon]
    assert all(value is not None for value in magnitudes)
    assert all(agent is not None for agent in agents)
    assert [value.value for value in magnitudes if value] == [10.0, 10.0]
    assert [agent.name for agent in agents if agent] == ["D-glucose", "xylose"]


def test_a_table1_condition_gaps_the_duration_it_never_states() -> None:
    """The primary column is a maximum over a 24 h-sampled time course."""
    env = kang.environment(kang.CONDITIONS["flask_m9_glc20"])
    assert env.duration_hours is None
    assert "duration_hours" in env.gapped_fields()
    assert env.temperature is not None
    assert env.temperature.value == 30.0
    assert env.aerobicity == "aerobic"


def test_the_one_table1_strain_with_a_stated_time_carries_it() -> None:
    """599 mg/L at 144 h is the only Table 1 row whose time the source gives."""
    env = kang.environment(kang.CONDITIONS["flask_m9hn_glcxyl_1010_pulse"])
    assert env.duration_hours == 144.0
    assert "duration_hours" not in env.gapped_fields()


def test_a_fed_batch_time_point_overrides_the_conditions_duration() -> None:
    """Each sampled time is its own environment, which is what separates the records."""
    env = kang.environment(
        kang.CONDITIONS["fedbatch_mops_ye_glcxyl_1337"], duration_hours=141.4
    )
    assert env.duration_hours == 141.4
    assert env.media is M9_MOPS_KANG2026


def test_the_mops_conditions_carry_the_stated_additions() -> None:
    """Yeast extract, the raised ammonium, both inducers and the Durasyn overlay."""
    env = kang.environment(kang.CONDITIONS["flask_mops_ye_glcxyl_137_pulse"])
    doses = {
        pert.compound.name: (pert.concentration.value, pert.concentration.unit)
        for pert in env.perturbations
        if isinstance(pert, SmallMoleculePerturbation)
    }
    assert doses["yeast extract"] == (1.0, ConcentrationUnit.g_per_l)
    assert doses["ammonium sulfate"] == (40.0, ConcentrationUnit.millimolar)
    assert doses["L-arabinose"] == (2.0, ConcentrationUnit.g_per_l)
    assert doses["salicylic acid"] == (62.5, ConcentrationUnit.micromolar)
    assert doses["m-toluic acid"] == (250.0, ConcentrationUnit.micromolar)
    assert doses["Durasyn 164"] == (20.0, ConcentrationUnit.percent_v_v)


def test_the_tube_condition_is_the_48_hour_m9_glucose_one() -> None:
    """Table S4's panel is a 5 mL tube endpoint, not a flask maximum."""
    env = kang.environment(kang.CONDITIONS["tube_m9_glc20"])
    assert env.media is M9_NREL_KANG2026
    assert env.duration_hours == 48.0
    assert kang.CULTURE_FORMATS["tube"]["working_volume_ml"] == 5.0


def test_every_condition_carries_its_own_provenance_quotes() -> None:
    """The environment a record holds is auditable condition by condition."""
    assert set(kang.CONDITION_PROVENANCE) == set(kang.CONDITIONS)
    for values in kang.CONDITION_PROVENANCE.values():
        assert values
        for value in values:
            assert value.provenance.sha256
            assert value.quote


# --------------------------------------------------------------------------- #
# Genotype
# --------------------------------------------------------------------------- #
def test_a_cassette_gene_must_appear_in_its_own_quoted_construct_string() -> None:
    """A changed quote cannot drift away from the identifiers it justifies."""
    broken = kang.PIY670.model_copy(update={"construct_string": "pRK2-Kan-araC-PBAD"})
    with pytest.raises(RuntimeError, match="is not in the quoted pIY670 description"):
        kang.cassette_perturbations(broken)


def test_the_xylab_operon_shorthand_is_how_two_genes_share_a_token() -> None:
    """The source writes the adjacent xylA and xylB as ``xylAB``."""
    tokens = {gene.token: gene.construct_token for gene in kang.XYL_CASSETTE_SPEC.genes}
    assert tokens["xylA"] == "xylAB"
    assert tokens["xylB"] == "xylAB"
    assert tokens["xylE"] is None
    perturbations = kang.cassette_perturbations(kang.XYL_CASSETTE_SPEC)
    assert [p.systematic_gene_name for p in perturbations] == [
        "xylE",
        "xylA",
        "xylB",
        "talB",
        "tktA",
    ]
    assert {p.localization for p in perturbations} == {"chromosomal_integration"}
    assert {p.integration_locus for p in perturbations} == {"PP_1444"}


def test_the_pathway_genes_carry_their_sourced_organisms_and_promoters() -> None:
    """The part order is where the promoter assignment comes from."""
    by_token = {gene.token: gene for gene in kang.PIY670.genes}
    assert by_token["MvaSef"].promoter == "PBAD"
    assert by_token["AphA"].promoter == "Ptrc1-O"
    assert by_token["MvaSef"].organism == "Enterococcus faecalis"
    assert by_token["MKmm"].organism == "Methanosarcina mazei"
    assert by_token["PMDHKQ"].variant == "HKQ"
    assert by_token["PMDHKQ"].organism_source == "carruthers"
    assert by_token["AphA"].organism_source == "paper"


def test_the_pmd_organism_is_sourced_by_deferral_to_a_mirrored_paper() -> None:
    """Kang names no organism for PMD; the mirrored sibling campaign does."""
    assert kang.CARRUTHERS_KEY != kang.CITATION_KEY
    assert "S. cerevisiae" in kang._Q_PMD_CARRUTHERS
    deferred = kang._carruthers("x", kang._Q_PMD_CARRUTHERS, page="Methods")
    assert deferred.provenance.citation_key == kang.CARRUTHERS_KEY
    assert deferred.provenance.sha256 == kang.CARRUTHERS_PAPER_MD_SHA256


def test_the_final_strain_genotype_is_every_edit_on_top_of_pipa() -> None:
    """Seven deletions plus four cassettes, and nothing from the background repeated."""
    genotype = kang.strain_genotype("PIPAxyl-E3-K3-O15", {})
    deletions = [
        pert
        for pert in genotype.perturbations
        if pert.perturbation_type == "bacterial_deletion"
    ]
    assert sorted(p.systematic_gene_name for p in deletions) == [
        "PP_1021",
        "PP_1127",
        "PP_1444",
        "PP_2876",
        "PP_3812",
        "PP_4218",
        "PP_5292",
    ]
    added = [
        pert
        for pert in genotype.perturbations
        if pert.perturbation_type == "heterologous_pathway"
    ]
    assert len(added) == 5 + 2 + 5 + 1
    assert {p.construct_name for p in added} == {"pXyl", "pO15", "pIY670", "pAAT1"}


def test_the_source_symbol_wins_over_the_annotations_where_the_source_gives_one() -> (
    None
):
    """Table 1 writes ``Δcrc (PP_5292)``; the annotation gives PP_5292 no symbol."""
    genotype = kang.strain_genotype("PIPAxyl-E3-K3", {"PP_5292": "something-else"})
    by_tag = {
        pert.systematic_gene_name: pert.perturbed_gene_name
        for pert in genotype.perturbations
        if pert.perturbation_type == "bacterial_deletion"
    }
    assert by_tag["PP_5292"] == "crc"
    assert by_tag["PP_1021"] == "hexR"


def test_a_locus_the_source_names_only_by_tag_takes_the_annotations_symbol() -> None:
    """PP_1127 is written as a bare tag, and the annotation carries ``estC``."""
    genotype = kang.strain_genotype("PIPA-E3", {"PP_1127": "estC"})
    by_tag = {
        pert.systematic_gene_name: pert.perturbed_gene_name
        for pert in genotype.perturbations
        if pert.perturbation_type == "bacterial_deletion"
    }
    assert by_tag["PP_1127"] == "estC"
    assert by_tag["PP_3812"] == "PP_3812"


def test_the_expected_perturbation_count_is_derived_from_the_strain_table() -> None:
    """The level that checks the store derives its oracle from the declarative table."""
    assert kang._expected_perturbations("PIPA-AAT1") == 6
    assert kang._expected_perturbations("PIPAxyl-E3-K3-O15") == 7 + 13


# --------------------------------------------------------------------------- #
# The three titer columns, and their internal consistency
# --------------------------------------------------------------------------- #
def test_the_primary_column_is_the_seven_table1_rows() -> None:
    """Table 1's titer column, in table order, with the trajectory it records."""
    assert [row.strain for row in kang.TABLE1_ROWS] == [
        "PIPA-AAT1",
        "PIPA-E3",
        "PIPAxyl-AAT1",
        "PIPAxyl-E3",
        "PIPAxyl-E3-K3",
        "PIPAxyl-E3-O15",
        "PIPAxyl-E3-K3-O15",
    ]
    assert [row.titer_mg_per_l for row in kang.TABLE1_ROWS] == [
        414.0,
        724.0,
        181.0,
        383.0,
        599.0,
        819.0,
        1500.0,
    ]
    for row in kang.TABLE1_ROWS:
        assert str(int(row.titer_mg_per_l)) == row.condition_quote.split(";")[0]
        assert row.condition in kang.CONDITIONS
        assert row.strain in kang.STRAIN_SPECS
        assert row.strain in kang.TABLE1_REFERENCE


def test_the_record_count_is_the_three_columns_added_up() -> None:
    """19 = 7 Table 1 + 5 Table S4 + 7 Table S9."""
    assert kang.EXPECTED_RECORDS == 19
    assert len(kang.TABLE1_ROWS) == 7
    assert len(kang.AAT_CASSETTES) == 5
    assert len(TABLE_S9_ROWS) == 7


def test_every_strain_spec_names_a_defined_cassette() -> None:
    """A strain spec cannot name a cassette object that does not exist."""
    for spec in kang.STRAIN_SPECS.values():
        for name in spec.cassettes:
            assert name in kang.CASSETTES
        for tag in spec.deletions:
            assert tag in kang.DELETION_SYMBOLS
            assert tag in kang.DELETION_QUOTES


def test_the_strains_with_no_titer_are_recorded_rather_than_forgotten() -> None:
    """Five Table 1 strains carry no number, and none of them is a record."""
    named = {name for name, _ in kang.STRAINS_WITHOUT_A_TITER}
    assert len(named) == 5
    assert named.isdisjoint({row.strain for row in kang.TABLE1_ROWS})


def test_the_panel_reader_joins_the_enzyme_order_to_the_plasmid_order(
    tmp_path: Path,
) -> None:
    """Table S4 names the enzyme; Table S2's plasmid order is what names the strain."""
    rows = kang.read_aat_panel(write_si_docx(tmp_path / "si.docx"))
    assert [row.strain for row in rows] == [f"PIPA-AAT{n}" for n in range(1, 6)]
    assert [row.titer_mg_per_l for row in rows] == [405.0, 157.0, 22.0, 119.0, 15.0]
    assert rows[2].organism.startswith("Fragaria")
    assert rows[4].identifier == "UniProtKB: P00484"


def test_an_enzyme_the_module_does_not_type_is_refused(tmp_path: Path) -> None:
    """A sixth AAT means the panel changed and the strain join is no longer sound."""
    path = write_si_docx(
        tmp_path / "si.docx",
        s4_rows=(("ATF9", "S. cerevisiae", "NCBI: x", "1"), *TABLE_S4_ROWS[1:]),
    )
    with pytest.raises(RuntimeError, match="names the enzyme 'ATF9'"):
        kang.read_aat_panel(path)


def test_a_reordered_panel_is_refused_rather_than_mis_joined(tmp_path: Path) -> None:
    """Row n must be pAAT<n>, or the enzyme-to-strain join is silently wrong."""
    reordered = (TABLE_S4_ROWS[1], TABLE_S4_ROWS[0], *TABLE_S4_ROWS[2:])
    path = write_si_docx(tmp_path / "si.docx", s4_rows=reordered)
    with pytest.raises(RuntimeError, match="plasmid order no longer matches"):
        kang.read_aat_panel(path)


def test_the_fed_batch_titer_is_the_sum_of_the_three_released_phases(
    tmp_path: Path,
) -> None:
    """Aqueous + organic + off-gas, which is the total the authors' own column uses."""
    rows = kang.read_fed_batch(write_si_docx(tmp_path / "si.docx"))
    assert len(rows) == 7
    assert rows[5].time_hours == 141.4
    assert rows[5].titer_mg_per_l == pytest.approx(1909.4)
    assert max(row.titer_mg_per_l for row in rows) == pytest.approx(1909.4)


def test_a_changed_table_s9_header_is_refused(tmp_path: Path) -> None:
    """The column meanings are asserted before a single value is read."""
    header = ("Time (h)", *kang.SI_TABLE9_COLUMNS[1:])
    path = write_si_docx(tmp_path / "si.docx", s9_header=("t", *header[1:]))
    with pytest.raises(RuntimeError, match="Table S9 header changed"):
        kang.read_fed_batch(path)


# --------------------------------------------------------------------------- #
# The cross-source oracles and their refusals
# --------------------------------------------------------------------------- #
def _fed_batch_rows() -> list[kang.FedBatchRow]:
    return [
        kang.FedBatchRow(
            time_hours=float(row[0]),
            glucose_g_per_l=float(row[1]),
            xylose_g_per_l=float(row[2]),
            total_sugar_g_per_l=float(row[3]),
            aqueous_mg_per_l=float(row[5]),
            organic_mg_per_l=float(row[6]),
            offgas_mg_per_l=float(row[7]),
            released_offgas_fraction_percent=float(row[8]),
        )
        for row in TABLE_S9_ROWS
    ]


def test_the_released_off_gas_fraction_confirms_the_stored_total() -> None:
    """The oracle: off-gas / the three-phase sum IS the released percent column."""
    rows = _fed_batch_rows()
    kang.IsoprenylAcetateTiterKang2026Dataset._assert_fed_batch_oracles(rows)
    derived = [
        0.0
        if row.titer_mg_per_l == 0
        else row.offgas_mg_per_l / row.titer_mg_per_l * 100.0
        for row in rows
    ]
    assert derived == pytest.approx(
        [row.released_offgas_fraction_percent for row in rows], abs=0.05
    )
    assert derived[-1] == pytest.approx(51.9, abs=0.05)


def test_an_off_gas_fraction_the_sum_cannot_reproduce_is_refused() -> None:
    """If the percent column disagrees, the sum is not the authors' total."""
    rows = _fed_batch_rows()
    rows[3] = rows[3].model_copy(update={"released_offgas_fraction_percent": 42.0})
    with pytest.raises(RuntimeError, match="is not the total the authors used"):
        kang.IsoprenylAcetateTiterKang2026Dataset._assert_fed_batch_oracles(rows)


def test_a_total_sugar_column_that_is_not_the_sum_of_its_parts_is_refused() -> None:
    """One decimal of rounding is tolerated; a real disagreement is not."""
    rows = _fed_batch_rows()
    rows[0] = rows[0].model_copy(update={"total_sugar_g_per_l": 20.0})
    with pytest.raises(RuntimeError, match="the released total reads 20.0"):
        kang.IsoprenylAcetateTiterKang2026Dataset._assert_fed_batch_oracles(rows)


def test_the_measured_rounding_tolerance_is_what_one_real_row_needs() -> None:
    """93.4 h: 2.1 + 3.2 = 5.3 against a released 5.2, which is three roundings."""
    assert kang._SUGAR_ROUNDING_TOL == 0.15
    rows = _fed_batch_rows()
    assert rows[3].glucose_g_per_l + rows[3].xylose_g_per_l == pytest.approx(5.3)
    assert rows[3].total_sugar_g_per_l == 5.2


def test_a_peak_that_is_not_the_stated_final_titer_is_refused() -> None:
    """The Results state 1.9 g/L; the sum's maximum has to be that number."""
    rows = [
        row.model_copy(update={"organic_mg_per_l": 0.0}) for row in _fed_batch_rows()
    ]
    rows = [
        row.model_copy(
            update={
                "released_offgas_fraction_percent": (
                    0.0
                    if row.titer_mg_per_l == 0
                    else row.offgas_mg_per_l / row.titer_mg_per_l * 100.0
                )
            }
        )
        for row in rows
    ]
    with pytest.raises(RuntimeError, match="the Results state 1900.0 mg/L"):
        kang.IsoprenylAcetateTiterKang2026Dataset._assert_fed_batch_oracles(rows)


def test_an_empty_table_s9_is_refused() -> None:
    """No sampled times means the table is not the one this loader reads."""
    with pytest.raises(RuntimeError, match="holds no sampled times"):
        kang.IsoprenylAcetateTiterKang2026Dataset._assert_fed_batch_oracles([])


def _panel_rows() -> list[kang.AatPanelRow]:
    return [
        kang.AatPanelRow(
            enzyme=name,
            strain=f"PIPA-AAT{index}",
            organism=origin,
            identifier=ident,
            titer_mg_per_l=float(titer),
        )
        for index, (name, origin, ident, titer) in enumerate(TABLE_S4_ROWS, start=1)
    ]


def test_the_panels_best_enzyme_is_the_one_the_results_name() -> None:
    """ATF1, and its tube titer is below Table 1's flask maximum for the same strain."""
    rows = _panel_rows()
    kang.IsoprenylAcetateTiterKang2026Dataset._assert_panel_agrees_with_table1(rows)
    best = max(rows, key=lambda row: row.titer_mg_per_l)
    assert best.enzyme == "ATF1"
    assert best.titer_mg_per_l == 405.0
    assert best.titer_mg_per_l < kang.TABLE1_ROWS[0].titer_mg_per_l


def test_a_panel_whose_best_enzyme_is_not_atf1_is_refused() -> None:
    """The Results say ATF1 was highest, so a different winner means a changed table."""
    rows = _panel_rows()
    rows[1] = rows[1].model_copy(update={"titer_mg_per_l": 999.0})
    with pytest.raises(RuntimeError, match="Table S4's highest titer is ATF2"):
        kang.IsoprenylAcetateTiterKang2026Dataset._assert_panel_agrees_with_table1(rows)


def test_a_tube_titer_above_the_flask_maximum_is_refused() -> None:
    """Both numbers are kept, and the tube endpoint must be the lower of the two."""
    rows = _panel_rows()
    rows[0] = rows[0].model_copy(update={"titer_mg_per_l": 500.0})
    with pytest.raises(RuntimeError, match="the tube endpoint is expected to be"):
        kang.IsoprenylAcetateTiterKang2026Dataset._assert_panel_agrees_with_table1(rows)


# --------------------------------------------------------------------------- #
# Retention bookkeeping and the record join
# --------------------------------------------------------------------------- #
def test_build_accounting_refuses_counts_that_do_not_add_up() -> None:
    """Kept + dropped must equal the candidates."""
    accounting = kang.BuildAccounting(
        dataset="probe",
        source_rows=19,
        candidate_records=19,
        kept_records=18,
        dropped_records=0,
    )
    with pytest.raises(RuntimeError, match="!= 19 candidates"):
        accounting.check()


def test_build_accounting_refuses_a_drop_no_rule_accounts_for() -> None:
    """A record that left the build without a named reason is a refusal."""
    accounting = kang.BuildAccounting(
        dataset="probe",
        source_rows=19,
        candidate_records=19,
        kept_records=18,
        dropped_records=1,
    )
    with pytest.raises(RuntimeError, match="rules total 0, 1 records are missing"):
        accounting.check()


def test_the_record_join_key_is_the_titer_and_the_duration_not_the_position() -> None:
    """LMDB keys are strings, so "10" sorts before "2" and a positional join misreads."""
    assert kang._record_signature(414.0, None) == (414.0, -1.0)
    assert kang._record_signature(1909.4, 141.4) == (1909.4, 141.4)
    assert kang._record_signature(414.0, None) != kang._record_signature(414.0, 48.0)


def _record(
    titer: float, hours: float | None, perturbations: list[Any]
) -> dict[str, Any]:
    return {
        "experiment": {
            "phenotype": {"titer": titer},
            "environment": {"duration_hours": hours},
            "genotype": {"perturbations": perturbations},
        }
    }


def test_a_ledger_with_a_repeated_signature_is_refused() -> None:
    """Two rows with one key would pair records to the wrong released column."""
    ledger = pd.DataFrame(
        [
            {"titer_mg_per_l": 414.0, "time_hours": None, "source": "Table 1"},
            {"titer_mg_per_l": 414.0, "time_hours": None, "source": "Table S4"},
        ]
    )
    records = [_record(414.0, None, []), _record(414.0, None, [])]
    with pytest.raises(RuntimeError, match="share the signature"):
        kang._records_by_source(records, ledger)


def test_a_store_record_no_ledger_row_carries_is_refused() -> None:
    """The ledger is written from the same rows the records were put from."""
    ledger = pd.DataFrame(
        [{"titer_mg_per_l": 414.0, "time_hours": None, "source": "Table 1"}]
    )
    with pytest.raises(RuntimeError, match="which no ledger row carries"):
        kang._records_by_source([_record(999.0, None, [])], ledger)


def test_a_record_count_that_differs_from_the_ledger_is_refused() -> None:
    """The store and its own build ledger cannot disagree on how many rows there are."""
    ledger = pd.DataFrame(
        [{"titer_mg_per_l": 414.0, "time_hours": None, "source": "Table 1"}]
    )
    with pytest.raises(RuntimeError, match="0 records and 1 ledger rows"):
        kang._records_by_source([], ledger)


def test_a_genotype_that_matches_no_strain_spec_is_refused() -> None:
    """A record names no strain, so the join back to the table must be exact."""
    with pytest.raises(RuntimeError, match="matches 0 strain specs"):
        kang._match_strain(
            _record(
                1.0,
                None,
                [
                    {
                        "perturbation_type": "bacterial_deletion",
                        "systematic_gene_name": "PP_9999",
                    }
                ],
            )
        )


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
def test_the_pmc_cloud_key_names_this_articles_bucket_prefix() -> None:
    """The recorded retrieval re-runs as-is against the PMC Article Datasets bucket."""
    assert kang.pmc_cloud_key() == f"{kang.PMCID}.1/mmc1.docx"
    assert kang.PMC_PREFIX == "PMC12996797.1"


def test_manifest_sha256_raises_on_a_path_the_manifest_does_not_list() -> None:
    """A pin can only be read for a file the mirror actually records."""
    manifest = Manifest(
        citation_key=kang.CITATION_KEY, doi=kang.DOI, title=kang.TITLE, files=[]
    )
    with pytest.raises(KeyError, match="is not in the raw-mirror manifest"):
        kang.manifest_sha256(manifest, kang.SI_DOCX_REL)


def test_the_deposit_is_idempotent_by_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching mirror file is left alone and the manifest records its retrieval."""
    source = write_si_docx(tmp_path / "source" / "si1.docx")
    monkeypatch.setattr(kang, "SI_DOCX_SHA256", _sha256(source))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    root = kang.deposit_raw_mirror(si_docx_path=source)
    first = (root / kang.SI_DOCX_REL).stat().st_mtime_ns
    kang.deposit_raw_mirror(si_docx_path=source)
    assert (root / kang.SI_DOCX_REL).stat().st_mtime_ns == first
    manifest = kang.load_manifest()
    assert len(manifest.files) == 1
    record = manifest.files[0]
    assert record.role == ROLE_SI_DATA
    assert record.retrieval is not None
    assert record.retrieval.method is RetrievalMethod.pmc_cloud
    assert record.original_filename == "mmc1.docx"
    assert any("PRIDE" in line for line in manifest.si_expected)


def test_a_differing_mirror_file_is_refused_rather_than_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream drift is detected, never silently followed."""
    source = write_si_docx(tmp_path / "source" / "si1.docx")
    monkeypatch.setattr(kang, "SI_DOCX_SHA256", _sha256(source))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    root = kang.deposit_raw_mirror(si_docx_path=source)
    (root / kang.SI_DOCX_REL).write_bytes(b"different bytes")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        kang.deposit_raw_mirror(si_docx_path=source)


# --------------------------------------------------------------------------- #
# The Table 1 quote audit
# --------------------------------------------------------------------------- #
def _synthetic_paper_md() -> str:
    """An OCR stand-in carrying every quote the loader reads out of ``paper.md``."""
    parts = [kang._Q_TABLE1_FOOTNOTE]
    parts.extend(row.condition_quote for row in kang.TABLE1_ROWS)
    parts.extend(
        kang.STRAIN_DESCRIPTIONS[row.strain]
        for row in kang.TABLE1_ROWS
        if kang.STRAIN_DESCRIPTION_SOURCE[row.strain] == "table1"
    )
    parts.extend(quote for _, quote in kang.TABLE1_PROSE_TITERS.values())
    return "\n\n".join(parts)


def _write_library(root: Path, text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    path = root / "torchcell-library" / kang.CITATION_KEY / kang.PAPER_MD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(kang, "PAPER_MD_SHA256", _sha256(path))


def test_the_quote_audit_hashes_the_ocr_before_it_trusts_a_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Table 1 is the one consumed column with no released data file behind it."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_library(tmp_path, _synthetic_paper_md(), monkeypatch)
    assert kang.verify_paper_quotes() == 15


def test_a_drifted_ocr_hash_stops_the_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed mirror is upstream drift, and every quote must be re-read."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_library(tmp_path, _synthetic_paper_md(), monkeypatch)
    monkeypatch.setattr(kang, "PAPER_MD_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="the OCR mirror moved"):
        kang.verify_paper_quotes()


def test_a_quote_the_ocr_no_longer_carries_stops_the_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Table 1 value whose quote vanished is not a value this loader will store."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    text = _synthetic_paper_md().replace(kang.TABLE1_ROWS[0].condition_quote, "")
    _write_library(tmp_path, text, monkeypatch)
    with pytest.raises(RuntimeError, match="Table 1 quotes are not verbatim"):
        kang.verify_paper_quotes()


# --------------------------------------------------------------------------- #
# Hermetic end to end: a synthetic KT2440 assembly, a synthetic SI document and a
# synthetic OCR mirror, with the loader built under tmp_path. No network, no $DATA_ROOT.
#
# The real tier cannot be used on a CI runner, so the real KT2440 class is built over a
# short synthetic replicon carrying the thirteen loci this loader names. ``phaC`` and
# ``crc`` are deliberately absent as SYMBOLS, which is exactly what the chassis builder
# asserts before it types them from Table 1's locus tags. The synthetic loci sit at low
# coordinates, so the Δ86kb span types nothing here; that arithmetic is pinned by
# ``test_the_span_report_reproduces_the_measured_three_way_disagreement`` instead.
# --------------------------------------------------------------------------- #
LOCUS_SPECS: tuple[tuple[str, str | None], ...] = (
    ("PP_1021", "hexR"),
    ("PP_1127", "estC"),
    ("PP_1444", "gcd"),
    ("PP_1649", "ldhA"),
    ("PP_2876", "ampC"),
    ("PP_3073", "hbdH"),
    ("PP_3540", "mvaB"),
    ("PP_3812", None),
    ("PP_4218", None),
    ("PP_5003", "phaA"),
    ("PP_5004", "phaB"),
    ("PP_5005", "phaC-II"),
    ("PP_5292", None),
)
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


def _synthetic_loci() -> list[Any]:
    """One :class:`SyntheticLocus` per :data:`LOCUS_SPECS` entry, laid end to end."""
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
        [fixtures.gaf_row("hbdH", "hbdH|PP_3073", "GO:0000001")],
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


def test_the_chassis_types_the_six_named_deletions_against_the_annotation(
    synthetic_kt2440: Any,
) -> None:
    """Table 1's locus tag per gene, and the annotation has to agree where it can."""
    background, span = kang.chassis_background(synthetic_kt2440)
    assert background.name == "PIPA"
    assert background.reference_strain == "KT2440"
    assert background.parents == ["KT2440"]
    assert background.genotype_statement == kang._Q_PIPA_SI
    by_tag = {allele.systematic_gene_name: allele for allele in background.alleles}
    assert sorted(by_tag) == [
        "PP_1649",
        "PP_3073",
        "PP_3540",
        "PP_5003",
        "PP_5004",
        "PP_5005",
    ]
    assert by_tag["PP_5005"].gene_name == "phaC"
    assert by_tag["PP_5005"].allele_name == "ΔphaABC"
    assert all(allele.edit is AlleleEdit.full_deletion for allele in by_tag.values())
    assert all(not allele.functional for allele in by_tag.values())
    assert all(allele.is_sourced for allele in by_tag.values())
    assert background.is_fully_sourced
    assert span.full_deletions == []


def test_phac_and_crc_carry_no_symbol_of_this_assembly(synthetic_kt2440: Any) -> None:
    """The documented reason Table 1's locus tags are load-bearing, asserted."""
    assert synthetic_kt2440.resolve_gene_name("phaC").status.value == "retired"
    assert synthetic_kt2440.resolve_gene_name("crc").status.value == "retired"
    symbols = kang._annotation_symbols(synthetic_kt2440)
    assert symbols["PP_5005"] == "phaC-II"
    assert "PP_5292" not in symbols


def test_a_table1_locus_tag_the_assembly_lacks_stops_the_background(
    synthetic_kt2440: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tag that is not a locus cannot carry an allele."""
    moved = (
        kang.ChassisDeletion(
            symbol="phaA",
            locus_tag="PP_9999",
            designation="ΔphaABC",
            symbol_resolves=True,
        ),
    )
    monkeypatch.setattr(kang, "CHASSIS_DELETIONS_TYPED", moved)
    with pytest.raises(RuntimeError, match="which is not a locus of"):
        kang.chassis_background(synthetic_kt2440)


def test_a_symbol_the_annotation_sends_elsewhere_stops_the_background(
    synthetic_kt2440: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where both the paper and the annotation name a locus, they must agree."""
    wrong = (
        kang.ChassisDeletion(
            symbol="phaA",
            locus_tag="PP_5004",
            designation="ΔphaABC",
            symbol_resolves=True,
        ),
    )
    monkeypatch.setattr(kang, "CHASSIS_DELETIONS_TYPED", wrong)
    with pytest.raises(RuntimeError, match="cannot be typed while the two disagree"):
        kang.chassis_background(synthetic_kt2440)


def test_a_symbol_that_starts_resolving_stops_the_documented_exception(
    synthetic_kt2440: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``phaC`` ever resolves, the reason recorded for PP_5005 is no longer true."""
    now_resolving = (
        kang.ChassisDeletion(
            symbol="phaA",
            locus_tag="PP_5003",
            designation="ΔphaABC",
            symbol_resolves=False,
        ),
    )
    monkeypatch.setattr(kang, "CHASSIS_DELETIONS_TYPED", now_resolving)
    with pytest.raises(RuntimeError, match="must become a symbol-resolved allele"):
        kang.chassis_background(synthetic_kt2440)


def test_the_reference_pins_the_genbank_assembly_and_survives_serialization(
    synthetic_kt2440: Any,
) -> None:
    """The assembly pin is only real where the field is narrowed; here it is."""
    reference = kang.chassis_reference(synthetic_kt2440)
    assert reference.species == "Pseudomonas putida"
    assert reference.strain == "PIPA"
    assert reference.assembly_set == "pputida_KT2440_ASM756v2"
    assert reference.assembly_accession == "GCA_000007565.2"
    dumped = ProductTiterExperimentReference(
        dataset_name="probe",
        genome_reference=reference,
        environment_reference=kang.environment(kang.CONDITIONS["flask_m9_glc20"]),
        phenotype_reference=kang.titer_phenotype(414.0, n_samples=3),
    ).model_dump()
    assert dumped["genome_reference"]["assembly_accession"] == "GCA_000007565.2"
    assert dumped["genome_reference"]["background"]["name"] == "PIPA"


@pytest.fixture
def built_store(
    synthetic_kt2440: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Deposit a synthetic mirror and build the loader end to end under ``tmp_path``."""
    data_root = tmp_path / "root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    source = write_si_docx(tmp_path / "source" / "si1.docx")
    monkeypatch.setattr(kang, "SI_DOCX_SHA256", _sha256(source))
    kang.deposit_raw_mirror(si_docx_path=source)
    _write_library(data_root, _synthetic_paper_md(), monkeypatch)
    root = str(data_root / "data/torchcell/isoprenyl_acetate_titer_kang2026")
    kang.IsoprenylAcetateTiterKang2026Dataset(
        root=root, pputida_genome=synthetic_kt2440
    )
    return root


def test_the_built_store_holds_one_record_per_released_number(built_store: str) -> None:
    """Nineteen records, and the three columns are all present in the ledger."""
    from torchcell.verification.runners import load_records

    records = load_records(built_store)
    assert len(records) == kang.EXPECTED_RECORDS
    ledger = pd.read_csv(osp.join(built_store, "preprocess", "titer_rows.csv"))
    assert dict(ledger["source"].value_counts()) == {
        "Table 1": 7,
        "Table S4": 5,
        "Table S9": 7,
    }
    assert set(ledger["column"]) == {
        "Titer (mg/L); culture conditions",
        kang.SI_TABLE4_COLUMN,
        kang.SI_TABLE9_SUM_LABEL,
    }
    for record in records:
        ProductTiterExperiment.model_validate(record["experiment"])


def test_the_built_store_passes_l0_to_l4(built_store: str) -> None:
    """The module's own runner, over the store it just built."""
    report = kang.verify_build(built_store, os.environ["DATA_ROOT"])
    assert report.passed, report.summary()
    assert {result.level.name for result in report.results} == {
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    }
    written = json.loads(
        Path(built_store, "preprocess", "verification_report.json").read_text()
    )
    assert written["provenance"]["citation_key"] == kang.CITATION_KEY


def test_the_build_accounting_records_zero_drops(built_store: str) -> None:
    """Every released isoprenyl acetate number in the mirror is a record."""
    accounting = json.loads(
        Path(built_store, "preprocess", "build_accounting.json").read_text()
    )
    assert accounting["candidate_records"] == kang.EXPECTED_RECORDS
    assert accounting["kept_records"] == kang.EXPECTED_RECORDS
    assert accounting["dropped_records"] == 0
    assert accounting["rules"] == []
    histogram = accounting["reconciliation"]["status_histogram"]
    assert histogram["current"] == 13
    assert sum(histogram.values()) == 13


def test_the_build_writes_the_four_audit_files(built_store: str) -> None:
    """The ledger, the strains with no titer, the span report and the condition quotes."""
    preprocess = Path(built_store, "preprocess")
    for name in (
        "titer_rows.csv",
        "strains_without_a_titer.csv",
        "span_disagreement.csv",
        "condition_provenance.json",
    ):
        assert (preprocess / name).exists(), name
    assert pd.read_csv(preprocess / "span_disagreement.csv").columns.tolist() == list(
        kang.SpanResolution.DISAGREEMENT_COLUMNS
    )
    assert len(pd.read_csv(preprocess / "strains_without_a_titer.csv")) == 5
    quotes = json.loads((preprocess / "condition_provenance.json").read_text())
    assert set(quotes) == set(kang.CONDITIONS)


def test_the_fed_batch_records_differ_only_in_their_sampled_time(
    built_store: str,
) -> None:
    """Seven records of one strain, separated by ``duration_hours``."""
    from torchcell.verification.runners import load_records

    ledger = pd.read_csv(osp.join(built_store, "preprocess", "titer_rows.csv"))
    grouped = kang._records_by_source(load_records(built_store), ledger)
    fed_batch = grouped["Table S9"]
    assert len(fed_batch) == 7
    hours = sorted(
        record["experiment"]["environment"]["duration_hours"] for record in fed_batch
    )
    assert hours == [21.4, 45.4, 69.4, 93.4, 117.4, 141.4, 165.4]
    with_yield = [
        record
        for record in fed_batch
        if record["experiment"]["phenotype"]["product_yield"] is not None
    ]
    assert len(with_yield) == 1
    assert with_yield[0]["experiment"]["phenotype"]["titer"] == pytest.approx(1909.4)


def test_the_four_references_are_the_papers_own_baselines(built_store: str) -> None:
    """Each family's reference is the measured baseline the paper divides by."""
    from torchcell.verification.runners import load_records

    titers = {
        record["reference"]["phenotype_reference"]["titer"]
        for record in load_records(built_store)
    }
    assert titers == {414.0, 181.0, 405.0, 1500.0}
    assert set(kang.REFERENCE_BASELINES) == {
        "flask_pipa",
        "flask_pipaxyl",
        "tube_pipa",
        "fedbatch",
    }


# --------------------------------------------------------------------------- #
# The real mirrors and the real built store
# --------------------------------------------------------------------------- #
@pytest.mark.data
@requires_mirror
def test_the_mirror_manifest_records_the_digest_the_module_pins() -> None:
    """The deposited bytes, their pin and the module constant are one number."""
    manifest = kang.load_manifest(DATA_ROOT)
    assert kang.manifest_sha256(manifest, kang.SI_DOCX_REL) == kang.SI_DOCX_SHA256
    assert _sha256(Path(DATA_ROOT, kang.RAW_DIR_REL, kang.SI_DOCX_REL)) == (
        kang.SI_DOCX_SHA256
    )
    record = manifest.files[0]
    assert record.retrieval is not None
    source_url = record.retrieval.source_url
    assert source_url is not None
    assert source_url.endswith(kang.pmc_cloud_key())


@pytest.mark.data
@requires_library
def test_every_sourced_quote_is_verbatim_in_its_pinned_mirror() -> None:
    """The audit the whole module rests on: 87 quotes, two mirrors, no paraphrase."""
    paper = Path(DATA_ROOT, "torchcell-library", kang.CITATION_KEY, kang.PAPER_MD)
    carruthers = Path(
        DATA_ROOT, "torchcell-library", kang.CARRUTHERS_KEY, kang.PAPER_MD
    )
    if not carruthers.exists():
        pytest.skip("the deferred-to Carruthers 2025 OCR mirror is not present")
    assert _sha256(paper) == kang.PAPER_MD_SHA256
    assert _sha256(carruthers) == kang.CARRUTHERS_PAPER_MD_SHA256
    paper_text = paper.read_text(encoding="utf-8")
    carruthers_text = carruthers.read_text(encoding="utf-8")
    docx = Path(DATA_ROOT, kang.RAW_DIR_REL, kang.SI_DOCX_REL)
    cells: set[str] = set()
    if docx.exists():
        cells = {
            cell
            for table in kang.read_si_tables(docx)
            for row in table.rows
            for cell in row
        }
    checked = 0
    for name, value in sorted(vars(kang).items()):
        if not name.startswith("_Q_"):
            continue
        checked += 1
        if name.endswith("CARRUTHERS"):
            assert value in carruthers_text, name
        elif value in paper_text:
            continue
        else:
            assert value in cells, name
    assert checked >= 30


@pytest.mark.data
@requires_genome
@requires_library
def test_the_real_annotation_agrees_with_table1s_own_locus_tags() -> None:
    """The five symbols the annotation resolves land exactly where Table 1 puts them."""
    from torchcell.datasets.bacteria_common import bacterial_genome

    genome = bacterial_genome("pputida", "KT2440", DATA_ROOT)
    for entry in kang.CHASSIS_DELETIONS_TYPED:
        resolved = genome.resolve_gene_name(entry.symbol)
        if entry.symbol_resolves:
            assert resolved.systematic_name == entry.locus_tag, entry.symbol
        else:
            assert resolved.status.value == "retired", entry.symbol
    assert genome.resolve_gene_name("crc").status.value == "retired"
    assert kang._annotation_symbols(genome)["PP_5005"] == "phaC-II"


@pytest.mark.data
@requires_genome
@requires_library
def test_the_real_chassis_carries_seventy_typed_alleles() -> None:
    """6 named individually plus 64 carrying the Δ86kb span, 1 of them a truncation."""
    from torchcell.datasets.bacteria_common import bacterial_genome

    genome = bacterial_genome("pputida", "KT2440", DATA_ROOT)
    background, _ = kang.chassis_background(genome)
    assert len(background.alleles) == 70
    with_span = [a for a in background.alleles if a.deleted_span is not None]
    assert len(with_span) == 64
    assert sum(1 for a in with_span if a.edit is AlleleEdit.partial_deletion) == 1
    assert background.is_fully_sourced


@pytest.mark.data
@requires_genome
@requires_library
def test_the_real_span_report_is_the_measurement_the_note_states() -> None:
    """67 loci inside the span, 63 typed, PP_4023 truncated, four span-only."""
    from torchcell.datasets.bacteria_common import bacterial_genome

    genome = bacterial_genome("pputida", "KT2440", DATA_ROOT)
    stub: Any = genome
    report = kang.resolve_span(stub)
    assert report.span_length == 91_743
    assert len(report.inside_span) == 67
    assert len(report.tag_range) == 64
    assert len(report.full_deletions) == 63
    assert report.partial_deletions == ["PP_4023"]
    assert report.span_only == ["PP_5652", "PP_5653", "PP_5654", "PP_mr45"]
    assert report.unannotated_tag_numbers == [4026, 4028, 4062, 4086, 4087, 4088]


@pytest.mark.data
@requires_store
@requires_mirror
@requires_library
def test_the_real_built_store_passes_l0_to_l4() -> None:
    """The verifier numbers the PR reports, from the module's own runner."""
    assert _STORE is not None
    report = kang.verify_build(str(_STORE), DATA_ROOT)
    assert report.passed, report.summary()
    counts = {result.level.name: 0 for result in report.results}
    for result in report.results:
        counts[result.level.name] += 1
    assert counts == {"L0": 1, "L1": 1, "L2": 2, "L3": 4, "L4": 2}


@pytest.mark.data
@requires_store
def test_the_real_build_manifest_names_this_loader_module() -> None:
    """The KG freshness gate reads exactly this file."""
    assert _STORE is not None
    manifest = json.loads((_STORE / "preprocess" / "build_manifest.json").read_text())
    assert manifest["loader_module"] == "torchcell.datasets.pputida.kang2026"
    assert manifest["loader_class"] == "IsoprenylAcetateTiterKang2026Dataset"


def main() -> int:
    """Print the L0-L4 report for the real built store (the PR's verifier numbers)."""
    from dotenv import load_dotenv

    load_dotenv()
    store = Path(
        os.environ["DATA_ROOT"], "data/torchcell/isoprenyl_acetate_titer_kang2026"
    )
    report = kang.verify_build(str(store), os.environ["DATA_ROOT"])
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
