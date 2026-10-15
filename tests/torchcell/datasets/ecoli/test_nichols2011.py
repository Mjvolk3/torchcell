# tests/torchcell/datasets/ecoli/test_nichols2011.py
# [[tests.torchcell.datasets.ecoli.test_nichols2011]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_nichols2011.py
"""Nichols 2011 loader: the Table S1 transcription, label matching, strains, records.

The synthetic tests run everywhere: the screen transcription and its validator, the
Table S2 label grammar and its one-to-one match onto Table S1, dose conversion, the
environment and phenotype of each screen kind, the row-label grammar, the row rules on a
stubbed BW25113 annotation, the genotype of each kept kind, the Table S2 reader on a
small OOXML workbook written under an ``.xls`` name, and the OCR manifest records.

The data tests (``--data``) audit every quote against the sha256-pinned OCR and Table
S1, read the real Tables S1 and S2 from the raw mirror, and check the built dev store.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from decimal import Decimal
from pathlib import Path
from typing import Any

import openpyxl
import pytest

import torchcell.datasets.ecoli.nichols2011 as n
from torchcell.datamodels.schema import (
    AssayType,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    MeasurementType,
    PhysicalExposurePerturbation,
    PhysicalFactor,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    SampleUnit,
    SmallMoleculePerturbation,
)
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.literature.manifest import ProcessingRecord
from torchcell.verification.sourced import audit_sourced_value


def _condition(label: str) -> n.Condition:
    return n.match_condition(label)


def _screen(row: str) -> n.Screen:
    (screen,) = [s for s in n.SCREENS if s.row == row]
    return screen


# --------------------------------------------------------------------------- #
# Table S1 transcription
# --------------------------------------------------------------------------- #
def test_the_transcription_holds_114_screens_summing_to_324_columns() -> None:
    assert len(n.SCREENS) == 114
    assert sum(s.screens for s in n.SCREENS) == n.RELEASED_CONDITIONS == 324
    assert len({s.row for s in n.SCREENS}) == 114
    kinds = {
        k: sum(s.screens for s in n.SCREENS if s.kind == k)
        for k in (
            "small_molecule",
            "carbon_source",
            "nitrogen_source",
            "ph",
            "temperature",
            "uv",
            "anaerobic",
        )
    }
    assert kinds == {
        "small_molecule": 296,
        "carbon_source": 7,
        "nitrogen_source": 1,
        "ph": 8,
        "temperature": 7,
        "uv": 4,
        "anaerobic": 1,
    }


def test_a_screen_refuses_a_column_count_that_disagrees_with_its_screens_cell() -> None:
    with pytest.raises(ValueError, match="2 columns transcribed, Table S1 says 3"):
        n.Screen(
            row="X",
            cell="1,2 µg/ml",
            screens=3,
            stem="X",
            kind="small_molecule",
            compounds=("x",),
            units=("ug/ml",),
            amounts=(("1",), ("2",)),
        )


def test_a_screen_refuses_a_number_its_cell_does_not_print() -> None:
    with pytest.raises(
        ValueError, match=r"reads \['1', '2'\], the transcription \['1', '3'\]"
    ):
        n.Screen(
            row="X",
            cell="1,2 µg/ml",
            screens=2,
            stem="X",
            kind="small_molecule",
            compounds=("x",),
            units=("ug/ml",),
            amounts=(("1",), ("3",)),
        )


def test_a_screen_refuses_a_column_missing_a_compound_amount() -> None:
    with pytest.raises(ValueError, match="every column carries 2 amount"):
        n.Screen(
            row="X",
            cell="1 + 2 µg/ml",
            screens=1,
            stem="X",
            kind="small_molecule",
            compounds=("x", "y"),
            units=("ug/ml", "ug/ml"),
            amounts=(("1",),),
        )


def test_check_screen_list_refuses_bytes_that_read_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [(s.row, s.cell, s.screens) for s in n.SCREENS]
    monkeypatch.setattr(n, "read_screen_list", lambda path: list(rows))
    n.check_screen_list("unused")
    rows[8] = ("Ampicillin", "1,2,4,16 µg/ml", 4)
    with pytest.raises(ValueError, match="disagrees with the transcription"):
        n.check_screen_list("unused")


def test_compounds_are_named_as_table_s1_writes_them() -> None:
    assert _screen("Nitrofurnatoin").compounds == ("Nitrofurnatoin",)
    assert _screen("Cobalt stress-CoCl2").compounds == ("CoCl2",)
    assert _screen("SDS+EDTA").compounds == ("SDS", "EDTA")
    assert _screen("Fosfomycin +Glucose 6P").compounds == ("Fosfomycin", "Glucose 6P")
    assert _screen("Calcofluor (F3543    \nFluorescent Brightener 28) ").compounds == (
        "Calcofluor (F3543 Fluorescent Brightener 28)",
    )


def test_released_compound_keeps_the_released_name_and_attaches_identity() -> None:
    ampicillin = n.released_compound("Ampicillin")
    assert ampicillin.name == "Ampicillin" and ampicillin.inchikey is None
    assert ampicillin.gapped_fields() == {"inchikey"}
    ethanol = n.released_compound("EtOH")
    assert ethanol.name == "EtOH"
    assert ethanol.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert ethanol.gapped_fields() == set()


# --------------------------------------------------------------------------- #
# Table S2 labels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "stem", "amounts", "percent"),
    [
        ("16C - ", "TEMPERATURE", ("16",), False),
        ("43.5C - ", "TEMPERATURE", ("43.5",), False),
        ("PH4.5 - UNSPECIFIED", "PH", ("4.5",), False),
        ("UV-12SEC - UNSPECIFIED", "UV", ("12",), False),
        ("SDS0.5%/EDTA0.1 - ", "SDS/EDTA", ("0.5", "0.1"), True),
        (
            "CEFSULODIN-6.0,MECILLINAM-0.03 - UNSPECIFIED",
            "CEFSULODIN,MECILLINAM",
            ("6.0", "0.03"),
            False,
        ),
        ("BILE-0.1% - UNSPECIFIED", "BILE", ("0.1",), True),
        ("N-ACETYLGLUCOSAMINE - UNSPECIFIED", "N-ACETYLGLUCOSAMINE", (), False),
        ("ACETATE - ", "ACETATE", (), False),
        ("CHIR090-0.025 - UNSPECIFIED", "CHIR090", ("0.025",), False),
    ],
)
def test_parse_label_splits_stem_and_numbers(
    label: str, stem: str, amounts: tuple[str, ...], percent: bool
) -> None:
    assert n.parse_label(label) == n.ParsedLabel(
        stem=stem, amounts=amounts, percent=percent
    )


@pytest.mark.parametrize(
    "label",
    ["AMPICILLIN-1.0", "AMPICILLIN-1.0 - SPECIFIED", "AMP ICILLIN-1.0 - UNSPECIFIED"],
)
def test_parse_label_refuses_a_label_outside_the_grammar(label: str) -> None:
    with pytest.raises(ValueError):
        n.parse_label(label)


def test_match_condition_takes_the_table_s1_dose_not_the_label_number() -> None:
    c = _condition("AMPICILLIN-8.0 - UNSPECIFIED")
    assert c.screen.row == "Ampicillin"
    assert c.amounts == ("8",)
    (dose,) = c.doses
    assert dose == n.Dose(compound="Ampicillin", amount="8", unit="ug/ml")


def test_match_condition_reads_the_mms_percent_against_the_bare_fraction() -> None:
    c = _condition("MMS-0.05% - UNSPECIFIED")
    assert c.amounts == ("0.0005",)
    (dose,) = c.doses
    assert dose.concentration.value == 0.05
    assert dose.concentration.unit is ConcentrationUnit.percent


def test_match_condition_splits_shared_stems_by_their_amounts() -> None:
    assert _condition("16C - ").screen.row == "Cold shock "
    assert _condition("45C - ").screen.row == "Heat shock "
    assert _condition("PH6 - UNSPECIFIED").screen.row == "acidic pH (MES-HOMOPIPES)"
    assert _condition("PH8 - UNSPECIFIED").screen.row == "basic pH (TAPS)"


def test_match_condition_pairs_a_combination_with_its_fixed_partner() -> None:
    c = _condition("GLUFOSFOMYCIN-0.2 - UNSPECIFIED")
    assert c.amounts == ("0.2", "50")
    assert [d.compound for d in c.doses] == ["Fosfomycin", "Glucose 6P"]
    assert c.doses[1].concentration.value == 50000.0
    assert c.doses[1].concentration.unit is ConcentrationUnit.ug_per_ml


def test_match_condition_refuses_a_dose_table_s1_does_not_list() -> None:
    with pytest.raises(ValueError, match="matches 0 Table S1 columns"):
        _condition("AMPICILLIN-3.0 - UNSPECIFIED")


def test_match_conditions_refuses_a_repeated_or_missing_column() -> None:
    labels = [_label_of(s, column) for s in n.SCREENS for column in s.amounts]
    assert len(n.match_conditions(labels)) == 324
    with pytest.raises(ValueError, match="do not map one to one"):
        n.match_conditions([*labels[:-1], labels[0]])


def _label_of(screen: n.Screen, column: tuple[str, ...]) -> str:
    """A Table S2-shaped label for one Table S1 column (the inverse of the matcher)."""
    if screen.kind == "temperature":
        return f"{column[0]}C - "
    if screen.kind == "ph":
        return f"PH{column[0]} - UNSPECIFIED"
    if screen.kind == "uv":
        return f"UV-{column[0]}SEC - UNSPECIFIED"
    if screen.stem == "SDS/EDTA":
        return f"SDS{column[0]}%/EDTA{column[1]} - "
    if screen.screens == 1 and screen.kind != "small_molecule":
        return f"{screen.stem} - UNSPECIFIED"
    if screen.stem == "MMS":
        return "MMS-0.05% - UNSPECIFIED"
    if "," in screen.stem:
        return (
            ",".join(
                f"{stem}-{a}"
                for stem, a in zip(screen.stem.split(","), column, strict=True)
            )
            + " - UNSPECIFIED"
        )
    if screen.stem in ("CALCOFLUOR", "HIGHFE", "LOWFE"):
        return f"{screen.stem} - UNSPECIFIED"
    return f"{screen.stem}-{column[0]} - UNSPECIFIED"


@pytest.mark.parametrize(
    ("unit", "amount", "value", "stored"),
    [
        ("ug/ml", "2.5", 2.5, ConcentrationUnit.ug_per_ml),
        ("ng/ml", "0.5", 0.0005, ConcentrationUnit.ug_per_ml),
        ("mg/ml", "50", 50000.0, ConcentrationUnit.ug_per_ml),
        ("uM", "24", 24.0, ConcentrationUnit.micromolar),
        ("mM", "0.4", 0.4, ConcentrationUnit.millimolar),
        ("percent", "0.01", 0.01, ConcentrationUnit.percent),
        ("fraction", "0.006", 0.6, ConcentrationUnit.percent),
    ],
)
def test_dose_converts_by_exact_decimal(
    unit: n.SourceUnit, amount: str, value: float, stored: ConcentrationUnit
) -> None:
    concentration = n.Dose(compound="x", amount=amount, unit=unit).concentration
    assert concentration.value == value
    assert concentration.unit is stored
    assert Decimal(amount) * n.UNIT_CONVERSIONS[unit][1] == Decimal(str(value))


# --------------------------------------------------------------------------- #
# Environment and phenotype
# --------------------------------------------------------------------------- #
def test_a_drug_plate_is_lb_agar_at_37_c_with_one_small_molecule() -> None:
    env = n.environment(_condition("AMPICILLIN-2.0 - UNSPECIFIED"))
    assert env.media is n.NICHOLS2011_LB_AGAR
    assert env.media.base_medium == "LB"
    assert all(c.concentration is None for c in env.media.components)
    assert env.temperature is not None and env.temperature.value == 37.0
    assert env.aerobicity == "aerobic"
    assert env.duration_hours is None
    assert env.gapped_fields() == {"duration_hours"}
    (p,) = env.perturbations
    assert isinstance(p, SmallMoleculePerturbation)
    assert p.concentration.value == 2.0
    assert p.concentration.unit is ConcentrationUnit.ug_per_ml


def test_a_shock_or_the_anaerobic_plate_carries_no_perturbation() -> None:
    cold = n.environment(_condition("16C - "))
    assert cold.perturbations == []
    assert cold.temperature is not None and cold.temperature.value == 16.0
    anaerobic = n.environment(_condition("ANAEROBIC - UNSPECIFIED"))
    assert anaerobic.perturbations == []
    assert anaerobic.aerobicity == "anaerobic"


def test_a_uv_plate_is_dosed_by_its_exposure_time_with_both_flux_gaps() -> None:
    (p,) = n.environment(_condition("UV-18SEC - UNSPECIFIED")).perturbations
    assert isinstance(p, PhysicalExposurePerturbation)
    assert p.exposure_duration_seconds == 18.0
    assert p.gapped_fields() == {"irradiance_w_per_m2", "fluence_j_per_m2"}


def test_the_acidic_buffer_is_a_gap_and_the_basic_one_is_taps_at_100_mm() -> None:
    (acidic,) = n.environment(_condition("PH4.5 - UNSPECIFIED")).perturbations
    assert isinstance(acidic, EnvironmentPhysicalPerturbation)
    assert acidic.factor is PhysicalFactor.ph and acidic.agent is None
    assert acidic.magnitude is not None and acidic.magnitude.value == 4.5
    assert acidic.gapped_fields() == {"agent"}
    ph, buffer = n.environment(_condition("PH9.5 - UNSPECIFIED")).perturbations
    assert isinstance(ph, EnvironmentPhysicalPerturbation)
    assert ph.agent is not None and ph.agent.name == "TAPS"
    assert ph.magnitude is not None and ph.magnitude.unit is ConcentrationUnit.ph
    assert isinstance(buffer, SmallMoleculePerturbation)
    assert buffer.concentration.value == 100.0
    assert buffer.concentration.unit is ConcentrationUnit.millimolar


def test_nutrient_plates_vary_their_carbon_or_nitrogen_source() -> None:
    m9 = n.environment(_condition("GLYCEROL - UNSPECIFIED"))
    assert m9.media is n.NICHOLS2011_M9_AGAR
    (carbon,) = m9.perturbations
    assert isinstance(carbon, EnvironmentPhysicalPerturbation)
    assert carbon.factor is PhysicalFactor.carbon_source
    assert carbon.agent is not None and carbon.agent.name == "Glycerol"
    assert carbon.magnitude is not None and carbon.magnitude.value == 0.4
    nag = n.environment(_condition("N-ACETYLGLUCOSAMINE - UNSPECIFIED"))
    assert nag.media is n.NICHOLS2011_LB_AGAR
    mops = n.environment(_condition("NH4CL - UNSPECIFIED"))
    assert mops.media is n.NICHOLS2011_MOPS_AGAR
    (nitrogen,) = mops.perturbations
    assert isinstance(nitrogen, EnvironmentPhysicalPerturbation)
    assert nitrogen.factor is PhysicalFactor.nitrogen_source
    assert nitrogen.magnitude is not None and nitrogen.magnitude.value == 9.5
    (iron,) = n.environment(_condition("LOWFE - UNSPECIFIED")).perturbations
    assert isinstance(iron, SmallMoleculePerturbation)
    assert iron.compound.name == "FeSO4"
    assert iron.concentration.unit is ConcentrationUnit.micromolar


def test_phenotype_is_the_rescaled_z_score_at_four_colonies() -> None:
    c = _condition("AMPICILLIN-2.0 - UNSPECIFIED")
    assert n.phenotype(1.0, c, "Table S2 row 586").replicate_id == "Table S2 row 586"
    p = n.phenotype(-3.25, c)
    assert p.measurement_type is MeasurementType.z_score
    assert p.assay_type is AssayType.colony_size_array
    assert p.environment_response == -3.25
    assert (p.n_samples, p.sample_unit) == (4, SampleUnit.colony)
    assert p.screen_id == "AMPICILLIN-2.0 - UNSPECIFIED"
    assert p.replicate_id is None
    assert p.gapped_fields() == {
        "environment_response_uncertainty",
        "environment_response_se",
    }
    ref = n.reference_phenotype(c)
    assert ref.environment_response == 0.0
    assert ref.screen_id == p.screen_id


# --------------------------------------------------------------------------- #
# Strain labels, stored as released
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "kind", "ecks", "symbol"),
    [
        ("ECK1963-HCHA", ReleasedKind.deletion, ("ECK1963",), "HCHA"),
        ("ECK2350-YFDM'", ReleasedKind.deletion, ("ECK2350",), "YFDM'"),
        ("ECK2331", ReleasedKind.deletion, ("ECK2331",), None),
        ("ECK5002-ISTR-1", ReleasedKind.deletion, ("ECK5002",), "ISTR-1"),
        (
            "ECK1205/1207/1209-RDLABC",
            ReleasedKind.deletion,
            ("ECK1205", "ECK1207", "ECK1209"),
            None,
        ),
        ("ECK4142-4143-ECNAB", ReleasedKind.deletion, ("ECK4142", "ECK4143"), None),
        ("ECK0095-C-FTSA-SPA - SPA", ReleasedKind.affinity_tag, ("ECK0095",), "FTSA"),
        ("ECK0125-CAN - SPA", ReleasedKind.affinity_tag, ("ECK0125",), "CAN"),
        ("ECK0086-C - SPA", ReleasedKind.affinity_tag, ("ECK0086",), None),
        ("ECK0055-B-IMP-DAS+4-KAN - DAS", ReleasedKind.degron, ("ECK0055",), "IMP"),
        (
            "ECK0097-A-LPXC-KAN - Linked",
            ReleasedKind.linked_marker,
            ("ECK0097",),
            "LPXC",
        ),
        (
            "ECK0097-B-LPXC(G210S)-KAN - Allele",
            ReleasedKind.sequence_allele,
            ("ECK0097",),
            "LPXC",
        ),
        (
            "ECK0055-D-IMP4213 - Allele",
            ReleasedKind.sequence_allele,
            ("ECK0055",),
            "IMP",
        ),
        ("ECK2593-A-YFIO* - Truncation", ReleasedKind.truncation, ("ECK2593",), "YFIO"),
    ],
)
def test_parse_strain_label(
    label: str, kind: ReleasedKind, ecks: tuple[str, ...], symbol: str | None
) -> None:
    parsed = n.parse_strain_label(7, label)
    assert parsed == n.StrainLabel(
        index=7, label=label, kind=kind, ecks=ecks, symbol=symbol
    )


@pytest.mark.parametrize("label", ["JW0001-THRL", "ECK0001/0002-AB - SPA"])
def test_parse_strain_label_refuses_what_its_grammar_does_not_cover(label: str) -> None:
    with pytest.raises(ValueError):
        n.parse_strain_label(0, label)


_LABELS = [
    "ECK4142-ECNA",
    "ECK4142-4143-ECNAB",
    "ECK2331",
    "ECK0005-GEN5",
    "ECK0005-GEN5",
    "ECK0097-B-LPXC(G210S)-KAN - Allele",
    "ECK0097-C-LPXC-SPA - SPA",
    "ECK0055-B-IMP-DAS+4-KAN - DAS",
]


def test_strain_rows_keep_every_row_under_its_released_label() -> None:
    rows = n.strain_rows(_LABELS)
    assert [r.label.label for r in rows] == _LABELS
    assert [r.gene_names for r in rows] == [
        ("ECNA",),
        ("ECNA", "ECK4143"),
        ("ECK2331",),
        ("GEN5",),
        ("GEN5",),
        ("LPXC",),
        ("LPXC",),
        ("IMP",),
    ]
    assert [r.replicate_id for r in rows] == [
        None,
        None,
        None,
        "Table S2 row 6",
        "Table S2 row 7",
        None,
        None,
        None,
    ]


def test_released_gene_names_refuses_an_accession_released_under_two_names() -> None:
    labels = [
        n.parse_strain_label(0, "ECK0001-AAA"),
        n.parse_strain_label(1, "ECK0001-BBB"),
    ]
    with pytest.raises(ValueError, match="released under two names"):
        n.released_gene_names(labels)


def test_genotype_stores_each_accession_on_the_released_strain_leaf() -> None:
    rows = {r.label.label: r for r in n.strain_rows(_LABELS)}
    multi = n.genotype(rows["ECK4142-4143-ECNAB"])
    assert multi.systematic_gene_names == ["ECK4142", "ECK4143"]
    for p in multi.perturbations:
        assert isinstance(p, ReleasedIdentifierPerturbation)
        assert p.strain_label == "ECK4142-4143-ECNAB"
        assert p.released_kind is ReleasedKind.deletion
        assert (p.cassette, p.collection) == ("kanamycin", None)
        assert p.gene_namespace == "ecoli_k12_bw25113_locus_tag"
        assert p.identifier_kind == "ECK accession"
    (allele,) = n.genotype(rows["ECK0097-B-LPXC(G210S)-KAN - Allele"]).perturbations
    assert isinstance(allele, ReleasedIdentifierPerturbation)
    assert (allele.systematic_gene_name, allele.perturbed_gene_name) == (
        "ECK0097",
        "LPXC",
    )
    assert allele.mechanism_so == ("SO:0001060", "sequence_variant")
    (spa,) = n.genotype(rows["ECK0097-C-LPXC-SPA - SPA"]).perturbations
    assert isinstance(spa, ReleasedIdentifierPerturbation)
    assert spa.collection == n.SPA_COLLECTION
    (das,) = n.genotype(rows["ECK0055-B-IMP-DAS+4-KAN - DAS"]).perturbations
    assert isinstance(das, ReleasedIdentifierPerturbation)
    assert (das.collection, das.mechanism_so) == (
        n.DAS_COLLECTION,
        ("SO:0001218", "transgenic_insertion"),
    )
    (unannotated,) = n.genotype(rows["ECK2331"]).perturbations
    assert unannotated.systematic_gene_name == "ECK2331"


# --------------------------------------------------------------------------- #
# Reading Table S2
# --------------------------------------------------------------------------- #
def _workbook(path: Path, rows: list[list[Any]], title: str = n.MATRIX_TITLE) -> Path:
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = n.MATRIX_SHEET
    sheet.append([title])
    for row in rows:
        sheet.append(row)
    book.save(path)
    return path


def test_read_score_matrix_reads_an_ooxml_workbook_under_an_xls_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(n, "RELEASED_STRAIN_ROWS", 2)
    monkeypatch.setattr(n, "RELEASED_CONDITIONS", 2)
    path = _workbook(
        tmp_path / "mmc2.xls",
        [
            ["Gene", "A-1 - ", "B-2 - "],
            ["ECK0001-THRL", -1.5, None],
            ["ECK0002", 2, 0.25],
        ],
    )
    matrix = n.read_score_matrix(path)
    assert matrix.conditions == ("A-1 - ", "B-2 - ")
    assert matrix.strains == ("ECK0001-THRL", "ECK0002")
    assert matrix.scores == ((-1.5, None), (2.0, 0.25))


def test_read_score_matrix_refuses_a_wrong_title_text_score_or_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(n, "RELEASED_STRAIN_ROWS", 1)
    monkeypatch.setattr(n, "RELEASED_CONDITIONS", 1)
    rows: list[list[Any]] = [["Gene", "A-1 - "], ["ECK0001", 1.0]]
    with pytest.raises(ValueError, match="title cell"):
        n.read_score_matrix(_workbook(tmp_path / "a.xls", rows, title="Table S9"))
    bad = [["Gene", "A-1 - "], ["ECK0001", "n/a"]]
    with pytest.raises(ValueError, match="non-numeric score"):
        n.read_score_matrix(_workbook(tmp_path / "b.xls", bad))
    two = [*rows, ["ECK0002", 2.0]]
    with pytest.raises(ValueError, match="2 x 1, expected 1 x 1"):
        n.read_score_matrix(_workbook(tmp_path / "c.xls", two))


# --------------------------------------------------------------------------- #
# The article OCR in the raw mirror
# --------------------------------------------------------------------------- #
def _ocr_tree(root: Path, input_sha: str) -> str:
    si = root / "si"
    (si / "images" / "mmc8").mkdir(parents=True)
    (si / "mmc8.md").write_text("the article\n")
    (si / "mmc8_content_list.json").write_text("[]")
    (si / "mmc8_middle.json").write_text("{}")
    (si / "images" / "mmc8" / "a.jpg").write_bytes(b"jpg")
    record = ProcessingRecord(
        processor="torchcell.literature.ocr.ocr_pdf",
        tool="mineru",
        version="2.7.6",
        input_sha256=[input_sha],
    )
    (si / "mmc8_ocr_provenance.json").write_text(record.model_dump_json())
    return hashlib.sha256(b"the article\n").hexdigest()


def test_article_ocr_records_pin_every_output_and_refuse_another_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    md_sha = _ocr_tree(tmp_path, n.ARTICLE_PDF_SHA256)
    monkeypatch.setattr(n, "ARTICLE_MD_SHA256", md_sha)
    records = n.article_ocr_records(tmp_path)
    assert [r.path for r in records] == [
        "si/mmc8.md",
        "si/mmc8_ocr_provenance.json",
        "si/mmc8_content_list.json",
        "si/mmc8_middle.json",
        "si/images/mmc8/a.jpg",
    ]
    assert records[0].processing is not None
    assert records[4].sha256 == hashlib.sha256(b"jpg").hexdigest()
    other = tmp_path / "other"
    _ocr_tree(other, "0" * 64)
    with pytest.raises(RuntimeError, match="is not the OCR of si/mmc8.pdf"):
        n.article_ocr_records(other)


def test_the_dataset_declares_its_strain_organism_files_and_classes() -> None:
    cls = dataset_registry["EnvChemgenNichols2011Dataset"]
    assert cls is n.EnvChemgenNichols2011Dataset
    assert cls.REFERENCE_STRAIN == "BW25113"
    assert cls.organism.value == "ecoli"
    assert "ecoli_genome" not in cls.__init__.__code__.co_varnames
    dataset = object.__new__(cls)
    assert dataset.raw_file_names == ["mmc1.xls", "mmc2.xls"]
    assert dataset.experiment_class is EnvironmentResponseExperiment


def test_sourced_values_point_at_the_raw_mirror(tmp_path: Path) -> None:
    for value in n.SOURCED_VALUES:
        root = n.sourced_value_root(value, str(tmp_path))
        assert root == tmp_path / "torchcell-raw"
    assert {v.provenance.source_uri for v in n.SOURCED_VALUES} == {"si/mmc8.md"}


# --------------------------------------------------------------------------- #
# Data tests (--data): the mirror and the built store
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    return os.environ["DATA_ROOT"]


@pytest.mark.data
def test_every_quote_is_verbatim_in_its_pinned_file() -> None:
    for value in n.SOURCED_VALUES:
        result = audit_sourced_value(value, n.sourced_value_root(value, _data_root()))
        assert result.passed, (value.quote, result.message)


@pytest.mark.data
def test_table_s1_reads_as_transcribed_and_table_s2_matches_it() -> None:
    root = n.raw_mirror_dir(_data_root())
    n.check_screen_list(root / n.SCREEN_LIST_REL)
    for screen in n.SCREENS:
        statement = n._screen_list_statement(screen.row, screen.cell)
        assert statement.quote == screen.cell
    matrix = n.read_score_matrix(root / n.MATRIX_REL)
    conditions = n.match_conditions(matrix.conditions)
    assert len({c.label for c in conditions}) == 324
    assert sum(v is not None for row in matrix.scores for v in row) == 1286962


@pytest.mark.data
def test_the_dev_store_holds_every_released_score() -> None:
    root = osp.join(_data_root(), "data/torchcell/ecoli_env_chemgen_nichols2011")
    cells = json.loads(Path(root, "preprocess", "cell_accounting.json").read_text())
    assert cells["released_cells"] == 1289196
    assert cells["records"] == n.EXPECTED_RECORDS == 1286962
    assert cells["empty_cells"] == 2234
    labels = json.loads(Path(root, "preprocess", "strain_labels.json").read_text())
    assert labels["released_strain_rows"] == 3979
    assert labels["genome_lookups"] == 0
    assert labels["rows_with_replicate_id"] == 24
    assert labels["rows_by_kind"] == {
        "affinity_tag": 117,
        "degron": 5,
        "deletion": 3837,
        "linked_marker": 9,
        "sequence_allele": 9,
        "truncation": 2,
    }
    report = json.loads(
        Path(root, "preprocess", "verification_report.json").read_text()
    )
    assert report["passed"] is True
