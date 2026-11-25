# tests/torchcell/datasets/ecoli/test_rachwalski2024.py
# [[tests.torchcell.datasets.ecoli.test_rachwalski2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_rachwalski2024.py
"""``torchcell/datasets/ecoli/rachwalski2024.py``: the readers, the plate-order join, the
three build checks, the drop rules, the records and the raw mirror.

Everything here is hermetic. The workbooks are written at a SMALL shape and the module's
released-shape constants are pinned to it, the identifier tests read the REAL
``EcoliK12BW25113Genome`` over the synthetic BW25113 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` with the network refused, and
the end-to-end build runs into a ``tmp_path`` DATA_ROOT. The ``--data`` tests at the
bottom read the dev-tree store the real release built.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl
import pytest

import torchcell.datasets.ecoli.rachwalski2024 as r
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    ComponentDefinition,
    ConcentrationUnit,
    CrisprInterferencePerturbation,
    DeletionStrainPerturbation,
    DoseBasis,
    Genotype,
    MediaComponentRole,
    SampleUnit,
    SmallMoleculePerturbation,
)
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.sequence.genome.ecoli.k12 import BW25113_ASSEMBLY, EcoliK12BW25113Genome
from torchcell.verification.report import Level
from torchcell.verification.sourced import ProvenanceGapReason

# --------------------------------------------------------------------------- #
# The synthetic release: 8 collection wells and 6 Keio deletion rows
# --------------------------------------------------------------------------- #
#: ``(well row, well column, target symbol, spacer)``. A1 is the empty vector; A7
#: repeats A2's spacer, so both A2 and A7 are one construct at two wells; A6 names
#: nothing the synthetic assembly carries. Three wells are writable: hokC, yaaX, thrL.
SYNTHETIC_WELLS: tuple[tuple[str, int, str, str | None], ...] = (
    ("A", 1, "Empty_Vector", None),
    ("A", 2, "thrA", "AAAAAAAAAAAAAAAAAAAA"),
    ("A", 3, "hokC", "CCCCCCCCCCCCCCCCCCCC"),
    ("A", 4, "yaaX", "GGGGGGGGGGGGGGGGGGGG"),
    ("A", 5, "thrL", "TTTTTTTTTTTTTTTTTTTT"),
    ("A", 6, "notAGene", "AGAGAGAGAGAGAGAGAGAG"),
    ("A", 7, "thrA", "AAAAAAAAAAAAAAAAAAAA"),
)
#: The growth tables' row labels: Table S1's symbols with the one transposition typo
#: this release carries, which stands in for ``gpsA`` spelled ``gspA`` at well G15.
TYPO_WELL = "A4"
TYPO_SYMBOL = "yaaX"
TYPO_LABEL = "yaxX"
SYNTHETIC_LABELS: tuple[str, ...] = tuple(
    TYPO_LABEL if f"{row}{column}" == TYPO_WELL else symbol
    for row, column, symbol, _ in SYNTHETIC_WELLS
)
#: Table S4A's deletion labels: four writable, one repeated, one unresolvable.
SYNTHETIC_KEIO: tuple[str, ...] = ("thrL", "thrA", "hokC", "yaaX", "hokC", "notAGene")
#: The three knockdowns Table S4A's blocks name, remapped onto the synthetic symbols.
SYNTHETIC_BLOCKS: tuple[tuple[str | None, str], ...] = (
    (None, r.S4A_BLOCKS[0][1]),
    ("thrA", r.S4A_BLOCKS[1][1]),
    ("hokC", r.S4A_BLOCKS[2][1]),
    ("yaaX", r.S4A_BLOCKS[3][1]),
)
#: The LB 100 ng/mL value given to the wells that count as a 50% reduction.
REDUCED = 0.25
#: The value every other growth cell carries, so an empty-vector mean of 1 is exact.
NEUTRAL = 1.0


def _six(reduced: bool) -> list[float]:
    """One six-dose row: all neutral, with the 100 ng/mL column optionally reduced."""
    values = [NEUTRAL] * 6
    if reduced:
        values[4] = REDUCED
    return values


def write_table_s1(path: Path) -> Path:
    """The synthetic Table S1: one header row then the eight wells."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Sheet1"
    sheet.append(list(r.S1_COLUMNS))
    for row, column, symbol, spacer in SYNTHETIC_WELLS:
        sheet.append(
            [row, column, symbol, "yes", "function", "operon", spacer, "p1", "p2"]
        )
    book.save(path)
    return path


def write_table_s2(
    path: Path, *, reduced: int = 2, labels: list[str] | None = None
) -> Path:
    """The synthetic Table S2 workbook: sheets ST2A and ST2C.

    ``reduced`` is how many guide rows carry a reduced LB 100 ng/mL value, which is what
    :func:`check_fifty_percent_reduction` counts.
    """
    names = labels or list(SYNTHETIC_LABELS)
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "ST2A"
    sheet.append([None, "Average Growth of Replicates"])
    sheet.append([None, *r.SIX_DOSE_HEADERS, *r.SIX_DOSE_HEADERS])
    seen = 0
    for name in names:
        if name == r.EMPTY_VECTOR:
            sheet.append([name, *_six(False), *_six(False)])
            continue
        sheet.append([name, *_six(seen < reduced), *_six(False)])
        seen += 1
    s2c = book.create_sheet("ST2C")
    s2c.append(["CRISPRi constructs"])
    s2c.append(
        [
            "Genes",
            "Normalized Growth LB with 500 ng/ml aTc (This study)",
            "Normalized Growth MOPS-glucose with 500 ng/ml aTc (This study)",
        ]
    )
    s2c.append(["hokC", NEUTRAL, NEUTRAL])
    s2c.append(["thrL", NEUTRAL, NEUTRAL])
    book.save(path)
    return path


def write_table_s3(path: Path, *, labels: list[str] | None = None) -> Path:
    """The synthetic Table S3: a WT block, a dlpp block and a fold-change block."""
    names = labels or list(SYNTHETIC_LABELS)
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "ST3"
    sheet.append(
        [
            None,
            "Normalized Growth of WT CRISPRi Collection",
            *[None] * 6,
            "Normalized Growth of dlpp CRISPRi Collection",
        ]
    )
    sheet.append([None, *r.SIX_DOSE_HEADERS, None, *r.SIX_DOSE_HEADERS])
    for name in names:
        sheet.append([name, *_six(False), None, *_six(False)])
    book.save(path)
    return path


def write_table_s4(path: Path, *, labels: tuple[str, ...] = SYNTHETIC_KEIO) -> Path:
    """The synthetic Table S4: sheet ST4A with four three-dose blocks."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "ST4A"
    group: list[Any] = [None]
    header: list[Any] = ["Keio_Deletion"]
    for _, label in SYNTHETIC_BLOCKS:
        group += [label, None, None, None]
        header += [*r.THREE_DOSE_HEADERS, None]
    sheet.append(group)
    sheet.append(header)
    for name in labels:
        row: list[Any] = [name]
        for _ in SYNTHETIC_BLOCKS:
            row += [NEUTRAL, NEUTRAL, NEUTRAL, None]
        sheet.append(row)
    book.save(path)
    return path


# --------------------------------------------------------------------------- #
# The synthetic Zenodo deposit: per-replicate Keio tables and the replicate legends
# --------------------------------------------------------------------------- #
#: The two replicate values of every synthetic Keio cell. Their mean is ``NEUTRAL``,
#: which is what the synthetic Table S4A carries, so the deposit reproduces the workbook
#: exactly as the real one does.
KEIO_R1 = NEUTRAL - 0.1
KEIO_R2 = NEUTRAL + 0.1
#: ``(R row index, Well.1536)`` per synthetic Keio row. Three colonies to a plate over
#: two plates, so the derived plates are 1, 1, 1, 2, 2, 2 and every (plate, well) is
#: distinct even where a label repeats.
SYNTHETIC_KEIO_POSITIONS: tuple[tuple[int, str], ...] = (
    (1, "0A01"),
    (2, "0A02"),
    (3, "0A03"),
    (4, "0A01"),
    (5, "0A02"),
    (6, "0A03"),
)
#: The synthetic 1,536 legend: two strains, each pinned four times.
SYNTHETIC_LEGEND: tuple[tuple[str, str, int], ...] = tuple(
    (f"0A0{index}", symbol, pinning)
    for index, (symbol, pinning) in enumerate(
        [
            (symbol, pinning)
            for pinning in (1, 2, 3, 4)
            for symbol in (r.EMPTY_VECTOR, "thrA")
        ],
        start=1,
    )
)


def write_keio_replicate_table(path: Path, *, cross: str, dose: int) -> Path:
    """One synthetic deposit table: the six Keio rows with both replicates."""
    rows = [list(r.keio_table_columns(dose, cross))]
    for (index, well), label in zip(
        SYNTHETIC_KEIO_POSITIONS, SYNTHETIC_KEIO, strict=True
    ):
        row = [
            str(index),
            "1000",
            "1000",
            well,
            "1",
            well[2],
            well[3:],
            label,
            cross,
            "1000",
            repr(KEIO_R1),
            repr(KEIO_R2),
            repr(KEIO_R1),
            repr(KEIO_R2),
            repr(NEUTRAL),
        ]
        if cross != "Vector":
            row += [repr(NEUTRAL), repr(NEUTRAL), repr(NEUTRAL)]
        rows.append(row)
    path.write_text("\n".join(",".join(cell) for cell in rows) + "\n")
    return path


def write_legend_1536(path: Path) -> Path:
    """The synthetic 1,536 layout: four pinning labels per strain."""
    lines = ["Well 1536,CRISPRi_Plasmid,Replicate"]
    lines += [
        f"{well},{symbol},{pinning}" for well, symbol, pinning in SYNTHETIC_LEGEND
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_wt_lpp_table(path: Path, *, replicate: int) -> Path:
    """One synthetic dose-response table: the wells of ONE legend pinning."""
    want = r.WT_LPP_TABLE_PINNINGS[replicate]
    lines = ["Well,Gene,0aTC,5aTC,10aTC,50aTC,100aTC,500aTC,Condition"]
    for well, symbol, pinning in SYNTHETIC_LEGEND:
        if str(pinning) != want:
            continue
        lines.append(f"{well},{symbol},1,1,1,1,1,1,WT_LB")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_deposit(out: Path) -> dict[str, Path]:
    """Every synthetic deposit member, under the mirror names the loader links."""
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for cross in r.KEIO_TABLE_CROSS_NAMES:
        for dose in (0, 50, 500):
            name = r._keio_table_name(cross, dose)
            written[name] = write_keio_replicate_table(
                out / name, cross=cross, dose=dose
            )
    written[r.LEGEND_1536_FILE] = write_legend_1536(out / r.LEGEND_1536_FILE)
    for background in ("WT", "Lpp"):
        for medium in ("LB", "MOPS"):
            for replicate in (1, 2):
                name = r._wt_lpp_table_name(background, medium, replicate)
                written[name] = write_wt_lpp_table(out / name, replicate=replicate)
    (out / r.KEIO_README_FILE).write_text("K1 = Keio 384 Plate 1\nR1 = Replicate 1\n")
    written[r.KEIO_README_FILE] = out / r.KEIO_README_FILE
    (out / r.SCANS_README_FILE).write_text(
        "Each strain pinned in technical quadruplicate\n"
    )
    written[r.SCANS_README_FILE] = out / r.SCANS_README_FILE
    return written


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the released-shape constants to the synthetic release's shape."""
    monkeypatch.setattr(r, "KEIO_TABLE_ROWS", len(SYNTHETIC_KEIO))
    monkeypatch.setattr(r, "COLONIES_PER_PLATE_1536", 3)
    monkeypatch.setattr(r, "KEIO_ASSAY_PLATES", 2)
    monkeypatch.setattr(r, "PINNINGS_PER_STRAIN", 4)
    monkeypatch.setattr(r, "WELLS_PER_PINNING", 2)
    monkeypatch.setattr(r, "S1_ROWS", len(SYNTHETIC_WELLS))
    monkeypatch.setattr(r, "S1_FILLED_WELLS", len(SYNTHETIC_WELLS))
    monkeypatch.setattr(r, "S1_EMPTY_VECTOR_WELLS", 1)
    monkeypatch.setattr(r, "S2A_ROWS", len(SYNTHETIC_WELLS))
    monkeypatch.setattr(r, "S3_ROWS", len(SYNTHETIC_WELLS))
    monkeypatch.setattr(r, "S4A_ROWS", len(SYNTHETIC_KEIO))
    monkeypatch.setattr(r, "S4A_DISTINCT_LABELS", len(set(SYNTHETIC_KEIO)))
    monkeypatch.setattr(r, "S2A_GUIDE_ROWS", len(SYNTHETIC_WELLS) - 1)
    monkeypatch.setattr(
        r,
        "S2A_DISTINCT_TARGETS",
        len({symbol for _, _, symbol, _ in SYNTHETIC_WELLS} - {r.EMPTY_VECTOR}),
    )
    monkeypatch.setattr(r, "FIFTY_PERCENT_COUNT", 2)
    monkeypatch.setattr(r, "S4A_BLOCKS", SYNTHETIC_BLOCKS)
    monkeypatch.setattr(r, "S1_TYPO_WELL", TYPO_WELL)
    monkeypatch.setattr(r, "S1_TYPO_SYMBOL", TYPO_SYMBOL)
    monkeypatch.setattr(r, "GROWTH_TABLE_TYPO_SYMBOL", TYPO_LABEL)
    # ``yaaP`` is the synthetic assembly's pseudogene and is no guide target here, so it
    # stands in for the ``lpp`` query strain of Table S3.
    monkeypatch.setattr(r, "LPP_SYMBOL", "yaaP")


@pytest.fixture
def release(tmp_path: Path, pinned: None) -> dict[str, Path]:
    """The four synthetic workbooks plus the synthetic deposit members, on disk."""
    out = tmp_path / "release"
    out.mkdir()
    files = {
        r.TABLE_S1_FILE: write_table_s1(out / r.TABLE_S1_FILE),
        r.TABLE_S2_FILE: write_table_s2(out / r.TABLE_S2_FILE),
        r.TABLE_S3_FILE: write_table_s3(out / r.TABLE_S3_FILE),
        r.TABLE_S4_FILE: write_table_s4(out / r.TABLE_S4_FILE),
    }
    files.update(
        {f"zenodo/{name}": path for name, path in write_deposit(out / "zenodo").items()}
    )
    return files


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
def test_the_two_media_derive_from_shared_library_bases() -> None:
    """Both media join the shared library at their base, which L3 media_membership wants."""
    assert r.RACHWALSKI2024_LB_AGAR.base_medium in MEDIA_LIBRARY
    assert r.RACHWALSKI2024_MOPS_GLUCOSE_AGAR.base_medium in MEDIA_LIBRARY
    assert r.RACHWALSKI2024_LB_AGAR.state == "solid"
    assert r.RACHWALSKI2024_MOPS_GLUCOSE_AGAR.state == "solid"


def test_the_lb_plate_is_the_shared_lb_plus_fifteen_grams_of_agar() -> None:
    agar = r.RACHWALSKI2024_LB_AGAR.components[-1]
    assert agar.compound.name == "agar"
    assert agar.role is MediaComponentRole.gelling_agent
    assert agar.concentration is not None
    assert (agar.concentration.value, agar.concentration.unit) == (
        15.0,
        ConcentrationUnit.g_per_l,
    )
    assert "15 g/L agar" in str(agar.provenance[0].quote).replace("$", "").replace(
        "1 5 { \\mathfrak { g } } / \\mathsf { L }", "15 g/L"
    )


def test_the_minimal_plate_carries_glucose_with_no_amount_and_says_why() -> None:
    """The carbon source is named only by a column header, and its amount nowhere."""
    glucose = next(
        component
        for component in r.RACHWALSKI2024_MOPS_GLUCOSE_AGAR.components
        if component.compound.name == "D-glucose"
    )
    assert glucose.role is MediaComponentRole.carbon_source
    assert glucose.definition is ComponentDefinition.defined
    assert glucose.concentration is None
    assert r.RACHWALSKI2024_MOPS_GLUCOSE_AGAR.open_gaps == ["D-glucose"]
    assert glucose.provenance[0].quote == (
        "Normalized Growth MOPS-glucose with 500 ng/ml aTc (This study)"
    )
    assert glucose.provenance[0].provenance.source_uri == f"data/{r.TABLE_S2_FILE}"


def test_the_minimal_plate_uses_the_percent_the_mops_sentence_states() -> None:
    agar = r.RACHWALSKI2024_MOPS_GLUCOSE_AGAR.components[-1]
    assert agar.concentration is not None
    assert (agar.concentration.value, agar.concentration.unit) == (
        1.5,
        ConcentrationUnit.percent_w_v,
    )


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def test_an_induced_plate_stores_its_dose_in_micrograms_per_millilitre() -> None:
    """ng/mL is not a schema unit, so a dose is stored as the exact ug/mL thousandth."""
    perturbation = r.atc_perturbation(500.0)
    assert perturbation.compound.name == "anhydrotetracycline"
    assert perturbation.concentration.value == 0.5
    assert perturbation.concentration.unit is ConcentrationUnit.ug_per_ml
    assert perturbation.concentration.basis is DoseBasis.fixed
    assert r.atc_perturbation(5.0).concentration.value == 0.005


def test_an_uninduced_plate_has_no_inducer_to_perturb_with() -> None:
    with pytest.raises(ValueError, match="carry no aTc"):
        r.atc_perturbation(0.0)


def test_the_zero_dose_environment_carries_no_perturbation_at_all() -> None:
    """No aTc was added to those plates, so none is asserted."""
    assert r.environment(r.MEDIUM_LB, 0.0, timed=True).perturbations == []
    induced = r.environment(r.MEDIUM_LB, 50.0, timed=True)
    (inducer,) = induced.perturbations
    assert isinstance(inducer, SmallMoleculePerturbation)
    assert inducer.compound.name == "anhydrotetracycline"


def test_a_timed_plate_states_sixteen_hours_and_an_untimed_one_gaps_it() -> None:
    timed = r.environment(r.MEDIUM_MOPS, 10.0, timed=True)
    assert timed.duration_hours == 16.0
    assert timed.provenance_gaps == []
    untimed = r.environment(r.MEDIUM_LB, 50.0, timed=False)
    assert untimed.duration_hours is None
    assert untimed.gapped_fields() == {"duration_hours", "duration_generations"}
    assert all(
        gap.reason is ProvenanceGapReason.not_reported_by_primary
        for gap in untimed.provenance_gaps
    )


def test_every_environment_is_an_aerobic_thirty_seven_degree_plate() -> None:
    env = r.environment(r.MEDIUM_MOPS, 100.0, timed=True)
    assert env.aerobicity == "aerobic"
    assert env.temperature is not None
    assert env.temperature.value == 37.0


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
def test_a_collection_record_combines_its_two_technical_pinnings() -> None:
    """Tables S2A and S3: one culture, its two pinnings averaged as the release does."""
    phenotype = r.combined_technical_phenotype(0.4321, r.SCREEN_S2A, "A1")
    assert phenotype.fitness == 0.4321
    assert phenotype.n_samples == 2
    assert phenotype.sample_unit is SampleUnit.technical_replicate
    assert phenotype.screen_id == r.SCREEN_S2A
    assert phenotype.replicate_id == "A1"
    assert phenotype.label_name == "fitness"


def test_a_keio_record_is_one_biological_replicate_with_no_dispersion() -> None:
    """Table S4A: one assay-plate replicate, so n = 1 and nothing is averaged."""
    phenotype = r.replicate_phenotype(0.5, "K3:0A01:R2")
    assert phenotype.fitness == 0.5
    assert phenotype.n_samples == 1
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    assert phenotype.screen_id == r.SCREEN_S4A
    assert phenotype.replicate_id == "K3:0A01:R2"


def test_a_single_replicate_types_its_absent_dispersion_as_not_applicable() -> None:
    """An n = 1 record has no dispersion of its own, which is not the same as unreported."""
    phenotype = r.replicate_phenotype(0.5, "K1:0A01:R1")
    assert phenotype.gapped_fields() == {
        "fitness_uncertainty",
        "fitness_uncertainty_type",
        "fitness_se",
        "fitness_std",
    }
    assert phenotype.sample_unit is not None
    assert all(
        gap.reason is ProvenanceGapReason.not_applicable_single_measurement
        for gap in phenotype.provenance_gaps
    )


def test_a_combined_record_types_its_absent_dispersion_as_unreported() -> None:
    """The two pinnings exist, so their spread is unreleased rather than inapplicable."""
    phenotype = r.combined_technical_phenotype(0.5, r.SCREEN_S2A, "A1")
    assert phenotype.gapped_fields() == {
        "fitness_uncertainty",
        "fitness_uncertainty_type",
        "fitness_se",
        "fitness_std",
    }
    assert all(
        gap.reason is ProvenanceGapReason.not_reported_by_primary
        for gap in phenotype.provenance_gaps
    )


def test_the_replicate_unit_of_each_arm_is_sourced_and_not_guessed() -> None:
    """The deposit types both units, so neither is a typed gap any more."""
    technical = r.SOURCED_VALUES["s2_replicate_is_technical"]
    biological = r.SOURCED_VALUES["s4_replicate_is_biological"]
    assert technical.value == "technical_replicate"
    assert technical.quote == "Each strain pinned in technical quadruplicate"
    assert biological.value == "biological_replicate"
    assert biological.quote == "R1 = Replicate 1"
    assert technical.provenance.source_uri == f"data/zenodo/{r.SCANS_README_FILE}"
    assert biological.provenance.source_uri == f"data/zenodo/{r.KEIO_README_FILE}"


def test_the_reference_is_the_empty_vector_at_one_over_its_seventeen_wells() -> None:
    reference = r.reference_phenotype()
    assert reference.fitness == 1.0
    assert reference.n_samples == r.S1_EMPTY_VECTOR_WELLS == 17
    assert reference.screen_id is None
    assert reference.replicate_id is None


# --------------------------------------------------------------------------- #
# Genotype
# --------------------------------------------------------------------------- #
def _well(
    symbol: str = "thrA", spacer: str = "AAAAAAAAAAAAAAAAAAAA"
) -> r.CollectionWell:
    return r.CollectionWell(
        well="A2", symbol=symbol, spacer=spacer, essential="yes", function="f"
    )


def test_a_knockdown_carries_the_wells_own_spacer_and_one_guide() -> None:
    perturbation = r.knockdown(_well(), "BW25113_0002")
    assert isinstance(perturbation, CrisprInterferencePerturbation)
    assert perturbation.systematic_gene_name == "BW25113_0002"
    assert perturbation.perturbed_gene_name == "thrA"
    assert perturbation.gene_namespace == "ecoli_k12_bw25113_locus_tag"
    assert perturbation.crispr.effector == "dCas9"
    assert perturbation.crispr.guide_sequence == "AAAAAAAAAAAAAAAAAAAA"
    assert perturbation.crispr.n_guides == 1
    assert perturbation.expression_direction == "decreased"
    assert perturbation.identifier_mapping is not None
    assert perturbation.identifier_mapping.route == "gene_symbol"


def test_a_deletion_carries_the_keio_collection_and_its_cassette() -> None:
    perturbation = r.deletion("thrL", "BW25113_0001")
    assert isinstance(perturbation, DeletionStrainPerturbation)
    assert perturbation.collection == "Keio collection"
    assert perturbation.cassette == "kanamycin-resistance cassette"
    assert perturbation.construction is None
    assert r.deletion("thrL", "BW25113_0001", well="A1").construction is not None


def test_the_crossed_genotype_is_one_knockdown_and_one_deletion() -> None:
    """The pairing the dataset exists for: an essential gene reachable only by
    knockdown beside a non-essential gene reachable only by deletion.
    """
    genotype = Genotype(
        perturbations=[
            r.knockdown(_well(), "BW25113_0002"),
            r.deletion("thrL", "BW25113_0001"),
        ]
    )
    assert sorted(genotype.perturbation_types) == [
        "crispr_interference",
        "deletion_strain",
    ]
    assert len(genotype) == 2


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def test_table_s1_reads_its_filled_wells_in_plate_order(
    release: dict[str, Path],
) -> None:
    wells = r.read_table_s1(release[r.TABLE_S1_FILE])
    assert [well.well for well in wells] == [
        f"{row}{column}" for row, column, _, _ in SYNTHETIC_WELLS
    ]
    assert wells[0].is_empty_vector
    assert wells[1].spacer == "AAAAAAAAAAAAAAAAAAAA"


def test_table_s1_refuses_a_header_that_moved(release: dict[str, Path]) -> None:
    path = release[r.TABLE_S1_FILE]
    book = openpyxl.load_workbook(path)
    book["Sheet1"]["C1"] = "guide"
    book.save(path)
    with pytest.raises(r.TableLayoutError, match="Table S1 header"):
        r.read_table_s1(path)


def test_table_s1_refuses_a_row_count_that_moved(
    release: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r, "S1_ROWS", 999)
    with pytest.raises(r.TableLayoutError, match="Table S1 has 7 rows"):
        r.read_table_s1(release[r.TABLE_S1_FILE])


def test_table_s1_refuses_a_guide_well_with_no_spacer(
    tmp_path: Path, pinned: None
) -> None:
    path = write_table_s1(tmp_path / "s1.xlsx")
    book = openpyxl.load_workbook(path)
    book["Sheet1"]["G3"] = None
    book.save(path)
    with pytest.raises(r.TableLayoutError, match="no guide sequence"):
        r.read_table_s1(path)


def test_table_s1_refuses_an_empty_vector_count_that_moved(
    release: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r, "S1_EMPTY_VECTOR_WELLS", 17)
    with pytest.raises(r.TableLayoutError, match="empty-vector wells"):
        r.read_table_s1(release[r.TABLE_S1_FILE])


def test_table_s2a_reads_two_six_dose_blocks(release: dict[str, Path]) -> None:
    labels, blocks = r.read_table_s2a(release[r.TABLE_S2_FILE])
    assert labels == list(SYNTHETIC_LABELS)
    assert [block.medium for block in blocks] == [r.MEDIUM_LB, r.MEDIUM_MOPS]
    assert blocks[0].doses == r.SIX_DOSES_NG_PER_ML
    assert len(blocks[0].values) == len(SYNTHETIC_WELLS)
    assert blocks[0].values[1][4] == REDUCED


def test_table_s2a_refuses_a_dose_header_that_moved(release: dict[str, Path]) -> None:
    path = release[r.TABLE_S2_FILE]
    book = openpyxl.load_workbook(path)
    book["ST2A"]["F2"] = "200 ng/ml aTC"
    book.save(path)
    with pytest.raises(r.TableLayoutError, match="Table S2A block 1 header"):
        r.read_table_s2a(path)


def test_table_s2a_refuses_a_blank_growth_cell(release: dict[str, Path]) -> None:
    path = release[r.TABLE_S2_FILE]
    book = openpyxl.load_workbook(path)
    book["ST2A"]["C4"] = None
    book.save(path)
    with pytest.raises(r.TableLayoutError, match="is blank"):
        r.read_table_s2a(path)


def test_table_s3_reads_a_wild_type_block_then_the_deletion_block(
    release: dict[str, Path],
) -> None:
    labels, blocks = r.read_table_s3(release[r.TABLE_S3_FILE])
    assert len(labels) == len(SYNTHETIC_WELLS)
    assert [block.background for block in blocks] == [None, r.LPP_SYMBOL]
    assert all(block.medium == r.MEDIUM_MOPS for block in blocks)


def test_table_s3_refuses_a_group_header_that_moved(release: dict[str, Path]) -> None:
    path = release[r.TABLE_S3_FILE]
    book = openpyxl.load_workbook(path)
    book["ST3"]["B1"] = "Something Else"
    book.save(path)
    with pytest.raises(r.TableLayoutError, match="group header"):
        r.read_table_s3(path)


def test_table_s4a_reads_the_vector_block_then_the_three_knockdowns(
    release: dict[str, Path],
) -> None:
    labels, blocks = r.read_table_s4a(release[r.TABLE_S4_FILE])
    assert labels == list(SYNTHETIC_KEIO)
    assert [block.knockdown for block in blocks] == [None, "thrA", "hokC", "yaaX"]
    assert all(block.doses == r.THREE_DOSES_NG_PER_ML for block in blocks)
    assert all(block.medium == r.MEDIUM_LB for block in blocks)


def test_table_s4a_refuses_a_block_header_that_moved(release: dict[str, Path]) -> None:
    path = release[r.TABLE_S4_FILE]
    book = openpyxl.load_workbook(path)
    book["ST4A"]["F1"] = "Average Normalized Growth of something else"
    book.save(path)
    with pytest.raises(r.TableLayoutError, match="Table S4A block at column 5"):
        r.read_table_s4a(path)


def test_table_s4a_refuses_a_distinct_label_count_that_moved(
    release: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r, "S4A_DISTINCT_LABELS", 4017)
    with pytest.raises(r.TableLayoutError, match="distinct labels"):
        r.read_table_s4a(release[r.TABLE_S4_FILE])


def test_a_missing_sheet_row_is_refused(release: dict[str, Path]) -> None:
    with pytest.raises(r.TableLayoutError, match="has no row"):
        r._header(release[r.TABLE_S1_FILE], "Sheet1", 99)


# --------------------------------------------------------------------------- #
# The plate-order join and the three build checks
# --------------------------------------------------------------------------- #
def test_the_growth_tables_are_the_collection_layout_position_for_position(
    release: dict[str, Path],
) -> None:
    """One label disagreement is expected -- the gpsA / gspA transposition -- and the
    join asserts it is the ONLY one.
    """
    wells = r.read_table_s1(release[r.TABLE_S1_FILE])
    labels, _ = r.read_table_s2a(release[r.TABLE_S2_FILE])
    join = r.join_growth_rows_to_wells(labels, wells, table=r.SCREEN_S2A)
    assert join.rows == len(wells)
    assert join.disagreements == [(TYPO_WELL, TYPO_SYMBOL, TYPO_LABEL)]


def test_a_second_label_disagreement_stops_the_build(release: dict[str, Path]) -> None:
    wells = r.read_table_s1(release[r.TABLE_S1_FILE])
    labels = list(SYNTHETIC_LABELS)
    labels[2] = "somethingElse"
    with pytest.raises(
        r.TableLayoutError, match="disagrees with Table S1's plate order"
    ):
        r.join_growth_rows_to_wells(labels, wells, table=r.SCREEN_S2A)


def test_a_growth_table_with_a_different_row_count_stops_the_build(
    release: dict[str, Path],
) -> None:
    wells = r.read_table_s1(release[r.TABLE_S1_FILE])
    with pytest.raises(r.TableLayoutError, match="against Table S1's"):
        r.join_growth_rows_to_wells(["thrA"], wells, table=r.SCREEN_S3)


def test_the_paper_s_own_fifty_percent_count_is_reproduced(
    release: dict[str, Path],
) -> None:
    wells = r.read_table_s1(release[r.TABLE_S1_FILE])
    _, blocks = r.read_table_s2a(release[r.TABLE_S2_FILE])
    check = r.check_fifty_percent_reduction(wells, blocks)
    assert check["at_or_below_half"] == 2
    assert check["guide_rows"] == len(SYNTHETIC_WELLS) - 1
    assert check["threshold"] == 0.5
    assert "272" in str(check["quote"])


def test_a_fifty_percent_count_that_moved_stops_the_build(
    tmp_path: Path, pinned: None
) -> None:
    wells = r.read_table_s1(write_table_s1(tmp_path / "s1.xlsx"))
    _, blocks = r.read_table_s2a(write_table_s2(tmp_path / "s2.xlsx", reduced=3))
    with pytest.raises(r.TableLayoutError, match="at or below 0.5"):
        r.check_fifty_percent_reduction(wells, blocks)


def test_a_guide_row_count_that_moved_stops_the_build(
    release: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r, "S2A_GUIDE_ROWS", 360)
    wells = r.read_table_s1(release[r.TABLE_S1_FILE])
    _, blocks = r.read_table_s2a(release[r.TABLE_S2_FILE])
    with pytest.raises(r.TableLayoutError, match="guide rows over"):
        r.check_fifty_percent_reduction(wells, blocks)


def test_the_empty_vector_mean_is_one_in_every_column(release: dict[str, Path]) -> None:
    wells = r.read_table_s1(release[r.TABLE_S1_FILE])
    _, blocks = r.read_table_s2a(release[r.TABLE_S2_FILE])
    check = r.check_empty_vector_mean_is_one(wells, blocks, table=r.SCREEN_S2A)
    assert check["columns"] == 12
    assert check["worst_abs_deviation"] == 0.0


def test_an_empty_vector_mean_off_one_stops_the_build(release: dict[str, Path]) -> None:
    path = release[r.TABLE_S2_FILE]
    book = openpyxl.load_workbook(path)
    book["ST2A"]["B3"] = 0.5
    book.save(path)
    wells = r.read_table_s1(release[r.TABLE_S1_FILE])
    _, blocks = r.read_table_s2a(path)
    with pytest.raises(r.TableLayoutError, match="empty-vector mean is not 1"):
        r.check_empty_vector_mean_is_one(wells, blocks, table=r.SCREEN_S2A)


def test_table_s2c_fixes_which_block_is_lb(release: dict[str, Path]) -> None:
    labels, blocks = r.read_table_s2a(release[r.TABLE_S2_FILE])
    check = r.check_s2a_block_order(release[r.TABLE_S2_FILE], labels, blocks)
    assert check["matched_rows"] == 2
    assert check["mismatched_rows"] == []


def test_a_block_order_table_s2c_disagrees_with_stops_the_build(
    release: dict[str, Path],
) -> None:
    path = release[r.TABLE_S2_FILE]
    book = openpyxl.load_workbook(path)
    book["ST2C"]["B3"] = 0.123
    book["ST2C"]["B4"] = 0.456
    book.save(path)
    labels, blocks = r.read_table_s2a(path)
    with pytest.raises(r.TableLayoutError, match="block order on only"):
        r.check_s2a_block_order(path, labels, blocks)


def test_table_s2c_refuses_a_header_that_moved(release: dict[str, Path]) -> None:
    path = release[r.TABLE_S2_FILE]
    book = openpyxl.load_workbook(path)
    book["ST2C"]["B2"] = "Normalized Growth LB"
    book.save(path)
    labels, blocks = r.read_table_s2a(path)
    with pytest.raises(r.TableLayoutError, match="Table S2C header"):
        r.check_s2a_block_order(path, labels, blocks)


# --------------------------------------------------------------------------- #
# Resolution against the synthetic BW25113 assembly
# --------------------------------------------------------------------------- #
@pytest.fixture
def bw25113(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12BW25113Genome:
    """The synthetic BW25113 genome; the network refuses."""
    files = write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    return EcoliK12BW25113Genome(genome_root=str(tmp_path / "bw25113"), overwrite=True)


def test_resolution_keeps_the_symbols_the_assembly_carries(
    bw25113: EcoliK12BW25113Genome,
) -> None:
    resolution = r.resolve_labels(
        ["thrA", "hokC", "yaaX", "thrL"], bw25113, label="synthetic", min_resolved=0.9
    )
    assert resolution.locus_by_label == {
        "thrA": "BW25113_0002",
        "hokC": "BW25113_4412",
        "yaaX": "BW25113_0008",
        "thrL": "BW25113_0001",
    }
    assert all(not names for names in resolution.unwritable.values())


def test_the_stored_spelling_is_the_annotations_own_symbol(
    bw25113: EcoliK12BW25113Genome,
) -> None:
    """One spelling per locus: a Keio strain id becomes the locus's gene symbol, and the
    released id is kept as the perturbation's source identifier.
    """
    assert r.canonical_symbol(bw25113, "BW25113_0002") == "thrA"
    resolution = r.resolve_labels(
        ["JW0001", "thrL"], bw25113, label="synthetic", min_resolved=0.5
    )
    assert resolution.locus_by_label["JW0001"] == "BW25113_0002"
    assert resolution.symbol_by_label["JW0001"] == "thrA"
    perturbation = r.deletion(
        "JW0001", "BW25113_0002", symbol=resolution.symbol_by_label["JW0001"]
    )
    assert perturbation.perturbed_gene_name == "thrA"
    assert perturbation.identifier_mapping is not None
    assert perturbation.identifier_mapping.source_identifier == "JW0001"


def test_a_pseudogene_locus_is_writable(bw25113: EcoliK12BW25113Genome) -> None:
    """A Keio deletion of a pseudogene is a real strain; the locus resolves to itself."""
    resolution = r.resolve_labels(
        ["thrA", "yaaP"], bw25113, label="synthetic", min_resolved=0.5
    )
    assert resolution.locus_by_label["yaaP"] == "BW25113_0004"


def test_each_resolution_rule_claims_its_own_labels(
    bw25113: EcoliK12BW25113Genome,
) -> None:
    """A retired label, a merged-locus collision and an ambiguous label, each named."""
    resolution = r.resolve_labels(
        ["thrA", "notAGene", "proB", "ECK0005", "ECK0099"],
        bw25113,
        label="synthetic",
        min_resolved=0.1,
    )
    assert "notAGene" in resolution.unwritable[r.DROP_NOT_IN_ANNOTATION]
    assert "proB" in resolution.unwritable[r.DROP_MERGED_LOCUS]
    assert "ECK0099" in resolution.unwritable[r.DROP_MERGED_LOCUS]
    assert "ECK0005" in resolution.unwritable[r.DROP_AMBIGUOUS]
    assert "proB" not in resolution.locus_by_label
    assert resolution.locus_by_label["thrA"] == "BW25113_0002"


def test_a_release_below_the_resolution_threshold_stops_instead_of_dropping(
    bw25113: EcoliK12BW25113Genome,
) -> None:
    with pytest.raises(Exception, match="resolv"):
        r.resolve_labels(
            ["notAGene", "alsoNot", "thrA"],
            bw25113,
            label="synthetic",
            min_resolved=0.99,
        )


def test_a_kept_tag_outside_the_assembly_stops_the_build(
    bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    from torchcell.datasets.bacteria_common import reconcile_locus_tags as real

    def stub(genome: Any, names: Any, *, label: str) -> Any:
        stored, report = real(genome, names, label=label)
        return pd.Series(["BW25113_9999"] * len(stored), index=stored.index), report

    monkeypatch.setattr(r, "reconcile_locus_tags", stub)
    with pytest.raises(RuntimeError, match="not loci of the pinned assembly"):
        r.resolve_labels(["thrA"], bw25113, label="synthetic", min_resolved=0.1)


# --------------------------------------------------------------------------- #
# Row selection
# --------------------------------------------------------------------------- #
def _resolution(mapping: dict[str, str]) -> r.Resolution:
    from torchcell.datasets.bacteria_common import LocusTagReconciliation

    return r.Resolution(
        locus_by_label=mapping,
        symbol_by_label=dict(mapping and {k: k for k in mapping}),
        unwritable={
            r.DROP_NOT_IN_ANNOTATION: [],
            r.DROP_MERGED_LOCUS: [],
            r.DROP_AMBIGUOUS: [],
        },
        reconciliation=LocusTagReconciliation.model_construct(
            label="synthetic",
            assembly_set="ecoli_K12_BW25113_ASM75055v1",
            gene_namespace="ecoli_k12_bw25113_locus_tag",
            unique_names=len(mapping),
            status_histogram={},
            layer_histogram={},
            remapped=len(mapping),
            kept_on_collision=(),
            retired_kept=(),
            ambiguous_kept={},
        ),
    )


def test_an_empty_vector_well_is_a_record_with_an_empty_genotype(
    release: dict[str, Path],
) -> None:
    """A control replicate the source releases is a record; its well tells it apart."""
    wells = r.read_table_s1(release[r.TABLE_S1_FILE])
    _, blocks = r.read_table_s2a(release[r.TABLE_S2_FILE])
    rows = list(
        r._guide_rows(
            wells,
            blocks,
            _resolution({"thrA": "BW25113_0002", "hokC": "BW25113_4412"}),
            screen_id=r.SCREEN_S2A,
        )
    )
    controls = [row for row in rows if not row.genotype.perturbations]
    assert len(controls) == 2  # the one empty-vector well, in each medium block
    assert {row.replicate_id for row in controls} == {"A1"}
    assert r.EMPTY_VECTOR not in {
        p.perturbed_gene_name for row in rows for p in row.genotype.perturbations
    }


def test_a_construct_at_two_wells_is_two_records_keyed_by_well(
    release: dict[str, Path],
) -> None:
    """A2 and A7 share a spacer, so they are two pinned strains at two positions."""
    wells = r.read_table_s1(release[r.TABLE_S1_FILE])
    _, blocks = r.read_table_s2a(release[r.TABLE_S2_FILE])
    rows = list(
        r._guide_rows(
            wells,
            blocks,
            _resolution({"thrA": "BW25113_0002", "hokC": "BW25113_4412"}),
            screen_id=r.SCREEN_S2A,
        )
    )
    thra = [
        row
        for row in rows
        if any(p.perturbed_gene_name == "thrA" for p in row.genotype.perturbations)
    ]
    assert len(thra) == 4  # two wells x two media blocks
    assert {row.replicate_id for row in thra} == {"A2", "A7"}


def test_a_background_block_puts_a_deletion_beside_every_knockdown(
    release: dict[str, Path],
) -> None:
    wells = r.read_table_s1(release[r.TABLE_S1_FILE])
    _, blocks = r.read_table_s3(release[r.TABLE_S3_FILE])
    rows = [
        row
        for row in r._guide_rows(
            wells,
            blocks,
            _resolution({"hokC": "BW25113_4412", r.LPP_SYMBOL: "BW25113_0005"}),
            screen_id=r.SCREEN_S3,
        )
        if any(p.perturbed_gene_name == "hokC" for p in row.genotype.perturbations)
    ]
    wild_type, deleted = rows
    assert len(wild_type.genotype) == 1
    assert len(deleted.genotype) == 2
    assert sorted(deleted.genotype.perturbation_types) == [
        "crispr_interference",
        "deletion_strain",
    ]


def _knockdowns() -> dict[str, CrisprInterferencePerturbation]:
    """One knockdown per Table S4A block, on the synthetic assembly's loci."""
    return {
        "thrA": r.knockdown(_well("thrA"), "BW25113_0002"),
        "hokC": r.knockdown(_well("hokC", "CCCCCCCCCCCCCCCCCCCC"), "BW25113_4412"),
        "yaaX": r.knockdown(_well("yaaX", "GGGGGGGGGGGGGGGGGGGG"), "BW25113_0008"),
    }


def _keio_tables(release: dict[str, Path]) -> dict[tuple[str, int], Any]:
    """Every synthetic deposit table, read the way the build reads them."""
    return {
        (cross, dose): r.read_keio_replicate_table(
            release[f"zenodo/{r._keio_table_name(cross, dose)}"], cross=cross, dose=dose
        )
        for cross in r.KEIO_TABLE_CROSS_NAMES
        for dose in (0, 50, 500)
    }


def test_a_deposit_table_derives_its_assay_plate_from_the_row_index(
    release: dict[str, Path],
) -> None:
    """Three colonies to a plate over two plates, so the plates run 1, 1, 1, 2, 2, 2."""
    rows = r.read_keio_replicate_table(
        release[f"zenodo/{r._keio_table_name('Vector', 0)}"], cross="Vector", dose=0
    )
    assert [row.plate for row in rows] == [1, 1, 1, 2, 2, 2]
    assert [row.values for row in rows] == [(KEIO_R1, KEIO_R2)] * len(SYNTHETIC_KEIO)
    assert len({(row.plate, row.well) for row in rows}) == len(SYNTHETIC_KEIO)


def test_a_deposit_table_refuses_a_header_that_moved(release: dict[str, Path]) -> None:
    path = release[f"zenodo/{r._keio_table_name('Vector', 0)}"]
    path.write_text(path.read_text().replace("TwoPass_Avg", "TwoPass_Average", 1))
    with pytest.raises(r.TableLayoutError, match="header"):
        r.read_keio_replicate_table(path, cross="Vector", dose=0)


def test_a_deposit_table_refuses_a_repeated_position(release: dict[str, Path]) -> None:
    """Without a unique position a replicate cannot be labeled, so the build stops."""
    path = release[f"zenodo/{r._keio_table_name('Vector', 0)}"]
    lines = path.read_text().splitlines()
    lines[2] = lines[2].replace(",0A02,", ",0A01,")
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(r.TableLayoutError, match="position"):
        r.read_keio_replicate_table(path, cross="Vector", dose=0)


def test_a_keio_row_is_two_records_one_per_assay_plate_replicate(
    release: dict[str, Path],
) -> None:
    resolution = _resolution({"thrL": "BW25113_0001"})
    records = list(
        r._keio_replicate_records(_keio_tables(release), resolution, _knockdowns())
    )
    # one writable label (thrL, one row) x 4 crosses x 3 doses x 2 replicates
    assert len(records) == 24
    assert {record.replicate_id for record in records} == {"K1:0A01:R1", "K1:0A01:R2"}
    assert {record.value for record in records} == {KEIO_R1, KEIO_R2}
    assert sorted({len(record.genotype) for record in records}) == [1, 2]


def test_a_keio_label_heading_two_rows_is_two_records_keyed_by_position(
    release: dict[str, Path],
) -> None:
    """The hokC label heads rows 3 and 6 at two distinct positions, so both stay."""
    resolution = _resolution({"thrL": "BW25113_0001", "hokC": "BW25113_4412"})
    records = [
        record
        for record in r._keio_replicate_records(
            _keio_tables(release), resolution, _knockdowns()
        )
        if any(
            p.perturbed_gene_name == "hokC" and p.perturbation_type == "deletion_strain"
            for p in record.genotype.perturbations
        )
    ]
    # two rows x 4 crosses x 3 doses x 2 replicates
    assert len(records) == 48
    assert {record.replicate_id for record in records} == {
        "K1:0A03:R1",
        "K1:0A03:R2",
        "K2:0A02:R1",
        "K2:0A02:R2",
    }


def test_the_deposit_must_reproduce_the_workbook_before_any_record_is_written(
    release: dict[str, Path],
) -> None:
    _, blocks = r.read_table_s4a(release[r.TABLE_S4_FILE])
    labels, _ = r.read_table_s4a(release[r.TABLE_S4_FILE])
    check = r.check_deposit_reproduces_table_s4a(_keio_tables(release), labels, blocks)
    assert check["cells_compared"] == len(SYNTHETIC_KEIO) * 12
    assert check["worst_abs_cell_difference"] <= r.DEPOSIT_TOLERANCE
    assert check["worst_abs_replicate_mean_difference"] <= r.DEPOSIT_TOLERANCE


def test_a_deposit_that_disagrees_with_the_workbook_stops_the_build(
    release: dict[str, Path],
) -> None:
    labels, blocks = r.read_table_s4a(release[r.TABLE_S4_FILE])
    tables = _keio_tables(release)
    moved = tables[("Vector", 0)]
    tables[("Vector", 0)] = [
        moved[0].model_copy(update={"released_mean": moved[0].released_mean + 0.5}),
        *moved[1:],
    ]
    with pytest.raises(r.TableLayoutError, match="does not reproduce"):
        r.check_deposit_reproduces_table_s4a(tables, labels, blocks)


def test_the_collection_tables_map_onto_one_pinning_label_each(
    release: dict[str, Path],
) -> None:
    """This is what types the Table S2A and S3 replicate as technical."""
    check = r.check_collection_replicate_is_technical(
        release[f"zenodo/{r.LEGEND_1536_FILE}"],
        {
            (background, medium, replicate): release[
                f"zenodo/{r._wt_lpp_table_name(background, medium, replicate)}"
            ]
            for background in ("WT", "Lpp")
            for medium in ("LB", "MOPS")
            for replicate in (1, 2)
        },
    )
    assert check["pinnings_per_strain"] == {"1": 2, "2": 2, "3": 2, "4": 2}
    assert set(check["table_pinning"].values()) == {"3", "4"}


def test_a_collection_table_on_two_pinnings_stops_the_build(
    release: dict[str, Path],
) -> None:
    path = release[f"zenodo/{r._wt_lpp_table_name('WT', 'LB', 1)}"]
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace("0A05", "0A07")
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(r.TableLayoutError, match="pinnings"):
        r.check_collection_replicate_is_technical(
            release[f"zenodo/{r.LEGEND_1536_FILE}"], {("WT", "LB", 1): path}
        )


# --------------------------------------------------------------------------- #
# The rule table and what is not loaded
# --------------------------------------------------------------------------- #
def test_only_the_genome_rules_still_drop_a_released_value() -> None:
    """Every replicate rule was retired by the deposit; the three left are a genome gate."""
    rules = [rule for rule, _ in r.ROW_RULES]
    assert len(rules) == len(set(rules)) == 3
    assert all(len(description) > 40 for _, description in r.ROW_RULES)
    assert set(rules) == {
        r.DROP_NOT_IN_ANNOTATION,
        r.DROP_MERGED_LOCUS,
        r.DROP_AMBIGUOUS,
    }
    assert all("GENOME GATE" in description for _, description in r.ROW_RULES)


def test_every_retired_rule_names_the_key_that_retired_it() -> None:
    retired = [rule for rule, _ in r.RETIRED_ROW_RULES]
    assert retired == [
        "row_is_an_empty_vector_control",
        "construct_sits_in_more_than_one_collection_well",
        "label_heads_more_than_one_row",
    ]
    assert all(
        description.startswith("RETIRED 2026-11-25")
        for _, description in r.RETIRED_ROW_RULES
    )


def test_the_four_consumed_workbooks_are_the_pmc_supplementary_files() -> None:
    assert [raw.name for raw in r.RAW_FILES] == [
        "mmc2.xlsx",
        "mmc3.xlsx",
        "mmc4.xlsx",
        "mmc5.xlsx",
    ]
    assert all(raw.mirror_relpath == f"data/{raw.name}" for raw in r.RAW_FILES)
    assert all(raw.bucket_key.startswith("PMC10832289.1/") for raw in r.RAW_FILES)
    assert all(
        raw.retrieval.method == "pmc_cloud"
        and raw.retrieval.retriever == "torchcell.literature.retrieve.pmc_cloud_object"
        for raw in r.RAW_FILES
    )


def test_the_deposit_members_are_read_out_of_the_two_pinned_archives() -> None:
    """23 members: the 12 per-replicate Keio tables, 8 collection tables, 3 legends."""
    assert len(r.ZENODO_FILES) == 23
    assert sum(1 for m in r.ZENODO_FILES if m.name.startswith("keio_")) == 13
    assert all(m.mirror_relpath == f"data/zenodo/{m.name}" for m in r.ZENODO_FILES)
    assert {m.archive for m in r.ZENODO_FILES} == {r.KEIO_ZIP, r.WT_LPP_ZIP}
    for member in r.ZENODO_FILES:
        record = member.retrieval
        assert record.method == "zenodo"
        assert record.retriever == "torchcell.literature.retrieve.zip_member"
        assert record.params["member"] == member.member
        assert record.params["container_sha256"] == r.ZIP_SHA256[member.archive]
        assert record.params["url"] == r.zenodo_zip_url(member.archive)
    assert set(r.ZIP_SHA256) == set(r.ZIP_BYTES) == {r.KEIO_ZIP, r.WT_LPP_ZIP}
    assert all(len(digest) == 64 for digest in r.ZIP_SHA256.values())


def test_what_is_not_loaded_names_the_images_the_subsets_and_the_ratios() -> None:
    text = " ".join(r.NOT_MIRRORED)
    assert "zenodo.10214517" in text
    assert "Table S4B" in text
    assert "Fold Change Growth" in text
    assert "GeneInteractionPhenotype" in text
    assert "mmc6.pdf" in text
    # the archives are no longer dismissed as images plus code (#1203 item 7)
    assert "ARE consumed" in text


def test_the_record_counts_add_up_from_the_writable_rows() -> None:
    """MEASURED on the real release: 112,872 records, 2.3x the 48,900 stored before."""
    assert r.EXPECTED_RECORDS == 112_872
    assert r.EXPECTED_CROSSED_RECORDS == 80_076
    assert r.EXPECTED_EMPTY_GENOTYPE_RECORDS == 306
    assert r.EXPECTED_WRITABLE_WELLS == 373
    assert r.EXPECTED_WRITABLE_KEIO_ROWS == 4330
    assert (
        r.EXPECTED_WRITABLE_WELLS * 12 * 2 + r.EXPECTED_WRITABLE_KEIO_ROWS * 4 * 3 * 2
        == r.EXPECTED_RECORDS
    )
    assert (
        r.EXPECTED_WRITABLE_WELLS - r.S1_EMPTY_VECTOR_WELLS
    ) * 6 + r.EXPECTED_WRITABLE_KEIO_ROWS * 3 * 3 * 2 == r.EXPECTED_CROSSED_RECORDS
    assert r.EXPECTED_EMPTY_GENOTYPE_RECORDS == r.S1_EMPTY_VECTOR_WELLS * 18


def test_the_dataset_is_registered_under_its_class_name() -> None:
    assert dataset_registry["CrispriCrossRachwalski2024Dataset"] is (
        r.CrispriCrossRachwalski2024Dataset
    )


def test_the_library_dir_points_at_this_keys_literature_mirror() -> None:
    assert r.library_dir("/root") == Path("/root/torchcell-library") / r.CITATION_KEY
    assert r.raw_mirror_dir("/root") == Path("/root/torchcell-raw") / r.CITATION_KEY


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
@pytest.fixture
def deposited(
    release: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Path], Path]:
    """A tmp DATA_ROOT whose raw mirror holds the synthetic workbooks and deposit."""
    files = tuple(
        r.RawFile(
            name=name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size,
            description=f"synthetic {name}",
        )
        for name, path in release.items()
        if not name.startswith("zenodo/")
    )
    members = tuple(
        r.ZenodoMember(
            name=name.removeprefix("zenodo/"),
            archive=(
                r.WT_LPP_ZIP
                if name.removeprefix("zenodo/")
                in {r.LEGEND_1536_FILE, r.SCANS_README_FILE}
                or "_R" in name
                else r.KEIO_ZIP
            ),
            member=f"synthetic/{name.removeprefix('zenodo/')}",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size,
            description=f"synthetic {name}",
        )
        for name, path in release.items()
        if name.startswith("zenodo/")
    )
    monkeypatch.setattr(r, "RAW_FILES", files)
    monkeypatch.setattr(r, "ZENODO_FILES", members)
    monkeypatch.setattr(
        r, "ZENODO_FILES_BY_NAME", {member.name: member for member in members}
    )
    monkeypatch.setattr(
        r,
        "DATA_SHA256",
        {f.name: f.sha256 for f in files}
        | {f"zenodo/{m.name}": m.sha256 for m in members},
    )
    monkeypatch.setattr(r, "RAW_FILES_BY_NAME", {f.name: f for f in files})
    data_root = tmp_path / "data_root"
    r.deposit_raw_mirror(sources=release, data_root=str(data_root))
    return release, data_root


def test_the_deposit_is_idempotent_and_refuses_a_changed_mirror_file(
    deposited: tuple[dict[str, Path], Path],
) -> None:
    release, data_root = deposited
    assert r.deposit_raw_mirror(sources=release, data_root=str(data_root)) == (
        r.raw_mirror_dir(str(data_root))
    )
    mirrored = r.raw_mirror_dir(str(data_root)) / f"data/{r.TABLE_S1_FILE}"
    mirrored.write_bytes(b"not the released bytes")
    with pytest.raises(RuntimeError, match="different sha256; refusing"):
        r.deposit_raw_mirror(sources=release, data_root=str(data_root))


def test_the_deposit_refuses_bytes_that_do_not_match_the_pin(
    deposited: tuple[dict[str, Path], Path], tmp_path: Path
) -> None:
    release, data_root = deposited
    other = tmp_path / "other.xlsx"
    other.write_bytes(b"not the released bytes")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        r.deposit_raw_mirror(
            sources={**release, r.TABLE_S1_FILE: other}, data_root=str(data_root)
        )


def test_the_deposit_refuses_a_missing_source(
    deposited: tuple[dict[str, Path], Path],
) -> None:
    release, data_root = deposited
    partial = {k: v for k, v in release.items() if k != r.TABLE_S4_FILE}
    with pytest.raises(KeyError, match=r.TABLE_S4_FILE):
        r.deposit_raw_mirror(sources=partial, data_root=str(data_root))


def test_the_manifest_records_the_rerunnable_pmc_retrieval(
    deposited: tuple[dict[str, Path], Path],
) -> None:
    _, data_root = deposited
    manifest = r.load_manifest(str(data_root))
    assert [record.path for record in manifest.files] == [
        f"data/{name}"
        for name in (r.TABLE_S1_FILE, r.TABLE_S2_FILE, r.TABLE_S3_FILE, r.TABLE_S4_FILE)
    ] + [member.mirror_relpath for member in r.ZENODO_FILES]
    record = manifest.files[0]
    assert record.retrieval is not None
    assert record.retrieval.method == "pmc_cloud"
    assert record.retrieval.params == {"key": f"PMC10832289.1/{r.TABLE_S1_FILE}"}
    assert (
        r.manifest_sha256(manifest, f"data/{r.TABLE_S1_FILE}")
        == (r.DATA_SHA256[r.TABLE_S1_FILE])
    )
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        r.manifest_sha256(manifest, "data/nope.xlsx")
    assert manifest.provenance_complete is True


# --------------------------------------------------------------------------- #
# The end-to-end build, on the synthetic release
# --------------------------------------------------------------------------- #
BW25113_PIN_ACCESSION = "GCA_000750555.1"


def _pin_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``assembly_reference`` so no assembly report has to be served."""
    from torchcell.datamodels.schema import ReferenceGenome

    reference = ReferenceGenome(
        species="Escherichia coli",
        strain="BW25113",
        assembly_set="ecoli_K12_BW25113_ASM75055v1",
        assembly_accession=BW25113_PIN_ACCESSION,
    )
    monkeypatch.setattr(r, "assembly_reference", lambda strain, **_: reference)


@pytest.fixture
def built(
    deposited: tuple[dict[str, Path], Path],
    bw25113: EcoliK12BW25113Genome,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Any:
    """Build the dataset from the synthetic release into the tmp DATA_ROOT."""
    _, data_root = deposited
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _pin_assembly(monkeypatch)
    # Only notAGene is unwritable, so six of the seven wells are records: the empty
    # vector, thrA at BOTH its wells, hokC, yaaX and thrL. Of the six Keio rows five are
    # writable, hokC's two rows among them, since their positions tell them apart.
    monkeypatch.setattr(r, "EXPECTED_WRITABLE_WELLS", 6)
    monkeypatch.setattr(r, "EXPECTED_WRITABLE_KEIO_ROWS", 5)
    monkeypatch.setattr(r, "EXPECTED_RECORDS", 6 * 12 * 2 + 5 * 4 * 3 * 2)
    monkeypatch.setattr(r, "EXPECTED_CROSSED_RECORDS", 5 * 6 + 5 * 3 * 3 * 2)
    monkeypatch.setattr(r, "EXPECTED_EMPTY_GENOTYPE_RECORDS", 1 * 18)
    monkeypatch.setattr(r, "MIN_RESOLVED_TARGETS", 0.5)
    monkeypatch.setattr(r, "MIN_RESOLVED_KEIO", 0.5)
    dataset = r.CrispriCrossRachwalski2024Dataset(
        root=str(tmp_path / "store"), ecoli_genome=bw25113
    )
    yield dataset
    dataset.close_lmdb()


def test_the_build_writes_one_record_per_strain_condition(built: Any) -> None:
    """6 writable wells x 24 columns, plus 5 Keio rows x 4 crosses x 3 doses x 2."""
    assert len(built) == 6 * 12 * 2 + 5 * 4 * 3 * 2 == 264


def test_the_build_refuses_a_record_count_that_does_not_match_the_tables(
    deposited: tuple[dict[str, Path], Path],
    bw25113: EcoliK12BW25113Genome,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, data_root = deposited
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _pin_assembly(monkeypatch)
    monkeypatch.setattr(r, "MIN_RESOLVED_TARGETS", 0.5)
    monkeypatch.setattr(r, "MIN_RESOLVED_KEIO", 0.5)
    monkeypatch.setattr(r, "EXPECTED_RECORDS", 1)
    monkeypatch.setattr(r, "EXPECTED_CROSSED_RECORDS", 1)
    with pytest.raises(RuntimeError, match="records .* expected"):
        r.CrispriCrossRachwalski2024Dataset(
            root=str(tmp_path / "store"), ecoli_genome=bw25113
        )


def test_the_build_refuses_a_genome_of_another_strain(
    deposited: tuple[dict[str, Path], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from torchcell.sequence.genome.ecoli.k12 import EcoliK12MG1655Genome

    _, data_root = deposited
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    wrong = object.__new__(EcoliK12MG1655Genome)
    with pytest.raises(TypeError, match="needs the BW25113 genome"):
        r.CrispriCrossRachwalski2024Dataset(
            root=str(tmp_path / "store"), ecoli_genome=wrong
        )


def test_every_built_record_is_a_fitness_experiment_with_its_screen_label(
    built: Any,
) -> None:
    screens = set()
    for index in range(len(built)):
        record = built[index]
        assert record["experiment"]["experiment_type"] == "fitness"
        assert record["reference"]["phenotype_reference"]["fitness"] == 1.0
        screens.add(record["experiment"]["phenotype"]["screen_id"])
    assert screens == {r.SCREEN_S2A, r.SCREEN_S3, r.SCREEN_S4A}


def test_the_build_writes_its_four_ledgers(built: Any) -> None:
    out = Path(built.preprocess_dir)
    dropped = json.loads((out / "dropped_records.json").read_text())
    assert [rule["rule"] for rule in dropped["rules"]] == [
        rule for rule, _ in r.ROW_RULES
    ]
    assert [rule["rule"] for rule in dropped["retired_rules"]] == [
        rule for rule, _ in r.RETIRED_ROW_RULES
    ]
    assert dropped["kept_records"] == len(built)
    assert dropped["empty_genotype_records"] == 18
    # the restored rules say how many records they now write rather than drop
    assert all(rule["records_now_written"] > 0 for rule in dropped["retired_rules"])
    extraction = json.loads((out / "extraction.json").read_text())
    assert [join["table"] for join in extraction["plate_order_joins"]] == [
        r.SCREEN_S2A,
        r.SCREEN_S3,
    ]
    assert [check["rule"] for check in extraction["checks"]] == [
        "table_s2c_fixes_the_block_order",
        "fifty_percent_reduction_count_reproduced",
        "empty_vector_mean_is_one_per_column",
        "empty_vector_mean_is_one_per_column",
        "deposit_reproduces_table_s4a",
        "collection_replicate_is_a_technical_pinning",
    ]
    identifiers = json.loads((out / "identifier_reconciliation.json").read_text())
    assert identifiers["identifier_route"] == "gene_symbol"
    accounting = json.loads((out / "build_accounting.json").read_text())
    assert accounting["unpinned_environment_values"][0]["screen"] == r.SCREEN_S4A
    assert accounting["unpinned_environment_values"][0]["reading"] == "LB"


def test_the_loader_refuses_to_build_without_a_resolvable_query_strain(
    deposited: tuple[dict[str, Path], Path],
    bw25113: EcoliK12BW25113Genome,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Table S3's dlpp block cannot be written if lpp does not resolve."""
    _, data_root = deposited
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _pin_assembly(monkeypatch)
    monkeypatch.setattr(r, "MIN_RESOLVED_TARGETS", 0.1)
    monkeypatch.setattr(r, "MIN_RESOLVED_KEIO", 0.5)
    monkeypatch.setattr(r, "LPP_SYMBOL", "noSuchGene")
    with pytest.raises(RuntimeError, match="lpp does not resolve"):
        r.CrispriCrossRachwalski2024Dataset(
            root=str(tmp_path / "store"), ecoli_genome=bw25113
        )


def test_create_experiment_is_not_the_entry_point(built: Any) -> None:
    with pytest.raises(NotImplementedError):
        built.create_experiment()


# --------------------------------------------------------------------------- #
# The two supplementary verification rows
# --------------------------------------------------------------------------- #
def _record(types: list[str]) -> dict[str, Any]:
    return {
        "experiment": {
            "genotype": {
                "perturbations": [
                    {"systematic_gene_name": "BW25113_0002", "perturbation_type": kind}
                    for kind in types
                ]
            }
        }
    }


def test_the_crossed_row_counts_the_knockdown_by_deletion_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(r, "EXPECTED_CROSSED_RECORDS", 1)
    result = r.crossed_genotypes_carry_both_kinds(
        [
            _record(["crispr_interference"]),
            _record(["deletion_strain"]),
            _record(["crispr_interference", "deletion_strain"]),
        ]
    )
    assert result.passed
    assert result.level is Level.L1
    assert result.details["shapes"]["crispr_interference+deletion_strain"] == 1


def test_an_unexpected_genotype_shape_fails_the_crossed_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(r, "EXPECTED_CROSSED_RECORDS", 0)
    result = r.crossed_genotypes_carry_both_kinds(
        [_record(["deletion_strain", "deletion_strain"])]
    )
    assert not result.passed


def test_the_stored_tag_row_accepts_a_gene_and_a_pseudogene_locus(
    bw25113: EcoliK12BW25113Genome,
) -> None:
    records: list[dict[str, Any]] = [
        {
            "experiment": {
                "genotype": {
                    "perturbations": [
                        {"systematic_gene_name": "BW25113_0002"},
                        {"systematic_gene_name": "BW25113_0004"},
                    ]
                }
            }
        }
    ]
    result = r.stored_tags_are_loci(records, bw25113)
    assert result.passed
    assert result.details["n_pseudogene_loci"] == 1


def test_the_stored_tag_row_fails_on_a_tag_the_assembly_does_not_carry(
    bw25113: EcoliK12BW25113Genome,
) -> None:
    records: list[dict[str, Any]] = [
        {
            "experiment": {
                "genotype": {
                    "perturbations": [{"systematic_gene_name": "BW25113_9999"}]
                }
            }
        }
    ]
    result = r.stored_tags_are_loci(records, bw25113)
    assert not result.passed
    assert result.details["outside"] == ["BW25113_9999"]


# --------------------------------------------------------------------------- #
# The real release (--data)
# --------------------------------------------------------------------------- #
@pytest.mark.data
def test_the_dev_store_holds_the_measured_record_count() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    root = os.path.join(os.environ["DATA_ROOT"], r.DATASET_ROOT_REL)
    if not os.path.exists(os.path.join(root, "processed", "lmdb")):
        pytest.skip(f"dev store at {root} is absent")
    dataset = r.CrispriCrossRachwalski2024Dataset(root=root)
    try:
        assert len(dataset) == r.EXPECTED_RECORDS
        record = dataset[0]
        phenotype = record["experiment"]["phenotype"]
        # the first record is a CRISPRi-collection culture: two technical pinnings
        assert phenotype["n_samples"] == 2
        assert phenotype["sample_unit"] == "technical_replicate"
        assert phenotype["replicate_id"] is not None
    finally:
        # The handle MUST be closed before anything re-reads this store: a held
        # environment fails with "already open in this process", and that only shows
        # in the full-suite run.
        dataset.close_lmdb()


@pytest.mark.data
def test_every_text_sourced_value_audits_against_its_pinned_bytes() -> None:
    """Each quote is still present in the mirrored file whose sha256 it pins."""
    from dotenv import load_dotenv

    from torchcell.verification.sourced import audit_sourced_value

    load_dotenv()
    # ``source_path`` prepends the citation key, so each root is the mirror's parent.
    # The paper and SI quotes live in the literature mirror; the deposit quotes live in
    # the raw mirror beside the files the loader consumes.
    library = r.library_dir().parent
    raw = r.raw_mirror_dir().parent
    if not r.library_dir().exists():
        pytest.skip(f"literature mirror at {r.library_dir()} is absent")
    # a workbook is a zip, so its quote is XML; the next row checks it as a cell
    audits = {
        key: audit_sourced_value(
            value,
            str(raw)
            if str(value.provenance.source_uri).startswith("data/")
            else str(library),
        )
        for key, value in r.SOURCED_VALUES.items()
        if not str(value.provenance.source_uri).endswith(".xlsx")
    }
    assert [key for key, audit in audits.items() if not audit.passed] == []
    assert len(audits) == len(r.SOURCED_VALUES) - 1
    # the three deposit quotes are audited, each against its own pinned member
    deposited_keys = {
        key
        for key, value in r.SOURCED_VALUES.items()
        if str(value.provenance.source_uri).startswith("data/zenodo/")
    }
    assert deposited_keys == {
        "s2_replicate_is_technical",
        "s2_four_pinnings_one_array",
        "s4_replicate_is_biological",
    }


@pytest.mark.data
def test_the_workbook_sourced_value_is_the_cell_it_names() -> None:
    """The carbon source's quote is a workbook cell, so it is read as a cell.

    ``audit_sourced_value`` reads the artifact as text, which a zipped xlsx is not, so
    this row pins the same two facts by hand: the sha256 of the released bytes and the
    verbatim cell the quote came from.
    """
    from dotenv import load_dotenv

    load_dotenv()
    value = r.SOURCED_VALUES["mops_glucose"]
    path = r.raw_mirror_dir() / str(value.provenance.source_uri)
    if not path.exists():
        pytest.skip(f"raw mirror file at {path} is absent")
    assert r._sha256(path) == value.provenance.sha256
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        header = next(book["ST2C"].iter_rows(min_row=2, max_row=2, values_only=True))
    finally:
        book.close()
    assert str(header[2]).strip() == value.quote


@pytest.mark.data
def test_the_deposit_in_the_raw_mirror_is_per_replicate_table_s4a() -> None:
    """The real deposit tables: 4,542 rows each, both replicates, unique positions.

    This is the count test of the conversion's source. It needs no LMDB: it reads the
    pinned mirror bytes and re-measures the three facts the build rests on, so a mirror
    that drifted or a release that moved is caught here as well as at build time.
    """
    from dotenv import load_dotenv

    load_dotenv()
    mirror = r.raw_mirror_dir()
    if not (mirror / "data" / "zenodo" / r.LEGEND_1536_FILE).exists():
        pytest.skip(f"deposit members are absent from {mirror}")
    worst_mean = 0.0
    rows_by_table: dict[tuple[str, int], int] = {}
    for cross in r.KEIO_TABLE_CROSS_NAMES:
        for dose in (0, 50, 500):
            path = mirror / "data" / "zenodo" / r._keio_table_name(cross, dose)
            rows = r.read_keio_replicate_table(path, cross=cross, dose=dose)
            rows_by_table[(cross, dose)] = len(rows)
            for row in rows:
                mean = (row.values[0] + row.values[1]) / 2.0
                worst_mean = max(worst_mean, abs(mean - row.released_mean))
    assert set(rows_by_table.values()) == {4542}
    assert len(rows_by_table) == 12
    assert worst_mean <= r.DEPOSIT_TOLERANCE

    # every deletion label that heads more than one row sits at distinct positions,
    # which is what makes those 1,010 rows storable as replicates
    rows = r.read_keio_replicate_table(
        mirror / "data" / "zenodo" / r._keio_table_name("Vector", 0),
        cross="Vector",
        dose=0,
    )
    counts = Counter(row.label for row in rows)
    repeated = {label for label, count in counts.items() if count > 1}
    assert len(repeated) == 485
    assert sum(counts[label] for label in repeated) == 1010
    positions = [(row.plate, row.well) for row in rows if row.label in repeated]
    assert len(set(positions)) == len(positions)


@pytest.mark.data
def test_the_real_legend_types_the_collection_replicate_as_technical() -> None:
    """The deposit's own legend and tables, re-measured: four pinnings of 384 wells."""
    from dotenv import load_dotenv

    load_dotenv()
    mirror = r.raw_mirror_dir() / "data" / "zenodo"
    if not (mirror / r.LEGEND_1536_FILE).exists():
        pytest.skip(f"deposit members are absent from {mirror}")
    check = r.check_collection_replicate_is_technical(
        mirror / r.LEGEND_1536_FILE,
        {
            (background, medium, replicate): mirror
            / r._wt_lpp_table_name(background, medium, replicate)
            for background in ("WT", "Lpp")
            for medium in ("LB", "MOPS")
            for replicate in (1, 2)
        },
    )
    assert check["legend_wells"] == 1536
    assert check["pinnings_per_strain"] == {"1": 384, "2": 384, "3": 384, "4": 384}
    assert set(check["table_pinning"].values()) == {"3", "4"}
    assert len(check["table_pinning"]) == 8
