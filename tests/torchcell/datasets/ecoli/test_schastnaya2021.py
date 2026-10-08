# tests/torchcell/datasets/ecoli/test_schastnaya2021.py
# [[tests.torchcell.datasets.ecoli.test_schastnaya2021]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_schastnaya2021.py
"""The Schastnaya 2021 deletion metabolome loader
(``torchcell.datasets.ecoli.schastnaya2021``).

Synthetic tests (run everywhere) write both workbooks into ``tmp_path`` and build over
the real ``EcoliK12MG1655Genome`` class on the synthetic MG1655 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` (thrA on b0002, proB on
b0005), served through a stubbed ``resolve`` with the network refused, and replace the
loader's ``verify_raw_files`` with a presence check (the synthetic files cannot carry
the real pins; the pins are asserted by the refusal test and by the data-gated tests).
The synthetic Supplementary Data 3 is four columns over five ions:

    column  strain            carbon source   kept?
    5       thrA knockout     GLUCOSE         yes, 4 of 5 ions detected
    6       proB knockout     ACETATE         yes, 4 of 5 ions detected
    7       ThrL-S2A          GLUCOSE         no, the phosphosite rule
    8       ThrL-S2E          GLUCOSE         no, the phosphosite rule

Data-gated tests (``@pytest.mark.data``) read the real raw mirror and the built
dev-tree LMDB under ``$DATA_ROOT`` (they never build it): the manifest pins, the
provenance audit of every sourced value, the record count and retention arithmetic, one
hand-checked record read off the workbook, and the L0-L4 verifier.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import openpyxl
import pytest

import torchcell.datasets.ecoli.schastnaya2021 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError, RawSha256MismatchError
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialStrainBackground,
    ConcentrationUnit,
    MediaComponentRole,
)
from torchcell.datasets.bacteria_common import LocusTagResolutionError
from torchcell.literature.manifest import RetrievalMethod
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.sourced import audit_sourced_value

ND = m.NOT_DETECTED
#: Five synthetic ions: three single-identity, one two-candidate, one 18-candidate.
IONS: list[tuple[str, str, str, float]] = [
    ("C00031", "D-Glucose", "C6H12O6", 179.0561),
    ("C00022; C00222", "Pyruvate; Malonate semialdehyde", "C3H4O3", 87.0088),
    ("C00042", "Succinate", "C4H6O4", 117.0193),
    ("C00074", "Phosphoenolpyruvate", "C3H5O6P", 166.9751),
    ("C00037", "Glycine", "C2H5NO2", 74.0248),
]
#: ``(strain, carbon source, fold changes, adjusted p-values)`` of each column.
COLUMNS: list[tuple[str, str, list[Any], list[Any]]] = [
    (
        "thrA knockout",
        "GLUCOSE",
        [0.5, -1.25, ND, 0.125, 2.0],
        [0.01, 0.02, ND, 0.03, 0.04],
    ),
    (
        "proB knockout",
        "ACETATE",
        [-0.5, 1.25, ND, -0.125, -2.0],
        [0.05, 0.06, ND, 0.07, 0.08],
    ),
    ("ThrL-S2A", "GLUCOSE", [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]),
    ("ThrL-S2E", "GLUCOSE", [0.6, 0.7, 0.8, 0.9, 1.0], [0.6, 0.7, 0.8, 0.9, 1.0]),
]
SD1_ROWS: list[tuple[str, str, str]] = [
    ("ThrL-S2A", "phospho-abolishing", "b0001"),
    ("ThrL-S2E", "phospho-mimicking", "b0001"),
]
KEPT_KEYS = ["C6H12O6", "C3H4O3", "C3H5O6P", "C2H5NO2"]
REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain=m.WILD_TYPE_BACKGROUND_NAME,
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
    background=m.wild_type_background(),
)


def _write_sd3(
    path: Path, columns: Sequence[tuple[str, str, list[Any], list[Any]]]
) -> None:
    """A synthetic Supplementary Data 3 with the pinned geometry."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.SD3_SHEET
    sheet.append([m.SD3_TITLE])
    n = len(columns)
    sheet.append(
        [*m.SD3_LABEL_COLUMNS, m.SD3_FOLD_CHANGE_HEADER, *[None] * (n - 1)]
        + [m.SD3_P_VALUE_HEADER, *[None] * (n - 1)]
    )
    sheet.append([None] * 4 + [c[1] for c in columns] * 2)
    sheet.append([None] * 4 + [c[0] for c in columns] * 2)
    for row, (kegg, annotation, formula, mz) in enumerate(IONS):
        sheet.append(
            [kegg, annotation, formula, mz]
            + [c[2][row] for c in columns]
            + [c[3][row] for c in columns]
        )
    workbook.save(path)


def _write_sd1(
    path: Path, rows: Sequence[tuple[str, str, str]] = tuple(SD1_ROWS)
) -> None:
    """A synthetic Supplementary Data 1 with the pinned geometry."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.SD1_SHEET
    sheet.append([m.SD1_TITLE])
    sheet.append(
        [
            "Phosphomutant",
            "Mutant state",
            "Previous evidence",
            "Occupancy",
            "Systematic protein name",
        ]
    )
    for strain, state, locus in rows:
        sheet.append([strain, state, "none", None, locus])
    workbook.save(path)


def _write_raw(raw: Path) -> None:
    """Both synthetic workbooks under a dataset root's ``raw/``."""
    raw.mkdir(parents=True, exist_ok=True)
    _write_sd3(raw / m.SD3_FILE, COLUMNS)
    _write_sd1(raw / m.SD1_FILE)


@pytest.fixture
def mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The real MG1655 class over the synthetic assembly; network refused."""
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "mg1655"
    root.mkdir()
    return EcoliK12MG1655Genome(genome_root=str(root), overwrite=False)


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
def synthetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mg1655: EcoliK12MG1655Genome,
    presence_only_pins: list[Mapping[str, str]],
) -> Path:
    """A dataset root whose raw/ holds both synthetic workbooks."""
    root = tmp_path / "metabolome_schastnaya2021"
    _write_raw(root / "raw")
    monkeypatch.setattr(m, "assembly_reference", lambda strain, **kwargs: REFERENCE)
    return root


# --------------------------------------------------------------------------- #
# Pure parsing
# --------------------------------------------------------------------------- #
def test_read_sd3_parses_both_blocks_and_marks_nd(tmp_path: Path) -> None:
    path = tmp_path / m.SD3_FILE
    _write_sd3(path, COLUMNS)
    table = m.read_sd3(path)
    assert [ion.formula for ion in table.ions] == [ion[2] for ion in IONS]
    assert [ion.n_candidates for ion in table.ions] == [1, 2, 1, 1, 1]
    assert table.ions[0].target_metabolite_id == "C00031"
    assert table.ions[1].target_metabolite_id is None
    first = table.columns[0]
    assert (first.column, first.strain, first.carbon_source) == (
        5,
        "thrA knockout",
        "GLUCOSE",
    )
    assert first.fold_change == (0.5, -1.25, None, 0.125, 2.0)
    assert first.adjusted_p_value == (0.01, 0.02, None, 0.03, 0.04)
    assert first.detected_rows == (0, 1, 3, 4)
    assert first.n_detected == 4
    assert first.is_knockout and first.gene_symbol == "thrA"
    assert first.extraction == "cold"
    assert table.columns[2].is_knockout is False


def test_strain_column_gene_symbol_refuses_a_phosphomutant(tmp_path: Path) -> None:
    path = tmp_path / m.SD3_FILE
    _write_sd3(path, COLUMNS)
    phosphomutant = m.read_sd3(path).columns[2]
    with pytest.raises(ValueError, match="is not a knockout column"):
        _ = phosphomutant.gene_symbol


def test_strain_column_extraction_reads_the_papers_hot_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "HOT_EXTRACTION_ENZYMES", ("ThrA",))
    path = tmp_path / m.SD3_FILE
    _write_sd3(path, COLUMNS)
    columns = m.read_sd3(path).columns
    assert columns[0].extraction == "hot"
    assert columns[1].extraction == "cold"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda rows: rows.__setitem__(0, ["not the title"]), "title"),
        (
            lambda rows: rows.__setitem__(1, ["Kegg ID", "x", "Formula", "m/z"]),
            "label columns",
        ),
    ],
)
def test_read_sd3_refuses_a_wrong_header(
    tmp_path: Path, mutate: Any, match: str
) -> None:
    path = tmp_path / m.SD3_FILE
    _write_sd3(path, COLUMNS)
    workbook = openpyxl.load_workbook(path)
    sheet = workbook[m.SD3_SHEET]
    rows = [list(r) for r in sheet.iter_rows(values_only=True)]
    mutate(rows)
    fresh = openpyxl.Workbook()
    new_sheet = fresh.active
    assert new_sheet is not None
    new_sheet.title = m.SD3_SHEET
    for row in rows:
        new_sheet.append(row)
    fresh.save(path)
    with pytest.raises(m.TableFormatError, match=match):
        m.read_sd3(path)


def test_read_sd3_refuses_an_unknown_carbon_source(tmp_path: Path) -> None:
    path = tmp_path / m.SD3_FILE
    columns = [("thrA knockout", "XYLOSE", [1.0] * 5, [0.1] * 5)]
    _write_sd3(path, columns)
    with pytest.raises(m.TableFormatError, match="are not the five"):
        m.read_sd3(path)


def test_read_sd3_refuses_a_value_without_its_p_value(tmp_path: Path) -> None:
    path = tmp_path / m.SD3_FILE
    columns = [("thrA knockout", "GLUCOSE", [1.0] * 5, [0.1, 0.1, ND, 0.1, 0.1])]
    _write_sd3(path, columns)
    with pytest.raises(m.TableFormatError, match="without its p-value"):
        m.read_sd3(path)


def test_read_sd3_refuses_a_repeated_formula(tmp_path: Path) -> None:
    path = tmp_path / m.SD3_FILE
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.SD3_SHEET
    sheet.append([m.SD3_TITLE])
    sheet.append([*m.SD3_LABEL_COLUMNS, m.SD3_FOLD_CHANGE_HEADER, m.SD3_P_VALUE_HEADER])
    sheet.append([None] * 4 + ["GLUCOSE", "GLUCOSE"])
    sheet.append([None] * 4 + ["thrA knockout", "thrA knockout"])
    sheet.append(["C00031", "D-Glucose", "C6H12O6", 179.0, 1.0, 0.1])
    sheet.append(["C00267", "a-D-Glucose", "C6H12O6", 179.0, 1.0, 0.1])
    workbook.save(path)
    with pytest.raises(m.TableFormatError, match="Formula column is not unique"):
        m.read_sd3(path)


def test_cell_float_accepts_nd_and_refuses_anything_else() -> None:
    assert m._cell_float(1.5, "where") == 1.5
    assert m._cell_float(ND, "where") is None
    with pytest.raises(m.TableFormatError, match="neither a number nor"):
        m._cell_float("0.5", "where")
    with pytest.raises(m.TableFormatError, match="is not a number"):
        m._cell_float(None, "where")


def test_read_sd1_keys_by_strain(tmp_path: Path) -> None:
    path = tmp_path / m.SD1_FILE
    _write_sd1(path)
    entries = m.read_sd1(path)
    assert sorted(entries) == ["ThrL-S2A", "ThrL-S2E"]
    assert entries["ThrL-S2A"].mutant_state == "phospho-abolishing"
    assert entries["ThrL-S2A"].enzyme_locus == "b0001"


def test_read_sd1_refuses_a_repeated_strain(tmp_path: Path) -> None:
    path = tmp_path / m.SD1_FILE
    _write_sd1(path, [*SD1_ROWS, SD1_ROWS[0]])
    with pytest.raises(m.TableFormatError, match="twice"):
        m.read_sd1(path)


def test_read_sd1_refuses_a_wrong_title(tmp_path: Path) -> None:
    path = tmp_path / m.SD1_FILE
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.SD1_SHEET
    sheet.append(["something else"])
    workbook.save(path)
    with pytest.raises(m.TableFormatError, match="title"):
        m.read_sd1(path)


# --------------------------------------------------------------------------- #
# Retention, identity and the detected-ion ledger
# --------------------------------------------------------------------------- #
def test_phosphomutant_drop_rule_names_each_strain_with_its_enzyme(
    tmp_path: Path,
) -> None:
    _write_sd3(tmp_path / m.SD3_FILE, COLUMNS)
    _write_sd1(tmp_path / m.SD1_FILE)
    table = m.read_sd3(tmp_path / m.SD3_FILE)
    entries = m.read_sd1(tmp_path / m.SD1_FILE)
    dropped = [c for c in table.columns if not c.is_knockout]
    rule = m.phosphomutant_drop_rule(dropped, entries)
    assert rule.rule == m.PHOSPHOMUTANT_DROP_RULE
    assert rule.n_records == 2
    assert rule.items == [
        "ThrL-S2A (b0001, phospho-abolishing): GLUCOSE",
        "ThrL-S2E (b0001, phospho-mimicking): GLUCOSE",
    ]


def test_phosphomutant_drop_rule_refuses_a_strain_absent_from_sd1(
    tmp_path: Path,
) -> None:
    _write_sd3(tmp_path / m.SD3_FILE, COLUMNS)
    _write_sd1(tmp_path / m.SD1_FILE, SD1_ROWS[:1])
    table = m.read_sd3(tmp_path / m.SD3_FILE)
    entries = m.read_sd1(tmp_path / m.SD1_FILE)
    dropped = [c for c in table.columns if not c.is_knockout]
    with pytest.raises(m.TableFormatError, match="is in no Supplementary Data 1 row"):
        m.phosphomutant_drop_rule(dropped, entries)


def test_detected_ion_ledger_groups_by_extraction_and_carbon_source(
    tmp_path: Path,
) -> None:
    _write_sd3(tmp_path / m.SD3_FILE, COLUMNS)
    kept = [c for c in m.read_sd3(tmp_path / m.SD3_FILE).columns if c.is_knockout]
    ledger = m.detected_ion_ledger(kept)
    assert ledger.n_ions == 5
    assert ledger.groups == {"cold|ACETATE": 4, "cold|GLUCOSE": 4}
    assert ledger.n_strains == {"cold|ACETATE": 1, "cold|GLUCOSE": 1}
    assert ledger.cold_set_shared_across_carbon_sources is True


def test_detected_ion_ledger_refuses_a_strain_dependent_set(tmp_path: Path) -> None:
    columns: list[tuple[str, str, list[Any], list[Any]]] = [
        (
            "thrA knockout",
            "GLUCOSE",
            [1.0, 1.0, ND, 1.0, 1.0],
            [0.1, 0.1, ND, 0.1, 0.1],
        ),
        (
            "proB knockout",
            "GLUCOSE",
            [1.0, ND, 1.0, 1.0, 1.0],
            [0.1, ND, 0.1, 0.1, 0.1],
        ),
    ]
    _write_sd3(tmp_path / m.SD3_FILE, columns)
    kept = [c for c in m.read_sd3(tmp_path / m.SD3_FILE).columns if c.is_knockout]
    with pytest.raises(m.TableFormatError, match="differs between strains"):
        m.detected_ion_ledger(kept)


def test_identity_ledger_leaves_a_merged_isobaric_set_out_of_the_map(
    tmp_path: Path,
) -> None:
    _write_sd3(tmp_path / m.SD3_FILE, COLUMNS)
    ledger = m.identity_ledger(m.read_sd3(tmp_path / m.SD3_FILE).ions)
    assert (ledger.n_ions, ledger.n_single_identity, ledger.n_merged_isobaric) == (
        5,
        4,
        1,
    )
    assert ledger.candidate_size_histogram == {1: 4, 2: 1}
    assert ledger.merged_candidates == {"C3H4O3": ("C00022", "C00222")}
    assert ledger.target_metabolite_ids_covered == 0.8


# --------------------------------------------------------------------------- #
# Media, environment and record construction
# --------------------------------------------------------------------------- #
def test_every_released_carbon_source_has_one_sourced_component() -> None:
    assert sorted(m.MEDIA) == ["ACETATE", "FRUCTOSE", "GLUCOSE", "GLYCEROL", "PYRUVATE"]
    for token, media in m.MEDIA.items():
        (component,) = media.components
        assert media.base_medium == "M9"
        assert media.is_synthetic and media.state == "liquid"
        assert component.role is MediaComponentRole.carbon_source
        assert component.concentration is not None
        assert component.concentration.unit is ConcentrationUnit.g_per_l
        assert component.concentration.value == m.CARBON_SOURCES[token][1]
        assert component.compound.name == m.CARBON_SOURCES[token][0]


def test_environment_is_the_medium_at_37c_aerobic() -> None:
    env = m.environment("ACETATE")
    assert env.media is m.MEDIA["ACETATE"]
    assert env.temperature is not None and env.temperature.value == 37.0
    assert env.aerobicity == "aerobic"
    assert env.perturbations == []
    assert env.duration_hours is None


def test_wild_type_background_is_the_mg1655_dmuts_strain() -> None:
    background = m.wild_type_background()
    assert isinstance(background, BacterialStrainBackground)
    assert background.name == m.WILD_TYPE_BACKGROUND_NAME
    assert background.reference_strain == "MG1655"
    assert background.parents == ["MG1655"]
    assert background.alleles == []
    assert background.provenance_gaps == []
    assert background.genotype_statement is not None
    assert "mutS" in background.genotype_statement


def test_metabolite_phenotype_drops_nd_and_maps_only_single_identities(
    tmp_path: Path,
) -> None:
    _write_sd3(tmp_path / m.SD3_FILE, COLUMNS)
    table = m.read_sd3(tmp_path / m.SD3_FILE)
    phenotype = m.metabolite_phenotype(table.ions, table.columns[0].fold_change)
    assert list(phenotype.metabolite_level) == KEPT_KEYS
    assert phenotype.metabolite_level["C6H12O6"] == 0.5
    assert "C4H6O4" not in phenotype.metabolite_level
    assert phenotype.target_metabolite_ids == {
        "C6H12O6": "C00031",
        "C3H5O6P": "C00074",
        "C2H5NO2": "C00037",
    }
    assert phenotype.metabolite_level_se is None
    assert set(phenotype.n_replicates.values()) == {3}
    assert phenotype.measurement_type == m.MEASUREMENT_TYPE
    assert [gap.field for gap in phenotype.provenance_gaps] == ["metabolite_level_se"]


def test_metabolite_phenotype_refuses_a_length_mismatch_and_an_empty_column(
    tmp_path: Path,
) -> None:
    _write_sd3(tmp_path / m.SD3_FILE, COLUMNS)
    ions = m.read_sd3(tmp_path / m.SD3_FILE).ions
    with pytest.raises(ValueError, match="ions for"):
        m.metabolite_phenotype(ions, [1.0])
    with pytest.raises(ValueError, match="at least one ion"):
        m.metabolite_phenotype(ions, [None] * len(ions))


def test_wild_type_phenotype_is_zero_on_the_same_keys(tmp_path: Path) -> None:
    _write_sd3(tmp_path / m.SD3_FILE, COLUMNS)
    table = m.read_sd3(tmp_path / m.SD3_FILE)
    phenotype = m.metabolite_phenotype(table.ions, table.columns[0].fold_change)
    reference = m.wild_type_phenotype(phenotype)
    assert set(reference.metabolite_level) == set(phenotype.metabolite_level)
    assert set(reference.metabolite_level.values()) == {0.0}
    assert reference.target_metabolite_ids == phenotype.target_metabolite_ids
    assert reference.measurement_type == m.MEASUREMENT_TYPE


def test_deletion_genotype_is_one_keio_deletion_on_the_mg1655_namespace(
    tmp_path: Path,
) -> None:
    _write_sd3(tmp_path / m.SD3_FILE, COLUMNS)
    column = m.read_sd3(tmp_path / m.SD3_FILE).columns[0]
    genotype = m.deletion_genotype(m.ResolvedDeletion(column=column, locus_tag="b0002"))
    (perturbation,) = genotype.perturbations
    dumped = perturbation.model_dump()
    assert dumped["systematic_gene_name"] == "b0002"
    assert dumped["perturbed_gene_name"] == "thrA"
    assert dumped["perturbation_type"] == "bacterial_deletion"
    assert dumped["gene_namespace"] == "ecoli_k12_mg1655_bnumber"
    assert dumped["collection"] == "KEIO collection"
    assert dumped["cassette"] == m.CASSETTE
    assert dumped["state"] == "absent"


def test_build_reference_keeps_the_assembly_pin_through_a_dump(tmp_path: Path) -> None:
    _write_sd3(tmp_path / m.SD3_FILE, COLUMNS)
    table = m.read_sd3(tmp_path / m.SD3_FILE)
    resolved = m.ResolvedDeletion(column=table.columns[0], locus_tag="b0002")
    experiment = m.build_experiment("ds", resolved, table.ions)
    reference = m.build_reference("ds", REFERENCE, experiment)
    dumped = reference.model_dump()
    assert dumped["genome_reference"]["assembly_accession"] == "GCA_000005845.2"
    assert dumped["genome_reference"]["strain"] == m.WILD_TYPE_BACKGROUND_NAME
    assert dumped["environment_reference"] == experiment.environment.model_dump()
    assert set(dumped["phenotype_reference"]["metabolite_level"].values()) == {0.0}


# --------------------------------------------------------------------------- #
# The hermetic build
# --------------------------------------------------------------------------- #
def test_build_keeps_the_two_deletions_and_ledgers_the_phosphomutants(
    synthetic: Path,
    mg1655: EcoliK12MG1655Genome,
    presence_only_pins: list[Mapping[str, str]],
) -> None:
    dataset = m.MetabolomeSchastnaya2021Dataset(
        root=str(synthetic), ecoli_genome=mg1655
    )
    assert presence_only_pins == [m.DATA_SHA256]
    assert len(dataset) == 2

    first = dataset[0]["experiment"]
    (perturbation,) = first["genotype"]["perturbations"]
    assert perturbation["systematic_gene_name"] == "b0002"
    assert perturbation["perturbed_gene_name"] == "thrA"
    assert first["phenotype"]["metabolite_level"] == {
        "C6H12O6": 0.5,
        "C3H4O3": -1.25,
        "C3H5O6P": 0.125,
        "C2H5NO2": 2.0,
    }
    assert "C3H4O3" not in (first["phenotype"]["target_metabolite_ids"] or {})
    assert first["environment"]["media"]["name"] == m.MEDIA["GLUCOSE"].name

    second = dataset[1]["experiment"]
    assert second["genotype"]["perturbations"][0]["systematic_gene_name"] == "b0005"
    assert second["environment"]["media"]["name"] == m.MEDIA["ACETATE"].name

    reference = dataset[0]["reference"]
    assert reference["genome_reference"] == REFERENCE.model_dump()
    assert set(reference["phenotype_reference"]["metabolite_level"].values()) == {0.0}
    assert dataset[0]["publication"] == m.PUBLICATION.model_dump()

    preprocess = synthetic / "preprocess"
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert (
        drops["released_columns"],
        drops["kept_records"],
        drops["dropped_records"],
    ) == (4, 2, 2)
    (rule,) = drops["rules"]
    assert rule["rule"] == m.PHOSPHOMUTANT_DROP_RULE
    assert rule["n_records"] == 2
    identity = json.loads((preprocess / "metabolite_identity.json").read_text())
    assert identity["merged_candidates"] == {"C3H4O3": ["C00022", "C00222"]}
    assert identity["target_metabolite_ids_covered"] == 0.8
    detected = json.loads((preprocess / "detected_ion_counts.json").read_text())
    assert detected["groups"] == {"cold|ACETATE": 4, "cold|GLUCOSE": 4}
    ions = (preprocess / "ions.csv").read_text().splitlines()
    assert ions[0] == (
        "row,formula,mz,n_candidates,kegg_ids,target_metabolite_id,annotation"
    )
    strains = (preprocess / "strains.csv").read_text().splitlines()
    assert strains[1].startswith("0,5,thrA knockout,thrA,b0002,GLUCOSE,cold,4,BW25113,")
    p_values = (preprocess / "adjusted_p_values.csv").read_text().splitlines()
    assert p_values[0] == "formula,thrA knockout|GLUCOSE,proB knockout|ACETATE"
    assert p_values[1] == "C6H12O6,0.01,0.05"
    assert p_values[3] == "C4H6O4,,"


def test_build_verifies_l0_to_l4_on_the_synthetic_tree(
    synthetic: Path, mg1655: EcoliK12MG1655Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    m.MetabolomeSchastnaya2021Dataset(root=str(synthetic), ecoli_genome=mg1655)
    monkeypatch.setattr(
        "torchcell.verification.runners._gene_set_for_reference",
        lambda reference, data_root: {"b0002", "b0005"},
    )
    monkeypatch.setattr(m, "SOURCED_VALUES", {})
    report = verify_without_audits(synthetic)
    assert report.passed, report.summary()
    names = [result.name for result in report.results]
    assert "genotype_environment_uniqueness" in names
    assert "gene_containment_mg1655_b_numbers" in names


def verify_without_audits(root: Path) -> Any:
    """``verify_build`` over the synthetic tree (``SOURCED_VALUES`` monkeypatched out)."""
    return m.verify_build(str(root), data_root=str(root))


def test_build_refuses_a_genome_of_another_strain(
    synthetic: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.torchcell.sequence.genome._bacterial_fixtures import BW25113_LOCI
    from torchcell.sequence.genome.ecoli.k12 import (
        BW25113_ASSEMBLY,
        EcoliK12BW25113Genome,
    )

    files = write_assembly(tmp_path / "tier2", BW25113_ASSEMBLY, BW25113_LOCI)
    serve_tier(monkeypatch, files)
    root = tmp_path / "bw25113"
    root.mkdir()
    other = EcoliK12BW25113Genome(genome_root=str(root), overwrite=False)
    with pytest.raises(ValueError, match="needs the ecoli_K12_MG1655_ASM584v2 genome"):
        m.MetabolomeSchastnaya2021Dataset(root=str(synthetic), ecoli_genome=other)


def test_resolve_deletions_stops_below_the_resolution_threshold(
    tmp_path: Path, mg1655: EcoliK12MG1655Genome
) -> None:
    columns = [("notAGene knockout", "GLUCOSE", [1.0] * 5, [0.1] * 5)]
    _write_sd3(tmp_path / m.SD3_FILE, columns)
    kept = list(m.read_sd3(tmp_path / m.SD3_FILE).columns)
    with pytest.raises(LocusTagResolutionError, match=r"0 of 1 names \(0\.000\)"):
        m.resolve_deletions(mg1655, kept, label="test")


def test_resolve_deletions_refuses_a_symbol_off_the_namespace(
    tmp_path: Path, mg1655: EcoliK12MG1655Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Above the threshold an unplaced symbol is still a build error, not a drop."""
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.5)
    columns = [
        ("thrA knockout", "GLUCOSE", [1.0] * 5, [0.1] * 5),
        ("notAGene knockout", "GLUCOSE", [1.0] * 5, [0.1] * 5),
    ]
    _write_sd3(tmp_path / m.SD3_FILE, columns)
    kept = list(m.read_sd3(tmp_path / m.SD3_FILE).columns)
    with pytest.raises(m.TableFormatError, match="not on an MG1655 locus"):
        m.resolve_deletions(mg1655, kept, label="test")


def test_process_verifies_the_real_pins(
    tmp_path: Path, mg1655: EcoliK12MG1655Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the presence-only stub the synthetic bytes fail the pinned sha256."""
    root = tmp_path / "metabolome_schastnaya2021"
    _write_raw(root / "raw")
    monkeypatch.setattr(m, "assembly_reference", lambda strain, **kwargs: REFERENCE)
    with pytest.raises(RawSha256MismatchError):
        m.MetabolomeSchastnaya2021Dataset(root=str(root), ecoli_genome=mg1655)


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
def _synthetic_raw_files(tmp_path: Path) -> dict[str, str | Path]:
    """Two files whose bytes hash to the pins, by monkeypatching nothing: unusable."""
    return {name: tmp_path / name for name in m.DATA_SHA256}


def test_deposit_refuses_a_file_whose_bytes_are_not_the_pin(tmp_path: Path) -> None:
    for name in m.DATA_SHA256:
        (tmp_path / name).write_bytes(b"not the released workbook")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        m.deposit_raw_mirror(
            sources=_synthetic_raw_files(tmp_path), data_root=str(tmp_path / "root")
        )


def test_deposit_refuses_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="no source given"):
        m.deposit_raw_mirror(sources={}, data_root=str(tmp_path))


def test_download_checks_the_manifest_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest whose recorded hash is not the module's pin stops the download."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    mirror = m.raw_mirror_dir(str(tmp_path))
    (mirror / "data").mkdir(parents=True)
    records = [
        {
            "path": raw.mirror_relpath,
            "role": "raw_data",
            "bytes": raw.bytes,
            "sha256": "0" * 64,
            "source": raw.retrieval.source_url,
        }
        for raw in m.RAW_FILES
    ]
    (mirror / "manifest.json").write_text(
        json.dumps(
            {
                "citation_key": m.CITATION_KEY,
                "doi": m.PAPER_DOI,
                "title": "t",
                "files": records,
                "provenance_complete": True,
            }
        )
    )
    dataset = m.MetabolomeSchastnaya2021Dataset.__new__(
        m.MetabolomeSchastnaya2021Dataset
    )
    object.__setattr__(dataset, "root", str(tmp_path / "ds"))
    with pytest.raises(ManifestPinMismatchError):
        m.MetabolomeSchastnaya2021Dataset.download(dataset)


def _one_pinned_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes = b"a workbook"
) -> Path:
    """Point ``RAW_FILES`` at one tmp file, pinned to the hash it really has."""
    source = tmp_path / "source" / m.SD3_FILE
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    pinned = m.RAW_FILES[1].model_copy(
        update={"sha256": m._sha256(source), "bytes": source.stat().st_size}
    )
    pinned = pinned.model_copy(
        update={
            "retrieval": pinned.retrieval.model_copy(update={"sha256": pinned.sha256})
        }
    )
    monkeypatch.setattr(m, "RAW_FILES", (pinned,))
    monkeypatch.setattr(m, "DATA_SHA256", {pinned.name: pinned.sha256})
    monkeypatch.setattr(m, "RAW_FILES_BY_NAME", {pinned.name: pinned})
    return source


def test_deposit_writes_the_mirror_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _one_pinned_file(tmp_path, monkeypatch)
    root = m.deposit_raw_mirror(
        sources={m.SD3_FILE: source}, data_root=str(tmp_path / "root")
    )
    assert (root / "data" / m.SD3_FILE).read_bytes() == source.read_bytes()
    manifest = m.load_manifest(str(tmp_path / "root"))
    assert manifest.citation_key == m.CITATION_KEY
    assert m.manifest_sha256(manifest, f"data/{m.SD3_FILE}") == m.RAW_FILES[0].sha256
    with pytest.raises(KeyError, match="is not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "data/nope.xlsx")
    # a second deposit of the same bytes leaves the mirror alone
    assert (
        m.deposit_raw_mirror(
            sources={m.SD3_FILE: source}, data_root=str(tmp_path / "root")
        )
        == root
    )


def test_deposit_refuses_to_overwrite_a_differing_mirror_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _one_pinned_file(tmp_path, monkeypatch)
    root = m.deposit_raw_mirror(
        sources={m.SD3_FILE: source}, data_root=str(tmp_path / "root")
    )
    (root / "data" / m.SD3_FILE).write_bytes(b"something else")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        m.deposit_raw_mirror(
            sources={m.SD3_FILE: source}, data_root=str(tmp_path / "root")
        )


def test_retrieve_runs_the_recorded_retriever_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _one_pinned_file(tmp_path, monkeypatch)
    calls: list[str] = []

    def fake_retriever(record: Any) -> bytes:
        calls.append(record.retriever)
        return source.read_bytes()

    monkeypatch.setattr(m, "run_retriever", fake_retriever)
    out = m.retrieve_raw_files(tmp_path / "download")
    assert calls == ["torchcell.literature.retrieve.pmc_cloud_object"]
    assert out[m.SD3_FILE].read_bytes() == source.read_bytes()
    assert m.retrieve_raw_files(tmp_path / "download2", names=[]) == {}


def test_main_dispatches_deposit_build_and_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        m, "retrieve_raw_files", lambda dest: seen.setdefault("retrieved", dest)
    )
    monkeypatch.setattr(
        m,
        "deposit_raw_mirror",
        lambda *, sources, data_root: seen.setdefault("sources", sorted(sources)),
    )
    assert m.main(["deposit", "--retrieve"]) == 0
    assert seen["retrieved"] == m.library_dir(str(tmp_path)) / "si"
    assert seen["sources"] == sorted(m.DATA_SHA256)

    monkeypatch.setattr(
        m, "MetabolomeSchastnaya2021Dataset", lambda root: seen.setdefault("root", [])
    )
    monkeypatch.setattr(m, "verify_build", lambda root, data_root: _FakeReport(True))
    assert m.main(["build"]) == 0
    assert seen["root"] == []
    assert m.main(["verify"]) == 0
    monkeypatch.setattr(m, "verify_build", lambda root, data_root: _FakeReport(False))
    assert m.main(["verify"]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out == ["['si4.xlsx', 'si6.xlsx']", "len = 0", "fake", "fake"]


class _FakeReport:
    """The two fields ``main`` reads off a verification report."""

    def __init__(self, passed: bool) -> None:
        self.passed = passed

    def summary(self) -> str:
        return "fake"


def test_every_consumed_file_is_pinned_once() -> None:
    assert [raw.name for raw in m.RAW_FILES] == [m.SD1_FILE, m.SD3_FILE]
    assert all(
        len(raw.sha256) == 64 and raw.sha256 == raw.retrieval.sha256
        for raw in m.RAW_FILES
    )
    assert {raw.retrieval.method for raw in m.RAW_FILES} == {RetrievalMethod.pmc_cloud}
    assert [raw.mirror_relpath for raw in m.RAW_FILES] == [
        f"data/{m.SD1_FILE}",
        f"data/{m.SD3_FILE}",
    ]


def test_sourced_value_root_is_the_literature_mirror(tmp_path: Path) -> None:
    value = m.SOURCED_VALUES["temperature_c"]
    assert m.sourced_value_root(value, str(tmp_path)) == tmp_path / "torchcell-library"


def test_the_two_deferred_values_cite_baba_not_schastnaya() -> None:
    for key in ("deletion_background", "cassette"):
        assert m.SOURCED_VALUES[key].provenance.citation_key == m.BABA_KEY
    assert m.SOURCED_VALUES["released_table"].provenance.source_uri == m.SI3_MD


# --------------------------------------------------------------------------- #
# Real data (``--data``): the mirror and the built dev-tree LMDB, never rebuilt
# --------------------------------------------------------------------------- #
def _real_data_root() -> str:
    return os.environ["DATA_ROOT"]


def _built_root() -> str:
    return osp.join(_real_data_root(), "data/torchcell/metabolome_schastnaya2021")


@pytest.mark.data
def test_real_mirror_matches_the_module_pins() -> None:
    from torchcell.data.experiment_dataset import file_sha256

    manifest = m.load_manifest(_real_data_root())
    for raw in m.RAW_FILES:
        assert m.manifest_sha256(manifest, raw.mirror_relpath) == raw.sha256
        path = m.raw_mirror_dir(_real_data_root()) / raw.mirror_relpath
        assert path.stat().st_size == raw.bytes
        assert file_sha256(path) == raw.sha256


@pytest.mark.data
@pytest.mark.parametrize("key", sorted(m.SOURCED_VALUES))
def test_real_sourced_values_are_verbatim(key: str) -> None:
    value = m.SOURCED_VALUES[key]
    result = audit_sourced_value(value, m.sourced_value_root(value, _real_data_root()))
    assert result.passed, result.message


@pytest.mark.data
def test_real_build_counts_and_the_gnd_record() -> None:
    """29 records (200 released columns - 171 phosphomutant); gnd on fructose is
    b2029 with values read off Supplementary Data 3 by hand.
    """
    from torchcell.verification.runners import load_records

    drops = json.loads(
        Path(_built_root(), "preprocess", "dropped_records.json").read_text()
    )
    assert (drops["released_columns"], drops["kept_records"]) == (200, 29)
    assert [rule["n_records"] for rule in drops["rules"]] == [171]
    records = load_records(_built_root())
    assert len(records) == 29
    (gnd,) = [
        record
        for record in records
        if record["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        == "b2029"
        and record["experiment"]["environment"]["media"]["name"]
        == m.MEDIA["FRUCTOSE"].name
    ]
    levels = gnd["experiment"]["phenotype"]["metabolite_level"]
    assert len(levels) == 284
    assert {k: levels[k] for k in HAND_CHECKED_GND} == HAND_CHECKED_GND
    assert set(gnd["experiment"]["phenotype"]["n_replicates"].values()) == {3}
    assert set(
        gnd["reference"]["phenotype_reference"]["metabolite_level"].values()
    ) == {0.0}
    deleted = {
        record["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for record in records
    }
    assert len(deleted) == 16


@pytest.mark.data
def test_real_build_passes_l0_to_l4() -> None:
    report = m.verify_build(_built_root(), data_root=_real_data_root())
    assert report.passed, report.summary()


#: Read off Supplementary Data 3 by hand: the 'gnd knockout' FRUCTOSE column is column
#: 65 of sheet SD3 (the LOG2(FC) block), and these are its data rows 5, 6, 7 and 288.
HAND_CHECKED_GND: dict[str, float] = {
    "C2H5NO": -0.1908,
    "C2H4O2": 0.0698,
    "C3H4O2": 0.1039,
    "C28H40N7O17P3S": -0.013,
}
