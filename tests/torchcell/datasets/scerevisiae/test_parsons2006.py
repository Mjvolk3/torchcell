# tests/torchcell/datasets/scerevisiae/test_parsons2006.py
# [[tests.torchcell.datasets.scerevisiae.test_parsons2006]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_parsons2006.py
"""Parsons 2006 loader: the release parsers, the doses, the compounds and one build.

``process()`` reads four legacy BIFF workbooks; no library in the env WRITES a BIFF
``.xls``, so the four ``_read_*`` methods (the file boundary) are stubbed with parsed
objects and everything downstream runs for real, including the pinned compound table.
The parsers themselves run on hand-built frames shaped like the release (title row 0,
header row 2, body from row 3). No genome is consulted (#1045): every strain is stored
under its released ORF.

Synthetic build (Table S7 rows x the five filled columns; every other cell blank):

    YAL001C  Sulfometuron (SC), MMS (YPD) blank, Thialysine, Extract 95-57, Agelasine E
    YDL039C  all five
    YDL038C  all five (an ORF SGD merged into YDL039C; stored as released)
    YGR272C  all five (an ORF SGD merged into YGR271C-A; stored as released)
    YIL170W  all five (a locus R64 does not call a gene; stored as released)

Released ratios: 4 + 5 + 5 + 5 + 5 = 24; 82 x 5 - 24 = 386 released blanks have none;
410 source cells. The 5 Extract 95-57 ratios are HELD OUT under
``UNIDENTIFIED_COMPOUND_RULE`` (a crude extract known only by its collection code); the
5 agelasine E ratios are stored, the named compound carrying its typed inchikey gap
(#1045 row 19). Records: 3 + 4 + 4 + 4 + 4 = 19, in build order: idx 0-2 YAL001C
(sulfometuron, thialysine, agelasine E), 3-6 YDL039C, 7-10 YDL038C, 11-14 YGR272C, 15-18
YIL170W, each in column order sulfometuron, MMS, thialysine, agelasine E.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SC, YPD_LIQUID
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    ConcentrationUnit,
    CultureEnvironment,
    EukaryoticStrainBackground,
    Genotype,
    MatingType,
    MeasurementType,
    PreCultureSource,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    Zygosity,
)
from torchcell.datamodels.strain_background import GIAEVER_2002
from torchcell.datasets.scerevisiae import parsons2006 as p
from torchcell.datasets.scerevisiae.hoepfner2014 import ReleasedIdentity

S7_LABELS = [c.s7_label for c in p.CONDITIONS]
COL = {label: i for i, label in enumerate(S7_LABELS)}
FILLED = ["Sulfometuron methyl", "MMS", "Thialysine", "Extract 95-57", "Agelasine E"]


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _Resolution:
    def __init__(self, status: str, systematic: str | None) -> None:
        self.status = status
        self.systematic_name = systematic

    @property
    def is_current_gene(self) -> bool:
        return self.status in ("current", "renamed")


def _ratio_matrix() -> p.RatioMatrix:
    orfs = ["YAL001C", "YDL039C", "YDL038C", "YGR272C", "YIL170W"]
    values: list[list[float | None]] = []
    for row, _orf in enumerate(orfs):
        cells: list[float | None] = [None] * len(S7_LABELS)
        for k, label in enumerate(FILLED):
            cells[COL[label]] = 0.25 * (row + 1) * (1 if k % 2 == 0 else -1)
        values.append(cells)
    values[0][COL["MMS"]] = None
    return p.RatioMatrix(orfs=orfs, labels=S7_LABELS, values=values)


def _p_value_matrix() -> p.PValueMatrix:
    neg: dict[str, list[float | None]] = {}
    for orf in ["YAL001C", "YDL039C", "YDL038C", "YGR272C", "YIL170W"]:
        cells: list[float | None] = [None] * len(S7_LABELS)
        for label in FILLED:
            cells[COL[label]] = 2.0
        neg[orf] = cells
    neg["YAL001C"][COL["MMS"]] = 1.5  # a p-value whose ratio is blank
    neg["YDL039C"][COL["Thialysine"]] = None  # a ratio without a p-value
    neg["YDL038C"][COL["Sulfometuron methyl"]] = 300.0  # the release's cap
    for orf in ["YKL212W", "YML008C"]:
        neg[orf] = [0.0] * len(S7_LABELS)
    return p.PValueMatrix(orfs=list(neg), labels=S7_LABELS, neg_log10=neg)


def _s1_rows() -> list[p.S1Row]:
    special: dict[str, tuple[str | float, str]] = {
        "Methanesulfonic acid methyl ester (MMS)": (4e-05, "0.000%"),
        "Sulfometuron methyl": ("0.2 nM", "General"),
        "Thialysine": ("3 μM", "General"),
        "Extract 95-57": ("30 μg/ml", "General"),
    }
    return [
        p.S1Row(
            cas=None,
            name=spec.s1_name,
            dose_value=special.get(spec.s1_name, ("1 μM", "General"))[0],
            dose_format=special.get(spec.s1_name, ("1 μM", "General"))[1],
        )
        for spec in p.CONDITIONS
    ]


def _build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> p.EnvChemgenParsons2006Dataset:
    """A real scratch build of the synthetic release (19 records)."""
    cls = p.EnvChemgenParsons2006Dataset
    monkeypatch.setattr(cls, "_read_ratio_matrix", lambda self: _ratio_matrix())
    monkeypatch.setattr(cls, "_read_p_value_matrix", lambda self: _p_value_matrix())
    monkeypatch.setattr(
        cls, "_read_multidrug_sensitive", lambda self: ["YML008C", "YKL212W"]
    )
    monkeypatch.setattr(cls, "_read_s1", lambda self: _s1_rows())
    raw = tmp_path / "parsons" / "raw"
    raw.mkdir(parents=True)
    for table in p.RAW_TABLES:
        (raw / table.filename).write_bytes(b"placeholder: the readers are stubbed")
    return cls(root=str(tmp_path / "parsons"))


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    dataset = _build(tmp_path, monkeypatch)
    yield dataset
    dataset.close_lmdb()


def _exp(dataset: Any, idx: int) -> StrainEnvironmentResponseExperiment:
    """The stored experiment as its schema object."""
    return StrainEnvironmentResponseExperiment.model_validate(
        dataset[idx]["experiment"]
    )


def _deletion(dataset: Any, idx: int) -> BarcodedKanMxDeletionPerturbation:
    """The record's one perturbation, narrowed to the barcoded kanMX leaf."""
    genotype = _exp(dataset, idx).genotype
    assert isinstance(genotype, Genotype)
    (perturbation,) = genotype.perturbations
    assert isinstance(perturbation, BarcodedKanMxDeletionPerturbation)
    return perturbation


def _edit(environment: CultureEnvironment) -> SmallMoleculePerturbation:
    """The environment's one compound edit."""
    (edit,) = environment.perturbations
    assert isinstance(edit, SmallMoleculePerturbation)
    return edit


def _solvent(edit: SmallMoleculePerturbation) -> Solvent:
    assert edit.solvent is not None
    return edit.solvent


def _ref(dataset: Any, idx: int) -> StrainEnvironmentResponseExperimentReference:
    """The stored reference as its schema object."""
    return StrainEnvironmentResponseExperimentReference.model_validate(
        dict(dataset[idx]["reference"])
    )


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
def test_drop_log_counts_every_cell_once(built: Any) -> None:
    """Every released ratio of an identified agent is a record (#1045); the released
    blanks and the held-out extract ratios have none, and the rules sum to the gap.
    """
    assert len(built) == 19
    log = json.loads((Path(built.preprocess_dir) / "dropped_records.json").read_text())
    assert (log["source_records"], log["released_ratios"]) == (410, 24)
    assert (log["kept_records"], log["dropped_records"]) == (19, 391)
    assert [(r["rule"], r["n_records"]) for r in log["rules"]] == [
        ("ratio_cell_blank_in_release", 386),
        (p.UNIDENTIFIED_COMPOUND_RULE, 5),
    ]
    assert log["source_records"] == log["kept_records"] + sum(
        r["n_records"] for r in log["rules"]
    )
    gapped = {
        g["condition"]: (g["compound_name"], g["gap_field"], g["gap_reason"])
        for g in log["compound_identity_gaps"]
    }
    names = {"192A4-Stichloroside": "Stichloroside"}
    assert gapped == {
        label: (names.get(label, label), "inchikey", "deferred_pending_source_review")
        for label in ["192A4-Stichloroside", "Agelasine E"]
    }
    assert {
        g["condition"]: g["n_records"]
        for g in log["compound_identity_gaps"]
        if g["n_records"]
    } == {"Agelasine E": 5}
    assert log["release"] == {
        "table_s4_strains": 7,
        "table_s4_p_values": 24 + 2 * 82,
        "multidrug_sensitive_strains_without_ratios": ["YKL212W", "YML008C"],
        "multidrug_sensitive_p_values": 164,
        "p_value_cells_without_a_ratio": 1,
        "ratio_cells_without_a_p_value": 1,
    }


def test_a_name_only_compound_is_stored_with_its_typed_inchikey_gap(built: Any) -> None:
    """YAL001C x Agelasine E (record 2) is stored (#1045 row 19): the named compound
    keeps its released label, carries no identifier, and declares the inchikey gap.
    """
    edit = _edit(_exp(built, 2).environment)
    assert (edit.compound.name, edit.compound.inchikey, edit.compound.pubchem_cid) == (
        "Agelasine E",
        None,
        None,
    )
    assert [(g.field, g.reason.value) for g in edit.compound.provenance_gaps] == [
        ("inchikey", "deferred_pending_source_review")
    ]
    assert _solvent(edit).name == "DMSO"
    assert _exp(built, 2).phenotype.environment_response == 0.25


def test_an_unidentified_extract_is_held_out_and_listed_in_the_ledger(
    built: Any,
) -> None:
    """Extract 95-57 (a collection code, no composition) has no stored record; the
    ledger lists it with its dose and its 5 ratios, quotes the source, and balances.
    """
    names = {_edit(_exp(built, i).environment).compound.name for i in range(len(built))}
    assert not any(name.startswith("Extract") for name in names)
    assert "Agelasine E" in names
    log = json.loads((Path(built.preprocess_dir) / "dropped_records.json").read_text())
    (rule,) = [r for r in log["rules"] if r["rule"] == p.UNIDENTIFIED_COMPOUND_RULE]
    held = log["unidentified_compounds_held_out"]
    assert (
        rule["items"]
        == [h["condition"] for h in held]
        == sorted(c.s7_label for c in p.CONDITIONS if c.s7_label.startswith("Extract"))
    )
    assert {h["condition"]: h["n_records"] for h in held if h["n_records"]} == {
        "Extract 95-57": 5
    }
    assert rule["n_records"] == sum(h["n_records"] for h in held) == 5
    (extract,) = [h for h in held if h["condition"] == "Extract 95-57"]
    assert extract == {
        "condition": "Extract 95-57",
        "s1_name": "Extract 95-57",
        "dose": {"value": 30.0, "unit": "ug/mL", "basis": None},
        "vehicle": "methanol",
        "medium": "ypd",
        "n_records": 5,
    }
    assert log["unidentified_compound_rule"] == (
        "compound_has_no_released_name_or_structure_not_representable"
    )
    assert log["unidentified_compound_policy"] == p.UNIDENTIFIED_COMPOUND_POLICY
    sources = log["unidentified_compound_sources"]
    assert [s["quote"] for s in sources] == [
        p.N_CRUDE_EXTRACTS.quote,
        p.EXTRACT_COLLECTION.quote,
        p.EXTRACT_TABLE_COMMENT.quote,
    ]
    assert {s["provenance"]["sha256"] for s in sources} == {p.PAPER_MD_SHA256}
    assert log["source_records"] == log["kept_records"] + sum(
        r["n_records"] for r in log["rules"]
    )


def test_a_released_identity_admits_an_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filled ``RELEASED_IDENTITIES`` entry stores the extract under the released
    name; the admission table ships empty.
    """
    assert p.RELEASED_IDENTITIES == {}
    monkeypatch.setitem(
        p.RELEASED_IDENTITIES,
        "Extract 95-57",
        ReleasedIdentity(name="released extract 95-57", source=p.EXTRACT_COLLECTION),
    )
    dataset = _build(tmp_path, monkeypatch)
    try:
        assert len(dataset) == 24
        log = json.loads(
            (Path(dataset.preprocess_dir) / "dropped_records.json").read_text()
        )
        assert [h["condition"] for h in log["unidentified_compounds_held_out"]] == [
            "Extract 00-132",
            "Extract 00-192",
            "Extract 00-243",
            "Extract 00-303C",
            "Extract 00-89",
            "Extract 6592",
        ]
        assert _edit(_exp(dataset, 2).environment).compound.name == (
            "released extract 95-57"
        )
    finally:
        dataset.close_lmdb()


def test_record_stores_the_released_ratio_and_the_s4_p_value(built: Any) -> None:
    # YDL038C x sulfometuron methyl, -log10 p at the cap
    phenotype = _exp(built, 7).phenotype
    assert phenotype.environment_response == 0.75
    assert phenotype.measurement_type is MeasurementType.log2_ratio
    assert phenotype.assay_type is AssayType.pooled_competitive_growth_barcode
    assert phenotype.environment_response_p_value == 1e-300
    assert (phenotype.n_samples, phenotype.sample_unit) == (
        2,
        SampleUnit.technical_replicate,
    )
    assert phenotype.units == p.MEASUREMENT_UNITS
    assert "POSITIVE = the deletion strain is depleted" in phenotype.units
    assert {g.field for g in phenotype.provenance_gaps} == {
        "environment_response_uncertainty",
        "environment_response_se",
    }
    assert _exp(built, 0).phenotype.environment_response_p_value == 0.01
    assert _exp(built, 5).phenotype.environment_response_p_value is None


def test_every_strain_is_stored_under_its_released_orf(built: Any) -> None:
    """Each strain's first record carries Table S7's ORF as both names: the merged
    YDL038C and YGR272C and the non-gene YIL170W included, no ConstructedOrf (#1045).
    """
    stored = [
        (d.systematic_gene_name, d.perturbed_gene_name, d.constructed_orf)
        for d in (_deletion(built, i) for i in (0, 3, 7, 11, 15))
    ]
    assert stored == [
        ("YAL001C", "YAL001C", None),
        ("YDL039C", "YDL039C", None),
        ("YDL038C", "YDL038C", None),
        ("YGR272C", "YGR272C", None),
        ("YIL170W", "YIL170W", None),
    ]
    assert sorted(built.gene_set) == [
        "YAL001C",
        "YDL038C",
        "YDL039C",
        "YGR272C",
        "YIL170W",
    ]


def test_deletion_is_a_barcoded_kanmx4_strain_with_typed_barcode_gaps(
    built: Any,
) -> None:
    perturbation = _deletion(built, 0)
    assert perturbation.perturbation_type == "barcoded_kanmx_deletion"
    assert (perturbation.systematic_gene_name, perturbation.perturbed_gene_name) == (
        "YAL001C",
        "YAL001C",
    )
    assert perturbation.cassette == "kanMX4"
    assert (
        perturbation.collection == "MATa haploid deletion mutants (Research Genetics)"
    )
    assert (perturbation.barcode, perturbation.downtag_barcode) == (None, None)
    assert {g.field for g in perturbation.provenance_gaps} == {
        "barcode",
        "downtag_barcode",
    }


def test_environment_carries_medium_compound_dose_and_vehicle(built: Any) -> None:
    sulfometuron = _exp(built, 0).environment
    mms = _exp(built, 4).environment
    thialysine = _exp(built, 1).environment
    assert sulfometuron.media.base_medium == "SC"
    assert mms.media.base_medium == "YPD"
    assert thialysine.media.name == p.PARSONS_SC_MINUS_LYSINE.name
    assert [c.name for c in thialysine.media.dropouts] == ["L-lysine"]
    edit = _edit(mms)
    assert edit.compound.name == "methyl methanesulfonate"
    assert edit.compound.inchikey == "MBABOKRGFJTBAE-UHFFFAOYSA-N"
    assert (edit.concentration.value, edit.concentration.unit) == (
        0.004,
        ConcentrationUnit.percent,
    )
    solvent = _solvent(edit)
    assert (solvent.name, solvent.percent, solvent.compound) == ("water", 1.0, None)
    edit = _edit(sulfometuron)
    assert edit.compound.name == "sulfometuron-methyl"
    assert (edit.concentration.value, edit.concentration.unit) == (
        0.2,
        ConcentrationUnit.nanomolar,
    )
    solvent = _solvent(edit)
    assert solvent.name == "DMSO"
    assert solvent.compound is not None
    assert solvent.compound.name == "dimethyl sulfoxide"
    assert _edit(thialysine).compound.inchikey == "CVHKULVNPGAEQM-WCCKRBBISA-N"
    assert sulfometuron.temperature is not None
    assert (sulfometuron.temperature.value, sulfometuron.duration_hours) == (30.0, 38.0)
    assert sulfometuron.aerobicity == "aerobic"
    culture = sulfometuron.culture_format
    assert culture is not None and sulfometuron.pre_culture is not None
    assert culture.inoculum_od600 == 0.025
    assert sulfometuron.pre_culture.source is PreCultureSource.frozen_stock
    assert {g.field for g in culture.provenance_gaps} == {
        "working_volume_ul",
        "shaking_rpm",
        "vessel",
    }


def test_one_reference_per_medium_on_the_by4741_background(built: Any) -> None:
    index = built.experiment_reference_index
    groups = sorted(sorted(int(i) for i in ref.member_indices) for ref in index)
    assert groups == [
        [0, 3, 7, 11, 15],
        [1, 5, 9, 13, 17],
        [2, 4, 6, 8, 10, 12, 14, 16, 18],
    ]
    reference = _ref(built, 0)
    assert reference.environment_reference.perturbations == []
    assert reference.phenotype_reference.environment_response == 0.0
    assert reference.phenotype_reference.units == p.REFERENCE_UNITS
    genome = reference.genome_reference
    assert (genome.strain, genome.ploidy) == ("BY4741", "haploid")
    background = genome.background
    assert isinstance(background, EukaryoticStrainBackground)
    assert background.mating_type is MatingType.a
    assert [a.allele_name for a in background.alleles] == [
        "his3Δ1",
        "leu2Δ0",
        "met15Δ0",
        "ura3Δ0",
    ]
    assert all(a.zygosity is Zygosity.haploid for a in background.alleles)
    assert all(a.provenance == [p.BY4741_GENOTYPE] for a in background.alleles)
    assert background.provenance is None
    (gap,) = background.provenance_gaps
    assert (gap.field, gap.resolve_with) == ("provenance", GIAEVER_2002)


def test_an_s4_strain_set_other_than_table_s2_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cls = p.EnvChemgenParsons2006Dataset
    monkeypatch.setattr(cls, "_read_multidrug_sensitive", lambda self: ["YKL212W"])
    with pytest.raises(RuntimeError, match="not Table S2's multidrug-sensitive set"):
        _build_with(tmp_path, monkeypatch, keep_mds=True)


def test_an_s7_orf_missing_from_s4_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def short() -> p.PValueMatrix:
        matrix = _p_value_matrix()
        del matrix.neg_log10["YAL001C"]
        return matrix

    monkeypatch.setattr(
        p.EnvChemgenParsons2006Dataset, "_read_p_value_matrix", lambda self: short()
    )
    with pytest.raises(RuntimeError, match="Table S7 ORFs absent from Table S4"):
        _build_with(tmp_path, monkeypatch, keep_p=True)


def test_the_loader_takes_no_genome() -> None:
    """No genome is consulted (#1045), so ``build_dataset_lmdb`` injects none: it injects
    by parameter name.
    """
    params = inspect.signature(p.EnvChemgenParsons2006Dataset.__init__).parameters
    assert "genome" not in params


def _build_with(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    keep_mds: bool = False,
    keep_p: bool = False,
) -> p.EnvChemgenParsons2006Dataset:
    """``_build`` that leaves an already-patched reader in place."""
    cls = p.EnvChemgenParsons2006Dataset
    monkeypatch.setattr(cls, "_read_ratio_matrix", lambda self: _ratio_matrix())
    if not keep_p:
        monkeypatch.setattr(cls, "_read_p_value_matrix", lambda self: _p_value_matrix())
    if not keep_mds:
        monkeypatch.setattr(
            cls, "_read_multidrug_sensitive", lambda self: ["YML008C", "YKL212W"]
        )
    monkeypatch.setattr(cls, "_read_s1", lambda self: _s1_rows())
    raw = tmp_path / "parsons" / "raw"
    raw.mkdir(parents=True)
    for table in p.RAW_TABLES:
        (raw / table.filename).write_bytes(b"x")
    return cls(root=str(tmp_path / "parsons"))


def test_inline_construction_hooks_are_inert(built: Any) -> None:
    frame = pd.DataFrame({"a": [1]})
    assert built.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        built.create_experiment()


# --------------------------------------------------------------------------- #
# Doses, conditions, compounds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "fmt", "expected"),
    [
        ("0.33 mM", "General", (0.33, ConcentrationUnit.millimolar)),
        ("61 μM", "General", (61.0, ConcentrationUnit.micromolar)),
        ("7 nM", "General", (7.0, ConcentrationUnit.nanomolar)),
        ("5.5 pM", "General", (0.0055, ConcentrationUnit.nanomolar)),
        ("71 pM", "General", (0.071, ConcentrationUnit.nanomolar)),
        ("130 μg/ml", "General", (130.0, ConcentrationUnit.ug_per_ml)),
        ("5% (v/v)", "General", (5.0, ConcentrationUnit.percent_v_v)),
        ("0.00001%(v/v)", "General", (1e-05, ConcentrationUnit.percent_v_v)),
        (4e-05, "0.000%", (0.004, ConcentrationUnit.percent)),
    ],
)
def test_parse_dose_converts_exactly(
    value: str | float, fmt: str, expected: tuple[float, ConcentrationUnit]
) -> None:
    dose = p.parse_dose(p.S1Row(cas=None, name="x", dose_value=value, dose_format=fmt))
    assert (dose.value, dose.unit) == expected


@pytest.mark.parametrize(
    ("value", "fmt", "match"),
    [
        (0.5, "General", "is not percent-formatted"),
        ("12 mg/ml", "General", "unparseable dose"),
        ("", "General", "unparseable dose"),
    ],
)
def test_parse_dose_refuses_what_it_cannot_read(
    value: str | float, fmt: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        p.parse_dose(p.S1Row(cas=None, name="x", dose_value=value, dose_format=fmt))


def test_conditions_are_82_unique_columns_with_their_footnotes_stripped() -> None:
    assert len(p.CONDITIONS) == p.N_CONDITIONS.value == 82
    assert len(set(S7_LABELS)) == 82
    assert len({c.s1_name for c in p.CONDITIONS}) == 82
    by_label = {c.s7_label: c for c in p.CONDITIONS}
    assert by_label["Alamethicin"].lookup_label == "Alamethicin"
    assert by_label["Oligomycin"].lookup_label == "Oligomycin"
    assert by_label["Tunicamycin"].lookup_label == "Tunicamycin"
    assert [c.s7_label for c in p.CONDITIONS if c.medium != "ypd"] == [
        "Sulfometuron methyl",
        "Brefeldin A",
        "Thialysine",
    ]
    vehicles: dict[str, list[str]] = {}
    for c in p.CONDITIONS:
        vehicles.setdefault(c.vehicle, []).append(c.s7_label)
    assert len(vehicles["water"]) == 18
    assert vehicles["ethanol"] == ["Artemisinin", "Brefeldin A", "Phenantroline"]
    assert vehicles["saline"] == ["Cisplatin"]
    assert vehicles["methanol"] == ["Wortmannin"] + [
        label for label in S7_LABELS if label.startswith("Extract")
    ]


def test_the_seven_crude_extracts_are_the_unidentified_conditions() -> None:
    held = [c.s7_label for c in p.unidentified_conditions()]
    assert held == [
        "Extract 95-57",
        "Extract 6592",
        "Extract 00-89",
        "Extract 00-303C",
        "Extract 00-243",
        "Extract 00-192",
        "Extract 00-132",
    ]
    assert len(held) == p.N_CRUDE_EXTRACTS.value
    for label in ["Agelasine E", "192A4-Stichloroside", "CG4-Theopalauamide"]:
        (spec,) = [c for c in p.CONDITIONS if c.s7_label == label]
        assert p.compound_is_identified(spec)


def test_an_extract_code_count_other_than_the_papers_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra = p.ConditionSpec(
        s7_label="Extract 99-1",
        s1_name="Extract 99-1",
        vehicle="methanol",
        medium="ypd",
    )
    monkeypatch.setattr(p, "CONDITIONS", (*p.CONDITIONS, extra))
    with pytest.raises(RuntimeError, match="8 Table S1 names are extract collection"):
        p.unidentified_conditions()


def test_every_condition_resolves_except_the_nine_stored_with_a_gap() -> None:
    unidentified = sorted(
        c.s7_label
        for c in p.CONDITIONS
        if (compound := resolved_compound(c.lookup_label)).inchikey is None
        and compound.pubchem_cid is None
        and compound.chebi_id is None
    )
    assert unidentified == [
        "192A4-Stichloroside",
        "Agelasine E",
        "Extract 00-132",
        "Extract 00-192",
        "Extract 00-243",
        "Extract 00-303C",
        "Extract 00-89",
        "Extract 6592",
        "Extract 95-57",
    ]
    names = [
        resolved_compound(c.lookup_label).name
        for c in p.CONDITIONS
        if c.s7_label not in unidentified
    ]
    assert len(set(names)) == len(names) == 73


@pytest.mark.parametrize(
    ("label", "name", "inchikey"),
    [
        # CAS 4099-35-8, the hydrochloride, not the SGA media's thialysine row
        ("Thialysine", "l-cysteine, s-(2-aminoethyl)-, hydrochloride",
         "CVHKULVNPGAEQM-WCCKRBBISA-N"),
        # CAS 15663-27-1 -> CID 5460033, the same key as the "cisplatin" row
        ("cis-diamminedichloroplatinum (II) Cisplatin", "azane;dichloroplatinum",
         "LXZZYRPGZAFOLE-UHFFFAOYSA-L"),
        ("Clomiphene citrate", "Clomiphene citrate", "PYTMYKVIJXPNBD-UHFFFAOYSA-N"),
        ("(+)-Usnic acid", "usnic acid, (r)-", "WEYVVCKOOFYHRW-SFHVURJKSA-N"),
        ("Basiliskamide", "basiliskamide a", "QAHZDGAHECEVRM-HXKCHMFGSA-N"),
        ("(1R, 2S, 5R) - ( - ) - Menthol", "(-)-menthol",
         "NOOLISFMXDJSKH-KXUCPTDWSA-N"),
    ],
)  # fmt: skip
def test_cas_identities_from_the_pinned_table(
    label: str, name: str, inchikey: str
) -> None:
    compound = resolved_compound(label)
    assert (compound.name, compound.inchikey) == (name, inchikey)


def test_the_two_mixtures_carry_an_identifier_and_no_inchikey() -> None:
    alamethicin = resolved_compound("Alamethicin")
    oligomycin = resolved_compound("Oligomycin")
    assert (alamethicin.inchikey, alamethicin.pubchem_cid) == (None, 76958618)
    assert (oligomycin.inchikey, oligomycin.pubchem_cid, oligomycin.chebi_id) == (
        None,
        None,
        "CHEBI:84314",
    )


def test_media_join_the_shared_library_at_their_base() -> None:
    assert p.PARSONS_YPD.components == YPD_LIQUID.components
    assert p.PARSONS_YPD.provenance[-1] == p.YPD_CULTURE
    assert p.PARSONS_SC.components == SC.components
    assert p.PARSONS_SC_MINUS_LYSINE.base_medium == "SC"
    assert "L-lysine" not in {
        c.compound.name for c in p.PARSONS_SC_MINUS_LYSINE.components
    }


@pytest.mark.parametrize(
    ("key", "name", "compound"),
    [
        ("dmso", "DMSO", "dimethyl sulfoxide"),
        ("ethanol", "ethanol", "ethanol"),
        ("methanol", "methanol", "methanol"),
        ("water", "water", None),
        ("saline", "0.9% sodium chloride solution", None),
    ],
)
def test_vehicle_is_the_methods_solvent_at_one_percent(
    key: p.Vehicle, name: str, compound: str | None
) -> None:
    solvent = p.vehicle(key)
    assert (solvent.name, solvent.percent) == (name, 1.0)
    assert (None if solvent.compound is None else solvent.compound.name) == compound


# --------------------------------------------------------------------------- #
# Release parsers
# --------------------------------------------------------------------------- #
def _frame(title: str, header: list[Any], body: list[list[Any]]) -> pd.DataFrame:
    width = len(header)
    rows = [[title] + [None] * (width - 1), [None] * width, header, *body]
    return pd.DataFrame(rows)


def _s7_frame(body: list[list[Any]]) -> pd.DataFrame:
    return _frame(
        p.TABLE_S7.title, ["ORF", *[f"{label} " for label in S7_LABELS]], body
    )


def test_ratio_matrix_reads_floats_and_the_two_space_blank() -> None:
    row = ["YAL001C ", 0.5, "  ", *([1.0] * 80)]
    matrix = p.parse_ratio_matrix(_s7_frame([row]))
    assert matrix.orfs == ["YAL001C"]
    assert matrix.labels == S7_LABELS
    assert matrix.values[0][:3] == [0.5, None, 1.0]
    assert (matrix.n_numeric, matrix.n_blank) == (81, 1)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ([["YAL001C", "x", *([1.0] * 81)]], "unexpected cell 'x'"),
        ([["YAL001C", math.inf, *([1.0] * 81)]], "non-finite"),
        ([["YAL001C", *([1.0] * 82)], ["YAL001C", *([1.0] * 82)]], "ORF twice"),
    ],
)
def test_ratio_matrix_refuses_cells_the_release_does_not_write(
    body: list[list[Any]], match: str
) -> None:
    with pytest.raises(RuntimeError, match=match):
        p.parse_ratio_matrix(_s7_frame(body))


def test_ratio_matrix_refuses_a_relabelled_column_and_a_wrong_title() -> None:
    header = ["ORF", *S7_LABELS[:-1], "Geldanamycin B"]
    with pytest.raises(RuntimeError, match="Geldanamycin B"):
        p.parse_ratio_matrix(_frame(p.TABLE_S7.title, header, []))
    with pytest.raises(RuntimeError, match="title"):
        p.parse_ratio_matrix(_frame("Table 7", ["ORF", *S7_LABELS], []))
    with pytest.raises(RuntimeError, match="first header cell"):
        p.parse_ratio_matrix(_frame(p.TABLE_S7.title, ["orf", *S7_LABELS], []))


def test_p_value_matrix_reads_blanks_and_refuses_out_of_range() -> None:
    header = ["ORF", "Standard Name", *S7_LABELS]
    body = [["YAL001C", "TFC3", 2.5, float("nan"), *([300] * 80)]]
    matrix = p.parse_p_value_matrix(_frame(p.TABLE_S4.title, header, body))
    assert matrix.neg_log10["YAL001C"][:3] == [2.5, None, 300.0]
    bad = [["YAL001C", "TFC3", 301.0, *([0.0] * 81)]]
    with pytest.raises(RuntimeError, match="-log10 p 301.0"):
        p.parse_p_value_matrix(_frame(p.TABLE_S4.title, header, bad))
    twice = [["YAL001C", "", *([0.0] * 82)], ["YAL001C", "", *([0.0] * 82)]]
    with pytest.raises(RuntimeError, match="ORF twice"):
        p.parse_p_value_matrix(_frame(p.TABLE_S4.title, header, twice))
    with pytest.raises(RuntimeError, match="header starts"):
        p.parse_p_value_matrix(_frame(p.TABLE_S4.title, ["ORF", *S7_LABELS, "x"], []))
    with pytest.raises(RuntimeError, match="labels differ"):
        p.parse_p_value_matrix(
            _frame(p.TABLE_S4.title, ["ORF", "Standard Name", *S7_LABELS[::-1]], [])
        )


def test_orf_list_stops_at_the_first_blank_row() -> None:
    frame = _frame(
        p.TABLE_S2.title,
        ["ORF", "Standard name"],
        [["YKL212W", "SAC1"], ["YML008C ", "ERG6"], [float("nan"), None], ["X", None]],
    )
    assert p.parse_orf_list(frame, p.TABLE_S2) == ["YKL212W", "YML008C"]
    with pytest.raises(RuntimeError, match="not 'ORF'"):
        p.parse_orf_list(_frame(p.TABLE_S2.title, ["Gene", "x"], []), p.TABLE_S2)


def test_s1_by_name_needs_every_condition_exactly_once() -> None:
    rows = _s1_rows()
    assert set(p.s1_by_name(rows)) == {c.s1_name for c in p.CONDITIONS}
    with pytest.raises(RuntimeError, match="missing \\['Wortmannin'\\]"):
        p.s1_by_name([r for r in rows if r.name != "Wortmannin"])
    with pytest.raises(RuntimeError, match="name twice"):
        p.s1_by_name([*rows, rows[0]])


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _synthetic_tables(library: Path) -> tuple[p.RawTable, ...]:
    tables = []
    for table in p.RAW_TABLES:
        payload = f"synthetic {table.table}".encode()
        path = library / table.library_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        tables.append(
            table.model_copy(
                update={
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
        )
    return tuple(tables)


def test_deposit_writes_the_four_tables_and_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "torchcell-library" / p.CITATION_KEY
    tables = _synthetic_tables(library)
    monkeypatch.setattr(p, "RAW_TABLES", tables)
    root = p.deposit_raw_mirror(data_root=str(tmp_path))
    assert root == tmp_path / "torchcell-raw" / p.CITATION_KEY
    manifest = p.load_manifest(str(tmp_path))
    assert [f.path for f in manifest.files] == [
        "data/mmc2.xls",
        "data/mmc3.xls",
        "data/mmc5.xls",
        "data/mmc8.xls",
    ]
    record = manifest.files[3]
    assert record.retrieval is not None
    assert record.retrieval.source_url == (
        "https://ars.els-cdn.com/content/image/1-s2.0-S0092867406009068-mmc8.xls"
    )
    assert record.retrieval.params == {"pii": p.PII, "filename": "mmc8.xls"}
    assert record.retrieval.retriever == "torchcell.literature.retrieve.elsevier_mmc"
    assert p.manifest_sha256(manifest, "data/mmc8.xls") == tables[3].sha256
    assert manifest.si_expected == list(p.NOT_CONSUMED)
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        p.manifest_sha256(manifest, "data/mmc9.xls")
    # idempotent on identical bytes
    assert p.deposit_raw_mirror(data_root=str(tmp_path)) == root


def test_deposit_refuses_before_writing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "torchcell-library" / p.CITATION_KEY
    tables = list(_synthetic_tables(library))
    tables[3] = tables[3].model_copy(update={"sha256": "0" * 64})
    monkeypatch.setattr(p, "RAW_TABLES", tuple(tables))
    with pytest.raises(RuntimeError, match="is not the pinned"):
        p.deposit_raw_mirror(data_root=str(tmp_path))
    assert not (tmp_path / "torchcell-raw").exists()
    (library / tables[0].library_rel).unlink()
    with pytest.raises(RuntimeError, match="library mirror is missing"):
        p.deposit_raw_mirror(data_root=str(tmp_path))


def test_deposit_refuses_an_existing_mirror_file_with_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "torchcell-library" / p.CITATION_KEY
    monkeypatch.setattr(p, "RAW_TABLES", _synthetic_tables(library))
    dest = tmp_path / "torchcell-raw" / p.CITATION_KEY / "data" / "mmc2.xls"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        p.deposit_raw_mirror(data_root=str(tmp_path))


def test_download_links_the_mirror_and_refuses_without_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "torchcell-library" / p.CITATION_KEY
    monkeypatch.setattr(p, "RAW_TABLES", _synthetic_tables(library))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="run deposit_raw_mirror"):
        p.load_manifest()
    p.deposit_raw_mirror()
    dataset = p.EnvChemgenParsons2006Dataset.__new__(p.EnvChemgenParsons2006Dataset)
    raw_dir = tmp_path / "build" / "raw"
    monkeypatch.setattr(
        p.EnvChemgenParsons2006Dataset, "raw_dir", property(lambda self: str(raw_dir))
    )
    dataset.download()
    assert sorted(os.listdir(raw_dir)) == [
        "mmc2.xls",
        "mmc3.xls",
        "mmc5.xls",
        "mmc8.xls",
    ]
    assert (raw_dir / "mmc8.xls").read_bytes() == b"synthetic S7"
    (tmp_path / "torchcell-raw" / p.CITATION_KEY / "data" / "mmc5.xls").unlink()
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        dataset.download()


# --------------------------------------------------------------------------- #
# Sourcing audit (skips when the mirror is absent)
# --------------------------------------------------------------------------- #
def _library_root() -> Path | None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        return None
    root = Path(data_root) / "torchcell-library"
    return root if (root / p.CITATION_KEY / p.SI_MD).exists() else None


@pytest.mark.skipif(_library_root() is None, reason="literature mirror not present")
def test_every_sourced_value_is_a_verbatim_quote_in_its_pinned_mirror() -> None:
    root = _library_root()
    assert root is not None
    for value in p.SOURCED_VALUES:
        path = value.source_path(root)
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == value.provenance.sha256, path
        assert value.quote in payload.decode("utf-8"), (value.value, path)
