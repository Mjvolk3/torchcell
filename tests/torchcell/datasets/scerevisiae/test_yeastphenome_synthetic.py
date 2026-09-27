# tests/torchcell/datasets/scerevisiae/test_yeastphenome_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_yeastphenome_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_yeastphenome_synthetic.py
"""Hermetic build tests for the YeastPhenome loader on two synthetic screens.

``SCREENS`` (44 pinned real screens) is monkeypatched to two synthetic entries, and each
screen's ``<pmid>_<stem>_valuez.txt`` is hand-written under ``<root>/raw/`` so PyG skips
``download()``. The compound resolver is the pinned in-repo table (no network). The
data-gated mirror test lives in ``test_yeastphenome.py``; this file pins the column
grammar and the exact records without any ``$DATA_ROOT``.

Screen A (PMID 99999901, stem ``synthetic_a``), header cells and their fate:

    1  hom | growth (colony size) | furfural [10 mM] | YPD | synthetic      kept, BY4743 diploid
    2  hap a | growth (culture turbidity) | quinine [2 mM] | SD | synthetic kept, BY4741 haploid
    3  het | growth (colony size) | furfural [10 mM] | YPD | synthetic      het_or_ambiguous_zygosity
    4  hom | growth (colony size) | heat [37 C] | YPD | synthetic           condition_unparseable (unit C)
    5  hom | growth (colony size) | furfural [10 mM] | synthetic            media_absent_or_unknown (4 fields)
    6  hom | expression (microarray) | furfural [10 mM] | YPD | synthetic   not_growth
    7  hom | growth                                                         malformed_header
    8  hom | growth (colony size) | furfural [10 mM] | LB | synthetic       media_absent_or_unknown (LB)
    9  (empty cell)                                                         skipped silently

    rows: YAL001C 1.5 / -2.25; YBR002C 0.5 / (blank); not_an_orf 3.0 / 3.0; Q0250 -0.75 / 0.125

Screen B (PMID 99999902, stem ``synthetic_b``): one column
``hap alpha | growth (pooled barseq) | furfural [10 mM] | SC + EtOH | synthetic`` (SC
synthetic liquid, BY4742 haploid), one row YAL001C 2.0.

Records in write order: A col 1 -> YAL001C, YBR002C, Q0250 (idx 0-2); A col 2 -> YAL001C,
Q0250 (idx 3-4); B col 1 -> YAL001C (idx 5). Six records, three distinct references.
"""

import json
import socket
from pathlib import Path
from typing import Literal

import pytest

from torchcell.datamodels.schema import (
    BiologicAgentClass,
    BiologicPerturbation,
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    Publication,
    ReferenceGenome,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import yeastphenome
from torchcell.datasets.scerevisiae.yeastphenome import (
    CURATION,
    NPV_UNITS,
    YeastPhenomeDataset,
    _column_meta,
    _parse_condition,
    _parse_media,
    _readout_method,
)
from torchcell.verification import ProvenanceGap, ProvenanceGapReason

SCREEN_A = {"pmid": "99999901", "stem": "synthetic_a", "valuez_sha256": "0" * 64}
SCREEN_B = {"pmid": "99999902", "stem": "synthetic_b", "valuez_sha256": "1" * 64}

HEADER_A = [
    "orf",
    "hom | growth (colony size) | furfural [10 mM] | YPD | synthetic",
    "hap a | growth (culture turbidity) | quinine [2 mM] | SD | synthetic",
    "het | growth (colony size) | furfural [10 mM] | YPD | synthetic",
    "hom | growth (colony size) | heat [37 C] | YPD | synthetic",
    "hom | growth (colony size) | furfural [10 mM] | synthetic",
    "hom | expression (microarray) | furfural [10 mM] | YPD | synthetic",
    "hom | growth",
    "hom | growth (colony size) | furfural [10 mM] | LB | synthetic",
    "",
]
ROWS_A = [
    ["YAL001C", "1.5", "-2.25", "1", "1", "1", "1", "1", "1", "1"],
    ["YBR002C", "0.5", "", "1", "1", "1", "1", "1", "1", "1"],
    ["not_an_orf", "3.0", "3.0", "1", "1", "1", "1", "1", "1", "1"],
    ["Q0250", "-0.75", "0.125", "1", "1", "1", "1", "1", "1", "1"],
]
HEADER_B = [
    "orf",
    "hap alpha | growth (pooled barseq) | furfural [10 mM] | SC + EtOH | synthetic",
]
ROWS_B = [["YAL001C", "2.0"]]

FURFURAL = Compound(
    name="furfural",
    inchikey="HYBBIBNJHNGZAN-UHFFFAOYSA-N",
    smiles="C1=COC(=C1)C=O",
    pubchem_cid=7362,
    chebi_id="CHEBI:30976",
)
QUININE_UNRESOLVED = Compound(
    name="quinine",
    provenance_gaps=[
        ProvenanceGap(
            field="inchikey", reason=ProvenanceGapReason.deferred_pending_source_review
        )
    ],
)
TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_carried_by_curation,
    looked_in=CURATION,
)


def _phenotype_gaps() -> list[ProvenanceGap]:
    return [
        ProvenanceGap(
            field=f,
            reason=ProvenanceGapReason.not_carried_by_curation,
            looked_in=CURATION,
        )
        for f in ("n_samples", "environment_response_uncertainty", "sample_unit")
    ]


def _environment(media: Media, compound: Compound, mm: float) -> Environment:
    return Environment(
        media=media,
        temperature=None,
        perturbations=[
            SmallMoleculePerturbation(
                compound=compound,
                concentration=Concentration(
                    value=mm, unit=ConcentrationUnit.millimolar
                ),
            )
        ],
        provenance_gaps=[TEMPERATURE_GAP],
    )


def _experiment(
    orf: str, npv: float, env: Environment, readout: str
) -> EnvironmentResponseExperiment:
    return EnvironmentResponseExperiment(
        dataset_name="YeastPhenomeDataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                )
            ]
        ),
        environment=env,
        phenotype=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            environment_response=npv,
            units=f"{NPV_UNITS} [readout: {readout}]",
            provenance_gaps=_phenotype_gaps(),
        ),
    )


def _reference(
    env: Environment, readout: str, strain: str, ploidy: Literal["haploid", "diploid"]
) -> EnvironmentResponseExperimentReference:
    return EnvironmentResponseExperimentReference(
        dataset_name="YeastPhenomeDataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=strain, ploidy=ploidy
        ),
        environment_reference=env,
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            environment_response=0.0,
            units=f"{NPV_UNITS} [readout: {readout}]",
        ),
    )


def _write_screen(
    raw: Path, screen: dict[str, str], header: list[str], rows: list[list[str]]
) -> None:
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    (raw / f"{screen['pmid']}_{screen['stem']}_valuez.txt").write_text(
        "\n".join(lines) + "\n"
    )


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[YeastPhenomeDataset, Path]:
    monkeypatch.setattr(yeastphenome, "SCREENS", [SCREEN_A, SCREEN_B])
    root = tmp_path / "yeastphenome"
    raw = root / "raw"
    raw.mkdir(parents=True)
    _write_screen(raw, SCREEN_A, HEADER_A, ROWS_A)
    _write_screen(raw, SCREEN_B, HEADER_B, ROWS_B)
    return YeastPhenomeDataset(root=str(root)), root


# --------------------------------------------------------------------------- #
# Column grammar helpers
# --------------------------------------------------------------------------- #


def test_column_meta_five_and_four_fields_and_malformed() -> None:
    """Media is present only with 5 fields; the author is always last; <4 fields -> None."""
    assert _column_meta(HEADER_A[1]) == {
        "zygosity": "hom",
        "phenotype": "growth (colony size)",
        "condition": "furfural [10 mM]",
        "media": "YPD",
    }
    assert _column_meta(
        "hom | growth (colony size) | furfural [10 mM] | synthetic"
    ) == {
        "zygosity": "hom",
        "phenotype": "growth (colony size)",
        "condition": "furfural [10 mM]",
        "media": None,
    }
    assert _column_meta("hom | growth | x") is None


@pytest.mark.parametrize(
    ("phenotype", "expected"),
    [
        ("growth (colony size)", "colony size"),
        ("growth (culture turbidity)", "culture turbidity"),
        ("growth", "growth"),
        ("  growth (pooled barseq)  ", "pooled barseq"),
    ],
)
def test_readout_method(phenotype: str, expected: str) -> None:
    """The parenthesized modality is the readout; without one the stripped string is."""
    assert _readout_method(phenotype) == expected


def test_parse_media_rich_synthetic_state_and_unknown() -> None:
    """Base = token before '+'; rich vs synthetic from the two tables; state from readout.

    ``colony`` / ``spot`` / ``killing`` in the phenotype -> solid, otherwise liquid;
    ``LB`` is in neither table -> None; a None field -> None.
    """
    assert _parse_media("YPD", "growth (colony size)") == Media(
        name="YPD", state="solid", is_synthetic=False
    )
    assert _parse_media("SC + EtOH", "growth (pooled barseq)") == Media(
        name="SC", state="liquid", is_synthetic=True
    )
    assert _parse_media("YPD", "growth (spot assay)") == Media(
        name="YPD", state="solid", is_synthetic=False
    )
    assert _parse_media("SD", "growth (killing zone)") == Media(
        name="SD", state="solid", is_synthetic=True
    )
    assert _parse_media("LB", "growth (colony size)") is None
    assert _parse_media(None, "growth (colony size)") is None


def test_parse_condition_small_molecule_resolved_and_unresolved() -> None:
    """``furfural [10 mM]`` resolves through the pinned table; ``quinine [2 mM]`` does not.

    Finding: quinine is absent from the compound table, so its ``Compound`` carries an
    ``inchikey`` gap with reason ``deferred_pending_source_review`` and no identifiers.
    Whether quinine is the condition of a real pinned screen is not verified here.
    """
    assert _parse_condition("furfural [10 mM]") == SmallMoleculePerturbation(
        compound=FURFURAL,
        concentration=Concentration(value=10.0, unit=ConcentrationUnit.millimolar),
    )
    assert _parse_condition("quinine [2 mM]") == SmallMoleculePerturbation(
        compound=QUININE_UNRESOLVED,
        concentration=Concentration(value=2.0, unit=ConcentrationUnit.millimolar),
    )


def test_parse_condition_defensin_and_unit_map() -> None:
    """A defensin name routes to a peptide biologic; every mapped unit spelling resolves."""
    assert _parse_condition("NaD1 [0.5 uM]") == BiologicPerturbation(
        agent_class=BiologicAgentClass.peptide,
        name="NaD1",
        concentration=Concentration(value=0.5, unit=ConcentrationUnit.micromolar),
    )
    units = {
        "M": ConcentrationUnit.molar,
        "mM": ConcentrationUnit.millimolar,
        "uM": ConcentrationUnit.micromolar,
        "µM": ConcentrationUnit.micromolar,
        "nM": ConcentrationUnit.nanomolar,
        "ug/ml": ConcentrationUnit.ug_per_ml,
        "ug/mL": ConcentrationUnit.ug_per_ml,
        "g/L": ConcentrationUnit.g_per_l,
        "g/l": ConcentrationUnit.g_per_l,
    }
    for spelling, unit in units.items():
        pert = _parse_condition(f"furfural [3 {spelling}]")
        assert isinstance(pert, SmallMoleculePerturbation)
        assert pert.concentration == Concentration(value=3.0, unit=unit)


def test_parse_condition_rejects_unencodable() -> None:
    """Unknown unit, multi-component, dose range, undosed, and nameless conditions -> None.

    Six shapes, in order: unit ``C`` is not in ``_UNIT_MAP``; a comma-joined second
    component fails the anchored regex; a dose range is not ``[0-9.]+``; no bracket at
    all; a ``+``-joined second dosed agent leaves text after the first ``]``; and an
    empty name fails the ``+?`` (one-or-more) name group.
    """
    conditions = [
        "heat [37 C]",
        "time [5 gen], furfural [1 uM]",
        "furfural [1-10 mM]",
        "furfural",
        "furfural [10 mM] + acetic acid [5 mM]",
        "[10 mM]",
    ]
    assert [_parse_condition(c) for c in conditions] == [None] * 6


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def test_len_gene_set_and_manifest(dataset: tuple[YeastPhenomeDataset, Path]) -> None:
    """Six records over two screens; the gene set is the three ORF-shaped row ids.

    ``not_an_orf`` rows are dropped; ``Q0250`` passes the Q-plus-four-digits alternative. No
    ``preprocess/data.csv`` is written by this loader (``df`` is None).
    """
    ds, root = dataset
    assert len(ds) == 6
    pre = root / "preprocess"
    assert json.loads((pre / "gene_set.json").read_text()) == [
        "Q0250",
        "YAL001C",
        "YBR002C",
    ]
    assert not (pre / "data.csv").exists()
    assert ds.df is None
    manifest = json.loads((pre / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "yeastphenome"
    assert manifest["loader_class"] == "YeastPhenomeDataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.yeastphenome"
    assert manifest["hostname"] == socket.gethostname()
    assert set(manifest["closure"]) >= {
        "EnvironmentResponseExperiment",
        "EnvironmentResponsePhenotype",
        "SmallMoleculePerturbation",
        "BiologicPerturbation",
    }


def test_hom_colony_size_records(dataset: tuple[YeastPhenomeDataset, Path]) -> None:
    """Records 0-2 are screen A column 1: YPD solid, furfural 10 mM, BY4743 diploid.

    NPVs 1.5, 0.5, -0.75 for YAL001C, YBR002C, Q0250. Temperature is None with a
    ``not_carried_by_curation`` gap anchored to ``CURATION``; the phenotype carries the
    three curation gaps; the reference NPV is 0.0 in the same environment.
    """
    ds, _ = dataset
    env = _environment(
        Media(name="YPD", state="solid", is_synthetic=False), FURFURAL, 10.0
    )
    for idx, orf, npv in [
        (0, "YAL001C", 1.5),
        (1, "YBR002C", 0.5),
        (2, "Q0250", -0.75),
    ]:
        record = ds[idx]
        assert (
            record["experiment"]
            == _experiment(orf, npv, env, "colony size").model_dump()
        )
        assert record["reference"] == (
            _reference(env, "colony size", "BY4743", "diploid").model_dump()
        )
        assert (
            record["publication"]
            == Publication(
                pubmed_id="99999901",
                pubmed_url="https://pubmed.ncbi.nlm.nih.gov/99999901/",
            ).model_dump()
        )
    assert (
        EnvironmentResponseExperiment.model_validate(ds[0]["experiment"]).model_dump()
        == ds[0]["experiment"]
    )


def test_hap_a_turbidity_records_skip_blank_cell(
    dataset: tuple[YeastPhenomeDataset, Path],
) -> None:
    """Records 3-4 are screen A column 2: SD liquid, unresolved quinine 2 mM, BY4741 haploid.

    YBR002C's blank cell yields no record, so the column contributes YAL001C (-2.25) and
    Q0250 (0.125) only. ``culture turbidity`` -> liquid state and the readout in units.
    """
    ds, _ = dataset
    env = _environment(
        Media(name="SD", state="liquid", is_synthetic=True), QUININE_UNRESOLVED, 2.0
    )
    for idx, orf, npv in [(3, "YAL001C", -2.25), (4, "Q0250", 0.125)]:
        record = ds[idx]
        assert record["experiment"] == (
            _experiment(orf, npv, env, "culture turbidity").model_dump()
        )
        assert record["reference"] == (
            _reference(env, "culture turbidity", "BY4741", "haploid").model_dump()
        )
        assert record["publication"]["pubmed_id"] == "99999901"


def test_second_screen_hap_alpha_record_and_publication(
    dataset: tuple[YeastPhenomeDataset, Path],
) -> None:
    """Record 5 is screen B: SC (from ``SC + EtOH``) liquid, BY4742 haploid, PMID 99999902."""
    ds, _ = dataset
    env = _environment(
        Media(name="SC", state="liquid", is_synthetic=True), FURFURAL, 10.0
    )
    record = ds[5]
    assert record["experiment"] == (
        _experiment("YAL001C", 2.0, env, "pooled barseq").model_dump()
    )
    assert record["reference"] == (
        _reference(env, "pooled barseq", "BY4742", "haploid").model_dump()
    )
    assert (
        record["publication"]
        == Publication(
            pubmed_id="99999902", pubmed_url="https://pubmed.ncbi.nlm.nih.gov/99999902/"
        ).model_dump()
    )


def test_reference_index_groups_by_environment(
    dataset: tuple[YeastPhenomeDataset, Path],
) -> None:
    """Three references: [0, 1, 2] (hom furfural), [3, 4] (hap a quinine), [5] (hap alpha)."""
    ds, root = dataset
    index = json.loads(
        (root / "preprocess" / "experiment_reference_index.json").read_text()
    )
    assert [e["member_indices"] for e in index] == [[0, 1, 2], [3, 4], [5]]
    assert [e["reference"]["genome_reference"]["strain"] for e in index] == [
        "BY4743",
        "BY4741",
        "BY4742",
    ]
    loaded = ds.experiment_reference_index
    assert loaded is not None
    assert [eri.member_indices for eri in loaded] == [[0, 1, 2], [3, 4], [5]]


def test_dropped_columns_are_logged_by_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The final log line counts every drop reason exactly once each for screen A.

    ``non_orf_row`` comes first in the Counter because column 1's rows are walked
    before column 3's het drop; then het (1), heat unit C (1), 4-field header + LB (2),
    expression (1), 2-field header (1). The two ``not_an_orf`` cells in the two kept
    columns give ``non_orf_row`` 2. Screen A alone is built here so the counts are the
    header's.
    """
    monkeypatch.setattr(yeastphenome, "SCREENS", [SCREEN_A])
    root = tmp_path / "yeastphenome"
    (root / "raw").mkdir(parents=True)
    _write_screen(root / "raw", SCREEN_A, HEADER_A, ROWS_A)
    with caplog.at_level("INFO", logger="torchcell.datasets.scerevisiae.yeastphenome"):
        ds = YeastPhenomeDataset(root=str(root))
    assert len(ds) == 5
    summary = [r for r in caplog.records if r.getMessage().startswith("Wrote ")]
    assert len(summary) == 1
    assert summary[0].getMessage() == (
        "Wrote 5 YeastPhenome records over 2 environments (1 screens); dropped "
        "columns/rows: {'non_orf_row': 2, 'het_or_ambiguous_zygosity': 1, "
        "'condition_unparseable': 1, 'media_absent_or_unknown': 2, 'not_growth': 1, "
        "'malformed_header': 1}"
    )
    drops = [
        r.getMessage() for r in caplog.records if "drop condition" in r.getMessage()
    ]
    assert drops == ["YeastPhenome synthetic_a col 4: drop condition 'heat [37 C]'"]


def test_raw_file_names_follow_patched_screens(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raw_file_names`` is derived from the module-level ``SCREENS`` at call time."""
    monkeypatch.setattr(yeastphenome, "SCREENS", [SCREEN_A, SCREEN_B])
    prop = vars(YeastPhenomeDataset)["raw_file_names"]
    names = prop.fget(object.__new__(YeastPhenomeDataset))
    assert names == [
        "99999901_synthetic_a_valuez.txt",
        "99999902_synthetic_b_valuez.txt",
    ]
