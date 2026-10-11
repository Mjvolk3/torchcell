# tests/torchcell/datasets/scerevisiae/test_boocock2025.py
# [[tests.torchcell.datasets.scerevisiae.test_boocock2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_boocock2025.py
"""Boocock 2025 loader built end to end on a synthetic two-panel release.

The release, its expected records and why each value is what it is are in
``_boocock2025_release.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.torchcell.datasets.scerevisiae._boocock2025_release import (
    _BY,
    _D_MARKERS,
    _LOOKUP,
    _PANELS,
    _RELEASE,
    _RM,
    _S288C_SHA,
    _SI1_ROWS,
    _TAR_SHA,
    _YJM,
    _YPS,
    _build,
    _genome,
    _gsub,
    _patch_release,
    _raw_records,
    _write_genomes_tier,
    _write_mirror_manifest,
    _write_raw,
    _write_si1,
)
from torchcell.datamodels.media import SM, YNB_SHEATH_FLUID_BOOCOCK2025
from torchcell.datamodels.schema import (
    ArtifactRef,
    HaplotypeBlock,
    SegregantExpressionExperiment,
    SegregantExpressionExperimentReference,
)
from torchcell.datasets.scerevisiae import boocock2025 as m
from torchcell.datasets.scerevisiae.bloom2019 import ParentAssemblyRefs, parse_marker


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> m.Boocock2025Dataset:
    return _build(tmp_path, monkeypatch)


# --------------------------------------------------------------------------- #
def _exp(dataset: m.Boocock2025Dataset, i: int) -> SegregantExpressionExperiment:
    return SegregantExpressionExperiment.model_validate(dataset[i]["experiment"])


def _ref(
    dataset: m.Boocock2025Dataset, i: int
) -> SegregantExpressionExperimentReference:
    return SegregantExpressionExperimentReference.model_validate(
        dataset[i]["reference"]
    )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def test_four_records_pooled_segregants_then_de_novo_cells(
    dataset: m.Boocock2025Dataset,
) -> None:
    assert len(dataset) == 4
    ids = [dataset[i]["experiment"]["genotype"]["segregant_id"] for i in range(4)]
    assert ids == ["S1", "S2", "LD:TTTA-1", "LD:TTTC-1"]
    assert {_exp(dataset, i).experiment_type for i in range(4)} == {
        "segregant_expression"
    }
    assert {_ref(dataset, i).experiment_reference_type for i in range(4)} == {
        "segregant_expression"
    }


def test_pooled_segregant_sums_its_matched_cells(dataset: m.Boocock2025Dataset) -> None:
    s1 = _exp(dataset, 0).phenotype
    s2 = _exp(dataset, 1).phenotype
    assert dict(s1.umi_count) == {"YAL002W": 3}
    assert (s1.total_umis, s1.n_cells) == (4, 1)
    assert (s1.cell_cycle_stage_counts, s1.library_cell_counts) == (
        {"S": 1},
        {"LP1": 1},
    )
    assert dict(s2.umi_count) == {"YAL001C": 3, "YAL002W": 1, "YBL001C": 1}
    assert (s2.total_umis, s2.n_cells) == (8, 2)
    assert s2.cell_cycle_stage_counts == {"G1": 2}
    assert s2.library_cell_counts == {"LP1": 1, "LP2": 1}
    assert s2.n_features == 3
    assert (
        s2.feature_universe_sha256
        == hashlib.sha256(b"YAL001C\nYAL002W\nYBL001C").hexdigest()
    )


def test_pooled_genotype_is_the_lookup_hard_call_with_posterior_one(
    dataset: m.Boocock2025Dataset,
) -> None:
    s2 = _exp(dataset, 1).genotype
    assert [
        (b.chromosome, b.start, b.end, b.parent, b.posterior, b.n_markers)
        for b in s2.blocks
    ] == [
        ("chrI", 100, 100, 2, 1.0, 1),
        ("chrI", 300, 300, 1, 1.0, 1),
        ("chrIII", 150, 150, 1, 1.0, 1),
    ]
    s1 = _exp(dataset, 0).genotype
    assert [(b.chromosome, b.start, b.end, b.parent) for b in s1.blocks] == [
        ("chrI", 100, 300, 1),
        ("chrIII", 150, 150, 1),
    ]
    assert s1.call_method == m.POOLED_CALL_METHOD
    assert (
        s1.marker_matrix_sha256
        == hashlib.sha256(f"placeholder {m.GDATA_NAME}".encode()).hexdigest()
    )
    assert (s1.cross, s1.parent_1.name, s1.parent_2.name) == ("A", "BY", "RM")
    assert s1.parent_1.peter_strain_id is None
    assert s1.parent_1.assembly_ref.key == "sgd_S288C_R64-4-1_20230830"
    assert s1.parent_2.peter_strain_id == "AAA"
    assert s1.parent_2.assembly_ref.member == "GENOMES_ASSEMBLED/AAA_6.re.fa"
    assert s1.parent_2.engineered_background == _RM


def test_de_novo_cell_blocks_carry_the_posterior_on_the_grid(
    dataset: m.Boocock2025Dataset,
) -> None:
    d1 = _exp(dataset, 2).genotype
    assert [
        (b.start, b.end, b.parent, b.posterior, b.n_markers) for b in d1.blocks
    ] == [
        (100, 100, 2, 1.0, 1),
        (200, 200, 2, 0.95, 1),
        (300, 300, 1, 0.8, 1),
        (150, 150, 2, 0.9, 1),
    ]
    d2 = _exp(dataset, 3).genotype
    assert [
        (b.start, b.end, b.parent, b.posterior, b.n_markers) for b in d2.blocks
    ] == [(100, 200, 1, 1.0, 2), (300, 300, 2, 0.6, 1), (150, 150, 1, 0.7, 1)]
    assert (d1.cross, d1.parent_1.name, d1.parent_2.name) == ("B", "YJM145", "YPS163")
    assert d1.parent_1.assembly_ref.member == "GENOMES_ASSEMBLED/AAS_3.re.fa"
    assert d1.parent_2.engineered_background == _YPS
    assert d1.call_method == m.DE_NOVO_CALL_METHOD
    assert (
        d1.marker_matrix_sha256
        == hashlib.sha256(b"placeholder segData_D.RDS").hexdigest()
    )
    phenotype = _exp(dataset, 2).phenotype
    assert dict(phenotype.umi_count) == {"YAL001C": 4}
    assert (phenotype.total_umis, phenotype.n_cells) == (6, 1)
    assert phenotype.cell_cycle_stage_counts == {"G2/M": 1}


def test_environments_are_minimal_ynb_and_the_sheath_fluid_medium(
    dataset: m.Boocock2025Dataset,
) -> None:
    pooled = _exp(dataset, 0).environment
    de_novo = _exp(dataset, 2).environment
    assert pooled.media.components == SM.components
    assert pooled.media.provenance[-1] == m.SOURCED["pilot_medium"]
    assert de_novo.media == YNB_SHEATH_FLUID_BOOCOCK2025
    assert pooled.temperature is not None and pooled.temperature.value == 30.0
    assert de_novo.aerobicity == "aerobic"


def test_reference_is_the_panel_pseudobulk(dataset: m.Boocock2025Dataset) -> None:
    pooled = _ref(dataset, 0).phenotype_reference
    assert dict(pooled.umi_count) == {"YAL001C": 3, "YAL002W": 4, "YBL001C": 1}
    assert (pooled.total_umis, pooled.n_cells) == (12, 3)
    assert pooled.cell_cycle_stage_counts == {"G1": 2, "S": 1}
    de_novo = _ref(dataset, 3).phenotype_reference
    assert dict(de_novo.umi_count) == {"YAL001C": 5, "YCL001W": 2}
    assert (de_novo.total_umis, de_novo.n_cells) == (9, 2)
    index = dataset.experiment_reference_index
    assert index is not None
    assert [list(e.member_indices) for e in index] == [[0, 1], [2, 3]]


def test_side_files_and_gene_set(dataset: m.Boocock2025Dataset) -> None:
    preprocess = Path(dataset.preprocess_dir)
    assert pd.read_csv(preprocess / "data.csv").to_dict("list") == {
        "panel": ["P", "P", "D", "D"],
        "segregant_id": ["S1", "S2", "LD:TTTA-1", "LD:TTTC-1"],
    }
    assert pd.read_csv(preprocess / "block_counts.csv").to_dict("list") == {
        "panel": ["P", "P", "D", "D"],
        "n_blocks": [2, 3, 4, 3],
    }
    assert sorted(dataset.gene_set) == ["YAL001C", "YCL001W"]
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YCL001W",
    ]


def test_genotype_is_stored_inline_and_the_reference_interned(
    dataset: m.Boocock2025Dataset,
) -> None:
    raw = _raw_records(dataset.root)
    assert len(raw) == 4
    assert "$ref" not in raw[0]["experiment"]["genotype"]
    assert set(raw[0]["reference"]) == {"$ref", "name"}
    assert raw[0]["reference"]["$ref"] == raw[1]["reference"]["$ref"]
    assert raw[2]["reference"]["$ref"] != raw[0]["reference"]["$ref"]


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_record_count_oracle_and_release_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_release(monkeypatch)
    spec = _PANELS["D"]
    release = m.read_panel_release(tmp_path / "segData_D.RDS")
    m.Boocock2025Dataset._check_release(spec, release)
    with pytest.raises(ValueError, match="D: 2 cells, expected 3"):
        m.Boocock2025Dataset._check_release(
            spec.model_copy(update={"expected_cells": 3}), release
        )
    with pytest.raises(ValueError, match="D: 2 transcripts, expected 9"):
        m.Boocock2025Dataset._check_release(
            spec.model_copy(update={"expected_transcripts": 9}), release
        )
    with pytest.raises(ValueError, match=r"D: libraries \['LD'\] != \['LX'\]"):
        m.Boocock2025Dataset._check_release(
            spec.model_copy(update={"libraries": ("LX",)}), release
        )


def test_the_build_refuses_a_wrong_record_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_release(monkeypatch)
    monkeypatch.setattr(m, "EXPECTED_RECORDS", 5)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    root = tmp_path / "boocock2025"
    _write_raw(root / "raw", _SI1_ROWS)
    _write_mirror_manifest(data_root, root / "raw")
    _write_genomes_tier(data_root)
    with pytest.raises(ValueError, match="wrote 4 records, expected 5"):
        m.Boocock2025Dataset(root=str(root), genome=_genome())


def test_table_s1_must_name_each_parent_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_release(monkeypatch)
    path = tmp_path / "si1.xlsx"
    _write_si1(path, [("YLKD", _YPS, _YPS, "B"), ("YLKP", _BY, _RM, "A")])
    with pytest.raises(ValueError, match="Table S1 YLKD: no unique cell for YJM145"):
        m.read_parent_backgrounds(path)
    _write_si1(path, [("YLKP", _BY, _RM, "A")])
    with pytest.raises(
        ValueError, match=r"Table S1 lacks the panel strains \['YLKD'\]"
    ):
        m.read_parent_backgrounds(path)


def test_release_tables_must_align_and_markers_sort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_release(monkeypatch)
    good = _RELEASE["segData_D.RDS"]
    shuffled = dict(good)
    shuffled["barcode.features"] = (
        good["barcode.features"].iloc[::-1].reset_index(drop=True)
    )
    monkeypatch.setitem(_RELEASE, "segData_D.RDS", shuffled)
    with pytest.raises(ValueError, match="rows differ"):
        m.read_panel_release(tmp_path / "segData_D.RDS")
    unsorted = dict(good)
    unsorted["Gsub"] = _gsub(
        [[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]],
        ["TTTA-1", "TTTC-1"],
        ["chrI_200_C_G_1", "chrI_100_A_T_2", "chrI_300_G_C_3", "chrIII_150_A_G_4"],
    )
    monkeypatch.setitem(_RELEASE, "segData_D.RDS", unsorted)
    with pytest.raises(ValueError, match="not in \\(chromosome, position\\) order"):
        m.read_panel_release(tmp_path / "segData_D.RDS")
    fractional = dict(good)
    fractional["Yr"] = SimpleNamespace(**{**vars(good["Yr"]), "x": good["Yr"].x + 0.5})
    monkeypatch.setitem(_RELEASE, "segData_D.RDS", fractional)
    with pytest.raises(ValueError, match="Yr: non-integral counts"):
        m.read_panel_release(tmp_path / "segData_D.RDS")


def test_lookup_genotypes_parse_sort_and_refuse_unknown_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_release(monkeypatch)
    ids, markers, calls = m.read_lookup_genotypes(tmp_path / m.GDATA_NAME)
    assert ids == ["S0", "S1", "S2"]
    assert [(x.chromosome, x.position, x.ref, x.alt) for x in markers] == [
        ("chrI", 100, "A", "T"),
        ("chrI", 300, "G", "C"),
        ("chrIII", 150, "A", "G"),
    ]
    assert calls.tolist() == [[1, 1, 1], [-1, -1, -1], [1, -1, -1]]
    bad = {"gdata": _LOOKUP["gdata"].copy(data=[[0.0, 1, 1], [1, 1, 1], [1, 1, 1]])}
    monkeypatch.setattr("rdata.read_rda", lambda path: bad)
    with pytest.raises(ValueError, match="outside"):
        m.read_lookup_genotypes(tmp_path / m.GDATA_NAME)
    named = {
        "gdata": _LOOKUP["gdata"].assign_coords(
            dim_1=["chrI-100", "chrI:300_G/C", "chrIII:150_A/G"]
        )
    }
    monkeypatch.setattr("rdata.read_rda", lambda path: named)
    with pytest.raises(ValueError, match="'chrI-100' is not chr:pos_ref/alt"):
        m.read_lookup_genotypes(tmp_path / m.GDATA_NAME)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_posterior_blocks_round_trip_within_half_a_step() -> None:
    markers = [parse_marker(name, i) for i, name in enumerate(_D_MARKERS)]
    posterior = np.asarray([0.524, 0.976, 0.474, 0.026])
    blocks = m.encode_posterior_blocks(posterior, markers)
    assert [(b.parent, b.posterior) for b in blocks] == [
        (2, 0.5),
        (2, 1.0),
        (1, 0.55),
        (1, 0.95),
    ]
    parent, level = m.expand_posterior_blocks(blocks, markers)
    assert parent.tolist() == [2, 2, 1, 1]
    q = np.where(parent == 2, posterior, 1 - posterior)
    assert float(np.max(np.abs(q - level))) <= m.POSTERIOR_STEP / 2


def test_a_posterior_of_exactly_one_half_has_no_called_parent() -> None:
    markers = [parse_marker(name, i) for i, name in enumerate(_D_MARKERS)]
    with pytest.raises(ValueError, match="exactly 0.5 has no called parent"):
        m.encode_posterior_blocks(np.asarray([0.9, 0.5, 0.1, 0.2]), markers)


def test_expand_posterior_blocks_refuses_misaligned_blocks() -> None:
    markers = [parse_marker(name, i) for i, name in enumerate(_D_MARKERS)]
    shifted = [
        HaplotypeBlock(chromosome="chrI", start=200, end=300, parent=1, n_markers=2)
    ]
    with pytest.raises(ValueError, match="does not align with the markers at 0"):
        m.expand_posterior_blocks(shifted, markers)
    short = [
        HaplotypeBlock(chromosome="chrI", start=100, end=200, parent=1, n_markers=2)
    ]
    with pytest.raises(ValueError, match="blocks cover 2 of 4 markers"):
        m.expand_posterior_blocks(short, markers)


def test_feature_universe_hash_ignores_input_order() -> None:
    assert m.feature_universe_sha256(["YB", "YA"]) == m.feature_universe_sha256(
        ["YA", "YB"]
    )
    assert (
        m.feature_universe_sha256(["YA", "YB"]) == hashlib.sha256(b"YA\nYB").hexdigest()
    )


def test_build_parent_refuses_a_peter_id_missing_from_the_index() -> None:
    assemblies = ParentAssemblyRefs(
        index={"AAA": "GENOMES_ASSEMBLED/AAA_6.re.fa"},
        tarball=ArtifactRef(
            tier="genomes",
            key="peter2018_1011_assemblies",
            path="1011Assemblies.tar.gz",
            sha256=_TAR_SHA,
            bytes=1,
        ),
        s288c=ArtifactRef(
            tier="genomes",
            key="sgd_S288C_R64-4-1_20230830",
            path="S288C_reference_sequence_R64-4-1_20230830.fsa",
            sha256=_S288C_SHA,
            bytes=2,
        ),
    )
    assert m.build_parent("RM", _RM, assemblies).peter_strain_id == "AAA"
    with pytest.raises(RuntimeError, match="parent YJM145: Peter id AAS is not"):
        m.build_parent("YJM145", _YJM, assemblies)


def test_record_builders_refuse_what_the_store_cannot_hold() -> None:
    with pytest.raises(NotImplementedError, match="carries no gene perturbations"):
        m.Boocock2025Dataset.extract_systematic_gene_names({})
    probe = m.Boocock2025Dataset.__new__(m.Boocock2025Dataset)
    with pytest.raises(NotImplementedError, match="builds its records in process"):
        probe.create_experiment()
    frame = pd.DataFrame({"a": [1]})
    assert probe.preprocess_raw(frame) is frame
    probe.genome = None
    with pytest.raises(RuntimeError, match="requires a genome"):
        probe.compute_gene_set()


def test_every_module_quote_is_verbatim_in_the_mirrored_paper() -> None:
    """The quotes are substrings of the bytes ``PAPER_MD_SHA256`` pins (skipped when
    the library mirror is not mounted).
    """
    import os

    path = Path(os.environ["DATA_ROOT"]) / m.LIBRARY_DIR_REL / "paper.md"
    if not path.exists():
        pytest.skip("library mirror not mounted")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == m.PAPER_MD_SHA256
    text = path.read_text(encoding="utf-8")
    missing = [key for key, sv in m.SOURCED.items() if sv.quote not in text]
    assert missing == []
    assert len(m.sourced_values()) == len(m.SOURCED) + len(
        [*m.PILOT_MEDIUM.provenance, *YNB_SHEATH_FLUID_BOOCOCK2025.provenance]
    ) + sum(
        len(c.provenance)
        for media in (m.PILOT_MEDIUM, YNB_SHEATH_FLUID_BOOCOCK2025)
        for c in media.components
    )
