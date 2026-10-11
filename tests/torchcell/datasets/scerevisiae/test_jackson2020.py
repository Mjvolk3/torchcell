# tests/torchcell/datasets/scerevisiae/test_jackson2020.py
# [[tests.torchcell.datasets.scerevisiae.test_jackson2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_jackson2020.py
"""Jackson 2020 loader: every reader and refusal on a miniature release, the full build
on it, and (``data``-marked) the real mirrors and the built dev store.

The miniature (``_jackson2020_release.py``) has four strains, nine count columns in the
real header's R-mangled spellings and sixteen cells, so every expected number below is
countable by hand from that module.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import os.path as osp
import re
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tests.torchcell.datasets.scerevisiae import _jackson2020_release as rel
from torchcell.datamodels.schema import (
    ConcentrationUnit,
    EndpointRule,
    PreCultureSource,
    PseudobulkUmiCountExperiment,
    PseudobulkUmiCountExperimentReference,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import jackson2020 as j
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.verification.sourced import ProvenanceGapReason


# --------------------------------------------------------------------------- #
# Small readers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("released", "source"),
    [
        ("YDL247W.A", "YDL247W-A"),
        ("X15S_rRNA", "15S_rRNA"),
        ("snR7.L", "snR7-L"),
        ("RDN25.1", "RDN25-1"),
        ("YAL001C", "YAL001C"),
        ("X", "X"),
        ("XBP1", "XBP1"),
    ],
)
def test_unmangle_undoes_make_names(released: str, source: str) -> None:
    assert j.unmangle(released) == source


def test_render_row_drops_empty_cells_and_keeps_openpyxl_str(tmp_path: Path) -> None:
    assert j.render_row((1.0, "FY4", "MATa", None)) == "1.0 | FY4 | MATa"
    rel.write_workbook(tmp_path / "w.xlsx", ["only row"])
    text = j.sheet_text(tmp_path / "w.xlsx")
    first, second = text.split("\n")
    assert first.startswith(
        "Strain ID (DGY) | Strain | Genotype | Parent | Barcode | Plate Location | "
        "Reference / 1 | FY4 | MATa / 2 | FY5 | MATα / "
    )
    assert second == "only row"


def test_read_strain_table_joins_each_diploid_to_its_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rel.install(tmp_path, monkeypatch)
    specs = j.read_strain_table(root / "raw" / j.SUPP1_NAME)
    assert [s.strain for s in specs] == list(rel.STRAINS)
    by_strain = {s.strain: s for s in specs}
    wt = by_strain["WT(ho)_A1"]
    assert (wt.gene_token, wt.well, wt.is_wild_type) == ("ho", "A1", True)
    assert (wt.parent_dgy, wt.parent_row, wt.diploid_row, wt.natmx_parent_row) == (
        1997,
        4,
        11,
        8,
    )
    assert wt.diploid_row_quote == (
        "N/A | WT(ho)_A1 | MATa/MATα Δho::KanMX/Δho::NatMX | AAAAGTTTGAAAGTTTGAA | "
        "BC_DEL_1_A1 | This work"
    )
    stp2 = by_strain["stp2_A12"]
    assert stp2.diploid_cell_disagrees is True
    assert stp2.parent_genotype == "MATa Δstp2::KanMX"
    assert [s.strain for s in specs if s.diploid_cell_disagrees] == ["stp2_A12"]


def _rewrite_s2(path: Path, row: int, column: int, value: Any) -> None:
    import openpyxl

    workbook = openpyxl.load_workbook(path)
    workbook[j.TABLE_S2_SHEET].cell(row=row, column=column, value=value)
    workbook.save(path)


@pytest.mark.parametrize(
    ("row", "column", "value", "message"),
    [
        (1, 1, "Strain", "unexpected Table S2 header"),
        (
            4,
            3,
            "MATa Δgat1::KanMX",
            "MATa parent genotype 'MATa Δgat1::KanMX' does not delete ho",
        ),
        (11, 5, "GATTACA", "WT(ho)_A1: diploid and MATa barcodes differ"),
        (11, 6, "BC_DEL_1_B7", "WT(ho)_A1: plate location 'BC_DEL_1_B7' is not A1"),
        (9, 2, "gat9", "no FY5 NatMX parent for gat1"),
        (
            14,
            3,
            "MATa/MATα Δstp2::KanMX/Δstp2::NatMX",
            "diploid Genotype cells naming another gene than their label: [], pinned "
            "['stp2_A12']",
        ),
    ],
)
def test_read_strain_table_refuses_an_inconsistent_sheet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row: int,
    column: int,
    value: Any,
    message: str,
) -> None:
    root = rel.install(tmp_path, monkeypatch)
    path = root / "raw" / j.SUPP1_NAME
    _rewrite_s2(path, row, column, value)
    with pytest.raises(j.StrainTableError) as caught:
        j.read_strain_table(path)
    assert message in str(caught.value)


def test_read_strain_table_refuses_a_short_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rel.install(tmp_path, monkeypatch)
    monkeypatch.setattr(j, "N_STRAINS", 5)
    with pytest.raises(j.StrainTableError, match="lists 4 diploid and 4 MATa strains"):
        j.read_strain_table(root / "raw" / j.SUPP1_NAME)


def test_read_barcode_fasta_and_its_refusal(tmp_path: Path) -> None:
    path = tmp_path / "bc.fasta"
    path.write_text(">a_A1\nACGT\n\n>b_B2\nTTTT\n")
    assert j.read_barcode_fasta(path) == {"a_A1": "ACGT", "b_B2": "TTTT"}
    path.write_text(">a_A1\nACGT\nGGGG\n")
    with pytest.raises(
        ValueError, match="sequence line without a fresh header: 'GGGG'"
    ):
        j.read_barcode_fasta(path)


def test_read_author_settings_parses_both_maps(tmp_path: Path) -> None:
    path = tmp_path / "s.R"
    path.write_text(rel.SETTINGS_TEXT)
    settings = j.read_author_settings(path)
    assert settings.condition_label["AmmoniumSulfate"] == "NLIM-NH4"
    assert settings.condition_label["YPDDiauxic"] == "DIAUXY"
    assert len(settings.condition_label) == 11
    assert settings.tf_systematic_name == {"gat1": "YFL021W", "stp2": "YHR006W"}
    assert settings.tf_lines["gat1"] == 'nitrogen.tfs <<- c("YFL021W", # gat1'
    assert settings.tf_lines["stp2"] == '"YHR006W"  # stp2'
    assert settings.condition_levels_line.startswith('CONDITION.LEVELS <<- c("CStarve"')
    path.write_text(rel.SETTINGS_TEXT + '"YFL021W", # gat1\n')
    with pytest.raises(ValueError, match="gat1 is listed twice"):
        j.read_author_settings(path)
    path.write_text(rel.SETTINGS_TEXT.replace('"YPD")\n', '"YPD", "X")\n', 1))
    with pytest.raises(ValueError, match="12 levels and 11 labels"):
        j.read_author_settings(path)


# --------------------------------------------------------------------------- #
# Conditions and environments
# --------------------------------------------------------------------------- #
def _settings(tmp_path: Path) -> j.AuthorSettings:
    path = tmp_path / "settings.R"
    path.write_text(rel.SETTINGS_TEXT)
    return j.read_author_settings(path)


def test_build_conditions_follows_table_1_and_the_authors_labels(
    tmp_path: Path,
) -> None:
    conditions = j.build_conditions(_settings(tmp_path))
    assert list(conditions) == [
        "YPD",
        "DIAUXY",
        "RAPA",
        "YPEtOH",
        "MMD",
        "MMEtOH",
        "NLIM-GLN",
        "NLIM-PRO",
        "NLIM-NH4",
        "NLIM-UREA",
        "CSTARVE",
    ]
    assert {a: c.release_label for a, c in conditions.items()} == {
        "YPD": "YPD",
        "DIAUXY": "YPDDiauxic",
        "RAPA": "YPDRapa",
        "YPEtOH": "YPEtOH",
        "MMD": "MinimalGlucose",
        "MMEtOH": "MinimalEtOH",
        "NLIM-GLN": "Glutamine",
        "NLIM-PRO": "Proline",
        "NLIM-NH4": "AmmoniumSulfate",
        "NLIM-UREA": "Urea",
        "CSTARVE": "CStarve",
    }
    assert [a for a, c in conditions.items() if c.rapamycin] == ["RAPA"]
    assert {
        a: c.duration_hours for a, c in conditions.items() if c.duration_hours != 4.0
    } == {"DIAUXY": 10.0}
    assert conditions["NLIM-UREA"].media.name == (
        "Jackson 2020 nitrogen-limited minimal medium, 2% glucose + 800 uM urea "
        "(NLIM-UREA)"
    )
    assert conditions["YPD"].media is j.JACKSON_YPD


def test_build_conditions_refuses_a_label_map_missing_a_condition(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    labels = dict(settings.condition_label)
    labels["CStarve"] = "STARVE"
    broken = settings.model_copy(update={"condition_label": labels})
    with pytest.raises(KeyError, match="CSTARVE"):
        j.build_conditions(broken)


def test_condition_environment_states_the_culture_and_gaps_the_temperature(
    tmp_path: Path,
) -> None:
    conditions = j.build_conditions(_settings(tmp_path))
    env = j.condition_environment(conditions["RAPA"])
    assert env.temperature is None
    (gap,) = env.provenance_gaps
    assert (gap.field, gap.reason) == (
        "temperature",
        ProvenanceGapReason.not_reported_by_primary,
    )
    assert "30 C is likely and unconfirmed" in (gap.note or "")
    assert env.duration_hours == 4.0
    assert env.auxotroph_supplements == []
    assert env.culture_format is not None
    assert (env.culture_format.vessel, env.culture_format.working_volume_ul) == (
        "shake flask",
        50_000.0,
    )
    assert env.culture_format.endpoint is EndpointRule.fixed_duration
    assert env.pre_culture is not None
    assert env.pre_culture.source is PreCultureSource.overnight_culture
    (rapamycin,) = env.perturbations
    assert isinstance(rapamycin, SmallMoleculePerturbation)
    assert rapamycin.compound.name == "rapamycin"
    assert rapamycin.concentration.value == 0.2
    assert rapamycin.concentration.unit is ConcentrationUnit.ug_per_ml
    assert rapamycin.solvent is not None
    assert (rapamycin.solvent.name, rapamycin.solvent.percent) == ("ethanol", 0.02)
    assert j.condition_environment(conditions["DIAUXY"]).duration_hours == 10.0
    assert j.condition_environment(conditions["MMD"]).perturbations == []


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_build_feature_map_keys_genes_and_keeps_markers_apart() -> None:
    features = j.build_feature_map(rel.RELEASED, rel.BULK_HEADER, rel.fake_genome())
    assert features.keys == rel.KEYS
    assert features.marker_columns == [7, 8]
    assert features.gene_columns == [0, 1, 2, 3, 4, 5, 6]
    assert features.gene_keys == rel.KEYS[:7]
    assert features.legacy() == {
        "YIL168W": "merged into YIL167W; kept distinct",
        "YIL080W": "retired: not in R64-4-1",
    }
    assert (
        features.universe_sha256
        == hashlib.sha256("\n".join(sorted(rel.KEYS[:7])).encode()).hexdigest()
    )
    assert features.status["snR10"] == "non_gene_feature"


def test_build_feature_map_refuses_a_header_or_an_ambiguous_id() -> None:
    with pytest.raises(j.FeatureMapError, match="differs from TRIZOL_BULK.tsv"):
        j.build_feature_map(rel.RELEASED, rel.BULK_HEADER[::-1], rel.fake_genome())
    no_marker = [*rel.RELEASED[:7], "KANMX", "YAL003W"]
    with pytest.raises(j.FeatureMapError, match="marker columns"):
        j.build_feature_map(
            no_marker, [j.unmangle(n) for n in no_marker], rel.fake_genome()
        )
    ambiguous = rel.fake_genome({"YIL080W": (GeneNameStatus.AMBIGUOUS, None)})
    with pytest.raises(j.FeatureMapError, match="YIL080W resolves to several features"):
        j.build_feature_map(rel.RELEASED, rel.BULK_HEADER, ambiguous)


# --------------------------------------------------------------------------- #
# Release reading
# --------------------------------------------------------------------------- #
def _strains(root: Path) -> dict[str, j.StrainSpec]:
    return {s.strain: s for s in j.read_strain_table(root / "raw" / j.SUPP1_NAME)}


def test_read_release_sums_each_strain_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rel.install(tmp_path, monkeypatch)
    labels = set(j.read_author_settings(root / "raw" / j.SETTINGS_NAME).condition_label)
    sums, n_cells = j.read_release(
        root / "raw" / j.MATRIX_NAME, _strains(root), labels, chunksize=5
    )
    assert n_cells == 16
    assert len(sums) == rel.N_RECORDS
    b1 = sums[("WT(ho)_B1", "YPD")]
    assert b1.counts.tolist() == [1, 3, 0, 0, 1, 0, 0, 1, 1]
    assert b1.cell_barcodes == ["BC01", "BC02"]
    assert b1.cell_total_umis == [3, 4]
    assert sums[("gat1_A5", "CStarve")].counts.tolist() == [2, 0, 0, 0, 0, 0, 0, 0, 0]
    assert ("stp2_A12", "Urea") not in sums
    assert j.read_release_header(root / "raw" / j.MATRIX_NAME) == rel.RELEASED


def _with_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cells: list[Any]
) -> tuple[Path, dict[str, j.StrainSpec], set[str]]:
    root = rel.install(tmp_path, monkeypatch)
    rel.write_matrix(root / "raw" / j.MATRIX_NAME, cells)
    labels = set(j.read_author_settings(root / "raw" / j.SETTINGS_NAME).condition_label)
    return root / "raw" / j.MATRIX_NAME, _strains(root), labels


@pytest.mark.parametrize(
    ("cell", "message"),
    [
        (
            ("WT(ho)_A1", "YPD", "AC10", [-1, 0, 0, 0, 0, 0, 0, 0, 0]),
            "a negative count",
        ),
        (
            ("WT(ho)_A1", "Mars", "ZZ01", [1] * 9),
            "condition 'Mars' is not a known label",
        ),
        (("WT(ho)_A1", "YPD", "AC10", [1] * 9), "YPD: barcode AC10 seen twice"),
    ],
)
def test_read_release_refuses_a_bad_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cell: tuple[str, str, str, list[int]],
    message: str,
) -> None:
    path, strains, labels = _with_cells(tmp_path, monkeypatch, [*rel.CELLS, cell])
    with pytest.raises(j.ReleaseShapeError, match=message):
        j.read_release(path, strains, labels)


def test_read_release_refuses_unknown_strains_and_mislabeled_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, strains, labels = _with_cells(tmp_path, monkeypatch, rel.CELLS)
    with pytest.raises(
        j.ReleaseShapeError, match="strain 'gat1_A5' is not in Table S2"
    ):
        j.read_release(
            path, {k: v for k, v in strains.items() if k != "gat1_A5"}, labels
        )
    shifted = {
        k: v.model_copy(update={"well": "H9"}) if k == "gat1_A5" else v
        for k, v in strains.items()
    }
    with pytest.raises(j.ReleaseShapeError, match="gat1_A5: Genotype_Group 'gat1'"):
        j.read_release(path, shifted, labels)
    monkeypatch.setattr(j, "PAPER_CELLS", 17)
    with pytest.raises(j.ReleaseShapeError, match="16 cells, the paper states 17"):
        j.read_release(path, strains, labels)


def test_read_release_refuses_non_integer_counts_and_a_wrong_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rel.install(tmp_path, monkeypatch)
    path = root / "raw" / j.MATRIX_NAME
    text = gzip.decompress(path.read_bytes()).decode()
    lines = text.splitlines()
    first = lines[1].split("\t")
    first[1] = "1.5"
    lines[1] = "\t".join(first)
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))
    strains = _strains(root)
    labels = set(j.read_author_settings(root / "raw" / j.SETTINGS_NAME).condition_label)
    with pytest.raises(j.ReleaseShapeError, match="non-integer counts"):
        j.read_release(path, strains, labels)
    path.write_bytes(gzip.compress(b'"YAL001C"\t"Genotype"\n"1_x"\t1\t"a"\n'))
    with pytest.raises(j.ReleaseShapeError, match="the last columns are"):
        j.read_release_header(path)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def test_pseudobulk_phenotype_and_pooled_sum_exactly() -> None:
    features = j.build_feature_map(rel.RELEASED, rel.BULK_HEADER, rel.fake_genome())
    a = j.StrainConditionSum(
        counts=np.array([1, 3, 0, 0, 1, 0, 0, 1, 1]),
        cell_barcodes=["BC01", "BC02"],
        cell_total_umis=[3, 4],
    )
    b = j.StrainConditionSum(
        counts=np.array([3, 0, 1, 2, 0, 1, 0, 4, 2]),
        cell_barcodes=["AC10"],
        cell_total_umis=[13],
    )
    phenotype = j.pseudobulk_phenotype(j.pooled([a, b]), features)
    assert dict(phenotype.umi_count) == {
        "YAL001C": 4,
        "YAL002W-A": 3,
        "YNCG0013W": 1,
        "YNCQ0002W": 2,
        "YIL168W": 1,
        "YIL167W": 1,
    }
    assert dict(phenotype.marker_umi_count) == {"KANMX": 5, "NATMX": 3}
    assert (phenotype.total_umis, phenotype.n_cells, phenotype.n_features) == (20, 3, 7)
    assert phenotype.cell_barcodes == ["BC01", "BC02", "AC10"]
    assert phenotype.cell_total_umis == [3, 4, 13]
    assert phenotype.feature_universe_sha256 == features.universe_sha256


def test_strain_genotype_is_one_barcoded_homozygous_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rel.install(tmp_path, monkeypatch)
    spec = _strains(root)["gat1_A5"]
    (pert,) = j.strain_genotype(spec, "YFL021W", "GAT1").perturbations
    dumped = pert.model_dump()
    assert dumped["perturbation_type"] == "barcoded_kanmx_deletion"
    assert (dumped["systematic_gene_name"], dumped["perturbed_gene_name"]) == (
        "YFL021W",
        "GAT1",
    )
    assert (dumped["barcode"], dumped["cassette"]) == (
        "GGGGGTTTGGGGGTTTGGG",
        "KanMX/NatMX",
    )
    assert dumped["construction"] == {
        "strain_accession": "gat1_A5",
        "lab": None,
        "batch": None,
        "plate": "BC_DEL_1",
        "well": "A5",
    }


def test_deleted_gene_ids_and_their_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rel.install(tmp_path, monkeypatch)
    specs = list(_strains(root).values())
    settings = j.read_author_settings(root / "raw" / j.SETTINGS_NAME)
    ids = j.TfDeletionPseudobulkJackson2020Dataset._deleted_gene_ids(
        specs, settings, rel.fake_genome()
    )
    assert ids == {
        "gat1": ("YFL021W", "GAT1"),
        "ho": ("YDL227C", "HO"),
        "stp2": ("YHR006W", "STP2"),
    }
    lost_ho = rel.fake_genome({"HO": (GeneNameStatus.RETIRED, "HO")})
    with pytest.raises(ValueError, match="HO did not resolve to one current gene"):
        j.TfDeletionPseudobulkJackson2020Dataset._deleted_gene_ids(
            specs, settings, lost_ho
        )
    stale_tf = rel.fake_genome({"YFL021W": (GeneNameStatus.RETIRED, "YFL021W")})
    with pytest.raises(ValueError, match="gat1: YFL021W is not a current R64 gene"):
        j.TfDeletionPseudobulkJackson2020Dataset._deleted_gene_ids(
            specs, settings, stale_tf
        )
    swapped = settings.model_copy(
        update={"tf_systematic_name": {"gat1": "YHR006W", "stp2": "YHR006W"}}
    )
    with pytest.raises(ValueError, match="YHR006W's standard name STP2 is not gat1"):
        j.TfDeletionPseudobulkJackson2020Dataset._deleted_gene_ids(
            specs, swapped, rel.fake_genome()
        )
    extra = settings.model_copy(
        update={
            "tf_systematic_name": {**settings.tf_systematic_name, "gln3": "YER040W"}
        }
    )
    with pytest.raises(ValueError, match="not the settings file's TFs"):
        j.TfDeletionPseudobulkJackson2020Dataset._deleted_gene_ids(
            specs, extra, rel.fake_genome()
        )


def test_release_ledger_refuses_an_unbalanced_account() -> None:
    kwargs: dict[str, Any] = {
        "matrix_sha256": "x",
        "n_cells_released": 16,
        "n_cells_served": 16,
        "n_wild_type_cells": 13,
        "n_transcript_counts_released": 100,
        "n_transcript_counts_served": 100,
        "paper_cells": 16,
        "paper_transcript_counts": 100,
        "n_features_released": 9,
        "n_gene_features": 7,
        "marker_features": ["KANMX", "NATMX"],
        "feature_status": {},
        "remapped_features": {},
        "legacy_features": {},
        "n_strains": 4,
        "n_conditions": 11,
        "n_records": 15,
        "n_references": 11,
        "strain_conditions_without_cells": ["x"] * 29,
        "cells_per_condition": {},
        "records_per_condition": {},
        "table_s2_genotype_disagreements": [],
        "harvest_cells_per_ml": {},
        "refused": {},
    }
    assert j.ReleaseLedger(**kwargs).n_records == 15
    for update, message in [
        ({"n_cells_served": 15}, "served + refused cells must equal released cells"),
        ({"paper_cells": 17}, "the release's cell count is not the paper's"),
        (
            {"paper_transcript_counts": 99},
            "the release's count total is not the paper's",
        ),
        ({"n_records": 14}, "records + empty pairs must equal strains x conditions"),
    ]:
        with pytest.raises(ValidationError, match=re.escape(message)):
            j.ReleaseLedger(**{**kwargs, **update})


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
def _build(root: Path) -> j.TfDeletionPseudobulkJackson2020Dataset:
    return j.TfDeletionPseudobulkJackson2020Dataset(
        root=str(root), genome=rel.fake_genome()
    )


def test_process_builds_one_record_per_strain_condition_with_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rel.install(tmp_path, monkeypatch)
    dataset = _build(root)
    assert len(dataset) == rel.N_RECORDS
    rows = pd.read_csv(root / "preprocess" / "data.csv")
    assert rows["strain"].tolist()[:3] == ["WT(ho)_A1", "WT(ho)_B1", "gat1_A5"]
    assert rows["condition"].tolist()[:3] == ["YPD", "YPD", "YPD"]
    assert int(rows["n_cells"].sum()) == 16
    first = dataset[0]
    experiment = PseudobulkUmiCountExperiment.model_validate(first["experiment"])
    reference = PseudobulkUmiCountExperimentReference.model_validate(first["reference"])
    assert experiment.phenotype.n_cells == 1
    assert experiment.phenotype.total_umis == 13
    assert reference.phenotype_reference.n_cells == 3
    assert reference.phenotype_reference.cell_barcodes == ["AC10", "BC01", "BC02"]
    assert reference.genome_reference.strain == "FY4/FY5"
    assert reference.environment_reference == experiment.environment
    ledger = json.loads((root / "preprocess" / "release_ledger.json").read_text())
    assert ledger["n_records"] == 15
    assert ledger["n_references"] == 11
    assert ledger["n_cells_served"] == ledger["n_cells_released"] == 16
    assert ledger["n_wild_type_cells"] == 13
    assert ledger["n_transcript_counts_served"] == rel.N_UMIS
    assert ledger["feature_status"] == {
        "current": 2,
        "non_gene_feature": 4,
        "retired": 1,
    }
    assert ledger["remapped_features"] == {
        "snR10": "YNCG0013W",
        "15S_rRNA": "YNCQ0002W",
    }
    assert ledger["legacy_features"] == {
        "YIL168W": "merged into YIL167W; kept distinct",
        "YIL080W": "retired: not in R64-4-1",
    }
    assert len(ledger["strain_conditions_without_cells"]) == 29
    assert "stp2_A12 x CSTARVE" in ledger["strain_conditions_without_cells"]
    assert ledger["cells_per_condition"]["YPD"] == 5
    assert ledger["records_per_condition"] == {
        "CSTARVE": 2,
        "NLIM-UREA": 1,
        "NLIM-NH4": 1,
        "NLIM-PRO": 1,
        "NLIM-GLN": 1,
        "MMEtOH": 1,
        "MMD": 1,
        "YPEtOH": 1,
        "RAPA": 1,
        "DIAUXY": 1,
        "YPD": 4,
    }
    assert ledger["table_s2_genotype_disagreements"] == ["stp2_A12"]
    assert ledger["harvest_cells_per_ml"]["NLIM-PRO"] == 4_000_000.0


def test_process_refuses_a_condition_without_a_control_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = [c for c in rel.CELLS if not (c[0] == "WT(ho)_A1" and c[1] == "Urea")]
    root = tmp_path / "store"
    pins = rel.write_raw(root / "raw", cells)
    rel.pin_constants(monkeypatch, pins)
    monkeypatch.setattr(j, "PAPER_CELLS", 15)
    with pytest.raises(
        j.ReleaseShapeError, match="NLIM-UREA: no HO-deletion control cell"
    ):
        _build(root)


def test_process_refuses_barcodes_that_disagree_with_the_fasta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    raw = root / "raw"
    pins = rel.write_raw(raw)
    (raw / j.BARCODES_NAME).write_text(">WT(ho)_A1\nAAAA\n")
    pins[j.BARCODES_NAME] = hashlib.sha256(
        (raw / j.BARCODES_NAME).read_bytes()
    ).hexdigest()
    rel.pin_constants(monkeypatch, pins)
    with pytest.raises(
        j.StrainTableError, match="barcodes differ from bcdel_1_barcodes"
    ):
        _build(root)


def test_the_dataset_needs_a_genome_and_builds_only_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rel.install(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="needs a genome to key the gene features"):
        j.TfDeletionPseudobulkJackson2020Dataset(root=str(root))
    dataset = _build(root)
    frame = pd.DataFrame({"a": [1]})
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError, match="builds its records in process"):
        dataset.create_experiment()
    assert dataset.raw_file_names == list(j.RAW_FILES)


def test_verify_build_hands_the_store_to_the_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torchcell.verification import runners

    calls: list[tuple[str, str, str | None]] = []

    def fake(name: str, data_root: str, abs_root: str | None = None) -> str:
        calls.append((name, data_root, abs_root))
        return "report"

    monkeypatch.setattr(runners, "verify_pseudobulk_umi_count_dataset", fake)
    assert j.verify_build("/store", data_root="/data") == "report"
    assert calls == [("tf_deletion_pseudobulk_jackson2020", "/data", "/store")]


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _tar(path: Path, members: dict[str, bytes]) -> str:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_deposit_raw_mirror_cuts_members_and_writes_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = tmp_path / "fetch"
    fetch.mkdir()
    files = {
        "settings": b"CONDITION.LEVELS <<- c()\n",
        "bulk": b"YAL001C\n",
        "fasta": b">a\nACGT\n",
        "matrix": b"matrix-bytes",
        "xlsx": b"xlsx-bytes",
    }
    code1 = _tar(
        fetch / j.CODE1_NAME,
        {j.SETTINGS_NAME: files["settings"], "data/TRIZOL_BULK.tsv": files["bulk"]},
    )
    code4 = _tar(fetch / j.CODE4_NAME, {j.BARCODES_NAME: files["fasta"]})
    (fetch / j.MATRIX_NAME).write_bytes(files["matrix"])
    (fetch / j.SUPP1_NAME).write_bytes(files["xlsx"])
    pins = {
        j.MATRIX_NAME: _sha(files["matrix"]),
        j.SUPP1_NAME: _sha(files["xlsx"]),
        j.SETTINGS_NAME: _sha(files["settings"]),
        j.BULK_NAME: _sha(files["bulk"]),
        j.BARCODES_NAME: _sha(files["fasta"]),
    }
    monkeypatch.setattr(j, "RAW_FILES", pins)
    monkeypatch.setattr(j, "CODE1_SHA256", code1)
    monkeypatch.setattr(j, "CODE4_SHA256", code4)
    monkeypatch.setattr(j, "MATRIX_SHA256", pins[j.MATRIX_NAME])
    monkeypatch.setattr(j, "SUPP1_SHA256", pins[j.SUPP1_NAME])
    monkeypatch.setattr(
        j,
        "RAW_SOURCES",
        {
            j.MATRIX_REL: (j.MATRIX_NAME, None, None),
            j.SUPP1_REL: (j.SUPP1_NAME, None, None),
            j.SETTINGS_REL: (j.CODE1_NAME, j.SETTINGS_NAME, code1),
            j.BULK_REL: (j.CODE1_NAME, "data/TRIZOL_BULK.tsv", code1),
            j.BARCODES_REL: (j.CODE4_NAME, j.BARCODES_NAME, code4),
        },
    )
    data_root = tmp_path / "data"
    root = j.deposit_raw_mirror(
        fetch_dir=fetch, retrieved_at="2026-10-11", data_root=str(data_root)
    )
    assert root == data_root / j.RAW_DIR_REL
    assert (root / j.SETTINGS_REL).read_bytes() == files["settings"]
    manifest = j.load_manifest(str(data_root))
    assert [f.path for f in manifest.files] == list(j.RAW_RELPATHS.values())
    assert j.manifest_sha256(manifest, j.BULK_REL) == pins[j.BULK_NAME]
    by_path = {f.path: f for f in manifest.files}
    settings = by_path[j.SETTINGS_REL]
    assert settings.retrieval is not None
    assert settings.retrieval.retriever == "torchcell.literature.retrieve.tar_member"
    assert settings.retrieval.params == {
        "url": f"{j.PMC_BUCKET_URL}{j.PMC_KEY_PREFIX}/{j.CODE1_NAME}",
        "member": j.SETTINGS_NAME,
        "container_sha256": code1,
    }
    matrix = by_path[j.MATRIX_REL]
    assert matrix.retrieval is not None
    assert matrix.retrieval.params == {"key": f"{j.PMC_KEY_PREFIX}/{j.MATRIX_NAME}"}
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        j.manifest_sha256(manifest, "data/other")
    # idempotent by sha256; a different file at a recorded path is refused
    j.deposit_raw_mirror(
        fetch_dir=fetch, retrieved_at="2026-10-11", data_root=str(data_root)
    )
    (root / j.BULK_REL).write_bytes(b"edited")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        j.deposit_raw_mirror(
            fetch_dir=fetch, retrieved_at="2026-10-11", data_root=str(data_root)
        )
    (fetch / j.MATRIX_NAME).write_bytes(b"drifted")
    with pytest.raises(ValueError, match="does not hash to its pin"):
        j.deposit_raw_mirror(
            fetch_dir=fetch, retrieved_at="2026-10-11", data_root=str(data_root)
        )


def test_download_links_the_mirror_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    mirror = data_root / j.RAW_DIR_REL
    staged = tmp_path / "staged"
    pins = rel.write_raw(staged)
    rel.pin_constants(monkeypatch, pins)
    from torchcell.literature.manifest import ArtifactRecord, Manifest

    records = []
    for name, relpath in j.RAW_RELPATHS.items():
        (mirror / relpath).parent.mkdir(parents=True, exist_ok=True)
        (mirror / relpath).write_bytes((staged / name).read_bytes())
        records.append(
            ArtifactRecord(path=relpath, role="raw_data", bytes=1, sha256=pins[name])
        )
    (mirror / "manifest.json").write_text(
        Manifest(citation_key=j.CITATION_KEY, files=records).model_dump_json()
    )
    monkeypatch.setattr(j, "_data_root", lambda: str(data_root))
    dataset = _build(tmp_path / "store")
    assert len(dataset) == rel.N_RECORDS
    linked = tmp_path / "store" / "raw" / j.MATRIX_NAME
    assert linked.is_symlink()
    assert os.readlink(linked) == str(mirror / j.MATRIX_REL)


# --------------------------------------------------------------------------- #
# The real mirrors and the built store (``--data``)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get("DATA_ROOT", "")


_MIRROR = osp.join(_data_root(), j.RAW_DIR_REL)
_STORE = osp.join(_data_root(), "data/torchcell/tf_deletion_pseudobulk_jackson2020")


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(osp.join(_MIRROR, "manifest.json")),
    reason="Jackson 2020 raw mirror not on disk",
)
def test_the_raw_mirror_holds_the_pinned_release() -> None:
    manifest = j.load_manifest(_data_root())
    for name, relpath in j.RAW_RELPATHS.items():
        assert j.manifest_sha256(manifest, relpath) == j.RAW_FILES[name]
    mirror = Path(_MIRROR)
    specs = j.read_strain_table(mirror / j.SUPP1_REL)
    assert len(specs) == 72
    assert sorted(s.strain for s in specs if s.diploid_cell_disagrees) == list(
        j.TABLE_S2_GENOTYPE_DISAGREEMENTS
    )
    assert j.read_barcode_fasta(mirror / j.BARCODES_REL) == {
        s.strain: s.barcode for s in specs
    }
    settings = j.read_author_settings(mirror / j.SETTINGS_REL)
    assert settings.tf_systematic_name["gat1"] == "YFL021W"
    assert len(j.read_release_header(mirror / j.MATRIX_REL)) == 6829


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(osp.join(_STORE, "preprocess", "release_ledger.json")),
    reason="Jackson 2020 dev store not built",
)
def test_the_built_store_accounts_for_every_released_cell() -> None:
    ledger = json.loads(Path(_STORE, "preprocess", "release_ledger.json").read_text())
    assert ledger["n_records"] == 784
    assert ledger["n_cells_served"] == 38225
    assert ledger["n_transcript_counts_served"] == 83703440
    assert ledger["n_wild_type_cells"] == 3306
    assert ledger["feature_status"] == {
        "current": 6599,
        "non_gene_feature": 227,
        "retired": 1,
    }
    assert len(ledger["remapped_features"]) == 114
    assert ledger["strain_conditions_without_cells"] == [
        "stp2_E12 x YPD",
        "stp2_G12 x YPD",
        "stp2_E12 x RAPA",
        "stp2_G12 x RAPA",
        "gcn4_G10 x NLIM-GLN",
        "gcn4_A10 x NLIM-NH4",
        "stp2_E12 x CSTARVE",
        "stp2_G12 x CSTARVE",
    ]
    assert ledger["cells_per_condition"] == {
        "YPD": 11037,
        "DIAUXY": 3332,
        "RAPA": 11805,
        "YPEtOH": 1427,
        "MMD": 1894,
        "MMEtOH": 581,
        "NLIM-GLN": 1644,
        "NLIM-PRO": 853,
        "NLIM-NH4": 1111,
        "NLIM-UREA": 992,
        "CSTARVE": 3549,
    }
