# tests/torchcell/datasets/scerevisiae/test_cooper2010.py
"""Unit tests for the Cooper 2010 CE-LIF amino-acid loader's record construction.

The loader's surface is how it turns one deposited Table 4 row into a typed metabolite
record, so these tests drive ``read_table4`` / ``retain_rows`` / ``create_experiment`` on
a synthetic Table 4 plus a fake resolver. What is pinned: the header is checked exactly,
missing cells drop the key rather than becoming NaN, the medium is a typed dropout of the
library SC (joinable at ``base_medium``), temperature is a typed gap, the reference is 1.0
per present key, replicate counts are the conservative 1, identifiers are normalized and
recorded, a duplicate identifier keeps its first row and ledgers the second, a RENAMED row
is a distinct strain of the current gene, non-gene and retired rows are dropped with their
status, and every sourced quote is still verbatim in the sha256-pinned mirror.
"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path
from typing import Any

import pytest

from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import Genotype
from torchcell.datasets.scerevisiae import cooper2010 as module
from torchcell.datasets.scerevisiae.cooper2010 import (
    COOPER_SC,
    EXPECTED_HEADER,
    LEGEND_SOURCED_VALUES,
    MEASUREMENT_TYPE,
    PEAK_KEYS,
    SOURCED_VALUES,
    AminoAcidCooper2010Dataset,
    normalize_identifier,
    read_table4,
    retain_rows,
)
from torchcell.sequence.genome.scerevisiae.s288c import GeneNameStatus


class _Resolution:
    """The three fields of ``GeneNameResolution`` the loader reads."""

    def __init__(self, status: GeneNameStatus, name: str | None, feature: str | None):
        self.status = status
        self.systematic_name = name
        self.feature_type = feature


_RESOLVER: dict[str, _Resolution] = {
    "YBR208C": _Resolution(GeneNameStatus.CURRENT, "YBR208C", None),
    "YBR020W": _Resolution(GeneNameStatus.CURRENT, "YBR020W", None),
    "YML048W-A": _Resolution(GeneNameStatus.CURRENT, "YML048W-A", None),
    "YLR228C": _Resolution(GeneNameStatus.CURRENT, "YLR228C", None),
    "YAL034C-B": _Resolution(GeneNameStatus.CURRENT, "YAL034C-B", None),
    "YAL035C-A": _Resolution(GeneNameStatus.RENAMED, "YAL034C-B", None),
    "YCL074W": _Resolution(GeneNameStatus.NON_GENE_FEATURE, "YCL074W", "pseudogene"),
    "R0010W": _Resolution(GeneNameStatus.RETIRED, "R0010W", None),
    "YGL180W": _Resolution(GeneNameStatus.CURRENT, "YGL180W", None),
}

_HEADER = "\t".join(EXPECTED_HEADER)


def _row(name: str, gene: str, values: list[str]) -> str:
    assert len(values) == len(PEAK_KEYS)
    return "\t".join([name, gene, *values])


def _full(seed: float) -> list[str]:
    return [f"{seed + i * 0.1:.3f}" for i in range(len(PEAK_KEYS))]


def _table(tmp_path: Path) -> str:
    """A Table 4 with one row per rule: full, ragged, duplicate, malformed, renamed,
    non-gene, retired, and an essentiality-store hit.
    """
    ragged = _full(0.5)
    ragged[0] = "-"  # RPS19 missing
    ragged[2] = "0"  # gshbiotin released zero
    ragged[-1] = "-"  # Asp missing
    rows = [
        _HEADER,
        _row("YBR208C", "DUR1,2", _full(1.0)),
        _row("YBR020W", "GAL1", ragged),
        _row("YBR020W", "GAL1", _full(2.0)),  # duplicate identifier, second row
        _row("YML048WA-", "GSF2", _full(3.0)),  # misplaced suffix
        _row("YLR228 C", "", _full(4.0)),  # internal whitespace, empty NAME
        _row("YAL035C-A", "YAL035C-A", _full(5.0)),  # RENAMED onto YAL034C-B
        _row("YAL034C-B", "YAL034C-B", _full(6.0)),  # the current gene's own row
        _row("YCL074W", "YCL074W", _full(7.0)),  # pseudogene
        _row("R0010W", "R0010W", _full(8.0)),  # retired
        _row("YGL180W", "ATG1", _full(9.0)),  # in the essentiality store, kept
    ]
    path = tmp_path / "SupplementalTable4.txt"
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def _dataset() -> AminoAcidCooper2010Dataset:
    """The loader without its PyG init: these tests never touch disk or an LMDB."""
    dataset = AminoAcidCooper2010Dataset.__new__(AminoAcidCooper2010Dataset)
    dataset.name = "AminoAcidCooper2010Dataset"
    return dataset


def _retained(tmp_path: Path) -> tuple[list[Any], Any]:
    rows = read_table4(_table(tmp_path))
    return retain_rows(
        rows,
        lambda name: _RESOLVER[name],
        frozenset({"YGL180W"}),
        dataset_name="AminoAcidCooper2010Dataset",
    )


# ---- parsing --------------------------------------------------------------------- #
def test_header_is_checked_exactly(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("Sysname\tNAME\tRPS19\targ1\tunknown\n")
    with pytest.raises(RuntimeError, match="unmapped columns \\['unknown'\\]"):
        read_table4(path)


def test_missing_cells_drop_the_key_and_zero_is_a_value(tmp_path: Path) -> None:
    rows = read_table4(_table(tmp_path))
    ragged = rows[1].levels
    assert "lysine_related_peak1" not in ragged and "aspartate" not in ragged
    assert ragged["gshbiotin"] == 0.0
    assert len(rows[0].levels) == len(PEAK_KEYS)


def test_identifier_normalization_is_recorded_verbatim(tmp_path: Path) -> None:
    assert normalize_identifier("YML048WA-") == "YML048W-A"
    assert normalize_identifier("YLR228 C") == "YLR228C"
    assert normalize_identifier("YML009c") == "YML009C"
    assert normalize_identifier("YJL038C ") == "YJL038C"
    _, ledger = _retained(tmp_path)
    assert {(n.source_name, n.normalized_name) for n in ledger.normalizations} == {
        ("YML048WA-", "YML048W-A"),
        ("YLR228 C", "YLR228C"),
    }


# ---- retention ------------------------------------------------------------------- #
def test_non_gene_and_retired_rows_are_dropped_with_their_status(
    tmp_path: Path,
) -> None:
    kept, ledger = _retained(tmp_path)
    assert {row.systematic_gene_name for row in kept} == {
        "YBR208C",
        "YBR020W",
        "YML048W-A",
        "YLR228C",
        "YAL034C-B",
        "YGL180W",
    }
    dropped = {d.source_name: d for d in ledger.dropped_orfs}
    assert set(dropped) == {"YCL074W", "R0010W"}
    assert (dropped["YCL074W"].status, dropped["YCL074W"].feature_type) == (
        "non_gene_feature",
        "pseudogene",
    )
    assert dropped["R0010W"].status == "retired"
    assert dropped["R0010W"].n_values == len(PEAK_KEYS)


def test_renamed_row_is_a_distinct_strain_of_the_current_gene(tmp_path: Path) -> None:
    kept, ledger = _retained(tmp_path)
    assert ledger.renamed_orfs == {"YAL035C-A": "YAL034C-B"}
    by_source = {row.perturbed_gene_name: row.systematic_gene_name for row in kept}
    assert by_source["YAL035C-A"] == "YAL034C-B"
    assert by_source["YAL034C-B"] == "YAL034C-B"


def test_duplicate_identifier_keeps_the_first_row_and_ledgers_the_second(
    tmp_path: Path,
) -> None:
    kept, ledger = _retained(tmp_path)
    gal1 = [row for row in kept if row.systematic_gene_name == "YBR020W"]
    assert len(gal1) == 1 and gal1[0].row_index == 1
    assert len(ledger.duplicate_strain_rows) == 1
    duplicate = ledger.duplicate_strain_rows[0]
    assert (duplicate.row_index, duplicate.kept_row_index) == (2, 1)
    assert duplicate.levels["arginine"] == 2.1  # never averaged into the kept row
    assert ledger.n_duplicate_rows == 1


def test_essentiality_hit_is_flagged_and_kept(tmp_path: Path) -> None:
    kept, ledger = _retained(tmp_path)
    assert "YGL180W" in {row.systematic_gene_name for row in kept}
    assert [f.systematic_gene_name for f in ledger.essentiality_flagged] == ["YGL180W"]
    assert ledger.n_excluded_non_deletion == 0


def test_parent_strain_row_is_excluded(tmp_path: Path) -> None:
    path = tmp_path / "t.txt"
    path.write_text(
        "\n".join([_HEADER, _row("BY4742", "wild type", _full(1.0))]) + "\n"
    )
    kept, ledger = retain_rows(
        read_table4(path), lambda name: _RESOLVER[name], frozenset(), dataset_name="x"
    )
    assert kept == []
    assert ledger.excluded_non_deletion[0].reason.startswith("parent-strain")


def test_ledger_counts_add_up(tmp_path: Path) -> None:
    _, ledger = _retained(tmp_path)
    assert ledger.n_source_rows == 10
    assert (
        ledger.n_kept
        + ledger.n_dropped_orfs
        + ledger.n_duplicate_rows
        + ledger.n_excluded_non_deletion
        == ledger.n_source_rows
    )


# ---- record ---------------------------------------------------------------------- #
def test_record_keys_follow_the_row_and_the_reference_is_one(tmp_path: Path) -> None:
    kept, _ = _retained(tmp_path)
    ragged = next(row for row in kept if row.systematic_gene_name == "YBR020W")
    experiment, reference, publication = _dataset().create_experiment(ragged)
    keys = list(experiment.phenotype.metabolite_level)
    assert len(keys) == len(PEAK_KEYS) - 2
    assert "lysine_related_peak1" not in keys
    assert keys == [k for k in PEAK_KEYS.values() if k in ragged.levels]
    assert experiment.phenotype.metabolite_level["gshbiotin"] == 0.0
    assert experiment.phenotype.n_replicates == dict.fromkeys(keys, 1)
    assert experiment.phenotype.metabolite_level_se is None
    assert experiment.phenotype.measurement_type == MEASUREMENT_TYPE
    assert reference.phenotype_reference.metabolite_level == dict.fromkeys(keys, 1.0)
    assert reference.genome_reference.strain == "BY4741"
    assert publication.pubmed_id == "20610602" and publication.doi == module.DOI


def test_phenotype_declares_its_typed_absences(tmp_path: Path) -> None:
    kept, _ = _retained(tmp_path)
    full = kept[0]
    experiment, _, _ = _dataset().create_experiment(full)
    gaps = {gap.field: gap for gap in experiment.phenotype.provenance_gaps}
    assert set(gaps) == {"metabolite_level_se", "target_metabolite_ids"}
    assert "gshbiotin" in str(gaps["target_metabolite_ids"].note)
    assert experiment.phenotype.target_metabolite_ids is None
    ragged = next(row for row in kept if row.systematic_gene_name == "YML048W-A")
    ragged.levels.pop("gshbiotin")
    experiment, _, _ = _dataset().create_experiment(ragged)
    gaps = {gap.field: gap for gap in experiment.phenotype.provenance_gaps}
    assert set(gaps) == {"metabolite_level_se", "target_metabolite_ids"}
    assert "gshbiotin" not in str(gaps["target_metabolite_ids"].note)


def test_genotype_is_a_kanmx_deletion_keyed_on_the_normalized_source(
    tmp_path: Path,
) -> None:
    kept, _ = _retained(tmp_path)
    renamed = next(row for row in kept if row.perturbed_gene_name == "YAL035C-A")
    experiment, _, _ = _dataset().create_experiment(renamed)
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    perturbation = genotype.perturbations[0]
    assert perturbation.perturbation_type == "kanmx_deletion"
    assert perturbation.systematic_gene_name == "YAL034C-B"
    assert perturbation.perturbed_gene_name == "YAL035C-A"


# ---- environment ----------------------------------------------------------------- #
def test_medium_is_a_typed_dropout_of_the_library_sc() -> None:
    assert COOPER_SC.base_medium == "SC"
    assert COOPER_SC.state == SC.state == "liquid"
    assert [d.name for d in COOPER_SC.dropouts] == [
        "L-alanine",
        "L-asparagine",
        "L-cysteine",
        "L-glutamine",
        "glycine",
        "L-proline",
        "uracil",
    ]
    names = {c.compound.name for c in COOPER_SC.components}
    assert "L-serine" in names and "adenine" in names and "uracil" not in names
    assert COOPER_SC.provenance[0].quote.startswith(
        "Yeast growth was in synthetic complete media (adenine"
    )


def test_temperature_is_a_typed_gap_and_duration_is_sixteen_hours() -> None:
    environment = AminoAcidCooper2010Dataset._environment()
    assert environment.temperature is None
    assert environment.duration_hours == 16.0
    assert [gap.field for gap in environment.provenance_gaps] == ["temperature"]
    assert environment.media == COOPER_SC


# ---- provenance ------------------------------------------------------------------ #
def test_every_sourced_value_quote_is_verbatim_in_the_mirrored_paper() -> None:
    """The audit anchor is only real if the quote is still in the sha256-pinned file."""
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    paper = osp.join(
        data_root, "torchcell-library", module.CITATION_KEY, module.PAPER_MD
    )
    if not osp.exists(paper):
        pytest.skip("the literature mirror is not mounted on this machine")
    text = Path(paper).read_text()
    for key, sourced in SOURCED_VALUES.items():
        assert sourced.quote in text, key
        assert sourced.provenance.sha256 == module.PAPER_MD_SHA256


def test_legend_quote_is_verbatim_in_the_deposited_legends_file() -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    legends = (
        module.raw_mirror_dir(data_root)
        / module._RAW_FILES[module.LEGENDS_NAME]["relpath"]
    )
    if not legends.exists():
        pytest.skip("the raw mirror is not deposited on this machine")
    text = legends.read_bytes().decode("latin-1")
    sourced = LEGEND_SOURCED_VALUES["table4_legend"]
    assert sourced.quote in text
    assert sourced.provenance.sha256 == module.LEGENDS_SHA256


def test_deposit_refuses_a_hash_mismatch(tmp_path: Path) -> None:
    staged = tmp_path / "deposit"
    staged.mkdir()
    (staged / module.TABLE4_NAME).write_text("Sysname\tNAME\n")
    (staged / module.LEGENDS_NAME).write_bytes(b"x")
    (staged / module.SHA256SUMS_NAME).write_text(
        f"{module.TABLE4_SHA256}  {module.TABLE4_NAME}\n"
        f"{module.LEGENDS_SHA256}  {module.LEGENDS_NAME}\n"
    )
    with pytest.raises(RuntimeError, match="refusing to deposit"):
        module.deposit_raw_mirror(source_dir=staged, data_root=str(tmp_path))
