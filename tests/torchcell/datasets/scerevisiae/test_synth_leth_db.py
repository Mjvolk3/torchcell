# tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py
# [[tests.torchcell.datasets.scerevisiae.test_synth_leth_db]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py
"""The SynLethDB provenance record: the decision, and the measurement behind it.

Synthetic tests run everywhere. They write CSVs in the released header, including a
counterfactual file that DOES carry a medium column, so the condition rule is shown to
separate the two rather than merely to find nothing in the real bytes.

The ``@pytest.mark.data`` tests read the pinned release files and the mirrored paper:
they verify both sha256 pins, re-measure every count the decision rests on, and audit
every quote against the pinned ``paper.md``.
"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path

import pandas as pd
import pytest

import torchcell.datasets.scerevisiae.synth_leth_db as s
from torchcell.data.experiment_dataset import RawSha256MismatchError
from torchcell.verification.sourced import audit_sourced_value

DATA_ROOT = os.environ.get("DATA_ROOT")

HEADER = ",".join(s.SYNLETHDB_COLUMNS)

#: What the pinned release files hold, re-measured by the ``data`` tests.
REAL_SL_ROWS = 14000
REAL_SR_ROWS = 6948
REAL_SL_SOURCE_PUBMED_IDS = 1741
REAL_SR_SOURCE_PUBMED_IDS = 1918
REAL_SL_ROWS_BY_EVIDENCE = {
    "High Throughput": 7721,
    "High Throughput|Low Throughput": 1248,
    "Low Throughput": 4944,
    "Text Mining": 87,
}
REAL_SL_SCORES_BY_EVIDENCE = {
    "High Throughput": [0.5],
    "High Throughput|Low Throughput": [0.9],
    "Low Throughput": [0.8],
    "Text Mining": [0.25, 0.5, 0.75, 1.0],
}
REAL_SR_SCORED_ROWS = 74


def _csv(path: Path, rows: list[str], *, header: str = HEADER) -> str:
    path.write_text("﻿" + header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def _sha256(path: str) -> str:
    import hashlib

    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


@pytest.fixture
def sl_csv(tmp_path: Path) -> tuple[str, str]:
    """A four-row SL file in the released header, and its sha256."""
    path = _csv(
        tmp_path / "Yeast_SL.csv",
        [
            "TFC3,851230,SAC1,853646,,18676811,High Throughput,0.5",
            "NTH2,852033,DAN4,850372,,14764870,Low Throughput,0.8",
            "ATG1,852754,ATG7,856767,,24470217,High Throughput|Low Throughput,0.9",
            "GSP1,850990,TRA1,856498,,24125552;19918932,Text Mining,0.25",
        ],
    )
    return path, _sha256(path)


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def test_the_module_registers_no_dataset() -> None:
    """No dataset class: no field states a medium, so no record can be built."""
    from torchcell.datasets.dataset_registry import dataset_registry

    assert not any(cls.__module__ == s.__name__ for cls in dataset_registry.values())
    assert "SynthLethalityYeastSynthLethDbDataset" not in dataset_registry
    assert "SynthRescueYeastSynthLethDbDataset" not in dataset_registry


def test_the_removed_classes_are_not_importable_from_the_dataset_package() -> None:
    """Both served classes are gone from the package namespace."""
    import torchcell.datasets as datasets

    for name in (
        "SynthLethalityYeastSynthLethDbDataset",
        "SynthRescueYeastSynthLethDbDataset",
    ):
        assert not hasattr(datasets, name)
        assert name not in datasets.__all__


def test_neither_class_is_in_the_public_adapter_map() -> None:
    """Nothing in the KG build can reach either dataset by class or by adapter."""
    from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map

    names = {cls.__name__ for cls in dataset_adapter_map}
    adapters = {adapter.__name__ for adapter in dataset_adapter_map.values()}
    assert not names & {
        "SynthLethalityYeastSynthLethDbDataset",
        "SynthRescueYeastSynthLethDbDataset",
    }
    assert not adapters & {
        "SynthLethalityYeastSynthLethDbAdapter",
        "SynthRescueYeastSynthLethDbAdapter",
    }


def test_the_recorded_header_names_no_growth_condition() -> None:
    """Clause 1 as a constant: eight columns, and not one of them is a condition."""
    assert len(s.SYNLETHDB_COLUMNS) == 8
    assert not [
        column
        for column in s.SYNLETHDB_COLUMNS
        if any(term in column.casefold() for term in s.CONDITION_HEADER_TERMS)
    ]


def test_the_old_resolution_record_keeps_both_measurements() -> None:
    """Clause 5: the wrong-ORF counts outlive the stores they were measured on."""
    assert s.OLD_RESOLUTION_WRONG_ORF == {"Yeast_SL.csv": 872, "Yeast_SR.csv": 119}
    assert s.OLD_RESOLUTION_DUPLICATE_PAIRS == {"Yeast_SL.csv": 7, "Yeast_SR.csv": 2}


def test_the_unmirrored_biogrid_reference_is_named_not_assumed() -> None:
    """The 74 SR scores are undefined because this document is not in the mirror."""
    assert s.BIOGRID_2019_REFERENCE.source_uri == "https://doi.org/10.1093/nar/gky1079"
    assert "not in the mirror" in str(s.BIOGRID_2019_REFERENCE.method)


def test_every_sourced_quote_names_the_pinned_paper() -> None:
    """Each clause rests on a sentence of one pinned document."""
    assert len(s.SYNLETHDB_SOURCED_QUOTES) == 4
    for sourced in s.SYNLETHDB_SOURCED_QUOTES:
        assert sourced.provenance.sha256 == s.SYNLETHDB_PAPER_SHA256
        assert sourced.provenance.citation_key == s.SYNLETHDB_CITATION_KEY
        assert sourced.quote


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
def test_the_inventory_finds_no_condition_and_no_storable_row(
    sl_csv: tuple[str, str],
) -> None:
    """Clause 1 on synthetic bytes, with the byte-order mark stripped from the header."""
    path, sha = sl_csv
    inventory = s.release_inventory(path, sha)
    assert inventory.columns == list(s.SYNLETHDB_COLUMNS)
    assert inventory.n_rows == 4
    assert inventory.n_condition_columns == 0
    assert inventory.n_cell_line_stated == 0
    assert inventory.n_scored == 4
    assert inventory.n_storable == 0


def test_a_release_that_grew_a_medium_column_is_refused(tmp_path: Path) -> None:
    """The counterfactual: a condition column invalidates the measurement, loudly."""
    path = _csv(
        tmp_path / "Yeast_SL.csv",
        ["TFC3,851230,SAC1,853646,,18676811,High Throughput,0.5,YPD"],
        header=HEADER + ",r.medium",
    )
    with pytest.raises(s.ConditionColumnFound, match="r.medium"):
        s.release_inventory(path, _sha256(path))


def test_a_stated_cell_line_is_counted(tmp_path: Path) -> None:
    """``r.cell_line`` is the one column that could carry a context, so it is counted."""
    path = _csv(
        tmp_path / "Yeast_SL.csv",
        ["TFC3,851230,SAC1,853646,BY4741,18676811,High Throughput,0.5"],
    )
    assert s.release_inventory(path, _sha256(path)).n_cell_line_stated == 1


def test_bytes_off_the_pin_are_refused(sl_csv: tuple[str, str]) -> None:
    """Every measurement is stated of verified bytes."""
    path, _ = sl_csv
    with pytest.raises(RawSha256MismatchError, match="sha256 mismatch"):
        s.release_inventory(path, "0" * 64)


def test_a_blank_pubmed_id_is_refused_by_name(tmp_path: Path) -> None:
    """Every row cites its PMID; a blank one is the decomposition's missing link."""
    path = _csv(
        tmp_path / "Yeast_SL.csv", ["TFC3,851230,SAC1,853646,,   ,High Throughput,0.5"]
    )
    with pytest.raises(s.BlankPubmedIdError, match="blank r.pubmed_id"):
        s.read_synlethdb_csv(path)


def test_the_pubmed_id_stays_text_so_a_multi_paper_row_survives(
    sl_csv: tuple[str, str],
) -> None:
    """Read as a number, one blank cell would turn every PMID into '111.0'."""
    path, _ = sl_csv
    frame = s.read_synlethdb_csv(path)
    assert frame["r.pubmed_id"].tolist() == [
        "18676811",
        "14764870",
        "24470217",
        "24125552;19918932",
    ]


def test_the_decomposition_counts_the_studies_to_load_instead(
    sl_csv: tuple[str, str],
) -> None:
    """Clause 3 on synthetic bytes: a ';'-joined row cites two studies, not one."""
    path, sha = sl_csv
    decomposition = s.source_decomposition(path, sha)
    assert decomposition.n_rows == 4
    assert decomposition.n_source_pubmed_ids == 5
    assert decomposition.n_multi_pubmed_rows == 1
    assert decomposition.rows_by_evidence == {
        "High Throughput": 1,
        "High Throughput|Low Throughput": 1,
        "Low Throughput": 1,
        "Text Mining": 1,
    }
    assert decomposition.pubmed_ids_by_evidence == {
        "High Throughput": 1,
        "High Throughput|Low Throughput": 1,
        "Low Throughput": 1,
        "Text Mining": 2,
    }


def test_the_score_is_a_function_of_the_evidence_label(sl_csv: tuple[str, str]) -> None:
    """Clause 4 on synthetic bytes: one score per label, not one per pair."""
    path, sha = sl_csv
    assert s.score_by_source(path, sha) == {
        "High Throughput": [0.5],
        "High Throughput|Low Throughput": [0.9],
        "Low Throughput": [0.8],
        "Text Mining": [0.25],
    }


def test_an_unscored_row_contributes_no_score(tmp_path: Path) -> None:
    """The SR file is mostly unscored, so a null must not read as a value."""
    path = _csv(
        tmp_path / "Yeast_SR.csv",
        [
            "TFC3,851230,SAC1,853646,,18676811,BioGRID,",
            "NTH2,852033,DAN4,850372,,24470217,BioGRID,-0.83",
        ],
    )
    sha = _sha256(path)
    assert s.release_inventory(path, sha).n_scored == 1
    assert s.score_by_source(path, sha) == {"BioGRID": [-0.83]}


# --------------------------------------------------------------------------- #
# Gene resolution (clause 5), the map a per-study loader reuses
# --------------------------------------------------------------------------- #
GFF_LINES = (
    "##gff-version 3",
    "chrI\tRefSeq\tgene\t1\t100\t.\t+\t.\tID=gene-A;Dbxref=GeneID:851230;"
    "locus_tag=YAL001C",
    "chrI\tRefSeq\tpseudogene\t200\t300\t.\t+\t.\tID=gene-B;Dbxref=GeneID:853646;"
    "locus_tag=YKL212W",
    "chrI\tRefSeq\tmRNA\t1\t100\t.\t+\t.\tID=rna-A;Dbxref=GeneID:999999;"
    "locus_tag=YZZ999W",
)


def _gff(tmp_path: Path, lines: tuple[str, ...] = GFF_LINES) -> str:
    path = tmp_path / "ncbi_genomic.gff"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_only_gene_and_pseudogene_features_map_an_entrez_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An mRNA carries the same GeneID; counting it would double-map the locus."""
    path = _gff(tmp_path)
    monkeypatch.setattr(s, "NCBI_GFF_SHA256", _sha256(path))
    assert s.load_entrez_to_orf(path) == {851230: "YAL001C", 853646: "YKL212W"}


def test_one_gene_id_naming_two_loci_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conflicting map would silently put records under the wrong ORF (#597)."""
    path = _gff(
        tmp_path,
        GFF_LINES
        + (
            "chrI\tRefSeq\tgene\t400\t500\t.\t+\t.\tID=gene-C;Dbxref=GeneID:851230;"
            "locus_tag=YBR001C",
        ),
    )
    monkeypatch.setattr(s, "NCBI_GFF_SHA256", _sha256(path))
    with pytest.raises(s.EntrezGeneIdConflictError, match="names both"):
        s.load_entrez_to_orf(path)


def test_a_feature_with_two_gene_ids_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two cross-references give no single answer, so none is chosen."""
    path = _gff(
        tmp_path,
        (
            "chrI\tRefSeq\tgene\t1\t100\t.\t+\t.\tID=gene-A;"
            "Dbxref=GeneID:851230,GeneID:853646;locus_tag=YAL001C",
        ),
    )
    monkeypatch.setattr(s, "NCBI_GFF_SHA256", _sha256(path))
    with pytest.raises(s.EntrezGeneIdConflictError, match="2 GeneID"):
        s.load_entrez_to_orf(path)


def test_a_gff_off_the_pin_is_refused(tmp_path: Path) -> None:
    """The annotation cannot be re-fetched, so the pin is the only acceptance test."""
    with pytest.raises(RawSha256MismatchError, match="sha256 mismatch"):
        s.load_entrez_to_orf(_gff(tmp_path))


def test_a_row_whose_side_is_absent_resolves_to_nothing() -> None:
    """Resolution by Entrez id alone, with no name fallback (clause 5)."""
    frame = pd.DataFrame(
        {"n1.identifier": [851230, 851230], "n2.identifier": [853646, 999]}
    )
    assert s.resolved_orf_pairs(frame, {851230: "YAL001C", 853646: "YKL212W"}) == [
        ("YAL001C", "YKL212W"),
        None,
    ]


def test_the_gff_path_hangs_off_the_genome_root(tmp_path: Path) -> None:
    """The pinned annotation sits beside the SGD release files, not in the tier set."""

    class _Genome:
        genome_root = str(tmp_path)

    from typing import cast

    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    assert s.ncbi_gff_path(cast(SCerevisiaeGenome, _Genome())) == osp.join(
        str(tmp_path), s.NCBI_GFF_RELPATH
    )


# --------------------------------------------------------------------------- #
# The pinned release
# --------------------------------------------------------------------------- #
@pytest.mark.data
@pytest.mark.skipif(DATA_ROOT is None, reason="DATA_ROOT unset")
def test_the_real_release_matches_the_pins_and_every_recorded_count() -> None:
    """Every number of clauses 1, 3 and 4, re-measured on the pinned release files."""
    assert DATA_ROOT is not None
    sl = osp.join(DATA_ROOT, "data/torchcell/syn_leth_db_yeast/raw", s.SL_CSV_NAME)
    sr = osp.join(DATA_ROOT, "data/torchcell/syn_rescue_db_yeast/raw", s.SR_CSV_NAME)
    for path in (sl, sr):
        if not osp.isfile(path):
            pytest.skip(f"{path} not on this machine")

    sl_inventory = s.release_inventory(sl, s.SL_CSV_SHA256)
    assert sl_inventory.n_rows == REAL_SL_ROWS
    assert sl_inventory.n_cell_line_stated == 0
    assert sl_inventory.n_scored == REAL_SL_ROWS
    assert sl_inventory.n_storable == 0

    sr_inventory = s.release_inventory(sr, s.SR_CSV_SHA256)
    assert sr_inventory.n_rows == REAL_SR_ROWS
    assert sr_inventory.n_cell_line_stated == 0
    assert sr_inventory.n_scored == REAL_SR_SCORED_ROWS
    assert sr_inventory.n_storable == 0

    sl_sources = s.source_decomposition(sl, s.SL_CSV_SHA256)
    assert sl_sources.n_source_pubmed_ids == REAL_SL_SOURCE_PUBMED_IDS
    assert sl_sources.rows_by_evidence == REAL_SL_ROWS_BY_EVIDENCE

    sr_sources = s.source_decomposition(sr, s.SR_CSV_SHA256)
    assert sr_sources.n_source_pubmed_ids == REAL_SR_SOURCE_PUBMED_IDS
    assert sr_sources.rows_by_evidence == {"BioGRID": REAL_SR_ROWS}

    assert s.score_by_source(sl, s.SL_CSV_SHA256) == REAL_SL_SCORES_BY_EVIDENCE
    assert list(s.score_by_source(sr, s.SR_CSV_SHA256)) == ["BioGRID"]


@pytest.mark.data
@pytest.mark.skipif(DATA_ROOT is None, reason="DATA_ROOT unset")
def test_every_quote_audits_against_the_pinned_paper() -> None:
    """Each clause's sentence is a verbatim substring of the mirrored OCR."""
    assert DATA_ROOT is not None
    library = osp.join(DATA_ROOT, "torchcell-library")
    paper = osp.join(library, s.SYNLETHDB_CITATION_KEY, "paper.md")
    if not osp.isfile(paper):
        pytest.skip(f"{paper} not on this machine")
    for sourced in s.SYNLETHDB_SOURCED_QUOTES:
        result = audit_sourced_value(sourced, library)
        assert result.passed, result.message
