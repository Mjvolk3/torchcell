# tests/torchcell/datasets/scerevisiae/test_xue2025.py
# [[tests.torchcell.datasets.scerevisiae.test_xue2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_xue2025.py
"""Xue 2025 FFA loader: genotype decoding, per-FFA replicate statistics, the WT reference.

``Supplementary Data 1_Raw titers.xlsx`` is written synthetically with openpyxl into
``<root>/raw/`` so PyG never calls ``download()``; ``process()`` runs for real against a
duck-typed genome stub carrying ``gene_set`` and ``alias_to_systematic``, the two
attributes ``_resolve_systematic`` reads. The stub is the authority for the ORF of every
common name used here (the ORFs are the SGD ones for these genes, but nothing in the test
depends on that).

Titer sheet fixture (col 0 = label, then 5 FFAs x 3 replicate columns in the module's
order C14:0, C16:0, C18:0, C16:1, C18:1); the WT row is placed second to show the
reference is found by label, not by position:

- ``+ve Ctrl``: [100,110,120] [200,200,200] [50,51,52] [8,10,12] [300,310,320]
- ``wt BY4741``: [10,12,14] [20,20,20] [5,6,7] [1,2,3] [30,40,50]
- ``F-G 5d``: third replicate blank everywhere; [1,3] [10,14] [5,5] [2,4] [7,9]
- ``P-S-Y 6dΔ`` (delta glyph): [4,4,4] [6,6,6] [8,8,8] [1,1,1] [9,9,9]
- ``T 4d``: one replicate only; 3, 4, 5, 6, 7
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
from pathlib import Path
from typing import Any, cast

import openpyxl
import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.gene_alias_resolution import (
    GeneNameRefusalReason,
    GeneNameRefused,
)
from torchcell.datasets.scerevisiae import xue2025 as m
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

_ORF = {
    "POX1": "YGL205W",
    "FAA1": "YOR317W",
    "FAA4": "YMR246W",
    "FKH1": "YIL131C",
    "GCN5": "YGR252W",
    "MED4": "YOR174W",
    "OPI1": "YHL020C",
    "RFX1": "YLR176C",
    "RGR1": "YLR071C",
    "RPD3": "YNL330C",
    "SPT3": "YDR392W",
    "YAP6": "YDR259C",
    "TFC7": "YOR110W",
}


class _StubGenome:
    """``gene_set`` (current ORFs), ``alias_to_systematic`` (common name -> [ORF]) and the
    ``feature_index`` standard-name map. ``TFC7`` is shaped as in R64 (#886): the
    standard name of YOR110W and an alias of YNL039W (BDP1), which the alias table
    lists FIRST, so a first-candidate resolver would store YNL039W.
    """

    gene_set = {*_ORF.values(), "YNL039W"}
    alias_to_systematic: dict[str, list[str]] = {
        **{k: [v] for k, v in _ORF.items()},
        "TFC7": ["YNL039W", "YOR110W"],
    }
    feature_index: dict[str, Any] = {
        "standard_to_ids": {k: [v] for k, v in _ORF.items()}
    }


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


_ABBREVIATIONS: list[tuple[str, str]] = [
    (gene, code) for code, gene in m._EXPECTED_CODE_TO_GENE.items()
]

Row = tuple[str, list[float | None]]
_ROWS: list[Row] = [
    ("+ve Ctrl", [100, 110, 120, 200, 200, 200, 50, 51, 52, 8, 10, 12, 300, 310, 320]),
    ("wt BY4741", [10, 12, 14, 20, 20, 20, 5, 6, 7, 1, 2, 3, 30, 40, 50]),
    ("F-G 5d", [1, 3, None, 10, 14, None, 5, 5, None, 2, 4, None, 7, 9, None]),
    ("P-S-Y 6dΔ", [4, 4, 4, 6, 6, 6, 8, 8, 8, 1, 1, 1, 9, 9, 9]),
    (
        "T 4d",
        [3, None, None, 4, None, None, 5, None, None, 6, None, None, 7, None, None],
    ),
]


def _write_workbook(
    raw: Path,
    rows: list[Row] = _ROWS,
    abbreviations: list[tuple[str, str]] = _ABBREVIATIONS,
) -> None:
    """Write the two-sheet workbook: ``Abbreviations`` (gene, code) and the titer sheet
    whose row 0 is a label row the loader skips.
    """
    workbook = openpyxl.Workbook()
    ab = workbook.active
    ab.title = m._ABBREV_SHEET
    for gene, code in abbreviations:
        ab.append([gene, code])
    titer = workbook.create_sheet(m._TITER_SHEET)
    header: list[str | None] = ["Strain"]
    for ffa in m._FFA_COLUMNS:
        header.extend([ffa, None, None])
    titer.append(header)
    for label, values in rows:
        titer.append([label, *values])
    workbook.save(raw / m.DATA_FILENAME)


def _root(
    tmp_path: Path,
    slug: str,
    rows: list[Row] = _ROWS,
    abbreviations: list[tuple[str, str]] = _ABBREVIATIONS,
) -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    _write_workbook(root / "raw", rows, abbreviations)
    return root


@pytest.fixture
def built(tmp_path: Path) -> m.FattyAcidXue2025Dataset:
    root = _root(tmp_path, "ffa_xue2025")
    return m.FattyAcidXue2025Dataset(root=str(root), genome=_genome())


_FFAS = ["C14:0", "C16:0", "C18:0", "C16:1", "C18:1"]
_ENVIRONMENT = Environment(
    media=Media(name="SC (FFA production)", state="liquid", is_synthetic=True),
    temperature=Temperature(value=30.0),
    aerobicity="aerobic",
)
_PUBLICATION = Publication(
    pubmed_id="23899824",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/23899824/",
    doi="10.1016/j.ymben.2013.07.003",
    doi_url="https://doi.org/10.1016/j.ymben.2013.07.003",
)


def _phenotype(levels: list[float], ses: list[float], n: int) -> MetabolitePhenotype:
    return MetabolitePhenotype(
        metabolite_level=dict(zip(_FFAS, levels, strict=True)),
        metabolite_level_se=dict(zip(_FFAS, ses, strict=True)),
        n_replicates=dict.fromkeys(_FFAS, n),
        measurement_type="titer_mg_per_l",
        target_metabolite_ids=None,
    )


def _genotype(genes: list[str]) -> Genotype:
    return Genotype(
        perturbations=[
            KanMxDeletionPerturbation(
                systematic_gene_name=_ORF[g], perturbed_gene_name=g
            )
            for g in genes
        ]
    )


def _pairs(record: dict[str, object]) -> list[tuple[str, str]]:
    """(ORF, perturbed name) per perturbation, in stored (systematic-name) order."""
    genotype = cast(dict[str, list[dict[str, str]]], record["genotype"])
    return [
        (p["systematic_gene_name"], p["perturbed_gene_name"])
        for p in genotype["perturbations"]
    ]


def test_one_record_per_non_wt_row_in_sheet_order(
    built: m.FattyAcidXue2025Dataset,
) -> None:
    """Five sheet rows give four records (the WT row is skipped) in sheet order, each with
    the three chassis deletions plus the decoded TF letters, perturbations sorted by ORF.

    ``+ve Ctrl`` is the chassis alone: means 110, 200, 51, 10, 310 with SE 10/sqrt(3),
    0, 1/sqrt(3), 2/sqrt(3), 10/sqrt(3) (sample SD over three replicates), n = 3. Its
    perturbations sort YGL205W (POX1), YMR246W (FAA4), YOR317W (FAA1).
    """
    assert len(built) == 4
    assert [_pairs(built[i]["experiment"]) for i in range(4)] == [
        [("YGL205W", "POX1"), ("YMR246W", "FAA4"), ("YOR317W", "FAA1")],
        [
            ("YGL205W", "POX1"),
            ("YGR252W", "GCN5"),
            ("YIL131C", "FKH1"),
            ("YMR246W", "FAA4"),
            ("YOR317W", "FAA1"),
        ],
        [
            ("YDR259C", "YAP6"),
            ("YDR392W", "SPT3"),
            ("YGL205W", "POX1"),
            ("YMR246W", "FAA4"),
            ("YNL330C", "RPD3"),
            ("YOR317W", "FAA1"),
        ],
        [
            ("YGL205W", "POX1"),
            ("YMR246W", "FAA4"),
            ("YOR110W", "TFC7"),
            ("YOR317W", "FAA1"),
        ],
    ]
    root3 = math.sqrt(3)
    assert (
        built[0]["experiment"]
        == MetaboliteExperiment(
            dataset_name="FattyAcidXue2025Dataset",
            genotype=_genotype(["POX1", "FAA1", "FAA4"]),
            environment=_ENVIRONMENT,
            phenotype=_phenotype(
                [110.0, 200.0, 51.0, 10.0, 310.0],
                [10.0 / root3, 0.0, 1.0 / root3, 2.0 / root3, 10.0 / root3],
                3,
            ),
        ).model_dump()
    )
    assert built[0]["publication"] == _PUBLICATION.model_dump()


def test_two_replicate_strain_uses_the_present_columns_only(
    built: m.FattyAcidXue2025Dataset,
) -> None:
    """``F-G 5d`` has a blank third replicate in every FFA: n = 2 per FFA, means 2, 12, 5,
    3, 8, and SE = |a - b| / 2 (sample SD of two values over sqrt(2)): 1, 2, 0, 1, 1.
    """
    phenotype = built[1]["experiment"]["phenotype"]
    assert (
        phenotype
        == _phenotype(
            [2.0, 12.0, 5.0, 3.0, 8.0], [1.0, 2.0, 0.0, 1.0, 1.0], 2
        ).model_dump()
    )


def test_single_replicate_strain_stores_nan_se(
    built: m.FattyAcidXue2025Dataset,
) -> None:
    """``T 4d`` has one replicate per FFA: the level is that value (3, 4, 5, 6, 7), n = 1,
    and the sample SD is undefined so every SE is NaN.
    """
    phenotype = built[3]["experiment"]["phenotype"]
    assert phenotype["metabolite_level"] == {
        "C14:0": 3.0,
        "C16:0": 4.0,
        "C18:0": 5.0,
        "C16:1": 6.0,
        "C18:1": 7.0,
    }
    assert phenotype["n_replicates"] == dict.fromkeys(_FFAS, 1)
    assert sorted(phenotype["metabolite_level_se"]) == sorted(_FFAS)
    assert all(math.isnan(se) for se in phenotype["metabolite_level_se"].values())
    assert phenotype["measurement_type"] == "titer_mg_per_l"
    assert phenotype["target_metabolite_ids"] is None


def test_wt_row_is_the_shared_measured_reference(
    built: m.FattyAcidXue2025Dataset,
) -> None:
    """The ``wt BY4741`` row (means 12, 20, 6, 2, 40; SE 2/sqrt(3), 0, 1/sqrt(3),
    1/sqrt(3), 10/sqrt(3); n = 3) is the phenotype reference of every record, and the
    reference index has one entry covering [0, 1, 2, 3].
    """
    root3 = math.sqrt(3)
    expected = MetaboliteExperimentReference(
        dataset_name="FattyAcidXue2025Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=_phenotype(
            [12.0, 20.0, 6.0, 2.0, 40.0],
            [2.0 / root3, 0.0, 1.0 / root3, 1.0 / root3, 10.0 / root3],
            3,
        ),
    ).model_dump()
    assert [built[i]["reference"] for i in range(4)] == [expected] * 4
    index = json.loads(
        (
            Path(built.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2, 3]]
    assert index[0]["reference"]["phenotype_reference"]["metabolite_level"] == {
        "C14:0": 12.0,
        "C16:0": 20.0,
        "C18:0": 6.0,
        "C16:1": 2.0,
        "C18:1": 40.0,
    }


def test_summary_csv_gene_set_and_build_manifest(
    built: m.FattyAcidXue2025Dataset,
) -> None:
    """``preprocess/data.csv`` lists each record's label, deletion count and ``;``-joined
    sorted ORFs; ``gene_set.json`` is the nine ORFs sorted; the manifest names the slug,
    the loader and this host.
    """
    preprocess = Path(built.root) / "preprocess"
    summary = pd.read_csv(preprocess / "data.csv")
    assert summary.columns.tolist() == ["genotype", "n_deletions", "orfs"]
    assert summary.values.tolist() == [
        ["+ve Ctrl", 3, "YGL205W;YMR246W;YOR317W"],
        ["F-G 5d", 5, "YGL205W;YGR252W;YIL131C;YMR246W;YOR317W"],
        ["P-S-Y 6dΔ", 6, "YDR259C;YDR392W;YGL205W;YMR246W;YNL330C;YOR317W"],
        ["T 4d", 4, "YGL205W;YMR246W;YOR110W;YOR317W"],
    ]
    assert built.df is not None
    assert built.df.shape == (4, 3)
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YDR259C",
        "YDR392W",
        "YGL205W",
        "YGR252W",
        "YIL131C",
        "YMR246W",
        "YNL330C",
        "YOR110W",
        "YOR317W",
    ]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "ffa_xue2025"
    assert manifest["loader_class"] == "FattyAcidXue2025Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.xue2025"
    assert manifest["hostname"] == socket.gethostname()
    assert {"MetaboliteExperiment", "KanMxDeletionPerturbation"} <= set(
        manifest["closure"]
    )


def test_resolve_systematic_prefers_gene_set_then_alias_then_raises(
    built: m.FattyAcidXue2025Dataset,
) -> None:
    """A name already in ``gene_set`` returns itself, in any case (#886 retired the
    case-sensitive membership test); a common name with one candidate follows the alias
    table (``pox1`` -> YGL205W); ``TFC7`` resolves through its pin to YOR110W, not to the
    alias table's first candidate; an unknown name raises a typed refusal.
    """
    assert built.experiment_class is MetaboliteExperiment
    assert built.reference_class is MetaboliteExperimentReference
    assert built._resolve_systematic(" YGL205W ") == "YGL205W"
    assert built._resolve_systematic("ygl205w") == "YGL205W"
    assert built._resolve_systematic("pox1") == "YGL205W"
    assert built._resolve_systematic("TFC7") == "YOR110W"
    with pytest.raises(GeneNameRefused) as info:
        built._resolve_systematic("NOPE")
    assert info.value.refusal.reason is GeneNameRefusalReason.NOT_IN_GENOME
    assert str(info.value) == (
        "gene name 'NOPE' refused (not_in_genome): no live gene, alias or standard "
        "name matches; candidates []"
    )


def test_tfc7_pin_is_the_r64_standard_name_and_is_rechecked(tmp_path: Path) -> None:
    """#886: the one pin is TFC7 -> YOR110W under the SGD standard-name rule, quoting
    the start of YOR110W's gene row in the R64-4-1 GFF. When the injected genome names a
    different standard-name owner, the build refuses before writing anything.
    """
    assert {
        alias: (pin.systematic_name, pin.rule.value, pin.quote, pin.provenance.sha256)
        for alias, pin in m.AMBIGUOUS_ALIAS_PINS.items()
    } == {
        "TFC7": (
            "YOR110W",
            "sgd_standard_name",
            "ID=YOR110W;Name=YOR110W;gene=TFC7;",
            "64f61e3153083a8ef6d853721c9e83e4469cdc120883ec281e51a0df4ba390fa",
        )
    }

    class _Contradicting(_StubGenome):
        feature_index: dict[str, Any] = {
            "standard_to_ids": {**_StubGenome.feature_index["standard_to_ids"]}
            | {"TFC7": ["YNL039W"]}
        }

    root = _root(tmp_path, "contradicted")
    with pytest.raises(GeneNameRefused) as info:
        m.FattyAcidXue2025Dataset(
            root=str(root), genome=cast(SCerevisiaeGenome, _Contradicting())
        )
    assert info.value.refusal.reason is GeneNameRefusalReason.PIN_RULE_CONTRADICTED
    assert info.value.refusal.candidates == ["YNL039W", "YOR110W"]
    assert not (root / "processed" / "lmdb").exists()


def test_decode_genotype_labels_and_its_two_error_paths(
    built: m.FattyAcidXue2025Dataset,
) -> None:
    """``wt BY4741`` and ``BY4741`` decode to no deletions, ``+ve Ctrl`` to the chassis
    triple, ``G-O-T 6d`` to the triple plus GCN5, OPI1, TFC7, and a trailing ``∆`` (the
    other delta code point) is stripped. Two TF letters declared as ``6d`` raise the count
    mismatch (2 + 3 != 6); a label with no ``<letters> <N>`` core is unparseable.
    """
    code_to_gene = built._code_to_gene()
    assert code_to_gene == m._EXPECTED_CODE_TO_GENE
    assert built._decode_genotype("wt BY4741", code_to_gene) == []
    assert built._decode_genotype("BY4741 ", code_to_gene) == []
    assert built._decode_genotype("+ve Ctrl", code_to_gene) == ["POX1", "FAA1", "FAA4"]
    assert built._decode_genotype("G-O-T 6d", code_to_gene) == [
        "POX1",
        "FAA1",
        "FAA4",
        "GCN5",
        "OPI1",
        "TFC7",
    ]
    assert built._decode_genotype("M 4d∆", code_to_gene) == [
        "POX1",
        "FAA1",
        "FAA4",
        "MED4",
    ]
    with pytest.raises(
        RuntimeError,
        match=r"'F-G 6d': TF letters \(2\) \+ baseline \(3\) != declared deletion count 6",
    ):
        built._decode_genotype("F-G 6d", code_to_gene)
    with pytest.raises(
        RuntimeError, match="unparseable genotype string 'weird' \\(core 'weir'\\)"
    ):
        built._decode_genotype("weird", code_to_gene)


def test_wt_row_count_must_be_exactly_one(tmp_path: Path) -> None:
    """No WT row raises ``found 0``; both ``wt BY4741`` and ``BY4741`` present raises
    ``found 2`` and lists the two labels.
    """
    no_wt = [row for row in _ROWS if row[0] != "wt BY4741"]
    with pytest.raises(
        RuntimeError, match=r"expected exactly 1 wild-type row, found 0"
    ):
        m.FattyAcidXue2025Dataset(
            root=str(_root(tmp_path, "none", no_wt)), genome=_genome()
        )
    two_wt = [*_ROWS, ("BY4741", _ROWS[1][1])]
    with pytest.raises(
        RuntimeError,
        match=r"expected exactly 1 wild-type row, found 2: \['wt BY4741', 'BY4741'\]",
    ):
        m.FattyAcidXue2025Dataset(
            root=str(_root(tmp_path, "two", two_wt)), genome=_genome()
        )


def test_abbreviation_sheet_must_match_the_expected_code_map(tmp_path: Path) -> None:
    """Dropping the TFC7 row from the Abbreviations sheet raises before any titer is read;
    the message carries the nine-entry map that was read.
    """
    short = [(gene, code) for gene, code in _ABBREVIATIONS if code != "T"]
    root = _root(tmp_path, "abbr", abbreviations=short)
    with pytest.raises(
        RuntimeError,
        match="Abbreviations sheet does not match the expected TF code map: got "
        + r"\{'F': 'FKH1', 'G': 'GCN5', 'M': 'MED4', 'O': 'OPI1', 'X': 'RFX1', 'R': 'RGR1', "
        + r"'P': 'RPD3', 'S': 'SPT3', 'Y': 'YAP6'\}",
    ):
        m.FattyAcidXue2025Dataset(root=str(root), genome=_genome())


def test_ffa_with_no_replicate_values_raises(tmp_path: Path) -> None:
    """A ``M 4d`` row whose three C18:1 cells are all blank raises naming the FFA and the
    row label (the other four FFAs are complete, so the failure is that column alone).
    """
    blank_c181: list[float | None] = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        1,
        2,
        3,
        None,
        None,
        None,
    ]
    rows = [*_ROWS, ("M 4d", blank_c181)]
    with pytest.raises(
        RuntimeError, match="FFA C18:1 has no replicate values in row 'M 4d'"
    ):
        m.FattyAcidXue2025Dataset(
            root=str(_root(tmp_path, "blank", rows)), genome=_genome()
        )


def test_genome_is_required(tmp_path: Path) -> None:
    """With the raw workbook present and ``genome=None``, ``process()`` raises before
    reading the sheets.
    """
    with pytest.raises(
        RuntimeError,
        match="FattyAcidXue2025Dataset requires an injected SCerevisiaeGenome",
    ):
        m.FattyAcidXue2025Dataset(root=str(_root(tmp_path, "nogenome")), genome=None)


def test_download_reads_the_library_mirror_and_verifies_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no raw file, ``download()`` copies ``$DATA_ROOT/torchcell-library/xue2025/
    data/<xlsx>`` and checks the pinned digest: a missing mirror file raises naming the
    path; ``b"not the real workbook"`` is rejected with ``RawSha256MismatchError``
    naming the mirror file, the pin and its own sha256 (55b4751d...) BEFORE the copy, so
    ``raw/`` stays empty.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = data_root / "torchcell-library" / "xue2025" / "data"
    with pytest.raises(
        RuntimeError, match=f"library mirror data file not found: {mirror}"
    ):
        m.FattyAcidXue2025Dataset(root=str(tmp_path / "a"), genome=_genome())
    mirror.mkdir(parents=True)
    (mirror / m.DATA_FILENAME).write_bytes(b"not the real workbook")
    digest = hashlib.sha256(b"not the real workbook").hexdigest()
    assert digest.startswith("55b4751d")
    with pytest.raises(RawSha256MismatchError) as err:
        m.FattyAcidXue2025Dataset(root=str(tmp_path / "b"), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {mirror / m.DATA_FILENAME}: expected {m.DATA_SHA256}, "
        f"observed {digest}"
    )
    assert list((tmp_path / "b" / "raw").iterdir()) == []


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): with the workbook already in ``raw/`` PyG skips
    ``download()``, so ``process()`` verifies it against ``DATA_SHA256`` first and raises
    ``RawSha256MismatchError`` naming it and both digests before a sheet is read; no
    store is written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(m, [m.DATA_FILENAME])
    raw = staged.root / "raw" / m.DATA_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        m.FattyAcidXue2025Dataset(root=str(staged.root), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected {m.DATA_SHA256}, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


# Phase 24: the verified-copy log line and the inert generic hooks


def test_download_copies_a_matching_mirror_file_and_logs_the_verified_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A mirror file whose digest matches the pin is copied byte for byte and logged.

    ``DATA_SHA256`` is patched to ``sha256(b"pinned titers")``; ``download`` is called
    unbound on a namespace holding only ``raw_dir``. The INFO line is
    ``Verified <raw_dir>/<file> (sha256 <digest>)``.
    """
    import logging
    from types import SimpleNamespace

    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = data_root / "torchcell-library" / m._LIBRARY_CITATION_KEY / "data"
    mirror.mkdir(parents=True)
    (mirror / m.DATA_FILENAME).write_bytes(b"pinned titers")
    digest = hashlib.sha256(b"pinned titers").hexdigest()
    monkeypatch.setattr(m, "DATA_SHA256", digest)
    raw_dir = tmp_path / "root" / "raw"
    with caplog.at_level(logging.INFO, logger=m.__name__):
        m.FattyAcidXue2025Dataset.download(
            cast(Any, SimpleNamespace(raw_dir=str(raw_dir)))
        )
    dest = raw_dir / m.DATA_FILENAME
    assert dest.read_bytes() == b"pinned titers"
    assert [r.getMessage() for r in caplog.records if r.name == m.__name__] == [
        f"Verified {dest} (sha256 {digest})"
    ]


def test_generic_hooks_pass_the_frame_through_and_refuse_create_experiment() -> None:
    """``preprocess_raw`` returns the frame it was given (``is``); ``create_experiment``
    raises a bare ``NotImplementedError`` (empty message).
    """
    frame = pd.DataFrame({"a": [1]})
    cls = m.FattyAcidXue2025Dataset
    assert cls.preprocess_raw(cast(Any, None), frame, {"k": 1}) is frame
    with pytest.raises(NotImplementedError) as err:
        cls.create_experiment(cast(Any, None))
    assert str(err.value) == ""
