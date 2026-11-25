# tests/torchcell/datasets/scerevisiae/test_cachera2023_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_cachera2023_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_cachera2023_synthetic.py
"""Hermetic build of the Cachera 2023 CRI-SPA betaxanthin loader on synthetic files.

``QD1_2_4_6.csv`` (per colony) and ``GA1_2_4_6.csv`` (per gene, only its ``WT`` row is
read) are written into ``<root>/raw/`` in a minimal column set, so PyG never calls
``download()``. No genome is read (#1045): every measured colony whose position names a
gene is a record under the token the ``gene`` cell releases, upper-cased; a systematic
name goes on ``KanMxDeletionPerturbation`` and any other name on
``ReleasedIdentifierPerturbation``.

Colonies (gene, repeat, plate, row, column, raw, corrected):

    TFC3       1 0 0 0   0.4   1.5    -> released symbol
    TFC3       2 0 0 0   0.4   1.1    -> released symbol, a second colony
    YBR001C    1 0 0 1   0.4   0.5    -> placed leaf
    ycr001w    1 0 0 2   0.4  -0.25   -> YCR001W on the placed leaf
    WT         4 0 0 3   0.4   0.2    -> control colony: the cassette, no deletion
    0          1 0 0 4   0.4   1.0    -> empty library well, no record
    (blank)    1 0 0 5   0.4   1.0    -> gene NaN, no record
    NTH2       1 0 0 6   0.4  (blank) -> withdrawn by the authors, no record
    OLDNAME    1 0 0 7  (blank)(blank)-> no colony measured, no record

Five records; the reference is the GA ``WT`` summary (0.2, std 0.4, count 4).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datasets.scerevisiae import cachera2023 as m

_QD_ROWS = [
    ["TFC3", "1", "0", "0", "0", "0.4", "1.5"],
    ["TFC3", "2", "0", "0", "0", "0.4", "1.1"],
    ["YBR001C", "1", "0", "0", "1", "0.4", "0.5"],
    ["ycr001w", "1", "0", "0", "2", "0.4", "-0.25"],
    ["WT", "4", "0", "0", "3", "0.4", "0.2"],
    ["0", "1", "0", "0", "4", "0.4", "1.0"],
    ["", "1", "0", "0", "5", "0.4", "1.0"],
    ["NTH2", "1", "0", "0", "6", "0.4", ""],
    ["OLDNAME", "1", "0", "0", "7", "", ""],
]
_GA_ROWS = [["TFC3", "1.3", "0.28", "2"], ["WT", "0.2", "0.4", "4"]]


def _root(tmp_path: Path, slug: str = "betaxanthin_cachera2023") -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    qd_header = ["gene", *m._POSITION, m._COLONY_RAW, m._COLONY_LEVEL]
    (root / "raw" / m.QD_FILENAME).write_text(
        "\n".join(",".join(r) for r in [qd_header, *_QD_ROWS]) + "\n"
    )
    ga_header = ["gene", m._LEVEL, m._STD, m._COUNT]
    (root / "raw" / m.DATA_FILENAME).write_text(
        "\n".join(",".join(r) for r in [ga_header, *_GA_ROWS]) + "\n"
    )
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> m.BetaxanthinCachera2023Dataset:
    return m.BetaxanthinCachera2023Dataset(root=str(_root(tmp_path)))


def test_five_colony_records_under_released_names(
    dataset: m.BetaxanthinCachera2023Dataset,
) -> None:
    """Nine positions give five records in source order: the two TFC3 colonies, the
    two systematic names on the placed leaf (``ycr001w`` upper-cased) and the WT
    control colony with no deletion. Each is one colony with its position as id.
    """
    assert len(dataset) == 5
    got = []
    for i in range(5):
        experiment = dataset[i]["experiment"]
        varying = [
            (p["perturbation_type"], p["systematic_gene_name"])
            for p in experiment["genotype"]["perturbations"]
            if p["perturbation_type"] != "gene_addition"
        ]
        phenotype = experiment["phenotype"]
        got.append(
            (
                varying,
                phenotype["metabolite_level"]["betaxanthin"],
                phenotype["n_replicates"]["betaxanthin"],
                phenotype["metabolite_level_se"],
                phenotype["replicate_id"],
            )
        )
    assert got == [
        (
            [("released_identifier", "TFC3")],
            1.5,
            1,
            None,
            "repeat1/plate0/row0/column0",
        ),
        (
            [("released_identifier", "TFC3")],
            1.1,
            1,
            None,
            "repeat2/plate0/row0/column0",
        ),
        ([("kanmx_deletion", "YBR001C")], 0.5, 1, None, "repeat1/plate0/row0/column1"),
        (
            [("kanmx_deletion", "YCR001W")],
            -0.25,
            1,
            None,
            "repeat1/plate0/row0/column2",
        ),
        ([], 0.2, 1, None, "repeat4/plate0/row0/column3"),
    ]


def test_the_reference_is_the_ga_wt_row(
    dataset: m.BetaxanthinCachera2023Dataset,
) -> None:
    """Every record's reference: level 0.2, SE 0.4 / sqrt(4) = 0.2, n 4."""
    for i in range(5):
        reference = dataset[i]["reference"]["phenotype_reference"]
        assert reference["metabolite_level"] == {"betaxanthin": 0.2}
        assert reference["metabolite_level_se"] == {"betaxanthin": 0.4 / math.sqrt(4)}
        assert reference["n_replicates"] == {"betaxanthin": 4}
        assert reference["replicate_id"] is None


def test_side_files(dataset: m.BetaxanthinCachera2023Dataset) -> None:
    """``data.csv`` carries one row per record; one shared reference covers all five;
    the ledger names the four positions without a record under their rules.
    """
    preprocess = Path(dataset.root) / "preprocess"
    assert (preprocess / "data.csv").read_text() == (
        "gene,replicate_id,repeat,plate,row,column,level\n"
        "TFC3,repeat1/plate0/row0/column0,1,0,0,0,1.5\n"
        "TFC3,repeat2/plate0/row0/column0,2,0,0,0,1.1\n"
        "YBR001C,repeat1/plate0/row0/column1,1,0,0,1,0.5\n"
        "YCR001W,repeat1/plate0/row0/column2,1,0,0,2,-0.25\n"
        ",repeat4/plate0/row0/column3,4,0,0,3,0.2\n"
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "CYP76AD1",
        "DOD",
        "TFC3",
        "YBR001C",
        "YBR249C",
        "YCR001W",
        "YPR060C",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2, 3, 4]]
    ledger = m.DropLog.model_validate_json(
        (preprocess / "dropped_records.json").read_text()
    )
    assert (
        ledger.kept_records,
        ledger.kept_control_records,
        ledger.dropped_records,
    ) == (5, 1, 4)
    assert [(r.rule, r.items) for r in ledger.rules] == [
        (m.RULE_NO_COLONY, ["repeat1/plate0/row0/column7"]),
        (m.RULE_WITHDRAWN, ["repeat1/plate0/row0/column6"]),
        (
            m.RULE_NO_GENE,
            ["repeat1/plate0/row0/column4", "repeat1/plate0/row0/column5"],
        ),
    ]


def test_process_rejects_a_present_file_with_the_wrong_sha256(
    dataset: m.BetaxanthinCachera2023Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild over the synthetic files under the real build-time check raises
    ``RawSha256MismatchError`` naming the first file and both digests.
    """
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    dest = Path(dataset.root) / "raw" / m.DATA_FILENAME
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    with pytest.raises(RawSha256MismatchError) as err:
        dataset.process()
    assert str(err.value) == (
        f"sha256 mismatch for {dest}: expected {m.DATA_SHA256}, observed {digest}"
    )
