# tests/torchcell/tasks/test_ecoli.py
# [[tests.torchcell.tasks.test_ecoli]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/tasks/test_ecoli.py
"""Tests for ``torchcell.tasks.ecoli`` with fake datasets and a fake genome.

No test opens a real dataset, reads ``DATA_ROOT`` or loads a genome. The fake genome
resolves nine stored names:

- ``geneA`` -> ``bA`` and ``geneB`` -> ``bB`` (current, with proteins ``MKV`` and
  ``MAT``); ``oldC`` -> ``bC`` and ``aliasA`` -> ``bA`` (renamed; ``bC`` has ``MQQ``);
- ``geneN`` -> ``bN`` (current, no protein) and ``geneE`` -> ``bE`` (current, empty
  protein sequence);
- ``geneR`` (retired, kept as ``geneR``), ``geneM`` (ambiguous, no systematic name) and
  ``geneF`` (non-gene feature ``featF``).

The four shipped split files are read through ``read_splits``; their row counts, split
columns and the part sizes of ``random_s0`` are pinned to the files whose sha256 is in
``SPLIT_SHA256``.
"""

from __future__ import annotations

import gzip
import hashlib
import io
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.tasks import ecoli
from torchcell.tasks.ecoli import (
    ECOLI_TASKS,
    SPLIT_SHA256,
    build_task,
    dataset_root,
    load_task,
    match_records,
    open_dataset,
    read_labels,
    read_splits,
    write_splits,
)
from torchcell.tasks.task import BenchmarkTask, TaskSpec

GROWTH = "ecoli-growth-rate-campos2018"
MORPHOLOGY = "ecoli-morphology-campos2018"
BABU = "ecoli-gene-interaction-babu2014"
FUHRER = "ecoli-metabolome-fuhrer2017"
BW25113 = "ecoli_K12_BW25113_ASM75055v1"
SINGLE_COLUMNS = [
    "record_index",
    "key",
    *(f"random_s{s}" for s in range(5)),
    *(f"{kind}_s{s}" for kind in ("seqid30", "seqid50", "position") for s in range(3)),
]
PAIR_COLUMNS = [
    "record_index",
    "key",
    *(f"random_s{s}" for s in range(5)),
    *(
        f"{kind}gene_holdout_s{s}"
        for kind in ("", "seqid30_", "seqid50_", "position_")
        for s in range(3)
    ),
]
#: (rows, columns, part sizes of random_s0) of each shipped split file.
SHIPPED = {
    GROWTH: (3573, SINGLE_COLUMNS, {"train": 2859, "val": 357, "test": 357}),
    MORPHOLOGY: (3573, SINGLE_COLUMNS, {"train": 2859, "val": 357, "test": 357}),
    BABU: (40658, PAIR_COLUMNS, {"train": 32527, "val": 4066, "test": 4065}),
    FUHRER: (3624, SINGLE_COLUMNS, {"train": 2900, "val": 362, "test": 362}),
}
SINGLE_GENE_TASKS = [GROWTH, MORPHOLOGY, FUHRER]
#: record_index of the last sample of each shipped split file.
LAST_RECORD = {GROWTH: 3662, MORPHOLOGY: 3662, BABU: 42704, FUHRER: 3805}


class FakeProtein:
    """A protein record: only ``seq`` is read."""

    def __init__(self, seq: str) -> None:
        """Hold the sequence."""
        self.seq = seq


class FakeGene:
    """A gene record: only ``protein`` is read."""

    def __init__(self, protein: FakeProtein | None) -> None:
        """Hold the protein, or None for a gene without one."""
        self.protein = protein


def _resolution(
    name: str, status: GeneNameStatus, systematic: str | None
) -> GeneNameResolution:
    return GeneNameResolution(
        input_name=name, status=status, systematic_name=systematic
    )


class FakeGenome:
    """The nine resolutions and five genes listed in the module docstring."""

    ASSEMBLY_SET = BW25113

    def __init__(self, assembly_set: str = BW25113) -> None:
        """Build the fake under ``assembly_set``; ``lookups`` logs gene reads."""
        self.ASSEMBLY_SET = assembly_set
        self.lookups: list[str] = []
        self._resolutions = {
            "geneA": _resolution("geneA", GeneNameStatus.CURRENT, "bA"),
            "geneB": _resolution("geneB", GeneNameStatus.CURRENT, "bB"),
            "oldC": _resolution("oldC", GeneNameStatus.RENAMED, "bC"),
            "aliasA": _resolution("aliasA", GeneNameStatus.RENAMED, "bA"),
            "geneN": _resolution("geneN", GeneNameStatus.CURRENT, "bN"),
            "geneE": _resolution("geneE", GeneNameStatus.CURRENT, "bE"),
            "geneR": _resolution("geneR", GeneNameStatus.RETIRED, "geneR"),
            "geneM": _resolution("geneM", GeneNameStatus.AMBIGUOUS, None),
            "geneF": _resolution("geneF", GeneNameStatus.NON_GENE_FEATURE, "featF"),
        }
        self._genes = {
            "bA": FakeGene(FakeProtein("MKV")),
            "bB": FakeGene(FakeProtein("MAT")),
            "bC": FakeGene(FakeProtein("MQQ")),
            "bN": FakeGene(None),
            "bE": FakeGene(FakeProtein("")),
        }

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        """The fixed resolution of ``name``."""
        return self._resolutions[name]

    def __getitem__(self, locus: str) -> FakeGene:
        """The gene at ``locus``, logged in ``lookups``."""
        self.lookups.append(locus)
        return self._genes[locus]


class FakeDataset:
    """Records built from (stored names, symbols, phenotype) triples."""

    def __init__(
        self, records: list[tuple[list[str], list[str], dict[str, Any]]]
    ) -> None:
        """Wrap each triple as a record with one perturbation per stored name."""
        self._items = [
            {
                "experiment": {
                    "genotype": {
                        "perturbations": [
                            {"systematic_gene_name": name, "perturbed_gene_name": sym}
                            for name, sym in zip(names, symbols, strict=True)
                        ]
                    },
                    "phenotype": phenotype,
                }
            }
            for names, symbols, phenotype in records
        ]
        self.gene_set = sorted({n for names, _, _ in records for n in names})

    def __len__(self) -> int:
        """Number of records."""
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """The record at ``index``."""
        return self._items[index]


def single_gene_dataset() -> FakeDataset:
    return FakeDataset(
        [
            (["geneA"], ["aaaA"], {"fitness": 0.5}),
            (["geneA", "geneB"], ["aaaA", "bbbB"], {"fitness": 0.6}),
            (["geneR"], ["rrrR"], {"fitness": 0.7}),
            (["geneM"], ["mmmM"], {"fitness": 0.8}),
            (["geneN"], ["nnnN"], {"fitness": 0.9}),
            (["geneE"], ["eeeE"], {"fitness": 1.0}),
            (["oldC"], ["cccC"], {"fitness": 1.25}),
            (["geneF"], ["fffF"], {"fitness": 1.5}),
            ([], [], {"fitness": 2.0}),
        ]
    )


NOT_RESOLVED = "a gene does not resolve to a gene of the genome"
NO_PROTEIN = "a gene has no protein sequence in the genome"


def test_match_records_single_gene_exact() -> None:
    genome = FakeGenome()
    out = match_records(single_gene_dataset(), genome, n_genes=1)
    expected = pd.DataFrame(
        {
            "record_index": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "n_perturbations": [1, 2, 1, 1, 1, 1, 1, 1, 0],
            "stored": [
                "geneA",
                "geneA|geneB",
                "geneR",
                "geneM",
                "geneN",
                "geneE",
                "oldC",
                "geneF",
                "",
            ],
            "symbol": [
                "aaaA",
                "aaaA|bbbB",
                "rrrR",
                "mmmM",
                "nnnN",
                "eeeE",
                "cccC",
                "fffF",
                "",
            ],
            "locus_tag": ["bA", "bA|bB", "geneR", "", "bN", "bE", "bC", "featF", ""],
            "status": [
                "current",
                "current|current",
                "retired",
                "ambiguous",
                "current",
                "current",
                "renamed",
                "non_gene_feature",
                "",
            ],
            "eligible": [True, False, False, False, False, False, True, False, False],
            "exclusion_reason": [
                "",
                "perturbs 2 genes, the task is about 1",
                NOT_RESOLVED,
                NOT_RESOLVED,
                NO_PROTEIN,
                NO_PROTEIN,
                "",
                NOT_RESOLVED,
                "perturbs 0 genes, the task is about 1",
            ],
        }
    )
    pd.testing.assert_frame_equal(out, expected)
    # Each resolved locus is looked up in the genome once.
    assert genome.lookups == ["bA", "bN", "bE", "bC"]


def test_match_records_gene_pair_exact() -> None:
    dataset = FakeDataset(
        [
            (["geneA", "geneB"], ["aaaA", "bbbB"], {"gene_interaction": -0.1}),
            (["geneA", "aliasA"], ["aaaA", "aaaA2"], {"gene_interaction": 0.2}),
            (["geneA"], ["aaaA"], {"gene_interaction": 0.3}),
            (["geneA", "geneN"], ["aaaA", "nnnN"], {"gene_interaction": 0.4}),
            (["geneB", "geneR"], ["bbbB", "rrrR"], {"gene_interaction": 0.5}),
            (["oldC", "geneB"], ["cccC", "bbbB"], {"gene_interaction": 0.6}),
        ]
    )
    out = match_records(dataset, FakeGenome(), n_genes=2)
    assert out["locus_tag"].tolist() == [
        "bA|bB",
        "bA|bA",
        "bA",
        "bA|bN",
        "bB|geneR",
        "bC|bB",
    ]
    assert out["status"].tolist() == [
        "current|current",
        "current|renamed",
        "current",
        "current|current",
        "current|retired",
        "renamed|current",
    ]
    assert out["eligible"].tolist() == [True, False, False, False, False, True]
    assert out["exclusion_reason"].tolist() == [
        "",
        "the perturbed genes resolve to the same gene",
        "perturbs 1 genes, the task is about 2",
        NO_PROTEIN,
        NOT_RESOLVED,
        "",
    ]


def test_read_labels_scalar() -> None:
    y, targets = read_labels(
        single_gene_dataset(), ECOLI_TASKS[GROWTH], np.array([6, 0, 2], dtype=np.int64)
    )
    assert targets == ["fitness"]
    assert y.dtype == np.float64
    np.testing.assert_array_equal(y, np.array([[1.25], [0.5], [0.7]]))


def test_read_labels_dict_with_missing_key_gives_nan() -> None:
    dataset = FakeDataset(
        [
            (["geneA"], ["aaaA"], {"morphology": {"length": 2.0, "width": 1.0}}),
            (["geneB"], ["bbbB"], {"morphology": {"width": 0.8, "area": 3.0}}),
            (["oldC"], ["cccC"], {"morphology": {"length": None, "area": 4.5}}),
        ]
    )
    y, targets = read_labels(
        dataset, ECOLI_TASKS[MORPHOLOGY], np.array([0, 1, 2], dtype=np.int64)
    )
    assert targets == ["length", "width", "area"]
    np.testing.assert_array_equal(
        y, np.array([[2.0, 1.0, np.nan], [np.nan, 0.8, 3.0], [np.nan, np.nan, 4.5]])
    )


def test_dataset_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "/fake/root")
    assert dataset_root("toy_slug") == "/fake/root/data/torchcell/toy_slug"
    monkeypatch.delenv("DATA_ROOT")
    assert dataset_root("toy_slug") == "./data/torchcell/toy_slug"


def test_open_dataset_calls_loader_with_root(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ECOLI_TASKS[GROWTH].model_copy(
        update={"dataset_module": "types", "dataset_class": "SimpleNamespace"}
    )
    assert open_dataset(spec, root="/given").root == "/given"
    monkeypatch.setenv("DATA_ROOT", "/fake/root")
    assert (
        open_dataset(spec).root
        == "/fake/root/data/torchcell/ecoli_growth_rate_campos2018"
    )


def _patch_sources(monkeypatch: pytest.MonkeyPatch, genome: FakeGenome) -> None:
    monkeypatch.setattr(ecoli, "open_dataset", lambda spec, root: single_gene_dataset())
    monkeypatch.setattr(ecoli, "genome_for", lambda dataset: genome)


def test_build_task_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    genome = FakeGenome()
    _patch_sources(monkeypatch, genome)
    task = build_task(GROWTH)
    assert task.spec == ECOLI_TASKS[GROWTH]
    assert task.genome is genome
    assert task.split_names == []
    pd.testing.assert_frame_equal(
        task.samples,
        pd.DataFrame(
            {
                "record_index": [0, 6],
                "key": ["bA", "bC"],
                "n_perturbations": [1, 1],
                "stored": ["geneA", "oldC"],
                "symbol": ["aaaA", "cccC"],
                "locus_tag": ["bA", "bC"],
                "status": ["current", "renamed"],
            }
        ),
    )
    np.testing.assert_array_equal(task.genes, np.array([["bA"], ["bC"]]))
    np.testing.assert_array_equal(task.y, np.array([[0.5], [1.25]]))
    assert task.targets == ["fitness"]
    assert task.excluded.columns.tolist() == [
        "record_index",
        "n_perturbations",
        "stored",
        "symbol",
        "locus_tag",
        "status",
        "exclusion_reason",
    ]
    assert task.excluded["record_index"].tolist() == [1, 2, 3, 4, 5, 7, 8]
    assert task.excluded.index.tolist() == [0, 1, 2, 3, 4, 5, 6]


def test_build_task_rejects_other_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(monkeypatch, FakeGenome(assembly_set="other_set"))
    with pytest.raises(ValueError) as info:
        build_task(GROWTH)
    assert str(info.value) == (
        f"{GROWTH} records name other_set, the task pins {BW25113}"
    )


def test_build_task_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        build_task("no-such-task")


def test_load_task_attaches_pinned_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(monkeypatch, FakeGenome())
    pinned = pd.DataFrame(
        {"record_index": ["0", "6"], "key": ["bA", "bC"], "random_s0": ["val", "test"]}
    )
    monkeypatch.setattr(ecoli, "read_splits", lambda name: pinned)
    task = load_task(GROWTH)
    assert task.split_names == ["random_s0"]
    split = task.split("random_s0")
    np.testing.assert_array_equal(split.train, np.array([], dtype=np.int64))
    np.testing.assert_array_equal(split.val, np.array([0]))
    assert split.sizes() == {"train": 0, "val": 1, "test": 1}


def test_load_task_rejects_mismatched_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(monkeypatch, FakeGenome())
    pinned = pd.DataFrame(
        {
            "record_index": ["0", "6", "7"],
            "key": ["bA", "bC", "featF"],
            "random_s0": ["train", "val", "test"],
        }
    )
    monkeypatch.setattr(ecoli, "read_splits", lambda name: pinned)
    with pytest.raises(ValueError) as info:
        load_task(GROWTH)
    assert str(info.value) == (
        f"the samples built for {GROWTH} differ from the pinned split file's "
        "(2 built, 3 pinned)"
    )


def _toy_task() -> BenchmarkTask:
    spec = TaskSpec(**{**ECOLI_TASKS[GROWTH].model_dump(), "name": "toy"})
    samples = pd.DataFrame({"record_index": [0, 6, 9], "key": ["bA", "bC", "bB"]})
    return BenchmarkTask(
        spec,
        samples,
        np.array([["bA"], ["bC"], ["bB"]]),
        np.zeros((3, 1)),
        ["fitness"],
        pd.DataFrame(),
        None,
    )


TOY_SPLITS = pd.DataFrame(
    {"random_s0": ["train", "test", "val"], "position_s0": ["val", "train", "test"]}
)
TOY_CSV = (
    "record_index,key,random_s0,position_s0\n"
    "0,bA,train,val\n"
    "6,bC,test,train\n"
    "9,bB,val,test\n"
)


def test_write_splits_is_byte_stable_and_round_trips(tmp_path: Path) -> None:
    first, second = tmp_path / "a.splits.csv.gz", tmp_path / "b.splits.csv.gz"
    digest_a = write_splits(str(first), _toy_task(), TOY_SPLITS)
    digest_b = write_splits(str(second), _toy_task(), TOY_SPLITS)
    data = first.read_bytes()
    assert digest_a == digest_b
    assert data == second.read_bytes()
    assert digest_a == hashlib.sha256(data).hexdigest()
    # gzip header: no FNAME flag (byte 3) and mtime zero (bytes 4-7).
    assert data[:2] == b"\x1f\x8b"
    assert data[3] == 0
    assert data[4:8] == b"\x00\x00\x00\x00"
    assert gzip.decompress(data).decode("utf-8") == TOY_CSV
    frame = pd.read_csv(io.BytesIO(gzip.decompress(data)), dtype=str)
    pd.testing.assert_frame_equal(
        frame,
        pd.DataFrame(
            {
                "record_index": ["0", "6", "9"],
                "key": ["bA", "bC", "bB"],
                "random_s0": ["train", "test", "val"],
                "position_s0": ["val", "train", "test"],
            }
        ),
    )


def test_task_registry_matches_shipped_files() -> None:
    assert list(ECOLI_TASKS) == [GROWTH, MORPHOLOGY, BABU, FUHRER]
    assert sorted(SPLIT_SHA256) == sorted(ECOLI_TASKS)
    assert {name: spec.kind for name, spec in ECOLI_TASKS.items()} == {
        GROWTH: "single_gene",
        MORPHOLOGY: "single_gene",
        BABU: "gene_pair",
        FUHRER: "single_gene",
    }
    shipped = sorted(
        entry.name
        for entry in resources.files("torchcell.tasks").joinpath("data").iterdir()
        if entry.name.endswith(".splits.csv.gz")
    )
    assert shipped == sorted(f"{name}.splits.csv.gz" for name in ECOLI_TASKS)
    # The two Campos 2018 tasks share one sample set, hence one split file.
    assert SPLIT_SHA256[GROWTH] == SPLIT_SHA256[MORPHOLOGY]


@pytest.mark.parametrize("name", list(SHIPPED))
def test_read_splits_shipped_file(name: str) -> None:
    data = (
        resources.files("torchcell.tasks")
        .joinpath(f"data/{name}.splits.csv.gz")
        .read_bytes()
    )
    assert hashlib.sha256(data).hexdigest() == SPLIT_SHA256[name]
    frame = read_splits(name)
    rows, columns, random_s0 = SHIPPED[name]
    assert frame.shape == (rows, len(columns))
    assert frame.columns.tolist() == columns
    indices = frame["record_index"].astype(int)
    # Record indices run from 0 to the last sample in increasing order, one per row.
    assert (int(indices.iloc[0]), int(indices.iloc[-1])) == (0, LAST_RECORD[name])
    assert int(indices.diff().min()) == 1
    parts = frame[columns[2:]]
    assert sorted(pd.unique(parts.to_numpy().ravel()).tolist()) == [
        "test",
        "train",
        "val",
    ]
    counts = frame["random_s0"].value_counts().to_dict()
    assert counts == random_s0


def test_read_splits_rejects_wrong_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = SPLIT_SHA256[GROWTH]
    monkeypatch.setitem(ecoli.SPLIT_SHA256, GROWTH, "0" * 64)
    with pytest.raises(ValueError) as info:
        read_splits(GROWTH)
    assert str(info.value) == (
        f"split file of {GROWTH} has sha256 {pinned}, pinned {'0' * 64}"
    )


@pytest.mark.parametrize("name", list(SHIPPED))
def test_no_key_in_two_parts_of_any_column(name: str) -> None:
    frame = read_splits(name)
    columns = SHIPPED[name][1][2:]
    parts_per_key = frame.groupby("key")[columns].nunique()
    assert parts_per_key.to_numpy().max() == 1
    if name in SINGLE_GENE_TASKS:
        # Single-gene tasks hold each gene once.
        assert frame["key"].nunique() == len(frame)


def test_gene_holdout_pairs_hold_an_unseen_gene() -> None:
    frame = read_splits(BABU)
    genes = frame["key"].str.split("|", expand=True)
    a, b = genes[0], genes[1]
    holdout = [c for c in PAIR_COLUMNS[2:] if "gene_holdout" in c]
    assert len(holdout) == 12
    for column in holdout:
        part = frame[column]
        seen_train = set(a[part == "train"]) | set(b[part == "train"])
        seen_fit = seen_train | set(a[part == "val"]) | set(b[part == "val"])
        held = part != "train"
        test = part == "test"
        # Every validation or test pair holds a gene no training pair holds, and
        # every test pair holds a gene no training or validation pair holds.
        assert int((a[held].isin(seen_train) & b[held].isin(seen_train)).sum()) == 0
        assert int((a[test].isin(seen_fit) & b[test].isin(seen_fit)).sum()) == 0
