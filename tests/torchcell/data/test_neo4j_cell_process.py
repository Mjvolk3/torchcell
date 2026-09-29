# tests/torchcell/data/test_neo4j_cell_process.py
# [[tests.torchcell.data.test_neo4j_cell_process]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_neo4j_cell_process.py
"""`Neo4jCellDataset.process` wiring, the index error branches, and the item path seams.

No Neo4j: `load_raw` is replaced on the class by a recorder that writes the raw LMDB a
`Neo4jQueryRaw` would (`<root>/raw/lmdb`, a DIRECTORY, keys `"<idx>"`, values a JSON list
of `{"experiment", "experiment_reference"}` dicts). The converter, deduplicator and
aggregator are recorder classes whose `process(input_path, output_path)` copies the input
LMDB to the output and appends `+<stage>` to every record's `dataset_name`, so the
`dataset_name_index` of the finished build spells out which stages ran, in which order,
and which LMDB the PROCESSED copy was taken from.

Two raw records, genes YAL001C..YAL004W, dataset `toy`:

* key 0: fitness 0.9, YAL001C deleted;
* key 1: gene interaction -0.2, YAL001C and YAL002W deleted.

Derived by hand:

* no stages: steps RAW -> PROCESSED, the processed LMDB is the raw one, so
  `dataset_name_index` is {toy: [0, 1]}; `label_df` is index [0, 1], fitness [0.9, NaN],
  gene_interaction [NaN, -0.2];
* all three stages: `toy+conversion+deduplication+aggregation` -> [0, 1], one
  `STAGE_COMPLETE` beside each of conversion/, deduplication/, aggregation/, processed/;
* a conversion marker present before the build: the converter is still INSTANTIATED but
  its `process` never runs, and deduplication reads the pre-existing conversion LMDB
  (`toy+precomputed`), giving `toy+precomputed+deduplication+aggregation`.

The error-branch store holds keys b"0", b"1", b"2", b"x" (LMDB orders bytes, so that is
the cursor order): 0 is a valid fitness record for YAL001C, 1 is `not json`, 2 is a list
of a valid fitness record for YAL002W followed by `{"experiment": {}}`, and x is a valid
record under a non-integer key. Each index therefore holds {0, 2}: key 2 is announced as
"Skipping this entry" but its first item was indexed before the second one raised.
"""

import errno
import json
import os
import pickle
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lmdb
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sortedcontainers import SortedDict

from torchcell.data.graph_processor import GraphProcessor, Perturbation
from torchcell.data.neo4j_cell import Neo4jCellDataset, ProcessingStep
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    ReferenceGenome,
)
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W", "YAL004W"])
LABELS = ["fitness", "gene_interaction"]
QUERY = "MATCH (e:Experiment) RETURN e"
ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _deletion(gene: str) -> KanMxDeletionPerturbation:
    return KanMxDeletionPerturbation(
        systematic_gene_name=gene, perturbed_gene_name=gene
    )


def _fitness(dataset_name: str, gene: str, fitness: float) -> dict[str, Any]:
    return {
        "experiment": FitnessExperiment(
            dataset_name=dataset_name,
            genotype=Genotype(perturbations=[_deletion(gene)]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=fitness),
        ).model_dump(mode="json"),
        "experiment_reference": FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0),
        ).model_dump(mode="json"),
    }


def _interaction(dataset_name: str, genes: list[str], value: float) -> dict[str, Any]:
    return {
        "experiment": GeneInteractionExperiment(
            dataset_name=dataset_name,
            genotype=Genotype(perturbations=[_deletion(g) for g in genes]),
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(gene_interaction=value),
        ).model_dump(mode="json"),
        "experiment_reference": GeneInteractionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=GeneInteractionPhenotype(gene_interaction=0.0),
        ).model_dump(mode="json"),
    }


def _metabolite(dataset_name: str, gene: str) -> dict[str, Any]:
    phenotype = MetabolitePhenotype(
        metabolite_level={"betaxanthin": 1.5},
        n_replicates={"betaxanthin": 2},
        measurement_type="cri_spa_corrected_fluorescence_intensity",
    )
    return {
        "experiment": MetaboliteExperiment(
            dataset_name=dataset_name,
            genotype=Genotype(perturbations=[_deletion(gene)]),
            environment=ENVIRONMENT,
            phenotype=phenotype,
        ).model_dump(mode="json"),
        "experiment_reference": MetaboliteExperimentReference(
            dataset_name=dataset_name,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=phenotype,
        ).model_dump(mode="json"),
    }


def _raw_records(dataset_name: str) -> dict[bytes, bytes]:
    """The two raw records of the module docstring, one per key."""
    return {
        b"0": json.dumps([_fitness(dataset_name, "YAL001C", 0.9)]).encode(),
        b"1": json.dumps(
            [_interaction(dataset_name, ["YAL001C", "YAL002W"], -0.2)]
        ).encode(),
    }


def _write_lmdb(path: Path, entries: dict[bytes, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=int(1e8))
    with env.begin(write=True) as txn:
        for key, value in entries.items():
            txn.put(key, value)
    env.close()


def _relabel_copy(src: str, dst: str, suffix: str) -> None:
    """Copy `src` to `dst`, appending `suffix` to every record's dataset_name."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src_env = lmdb.open(src, readonly=True, lock=False)
    entries: dict[bytes, bytes] = {}
    with src_env.begin() as txn:
        for key, value in txn.cursor():
            items = json.loads(value)
            for item in items:
                item["experiment"]["dataset_name"] += suffix
            entries[bytes(key)] = json.dumps(items).encode()
    src_env.close()
    _write_lmdb(Path(dst), entries)


def _stage_classes(calls: list[tuple[Any, ...]]) -> tuple[Any, Any, Any]:
    """Converter, deduplicator and aggregator recorders sharing one call log."""

    class FakeConverter:
        def __init__(self, root: str, query: Any) -> None:
            calls.append(("Converter.__init__", root, query))

        def process(self, input_path: str, output_path: str) -> None:
            calls.append(("Converter.process", input_path, output_path))
            _relabel_copy(input_path, output_path, "+conversion")

    class FakeDeduplicator:
        def __init__(self, root: str) -> None:
            calls.append(("Deduplicator.__init__", root))

        def process(self, input_path: str, output_path: str) -> None:
            calls.append(("Deduplicator.process", input_path, output_path))
            _relabel_copy(input_path, output_path, "+deduplication")

    class FakeAggregator:
        def __init__(self, root: str) -> None:
            calls.append(("Aggregator.__init__", root))

        def process(self, input_path: str, output_path: str) -> None:
            calls.append(("Aggregator.process", input_path, output_path))
            _relabel_copy(input_path, output_path, "+aggregation")

    return FakeConverter, FakeDeduplicator, FakeAggregator


@pytest.fixture
def raw_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Patch `load_raw` to write the raw LMDB and log its arguments; return the log."""
    calls: list[tuple[Any, ...]] = []

    def fake_load_raw(
        uri: str,
        username: str,
        password: str,
        root_dir: str,
        query: str,
        gene_set: GeneSet,
    ) -> SimpleNamespace:
        calls.append(
            ("load_raw", uri, username, password, root_dir, query, list(gene_set))
        )
        _write_lmdb(Path(root_dir) / "raw" / "lmdb", _raw_records("toy"))
        raw_db = SimpleNamespace(env="open raw env")
        calls.append(("raw_db", raw_db))
        return raw_db

    monkeypatch.setattr(Neo4jCellDataset, "load_raw", staticmethod(fake_load_raw))
    return calls


def _build(root: Path, **overrides: Any) -> Neo4jCellDataset:
    kwargs: dict[str, Any] = {
        "root": str(root),
        "query": QUERY,
        "gene_set": GENES,
        "graph_processor": Perturbation(),
        "phenotype_labels": LABELS,
        "uri": "bolt://unused:1",
        "username": "u",
        "password": "p",
    }
    kwargs.update(overrides)
    return Neo4jCellDataset(**kwargs)


def _markers(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("STAGE_COMPLETE"))


def test_process_without_stages_copies_raw_to_processed(
    tmp_path: Path, raw_calls: list[tuple[Any, ...]]
) -> None:
    """RAW -> PROCESSED: `load_raw` gets the six connection/query/gene-set arguments, the
    processed LMDB is a copy of the raw one, and the build leaves one marker, the
    experiment types, the label frame, and a released raw handle.
    """
    ds = _build(tmp_path)
    raw_db = raw_calls[1][1]
    assert raw_calls[0] == (
        "load_raw",
        "bolt://unused:1",
        "u",
        "p",
        str(tmp_path),
        QUERY,
        ["YAL001C", "YAL002W", "YAL003W", "YAL004W"],
    )
    assert raw_db.env is None
    assert [s.name for s in ds.processing_steps] == ["RAW", "PROCESSED"]
    assert (ds.converter, ds.deduplicator, ds.aggregator) == (None, None, None)
    assert _markers(tmp_path) == ["processed/STAGE_COMPLETE"]
    assert (tmp_path / "processed" / "STAGE_COMPLETE").read_text() == ""
    assert sorted(
        json.loads((tmp_path / "processed" / "experiment_types.json").read_text())
    ) == ["fitness", "gene interaction"]
    assert ds.dataset_name_index == {"toy": [0, 1]}
    assert ds.phenotype_label_index == {"fitness": [0], "gene_interaction": [1]}
    assert ds.perturbation_count_index == {1: [0], 2: [1]}
    assert len(ds) == 2
    expected = pd.DataFrame(
        {"index": [0, 1], "fitness": [0.9, np.nan], "gene_interaction": [np.nan, -0.2]}
    )
    pd.testing.assert_frame_equal(ds.label_df, expected)
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "processed" / "label_df.parquet"), expected
    )


def test_process_runs_every_stage_in_order_and_copies_the_last(
    tmp_path: Path, raw_calls: list[tuple[Any, ...]]
) -> None:
    """Each stage class is instantiated (converter with the raw db as `query`), then each
    `process` reads the previous stage's LMDB; PROCESSED is the aggregator's output.
    """
    stage_calls: list[tuple[Any, ...]] = []
    converter, deduplicator, aggregator = _stage_classes(stage_calls)
    ds = _build(
        tmp_path, converter=converter, deduplicator=deduplicator, aggregator=aggregator
    )
    raw_db = raw_calls[1][1]
    root = str(tmp_path)
    raw, conv, dedup, agg = (
        f"{root}/raw/lmdb",
        f"{root}/conversion/lmdb",
        f"{root}/deduplication/lmdb",
        f"{root}/aggregation/lmdb",
    )
    assert stage_calls == [
        ("Converter.__init__", root, raw_db),
        ("Deduplicator.__init__", root),
        ("Aggregator.__init__", root),
        ("Converter.process", raw, conv),
        ("Deduplicator.process", conv, dedup),
        ("Aggregator.process", dedup, agg),
    ]
    assert [type(x) for x in (ds.converter, ds.deduplicator, ds.aggregator)] == [
        converter,
        deduplicator,
        aggregator,
    ]
    assert ds.processing_steps == list(ProcessingStep)
    assert _markers(tmp_path) == [
        "aggregation/STAGE_COMPLETE",
        "conversion/STAGE_COMPLETE",
        "deduplication/STAGE_COMPLETE",
        "processed/STAGE_COMPLETE",
    ]
    assert ds.dataset_name_index == {"toy+conversion+deduplication+aggregation": [0, 1]}


def test_process_skips_a_stage_whose_marker_exists(
    tmp_path: Path, raw_calls: list[tuple[Any, ...]]
) -> None:
    """A finished conversion is not redone: its class is still built, its `process` is
    not called, and deduplication consumes the conversion LMDB already on disk.
    """
    _write_lmdb(tmp_path / "conversion" / "lmdb", _raw_records("toy+precomputed"))
    (tmp_path / "conversion" / "STAGE_COMPLETE").write_text("")
    stage_calls: list[tuple[Any, ...]] = []
    converter, deduplicator, aggregator = _stage_classes(stage_calls)
    ds = _build(
        tmp_path, converter=converter, deduplicator=deduplicator, aggregator=aggregator
    )
    raw_db = raw_calls[1][1]
    root = str(tmp_path)
    assert stage_calls == [
        ("Converter.__init__", root, raw_db),
        ("Deduplicator.__init__", root),
        ("Aggregator.__init__", root),
        (
            "Deduplicator.process",
            f"{root}/conversion/lmdb",
            f"{root}/deduplication/lmdb",
        ),
        (
            "Aggregator.process",
            f"{root}/deduplication/lmdb",
            f"{root}/aggregation/lmdb",
        ),
    ]
    assert ds.dataset_name_index == {
        "toy+precomputed+deduplication+aggregation": [0, 1]
    }
    assert [c[0] for c in raw_calls] == ["load_raw", "raw_db"]


def test_overwrite_intermediates_fails_on_an_lmdb_directory(
    tmp_path: Path, raw_calls: list[tuple[Any, ...]]
) -> None:
    """Finding: `overwrite_intermediates=True` calls `os.remove(input_path)`
    (`neo4j_cell.py:504-505`), but every stage LMDB is a directory (`lmdb.open` default
    `subdir=True`, and `Neo4jQueryRaw` makes `raw/lmdb` with `os.makedirs`), so the first
    finished stage raises an OSError naming the raw path: IsADirectoryError (EISDIR) on
    Linux, PermissionError (EPERM) on macOS, where unlink(2) refuses a directory with
    EPERM. Its marker is already written, so a rerun without the flag resumes past it and
    completes.
    """
    stage_calls: list[tuple[Any, ...]] = []
    converter, _, _ = _stage_classes(stage_calls)
    raw_path = str(tmp_path / "raw" / "lmdb")
    with pytest.raises(OSError, match=re.escape(raw_path)) as exc:
        _build(tmp_path, converter=converter, overwrite_intermediates=True)
    assert exc.value.errno in (errno.EISDIR, errno.EPERM)
    assert _markers(tmp_path) == ["conversion/STAGE_COMPLETE"]
    assert (tmp_path / "raw" / "lmdb").is_dir()
    assert not (tmp_path / "processed" / "lmdb").exists()
    stage_calls.clear()
    ds = _build(tmp_path, converter=converter)
    assert [c[0] for c in stage_calls] == ["Converter.__init__"]
    assert ds.dataset_name_index == {"toy+conversion": [0, 1]}


def test_load_raw_passes_the_query_and_sorted_gene_set_to_neo4j_query_raw(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`load_raw` builds exactly one `Neo4jQueryRaw` with 10 io and 10 cpu workers and the
    gene set as a sorted list under the `gene_set` cypher parameter, and banners root_dir.
    """
    seen: list[dict[str, Any]] = []

    class RecordingQueryRaw:
        def __init__(self, **kwargs: Any) -> None:
            seen.append(kwargs)

    monkeypatch.setattr("torchcell.data.neo4j_cell.Neo4jQueryRaw", RecordingQueryRaw)
    raw_db = Neo4jCellDataset.load_raw(
        "bolt://h:7687", "user", "pw", "/some/root", QUERY, GENES
    )
    assert type(cast(Any, raw_db)) is RecordingQueryRaw
    assert seen == [
        {
            "uri": "bolt://h:7687",
            "username": "user",
            "password": "pw",
            "root_dir": "/some/root",
            "query": QUERY,
            "io_workers": 10,
            "num_workers": 10,
            "cypher_kwargs": {"gene_set": ["YAL001C", "YAL002W", "YAL003W", "YAL004W"]},
        }
    ]
    assert capsys.readouterr().out == (
        "================\nraw root_dir: /some/root\n================\n"
    )


def _write_processed(root: Path, entries: dict[bytes, bytes], types: list[str]) -> None:
    _write_lmdb(root / "processed" / "lmdb", entries)
    (root / "processed" / "experiment_types.json").write_text(json.dumps(types))


def _bad_store() -> dict[bytes, bytes]:
    return {
        b"0": json.dumps([_fitness("toy", "YAL001C", 0.9)]).encode(),
        b"1": b"not json",
        b"2": json.dumps(
            [_fitness("toy", "YAL002W", 0.5), {"experiment": {}}]
        ).encode(),
        b"x": json.dumps([_fitness("toy", "YAL003W", 0.1)]).encode(),
    }


def test_constructor_indices_skip_bad_entries_but_keep_partial_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: each index loop prints "Skipping this entry" for key 2 after its first
    item was already added (`neo4j_cell.py:750-769`, `806-823`, `860-879`), so a
    half-malformed entry is indexed on its good prefix rather than skipped. The three
    handlers print the exact messages below: JSON error for key 1, integer conversion for
    key x, and the generic handler with the missing field for key 2.
    """
    _write_processed(tmp_path, _bad_store(), ["fitness"])
    ds = Neo4jCellDataset(
        root=str(tmp_path), gene_set=GENES, graph_processor=Perturbation()
    )
    assert ds.phenotype_label_index == {"fitness": [0, 2]}
    assert ds.dataset_name_index == {"toy": [0, 2]}
    assert ds.perturbation_count_index == {1: [0, 2]}
    skipping = (
        "Error decoding JSON for entry b'1'. Skipping this entry.\n"
        "Error processing entry b'2': '{field}'. Skipping this entry.\n"
        "Error converting key to integer: b'x'. Skipping this entry.\n"
    )
    assert capsys.readouterr().out == (
        "Computing phenotype label index...\n"
        + skipping.format(field="phenotype")
        + "Computing dataset name index...\n"
        + skipping.format(field="dataset_name")
        + "Computing perturbation count index...\n"
        + skipping.format(field="genotype")
    )


def test_perturbed_gene_index_skips_bad_entries_but_deletion_index_raises(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: the perturbed-gene index catches JSON/ValueError/KeyError per entry
    (`neo4j_cell.py:935-936`), while the deletion-gene index has no handler
    (`neo4j_cell.py:999-1007`) and fails on key 1 with the raw JSON error; its `finally`
    still closes the LMDB handle.
    """
    _write_processed(tmp_path, _bad_store(), ["fitness"])
    ds = Neo4jCellDataset(
        root=str(tmp_path), gene_set=GENES, graph_processor=Perturbation()
    )
    capsys.readouterr()
    assert ds.compute_is_any_perturbed_gene_index() == {"YAL001C": [0], "YAL002W": [2]}
    assert capsys.readouterr().out == (
        "Computing is any perturbed gene index...\n"
        "Error processing entry b'1': Expecting value: line 1 column 1 (char 0)\n"
        "Error processing entry b'2': 'genotype'\n"
        "Error processing entry b'x': invalid literal for int() with base 10: 'x'\n"
    )
    with pytest.raises(json.JSONDecodeError, match="Expecting value"):
        ds.compute_is_any_deletion_gene_index()
    assert ds.env is None


def test_label_df_leaves_dict_valued_labels_as_nan(tmp_path: Path) -> None:
    """A metabolite record's `metabolite_level` is a dict, so its cell stays NaN
    (`neo4j_cell.py:715-716`), while `phenotype_label_index` still records its presence.
    Key 0 metabolite (YAL001C), key 1 fitness 0.9 (YAL002W).
    """
    entries = {
        b"0": json.dumps([_metabolite("toy", "YAL001C")]).encode(),
        b"1": json.dumps([_fitness("toy", "YAL002W", 0.9)]).encode(),
    }
    _write_processed(tmp_path, entries, ["metabolite", "fitness"])
    ds = Neo4jCellDataset(
        root=str(tmp_path),
        gene_set=GENES,
        graph_processor=Perturbation(),
        phenotype_labels=["metabolite_level", "fitness"],
    )
    pd.testing.assert_frame_equal(
        ds.label_df,
        pd.DataFrame(
            {
                "index": [0, 1],
                "metabolite_level": [np.nan, np.nan],
                "fitness": [np.nan, 0.9],
            }
        ),
    )
    assert ds.phenotype_label_index == {"metabolite_level": [0], "fitness": [1]}


class _RecordingProcessor(GraphProcessor):
    """Records what `get` hands a graph processor and returns a fixed marker."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, Any]] = []

    def process(self, cell_graph: Any, phenotype_info: Any, data: Any) -> Any:
        self.calls.append((cell_graph, phenotype_info, data))
        return {"marker": len(self.calls)}


@pytest.fixture
def two_record_build(tmp_path: Path) -> Path:
    """processed/ holding the two raw records of the module docstring."""
    _write_processed(tmp_path, _raw_records("toy"), ["fitness", "gene interaction"])
    return tmp_path


def test_get_hands_the_processor_the_cell_graph_labels_and_rebuilt_records(
    two_record_build: Path,
) -> None:
    """`get(1)` passes the dataset's own `cell_graph` object, `phenotype_info` in label
    order, and the record rebuilt as pydantic objects, and returns the processor's value;
    `ds[0]` then applies `transform` to what `get(0)` returned.
    """
    processor = _RecordingProcessor()
    ds = Neo4jCellDataset(
        root=str(two_record_build),
        gene_set=GENES,
        graph_processor=processor,
        phenotype_labels=LABELS,
        transform=lambda item: {"transformed": item},
    )
    assert ds.get(1) == {"marker": 1}
    cell_graph, info, data = processor.calls[0]
    assert cell_graph is ds.cell_graph
    assert info == [FitnessPhenotype, GeneInteractionPhenotype]
    expected = json.loads(_raw_records("toy")[b"1"])[0]
    assert [type(d["experiment"]) for d in data] == [GeneInteractionExperiment]
    assert [type(d["experiment_reference"]) for d in data] == [
        GeneInteractionExperimentReference
    ]
    assert data[0]["experiment"].model_dump(mode="json") == expected["experiment"]
    assert (
        data[0]["experiment_reference"].model_dump(mode="json")
        == expected["experiment_reference"]
    )
    assert ds[0] == {"transformed": {"marker": 2}}
    ds.close_lmdb()


def test_len_and_repr_count_lmdb_entries_and_len_closes_an_open_handle(
    two_record_build: Path,
) -> None:
    """Two entries: `len` 2 and PyG's repr "Neo4jCellDataset(2)". `len` reuses the handle
    `get` opened and then closes it (`neo4j_cell.py:641-646`).
    """
    ds = Neo4jCellDataset(
        root=str(two_record_build), gene_set=GENES, graph_processor=Perturbation()
    )
    ds.get(0)
    env = ds.env
    assert env is not None
    assert ds.len() == 2
    assert ds.env is None
    assert repr(ds) == "Neo4jCellDataset(2)"
    assert len(ds) == 2


def test_real_perturbed_gene_cache_travels_to_workers(two_record_build: Path) -> None:
    """Finding: `_WORKER_DROPPED_CACHES` names `_is_any_perturbed_gene_index`
    (`neo4j_cell.py:1037`), which `__init__` sets to None and nothing ever fills; the
    property caches into `_is_any_perturbed_gene_index_cache` (`neo4j_cell.py:951-967`),
    which is NOT dropped, so the full gene index is pickled to every worker. The deletion
    cache, named correctly, is dropped.
    """
    ds = Neo4jCellDataset(
        root=str(two_record_build), gene_set=GENES, graph_processor=Perturbation()
    )
    perturbed = {"YAL001C": [0, 1], "YAL002W": [1]}
    assert ds.is_any_perturbed_gene_index == perturbed
    assert ds.is_any_deletion_gene_index == perturbed
    state = ds.__getstate__()
    assert state["_is_any_perturbed_gene_index"] is None
    assert state["_is_any_perturbed_gene_index_cache"] == perturbed
    assert state["_is_any_deletion_gene_index_cache"] is None
    restored = pickle.loads(pickle.dumps(ds))
    assert restored._is_any_perturbed_gene_index_cache == perturbed


def test_connection_settings_fall_back_to_the_local_instance(
    two_record_build: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD unset the fields are the local defaults
    from `torchcell.database.connection`: bolt://localhost:7687, neo4j, torchcell.
    """
    for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    ds = Neo4jCellDataset(
        root=str(two_record_build), gene_set=GENES, graph_processor=Perturbation()
    )
    assert (ds.uri, ds.username, ds.password) == (
        "bolt://localhost:7687",
        "neo4j",
        "torchcell",
    )


def test_a_caller_base_graph_replaces_the_gene_set_nodes(
    two_record_build: Path,
) -> None:
    """Finding: a `graphs` argument that already has "base" keeps it
    (`neo4j_cell.py:296-297`), and `to_cell_data` takes the node list from "base", so a
    two-gene base shrinks the cell graph to 2 nodes although `gene_set` has 4; the base
    graph's own edge produces no edge type.
    """
    small = GeneSet(["YAL001C", "YAL002W"])
    base = nx.Graph()
    base.add_nodes_from(small)
    base.add_edge("YAL001C", "YAL002W")
    multigraph = GeneMultiGraph(
        graphs=SortedDict(
            {"base": GeneGraph(name="base", graph=base, max_gene_set=small)}
        )
    )
    ds = Neo4jCellDataset(
        root=str(two_record_build),
        gene_set=GENES,
        graphs=multigraph,
        graph_processor=Perturbation(),
    )
    assert ds.gene_set == GENES
    assert ds.cell_graph["gene"].node_ids == ["YAL001C", "YAL002W"]
    assert ds.cell_graph["gene"].num_nodes == 2
    assert ds.cell_graph.edge_types == []


def test_raw_file_names_and_get_init_graphs(two_record_build: Path) -> None:
    """`raw_file_names` is the single "lmdb"; `get_init_graphs` is the edgeless base graph
    over the given genes.
    """
    ds = Neo4jCellDataset(
        root=str(two_record_build), gene_set=GENES, graph_processor=Perturbation()
    )
    assert ds.raw_file_names == "lmdb"
    assert ds.processed_file_names == "lmdb"
    graph = ds.get_init_graphs(GeneSet(["YAL003W", "YAL001C"]))
    assert graph.name == "base"
    assert list(graph.graph.nodes) == ["YAL001C", "YAL003W"]
    assert graph.graph.number_of_edges() == 0
    assert cast(Any, graph.max_gene_set) == GeneSet(["YAL001C", "YAL003W"])
