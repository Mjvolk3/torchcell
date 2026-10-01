# tests/torchcell/data/test_neo4j_cell.py
# [[tests.torchcell.data.test_neo4j_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_neo4j_cell.py
"""The Neo4jCellDataset's pure helpers and its worker pickle, no LMDB and no Neo4j.

Two groups. The pickle group pins `__getstate__`, which decides what a spawned DataLoader
worker receives: on the 13.5M-record 025 build the bulk index caches are 0.62 GB of a
0.65 GB pickle, and 56 worker copies of them OOM-killed a 250 GB allocation (job 1597)
during worker spawn. The properties that would silently make the caches reappear in the
pickle are the risk, so the attribute list and the drop are asserted rather than trusted.

The helper group pins the module-level functions and the two per-instance path helpers,
with every value worked by hand:

* `min_max_normalize_embedding` on [[1, 5, 2], [3, 5, 6]]: column 0 spans 1..3 so
  (1-1)/2 = 0 and (3-1)/2 = 1; column 1 is constant and becomes 0.5; column 2 spans 2..6 so
  0 and 1. Result [[0, 0.5, 0], [1, 0.5, 1]].
* `normalize_tensor_row` on [1, 3, 5]: minus the min gives [0, 2, 4], sum 6, so
  [0, 1/3, 2/3]. On [0.2, 0.5] the shifted row [0, 0.3] sums to 0.3, which the
  `clamp(min=1.0)` lifts to 1, so the row is returned UNSCALED (a Finding, see the test).
* `create_embedding_graph` over ids [YAL001C, YAL002W, YZZ999X] with key "e" as the
  matrix above plus [2, 5, 4], and key "f" = [[10], [20], [30]]: only "e" is normalized
  (column 0 now spans 1..3 with 2 -> 0.5), "f" is concatenated raw, and YZZ999X is
  outside the gene set so it gets no node.
* `_determine_processing_steps` is RAW, then CONVERSION / DEDUPLICATION / AGGREGATION
  for whichever of converter / deduplicator / aggregator is set, then PROCESSED.
* `_get_lmdb_path` puts RAW under `<root>/raw/lmdb`, PROCESSED under
  `<root>/processed/lmdb`, and every intermediate stage under `<root>/<stage>/lmdb`.

The item path against a real on-disk build lives in
`tests/torchcell/data/test_neo4j_cell_hermetic_build.py`.
"""

import pickle
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from torch_geometric.data import Data

from torchcell.data.aggregate import Aggregator
from torchcell.data.deduplicate import Deduplicator
from torchcell.data.embedding import BaseEmbeddingDataset
from torchcell.data.neo4j_cell import (
    Neo4jCellDataset,
    ParsedGenome,
    ProcessingStep,
    _label_values,
    _print_label_stats,
    create_embedding_graph,
    create_graph_from_gene_set,
    min_max_normalize_dataset,
    min_max_normalize_embedding,
    normalize_tensor_row,
    parse_genome,
)
from torchcell.datamodels import Converter
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W", "YAL004W"])
EMBEDDING_E = torch.tensor([[1.0, 5.0, 2.0], [3.0, 5.0, 6.0], [2.0, 5.0, 4.0]])
EMBEDDING_F = torch.tensor([[10.0], [20.0], [30.0]])


class _FakeEmbeddingDataset:
    """The slice of `BaseEmbeddingDataset` the normalizers and `create_embedding_graph` read.

    `_data.embeddings[key]` is the collated tensor, `[n, D]` or flat `[n * D]`; an item is
    a `Data` with `id` and per-key `[1, D]` embeddings (flat storage yields `[D]`), which
    is the layout `BaseEmbeddingDataset.__getitem__` produces.
    """

    def __init__(
        self, ids: list[str], embeddings: dict[str, torch.Tensor], feat_dim: int
    ) -> None:
        self._data = Data(id=ids, embeddings=embeddings)
        self.feat_dim = feat_dim

    def __len__(self) -> int:
        return len(self._data.id)

    def __getitem__(self, idx: int) -> Data:
        items: dict[str, torch.Tensor] = {}
        for key, value in self._data.embeddings.items():
            if value.dim() == 1:
                items[key] = value.view(-1, self.feat_dim)[idx]
            else:
                items[key] = value[idx : idx + 1]
        return Data(id=self._data.id[idx], embeddings=items)

    def __iter__(self) -> Iterator[Data]:
        for idx in range(len(self)):
            yield self[idx]


def _fake_embeddings() -> _FakeEmbeddingDataset:
    return _FakeEmbeddingDataset(
        ["YAL001C", "YAL002W", "YZZ999X"],
        {"e": EMBEDDING_E.clone(), "f": EMBEDDING_F.clone()},
        feat_dim=3,
    )


def _bare(root: str = "/r") -> Neo4jCellDataset:
    """An instance with only `root`, for the path helpers: `__init__` opens a real build."""
    dataset = object.__new__(Neo4jCellDataset)
    dataset.root = root
    return dataset


def test_min_max_normalize_embedding_scales_each_column_to_unit_range() -> None:
    """[[1, 5, 2], [3, 5, 6]] -> [[0, 0.5, 0], [1, 0.5, 1]]; the constant column is 0.5."""
    out = min_max_normalize_embedding(EMBEDDING_E[:2])
    assert out.tolist() == [[0.0, 0.5, 0.0], [1.0, 0.5, 1.0]]
    assert out.dtype == torch.float32


def test_normalize_tensor_row_divides_by_the_range_sum() -> None:
    """[1, 3, 5] -> [0, 2, 4] / 6 = [0, 1/3, 2/3], returned as a one-element list of rows."""
    out = normalize_tensor_row(torch.tensor([1.0, 3.0, 5.0]))
    assert len(out) == 1
    torch.testing.assert_close(out[0], torch.tensor([0.0, 1 / 3, 2 / 3]))


def test_normalize_tensor_row_does_not_reach_a_unit_sum_below_one() -> None:
    """Finding: the docstring says "sum to 1", but `clamp(min=1.0)` on the row sum leaves
    a row whose range-sum is below 1 unscaled ([0.2, 0.5] -> [0, 0.3], sum 0.3), a 2-row
    input comes back as a list of ROWS (`flatten(-1)` on a 2-D tensor is a no-op), and a
    list input keeps only its first tensor (`neo4j_cell.py:73-82`).
    """
    torch.testing.assert_close(
        normalize_tensor_row(torch.tensor([0.2, 0.5]))[0], torch.tensor([0.0, 0.3])
    )
    rows = normalize_tensor_row(torch.tensor([[1.0, 3.0], [2.0, 8.0]]))
    # min over the WHOLE tensor is 1: [[0, 2], [1, 7]] / row sums [2, 8]
    assert [row.tolist() for row in rows] == [[0.0, 1.0], [0.125, 0.875]]
    only_first = normalize_tensor_row(
        [torch.tensor([1.0, 3.0]), torch.tensor([9.0, 9.0])]
    )
    assert [row.tolist() for row in only_first] == [[0.0, 1.0]]


def test_label_values_drops_missing_rows_and_returns_a_float_array() -> None:
    """``_label_values`` returns the label column's non-missing values, in row order,
    as a float64 numpy array (the standardization demo reads its statistics from it).
    """
    label_df = pd.DataFrame(
        {"fitness": [0.25, float("nan"), 1.5], "gene_interaction": [0.0, 0.1, 0.2]}
    )

    values = _label_values(label_df, "fitness")

    assert type(values) is np.ndarray
    assert values.dtype == np.float64
    assert values.tolist() == [0.25, 1.5]


def test_print_label_stats_prints_exact_statistics_per_label(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``_print_label_stats`` prints, per label, the count, min, max, mean and population
    std (ddof 0) of the non-missing values, to four decimals.

    By hand: ``fitness`` = [1.0, NaN, 3.0] drops the NaN -> [1.0, 3.0]: count 2, min 1,
    max 3, mean (1 + 3) / 2 = 2, std sqrt(((1 - 2)^2 + (3 - 2)^2) / 2) = 1.
    ``gene_interaction`` = [0.0, 0.5, 1.0]: count 3, min 0, max 1, mean 1.5 / 3 = 0.5,
    std sqrt((0.25 + 0 + 0.25) / 3) = sqrt(1 / 6) = 0.408248... -> 0.4082.
    """
    label_df = pd.DataFrame(
        {"fitness": [1.0, float("nan"), 3.0], "gene_interaction": [0.0, 0.5, 1.0]}
    )

    _print_label_stats(label_df, ["fitness", "gene_interaction"])

    assert capsys.readouterr().out.split("\n") == [
        "",
        "fitness statistics (original):",
        "  Count: 2",
        "  Min: 1.0000",
        "  Max: 3.0000",
        "  Mean: 2.0000",
        "  Std: 1.0000",
        "",
        "gene_interaction statistics (original):",
        "  Count: 3",
        "  Min: 0.0000",
        "  Max: 1.0000",
        "  Mean: 0.5000",
        "  Std: 0.4082",
        "",
    ]


def test_min_max_normalize_dataset_rewrites_the_first_key_in_place() -> None:
    """Finding: only the FIRST embedding key is normalized ("e" here); "f" is untouched
    although the docstring says "across the entire dataset" (`neo4j_cell.py:114`).
    """
    fake = _fake_embeddings()
    min_max_normalize_dataset(cast(BaseEmbeddingDataset, fake))
    assert fake._data.embeddings["e"].tolist() == [
        [0.0, 0.5, 0.0],
        [1.0, 0.5, 1.0],
        [0.5, 0.5, 0.5],
    ]
    assert fake._data.embeddings["f"].tolist() == [[10.0], [20.0], [30.0]]


def test_min_max_normalize_dataset_handles_flat_storage() -> None:
    """A flat [n * D] store is viewed as [n, D], normalized per column, written back flat."""
    fake = _FakeEmbeddingDataset(
        ["a", "b"], {"e": torch.tensor([1.0, 5.0, 2.0, 3.0, 5.0, 6.0])}, feat_dim=3
    )
    min_max_normalize_dataset(cast(BaseEmbeddingDataset, fake))
    assert fake._data.embeddings["e"].tolist() == [0.0, 0.5, 0.0, 1.0, 0.5, 1.0]
    assert fake._data.embeddings["e"].shape == (6,)


def test_create_embedding_graph_concatenates_keys_for_genes_in_the_set() -> None:
    """Nodes YAL001C -> [0, 0.5, 0, 10] and YAL002W -> [1, 0.5, 1, 20]; YZZ999X is dropped."""
    fake = _fake_embeddings()
    graph = create_embedding_graph(GENES, cast(BaseEmbeddingDataset, fake))
    assert graph.name == "_FakeEmbeddingDataset"
    assert graph.max_gene_set == GENES
    assert sorted(graph.graph.nodes) == ["YAL001C", "YAL002W"]
    assert graph.graph.number_of_edges() == 0
    assert graph.graph.nodes["YAL001C"]["embedding"].tolist() == [0.0, 0.5, 0.0, 10.0]
    assert graph.graph.nodes["YAL002W"]["embedding"].tolist() == [1.0, 0.5, 1.0, 20.0]


def test_create_graph_from_gene_set_has_one_node_per_gene_and_no_edges() -> None:
    """The base graph is named "base" and carries the gene set as nodes only."""
    graph = create_graph_from_gene_set(GENES)
    assert graph.name == "base"
    assert list(graph.graph.nodes) == ["YAL001C", "YAL002W", "YAL003W", "YAL004W"]
    assert graph.graph.number_of_edges() == 0
    assert graph.max_gene_set == GENES
    assert isinstance(graph.graph, nx.Graph)


def test_parse_genome_returns_none_or_wraps_the_gene_set() -> None:
    """None passes through; a genome is reduced to its `gene_set`."""
    assert parse_genome(None) is None
    parsed = parse_genome(cast(SCerevisiaeGenome, SimpleNamespace(gene_set=GENES)))
    assert parsed == ParsedGenome(gene_set=GENES)
    assert parsed is not None and parsed.gene_set == GENES


def test_parsed_genome_rejects_a_non_gene_set() -> None:
    """Finding: the custom "gene_set must be a GeneSet" message in `validate_gene_set`
    (`neo4j_cell.py:61-65`) is unreachable. `GeneSet` is an arbitrary type, so pydantic's
    own isinstance check rejects a plain set first with its stock message.
    """
    with pytest.raises(ValidationError, match="Input should be an instance of GeneSet"):
        ParsedGenome(gene_set=cast(GeneSet, {"YAL001C"}))


def test_processing_step_members_in_pipeline_order() -> None:
    """RAW=1 through PROCESSED=5, in the order the pipeline runs them."""
    assert [(step.name, step.value) for step in ProcessingStep] == [
        ("RAW", 1),
        ("CONVERSION", 2),
        ("DEDUPLICATION", 3),
        ("AGGREGATION", 4),
        ("PROCESSED", 5),
    ]


@pytest.mark.parametrize(
    ("converter", "deduplicator", "aggregator", "expected"),
    [
        (None, None, None, ["RAW", "PROCESSED"]),
        (Converter, None, None, ["RAW", "CONVERSION", "PROCESSED"]),
        (None, Deduplicator, None, ["RAW", "DEDUPLICATION", "PROCESSED"]),
        (None, None, Aggregator, ["RAW", "AGGREGATION", "PROCESSED"]),
        (
            Converter,
            Deduplicator,
            None,
            ["RAW", "CONVERSION", "DEDUPLICATION", "PROCESSED"],
        ),
        (
            Converter,
            None,
            Aggregator,
            ["RAW", "CONVERSION", "AGGREGATION", "PROCESSED"],
        ),
        (
            None,
            Deduplicator,
            Aggregator,
            ["RAW", "DEDUPLICATION", "AGGREGATION", "PROCESSED"],
        ),
        (
            Converter,
            Deduplicator,
            Aggregator,
            ["RAW", "CONVERSION", "DEDUPLICATION", "AGGREGATION", "PROCESSED"],
        ),
    ],
)
def test_determine_processing_steps_follows_the_configured_stages(
    converter: type[Converter] | None,
    deduplicator: type[Deduplicator] | None,
    aggregator: type[Aggregator] | None,
    expected: list[str],
) -> None:
    """Every stage whose class is set appears, in fixed order, between RAW and PROCESSED."""
    dataset = _bare()
    dataset.converter = converter
    dataset.deduplicator = deduplicator
    dataset.aggregator = aggregator
    assert [s.name for s in dataset._determine_processing_steps()] == expected


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (ProcessingStep.RAW, "/r/raw/lmdb"),
        (ProcessingStep.CONVERSION, "/r/conversion/lmdb"),
        (ProcessingStep.DEDUPLICATION, "/r/deduplication/lmdb"),
        (ProcessingStep.AGGREGATION, "/r/aggregation/lmdb"),
        (ProcessingStep.PROCESSED, "/r/processed/lmdb"),
    ],
)
def test_get_lmdb_path_places_each_stage_under_root(
    step: ProcessingStep, expected: str
) -> None:
    """RAW and PROCESSED use the PyG raw/processed dirs; intermediates use the stage name."""
    assert _bare("/r")._get_lmdb_path(step) == expected


def test_gene_set_getter_without_a_file_uses_memory_or_raises(tmp_path: Path) -> None:
    """No gene_set.json: the in-memory set is returned as a GeneSet, or the exact error."""
    dataset = _bare(str(tmp_path))
    dataset._gene_set = {"YAL002W", "YAL001C"}
    assert dataset.gene_set == GeneSet(["YAL001C", "YAL002W"])
    dataset._gene_set = None
    with pytest.raises(
        ValueError,
        match=(
            "gene_set not written during process. "
            "Please call compute_gene_set in process."
        ),
    ):
        dataset.gene_set


def test_gene_set_setter_rejects_an_empty_set(tmp_path: Path) -> None:
    """An empty GeneSet is refused before anything is written under root."""
    dataset = _bare(str(tmp_path))
    with pytest.raises(
        ValueError, match="Cannot set an empty or None value for gene_set"
    ):
        dataset.gene_set = GeneSet()
    assert list(tmp_path.iterdir()) == []


def _stub(**overrides: Any) -> Neo4jCellDataset:
    """A dataset instance carrying only the attributes `__getstate__` reads.

    Built with `object.__new__` on purpose: `__init__` opens a real build, and
    `__getstate__` is a pure transform of `__dict__` that needs none of it.
    """
    dataset = object.__new__(Neo4jCellDataset)
    dataset.__dict__.update(
        {
            "env": "an open lmdb environment",
            "root": "/some/build",
            "phenotype_labels": ["gene_interaction"],
            "_phenotype_label_index": {"gene_interaction": [0, 1, 2]},
            "_dataset_name_index": {"kuzmin2020": [0, 1]},
            "_perturbation_count_index": {3: [0, 1, 2]},
            "_is_any_perturbed_gene_index": {"YAL001C": [0]},
            "_is_any_deletion_gene_index_cache": {"YAL001C": [0]},
            "_label_df": pd.DataFrame(
                {"index": [0, 1], "gene_interaction": [0.1, 0.2]}
            ),
            **overrides,
        }
    )
    return dataset


def test_getstate_drops_the_lmdb_environment() -> None:
    """The pre-existing contract: an open LMDB env cannot be pickled."""
    assert _stub().__getstate__()["env"] is None


def test_getstate_drops_every_bulk_cache() -> None:
    """None of the six bulk caches may travel to a worker."""
    state = _stub().__getstate__()
    for name in Neo4jCellDataset._WORKER_DROPPED_CACHES:
        assert state[name] is None, f"{name} would be pickled to every worker"


def test_getstate_keeps_everything_else() -> None:
    """The drop is targeted: attributes a worker needs are untouched.

    `get()` reads the LMDB record, `cell_graph`, and `phenotype_info`; dropping more than
    the caches would break item construction in the worker rather than in the parent,
    which is the harder place to see it.
    """
    state = _stub(
        cell_graph="hetero-data", _phenotype_info=["GeneInteraction"]
    ).__getstate__()
    assert state["cell_graph"] == "hetero-data"
    assert state["_phenotype_info"] == ["GeneInteraction"]
    assert state["root"] == "/some/build"
    assert state["phenotype_labels"] == ["gene_interaction"]


def test_getstate_does_not_mutate_the_live_dataset() -> None:
    """Pickling must not empty the caches in the PARENT process.

    `__getstate__` copies `__dict__` before clearing. Clearing in place would work once
    and then quietly strip the parent, whose `CellDataModule` reads these indices.
    """
    dataset = _stub()
    dataset.__getstate__()
    assert dataset._phenotype_label_index == {"gene_interaction": [0, 1, 2]}
    assert dataset.env == "an open lmdb environment"


def test_pickle_round_trip_is_small_and_restores() -> None:
    """End to end: the caches are absent from the bytes, and the rest survives."""
    dataset = _stub(cell_graph="hetero-data")
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.cell_graph == "hetero-data"
    assert restored.env is None
    assert restored._label_df is None
    assert restored._perturbation_count_index is None


def test_dropped_cache_names_are_real_attributes() -> None:
    """Guards against a rename silently un-dropping a cache.

    A typo in `_WORKER_DROPPED_CACHES` is invisible: `__getstate__` skips a name that is
    not in `__dict__`, so the real attribute keeps travelling and the pickle quietly
    grows back to its old size.
    """
    live = _stub().__dict__
    for name in Neo4jCellDataset._WORKER_DROPPED_CACHES:
        assert name in live, f"{name} is not an attribute of Neo4jCellDataset"
