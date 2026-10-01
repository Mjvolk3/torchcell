# tests/torchcell/datasets/test_sgd_gene_graph.py
# [[tests.torchcell.datasets.test_sgd_gene_graph]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_sgd_gene_graph.py
"""Hermetic build of ``GraphEmbeddingDataset`` on a hand-built three-gene SGD gene graph.

The loader reads no files: ``process()`` walks ``graph.nodes(data=True)`` and reads nine
node attributes, the layout ``SCerevisiaeGraph.G_gene`` carries after
``add_gene_protein_overview``, ``add_loci_information`` and ``add_pathway_annotation``
(``length``, ``molecular_weight``, ``pi``, ``median_value``, ``median_abs_dev_value``,
``start``, ``end``, ``chromosome``, ``pathways``). The graph here is built in memory and
the dataset root is under ``tmp_path``.

Fixture (node insertion order is the output order):

    node      length  mw      pi   median  mad  start  end  chromosome  pathways
    YAL001C   100     1000.0  4.0  10.0    1.0  100    400  2           [20]
    YBR001C   None    3000.0  0    30.0    3.0  300    600  1           None
    YCR001W   300     5000.0  8.0  50.0    5.0  500    800  2           [10, 20]

Chromosomes and pathways are small integers here; SGD's real values are display-name
strings, and the ``PYTHONHASHSEED`` test builds on strings in two subprocesses.

Expected values (issue #518 contract, fixed 2026.10.01):

* ``length`` values collected are [100, 300] (None skipped); ``torch.median`` of an even
  count returns the LOWER middle, 100, so YBR001C's missing length becomes 100.
* ``pi`` values collected are [4.0, 0, 8.0]; YBR001C's 0 is a value, not a missing
  entry, so it is stored as 0 (only None takes the median).
* Unnormalized rows (``chrom_pathways``) are the raw values with the one length fill.
* Normalized rows (``normalized_chrom_pathways``) are ``(x - min) / (max - min)`` with
  min/max over the COLLECTED values: length (100, 300), mw (1000, 5000), pi (0, 8),
  median (10, 50), mad (1, 5), start (100, 500), end (400, 800). YBR001C: length
  (100-100)/200 = 0, mw 2000/4000 = 0.5, pi 0/8 = 0, the rest 0.5; YAL001C: pi
  4/8 = 0.5, the rest 0; YCR001W: all 1.
* ``chromosome_index`` is the position in the sorted vocabulary [1, 2]: YAL001C (2) -> 1,
  YBR001C (1) -> 0, YCR001W (2) -> 1. So [1, 0, 1].
* ``pathways_indices`` are positions in the sorted vocabulary [10, 20]: YAL001C [20] ->
  [1]; YBR001C none -> []; YCR001W [10, 20] -> [0, 1]. Collated: [1, 0, 1] with slices
  [0, 1, 1, 3].
* A constant feature under normalization raises ``ConstantFeatureError``.
* ``categorical_features`` = {"chromosome": {"num_values": 2}, "pathways":
  {"num_values": 2}}.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data

from torchcell.datasets.sgd_gene_graph import (
    ConstantFeatureError,
    GraphEmbeddingDataset,
)

_FEATURES = (
    "length",
    "molecular_weight",
    "pi",
    "median_value",
    "median_abs_dev_value",
    "start",
    "end",
)


def _node(
    values: tuple[Any, ...], chromosome: Any, pathways: list[Any] | None
) -> dict[str, Any]:
    attrs = dict(zip(_FEATURES, values, strict=True))
    attrs["chromosome"] = chromosome
    attrs["pathways"] = pathways
    return attrs


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("YAL001C", **_node((100, 1000.0, 4.0, 10.0, 1.0, 100, 400), 2, [20]))
    graph.add_node("YBR001C", **_node((None, 3000.0, 0, 30.0, 3.0, 300, 600), 1, None))
    graph.add_node(
        "YCR001W", **_node((300, 5000.0, 8.0, 50.0, 5.0, 500, 800), 2, [10, 20])
    )
    return graph


RAW_ROWS = [
    [100.0, 1000.0, 4.0, 10.0, 1.0, 100.0, 400.0],
    [100.0, 3000.0, 0.0, 30.0, 3.0, 300.0, 600.0],
    [300.0, 5000.0, 8.0, 50.0, 5.0, 500.0, 800.0],
]
NORMALIZED_ROWS = [
    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
]


def test_unnormalized_build_fills_only_none_with_the_lower_median(
    tmp_path: Path,
) -> None:
    """``chrom_pathways``: exact ids, feature rows, categorical indices and file set.

    YBR001C's missing length takes the lower median 100; its pI of 0 is kept as 0, not
    replaced by the median 4.0 (issue #518).
    """
    dataset = GraphEmbeddingDataset(
        root=str(tmp_path / "raw_build"), graph=_graph(), model_name="chrom_pathways"
    )
    data = dataset._data
    assert data.id == ["YAL001C", "YBR001C", "YCR001W"]
    assert list(data.embeddings) == ["chrom_pathways"]
    assert torch.equal(data.embeddings["chrom_pathways"], torch.tensor(RAW_ROWS))
    assert torch.equal(data.chromosome_index, torch.tensor([1, 0, 1]))
    assert torch.equal(data.pathways_indices, torch.tensor([1, 0, 1]))
    assert torch.equal(dataset.slices["pathways_indices"], torch.tensor([0, 1, 1, 3]))
    assert dataset.categorical_features == {
        "chromosome": {"num_values": 2},
        "pathways": {"num_values": 2},
    }
    assert sorted(os.listdir(tmp_path / "raw_build" / "processed")) == [
        "categorical_features.pt",
        "chrom_pathways.pt",
        "pre_filter.pt",
        "pre_transform.pt",
    ]
    assert len(dataset) == 3


def test_integer_index_returns_one_gene_with_its_own_slices(tmp_path: Path) -> None:
    """``dataset[1]`` is YBR001C: a 1 x 7 row, chromosome index 0, no pathway indices."""
    dataset = GraphEmbeddingDataset(
        root=str(tmp_path / "item"), graph=_graph(), model_name="chrom_pathways"
    )
    item = dataset[1]
    assert item.id == "YBR001C"
    assert torch.equal(item.embeddings["chrom_pathways"], torch.tensor([RAW_ROWS[1]]))
    assert torch.equal(item.chromosome_index, torch.tensor([0]))
    assert torch.equal(item.pathways_indices, torch.tensor([], dtype=torch.long))
    third = dataset[2]
    assert third.id == "YCR001W"
    assert torch.equal(third.pathways_indices, torch.tensor([0, 1]))


def test_normalized_build_min_max_scales_every_feature(tmp_path: Path) -> None:
    """``normalized_chrom_pathways``: the rows worked in the module docstring, exactly."""
    dataset = GraphEmbeddingDataset(
        root=str(tmp_path / "norm"),
        graph=_graph(),
        model_name="normalized_chrom_pathways",
    )
    assert list(dataset._data.embeddings) == ["normalized_chrom_pathways"]
    assert torch.equal(
        dataset._data.embeddings["normalized_chrom_pathways"],
        torch.tensor(NORMALIZED_ROWS),
    )


def test_categorical_index_is_a_function_of_the_category(tmp_path: Path) -> None:
    """One category, one index: both chromosome-2 genes get 1, chromosome 1 gets 0, and
    pathway 20 is index 1 wherever it appears (sorted vocabularies, issue #518).
    """
    dataset = GraphEmbeddingDataset(
        root=str(tmp_path / "cat"), graph=_graph(), model_name="chrom_pathways"
    )
    index_of = dict(
        zip(dataset._data.id, dataset._data.chromosome_index.tolist(), strict=True)
    )
    assert index_of == {"YAL001C": 1, "YBR001C": 0, "YCR001W": 1}
    assert dataset[0].pathways_indices.tolist() == [1]
    assert dataset[2].pathways_indices.tolist() == [0, 1]


_HASHSEED_SCRIPT = """
import sys
import networkx as nx
from torchcell.datasets.sgd_gene_graph import GraphEmbeddingDataset

chromosomes = ["Chromosome XVI", "Chromosome I", "Chromosome Mito", "Chromosome IV"]
pathways = [f"pathway {name}" for name in "qwertyuiopasdfgh"]
graph = nx.Graph()
for i in range(8):
    graph.add_node(
        f"Y{i:03d}",
        length=100 + i, molecular_weight=1000.0 + i, pi=4.0 + i, median_value=10.0 + i,
        median_abs_dev_value=1.0 + i, start=100 + i, end=400 + i,
        chromosome=chromosomes[i % 4],
        pathways=[pathways[(3 * i) % 16], pathways[(5 * i + 1) % 16]],
    )
dataset = GraphEmbeddingDataset(root=sys.argv[1], graph=graph, model_name="chrom_pathways")
print(dataset._data.chromosome_index.tolist())
print(dataset._data.pathways_indices.tolist())
"""


def test_categorical_indices_do_not_depend_on_pythonhashseed(tmp_path: Path) -> None:
    """The same string-keyed graph built in two interpreters with PYTHONHASHSEED 1 and
    2 yields identical indices, equal to the sorted-vocabulary positions: chromosomes
    sort I, IV, Mito, XVI; the 12 pathways in use sort by their letter.
    """
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", _HASHSEED_SCRIPT, str(tmp_path / f"seed{seed}")],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout.splitlines()[-2:])
    order = "qwertyuiopasdfgh"
    used = [order[k] for i in range(8) for k in ((3 * i) % 16, (5 * i + 1) % 16)]
    vocabulary = sorted(set(used))
    expected_pathways = [vocabulary.index(letter) for letter in used]
    assert outputs[0] == outputs[1]
    assert outputs[0] == [str([3, 0, 2, 1, 3, 0, 2, 1]), str(expected_pathways)]


def test_second_construction_loads_from_disk_without_reprocessing(
    tmp_path: Path,
) -> None:
    """A processed root is not rebuilt: an EMPTY graph on the second call would raise in
    ``process`` (``min()`` of an empty tensor), yet the second dataset returns the first
    build's rows and the ``categorical_features`` saved to disk.
    """
    root = str(tmp_path / "reload")
    GraphEmbeddingDataset(root=root, graph=_graph(), model_name="chrom_pathways")
    reloaded = GraphEmbeddingDataset(
        root=root, graph=nx.Graph(), model_name="chrom_pathways"
    )
    assert reloaded._data.id == ["YAL001C", "YBR001C", "YCR001W"]
    assert torch.equal(
        reloaded._data.embeddings["chrom_pathways"], torch.tensor(RAW_ROWS)
    )
    assert reloaded.categorical_features == {
        "chromosome": {"num_values": 2},
        "pathways": {"num_values": 2},
    }


def test_caller_categorical_features_are_merged_and_mutated_in_place(
    tmp_path: Path,
) -> None:
    """A non-empty ``categorical_features`` keeps its extra keys, gains ``num_values``,
    and is the SAME dict object the build wrote into (line 39 aliases it, lines 163 to
    169 mutate it); the saved file holds the merged dict.
    """
    caller = {"chromosome": {"embedding_dim": 4}}
    dataset = GraphEmbeddingDataset(
        root=str(tmp_path / "merge"),
        graph=_graph(),
        model_name="chrom_pathways",
        categorical_features=caller,
    )
    merged = {
        "chromosome": {"embedding_dim": 4, "num_values": 2},
        "pathways": {"num_values": 2},
    }
    assert caller == merged
    assert dataset.categorical_features == merged
    saved = torch.load(
        tmp_path / "merge" / "processed" / "categorical_features.pt", weights_only=False
    )
    assert saved == merged


def test_pre_transform_runs_on_every_item_before_collation(tmp_path: Path) -> None:
    """``pre_transform`` sees each gene's ``Data``; its output is what is saved."""

    def tag(data: Data) -> Data:
        data.n_pathways = torch.tensor([data.pathways_indices.numel()])
        return data

    dataset = GraphEmbeddingDataset(
        root=str(tmp_path / "pre"),
        graph=_graph(),
        model_name="chrom_pathways",
        pre_transform=tag,
    )
    assert torch.equal(dataset._data.n_pathways, torch.tensor([1, 0, 2]))


def test_string_lookup_raises_because_there_are_no_dna_windows(tmp_path: Path) -> None:
    """Finding: the base class's string lookup (``torchcell/data/embedding.py`` lines 77 to
    90) reads ``_data.dna_windows``, which this dataset never stores, so ``dataset["YAL001C"]``
    raises ``AttributeError`` for a gene that IS present. Pinned until the subclass
    overrides ``__getitem__`` or the base tolerates the missing field.
    """
    dataset = GraphEmbeddingDataset(
        root=str(tmp_path / "str"), graph=_graph(), model_name="chrom_pathways"
    )
    with pytest.raises(
        AttributeError, match="^'GlobalStorage' object has no attribute 'dna_windows'$"
    ):
        dataset["YAL001C"]


def test_invalid_model_name_is_refused_before_any_file_is_written(
    tmp_path: Path,
) -> None:
    """The base validator names the bad name, then the valid names after a space."""
    with pytest.raises(ValueError) as excinfo:
        GraphEmbeddingDataset(
            root=str(tmp_path / "bad"), graph=_graph(), model_name="window_5979"
        )
    assert str(excinfo.value) == (
        "Invalid model_name 'window_5979'. Valid options are: "
        "normalized_chrom_pathways, chrom_pathways"
    )
    assert not (tmp_path / "bad").exists()


def test_model_name_none_fails_in_process_with_key_error(tmp_path: Path) -> None:
    """``model_name`` defaults to None, which the base validator lets through; ``process``
    then looks up ``MODEL_TO_WINDOW[None]`` (line 67) and raises ``KeyError(None)``.
    """
    with pytest.raises(KeyError) as excinfo:
        GraphEmbeddingDataset(root=str(tmp_path / "none"), graph=_graph())
    assert excinfo.value.args == (None,)


@pytest.mark.parametrize("missing", ["pi", "chromosome", "pathways"])
def test_missing_node_attribute_raises_key_error_naming_it(
    tmp_path: Path, missing: str
) -> None:
    """Every attribute is read with ``node_data[...]``, so a node without one raises
    ``KeyError`` with that attribute's name (``pi`` in the collection loop;
    ``chromosome`` and ``pathways`` in the pass that builds the sorted vocabularies).
    """
    graph = _graph()
    del graph.nodes["YBR001C"][missing]
    with pytest.raises(KeyError) as excinfo:
        GraphEmbeddingDataset(
            root=str(tmp_path / missing), graph=graph, model_name="chrom_pathways"
        )
    assert excinfo.value.args == (missing,)


def test_feature_missing_on_every_node_raises_in_min(tmp_path: Path) -> None:
    """A feature that is None on every gene leaves an empty value list; ``median`` of it
    does not raise but ``min`` does, with torch's empty-reduction message.
    """
    graph = _graph()
    for node in graph.nodes:
        graph.nodes[node]["median_abs_dev_value"] = None
    with pytest.raises(
        RuntimeError,
        match=r"^min\(\): Expected reduction dim to be specified for input.numel\(\) == 0",
    ):
        GraphEmbeddingDataset(
            root=str(tmp_path / "empty"), graph=graph, model_name="chrom_pathways"
        )


def test_single_gene_normalization_refuses_every_constant_feature(
    tmp_path: Path,
) -> None:
    """With one gene every feature has max == min: normalization raises
    ``ConstantFeatureError`` naming all seven features and their value, and writes no
    feature file, instead of storing 0 / 0 = NaN.
    """
    graph = nx.Graph()
    graph.add_node("YAL001C", **_node((100, 1000.0, 4.0, 10.0, 1.0, 100, 400), 2, [20]))
    with pytest.raises(ConstantFeatureError) as excinfo:
        GraphEmbeddingDataset(
            root=str(tmp_path / "one"),
            graph=graph,
            model_name="normalized_chrom_pathways",
        )
    assert str(excinfo.value) == (
        "features {'length': 100, 'molecular_weight': 1000.0, 'pi': 4.0, "
        "'median_value': 10.0, 'median_abs_dev_value': 1.0, 'start': 100, 'end': 400} "
        "take one value on every gene; min-max normalization would divide 0 by 0, "
        "refusing to store NaN"
    )
    assert not (
        tmp_path / "one" / "processed" / "normalized_chrom_pathways.pt"
    ).exists()


def test_a_constant_feature_is_kept_when_not_normalizing(tmp_path: Path) -> None:
    """The refusal belongs to min-max scaling only: one gene builds unnormalized."""
    graph = nx.Graph()
    graph.add_node("YAL001C", **_node((100, 1000.0, 4.0, 10.0, 1.0, 100, 400), 2, [20]))
    dataset = GraphEmbeddingDataset(
        root=str(tmp_path / "one_raw"), graph=graph, model_name="chrom_pathways"
    )
    assert torch.equal(
        dataset._data.embeddings["chrom_pathways"], torch.tensor([RAW_ROWS[0]])
    )
