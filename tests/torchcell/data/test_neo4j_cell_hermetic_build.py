# tests/torchcell/data/test_neo4j_cell_hermetic_build.py
# [[tests.torchcell.data.test_neo4j_cell_hermetic_build]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_neo4j_cell_hermetic_build.py
"""`Neo4jCellDataset` over a processed build written by hand under tmp_path, no Neo4j.

PyG's `Dataset._process` skips `process()` when every `processed_file_names` entry exists,
and `Neo4jCellDataset` names exactly one, `processed/lmdb`. The fixture writes that LMDB
with the JSON-array shape the aggregator emits (`aggregate.py:157-160`: key `"<idx>"`,
value `[{"experiment": ..., "experiment_reference": ...}, ...]`), plus
`experiment_types.json`, which `process()` would have produced. No query runs; the
constructor's `neo4j_connection_settings()` only reads environment variables.

Four genes, YAL001C..YAL004W, so the cell graph has 4 nodes. Four records, one per key:

* 0: `toy_a`, YAL001C deleted, fitness 0.9.
* 1: `toy_a`, YAL002W and YAL003W deleted, fitness 0.4 with fitness_se 0.05.
* 2: `toy_b`, YAL003W deleted and YAL004W as an SGA DAmP allele (NOT a deletion),
  fitness 0.7 with fitness_std 0.1 (fitness_se stays None: the statistic is `fitness_se`).
* 3: `toy_b`, a gene-interaction record, YAL001C and YAL004W deleted, gene_interaction -0.2.

Derived by hand: `len` 4; `phenotype_label_index` {fitness: [0, 1, 2], gene_interaction:
[3]}; `dataset_name_index` {toy_a: [0, 1], toy_b: [2, 3]}; `perturbation_count_index`
{1: [0], 2: [1, 2, 3]}; `is_any_perturbed_gene_index` has YAL004W -> [2, 3] where
`is_any_deletion_gene_index` has YAL004W -> [3], the DAmP record being the difference the
deletion index exists for. With `phenotype_labels=["fitness", "gene_interaction"]` the
COO type index is 0 for fitness and 1 for gene_interaction; `label_df` is index 0..3,
fitness [0.9, 0.4, 0.7, NaN], gene_interaction [NaN, NaN, NaN, -0.2].
"""

import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import lmdb
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from sortedcontainers import SortedDict
from torch_geometric.data import Data, HeteroData

from torchcell.data.graph_processor import Perturbation
from torchcell.data.neo4j_cell import Neo4jCellDataset
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
    ReferenceGenome,
    SgaDampPerturbation,
)
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W", "YAL004W"])
LABELS = ["fitness", "gene_interaction"]
ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _deletion(gene: str) -> KanMxDeletionPerturbation:
    return KanMxDeletionPerturbation(
        systematic_gene_name=gene, perturbed_gene_name=gene
    )


def _damp(gene: str) -> SgaDampPerturbation:
    return SgaDampPerturbation(
        systematic_gene_name=gene, perturbed_gene_name=gene, strain_id="damp-1"
    )


def _fitness_record(
    dataset_name: str, perturbations: list[Any], phenotype: FitnessPhenotype
) -> dict[str, Any]:
    return {
        "experiment": FitnessExperiment(
            dataset_name=dataset_name,
            genotype=Genotype(perturbations=perturbations),
            environment=ENVIRONMENT,
            phenotype=phenotype,
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0),
        ),
    }


def _interaction_record(
    dataset_name: str, perturbations: list[Any], gene_interaction: float
) -> dict[str, Any]:
    return {
        "experiment": GeneInteractionExperiment(
            dataset_name=dataset_name,
            genotype=Genotype(perturbations=perturbations),
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(gene_interaction=gene_interaction),
        ),
        "experiment_reference": GeneInteractionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=GeneInteractionPhenotype(gene_interaction=0.0),
        ),
    }


# One record per LMDB key; the module docstring lists them.
RECORDS: list[dict[str, Any]] = [
    _fitness_record("toy_a", [_deletion("YAL001C")], FitnessPhenotype(fitness=0.9)),
    _fitness_record(
        "toy_a",
        [_deletion("YAL002W"), _deletion("YAL003W")],
        FitnessPhenotype(fitness=0.4, fitness_se=0.05),
    ),
    _fitness_record(
        "toy_b",
        [_deletion("YAL003W"), _damp("YAL004W")],
        FitnessPhenotype(fitness=0.7, fitness_std=0.1),
    ),
    _interaction_record(
        "toy_b", [_deletion("YAL001C"), _deletion("YAL004W")], gene_interaction=-0.2
    ),
]


def _write_build(root: Path) -> Path:
    """Write `processed/lmdb` and `experiment_types.json` the way a finished build has them."""
    processed = root / "processed"
    processed.mkdir()
    env = lmdb.open(str(processed / "lmdb"), map_size=int(1e8))
    with env.begin(write=True) as txn:
        for idx, record in enumerate(RECORDS):
            payload = [
                {
                    "experiment": record["experiment"].model_dump(mode="json"),
                    "experiment_reference": record["experiment_reference"].model_dump(
                        mode="json"
                    ),
                }
            ]
            txn.put(str(idx).encode(), json.dumps(payload).encode())
    env.close()
    (processed / "experiment_types.json").write_text(
        json.dumps(["fitness", "gene interaction"])
    )
    return processed


def _info(dataset: Neo4jCellDataset) -> list[Any]:
    """`phenotype_info` holds phenotype CLASSES; the source annotates it as instances."""
    return cast(list[Any], dataset.phenotype_info)


def _dataset(root: Path, **overrides: Any) -> Neo4jCellDataset:
    kwargs: dict[str, Any] = {
        "root": str(root),
        "gene_set": GENES,
        "graph_processor": Perturbation(),
        "phenotype_labels": LABELS,
    }
    kwargs.update(overrides)
    return Neo4jCellDataset(**kwargs)


@pytest.fixture
def build_root(tmp_path: Path) -> Path:
    """A root whose `processed/` holds the LMDB and experiment types, nothing else."""
    _write_build(tmp_path)
    return tmp_path


@pytest.fixture
def dataset(build_root: Path) -> Iterator[Neo4jCellDataset]:
    """The dataset over `build_root`; closes the LMDB handle so a re-open is legal."""
    ds = _dataset(build_root)
    yield ds
    ds.close_lmdb()


def _expected_label_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            "fitness": [0.9, 0.4, 0.7, np.nan],
            "gene_interaction": [np.nan, np.nan, np.nan, -0.2],
        }
    )


def test_constructor_skips_process_and_persists_the_gene_set(build_root: Path) -> None:
    """The LMDB satisfies `processed_file_names`, so `process()` never runs (no
    `pre_transform.pt`), the setter writes `gene_set.json` sorted, and the constructor tail
    writes the three index files. Lock files are the FileLockHelper's side effect.
    """
    ds = _dataset(build_root)
    processed = build_root / "processed"
    assert sorted(p.name for p in processed.iterdir()) == [
        "dataset_name_index.json",
        "dataset_name_index.json.lock",
        "experiment_types.json",
        "gene_set.json",
        "gene_set.json.lock",
        "lmdb",
        "perturbation_count_index.json",
        "perturbation_count_index.json.lock",
        "phenotype_label_index.json",
        "phenotype_label_index.json.lock",
    ]
    assert json.loads((processed / "gene_set.json").read_text()) == [
        "YAL001C",
        "YAL002W",
        "YAL003W",
        "YAL004W",
    ]
    assert ds.gene_set == GENES
    assert ds.env is None
    assert ds.cell_graph["gene"].num_nodes == 4
    assert ds.cell_graph["gene"].node_ids == [
        "YAL001C",
        "YAL002W",
        "YAL003W",
        "YAL004W",
    ]
    assert ds.cell_graph["gene"].x.shape == (4, 0)
    assert ds.cell_graph.edge_types == []


def test_none_gene_set_is_rejected_before_anything_is_written(tmp_path: Path) -> None:
    """`gene_set=None` hits the setter's guard first; the root stays empty."""
    with pytest.raises(
        ValueError, match="Cannot set an empty or None value for gene_set"
    ):
        Neo4jCellDataset(
            root=str(tmp_path), gene_set=None, graph_processor=Perturbation()
        )
    assert list(tmp_path.iterdir()) == []


def test_len_closes_the_handle_and_get_leaves_it_open(
    dataset: Neo4jCellDataset,
) -> None:
    """`len` counts the 4 LMDB entries and closes; `get` opens lazily and stays open."""
    assert len(dataset) == 4
    assert dataset.env is None
    dataset.get(0)
    assert dataset.env is not None
    dataset.close_lmdb()
    assert dataset.env is None
    dataset.close_lmdb()  # idempotent on a closed handle
    assert dataset.env is None


@pytest.mark.parametrize(
    ("idx", "indices", "values", "type_indices", "stat_values", "stat_type_indices"),
    [
        (0, [0], [0.9], [0], [], []),
        (1, [1, 2], [0.4], [0], [0.05], [0]),
        (2, [2, 3], [0.7], [0], [], []),
        (3, [0, 3], [-0.2], [1], [], []),
    ],
)
def test_get_yields_the_perturbation_processor_item(
    dataset: Neo4jCellDataset,
    idx: int,
    indices: list[int],
    values: list[float],
    type_indices: list[int],
    stat_values: list[float],
    stat_type_indices: list[int],
) -> None:
    """Perturbed gene positions in the 4-node graph, COO phenotype and statistic tensors.

    Record 2's DAmP allele counts as perturbed (index 3); record 3 is the only
    gene_interaction so its type index is 1; only record 1 carries a `fitness_se`.
    """
    item = dataset.get(idx)
    assert isinstance(item, HeteroData)
    gene = item["gene"]
    assert gene.num_nodes == 4
    assert gene.perturbation_indices.tolist() == indices
    assert gene.pert_mask.tolist() == [i in indices for i in range(4)]
    assert gene.mask.tolist() == [i not in indices for i in range(4)]
    torch.testing.assert_close(gene.phenotype_values, torch.tensor(values))
    assert gene.phenotype_type_indices.tolist() == type_indices
    assert gene.phenotype_sample_indices.tolist() == [0]
    assert gene.phenotype_types == ["fitness", "gene_interaction"]
    torch.testing.assert_close(gene.phenotype_stat_values, torch.tensor(stat_values))
    assert gene.phenotype_stat_type_indices.tolist() == stat_type_indices
    assert gene.phenotype_stat_types == ["fitness_se", "gene_interaction_p_value"]


def test_get_returns_none_for_a_missing_key(dataset: Neo4jCellDataset) -> None:
    """A key the LMDB does not hold is None, not an error."""
    assert dataset.get(99) is None


def test_getitem_goes_through_get(dataset: Neo4jCellDataset) -> None:
    """`ds[3]` is `get(indices()[3])`, the gene-interaction record."""
    item = dataset[3]
    assert isinstance(item, HeteroData)
    torch.testing.assert_close(item["gene"].phenotype_values, torch.tensor([-0.2]))
    assert sorted(item["gene"].perturbed_genes) == ["YAL001C", "YAL004W"]


def test_read_deserialize_reconstruct_round_trip(dataset: Neo4jCellDataset) -> None:
    """The three item-path steps rebuild record 2 exactly, as pydantic objects."""
    dataset._init_lmdb_read()
    raw = dataset._read_from_lmdb(2)
    assert raw is not None
    data_list = dataset._deserialize_json(raw)
    assert [sorted(d) for d in data_list] == [["experiment", "experiment_reference"]]
    rebuilt = dataset._reconstruct_experiments(data_list)
    assert len(rebuilt) == 1
    assert type(rebuilt[0]["experiment"]) is FitnessExperiment
    assert type(rebuilt[0]["experiment_reference"]) is FitnessExperimentReference
    assert (
        rebuilt[0]["experiment"].model_dump() == RECORDS[2]["experiment"].model_dump()
    )
    assert (
        rebuilt[0]["experiment_reference"].model_dump()
        == RECORDS[2]["experiment_reference"].model_dump()
    )
    assert rebuilt[0]["experiment"].phenotype.fitness_std == 0.1
    assert rebuilt[0]["experiment"].phenotype.fitness_se is None


def test_label_df_is_exact_and_cached_to_parquet(
    dataset: Neo4jCellDataset, build_root: Path
) -> None:
    """One row per key, one column per label in `phenotype_labels` order, NaN where absent."""
    pd.testing.assert_frame_equal(dataset.label_df, _expected_label_df())
    assert dataset.env is None
    pd.testing.assert_frame_equal(
        pd.read_parquet(build_root / "processed" / "label_df.parquet"),
        _expected_label_df(),
    )


def test_second_instance_reads_the_cached_label_df(
    dataset: Neo4jCellDataset, build_root: Path
) -> None:
    """The parquet wins over recomputation: a sentinel frame written to it is what a new
    instance returns, while the first instance keeps its in-memory frame.
    """
    pd.testing.assert_frame_equal(dataset.label_df, _expected_label_df())
    sentinel = pd.DataFrame({"index": [7], "fitness": [0.1], "gene_interaction": [0.2]})
    sentinel.to_parquet(build_root / "processed" / "label_df.parquet")
    pd.testing.assert_frame_equal(_dataset(build_root).label_df, sentinel)
    pd.testing.assert_frame_equal(dataset.label_df, _expected_label_df())


def test_label_df_cache_ignores_a_different_label_order(
    dataset: Neo4jCellDataset, build_root: Path
) -> None:
    """Finding: `label_df` is keyed on the path alone (`neo4j_cell.py:659-668`), so an
    instance built with `phenotype_labels` reversed reads the cached frame in the FIRST
    instance's column order; its `phenotype_info` and the cached columns disagree.
    """
    assert list(dataset.label_df.columns) == ["index", "fitness", "gene_interaction"]
    reversed_ds = _dataset(build_root, phenotype_labels=list(reversed(LABELS)))
    assert _info(reversed_ds) == [GeneInteractionPhenotype, FitnessPhenotype]
    assert list(reversed_ds.label_df.columns) == [
        "index",
        "fitness",
        "gene_interaction",
    ]


def test_compute_phenotype_label_index(dataset: Neo4jCellDataset) -> None:
    """Fitness records are keys 0-2; the gene-interaction record is key 3."""
    assert dataset.compute_phenotype_label_index() == {
        "fitness": [0, 1, 2],
        "gene_interaction": [3],
    }
    assert dataset.env is None


def test_compute_dataset_name_index(dataset: Neo4jCellDataset) -> None:
    """toy_a holds keys 0-1, toy_b keys 2-3."""
    assert dataset.compute_dataset_name_index() == {"toy_a": [0, 1], "toy_b": [2, 3]}


def test_compute_perturbation_count_index_and_its_json_keys(
    dataset: Neo4jCellDataset, build_root: Path
) -> None:
    """Counts are int keys in memory and string keys on disk; the property converts back."""
    assert dataset.compute_perturbation_count_index() == {1: [0], 2: [1, 2, 3]}
    on_disk = json.loads(
        (build_root / "processed" / "perturbation_count_index.json").read_text()
    )
    assert on_disk == {"1": [0], "2": [1, 2, 3]}
    assert dataset.perturbation_count_index == {1: [0], 2: [1, 2, 3]}


def test_perturbed_and_deletion_gene_indices_differ_on_the_damp_allele(
    dataset: Neo4jCellDataset, build_root: Path
) -> None:
    """YAL004W is perturbed in records 2 (DAmP) and 3 (deletion) but deleted only in 3."""
    perturbed = {
        "YAL001C": [0, 3],
        "YAL002W": [1],
        "YAL003W": [1, 2],
        "YAL004W": [2, 3],
    }
    deletion = {"YAL001C": [0, 3], "YAL002W": [1], "YAL003W": [1, 2], "YAL004W": [3]}
    assert dataset.compute_is_any_perturbed_gene_index() == perturbed
    assert dataset.compute_is_any_deletion_gene_index() == deletion
    assert dataset.is_any_perturbed_gene_index == perturbed
    assert dataset.is_any_deletion_gene_index == deletion
    processed = build_root / "processed"
    assert (
        json.loads((processed / "is_any_perturbed_gene_index.json").read_text())
        == perturbed
    )
    assert (
        json.loads((processed / "is_any_deletion_gene_index.json").read_text())
        == deletion
    )


def test_index_properties_reread_their_json_on_every_access(
    dataset: Neo4jCellDataset, build_root: Path
) -> None:
    """The three constructor-tail indices are read from disk on each access, so an edited
    file changes what the SAME instance returns (the behavior `__getstate__` documents).
    """
    processed = build_root / "processed"
    (processed / "phenotype_label_index.json").write_text(json.dumps({"fitness": [42]}))
    (processed / "dataset_name_index.json").write_text(json.dumps({"other": [7]}))
    (processed / "perturbation_count_index.json").write_text(json.dumps({"9": [1]}))
    assert dataset.phenotype_label_index == {"fitness": [42]}
    assert dataset.dataset_name_index == {"other": [7]}
    assert dataset.perturbation_count_index == {9: [1]}
    fresh = _dataset(build_root)
    assert fresh.phenotype_label_index == {"fitness": [42]}
    assert fresh._phenotype_label_index == {"fitness": [42]}


def test_gene_index_properties_cache_in_memory_and_load_from_disk(
    dataset: Neo4jCellDataset, build_root: Path
) -> None:
    """Unlike the three above, the two gene indices are memory-cached after first access:
    an edited file does not reach the same instance but is what a new instance loads.
    """
    assert dataset.is_any_perturbed_gene_index["YAL004W"] == [2, 3]
    assert dataset.is_any_deletion_gene_index["YAL004W"] == [3]
    processed = build_root / "processed"
    (processed / "is_any_perturbed_gene_index.json").write_text(json.dumps({"P": [9]}))
    (processed / "is_any_deletion_gene_index.json").write_text(json.dumps({"D": [8]}))
    assert dataset.is_any_perturbed_gene_index["YAL004W"] == [2, 3]
    assert dataset.is_any_deletion_gene_index["YAL004W"] == [3]
    fresh = _dataset(build_root)
    assert fresh.is_any_perturbed_gene_index == {"P": [9]}
    assert fresh.is_any_deletion_gene_index == {"D": [8]}


def test_phenotype_info_order_follows_phenotype_labels(build_root: Path) -> None:
    """The class list is in `phenotype_labels` order; None gives the unordered full set.

    For None the source returns `list(set_of_classes)` (neo4j_cell.py:406), whose order
    the docstring disclaims, so that branch asserts the length and the set only.
    """
    assert _info(_dataset(build_root)) == [FitnessPhenotype, GeneInteractionPhenotype]
    assert _info(_dataset(build_root, phenotype_labels=list(reversed(LABELS)))) == [
        GeneInteractionPhenotype,
        FitnessPhenotype,
    ]
    # `list(set_of_classes)`: the docstring says this order is not guaranteed, so only
    # membership is a contract.
    unordered = _info(_dataset(build_root, phenotype_labels=None))
    assert len(unordered) == 2
    assert set(unordered) == {FitnessPhenotype, GeneInteractionPhenotype}


def test_unknown_phenotype_label_raises_on_first_phenotype_info_access(
    build_root: Path,
) -> None:
    """Finding: an unknown label is not rejected by `__init__`; the constructor tail never
    touches `phenotype_info`, so the error surfaces on the first `get`/`label_df`
    (`neo4j_cell.py:405-417` is reached only through `_load_phenotype_info`).
    """
    ds = _dataset(build_root, phenotype_labels=["nope"])
    assert ds.phenotype_labels == ["nope"]
    with pytest.raises(
        ValueError,
        match=(
            r"phenotype_labels \['nope'\] are not in this build "
            r"\(available: \['fitness', 'gene_interaction'\]\)"
        ),
    ):
        ds.phenotype_info


def test_fitness_only_labels_yield_one_value_per_record(build_root: Path) -> None:
    """`["fitness"]` selects one value per fitness record; the gene-interaction record has
    no fitness so the processor emits its NaN placeholder, and `label_df` has one label.
    """
    ds = _dataset(build_root, phenotype_labels=["fitness"])
    assert _info(ds) == [FitnessPhenotype]
    values = []
    for idx in range(3):
        item = ds.get(idx)
        assert isinstance(item, HeteroData)
        assert item["gene"].phenotype_types == ["fitness"]
        assert item["gene"].phenotype_stat_types == ["fitness_se"]
        assert item["gene"].phenotype_type_indices.tolist() == [0]
        values.append(item["gene"].phenotype_values)
    torch.testing.assert_close(torch.cat(values), torch.tensor([0.9, 0.4, 0.7]))
    placeholder = ds.get(3)
    assert isinstance(placeholder, HeteroData)
    assert torch.isnan(placeholder["gene"].phenotype_values).tolist() == [True]
    assert placeholder["gene"].phenotype_type_indices.tolist() == [0]
    ds.close_lmdb()
    pd.testing.assert_frame_equal(
        ds.label_df,
        pd.DataFrame({"index": [0, 1, 2, 3], "fitness": [0.9, 0.4, 0.7, np.nan]}),
    )


def test_compute_phenotype_info_writes_experiment_types(
    dataset: Neo4jCellDataset, build_root: Path
) -> None:
    """Without the file `phenotype_info` raises; `compute_phenotype_info` rebuilds it from
    the LMDB as the set of experiment types (a set, so the order on disk is not pinned).
    """
    path = build_root / "processed" / "experiment_types.json"
    path.unlink()
    with pytest.raises(
        FileNotFoundError,
        match="experiment_types.json not found. Please process the dataset first.",
    ):
        dataset.phenotype_info
    dataset.compute_phenotype_info()
    assert sorted(json.loads(path.read_text())) == ["fitness", "gene interaction"]
    assert dataset.env is None
    assert _info(dataset) == [FitnessPhenotype, GeneInteractionPhenotype]


def test_copy_lmdb_copies_every_entry_byte_for_byte(
    dataset: Neo4jCellDataset, build_root: Path
) -> None:
    """The copy holds the same 4 keys with identical values, under a created parent dir."""
    src = build_root / "processed" / "lmdb"
    dst = build_root / "copy" / "lmdb"
    dataset._copy_lmdb(str(src), str(dst))
    src_env = lmdb.open(str(src), readonly=True, lock=False)
    dst_env = lmdb.open(str(dst), readonly=True, lock=False)
    with src_env.begin() as src_txn, dst_env.begin() as dst_txn:
        src_entries = list(src_txn.cursor())
        dst_entries = list(dst_txn.cursor())
    src_env.close()
    dst_env.close()
    assert [k for k, _ in dst_entries] == [b"0", b"1", b"2", b"3"]
    assert dst_entries == src_entries
    assert json.loads(dst_entries[1][1])[0]["experiment"]["phenotype"]["fitness"] == 0.4


def test_pickle_round_trip_restores_the_read_path(dataset: Neo4jCellDataset) -> None:
    """A worker's copy arrives with no handle and no caches, and `get` reopens the LMDB."""
    dataset.label_df
    dataset.get(0)
    assert dataset.env is not None
    restored = pickle.loads(pickle.dumps(dataset))
    dataset.close_lmdb()
    assert restored.env is None
    assert restored._label_df is None
    assert restored._phenotype_label_index is None
    assert restored.phenotype_labels == LABELS
    assert restored.cell_graph["gene"].node_ids == dataset.cell_graph["gene"].node_ids
    item = restored.get(1)
    assert isinstance(item, HeteroData)
    torch.testing.assert_close(item["gene"].phenotype_values, torch.tensor([0.4]))
    assert item["gene"].perturbation_indices.tolist() == [1, 2]
    restored.close_lmdb()


def test_connection_settings_come_from_env_unless_given(
    build_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD fill the fields; explicit arguments win."""
    monkeypatch.setenv("NEO4J_URI", "bolt://sentinel-host:1")
    monkeypatch.setenv("NEO4J_USER", "sentinel-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "sentinel-password")
    from_env = _dataset(build_root)
    assert (from_env.uri, from_env.username, from_env.password) == (
        "bolt://sentinel-host:1",
        "sentinel-user",
        "sentinel-password",
    )
    assert from_env.query is None
    explicit = _dataset(
        build_root, uri="bolt://given:2", username="given", password="pw", query="MATCH"
    )
    assert (explicit.uri, explicit.username, explicit.password, explicit.query) == (
        "bolt://given:2",
        "given",
        "pw",
        "MATCH",
    )


def _physical_multigraph() -> GeneMultiGraph:
    physical = nx.Graph()
    physical.add_nodes_from(GENES)
    physical.add_edge("YAL001C", "YAL002W")
    return GeneMultiGraph(
        graphs=SortedDict(
            {"physical": GeneGraph(name="physical", graph=physical, max_gene_set=GENES)}
        )
    )


def test_graphs_argument_is_copied_and_given_a_base_graph(build_root: Path) -> None:
    """The one physical edge (0, 1) plus a self loop per node is the edge_index; the
    caller's multigraph does not receive the added base graph.
    """
    multigraph = _physical_multigraph()
    ds = _dataset(build_root, graphs=multigraph)
    assert list(multigraph.graphs.keys()) == ["physical"]
    assert ds.cell_graph.edge_types == [("gene", "physical_interaction", "gene")]
    assert ds.cell_graph[
        "gene", "physical_interaction", "gene"
    ].edge_index.tolist() == [[0, 0, 1, 2, 3], [1, 0, 1, 2, 3]]
    no_loops = _dataset(
        build_root, graphs=multigraph, add_remaining_gene_self_loops=False
    )
    assert no_loops.cell_graph[
        "gene", "physical_interaction", "gene"
    ].edge_index.tolist() == [[0], [1]]


class _FakeEmbeddingDataset:
    """`_data.embeddings[key]` as `[n, D]`, items as `Data(id, embeddings={key: [1, D]})`,
    the layout `BaseEmbeddingDataset.__getitem__` produces.
    """

    def __init__(self, ids: list[str], embeddings: dict[str, torch.Tensor]) -> None:
        self._data = Data(id=ids, embeddings=embeddings)

    def __len__(self) -> int:
        return len(self._data.id)

    def __getitem__(self, idx: int) -> Data:
        return Data(
            id=self._data.id[idx],
            embeddings={
                key: value[idx : idx + 1]
                for key, value in self._data.embeddings.items()
            },
        )

    def __iter__(self) -> Iterator[Data]:
        for idx in range(len(self)):
            yield self[idx]


def test_node_embeddings_become_gene_features(build_root: Path) -> None:
    """Key "e" is min-max normalized per column ([[1,5,2],[3,5,6],[2,5,4]] -> rows
    [0,.5,0], [1,.5,1], [.5,.5,.5]), key "f" is concatenated raw, genes without an
    embedding (YAL003W, YAL004W) get zero rows, and YZZ999X is outside the gene set.
    """
    fake = _FakeEmbeddingDataset(
        ["YAL001C", "YAL002W", "YZZ999X"],
        {
            "e": torch.tensor([[1.0, 5.0, 2.0], [3.0, 5.0, 6.0], [2.0, 5.0, 4.0]]),
            "f": torch.tensor([[10.0], [20.0], [30.0]]),
        },
    )
    ds = _dataset(build_root, node_embeddings={"fake": fake})
    assert ds.cell_graph["gene"].x.tolist() == [
        [0.0, 0.5, 0.0, 10.0],
        [1.0, 0.5, 1.0, 20.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    assert ds.cell_graph.edge_types == []
    # the normalizer rewrote the caller's store in place; "f" was never touched
    assert fake._data.embeddings["e"].tolist() == [
        [0.0, 0.5, 0.0],
        [1.0, 0.5, 1.0],
        [0.5, 0.5, 0.5],
    ]
    assert fake._data.embeddings["f"].tolist() == [[10.0], [20.0], [30.0]]
