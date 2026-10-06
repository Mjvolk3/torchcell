# tests/torchcell/data/test_neo4j_preprocessed_cell_full_masks.py
# [[tests.torchcell.data.test_neo4j_preprocessed_cell_full_masks]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_neo4j_preprocessed_cell_full_masks.py
"""`Neo4jPreprocessedCellDatasetFullMasks`: the uint8 full-mask store the 006
`_lazy_preprocessed` runs read when `use_full_masks` is set.

The store is written by `extract_full_masks` in
`experiments/006-kuzmin-tmi/scripts/preprocess_lazy_dataset_full_masks.py` (the only
writer; loaded here with `importlib`, `dotenv.load_dotenv` and `logging.basicConfig`
stubbed so importing it touches neither the environment nor the root logger). Records are
written from the live `Neo4jCellDataset` + `LazySubgraphRepresentation` item exactly as
that script's loop does (`pickle.dumps(extract_full_masks(item))` under key `"<idx>"`),
with its metadata `{"length", "storage_type": "full_masks", "created_at"}`.

Fixture: the same five-gene source build as `test_neo4j_preprocessed_cell.py`, copied
here (test modules do not import each other). Genes YAL001C=0, YAL002W=1, YAL003W=2,
YAL004W=3, YAL005C=4; physical edge_index [[0,1,3,0,1,2,3,4],[1,2,4,0,1,2,3,4]];
regulatory [[2,4,0,1,2,3,4],[0,2,0,1,2,3,4]]; reactions r_A (genes 0,1), r_B (gene 3),
r_C (no genes); GPR [[0,1,3],[0,0,1]]; RMR [[0,0,1,2],[0,1,1,0]].

Genotypes and hand-derived masks (edge kept iff neither endpoint deleted; reaction kept
iff all its genes kept; RMR kept iff its reaction kept; GPR kept iff its gene kept):

* 0: delete {1}: physical [F,F,T,T,F,T,T,T], regulatory [T,T,T,F,T,T,T],
  reactions [F,T,T], GPR [T,F,T], RMR [F,F,T,T].
* 1: delete {0,3}: physical [F,T,F,F,T,T,F,T], regulatory [F,T,F,T,T,F,T],
  reactions [F,F,T], GPR [F,T,F], RMR [F,F,F,T].
* 2: delete {4}: physical [T,T,F,T,T,T,T,F], regulatory [T,F,T,T,T,T,F], all
  reaction, GPR and RMR entries kept.
"""

import functools
import importlib.util
import json
import logging
import os
import pickle
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import dotenv
import lmdb
import networkx as nx
import pytest
import torch
from sortedcontainers import SortedDict
from torch_geometric.data import HeteroData

from torchcell.data.graph_processor import LazySubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.data.neo4j_preprocessed_cell import Neo4jPreprocessedCellDataset
from torchcell.data.neo4j_preprocessed_cell_full_masks import (
    Neo4jPreprocessedCellDatasetFullMasks,
)
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
)
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet

WRITER = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "006-kuzmin-tmi"
    / "scripts"
    / "preprocess_lazy_dataset_full_masks.py"
)

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W", "YAL004W", "YAL005C"])
ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
PHYS = ("gene", "physical_interaction", "gene")
REG = ("gene", "regulatory_interaction", "gene")
GPR = ("gene", "gpr", "reaction")
RMR = ("reaction", "rmr", "metabolite")


def _deletion(gene: str) -> KanMxDeletionPerturbation:
    return KanMxDeletionPerturbation(
        systematic_gene_name=gene, perturbed_gene_name=gene
    )


def _fitness(genes: list[str], fitness: float, se: float | None) -> dict[str, Any]:
    return {
        "experiment": FitnessExperiment(
            dataset_name="toy",
            genotype=Genotype(perturbations=[_deletion(g) for g in genes]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=fitness, fitness_se=se),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name="toy",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0),
        ),
    }


def _interaction(genes: list[str], value: float, p_value: float) -> dict[str, Any]:
    return {
        "experiment": GeneInteractionExperiment(
            dataset_name="toy",
            genotype=Genotype(perturbations=[_deletion(g) for g in genes]),
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=value, gene_interaction_p_value=p_value
            ),
        ),
        "experiment_reference": GeneInteractionExperimentReference(
            dataset_name="toy",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=GeneInteractionPhenotype(gene_interaction=0.0),
        ),
    }


RECORDS: list[dict[str, Any]] = [
    _fitness(["YAL002W"], 0.9, 0.01),
    _fitness(["YAL001C", "YAL004W"], 0.4, 0.05),
    _interaction(["YAL005C"], -0.2, 0.03),
]
LABELS = ["fitness", "gene_interaction"]


def _write_source(root: Path, records: list[dict[str, Any]], types: list[str]) -> None:
    processed = root / "processed"
    processed.mkdir(parents=True)
    env = lmdb.open(str(processed / "lmdb"), map_size=int(1e8))
    with env.begin(write=True) as txn:
        for idx, record in enumerate(records):
            payload = [{k: v.model_dump(mode="json") for k, v in record.items()}]
            txn.put(str(idx).encode(), json.dumps(payload).encode())
    env.close()
    (processed / "experiment_types.json").write_text(json.dumps(types))


def _multigraph() -> GeneMultiGraph:
    physical = nx.Graph()
    physical.add_nodes_from(GENES)
    physical.add_edges_from(
        [("YAL001C", "YAL002W"), ("YAL002W", "YAL003W"), ("YAL004W", "YAL005C")]
    )
    regulatory = nx.DiGraph()
    regulatory.add_nodes_from(GENES)
    regulatory.add_edges_from([("YAL003W", "YAL001C"), ("YAL005C", "YAL003W")])
    return GeneMultiGraph(
        graphs=SortedDict(
            {
                "physical": GeneGraph(
                    name="physical", graph=physical, max_gene_set=GENES
                ),
                "regulatory": GeneGraph(
                    name="regulatory", graph=regulatory, max_gene_set=GENES
                ),
            }
        )
    )


def _bipartite() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(
        "r_A",
        node_type="reaction",
        subsystem="Glycolysis",
        genes=["YAL001C", "YAL002W"],
    )
    graph.add_node(
        "r_B", node_type="reaction", subsystem="Glycolysis", genes=["YAL004W"]
    )
    graph.add_node("r_C", node_type="reaction", subsystem="Growth", genes=[])
    graph.add_node("m_x", node_type="metabolite")
    graph.add_node("m_y", node_type="metabolite")
    graph.add_edge("r_A", "m_x", edge_type="reactant", stoichiometry=1.0)
    graph.add_edge("r_A", "m_y", edge_type="product", stoichiometry=2.0)
    graph.add_edge("r_B", "m_y", edge_type="reactant", stoichiometry=1.0)
    graph.add_edge("r_C", "m_x", edge_type="product", stoichiometry=0.5)
    return graph


def _source(
    root: Path,
    records: list[dict[str, Any]] = RECORDS,
    labels: list[str] = LABELS,
    processor: LazySubgraphRepresentation | None = None,
    reopen_safe: bool = True,
) -> Neo4jCellDataset:
    if not (root / "processed" / "lmdb").exists():
        types = ["fitness", "gene interaction"] if len(labels) == 2 else ["fitness"]
        _write_source(root, records, types)
    kwargs: dict[str, Any] = {
        "root": str(root),
        "gene_set": GENES,
        "graphs": _multigraph(),
        "incidence_graphs": {"metabolism_bipartite": _bipartite()},
        "graph_processor": processor,
        "phenotype_labels": labels,
    }
    ds = Neo4jCellDataset(**kwargs)
    if reopen_safe:
        # Workaround for the lmdb >= 2 double-open (see the Finding test
        # `test_preprocess_reopens_the_source_lmdb_per_record`): close before reopen.
        ds.__dict__["_init_lmdb_read"] = functools.partial(_close_then_open, ds)
    return ds


def _close_then_open(ds: Neo4jCellDataset, readahead: bool = False) -> None:
    """`Neo4jCellDataset._init_lmdb_read`, preceded by `close_lmdb()`."""
    ds.close_lmdb()
    Neo4jCellDataset._init_lmdb_read(ds, readahead=readahead)


def _load_writer() -> ModuleType:
    """Import the experiment script with its import-time side effects stubbed.

    The stubs are live only around `exec_module`. Modules first imported during it that
    do `from dotenv import load_dotenv` (yeast_GEM, dcell, sgd, kemmeren2014,
    sameith2015) would keep the stub bound for the rest of the process, so every module
    holding the stub gets the real function back before this returns.
    """
    real_load_dotenv = dotenv.load_dotenv
    real_basic_config = logging.basicConfig

    def no_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

    def no_basic_config(*args: Any, **kwargs: Any) -> None:
        return None

    spec = importlib.util.spec_from_file_location("_full_mask_writer", WRITER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    dotenv.load_dotenv = no_dotenv
    logging.basicConfig = no_basic_config
    try:
        spec.loader.exec_module(module)
    finally:
        dotenv.load_dotenv = real_load_dotenv
        logging.basicConfig = real_basic_config
        # Read each module's own namespace, never `getattr`: a lazy module (transformers'
        # `_LazyModule`) answers an unknown attribute by importing submodules, which on a
        # runner without torchvision raises ModuleNotFoundError (PR #662 CI; the same
        # trap as PR #585 in `_run_mineru._patch_dpi`). `None` entries are blocked imports.
        for loaded in [*sys.modules.values(), module]:
            namespace = getattr(loaded, "__dict__", None)
            if namespace is not None and namespace.get("load_dotenv") is no_dotenv:
                namespace["load_dotenv"] = real_load_dotenv
    return module


def _write_store(
    root: Path, records: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(processed / "lmdb"), map_size=int(1e8))
    with env.begin(write=True) as txn:
        for idx, record in enumerate(records):
            txn.put(f"{idx}".encode(), pickle.dumps(record))
    env.close()
    (processed / "metadata.json").write_text(json.dumps(metadata, indent=2))


@pytest.fixture
def source(tmp_path: Path) -> Neo4jCellDataset:
    """Source build with no graph processor, as the 006 preprocessing script makes it."""
    return _source(tmp_path / "src")


@pytest.fixture
def live(tmp_path: Path, source: Neo4jCellDataset) -> Iterator[Neo4jCellDataset]:
    """The live path the store replaces: same build, Lazy processor.

    The source handle is closed first and this one at teardown: lmdb >= 2 refuses a
    second open of one path in one process.
    """
    source.close_lmdb()
    ds = _source(tmp_path / "src", processor=LazySubgraphRepresentation())
    yield ds
    ds.close_lmdb()


@pytest.fixture
def written(tmp_path: Path, live: Neo4jCellDataset) -> list[dict[str, Any]]:
    """The writer's record for each of the three live items, stored under `fm/`."""
    writer = _load_writer()
    records: list[dict[str, Any]] = []
    for idx in range(3):
        item = live.get(idx)
        assert isinstance(item, HeteroData)
        records.append(writer.extract_full_masks(item))
    live.close_lmdb()  # a later preprocess of the source must not meet this handle
    _write_store(
        tmp_path / "fm",
        records,
        {"length": 3, "storage_type": "full_masks", "created_at": "2026-10-01"},
    )
    return records


@pytest.fixture
def dataset(
    tmp_path: Path, source: Neo4jCellDataset, written: list[dict[str, Any]]
) -> Neo4jPreprocessedCellDatasetFullMasks:
    """The full-mask dataset over the written store."""
    return Neo4jPreprocessedCellDatasetFullMasks(
        root=str(tmp_path / "fm"), source_dataset=source
    )


_MISSING = "<missing>"


def _differences(
    a: HeteroData, b: HeteroData
) -> dict[tuple[Any, str], tuple[Any, Any]]:
    """Every (store, key) whose value differs, by dtype + exact tensor equality."""
    assert sorted(a.node_types) == sorted(b.node_types)
    assert sorted(a.edge_types) == sorted(b.edge_types)
    diffs: dict[tuple[Any, str], tuple[Any, Any]] = {}
    for store in [*a.node_types, *a.edge_types]:
        for key in sorted(set(a[store].keys()) | set(b[store].keys())):
            va = a[store][key] if key in a[store] else _MISSING
            vb = b[store][key] if key in b[store] else _MISSING
            if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                same = va.dtype == vb.dtype and torch.equal(va, vb)
            else:
                same = va == vb
            if not same:
                diffs[(store, key)] = (va, vb)
    return diffs


def _pert(item: HeteroData) -> list[int]:
    indices: list[int] = item["gene"].perturbation_indices.tolist()
    return indices


U8 = torch.uint8


def test_writer_record_layout(written: list[dict[str, Any]]) -> None:
    """Record 0 as the writer stores it: gene payload with `x_pert` None (the Lazy item
    has none) and `pert_mask` uint8 [0,1,0,0,0]; full uint8 node and edge masks; the
    non-gene `pert_mask` under top-level "reaction"/"metabolite" keys.
    """
    rec = written[0]
    assert list(rec) == ["gene", "node_masks", "reaction", "metabolite", "edge_masks"]
    gene = rec["gene"]
    assert list(gene)[-2:] == ["x_pert", "pert_mask"]
    assert gene["x_pert"] is None
    assert gene["pert_mask"].dtype == U8
    assert gene["pert_mask"].tolist() == [0, 1, 0, 0, 0]
    assert gene["perturbation_indices"].tolist() == [1]
    assert {k: (v.dtype, v.tolist()) for k, v in rec["node_masks"].items()} == {
        "gene": (U8, [1, 0, 1, 1, 1]),
        "reaction": (U8, [0, 1, 1]),
        "metabolite": (U8, [1, 1]),
    }
    assert {k: (v.dtype, v.tolist()) for k, v in rec["reaction"].items()} == {
        "pert_mask": (U8, [1, 0, 0])
    }
    assert {k: (v.dtype, v.tolist()) for k, v in rec["metabolite"].items()} == {
        "pert_mask": (U8, [0, 0])
    }
    assert {k: (v.dtype, v.tolist()) for k, v in rec["edge_masks"].items()} == {
        PHYS: (U8, [0, 0, 1, 1, 0, 1, 1, 1]),
        REG: (U8, [1, 1, 1, 0, 1, 1, 1]),
        GPR: (U8, [1, 0, 1]),
        RMR: (U8, [0, 0, 1, 1]),
    }


@pytest.mark.parametrize(
    ("idx", "masks"),
    [
        (
            0,
            {
                PHYS: [0, 0, 1, 1, 0, 1, 1, 1],
                REG: [1, 1, 1, 0, 1, 1, 1],
                GPR: [1, 0, 1],
                RMR: [0, 0, 1, 1],
                "gene": [1, 0, 1, 1, 1],
                "reaction": [0, 1, 1],
                "metabolite": [1, 1],
            },
        ),
        (
            1,
            {
                PHYS: [0, 1, 0, 0, 1, 1, 0, 1],
                REG: [0, 1, 0, 1, 1, 0, 1],
                GPR: [0, 1, 0],
                RMR: [0, 0, 0, 1],
                "gene": [0, 1, 1, 0, 1],
                "reaction": [0, 0, 1],
                "metabolite": [1, 1],
            },
        ),
        (
            2,
            {
                PHYS: [1, 1, 0, 1, 1, 1, 1, 0],
                REG: [1, 0, 1, 1, 1, 1, 0],
                GPR: [1, 1, 1],
                RMR: [1, 1, 1, 1],
                "gene": [1, 1, 1, 1, 0],
                "reaction": [1, 1, 1],
                "metabolite": [1, 1],
            },
        ),
    ],
)
def test_loaded_masks_equal_the_hand_derived_sets(
    dataset: Neo4jPreprocessedCellDatasetFullMasks,
    idx: int,
    masks: dict[Any, list[int]],
) -> None:
    """uint8 on disk comes back as bool masks equal to the module docstring's sets; every
    node `pert_mask` is bool and exactly `~mask`; no kept edge touches a deleted gene.
    """
    item = dataset[idx]
    for store, expected in masks.items():
        assert item[store].mask.dtype == torch.bool
        assert item[store].mask.tolist() == [bool(v) for v in expected]
    for node_type in ["gene", "reaction", "metabolite"]:
        assert item[node_type].pert_mask.dtype == torch.bool
        assert item[node_type].pert_mask.tolist() == (~item[node_type].mask).tolist()
    deleted = set(_pert(item))
    for et in [PHYS, REG]:
        kept = item[et].edge_index[:, item[et].mask].tolist()
        assert not deleted & (set(kept[0]) | set(kept[1]))


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_item_equals_the_live_lazy_item_except_two_fields(
    dataset: Neo4jPreprocessedCellDatasetFullMasks, live: Neo4jCellDataset, idx: int
) -> None:
    """Every key of every store, dtype and value exact, against the live Lazy item
    (`x_pert` None is skipped, so the key sets match). Exactly the two keys the next two
    Finding tests pin differ.
    """
    live_item = live.get(idx)
    assert isinstance(live_item, HeteroData)
    diffs = _differences(live_item, dataset[idx])
    assert sorted(diffs, key=str) == sorted(
        [("reaction", "node_ids"), (GPR, "num_edges")], key=str
    )


def test_reaction_node_ids_differ_from_the_live_item(
    dataset: Neo4jPreprocessedCellDatasetFullMasks, live: Neo4jCellDataset
) -> None:
    """Finding: `_load_full_masks` copies the cell graph's reaction names
    (`neo4j_preprocessed_cell_full_masks.py:215`) where the live Lazy item has positions
    `valid_reactions.tolist()` (`graph_processor.py:1724`). Latent: the lazy collater
    keeps reaction `node_ids` as a per-sample list the lazy model never reads; reached
    only in that the 006 configs 077 (full masks) and 081/082 (compact) read these
    stores (audit 2). Pinned until one side is changed to match the other.
    """
    live_item = live.get(1)
    assert isinstance(live_item, HeteroData)
    assert live_item["reaction"].node_ids == [0, 1, 2]
    assert dataset[1]["reaction"].node_ids == ["r_A", "r_B", "r_C"]


def test_gpr_num_edges_differs_from_the_live_item(
    dataset: Neo4jPreprocessedCellDatasetFullMasks, live: Neo4jCellDataset
) -> None:
    """Finding: GPR `num_edges` is copied from the cell graph
    (`neo4j_preprocessed_cell_full_masks.py:204-207`), where it counts unique reactions
    (2), while the live item and the mask itself count hyperedges (3). Latent: the lazy
    model computes edge counts from `edge_index`
    (`hetero_cell_bipartite_dango_gi_lazy.py:1199`), not from `num_edges`; reached only
    in that 006 configs 077, 081 and 082 read these stores (audit 2). Pinned until the
    loader recomputes it from the hyperedge_index.
    """
    live_item = live.get(1)
    assert isinstance(live_item, HeteroData)
    assert live_item[GPR].num_edges == 3
    item = dataset[1]
    assert item[GPR].num_edges == 2
    assert item[GPR].mask.numel() == 3


def test_full_mask_item_equals_the_compact_item(
    tmp_path: Path,
    dataset: Neo4jPreprocessedCellDatasetFullMasks,
    source: Neo4jCellDataset,
) -> None:
    """The two preprocessed variants serve identical items for the same build."""
    compact = Neo4jPreprocessedCellDataset(
        root=str(tmp_path / "compact"), source_dataset=source
    )
    compact.preprocess_from_source(source, LazySubgraphRepresentation())
    for idx in range(3):
        assert _differences(compact[idx], dataset[idx]) == {}


def test_hand_built_record_without_gene_pert_mask(
    tmp_path: Path, source: Neo4jCellDataset
) -> None:
    """A record whose gene `pert_mask` is None (the writer's value when the item lacks
    one): the key is skipped, then derived as `~mask` from `node_masks["gene"]`; a
    bool-typed edge mask is kept bool; a non-pert_mask uint8 gene tensor is NOT cast
    (only the key "pert_mask" is converted).
    """
    record = {
        "gene": {
            "perturbation_indices": torch.tensor([2]),
            "pert_mask": None,
            "x_pert": None,
            "extra_u8": torch.tensor([1, 0], dtype=U8),
            "ids_pert": ["YAL003W"],
        },
        "node_masks": {"gene": torch.tensor([1, 1, 0, 1, 1], dtype=U8)},
        "edge_masks": {PHYS: torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)},
    }
    _write_store(tmp_path / "hb", [record], {"length": 1, "storage_type": "full_masks"})
    ds = Neo4jPreprocessedCellDatasetFullMasks(
        root=str(tmp_path / "hb"), source_dataset=source
    )
    item = ds[0]
    gene = item["gene"]
    assert "x_pert" not in gene
    assert gene.mask.tolist() == [True, True, False, True, True]
    assert gene.pert_mask.tolist() == [False, False, True, False, False]
    assert gene.extra_u8.dtype == U8
    assert gene.ids_pert == ["YAL003W"]
    assert item[PHYS].mask.tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert "mask" not in item[REG]
    assert "pert_mask" not in item["reaction"]
    assert item["reaction"].node_ids == ["r_A", "r_B", "r_C"]


# ---------------------------------------------------------------- construction, I/O


def test_constructor_creates_processed_dir_and_refuses_reads(tmp_path: Path) -> None:
    """Unlike the compact variant, the constructor makes `<root>/processed` at once
    (`os.makedirs`), skips PyG's `Dataset.__init__` entirely (no `pre_transform`
    attribute), and both read paths refuse with the same full message.
    """
    root = tmp_path / "fm"
    ds = Neo4jPreprocessedCellDatasetFullMasks(root=str(root))
    assert sorted(os.listdir(root)) == ["processed"]
    assert os.listdir(root / "processed") == []
    assert ds.processed_dir == str(root / "processed")
    assert ds.transform is None
    assert ds._indices is None
    assert not hasattr(ds, "pre_transform")
    assert ds.cell_graph is None
    assert ds.phenotype_info is None
    msg = r"^Dataset not preprocessed\. Run preprocessing script first\.$"
    with pytest.raises(RuntimeError, match=msg):
        len(ds)
    with pytest.raises(RuntimeError, match=msg):
        ds.get(0)
    with pytest.raises(
        RuntimeError,
        match=r"^label_df not available\. Provide source_dataset at initialization\.$",
    ):
        ds.label_df


def test_without_a_source_get_fails_on_none_cell_graph(
    tmp_path: Path, written: list[dict[str, Any]]
) -> None:
    """Finding: with no source the constructor sets `cell_graph = None`
    (`neo4j_preprocessed_cell_full_masks.py:70`) and `get` dereferences it at line 177,
    a bare TypeError, where the compact variant raises a RuntimeError naming the missing
    source. Pinned until the full-mask loader refuses with a message.
    """
    ds = Neo4jPreprocessedCellDatasetFullMasks(root=str(tmp_path / "fm"))
    assert len(ds) == 3
    with pytest.raises(TypeError, match="^'NoneType' object is not subscriptable$"):
        ds.get(0)


def test_source_attributes_and_length_from_metadata(
    dataset: Neo4jPreprocessedCellDatasetFullMasks, source: Neo4jCellDataset
) -> None:
    """`cell_graph`/`phenotype_info` are the source's own objects, bound in the
    constructor; `_length` comes from metadata at construction; `label_df` is the
    source's frame.
    """
    assert dataset.cell_graph is source.cell_graph
    assert dataset.phenotype_info is source.phenotype_info
    assert dataset._length == 3
    assert len(dataset) == 3
    assert dataset.label_df["fitness"].tolist()[:2] == [0.9, 0.4]


def test_missing_key_and_index_mapping(
    dataset: Neo4jPreprocessedCellDatasetFullMasks,
) -> None:
    """`get(3)` raises with the key named (the compact variant returns None); `ds[3]`
    stops at `range(3)`; `ds[-1]` is record 2; a transform set after construction runs in
    `__getitem__` only.
    """
    with pytest.raises(IndexError, match="^Sample 3 not found in LMDB$"):
        dataset.get(3)
    with pytest.raises(IndexError, match="^range object index out of range$"):
        dataset[3]
    assert _pert(dataset[-1]) == [4]
    view = dataset[[1, 0]]
    # a `copy.copy` through `__getstate__`: no shared handle, so close the parent's
    # first (lmdb >= 2 refuses two opens of one path in one process)
    assert view.env is None
    dataset.close_lmdb()
    assert len(view) == 2
    assert _pert(view[0]) == [0, 3]
    view.close_lmdb()

    def tag(item: HeteroData) -> HeteroData:
        item["gene"].tag = 5
        return item

    dataset.__dict__["transform"] = tag  # the constructor binds it as None
    assert dataset[0]["gene"].tag == 5
    assert "tag" not in dataset.get(0)["gene"]


def test_env_lifecycle_and_pickle(
    dataset: Neo4jPreprocessedCellDatasetFullMasks,
) -> None:
    """`get` opens the env lazily; a pickled copy carries `env=None` and reopens on read;
    `close_lmdb` is idempotent.
    """
    assert dataset.env is None
    dataset.get(1)
    assert dataset.env is not None
    clone = pickle.loads(pickle.dumps(dataset))
    assert clone.env is None
    assert dataset.env is not None
    dataset.close_lmdb()  # lmdb >= 2: one open handle per path per process
    assert dataset.env is None
    assert _pert(clone[1]) == [0, 3]
    assert clone.env is not None
    clone.close_lmdb()
    dataset.close_lmdb()
    assert dataset.env is None


def test_storage_type_mismatch_only_warns_then_fails_on_read(
    tmp_path: Path, source: Neo4jCellDataset, caplog: pytest.LogCaptureFixture
) -> None:
    """Finding: pointing the full-mask loader at a COMPACT store (metadata has no
    `storage_type`) only logs a warning (`neo4j_preprocessed_cell_full_masks.py:111-112`);
    the first read then fails deep in `_load_full_masks` because a node mask entry is the
    compact `{"false_indices", "mask_size"}` dict. Pinned until a storage-type mismatch
    is a refusal.
    """
    compact = Neo4jPreprocessedCellDataset(
        root=str(tmp_path / "c"), source_dataset=source
    )
    compact.preprocess_from_source(source, LazySubgraphRepresentation())
    with caplog.at_level(
        logging.WARNING, logger="torchcell.data.neo4j_preprocessed_cell_full_masks"
    ):
        ds = Neo4jPreprocessedCellDatasetFullMasks(
            root=str(tmp_path / "c"), source_dataset=source
        )
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["Expected storage_type 'full_masks', got 'unknown'"]
    assert len(ds) == 3
    with pytest.raises(
        AttributeError,
        match="^" + re.escape("'dict' object has no attribute 'to'") + "$",
    ):
        ds.get(0)


def test_existing_store_is_served_for_a_different_source(
    tmp_path: Path, written: list[dict[str, Any]]
) -> None:
    """Finding: as in the compact variant, nothing ties the store to its source. A
    one-record source (YAL003W deleted) gets length 3 and the old record 0 (YAL002W).
    Reached: the 006 configs 077 (full masks) and 081/082 (compact) read these stores.
    Pinned until the metadata records a source fingerprint checked on load.
    """
    other = _source(tmp_path / "other", records=[_fitness(["YAL003W"], 0.5, 0.02)])
    ds = Neo4jPreprocessedCellDatasetFullMasks(
        root=str(tmp_path / "fm"), source_dataset=other
    )
    assert len(other) == 1
    assert len(ds) == 3
    assert _pert(ds[0]) == [1]


def test_len_loads_metadata_written_after_construction(
    tmp_path: Path, written: list[dict[str, Any]]
) -> None:
    """An instance built before the store exists reads the metadata on its first `len`."""
    early = Neo4jPreprocessedCellDatasetFullMasks(root=str(tmp_path / "late"))
    assert early._length is None
    _write_store(
        tmp_path / "late", written, {"length": 3, "storage_type": "full_masks"}
    )
    assert len(early) == 3
    assert early._length == 3


def test_writer_import_leaves_no_stubbed_load_dotenv(
    written: list[dict[str, Any]],
) -> None:
    """After `_load_writer` runs, `dotenv.load_dotenv` and every module that bound it by
    name during the import (dcell, yeast_GEM, sgd, kemmeren2014, sameith2015) hold the
    real function again, and `logging.basicConfig` is the real one.
    """
    import torchcell.datasets.scerevisiae.kemmeren2014 as kemmeren2014
    import torchcell.datasets.scerevisiae.sameith2015 as sameith2015
    import torchcell.graph.sgd as sgd
    import torchcell.metabolism.yeast_GEM as yeast_gem
    import torchcell.models.dcell as dcell

    assert len(written) == 3
    assert dotenv.load_dotenv.__module__ == "dotenv.main"
    for module in [dcell, yeast_gem, sgd, kemmeren2014, sameith2015]:
        assert module.load_dotenv is dotenv.load_dotenv, module.__name__
    assert logging.basicConfig.__module__ == "logging"


class _LazyReachesTorchvision(ModuleType):
    """A `sys.modules` entry shaped like transformers' `_LazyModule` on a runner without
    torchvision: any non-dunder attribute access raises ModuleNotFoundError.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise ModuleNotFoundError("No module named 'torchvision'")


def test_writer_loader_never_probes_module_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stub restore reads each module's `__dict__`, so a lazy module that would import
    torchvision on attribute access is never asked (PR #662 CI: a `getattr` scan errored
    every test using `written` with `No module named 'torchvision'`). The loaded writer is
    still the real script: `extract_full_masks` is defined in it.
    """
    lazy = _LazyReachesTorchvision("lazy_reaches_torchvision")
    monkeypatch.setitem(sys.modules, "lazy_reaches_torchvision", lazy)
    monkeypatch.setitem(sys.modules, "blocked_import", None)
    with pytest.raises(ModuleNotFoundError, match="^No module named 'torchvision'$"):
        lazy.load_dotenv  # noqa: B018  (the probe the loader must not make)
    writer = _load_writer()
    assert writer.extract_full_masks.__module__ == "_full_mask_writer"
    assert writer.extract_full_masks.__code__.co_filename == str(WRITER)
    assert vars(writer)["load_dotenv"] is dotenv.load_dotenv
