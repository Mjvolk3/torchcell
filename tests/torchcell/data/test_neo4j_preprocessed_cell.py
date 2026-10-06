# tests/torchcell/data/test_neo4j_preprocessed_cell.py
# [[tests.torchcell.data.test_neo4j_preprocessed_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_neo4j_preprocessed_cell.py
"""`Neo4jPreprocessedCellDataset`: the compact mask-index store the 006 `_lazy_preprocessed`
runs train from, checked against the live `Neo4jCellDataset` + `LazySubgraphRepresentation`
item it replaces.

Fixture (no Neo4j): a hand-written source build, the same shape as
`test_neo4j_cell_hermetic_build.py` (`processed/lmdb` with the aggregator's JSON array per
key, plus `experiment_types.json`, so PyG never runs `Neo4jCellDataset.process`).

Five genes, sorted by `GeneSet`: YAL001C=0, YAL002W=1, YAL003W=2, YAL004W=3, YAL005C=4.

Gene graphs (`to_cell_data` appends one self loop per node after the given edges):

* physical (undirected, networkx edge order): (0,1) (1,2) (3,4), then loops 0..4, so
  edge_index [[0,1,3,0,1,2,3,4],[1,2,4,0,1,2,3,4]], 8 edges.
* regulatory (directed): 2->0, 4->2, then loops 0..4, so
  edge_index [[2,4,0,1,2,3,4],[0,2,0,1,2,3,4]], 7 edges.

Metabolism bipartite (reactions inserted first): r_A genes [YAL001C, YAL002W], r_B genes
[YAL004W], r_C no genes; metabolites m_x, m_y. Sorted reactions r_A=0, r_B=1, r_C=2, so
GPR hyperedge_index [[0,1,3],[0,0,1]] (cell-graph num_edges = 2 unique reactions) and RMR
hyperedge_index [[0,0,1,2],[0,1,1,0]] (r_A-m_x, r_A-m_y, r_B-m_y, r_C-m_x).

Three genotypes (one LMDB key each); an edge survives iff NEITHER endpoint is deleted, a
reaction survives iff ALL its genes survive (r_C has none, always kept), an RMR edge
survives iff its reaction survives, a GPR edge survives iff its gene survives:

* 0: delete {1}; fitness 0.9, fitness_se 0.01.
  physical [F,F,T,T,F,T,T,T] (False at 0,1,4); regulatory [T,T,T,F,T,T,T] (False at 3);
  reactions [F,T,T]; GPR [T,F,T]; RMR [F,F,T,T].
* 1: delete {0,3}; fitness 0.4, fitness_se 0.05.
  physical [F,T,F,F,T,T,F,T] (False at 0,2,3,6); regulatory [F,T,F,T,T,F,T] (0,2,5);
  reactions [F,F,T]; GPR [F,T,F]; RMR [F,F,F,T].
* 2: delete {4}; gene_interaction -0.2, p-value 0.03 (type index 1).
  physical False at 2,7; regulatory False at 1,6; every reaction, GPR and RMR edge kept.
"""

import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import lmdb
import networkx as nx
import pytest
import torch
from sortedcontainers import SortedDict
from torch_geometric.data import HeteroData

from torchcell.data.graph_processor import LazySubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.data.neo4j_preprocessed_cell import Neo4jPreprocessedCellDataset
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
    return Neo4jCellDataset(**kwargs)


@pytest.fixture
def source(tmp_path: Path) -> Neo4jCellDataset:
    """Source build with no graph processor, as the 006 preprocessing script makes it."""
    return _source(tmp_path / "src")


@pytest.fixture
def preprocessed(
    tmp_path: Path, source: Neo4jCellDataset
) -> Neo4jPreprocessedCellDataset:
    """The compact store written from `source` with a fresh LazySubgraphRepresentation."""
    ds = Neo4jPreprocessedCellDataset(root=str(tmp_path / "pre"), source_dataset=source)
    ds.preprocess_from_source(source, LazySubgraphRepresentation())
    return ds


@pytest.fixture
def live(tmp_path: Path, source: Neo4jCellDataset) -> Neo4jCellDataset:
    """The live path the preprocessed store replaces: same build, Lazy processor."""
    return _source(tmp_path / "src", processor=LazySubgraphRepresentation())


def _read_raw(root: Path, idx: int) -> dict[str, Any]:
    env = lmdb.open(str(root / "processed" / "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        raw = txn.get(str(idx).encode())
    env.close()
    assert raw is not None
    loaded: dict[str, Any] = pickle.loads(raw)
    return loaded


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


# ----------------------------------------------------------------------------- fixture pin


def test_fixture_cell_graph_matches_the_hand_derivation(
    source: Neo4jCellDataset,
) -> None:
    """The edge lists the module docstring derives every mask from."""
    cg = source.cell_graph
    assert cg[PHYS].edge_index.tolist() == [
        [0, 1, 3, 0, 1, 2, 3, 4],
        [1, 2, 4, 0, 1, 2, 3, 4],
    ]
    assert cg[REG].edge_index.tolist() == [[2, 4, 0, 1, 2, 3, 4], [0, 2, 0, 1, 2, 3, 4]]
    assert cg[GPR].hyperedge_index.tolist() == [[0, 1, 3], [0, 0, 1]]
    assert cg[GPR].num_edges == 2
    assert cg[RMR].hyperedge_index.tolist() == [[0, 0, 1, 2], [0, 1, 1, 0]]
    assert cg["reaction"].node_ids == ["r_A", "r_B", "r_C"]


# ----------------------------------------------------------------------- construction


def test_construction_on_an_empty_root_writes_nothing(tmp_path: Path) -> None:
    """The class defines no `process`/`download`, so PyG's constructor runs neither and
    the root is never created; `processed_dir` is PyG's `<root>/processed`.
    """
    root = tmp_path / "pre"
    ds = Neo4jPreprocessedCellDataset(root=str(root))
    assert ds.has_process is False
    assert ds.has_download is False
    assert not root.exists()
    assert ds.processed_dir == str(root / "processed")
    assert ds.raw_file_names == []
    assert ds.processed_file_names == ["lmdb", "metadata.json"]
    assert ds._is_preprocessed() is False
    assert ds._length is None
    assert ds.env is None


@pytest.mark.parametrize(
    ("accessor", "message"),
    [
        (
            "cell_graph",
            "cell_graph not available. Either provide source_dataset at "
            "initialization or call preprocess_from_source first.",
        ),
        (
            "phenotype_info",
            "phenotype_info not available. Either provide source_dataset at "
            "initialization or call preprocess_from_source first.",
        ),
        (
            "label_df",
            "label_df not available. Provide source_dataset at initialization.",
        ),
    ],
)
def test_source_properties_refuse_without_a_source(
    tmp_path: Path, accessor: str, message: str
) -> None:
    """Each source-backed property raises its own full message when no source is set."""
    ds = Neo4jPreprocessedCellDataset(root=str(tmp_path / "pre"))
    with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
        getattr(ds, accessor)


def test_len_and_get_refuse_before_preprocessing(tmp_path: Path) -> None:
    """Both read paths name the missing step; `len` does not cache a failed load."""
    ds = Neo4jPreprocessedCellDataset(root=str(tmp_path / "pre"))
    msg = r"^Dataset not preprocessed\. Call preprocess_from_source\(\) first\.$"
    with pytest.raises(RuntimeError, match=msg):
        len(ds)
    with pytest.raises(RuntimeError, match=msg):
        ds.get(0)
    assert ds._length is None
    assert ds.env is None


def test_source_properties_are_read_once_and_cached(
    tmp_path: Path, source: Neo4jCellDataset
) -> None:
    """`cell_graph` / `phenotype_info` are the source's objects, cached on first access;
    `label_df` is re-read from the source every time (no cache attribute).
    """
    ds = Neo4jPreprocessedCellDataset(root=str(tmp_path / "pre"), source_dataset=source)
    assert ds._cell_graph is None
    assert ds.cell_graph is source.cell_graph
    assert ds._cell_graph is source.cell_graph
    assert ds.phenotype_info is source.phenotype_info
    info: list[Any] = list(ds.phenotype_info)  # classes, annotated as instances
    assert info == [FitnessPhenotype, GeneInteractionPhenotype]
    assert ds.label_df["fitness"].tolist()[:2] == [0.9, 0.4]
    assert ds.label_df["gene_interaction"].tolist()[2] == -0.2


# ------------------------------------------------------------------- on-disk layout


def test_preprocess_writes_exactly_the_lmdb_and_metadata(
    tmp_path: Path, preprocessed: Neo4jPreprocessedCellDataset
) -> None:
    """`processed/` holds the LMDB dir, `metadata.json` = {"length": 3}, and the
    FileLockHelper lock; the LMDB has exactly the keys b"0", b"1", b"2".
    """
    processed = tmp_path / "pre" / "processed"
    assert sorted(os.listdir(processed)) == [
        "lmdb",
        "metadata.json",
        "metadata.json.lock",
    ]
    assert json.loads((processed / "metadata.json").read_text()) == {"length": 3}
    env = lmdb.open(str(processed / "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        keys = [k for k, _ in txn.cursor()]
    env.close()
    assert keys == [b"0", b"1", b"2"]
    assert preprocessed._length == 3
    assert len(preprocessed) == 3


@pytest.mark.parametrize(
    ("idx", "node_false", "edge_false"),
    [
        (
            0,
            {"gene": [1], "reaction": [0], "metabolite": []},
            {PHYS: [0, 1, 4], REG: [3], GPR: [1], RMR: [0, 1]},
        ),
        (
            1,
            {"gene": [0, 3], "reaction": [0, 1], "metabolite": []},
            {PHYS: [0, 2, 3, 6], REG: [0, 2, 5], GPR: [0, 2], RMR: [0, 1, 2]},
        ),
        (
            2,
            {"gene": [4], "reaction": [], "metabolite": []},
            {PHYS: [2, 7], REG: [1, 6], GPR: [], RMR: []},
        ),
    ],
)
def test_compact_record_stores_exactly_the_false_indices(
    tmp_path: Path,
    preprocessed: Neo4jPreprocessedCellDataset,
    idx: int,
    node_false: dict[str, list[int]],
    edge_false: dict[Any, list[int]],
) -> None:
    """The pickled record: top keys gene/node_masks/edge_masks; per node and edge type
    the int64 positions where the live mask is False (module docstring derivation) and the
    full mask length (genes 5, reactions 3, metabolites 2; edges 8/7/3/4).
    """
    record = _read_raw(tmp_path / "pre", idx)
    assert list(record) == ["gene", "node_masks", "edge_masks"]
    sizes = {"gene": 5, "reaction": 3, "metabolite": 2, PHYS: 8, REG: 7, GPR: 3, RMR: 4}
    assert sorted(record["node_masks"]) == ["gene", "metabolite", "reaction"]
    for node_type, false in node_false.items():
        entry = record["node_masks"][node_type]
        assert entry["false_indices"].dtype == torch.int64
        assert entry["false_indices"].tolist() == false
        assert entry["mask_size"] == sizes[node_type]
    assert sorted(record["edge_masks"]) == sorted([PHYS, REG, GPR, RMR])
    for edge_type, false in edge_false.items():
        entry = record["edge_masks"][edge_type]
        assert entry["false_indices"].dtype == torch.int64
        assert entry["false_indices"].tolist() == false
        assert entry["mask_size"] == sizes[edge_type]


@pytest.mark.parametrize(
    ("idx", "genes", "value", "type_idx", "stat", "stat_type_idx"),
    [
        (0, ["YAL002W"], 0.9, 0, 0.01, 0),
        (1, ["YAL001C", "YAL004W"], 0.4, 0, 0.05, 0),
        (2, ["YAL005C"], -0.2, 1, 0.03, 1),
    ],
)
def test_compact_record_gene_payload(
    tmp_path: Path,
    preprocessed: Neo4jPreprocessedCellDataset,
    idx: int,
    genes: list[str],
    value: float,
    type_idx: int,
    stat: float,
    stat_type_idx: int,
) -> None:
    """The ten gene keys in writer order, float32 values, int64 indices, label lists."""
    gene = _read_raw(tmp_path / "pre", idx)["gene"]
    assert list(gene) == [
        "ids_pert",
        "perturbation_indices",
        "phenotype_values",
        "phenotype_type_indices",
        "phenotype_sample_indices",
        "phenotype_types",
        "phenotype_stat_values",
        "phenotype_stat_type_indices",
        "phenotype_stat_sample_indices",
        "phenotype_stat_types",
    ]
    assert sorted(gene["ids_pert"]) == genes
    assert gene["perturbation_indices"].tolist() == [GENES.index(g) for g in genes]
    assert gene["perturbation_indices"].dtype == torch.int64
    assert gene["phenotype_values"].dtype == torch.float32
    torch.testing.assert_close(gene["phenotype_values"], torch.tensor([value]))
    assert gene["phenotype_type_indices"].tolist() == [type_idx]
    assert gene["phenotype_sample_indices"].tolist() == [0]
    assert gene["phenotype_types"] == ["fitness", "gene_interaction"]
    torch.testing.assert_close(gene["phenotype_stat_values"], torch.tensor([stat]))
    assert gene["phenotype_stat_type_indices"].tolist() == [stat_type_idx]
    assert gene["phenotype_stat_sample_indices"].tolist() == [0]
    assert gene["phenotype_stat_types"] == ["fitness_se", "gene_interaction_p_value"]


# ----------------------------------------------------------------- equals the live item


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_item_equals_the_live_lazy_item_except_two_fields(
    preprocessed: Neo4jPreprocessedCellDataset, live: Neo4jCellDataset, idx: int
) -> None:
    """Every key of every node and edge store, dtype and value exact, against
    `Neo4jCellDataset(graph_processor=LazySubgraphRepresentation()).get(idx)`.

    Exactly two keys differ, the same two for every record (see the two Finding tests
    below): reaction `node_ids` and the GPR `num_edges`.
    """
    live_item = live.get(idx)
    assert isinstance(live_item, HeteroData)
    diffs = _differences(live_item, preprocessed[idx])
    assert sorted(diffs, key=str) == sorted(
        [("reaction", "node_ids"), (GPR, "num_edges")], key=str
    )


def test_reaction_node_ids_differ_from_the_live_item(
    preprocessed: Neo4jPreprocessedCellDataset, live: Neo4jCellDataset
) -> None:
    """Finding: the reconstructed item copies the cell graph's reaction ids
    (`neo4j_preprocessed_cell.py:255`, sorted names) while the live Lazy item carries
    `valid_reactions.tolist()` (`graph_processor.py:1724`, positions 0..R-1), so a model
    reading reaction `node_ids` sees names from the store and integers live. Latent: the
    lazy collater keeps reaction `node_ids` as a per-sample list the lazy model never
    reads; reached only in that the 006 configs 077 (full masks) and 081/082 (compact)
    read these stores (audit 2). Pinned until one side is changed to match the other.
    """
    live_item = live.get(0)
    assert isinstance(live_item, HeteroData)
    assert live_item["reaction"].node_ids == [0, 1, 2]
    assert preprocessed[0]["reaction"].node_ids == ["r_A", "r_B", "r_C"]


def test_gpr_num_edges_differs_from_the_live_item(
    preprocessed: Neo4jPreprocessedCellDataset, live: Neo4jCellDataset
) -> None:
    """Finding: the reconstructed GPR `num_edges` is the cell graph's value, the count of
    unique reactions with a gene (`cell_data.py:557-559`: 2 for r_A, r_B), copied at
    `neo4j_preprocessed_cell.py:244-247`; the live Lazy item sets the hyperedge count
    `gpr_edge_index.size(1)` = 3 (`graph_processor.py:1691-1693`). They differ whenever
    a reaction has more than one gene. Latent: the lazy model computes edge counts from
    `edge_index` (`hetero_cell_bipartite_dango_gi_lazy.py:1199`), not from `num_edges`;
    reached only in that 006 configs 077, 081 and 082 read these stores (audit 2).
    Pinned until the reconstruction recomputes num_edges from the hyperedge_index.
    """
    live_item = live.get(0)
    assert isinstance(live_item, HeteroData)
    assert live_item[GPR].num_edges == 3
    item = preprocessed[0]
    assert item[GPR].num_edges == 2
    assert item[GPR].hyperedge_index.size(1) == 3
    assert item[GPR].mask.numel() == 3


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
def test_reconstructed_masks_equal_the_hand_derived_sets(
    preprocessed: Neo4jPreprocessedCellDataset, idx: int, masks: dict[Any, list[int]]
) -> None:
    """Bool masks from the module docstring; node `pert_mask` is exactly `~mask`; no
    surviving edge touches a deleted gene (checked against the edge_index directly).
    """
    item = preprocessed[idx]
    for store, expected in masks.items():
        assert item[store].mask.dtype == torch.bool
        assert item[store].mask.tolist() == [bool(v) for v in expected]
    for node_type in ["gene", "reaction", "metabolite"]:
        assert item[node_type].pert_mask.tolist() == (~item[node_type].mask).tolist()
    deleted = set(_pert(item))
    for et in [PHYS, REG]:
        ei = item[et].edge_index
        kept = ei[:, item[et].mask].tolist()
        assert not deleted & (set(kept[0]) | set(kept[1]))


def test_item_shares_the_cell_graph_tensors(
    preprocessed: Neo4jPreprocessedCellDataset, source: Neo4jCellDataset
) -> None:
    """Zero-copy: x, edge_index, hyperedge_index, stoichiometry and w_growth are the cell
    graph's own tensor objects, not copies.
    """
    item = preprocessed[1]
    cg = source.cell_graph
    assert item["gene"].x is cg["gene"].x
    assert item[PHYS].edge_index is cg[PHYS].edge_index
    assert item[REG].edge_index is cg[REG].edge_index
    assert item[GPR].hyperedge_index is cg[GPR].hyperedge_index
    assert item[RMR].hyperedge_index is cg[RMR].hyperedge_index
    assert item[RMR].stoichiometry is cg[RMR].stoichiometry
    assert item["reaction"].w_growth is cg["reaction"].w_growth
    assert item["reaction"].w_growth.tolist() == [0.0, 0.0, 1.0]


# --------------------------------------------------------------- index mapping, I/O


def test_index_mapping_slicing_and_transform(
    preprocessed: Neo4jPreprocessedCellDataset,
) -> None:
    """`ds[i]` is `get(indices()[i])`: -1 is record 2, `ds[[2, 0]]` is a 2-long view whose
    item 0 is record 2; `transform` runs in `__getitem__`, never in `get`.
    """
    assert _pert(preprocessed[-1]) == [4]
    view = preprocessed[[2, 0]]
    assert isinstance(view, Neo4jPreprocessedCellDataset)
    assert len(view) == 2
    assert _pert(view[0]) == [4]
    assert _pert(view[1]) == [1]
    seen: list[list[int]] = []

    def record(item: HeteroData) -> HeteroData:
        seen.append(_pert(item))
        item["gene"].tag = 7
        return item

    preprocessed.transform = record
    assert preprocessed[1]["gene"].tag == 7
    untransformed = preprocessed.get(1)
    assert isinstance(untransformed, HeteroData)
    assert "tag" not in untransformed["gene"]
    assert seen == [[0, 3]]


def test_get_returns_none_for_a_key_past_the_store(
    preprocessed: Neo4jPreprocessedCellDataset,
) -> None:
    """`get(3)` is None (no key b"3"); `ds[3]` stops earlier at `range(3)[3]`."""
    assert preprocessed.get(3) is None
    with pytest.raises(IndexError, match="^range object index out of range$"):
        preprocessed[3]


def test_env_opens_lazily_closes_idempotently_and_is_dropped_by_pickle(
    preprocessed: Neo4jPreprocessedCellDataset,
) -> None:
    """`get` opens the read env; pickling ships `env=None`; the copy reopens and reads
    the same record; `close_lmdb` twice is a no-op the second time.
    """
    assert preprocessed.env is None
    preprocessed.get(0)
    assert preprocessed.env is not None
    clone = pickle.loads(pickle.dumps(preprocessed))
    assert clone.env is None
    assert preprocessed.env is not None
    assert _pert(clone[1]) == [0, 3]
    assert clone.env is not None
    clone.close_lmdb()
    preprocessed.close_lmdb()
    assert preprocessed.env is None
    preprocessed.close_lmdb()
    assert preprocessed.env is None


def test_a_second_instance_loads_length_from_metadata(
    tmp_path: Path, preprocessed: Neo4jPreprocessedCellDataset
) -> None:
    """A fresh instance on the finished root reads `_length` = 3 in the constructor."""
    again = Neo4jPreprocessedCellDataset(root=str(tmp_path / "pre"))
    assert again._length == 3
    assert len(again) == 3


# ----------------------------------------------------------------- stale-store hazards


def test_existing_store_is_served_for_a_different_source(
    tmp_path: Path, preprocessed: Neo4jPreprocessedCellDataset
) -> None:
    """Finding: nothing ties the store to the source it was built from. A source with ONE
    record (YAL003W deleted) handed to an instance on the finished root gets length 3
    and record 0 = YAL002W deleted, the OLD build's genotype, with no warning
    (`neo4j_preprocessed_cell.py:83-84` loads metadata only; `_load_metadata` reads only
    `length`). Reached: the 006 configs 077 (full masks) and 081/082 (compact) read
    these stores. Pinned until the metadata records a source fingerprint checked on
    load.
    """
    other = _source(tmp_path / "other", records=[_fitness(["YAL003W"], 0.5, 0.02)])
    assert len(other) == 1
    stale = Neo4jPreprocessedCellDataset(
        root=str(tmp_path / "pre"), source_dataset=other
    )
    assert len(stale) == 3
    assert _pert(stale[0]) == [1]
    torch.testing.assert_close(stale[0]["gene"].phenotype_values, torch.tensor([0.9]))


def test_re_preprocessing_a_shorter_source_leaves_stale_keys(
    tmp_path: Path, preprocessed: Neo4jPreprocessedCellDataset
) -> None:
    """Finding: `preprocess_from_source` writes into the existing LMDB without clearing it
    (`neo4j_preprocessed_cell.py:336-359`). After re-running on a 1-record source,
    length is 1 and key 0 is the new record, but keys 1 and 2 still hold the old build
    and `get(1)` returns it. Reached: the 006 configs 077 (full masks) and 081/082
    (compact) read these stores. Pinned until preprocessing clears or replaces the
    store.
    """
    other = _source(tmp_path / "other", records=[_fitness(["YAL003W"], 0.5, 0.02)])
    assert len(other) == 1
    # class-qualified so the quality checker sees the instance handed to the call
    Neo4jPreprocessedCellDataset.preprocess_from_source(
        preprocessed, other, LazySubgraphRepresentation()
    )
    assert len(preprocessed) == 1
    assert _pert(preprocessed[0]) == [2]
    stale = preprocessed.get(1)
    assert isinstance(stale, HeteroData)
    assert _pert(stale) == [0, 3]
    with pytest.raises(IndexError, match="^range object index out of range$"):
        preprocessed[1]


def test_item_without_a_statistic_value_crashes_preprocessing(tmp_path: Path) -> None:
    """Finding: `LazySubgraphRepresentation._add_phenotype_data` sets the
    `phenotype_stat_*` keys only when some statistic is present
    (`graph_processor.py:1858-1868`), but `_extract_mask_indices` reads
    `phenotype_stat_values` unconditionally (`neo4j_preprocessed_cell.py:170`), so one
    record with `fitness_se=None` aborts the whole run. The aborted write transaction
    leaves an empty `processed/lmdb` and no metadata, so the root still reads as not
    preprocessed. Pinned until the extractor tolerates a missing statistic.
    """
    src = _source(
        tmp_path / "src", records=[_fitness(["YAL002W"], 0.9, None)], labels=["fitness"]
    )
    ds = Neo4jPreprocessedCellDataset(root=str(tmp_path / "pre"))
    with pytest.raises(
        AttributeError,
        match="^'NodeStorage' object has no attribute 'phenotype_stat_values'$",
    ):
        ds.preprocess_from_source(src, LazySubgraphRepresentation())
    assert sorted(os.listdir(tmp_path / "pre" / "processed")) == ["lmdb"]
    assert ds._is_preprocessed() is False
    env = lmdb.open(
        str(tmp_path / "pre" / "processed" / "lmdb"), readonly=True, lock=False
    )
    with env.begin() as txn:
        assert txn.stat()["entries"] == 0
    env.close()


def test_len_loads_metadata_written_after_construction(
    tmp_path: Path, source: Neo4jCellDataset
) -> None:
    """An instance built on an empty root has no length; once another instance finishes
    preprocessing, its first `len` reads metadata.json (3) and caches it.
    """
    early = Neo4jPreprocessedCellDataset(root=str(tmp_path / "pre"))
    assert early._length is None
    writer = Neo4jPreprocessedCellDataset(root=str(tmp_path / "pre"))
    writer.preprocess_from_source(source, LazySubgraphRepresentation())
    assert len(writer) == 3
    assert len(early) == 3
    assert early._length == 3


_IDS_PERT_PROBE = """
import json, torch
from torch_geometric.data import HeteroData
from torchcell.data.graph_processor import LazySubgraphRepresentation
from torchcell.datamodels.schema import (
    Environment, FitnessExperiment, FitnessPhenotype, Genotype,
    KanMxDeletionPerturbation, Media,
)
env = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))
genes = ["YAL001C", "YAL004W"]
exp = FitnessExperiment(
    dataset_name="toy",
    genotype=Genotype(perturbations=[
        KanMxDeletionPerturbation(systematic_gene_name=g, perturbed_gene_name=g)
        for g in genes
    ]),
    environment=env,
    phenotype=FitnessPhenotype(fitness=0.4, fitness_se=0.05),
)
cg = HeteroData()
cg["gene"].node_ids = ["YAL001C", "YAL002W", "YAL003W", "YAL004W", "YAL005C"]
cg["gene"].num_nodes = 5
cg["gene"].x = torch.zeros(5, 0)
item = LazySubgraphRepresentation().process(cg, [FitnessPhenotype], [{"experiment": exp}])
print(json.dumps([item["gene"].ids_pert, item["gene"].perturbation_indices.tolist()]))
"""


def _ids_pert_under_hash_seed(seed: str) -> list[Any]:
    env = {k: v for k, v in os.environ.items() if k != "DATA_ROOT"}
    env.update(
        PYTHONHASHSEED=seed,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(Path(__file__).resolve().parents[3]),
    )
    out = subprocess.run(
        [sys.executable, "-c", _IDS_PERT_PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result: list[Any] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


def test_ids_pert_order_depends_on_the_hash_seed() -> None:
    """Finding: Lazy builds `ids_pert` as `list(gene_info["perturbed_names"])`, a list of
    a SET (`graph_processor.py:1550`), so its order follows `PYTHONHASHSEED` while
    `perturbation_indices` is always ascending node order. Record 1 (YAL001C, YAL004W)
    in two fresh interpreters: seed 0 gives [YAL001C, YAL004W], seed 1 gives
    [YAL004W, YAL001C], indices [0, 3] both times. The compact store freezes whichever
    order the preprocessing process had, so `ids_pert[k]` is not `perturbation_indices[k]`
    and a stored item can disagree with a live item built under another seed. Pinned
    until the processor emits `ids_pert` in node order.
    """
    assert _ids_pert_under_hash_seed("0") == [["YAL001C", "YAL004W"], [0, 3]]
    assert _ids_pert_under_hash_seed("1") == [["YAL004W", "YAL001C"], [0, 3]]


def test_get_after_close_reopens_a_new_environment(
    preprocessed: Neo4jPreprocessedCellDataset,
) -> None:
    """`close_lmdb` drops the handle; the next `get` opens a fresh read env (a different
    object) and serves the same record as before the close.
    """
    before = preprocessed.get(1)
    assert isinstance(before, HeteroData)
    first_env = preprocessed.env
    assert first_env is not None
    preprocessed.close_lmdb()
    assert preprocessed.env is None
    after = preprocessed.get(1)
    assert isinstance(after, HeteroData)
    assert preprocessed.env is not None
    assert preprocessed.env is not first_env
    assert _pert(after) == [0, 3]
    assert _differences(before, after) == {}
    preprocessed.close_lmdb()
