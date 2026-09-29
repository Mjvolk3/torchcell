# tests/torchcell/adapters/test_costanzo2016_adapter.py
# [[tests.torchcell.adapters.test_costanzo2016_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_costanzo2016_adapter.py
"""The three Costanzo 2016 adapters and the module's ``main``.

Fixture: two records per dataset shaped as the loaders' ``create_experiment`` build them
(SGA double-mutant selection medium at 30 C, PubMed 27708008, ``strain_id`` on every
perturbation). Single mutants carry one perturbation and a bootstrap-SE fitness
uncertainty over 17 screens, as the SMF loader records; doubles carry two. Fitness
datasets use ``FitnessPhenotype`` with a 1.0 reference, the interaction dataset
``GeneInteractionPhenotype`` with a 0.0 reference at ``graph_level`` "edge".

With ``P`` perturbations per record the conf yields ``21 + 2P`` nodes and ``20 + 2P``
edges (derivation in ``test_kuzmin2018_adapter``), compared element by element against
``_sga_adapter_harness.expected_nodes`` / ``expected_edges``. The dmf and dmi confs set
``memory_reduction_factor: 0.5`` on the publication node and on every chunked edge, so
those factors read 0.5 while smf reads 1.0, in the method table and in the lookups. 8
chunked node methods + 9 chunked edge methods close the LMDB 17 times; the 15 node
methods log events 1-15 and the 13 edge methods events 16-28, in that order.

``main`` is run with the module's ``BioCypher``, ``DmiCostanzo2016Dataset`` and
``DmiCostanzo2016Adapter`` replaced by recorders and ``dotenv.load_dotenv`` stubbed, so
the exact root, ``subset_n`` 500000 and worker sizes it hardcodes are pinned without a
dataset build.
"""

from __future__ import annotations

import gc
import os.path as osp
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import torchcell.adapters.cell_adapter as cell_adapter_module
import torchcell.adapters.costanzo2016_adapter as adapter_module
from tests.torchcell.adapters._sga_adapter_harness import (
    EDGE_METHODS,
    FITNESS_NODE_METHODS,
    GENOME,
    INTERACTION_NODE_METHODS,
    FixedDatetime,
    WandbRecorder,
    assert_method_table,
    conf_method_names,
    expected_edges,
    expected_events,
    expected_nodes,
    make_dataset,
)
from torchcell.adapters.costanzo2016_adapter import (
    DmfCostanzo2016Adapter,
    DmiCostanzo2016Adapter,
    SmfCostanzo2016Adapter,
)
from torchcell.datamodels import schema as s
from torchcell.datamodels.media import SGA_DM_SELECTION

ENVIRONMENT = s.Environment(media=SGA_DM_SELECTION, temperature=s.Temperature(value=30))
PUBLICATION = s.Publication(
    pubmed_id="27708008",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/27708008/",
    doi="10.1126/science.aaf1420",
    doi_url="https://www.science.org/doi/10.1126/science.aaf1420",
)
TS_ALLELE = s.SgaTsAllelePerturbation(
    systematic_gene_name="YBR001C",
    perturbed_gene_name="nth2-ts",
    strain_id="YBR001C_tsq2",
)
NATMX = s.SgaNatMxDeletionPerturbation(
    systematic_gene_name="YBR002C", perturbed_gene_name="RER2", strain_id="YBR002C_dma3"
)
DAMP = s.SgaDampPerturbation(
    systematic_gene_name="YBR003W",
    perturbed_gene_name="COQ1",
    strain_id="YBR003W_damp4",
)
SUPPRESSOR = s.SgaSuppressorAllelePerturbation(
    systematic_gene_name="YBR004C",
    perturbed_gene_name="GPI18",
    strain_id="YBR004C_sup5",
)
KANMX = s.SgaKanMxDeletionPerturbation(
    systematic_gene_name="YBR005W", perturbed_gene_name="RCR1", strain_id="YBR005W_dma6"
)


def _smf_phenotype(fitness: float, std: float) -> s.FitnessPhenotype:
    """The SMF loader's phenotype: the stddev column is a bootstrap SE over 17 screens."""
    return s.FitnessPhenotype(
        fitness=fitness,
        fitness_std=std,
        fitness_uncertainty=std,
        fitness_uncertainty_type=s.UncertaintyType.bootstrap_se,
        n_samples=17,
        sample_unit=s.SampleUnit.screen,
    )


def _fitness(
    name: str, genotypes: list[list[Any]], phenotypes: list[s.FitnessPhenotype]
) -> tuple[list[Any], Any]:
    experiments = [
        s.FitnessExperiment(
            dataset_name=name,
            genotype=s.Genotype(perturbations=perturbations),
            environment=ENVIRONMENT,
            phenotype=phenotype,
        )
        for perturbations, phenotype in zip(genotypes, phenotypes, strict=True)
    ]
    reference = s.FitnessExperimentReference(
        dataset_name=name,
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT.model_copy(),
        phenotype_reference=_smf_phenotype(1.0, 0.04),
    )
    return experiments, reference


def _interaction(name: str, genotypes: list[list[Any]]) -> tuple[list[Any], Any]:
    experiments = [
        s.GeneInteractionExperiment(
            dataset_name=name,
            genotype=s.Genotype(perturbations=perturbations),
            environment=ENVIRONMENT,
            phenotype=s.GeneInteractionPhenotype(
                gene_interaction=score,
                gene_interaction_p_value=p_value,
                graph_level="edge",
            ),
        )
        for perturbations, score, p_value in zip(
            genotypes, [-0.2, 0.15], [0.01, 0.3], strict=True
        )
    ]
    reference = s.GeneInteractionExperimentReference(
        dataset_name=name,
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT.model_copy(),
        phenotype_reference=s.GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None, graph_level="edge"
        ),
    )
    return experiments, reference


SINGLE: list[list[Any]] = [[TS_ALLELE], [DAMP]]
DOUBLE: list[list[Any]] = [[TS_ALLELE, NATMX], [DAMP, SUPPRESSOR]]
DOUBLE_KANMX: list[list[Any]] = [[KANMX, NATMX], [DAMP, SUPPRESSOR]]

CASES: list[tuple[Any, str, str, list[list[Any]], bool, float]] = [
    (SmfCostanzo2016Adapter, "SmfCostanzo2016Dataset", "smf", SINGLE, True, 1.0),
    (DmfCostanzo2016Adapter, "DmfCostanzo2016Dataset", "dmf", DOUBLE, True, 0.5),
    (DmiCostanzo2016Adapter, "DmiCostanzo2016Dataset", "dmi", DOUBLE_KANMX, False, 0.5),
]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> WandbRecorder:
    """Install the recorder as ``cell_adapter``'s ``wandb`` and pin ``datetime.now``."""
    rec = WandbRecorder()
    monkeypatch.setattr(cell_adapter_module, "wandb", rec)
    monkeypatch.setattr(cell_adapter_module, "datetime", FixedDatetime)
    return rec


@pytest.fixture(autouse=True)
def _unfreeze_gc() -> Iterator[None]:
    """The adapter and its loader call ``gc.freeze()``; undo it for later tests."""
    yield
    gc.unfreeze()


@pytest.mark.parametrize(
    ("adapter_cls", "dataset_name", "slug", "genotypes", "is_fitness", "factor"),
    CASES,
    ids=[case[2] for case in CASES],
)
def test_adapter_emits_the_exact_graph_for_its_record_type(
    recorder: WandbRecorder,
    capsys: pytest.CaptureFixture[str],
    adapter_cls: Any,
    dataset_name: str,
    slug: str,
    genotypes: list[list[Any]],
    is_fitness: bool,
    factor: float,
) -> None:
    """Conf ``conf/<slug>_costanzo2016_adapter.yaml`` enables the 15 fitness (or gene
    interaction) node methods and 13 edge methods; dmf and dmi halve the publication node
    and every chunked edge (factor 0.5), smf sets no factor (1.0). The constructor stores
    the worker sizes, starts wandb once, prints its debug line, and the graph over two
    records is ``21 + 2P`` nodes and ``20 + 2P`` edges, exactly.
    """
    if is_fitness:
        experiments, reference = _fitness(
            dataset_name,
            genotypes,
            [_smf_phenotype(0.8, 0.02), _smf_phenotype(1.2, 0.05)],
        )
        experiment_cls: type[Any] = s.FitnessExperiment
        reference_cls: type[Any] = s.FitnessExperimentReference
        node_methods = FITNESS_NODE_METHODS
    else:
        experiments, reference = _interaction(dataset_name, genotypes)
        experiment_cls = s.GeneInteractionExperiment
        reference_cls = s.GeneInteractionExperimentReference
        node_methods = INTERACTION_NODE_METHODS
    dataset = make_dataset(
        dataset_name, experiments, reference, PUBLICATION, experiment_cls, reference_cls
    )
    adapter = adapter_cls(
        dataset=dataset,
        process_workers=1,
        io_workers=1,
        chunk_size=2,
        loader_batch_size=2,
    )
    assert capsys.readouterr().out.startswith(
        f"{adapter_cls.__name__} initialized with config: "
    )
    assert (adapter.dataset, adapter.process_workers, adapter.io_workers) == (
        dataset,
        1,
        1,
    )
    assert (adapter.chunk_size, adapter.loader_batch_size) == (2, 2)
    assert conf_method_names(adapter) == (node_methods, EDGE_METHODS)
    assert adapter.get_memory_reduction_factor("publication (chunked)") == factor
    assert adapter.get_memory_reduction_factor("experiment (chunked)") == 1.0
    for name in EDGE_METHODS:
        expected_factor = factor if name.endswith("(chunked)") else 1.0
        assert (
            adapter.get_memory_reduction_factor(name, is_edge=True) == expected_factor
        )
    assert recorder.init_calls == 1
    assert recorder.logged[1] == {
        "current_adapter_dataset_name": dataset_name,
        "current_adapter_dataset_start_time": "2026-09-28 12:00:00",
    }

    nodes = list(adapter.get_nodes())
    edges = list(adapter.get_edges())
    n_perturbations = len(genotypes[0])
    assert len(nodes) == 21 + 2 * n_perturbations
    assert len(edges) == 20 + 2 * n_perturbations
    assert nodes == expected_nodes(dataset_name, experiments, reference, PUBLICATION)
    assert edges == expected_edges(dataset_name, experiments, reference, PUBLICATION)
    assert dataset.close_calls == 17
    assert adapter.event == 28
    assert recorder.logged[2:] == expected_events(node_methods, EDGE_METHODS)
    assert_method_table(
        recorder.logged[0][f"{dataset_name}_method_table"],
        node_methods,
        EDGE_METHODS,
        node_factor=lambda name: factor if name == "publication (chunked)" else 1.0,
        edge_factor=lambda name: factor,
    )


@pytest.mark.parametrize(
    ("adapter_cls", "conf_name"),
    [
        (SmfCostanzo2016Adapter, "smf_costanzo2016_adapter.yaml"),
        (DmfCostanzo2016Adapter, "dmf_costanzo2016_adapter.yaml"),
        (DmiCostanzo2016Adapter, "dmi_costanzo2016_adapter.yaml"),
    ],
    ids=["smf", "dmf", "dmi"],
)
def test_missing_conf_is_refused_before_wandb_starts(
    recorder: WandbRecorder,
    monkeypatch: pytest.MonkeyPatch,
    adapter_cls: Any,
    conf_name: str,
) -> None:
    """With the conf path absent each constructor raises naming its exact path
    ``<adapters dir>/conf/<conf_name>`` and never reaches ``wandb.init``.
    """
    fake_osp = SimpleNamespace(
        dirname=osp.dirname, abspath=osp.abspath, join=osp.join, exists=lambda p: False
    )
    monkeypatch.setattr(adapter_module, "osp", fake_osp)
    conf_path = osp.join(
        osp.dirname(osp.abspath(adapter_module.__file__)), "conf", conf_name
    )
    dataset: Any = None
    with pytest.raises(
        FileNotFoundError, match=f"^{re.escape(f'Config file not found: {conf_path}')}$"
    ):
        adapter_cls(dataset=dataset, process_workers=1, io_workers=1)
    assert recorder.init_calls == 0


# --------------------------------------------------------------------------- main()


class _FakeBioCypher:
    """Records the constructor kwargs and every write call, in order."""

    instances: list[_FakeBioCypher] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[Any, ...]] = []
        _FakeBioCypher.instances.append(self)

    def write_nodes(self, nodes: Any) -> None:
        self.calls.append(("write_nodes", list(nodes)))

    def write_edges(self, edges: Any) -> None:
        self.calls.append(("write_edges", list(edges)))

    def write_import_call(self) -> None:
        self.calls.append(("write_import_call",))

    def write_schema_info(self, as_node: bool) -> None:
        self.calls.append(("write_schema_info", as_node))

    def summary(self) -> None:
        self.calls.append(("summary",))


class _FakeDataset:
    instances: list[_FakeDataset] = []

    def __init__(self, root: str, subset_n: int) -> None:
        self.root = root
        self.subset_n = subset_n
        _FakeDataset.instances.append(self)


class _FakeAdapter:
    instances: list[_FakeAdapter] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _FakeAdapter.instances.append(self)

    def get_nodes(self) -> Iterator[str]:
        yield from ["node-1", "node-2"]

    def get_edges(self) -> Iterator[str]:
        yield from ["edge-1"]


def test_main_builds_the_dmi_5e5_subset_and_writes_everything(  # test-quality: allow main() returns None; its effects are asserted on the recorders
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``main`` loads the dotenv once, opens BioCypher under
    ``$DATA_ROOT/database/biocypher-out/<YYYY-mm-dd_HH-MM-SS>``, builds
    ``DmiCostanzo2016Dataset`` at ``data/torchcell/dmi_costanzo2016_5e5`` with
    ``subset_n`` 500000, wraps it in the adapter with 10/10 workers, chunk 100 and
    loader batch 10, then writes nodes, edges, the import call, the schema node and the
    summary, in that order.
    """
    _FakeBioCypher.instances.clear()
    _FakeDataset.instances.clear()
    _FakeAdapter.instances.clear()
    dotenv_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "dotenv.load_dotenv", lambda *args, **kwargs: dotenv_calls.append(args)
    )
    monkeypatch.setattr(adapter_module, "BioCypher", _FakeBioCypher)
    monkeypatch.setattr(adapter_module, "DmiCostanzo2016Dataset", _FakeDataset)
    monkeypatch.setattr(adapter_module, "DmiCostanzo2016Adapter", _FakeAdapter)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("BIOCYPHER_CONFIG_PATH", "bc-config.yaml")
    monkeypatch.setenv("SCHEMA_CONFIG_PATH", "schema-config.yaml")

    adapter_module.main()

    assert dotenv_calls == [()]
    (bc,) = _FakeBioCypher.instances
    out_dir = bc.kwargs.pop("output_directory")
    prefix = osp.join(str(tmp_path), "database/biocypher-out") + "/"
    assert out_dir.startswith(prefix)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", out_dir[len(prefix) :])
    assert bc.kwargs == {
        "biocypher_config_path": "bc-config.yaml",
        "schema_config_path": "schema-config.yaml",
    }
    (dataset,) = _FakeDataset.instances
    assert dataset.root == osp.join(
        str(tmp_path), "data/torchcell/dmi_costanzo2016_5e5"
    )
    assert dataset.subset_n == 500000
    (adapter,) = _FakeAdapter.instances
    assert adapter.kwargs == {
        "dataset": dataset,
        "process_workers": 10,
        "io_workers": 10,
        "chunk_size": 100,
        "loader_batch_size": 10,
    }
    assert bc.calls == [
        ("write_nodes", ["node-1", "node-2"]),
        ("write_edges", ["edge-1"]),
        ("write_import_call",),
        ("write_schema_info", True),
        ("summary",),
    ]
