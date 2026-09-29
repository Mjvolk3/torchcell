# tests/torchcell/adapters/test_kuzmin2020_adapter.py
# [[tests.torchcell.adapters.test_kuzmin2020_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_kuzmin2020_adapter.py
"""The five Kuzmin 2020 adapters: conf loading, and the exact graph over two records.

Fixture: two records per dataset shaped as the loaders' ``create_experiment`` build them
(SGA triple-mutant selection medium at 26 C, PubMed 32586993, ``strain_id`` on every
perturbation; Kuzmin 2020 array alleles are trimmed of their ``_suffix`` by the loader,
so ``perturbed_gene_name`` here is the bare allele). Single mutants carry one
perturbation, doubles two, triples three; fitness datasets use ``FitnessPhenotype``
with a 1.0 reference, interaction datasets ``GeneInteractionPhenotype`` with a 0.0
reference (digenic ``graph_level`` "edge", trigenic the schema default "hyperedge").

With ``P`` perturbations per record the conf yields ``21 + 2P`` nodes and ``20 + 2P``
edges (derivation in ``test_kuzmin2018_adapter``), compared element by element against
``_sga_adapter_harness.expected_nodes`` / ``expected_edges``. 8 chunked node methods
(experiment, genotype, perturbation, environment, media, temperature, phenotype,
publication) + 9 chunked edge methods close the LMDB 17 times; the 15 node methods log
events 1-15 and the 13 edge methods events 16-28, in that order.

The runs fork one pool worker per chunked method, each forking one loader worker.
"""

from __future__ import annotations

import gc
import os.path as osp
import re
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

import torchcell.adapters.cell_adapter as cell_adapter_module
import torchcell.adapters.kuzmin2020_adapter as adapter_module
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
from torchcell.adapters.kuzmin2020_adapter import (
    DmfKuzmin2020Adapter,
    DmiKuzmin2020Adapter,
    SmfKuzmin2020Adapter,
    TmfKuzmin2020Adapter,
    TmiKuzmin2020Adapter,
)
from torchcell.datamodels import schema as s
from torchcell.datamodels.media import SGA_TM_SELECTION

ENVIRONMENT = s.Environment(media=SGA_TM_SELECTION, temperature=s.Temperature(value=26))
PUBLICATION = s.Publication(
    pubmed_id="32586993",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
    doi="10.1126/science.aaz5667",
    doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
)
KANMX = s.SgaKanMxDeletionPerturbation(
    systematic_gene_name="YBR001C", perturbed_gene_name="NTH2", strain_id="YBR001C_dma1"
)
ALLELE = s.SgaAllelePerturbation(
    systematic_gene_name="YBR002C",
    perturbed_gene_name="rer2-2",
    strain_id="YBR002C_sn1",
)
TS_ALLELE = s.SgaTsAllelePerturbation(
    systematic_gene_name="YBR003W",
    perturbed_gene_name="coq1-ts",
    strain_id="YBR003W_tsa1",
)
ARRAY_KANMX = s.SgaKanMxDeletionPerturbation(
    systematic_gene_name="YBR004C",
    perturbed_gene_name="GPI18",
    strain_id="YBR004C_dma2",
)


def _fitness(
    name: str, genotypes: list[list[Any]], fitness: list[float], std: list[float | None]
) -> tuple[list[Any], Any]:
    experiments = [
        s.FitnessExperiment(
            dataset_name=name,
            genotype=s.Genotype(perturbations=perturbations),
            environment=ENVIRONMENT,
            phenotype=s.FitnessPhenotype(fitness=f, fitness_std=sd),
        )
        for perturbations, f, sd in zip(genotypes, fitness, std, strict=True)
    ]
    reference = s.FitnessExperimentReference(
        dataset_name=name,
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT.model_copy(),
        phenotype_reference=s.FitnessPhenotype(fitness=1.0, fitness_std=None),
    )
    return experiments, reference


def _gi_phenotype(
    score: float, p_value: float | None, graph_level: str | None
) -> s.GeneInteractionPhenotype:
    """The loaders' phenotype: digenic sets ``graph_level`` "edge", trigenic leaves it."""
    if graph_level is None:
        return s.GeneInteractionPhenotype(
            gene_interaction=score, gene_interaction_p_value=p_value
        )
    return s.GeneInteractionPhenotype(
        gene_interaction=score,
        gene_interaction_p_value=p_value,
        graph_level=graph_level,
    )


def _interaction(
    name: str, genotypes: list[list[Any]], graph_level: str | None
) -> tuple[list[Any], Any]:
    experiments = [
        s.GeneInteractionExperiment(
            dataset_name=name,
            genotype=s.Genotype(perturbations=perturbations),
            environment=ENVIRONMENT,
            phenotype=_gi_phenotype(score, p_value, graph_level),
        )
        for perturbations, score, p_value in zip(
            genotypes, [-0.35, 0.08], [0.001, 0.4], strict=True
        )
    ]
    reference = s.GeneInteractionExperimentReference(
        dataset_name=name,
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT.model_copy(),
        phenotype_reference=_gi_phenotype(0.0, None, graph_level),
    )
    return experiments, reference


SINGLE: list[list[Any]] = [[KANMX], [ALLELE]]
DOUBLE: list[list[Any]] = [[KANMX, TS_ALLELE], [ALLELE, ARRAY_KANMX]]
TRIPLE: list[list[Any]] = [[KANMX, ALLELE, TS_ALLELE], [ALLELE, ARRAY_KANMX, TS_ALLELE]]

CASES: list[tuple[Any, str, str, list[list[Any]], bool, str | None]] = [
    (SmfKuzmin2020Adapter, "SmfKuzmin2020Dataset", "smf", SINGLE, True, None),
    (DmfKuzmin2020Adapter, "DmfKuzmin2020Dataset", "dmf", DOUBLE, True, None),
    (TmfKuzmin2020Adapter, "TmfKuzmin2020Dataset", "tmf", TRIPLE, True, None),
    (DmiKuzmin2020Adapter, "DmiKuzmin2020Dataset", "dmi", DOUBLE, False, "edge"),
    (TmiKuzmin2020Adapter, "TmiKuzmin2020Dataset", "tmi", TRIPLE, False, None),
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
    ("adapter_cls", "dataset_name", "slug", "genotypes", "is_fitness", "graph_level"),
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
    graph_level: str | None,
) -> None:
    """Conf ``conf/<slug>_kuzmin2020_adapter.yaml`` enables the 15 fitness (or gene
    interaction) node methods and 13 edge methods with no memory reduction factor; the
    constructor stores the worker sizes, starts wandb once, prints its debug line, and
    the graph over two records is ``21 + 2P`` nodes and ``20 + 2P`` edges, exactly.
    """
    if is_fitness:
        experiments, reference = _fitness(
            dataset_name, genotypes, [0.75, 1.05], [0.03, 0.01]
        )
        experiment_cls: type[Any] = s.FitnessExperiment
        reference_cls: type[Any] = s.FitnessExperimentReference
        node_methods = FITNESS_NODE_METHODS
    else:
        experiments, reference = _interaction(dataset_name, genotypes, graph_level)
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
    assert adapter.get_memory_reduction_factor("publication (chunked)") == 1.0
    assert (
        adapter.get_memory_reduction_factor(
            "experiment to dataset (chunked)", is_edge=True
        )
        == 1.0
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
        node_factor=lambda name: 1.0,
        edge_factor=lambda name: 1.0,
    )


@pytest.mark.parametrize(
    ("adapter_cls", "conf_name"),
    [
        (SmfKuzmin2020Adapter, "smf_kuzmin2020_adapter.yaml"),
        (DmfKuzmin2020Adapter, "dmf_kuzmin2020_adapter.yaml"),
        (TmfKuzmin2020Adapter, "tmf_kuzmin2020_adapter.yaml"),
        (DmiKuzmin2020Adapter, "dmi_kuzmin2020_adapter.yaml"),
        (TmiKuzmin2020Adapter, "tmi_kuzmin2020_adapter.yaml"),
    ],
    ids=["smf", "dmf", "tmf", "dmi", "tmi"],
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
