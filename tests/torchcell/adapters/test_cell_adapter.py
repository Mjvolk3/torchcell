# tests/torchcell/adapters/test_cell_adapter.py
# [[tests.torchcell.adapters.test_cell_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_cell_adapter.py
"""Exact node, edge and bookkeeping behavior of ``CellAdapter``.

Fixture: a tiny in-memory dataset (``_MemoryDataset``) exposing exactly what the adapter
reads from an ``ExperimentDataset``: ``name``, ``len``, integer and slice indexing,
``transform_item`` (raw dicts back into typed ``FitnessExperiment`` /
``FitnessExperimentReference`` / ``Publication``, as the real base class does),
``experiment_reference_index`` and ``close_lmdb``. Its records are real schema objects:
two KanMX deletions (one plain, one SGA with a ``strain_id``) on YPD at 30 C.

``wandb`` is replaced at the boundary by ``_WandbRecorder`` bound to the name the module
imported, so every payload the adapter logs is asserted verbatim; ``datetime`` is pinned
to 2026-09-27 08:30:05 so the start-time payload is a fixed string.

Every node id is the adapter's content address, ``sha256(json.dumps(model_dump()))``
(``_sha``), except the environment-side ids, which are ``identity_sha256`` of the
identity projection, and the ``interned constant`` id, which is the sha256 of the
constant's exact JSON payload. Only the record-level blobs (experiment, experiment
reference, interned constant, genome, environment side, publication) carry
``serialized_data``; sub-object nodes (genotype, perturbation, environment
perturbation, phenotype) carry only their scalar properties. Expected nodes and edges are built by hand as ``BioCypherNode`` /
``BioCypherEdge`` dataclasses and compared by equality, which covers id, label,
preferred id and the full property dict (BioCypher adds ``id`` and ``preferred_id`` to
the properties on both sides).

Builders already pinned elsewhere are not repeated here: the CRISPR construct node and
edge (``test_crispr_construct_nodes``), the gapped-temperature paths
(``test_optional_temperature_guard``) and the environment-side id agreement
(``test_environment_node_identity``).

The only multiprocessing is the adapter's own: the ``data_chunker`` decorator forks one
``CpuExperimentLoaderMultiprocessing`` worker per ``io_workers`` (1 here), and
``get_data_by_type`` forks one ``ProcessPoolExecutor`` worker per group of
``process_workers * CHUNKS_PER_WORKER = 2`` chunks (so 3 one-record chunks fork two pools,
each of whose chunks forks one loader worker).
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import re
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from biocypher._create import BioCypherEdge, BioCypherNode
from omegaconf import DictConfig, OmegaConf

import torchcell.adapters.cell_adapter as cell_adapter_module
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.data.data import ExperimentReferenceIndex
from torchcell.datamodels import schema as s
from torchcell.datamodels.identity import (
    environment_identity,
    environment_perturbation_identity,
    identity_sha256,
    media_identity,
    temperature_identity,
)
from torchcell.fast_csv import RenderedChunk
from torchcell.loader import CpuExperimentLoaderMultiprocessing

GENOME = s.ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
MEDIA = s.Media(name="YPD", state="liquid", is_synthetic=False)
NACL = s.Compound(name="NaCl", inchikey="FAPWRFPIFSIZLT-UHFFFAOYSA-M")
HCL = s.Compound(name="HCl", inchikey="VEXZGXHMUGYJMC-UHFFFAOYSA-N")
PUBLICATION = s.Publication(
    pubmed_id="27708008",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/27708008",
    doi="10.1126/science.aaf1420",
    doi_url="https://doi.org/10.1126/science.aaf1420",
)
SMALL_MOLECULE = s.SmallMoleculePerturbation(
    compound=NACL,
    concentration=s.Concentration(value=0.4, unit=s.ConcentrationUnit.molar),
)
ENVIRONMENT = s.Environment(
    media=MEDIA, temperature=s.Temperature(value=30.0), perturbations=[SMALL_MOLECULE]
)
PLAIN_DELETION = s.KanMxDeletionPerturbation(
    systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
)
SGA_DELETION = s.SgaKanMxDeletionPerturbation(
    systematic_gene_name="YAL002W", perturbed_gene_name="VPS8", strain_id="YAL002W_dma1"
)
FIXED_NOW = datetime(2026, 9, 27, 8, 30, 5)


def _sha(model: Any) -> str:
    """The adapter's content address: sha256 of the json-dumped ``model_dump``."""
    return hashlib.sha256(json.dumps(model.model_dump()).encode("utf-8")).hexdigest()


def _experiment(fitness: float) -> s.FitnessExperiment:
    return s.FitnessExperiment(
        dataset_name="ToyDataset",
        genotype=s.Genotype(perturbations=[SGA_DELETION, PLAIN_DELETION]),
        environment=ENVIRONMENT,
        phenotype=s.FitnessPhenotype(fitness=fitness, fitness_std=0.1),
    )


def _reference(fitness: float = 1.0) -> s.FitnessExperimentReference:
    return s.FitnessExperimentReference(
        dataset_name="ToyDataset",
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=s.FitnessPhenotype(fitness=fitness),
    )


def _raw(fitness: float) -> dict[str, Any]:
    """One stored item, as the LMDB would hand it back (plain dumps, not models)."""
    return {
        "experiment": _experiment(fitness).model_dump(),
        "reference": _reference().model_dump(),
        "publication": PUBLICATION.model_dump(),
    }


class _MemoryDataset:
    """The slice of ``ExperimentDataset`` the adapter touches, held in memory."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        reference_index: list[ExperimentReferenceIndex],
        name: str = "ToyDataset",
    ) -> None:
        self.name = name
        self._items = items
        self.experiment_reference_index = reference_index
        self.slices: list[tuple[int, int]] = []
        self.close_calls = 0

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            self.slices.append((idx.start, idx.stop))
            return _MemoryDataset(
                self._items[idx], self.experiment_reference_index, self.name
            )
        return self._items[idx]

    def transform_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "experiment": s.FitnessExperiment(**item["experiment"]),
            "reference": s.FitnessExperimentReference(**item["reference"]),
            "publication": s.Publication(**item["publication"]),
        }

    def close_lmdb(self) -> None:
        self.close_calls += 1


class _Table:
    """Stands in for ``wandb.Table``: keeps what it was built from."""

    def __init__(self, columns: list[str], data: list[list[Any]]) -> None:
        self.columns = columns
        self.data = data


class _WandbRecorder:
    """Records ``wandb.init`` calls and every ``wandb.log`` payload."""

    Table = _Table

    def __init__(self) -> None:
        self.init_calls = 0
        self.logged: list[dict[str, Any]] = []

    def init(self) -> None:
        self.init_calls += 1

    def log(self, payload: dict[str, Any]) -> None:
        self.logged.append(payload)


class _FixedDatetime:
    @staticmethod
    def now() -> datetime:
        return FIXED_NOW


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _WandbRecorder:
    """Install the recorder as the module's ``wandb`` and pin ``datetime.now``."""
    rec = _WandbRecorder()
    monkeypatch.setattr(cell_adapter_module, "wandb", rec)
    monkeypatch.setattr(cell_adapter_module, "datetime", _FixedDatetime)
    return rec


@pytest.fixture(autouse=True)
def _unfreeze_gc() -> Iterator[None]:
    """The adapter and its loader call ``gc.freeze()``; undo it for later tests."""
    yield
    gc.unfreeze()


def _conf(
    node_methods: list[dict[str, Any]], edge_methods: list[dict[str, Any]]
) -> DictConfig:
    conf = OmegaConf.create(
        {"cell_adapter": {"node_methods": node_methods, "edge_methods": edge_methods}}
    )
    assert isinstance(conf, DictConfig)
    return conf


def _index(*references: Any) -> list[ExperimentReferenceIndex]:
    return [
        ExperimentReferenceIndex(reference=ref, member_indices=[i])
        for i, ref in enumerate(references)
    ]


def _dataset(n: int = 2) -> _MemoryDataset:
    """``n`` records with fitness 0.5, 0.25, 0.125, ... and one shared reference."""
    return _MemoryDataset([_raw(0.5 / 2**i) for i in range(n)], _index(_reference()))


def _adapter(
    dataset: Any,
    node_methods: list[dict[str, Any]] | None = None,
    edge_methods: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> CellAdapter:
    return CellAdapter(
        _conf(node_methods or [], edge_methods or []),
        dataset,
        process_workers=kwargs.pop("process_workers", 1),
        io_workers=kwargs.pop("io_workers", 1),
        chunk_size=kwargs.pop("chunk_size", 2),
        loader_batch_size=kwargs.pop("loader_batch_size", 2),
    )


def _bare() -> CellAdapter:
    """An adapter with no config and no dataset, for builders that read only the record."""
    return CellAdapter.__new__(CellAdapter)


def _undecorated(method: Any) -> Any:
    """The chunk handler behind ``@data_chunker``."""
    return method.__wrapped__


# ----------------------------------------------------------------- construction


def test_loader_batch_larger_than_chunk_is_refused_before_wandb_starts(
    recorder: _WandbRecorder,
) -> None:
    """Finding: the two literals at cell_adapter.py lines 72-73 join with no space, so
    the message reads "...loader_batch_size.Our recommendation...".
    """
    message = (
        "chunk_size must be greater than or equal to loader_batch_size."
        "Our recommendation are chunk_size 2-3 order of magnitude in size."
    )
    dataset: Any = _dataset()
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        CellAdapter(
            _conf([], []),
            dataset,
            process_workers=1,
            io_workers=1,
            chunk_size=10,
            loader_batch_size=11,
        )
    assert recorder.init_calls == 0
    assert recorder.logged == []


def test_constructor_starts_wandb_and_logs_the_method_table_then_the_start(
    recorder: _WandbRecorder,
) -> None:
    """Events count nodes then edges from 1. A chunked method takes its configured
    factor (0.5) or 1.0 when none is set; an unchunked one gets NaN.
    """
    adapter = _adapter(
        _dataset(),
        node_methods=[
            {"method_name": "experiment (chunked)", "memory_reduction_factor": 0.5},
            {"method_name": "dataset"},
        ],
        edge_methods=[{"method_name": "genotype to experiment (chunked)"}],
    )
    assert recorder.init_calls == 1
    assert len(recorder.logged) == 2
    assert list(recorder.logged[0]) == ["ToyDataset_method_table"]
    table = recorder.logged[0]["ToyDataset_method_table"]
    assert isinstance(table, _Table)
    assert table.columns == ["event", "method", "data_type", "memory_reduction_factor"]
    assert table.data[0] == [1, "experiment (chunked)", "node", 0.5]
    assert table.data[1][:3] == [2, "dataset", "node"]
    assert math.isnan(table.data[1][3])
    assert table.data[2] == [3, "genotype to experiment (chunked)", "edge", 1.0]
    assert len(table.data) == 3
    assert recorder.logged[1] == {
        "current_adapter_dataset_name": "ToyDataset",
        "current_adapter_dataset_start_time": "2026-09-27 08:30:05",
    }
    assert (adapter.process_workers, adapter.io_workers) == (1, 1)
    assert (adapter.chunk_size, adapter.loader_batch_size, adapter.event) == (2, 2, 0)


def test_memory_reduction_factor_reads_the_matching_list_and_defaults_to_one(
    recorder: _WandbRecorder,
) -> None:
    """The node list answers without ``is_edge``, the edge list with it; a method in
    neither list, or listed without the key, gets 1.0.
    """
    adapter = _adapter(
        _dataset(),
        node_methods=[
            {"method_name": "genotype (chunked)", "memory_reduction_factor": 0.25},
            {"method_name": "experiment (chunked)"},
        ],
        edge_methods=[
            {
                "method_name": "genotype to experiment (chunked)",
                "memory_reduction_factor": 0.125,
            }
        ],
    )
    assert adapter.get_memory_reduction_factor("genotype (chunked)") == 0.25
    assert adapter.get_memory_reduction_factor("experiment (chunked)") == 1.0
    assert (
        adapter.get_memory_reduction_factor(
            "genotype to experiment (chunked)", is_edge=True
        )
        == 0.125
    )
    # the same name looked up on the wrong side is absent there
    assert (
        adapter.get_memory_reduction_factor("genotype to experiment (chunked)") == 1.0
    )
    assert (
        adapter.get_memory_reduction_factor("genotype (chunked)", is_edge=True) == 1.0
    )


NODE_METHODS = [
    "experiment reference",
    "genome",
    "experiment (chunked)",
    "genotype (chunked)",
    "segregant genotype (chunked)",
    "perturbation (chunked)",
    "bacterial perturbation (chunked)",
    "bacterial sequence variant perturbation (chunked)",
    "crispr construct (chunked)",
    "environment (chunked)",
    "environment reference",
    "media (chunked)",
    "media reference",
    "temperature (chunked)",
    "temperature reference",
    "environment perturbation (chunked)",
    "environment perturbation reference",
    "phage perturbation (chunked)",
    "phage perturbation reference",
    "fitness phenotype (chunked)",
    "gene interaction phenotype (chunked)",
    "gene essentiality phenotype (chunked)",
    "synthetic lethality phenotype (chunked)",
    "synthetic rescue phenotype (chunked)",
    "calmorph phenotype (chunked)",
    "microarray expression phenotype (chunked)",
    "rnaseq expression phenotype (chunked)",
    "pseudobulk expression phenotype (chunked)",
    "visual score phenotype (chunked)",
    "metabolite phenotype (chunked)",
    "protein abundance phenotype (chunked)",
    "protein fold change phenotype (chunked)",
    "environment response phenotype (chunked)",
    "product titer phenotype (chunked)",
    "protein turnover phenotype (chunked)",
    "flux phenotype (chunked)",
    "promoter activity phenotype (chunked)",
    "bacterial morphology phenotype (chunked)",
    "fitness phenotype reference",
    "gene interaction phenotype reference",
    "gene essentiality phenotype reference",
    "synthetic lethality phenotype reference",
    "synthetic rescue phenotype reference",
    "calmorph phenotype reference",
    "microarray expression phenotype reference",
    "rnaseq expression phenotype reference",
    "pseudobulk expression phenotype reference",
    "visual score phenotype reference",
    "metabolite phenotype reference",
    "protein abundance phenotype reference",
    "protein fold change phenotype reference",
    "environment response phenotype reference",
    "product titer phenotype reference",
    "protein turnover phenotype reference",
    "flux phenotype reference",
    "promoter activity phenotype reference",
    "bacterial morphology phenotype reference",
    "dataset",
    "publication (chunked)",
]
EDGE_METHODS = [
    "experiment reference to dataset",
    "experiment to dataset (chunked)",
    "experiment reference to experiment (chunked)",
    "genotype to experiment (chunked)",
    "perturbation to genotype (chunked)",
    "crispr construct to perturbation (chunked)",
    "environment to experiment (chunked)",
    "environment to experiment reference",
    "phenotype to experiment (chunked)",
    "media to environment (chunked)",
    "temperature to environment (chunked)",
    "environment perturbation to environment (chunked)",
    "environment perturbation to environment reference",
    "genome to experiment reference",
    "phenotype to experiment reference",
    "publication to experiment (chunked)",
]


def test_supported_method_names_are_the_registration_tables_in_order(
    recorder: _WandbRecorder,
) -> None:
    """56 node methods and 16 edge methods, in the order ``__init__`` registers them;
    every "(chunked)" name maps to a decorated handler and every other name to a
    ``_get_`` collector, which is how ``get_nodes`` routes them.
    """
    adapter = _adapter(_dataset())
    assert adapter.supported_node_methods == NODE_METHODS
    assert adapter.supported_edge_methods == EDGE_METHODS
    for name, method in adapter.node_methods + adapter.edge_methods:
        assert method.__name__.startswith("_get_") == ("(chunked)" not in name), name


# ------------------------------------------------------------- record node builders


def _environment_payload(experiment: s.FitnessExperiment) -> str:
    """The toy environment's JSON (733 bytes), at or above the 512-byte pointer floor."""
    payload = json.dumps(experiment.model_dump()["environment"])
    assert len(payload) == 733
    return payload


def _environment_constant_id(experiment: s.FitnessExperiment) -> str:
    return hashlib.sha256(_environment_payload(experiment).encode("utf-8")).hexdigest()


def test_experiment_genotype_and_publication_nodes_are_content_addressed() -> None:
    """The experiment handler returns the experiment node, whose id is the sha256 of
    the fully inlined record but whose blob holds a ``$ref`` pointer in place of the
    733-byte environment (floor 512), followed by one ``interned constant`` node
    carrying that environment. The 922-byte genotype stays inline (floor 8192).
    Genotype names come back sorted by systematic name (YAL001C before YAL002W);
    the genotype node carries no ``serialized_data``.
    """
    experiment = _experiment(0.5)
    data = {"experiment": experiment, "publication": PUBLICATION}
    adapter = _bare()
    dump = experiment.model_dump()
    assert len(json.dumps(dump["genotype"])) == 922
    env_payload = _environment_payload(experiment)
    env_id = _environment_constant_id(experiment)
    pointered = {**dump, "environment": {"$ref": env_id, "kind": "environment"}}
    assert _undecorated(CellAdapter._experiment_node)(
        adapter, data, "experiment (chunked)"
    ) == [
        BioCypherNode(
            node_id=_sha(experiment),
            preferred_id="experiment",
            node_label="experiment",
            properties={"serialized_data": json.dumps(pointered)},
        ),
        BioCypherNode(
            node_id=env_id,
            preferred_id="interned constant",
            node_label="interned constant",
            properties={"kind": "environment", "serialized_data": env_payload},
        ),
    ]
    genotype = experiment.genotype
    assert isinstance(genotype, s.Genotype)
    assert _undecorated(CellAdapter._genotype_node)(
        adapter, data, "genotype (chunked)"
    ) == BioCypherNode(
        node_id=_sha(genotype),
        preferred_id="genotype",
        node_label="genotype",
        properties={
            "systematic_gene_names": ["YAL001C", "YAL002W"],
            "perturbed_gene_names": ["TFC3", "VPS8"],
            "perturbation_types": ["kanmx_deletion", "sga_kanmx_deletion"],
        },
    )
    assert _undecorated(CellAdapter._publication_node)(
        adapter, data, "publication (chunked)"
    ) == BioCypherNode(
        node_id=_sha(PUBLICATION),
        preferred_id="publication_27708008",
        node_label="publication",
        properties={
            "pubmed_id": "27708008",
            "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/27708008",
            "doi": "10.1126/science.aaf1420",
            "doi_url": "https://doi.org/10.1126/science.aaf1420",
            "source_type": "journal_article",
            "title": None,
            "identifier": None,
            "identifier_url": None,
            "serialized_data": json.dumps(PUBLICATION.model_dump()),
        },
    )


def test_perturbation_nodes_read_strain_id_only_where_the_leaf_declares_it() -> None:
    """One node per perturbation in genotype order; the plain KanMX leaf has no
    ``strain_id`` and projects None, the SGA leaf projects its id.
    """
    data = {"experiment": _experiment(0.5)}
    nodes = _undecorated(CellAdapter._perturbation_node)(
        _bare(), data, "perturbation (chunked)"
    )
    assert nodes == [
        BioCypherNode(
            node_id=_sha(PLAIN_DELETION),
            preferred_id="kanmx_deletion",
            node_label="perturbation",
            properties={
                "systematic_gene_name": "YAL001C",
                "perturbed_gene_name": "TFC3",
                "perturbation_type": "kanmx_deletion",
                "description": PLAIN_DELETION.description,
                "strain_id": None,
            },
        ),
        BioCypherNode(
            node_id=_sha(SGA_DELETION),
            preferred_id="sga_kanmx_deletion",
            node_label="perturbation",
            properties={
                "systematic_gene_name": "YAL002W",
                "perturbed_gene_name": "VPS8",
                "perturbation_type": "sga_kanmx_deletion",
                "description": SGA_DELETION.description,
                "strain_id": "YAL002W_dma1",
            },
        ),
    ]
    assert PLAIN_DELETION.description == "Deletion via KanMX or NatMX gene replacement"


def _segregant() -> s.SegregantGenotype:
    parent: dict[str, Any] = {
        "assembly_ref": s.ArtifactRef(
            tier="genomes",
            key="sgd_S288C_R64-4-1_20230830",
            path="S288C_reference_sequence_R64-4-1_20230830.fsa",
            sha256="a" * 64,
        ),
        "engineered_background": "MATa his3 leu2",
    }
    return s.SegregantGenotype(
        cross="BYxRM",
        segregant_id="A01_01",
        parent_1=s.SegregantParent(name="BYa", **parent),
        parent_2=s.SegregantParent(name="RMx", peter_strain_id="AAA", **parent),
        blocks=[
            s.HaplotypeBlock(
                chromosome="chrI", start=1, end=100, parent=1, n_markers=3
            ),
            s.HaplotypeBlock(
                chromosome="chrI", start=150, end=300, parent=2, n_markers=2
            ),
            s.HaplotypeBlock(chromosome="chrII", start=5, end=9, parent=2, n_markers=1),
        ],
        call_method="hard calls",
        marker_matrix_sha256="b" * 64,
    )


def test_segregant_genotype_node_projects_the_cross_and_counts_blocks() -> None:
    """Three blocks -> ``n_blocks`` 3; the chunk handler delegates to ``_from``."""
    genotype = _segregant()
    expected = BioCypherNode(
        node_id=_sha(genotype),
        preferred_id="segregant genotype",
        node_label="segregant genotype",
        properties={
            "cross": "BYxRM",
            "segregant_id": "A01_01",
            "parent_1": "BYa",
            "parent_2": "RMx",
            "n_blocks": 3,
        },
    )
    assert CellAdapter._segregant_genotype_node_from(genotype) == expected
    data = {"experiment": SimpleNamespace(genotype=genotype)}
    assert (
        _undecorated(CellAdapter._segregant_genotype_node)(
            _bare(), data, "segregant genotype (chunked)"
        )
        == expected
    )


# ----------------------------------------------------------- environment builders


def _env_perturbation_node(perturbation: Any, props: dict[str, Any]) -> BioCypherNode:
    return BioCypherNode(
        node_id=identity_sha256(environment_perturbation_identity(perturbation)),
        preferred_id=perturbation.perturbation_type,
        node_label="environment perturbation",
        properties={
            "perturbation_type": perturbation.perturbation_type,
            "description": perturbation.description,
            **props,
        },
    )


def test_environment_perturbation_node_projects_compound_or_agent_and_dose() -> None:
    """A small molecule fills compound + concentration; a physical factor fills factor,
    agent (as the compound columns) and magnitude (as the dose columns); a basis-only
    dose has no unit, and a biologic has no compound at all.
    """
    small = SMALL_MOLECULE
    assert CellAdapter._environment_perturbation_node_from(
        small
    ) == _env_perturbation_node(
        small,
        {
            "factor": None,
            "compound_name": "NaCl",
            "inchikey": "FAPWRFPIFSIZLT-UHFFFAOYSA-M",
            "concentration_value": 0.4,
            "concentration_unit": "M",
        },
    )
    ph = s.EnvironmentPhysicalPerturbation(
        factor=s.PhysicalFactor.ph,
        magnitude=s.Concentration(value=4.5, unit=s.ConcentrationUnit.ph),
        agent=HCL,
    )
    assert CellAdapter._environment_perturbation_node_from(
        ph
    ) == _env_perturbation_node(
        ph,
        {
            "factor": "pH",
            "compound_name": "HCl",
            "inchikey": "VEXZGXHMUGYJMC-UHFFFAOYSA-N",
            "concentration_value": 4.5,
            "concentration_unit": "pH",
        },
    )
    bare_factor = s.EnvironmentPhysicalPerturbation(factor=s.PhysicalFactor.osmolarity)
    assert CellAdapter._environment_perturbation_node_from(
        bare_factor
    ) == _env_perturbation_node(
        bare_factor,
        {
            "factor": "osmolarity",
            "compound_name": None,
            "inchikey": None,
            "concentration_value": None,
            "concentration_unit": None,
        },
    )
    basis_only = s.SmallMoleculePerturbation(
        compound=NACL, concentration=s.Concentration(basis=s.DoseBasis.IC30)
    )
    assert CellAdapter._environment_perturbation_node_from(
        basis_only
    ) == _env_perturbation_node(
        basis_only,
        {
            "factor": None,
            "compound_name": "NaCl",
            "inchikey": "FAPWRFPIFSIZLT-UHFFFAOYSA-M",
            "concentration_value": None,
            "concentration_unit": None,
        },
    )
    biologic = s.BiologicPerturbation(
        agent_class=s.BiologicAgentClass.peptide,
        name="plant defensin DmAMP1",
        concentration=s.Concentration(value=2.0, unit=s.ConcentrationUnit.ug_per_ml),
    )
    assert CellAdapter._environment_perturbation_node_from(
        biologic
    ) == _env_perturbation_node(
        biologic,
        {
            "factor": None,
            "compound_name": None,
            "inchikey": None,
            "concentration_value": 2.0,
            "concentration_unit": "ug/mL",
        },
    )


def test_environment_side_reference_nodes_are_deduplicated_by_identity(
    recorder: _WandbRecorder,
) -> None:
    """Three references: two on ENVIRONMENT (fitness 1.0 and 0.9) and one on YPD with
    no temperature and no perturbation. Environment and media collapse by identity
    (2 environments, 1 medium: both environments use the same YPD), temperature skips
    the gapped one (1 node), the NaCl perturbation appears once.
    """
    gapped = s.Environment(media=MEDIA)
    third = _reference().model_copy(update={"environment_reference": gapped})
    adapter = _adapter(
        _MemoryDataset([], _index(_reference(1.0), _reference(0.9), third))
    )

    def environment_node(environment: s.Environment) -> BioCypherNode:
        temperature = environment.temperature
        return BioCypherNode(
            node_id=identity_sha256(environment_identity(environment)),
            preferred_id="environment",
            node_label="environment",
            properties={
                "temperature": temperature.value if temperature is not None else None,
                "media": json.dumps(MEDIA.model_dump()),
                "serialized_data": json.dumps(environment.model_dump()),
            },
        )

    assert adapter._get_environment_reference_nodes() == [
        environment_node(ENVIRONMENT),
        environment_node(gapped),
    ]
    assert adapter._get_media_reference_nodes() == [
        BioCypherNode(
            node_id=identity_sha256(media_identity(MEDIA)),
            preferred_id="media",
            node_label="media",
            properties={
                "name": "YPD",
                "state": "liquid",
                "serialized_data": json.dumps(MEDIA.model_dump()),
            },
        )
    ]
    temperature = s.Temperature(value=30.0)
    assert adapter._get_temperature_reference_nodes() == [
        BioCypherNode(
            node_id=identity_sha256(temperature_identity(temperature)),
            preferred_id="temperature",
            node_label="temperature",
            properties={
                "value": 30.0,
                "unit": s.TemperatureUnit.celsius,
                "serialized_data": json.dumps(temperature.model_dump()),
            },
        )
    ]
    assert adapter._get_environment_perturbation_reference_nodes() == [
        CellAdapter._environment_perturbation_node_from(SMALL_MOLECULE)
    ]


def test_environment_perturbation_chunk_handler_emits_one_node_per_edit() -> None:
    """Two edits on one environment -> two nodes in the environment's order."""
    osmotic = s.EnvironmentPhysicalPerturbation(factor=s.PhysicalFactor.osmolarity)
    environment = s.Environment(media=MEDIA, perturbations=[SMALL_MOLECULE, osmotic])
    data = {"experiment": SimpleNamespace(environment=environment)}
    assert _undecorated(CellAdapter._environment_perturbation_node)(
        _bare(), data, "environment perturbation (chunked)"
    ) == [
        CellAdapter._environment_perturbation_node_from(SMALL_MOLECULE),
        CellAdapter._environment_perturbation_node_from(osmotic),
    ]


# -------------------------------------------------------------- phenotype builders


PHENOTYPE_CASES: list[tuple[str, str, Any, Any, dict[str, Any]]] = [
    (
        "fitness",
        "fitness phenotype",
        s.FitnessExperimentReference,
        s.FitnessPhenotype(fitness=0.5, fitness_std=0.1),
        {
            "graph_level": "global",
            "label_name": "fitness",
            "label_statistic_name": "fitness_se",
            "fitness": 0.5,
            "fitness_std": 0.1,
            "screen_id": None,
        },
    ),
    (
        "gene_interaction",
        "gene interaction phenotype",
        s.GeneInteractionExperimentReference,
        s.GeneInteractionPhenotype(
            gene_interaction=-0.2, gene_interaction_p_value=0.01
        ),
        {
            "graph_level": "hyperedge",
            "label_name": "gene_interaction",
            "label_statistic_name": "gene_interaction_p_value",
            "gene_interaction": -0.2,
            "gene_interaction_p_value": 0.01,
            "screen_id": None,
            # #793: the replicate-design quartet is projected, null here
            "n_samples": None,
            "sample_unit": None,
            "gene_interaction_uncertainty": None,
            "gene_interaction_uncertainty_type": None,
        },
    ),
    (
        "gene_essentiality",
        "gene essentiality phenotype",
        s.GeneEssentialityExperimentReference,
        s.GeneEssentialityPhenotype(is_essential=True),
        {"graph_level": "node", "label_name": "is_essential", "is_essential": True},
    ),
    (
        "synthetic_lethality",
        "synthetic lethality phenotype",
        s.SyntheticLethalityExperimentReference,
        s.SyntheticLethalityPhenotype(
            is_synthetic_lethal=True, synthetic_lethality_statistic_score=0.9
        ),
        {
            "graph_level": "edge",
            "label_name": "is_synthetic_lethal",
            "label_statistic_name": "synthetic_lethality_statistic_score",
            "is_synthetic_lethal": True,
            "synthetic_lethality_statistic_score": 0.9,
        },
    ),
    (
        "synthetic_rescue",
        "synthetic rescue phenotype",
        s.SyntheticRescueExperimentReference,
        s.SyntheticRescuePhenotype(
            is_synthetic_rescue=False, synthetic_rescue_statistic_score=0.3
        ),
        {
            "graph_level": "edge",
            "label_name": "is_synthetic_rescue",
            "label_statistic_name": "synthetic_rescue_statistic_score",
            "is_synthetic_rescue": False,
            "synthetic_rescue_statistic_score": 0.3,
        },
    ),
    (
        "calmorph",
        "calmorph phenotype",
        s.CalMorphExperimentReference,
        s.CalMorphPhenotype(
            calmorph={"A101_A": 1.5},
            calmorph_coefficient_of_variation={"CCV101_C": 0.2},
        ),
        {
            "graph_level": "global",
            "label_name": "calmorph",
            "label_statistic_name": "calmorph_coefficient_of_variation",
            "calmorph": '{"A101_A": 1.5}',
            "calmorph_coefficient_of_variation": '{"CCV101_C": 0.2}',
        },
    ),
    (
        "calmorph",
        "calmorph phenotype",
        s.CalMorphExperimentReference,
        s.CalMorphPhenotype(calmorph={"A101_A": 1.5}),
        {
            "graph_level": "global",
            "label_name": "calmorph",
            "label_statistic_name": "calmorph_coefficient_of_variation",
            "calmorph": '{"A101_A": 1.5}',
            "calmorph_coefficient_of_variation": None,
        },
    ),
    (
        "microarray_expression",
        "microarray expression phenotype",
        s.MicroarrayExpressionExperimentReference,
        s.MicroarrayExpressionPhenotype(
            expression={"YAL001C": 2.0},
            expression_log2_ratio={"YAL001C": 1.0},
            expression_log2_ratio_se={"YAL001C": 0.1},
            n_replicates={"YAL001C": 2},
        ),
        {
            "graph_level": "node",
            "label_name": "expression_log2_ratio",
            "label_statistic_name": "expression_log2_ratio_se",
            "expression_log2_ratio": '{"YAL001C": 1.0}',
            "expression_log2_ratio_se": '{"YAL001C": 0.1}',
        },
    ),
    (
        "microarray_expression",
        "microarray expression phenotype",
        s.MicroarrayExpressionExperimentReference,
        s.MicroarrayExpressionPhenotype(
            expression={"YAL001C": 2.0},
            expression_log2_ratio={"YAL001C": 1.0},
            n_replicates={"YAL001C": 2},
        ),
        {
            "graph_level": "node",
            "label_name": "expression_log2_ratio",
            "label_statistic_name": "expression_log2_ratio_se",
            "expression_log2_ratio": '{"YAL001C": 1.0}',
            "expression_log2_ratio_se": None,
        },
    ),
    (
        "rnaseq_expression",
        "rnaseq expression phenotype",
        s.RNASeqExpressionExperimentReference,
        s.RNASeqExpressionPhenotype(
            expression_tpm={"YAL001C": 10.0},
            expression_count={"YAL001C": 5},
            measurement_type="tpm",
            n_mapped_reads=100,
        ),
        {
            "graph_level": "node",
            "label_name": "expression_tpm",
            "label_statistic_name": None,
            "expression_tpm": '{"YAL001C": 10.0}',
            "measurement_type": "tpm",
            "n_mapped_reads": 100,
        },
    ),
    (
        "pseudobulk_expression",
        "pseudobulk expression phenotype",
        s.PseudobulkExpressionExperimentReference,
        s.PseudobulkExpressionPhenotype(
            expression_log2_ratio={"YAL001C": 0.5},
            dispersion=0.25,
            n_cells=40,
            measurement_type="log2fc",
        ),
        {
            "graph_level": "node",
            "label_name": "expression_log2_ratio",
            "label_statistic_name": "dispersion",
            "expression_log2_ratio": '{"YAL001C": 0.5}',
            "dispersion": 0.25,
            "n_cells": 40,
            "measurement_type": "log2fc",
        },
    ),
    (
        "visual_score",
        "visual score phenotype",
        s.VisualScoreExperimentReference,
        s.VisualScorePhenotype(
            visual_score=2.0,
            n_replicates=3,
            score_scale_min=0,
            score_scale_max=4,
            score_semantics="higher is more",
            target_product="betaxanthin",
        ),
        {
            "graph_level": "global",
            "label_name": "visual_score",
            "label_statistic_name": None,
            "visual_score": 2.0,
            "n_replicates": 3,
            "target_product": "betaxanthin",
            "target_metabolite_id": None,
        },
    ),
    (
        "metabolite",
        "metabolite phenotype",
        s.MetaboliteExperimentReference,
        s.MetabolitePhenotype(
            metabolite_level={"s_0001": 1.25},
            metabolite_level_se={"s_0001": 0.5},
            n_replicates={"s_0001": 3},
            measurement_type="ms_abundance",
        ),
        {
            "graph_level": "metabolism",
            "label_name": "metabolite_level",
            "label_statistic_name": "metabolite_level_se",
            "metabolite_level": '{"s_0001": 1.25}',
            "metabolite_level_se": '{"s_0001": 0.5}',
            "measurement_type": "ms_abundance",
        },
    ),
    (
        "metabolite",
        "metabolite phenotype",
        s.MetaboliteExperimentReference,
        s.MetabolitePhenotype(
            metabolite_level={"s_0001": 1.25},
            n_replicates={"s_0001": 3},
            measurement_type="ms_abundance",
        ),
        {
            "graph_level": "metabolism",
            "label_name": "metabolite_level",
            "label_statistic_name": "metabolite_level_se",
            "metabolite_level": '{"s_0001": 1.25}',
            "metabolite_level_se": None,
            "measurement_type": "ms_abundance",
        },
    ),
    (
        "protein_abundance",
        "protein abundance phenotype",
        s.ProteinAbundanceExperimentReference,
        s.ProteinAbundancePhenotype(
            protein_abundance={"YAL001C": 7.5},
            protein_abundance_se={"YAL001C": 0.5},
            n_replicates={"YAL001C": 2},
            measurement_type="swath",
        ),
        {
            "graph_level": "node",
            "label_name": "protein_abundance",
            "label_statistic_name": "protein_abundance_se",
            "protein_abundance": '{"YAL001C": 7.5}',
            "protein_abundance_se": '{"YAL001C": 0.5}',
            "measurement_type": "swath",
        },
    ),
    (
        "protein_abundance",
        "protein abundance phenotype",
        s.ProteinAbundanceExperimentReference,
        s.ProteinAbundancePhenotype(
            protein_abundance={"YAL001C": 7.5},
            n_replicates={"YAL001C": 2},
            measurement_type="swath",
        ),
        {
            "graph_level": "node",
            "label_name": "protein_abundance",
            "label_statistic_name": "protein_abundance_se",
            "protein_abundance": '{"YAL001C": 7.5}',
            "protein_abundance_se": None,
            "measurement_type": "swath",
        },
    ),
    (
        "environment_response",
        "environment response phenotype",
        s.EnvironmentResponseExperimentReference,
        s.EnvironmentResponsePhenotype(
            measurement_type=s.MeasurementType.log2_ratio,
            environment_response=0.75,
            environment_response_se=0.125,
        ),
        {
            "graph_level": "global",
            "label_name": "environment_response",
            "label_statistic_name": "environment_response_se",
            "environment_response": 0.75,
            "environment_response_se": 0.125,
            "measurement_type": "log2_ratio",
            "assay_type": None,
            "category": None,
            "category_label": None,
            "screen_id": None,
            # #776: both released confidence limits, the level and the replicate id
            "environment_response_lower": None,
            "environment_response_upper": None,
            "confidence_level": None,
            "replicate_id": None,
        },
    ),
    (
        "environment_response",
        "environment response phenotype",
        s.EnvironmentResponseExperimentReference,
        s.EnvironmentResponsePhenotype(
            measurement_type=s.MeasurementType.categorical,
            assay_type=s.AssayType.spot_dilution,
            category=s.ResponseCategory.sensitive,
            category_label="S",
            screen_id="screen-7",
        ),
        {
            "graph_level": "global",
            "label_name": "environment_response",
            "label_statistic_name": "environment_response_se",
            "environment_response": None,
            "environment_response_se": None,
            "measurement_type": "categorical",
            "assay_type": "spot_dilution",
            "category": "sensitive",
            "category_label": "S",
            "screen_id": "screen-7",
            # #776: both released confidence limits, the level and the replicate id
            "environment_response_lower": None,
            "environment_response_upper": None,
            "confidence_level": None,
            "replicate_id": None,
        },
    ),
]


def _phenotype_node(
    phenotype: Any, label: str, props: dict[str, Any], preferred_id: str
) -> BioCypherNode:
    return BioCypherNode(
        node_id=_sha(phenotype),
        preferred_id=preferred_id,
        node_label=label,
        properties=dict(props),
    )


@pytest.mark.parametrize(
    ("stem", "label", "reference_cls", "phenotype", "props"), PHENOTYPE_CASES
)
def test_phenotype_chunk_handler_node_is_exact(
    stem: str, label: str, reference_cls: Any, phenotype: Any, props: dict[str, Any]
) -> None:
    """A record's phenotype node carries ``phenotype_<id>`` as its preferred id; dict
    phenotypes are JSON strings, and an absent optional dict is None, not "null".
    """
    handler = getattr(CellAdapter, f"_{stem}_phenotype_node")
    data = {"experiment": SimpleNamespace(phenotype=phenotype)}
    node = _undecorated(handler)(_bare(), data, f"{label} (chunked)")
    assert node == _phenotype_node(
        phenotype, label, props, preferred_id=f"phenotype_{_sha(phenotype)}"
    )


@pytest.mark.parametrize(
    ("stem", "label", "reference_cls", "phenotype", "props"), PHENOTYPE_CASES
)
def test_phenotype_reference_collector_node_is_exact(
    recorder: _WandbRecorder,
    stem: str,
    label: str,
    reference_cls: Any,
    phenotype: Any,
    props: dict[str, Any],
) -> None:
    """The reference variant has the same properties with the label as preferred id."""
    reference = reference_cls(
        dataset_name="ToyDataset",
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=phenotype,
    )
    adapter = _adapter(_MemoryDataset([], _index(reference)))
    collector = getattr(adapter, f"_get_{stem}_phenotype_reference_nodes")
    assert collector() == [_phenotype_node(phenotype, label, props, preferred_id=label)]


def test_only_the_environment_response_reference_collector_deduplicates(
    recorder: _WandbRecorder,
) -> None:
    """Finding: two index entries with the SAME phenotype reference give one node from
    ``_get_environment_response_phenotype_reference_nodes`` (cell_adapter.py line 1103
    skips a seen id) but two identical nodes from every other phenotype collector,
    e.g. ``_get_fitness_phenotype_reference_nodes`` (line 1118 has no seen set).
    """
    response = s.EnvironmentResponsePhenotype(
        measurement_type=s.MeasurementType.z_score, environment_response=1.0
    )
    response_ref = s.EnvironmentResponseExperimentReference(
        dataset_name="ToyDataset",
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=response,
    )
    adapter = _adapter(_MemoryDataset([], _index(response_ref, response_ref)))
    assert adapter._get_environment_response_phenotype_reference_nodes() == [
        _phenotype_node(
            response,
            "environment response phenotype",
            {
                "graph_level": "global",
                "label_name": "environment_response",
                "label_statistic_name": "environment_response_se",
                "environment_response": 1.0,
                "environment_response_se": None,
                "measurement_type": "z_score",
                "assay_type": None,
                "category": None,
                "category_label": None,
                "screen_id": None,
                # #776: both released confidence limits, the level and the replicate id
                "environment_response_lower": None,
                "environment_response_upper": None,
                "confidence_level": None,
                "replicate_id": None,
            },
            preferred_id="environment response phenotype",
        )
    ]
    fitness_adapter = _adapter(_MemoryDataset([], _index(_reference(), _reference())))
    nodes = fitness_adapter._get_fitness_phenotype_reference_nodes()
    assert len(nodes) == 2
    assert nodes[0] == nodes[1]


# ----------------------------------------------------- reference node collectors


def test_experiment_reference_genome_and_dataset_nodes(
    recorder: _WandbRecorder,
) -> None:
    """Two references (fitness 1.0, 0.9) give two experiment-reference nodes but one
    genome node, since both name S288C; the dataset node is the dataset's name.
    """
    first, second = _reference(1.0), _reference(0.9)
    adapter = _adapter(_MemoryDataset([], _index(first, second)))
    assert adapter._get_experiment_reference_nodes() == [
        BioCypherNode(
            node_id=_sha(ref),
            preferred_id="experiment reference",
            node_label="experiment reference",
            properties={"serialized_data": json.dumps(ref.model_dump())},
        )
        for ref in (first, second)
    ]
    assert adapter._get_genome_nodes() == [
        BioCypherNode(
            node_id=_sha(GENOME),
            preferred_id="genome",
            node_label="genome",
            properties={
                "species": "Saccharomyces cerevisiae",
                "strain": "S288C",
                "serialized_data": json.dumps(GENOME.model_dump()),
            },
        )
    ]
    assert adapter._get_dataset_nodes() == [
        BioCypherNode(
            node_id="ToyDataset", preferred_id="ToyDataset", node_label="dataset"
        )
    ]


# ------------------------------------------------------------------------- edges


def test_record_edges_point_from_part_to_whole() -> None:
    """Every record edge's source and target are the ids the node builders emit."""
    experiment = _experiment(0.5)
    reference = _reference()
    data = {
        "experiment": experiment,
        "reference": reference,
        "publication": PUBLICATION,
    }
    adapter = _bare()
    named: Any = SimpleNamespace(name="ToyDataset")
    adapter.dataset = named
    experiment_id = _sha(experiment)
    genotype_id = _sha(experiment.genotype)

    def call(name: str) -> Any:
        return _undecorated(getattr(CellAdapter, name))(adapter, data, name)

    assert call("_experiment_to_dataset_edge") == BioCypherEdge(
        source_id=experiment_id,
        target_id="ToyDataset",
        relationship_label="experiment member of",
    )
    assert call("_experiment_reference_to_experiment_edge") == BioCypherEdge(
        source_id=_sha(reference),
        target_id=experiment_id,
        relationship_label="experiment reference of",
    )
    assert call("_genotype_to_experiment_edge") == BioCypherEdge(
        source_id=genotype_id,
        target_id=experiment_id,
        relationship_label="genotype member of",
    )
    assert call("_perturbation_to_genotype_edges") == [
        BioCypherEdge(
            source_id=_sha(perturbation),
            target_id=genotype_id,
            relationship_label="perturbation member of",
        )
        for perturbation in (PLAIN_DELETION, SGA_DELETION)
    ]
    assert call("_phenotype_to_experiment_edge") == BioCypherEdge(
        source_id=_sha(experiment.phenotype),
        target_id=experiment_id,
        relationship_label="phenotype member of",
    )
    assert call("_publication_to_experiment_edge") == BioCypherEdge(
        source_id=_sha(PUBLICATION),
        target_id=experiment_id,
        relationship_label="mentions",
    )
    assert call("_environment_to_experiment_edge") == BioCypherEdge(
        source_id=identity_sha256(environment_identity(ENVIRONMENT)),
        target_id=experiment_id,
        relationship_label="environment member of",
    )


def test_reference_edges_are_deduplicated_per_pair(recorder: _WandbRecorder) -> None:
    """Index: fitness 1.0, then 0.9, then 1.0 again. The dataset edges keep all three
    (no dedup); every other reference edge keeps one per (source, target) pair, so the
    repeated reference collapses and two distinct references remain.
    """
    first, second = _reference(1.0), _reference(0.9)
    adapter = _adapter(_MemoryDataset([], _index(first, second, first)))
    env_id = identity_sha256(environment_identity(ENVIRONMENT))
    ref_ids = [_sha(first), _sha(second)]
    assert adapter._get_experiment_reference_to_dataset_edges() == [
        BioCypherEdge(
            source_id=ref_id,
            target_id="ToyDataset",
            relationship_label="experiment reference member of",
        )
        for ref_id in (ref_ids[0], ref_ids[1], ref_ids[0])
    ]
    assert adapter._get_environment_to_experiment_reference_edges() == [
        BioCypherEdge(
            source_id=env_id,
            target_id=ref_id,
            relationship_label="environment member of",
        )
        for ref_id in ref_ids
    ]
    assert adapter._get_genome_to_experiment_reference_edges() == [
        BioCypherEdge(
            source_id=_sha(GENOME),
            target_id=ref_id,
            relationship_label="genome member of",
        )
        for ref_id in ref_ids
    ]
    assert adapter._get_phenotype_to_experiment_reference_edges() == [
        BioCypherEdge(
            source_id=_sha(ref.phenotype_reference),
            target_id=ref_id,
            relationship_label="phenotype member of",
        )
        for ref, ref_id in zip((first, second), ref_ids, strict=True)
    ]
    # one environment across the three references -> one perturbation edge
    assert adapter._get_environment_perturbation_to_environment_reference_edges() == [
        BioCypherEdge(
            source_id=identity_sha256(
                environment_perturbation_identity(SMALL_MOLECULE)
            ),
            target_id=env_id,
            relationship_label="environment perturbation member of",
        )
    ]


# ----------------------------------------------------- chunking and orchestration


class _RecordingLoader(CpuExperimentLoaderMultiprocessing):
    """The real loader, remembering the batch size and worker count it was built with."""

    built: list[tuple[int, int]] = []

    def __init__(self, dataset: Any, batch_size: int, num_workers: int) -> None:
        _RecordingLoader.built.append((batch_size, num_workers))
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers)


def test_data_chunker_loads_transforms_and_flattens_in_process(
    recorder: _WandbRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two records through the decorated handlers, called directly (one forked loader
    worker each). The node method's factor 0.5 halves the loader batch, 2 -> 1; a list
    result is flattened (2 records x 2 perturbations -> 4 nodes).
    """
    _RecordingLoader.built = []
    monkeypatch.setattr(
        cell_adapter_module, "CpuExperimentLoaderMultiprocessing", _RecordingLoader
    )
    dataset = _dataset(2)
    adapter = _adapter(
        dataset,
        node_methods=[
            {"method_name": "experiment (chunked)", "memory_reduction_factor": 0.5}
        ],
    )
    nodes = adapter._experiment_node(dataset, "experiment (chunked)")
    # each record contributes [experiment, interned environment]; the chunk does not
    # dedup the shared environment constant (the writer does)
    env_id = _environment_constant_id(_experiment(0.5))
    assert [(node.get_label(), node.get_id()) for node in nodes] == [
        ("experiment", _sha(_experiment(0.5))),
        ("interned constant", env_id),
        ("experiment", _sha(_experiment(0.25))),
        ("interned constant", env_id),
    ]
    perturbations = adapter._perturbation_node(dataset, "perturbation (chunked)")
    assert [node.get_id() for node in perturbations] == [
        _sha(PLAIN_DELETION),
        _sha(SGA_DELETION),
        _sha(PLAIN_DELETION),
        _sha(SGA_DELETION),
    ]
    assert _RecordingLoader.built == [(1, 1), (2, 1)]


def test_data_chunker_sizes_an_edge_batch_from_the_node_list(
    recorder: _WandbRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: the decorator calls ``get_memory_reduction_factor(method_name)`` without
    ``is_edge`` (cell_adapter.py line 404), so an EDGE method's configured factor never
    reaches its loader batch: factor 0.5 on the edge leaves the batch at 2, while
    ``get_data_by_type`` does apply it to the chunk size.
    """
    _RecordingLoader.built = []
    monkeypatch.setattr(
        cell_adapter_module, "CpuExperimentLoaderMultiprocessing", _RecordingLoader
    )
    dataset = _dataset(2)
    adapter = _adapter(
        dataset,
        edge_methods=[
            {
                "method_name": "experiment to dataset (chunked)",
                "memory_reduction_factor": 0.5,
            }
        ],
    )
    edges = adapter._experiment_to_dataset_edge(
        dataset, "experiment to dataset (chunked)"
    )
    assert [edge.get_source_id() for edge in edges] == [
        _sha(_experiment(0.5)),
        _sha(_experiment(0.25)),
    ]
    assert _RecordingLoader.built == [(2, 1)]


def test_get_nodes_runs_enabled_collectors_in_table_order_and_logs_events(
    recorder: _WandbRecorder,
) -> None:
    """Finding: the docstring (cell_adapter.py line 446) says "in config order", but the
    loop walks the registration table (line 447), so a conf listing dataset before
    genome still yields the genome node first. ``get_edges`` has the same wrong "in
    config order" docstring (line 461) over the same table walk. Events count from 1 and
    continue into ``get_edges``.
    """
    adapter = _adapter(
        _dataset(),
        node_methods=[{"method_name": "dataset"}, {"method_name": "genome"}],
        edge_methods=[{"method_name": "genome to experiment reference"}],
    )
    recorder.logged.clear()
    nodes = list(adapter.get_nodes())
    assert [node.get_label() for node in nodes] == ["genome", "dataset"]
    edges = list(adapter.get_edges())
    assert edges == [
        BioCypherEdge(
            source_id=_sha(GENOME),
            target_id=_sha(_reference()),
            relationship_label="genome member of",
        )
    ]
    assert recorder.logged == [
        {"event": 1, "method": "genome", "type": "node"},
        {"event": 2, "method": "dataset", "type": "node"},
        {"event": 3, "method": "genome to experiment reference", "type": "edge"},
    ]
    assert adapter.event == 3


def test_get_data_by_type_forks_one_pool_per_group_and_keeps_record_order(
    recorder: _WandbRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three records, chunk_size 2 scaled by the edge factor 0.5 -> 1: chunks (0,1),
    (1,2), (2,3), all sliced before any submission, then the LMDB closed once. With
    ``process_workers`` 1 a group is 2 chunks, so two pools are forked (this forks: two
    ProcessPoolExecutor workers, each forking one loader worker per chunk); the pools
    are counted by wrapping ``ProcessPoolExecutor`` at its import site. The edges come
    back in record order.
    """
    pools: list[int] = []

    class _CountingPool(ProcessPoolExecutor):
        def __init__(self, max_workers: int | None = None) -> None:
            pools.append(0 if max_workers is None else max_workers)
            super().__init__(max_workers=max_workers)

    monkeypatch.setattr(
        "torchcell.adapters.cell_adapter.ProcessPoolExecutor", _CountingPool
    )
    dataset = _dataset(3)
    adapter = _adapter(
        dataset,
        edge_methods=[
            {
                "method_name": "experiment to dataset (chunked)",
                "memory_reduction_factor": 0.5,
            }
        ],
        chunk_size=2,
        loader_batch_size=1,
    )
    recorder.logged.clear()
    edges = list(adapter.get_edges())
    assert dataset.slices == [(0, 1), (1, 2), (2, 3)]
    assert dataset.close_calls == 1
    assert pools == [1, 1]  # two pools, one worker each
    assert edges == [
        BioCypherEdge(
            source_id=_sha(_experiment(fitness)),
            target_id="ToyDataset",
            relationship_label="experiment member of",
        )
        for fitness in (0.5, 0.25, 0.125)
    ]
    assert recorder.logged == [
        {"event": 1, "method": "experiment to dataset (chunked)", "type": "edge"}
    ]


def test_get_nodes_routes_a_chunked_method_through_the_pool(
    recorder: _WandbRecorder,
) -> None:
    """Two records, chunk_size 2 -> one chunk, one forked pool worker (which forks one
    loader worker); the node list is the two experiment nodes in record order.
    """
    dataset = _dataset(2)
    adapter = _adapter(dataset, node_methods=[{"method_name": "experiment (chunked)"}])
    recorder.logged.clear()
    nodes = list(adapter.get_nodes())
    assert dataset.slices == [(0, 2)]
    env_id = _environment_constant_id(_experiment(0.5))
    assert [(node.get_label(), node.get_id()) for node in nodes] == [
        ("experiment", _sha(_experiment(0.5))),
        ("interned constant", env_id),
        ("experiment", _sha(_experiment(0.25))),
        ("interned constant", env_id),
    ]
    assert recorder.logged == [
        {"event": 1, "method": "experiment (chunked)", "type": "node"}
    ]


# ------------------------------------------------- 2026.10.01 (phase 19): chunking
#
# The pool-side tests below replace ``ProcessPoolExecutor`` at its import site with
# ``_SyncPool``, which runs each submitted chunk at once in this process and hands back
# a finished ``Future``; nothing forks. The chunk function is ``_echo``: it returns one
# tuple per chunk, (method name, first record's ``i``, chunk length), so the yielded
# sequence IS the chunk boundary list. Records of ``_SizedDataset`` are ``{"i": i,
# "x": "a" * k}``; ``json.dumps`` of one is ``{"i": <i>, "x": "<k a's>"}``, i.e.
# 16 + len(str(i)) + k characters (``{"i": `` 6, ``, "x": "`` 8, ``"}`` 2); each test
# also checks its sizes with ``len(json.dumps(...))`` as an independent oracle.


class _SizedDataset:
    """Records ``{"i": i, "x": "a" * sizes[i]}``; slices and LMDB closes are recorded."""

    def __init__(self, sizes: list[int], start: int = 0) -> None:
        self.name = "Sized"
        self.sizes = sizes
        self.start = start
        self.slices: list[tuple[int, int]] = []
        self.close_calls = 0
        self.gets: list[int] = []

    def __len__(self) -> int:
        return len(self.sizes)

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            self.slices.append((idx.start, idx.stop))
            return _SizedDataset(self.sizes[idx], start=self.start + idx.start)
        self.gets.append(idx)
        return {"i": self.start + idx, "x": "a" * self.sizes[idx]}

    def close_lmdb(self) -> None:
        self.close_calls += 1


def _echo(chunk: _SizedDataset, method_name: str) -> list[tuple[str, int, int]]:
    return [(method_name, chunk.start, len(chunk))]


class _SyncPool:
    """``ProcessPoolExecutor`` stand-in: runs each task on submit; records pools."""

    pools: list[dict[str, Any]] = []

    def __init__(self, max_workers: int | None = None) -> None:
        self.record: dict[str, Any] = {"max_workers": max_workers, "submitted": []}
        _SyncPool.pools.append(self.record)

    def __enter__(self) -> _SyncPool:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def submit(self, fn: Any, chunk: Any, method_name: str) -> Any:
        from concurrent.futures import Future

        self.record["submitted"].append(chunk.start)
        future: Future[Any] = Future()
        future.set_result(fn(chunk, method_name))
        return future


@pytest.fixture
def sync_pool(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Install ``_SyncPool`` at the module's ``ProcessPoolExecutor`` name; returns the
    per-pool records (max_workers and the first record of each submitted chunk).
    """
    _SyncPool.pools = []
    monkeypatch.setattr(cell_adapter_module, "ProcessPoolExecutor", _SyncPool)
    return _SyncPool.pools


def _sized_adapter(
    sizes: list[int], node_methods: list[dict[str, Any]] | None = None, **kwargs: Any
) -> tuple[CellAdapter, _SizedDataset]:
    dataset = _SizedDataset(sizes)
    as_dataset: Any = dataset
    adapter = CellAdapter(
        _conf(node_methods or [], []),
        as_dataset,
        process_workers=kwargs.pop("process_workers", 1),
        io_workers=1,
        chunk_size=kwargs.pop("chunk_size", 2),
        loader_batch_size=1,
        **kwargs,
    )
    return adapter, dataset


def test_estimate_record_bytes_is_the_median_of_evenly_spaced_samples(
    recorder: _WandbRecorder,
) -> None:
    """Sizes k = [10, 0, 30, 0, 20]: record i dumps to 17 + k characters
    (``{"i": 0, "x": ""}`` is 17). With ``samples=2`` the step is 5 // 2 = 2, so records
    0, 2, 4 give [27, 47, 37], median (index 3 // 2 = 1 of the sorted list) 37. The LMDB
    is closed once and the value is cached: a later call with other samples or other
    data returns 37 without reading.
    """
    adapter, dataset = _sized_adapter([10, 0, 30, 0, 20])
    assert [len(json.dumps(dataset[i])) for i in range(5)] == [27, 17, 47, 17, 37]
    dataset.gets.clear()
    assert adapter._estimate_record_bytes(samples=2) == 37
    assert dataset.gets == [0, 2, 4]
    assert dataset.close_calls == 1
    dataset.sizes[2] = 1000
    assert adapter._estimate_record_bytes() == 37
    assert dataset.gets == [0, 2, 4]


def test_estimate_record_bytes_default_samples_reads_every_small_record(
    recorder: _WandbRecorder,
) -> None:
    """With the default 64 samples and 5 records the step is max(1, 0) = 1: all five
    sizes [27, 17, 47, 17, 37], sorted [17, 17, 27, 37, 47], median 27.
    """
    adapter, dataset = _sized_adapter([10, 0, 30, 0, 20])
    assert adapter._estimate_record_bytes() == 27
    assert dataset.gets == [0, 1, 2, 3, 4]


def test_single_pass_factor_is_the_smallest_factor_over_the_method_count(
    recorder: _WandbRecorder,
) -> None:
    """Folded methods experiment (0.5), genotype (no factor, 1.0) and perturbation
    (0.8): min 0.5 / 3 methods = 0.1666667, for nodes; for edges the edge list is read,
    where none is configured, so min(1, 1, 1) / 3.
    """
    adapter, _ = _sized_adapter(
        [0],
        node_methods=[
            {"method_name": "experiment (chunked)", "memory_reduction_factor": 0.5},
            {"method_name": "genotype (chunked)"},
            {"method_name": "perturbation (chunked)", "memory_reduction_factor": 0.8},
        ],
    )
    adapter._single_pass_methods = [
        ("experiment (chunked)", adapter._experiment_node),
        ("genotype (chunked)", adapter._genotype_node),
        ("perturbation (chunked)", adapter._perturbation_node),
    ]
    factor = adapter.get_memory_reduction_factor(cell_adapter_module.SINGLE_PASS_NODES)
    assert factor == pytest.approx(0.5 / 3)
    edge_factor = adapter.get_memory_reduction_factor(
        cell_adapter_module.SINGLE_PASS_EDGES, is_edge=True
    )
    assert edge_factor == pytest.approx(1 / 3)


def test_single_pass_chunk_shrinks_to_the_byte_budget(
    recorder: _WandbRecorder,
    sync_pool: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """601 records with k = 11: 16 + digits(i) + 11 bytes, 28 for one-digit i and 30
    for three digits; the 601 // 64 = 9-step sample (i = 0, 9, ..., 594) has 2 + 10 +
    55 members (i = 0, 9 one-digit; 18..99 two-digit; 108..594 three-digit), so its
    median (index 33) is 30 (the oracle computes it). chunk_size 1000 with one
    folded method of factor 1 gives 1000 / 1 = 1000; budget 300 * R // R = 300 < 1000,
    so chunks are [0, 300), [300, 600), [600, 601) and the shrink is logged.
    """
    adapter, dataset = _sized_adapter([11] * 601, chunk_size=1000)
    sample = sorted(
        len(json.dumps({"i": i, "x": "a" * 11})) for i in range(0, 601, 601 // 64)
    )
    record_bytes = sample[len(sample) // 2]
    assert record_bytes == 30
    adapter.single_pass_chunk_budget_bytes = 300 * record_bytes
    adapter._single_pass_methods = [("genotype (chunked)", adapter._genotype_node)]
    with caplog.at_level(logging.INFO, logger="torchcell.adapters.cell_adapter"):
        out = list(
            adapter.get_data_by_type(_echo, cell_adapter_module.SINGLE_PASS_NODES)
        )
    name = cell_adapter_module.SINGLE_PASS_NODES
    assert out == [(name, 0, 300), (name, 300, 300), (name, 600, 1)]
    assert dataset.slices == [(0, 300), (300, 600), (600, 900)]
    assert [r.getMessage() for r in caplog.records] == [
        "single-pass chunk 1000 -> 300 records (30 resolved bytes per record)"
    ]
    # one pool (group = 1 worker x 2 chunks per worker = 2), so two pools for 3 chunks
    assert [p["submitted"] for p in sync_pool] == [[0, 300], [600]]


def test_single_pass_chunk_never_drops_below_the_floor_of_256(
    recorder: _WandbRecorder, sync_pool: list[dict[str, Any]]
) -> None:
    """A budget of 10 records gives max(256, 10) = 256 < 1000: chunks of 256."""
    adapter, _ = _sized_adapter([11] * 600, chunk_size=1000)
    adapter.single_pass_chunk_budget_bytes = 10 * 30
    adapter._single_pass_methods = [("genotype (chunked)", adapter._genotype_node)]
    out = list(adapter.get_data_by_type(_echo, cell_adapter_module.SINGLE_PASS_NODES))
    assert [(start, n) for _, start, n in out] == [(0, 256), (256, 256), (512, 88)]


@pytest.mark.parametrize("budget_records", [800, 500])
def test_single_pass_budget_not_below_the_chunk_leaves_it_unchanged(
    recorder: _WandbRecorder,
    sync_pool: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    budget_records: int,
) -> None:
    """Two folded methods of factor 1: chunk 1000 * 1 / 2 = 500. A budget of 800
    records, or of exactly 500 (the shrink rule is strict ``<``), is not smaller, so
    the chunk stays 500 and nothing is logged (a ``<=`` would log "500 -> 500").
    """
    adapter, _ = _sized_adapter([11] * 600, chunk_size=1000)
    adapter.single_pass_chunk_budget_bytes = budget_records * 30
    adapter._single_pass_methods = [
        ("genotype (chunked)", adapter._genotype_node),
        ("experiment (chunked)", adapter._experiment_node),
    ]
    with caplog.at_level(logging.INFO, logger="torchcell.adapters.cell_adapter"):
        out = list(
            adapter.get_data_by_type(_echo, cell_adapter_module.SINGLE_PASS_NODES)
        )
    assert [(start, n) for _, start, n in out] == [(0, 500), (500, 100)]
    assert caplog.records == []


def _inprocess_echo(calls: list[Any]) -> Any:
    def fn(chunk: Any, method_name: str, inprocess: bool = False) -> list[Any]:
        calls.append((chunk.start, len(chunk), method_name, inprocess))
        return ["a", "b"]

    return fn


@pytest.mark.parametrize(
    ("max_records", "max_bytes", "inprocess"),
    [
        (3, 0, True),  # 0 < 3 <= 3, no byte rule
        (2, 0, False),  # 3 > 2
        (3, 3 * 27, True),  # 3 records * 27 bytes = 81 <= 81
        (3, 3 * 27 - 1, False),  # 81 > 80
    ],
)
def test_small_datasets_run_in_process_by_records_and_optionally_bytes(
    recorder: _WandbRecorder,
    sync_pool: list[dict[str, Any]],
    max_records: int,
    max_bytes: int,
    inprocess: bool,
) -> None:
    """Three 27-byte records (k = 10). In process: the whole dataset [0, 3) goes to the
    chunk function once with ``inprocess=True``, the LMDB is closed, no pool is built.
    Otherwise chunk_size 2 gives pool chunks [0, 2) and [2, 3).
    """
    adapter, dataset = _sized_adapter([10, 10, 10], inprocess_max_records=max_records)
    adapter.inprocess_max_bytes = max_bytes
    calls: list[Any] = []
    out = list(adapter.get_data_by_type(_inprocess_echo(calls), "genotype (chunked)"))
    if inprocess:
        assert calls == [(0, 3, "genotype (chunked)", True)]
        assert out == ["a", "b"]
        assert dataset.slices == [(0, 3)]
        assert sync_pool == []
    else:
        assert calls == [
            (0, 2, "genotype (chunked)", False),
            (2, 1, "genotype (chunked)", False),
        ]
        assert out == ["a", "b", "a", "b"]
        assert dataset.slices == [(0, 2), (2, 4)]
        assert [p["submitted"] for p in sync_pool] == [[0, 2]]
    # one close after slicing (plus one by the byte estimate when it ran)
    assert dataset.close_calls == 1 + (max_bytes > 0)


def test_an_empty_dataset_yields_nothing_and_builds_no_pool(
    recorder: _WandbRecorder, sync_pool: list[dict[str, Any]]
) -> None:
    """0 records: not in process (the rule needs 0 < len), no chunk, no pool."""
    adapter, dataset = _sized_adapter([], inprocess_max_records=5)
    assert list(adapter.get_data_by_type(_echo, "genotype (chunked)")) == []
    assert sync_pool == []
    assert dataset.close_calls == 1


def test_pools_recycle_on_cgroup_memory_after_every_worker_had_a_chunk(
    recorder: _WandbRecorder,
    sync_pool: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """10 one-record chunks, 2 workers, 3 chunks per worker (group 6), threshold 0.5.

    The fraction is checked when the group asks for chunk n + 1 with n >= 2 workers.
    Scripted readings 0.3, 0.6 (then 0.6 again for the log line): pool 1 stops after 3
    chunks. Pool 2 reads 0.2 at n = 2..5 and stops at the group size, 6 chunks. Pool 3
    takes the last chunk (n = 1, below the worker count, so no reading). Seven readings,
    three pools, three ``gc.freeze`` calls, records in order.
    """
    readings = iter([0.3, 0.6, 0.6, 0.2, 0.2, 0.2, 0.2])
    taken: list[float] = []

    def fraction() -> float:
        value = next(readings)
        taken.append(value)
        return value

    freezes: list[int] = []
    monkeypatch.setattr(cell_adapter_module, "cgroup_memory_fraction", fraction)
    monkeypatch.setattr(gc, "freeze", lambda: freezes.append(1))
    adapter, _ = _sized_adapter([0] * 10, chunk_size=1, process_workers=2)
    adapter.chunks_per_worker = 3
    adapter.pool_memory_fraction = 0.5
    with caplog.at_level(logging.INFO, logger="torchcell.adapters.cell_adapter"):
        out = list(adapter.get_data_by_type(_echo, "genotype (chunked)"))
    assert [start for _, start, _ in out] == list(range(10))
    assert [p["submitted"] for p in sync_pool] == [[0, 1, 2], [3, 4, 5, 6, 7, 8], [9]]
    assert [p["max_workers"] for p in sync_pool] == [2, 2, 2]
    assert taken == [0.3, 0.6, 0.6, 0.2, 0.2, 0.2, 0.2]
    assert len(freezes) == 3
    assert [r.getMessage() for r in caplog.records] == [
        "pool recycled at 3 chunks: cgroup memory at 0.60 of its limit"
    ]


def test_in_order_window_submits_workers_plus_two_then_one_per_consumed_chunk(
    recorder: _WandbRecorder, sync_pool: list[dict[str, Any]]
) -> None:
    """One worker, group 10, 5 one-record chunks: before the first result is handed out
    3 chunks (1 + 2) are submitted; consuming chunk 0 submits chunk 3, and so on, and
    results come back in submission order.
    """
    adapter, _ = _sized_adapter([0] * 5, chunk_size=1)
    adapter.chunks_per_worker = 10
    gen = adapter.get_data_by_type(_echo, "genotype (chunked)")
    seen = []
    submitted = []
    for _ in range(5):
        seen.append(next(gen)[1])
        submitted.append(len(sync_pool[0]["submitted"]))
    assert seen == [0, 1, 2, 3, 4]
    assert submitted == [3, 4, 5, 5, 5]
    assert next(gen, None) is None


def test_completion_order_yields_each_chunk_once_with_the_same_window(
    recorder: _WandbRecorder, sync_pool: list[dict[str, Any]]
) -> None:
    """``completion_order``: every submitted future here is already finished, so the
    first ``wait`` returns chunks {0, 1, 2} in set order; each one consumed submits the
    next chunk into the pending set. The first three results are a permutation of
    {0, 1, 2}, the last two of {3, 4}; the submission counts match the in-order path.
    """
    adapter, _ = _sized_adapter([0] * 5, chunk_size=1)
    adapter.chunks_per_worker = 10
    adapter.completion_order = True
    gen = adapter.get_data_by_type(_echo, "genotype (chunked)")
    seen = []
    submitted = []
    for _ in range(5):
        seen.append(next(gen)[1])
        submitted.append(len(sync_pool[0]["submitted"]))
    assert sorted(seen[:3]) == [0, 1, 2]
    assert sorted(seen[3:]) == [3, 4]
    assert submitted == [3, 4, 5, 5, 5]
    assert next(gen, None) is None


def test_data_chunker_in_process_transforms_every_record_and_closes_the_chunk(
    recorder: _WandbRecorder,
) -> None:
    """``inprocess=True``: no loader; each record goes through ``transform_item`` and
    the handler; a list result is flattened, a single node appended; the chunk's LMDB is
    closed once. Two records: the experiment handler gives [experiment, interned
    environment] per record, the publication handler one node per record.
    """
    dataset = _dataset(2)
    adapter = _adapter(dataset)
    nodes = adapter._experiment_node(dataset, "experiment (chunked)", inprocess=True)
    env_id = _environment_constant_id(_experiment(0.5))
    assert [(n.get_label(), n.get_id()) for n in nodes] == [
        ("experiment", _sha(_experiment(0.5))),
        ("interned constant", env_id),
        ("experiment", _sha(_experiment(0.25))),
        ("interned constant", env_id),
    ]
    assert dataset.close_calls == 1
    publications = adapter._publication_node(
        dataset, "publication (chunked)", inprocess=True
    )
    assert [(n.get_label(), n.get_id()) for n in publications] == [
        ("publication", _sha(PUBLICATION)),
        ("publication", _sha(PUBLICATION)),
    ]
    assert dataset.close_calls == 2


def test_all_chunked_applies_every_folded_method_per_record_in_table_order(
    recorder: _WandbRecorder,
) -> None:
    """The single-pass body calls each folded handler's ``__wrapped__`` on the same
    transformed record: per record, experiment's two nodes then publication's one.
    """
    dataset = _dataset(2)
    adapter = _adapter(dataset)
    adapter._single_pass_methods = [
        ("experiment (chunked)", adapter._experiment_node),
        ("publication (chunked)", adapter._publication_node),
    ]
    nodes = adapter._all_chunked(
        dataset, cell_adapter_module.SINGLE_PASS_NODES, inprocess=True
    )
    env_id = _environment_constant_id(_experiment(0.5))
    expected = []
    for fitness in (0.5, 0.25):
        expected += [
            ("experiment", _sha(_experiment(fitness))),
            ("interned constant", env_id),
            ("publication", _sha(PUBLICATION)),
        ]
    assert [(n.get_label(), n.get_id()) for n in nodes] == expected


def test_pack_chunk_renders_one_chunk_when_row_specs_are_set(
    recorder: _WandbRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``row_specs`` the list comes back as is (the same object); with them the
    output is ``[RenderedChunk.from_rows(datas, row_specs)]``.
    """
    adapter = _adapter(_dataset(1))
    datas = ["n1", "n2"]
    assert adapter._pack_chunk(datas) is datas
    rendered: list[Any] = []

    def from_rows(rows: Any, specs: Any) -> str:
        rendered.append((rows, specs))
        return "rendered"

    monkeypatch.setattr(RenderedChunk, "from_rows", from_rows)
    specs: Any = object()
    adapter.row_specs = specs
    assert adapter._pack_chunk(datas) == ["rendered"]
    assert rendered == [(datas, specs)]


class _PhaseRecorder:
    calls: list[tuple[str, str, str]] = []

    @classmethod
    def set(cls, adapter: str, method: str, kind: str) -> None:
        cls.calls.append((adapter, method, kind))


@pytest.mark.parametrize("kind", ["node", "edge"])
def test_single_pass_runs_reference_methods_then_one_folded_traversal(
    recorder: _WandbRecorder, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """``single_pass``: the enabled ``_get_*`` collector runs in the table loop; the
    enabled chunked methods are skipped there and folded, in table order, into ONE
    ``get_data_by_type(self._all_chunked, <pass name>, is_edge=...)`` call. Events 1
    (the collector) and 2 (the pass) are logged; ``BuildPhase`` sees both. Disabled
    methods (here ``publication`` / ``publication to experiment``) are not folded.
    """
    _PhaseRecorder.calls = []
    monkeypatch.setattr(cell_adapter_module, "BuildPhase", _PhaseRecorder)
    if kind == "node":
        config = [
            {"method_name": "genotype (chunked)"},
            {"method_name": "dataset"},
            {"method_name": "experiment (chunked)"},
        ]
        adapter = _adapter(_dataset(1), node_methods=config)
        folded = [
            ("experiment (chunked)", adapter._experiment_node),
            ("genotype (chunked)", adapter._genotype_node),
        ]
        reference = "dataset"
        pass_name = cell_adapter_module.SINGLE_PASS_NODES
    else:
        config = [
            {"method_name": "experiment to dataset (chunked)"},
            {"method_name": "genome to experiment reference"},
            {"method_name": "genotype to experiment (chunked)"},
        ]
        adapter = _adapter(_dataset(1), edge_methods=config)
        folded = [
            ("experiment to dataset (chunked)", adapter._experiment_to_dataset_edge),
            ("genotype to experiment (chunked)", adapter._genotype_to_experiment_edge),
        ]
        reference = "genome to experiment reference"
        pass_name = cell_adapter_module.SINGLE_PASS_EDGES
    adapter.single_pass = True
    calls: list[Any] = []

    def get_data_by_type(method: Any, name: str, is_edge: bool = False) -> Any:
        calls.append((method, name, is_edge))
        yield "folded"

    monkeypatch.setattr(adapter, "get_data_by_type", get_data_by_type)
    recorder.logged.clear()
    out = list(adapter.get_nodes() if kind == "node" else adapter.get_edges())
    assert len(out) == 2
    assert out[1] == "folded"
    if kind == "node":
        assert (out[0].get_label(), out[0].get_id()) == ("dataset", "ToyDataset")
    else:
        assert out[0] == BioCypherEdge(
            source_id=_sha(GENOME),
            target_id=_sha(_reference()),
            relationship_label="genome member of",
        )
    assert calls == [(adapter._all_chunked, pass_name, kind == "edge")]
    assert adapter._single_pass_methods == folded
    assert recorder.logged == [
        {"event": 1, "method": reference, "type": kind},
        {"event": 2, "method": pass_name, "type": kind},
    ]
    assert _PhaseRecorder.calls == [
        ("CellAdapter", reference, kind),
        ("CellAdapter", pass_name, kind),
    ]


def test_single_pass_with_no_chunked_method_enabled_runs_no_traversal(
    recorder: _WandbRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a collector enabled: one event, no ``get_data_by_type`` call."""
    adapter = _adapter(_dataset(1), node_methods=[{"method_name": "dataset"}])
    adapter.single_pass = True
    calls: list[Any] = []

    def no_traversal(*args: Any, **kwargs: Any) -> Iterator[Any]:
        calls.append(args)
        return iter(())

    monkeypatch.setattr(adapter, "get_data_by_type", no_traversal)
    recorder.logged.clear()
    assert [n.get_label() for n in adapter.get_nodes()] == ["dataset"]
    assert calls == []
    assert recorder.logged == [{"event": 1, "method": "dataset", "type": "node"}]


def _cgroup_files(
    monkeypatch: pytest.MonkeyPatch, root: Any, files: dict[str, str]
) -> None:
    """Write ``files`` under ``root`` and point the two module paths at the v2 names."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(
        cell_adapter_module, "CGROUP_MEMORY_CURRENT", str(root / "memory.current")
    )
    monkeypatch.setattr(
        cell_adapter_module, "CGROUP_MEMORY_MAX", str(root / "memory.max")
    )


def test_cgroup_v2_fraction_is_current_over_max(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """memory.current 2147483648 (2 GiB) over memory.max 8589934592 (8 GiB) = 0.25;
    trailing newlines are stripped.
    """
    _cgroup_files(
        monkeypatch,
        tmp_path,
        {"memory.max": "8589934592\n", "memory.current": "2147483648\n"},
    )
    assert cell_adapter_module.cgroup_memory_fraction() == 0.25


def test_cgroup_without_a_limit_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """memory.max ``max`` (no limit) raises before memory.current is read (it is
    absent here, which would otherwise raise FileNotFoundError).
    """
    _cgroup_files(monkeypatch, tmp_path, {"memory.max": "max\n"})
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "pool_memory_fraction needs a cgroup memory limit; memory.max is 'max'"
        ),
    ):
        cell_adapter_module.cgroup_memory_fraction()


def test_cgroup_v1_layout_is_not_read_and_raises_file_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Only the v2 files are read. A v1 tree (``memory/memory.limit_in_bytes``,
    ``memory/memory.usage_in_bytes``) has no ``memory.max``, so the call raises
    ``FileNotFoundError`` naming that path; with ``memory.max`` present but
    ``memory.current`` missing it names ``memory.current``.
    """
    _cgroup_files(
        monkeypatch,
        tmp_path,
        {
            "memory/memory.limit_in_bytes": "8589934592\n",
            "memory/memory.usage_in_bytes": "2147483648\n",
        },
    )
    with pytest.raises(FileNotFoundError) as missing_max:
        cell_adapter_module.cgroup_memory_fraction()
    assert missing_max.value.filename == str(tmp_path / "memory.max")
    (tmp_path / "memory.max").write_text("100\n")
    with pytest.raises(FileNotFoundError) as missing_current:
        cell_adapter_module.cgroup_memory_fraction()
    assert missing_current.value.filename == str(tmp_path / "memory.current")


def test_single_pass_over_an_empty_dataset_fails_in_the_byte_estimate(
    recorder: _WandbRecorder, sync_pool: list[dict[str, Any]]
) -> None:
    """Finding: ``_estimate_record_bytes`` takes ``sizes[len(sizes) // 2]`` of an empty
    sample list when the dataset has no records (cell_adapter.py:617-626), and the
    single-pass branch of ``get_data_by_type`` calls it before anything checks the
    length (:411), so a folded pass over an empty dataset raises ``IndexError`` where the
    per-method path yields nothing (``test_an_empty_dataset_yields_nothing...``). Not
    measured: whether any served dataset can be empty at build time. Pinned until the
    estimate (or the single-pass branch) handles zero records.
    """
    adapter, _ = _sized_adapter([])
    adapter._single_pass_methods = [("genotype (chunked)", adapter._genotype_node)]
    with pytest.raises(IndexError, match=re.escape("list index out of range")):
        list(adapter.get_data_by_type(_echo, cell_adapter_module.SINGLE_PASS_NODES))
    assert sync_pool == []


def test_estimate_record_bytes_takes_the_upper_median_of_an_even_sample(
    recorder: _WandbRecorder,
) -> None:
    """Four records k = [30, 0, 20, 10] dump to [47, 17, 37, 27]; sorted
    [17, 27, 37, 47], index 4 // 2 = 2, so 37 (the upper of the two middle values,
    not 27 and not their mean 32).
    """
    adapter, dataset = _sized_adapter([30, 0, 20, 10])
    assert [len(json.dumps(dataset[i])) for i in range(4)] == [47, 17, 37, 27]
    assert adapter._estimate_record_bytes() == 37


def test_shipped_costanzo_edge_factors_never_reach_the_loader_batch(
    recorder: _WandbRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding (reach of ``test_data_chunker_sizes_an_edge_batch_from_the_node_list``):
    ``data_chunker`` asks ``get_memory_reduction_factor(method_name)`` without
    ``is_edge`` (cell_adapter.py:584), so it searches the NODE list, finds no edge
    name there and returns 1.0. With the shipped
    ``torchcell/adapters/conf/dmf_costanzo2016_adapter.yaml`` (loaded by
    ``costanzo2016_adapter.py``), whose chunked edge methods carry 0.5, the edge
    lookup with ``is_edge=True`` is 0.5 while the decorator's lookup is 1.0, and the
    "experiment to dataset" loader is built with the full batch 2 instead of 1. Not
    measured: the memory this costs on a real build. Pinned until the decorator passes
    ``is_edge``.
    """
    _RecordingLoader.built = []
    monkeypatch.setattr(
        cell_adapter_module, "CpuExperimentLoaderMultiprocessing", _RecordingLoader
    )
    root = Path(cell_adapter_module.__file__).parent
    conf = OmegaConf.load(root / "conf" / "dmf_costanzo2016_adapter.yaml")
    assert isinstance(conf, DictConfig)
    dataset = _dataset(2)
    adapter = CellAdapter(
        conf,
        dataset,
        process_workers=1,
        io_workers=1,
        chunk_size=2,
        loader_batch_size=2,
    )
    name = "experiment to dataset (chunked)"
    assert adapter.get_memory_reduction_factor(name, is_edge=True) == 0.5
    assert adapter.get_memory_reduction_factor(name) == 1.0
    edges = adapter._experiment_to_dataset_edge(dataset, name)
    assert [edge.get_source_id() for edge in edges] == [
        _sha(_experiment(0.5)),
        _sha(_experiment(0.25)),
    ]
    assert _RecordingLoader.built == [(2, 1)]


def test_publication_preferred_id_falls_back_from_pubmed_to_doi_to_identifier() -> None:
    """A non-journal source has no PubMed id, so the preferred id must not say ``None``.

    The fallback order is pubmed -> doi -> identifier. A source with none of the three
    cannot exist: ``Publication`` requires a doi/pmid for a journal article and an
    identifier for every other source type.
    """
    adapter = _bare()
    build = _undecorated(CellAdapter._publication_node)

    doi_only = s.Publication(doi="10.1/x", doi_url="https://doi.org/10.1/x")
    node = build(adapter, {"publication": doi_only}, "publication (chunked)")
    assert node.get_preferred_id() == "publication_10.1/x"
    assert node.get_properties()["source_type"] == "journal_article"

    identifier = "si/prelim-report.pdf sha256:" + "b" * 64
    report = s.Publication(
        source_type=s.SourceType.preliminary_report,
        title="Preliminary examination report",
        identifier=identifier,
        identifier_url="https://example.invalid/report.pdf",
    )
    node = build(adapter, {"publication": report}, "publication (chunked)")
    assert node.get_preferred_id() == f"publication_{identifier}"
    properties = node.get_properties()
    assert properties["pubmed_id"] is None
    assert properties["doi"] is None
    assert properties["source_type"] == "preliminary_report"
    assert properties["title"] == "Preliminary examination report"
    assert properties["identifier"] == identifier
    assert properties["identifier_url"] == "https://example.invalid/report.pdf"
    back = s.Publication.model_validate_json(properties["serialized_data"])
    assert back == report
    # The node id is content-addressed, so two source kinds never collide.
    assert node.get_id() == _sha(report)
    assert node.get_id() != _sha(doi_only)
