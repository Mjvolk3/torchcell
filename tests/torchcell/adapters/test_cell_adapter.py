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
import math
import re
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
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
    "crispr construct (chunked)",
    "environment (chunked)",
    "environment reference",
    "media (chunked)",
    "media reference",
    "temperature (chunked)",
    "temperature reference",
    "environment perturbation (chunked)",
    "environment perturbation reference",
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
    "environment response phenotype (chunked)",
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
    "environment response phenotype reference",
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
    """43 node methods and 16 edge methods, in the order ``__init__`` registers them;
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
    """The toy environment's JSON (701 bytes), at or above the 512-byte pointer floor."""
    payload = json.dumps(experiment.model_dump()["environment"])
    assert len(payload) == 701
    return payload


def _environment_constant_id(experiment: s.FitnessExperiment) -> str:
    return hashlib.sha256(_environment_payload(experiment).encode("utf-8")).hexdigest()


def test_experiment_genotype_and_publication_nodes_are_content_addressed() -> None:
    """The experiment handler returns the experiment node, whose id is the sha256 of
    the fully inlined record but whose blob holds a ``$ref`` pointer in place of the
    701-byte environment (floor 512), followed by one ``interned constant`` node
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
    parent = {
        "assembly_member": "S288C_reference.fa",
        "assembly_sha256": "a" * 64,
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
