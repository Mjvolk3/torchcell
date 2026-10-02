# tests/torchcell/adapters/_sga_adapter_harness.py
# [[tests.torchcell.adapters._sga_adapter_harness]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/_sga_adapter_harness.py
"""Shared harness for the SGA-family adapter tests (Kuzmin 2018/2020, Costanzo 2016).

Not a test module. It provides the in-memory dataset the adapters read
(``MemoryDataset``: ``name``, ``len``, integer and slice indexing, ``transform_item``,
``experiment_reference_index``, ``close_lmdb``), a ``wandb`` recorder and a pinned
``datetime`` for ``CellAdapter.__init__``, and hand-built expectations for the exact
node and edge lists the SGA adapter confs produce over a two-record dataset.

The expectation builders encode the adapter contract independently of ``CellAdapter``:
every content id is ``sha256(json.dumps(model_dump()))`` (``sha``) except the
environment, media and temperature ids, which are ``identity_sha256`` of the identity
projection. The node order is the registration-table order of ``CellAdapter.__init__``
restricted to the conf, and within a chunked method the record order.

Payload layout (torchcell/datamodels/interned_constant.py): every fixture's
environment is the SGA selection medium at 30 C, whose JSON is about 9.5 KB, above the
512-byte environment floor, so each Experiment blob carries an environment pointer and
the experiment method emits the environment as an ``interned constant`` node right
after its Experiment node. Every fixture genotype (one to three gene perturbations) is
well under the 8192-byte genotype floor and stays inline. Genotype, perturbation and
phenotype nodes carry no ``serialized_data``; experiment, experiment reference,
genome, environment, media, temperature and publication nodes keep it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from datetime import datetime
from typing import Any

from biocypher._create import BioCypherEdge, BioCypherNode

from torchcell.data.data import ExperimentReferenceIndex
from torchcell.datamodels import schema as s
from torchcell.datamodels.identity import (
    environment_identity,
    identity_sha256,
    media_identity,
    temperature_identity,
)

GENOME = s.ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
FIXED_NOW = datetime(2026, 9, 28, 12, 0, 0)


def sha(model: Any) -> str:
    """The adapter's content address: sha256 of the json-dumped ``model_dump``."""
    return hashlib.sha256(json.dumps(model.model_dump()).encode("utf-8")).hexdigest()


class MemoryDataset:
    """The slice of ``ExperimentDataset`` the adapter touches, held in memory."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        reference_index: list[ExperimentReferenceIndex],
        name: str,
        experiment_cls: type[Any],
        reference_cls: type[Any],
    ) -> None:
        self.name = name
        self._items = items
        self.experiment_reference_index = reference_index
        self._experiment_cls = experiment_cls
        self._reference_cls = reference_cls
        self.close_calls = 0

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            return MemoryDataset(
                self._items[idx],
                self.experiment_reference_index,
                self.name,
                self._experiment_cls,
                self._reference_cls,
            )
        return self._items[idx]

    def transform_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "experiment": self._experiment_cls(**item["experiment"]),
            "reference": self._reference_cls(**item["reference"]),
            "publication": s.Publication(**item["publication"]),
        }

    def close_lmdb(self) -> None:
        self.close_calls += 1


class Table:
    """Stands in for ``wandb.Table``: keeps what it was built from."""

    def __init__(self, columns: list[str], data: list[list[Any]]) -> None:
        self.columns = columns
        self.data = data


class WandbRecorder:
    """Records ``wandb.init`` calls and every ``wandb.log`` payload."""

    Table = Table

    def __init__(self) -> None:
        self.init_calls = 0
        self.logged: list[dict[str, Any]] = []

    def init(self) -> None:
        self.init_calls += 1

    def log(self, payload: dict[str, Any]) -> None:
        self.logged.append(payload)


class FixedDatetime:
    """``datetime`` stand-in whose ``now`` is pinned to ``FIXED_NOW``."""

    @staticmethod
    def now() -> datetime:
        return FIXED_NOW


def make_dataset(
    name: str,
    experiments: list[Any],
    reference: Any,
    publication: s.Publication,
    experiment_cls: type[Any],
    reference_cls: type[Any],
) -> MemoryDataset:
    """A dataset of ``experiments`` sharing one ``reference`` and one ``publication``."""
    items = [
        {
            "experiment": experiment.model_dump(),
            "reference": reference.model_dump(),
            "publication": publication.model_dump(),
        }
        for experiment in experiments
    ]
    index = [
        ExperimentReferenceIndex(
            reference=reference, member_indices=list(range(len(experiments)))
        )
    ]
    return MemoryDataset(items, index, name, experiment_cls, reference_cls)


def _phenotype_props(phenotype: Any) -> dict[str, Any]:
    """The projected columns of a fitness or gene-interaction phenotype node."""
    props: dict[str, Any] = {
        "graph_level": phenotype.graph_level,
        "label_name": phenotype.label_name,
        "label_statistic_name": phenotype.label_statistic_name,
    }
    if isinstance(phenotype, s.FitnessPhenotype):
        props["fitness"] = phenotype.fitness
        props["fitness_std"] = phenotype.fitness_std
    else:
        props["gene_interaction"] = phenotype.gene_interaction
        props["gene_interaction_p_value"] = phenotype.gene_interaction_p_value
    # Issue #602: both phenotypes project the screen a measurement came from.
    props["screen_id"] = phenotype.screen_id
    return props


def _phenotype_label(phenotype: Any) -> str:
    return (
        "fitness phenotype"
        if isinstance(phenotype, s.FitnessPhenotype)
        else "gene interaction phenotype"
    )


def environment_node(environment: s.Environment) -> BioCypherNode:
    """The environment node: identity id, temperature value, media JSON, serialized."""
    temperature = environment.temperature
    return BioCypherNode(
        node_id=identity_sha256(environment_identity(environment)),
        preferred_id="environment",
        node_label="environment",
        properties={
            "temperature": temperature.value if temperature is not None else None,
            "media": json.dumps(environment.media.model_dump()),
            "serialized_data": json.dumps(environment.model_dump()),
        },
    )


def media_node(media: s.Media) -> BioCypherNode:
    """The media node: identity id by composition, name, state, serialized."""
    return BioCypherNode(
        node_id=identity_sha256(media_identity(media)),
        preferred_id="media",
        node_label="media",
        properties={
            "name": media.name,
            "state": media.state,
            "serialized_data": json.dumps(media.model_dump()),
        },
    )


def temperature_node(temperature: s.Temperature) -> BioCypherNode:
    """The temperature node: identity id, value, typed unit, serialized."""
    return BioCypherNode(
        node_id=identity_sha256(temperature_identity(temperature)),
        preferred_id="temperature",
        node_label="temperature",
        properties={
            "value": temperature.value,
            "unit": temperature.unit,
            "serialized_data": json.dumps(temperature.model_dump()),
        },
    )


def perturbation_node(perturbation: Any) -> BioCypherNode:
    """One gene-perturbation node; every SGA leaf declares ``strain_id``."""
    return BioCypherNode(
        node_id=sha(perturbation),
        preferred_id=perturbation.perturbation_type,
        node_label="perturbation",
        properties={
            "systematic_gene_name": perturbation.systematic_gene_name,
            "perturbed_gene_name": perturbation.perturbed_gene_name,
            "perturbation_type": perturbation.perturbation_type,
            "description": perturbation.description,
            "strain_id": perturbation.strain_id,
        },
    )


ENVIRONMENT_POINTER_MIN_BYTES = 512
GENOTYPE_POINTER_MIN_BYTES = 8192


def experiment_nodes(experiment: Any) -> list[BioCypherNode]:
    """The experiment method's output for one record: its Experiment node, then the
    environment as an ``interned constant``.

    The Experiment id is the sha256 of the fully inlined dump; its blob replaces the
    environment with ``{"$ref": <sha256 of the environment JSON>, "kind":
    "environment"}``. The genotype stays inline (asserted under its floor).
    """
    dump = experiment.model_dump()
    environment_payload = json.dumps(dump["environment"])
    assert len(environment_payload) >= ENVIRONMENT_POINTER_MIN_BYTES
    assert len(json.dumps(dump["genotype"])) < GENOTYPE_POINTER_MIN_BYTES
    ref = hashlib.sha256(environment_payload.encode("utf-8")).hexdigest()
    pointered = {**dump, "environment": {"$ref": ref, "kind": "environment"}}
    return [
        BioCypherNode(
            node_id=sha(experiment),
            preferred_id="experiment",
            node_label="experiment",
            properties={"serialized_data": json.dumps(pointered)},
        ),
        BioCypherNode(
            node_id=ref,
            preferred_id="interned constant",
            node_label="interned constant",
            properties={"kind": "environment", "serialized_data": environment_payload},
        ),
    ]


def expected_nodes(
    dataset_name: str,
    experiments: list[Any],
    reference: Any,
    publication: s.Publication,
) -> list[BioCypherNode]:
    """The exact node list an SGA adapter conf yields over ``experiments``.

    Conf order (the registration table restricted to the SGA confs): experiment
    reference, genome, experiment, genotype, perturbation, environment, environment
    reference, media, media reference, temperature, temperature reference, phenotype,
    phenotype reference, dataset, publication. Chunked methods emit one node per record
    (so the shared environment, its interned constant, media, temperature and
    publication repeat once per record; the sink dedups them by id); the reference
    collectors emit one node per distinct id.
    """
    environment = experiments[0].environment
    reference_environment = reference.environment_reference
    nodes: list[BioCypherNode] = [
        BioCypherNode(
            node_id=sha(reference),
            preferred_id="experiment reference",
            node_label="experiment reference",
            properties={"serialized_data": json.dumps(reference.model_dump())},
        ),
        BioCypherNode(
            node_id=sha(GENOME),
            preferred_id="genome",
            node_label="genome",
            properties={
                "species": GENOME.species,
                "strain": GENOME.strain,
                "serialized_data": json.dumps(GENOME.model_dump()),
            },
        ),
    ]
    for experiment in experiments:
        nodes += experiment_nodes(experiment)
    for experiment in experiments:
        genotype = experiment.genotype
        ordered = sorted(genotype.perturbations, key=lambda p: p.systematic_gene_name)
        nodes.append(
            BioCypherNode(
                node_id=sha(genotype),
                preferred_id="genotype",
                node_label="genotype",
                properties={
                    "systematic_gene_names": [p.systematic_gene_name for p in ordered],
                    "perturbed_gene_names": [p.perturbed_gene_name for p in ordered],
                    "perturbation_types": [p.perturbation_type for p in ordered],
                },
            )
        )
    for experiment in experiments:
        nodes += [perturbation_node(p) for p in experiment.genotype.perturbations]
    nodes += [environment_node(experiment.environment) for experiment in experiments]
    nodes.append(environment_node(reference_environment))
    nodes += [media_node(experiment.environment.media) for experiment in experiments]
    nodes.append(media_node(reference_environment.media))
    nodes += [
        temperature_node(experiment.environment.temperature)
        for experiment in experiments
    ]
    nodes.append(temperature_node(reference_environment.temperature))
    for experiment in experiments:
        phenotype = experiment.phenotype
        nodes.append(
            BioCypherNode(
                node_id=sha(phenotype),
                preferred_id=f"phenotype_{sha(phenotype)}",
                node_label=_phenotype_label(phenotype),
                properties=_phenotype_props(phenotype),
            )
        )
    reference_phenotype = reference.phenotype_reference
    nodes.append(
        BioCypherNode(
            node_id=sha(reference_phenotype),
            preferred_id=_phenotype_label(reference_phenotype),
            node_label=_phenotype_label(reference_phenotype),
            properties=_phenotype_props(reference_phenotype),
        )
    )
    nodes.append(
        BioCypherNode(
            node_id=dataset_name, preferred_id=dataset_name, node_label="dataset"
        )
    )
    nodes += [
        BioCypherNode(
            node_id=sha(publication),
            preferred_id=f"publication_{publication.pubmed_id}",
            node_label="publication",
            properties={
                "pubmed_id": publication.pubmed_id,
                "pubmed_url": publication.pubmed_url,
                "doi": publication.doi,
                "doi_url": publication.doi_url,
                "serialized_data": json.dumps(publication.model_dump()),
            },
        )
        for _ in experiments
    ]
    assert environment == reference_environment
    return nodes


def expected_edges(
    dataset_name: str,
    experiments: list[Any],
    reference: Any,
    publication: s.Publication,
) -> list[BioCypherEdge]:
    """The exact edge list an SGA adapter conf yields over ``experiments``.

    Conf order: experiment reference to dataset, experiment to dataset, experiment
    reference to experiment, genotype to experiment, perturbation to genotype,
    environment to experiment, environment to experiment reference, phenotype to
    experiment, media to environment, temperature to environment, genome to experiment
    reference, phenotype to experiment reference, publication to experiment.
    """
    reference_id = sha(reference)
    environment = reference.environment_reference
    environment_id = identity_sha256(environment_identity(environment))
    media_id = identity_sha256(media_identity(environment.media))
    temperature_id = identity_sha256(temperature_identity(environment.temperature))
    experiment_ids = [sha(experiment) for experiment in experiments]
    edges: list[BioCypherEdge] = [
        BioCypherEdge(
            source_id=reference_id,
            target_id=dataset_name,
            relationship_label="experiment reference member of",
        )
    ]
    edges += [
        BioCypherEdge(
            source_id=experiment_id,
            target_id=dataset_name,
            relationship_label="experiment member of",
        )
        for experiment_id in experiment_ids
    ]
    edges += [
        BioCypherEdge(
            source_id=reference_id,
            target_id=experiment_id,
            relationship_label="experiment reference of",
        )
        for experiment_id in experiment_ids
    ]
    edges += [
        BioCypherEdge(
            source_id=sha(experiment.genotype),
            target_id=experiment_id,
            relationship_label="genotype member of",
        )
        for experiment, experiment_id in zip(experiments, experiment_ids, strict=True)
    ]
    for experiment in experiments:
        genotype_id = sha(experiment.genotype)
        edges += [
            BioCypherEdge(
                source_id=sha(perturbation),
                target_id=genotype_id,
                relationship_label="perturbation member of",
            )
            for perturbation in experiment.genotype.perturbations
        ]
    edges += [
        BioCypherEdge(
            source_id=environment_id,
            target_id=experiment_id,
            relationship_label="environment member of",
        )
        for experiment_id in experiment_ids
    ]
    edges.append(
        BioCypherEdge(
            source_id=environment_id,
            target_id=reference_id,
            relationship_label="environment member of",
        )
    )
    edges += [
        BioCypherEdge(
            source_id=sha(experiment.phenotype),
            target_id=experiment_id,
            relationship_label="phenotype member of",
        )
        for experiment, experiment_id in zip(experiments, experiment_ids, strict=True)
    ]
    edges += [
        BioCypherEdge(
            source_id=media_id,
            target_id=environment_id,
            relationship_label="media member of",
        )
        for _ in experiments
    ]
    edges += [
        BioCypherEdge(
            source_id=temperature_id,
            target_id=environment_id,
            relationship_label="temperature member of",
        )
        for _ in experiments
    ]
    edges.append(
        BioCypherEdge(
            source_id=sha(GENOME),
            target_id=reference_id,
            relationship_label="genome member of",
        )
    )
    edges.append(
        BioCypherEdge(
            source_id=sha(reference.phenotype_reference),
            target_id=reference_id,
            relationship_label="phenotype member of",
        )
    )
    edges += [
        BioCypherEdge(
            source_id=sha(publication),
            target_id=experiment_id,
            relationship_label="mentions",
        )
        for experiment_id in experiment_ids
    ]
    return edges


# The method lists the SGA confs enable, in conf order. ``FITNESS_*`` is the smf/dmf/tmf
# conf shared verbatim by Kuzmin 2018, Kuzmin 2020 and Costanzo 2016 (Costanzo dmf adds
# memory_reduction_factor 0.5 to its chunked edges and its publication node);
# ``INTERACTION_*`` swaps the two fitness phenotype methods for gene interaction.
FITNESS_NODE_METHODS = [
    "experiment reference",
    "genome",
    "experiment (chunked)",
    "genotype (chunked)",
    "perturbation (chunked)",
    "environment (chunked)",
    "environment reference",
    "media (chunked)",
    "media reference",
    "temperature (chunked)",
    "temperature reference",
    "fitness phenotype (chunked)",
    "fitness phenotype reference",
    "dataset",
    "publication (chunked)",
]
INTERACTION_NODE_METHODS = [
    name.replace("fitness phenotype", "gene interaction phenotype")
    for name in FITNESS_NODE_METHODS
]
EDGE_METHODS = [
    "experiment reference to dataset",
    "experiment to dataset (chunked)",
    "experiment reference to experiment (chunked)",
    "genotype to experiment (chunked)",
    "perturbation to genotype (chunked)",
    "environment to experiment (chunked)",
    "environment to experiment reference",
    "phenotype to experiment (chunked)",
    "media to environment (chunked)",
    "temperature to environment (chunked)",
    "genome to experiment reference",
    "phenotype to experiment reference",
    "publication to experiment (chunked)",
]


def conf_method_names(adapter: Any) -> tuple[list[str], list[str]]:
    """The node and edge method names an adapter's loaded conf enables, in conf order."""
    conf = adapter.config.cell_adapter
    return (
        [entry["method_name"] for entry in conf.node_methods],
        [entry["method_name"] for entry in conf.edge_methods],
    )


def expected_events(
    node_methods: list[str], edge_methods: list[str]
) -> list[dict[str, Any]]:
    """The ``wandb.log`` payloads ``get_nodes`` then ``get_edges`` emit: one per enabled
    method in registration order, ``event`` counting from 1 across both passes.
    """
    events: list[dict[str, Any]] = [
        {"event": i + 1, "method": method, "type": "node"}
        for i, method in enumerate(node_methods)
    ]
    offset = len(node_methods)
    events += [
        {"event": offset + i + 1, "method": method, "type": "edge"}
        for i, method in enumerate(edge_methods)
    ]
    return events


def assert_method_table(
    table: Any,
    node_methods: list[str],
    edge_methods: list[str],
    node_factor: Callable[[str], float],
    edge_factor: Callable[[str], float],
) -> None:
    """The constructor's method table: columns ``event, method, data_type,
    memory_reduction_factor`` and one row per conf method in conf order (nodes then
    edges, events from 1); a chunked method's factor is its configured value or 1.0,
    an unchunked method's is NaN.
    """
    assert table.columns == ["event", "method", "data_type", "memory_reduction_factor"]
    expected = [(m, "node", node_factor(m)) for m in node_methods]
    expected += [(m, "edge", edge_factor(m)) for m in edge_methods]
    assert len(table.data) == len(expected)
    for i, (row, (method, kind, factor)) in enumerate(
        zip(table.data, expected, strict=True)
    ):
        assert row[:3] == [i + 1, method, kind], row
        if "(chunked)" in method:
            assert row[3] == factor, row
        else:
            assert math.isnan(row[3]), row
