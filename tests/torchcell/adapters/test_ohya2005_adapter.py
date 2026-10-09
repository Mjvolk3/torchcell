# tests/torchcell/adapters/test_ohya2005_adapter.py
# [[tests.torchcell.adapters.test_ohya2005_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_ohya2005_adapter.py
"""The Ohya 2005 adapter end to end: its own conf, on a REAL two-record CalMorph store.

``ScmdOhya2005Adapter`` adds no builder of its own; what it adds is the conf
``conf/scmd_ohya2005_adapter.yaml`` (which of the ``CellAdapter`` methods run) and the
constructor wiring. So the fixture is the real ``ScmdOhya2005Dataset`` built hermetically
in ``tmp_path`` from two synthetic SCMD matrices (the ``test_ohya2005`` recipe: both TSVs
in ``raw/`` so ``download()`` never runs, a genome stub whose ``resolve_gene_name`` says
every ORF is current), and the adapter's full ``get_nodes`` / ``get_edges`` output is
compared, node by node and edge by edge, with ``BioCypherNode`` / ``BioCypherEdge``
objects built by hand from hand-built schema objects.

Features (4 of the 501): base ``A101_A``, ``C103_A1B``; CV ``ACV103_A1B``, ``CCV103_A1B``.

``mt4718data.tsv``: YAL001C 1.0 2.0 0.5 0.25; YBR001C 3.0 4.0 0.25 0.5.
``wt122data.tsv``: wt1 1.0 3.0 0.25 0.5; wt2 3.0 5.0 0.75 1.0, so the ONE shared
reference phenotype is the WT mean: A101_A (1 + 3) / 2 = 2.0, C103_A1B (3 + 5) / 2 = 4.0,
ACV103_A1B (0.25 + 0.75) / 2 = 0.5, CCV103_A1B (0.5 + 1.0) / 2 = 0.75.

Ids: every record-side id is ``sha256(json.dumps(model_dump()))`` (``cell_adapter.py``
line 527 for the experiment, the same formula for genome, genotype, perturbation,
phenotype, reference and publication); environment, media and temperature ids are
``identity_sha256`` of the identity projection. The genome and temperature ids are also
pinned against their literal JSON, so a schema field reorder (which silently re-keys the
served graph) fails here too.

What the conf decides and this file pins: 15 node methods and 13 edge methods, in the
registration-table order; per-record (chunked) methods emit one node per record, so the
two records give duplicate environment, media, temperature and publication nodes (BioCypher
deduplicates on write); the CalMorph phenotype nodes carry the base and CV dictionaries
as JSON strings, with preferred id ``phenotype_<id>`` per record and ``calmorph phenotype``
for the reference.

Payloads: genotype, perturbation and phenotype nodes carry no ``serialized_data``;
experiment, experiment reference, genome, environment, media, temperature and
publication nodes keep it. Since ca0734254 (#622) the medium is the loader's sourced
``OHYA_YPD`` (the library ``YPD_LIQUID`` node, restated with the paper's quotes), so
the environment's JSON is 5200 bytes, over the 512-byte environment floor: each
Experiment blob points at it and the experiment method also emits it as one
``interned constant`` node per record. The one-deletion genotype is 395 bytes, under
the 8192-byte genotype floor (``EXPERIMENT_POINTER_MIN_BYTES``), and stays inline.

``wandb`` is replaced by a recorder bound to the name ``cell_adapter`` imported.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import os.path as osp
import pickle
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import lmdb
import pytest
from biocypher._create import BioCypherEdge, BioCypherNode

import torchcell.adapters.cell_adapter as cell_adapter_module
import torchcell.adapters.ohya2005_adapter as adapter_module
from torchcell.adapters.ohya2005_adapter import ScmdOhya2005Adapter
from torchcell.datamodels.identity import (
    environment_identity,
    identity_sha256,
    media_identity,
    temperature_identity,
)
from torchcell.datamodels.interned_constant import EXPERIMENT_POINTER_MIN_BYTES
from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae.ohya2005 import OHYA_YPD, ScmdOhya2005Dataset
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
)

DATASET = "ScmdOhya2005Dataset"
FEATURES = ["A101_A", "C103_A1B", "ACV103_A1B", "CCV103_A1B"]
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="BY4741")
MEDIA = OHYA_YPD
TEMPERATURE = Temperature(value=25)
ENVIRONMENT = Environment(media=MEDIA, temperature=TEMPERATURE)
PUBLICATION = Publication(
    pubmed_id="16365294",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/16365294/",
    doi="10.1073/pnas.0509436102",
    doi_url="https://www.pnas.org/doi/10.1073/pnas.0509436102",
)
WT_PHENOTYPE = CalMorphPhenotype(
    calmorph={"A101_A": 2.0, "C103_A1B": 4.0},
    calmorph_coefficient_of_variation={"ACV103_A1B": 0.5, "CCV103_A1B": 0.75},
)
REFERENCE = CalMorphExperimentReference(
    dataset_name=DATASET,
    genome_reference=GENOME,
    environment_reference=ENVIRONMENT,
    phenotype_reference=WT_PHENOTYPE,
)


def _deletion(orf: str) -> KanMxDeletionPerturbation:
    return KanMxDeletionPerturbation(systematic_gene_name=orf, perturbed_gene_name=orf)


def _phenotype(values: tuple[float, float, float, float]) -> CalMorphPhenotype:
    return CalMorphPhenotype(
        calmorph={"A101_A": values[0], "C103_A1B": values[1]},
        calmorph_coefficient_of_variation={
            "ACV103_A1B": values[2],
            "CCV103_A1B": values[3],
        },
    )


def _experiment(
    orf: str, values: tuple[float, float, float, float]
) -> CalMorphExperiment:
    return CalMorphExperiment(
        dataset_name=DATASET,
        genotype=Genotype(perturbations=[_deletion(orf)]),
        environment=ENVIRONMENT,
        phenotype=_phenotype(values),
    )


RECORDS = [("YAL001C", (1.0, 2.0, 0.5, 0.25)), ("YBR001C", (3.0, 4.0, 0.25, 0.5))]
EXPERIMENTS = [_experiment(orf, values) for orf, values in RECORDS]

# The calmorph JSON strings, written out rather than dumped, per record then reference.
CALMORPH_JSON = [
    ('{"A101_A": 1.0, "C103_A1B": 2.0}', '{"ACV103_A1B": 0.5, "CCV103_A1B": 0.25}'),
    ('{"A101_A": 3.0, "C103_A1B": 4.0}', '{"ACV103_A1B": 0.25, "CCV103_A1B": 0.5}'),
    ('{"A101_A": 2.0, "C103_A1B": 4.0}', '{"ACV103_A1B": 0.5, "CCV103_A1B": 0.75}'),
]


def _sha(model: Any) -> str:
    """The adapter's content address: sha256 of the json-dumped ``model_dump``."""
    return hashlib.sha256(json.dumps(model.model_dump()).encode("utf-8")).hexdigest()


ENVIRONMENT_ID = identity_sha256(environment_identity(ENVIRONMENT))
MEDIA_ID = identity_sha256(media_identity(MEDIA))
TEMPERATURE_ID = identity_sha256(temperature_identity(TEMPERATURE))
REFERENCE_ID = _sha(REFERENCE)

# The environment's JSON, served as one interned constant since ca0734254 (#622) made
# it larger than the 512-byte pointer floor; its id is the sha256 of these bytes. Both
# the byte count and the id moved in the #753 wave, where ``Environment`` gained
# ``dilution_rate_per_hour``: a content address over the whole record is supposed to
# move when the record's shape does, which is the BREAKING verdict
# ``scripts/schema_impact_check.py`` reports for all 81 datasets. ``ENVIRONMENT_ID``
# ``ENVIRONMENT_ID`` moved for the same reason: ``environment_identity`` projects the
# dilution rate, because in a chemostat it is a controlled variable and two cultures
# differing only in it must be two environment nodes (Ishii 2007's recovered wild-type
# series is four such records). ``MEDIA_ID`` is UNCHANGED, which is the evidence the
# change stayed on the environment and did not disturb the medium-level join.
ENVIRONMENT_JSON = json.dumps(ENVIRONMENT.model_dump())
ENVIRONMENT_CONSTANT_ID = (
    "ee9652f19e630ed76c05e0b8ec524567f62e0e6ad7798b2fea6950e1c4745f98"
)


def _experiment_blob(experiment: CalMorphExperiment) -> str:
    """The Experiment node's ``serialized_data``: the record with its environment
    replaced by a ``{"$ref", "kind"}`` pointer to the interned constant.
    """
    dump = experiment.model_dump()
    dump["environment"] = {"$ref": ENVIRONMENT_CONSTANT_ID, "kind": "environment"}
    return json.dumps(dump)


# ------------------------------------------------------------------- fixtures


class _StubGenome:
    """Only ``resolve_gene_name``: every ORF is current under its uppercased name."""

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        upper = name.strip().upper()
        return GeneNameResolution(
            input_name=name, status=GeneNameStatus.CURRENT, systematic_name=upper
        )


class _Table:
    def __init__(self, columns: list[str], data: list[list[Any]]) -> None:
        self.columns = columns
        self.data = data


class _WandbRecorder:
    Table = _Table

    def __init__(self) -> None:
        self.init_calls = 0
        self.logged: list[dict[str, Any]] = []

    def init(self) -> None:
        self.init_calls += 1

    def log(self, payload: dict[str, Any]) -> None:
        self.logged.append(payload)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _WandbRecorder:
    rec = _WandbRecorder()
    monkeypatch.setattr(cell_adapter_module, "wandb", rec)
    return rec


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No inherited ``TC_DATA_URL`` (it would send the build to tc-data), and undo the
    ``gc.freeze()`` the adapter and its loader apply.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    yield
    gc.unfreeze()


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text("\n".join("\t".join(r) for r in [header, *rows]) + "\n")


def _build(tmp_path: Path) -> Path:
    root = tmp_path / "scmd_ohya2005"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / "mt4718data.tsv",
        ["ORF", *FEATURES],
        [[orf, *(str(v) for v in values)] for orf, values in RECORDS],
    )
    _write_tsv(
        root / "raw" / "wt122data.tsv",
        ["NAME", *FEATURES],
        [["wt1", "1.0", "3.0", "0.25", "0.5"], ["wt2", "3.0", "5.0", "0.75", "1.0"]],
    )
    return root


def _dataset(root: Path) -> ScmdOhya2005Dataset:
    return ScmdOhya2005Dataset(
        root=str(root), genome=cast(SCerevisiaeGenome, _StubGenome())
    )


def _adapter(dataset: ScmdOhya2005Dataset) -> ScmdOhya2005Adapter:
    return ScmdOhya2005Adapter(
        dataset=dataset,
        process_workers=1,
        io_workers=1,
        chunk_size=2,
        loader_batch_size=2,
    )


# ------------------------------------------------------------ expected graph


def _serialized(model: Any) -> dict[str, Any]:
    return {"serialized_data": json.dumps(model.model_dump())}


def _calmorph_node(
    phenotype: CalMorphPhenotype, json_pair: tuple[str, str], preferred: str | None
) -> BioCypherNode:
    phenotype_id = _sha(phenotype)
    return BioCypherNode(
        node_id=phenotype_id,
        preferred_id=preferred or f"phenotype_{phenotype_id}",
        node_label="calmorph phenotype",
        properties={
            "graph_level": "global",
            "label_name": "calmorph",
            "label_statistic_name": "calmorph_coefficient_of_variation",
            "calmorph": json_pair[0],
            "calmorph_coefficient_of_variation": json_pair[1],
        },
    )


def _expected_nodes() -> list[BioCypherNode]:
    environment = BioCypherNode(
        node_id=ENVIRONMENT_ID,
        preferred_id="environment",
        node_label="environment",
        properties={
            "temperature": 25.0,
            "media": json.dumps(MEDIA.model_dump()),
            **_serialized(ENVIRONMENT),
        },
    )
    media = BioCypherNode(
        node_id=MEDIA_ID,
        preferred_id="media",
        node_label="media",
        properties={
            "name": "YPD (yeast extract / peptone / dextrose), liquid",
            "state": "liquid",
            **_serialized(MEDIA),
        },
    )
    temperature = BioCypherNode(
        node_id=TEMPERATURE_ID,
        preferred_id="temperature",
        node_label="temperature",
        properties={"value": 25.0, "unit": "Celsius", **_serialized(TEMPERATURE)},
    )
    publication = BioCypherNode(
        node_id=_sha(PUBLICATION),
        preferred_id="publication_16365294",
        node_label="publication",
        properties={
            "pubmed_id": "16365294",
            "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/16365294/",
            "doi": "10.1073/pnas.0509436102",
            "doi_url": "https://www.pnas.org/doi/10.1073/pnas.0509436102",
            "source_type": "journal_article",
            "title": None,
            "identifier": None,
            "identifier_url": None,
            **_serialized(PUBLICATION),
        },
    )
    return [
        BioCypherNode(
            node_id=REFERENCE_ID,
            preferred_id="experiment reference",
            node_label="experiment reference",
            properties=_serialized(REFERENCE),
        ),
        BioCypherNode(
            node_id=_sha(GENOME),
            preferred_id="genome",
            node_label="genome",
            properties={
                "species": "Saccharomyces cerevisiae",
                "strain": "BY4741",
                **_serialized(GENOME),
            },
        ),
        *[
            node
            for experiment in EXPERIMENTS
            for node in (
                BioCypherNode(
                    node_id=_sha(experiment),
                    preferred_id="experiment",
                    node_label="experiment",
                    properties={"serialized_data": _experiment_blob(experiment)},
                ),
                # once per record; the sink dedups by id
                BioCypherNode(
                    node_id=ENVIRONMENT_CONSTANT_ID,
                    preferred_id="interned constant",
                    node_label="interned constant",
                    properties={
                        "kind": "environment",
                        "serialized_data": ENVIRONMENT_JSON,
                    },
                ),
            )
        ],
        *[
            BioCypherNode(
                node_id=_sha(experiment.genotype),
                preferred_id="genotype",
                node_label="genotype",
                properties={
                    "systematic_gene_names": [orf],
                    "perturbed_gene_names": [orf],
                    "perturbation_types": ["kanmx_deletion"],
                },
            )
            for (orf, _), experiment in zip(RECORDS, EXPERIMENTS, strict=True)
        ],
        *[
            BioCypherNode(
                node_id=_sha(_deletion(orf)),
                preferred_id="kanmx_deletion",
                node_label="perturbation",
                properties={
                    "systematic_gene_name": orf,
                    "perturbed_gene_name": orf,
                    "perturbation_type": "kanmx_deletion",
                    "description": "Deletion via KanMX or NatMX gene replacement",
                    "strain_id": None,
                },
            )
            for orf, _ in RECORDS
        ],
        environment,
        environment,
        environment,  # two per-record nodes, then the reference collector's one
        media,
        media,
        media,
        temperature,
        temperature,
        temperature,
        _calmorph_node(EXPERIMENTS[0].phenotype, CALMORPH_JSON[0], None),
        _calmorph_node(EXPERIMENTS[1].phenotype, CALMORPH_JSON[1], None),
        _calmorph_node(WT_PHENOTYPE, CALMORPH_JSON[2], "calmorph phenotype"),
        BioCypherNode(node_id=DATASET, preferred_id=DATASET, node_label="dataset"),
        publication,
        publication,
    ]


def _edge(source: str, target: str, label: str) -> BioCypherEdge:
    return BioCypherEdge(source_id=source, target_id=target, relationship_label=label)


def _expected_edges() -> list[BioCypherEdge]:
    exp = [_sha(experiment) for experiment in EXPERIMENTS]
    genotypes = [_sha(experiment.genotype) for experiment in EXPERIMENTS]
    return [
        _edge(REFERENCE_ID, DATASET, "experiment reference member of"),
        *[_edge(e, DATASET, "experiment member of") for e in exp],
        *[_edge(REFERENCE_ID, e, "experiment reference of") for e in exp],
        *[
            _edge(g, e, "genotype member of")
            for g, e in zip(genotypes, exp, strict=True)
        ],
        *[
            _edge(_sha(_deletion(orf)), g, "perturbation member of")
            for (orf, _), g in zip(RECORDS, genotypes, strict=True)
        ],
        *[_edge(ENVIRONMENT_ID, e, "environment member of") for e in exp],
        _edge(ENVIRONMENT_ID, REFERENCE_ID, "environment member of"),
        *[
            _edge(_sha(experiment.phenotype), e, "phenotype member of")
            for experiment, e in zip(EXPERIMENTS, exp, strict=True)
        ],
        _edge(MEDIA_ID, ENVIRONMENT_ID, "media member of"),
        _edge(MEDIA_ID, ENVIRONMENT_ID, "media member of"),
        _edge(TEMPERATURE_ID, ENVIRONMENT_ID, "temperature member of"),
        _edge(TEMPERATURE_ID, ENVIRONMENT_ID, "temperature member of"),
        _edge(_sha(GENOME), REFERENCE_ID, "genome member of"),
        _edge(_sha(WT_PHENOTYPE), REFERENCE_ID, "phenotype member of"),
        *[_edge(_sha(PUBLICATION), e, "mentions") for e in exp],
    ]


NODE_METHODS = [
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
    "calmorph phenotype (chunked)",
    "calmorph phenotype reference",
    "dataset",
    "publication (chunked)",
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


# ---------------------------------------------------------------------- tests


def test_ids_are_the_sha256_of_the_literal_json() -> None:
    """The genome and temperature ids hash these exact strings: a field added,
    renamed or reordered in ``ReferenceGenome`` / ``Temperature`` changes the id every
    served Ohya node is keyed on, and this is the test that notices.
    """
    genome_json = (
        '{"species": "Saccharomyces cerevisiae", "strain": "BY4741", '
        '"ploidy": "haploid"}'
    )
    assert _sha(GENOME) == hashlib.sha256(genome_json.encode()).hexdigest()
    assert TEMPERATURE_ID == identity_sha256({"value": 25.0, "unit": "Celsius"})
    assert json.dumps(TEMPERATURE.model_dump()) == '{"value": 25.0, "unit": "Celsius"}'


def test_get_nodes_emits_the_exact_ohya_node_list(
    tmp_path: Path, recorder: _WandbRecorder
) -> None:
    """Two records, one shared reference: 25 nodes, in the registration-table order,
    compared by value (id, label, preferred id and every property). The event log names
    the 15 node methods the conf enables, which is the conf's whole contract.

    The medium is the loader's sourced ``OHYA_YPD`` since ca0734254 (#622): library
    ``YPD_LIQUID`` (1% yeast extract, 2% peptone, 2% D-glucose, w/v) restated with Ohya
    2005's growth sentence and the Ohya-lab recipe Ohnuki 2018 attributes to it, in
    place of the componentless ``Media(name="YPD")`` stub. Its media node keeps
    ``YPD_LIQUID``'s identity, so it is the library YPD node, not a new one.
    """
    assert MEDIA_ID == identity_sha256(media_identity(YPD_LIQUID))
    assert MEDIA_ID == (
        "aea23796700e8b3a95fce85defa3c999208717499a582a5b7862db7f3f605232"
    )
    assert ENVIRONMENT_ID == (
        "9473b25614c31cc2deae2f61d1d6964c5d08e8a0bbebb35d366fa0447350f591"
    )
    # The sourced medium takes the environment past the Neo4j pointer floor, so each
    # Experiment node points at one ``interned constant`` node (emitted per record,
    # deduplicated by the sink: 23 nodes before #622, 25 now); the genotype is not.
    assert (
        ENVIRONMENT_CONSTANT_ID == hashlib.sha256(ENVIRONMENT_JSON.encode()).hexdigest()
    )
    assert len(json.dumps(ENVIRONMENT.model_dump())) == 5200
    assert EXPERIMENT_POINTER_MIN_BYTES["environment"] == 512
    genotypes = [e.genotype for e in EXPERIMENTS]
    assert all(isinstance(g, Genotype) for g in genotypes)
    assert [len(json.dumps(cast(Genotype, g).model_dump())) for g in genotypes] == [
        395,
        395,
    ]
    assert EXPERIMENT_POINTER_MIN_BYTES["genotype"] == 8192
    adapter = _adapter(_dataset(_build(tmp_path)))
    recorder.logged.clear()
    nodes = list(adapter.get_nodes())
    assert nodes == _expected_nodes()
    assert [entry["method"] for entry in recorder.logged] == NODE_METHODS
    assert [entry["event"] for entry in recorder.logged] == list(range(1, 16))


def test_get_edges_emits_the_exact_ohya_edge_list(
    tmp_path: Path, recorder: _WandbRecorder
) -> None:
    """22 edges from the 13 edge methods the conf enables, part to whole, in table
    order; per-record media and temperature edges repeat once per record. The media
    endpoint is the library YPD node of the sourced ``OHYA_YPD`` medium (ca0734254,
    #622; see the node test).
    """
    adapter = _adapter(_dataset(_build(tmp_path)))
    recorder.logged.clear()
    edges = list(adapter.get_edges())
    assert edges == _expected_edges()
    assert [entry["method"] for entry in recorder.logged] == EDGE_METHODS


def test_the_loader_interns_the_reference_and_the_graph_is_unchanged(
    tmp_path: Path, recorder: _WandbRecorder
) -> None:
    """Contract (issue #546): ``ScmdOhya2005Dataset.process`` writes through the base
    writer ``_intern_record``, as the thirteen other interning loaders do. In the built
    store the reference (>= 512 bytes of canonical JSON) is a ``{"$ref", "name"}``
    pointer whose body sits in the sibling ``interned`` env. Since ca0734254 (#622)
    the environment carries the sourced ``OHYA_YPD`` medium and is interned too (named
    by its medium); the publication stays inline. ``get_single_item`` splices the
    pointers back, so the adapter's node and edge lists are exactly the hand-built
    graph.
    """
    root = _build(tmp_path)
    records_path = osp.join(root, "processed", "lmdb")
    _dataset(root).close_lmdb()
    raw_env = lmdb.open(records_path, readonly=True, lock=False)
    with raw_env.begin() as txn:
        raw = txn.get(b"1")
        assert raw is not None, "the record is missing from the store"
        stored = pickle.loads(raw)
    raw_env.close()
    reference_digest = hashlib.sha256(
        json.dumps(REFERENCE.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()
    assert stored["reference"] == {"$ref": reference_digest, "name": DATASET}
    environment_digest = hashlib.sha256(
        json.dumps(ENVIRONMENT.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()
    assert stored["experiment"]["environment"] == {
        "$ref": environment_digest,
        "name": "YPD (yeast extract / peptone / dextrose), liquid",
    }
    assert stored["publication"] == PUBLICATION.model_dump()

    adapter = _adapter(_dataset(root))
    assert list(adapter.get_nodes()) == _expected_nodes()
    assert list(adapter.get_edges()) == _expected_edges()


def test_constructor_passes_chunk_and_batch_sizes_in_order(
    tmp_path: Path, recorder: _WandbRecorder
) -> None:
    """The subclass forwards its sizes positionally, so a swap would invert the
    base's guard: loader batch 3 over chunk 2 is refused with the base message, while
    chunk 3 over batch 2 constructs and logs the conf's 28-row method table (factor 1.0
    on each chunked method, NaN on each collector) under ``<dataset>_method_table``.
    """
    dataset = _dataset(_build(tmp_path))
    with pytest.raises(ValueError) as refused:
        ScmdOhya2005Adapter(
            dataset=dataset,
            process_workers=1,
            io_workers=1,
            chunk_size=2,
            loader_batch_size=3,
        )
    assert str(refused.value) == (
        "chunk_size must be greater than or equal to loader_batch_size."
        "Our recommendation are chunk_size 2-3 order of magnitude in size."
    )
    assert recorder.init_calls == 0

    adapter = ScmdOhya2005Adapter(
        dataset=dataset,
        process_workers=4,
        io_workers=2,
        chunk_size=3,
        loader_batch_size=2,
    )
    assert (adapter.process_workers, adapter.io_workers) == (4, 2)
    assert (adapter.chunk_size, adapter.loader_batch_size) == (3, 2)
    assert recorder.init_calls == 1
    table = recorder.logged[0][f"{DATASET}_method_table"]
    rows = [
        (row[0], row[1], row[2], "nan" if row[3] != row[3] else row[3])
        for row in table.data
    ]
    expected_rows = [
        (i + 1, name, kind, 1.0 if "(chunked)" in name else "nan")
        for i, (name, kind) in enumerate(
            [(n, "node") for n in NODE_METHODS] + [(e, "edge") for e in EDGE_METHODS]
        )
    ]
    assert rows == expected_rows


def test_a_missing_conf_is_refused_before_wandb_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _WandbRecorder
) -> None:
    """The conf path is resolved from the module's ``__file__`` at call time; a module
    relocated without its ``conf/`` directory raises before the base constructor runs.
    """
    monkeypatch.setattr(adapter_module, "__file__", str(tmp_path / "moved.py"))
    with pytest.raises(FileNotFoundError) as refused:
        ScmdOhya2005Adapter(dataset=cast(Any, None), process_workers=1, io_workers=1)
    assert str(refused.value) == (
        f"Config file not found: {tmp_path}/conf/scmd_ohya2005_adapter.yaml"
    )
    assert recorder.init_calls == 0


# ----------------------------------------------------------------------- main


class _FakeBioCypher:
    instances: list[_FakeBioCypher] = []
    calls: list[tuple[Any, ...]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _FakeBioCypher.instances.append(self)

    def write_nodes(self, nodes: Any) -> None:
        _FakeBioCypher.calls.append(("write_nodes", list(nodes)))

    def write_edges(self, edges: Any) -> None:
        _FakeBioCypher.calls.append(("write_edges", list(edges)))

    def write_import_call(self) -> None:
        _FakeBioCypher.calls.append(("write_import_call",))

    def write_schema_info(self, as_node: bool) -> None:
        _FakeBioCypher.calls.append(("write_schema_info", as_node))


class _FakeDataset:
    instances: list[_FakeDataset] = []

    def __init__(self, root: str) -> None:
        self.root = root
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


def test_main_builds_the_store_and_writes_everything_then_finishes_wandb(  # test-quality: allow main() returns None; its effects are asserted on the recorders
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``main`` loads the dotenv once, opens BioCypher under
    ``$DATA_ROOT/database/biocypher-out/<YYYY-mm-dd_HH-MM-SS>``, builds the dataset at
    ``data/torchcell/scmd_ohya2005``, splits 7 CPUs into ceil(0.2 * 7) = ceil(1.4) = 2
    io workers and 7 - 2 = 5 process workers (a floor or a round would give 1 and 6),
    wraps it with chunk and loader batch both 10000, writes nodes, edges, the import call
    and the schema node, and finishes wandb last.
    """
    _FakeBioCypher.instances.clear()
    _FakeBioCypher.calls.clear()
    _FakeDataset.instances.clear()
    _FakeAdapter.instances.clear()
    dotenv_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "dotenv.load_dotenv", lambda *args, **kwargs: dotenv_calls.append(args)
    )
    monkeypatch.setattr(
        "wandb.finish", lambda: _FakeBioCypher.calls.append(("wandb.finish",))
    )
    monkeypatch.setattr("multiprocessing.cpu_count", lambda: 7)
    monkeypatch.setattr(adapter_module, "BioCypher", _FakeBioCypher)
    monkeypatch.setattr(adapter_module, "ScmdOhya2005Dataset", _FakeDataset)
    monkeypatch.setattr(adapter_module, "ScmdOhya2005Adapter", _FakeAdapter)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("BIOCYPHER_CONFIG_PATH", "bc-config.yaml")
    monkeypatch.setenv("SCHEMA_CONFIG_PATH", "schema-config.yaml")

    adapter_module.main()

    assert dotenv_calls == [()]
    (bc,) = _FakeBioCypher.instances
    out_dir = bc.kwargs.pop("output_directory")
    prefix = osp.join(str(tmp_path), "database/biocypher-out") + os.sep
    assert out_dir.startswith(prefix)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", out_dir[len(prefix) :])
    assert bc.kwargs == {
        "biocypher_config_path": "bc-config.yaml",
        "schema_config_path": "schema-config.yaml",
    }
    (dataset,) = _FakeDataset.instances
    assert dataset.root == osp.join(str(tmp_path), "data/torchcell/scmd_ohya2005")
    (adapter,) = _FakeAdapter.instances
    assert adapter.kwargs == {
        "dataset": dataset,
        "process_workers": 5,
        "io_workers": 2,
        "chunk_size": 10000,
        "loader_batch_size": 10000,
    }
    assert _FakeBioCypher.calls == [
        ("write_nodes", ["node-1", "node-2"]),
        ("write_edges", ["edge-1"]),
        ("write_import_call",),
        ("write_schema_info", True),
        ("wandb.finish",),
    ]
