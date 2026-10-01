# tests/torchcell/adapters/test_experiment_node_interned_constant.py
"""The Experiment blob's interned-constant pointers round-trip exactly.

``CellAdapter._experiment_node`` writes the record's large sub-objects (its
environment, a large genotype) as ``interned constant`` nodes and leaves pointers in
the Experiment blob; ``torchcell.data.neo4j_query_raw`` splices them back on query.
What is pinned: the Experiment id is still the sha256 of the inlined record, the
pointered blob is smaller than the inlined one, every pointed-to node's id is the
sha256 of its payload, resolving the pointers reproduces the inlined record byte for
byte (so the id can be recomputed from a query result), and a small sub-object stays
inline. The query-side batch writer is exercised with a stub store, so a missing or
corrupt constant is a hard error and never a silently dropped record.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from typing import Any, cast

import pytest

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw
from torchcell.datamodels import schema as s
from torchcell.datamodels.interned_constant import (
    EXPERIMENT_POINTER_MIN_BYTES,
    INTERNED_CONSTANT_LABEL,
    POINTER_KEY,
    collect_pointers,
    constant_id,
    resolve_pointers,
    split_experiment_dump,
    verified_constant,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

_SHA = "0" * 64


def _sourced(quote: str) -> SourcedValue:
    return SourcedValue(
        value="20 g/L",
        quote=quote,
        provenance=Provenance(
            source_uri="paper.md", citation_key="test2026", sha256=_SHA
        ),
    )


def _media(n_components: int) -> s.Media:
    return s.Media(
        name="YPD plates",
        state="solid",
        is_synthetic=False,
        base_medium="YPD",
        components=[
            s.MediaComponent(
                compound=s.Compound(name=f"component {i}"),
                role=s.MediaComponentRole.carbon_source,
                concentration=s.Concentration(
                    value=float(i + 1), unit=s.ConcentrationUnit.percent_w_v
                ),
                provenance=[_sourced(f"component {i} at {i + 1}% w/v in the plates")],
                note=f"component {i} stated by the methods",
            )
            for i in range(n_components)
        ],
        provenance=[_sourced("YPD plates")],
    )


def _experiment(n_components: int) -> s.FitnessExperiment:
    return s.FitnessExperiment(
        dataset_name="TestDataset",
        genotype=s.Genotype(
            perturbations=[
                s.SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAL001C",
                    perturbed_gene_name="TFC3",
                    strain_id="DMA1",
                )
            ]
        ),
        environment=s.Environment(
            media=_media(n_components), temperature=s.Temperature(value=30.0)
        ),
        phenotype=s.FitnessPhenotype(fitness=0.9, fitness_std=0.01),
    )


def _reference(experiment: s.FitnessExperiment) -> s.FitnessExperimentReference:
    return s.FitnessExperimentReference(
        dataset_name="TestDataset",
        genome_reference=s.ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=experiment.environment,
        phenotype_reference=s.FitnessPhenotype(fitness=1.0),
    )


def _emit(experiment: s.FitnessExperiment) -> list[Any]:
    body = cast(Any, CellAdapter._experiment_node).__wrapped__
    return cast(
        list[Any], body(None, {"experiment": experiment}, "experiment (chunked)")
    )


def test_large_environment_becomes_a_pointer_and_resolves_to_the_same_record() -> None:
    experiment = _experiment(n_components=8)
    inline = json.dumps(experiment.model_dump())
    assert (
        len(json.dumps(experiment.model_dump()["environment"]))
        >= (EXPERIMENT_POINTER_MIN_BYTES["environment"])
    )
    nodes = _emit(experiment)
    experiment_node, constants = nodes[0], nodes[1:]

    assert experiment_node.get_label() == "experiment"
    assert experiment_node.get_id() == hashlib.sha256(inline.encode()).hexdigest()
    blob = experiment_node.get_properties()["serialized_data"]
    assert len(blob) < len(inline) // 2
    assert json.loads(blob)["environment"][POINTER_KEY] == constants[0].get_id()

    assert [c.get_label() for c in constants] == [INTERNED_CONSTANT_LABEL]
    payload = constants[0].get_properties()["serialized_data"]
    assert constants[0].get_properties()["kind"] == "environment"
    assert constants[0].get_id() == constant_id(payload)

    store = {constants[0].get_id(): verified_constant(constants[0].get_id(), payload)}
    resolved = resolve_pointers(json.loads(blob), store)
    assert json.dumps(resolved) == inline


def test_small_genotype_and_phenotype_stay_inline() -> None:
    experiment = _experiment(n_components=8)
    dump = experiment.model_dump()
    assert len(json.dumps(dump["genotype"])) < EXPERIMENT_POINTER_MIN_BYTES["genotype"]
    nodes = _emit(experiment)
    blob = json.loads(nodes[0].get_properties()["serialized_data"])
    assert blob["genotype"] == dump["genotype"]
    assert blob["phenotype"] == dump["phenotype"]
    assert [c.get_properties()["kind"] for c in nodes[1:]] == ["environment"]


def test_split_does_not_mutate_the_dump_and_collects_every_pointer() -> None:
    dump = _experiment(n_components=8).model_dump()
    before = json.dumps(dump)
    pointered, constants = split_experiment_dump(dump)
    assert json.dumps(dump) == before
    refs: set[str] = set()
    collect_pointers(pointered, refs)
    assert refs == {ref for ref, _, _ in constants}


def test_verified_constant_rejects_a_payload_that_does_not_hash_to_its_id() -> None:
    with pytest.raises(ValueError, match="hashing to"):
        verified_constant(constant_id('{"a": 1}'), '{"a": 2}')


class _StoreQueryRaw(Neo4jQueryRaw):
    """``Neo4jQueryRaw`` over an in-memory constant store instead of a bolt connection."""

    def __attrs_post_init__(self) -> None:
        self.raw_dir = osp.join(self.root_dir, "raw")
        self.lmdb_dir = osp.join(self.raw_dir, "lmdb")
        os.makedirs(self.lmdb_dir, exist_ok=True)
        self._init_lmdb(readonly=False)
        self.store: dict[str, str] = {}
        self.fetched: list[list[str]] = []

    def fetch_constants(self, refs: list[str]) -> dict[str, Any]:
        self.fetched.append(list(refs))
        missing = [r for r in refs if r not in self.store]
        if missing:
            raise KeyError(f"{len(missing)} interned constants are missing")
        return {r: verified_constant(r, self.store[r]) for r in refs}


def _query(tmp_path: Any) -> _StoreQueryRaw:
    return _StoreQueryRaw(
        uri="bolt://none", username="", password="", root_dir=str(tmp_path), query=""
    )


def test_query_side_batch_resolves_pointers_once_per_id(tmp_path: Any) -> None:
    experiments = [_experiment(n_components=8) for _ in range(3)]
    emitted = [_emit(e) for e in experiments]
    store = {
        c.get_id(): c.get_properties()["serialized_data"]
        for nodes in emitted
        for c in nodes[1:]
    }
    assert len(store) == 1  # three records, one shared environment
    reference = _reference(experiments[0])
    query = _query(tmp_path)
    query.store = store
    batch = [
        (
            i,
            json.loads(nodes[0].get_properties()["serialized_data"]),
            json.dumps(reference.model_dump()),
        )
        for i, nodes in enumerate(emitted)
    ]
    constants: dict[str, Any] = {}
    query._write_batch(batch, constants)
    query._write_batch(batch, constants)  # a second batch fetches nothing new
    assert query.fetched == [list(store)]
    query.close_lmdb()
    for i, experiment in enumerate(experiments):
        record = query[i]
        assert record["experiment"] == experiment
        assert record["experiment_reference"] == reference


def test_query_side_batch_fails_on_a_missing_constant(tmp_path: Any) -> None:
    nodes = _emit(_experiment(n_components=8))
    query = _query(tmp_path)
    blob = json.loads(nodes[0].get_properties()["serialized_data"])
    with pytest.raises(KeyError, match="missing"):
        query._write_batch([(0, blob, "{}")], {})
