# tests/torchcell/data/test_neo4j_query_raw_single_pass.py
"""The single-pass raw stage writes exactly what the per-record path wrote.

``Neo4jQueryRaw.process`` validates only the experiment per record, takes the
environment and reference from caches of validated models, splices their cached JSON
into the written value, and computes the reference index and gene set while streaming
instead of re-reading the LMDB twice. These tests keep the previous implementation
(per-record validation of both halves, then the two streaming passes) as a reference
and pin that every LMDB value, ``experiment_reference_index.json`` and
``gene_set.json`` are byte-identical to it, on both Cypher layouts (environment inline,
as the served store holds it, and the interned ``$ref`` pointer layout), across batch
boundaries, with the caches emptied mid-stream, and with two references whose JSON
differs only in key order.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections.abc import Iterator
from typing import Any

import lmdb
import pytest
from pydantic import BaseModel

from torchcell.data import neo4j_query_raw as nqr
from torchcell.data.data import ExperimentReferenceIndex, compute_sha256_hash
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw
from torchcell.datamodels import schema as s
from torchcell.datamodels.interned_constant import (
    collect_pointers,
    resolve_pointers,
    split_experiment_dump,
    verified_constant,
)
from torchcell.sequence import GeneSet
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

N_RECORDS = 27
GENES = ["YAL001C", "YBR002C", "YCL003W", "YDR004W", "YER005C", "YFL006W"]


def _environment(n_components: int, temperature: float) -> s.Environment:
    sourced = SourcedValue(
        value="20 g/L",
        quote="20 g/L of each component",
        provenance=Provenance(
            source_uri="paper.md", citation_key="test2026", sha256="0" * 64
        ),
    )
    return s.Environment(
        media=s.Media(
            name="YPD",
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
                    provenance=[sourced],
                )
                for i in range(n_components)
            ],
        ),
        temperature=s.Temperature(value=temperature),
    )


# Index 0 is small enough to stay inline even on the pointer layout.
ENVIRONMENTS = [
    _environment(0, 30.0),
    _environment(4, 30.0),
    _environment(4, 37.0),
    _environment(6, 30.0),
]


def _experiment(i: int) -> s.FitnessExperiment:
    genes = GENES[i % len(GENES) : i % len(GENES) + 1 + i % 2]
    return s.FitnessExperiment(
        dataset_name="TestDataset",
        genotype=s.Genotype(
            perturbations=[
                s.SgaKanMxDeletionPerturbation(
                    systematic_gene_name=g, perturbed_gene_name=g, strain_id=f"S{i}"
                )
                for g in genes
            ]
        ),
        environment=ENVIRONMENTS[i % len(ENVIRONMENTS)],
        phenotype=s.FitnessPhenotype(fitness=0.5 + i / 100, fitness_std=0.01),
    )


def _reference(i: int) -> s.FitnessExperimentReference:
    return s.FitnessExperimentReference(
        dataset_name="TestDataset",
        genome_reference=s.ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=["BY4741", "BY4742"][i % 2]
        ),
        environment_reference=ENVIRONMENTS[i % len(ENVIRONMENTS)],
        phenotype_reference=s.FitnessPhenotype(fitness=1.0),
    )


def _ref_serialized(i: int) -> str:
    """Three references; reference 2 is sent with its keys in two different orders."""
    dump = _reference(i % 3).model_dump()
    if i % 3 == 2 and i % 2 == 1:
        dump = dict(reversed(list(dump.items())))
    return json.dumps(dump)


def _records(layout: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Cypher-shaped records and the interned-constant store for one layout."""
    store: dict[str, str] = {}
    records = []
    for i in range(N_RECORDS):
        dump = _experiment(i).model_dump()
        if layout == "pointer":
            pointered, constants = split_experiment_dump(dump)
            store.update({ref: payload for ref, _, payload in constants})
            dump = pointered
        records.append(
            {"e_serialized": json.dumps(dump), "ref_serialized": _ref_serialized(i)}
        )
    return records, store


class _StoreQueryRaw(Neo4jQueryRaw):
    """``Neo4jQueryRaw`` over in-memory records and constants, no bolt connection."""

    def __attrs_post_init__(self) -> None:
        # No store is opened here: ``process`` writes ``raw/lmdb.partial`` and moves it
        # into place, so the target must not exist until then.
        self.raw_dir = osp.join(self.root_dir, "raw")
        self.lmdb_dir = osp.join(self.raw_dir, "lmdb")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.records: list[dict[str, str]] = []
        self.store: dict[str, str] = {}

    def fetch_data(self) -> Iterator[Any]:
        yield from self.records

    def fetch_constants(self, refs: list[str]) -> dict[str, Any]:
        return {r: verified_constant(r, self.store[r]) for r in refs}


def _old_build(query: _StoreQueryRaw) -> None:
    """The previous raw stage: per-record validation, then the two streaming passes."""
    constants: dict[str, Any] = {}
    batch = []
    for i, record in enumerate(query.records):
        batch.append(
            (
                i,
                json.loads(record["e_serialized"]),
                json.loads(record["ref_serialized"]),
            )
        )
    refs: set[str] = set()
    for _, e, r in batch:
        collect_pointers(e, refs)
        collect_pointers(r, refs)
    constants.update(query.fetch_constants(sorted(refs)))
    os.makedirs(query.lmdb_dir, exist_ok=True)
    query._init_lmdb(readonly=False)
    with query.env.begin(write=True) as txn:
        for i, e, r in batch:
            e = resolve_pointers(e, constants)
            r = resolve_pointers(r, constants)
            experiment = s.EXPERIMENT_TYPE_MAP[e["experiment_type"]](
                dataset_name=e["dataset_name"],
                genotype=e["genotype"],
                environment=e["environment"],
                phenotype=e["phenotype"],
            )
            reference = s.EXPERIMENT_REFERENCE_TYPE_MAP[r["experiment_reference_type"]](
                **r
            )
            data_json = json.dumps(
                {"experiment": experiment, "experiment_reference": reference},
                default=lambda o: o.model_dump(),
            )
            txn.put(f"data_{i}".encode(), data_json.encode())
    query.close_lmdb()

    # experiment_reference_index, as the streaming property computed it
    env = lmdb.open(query.lmdb_dir, readonly=True, lock=False)
    hash_to_indices: dict[str, list[int]] = {}
    genes = GeneSet()
    with env.begin() as txn:
        for key, value in txn.cursor():
            idx = int(key.decode().split("_")[1])
            stored = json.loads(value.decode("utf-8"))
            ref_dict = stored["experiment_reference"]
            h = compute_sha256_hash(json.dumps(ref_dict, sort_keys=True))
            hash_to_indices.setdefault(h, []).append(idx)
            for p in stored["experiment"]["genotype"]["perturbations"]:
                genes.add(p["systematic_gene_name"])
    env.close()
    index = [
        ExperimentReferenceIndex(
            reference=query[indices[0]]["experiment_reference"].model_dump(),
            member_indices=sorted(indices),
        )
        for indices in hash_to_indices.values()
    ]
    with open(osp.join(query.raw_dir, "experiment_reference_index.json"), "w") as f:
        json.dump([eri.model_dump() for eri in index], f)
    with open(osp.join(query.raw_dir, "gene_set.json"), "w") as f:
        json.dump(list(sorted(genes)), f, indent=0)
    query.close_lmdb()


def _query(root: str, layout: str) -> _StoreQueryRaw:
    query = _StoreQueryRaw(
        uri="bolt://none", username="", password="", root_dir=root, query=""
    )
    query.records, query.store = _records(layout)
    return query


def _outputs(query: _StoreQueryRaw) -> tuple[dict[bytes, bytes], bytes, bytes]:
    env = lmdb.open(query.lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        values = {bytes(k): bytes(v) for k, v in txn.cursor()}
    env.close()
    with open(osp.join(query.raw_dir, "experiment_reference_index.json"), "rb") as f:
        index = f.read()
    with open(osp.join(query.raw_dir, "gene_set.json"), "rb") as f:
        genes = f.read()
    return values, index, genes


@pytest.mark.parametrize("layout", ["inline", "pointer"])
@pytest.mark.parametrize("batch, cache_max", [(1000, 200_000), (4, 2)])
def test_single_pass_outputs_are_byte_identical_to_the_per_record_path(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    batch: int,
    cache_max: int,
) -> None:
    monkeypatch.setattr(nqr, "PROCESS_BATCH", batch)
    monkeypatch.setattr(nqr, "CONSTANT_CACHE_MAX", cache_max)
    old = _query(str(tmp_path / "old"), layout)
    _old_build(old)
    new = _query(str(tmp_path / "new"), layout)
    new.process()
    new.close_lmdb()

    old_values, old_index, old_genes = _outputs(old)
    new_values, new_index, new_genes = _outputs(new)
    assert len(new_values) == N_RECORDS
    assert new_values == old_values
    assert new_index == old_index
    assert new_genes == old_genes
    # the fixture exercises what it claims to
    if layout == "pointer":
        assert any(
            "$ref" in json.loads(r["e_serialized"])["environment"] for r in new.records
        )
        assert any(
            "$ref" not in json.loads(r["e_serialized"])["environment"]
            for r in new.records
        )
    groups = json.loads(old_index)
    assert len(groups) == 3
    # cursor order is not numeric order for some group's first member
    assert any(
        min(g["member_indices"], key=str) != min(g["member_indices"]) for g in groups
    )
    assert len({r["ref_serialized"] for r in new.records}) == 4


def test_properties_load_the_files_process_wrote(tmp_path: Any) -> None:
    query = _query(str(tmp_path), "pointer")
    query.process()
    query.close_lmdb()
    index = query.experiment_reference_index
    assert sorted(i for eri in index for i in eri.member_indices) == list(
        range(N_RECORDS)
    )
    expected: set[str] = set()
    for i in range(N_RECORDS):
        genotype = _experiment(i).genotype
        assert isinstance(genotype, s.Genotype)
        expected |= {p.systematic_gene_name for p in genotype.perturbations}
    assert set(query.gene_set) == expected


def test_records_read_back_equal_the_models_they_came_from(tmp_path: Any) -> None:
    query = _query(str(tmp_path), "pointer")
    query.process()
    query.close_lmdb()
    for i in range(N_RECORDS):
        record = query[i]
        assert record["experiment"] == _experiment(i)
        assert record["experiment_reference"] == _reference(i % 3)


def test_cached_environment_is_safe_to_pass_unvalidated() -> None:
    """Every experiment class takes ``Environment`` as is: no validators, no revalidation.

    The single pass hands a cached, already-validated ``Environment`` to the
    experiment constructor, which pydantic accepts without re-running validation. That
    is only equivalent to validating the dict if no experiment-level validator reads
    or rewrites the environment and instances are not revalidated.
    """
    for cls in s.EXPERIMENT_TYPE_MAP.values():
        assert isinstance(cls, type) and issubclass(cls, BaseModel)
        assert cls.model_fields["environment"].annotation is s.Environment
        decorators = cls.__pydantic_decorators__
        assert not decorators.model_validators, cls.__name__
        assert not decorators.field_validators, cls.__name__
        assert cls.model_config.get("revalidate_instances", "never") == "never"
        assert not cls.model_computed_fields, cls.__name__
