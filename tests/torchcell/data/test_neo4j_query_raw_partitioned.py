# tests/torchcell/data/test_neo4j_query_raw_partitioned.py
"""The partitioned raw stage writes exactly what the single-session raw stage writes.

``Neo4jQueryRaw(fetch_workers=N)`` splits a query carrying ``{partition}`` once per
``UNION ALL`` block into ``e.id``-prefix partitions, renders them in N forked worker
processes, and writes them in (block, prefix) order. The stub here answers queries the
way the server does: a whole query returns its blocks in order, each sorted by
``e.id``; a partition query returns its block's records whose id starts with the
prefix, sorted by ``e.id``. Experiment ids are the real ones, the sha256 of the inlined
record, so the records spread over the hex prefixes as served ids do. Every LMDB value,
``experiment_reference_index.json``, ``gene_set.json`` and a ``SplitRecordObserver``'s
state must be byte-identical between ``fetch_workers=0`` and ``fetch_workers=3``, on
both Cypher layouts.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
import re
from collections.abc import Iterator
from typing import Any

import lmdb
import pytest

from torchcell.data import neo4j_query_raw as nqr
from torchcell.data.neo4j_cell import RawStageGrouping
from torchcell.data.neo4j_query_raw import (
    PARTITION_MARKER,
    Neo4jQueryRaw,
    partition_queries,
)
from torchcell.datamodels import schema as s
from torchcell.datamodels.interned_constant import (
    split_experiment_dump,
    verified_constant,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

BLOCK = """
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = '{dataset}'{partition}
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized
"""
DATASETS = ["BlockADataset", "BlockBDataset"]
QUERY = "\nUNION ALL\n".join(
    BLOCK.replace("{dataset}", d) for d in DATASETS
)  # an example partitioned query: the marker once per block, inside the WHERE
GENES = ["YAL001C", "YBR002C", "YCL003W", "YDR004W", "YER005C", "YFL006W"]
PER_BLOCK = 30


def _environment(n_components: int) -> s.Environment:
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
        temperature=s.Temperature(value=30.0),
    )


ENVIRONMENTS = [_environment(0), _environment(4), _environment(6)]


def _record(dataset: str, i: int) -> tuple[s.FitnessExperiment, str]:
    genes = GENES[i % len(GENES) : i % len(GENES) + 1 + i % 2]
    experiment = s.FitnessExperiment(
        dataset_name=dataset,
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
    reference = s.FitnessExperimentReference(
        dataset_name=dataset,
        genome_reference=s.ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=["BY4741", "BY4742"][i % 2]
        ),
        environment_reference=ENVIRONMENTS[i % len(ENVIRONMENTS)],
        phenotype_reference=s.FitnessPhenotype(fitness=1.0),
    )
    return experiment, json.dumps(reference.model_dump())


def _blocks(layout: str) -> tuple[list[list[dict[str, str]]], dict[str, str]]:
    """Per block, the records sorted by e.id, as the server orders them."""
    store: dict[str, str] = {}
    blocks = []
    for dataset in DATASETS:
        rows = []
        for i in range(PER_BLOCK):
            experiment, ref_serialized = _record(dataset, i)
            dump = experiment.model_dump()
            e_id = hashlib.sha256(json.dumps(dump).encode()).hexdigest()
            if layout == "pointer":
                dump, constants = split_experiment_dump(dump)
                store.update({ref: payload for ref, _, payload in constants})
            rows.append(
                {
                    "e_id": e_id,
                    "e_serialized": json.dumps(dump),
                    "ref_serialized": ref_serialized,
                }
            )
        blocks.append(sorted(rows, key=lambda r: r["e_id"]))
    return blocks, store


class _ServerStub(Neo4jQueryRaw):
    """Answers single-session and partition queries from class state, no bolt."""

    BLOCKS: list[list[dict[str, str]]] = []
    STORE: dict[str, str] = {}

    def fetch_query(self, query: str) -> Iterator[Any]:
        if re.search(r"\bUNION\s+ALL\b", query):
            for block in type(self).BLOCKS:
                yield from block
            return
        dataset = re.search(r"dataset\.id = '(\w+)'", query)
        assert dataset is not None
        block = type(self).BLOCKS[DATASETS.index(dataset.group(1))]
        prefix = re.search(r"e\.id STARTS WITH '([0-9a-f]+)'", query)
        guard = re.search(r"NOT e\.id =~ '(.*)'", query)
        if prefix is not None:
            yield from (r for r in block if r["e_id"].startswith(prefix.group(1)))
        else:
            assert guard is not None, query
            yield from (r for r in block if not re.fullmatch(guard.group(1), r["e_id"]))

    def fetch_constants(self, refs: list[str]) -> dict[str, Any]:
        return {r: verified_constant(r, type(self).STORE[r]) for r in refs}


def _key(record: dict[str, Any]) -> str:
    return json.dumps(
        sorted(
            p["systematic_gene_name"]
            for p in record["experiment"]["genotype"]["perturbations"]
        )
    )


def _run(
    root: str, fetch_workers: int, prefix_length: int = 1
) -> tuple[_ServerStub, RawStageGrouping]:
    grouping = RawStageGrouping(key_fn=_key, summarize=True)
    raw = _ServerStub(
        uri="bolt://none",
        username="",
        password="",
        root_dir=root,
        query=QUERY,
        record_observers=[grouping],
        fetch_workers=fetch_workers,
        partition_prefix_length=prefix_length,
    )
    raw.close_lmdb()
    return raw, grouping


def _outputs(raw: Neo4jQueryRaw) -> tuple[list[tuple[bytes, bytes]], bytes, bytes]:
    env = lmdb.open(raw.lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        items = [(bytes(k), bytes(v)) for k, v in txn.cursor()]
    env.close()
    with open(osp.join(raw.raw_dir, "experiment_reference_index.json"), "rb") as f:
        index = f.read()
    with open(osp.join(raw.raw_dir, "gene_set.json"), "rb") as f:
        genes = f.read()
    return items, index, genes


@pytest.mark.parametrize("layout", ["inline", "pointer"])
@pytest.mark.parametrize("batch, prefix_length", [(1000, 1), (3, 1), (1000, 2)])
def test_partitioned_outputs_are_byte_identical_to_one_session(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    batch: int,
    prefix_length: int,
) -> None:
    blocks, store = _blocks(layout)
    monkeypatch.setattr(_ServerStub, "BLOCKS", blocks)
    monkeypatch.setattr(_ServerStub, "STORE", store)
    monkeypatch.setattr(nqr, "PROCESS_BATCH", batch)
    one, one_grouping = _run(str(tmp_path / "one"), 0)
    many, many_grouping = _run(str(tmp_path / "many"), 3, prefix_length)

    one_items, one_index, one_genes = _outputs(one)
    assert len(one_items) == 2 * PER_BLOCK
    assert _outputs(many) == (one_items, one_index, one_genes)
    assert many_grouping.groups == one_grouping.groups
    assert many_grouping.summaries == one_grouping.summaries
    # the fixture spans several prefixes in each block, and the blocks interleave
    # in id order, so a merge by id instead of by block would be caught
    for block in blocks:
        assert len({r["e_id"][0] for r in block}) >= 8
    assert blocks[0][-1]["e_id"] > blocks[1][0]["e_id"]
    # record order is the server's: block by block, each by e.id
    stored = [
        json.loads(v)["experiment"]
        for _, v in sorted(one_items, key=lambda kv: int(kv[0].decode()[5:]))
    ]
    served = [r for block in blocks for r in block]
    assert [e["dataset_name"] for e in stored] == [
        d for d in DATASETS for _ in range(PER_BLOCK)
    ]
    assert [hashlib.sha256(json.dumps(e).encode()).hexdigest() for e in stored] == [
        r["e_id"] for r in served
    ]


def test_partition_queries_order_and_guard() -> None:
    parts = partition_queries(QUERY, 1)
    assert [(b, p) for b, p, _ in parts] == [
        (b, p) for b in range(2) for p in [*"0123456789abcdef", ""]
    ]
    assert "STARTS WITH 'a'" in parts[10][2] and PARTITION_MARKER not in parts[10][2]
    assert "NOT e.id =~ '[0-9a-f]{1}.*'" in parts[16][2]
    assert "UNION" not in parts[0][2]
    assert len(partition_queries(QUERY, 2)) == 2 * 257


def test_a_query_without_the_marker_cannot_be_partitioned(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="partition marker"):
        _run_query(str(tmp_path), QUERY.replace(PARTITION_MARKER, ""), 2)
    with pytest.raises(ValueError, match="partition marker"):
        partition_queries(QUERY.replace(PARTITION_MARKER, "", 1), 1)


def _run_query(root: str, query: str, fetch_workers: int) -> Neo4jQueryRaw:
    return _ServerStub(
        uri="bolt://none",
        username="",
        password="",
        root_dir=root,
        query=query,
        fetch_workers=fetch_workers,
    )


def test_a_plain_observer_cannot_run_partitioned(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, store = _blocks("inline")
    monkeypatch.setattr(_ServerStub, "BLOCKS", blocks)

    def plain(index: int, record: dict[str, Any]) -> None:
        return None

    with pytest.raises(TypeError, match="SplitRecordObserver"):
        _ServerStub(
            uri="bolt://none",
            username="",
            password="",
            root_dir=str(tmp_path),
            query=QUERY,
            record_observers=[plain],
            fetch_workers=2,
        )


def test_an_id_outside_the_hex_prefixes_stops_the_partitioned_stage(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks, store = _blocks("inline")
    stray = dict(blocks[1][0], e_id="G" + blocks[1][0]["e_id"][1:])
    monkeypatch.setattr(_ServerStub, "BLOCKS", [blocks[0], [*blocks[1], stray]])
    one = _run_query(str(tmp_path / "one"), QUERY, 0)
    assert len(one) == 2 * PER_BLOCK + 1  # one session returns it
    with pytest.raises(ValueError, match="lowercase hex prefix"):
        _run_query(str(tmp_path / "many"), QUERY, 2)
