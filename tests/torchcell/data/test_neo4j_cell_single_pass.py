# tests/torchcell/data/test_neo4j_cell_single_pass.py
"""The cell dataset's stages read nothing back that the raw stage already saw.

``Neo4jCellDataset.process`` registers a ``RawStageGrouping`` on the raw stage, so the
first grouping stage (the aggregator, or the deduplicator when there is one) gets its
pass-1 grouping computed while the raw LMDB is written, and, when that stage is the
aggregation, the processed store's per-record passes (``compute_phenotype_info``,
``label_df``, the phenotype-label, dataset-name and perturbation-count indices) are
written from the same observations. These tests build the same dataset twice from
the same synthetic Cypher records: once on the new path, and once with the raw
stage's grouping withheld (``raw_stage_ran`` False, which is also what a raw LMDB left
by an earlier run gives), which runs the unchanged pass 1 and standalone passes. Every
LMDB key and value and every index and label file must be byte-identical, on both
Cypher layouts, with the aggregator alone and with a deduplicator in front of it.
"""

from __future__ import annotations

import json
import os.path as osp
from collections.abc import Iterator
from typing import Any

import lmdb
import pandas as pd
import pytest

from torchcell.data import neo4j_cell as nc
from torchcell.data.aggregate import Aggregator
from torchcell.data.deduplicate import Deduplicator
from torchcell.data.genotype_aggregate import GenotypeAggregator
from torchcell.data.mean_experiment_deduplicate import MeanExperimentDeduplicator
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw, RecordObserver
from torchcell.datamodels import schema as s
from torchcell.datamodels.interned_constant import (
    split_experiment_dump,
    verified_constant,
)
from torchcell.sequence import GeneSet
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

N_RECORDS = 40
GENES = ["YAL001C", "YBR002C", "YCL003W", "YDR004W", "YER005C"]


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


ENVIRONMENTS = [_environment(0), _environment(5)]
GENOME = s.ReferenceGenome(species="Saccharomyces cerevisiae", strain="BY4741")


def _genotype(i: int) -> s.Genotype:
    # 7 distinct gene sets over 40 records: repeated genotypes spread across the
    # index range, so groups have members on both sides of the lexicographic split.
    k = i % 7
    genes = [GENES[k % len(GENES)]] + ([GENES[(k + 2) % len(GENES)]] if k > 4 else [])
    return s.Genotype(
        perturbations=[
            s.SgaKanMxDeletionPerturbation(
                systematic_gene_name=g, perturbed_gene_name=g, strain_id=f"S{g}"
            )
            for g in genes
        ]
    )


def _record(i: int) -> tuple[Any, Any]:
    environment = ENVIRONMENTS[i % 2]
    if i % 3 == 2:
        gi = s.GeneInteractionExperiment(
            dataset_name="GIDataset",
            genotype=_genotype(i),
            environment=environment,
            phenotype=s.GeneInteractionPhenotype(
                gene_interaction=-0.1 * (i % 5), gene_interaction_p_value=0.01
            ),
        )
        gi_ref = s.GeneInteractionExperimentReference(
            dataset_name="GIDataset",
            genome_reference=GENOME,
            environment_reference=environment,
            phenotype_reference=s.GeneInteractionPhenotype(gene_interaction=0.0),
        )
        return gi, gi_ref
    fitness = s.FitnessExperiment(
        dataset_name="FitnessDataset",
        genotype=_genotype(i),
        environment=environment,
        phenotype=s.FitnessPhenotype(fitness=0.5 + i / 100, fitness_std=0.01),
    )
    fitness_ref = s.FitnessExperimentReference(
        dataset_name="FitnessDataset",
        genome_reference=GENOME,
        environment_reference=environment,
        phenotype_reference=s.FitnessPhenotype(fitness=1.0),
    )
    return fitness, fitness_ref


def _records(layout: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    store: dict[str, str] = {}
    records = []
    for i in range(N_RECORDS):
        experiment, reference = _record(i)
        dump = experiment.model_dump()
        if layout == "pointer":
            dump, constants = split_experiment_dump(dump)
            store.update({ref: payload for ref, _, payload in constants})
        records.append(
            {
                "e_serialized": json.dumps(dump),
                "ref_serialized": json.dumps(reference.model_dump()),
            }
        )
    return records, store


class _MemoryQueryRaw(Neo4jQueryRaw):
    """``Neo4jQueryRaw`` whose Cypher result and constants come from class state."""

    RECORDS: list[dict[str, str]] = []
    STORE: dict[str, str] = {}

    def fetch_data(self) -> Iterator[Any]:
        yield from type(self).RECORDS

    def fetch_constants(self, refs: list[str]) -> dict[str, Any]:
        return {r: verified_constant(r, type(self).STORE[r]) for r in refs}


def _build(
    root: str,
    monkeypatch: pytest.MonkeyPatch,
    withhold_grouping: bool,
    deduplicator: type[Deduplicator] | None,
    aggregator: type[Aggregator],
) -> nc.Neo4jCellDataset:
    def load_raw(
        uri: str,
        username: str,
        password: str,
        root_dir: str,
        query: str,
        gene_set: GeneSet,
        record_observers: list[RecordObserver],
        fetch_workers: int,
    ) -> Neo4jQueryRaw:
        raw = _MemoryQueryRaw(
            uri=uri,
            username=username,
            password=password,
            root_dir=root_dir,
            query=query,
            record_observers=list(record_observers),
            fetch_workers=fetch_workers,
        )
        if withhold_grouping:
            raw.raw_stage_ran = False
        return raw

    monkeypatch.setattr(nc.Neo4jCellDataset, "load_raw", staticmethod(load_raw))
    return nc.Neo4jCellDataset(
        root=root,
        query="MATCH (n) RETURN n",
        gene_set=GeneSet(GENES),
        deduplicator=deduplicator,
        aggregator=aggregator,
        uri="bolt://none",
        username="",
        password="",
    )


def _lmdb_items(path: str) -> list[tuple[bytes, bytes]]:
    env = lmdb.open(path, readonly=True, lock=False)
    with env.begin() as txn:
        items = [(bytes(k), bytes(v)) for k, v in txn.cursor()]
    env.close()
    return items


PROCESSED_FILES = [
    "experiment_types.json",
    "phenotype_label_index.json",
    "dataset_name_index.json",
    "perturbation_count_index.json",
    "label_df.parquet",
]


@pytest.mark.parametrize("layout", ["inline", "pointer"])
@pytest.mark.parametrize("with_deduplicator", [False, True])
def test_cell_dataset_outputs_are_byte_identical_without_the_re_reads(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, layout: str, with_deduplicator: bool
) -> None:
    records, store = _records(layout)
    monkeypatch.setattr(_MemoryQueryRaw, "RECORDS", records)
    monkeypatch.setattr(_MemoryQueryRaw, "STORE", store)
    deduplicator = MeanExperimentDeduplicator if with_deduplicator else None

    passes: dict[str, int] = {}

    def counting(name: str, method: Any) -> Any:
        def wrapped(self: Any, *args: Any) -> Any:
            passes[name] = passes.get(name, 0) + 1
            return method(self, *args)

        return wrapped

    for name in ("compute_phenotype_info", "compute_phenotype_label_index"):
        monkeypatch.setattr(
            nc.Neo4jCellDataset,
            name,
            counting(name, getattr(nc.Neo4jCellDataset, name)),
        )
    for cls, name in (
        (GenotypeAggregator, "aggregate_key_raw"),
        (MeanExperimentDeduplicator, "duplicate_key"),
    ):
        monkeypatch.setattr(cls, name, counting(name, getattr(cls, name)))

    old = _build(
        str(tmp_path / "old"), monkeypatch, True, deduplicator, GenotypeAggregator
    )
    old_passes = dict(passes)
    passes.clear()
    new = _build(
        str(tmp_path / "new"), monkeypatch, False, deduplicator, GenotypeAggregator
    )
    new_passes = dict(passes)

    stages = ["raw", "aggregation"] + (["deduplication"] if with_deduplicator else [])
    for stage in stages:
        old_items = _lmdb_items(osp.join(old.root, stage, "lmdb"))
        assert old_items, stage
        assert _lmdb_items(osp.join(new.root, stage, "lmdb")) == old_items, stage
    assert _lmdb_items(osp.join(new.processed_dir, "lmdb")) == _lmdb_items(
        osp.join(old.processed_dir, "lmdb")
    )
    for name in PROCESSED_FILES:
        with open(osp.join(old.processed_dir, name), "rb") as f:
            old_bytes = f.read()
        with open(osp.join(new.processed_dir, name), "rb") as f:
            assert f.read() == old_bytes, name
    for name in ("experiment_reference_index.json", "gene_set.json"):
        with open(osp.join(old.raw_dir, name), "rb") as f:
            old_bytes = f.read()
        with open(osp.join(new.raw_dir, name), "rb") as f:
            assert f.read() == old_bytes, name
    pd.testing.assert_frame_equal(new.label_df, old.label_df)

    # The new path skipped what it claims to skip. With a deduplicator the grouping
    # goes to it (its pass 1 is skipped); the aggregation reads the deduplicated
    # store and runs its own pass 1, and the processed-store passes run as before.
    assert old_passes["compute_phenotype_info"] == 1
    assert old_passes["compute_phenotype_label_index"] == 1
    n_groups = len(_lmdb_items(osp.join(new.root, "aggregation", "lmdb")))
    assert n_groups == 7
    if with_deduplicator:
        assert old_passes["duplicate_key"] == 2 * N_RECORDS  # observer + pass 1
        assert new_passes["duplicate_key"] == N_RECORDS  # observer only
        assert new_passes["aggregate_key_raw"] == old_passes["aggregate_key_raw"]
        assert new_passes["compute_phenotype_info"] == 1
    else:
        assert old_passes["aggregate_key_raw"] == 2 * N_RECORDS
        assert new_passes["aggregate_key_raw"] == N_RECORDS
        assert "compute_phenotype_info" not in new_passes
        assert "compute_phenotype_label_index" not in new_passes
    # The fixture exercises the lexicographic ordering: some multi-member group's
    # first key differs between cursor (string) and numeric order.
    raw_keys = [k for k, _ in _lmdb_items(osp.join(new.root, "raw", "lmdb"))]
    assert raw_keys[:3] == [b"data_0", b"data_1", b"data_10"]
    label_df = new.label_df
    assert set(label_df.columns) == {"index", "fitness", "gene_interaction"}


def test_label_values_read_off_the_dict_equal_the_validated_attribute() -> None:
    """``label_df`` read ``getattr(validated experiment.phenotype, name)``; the fold
    reads the stored dict. For every experiment class and every label name, a name is
    a phenotype attribute exactly when it is a dumped field, so the two agree on which
    records carry a value, and the values round-trip unchanged.
    """
    for experiment_class in s.EXPERIMENT_TYPE_MAP.values():
        phenotype_class = experiment_class.__annotations__["phenotype"]
        for name in nc.LABEL_NAME_CANDIDATES:
            is_field = name in phenotype_class.model_fields
            is_attribute = hasattr(phenotype_class, name) and not is_field
            assert not is_attribute, (phenotype_class.__name__, name)
    for i in range(N_RECORDS):
        experiment, _ = _record(i)
        stored = json.loads(json.dumps(experiment.model_dump()))
        revalidated = type(experiment)(**stored)
        for name in nc.LABEL_NAME_CANDIDATES:
            if name in stored["phenotype"]:
                assert stored["phenotype"][name] == getattr(revalidated.phenotype, name)
