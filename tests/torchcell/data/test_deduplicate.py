# tests/torchcell/data/test_deduplicate.py
# [[tests.torchcell.data.test_deduplicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_deduplicate.py
"""The abstract ``Deduplicator`` LMDB stage, driven through ``MeanExperimentDeduplicator``.

``tests/torchcell/data/test_mean_experiment_deduplicate.py`` pins the merge arithmetic on
in-memory records; this file pins the LMDB stage around it: ``process`` on a hand-written
raw store, the readers, and the environment lifecycle.

The raw store holds four records under the keys ``Neo4jQueryRaw.process`` writes,
``data_0`` to ``data_3``, each ``json.dumps({"experiment": model_dump(),
"experiment_reference": model_dump()})``:

* ``data_0`` fitness, dataset ``a``, YAL001C deleted, fitness 0.75, std 1.0
* ``data_1`` fitness, dataset ``b``, YAL002W deleted, fitness 0.5 (singleton)
* ``data_2`` fitness, dataset ``c``, YAL001C deleted, fitness 0.25, std 0.0 (duplicate of 0)
* ``data_3`` gene interaction, dataset ``d``, YAL001C + YAL002W (singleton, other type)

The duplicate key is ``sha256("<experiment_type>:<sorted gene names>")``, so records 0
and 2 share a group and record 3 does not. Groups are written in first-occurrence order
under the keys ``0``, ``1``, ``2``: the merged pair, then ``data_1`` and ``data_3``
passed through byte for byte. The merged record has fitness (0.75 + 0.25) / 2 = 0.5,
RMS-pooled std sqrt((1.0^2 + 0.0^2) / 2) = sqrt(0.5) = 0.7071067811865476, a
``MeanDeletionPerturbation`` with ``num_duplicates`` 2, dataset name ``a+c``, and a
reference with fitness (1.0 + 1.0) / 2 = 1.0 and no std (both inputs carry none).
"""

import json
import math
import os.path as osp
from pathlib import Path
from typing import Any

import lmdb
import pytest

from torchcell.data.mean_experiment_deduplicate import MeanExperimentDeduplicator
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeanDeletionPerturbation,
    Media,
    ReferenceGenome,
    Temperature,
)

ENVIRONMENT = Environment(
    media=Media(name="YPD", state="solid", is_synthetic=False),
    temperature=Temperature(value=30.0),
)
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _deletion(gene: str) -> KanMxDeletionPerturbation:
    return KanMxDeletionPerturbation(
        systematic_gene_name=gene, perturbed_gene_name=gene
    )


def _fitness(
    dataset: str, genes: list[str], fitness: float, std: float | None
) -> dict[str, Any]:
    return {
        "experiment": FitnessExperiment(
            dataset_name=dataset,
            genotype=Genotype(perturbations=[_deletion(g) for g in genes]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=fitness, fitness_std=std),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name=dataset,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0),
        ),
    }


def _interaction(dataset: str, genes: list[str], score: float) -> dict[str, Any]:
    return {
        "experiment": GeneInteractionExperiment(
            dataset_name=dataset,
            genotype=Genotype(perturbations=[_deletion(g) for g in genes]),
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=score, gene_interaction_p_value=0.5
            ),
        ),
        "experiment_reference": GeneInteractionExperimentReference(
            dataset_name=dataset,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=GeneInteractionPhenotype(gene_interaction=0.0),
        ),
    }


RECORDS = [
    _fitness("a", ["YAL001C"], 0.75, 1.0),
    _fitness("b", ["YAL002W"], 0.5, None),
    _fitness("c", ["YAL001C"], 0.25, 0.0),
    _interaction("d", ["YAL001C", "YAL002W"], 0.1),
]


def _serialize(record: dict[str, Any]) -> bytes:
    return json.dumps({k: v.model_dump() for k, v in record.items()}).encode()


RAW_BYTES = [_serialize(r) for r in RECORDS]


def _write_raw(path: str) -> None:
    env = lmdb.open(path, map_size=int(1e8))
    with env.begin(write=True) as txn:
        for i, raw in enumerate(RAW_BYTES):
            txn.put(f"data_{i}".encode(), raw)
    env.close()


def _merged_record() -> dict[str, Any]:
    """The expected merge of records 0 and 2, built by hand."""
    genotype = Genotype(
        perturbations=[
            MeanDeletionPerturbation(
                systematic_gene_name="YAL001C",
                perturbed_gene_name="YAL001C",
                num_duplicates=2,
            )
        ]
    )
    return {
        "experiment": FitnessExperiment(
            dataset_name="a+c",
            genotype=genotype,
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=0.5, fitness_std=math.sqrt(0.5)),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name="a+c",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0, fitness_std=None),
        ),
    }


def _dump(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v.model_dump() for k, v in record.items()}


@pytest.fixture
def processed(tmp_path: Path) -> MeanExperimentDeduplicator:
    """A deduplicator whose store holds the three groups of ``RECORDS``."""
    raw = str(tmp_path / "raw")
    _write_raw(raw)
    dedup = MeanExperimentDeduplicator(root=str(tmp_path))
    dedup.process(raw, dedup.lmdb_dir)
    return dedup


def test_process_writes_groups_in_first_occurrence_order(
    processed: MeanExperimentDeduplicator,
) -> None:
    """Keys 0, 1, 2: the merged pair, then the two singletons as their stored bytes."""
    env = lmdb.open(processed.lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        stored = {bytes(k): bytes(v) for k, v in txn.cursor()}
    env.close()
    assert sorted(stored) == [b"0", b"1", b"2"]
    assert json.loads(stored[b"0"]) == _dump(_merged_record())
    assert stored[b"1"] == RAW_BYTES[1]
    assert stored[b"2"] == RAW_BYTES[3]
    assert processed.env is None  # process closes the environment it opened


def test_getitem_by_int_returns_pydantic_objects(
    processed: MeanExperimentDeduplicator,
) -> None:
    """Index 0 is the merged record, index 2 the interaction, reconstructed as models."""
    merged = processed[0]
    assert isinstance(merged, dict)
    assert type(merged["experiment"]) is FitnessExperiment
    assert _dump(merged) == _dump(_merged_record())
    interaction = processed[2]
    assert isinstance(interaction, dict)
    assert type(interaction["experiment"]) is GeneInteractionExperiment
    assert _dump(interaction) == _dump(RECORDS[3])


def test_getitem_by_slice_and_list(processed: MeanExperimentDeduplicator) -> None:
    """A slice walks ``slice.indices(len)``; a list keeps the order it was given."""
    every_other = processed[0:10:2]
    assert isinstance(every_other, list)
    assert [_dump(r) for r in every_other] == [
        _dump(_merged_record()),
        _dump(RECORDS[3]),
    ]
    last = processed[-1:]
    assert isinstance(last, list)
    assert [_dump(r) for r in last] == [_dump(RECORDS[3])]
    reversed_pair = processed[[2, 1]]
    assert isinstance(reversed_pair, list)
    assert [_dump(r) for r in reversed_pair] == [_dump(RECORDS[3]), _dump(RECORDS[1])]
    assert processed[5:9] == []


def test_getitem_errors(processed: MeanExperimentDeduplicator) -> None:
    """A string index is a TypeError, a missing key an IndexError."""
    with pytest.raises(TypeError, match=r"Invalid index type: <class 'str'>"):
        processed["0"]  # type: ignore[index]
    with pytest.raises(IndexError, match="No item found at index 7"):
        processed[7]
    processed.close_lmdb()
    with pytest.raises(ValueError, match="LMDB environment is not initialized."):
        processed._get_record_by_index(0)
    with pytest.raises(ValueError, match="LMDB environment is not initialized."):
        processed._get_records_by_slice(slice(0, 1))


def test_len_bool_and_repr_before_and_after_process(tmp_path: Path) -> None:
    """Finding: ``__repr__`` is the literal ``Deduplicator(root=...)`` for every
    subclass; it does not name ``MeanExperimentDeduplicator``. Before ``process`` the
    store is absent: ``len`` is 0 and ``bool`` False.
    """
    dedup = MeanExperimentDeduplicator(root=str(tmp_path))
    assert dedup.lmdb_dir == osp.join(str(tmp_path), "deduplication", "lmdb")
    assert repr(dedup) == f"Deduplicator(root={tmp_path})"
    assert len(dedup) == 0
    assert bool(dedup) is False
    raw = str(tmp_path / "raw")
    _write_raw(raw)
    dedup.process(raw, dedup.lmdb_dir)
    assert len(dedup) == 3
    assert bool(dedup) is True
    dedup.close_lmdb()


def test_init_lmdb_readonly_needs_an_existing_store_and_writable_creates_one(
    tmp_path: Path,
) -> None:
    """Readonly on an absent store leaves ``env`` None but makes the parent directory;
    writable creates the store; a second init closes the first environment.
    """
    absent = MeanExperimentDeduplicator(root=str(tmp_path / "absent"))
    absent._init_lmdb(readonly=True)
    assert absent.env is None
    assert osp.isdir(osp.join(str(tmp_path), "absent", "deduplication"))
    assert not osp.exists(absent.lmdb_dir)

    dedup = MeanExperimentDeduplicator(root=str(tmp_path))
    dedup._init_lmdb(readonly=False)
    first = dedup.env
    assert first.path() == dedup.lmdb_dir
    assert osp.isdir(dedup.lmdb_dir)

    dedup._init_lmdb(readonly=True)
    second = dedup.env
    assert second is not first
    assert second.path() == dedup.lmdb_dir
    with pytest.raises(
        lmdb.Error, match="Attempt to operate on closed/deleted/dropped"
    ):
        first.begin()

    dedup.close_lmdb()
    assert dedup.env is None
    dedup.close_lmdb()  # idempotent
    assert dedup.env is None
