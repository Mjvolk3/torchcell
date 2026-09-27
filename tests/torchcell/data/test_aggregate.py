# tests/torchcell/data/test_aggregate.py
# [[tests.torchcell.data.test_aggregate]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_aggregate.py
"""The abstract ``Aggregator`` LMDB stage, driven through ``GenotypeAggregator``.

``tests/torchcell/data/test_genotype_aggregate.py`` pins the ``aggregate_check`` key on
in-memory objects; this file pins the raw-JSON key, the LMDB stage around it, and the
readers.

The raw store holds four records under ``data_0`` to ``data_3``, each
``json.dumps({"experiment": model_dump(), "experiment_reference": model_dump()})``:

* ``data_0`` fitness, dataset ``a``, YAL001C deleted
* ``data_1`` fitness, dataset ``b``, YAL002W + YAL001C deleted
* ``data_2`` gene interaction, dataset ``c``, YAL001C + YAL002W (same gene set as 1)
* ``data_3`` fitness, dataset ``d``, YAL001C deleted (same gene set as 0)

The key is ``sha256(str(sorted(set(gene names))))`` and ignores the experiment type, so
records 0 and 3 form group ``0`` and records 1 and 2 form group ``1`` (first-occurrence
order). A group is stored as the JSON array of its members' stored bytes joined by a
comma, so group ``0`` is exactly ``b"[" + raw_0 + b"," + raw_3 + b"]"``. The two keys are
``sha256("['YAL001C']")`` = ``7cabd79d...`` and ``sha256("['YAL001C', 'YAL002W']")`` =
``45978da8...`` (full digests in the tests, computed with ``hashlib`` in a shell).
"""

import json
import os.path as osp
from pathlib import Path
from typing import Any, cast

import lmdb
import pytest
from pydantic import ValidationError

from torchcell.data.aggregate import Aggregator, ExperimentInfo
from torchcell.data.genotype_aggregate import (
    DeletionKeyedGenotypeAggregator,
    GenotypeAggregator,
)
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
    Media,
    ReferenceGenome,
    Temperature,
)

ENVIRONMENT = Environment(
    media=Media(name="YPD", state="solid", is_synthetic=False),
    temperature=Temperature(value=30.0),
)
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
KEY_SINGLE = "7cabd79d8be173ac803e1e8e7f855fb53b978b6e2c1eefb3c6f284671837ec59"
KEY_DOUBLE = "45978da82cc80da11902760351f07b009538a664546c1f8af829cbf1d0f13e5b"


def _deletion(gene: str) -> KanMxDeletionPerturbation:
    return KanMxDeletionPerturbation(
        systematic_gene_name=gene, perturbed_gene_name=gene
    )


def _fitness(dataset: str, genes: list[str], fitness: float) -> dict[str, Any]:
    return {
        "experiment": FitnessExperiment(
            dataset_name=dataset,
            genotype=Genotype(perturbations=[_deletion(g) for g in genes]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=fitness),
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
            phenotype=GeneInteractionPhenotype(gene_interaction=score),
        ),
        "experiment_reference": GeneInteractionExperimentReference(
            dataset_name=dataset,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=GeneInteractionPhenotype(gene_interaction=0.0),
        ),
    }


RECORDS = [
    _fitness("a", ["YAL001C"], 0.9),
    _fitness("b", ["YAL002W", "YAL001C"], 0.4),
    _interaction("c", ["YAL001C", "YAL002W"], 0.1),
    _fitness("d", ["YAL001C"], 0.8),
]


def _dump(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v.model_dump() for k, v in record.items()}


RAW_BYTES = [json.dumps(_dump(r)).encode() for r in RECORDS]


def _write_raw(path: str, raws: list[bytes]) -> None:
    env = lmdb.open(path, map_size=int(1e8))
    with env.begin(write=True) as txn:
        for i, raw in enumerate(raws):
            txn.put(f"data_{i}".encode(), raw)
    env.close()


@pytest.fixture
def processed(tmp_path: Path) -> GenotypeAggregator:
    """An aggregator whose store holds the two groups of ``RECORDS``."""
    raw = str(tmp_path / "raw")
    _write_raw(raw, RAW_BYTES)
    agg = GenotypeAggregator(root=str(tmp_path))
    agg.process(raw, agg.lmdb_dir)
    return agg


def test_aggregate_key_raw_is_the_sorted_gene_set_hash_and_matches_aggregate_check(
    tmp_path: Path,
) -> None:
    """Both keys are sha256 of the sorted gene-name set; type and order do not enter."""
    agg = GenotypeAggregator(root=str(tmp_path))
    assert agg.aggregate_key_raw(_dump(RECORDS[0])) == KEY_SINGLE
    assert agg.aggregate_key_raw(_dump(RECORDS[3])) == KEY_SINGLE
    assert agg.aggregate_key_raw(_dump(RECORDS[1])) == KEY_DOUBLE
    assert agg.aggregate_key_raw(_dump(RECORDS[2])) == KEY_DOUBLE
    assert agg.aggregate_check(RECORDS[1]) == KEY_DOUBLE


def test_deletion_keyed_aggregator_ignores_non_deletion_perturbations(
    tmp_path: Path,
) -> None:
    """A cassette ``gene_addition`` beside a YAL001C deletion keys as {YAL001C} alone."""
    agg = DeletionKeyedGenotypeAggregator(root=str(tmp_path))
    record = {
        "experiment": {
            "genotype": {
                "perturbations": [
                    {
                        "systematic_gene_name": "CYP76AD1",
                        "perturbation_type": "gene_addition",
                    },
                    {
                        "systematic_gene_name": "YAL001C",
                        "perturbation_type": "sga_kanmx_deletion",
                    },
                ]
            }
        }
    }
    assert agg.aggregate_key_raw(record) == KEY_SINGLE
    # the full-set key sees the cassette gene and lands elsewhere
    assert (
        GenotypeAggregator(root=str(tmp_path)).aggregate_key_raw(record) != KEY_SINGLE
    )


def test_process_joins_stored_bytes_into_one_json_array_per_group(
    processed: GenotypeAggregator,
) -> None:
    """Group 0 is records 0 and 3, group 1 is records 1 and 2, byte for byte."""
    env = lmdb.open(processed.lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        stored = {bytes(k): bytes(v) for k, v in txn.cursor()}
    env.close()
    assert stored == {
        b"0": b"[" + RAW_BYTES[0] + b"," + RAW_BYTES[3] + b"]",
        b"1": b"[" + RAW_BYTES[1] + b"," + RAW_BYTES[2] + b"]",
    }
    assert processed.env is None


def test_getitem_int_returns_the_group_as_pydantic_pairs(
    processed: GenotypeAggregator,
) -> None:
    """Group 1 holds a FitnessExperiment then a GeneInteractionExperiment."""
    group = cast(list[dict[str, Any]], processed[1])
    assert [type(pair["experiment"]) for pair in group] == [
        FitnessExperiment,
        GeneInteractionExperiment,
    ]
    assert [_dump(pair) for pair in group] == [_dump(RECORDS[1]), _dump(RECORDS[2])]


def test_getitem_slice_returns_groups_but_a_list_index_flattens_them(
    processed: GenotypeAggregator,
) -> None:
    """Finding: ``agg[[1, 0]]`` returns ONE flat list of the four records, not a list
    of two groups, while ``agg[0:2]`` returns two groups. The two container indexings
    disagree on nesting (``aggregate.py:185`` flattens).
    """
    by_slice = processed[0:2]
    assert [[_dump(p) for p in g] for g in by_slice] == [  # type: ignore[arg-type]
        [_dump(RECORDS[0]), _dump(RECORDS[3])],
        [_dump(RECORDS[1]), _dump(RECORDS[2])],
    ]
    by_list = processed[[1, 0]]
    assert [_dump(p) for p in by_list] == [  # type: ignore[arg-type]
        _dump(RECORDS[1]),
        _dump(RECORDS[2]),
        _dump(RECORDS[0]),
        _dump(RECORDS[3]),
    ]
    assert processed[7:9] == []


def test_getitem_errors(processed: GenotypeAggregator) -> None:
    """A string index is a TypeError, a missing group an IndexError."""
    with pytest.raises(TypeError, match=r"Invalid index type: <class 'str'>"):
        processed["0"]  # type: ignore[index]
    with pytest.raises(IndexError, match="No item found at index 5"):
        processed[5]
    processed.close_lmdb()
    with pytest.raises(ValueError, match="LMDB environment is not initialized."):
        processed._get_record_by_index(0)
    with pytest.raises(ValueError, match="LMDB environment is not initialized."):
        processed._get_records_by_slice(slice(0, 1))


def test_phenotype_info_lists_each_phenotype_class_once_and_is_not_reset_by_process(
    tmp_path: Path,
) -> None:
    """Finding: ``process`` resets ``self._experiment_info``, an attribute nothing reads,
    and leaves ``_phenotype_info`` cached (``aggregate.py:166``). A store re-processed
    with a new phenotype family keeps reporting the old classes.
    """
    agg = GenotypeAggregator(root=str(tmp_path))
    assert agg.phenotype_info == []  # no store yet
    agg._phenotype_info = None  # the empty answer was cached too

    fitness_only = str(tmp_path / "raw_fitness")
    _write_raw(fitness_only, [RAW_BYTES[0], RAW_BYTES[3]])
    agg.process(fitness_only, agg.lmdb_dir)
    assert agg.phenotype_info == [FitnessPhenotype]
    assert agg.phenotype_info is agg.phenotype_info  # cached list object

    both = str(tmp_path / "raw_both")
    _write_raw(both, RAW_BYTES)
    agg.process(both, agg.lmdb_dir)
    assert agg._experiment_info is None
    assert agg.phenotype_info == [FitnessPhenotype]
    assert set(agg._get_phenotype_info()) == {
        FitnessPhenotype,
        GeneInteractionPhenotype,
    }
    assert agg.env is None  # _get_phenotype_info closes what it opened


def test_len_bool_repr_and_create_aggregate_entry(tmp_path: Path) -> None:
    """Finding: ``__repr__`` is the literal ``Aggregator(root=...)`` for every subclass.
    ``create_aggregate_entry`` flattens nested lists in order.
    """
    agg = GenotypeAggregator(root=str(tmp_path))
    assert agg.lmdb_dir == osp.join(str(tmp_path), "aggregation", "lmdb")
    assert repr(agg) == f"Aggregator(root={tmp_path})"
    assert len(agg) == 0
    assert bool(agg) is False
    raw = str(tmp_path / "raw")
    _write_raw(raw, RAW_BYTES)
    agg.process(raw, agg.lmdb_dir)
    assert len(agg) == 2
    assert bool(agg) is True
    agg.close_lmdb()
    assert Aggregator.create_aggregate_entry(
        agg, [[RECORDS[0]], [RECORDS[1], RECORDS[2]]]
    ) == [RECORDS[0], RECORDS[1], RECORDS[2]]


def test_init_lmdb_readonly_needs_an_existing_store(tmp_path: Path) -> None:
    """Readonly on an absent store leaves ``env`` None; writable creates the store."""
    absent = GenotypeAggregator(root=str(tmp_path / "absent"))
    absent._init_lmdb(readonly=True)
    assert absent.env is None
    assert osp.isdir(osp.join(str(tmp_path), "absent", "aggregation"))
    assert not osp.exists(absent.lmdb_dir)

    agg = GenotypeAggregator(root=str(tmp_path))
    agg._init_lmdb(readonly=False)
    assert agg.env.path() == agg.lmdb_dir
    agg.close_lmdb()
    assert agg.env is None


def test_experiment_info_is_strict() -> None:
    """Two string fields; an extra key is refused."""
    info = ExperimentInfo(
        experiment_type="fitness", experiment_reference_type="fitness"
    )
    assert info.model_dump() == {
        "experiment_type": "fitness",
        "experiment_reference_type": "fitness",
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentInfo(
            experiment_type="fitness",
            experiment_reference_type="fitness",
            phenotype="fitness",  # type: ignore[call-arg]
        )
