# tests/torchcell/datamodels/test_conversion.py
# [[tests.torchcell.datamodels.test_conversion]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_conversion.py
"""The abstract ``Converter`` LMDB stage through its concrete fitness converters.

``Converter`` only stores the ``Neo4jQueryRaw`` it is handed, so a sentinel ``object()``
stands in for it and no Neo4j is touched. The raw store holds records under ``data_0``
onward, each ``json.dumps({"experiment": model_dump(), "experiment_reference":
model_dump()})``, and ``process`` writes its output under the SAME keys.

Expected values:

* ``_compute_hash({"b": 1, "a": [1, 2]})`` hashes the sorted-key dump ``{"a": [1, 2],
  "b": 1}``: ``3813f9409eb0095bf9c8e1225c627acf867d89bd18b8856fc38588a0c8c490f3``
  (``hashlib.sha256`` in a shell).
* An essential-gene record converts to a ``FitnessExperiment`` with fitness 0.0 and no
  std, keeping its dataset name, genotype and environment; its reference becomes a
  ``FitnessExperimentReference`` with fitness 1.0.
* A record no entry matches passes through unchanged, extra keys included.
* A non-essential record makes the conversion function return None, which ``convert``
  refuses as a TypeError; ``process`` catches that, logs it and skips the record, so the
  documented "excluded from the converted dataset" happens through the error path.
"""

import json
import logging
import os.path as osp
from pathlib import Path
from typing import Any

import lmdb
import pytest
from pydantic import ValidationError

from torchcell.datamodels.conversion import ConversionEntry, ConversionMap, Converter
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.datamodels.gene_essentiality_to_fitness_conversion import (
    GeneEssentialityToFitnessConverter,
    gene_essentiality_to_fitness_experiment,
    gene_essentiality_to_fitness_reference,
)
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneEssentialityExperiment,
    GeneEssentialityExperimentReference,
    GeneEssentialityPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ReferenceGenome,
    SyntheticLethalityExperiment,
    SyntheticLethalityExperimentReference,
    SyntheticLethalityPhenotype,
)

ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
QUERY: Any = object()  # Converter stores it and reads nothing from it
FIXED_HASH = "3813f9409eb0095bf9c8e1225c627acf867d89bd18b8856fc38588a0c8c490f3"


def _genotype(genes: list[str]) -> Genotype:
    return Genotype(
        perturbations=[
            KanMxDeletionPerturbation(systematic_gene_name=g, perturbed_gene_name=g)
            for g in genes
        ]
    )


def _essentiality(gene: str, essential: bool) -> dict[str, Any]:
    return {
        "experiment": GeneEssentialityExperiment(
            dataset_name="sgd",
            genotype=_genotype([gene]),
            environment=ENVIRONMENT,
            phenotype=GeneEssentialityPhenotype(is_essential=essential),
        ),
        "experiment_reference": GeneEssentialityExperimentReference(
            dataset_name="sgd",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=GeneEssentialityPhenotype(is_essential=False),
        ),
    }


def _fitness(gene: str, fitness: float) -> dict[str, Any]:
    return {
        "experiment": FitnessExperiment(
            dataset_name="toy",
            genotype=_genotype([gene]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=fitness),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name="toy",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0),
        ),
    }


def _synthetic_lethal(genes: list[str]) -> dict[str, Any]:
    return {
        "experiment": SyntheticLethalityExperiment(
            dataset_name="synlethdb",
            genotype=_genotype(genes),
            environment=ENVIRONMENT,
            phenotype=SyntheticLethalityPhenotype(is_synthetic_lethal=True),
        ),
        "experiment_reference": SyntheticLethalityExperimentReference(
            dataset_name="synlethdb",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=SyntheticLethalityPhenotype(is_synthetic_lethal=False),
        ),
    }


def _converted_essential(gene: str, dataset: str = "sgd") -> dict[str, Any]:
    """The fitness pair an essential-gene record converts to, built by hand."""
    return {
        "experiment": FitnessExperiment(
            dataset_name=dataset,
            genotype=_genotype([gene]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=0.0, fitness_std=None),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name=dataset,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0, fitness_std=None),
        ),
    }


def _dump(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v.model_dump() for k, v in record.items()}


def _write_raw(path: str, raws: list[bytes]) -> None:
    env = lmdb.open(path, map_size=int(1e8))
    with env.begin(write=True) as txn:
        for i, raw in enumerate(raws):
            txn.put(f"data_{i}".encode(), raw)
    env.close()


def _read_all(path: str) -> dict[bytes, bytes]:
    env = lmdb.open(path, readonly=True, lock=False)
    with env.begin() as txn:
        stored = {bytes(k): bytes(v) for k, v in txn.cursor()}
    env.close()
    return stored


def test_compute_hash_is_sha256_of_the_sorted_key_dump() -> None:
    """Key order does not enter the hash; the value does."""
    assert Converter._compute_hash({"b": 1, "a": [1, 2]}) == FIXED_HASH
    assert Converter._compute_hash({"a": [1, 2], "b": 1}) == FIXED_HASH
    assert Converter._compute_hash({"a": [1, 2], "b": 2}) != FIXED_HASH


def test_conversion_map_holds_typed_entries_and_refuses_extras() -> None:
    """The essentiality converter's map has one entry naming the four types."""
    converter = GeneEssentialityToFitnessConverter(root="unused", query=QUERY)
    (entry,) = converter.conversion_map.entries
    assert entry.experiment_input_type is GeneEssentialityExperiment
    assert entry.experiment_output_type is FitnessExperiment
    assert entry.experiment_reference_input_type is GeneEssentialityExperimentReference
    assert entry.experiment_reference_output_type is FitnessExperimentReference
    assert (
        entry.experiment_conversion_function is gene_essentiality_to_fitness_experiment
    )
    assert (
        entry.experiment_reference_conversion_function
        is gene_essentiality_to_fitness_reference
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ConversionMap(entries=[entry], name="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ConversionEntry(**entry.model_dump(), extra=1)  # type: ignore[call-arg]


def test_converter_stores_root_and_query_and_derives_the_lmdb_dir(
    tmp_path: Path,
) -> None:
    """Finding: ``__repr__`` is the literal ``Converter(root=...)`` for every subclass."""
    converter = GeneEssentialityToFitnessConverter(root=str(tmp_path), query=QUERY)
    assert converter.root == str(tmp_path)
    assert converter.query is QUERY
    assert converter.lmdb_dir == osp.join(str(tmp_path), "conversion", "lmdb")
    assert converter.env is None
    assert repr(converter) == f"Converter(root={tmp_path})"
    assert len(converter) == 0
    assert bool(converter) is False


def test_convert_maps_an_essential_gene_to_fitness_zero() -> None:
    """dataset, genotype and environment carry over; fitness 0.0, reference fitness 1.0."""
    converter = GeneEssentialityToFitnessConverter(root="unused", query=QUERY)
    out = converter.convert(_essentiality("YAL001C", True))
    assert type(out["experiment"]) is FitnessExperiment
    assert type(out["experiment_reference"]) is FitnessExperimentReference
    assert _dump(out) == _dump(_converted_essential("YAL001C"))


def test_convert_passes_an_unmatched_record_through_with_its_extra_keys() -> None:
    """A fitness record matches no entry, so the same objects come back, plus extras."""
    converter = GeneEssentialityToFitnessConverter(root="unused", query=QUERY)
    record = _fitness("YAL002W", 0.7)
    record["note"] = "kept"
    out = converter.convert(record)
    assert out["experiment"] is record["experiment"]
    assert out["experiment_reference"] is record["experiment_reference"]
    note: Any = out["note"]
    assert note == "kept"
    assert list(out) == ["experiment", "experiment_reference", "note"]


def test_convert_refuses_a_missing_key_and_a_none_result() -> None:
    """Finding: a non-essential gene is documented as "excluded", but ``convert`` raises
    ``TypeError`` because the entry's function returns None, which is not the declared
    output type (``conversion.py:124``). Only ``process`` turns that into a skip.
    """
    converter = GeneEssentialityToFitnessConverter(root="unused", query=QUERY)
    with pytest.raises(
        ValueError,
        match="Input data must contain both 'experiment' and 'experiment_reference'",
    ):
        converter.convert({"experiment": _fitness("YAL001C", 0.5)["experiment"]})
    with pytest.raises(
        TypeError,
        match="Conversion function did not return expected type for experiment",
    ):
        converter.convert(_essentiality("YAL003W", False))


def test_process_converts_in_place_and_skips_records_that_fail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Four inputs: essential (converted), fitness (unchanged bytes), non-essential
    (skipped through the TypeError), corrupt JSON (skipped through JSONDecodeError).
    Output keeps the input keys; the counts are logged as 1 converted of 2 processed.
    """
    raws = [
        json.dumps(_dump(_essentiality("YAL001C", True))).encode(),
        json.dumps(_dump(_fitness("YAL002W", 0.7))).encode(),
        json.dumps(_dump(_essentiality("YAL003W", False))).encode(),
        b"{not json",
    ]
    raw = str(tmp_path / "raw")
    _write_raw(raw, raws)
    converter = GeneEssentialityToFitnessConverter(root=str(tmp_path), query=QUERY)
    with caplog.at_level(logging.ERROR, logger="torchcell.datamodels.conversion"):
        converter.process(raw, converter.lmdb_dir)

    stored = _read_all(converter.lmdb_dir)
    assert sorted(stored) == [b"data_0", b"data_1"]
    assert json.loads(stored[b"data_0"]) == _dump(_converted_essential("YAL001C"))
    assert stored[b"data_1"] == raws[1]
    assert converter.env is None
    assert caplog.messages == [
        "Error processing entry 2: Conversion function did not return expected type "
        "for experiment. Skipping this entry.",
        "Error decoding JSON for entry 3. Skipping this entry.",
    ]


def test_process_logs_the_conversion_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One of two records changes hash, so 1 converted and 2 processed."""
    raws = [
        json.dumps(_dump(_essentiality("YAL001C", True))).encode(),
        json.dumps(_dump(_fitness("YAL002W", 0.7))).encode(),
    ]
    raw = str(tmp_path / "raw")
    _write_raw(raw, raws)
    converter = GeneEssentialityToFitnessConverter(root=str(tmp_path), query=QUERY)
    with caplog.at_level(logging.INFO, logger="torchcell.datamodels.conversion"):
        converter.process(raw, converter.lmdb_dir)
    module_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "torchcell.datamodels.conversion"
    ]
    assert module_messages == [
        f"Conversion complete. LMDB database written to {converter.lmdb_dir}",
        "Number of instances converted: 1",
        "Total number of instances processed: 2",
    ]


@pytest.fixture
def processed(tmp_path: Path) -> GeneEssentialityToFitnessConverter:
    """A converter whose store holds ``data_0`` (converted) and ``data_1`` (fitness)."""
    raws = [
        json.dumps(_dump(_essentiality("YAL001C", True))).encode(),
        json.dumps(_dump(_fitness("YAL002W", 0.7))).encode(),
    ]
    raw = str(tmp_path / "raw")
    _write_raw(raw, raws)
    converter = GeneEssentialityToFitnessConverter(root=str(tmp_path), query=QUERY)
    converter.process(raw, converter.lmdb_dir)
    return converter


def test_getitem_int_slice_and_list(
    processed: GeneEssentialityToFitnessConverter,
) -> None:
    """Records come back as pydantic objects under ``data_<i>`` keys."""
    first = processed[0]
    assert isinstance(first, dict)
    assert type(first["experiment"]) is FitnessExperiment
    assert _dump(first) == _dump(_converted_essential("YAL001C"))
    both = processed[0:2]
    assert isinstance(both, list)
    assert [_dump(r) for r in both] == [
        _dump(_converted_essential("YAL001C")),
        _dump(_fitness("YAL002W", 0.7)),
    ]
    reversed_pair = processed[[1, 0]]
    assert isinstance(reversed_pair, list)
    assert [_dump(r) for r in reversed_pair] == [
        _dump(_fitness("YAL002W", 0.7)),
        _dump(_converted_essential("YAL001C")),
    ]
    assert processed[5:7] == []
    assert len(processed) == 2
    assert bool(processed) is True


def test_getitem_errors(processed: GeneEssentialityToFitnessConverter) -> None:
    """A string index is a TypeError, a missing key an IndexError."""
    with pytest.raises(TypeError, match=r"Invalid index type: <class 'str'>"):
        processed["0"]  # type: ignore[index]
    with pytest.raises(IndexError, match="Record not found at index: 9"):
        processed[9]
    processed.close_lmdb()
    with pytest.raises(ValueError, match="LMDB environment is not initialized."):
        processed._get_record_by_index(0)
    with pytest.raises(ValueError, match="LMDB environment is not initialized."):
        processed._get_records_by_slice(slice(0, 1))


def test_init_lmdb_readonly_needs_an_existing_store(tmp_path: Path) -> None:
    """Readonly on an absent store leaves ``env`` None; writable creates the store."""
    absent = GeneEssentialityToFitnessConverter(
        root=str(tmp_path / "absent"), query=QUERY
    )
    absent._init_lmdb(readonly=True)
    assert absent.env is None
    assert osp.isdir(osp.join(str(tmp_path), "absent", "conversion"))
    assert not osp.exists(absent.lmdb_dir)

    converter = GeneEssentialityToFitnessConverter(root=str(tmp_path), query=QUERY)
    converter._init_lmdb(readonly=False)
    env = converter.env
    assert env is not None  # narrows lmdb.Environment | None for the path check
    assert env.path() == converter.lmdb_dir
    converter.close_lmdb()
    assert converter.env is None


def test_composite_converter_tries_essentiality_then_synthetic_lethality() -> None:
    """Two entries; an essential gene and a lethal pair both become fitness 0.0; a
    fitness record is returned unchanged.
    """
    composite = CompositeFitnessConverter(root="unused", query=QUERY)
    assert [e.experiment_input_type for e in composite.conversion_map.entries] == [
        GeneEssentialityExperiment,
        SyntheticLethalityExperiment,
    ]
    essential = composite.convert(_essentiality("YAL001C", True))
    assert _dump(essential) == _dump(_converted_essential("YAL001C"))

    lethal = composite.convert(_synthetic_lethal(["YAL001C", "YAL002W"]))
    expected = {
        "experiment": FitnessExperiment(
            dataset_name="synlethdb",
            genotype=_genotype(["YAL001C", "YAL002W"]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=0.0, fitness_std=None),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name="synlethdb",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0, fitness_std=None),
        ),
    }
    assert _dump(lethal) == _dump(expected)

    record = _fitness("YAL003W", 0.9)
    unchanged = composite.convert(record)
    assert unchanged == record
