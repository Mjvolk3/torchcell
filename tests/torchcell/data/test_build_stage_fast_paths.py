"""Fast paths of the dataset build stages.

- conversion: records whose experiment type the converter cannot change are copied
  byte for byte (the 025 build parsed, validated, hashed twice and re-serialized
  43.8 million records for a few thousand conversions);
- raw: query results are committed to LMDB in batches, and the gene set is
  accumulated during the stream rather than by a second pass;
- aggregation: the grouping key is read off the stored bytes, guarded against the
  full parse on the first records.
"""

import json
import os
import os.path as osp
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lmdb
import pytest

from torchcell.data.genotype_aggregate import (
    DeletionKeyedGenotypeAggregator,
    GenotypeAggregator,
)
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneEssentialityExperiment,
    GeneEssentialityExperimentReference,
    GeneEssentialityPhenotype,
    Genotype,
    Media,
    ReferenceGenome,
    SgaDampPerturbation,
    SgaKanMxDeletionPerturbation,
    Temperature,
)


def _environment() -> Environment:
    return Environment(
        media=Media(name="YEPD", state="solid", is_synthetic=False),
        temperature=Temperature(value=30.0),
    )


def _genome() -> ReferenceGenome:
    return ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _deletions(genes: list[str]) -> Genotype:
    return Genotype(
        perturbations=[
            SgaKanMxDeletionPerturbation(
                systematic_gene_name=g,
                perturbed_gene_name=g.lower(),
                strain_id=f"{g}_dma1",
            )
            for g in genes
        ]
    )


def _fitness_record(genes: list[str], fitness: float) -> dict[str, Any]:
    experiment = FitnessExperiment(
        dataset_name="DmfTest",
        genotype=_deletions(genes),
        environment=_environment(),
        phenotype=FitnessPhenotype(fitness=fitness, fitness_std=0.01),
    )
    reference = FitnessExperimentReference(
        dataset_name="DmfTest",
        genome_reference=_genome(),
        environment_reference=_environment(),
        phenotype_reference=FitnessPhenotype(fitness=1.0, fitness_std=0.01),
    )
    return {"experiment": experiment, "experiment_reference": reference}


def _essentiality_record(gene: str) -> dict[str, Any]:
    experiment = GeneEssentialityExperiment(
        dataset_name="EssTest",
        genotype=_deletions([gene]),
        environment=_environment(),
        phenotype=GeneEssentialityPhenotype(is_essential=True),
    )
    reference = GeneEssentialityExperimentReference(
        dataset_name="EssTest",
        genome_reference=_genome(),
        environment_reference=_environment(),
        phenotype_reference=GeneEssentialityPhenotype(is_essential=False),
    )
    return {"experiment": experiment, "experiment_reference": reference}


def _dump(record: dict[str, Any]) -> bytes:
    return json.dumps(
        {
            "experiment": record["experiment"].model_dump(),
            "experiment_reference": record["experiment_reference"].model_dump(),
        }
    ).encode()


def _write_lmdb(path: str, items: list[tuple[bytes, bytes]]) -> None:
    os.makedirs(path, exist_ok=True)
    env = lmdb.open(path, map_size=1 << 30)
    with env.begin(write=True) as txn:
        for k, v in items:
            txn.put(k, v)
    env.close()


def _read_all(path: str) -> dict[bytes, bytes]:
    env = lmdb.open(path, readonly=True, lock=False)
    with env.begin() as txn:
        out = {bytes(k): bytes(v) for k, v in txn.cursor()}
    env.close()
    return out


# --------------------------------------------------------------------------- conversion
def test_conversion_copies_unconvertible_records_byte_for_byte(tmp_path: Path) -> None:
    fitness = _dump(_fitness_record(["YAL001C", "YAL002W"], 0.85))
    essential = _dump(_essentiality_record("YAL003W"))
    raw = str(tmp_path / "raw" / "lmdb")
    _write_lmdb(raw, [(b"data_0", fitness), (b"data_1", essential)])

    converter = CompositeFitnessConverter(root=str(tmp_path), query=None)  # type: ignore[arg-type]
    assert converter.convertible_experiment_types == {
        "gene essentiality",
        "synthetic lethality",
    }
    out = str(tmp_path / "conversion" / "lmdb")
    converter.process(raw, out)

    got = _read_all(out)
    assert got[b"data_0"] == fitness, (
        "an unconvertible record must pass through unchanged"
    )
    converted = json.loads(got[b"data_1"].decode())
    assert converted["experiment"]["experiment_type"] == "fitness"
    assert converted["experiment"]["phenotype"]["fitness"] == 0.0
    assert converted["experiment_reference"]["experiment_reference_type"] == "fitness"
    assert converted["experiment_reference"]["phenotype_reference"]["fitness"] == 1.0


def test_experiment_type_is_read_off_the_bytes() -> None:
    value = _dump(_fitness_record(["YAL001C"], 0.9))
    assert CompositeFitnessConverter._experiment_type_of(value) == "fitness"
    assert CompositeFitnessConverter._experiment_type_of(b"{}") is None


# --------------------------------------------------------------------------- aggregation
def test_aggregate_key_bytes_matches_the_parsed_key() -> None:
    agg = GenotypeAggregator(root="/tmp/unused")
    records = [
        _fitness_record(["YAL001C", "YAL002W"], 0.85),
        _fitness_record(["YAL002W", "YAL001C"], 0.70),  # same set, other order
        _essentiality_record("YAL003W"),
    ]
    keys_bytes = [agg.aggregate_key_bytes(_dump(r)) for r in records]
    keys_raw = [agg.aggregate_key_raw(json.loads(_dump(r).decode())) for r in records]
    assert keys_bytes == keys_raw
    assert keys_bytes[0] == keys_bytes[1] and keys_bytes[0] != keys_bytes[2]


def test_deletion_keyed_aggregator_parses_because_its_key_needs_the_type() -> None:
    experiment = FitnessExperiment(
        dataset_name="DmfTest",
        genotype=Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAL001C",
                    perturbed_gene_name="tfc3",
                    strain_id="d1",
                ),
                SgaDampPerturbation(
                    systematic_gene_name="YAL003W",
                    perturbed_gene_name="efb1",
                    strain_id="d2",
                ),
            ]
        ),
        environment=_environment(),
        phenotype=FitnessPhenotype(fitness=0.5, fitness_std=0.01),
    )
    record = {
        "experiment": experiment,
        "experiment_reference": _fitness_record(["YAL001C"], 1.0)[
            "experiment_reference"
        ],
    }
    value = _dump(record)
    deletion_keyed = DeletionKeyedGenotypeAggregator(root="/tmp/unused")
    assert deletion_keyed.aggregate_key_bytes(
        value
    ) == deletion_keyed.aggregate_key_raw(json.loads(value.decode()))
    # the gene-set key sees both genes, the deletion key only the deletion
    assert deletion_keyed.aggregate_key_bytes(value) != GenotypeAggregator(
        root="/tmp/unused"
    ).aggregate_key_bytes(value)


def test_aggregation_groups_records_by_gene_set(tmp_path: Path) -> None:
    items = [
        (b"data_0", _dump(_fitness_record(["YAL001C", "YAL002W"], 0.85))),
        (b"data_1", _dump(_fitness_record(["YAL002W", "YAL001C"], 0.70))),
        (b"data_2", _dump(_fitness_record(["YAL005C"], 0.95))),
    ]
    src = str(tmp_path / "conversion" / "lmdb")
    _write_lmdb(src, items)
    agg = GenotypeAggregator(root=str(tmp_path))
    out = str(tmp_path / "aggregation" / "lmdb")
    agg.process(src, out)
    groups = {k: json.loads(v.decode()) for k, v in _read_all(out).items()}
    assert len(groups) == 2
    sizes = sorted(len(g) for g in groups.values())
    assert sizes == [1, 2]


def test_aggregation_guard_stops_a_wrong_byte_reader(tmp_path: Path) -> None:
    class Broken(GenotypeAggregator):
        def aggregate_key_bytes(self, value: bytes) -> str:
            return "not-the-key"

    src = str(tmp_path / "conversion" / "lmdb")
    _write_lmdb(src, [(b"data_0", _dump(_fitness_record(["YAL001C"], 0.9)))])
    with pytest.raises(ValueError, match="disagrees"):
        Broken(root=str(tmp_path)).process(src, str(tmp_path / "aggregation" / "lmdb"))


# --------------------------------------------------------------------------- raw
def test_raw_stage_commits_in_batches_and_accumulates_the_gene_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n = 2 * Neo4jQueryRaw.WRITE_BATCH + 17  # three commits, the last one partial
    genes = [f"YAL00{k}C" for k in range(7)]

    def fake_fetch(self: Neo4jQueryRaw) -> Iterator[dict[str, str]]:
        for i in range(n):
            rec = _fitness_record([genes[i % 7], genes[(i + 1) % 7]], 0.5)
            yield {
                "e_serialized": json.dumps(rec["experiment"].model_dump()),
                "ref_serialized": json.dumps(rec["experiment_reference"].model_dump()),
            }

    monkeypatch.setattr(Neo4jQueryRaw, "fetch_data", fake_fetch)
    raw = Neo4jQueryRaw(
        uri="bolt://unused",
        username="u",
        password="p",
        root_dir=str(tmp_path),
        query="unused",
    )
    assert len(raw) == n
    last = raw[n - 1]
    assert last["experiment"].phenotype.fitness == 0.5
    assert set(raw.gene_set) == set(genes)
    assert osp.exists(osp.join(tmp_path, "raw", "gene_set.json"))
    # the reference index is lazy now: nothing on disk until a caller asks
    assert not osp.exists(osp.join(tmp_path, "raw", "experiment_reference_index.json"))
    raw.close_lmdb()
