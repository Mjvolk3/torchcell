# experiments/043-expression-proteome-morphology/scripts/build_043_store.py
# [[experiments.043-expression-proteome-morphology.scripts.build_043_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/build_043_store
"""Build the experiment 043 raw store: one sample per served experiment, nothing merged.

The query (``queries/expression_proteome_morphology_043.cql``, from
``generate_043_cql.py``) unions 15 expression, proteome and morphology datasets of KG 4.0.
The owner's rule for this first query is to keep the data as served: no converter, no
deduplicator, no genotype aggregation. Each served ``(Experiment, ExperimentReference)``
pair becomes one sample.

``RecordAggregator`` and why it is here: on main (v1.8.4) ``Neo4jCellDataset`` with
``aggregator=None`` copies the raw LMDB into ``processed/`` unchanged, keys ``data_<i>``
with one JSON object per value, while every processed-store reader (``get``,
``compute_phenotype_info``, the three index passes) expects keys ``<i>`` with a JSON
list per value. Measured offline on a one-record LMDB: ``get(0)`` reads ``None`` and
``compute_phenotype_info`` raises ``TypeError: string indices must be integers``, so the
build would die after the full raw fetch. ``RecordAggregator`` keys every record by the
sha256 of its own content, so each group holds exactly one record and the processed store
is the raw records, one per entry, in the list format the readers expect. The script
asserts that the number of processed entries equals the number of raw records; a
collision (two byte-identical served records) fails the build instead of merging.

The ``aggregator=None`` defect is a torchcell bug, to be fixed separately and not in the
043 PR: on main ``aggregator=None`` (with no deduplicator) produces a processed store
that no reader can open.

Genome, graphs and the ``calm`` node embedding are set up as in the 019
``build_fig3_proteome.py`` build. The Neo4j connection resolves from ``NEO4J_URI`` /
``NEO4J_USER`` / ``NEO4J_PASSWORD`` in ``.env`` through
``torchcell.database.connection.neo4j_connection_settings`` (local served instance when
unset); only the URI is recorded.

Writes ``results/build_043_store_census.json``: build time, ``len(dataset)``, raw and
processed entry counts, records per dataset (raw LMDB scan and ``dataset_name_index``),
records per phenotype label, and for the first sample of every dataset the
``Perturbation`` processor's output (or the exception it raised).

Run from the repo root (under slurm on GilaHyper, ``gh_build_043_store.slurm``):
    python experiments/043-expression-proteome-morphology/scripts/build_043_store.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import socket
import time
from collections import Counter
from typing import Any

import lmdb
from dotenv import load_dotenv

load_dotenv()

from torchcell.data import Neo4jCellDataset  # noqa: E402
from torchcell.data.aggregate import Aggregator  # noqa: E402
from torchcell.data.graph_processor import Perturbation  # noqa: E402
from torchcell.database.connection import neo4j_connection_settings  # noqa: E402
from torchcell.datamodels.schema import (  # noqa: E402
    ExperimentReferenceType,
    ExperimentType,
)
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"
DATASET_TAG = "raw_043"
QUERY_NAME = "expression_proteome_morphology_043.cql"
GRAPHS = [
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
]


def _record_key(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class RecordAggregator(Aggregator):
    """Identity grouping: every record is its own group, keyed by its content hash."""

    def aggregate_check(
        self, data: dict[str, ExperimentType | ExperimentReferenceType]
    ) -> str:
        """The content hash of the record's ``model_dump``."""
        return _record_key({k: v.model_dump() for k, v in data.items()})

    def aggregate_key_raw(self, record: dict[str, Any]) -> str:
        """The content hash of the stored record."""
        return _record_key(record)


def _lmdb_entries(path: str) -> int:
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        n = int(txn.stat()["entries"])
    env.close()
    return n


def _raw_dataset_counts(path: str) -> dict[str, int]:
    """Records per ``dataset_name`` in the raw LMDB (one JSON object per value)."""
    counts: Counter[str] = Counter()
    env = lmdb.open(path, readonly=True, lock=False, subdir=True, readahead=True)
    with env.begin() as txn:
        for _, value in txn.cursor():
            counts[json.loads(value)["experiment"]["dataset_name"]] += 1
    env.close()
    return dict(sorted(counts.items()))


def _first_record(path: str, index: int) -> dict[str, Any]:
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        (record,) = json.loads(txn.get(f"{index}".encode()))
    env.close()
    return record


def _describe(dataset: Neo4jCellDataset, index: int, record: dict[str, Any]) -> dict:
    """What the processed record holds and what ``dataset[index]`` returns."""
    experiment = record["experiment"]
    phenotype = experiment["phenotype"]
    label = phenotype["label_name"]
    value = phenotype.get(label)
    # A SegregantGenotype has no perturbation list.
    perts = experiment["genotype"].get("perturbations") or []
    summary: dict[str, Any] = {
        "index": index,
        "dataset_name": experiment["dataset_name"],
        "experiment_type": experiment["experiment_type"],
        "genotype_fields": sorted(experiment["genotype"]),
        "label_name": label,
        "label_length": len(value) if isinstance(value, (dict, list)) else None,
        "n_perturbations": len(perts),
        "perturbation_types": dict(
            Counter(p.get("perturbation_type") for p in perts).most_common()
        ),
        "media": experiment["environment"]["media"]["name"],
        # A typed ProvenanceGap can stand in for the temperature, so the object is kept.
        "temperature": experiment["environment"]["temperature"],
    }
    try:
        item = dataset[index]
    except Exception as exc:  # recorded, not swallowed: the census reports it
        summary["get_error"] = f"{type(exc).__name__}: {exc}"
        return summary
    summary["get_repr"] = repr(item)
    summary["perturbation_indices_length"] = len(item["gene"].perturbation_indices)
    summary["pert_mask_sum"] = int(item["gene"].pert_mask.sum())
    summary["phenotype_types"] = list(item["gene"].phenotype_types)
    return summary


def main() -> None:
    """Build the store and write the census."""
    data_root = os.environ["DATA_ROOT"]
    dataset_root = osp.join(
        data_root, "data/torchcell/experiments", EXPERIMENT, DATASET_TAG
    )
    here = osp.dirname(osp.abspath(__file__))
    query_path = osp.abspath(osp.join(here, "..", "queries", QUERY_NAME))
    with open(query_path) as f:
        query = f.read()
    uri = neo4j_connection_settings().uri

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )
    t0 = time.time()
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=build_gene_multigraph(graph=graph, graph_names=GRAPHS),
        incidence_graphs=None,
        node_embeddings=NodeEmbeddingBuilder.build(
            embedding_names=["calm"], data_root=data_root, genome=genome, graph=graph
        ),
        converter=None,
        deduplicator=None,
        aggregator=RecordAggregator,
        graph_processor=Perturbation(),
        transform=None,
    )
    build_seconds = time.time() - t0
    print(f"len(dataset) = {len(dataset)}  ({build_seconds / 60:.1f} min)")

    raw_path = osp.join(dataset_root, "raw", "lmdb")
    processed_path = osp.join(dataset_root, "processed", "lmdb")
    raw_n = _lmdb_entries(raw_path)
    processed_n = _lmdb_entries(processed_path)
    if raw_n != processed_n:
        raise RuntimeError(
            f"processed store has {processed_n} entries for {raw_n} raw records: "
            "RecordAggregator merged byte-identical records"
        )

    name_index = dataset.dataset_name_index
    first_by_dataset = {name: min(idx) for name, idx in sorted(name_index.items())}
    samples = {
        name: _describe(dataset, index, _first_record(processed_path, index))
        for name, index in first_by_dataset.items()
    }
    census = {
        "generated_by": (f"experiments/{EXPERIMENT}/scripts/build_043_store.py"),
        "dataset_root": dataset_root,
        "query_file": query_path,
        "neo4j_uri": uri,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "build_seconds": build_seconds,
        "len_dataset": len(dataset),
        "n_raw_records": raw_n,
        "n_processed_records": processed_n,
        "n_records_by_dataset_raw": _raw_dataset_counts(raw_path),
        "n_records_by_dataset_index": {
            k: len(v) for k, v in sorted(name_index.items())
        },
        "n_records_by_label": {
            k: len(v) for k, v in sorted(dataset.phenotype_label_index.items())
        },
        "first_sample": samples[min(first_by_dataset, key=first_by_dataset.get)],
        "first_sample_by_dataset": samples,
    }
    out = osp.join(
        experiment_results_dir(EXPERIMENT, __file__), "build_043_store_census.json"
    )
    with open(out, "w") as f:
        json.dump(census, f, indent=1)
    print(json.dumps(census, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
