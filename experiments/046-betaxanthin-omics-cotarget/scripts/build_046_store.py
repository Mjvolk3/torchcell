# experiments/046-betaxanthin-omics-cotarget/scripts/build_046_store.py
# [[experiments.046-betaxanthin-omics-cotarget.scripts.build_046_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/046-betaxanthin-omics-cotarget/scripts/build_046_store
"""Build the experiment 046 store: betaxanthin, amino acids, knockout expression and
knockout proteome from KG 4.0, one sample per served experiment, nothing merged.

The query (``queries/betaxanthin_omics_046.cql``, from ``generate_046_cql.py``) unions
Cachera 2023, Mulleder 2016, Kemmeren 2014 and Messner 2023. The rule for this store is the
one the owner set for 043's first query: keep the data as served. No converter, no
deduplicator, no genotype aggregation; a strain Messner measured twice is two records, and a
deletion that Cachera and Messner both made is two records of two experiments with the same
genotype, which is what a model that predicts both phenotypes from the genotype trains on.
This differs from the 019 ``fig6_build`` the 031 program trains on, which averaged replicate
experiments (``MeanExperimentDeduplicator``) and merged a genotype's datasets into one
sample (``GenotypeAggregator``); a later experiment that needs that form builds it from this
query with those two classes and says so.

``RecordAggregator`` keys every record by the sha256 of its own content, so each group
holds exactly one record and the processed store is the raw records in the list format the
readers expect (``aggregator=None`` on main writes a processed store no reader can open;
043 measured it). The script asserts processed entries == raw records.

The graph processor is this experiment's ``ResolvingPerturbation``
(``resolving_perturbation_046.py``): a strain whose source released a gene SYMBOL sits on
the released-identifier leaf with the symbol as its name (3,592 of Cachera's 4,759), and
the stock processor would give it no perturbed gene; the resolving processor maps the
symbol to its R64 ORF at read time and lists every refusal on the record. The census counts
the outcomes per dataset from the raw records (``resolution_by_dataset``).

Genome and the nine gene graphs are set up as in 043's build; the metabolism bipartite
incidence graph (Yeast9 through ``YeastGEM``) is attached as the 031 program's
``fig6_build`` attached it, because the betaxanthin block reads reaction and metabolite
nodes. Node embeddings are built at open time and not stored, so the build attaches one
(``calm``) only to exercise the path. The Neo4j connection resolves from ``NEO4J_URI`` /
``NEO4J_USER`` / ``NEO4J_PASSWORD`` in ``.env`` through
``torchcell.database.connection.neo4j_connection_settings``; only the URI is recorded.

Writes ``results/build_046_store_census.json``: build time, record counts per dataset (raw
LMDB scan and ``dataset_name_index``) beside the served counts of the release, records per
phenotype label, and for the first record of every dataset what the processed record holds
and what ``dataset[i]`` returns.

Run from the repo root (under slurm on GilaHyper, ``gh_build_046_store.slurm``):
    python experiments/046-betaxanthin-omics-cotarget/scripts/build_046_store.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import socket
import sys
import time
from collections import Counter
from typing import Any

import lmdb
from dotenv import load_dotenv

load_dotenv()

from torchcell.data import Neo4jCellDataset  # noqa: E402
from torchcell.data.aggregate import Aggregator  # noqa: E402
from torchcell.database.connection import neo4j_connection_settings  # noqa: E402
from torchcell.datamodels.schema import (  # noqa: E402
    ExperimentReferenceType,
    ExperimentType,
)
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from generate_046_cql import QUERY_NAME, SERVED_COUNTS  # noqa: E402
from resolving_perturbation_046 import (  # noqa: E402
    RULE_REFUSED,
    ReleasedNameResolver,
    ResolvingPerturbation,
)

EXPERIMENT = "046-betaxanthin-omics-cotarget"
DATASET_TAG = "raw_046"
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
#: Partitioned raw stage: 4 forked fetch workers, one hex character of ``e.id`` per
#: partition (16 partitions a block). 15,620 served experiments is small; the workers
#: exercise the marker, not the clock.
FETCH_WORKERS = 4


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


def _resolution_census(path: str, resolver: ReleasedNameResolver) -> dict[str, Any]:
    """Resolution outcomes per dataset, from the raw LMDB.

    Records, records whose every deletion name is a gene node or resolved, records with a
    refused name, and the rule counts over perturbations.
    """
    per: dict[str, Counter[str]] = {}
    refused_names: dict[str, Counter[str]] = {}
    env = lmdb.open(path, readonly=True, lock=False, subdir=True, readahead=True)
    with env.begin() as txn:
        for _, value in txn.cursor():
            experiment = json.loads(value)["experiment"]
            name = experiment["dataset_name"]
            c = per.setdefault(name, Counter())
            c["n_records"] += 1
            refused_here = False
            for p in experiment["genotype"].get("perturbations") or []:
                gene = p["systematic_gene_name"]
                if gene in resolver.gene_set:
                    c["perturbations_gene_node"] += 1
                    continue
                if not (
                    p["perturbation_type"] == "released_identifier"
                    or p["perturbation_type"].endswith("deletion")
                ):
                    c["perturbations_other_type"] += 1
                    continue
                orf, rule, _ = resolver.resolve(gene)
                c[f"perturbations_{rule}"] += 1
                if orf is None:
                    refused_here = True
                    refused_names.setdefault(name, Counter())[gene] += 1
            c["n_records_with_refusal" if refused_here else "n_records_resolved"] += 1
    env.close()
    return {
        name: {
            **dict(sorted(c.items())),
            "refused_names": dict(refused_names.get(name, {})),
        }
        for name, c in sorted(per.items())
    }


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
    perts = experiment["genotype"].get("perturbations") or []
    summary: dict[str, Any] = {
        "index": index,
        "dataset_name": experiment["dataset_name"],
        "experiment_type": experiment["experiment_type"],
        "label_name": label,
        "graph_level": phenotype.get("graph_level"),
        "label_length": len(value) if isinstance(value, (dict, list)) else None,
        "n_perturbations": len(perts),
        "perturbation_types": dict(
            Counter(p.get("perturbation_type") for p in perts).most_common()
        ),
        "media": experiment["environment"]["media"]["name"],
        "temperature": experiment["environment"]["temperature"],
    }
    item = dataset[index]
    summary["get_repr"] = repr(item)
    summary["perturbation_indices_length"] = len(item["gene"].perturbation_indices)
    summary["pert_mask_sum"] = int(item["gene"].pert_mask.sum())
    summary["phenotype_types"] = list(item["gene"].phenotype_types)
    summary["unresolved_perturbations"] = list(item["gene"].unresolved_perturbations)
    summary["resolution_rules"] = dict(item["gene"].resolution_rules)
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
    processor = ResolvingPerturbation(genome)
    t0 = time.time()
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=build_gene_multigraph(graph=graph, graph_names=GRAPHS),
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings=NodeEmbeddingBuilder.build(
            embedding_names=["calm"], data_root=data_root, genome=genome, graph=graph
        ),
        converter=None,
        deduplicator=None,
        aggregator=RecordAggregator,
        graph_processor=processor,
        transform=None,
        fetch_workers=FETCH_WORKERS,
    )
    build_seconds = time.time() - t0
    print(f"len(dataset) = {len(dataset)}  ({build_seconds / 60:.1f} min)", flush=True)

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
    returned = {k: len(v) for k, v in sorted(name_index.items())}
    resolution = _resolution_census(raw_path, processor.resolver)
    n_refused = sum(
        c.get(f"perturbations_{RULE_REFUSED}", 0) for c in resolution.values()
    )
    census = {
        "generated_by": f"experiments/{EXPERIMENT}/scripts/build_046_store.py",
        "dataset_root": dataset_root,
        "query_file": query_path,
        "neo4j_uri": uri,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "fetch_workers": FETCH_WORKERS,
        "build_seconds": build_seconds,
        "len_dataset": len(dataset),
        "n_raw_records": raw_n,
        "n_processed_records": processed_n,
        "n_records_by_dataset_raw": _raw_dataset_counts(raw_path),
        "n_records_by_dataset_index": returned,
        "n_experiments_served_release_2026_10_17": SERVED_COUNTS,
        "n_dropped_by_gene_clause": {
            k: SERVED_COUNTS[k] - returned.get(k, 0) for k in SERVED_COUNTS
        },
        "resolution_by_dataset": resolution,
        "n_perturbations_refused": n_refused,
        "n_records_by_label": {
            k: len(v) for k, v in sorted(dataset.phenotype_label_index.items())
        },
        "first_sample_by_dataset": samples,
    }
    out = osp.join(
        experiment_results_dir(EXPERIMENT, __file__), "build_046_store_census.json"
    )
    with open(out, "w") as f:
        json.dump(census, f, indent=1)
    print(json.dumps(census, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
