# experiments/040-inhibitor-synergy-wetlab/scripts/build_040_store.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.build_040_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/build_040_store
"""Build the experiment 040 store from KG 4.0: one entry per served record, with a manifest.

The query (``queries/env_chemgen_mixture_040.cql``, written by ``generate_040_cql.py``)
unions the four pooled chemogenomic screens the mixture trainer reads and the private
2021 Bioscreen wells. The store is a ``Neo4jCellDataset`` under
``$DATA_ROOT/data/torchcell/experiments/040-inhibitor-synergy-wetlab/<tag>``; the
trainer's data layer (``mixture_data.py``) reads its ``processed/lmdb`` and nothing else,
and the three hand-carried inputs the Delta launcher used to ship (the 033 cell table, the
031 embedding tables, the Vanacloig dev store) are replaced by this one directory.

WHAT ONE ENTRY IS. The owner's rule for a first query is to keep the data as served, so
there is no converter, no deduplicator and no genotype aggregation: each served
``(Experiment, ExperimentReference)`` pair is one entry, keyed by the sha256 of its own
content (``RecordAggregator``, the 043 build's device). The reason it is an aggregator at
all: ``Neo4jCellDataset`` with ``aggregator=None`` copies the raw LMDB unchanged, keys
``data_<i>`` with one JSON object per value, while every processed-store reader expects
keys ``<i>`` with a JSON LIST per value, so a store built without one cannot be read
(a torchcell defect, measured by the 043 build, not fixed here). The 033 build folded
repeated measurements of one (genotype, environment) cell into one entry with its own
aggregator; here that fold is the trainer's read-time label policy
(``mixture_data.load_gene_source``: ``single`` asserts one measurement per cell,
``mean`` averages), which is where 033's own post-query analysis said it belongs.

THE MANIFEST. ``processed/store_manifest.json`` (``StoreManifest``) records what the
store was built from and what it holds: the release id and version read from the
``KgRelease`` node of the store it was queried from, the torchcell commit and version the
build imported, the query file and its sha256, the gene set size, the records returned per
dataset against the counts the release serves and against what each block's clause admits
(a count run on the server with the same clause text; the build asserts returned equals
admitted, so the gap to the served count is the clause's exclusion, listed per perturbed
gene outside the gene set), the processed entry count, and the sha256
and byte size of ``processed/lmdb/data.mdb``. The launchers verify that sha256 before a
training job starts, which is what turns a damaged copy (Delta's Lustre target 106 left
about one file in a hundred unreadable on 2026-10-16, and a job that touched one hung to
its 12 h limit) into a preflight failure instead of a lost allocation.

The build runs on the release's own commit (``TORCHCELL_SRC`` in the slurm launcher
points ``PYTHONPATH`` at the ``build/kg-4-0-v1-8-0`` worktree, 1.8.3), the package the
release is paired with; ``Neo4jQueryRaw`` refuses a release the installed schema does not
reproduce. The Neo4j connection resolves from ``NEO4J_URI`` / ``NEO4J_USER`` /
``NEO4J_PASSWORD`` in ``.env`` (the local served instance when unset); only the URI is
recorded.

A smoke build is the same script with ``TC040_GENES`` set to a gene list (separated by
spaces or semicolons: sbatch splits ``--export`` on commas, so a comma-separated list
arrives as its first gene) and ``TC040_STORE_TAG`` set to a scratch tag; the private block carries no gene clause and
returns all 977 wells regardless.

Run under slurm on GilaHyper (``gh_build_040_store.slurm``), from the repository root.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import re
import socket
import subprocess
import time
from collections import Counter
from typing import Any

import lmdb
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

from generate_040_cql import (  # noqa: E402
    BLOCKS,
    RELEASE,
    SERVED_COUNTS,
    count_query,
    empty_genotype_query,
    outside_gene_set_query,
    query_path,
)
from store_manifest_040 import (  # noqa: E402
    DatasetCount,
    StoreManifest,
    manifest_path,
    sha256_file,
)

from torchcell.data import Neo4jCellDataset  # noqa: E402
from torchcell.data.aggregate import Aggregator  # noqa: E402
from torchcell.data.graph_processor import Perturbation  # noqa: E402
from torchcell.database.connection import neo4j_connection_settings  # noqa: E402
from torchcell.datamodels.schema import (  # noqa: E402
    ExperimentReferenceType,
    ExperimentType,
)
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.knowledge_graphs import releases  # noqa: E402
from torchcell.sequence import GeneSet  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

EXPERIMENT = "040-inhibitor-synergy-wetlab"
STORE_TAG = os.environ.get("TC040_STORE_TAG", "mixture_040")
#: The raw stage partitioned over this many fetch processes, 256 e.id-prefix partitions
#: per UNION ALL block; measured at 0.307 ms per record end to end on the Hoepfner block
#: (tcdb-002, slurm 3138) against 4.39 ms single-session.
FETCH_WORKERS = int(os.environ.get("TC040_FETCH_WORKERS", "8"))
PARTITION_PREFIX_LENGTH = 2
#: The gene graphs the trainer's cell graph is built from (``conf/mixture/cgt_model.yaml``);
#: the store carries them so ``dataset[i]`` is a cell for the per-record census.
GRAPHS = ["physical", "regulatory", "tflink"]


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


def _clause_counts(
    uri: str, username: str, password: str, database: str, gene_set: GeneSet
) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, int]]:
    """Per block: what its clause admits, the genes it excludes, the empty genotypes.

    The same clause text the query carries (``generate_040_cql.count_query``), run as a
    count against the same store, so the number the raw stage is held to was produced by
    the server and not by this script.
    """
    genes = sorted(gene_set)
    admitted: dict[str, int] = {}
    outside: dict[str, dict[str, int]] = {}
    empty: dict[str, int] = {}
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            for dataset, _, clause in BLOCKS:
                t0 = time.time()
                record = session.run(
                    count_query(dataset, clause), gene_set=genes
                ).single()
                assert record is not None
                admitted[dataset] = int(record["n"])
                outside[dataset] = {
                    r["gene"]: int(r["n"])
                    for r in session.run(
                        outside_gene_set_query(dataset), gene_set=genes
                    )
                }
                record = session.run(empty_genotype_query(dataset)).single()
                assert record is not None
                empty[dataset] = int(record["n"])
                print(
                    f"{dataset}: clause admits {admitted[dataset]:,}, "
                    f"{len(outside[dataset])} perturbed genes outside the gene set, "
                    f"{empty[dataset]:,} empty genotypes ({time.time() - t0:.0f} s)",
                    flush=True,
                )
    finally:
        driver.close()
    return admitted, outside, empty


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
    perts = experiment["genotype"].get("perturbations") or []
    small = [
        p
        for p in experiment["environment"]["perturbations"]
        if p["perturbation_type"] == "small_molecule"
    ]
    summary: dict[str, Any] = {
        "index": index,
        "dataset_name": experiment["dataset_name"],
        "experiment_type": experiment["experiment_type"],
        "label_name": phenotype["label_name"],
        "n_perturbations": len(perts),
        "perturbation_types": dict(
            Counter(p.get("perturbation_type") for p in perts).most_common()
        ),
        "compounds": [
            {
                "name": p["compound"]["name"],
                "inchikey": p["compound"]["inchikey"],
                "smiles": p["compound"]["smiles"],
                "concentration": p["concentration"],
            }
            for p in small
        ],
        "media": experiment["environment"]["media"]["name"],
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


def _git_head(path: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    """Build the store, write the manifest and the census."""
    import torchcell

    expected_src = os.environ.get("TORCHCELL_SRC")
    torchcell_src = osp.dirname(osp.dirname(osp.abspath(torchcell.__file__)))
    assert expected_src is None or torchcell_src == osp.abspath(expected_src), (
        f"torchcell resolved from {torchcell_src}, not TORCHCELL_SRC {expected_src}"
    )
    data_root = os.environ["DATA_ROOT"]
    dataset_root = osp.join(
        data_root, "data/torchcell/experiments", EXPERIMENT, STORE_TAG
    )
    path = query_path()
    with open(path) as f:
        query = f.read()
    settings = neo4j_connection_settings()
    database = releases.resolve_database(
        settings.version, settings.uri, settings.username, settings.password
    )
    release = releases.read_release(
        settings.uri, settings.username, settings.password, database
    )
    assert release is not None, f"{settings.uri}/{database} carries no KgRelease node"
    assert release.release == RELEASE, (
        f"the served store is release {release.release}; the query is written against "
        f"{RELEASE} (generate_040_cql.RELEASE): regenerate the query and its served "
        "counts against the served release before building"
    )

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    smoke = os.environ.get("TC040_GENES")
    smoke_genes = (
        None if not smoke else sorted(set(g for g in re.split(r"[,;\s]+", smoke) if g))
    )
    gene_set = genome.gene_set if smoke_genes is None else GeneSet(smoke_genes)
    print(f"gene_set: {len(gene_set)} genes", flush=True)
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
        gene_set=gene_set,
        graphs=build_gene_multigraph(graph=graph, graph_names=GRAPHS),
        incidence_graphs=None,
        node_embeddings=None,
        converter=None,
        deduplicator=None,
        aggregator=RecordAggregator,
        graph_processor=Perturbation(),
        transform=None,
        fetch_workers=FETCH_WORKERS,
        partition_prefix_length=PARTITION_PREFIX_LENGTH,
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
    returned = _raw_dataset_counts(raw_path)
    unexpected = sorted(set(returned) - set(SERVED_COUNTS))
    assert not unexpected, f"the query returned datasets it does not name: {unexpected}"
    admitted, outside, empty = _clause_counts(
        settings.uri, settings.username, settings.password, database, gene_set
    )
    counts = [
        DatasetCount(
            dataset=name,
            served=served,
            admitted=admitted[name],
            returned=returned.get(name, 0),
            dropped=served - returned.get(name, 0),
            outside_gene_set=outside[name],
            empty_genotype=empty[name],
        )
        for name, served in SERVED_COUNTS.items()
    ]
    for c in counts:
        print(
            f"{c.dataset}: returned {c.returned:,} of {c.served:,} served, "
            f"clause admits {c.admitted:,}",
            flush=True,
        )
    lost = [c for c in counts if c.returned != c.admitted]
    assert not lost, (
        "the raw stage returned a different count than the clause admits in the "
        "served graph (records lost or duplicated between the partitions): "
        + ", ".join(f"{c.dataset} {c.returned:,}/{c.admitted:,}" for c in lost)
    )

    data_mdb = osp.join(processed_path, "data.mdb")
    manifest = StoreManifest(
        experiment=EXPERIMENT,
        store_tag=STORE_TAG,
        dataset_root=dataset_root,
        built_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        host=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        build_seconds=build_seconds,
        neo4j_uri=settings.uri,
        kg_release=release.release,
        kg_version=release.version,
        kg_torchcell_version=release.torchcell_version,
        torchcell_src=torchcell_src,
        torchcell_commit=_git_head(torchcell_src),
        torchcell_version=torchcell.__version__,
        query_file=path,
        query_sha256=hashlib.sha256(query.encode()).hexdigest(),
        gene_set_size=len(gene_set),
        smoke_genes=smoke_genes,
        counts=counts,
        n_raw_records=raw_n,
        n_processed_entries=processed_n,
        data_mdb_bytes=osp.getsize(data_mdb),
        data_mdb_sha256=sha256_file(data_mdb),
    )
    with open(manifest_path(dataset_root), "w") as f:
        f.write(manifest.model_dump_json(indent=1))
    print(f"wrote {manifest_path(dataset_root)}", flush=True)

    name_index = dataset.dataset_name_index
    first_by_dataset = {name: min(idx) for name, idx in sorted(name_index.items())}
    census = {
        "generated_by": f"experiments/{EXPERIMENT}/scripts/build_040_store.py",
        "manifest": manifest.model_dump(),
        "n_records_by_dataset_index": {
            k: len(v) for k, v in sorted(name_index.items())
        },
        "n_records_by_label": {
            k: len(v) for k, v in sorted(dataset.phenotype_label_index.items())
        },
        "first_sample_by_dataset": {
            name: _describe(dataset, index, _first_record(processed_path, index))
            for name, index in first_by_dataset.items()
        },
    }
    here = osp.dirname(osp.abspath(__file__))
    results = osp.join(here, "..", "results")
    out = osp.abspath(osp.join(results, f"build_040_store_census_{STORE_TAG}.json"))
    with open(out, "w") as f:
        json.dump(census, f, indent=1)
    print(json.dumps(census["first_sample_by_dataset"], indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
