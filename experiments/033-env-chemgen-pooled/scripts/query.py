# experiments/033-env-chemgen-pooled/scripts/query.py
# [[experiments.033-env-chemgen-pooled.scripts.query]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/033-env-chemgen-pooled/scripts/query
r"""Build the pooled four-dataset chemogenomic store from the served knowledge graph.

The dataset the experiment 031 planning document (notes-tex/031-unified-representation)
asks for: every environment-response record of Vanacloig 2022, Hillenmeyer 2008 HET,
Hoepfner 2014 and Wildenhain 2015 whose genes are all in the S288C gene set
(queries/001_env_chemgen_pooled.cql), one processed entry per (genotype, environment)
cell, every measurement of a cell kept side by side.

Three choices, each against the solid-growth builds this one is modeled on
(experiments/029-solid-growth-ko, experiments/030-solid-growth-multi):

1. No conversion stage. The composite converter rewrites essentiality and synthetic
   lethality to fitness; nothing here is either, so the stage would copy 90 GB byte for
   byte. ``converter=None`` runs raw -> aggregation -> processed.
2. Aggregation by the (genotype, environment) cell, not by the gene set.
   ``GenotypeAggregator`` keys on the perturbed genes alone, which for a chemogenomic
   record would put every condition of a gene into one entry (41 to 5,170 of them) and
   the label table would keep the first. ``GenotypeEnvironmentAggregator`` keys on the
   perturbations with their type and copy number, the reference ploidy, and the
   environment's content identity (torchcell/data/genotype_environment_aggregate.py).
   Within these datasets the only records that share a key are repeated screens of one
   compound at one concentration (Hoepfner's fourteen compounds run in more than one
   study); a heterozygous and a homozygous deletion of one gene are two cells.
3. No deduplication, as in 029 and 030: which measurement of a cell a trainer reads is a
   read-time policy beside the split and the seed, not a build decision.

The raw stage runs on the single-threaded ``Neo4jQueryRaw`` of main (one commit per
record), which the 025 build measured at about 630 records per second; 6.39 million
records is about three hours. The 030 branch's batched raw stage is not on main.

Stages: raw query -> aggregation -> processed, then the phenotype, perturbation-count and
dataset-name indices and the label table, and a pass over the processed store that
histograms the measurements per entry, so the multi-measurement cells are counted rather
than assumed.

Run from the repo root under slurm (scripts/gh_query_build_001.slurm):

    python experiments/033-env-chemgen-pooled/scripts/query.py

A smoke build against the live graph on a handful of genes (scripts/gh_query_smoke.slurm),
which must include Vanacloig's three sensitized-host deletions or no Vanacloig record
passes the gene filter:

    python experiments/033-env-chemgen-pooled/scripts/query.py \\
        --root <smoke root> \\
        --genes YAL004W,YBR058C,YAL001C,YAL013W,YMR263W,YGL013C,YBL005W,YDR011W
"""

import argparse
import json
import os
import os.path as osp
from collections import Counter

import lmdb
from dotenv import load_dotenv

from torchcell.data import GenotypeEnvironmentAggregator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

EXPERIMENT = "033-env-chemgen-pooled"
QUERY = f"experiments/{EXPERIMENT}/queries/001_env_chemgen_pooled.cql"
BUILD_NAME = "001-pooled-build"

#: Served record counts per dataset, from kg_manifest.json at release 2026.09.21-ab6d8c5d,
#: so the build can report what the gene filter dropped.
SERVED_COUNTS: dict[str, int] = {
    "EnvChemgenVanacloig2022Dataset": 143_218,
    "HetHillenmeyer2008Dataset": 2_698_797,
    "EnvChemgenHoepfner2014Dataset": 3_124_319,
    "EnvChemgenWildenhain2015Dataset": 428_206,
}


def measurements_per_entry(processed_lmdb: str) -> tuple[Counter[int], Counter[str]]:
    """Histogram the experiments per processed entry, and count experiments per dataset.

    The dataset-name index counts ENTRIES per dataset; this counts the experiments inside
    them, which is the number to read against ``SERVED_COUNTS``.
    """
    sizes: Counter[int] = Counter()
    experiments: Counter[str] = Counter()
    env = lmdb.open(processed_lmdb, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        for _key, value in txn.cursor():
            records = json.loads(value.decode("utf-8"))
            sizes[len(records)] += 1
            for record in records:
                experiments[record["experiment"]["dataset_name"]] += 1
    env.close()
    return sizes, experiments


def main() -> None:
    """Query, aggregate, and report the index breakdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="dataset root (default: $DATA_ROOT/data/torchcell/experiments/"
        f"{EXPERIMENT}/{BUILD_NAME})",
    )
    parser.add_argument(
        "--genes",
        default=None,
        help="comma-separated systematic gene names for a smoke build "
        "(default: the S288C genome gene set)",
    )
    args = parser.parse_args()

    load_dotenv()
    data_root = os.getenv("DATA_ROOT")
    assert data_root is not None, "DATA_ROOT must be set in .env"
    with open(QUERY) as f:
        query = f.read()

    if args.genes:
        gene_set = GeneSet(args.genes.split(","))
    else:
        # go_root must be explicit: the relative default resolves against cwd, misses
        # the mirror, and falls into a live GO download that 403s.
        genome = SCerevisiaeGenome(
            genome_root=osp.join(data_root, "data/sgd/genome"),
            go_root=osp.join(data_root, "data/go"),
        )
        gene_set = genome.gene_set
    print(f"gene_set: {len(gene_set)} genes", flush=True)

    dataset_root = args.root or osp.join(
        data_root, f"data/torchcell/experiments/{EXPERIMENT}/{BUILD_NAME}"
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=gene_set,
        graphs=None,
        incidence_graphs=None,
        node_embeddings=None,
        converter=None,
        deduplicator=None,
        aggregator=GenotypeEnvironmentAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    print(f"dataset length: {len(dataset)}", flush=True)

    sizes, experiments = measurements_per_entry(osp.join(dataset.processed_dir, "lmdb"))
    summary = {
        "length": len(dataset),
        "phenotype_label_index": {
            k: len(v) for k, v in dataset.phenotype_label_index.items()
        },
        "perturbation_count_index": {
            str(k): len(v) for k, v in dataset.perturbation_count_index.items()
        },
        "dataset_name_index": {
            k: len(v) for k, v in dataset.dataset_name_index.items()
        },
        "experiments_per_dataset": dict(experiments),
        "served_per_dataset": SERVED_COUNTS,
        "measurements_per_entry": {str(k): v for k, v in sorted(sizes.items())},
    }
    out = (
        osp.join(dataset_root, "dataset_index_summary.json")
        if args.genes
        else f"experiments/{EXPERIMENT}/results/dataset_index_summary.json"
    )
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    dataset.close_lmdb()
    print("finished")


if __name__ == "__main__":
    main()
