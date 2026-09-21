# experiments/030-solid-growth-multi/scripts/query.py
# [[experiments.030-solid-growth-multi.scripts.query]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/query
"""Build the all-allele, multi-measurement solid-growth dataset from the served graph.

030 is the 025 query with the 029 build. The query decides what the dataset IS: the gene
universe, the phenotypes, the medium, the record shape. It does not decide which
measurement of a genotype counts, because that is a modeling assumption and belongs with
the run, beside the split and the seed. A build that has already chosen has thrown the
alternatives away, and the store is the expensive thing to make.

Against 025 (experiments/025-solid-growth/scripts/query.py): no deduplication stage.
025's MeanExperimentDeduplicator merged every entry sharing an experiment type and a gene
set into one mean and replaced the source p-values with a t-test over the duplicates.
Here every source entry survives with its own fitness, standard deviation, score,
p-value, temperature, marker and screen, and the value a trainer reads comes from a
hashed label policy at read time.

Against 029 (experiments/029-solid-growth-ko/scripts/query.py): no perturbation-type
filter. All alleles restores the 376,732 triple identities of 010 and 025, so the pinned
random split and the query-pair-disjoint split transfer by gene-set identity exactly, and
deletions-only becomes a subset index over this build rather than a separate build.

The measurement that motivated 030 needed no query change. The double-mutant query
strain's fitness, which every Kuzmin trigenic row reports and which carries the trigenic
score, is double-mutant fitness; it now enters through DmfKuzmin2018Dataset and
DmfKuzmin2020Dataset, which had the code and never ran it (main 4c4a4f950). Those two
datasets must be re-served before this build runs, or 030 is 025 without the merge.

Stages: raw query -> conversion (essentiality and synthetic lethality to fitness 0, every
other record copied byte for byte) -> aggregation -> processed, then the phenotype,
perturbation-count and dataset-name indices and the label table.

Run from the repo root under slurm (scripts/gh_query_build_001.slurm):

    python experiments/030-solid-growth-multi/scripts/query.py

A smoke build against the live graph on a handful of genes:

    python experiments/030-solid-growth-multi/scripts/query.py \\
        --root /tmp/030-smoke --genes YAL002W,YAL004W,YBR001C
"""

import argparse
import json
import os
import os.path as osp

from dotenv import load_dotenv

from torchcell.data import GenotypeAggregator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

EXPERIMENT = "030-solid-growth-multi"
QUERY = f"experiments/{EXPERIMENT}/queries/001_multi_measurement.cql"
BUILD_NAME = "001-multi-build"


def main() -> None:
    """Query, convert, aggregate, and report the index breakdown."""
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
        converter=CompositeFitnessConverter,
        deduplicator=None,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    print(f"dataset length: {len(dataset)}", flush=True)

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
