# experiments/029-solid-growth-ko/scripts/query.py
# [[experiments.029-solid-growth-ko.scripts.query]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/029-solid-growth-ko/scripts/query
"""Build the deletion-only solid-growth dataset from the served knowledge graph.

The 029 build differs from 025 (experiments/025-solid-growth/scripts/query.py) in
three ways, all decided from the S3 closure recompute of 2026-09-18
(notes-tex/025-s3-closure):

1. The query keeps only records whose perturbations are all deletions
   (queries/001_ko_solid_growth.cql), so no temperature-sensitive, DAmP or suppressor
   allele shares a gene name with a deletion, and no essentiality 0 can be averaged
   into a measured allele fitness.
2. There is NO deduplication stage. 025's MeanExperimentDeduplicator merged every
   entry sharing an experiment type and a gene set into one mean, and replaced the
   source p-values with a t-test over the duplicates. Here the GenotypeAggregator
   groups the raw entries per genotype and every source entry survives with its own
   fitness, standard deviation, score, p-value, temperature and screen; which entry
   becomes the training label is a label policy applied at read time, not a build
   decision.
3. The dataset root lives on /db (a build is regenerable) behind a symlink under
   $DATA_ROOT, as 025's did.

Stages: raw query -> conversion (essentiality and synthetic lethality to fitness 0,
every other record copied byte for byte) -> aggregation -> processed, then the
phenotype, perturbation-count and dataset-name indices and the label table.

Run from the repo root under slurm (scripts/gh_query_build_001.slurm):

    python experiments/029-solid-growth-ko/scripts/query.py

A smoke build against the live graph on a handful of genes:

    python experiments/029-solid-growth-ko/scripts/query.py \\
        --root /tmp/029-smoke --genes YAL002W,YAL004W,YBR001C,...
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

EXPERIMENT = "029-solid-growth-ko"
QUERY = f"experiments/{EXPERIMENT}/queries/001_ko_solid_growth.cql"
BUILD_NAME = "001-ko-build"


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
    print(f"gene_set: {len(gene_set)} genes")

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
    print(f"dataset length: {len(dataset)}")

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
