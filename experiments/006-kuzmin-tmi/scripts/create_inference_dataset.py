#!/usr/bin/env python
"""
Example script showing how to create an InferenceDataset from triple combinations.
"""

import os
import os.path as osp
import sys
from dotenv import load_dotenv

# Add the current script's directory to Python path
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from inference_dataset import InferenceDataset
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.data.graph_processor import SubgraphRepresentation

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


def main():
    # Setup genome
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_chrmt()
    genome.drop_empty_go()

    # Setup graph
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Build gene multigraph using the same graphs as in hetero_cell_bipartite_dango_gi.py
    graph_names = [
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
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)

    # Build node embeddings using NodeEmbeddingBuilder
    # Using learnable embeddings as in the hetero_cell_bipartite_dango_gi.py config
    node_embeddings = NodeEmbeddingBuilder.build(
        embedding_names=["learnable"], data_root=DATA_ROOT, genome=genome, graph=graph
    )

    # Setup incidence graphs for metabolism
    incidence_graphs = {}
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    incidence_graphs["metabolism_bipartite"] = yeast_gem.bipartite_graph

    # Create dataset for inference - use relative path
    inference_root = osp.join(DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/inference_0")
    dataset = InferenceDataset(
        root=inference_root,
        gene_set=genome.gene_set,
        graphs=gene_multigraph,
        incidence_graphs=incidence_graphs,
        node_embeddings=node_embeddings,
        graph_processor=SubgraphRepresentation(),
    )

    # Load triple combinations from the raw subdirectory
    raw_dir = osp.join(inference_root, "raw")
    
    # Look for the most recent triple_combinations_list file
    if osp.exists(raw_dir):
        triple_files = [f for f in os.listdir(raw_dir) if f.startswith("triple_combinations_list_") and f.endswith(".txt")]
        if triple_files:
            # Get the most recent file
            latest_file = sorted(triple_files)[-1]
            triple_csv = osp.join(raw_dir, latest_file)
            print(f"Found triple combinations file: {triple_csv}")
        else:
            print(f"No triple combinations files found in {raw_dir}")
            print("Please run generate_triple_combinations.py first")
            return
    else:
        print(f"Raw directory not found at {raw_dir}")
        print("Please run generate_triple_combinations.py first")
        return

    # Convert CSV to experiments and load to LMDB
    experiments = dataset._load_from_csv(triple_csv)
    print(f"Loaded {len(experiments)} experiments from CSV")

    # Store in LMDB
    dataset.load_experiments_to_lmdb(experiments)

    # Test the dataset
    print(f"Dataset size: {len(dataset)}")
    print("=================")
    print(f"First data sample: {dataset.cell_graph}")
    print("=================")
    print(f"First data sample: {dataset[0]}")

    # Close LMDB
    dataset.close_lmdb()


if __name__ == "__main__":
    main()
