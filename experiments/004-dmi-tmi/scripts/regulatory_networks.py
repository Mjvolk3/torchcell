# experiments/004-dmi-tmi/scripts/regulatory_networks
# [[experiments.004-dmi-tmi.scripts.regulatory_networks]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/004-dmi-tmi/scripts/regulatory_networks
# Test file: experiments/004-dmi-tmi/scripts/test_regulatory_networks.py

import os
import os.path as osp
from dotenv import load_dotenv
import pandas as pd
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph import SCerevisiaeGraph

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


def inspect_regulatory_edges(graph, num_samples=10):
    """
    Print data from a sample of regulatory edges

    Args:
        graph: SCerevisiaeGraph instance
        num_samples: Number of edges to sample
    """
    regulatory_graph = graph.G_regulatory

    # Get total edge count
    total_edges = regulatory_graph.number_of_edges()
    print(f"Total regulatory edges: {total_edges}")

    # Collect unique sources and annotation types
    sources = {}
    annotation_types = {}
    experiments = {}

    # Collect edge data sample
    edge_samples = []

    # Iterate through edges
    for i, (source, target, data) in enumerate(regulatory_graph.edges(data=True)):
        # Count sources
        if "source" in data and "display_name" in data["source"]:
            source_name = data["source"]["display_name"]
            sources[source_name] = sources.get(source_name, 0) + 1

        # Count annotation types
        if "annotation_type" in data:
            annot_type = data["annotation_type"]
            annotation_types[annot_type] = annotation_types.get(annot_type, 0) + 1

        # Count experiment types
        if "experiment" in data and "display_name" in data["experiment"]:
            exp_name = data["experiment"]["display_name"]
            experiments[exp_name] = experiments.get(exp_name, 0) + 1

        # Save some sample edges for detailed inspection
        if i < num_samples:
            edge_sample = {
                "source_gene": source,
                "target_gene": target,
                "edge_data": data,
            }
            edge_samples.append(edge_sample)

    # Print source statistics
    print("\nSources of regulatory interactions:")
    for source_name, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"{source_name}: {count} interactions ({count/total_edges*100:.1f}%)")

    # Print annotation type statistics
    print("\nAnnotation types for regulatory interactions:")
    for annot_type, count in sorted(
        annotation_types.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{annot_type}: {count} interactions ({count/total_edges*100:.1f}%)")

    # Print experiment type statistics
    print("\nExperiment types for regulatory interactions:")
    for exp_name, count in sorted(
        experiments.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        print(f"{exp_name}: {count} interactions ({count/total_edges*100:.1f}%)")

    # Print detailed information for a few sample edges
    print(f"\nDetailed data for {num_samples} sample regulatory edges:")
    for i, edge in enumerate(edge_samples):
        print(f"\nEdge {i+1}: {edge['source_gene']} → {edge['target_gene']}")

        # Extract and print the most important fields
        data = edge["edge_data"]

        # Reference info
        if "reference" in data:
            ref = data["reference"]
            print(
                f"  Reference: {ref.get('display_name', 'N/A')}, PubMed ID: {ref.get('pubmed_id', 'N/A')}"
            )

        # Experimental method
        if "experiment" in data and "display_name" in data["experiment"]:
            print(f"  Experiment: {data['experiment']['display_name']}")

        # Regulation type
        if "regulation_type" in data:
            print(f"  Regulation type: {data['regulation_type']}")

        # Annotation type
        if "annotation_type" in data:
            print(f"  Annotation type: {data['annotation_type']}")

        # Source
        if "source" in data and "display_name" in data["source"]:
            print(f"  Source: {data['source']['display_name']}")


def main():
    # Load genome and graph
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    # Inspect regulatory edges
    inspect_regulatory_edges(graph, num_samples=5)


if __name__ == "__main__":
    main()
