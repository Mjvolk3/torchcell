# tests/torchcell/graph/test_graph.py
# [[tests.torchcell.graph.test_graph]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/graph/test_graph.py
# Test file: tests/torchcell/graph/test_test_graph.py

import os
from datetime import datetime

import networkx as nx
import pytest
from dotenv import load_dotenv
from tqdm import tqdm

# Assuming you've imported your necessary classes/methods at the top
from torchcell.graph import SCerevisiaeGraph, filter_by_date, filter_go_IGI
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


@pytest.fixture
def get_sample_graph() -> nx.DiGraph:
    # TODO should I be setting up test/data
    """Fixture to generate a sample graph for testing."""
    genome = SCerevisiaeGenome(
        data_root=os.path.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    graph = SCerevisiaeGraph(
        data_root=os.path.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    return graph.G_go


def check_no_IGI_in_graph(G: nx.DiGraph) -> bool:
    """Utility function to check if any node in G has a gene with the 'IGI' display name."""
    for node in tqdm(G.nodes()):
        if "genes" in G.nodes[node]:
            for v in G.nodes[node]["genes"].values():
                if v["go_details"]["experiment"]["display_name"] == "IGI":
                    # Found an IGI
                    return False
    # No IGI found
    return True


def test_filter_go_IGI(get_sample_graph):
    G_sample = get_sample_graph

    G_filtered = filter_go_IGI(G_sample)

    assert check_no_IGI_in_graph(
        G_filtered
    ), "Found 'IGI' display name in the filtered graph."


def check_no_genes_after_date(G: nx.DiGraph, cutoff_date: str) -> bool:
    """Utility function to check if any node in G has a gene with a date after the cutoff."""
    for node in tqdm(G.nodes()):
        if "genes" in G.nodes[node]:
            for v in G.nodes[node]["genes"].values():
                gene_date = datetime.strptime(
                    v["go_details"]["date_created"], "%Y-%m-%d"
                )
                cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")
                if gene_date > cutoff:
                    # Found a gene after the cutoff date
                    return False
    # No genes found after the cutoff date
    return True


def test_filter_by_date(get_sample_graph):
    G_sample = get_sample_graph
    cutoff_date = "2018-02-01"

    G_filtered = filter_by_date(G_sample, cutoff_date)

    assert check_no_genes_after_date(
        G_filtered, cutoff_date
    ), f"Found genes annotated after {cutoff_date} in the filtered graph."


# if __name__ == "__main__":
#     test_filter_go_IGI(get_sample_graph)
