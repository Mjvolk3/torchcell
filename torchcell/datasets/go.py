import datetime
import os
import os.path as osp
import pickle
from abc import ABC, abstractmethod

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from goatools.obo_parser import GODag
from sortedcontainers import SortedSet
from torch_geometric.data import download_url

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

import torchcell

style_file_path = osp.join(osp.dirname(torchcell.__file__), 'torchcell.mplstyle')
plt.style.use(style_file_path)

class GoPlot(ABC):
    @abstractmethod
    def plot(self):
        raise NotImplementedError("Subclasses must implement plot() method.")

    def save(self, as_pdf=False, as_pickle=False):
        current_dir = osp.dirname(osp.abspath(__file__))
        timestamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        file_path = osp.join(
            ".notes/assets/images",
            osp.relpath(current_dir),
            f"{self.__class__.__name__}-{timestamp}",
        )
        dir_name = osp.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f"{file_path}.png", format="png", bbox_inches="tight")
        if as_pdf:
            plt.savefig(f"{file_path}.pdf", format="pdf", bbox_inches="tight")
        if as_pickle:
            with open(f"{file_path}.pkl", "wb") as f:
                pickle.dump(plt.gcf(), f)


class GOTermGraph:
    def __init__(self, gene_set, add_fantasy_root=False):
        self.gene_set = gene_set
        self.graph = nx.DiGraph()

        data_dir = "data/go"
        obo_path = "data/go/go.obo"
        if not osp.exists(obo_path):
            os.makedirs(data_dir, exist_ok=True)
            download_url("http://current.geneontology.org/ontology/go.obo", data_dir)
            # os.rename(f"{data_dir}/go.obo", obo_path)

        self.godag = GODag(obo_path)

        for term in gene_set:
            # Add a check here
            if term in self.godag:
                self.graph.add_node(term, category=self.godag[term].namespace)
                parents = self.getAllParents(term)
                self.graph.add_edges_from([(term, parent) for parent in parents])
            else:
                print(f"Warning: {term} not found in GO DAG.")

        if add_fantasy_root:
            self.root = "FantasyRoot"
            self.graph.add_node(self.root, category="root")
            for node in self.graph.nodes:
                if self.graph.out_degree(node) == 0 and node != self.root:
                    self.graph.add_edge(node, self.root)
        else:
            self.root = None

    def getAllParents(self, term):
        term_obj = self.godag[term]
        return term_obj.get_all_parents()

    def removeTerm(self, term):
        if term in self.graph:
            self.graph.remove_node(term)


class SimpleTreePlot(GoPlot):
    def __init__(self, go_graph):
        self.go_graph = go_graph

    def plot(self):
        if self.go_graph.root == "FantasyRoot":
            circular_pos = nx.circular_layout(self.go_graph.graph)
            circular_pos[self.go_graph.root] = (0, 0)
        else:
            circular_pos = nx.circular_layout(self.go_graph.graph)

        color_map = []
        for node in self.go_graph.graph.nodes(data=True):
            category = node[1].get("category", "")
            if category == "biological_process":
                color_map.append("green")
            elif category == "cellular_component":
                color_map.append("red")
            elif category == "molecular_function":
                color_map.append("blue")
            elif category == "root":
                color_map.append("purple")
            else:
                color_map.append("lightgray")

        nx.draw(
            self.go_graph.graph,
            pos=circular_pos,
            with_labels=True,
            node_color=color_map,
            arrows=True,
        )

        # Add Legend
        legend_labels = [
            mpatches.Patch(color="green", label="Biological Process"),
            mpatches.Patch(color="red", label="Cellular Component"),
            mpatches.Patch(color="blue", label="Molecular Function"),
            mpatches.Patch(color="purple", label="Fantasy Root"),
        ]
        plt.legend(handles=legend_labels)

    def show(self):
        plt.show()


def main():
    import random

    genome = SCerevisiaeGenome()

    # Take a random sample for testing
    random.seed(0)
    keys_list = list(genome.go_genes.keys())
    sampled_keys = random.sample(keys_list, 10)
    sampled_dict = {key: genome.go_genes[key] for key in sampled_keys}

    go_graph = GOTermGraph(gene_set=sampled_dict, add_fantasy_root=True)
    plotter = SimpleTreePlot(go_graph)
    plotter.plot()
    plotter.show()


if __name__ == "__main__":
    main()
