# src/torchcell/models/dcell.py
# [[src.torchcell.models.dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/models/dcell.py
# Test file: src/torchcell/models/test_dcell.py

from collections import OrderedDict

import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx

from torchcell.graph import (
    SCerevisiaeGraph,
    filter_by_contained_genes,
    filter_by_date,
    filter_go_IGI,
    filter_redundant_terms,
)
from torchcell.sequence import GeneSet


class SubsystemModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size  # Store the output size as an attribute
        self.linear = nn.Linear(input_size, output_size)
        self.tanh = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.tanh(x)
        x = self.batchnorm(x)
        return x


class DCell(nn.Module):
    def __init__(self, go_graph):
        super().__init__()
        self.go_graph = self.add_boolean_state(go_graph)
        self.subsystems = nn.ModuleDict()
        self.build_subsystems()

    @staticmethod
    def add_boolean_state(go_graph: nx.Graph) -> nx.Graph:
        for node_id in go_graph.nodes:
            if node_id == "GO:ROOT":
                go_graph.nodes[node_id]["mutant_state"] = torch.tensor(
                    [], dtype=torch.float32
                )
            else:
                subsystem_state_size = len(go_graph.nodes[node_id]["gene_set"])
                go_graph.nodes[node_id]["mutant_state"] = torch.ones(
                    subsystem_state_size, dtype=torch.float32
                )
        return go_graph

    def build_subsystems(self):
        # Start with a topological sort to ensure correct processing order
        nodes_sorted = list(nx.topological_sort(self.go_graph))

        for node_id in nodes_sorted:
            # Skip the 'GO:ROOT' node or handle it separately
            if node_id == "GO:ROOT":
                continue

            data = self.go_graph.nodes[node_id]
            # Use 'gene_set' to set the input size for each SubsystemModel
            if "gene_set" in data:
                gene_set = data["gene_set"]
                input_size = len(gene_set)  # The input size is the number of gene_set
                output_size = max(
                    20, int(0.3 * input_size)
                )  # Apply your original formula
                self.subsystems[node_id] = SubsystemModel(input_size, output_size)

    def calculate_input_size(self, node_id):
        # Calculate the input size based on the number of gene_set
        # If the node has children, sum their gene_set; otherwise,
        # just use this node's gene_set
        gene_set = self.go_graph.nodes[node_id].get("gene_set", [])
        return len(gene_set)

    def forward(self, batch: Batch):
        # Flatten ids if batch.id is a list of lists; otherwise, use it directly
        ids = (
            [id for sublist in batch.id for id in sublist]
            if isinstance(batch.id[0], list)
            else batch.id
        )

        # Initialize a dictionary to store the outputs of each subsystem
        subsystem_outputs = {}

        # Iterate over the subsystems in the model
        for subsystem_name, subsystem_model in self.subsystems.items():
            # Find the indices of all nodes across the batch that correspond to this subsystem
            subsystem_indices = [
                (i, node_id)
                for i, node_id in enumerate(ids)
                if node_id == subsystem_name
            ]

            # Stack the mutant states for all nodes corresponding to this subsystem
            subsystem_states = torch.stack(
                [batch.x[i][batch.mask[i].bool()].float() for i, _ in subsystem_indices]
            )

            # Pass the stacked mutant states through the subsystem model
            subsystem_output = subsystem_model(subsystem_states)

            # Store the output in the dictionary using the subsystem name as the key
            subsystem_outputs[subsystem_name] = subsystem_output

        # The final output is a dictionary of tensors, where each tensor is the output of a subsystem
        return subsystem_outputs


class DCellLinear(nn.Module):
    def __init__(self, subsystems: nn.ModuleDict, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.subsystem_linears = nn.ModuleDict()

        # Create a linear layer for each subsystem with the appropriate input size
        for subsystem_name, subsystem in subsystems.items():
            in_features = subsystem.output_size
            self.subsystem_linears[subsystem_name] = nn.Linear(
                in_features, self.output_size
            )

    def forward(self, subsystem_outputs: dict):
        # Initialize an empty tensor to store the concatenated outputs
        concatenated_outputs = torch.empty(
            0,
            self.output_size,
            device=subsystem_outputs[next(iter(subsystem_outputs))].device,
        )

        # Apply the linear transformation to each subsystem output and concatenate them
        for subsystem_name, subsystem_output in subsystem_outputs.items():
            transformed_output = self.subsystem_linears[subsystem_name](
                subsystem_output
            )
            concatenated_outputs = torch.cat(
                (concatenated_outputs, transformed_output), dim=0
            )

        return concatenated_outputs


def delete_genes(go_graph: nx.Graph, deletion_gene_set: GeneSet):
    G_mutant = go_graph.copy()
    for node in G_mutant.nodes:
        if node == "GO:ROOT":
            G_mutant.nodes[node]["mutant_state"] = torch.tensor([], dtype=torch.int32)
        else:
            gene_set = G_mutant.nodes[node]["gene_set"]
            # Replace the genes in the knockout set with 0
            G_mutant.nodes[node]["mutant_state"] = torch.tensor(
                [1 if gene not in deletion_gene_set else 0 for gene in gene_set],
                dtype=torch.int32,
            )
    return G_mutant


def dcell_from_networkx(G_mutant):
    G_mutant_copy = G_mutant.copy()

    # Initialize maximum length to zero
    max_length = 0
    # Find the maximum length of the 'mutant_state' tensor and simplify node data
    for node_id, node_data in G_mutant_copy.nodes(data=True):
        mutant_state = node_data.get(
            "mutant_state", torch.tensor([], dtype=torch.float32)
        )
        max_length = max(max_length, mutant_state.size(0))
        simplified_data = {"id": node_id, "mutant_state": mutant_state}
        G_mutant_copy.nodes[node_id].clear()
        G_mutant_copy.nodes[node_id].update(simplified_data)

    # Pad the mutant_state tensors to the maximum length and create masks
    mask_list = []
    for node_id, node_data in G_mutant_copy.nodes(data=True):
        mutant_state = node_data["mutant_state"]
        mask = torch.ones(max_length, dtype=torch.uint8)
        if mutant_state.size(0) < max_length:
            padding = max_length - mutant_state.size(0)
            padded_mutant_state = torch.cat([mutant_state, torch.full((padding,), -1)])
            node_data["mutant_state"] = padded_mutant_state
            # Update the mask to indicate which entries are actual data
            mask[-padding:] = 0
        mask_list.append(mask)

    # Convert the NetworkX graph to a PyTorch Geometric Data object
    data = from_networkx(G_mutant_copy)
    data.x = torch.stack(
        [node_data["mutant_state"] for _, node_data in G_mutant_copy.nodes(data=True)]
    )
    data.mask = torch.stack(mask_list)
    del data.mutant_state
    return data


def main():
    import os
    import os.path as osp
    import random

    import matplotlib.pyplot as plt
    import pandas as pd
    from dotenv import load_dotenv

    from torchcell.datasets.scerevisiae import (
        DmfCostanzo2016Dataset,
        SmfCostanzo2016Dataset,
    )
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    dmf_dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e5"),
        preprocess={"duplicate_resolution": "low_dmf_std"},
        # subset_n=100,
    )
    smf_dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_smf"),
        preprocess={"duplicate_resolution": "low_std_both"},
        skip_process_file_exist_check=True,
    )
    gene_set = smf_dataset.gene_set.union(dmf_dataset.gene_set)
    #
    print(graph.G_go.number_of_nodes())
    G = graph.G_go.copy()

    # Filtering
    G = filter_by_date(G, "2017-07-19")
    print(f"After date filter: {G.number_of_nodes()}")
    G = filter_go_IGI(G)
    print(f"After IGI filter: {G.number_of_nodes()}")
    G = filter_redundant_terms(G)
    print(f"After redundant filter: {G.number_of_nodes()}")
    G = filter_by_contained_genes(G, n=2, gene_set=gene_set)
    print(f"After containment filter: {G.number_of_nodes()}")

    # Instantiate the model
    dcell = DCell(go_graph=G)

    print(dcell)
    print(dcell)

    G_mutant = delete_genes(
        go_graph=dcell.go_graph, deletion_gene_set=GeneSet(("YDL029W", "YDR150W"))
    )
    print()
    # forward method
    G_mutant = dcell_from_networkx(G_mutant)

    batch = Batch.from_data_list([G_mutant, G_mutant])
    subsystem_outputs = dcell(batch)
    dcell_linear = DCellLinear(dcell.subsystems, output_size=2)
    output = dcell_linear(subsystem_outputs)
    print(output.size)
    print()


if __name__ == "__main__":
    main()
