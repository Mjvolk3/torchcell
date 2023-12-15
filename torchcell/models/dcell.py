# torchcell/models/dcell.py
# [[torchcell.models.dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/dcell.py
# Test file: torchcell/models/test_dcell.py

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
    def __init__(
        self,
        go_graph,
        subsystem_output_min: int = 20,
        subsystem_output_max_mult: float = 0.3,
    ):
        super().__init__()
        # HACK probably should reverse the edges in original graph
        go_graph = nx.reverse(go_graph, copy=True)
        self.subsystem_output_min = subsystem_output_min
        self.subsystem_output_max_mult = subsystem_output_max_mult
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
        nodes_sorted = list(nx.topological_sort(self.go_graph))

        current_nodes = []
        # Initialize subsystems based on descendants
        for node_id in nodes_sorted:
            descendants = set(nx.descendants(self.go_graph, node_id))
            # If there are no descendants, this is a true leaf node
            if not descendants:
                genes = self.go_graph.nodes[node_id]["gene_set"]
                self.subsystems[node_id] = SubsystemModel(
                    input_size=len(genes),
                    output_size=max(
                        self.subsystem_output_min,
                        int(self.subsystem_output_max_mult * len(genes)),
                    ),
                )
                current_nodes.append(node_id)

                # Process non-leaf nodes in reverse topological order
        # Reverse order to move up from leaf nodes towards root node
        for node_id in reversed(nodes_sorted):
            if node_id in self.subsystems:
                # Skip if subsystem is already initialized (as a leaf)
                continue

            # TODO pick up here with nodes parent to leaf nodes
            # Calculate the input size as the sum of the output sizes of the child subsystems
            children_output_sizes = sum(
                self.subsystems[child].output_size
                for child in self.go_graph.successors(node_id)
            )
            # Add the boolean state vector size for the current node
            # make empty gene set
            genes = self.go_graph.nodes[node_id].get("gene_set", [])
            total_input_size = children_output_sizes + len(genes)

            # Calculate the output size with a minimum of 20
            output_size = max(
                self.subsystem_output_min,
                int(self.subsystem_output_max_mult * len(genes)),
            )

            # Initialize the subsystem for the current node
            self.subsystems[node_id] = SubsystemModel(
                input_size=total_input_size, output_size=output_size
            )

    def calculate_input_size(self, node_id):
        # Sum output sizes of child subsystems and the boolean state vector
        input_size_from_children = sum(
            self.subsystems[child].output_size
            for child in self.go_graph.successors(node_id)
        )
        genes = self.go_graph.nodes[node_id]["gene_set"]
        return input_size_from_children + len(genes)

    def forward(self, batch: Batch):
        # HACK should probably move device to a more appropriate location
        subsystem_outputs = {}
        sorted_subsystems = reversed(list(nx.topological_sort(self.go_graph)))
        go_ids = [i for ids in batch.id for i in ids]
        device = batch.x.device
        for subsystem_name in sorted_subsystems:
            subsystem_model = self.subsystems[subsystem_name]

            # Find indices in the batch that correspond to this subsystem
            subsystem_indices = [
                i for i, go in enumerate(go_ids) if go == subsystem_name
            ]

            # Convert the mask to boolean if it's not already
            bool_mask = batch.mask.type(torch.bool)

            # Gather the mutant states for the subsystem
            mutant_states = torch.stack(
                [batch.x[idx][bool_mask[idx]].to(device) for idx in subsystem_indices]
            ).to(torch.float32)

            # Gather and concatenate outputs of child subsystems
            child_outputs = torch.tensor([], dtype=torch.float32).to(device)
            for child in self.go_graph.successors(subsystem_name):
                assert child in subsystem_outputs, "children must be processed first"
                child_output = subsystem_outputs[child].to(device)
                child_outputs = torch.cat((child_outputs, child_output), dim=1)

            # Concatenate child outputs with mutant states
            if len(child_outputs) > 0:
                subsystem_input = torch.cat((mutant_states, child_outputs), dim=1)
            else:
                subsystem_input = mutant_states

            # Compute the subsystem output
            subsystem_output = subsystem_model(subsystem_input.to(torch.float32))
            subsystem_outputs[subsystem_name] = subsystem_output

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
        # Initialize a dictionary to store the outputs for each subsystem
        linear_outputs = {}

        # Apply the linear transformation to each subsystem output
        for subsystem_name, subsystem_output in subsystem_outputs.items():
            transformed_output = self.subsystem_linears[subsystem_name](
                subsystem_output
            )
            linear_outputs[subsystem_name] = transformed_output

        return linear_outputs


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
    from torchcell.losses import DCellLoss
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
    # dcell = DCell(go_graph=G)

    dcell = DCell(G)
    target = torch.rand(2, 1)
    # target = target.repeat_interleave(G.number_of_nodes())
    # Define the loss function
    criterion = DCellLoss()

    G_mutant = delete_genes(
        go_graph=dcell.go_graph, deletion_gene_set=GeneSet(("YDL029W", "YDR150W"))
    )
    # forward method
    G_mutant = dcell_from_networkx(G_mutant)

    batch = Batch.from_data_list([G_mutant, G_mutant])
    subsystem_outputs = dcell(batch)
    dcell_linear = DCellLinear(dcell.subsystems, output_size=1)
    output = dcell_linear(subsystem_outputs)
    # Compute the loss
    loss = criterion(output, target, dcell.parameters())

    # Backward pass
    loss.backward()

    print(f"Loss: {loss.item()}")
    print()


if __name__ == "__main__":
    main()
