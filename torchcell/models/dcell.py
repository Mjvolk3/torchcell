# torchcell/models/dcell
# [[torchcell.models.dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/dcell
# Test file: tests/torchcell/models/test_dcell.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
from torch_geometric.data import HeteroData, Batch
import networkx as nx
import hydra
import os
import os.path as osp
import numpy as np
import time
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
# Previously commented out to avoid circular import issue
# Now we need this for the demo script
from torchcell.losses.dcell import DCellLoss


class SubsystemModel(nn.Module):
    """
    Neural network model representing a subsystem in the GO hierarchy.

    When num_layers=1, this module behaves like the original DCell subsystem.
    When num_layers>1, this becomes a multi-layer perceptron (MLP) with the
    specified number of layers, all with the same output size.

    This module processes the inputs for each GO term, which consists of:
    1. The mutant state for genes directly annotated to this term
    2. Outputs from child subsystems in the GO hierarchy

    Args:
        input_size: Total input size (mutant states + child outputs)
        output_size: Size of the output vector for this subsystem
        norm_type: Type of normalization to use ('batch', 'layer', 'instance', or 'none')
        norm_before_act: Whether to apply normalization before activation
        num_layers: Number of layers in this subsystem (default: 1 as in original DCell)
                    When >1, creates an MLP with multiple linear+norm+activation blocks
        activation: Activation function to use (default: nn.Tanh as in original DCell)
                   The same activation function is used across all layers
        init_range: Range for uniform weight initialization [-init_range, init_range]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        norm_type: str = "batch",
        norm_before_act: bool = False,
        num_layers: int = 1,
        activation: nn.Module = None,
        init_range: float = 0.001,
    ):
        super().__init__()
        self.output_size = output_size  # Store output size as an attribute
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.norm_before_act = norm_before_act

        # Set default activation to Tanh if none provided (as in original DCell)
        self.activation = activation if activation is not None else nn.Tanh()

        # Create layers - using ModuleList to track parameters properly
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            # Single layer case (original DCell architecture)
            self.layers.append(nn.Linear(input_size, output_size))
            self.norms.append(self._create_norm(output_size, norm_type))
        else:
            # Multi-layer MLP case

            # First layer: input_size -> output_size
            self.layers.append(nn.Linear(input_size, output_size))
            self.norms.append(self._create_norm(output_size, norm_type))

            # Additional hidden layers: output_size -> output_size
            # Each layer forms a block in the MLP with the same activation
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(output_size, output_size))
                self.norms.append(self._create_norm(output_size, norm_type))

        # Initialize weights according to the DCell paper - uniform random in small range
        for layer in self.layers:
            nn.init.uniform_(layer.weight, -init_range, init_range)
            nn.init.uniform_(layer.bias, -init_range, init_range)

    def _create_norm(self, size: int, norm_type: str):
        """Helper function to create normalization layers"""
        if norm_type == "batch":
            # Use standard BatchNorm with proper momentum
            return nn.BatchNorm1d(size, momentum=0.1, track_running_stats=True)
        elif norm_type == "instance":
            # Instance normalization (normalizes each sample independently)
            return nn.InstanceNorm1d(size, affine=True)
        elif norm_type == "layer":
            # Use LayerNorm as an alternative
            return nn.LayerNorm(size)
        elif norm_type == "none":
            return None
        else:
            raise ValueError(
                f"Unknown norm_type: {norm_type}. Expected 'batch', 'layer', 'instance', or 'none'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the subsystem.

        When num_layers=1, this behaves like the original DCell subsystem.
        When num_layers>1, this functions like an MLP with multiple blocks, where
        each block consists of: Linear -> [Normalization] -> Activation
        or Linear -> Activation -> [Normalization], depending on norm_before_act.

        Args:
            x: Input tensor containing mutant states and child outputs [batch_size, input_size]

        Returns:
            Processed subsystem output [batch_size, output_size]
        """
        # Apply each layer of the MLP in sequence
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # Apply linear transformation
            x = layer(x)

            # Apply normalization and activation in the specified order
            if self.norm_before_act:
                # Norm -> Act order
                if norm is not None:
                    # Handle batch size safely for BatchNorm - this is a legitimate technical constraint
                    if self.norm_type == "batch" and x.size(0) == 1:
                        # Skip BatchNorm for single samples to avoid the error
                        pass
                    else:
                        x = norm(x)
                x = self.activation(x)
            else:
                # Act -> Norm order (original DCell approach)
                x = self.activation(x)
                if norm is not None:
                    # Handle batch size safely for BatchNorm - this is a legitimate technical constraint
                    if self.norm_type == "batch" and x.size(0) == 1:
                        # Skip BatchNorm for single samples to avoid the error
                        pass
                    else:
                        x = norm(x)

        return x


class DCell(nn.Module):
    """
    Reimplementation of DCell model that works directly with PyTorch Geometric HeteroData.

    This implementation:
    1. Uses the gene_ontology structure from cell_graph
    2. Processes mutant state tensors generated by DCellGraphProcessor
    3. Handles batches of multiple samples efficiently using vectorized operations
    4. Supports multi-layer MLPs for each subsystem (when subsystem_num_layers > 1)

    Args:
        gene_num: Total number of genes (used for tensor allocations)
        subsystem_output_min: Minimum output size for any subsystem
        subsystem_output_max_mult: Multiplier for scaling subsystem output sizes
        norm_type: Type of normalization to use ('batch', 'layer', 'instance', or 'none')
        norm_before_act: Whether to apply normalization before activation
        subsystem_num_layers: Number of layers per subsystem (default: 1 as in original DCell)
                             When >1, each subsystem becomes an MLP with multiple
                             Linear -> [Norm] -> Activation layers (or reverse order)
        activation: Activation function to use (default: nn.Tanh as in original DCell)
                    The same activation is used throughout all layers of each subsystem
        init_range: Range for uniform weight initialization [-init_range, init_range]
    """

    def __init__(
        self,
        gene_num: int,
        subsystem_output_min: int = 20,
        subsystem_output_max_mult: float = 0.3,
        norm_type: str = "batch",
        norm_before_act: bool = False,
        subsystem_num_layers: int = 1,
        activation: nn.Module = None,
        init_range: float = 0.001,
        learnable_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.gene_num = gene_num
        self.subsystem_output_min = subsystem_output_min
        self.subsystem_output_max_mult = subsystem_output_max_mult
        self.subsystems = nn.ModuleDict()
        self.go_graph = None
        self.norm_type = norm_type
        self.norm_before_act = norm_before_act
        self.subsystem_num_layers = subsystem_num_layers
        self.activation = activation
        self.init_range = init_range
        self.learnable_embedding_dim = learnable_embedding_dim

        # Initialize gene embeddings if enabled
        if self.learnable_embedding_dim is not None:
            self.gene_embeddings = nn.Embedding(gene_num, learnable_embedding_dim)
            # Initialize embeddings with small random values
            nn.init.uniform_(self.gene_embeddings.weight, -init_range, init_range)
        else:
            self.gene_embeddings = None

        # Flag to track if we've initialized from real data
        self.initialized = False

        # Cache for topological order
        self.sorted_subsystems = None

    def _initialize_from_cell_graph(self, cell_graph: HeteroData) -> None:
        """
        Initialize subsystems from the cell_graph structure.
        Must be called before the first forward pass.

        Args:
            cell_graph: The cell graph containing gene ontology structure
        """
        # Verify we have gene ontology data
        if "gene_ontology" not in cell_graph.node_types:
            raise ValueError("Cell graph must contain gene_ontology nodes")

        # Build the NetworkX graph from the HeteroData structure for traversal
        go_graph = nx.DiGraph()

        # Add GO terms as nodes with full data
        for i, term_id in enumerate(cell_graph["gene_ontology"].node_ids):
            # Extract gene indices for this term
            gene_indices = []
            if hasattr(cell_graph["gene_ontology"], "term_to_gene_dict"):
                gene_indices = cell_graph["gene_ontology"].term_to_gene_dict.get(i, [])

            # Add gene names if available
            gene_names = []
            if hasattr(cell_graph["gene"], "node_ids"):
                gene_names = [
                    cell_graph["gene"].node_ids[idx]
                    for idx in gene_indices
                    if idx < len(cell_graph["gene"].node_ids)
                ]

            # Add node with all required attributes
            go_graph.add_node(
                term_id,
                id=i,
                gene_set=gene_names if gene_names else gene_indices,
                namespace="biological_process",  # Default, might be updated later if info is available
                # Initialize empty mutant state
                mutant_state=torch.ones(len(gene_indices), dtype=torch.float32),
            )

        # Add hierarchical edges (child -> parent)
        if ("gene_ontology", "is_child_of", "gene_ontology") in cell_graph.edge_types:
            edge_index = cell_graph[
                "gene_ontology", "is_child_of", "gene_ontology"
            ].edge_index
            for i in range(edge_index.size(1)):
                child_idx = edge_index[0, i].item()
                parent_idx = edge_index[1, i].item()
                child_id = cell_graph["gene_ontology"].node_ids[child_idx]
                parent_id = cell_graph["gene_ontology"].node_ids[parent_idx]
                go_graph.add_edge(child_id, parent_id)

        # Add root node if not already present
        if "GO:ROOT" not in go_graph.nodes:
            # Find all nodes without parents (current roots)
            root_nodes = [
                node for node in go_graph.nodes if go_graph.in_degree(node) == 0
            ]

            # Add super-root node
            go_graph.add_node(
                "GO:ROOT",
                name="GO Super Node",
                namespace="super_root",
                level=-1,
                gene_set=[],
                mutant_state=torch.tensor([], dtype=torch.float32),
            )

            # Connect all current roots to the super-root
            for node in root_nodes:
                go_graph.add_edge("GO:ROOT", node)

        # Reverse the graph to make traversal easier (parent -> child)
        go_graph = nx.reverse(go_graph, copy=True)
        self.go_graph = go_graph

        # Clear default subsystem
        self.subsystems.clear()

        # Build the subsystems in topological order
        self._build_subsystems(go_graph)
        self.initialized = True

        # Cache sorted subsystems for faster forward pass
        self.sorted_subsystems = list(
            reversed(list(nx.topological_sort(self.go_graph)))
        )

        # Log the number of subsystems created
        print(
            f"Created {len(self.subsystems)} subsystems from GO graph with {len(go_graph.nodes)} nodes"
        )

    def _build_subsystems(self, go_graph: nx.DiGraph) -> None:
        """
        Build the subsystem modules based on the GO graph structure.
        Accounts for gene embeddings if enabled.

        Args:
            go_graph: NetworkX DiGraph representing the GO hierarchy
        """
        # Sort nodes topologically to ensure we process all nodes systematically
        nodes_sorted = list(nx.topological_sort(go_graph))

        # Create a mapping to track input and output sizes for validation
        self.input_sizes = {}

        # First pass: count children and identify leaf nodes
        node_children = {}
        leaf_nodes = []
        for node_id in nodes_sorted:
            successors = list(go_graph.successors(node_id))
            node_children[node_id] = successors
            if not successors:  # No children = leaf node
                leaf_nodes.append(node_id)

        # Debug information
        print(
            f"Found {len(leaf_nodes)} leaf nodes out of {len(nodes_sorted)} total nodes"
        )

        # First process leaf nodes (no children)
        for node_id in leaf_nodes:
            # Get gene set for this term
            genes = go_graph.nodes[node_id].get("gene_set", [])

            # Calculate input size based on embeddings or binary state
            if self.learnable_embedding_dim is not None:
                # When using embeddings, each gene contributes gene_embedding_dim values
                input_size = max(1, len(genes) * self.learnable_embedding_dim)
            else:
                # Standard binary representation - one value per gene
                input_size = max(1, len(genes))

            # Store input size for validation later
            self.input_sizes[node_id] = input_size

            # Calculate output size with minimum threshold
            output_size = max(
                self.subsystem_output_min,
                int(self.subsystem_output_max_mult * len(genes)),
            )

            # Create subsystem
            self.subsystems[node_id] = SubsystemModel(
                input_size=input_size,
                output_size=output_size,
                norm_type=self.norm_type,
                norm_before_act=self.norm_before_act,
                num_layers=self.subsystem_num_layers,
                activation=self.activation,
                init_range=self.init_range,
            )

        # Second pass: process non-leaf nodes in reverse topological order
        # Process nodes from leaves toward the root
        for node_id in reversed(nodes_sorted):
            # Skip if already processed
            if node_id in self.subsystems:
                continue

            # Get children of this node
            children = node_children.get(node_id, [])

            # Make sure all children have been processed
            all_children_processed = all(child in self.subsystems for child in children)
            if not all_children_processed:
                print(f"Warning: Not all children processed for {node_id}")
                # Add any missing children as simple pass-through
                for child in children:
                    if child not in self.subsystems:
                        genes_child = go_graph.nodes[child].get("gene_set", [])
                        self.subsystems[child] = SubsystemModel(
                            input_size=max(1, len(genes_child)),
                            output_size=self.subsystem_output_min,
                            norm_type=self.norm_type,
                            norm_before_act=self.norm_before_act,
                            num_layers=self.subsystem_num_layers,
                            activation=self.activation,
                            init_range=self.init_range,
                        )
                        self.input_sizes[child] = max(1, len(genes_child))

            # Calculate total input size: sum of child outputs + genes for this term
            children_output_size = sum(
                self.subsystems[child].output_size
                for child in children
                if child in self.subsystems
            )

            # Get genes for this term
            genes = go_graph.nodes[node_id].get("gene_set", [])

            # Calculate gene input size with embeddings if enabled
            if self.learnable_embedding_dim is not None:
                # When using embeddings, each gene contributes gene_embedding_dim values
                gene_input_size = max(1, len(genes) * self.learnable_embedding_dim)
            else:
                # Standard binary representation - one value per gene
                gene_input_size = max(1, len(genes))

            # Calculate total input size (child outputs + gene states/embeddings)
            total_input_size = children_output_size + gene_input_size
            # Ensure at least size 1 to avoid errors with empty terms
            total_input_size = max(1, total_input_size)

            # Store for validation
            self.input_sizes[node_id] = total_input_size

            # Calculate output size with minimum threshold
            output_size = max(
                self.subsystem_output_min,
                int(self.subsystem_output_max_mult * len(genes)),
            )

            # Initialize subsystem with proper sizes
            self.subsystems[node_id] = SubsystemModel(
                input_size=total_input_size,
                output_size=output_size,
                norm_type=self.norm_type,
                norm_before_act=self.norm_before_act,
                num_layers=self.subsystem_num_layers,
                activation=self.activation,
                init_range=self.init_range,
            )

        # Add special handling for root node if present
        if "GO:ROOT" in go_graph.nodes and "GO:ROOT" not in self.subsystems:
            # Get children of root
            root_children = list(go_graph.successors("GO:ROOT"))
            # Sum output sizes of all root's children
            root_input_size = sum(
                self.subsystems[child].output_size
                for child in root_children
                if child in self.subsystems
            )

            # We don't add an extra input for the gene state for ROOT anymore
            # The gene state will be created with the right size during forward pass

            # Add a small adjustment to prevent dimension mismatch for the root node
            # This padding dimension (size 1) ensures consistent processing in forward pass
            root_input_size += 1  # Always add a padding dimension for root node

            # Log the resulting input size for debugging
            print(f"GO:ROOT input size after padding: {root_input_size}")

            # Ensure at least size 1
            root_input_size = max(1, root_input_size)
            # Store for validation
            self.input_sizes["GO:ROOT"] = root_input_size

            # Create root subsystem
            self.subsystems["GO:ROOT"] = SubsystemModel(
                input_size=root_input_size,
                output_size=self.subsystem_output_min,
                norm_type=self.norm_type,
                norm_before_act=self.norm_before_act,
                num_layers=self.subsystem_num_layers,
                activation=self.activation,
                init_range=self.init_range,
            )

        # Verify subsystem creation
        num_created = len(self.subsystems)
        print(f"Created {num_created} subsystems out of {len(nodes_sorted)} nodes")

        # Print parameter count
        param_count = sum(
            sum(p.numel() for p in s.parameters() if p.requires_grad)
            for s in self.subsystems.values()
        )
        print(f"Total parameters in subsystems: {param_count:,}")

        # If we somehow ended up with no subsystems, fail explicitly
        if len(self.subsystems) == 0:
            raise ValueError(
                "No subsystems were created from the GO hierarchy. "
                "This indicates a problem with the GO hierarchy structure or filtering."
            )

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vectorized forward pass for the DCell model.
        This implementation avoids loops over individual samples in a batch.

        Args:
            cell_graph: The cell graph containing gene ontology structure
            batch: HeteroDataBatch containing perturbation information and mutant states

        Returns:
            Tuple of (root_output, outputs_dictionary)
        """
        # Initialize subsystems from cell_graph if not done yet
        if not self.initialized:
            self._initialize_from_cell_graph(cell_graph)

        # Double-check that we're initialized properly
        if len(self.subsystems) == 0:
            raise ValueError(
                "Model has no subsystems after initialization. "
                "This indicates a problem with the GO hierarchy structure or filtering."
            )

        # Set device - get device from any tensor in the batch
        if hasattr(batch, "device"):
            device = batch.device
        elif hasattr(batch["gene"], "perturbation_indices"):
            device = batch["gene"].perturbation_indices.device
        elif hasattr(batch["gene_ontology"], "mutant_state"):
            device = batch["gene_ontology"].mutant_state.device
        else:
            # Default to CPU if we can't determine the device
            device = torch.device("cpu")
            
        num_graphs = batch.num_graphs

        # Get mutant state tensor - COO format containing [term_idx, gene_idx, state]
        if not hasattr(batch["gene_ontology"], "mutant_state"):
            raise ValueError(
                "Batch must contain gene_ontology.mutant_state for DCell model"
            )

        mutant_state = batch["gene_ontology"].mutant_state

        # Extract batch information
        if hasattr(batch["gene_ontology"], "mutant_state_batch"):
            # Use provided batch indices for the mutant states
            batch_indices = batch["gene_ontology"].mutant_state_batch
        else:
            # If no batch index, assume single graph
            batch_indices = torch.zeros(
                mutant_state.size(0), dtype=torch.long, device=device
            )

        # Map from term IDs to indices in the cell_graph
        term_id_to_idx = {
            term_id: idx
            for idx, term_id in enumerate(cell_graph["gene_ontology"].node_ids)
        }

        # Dictionary to store all subsystem outputs
        subsystem_outputs = {}

        # We no longer support a default fallback subsystem
        # If we're missing subsystems, the initialization should have failed earlier
        if len(self.subsystems) == 0:
            raise ValueError(
                "Model has no subsystems. This should not happen as initialization should have failed earlier."
            )

        # Create a dictionary to hold gene states/embeddings for each term and batch item
        term_gene_states = {}

        # Pre-process mutant states for all terms and batch items
        # Group by term_idx to avoid duplication in the loop
        term_indices = mutant_state[:, 0].long()
        term_indices_set = torch.unique(term_indices).tolist()

        # For each term, create a tensor of gene states/embeddings for all batch items
        for term_idx in term_indices_set:
            term_idx = (
                term_idx.item() if isinstance(term_idx, torch.Tensor) else term_idx
            )

            # Get genes for this term
            genes = cell_graph["gene_ontology"].term_to_gene_dict.get(term_idx, [])
            num_genes = max(1, len(genes))  # Ensure at least 1 gene

            if self.learnable_embedding_dim is not None:
                # Initialize tensor to hold embeddings for all genes in this term
                # Shape: [batch_size, num_genes, embedding_dim]
                term_gene_states[term_idx] = torch.zeros(
                    (num_graphs, num_genes, self.learnable_embedding_dim),
                    dtype=torch.float,
                    device=device,
                )

                # Set default gene embeddings (for non-perturbed genes)
                for gene_local_idx, gene_idx in enumerate(genes):
                    # Copy embeddings to all batch items (will be zeroed out for perturbed genes)
                    term_gene_states[term_idx][:, gene_local_idx] = (
                        self.gene_embeddings.weight[gene_idx]
                    )
            else:
                # Standard binary encoding - initialize to all 1.0 (not perturbed)
                term_gene_states[term_idx] = torch.ones(
                    (num_graphs, num_genes), dtype=torch.float, device=device
                )

            # Get all mutant states for this term
            term_mask = term_indices == term_idx
            term_mutant_states = mutant_state[term_mask]

            if term_mutant_states.size(0) > 0:
                # Get batch indices for these states
                if hasattr(batch["gene_ontology"], "mutant_state_batch"):
                    states_batch_indices = batch_indices[term_mask]
                else:
                    # If no batch info, assume all from batch 0
                    states_batch_indices = torch.zeros(
                        term_mutant_states.size(0), dtype=torch.long, device=device
                    )

                # Apply perturbations to gene states/embeddings
                for i in range(term_mutant_states.size(0)):
                    batch_idx = states_batch_indices[i].item()
                    gene_idx = term_mutant_states[i, 1].long().item()
                    state_value = term_mutant_states[i, 2].item()

                    # Only process valid gene indices
                    if gene_idx < len(genes):
                        # Find local index within the term's genes
                        gene_local_idx = (
                            genes.index(gene_idx) if gene_idx in genes else -1
                        )
                        if gene_local_idx >= 0 and state_value != 1.0:  # Perturbed gene
                            if self.learnable_embedding_dim is not None:
                                # Zero out embedding for perturbed gene
                                term_gene_states[term_idx][
                                    batch_idx, gene_local_idx
                                ] = 0.0
                            else:
                                # Standard binary encoding - set to 0.0 (perturbed)
                                term_gene_states[term_idx][
                                    batch_idx, gene_local_idx
                                ] = 0.0

        # We no longer need special handling for root node here
        # The root node will get only child outputs, no gene states during the forward pass

        # Now process nodes in reverse topological order (leaves to root)
        # using the cached sorted order for efficiency
        for subsystem_name in self.sorted_subsystems:
            # Skip subsystems that weren't initialized
            if subsystem_name not in self.subsystems:
                continue

            # Get the model for this subsystem
            subsystem_model = self.subsystems[subsystem_name]

            # Get the index for this term in the cell_graph
            if subsystem_name == "GO:ROOT":
                # Handle root node specially
                term_idx = -1  # Special value for root
            else:
                # Get the index from the term_id
                term_idx = term_id_to_idx.get(subsystem_name, -1)
                if term_idx == -1:
                    # Skip if term is not in the cell graph
                    continue

            # Get or create gene states/embeddings tensor for this term
            if term_idx in term_gene_states:
                gene_states = term_gene_states[term_idx]
            else:
                # For terms not encountered in mutant_state, use defaults
                genes = cell_graph["gene_ontology"].term_to_gene_dict.get(term_idx, [])

                if self.learnable_embedding_dim is not None:
                    # Create tensor with embeddings for each gene
                    gene_states = torch.zeros(
                        (num_graphs, max(1, len(genes)), self.learnable_embedding_dim),
                        dtype=torch.float,
                        device=device,
                    )

                    # Set gene embeddings (all genes are present by default)
                    for gene_local_idx, gene_idx in enumerate(genes):
                        if gene_idx < self.gene_num:  # Check index bounds
                            gene_states[:, gene_local_idx] = (
                                self.gene_embeddings.weight[gene_idx]
                            )
                else:
                    # Standard binary encoding - all genes present (1.0)
                    gene_states = torch.ones(
                        (num_graphs, max(1, len(genes))),
                        dtype=torch.float,
                        device=device,
                    )

            # Get children outputs and concatenate them
            child_outputs = []
            for child in self.go_graph.successors(subsystem_name):
                if child in subsystem_outputs:
                    child_outputs.append(subsystem_outputs[child])

            # Handle reshaping of gene embeddings if needed
            if self.learnable_embedding_dim is not None and len(gene_states.shape) == 3:
                # Reshape [batch_size, num_genes, embedding_dim] to [batch_size, num_genes * embedding_dim]
                batch_size = gene_states.size(0)
                gene_states = gene_states.reshape(batch_size, -1)

            # Combine gene states/embeddings with child outputs
            if child_outputs:
                # Concatenate all child outputs along feature dimension
                child_tensor = torch.cat(child_outputs, dim=1)

                # Special handling for root node - don't add gene states for GO:ROOT
                if subsystem_name == "GO:ROOT":
                    # For root, we just use the child outputs directly
                    combined_input = child_tensor

                    # Always add padding dimension for root node to match initialization
                    # During initialization, a +1 padding is always added to root_input_size
                    # Add a padding dimension of zeros to match initialization sizing
                    padding = torch.zeros(
                        (combined_input.size(0), 1), device=combined_input.device
                    )
                    combined_input = torch.cat([combined_input, padding], dim=1)
                else:
                    # For regular nodes, combine gene states/embeddings with child outputs
                    combined_input = torch.cat([gene_states, child_tensor], dim=1)
            else:
                # Use only gene states/embeddings if no children
                combined_input = gene_states

            # Check for size mismatch and fail explicitly
            # First layer of the subsystem (we're using the first layer's input size)
            expected_size = subsystem_model.layers[0].weight.size(1)
            actual_size = combined_input.size(1)

            if actual_size != expected_size:
                # For ROOT node specifically, add detailed diagnostic information
                if subsystem_name == "GO:ROOT":
                    print(f"\nDiagnostic information for GO:ROOT node:")
                    print(f"  Expected input size: {expected_size}")
                    print(f"  Actual input size: {actual_size}")

                    if hasattr(self, "input_sizes") and "GO:ROOT" in self.input_sizes:
                        print(
                            f"  Stored input size during initialization: {self.input_sizes['GO:ROOT']}"
                        )

                    # Check if children outputs match expectations
                    child_output_sum = 0
                    for child in self.go_graph.successors(subsystem_name):
                        if child in subsystem_outputs:
                            child_size = subsystem_outputs[child].size(1)
                            child_output_sum += child_size
                            print(f"  Child '{child}' output size: {child_size}")

                    print(f"  Sum of child output sizes: {child_output_sum}")
                    print(
                        f"  Difference (expected - actual): {expected_size - actual_size}"
                    )

                # Fail explicitly with clear error message
                raise ValueError(
                    f"Size mismatch for subsystem '{subsystem_name}': expected {expected_size}, got {actual_size}. "
                    f"This indicates a mismatch between the graph structure used during model initialization "
                    f"and the graph structure in the current batch. "
                    f"Gene embeddings enabled: {self.learnable_embedding_dim is not None}, "
                    f"embedding_dim: {self.learnable_embedding_dim if self.learnable_embedding_dim is not None else 'N/A'}"
                )

            # Forward through subsystem model
            output = subsystem_model(combined_input)
            subsystem_outputs[subsystem_name] = output

        # Find the root output - fail explicitly if not found
        if "GO:ROOT" in subsystem_outputs:
            root_output = subsystem_outputs["GO:ROOT"]
        else:
            # No fallback - raise an error if root node is missing
            raise ValueError(
                "Root node 'GO:ROOT' not found in subsystem outputs. "
                "This indicates a problem with the GO hierarchy structure."
            )

        # Return root output and all subsystem outputs
        return root_output, {"subsystem_outputs": subsystem_outputs}


class DCellLinear(nn.Module):
    """
    Linear prediction head for DCell that takes subsystem outputs and makes final predictions.

    Args:
        subsystems: ModuleDict of subsystems from DCell model
        output_size: Size of the final output (usually 1 for fitness prediction)
    """

    def __init__(self, subsystems: nn.ModuleDict, output_size: int = 1):
        super().__init__()
        self.output_size = output_size
        self.subsystem_linears = nn.ModuleDict()

        # Create a linear layer for each subsystem
        for subsystem_name, subsystem in subsystems.items():
            in_features = subsystem.output_size
            linear = nn.Linear(in_features, self.output_size)

            # Use default initialization to match original DCell behavior
            # Default PyTorch initialization should provide sufficient randomness

            self.subsystem_linears[subsystem_name] = linear

    def forward(
        self, subsystem_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass applying linear transformation to each subsystem output.

        Args:
            subsystem_outputs: Dictionary mapping subsystem names to their outputs

        Returns:
            Dictionary mapping subsystem names to transformed outputs
        """
        linear_outputs = {}

        for subsystem_name, subsystem_output in subsystem_outputs.items():
            if subsystem_name in self.subsystem_linears:
                transformed_output = self.subsystem_linears[subsystem_name](
                    subsystem_output
                )
                linear_outputs[subsystem_name] = transformed_output

        return linear_outputs


class DCellModel(nn.Module):
    """
    Complete DCell model that integrates DCell with prediction head.

    This model:
    1. Processes gene perturbations through the GO hierarchy
    2. Makes phenotype predictions based on the subsystem outputs
    3. Supports multi-layer MLPs for each subsystem (when subsystem_num_layers > 1)
    4. Optionally uses learnable gene embeddings instead of binary state vectors

    Args:
        gene_num: Total number of genes
        subsystem_output_min: Minimum output size for any subsystem
        subsystem_output_max_mult: Multiplier for scaling subsystem output sizes
        output_size: Size of the final output (usually 1 for fitness prediction)
        norm_type: Type of normalization to use ('batch', 'layer', 'instance', or 'none')
        norm_before_act: Whether to apply normalization before activation
        subsystem_num_layers: Number of layers per subsystem (default: 1 as in original DCell)
                             When >1, each subsystem becomes an MLP with multiple
                             Linear -> [Norm] -> Activation layers (or reverse order)
        activation: Activation function to use (default: nn.Tanh as in original DCell)
                    The same activation is used throughout all layers of each subsystem
        init_range: Range for uniform weight initialization [-init_range, init_range]
        learnable_embedding_dim: Dimension for learnable gene embeddings. If None, uses binary states.
    """

    def __init__(
        self,
        gene_num: int,
        subsystem_output_min: int = 20,
        subsystem_output_max_mult: float = 0.3,
        output_size: int = 1,
        norm_type: str = "batch",
        norm_before_act: bool = False,
        subsystem_num_layers: int = 1,
        activation: nn.Module = None,
        init_range: float = 0.001,
        learnable_embedding_dim: Optional[int] = None,
    ):
        super().__init__()

        # DCell component for processing GO hierarchy
        self.dcell = DCell(
            gene_num=gene_num,
            subsystem_output_min=subsystem_output_min,
            subsystem_output_max_mult=subsystem_output_max_mult,
            norm_type=norm_type,
            norm_before_act=norm_before_act,
            subsystem_num_layers=subsystem_num_layers,
            activation=activation,
            init_range=init_range,
            learnable_embedding_dim=learnable_embedding_dim,
        )

        # We'll initialize dcell_linear in the forward pass after DCell is properly initialized
        self.dcell_linear = None
        self.output_size = output_size
        self._initialized = False

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for the complete DCellModel.

        Args:
            cell_graph: The cell graph containing gene ontology structure
            batch: HeteroDataBatch containing perturbation information

        Returns:
            Tuple of (predictions, outputs dictionary)
        """
        # Process through DCell
        root_output, outputs = self.dcell(cell_graph, batch)
        subsystem_outputs = outputs["subsystem_outputs"]

        # Initialize or update DCellLinear using the DCell component's actual subsystems
        if not self._initialized:
            # Replace the dummy DCellLinear with one that uses the actual subsystems
            self.dcell_linear = DCellLinear(
                subsystems=self.dcell.subsystems, output_size=self.output_size
            ).to(root_output.device)
            self._initialized = True

        # Apply linear transformation to all subsystem outputs
        linear_outputs = self.dcell_linear(subsystem_outputs)
        outputs["linear_outputs"] = linear_outputs

        # Find the root prediction - fail explicitly if not found
        # First try to get prediction for "GO:ROOT"
        if "GO:ROOT" in linear_outputs:
            predictions = linear_outputs["GO:ROOT"]
        else:
            # No fallback - raise an error if root node prediction is missing
            raise ValueError(
                "Root node 'GO:ROOT' prediction not found in linear outputs. "
                "This indicates a problem with the DCellLinear component or GO hierarchy."
            )

        return predictions, outputs

    @property
    def num_parameters(self) -> Dict[str, int]:
        """
        Count the number of trainable parameters in the model
        """

        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {"dcell": count_params(self.dcell)}

        # Count parameters in each subsystem
        subsystem_counts = {}
        for name, subsystem in self.dcell.subsystems.items():
            subsystem_counts[name] = count_params(subsystem)

        counts["subsystems"] = sum(subsystem_counts.values())

        if self.dcell_linear is not None:
            counts["dcell_linear"] = count_params(self.dcell_linear)

        # Calculate overall total
        counts["total"] = sum(v for k, v in counts.items() if k not in ["subsystems"])

        # Additional useful information
        if hasattr(self.dcell, "go_graph") and self.dcell.go_graph is not None:
            counts["num_go_terms"] = len(self.dcell.go_graph.nodes())
            counts["num_subsystems"] = len(self.dcell.subsystems)

        return counts


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/005-kuzmin2018-tmi/conf"),
    config_name="dcell_kuzmin2018_tmi",
)
def main(cfg: DictConfig):
    """
    Main function to test the DCellModel on a batch of data.
    Overfits the model on a single batch and produces loss component plots.
    """
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from torchcell.scratch.load_batch_005 import load_sample_data_batch
    import time
    from tqdm.auto import tqdm

    # Load environment variables to get ASSET_IMAGES_DIR
    load_dotenv()
    print("\n" + "=" * 80)
    print("DATA LOADING")
    print("=" * 80)

    print("Loading sample data with DCellGraphProcessor...")
    # Load sample data with DCellGraphProcessor - respecting config
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=32,  # Fixed for overfitting test
        num_workers=cfg.data_module.num_workers,
        config="dcell",
        is_dense=False,
    )

    # Set device based on config
    if cfg.trainer.accelerator == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")  # Default to CPU
    print(f"Using device: {device}")

    # Print detailed dataset information
    print(f"Dataset: {len(dataset)} samples")
    print(f"Batch: {batch.num_graphs} graphs")
    print(f"Max Number of Nodes: {max_num_nodes}")
    print(f"Input Channels: {input_channels}")

    # Move data to device
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Parse activation function from config
    def get_activation_from_config(activation_name: str) -> nn.Module:
        """Get activation function from string name in config"""
        activation_map = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "elu": nn.ELU(),
            "linear": nn.Identity(),  # No activation
        }
        return activation_map.get(activation_name.lower(), nn.Tanh())

    # Get activation function from config
    activation_name = cfg.model.get("activation", "tanh")
    activation = get_activation_from_config(activation_name)

    # Get other parameters from config with defaults
    norm_type = cfg.model.get("norm_type", "batch")
    norm_before_act = cfg.model.get("norm_before_act", False)
    subsystem_num_layers = cfg.model.get("subsystem_num_layers", 1)
    init_range = cfg.model.get("init_range", 0.001)

    # Get gene embedding parameters
    learnable_embedding_dim = cfg.model.get("learnable_embedding_dim", None)

    # Log model configuration
    print(f"Model configuration:")
    print(f"  - Normalization type: {norm_type}")
    print(f"  - Normalization order: {'norm->act' if norm_before_act else 'act->norm'}")

    # Provide more detailed information about the number of layers
    if subsystem_num_layers == 1:
        print(f"  - Subsystem architecture: Single layer (original DCell)")
    else:
        print(f"  - Subsystem architecture: {subsystem_num_layers}-layer MLP")
        print(
            f"    Each subsystem uses multiple linear layers with {activation_name} activation"
        )

    print(f"  - Activation function: {activation_name}")
    print(f"  - Weight initialization range: ±{init_range}")

    # Log gene embedding configuration
    if learnable_embedding_dim is not None:
        print(
            f"  - Using learnable gene embeddings with dimension: {learnable_embedding_dim}"
        )
    else:
        print(f"  - Using standard binary gene state encoding")

    # Initialize model
    model = DCellModel(
        gene_num=max_num_nodes,
        subsystem_output_min=cfg.model.subsystem_output_min,
        subsystem_output_max_mult=cfg.model.subsystem_output_max_mult,
        output_size=cfg.model.output_size,
        norm_type=norm_type,
        norm_before_act=norm_before_act,
        subsystem_num_layers=subsystem_num_layers,
        activation=activation,
        init_range=init_range,
        learnable_embedding_dim=learnable_embedding_dim,
    ).to(device)

    # Model has no verbose_debug flag anymore - removed for stricter error handling

    # Run a forward pass to initialize the model
    with torch.no_grad():
        predictions, _ = model(cell_graph, batch)

        # Briefly check prediction diversity
        diversity = predictions.std().item()
        print(f"Initial predictions diversity: {diversity:.6f}")

        if diversity < 1e-6:
            print("WARNING: Predictions lack diversity!")
        else:
            print("✓ Predictions are diverse")

    # Print basic parameter information
    param_info = model.num_parameters
    total_params = param_info.get("total", 0)
    print(f"Model parameters: {total_params:,}")
    print(f"Subsystems: {param_info.get('num_subsystems', 0):,}")

    # Create optimizer based on config
    if hasattr(cfg.regression_task, "optimizer"):
        optimizer_type = cfg.regression_task.optimizer.type
        optimizer_lr = cfg.regression_task.optimizer.lr
        optimizer_weight_decay = cfg.regression_task.optimizer.weight_decay

        print(f"\nOptimizer Configuration:")
        print(f"  Type: {optimizer_type}")
        print(f"  Learning Rate: {optimizer_lr}")
        print(f"  Weight Decay: {optimizer_weight_decay}")

        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay
            )
            print(
                f"Using Adam optimizer with lr={optimizer_lr}, weight_decay={optimizer_weight_decay}"
            )
        elif optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(
                model.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay
            )
            print(
                f"Using AdamW optimizer with lr={optimizer_lr}, weight_decay={optimizer_weight_decay}"
            )
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay
            )
            print(
                f"Using SGD optimizer with lr={optimizer_lr}, weight_decay={optimizer_weight_decay}"
            )
        else:
            # Default to Adam
            optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)
            print(
                f"Unknown optimizer type '{optimizer_type}', defaulting to Adam with lr={optimizer_lr}"
            )
    else:
        # Fallback to Adam with default parameters
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("No optimizer config found, using default Adam optimizer with lr=0.001")

    # Create loss function
    if hasattr(cfg.regression_task, "dcell_loss"):
        alpha = cfg.regression_task.dcell_loss.alpha
        use_auxiliary_losses = cfg.regression_task.dcell_loss.use_auxiliary_losses
        criterion = DCellLoss(alpha=alpha, use_auxiliary_losses=use_auxiliary_losses)
        print(
            f"Using DCellLoss with alpha={alpha}, use_auxiliary_losses={use_auxiliary_losses}"
        )
    else:
        criterion = DCellLoss(alpha=0.3)
        print("Using default DCellLoss with alpha=0.3")

    # Get target if phenotype_values exists
    if hasattr(batch["gene"], "phenotype_values"):
        target = batch["gene"].phenotype_values.view_as(
            torch.zeros(batch.num_graphs, 1, device=device)
        )
    else:
        # Use a placeholder target for demo or initialization
        target = torch.zeros(batch.num_graphs, 1, device=device)

    # Overfit the model on a single batch
    print("\nOverfitting model on a single batch for 500 epochs...")
    num_epochs = 500
    plot_every = 50  # Plot the loss curve every 50 epochs

    # Initialize history for loss tracking
    history = {
        "total_loss": [],
        "primary_loss": [],
        "auxiliary_loss": [],
        "weighted_auxiliary_loss": [],
        "epochs": [],
        "time_per_epoch": [],
    }

    # Training loop with tqdm progress bar
    start_time = time.time()
    progress_bar = tqdm(range(num_epochs), desc="Training")

    # Add correlation tracking to history
    history["correlation"] = []

    for epoch in progress_bar:
        epoch_start = time.time()

        # Forward pass
        predictions, outputs = model(cell_graph, batch)

        # Extract actual prediction and target values for tracking
        pred_values = predictions.detach().cpu().numpy().flatten()
        target_values = target.detach().cpu().numpy().flatten()

        # Compute loss
        loss, loss_components = criterion(predictions, outputs, target)

        # Store loss components
        history["total_loss"].append(loss.item())
        history["primary_loss"].append(loss_components["primary_loss"].item())
        history["auxiliary_loss"].append(loss_components["auxiliary_loss"].item())
        history["weighted_auxiliary_loss"].append(
            loss_components["weighted_auxiliary_loss"].item()
        )
        history["epochs"].append(epoch)

        # Calculate and store correlation
        correlation = None
        if len(pred_values) > 1:  # Only if we have more than one sample
            try:
                correlation = np.corrcoef(pred_values, target_values)[0, 1]
                history["correlation"].append(correlation)
                corr_str = f", Corr: {correlation:.4f}"
            except:
                corr_str = ""
                history["correlation"].append(float("nan"))
        else:
            corr_str = ""
            history["correlation"].append(float("nan"))

        # Record time taken for this epoch
        epoch_time = time.time() - epoch_start
        history["time_per_epoch"].append(epoch_time)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping if enabled in config
        if (
            hasattr(cfg.regression_task, "clip_grad_norm")
            and cfg.regression_task.clip_grad_norm
        ):
            max_norm = cfg.regression_task.get("clip_grad_norm_max_norm", 10.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # Update progress bar description
        progress_bar.set_description(
            f"Loss: {loss.item():.6f}{corr_str}, Time: {epoch_time:.3f}s/epoch"
        )

        # Plot curves at regular intervals
        if (epoch + 1) % plot_every == 0:
            # Create output directory if needed
            os.makedirs("outputs", exist_ok=True)

            # 1. Plot loss components
            plt.figure(figsize=(10, 6))
            # Use semilogy for logarithmic y-axis
            plt.semilogy(
                history["epochs"], history["total_loss"], "b-", label="Total Loss"
            )
            plt.semilogy(
                history["epochs"], history["primary_loss"], "r-", label="Primary Loss"
            )
            plt.semilogy(
                history["epochs"],
                history["weighted_auxiliary_loss"],
                "g-",
                label="Weighted Aux Loss",
            )

            plt.xlabel("Epoch")
            plt.ylabel("Loss (log scale)")

            # Include number of layers in title if using multi-layer model
            if subsystem_num_layers > 1:
                plt.title(
                    f"DCell Loss Components - Epoch {epoch+1} ({norm_type}, {activation_name}, {subsystem_num_layers}-layer)"
                )
            else:
                plt.title(
                    f"DCell Loss Components - Epoch {epoch+1} ({norm_type}, {activation_name})"
                )

            plt.legend()
            plt.grid(True)

            # Save figure properly using ASSET_IMAGES_DIR and timestamp
            if subsystem_num_layers > 1:
                title = f"dcell_{norm_type}_{activation_name}_{subsystem_num_layers}layer_loss_components_epoch_{epoch+1}"
            else:
                title = f"dcell_{norm_type}_{activation_name}_loss_components_epoch_{epoch+1}"

            save_path = osp.join(
                os.environ["ASSET_IMAGES_DIR"], f"{title}_{timestamp()}.png"
            )
            plt.savefig(save_path)
            print(f"Saved loss components plot to {save_path}")
            plt.close()

            # 2. Plot correlation progress
            plt.figure(figsize=(10, 6))
            valid_indices = ~np.isnan(history["correlation"])
            if np.any(valid_indices):
                valid_epochs = np.array(history["epochs"])[valid_indices]
                valid_correlations = np.array(history["correlation"])[valid_indices]
                plt.plot(valid_epochs, valid_correlations, "g-", linewidth=2)

                # Add horizontal line at 0.45 for reference
                plt.axhline(y=0.45, color="r", linestyle="--", label="0.45 threshold")

                # Get the max correlation achieved
                max_corr = (
                    np.max(valid_correlations) if len(valid_correlations) > 0 else 0
                )
                plt.title(
                    f"Correlation Progress - Epoch {epoch+1} (Max: {max_corr:.4f})"
                )
                plt.xlabel("Epoch")
                plt.ylabel("Pearson Correlation")
                plt.grid(True)
                plt.legend()

                # Save correlation plot
                if subsystem_num_layers > 1:
                    corr_title = f"dcell_{norm_type}_{activation_name}_{subsystem_num_layers}layer_correlation_epoch_{epoch+1}"
                else:
                    corr_title = f"dcell_{norm_type}_{activation_name}_correlation_epoch_{epoch+1}"

                corr_save_path = osp.join(
                    os.environ["ASSET_IMAGES_DIR"], f"{corr_title}_{timestamp()}.png"
                )
                plt.savefig(corr_save_path)
                print(f"Saved correlation plot to {corr_save_path}")
                plt.close()

            # Skip time per epoch plot

    # Create output directory for plots
    os.makedirs("outputs", exist_ok=True)

    # Create a more detailed loss components plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Primary plot: Main loss components - using log scale for y-axis
    ax1.semilogy(history["total_loss"], "b-", linewidth=2, label="Total Loss")
    ax1.semilogy(history["primary_loss"], "r-", linewidth=2, label="Primary Loss")
    ax1.semilogy(
        history["weighted_auxiliary_loss"],
        "g-",
        linewidth=2,
        label="Weighted Auxiliary Loss",
    )
    ax1.set_ylabel("Loss Value (log scale)", fontsize=12)

    # Include layers info in title if using multi-layer subsystems
    if subsystem_num_layers > 1:
        ax1.set_title(
            f"DCell Loss Components During Training ({norm_type}, {activation_name}, {subsystem_num_layers}-layer MLP)",
            fontsize=14,
        )
    else:
        ax1.set_title(
            f"DCell Loss Components During Training ({norm_type}, {activation_name})",
            fontsize=14,
        )

    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Add epoch markers
    for epoch in range(0, num_epochs, 50):
        if epoch > 0:  # Skip the first epoch for clarity
            ax1.axvline(x=epoch, color="gray", linestyle="--", alpha=0.3)

    # Secondary plot: Auxiliary loss (different scale) - with log scale
    ax2.semilogy(
        history["auxiliary_loss"],
        "orange",
        linewidth=2,
        label="Auxiliary Loss (Unweighted)",
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Auxiliary Loss Value (log scale)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(loc="upper right", fontsize=10)

    # Clear existing legends
    ax1.get_legend().remove()
    ax2.get_legend().remove()

    # Add final values to the legend instead of using annotations
    ax1.legend(
        [
            f"Total Loss: {history['total_loss'][-1]:.6f}",
            f"Primary Loss: {history['primary_loss'][-1]:.6f}",
            f"Weighted Aux Loss: {history['weighted_auxiliary_loss'][-1]:.6f}",
        ],
        loc="upper right",
        fontsize=10,
    )

    ax2.legend(
        [f"Auxiliary Loss: {history['auxiliary_loss'][-1]:.6f}"],
        loc="upper right",
        fontsize=10,
    )

    plt.tight_layout()

    # Save figure properly using ASSET_IMAGES_DIR and timestamp
    if subsystem_num_layers > 1:
        title = f"dcell_{norm_type}_{activation_name}_{subsystem_num_layers}layer_loss_components_final"
    else:
        title = f"dcell_{norm_type}_{activation_name}_loss_components_final"

    save_path = osp.join(os.environ["ASSET_IMAGES_DIR"], f"{title}_{timestamp()}.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nDetailed loss components plot saved to '{save_path}'")

    # Skip time per epoch plot - not needed

    # Check if predictions are converging to targets
    # If we have multiple samples, create a predictions vs targets plot
    if len(pred_values) > 1:
        # Create a larger figure to accommodate annotations
        plt.figure(figsize=(12, 9))

        # Plot points with a single color for better contrast with labels
        scatter = plt.scatter(
            target_values,
            pred_values,
            c="blue",
            alpha=0.7,
            s=100,  # Larger point size for better visibility
            label="Predictions",
        )

        # Add sample indices as annotations directly on each point
        for i, (x, y) in enumerate(zip(target_values, pred_values)):
            plt.annotate(
                f"{i}",  # Sample number
                (x, y),
                xytext=(3, 3),  # Small offset
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"),
            )

        # Add perfect prediction line
        min_val = min(min(target_values), min(pred_values))
        max_val = max(max(target_values), max(pred_values))
        plt.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction"
        )

        # Calculate and display correlation
        correlation = np.corrcoef(pred_values, target_values)[0, 1]

        # Include layers info in title if using multi-layer subsystems
        if subsystem_num_layers > 1:
            plt.title(
                f"DCell Predictions vs Targets - {norm_type}, {activation_name}, {subsystem_num_layers}-layer MLP (Correlation: {correlation:.4f})",
                fontsize=16,
            )
        else:
            plt.title(
                f"DCell Predictions vs Targets - {norm_type}, {activation_name} (Correlation: {correlation:.4f})",
                fontsize=16,
            )
        plt.xlabel("Target Values", fontsize=14)
        plt.ylabel("Predicted Values", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add correlation information as text box
        plt.text(
            0.05,
            0.95,
            f"Pearson Correlation: {correlation:.4f}\n"
            f"Number of samples: {len(target_values)}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        # Save figure properly using ASSET_IMAGES_DIR and timestamp
        if subsystem_num_layers > 1:
            title = f"dcell_{norm_type}_{activation_name}_{subsystem_num_layers}layer_predictions_vs_targets"
        else:
            title = f"dcell_{norm_type}_{activation_name}_predictions_vs_targets"

        save_path = osp.join(
            os.environ["ASSET_IMAGES_DIR"], f"{title}_{timestamp()}.png"
        )
        plt.savefig(save_path, dpi=300)
        print(f"Predictions vs targets plot saved to '{save_path}'")

    # Create final correlation plot
    plt.figure(figsize=(12, 8))
    valid_indices = ~np.isnan(history["correlation"])
    if np.any(valid_indices):
        valid_epochs = np.array(history["epochs"])[valid_indices]
        valid_correlations = np.array(history["correlation"])[valid_indices]
        plt.plot(
            valid_epochs, valid_correlations, "g-", linewidth=2, label="Correlation"
        )

        # Add horizontal line at 0.45 for reference
        plt.axhline(y=0.45, color="r", linestyle="--", label="0.45 threshold")

        # Get the max correlation achieved and its epoch
        max_corr = np.max(valid_correlations) if len(valid_correlations) > 0 else 0
        max_corr_epoch = (
            valid_epochs[np.argmax(valid_correlations)]
            if len(valid_correlations) > 0
            else 0
        )

        # Add annotation for max correlation
        plt.annotate(
            f"Max: {max_corr:.4f} (epoch {max_corr_epoch})",
            xy=(max_corr_epoch, max_corr),
            xytext=(max_corr_epoch + 5, max_corr + 0.02),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
            fontsize=12,
        )

        plt.title(
            f"Correlation Progress Over Training ({norm_type}, {activation_name}, {subsystem_num_layers}-layer)"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Pearson Correlation")
        plt.grid(True)
        plt.legend(loc="lower right")

        # Save final correlation plot
        if subsystem_num_layers > 1:
            corr_title = f"dcell_{norm_type}_{activation_name}_{subsystem_num_layers}layer_correlation_final"
        else:
            corr_title = f"dcell_{norm_type}_{activation_name}_correlation_final"

        corr_save_path = osp.join(
            os.environ["ASSET_IMAGES_DIR"], f"{corr_title}_{timestamp()}.png"
        )
        plt.savefig(corr_save_path, dpi=300)
        print(f"\nFinal correlation plot saved to '{corr_save_path}'")
        plt.close()

    # Print final loss and correlation values
    print("\nFinal values:")
    print(f"  Total Loss: {history['total_loss'][-1]:.6f}")
    print(f"  Primary Loss: {history['primary_loss'][-1]:.6f}")
    print(f"  Auxiliary Loss: {history['auxiliary_loss'][-1]:.6f}")
    print(f"  Weighted Auxiliary Loss: {history['weighted_auxiliary_loss'][-1]:.6f}")

    # Print correlation statistics
    valid_correlations = [c for c in history["correlation"] if not np.isnan(c)]
    if valid_correlations:
        max_corr = max(valid_correlations)
        max_corr_epoch = history["epochs"][history["correlation"].index(max_corr)]
        print(f"  Final Correlation: {valid_correlations[-1]:.6f}")
        print(f"  Max Correlation: {max_corr:.6f} (at epoch {max_corr_epoch})")

    # Print time statistics
    avg_time = sum(history["time_per_epoch"]) / len(history["time_per_epoch"])
    print(f"\nTime statistics:")
    print(f"  Average time per epoch: {avg_time:.3f}s")
    print(f"  First epoch time: {history['time_per_epoch'][0]:.3f}s")
    print(f"  Last epoch time: {history['time_per_epoch'][-1]:.3f}s")
    print(f"  Min epoch time: {min(history['time_per_epoch']):.3f}s")
    print(f"  Max epoch time: {max(history['time_per_epoch']):.3f}s")

    # Verify BatchNorm handling with batch size 1
    print("\nVerifying BatchNorm handling with batch size 1...")
    if batch.num_graphs > 1:
        # Extract single sample for testing
        single_batch = batch[0]
        single_batch.num_graphs = 1
        try:
            predictions_single, _ = model(cell_graph, single_batch)
            print(
                f"✓ Single batch forward pass succeeded, shape: {predictions_single.shape}"
            )
            print("  BatchNorm handling for single samples is working correctly")
        except Exception as e:
            print(f"❌ Single batch forward pass failed: {e}")
            print("  This suggests a problem with the BatchNorm safety handling")

    return model, history


if __name__ == "__main__":
    main()
