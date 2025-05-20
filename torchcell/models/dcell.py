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
        # Set device (use CUDA if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure input tensor is on the same device
        x = x.to(device)

        # Apply each layer of the MLP in sequence
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # Apply linear transformation - ensure on correct device
            layer = layer.to(device)
            x = layer(x)

            # Apply normalization and activation in the specified order
            if self.norm_before_act:
                # Norm -> Act order
                if norm is not None:
                    # Move normalization to correct device
                    norm = norm.to(device)
                    # Handle batch size safely for BatchNorm - this is a legitimate technical constraint
                    if self.norm_type == "batch" and x.size(0) == 1:
                        # Skip BatchNorm for single samples to avoid the error
                        pass
                    else:
                        x = norm(x)
                # Apply activation (activation functions don't have device-specific parameters)
                x = self.activation(x)
            else:
                # Act -> Norm order (original DCell approach)
                x = self.activation(x)
                if norm is not None:
                    # Move normalization to correct device
                    norm = norm.to(device)
                    # Handle batch size safely for BatchNorm - this is a legitimate technical constraint
                    if self.norm_type == "batch" and x.size(0) == 1:
                        # Skip BatchNorm for single samples to avoid the error
                        pass
                    else:
                        x = norm(x)

        # Make sure output is on the correct device before returning
        return x.to(device)


class DCell(nn.Module):
    """
    Reimplementation of DCell model that works directly with PyTorch Geometric HeteroData.

    This implementation:
    1. Uses the gene_ontology structure from cell_graph
    2. Processes mutant state tensors generated by DCellGraphProcessor
    3. Handles batches of multiple samples efficiently using vectorized operations
    4. Supports multi-layer MLPs for each subsystem (when subsystem_num_layers > 1)
    5. Can use stratum-based parallel processing to optimize performance

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
        use_stratum_optimization: Whether to use stratum-based optimization for parallel processing
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

        # Cache for strata organization
        self.stratum_to_systems = None
        self.sorted_strata = None

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

        # Verify we have stratum information for optimization
        if not hasattr(cell_graph["gene_ontology"], "strata"):
            raise ValueError(
                "Gene ontology must have strata information for stratum-based optimization"
            )
        else:
            print(
                f"Using stratum-based processing with {len(torch.unique(cell_graph['gene_ontology'].strata))} strata"
            )

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

            # Add stratum information if available
            node_attrs = {
                "id": i,
                "gene_set": gene_names if gene_names else gene_indices,
                "namespace": "biological_process",  # Default, might be updated later if info is available
                # Initialize empty mutant state
                "mutant_state": torch.ones(len(gene_indices), dtype=torch.float32),
            }

            # Add stratum information if available
            if hasattr(cell_graph["gene_ontology"], "strata") and i < len(
                cell_graph["gene_ontology"].strata
            ):
                stratum = cell_graph["gene_ontology"].strata[i].item()
                node_attrs["stratum"] = stratum

            # Add node with all required attributes
            go_graph.add_node(term_id, **node_attrs)

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

            # Find max stratum if available for assigning to ROOT
            max_stratum = -1
            if hasattr(cell_graph["gene_ontology"], "strata"):
                max_stratum = cell_graph["gene_ontology"].strata.max().item()

            # Add super-root node
            go_graph.add_node(
                "GO:ROOT",
                name="GO Super Node",
                namespace="super_root",
                level=-1,
                stratum=max_stratum + 1,  # Place ROOT in the highest stratum
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

        # Group subsystems by stratum for parallel processing
        from collections import defaultdict

        self.stratum_to_systems = None
        if hasattr(cell_graph["gene_ontology"], "strata"):
            self.stratum_to_systems = defaultdict(list)

            # Map each subsystem to its stratum
            for term_id, subsystem in self.subsystems.items():
                if term_id == "GO:ROOT":
                    # ROOT is always in the highest stratum
                    stratum = go_graph.nodes[term_id].get("stratum", 0)
                else:
                    # Get the stratum from the graph node
                    stratum = go_graph.nodes[term_id].get("stratum", 0)

                # Store subsystem with its stratum
                self.stratum_to_systems[stratum].append((term_id, subsystem))

            # Sort strata in reverse order to process from leaves to root
            # Higher strata numbers (leaves) should be processed before lower strata (root)
            self.sorted_strata = sorted(self.stratum_to_systems.keys(), reverse=True)

            # Print statistics about stratum grouping
            print(
                f"Grouped subsystems into {len(self.stratum_to_systems)} strata for parallel processing"
            )
            for stratum in sorted(self.stratum_to_systems.keys())[:5]:
                print(
                    f"  Stratum {stratum}: {len(self.stratum_to_systems[stratum])} subsystems"
                )
            if len(self.stratum_to_systems) > 5:
                print(f"  ... and {len(self.stratum_to_systems)-5} more strata")

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
        Vectorized forward pass for the DCell model using stratum-based parallelization.
        Processes GO terms in parallel by stratum, processing each stratum sequentially.
        This optimized version groups subsystems by input/output dimensions for batch processing.
        
        Optimized version with strict device management to ensure GPU compatibility.

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

        # Set device (use CUDA if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Log the device we're using
        if not hasattr(self, '_device_logged') or not self._device_logged:
            print(f"DCell model running on device: {device}")
            self._device_logged = True

        # Ensure data is on the correct device - create a recursive function to handle all nested tensors
        def ensure_tensor_on_device(data_obj, target_device):
            """Recursively ensure all tensors in a nested structure are on the target device."""
            if torch.is_tensor(data_obj):
                return data_obj.to(target_device)
            elif isinstance(data_obj, dict):
                return {k: ensure_tensor_on_device(v, target_device) for k, v in data_obj.items()}
            elif isinstance(data_obj, (list, tuple)):
                return type(data_obj)(ensure_tensor_on_device(x, target_device) for x in data_obj)
            elif hasattr(data_obj, 'to'):  # For HeteroData and other PyG objects
                return data_obj.to(target_device)
            return data_obj
            
        # Move all data to device
        cell_graph = cell_graph.to(device)
        batch = batch.to(device)
        num_graphs = batch.num_graphs

        # Verify mutant state exists and has the right format
        if not hasattr(batch["gene_ontology"], "mutant_state"):
            raise ValueError(
                "Batch must contain gene_ontology.mutant_state for DCell model"
            )

        # Make sure mutant_state is on the correct device
        mutant_state = batch["gene_ontology"].mutant_state.to(device)

        # Verify mutant state has the right number of columns
        # (either 4 columns [term_idx, gene_idx, stratum, gene_state] or
        #  5 columns [term_idx, gene_idx, stratum, level, gene_state])
        if mutant_state.shape[1] != 4 and mutant_state.shape[1] != 5:
            raise ValueError(
                f"Mutant state must have 4 or 5 columns, got {mutant_state.shape[1]}"
            )

        # Verify strata groups are available
        if self.stratum_to_systems is None:
            raise ValueError("Stratum to systems mapping is not initialized")

        # Map from term IDs to indices in the cell_graph
        term_id_to_idx = {
            term_id: idx
            for idx, term_id in enumerate(cell_graph["gene_ontology"].node_ids)
        }

        # Dictionary to store all subsystem outputs
        subsystem_outputs = {}

        # Extract information from mutant state - ensure on correct device
        term_indices = mutant_state[:, 0].long().to(device)
        gene_indices = mutant_state[:, 1].long().to(device)
        strata_indices = mutant_state[:, 2].long().to(device)

        # Handle either 4-column or 5-column format
        if mutant_state.shape[1] == 4:
            # 4-column format: [term_idx, gene_idx, stratum, gene_state]
            gene_states = mutant_state[:, 3].to(device)
        else:
            # 5-column format: [term_idx, gene_idx, stratum, level, gene_state]
            gene_states = mutant_state[:, 4].to(device)

        # Extract batch information if available
        if hasattr(batch["gene_ontology"], "mutant_state_batch"):
            batch_indices = batch["gene_ontology"].mutant_state_batch.to(device)
        else:
            batch_indices = torch.zeros(
                mutant_state.size(0), dtype=torch.long, device=device
            )

        # Pre-process gene states for all terms - make sure everything is on device
        term_gene_states = {}
        term_indices_set = torch.unique(term_indices).to(device).tolist()

        # Efficiently process gene states for each term
        for term_idx in term_indices_set:
            term_idx = term_idx if isinstance(term_idx, int) else term_idx.item()

            # Get genes for this term - ensure all data structure on the right device
            term_to_gene_dict = ensure_tensor_on_device(
                cell_graph["gene_ontology"].term_to_gene_dict, device
            )
            genes = term_to_gene_dict.get(term_idx, [])
            num_genes = max(1, len(genes))

            # Create gene state tensor based on embedding mode - directly on device
            if self.learnable_embedding_dim is not None:
                # Initialize tensor with embeddings - [batch_size, num_genes, embedding_dim]
                term_gene_states[term_idx] = torch.zeros(
                    (num_graphs, num_genes, self.learnable_embedding_dim),
                    dtype=torch.float,
                    device=device,
                )

                # Set default embeddings
                for gene_local_idx, gene_idx in enumerate(genes):
                    if gene_idx < self.gene_embeddings.weight.size(0):
                        term_gene_states[term_idx][:, gene_local_idx] = (
                            self.gene_embeddings.weight[gene_idx]
                        )
            else:
                # Binary encoding - all genes present (1.0) by default
                term_gene_states[term_idx] = torch.ones(
                    (num_graphs, num_genes), dtype=torch.float, device=device
                )

            # Apply perturbations from mutant state
            term_mask = term_indices == term_idx
            term_data = mutant_state[term_mask]

            if term_data.size(0) > 0:
                # Get batch indices for this term
                term_batch_indices = (
                    batch_indices[term_mask]
                    if batch_indices is not None
                    else torch.zeros(term_data.size(0), dtype=torch.long, device=device)
                )

                # Apply perturbations
                for i in range(term_data.size(0)):
                    batch_idx = term_batch_indices[i].item()
                    gene_idx = term_data[i, 1].long().item()

                    # Get state value based on mutant_state format
                    if mutant_state.shape[1] == 4:
                        state_value = term_data[i, 3].item()
                    else:
                        state_value = term_data[i, 4].item()

                    # Find gene in the term's gene list
                    if gene_idx < len(genes):
                        gene_local_idx = (
                            genes.index(gene_idx) if gene_idx in genes else -1
                        )
                        if gene_local_idx >= 0 and state_value != 1.0:
                            # Zero out gene or embedding for perturbed genes
                            if self.learnable_embedding_dim is not None:
                                term_gene_states[term_idx][
                                    batch_idx, gene_local_idx
                                ] = 0.0
                            else:
                                term_gene_states[term_idx][
                                    batch_idx, gene_local_idx
                                ] = 0.0

        # Process strata in order (from lowest to highest)
        for stratum in self.sorted_strata:
            # Get all systems at this stratum
            stratum_systems = self.stratum_to_systems[stratum]

            # Prepare for batch processing of subsystems in this stratum
            # Group subsystems by input and output sizes for efficient batch processing
            term_ids = []
            subsystem_models = []
            input_sizes = []
            output_sizes = []
            combined_inputs = []

            # First, collect data for all subsystems in this stratum
            for term_id, subsystem_model in stratum_systems:
                # Get the term index
                if term_id == "GO:ROOT":
                    term_idx = -1
                else:
                    term_idx = term_id_to_idx.get(term_id, -1)
                    if term_idx == -1:
                        continue

                # Get gene states for this term
                if term_idx in term_gene_states:
                    gene_states = term_gene_states[term_idx]
                else:
                    # Create default state tensor for terms not in mutant_state
                    term_to_gene_dict = ensure_tensor_on_device(
                        cell_graph["gene_ontology"].term_to_gene_dict, device
                    )
                    genes = term_to_gene_dict.get(term_idx, [])

                    if self.learnable_embedding_dim is not None:
                        gene_states = torch.zeros(
                            (
                                num_graphs,
                                max(1, len(genes)),
                                self.learnable_embedding_dim,
                            ),
                            dtype=torch.float,
                            device=device,
                        )

                        for gene_local_idx, gene_idx in enumerate(genes):
                            if gene_idx < self.gene_num:
                                gene_states[:, gene_local_idx] = (
                                    self.gene_embeddings.weight[gene_idx]
                                )
                    else:
                        gene_states = torch.ones(
                            (num_graphs, max(1, len(genes))),
                            dtype=torch.float,
                            device=device,
                        )

                # Reshape embeddings if needed
                if (
                    self.learnable_embedding_dim is not None
                    and len(gene_states.shape) == 3
                ):
                    batch_size = gene_states.size(0)
                    gene_states = gene_states.reshape(batch_size, -1)

                # Get outputs from child nodes
                child_outputs = []
                for child in self.go_graph.successors(term_id):
                    if child in subsystem_outputs:
                        child_outputs.append(subsystem_outputs[child])

                # Combine inputs
                if child_outputs:
                    child_tensor = torch.cat(child_outputs, dim=1)

                    if term_id == "GO:ROOT":
                        # Special handling for root node
                        combined_input = child_tensor
                        padding = torch.zeros(
                            (combined_input.size(0), 1), device=device
                        )
                        combined_input = torch.cat([combined_input, padding], dim=1)
                    else:
                        combined_input = torch.cat([gene_states, child_tensor], dim=1)
                else:
                    combined_input = gene_states

                # Validate input size
                expected_size = subsystem_model.layers[0].weight.size(1)
                actual_size = combined_input.size(1)

                if actual_size != expected_size:
                    # Provide helpful diagnostics for size mismatch
                    if term_id == "GO:ROOT":
                        print(f"\nDiagnostic information for GO:ROOT node:")
                        print(f"  Expected input size: {expected_size}")
                        print(f"  Actual input size: {actual_size}")

                        child_output_sum = 0
                        for child in self.go_graph.successors(term_id):
                            if child in subsystem_outputs:
                                child_size = subsystem_outputs[child].size(1)
                                child_output_sum += child_size
                                print(f"  Child '{child}' output size: {child_size}")

                        print(f"  Sum of child output sizes: {child_output_sum}")

                    # Fail with clear error message
                    raise ValueError(
                        f"Size mismatch for subsystem '{term_id}' in stratum {stratum}: "
                        f"expected {expected_size}, got {actual_size}."
                    )

                # Ensure the combined input is on the correct device
                combined_input = combined_input.to(device)
                
                # Save all data for batch processing
                term_ids.append(term_id)
                subsystem_models.append(subsystem_model)
                combined_inputs.append(combined_input)
                input_sizes.append(expected_size)
                output_sizes.append(subsystem_model.output_size)

            # Now group by input size and output size for parallel processing
            # This way we can batch process subsystems with the same input/output dimensions
            groups = {}
            for i, (term_id, model, inp_size, out_size, combined_input) in enumerate(
                zip(
                    term_ids,
                    subsystem_models,
                    input_sizes,
                    output_sizes,
                    combined_inputs,
                )
            ):
                key = (inp_size, out_size)
                if key not in groups:
                    groups[key] = []
                groups[key].append((i, term_id, model, combined_input))

            # Process each group in batches
            for (inp_size, out_size), group in groups.items():
                indices = [g[0] for g in group]
                batch_term_ids = [g[1] for g in group]
                batch_models = [g[2] for g in group]
                batch_inputs = [g[3] for g in group]

                # Skip batch processing for single subsystems
                if len(batch_inputs) == 1:
                    term_id = batch_term_ids[0]
                    model = batch_models[0]
                    combined_input = batch_inputs[0]
                    # Ensure input is on the right device
                    combined_input = combined_input.to(device)
                    output = model(combined_input)
                    subsystem_outputs[term_id] = output
                    continue

                # Process in batches when we have multiple subsystems with the same dimensions
                # This will use GPU parallelism to speed up computation
                try:
                    # Stack inputs (these all have the same shape)
                    batch_size = batch_inputs[0].size(0)
                    num_subsystems = len(batch_inputs)
                    
                    # Ensure all inputs are on the correct device
                    batch_inputs = [input_tensor.to(device) for input_tensor in batch_inputs]

                    # Stack inputs into a single tensor
                    stacked_inputs = torch.cat(batch_inputs, dim=0).to(device)

                    # For simple models (1 layer), we can optimize with a batched matrix multiply
                    if all(model.num_layers == 1 for model in batch_models):
                        # Stack all model weights and biases for the first layer
                        weights = torch.stack(
                            [model.layers[0].weight.to(device) for model in batch_models]
                        )
                        biases = torch.stack(
                            [model.layers[0].bias.to(device) for model in batch_models]
                        )

                        # Create a big batch multiplication
                        # Reshape inputs to (num_subsystems, batch_size, input_size)
                        reshaped_inputs = stacked_inputs.view(
                            num_subsystems, batch_size, inp_size
                        )

                        # Perform batch matrix multiplication
                        # weights: (num_subsystems, out_size, inp_size)
                        # reshaped_inputs: (num_subsystems, batch_size, inp_size)
                        # Result: (num_subsystems, batch_size, out_size)
                        outputs = torch.bmm(reshaped_inputs, weights.transpose(1, 2))

                        # Add biases: biases is (num_subsystems, out_size)
                        # Need to reshape to (num_subsystems, 1, out_size) for broadcasting
                        outputs = outputs + biases.unsqueeze(1)

                        # Apply activation and normalization
                        normalized_outputs = []
                        for i, model in enumerate(batch_models):
                            # Extract this model's output
                            output = outputs[i].to(device)  # Shape: (batch_size, out_size)

                            # Apply activation
                            output = model.activation(output)

                            # Apply normalization if needed
                            if model.norms[0] is not None and model.norm_type != "none":
                                # Skip BatchNorm for single samples
                                if model.norm_type == "batch" and output.size(0) == 1:
                                    pass
                                else:
                                    output = model.norms[0](output)

                            normalized_outputs.append(output)

                        # Store the results
                        for i, term_id in enumerate(batch_term_ids):
                            subsystem_outputs[term_id] = normalized_outputs[i]

                    # For multi-layer models, process each subsystem individually
                    else:
                        for i, (term_id, model, combined_input) in enumerate(
                            zip(batch_term_ids, batch_models, batch_inputs)
                        ):
                            # Ensure input is on the right device
                            combined_input = combined_input.to(device)
                            output = model(combined_input)
                            subsystem_outputs[term_id] = output

                except Exception as e:
                    # Fallback to sequential processing if batched processing fails
                    print(
                        f"Batch processing failed with error: {e}. Falling back to sequential processing."
                    )
                    for i, (term_id, model, combined_input) in enumerate(
                        zip(batch_term_ids, batch_models, batch_inputs)
                    ):
                        # Ensure input is on the right device
                        combined_input = combined_input.to(device)
                        output = model(combined_input)
                        subsystem_outputs[term_id] = output

        # Get the root output
        if "GO:ROOT" in subsystem_outputs:
            root_output = subsystem_outputs["GO:ROOT"].to(device)
        else:
            raise ValueError("Root node 'GO:ROOT' not found in outputs")

        # Add flag to track first device logging
        if not hasattr(self, '_device_logged'):
            self._device_logged = True

        # Return both the root output and all subsystem outputs for auxiliary loss calculation
        # Ensure everything is on the right device
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
        Uses batch processing for improved performance.
        
        Optimized version with strict device management to ensure GPU compatibility.

        Args:
            subsystem_outputs: Dictionary mapping subsystem names to their outputs

        Returns:
            Dictionary mapping subsystem names to transformed outputs
        """
        # Set device (use CUDA if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dictionary to store outputs
        linear_outputs = {}

        # Group subsystems by output size for batch processing
        subsystems_by_size = {}
        for subsystem_name, subsystem_output in subsystem_outputs.items():
            if subsystem_name in self.subsystem_linears:
                # Ensure output is on correct device before measuring shape
                subsystem_output = subsystem_output.to(device)
                output_size = subsystem_output.shape[1]
                if output_size not in subsystems_by_size:
                    subsystems_by_size[output_size] = []
                subsystems_by_size[output_size].append(
                    (subsystem_name, subsystem_output)
                )

        # Process each group in batches
        for output_size, subsystem_group in subsystems_by_size.items():
            # Get all subsystem names and outputs for this size
            names = [item[0] for item in subsystem_group]
            outputs = [item[1].to(device) for item in subsystem_group]  # Ensure all outputs on same device

            # Skip batch processing for small groups to avoid overhead
            if len(names) <= 1:
                for name, output in subsystem_group:
                    # Ensure both the linear layer and input are on the same device
                    linear = self.subsystem_linears[name].to(device)
                    output = output.to(device)
                    linear_outputs[name] = linear(output)
                continue

            # Create a batch of outputs for parallel processing
            stacked_outputs = torch.cat(outputs, dim=0).to(device)
            batch_sizes = [output.shape[0] for output in outputs]

            # Perform the transformation for each subsystem separately (more robust approach)
            start_idx = 0
            for i, (name, output) in enumerate(subsystem_group):
                # Ensure the linear layer is on the device
                linear = self.subsystem_linears[name].to(device)
                # Ensure output is on the device and apply transformation
                output = output.to(device)
                result = linear(output)
                linear_outputs[name] = result.to(device)

        # Ensure all outputs are on the correct device
        for name in linear_outputs:
            linear_outputs[name] = linear_outputs[name].to(device)
            
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
        
        Optimized version with strict device management to ensure GPU compatibility.

        Args:
            cell_graph: The cell graph containing gene ontology structure
            batch: HeteroDataBatch containing perturbation information

        Returns:
            Tuple of (predictions, outputs dictionary)
        """
        # Select target device - use CUDA if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure input data is on the correct device
        cell_graph = cell_graph.to(device)
        batch = batch.to(device)
        
        # Make sure DCell is on the right device
        self.dcell = self.dcell.to(device)
        
        # Process through DCell - this will initialize all parameters
        root_output, outputs = self.dcell(cell_graph, batch)
        subsystem_outputs = outputs["subsystem_outputs"]

        # Initialize or update DCellLinear using the DCell component's actual subsystems
        if not self._initialized:
            # Replace the dummy DCellLinear with one that uses the actual subsystems
            self.dcell_linear = DCellLinear(
                subsystems=self.dcell.subsystems, output_size=self.output_size
            ).to(device)
            self._initialized = True
        else:
            # Ensure DCellLinear is on the right device
            self.dcell_linear = self.dcell_linear.to(device)

        # Apply linear transformation to all subsystem outputs
        linear_outputs = self.dcell_linear(subsystem_outputs)
        outputs["linear_outputs"] = linear_outputs

        # Find the root prediction - fail explicitly if not found
        # First try to get prediction for "GO:ROOT"
        if "GO:ROOT" in linear_outputs:
            predictions = linear_outputs["GO:ROOT"].to(device)
        else:
            # No fallback - raise an error if root node prediction is missing
            raise ValueError(
                "Root node 'GO:ROOT' prediction not found in linear outputs. "
                "This indicates a problem with the DCellLinear component or GO hierarchy."
            )

        # Make sure everything in the outputs dictionary is on the correct device
        # This is a shallow check that ensures at least the top-level tensors are on the right device
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value.to(device)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        outputs[key][subkey] = subvalue.to(device)

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


def main():
    """
    Main function to test the DCellModel on a batch of data.
    Overfits the model on a single batch and produces loss component plots.
    """
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import time
    import os
    import os.path as osp
    import numpy as np
    from torchcell.scratch.load_batch_005 import load_sample_data_batch
    from torchcell.losses.dcell import DCellLoss
    from torchcell.timestamp import timestamp
    from collections import defaultdict
    from tqdm.auto import tqdm
    from dotenv import load_dotenv

    # Load environment variables for asset paths
    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "assets/images")

    print("\n" + "=" * 80)
    print("DCELL TRAINING TEST")
    print("=" * 80)

    # Configure default parameters for the test
    batch_size = 32
    norm_type = "batch"
    norm_before_act = False
    subsystem_num_layers = 1
    activation = nn.Tanh()
    activation_name = "tanh"
    init_range = 0.001
    num_workers = 0
    num_epochs = 500
    plot_every = 50

    # Load test data
    print("\nLoading test data...")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=batch_size, num_workers=num_workers, config="dcell", is_dense=False
    )

    # Set device (default to CPU for testing)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print dataset information
    print(f"Dataset: {len(dataset)} samples")
    print(f"Batch: {batch.num_graphs} graphs")
    print(f"Max Number of Nodes: {max_num_nodes}")

    # Move data to device
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Print GO hierarchy information
    if hasattr(cell_graph["gene_ontology"], "strata"):
        strata = cell_graph["gene_ontology"].strata
        num_strata = len(torch.unique(strata))
        print(f"\nGO hierarchy has {num_strata} strata for parallel processing")

        # Count terms per stratum
        stratum_counts = defaultdict(int)
        for s in strata:
            stratum_counts[s.item()] += 1

        print("Distribution of GO terms by stratum:")
        for stratum in sorted(stratum_counts.keys())[:5]:
            print(f"  Stratum {stratum}: {stratum_counts[stratum]} terms")
        if len(stratum_counts) > 5:
            print(f"  ... and {len(stratum_counts)-5} more strata")

    # Print model configuration
    print(f"\nModel configuration:")
    print(f"  - Normalization type: {norm_type}")
    print(f"  - Normalization order: {'norm->act' if norm_before_act else 'act->norm'}")

    if subsystem_num_layers == 1:
        print(f"  - Subsystem architecture: Single layer (original DCell)")
    else:
        print(f"  - Subsystem architecture: {subsystem_num_layers}-layer MLP")
        print(
            f"    Each subsystem uses multiple linear layers with {activation_name} activation"
        )

    print(f"  - Activation function: {activation_name}")
    print(f"  - Weight initialization range: ±{init_range}")

    # Initialize model with default parameters
    print("\nCreating DCellModel with stratum-based optimization...")
    model_params = {
        "gene_num": max_num_nodes,
        "subsystem_output_min": 20,  # Default value
        "subsystem_output_max_mult": 0.3,  # Default value
        "output_size": 1,  # Single output for fitness prediction
        "norm_type": norm_type,
        "norm_before_act": norm_before_act,
        "subsystem_num_layers": subsystem_num_layers,
        "activation": activation,
        "init_range": init_range,
    }

    # Time model initialization and initial forward pass
    start_time = time.time()
    model = DCellModel(**model_params).to(device)
    init_time = time.time() - start_time
    print(f"Model initialization time: {init_time:.3f}s")

    # Run a forward pass to initialize the model
    print("\nRunning initial forward pass to initialize model...")
    with torch.no_grad():
        predictions, _ = model(cell_graph, batch)

        # Check prediction diversity
        diversity = predictions.std().item()
        print(f"Initial predictions diversity: {diversity:.6f}")

        if diversity < 1e-6:
            print("WARNING: Predictions lack diversity!")
        else:
            print("✓ Predictions are diverse")

    # Print parameter count
    param_info = model.num_parameters
    total_params = param_info.get("total", 0)
    print(f"Model parameters: {total_params:,}")
    print(f"Subsystems: {param_info.get('num_subsystems', 0):,}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    print(f"Using Adam optimizer with lr=0.001, weight_decay=1e-5")

    # Create loss function
    criterion = DCellLoss(alpha=0.3, use_auxiliary_losses=True)
    print(f"Using DCellLoss with alpha=0.3, use_auxiliary_losses=True")

    # Get target
    target = batch["gene"].phenotype_values.view_as(
        torch.zeros(batch.num_graphs, 1, device=device)
    )

    # Initialize history for loss tracking
    history = {
        "total_loss": [],
        "primary_loss": [],
        "auxiliary_loss": [],
        "weighted_auxiliary_loss": [],
        "epochs": [],
        "time_per_epoch": [],
        "correlation": [],
    }

    # Training loop with tqdm progress bar
    print(f"\nOverfitting model on a single batch for {num_epochs} epochs...")
    progress_bar = tqdm(range(num_epochs), desc="Training")

    for epoch in progress_bar:
        epoch_start = time.time()

        # Forward pass
        predictions, outputs = model(cell_graph, batch)

        # Extract prediction and target values for correlation tracking
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
        if len(pred_values) > 1:  # Only if more than one sample
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

        # Record time for this epoch
        epoch_time = time.time() - epoch_start
        history["time_per_epoch"].append(epoch_time)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        progress_bar.set_description(
            f"Loss: {loss.item():.6f}{corr_str}, Time: {epoch_time:.3f}s/epoch"
        )

        # Plot curves at regular intervals
        if (epoch + 1) % plot_every == 0:
            # Create output directory if needed
            os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

            # Plot loss components
            plt.figure(figsize=(10, 6))
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

            title_suffix = f" ({norm_type}, {activation_name}"
            if subsystem_num_layers > 1:
                title_suffix += f", {subsystem_num_layers}-layer)"
            else:
                title_suffix += ")"

            plt.title(f"DCellNew Loss Components - Epoch {epoch+1}{title_suffix}")
            plt.legend()
            plt.grid(True)

            # Save figure
            if subsystem_num_layers > 1:
                title = f"dcell_{norm_type}_{activation_name}_{subsystem_num_layers}layer_loss_components_epoch_{epoch+1}"
            else:
                title = f"dcell_{norm_type}_{activation_name}_loss_components_epoch_{epoch+1}"

            save_path = osp.join(ASSET_IMAGES_DIR, f"{title}_{timestamp()}.png")
            plt.savefig(save_path)
            print(f"Saved loss components plot to {save_path}")
            plt.close()

            # Plot correlation progress
            plt.figure(figsize=(10, 6))
            valid_indices = ~np.isnan(np.array(history["correlation"]))
            if np.any(valid_indices):
                valid_epochs = np.array(history["epochs"])[valid_indices]
                valid_correlations = np.array(history["correlation"])[valid_indices]
                plt.plot(valid_epochs, valid_correlations, "g-", linewidth=2)

                # Add reference line at 0.45
                plt.axhline(y=0.45, color="r", linestyle="--", label="0.45 threshold")

                # Get max correlation
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
                    ASSET_IMAGES_DIR, f"{corr_title}_{timestamp()}.png"
                )
                plt.savefig(corr_save_path)
                print(f"Saved correlation plot to {corr_save_path}")
                plt.close()

    # Create final detailed loss components plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Main loss components (log scale)
    ax1.semilogy(history["total_loss"], "b-", linewidth=2, label="Total Loss")
    ax1.semilogy(history["primary_loss"], "r-", linewidth=2, label="Primary Loss")
    ax1.semilogy(
        history["weighted_auxiliary_loss"],
        "g-",
        linewidth=2,
        label="Weighted Auxiliary Loss",
    )
    ax1.set_ylabel("Loss Value (log scale)", fontsize=12)

    title_suffix = f" ({norm_type}, {activation_name}"
    if subsystem_num_layers > 1:
        title_suffix += f", {subsystem_num_layers}-layer MLP)"
    else:
        title_suffix += ")"

    ax1.set_title(f"DCell Loss Components During Training{title_suffix}", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Add epoch markers
    for e in range(0, num_epochs, 50):
        if e > 0:
            ax1.axvline(x=e, color="gray", linestyle="--", alpha=0.3)

    # Auxiliary loss plot
    ax2.semilogy(
        history["auxiliary_loss"],
        "orange",
        linewidth=2,
        label="Auxiliary Loss (Unweighted)",
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Auxiliary Loss Value (log scale)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Add final values to the legend
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

    # Save final loss plot
    if subsystem_num_layers > 1:
        title = f"dcell_{norm_type}_{activation_name}_{subsystem_num_layers}layer_loss_components_final"
    else:
        title = f"dcell_{norm_type}_{activation_name}_loss_components_final"

    save_path = osp.join(ASSET_IMAGES_DIR, f"{title}_{timestamp()}.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nDetailed loss components plot saved to '{save_path}'")
    plt.close()

    # Create predictions vs targets plot
    if len(pred_values) > 1:
        plt.figure(figsize=(12, 9))

        # Scatter plot
        scatter = plt.scatter(
            target_values, pred_values, c="blue", alpha=0.7, s=100, label="Predictions"
        )

        # Add sample indices as annotations
        for i, (x, y) in enumerate(zip(target_values, pred_values)):
            plt.annotate(
                f"{i}",
                (x, y),
                xytext=(3, 3),
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

        # Calculate correlation
        correlation = np.corrcoef(pred_values, target_values)[0, 1]

        # Title
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

        # Add correlation text box
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

        # Save predictions plot
        if subsystem_num_layers > 1:
            title = f"dcell_{norm_type}_{activation_name}_{subsystem_num_layers}layer_predictions_vs_targets"
        else:
            title = f"dcell_{norm_type}_{activation_name}_predictions_vs_targets"

        save_path = osp.join(ASSET_IMAGES_DIR, f"{title}_{timestamp()}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Predictions vs targets plot saved to '{save_path}'")
        plt.close()

    # Create final correlation plot
    plt.figure(figsize=(12, 8))
    valid_indices = ~np.isnan(np.array(history["correlation"]))
    if np.any(valid_indices):
        valid_epochs = np.array(history["epochs"])[valid_indices]
        valid_correlations = np.array(history["correlation"])[valid_indices]
        plt.plot(
            valid_epochs, valid_correlations, "g-", linewidth=2, label="Correlation"
        )

        # Add reference line
        plt.axhline(y=0.45, color="r", linestyle="--", label="0.45 threshold")

        # Annotate max correlation
        max_corr = np.max(valid_correlations) if len(valid_correlations) > 0 else 0
        max_corr_epoch = (
            valid_epochs[np.argmax(valid_correlations)]
            if len(valid_correlations) > 0
            else 0
        )

        plt.annotate(
            f"Max: {max_corr:.4f} (epoch {max_corr_epoch})",
            xy=(max_corr_epoch, max_corr),
            xytext=(max_corr_epoch + 5, max_corr + 0.02),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
            fontsize=12,
        )

        plt.title(f"Correlation Progress Over Training{title_suffix}")
        plt.xlabel("Epoch")
        plt.ylabel("Pearson Correlation")
        plt.grid(True)
        plt.legend(loc="lower right")

        # Save final correlation plot
        if subsystem_num_layers > 1:
            corr_title = f"dcell_{norm_type}_{activation_name}_{subsystem_num_layers}layer_correlation_final"
        else:
            corr_title = f"dcell_{norm_type}_{activation_name}_correlation_final"

        corr_save_path = osp.join(ASSET_IMAGES_DIR, f"{corr_title}_{timestamp()}.png")
        plt.savefig(corr_save_path, dpi=300)
        print(f"\nFinal correlation plot saved to '{corr_save_path}'")
        plt.close()

    # Print final values
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
    print(f"  Total training time: {sum(history['time_per_epoch']):.3f}s")
    print(f"  Samples per epoch: {batch_size}")
    print(f"  Samples per second: {batch_size / avg_time:.1f}")

    return model, history


if __name__ == "__main__":
    main()
