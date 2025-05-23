import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class AdaptiveDAGMessagePassing(MessagePassing):
    def __init__(
        self,
        node_type_dims: Dict[int, int],
        edge_index: torch.Tensor,
        hidden_dim_factor: float = 0.3,
        min_hidden_dim: int = 20,
        out_dim: int = 32,
        root_id: int = 0,
    ):
        """
        Message passing for a DAG with adaptive hidden dimensions based on the DCell approach.

        Args:
            node_type_dims: Dictionary mapping node IDs to their feature dimensions
            edge_index: Edge indices [2, num_edges] where edge_index[0] is source and
                        edge_index[1] is target
            hidden_dim_factor: Factor to scale hidden dimensions (0.3 in DCell)
            min_hidden_dim: Minimum hidden dimension (20 in DCell)
            out_dim: Output dimension after processing
            root_id: ID of the root node
        """
        super().__init__(aggr="add")

        self.node_type_dims = node_type_dims
        self.hidden_dim_factor = hidden_dim_factor
        self.min_hidden_dim = min_hidden_dim
        self.out_dim = out_dim
        self.root_id = root_id

        # Store edge_index for reference
        self.register_buffer("edge_index", edge_index)

        # Build adjacency structure
        self.child_to_parent = defaultdict(list)
        self.parent_to_children = defaultdict(list)

        for i in range(edge_index.shape[1]):
            child = edge_index[0, i].item()
            parent = edge_index[1, i].item()
            self.child_to_parent[child].append(parent)
            self.parent_to_children[parent].append(child)

        # Compute node levels and organize by level
        self.node_levels, self.max_level = self._compute_node_levels()
        self.nodes_by_level = self._group_nodes_by_level()

        # Calculate adaptive hidden dimensions for each node
        self.hidden_dims = self._calculate_hidden_dimensions()
        print(f"Adaptive hidden dimensions: {self.hidden_dims}")

        # Create input MLPs for each node
        self.input_mlps = nn.ModuleDict()
        for node_id, dim in node_type_dims.items():
            if dim > 0:  # Skip nodes with no input features
                self.input_mlps[str(node_id)] = nn.Sequential(
                    nn.Linear(dim, self.hidden_dims[node_id]),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dims[node_id], self.hidden_dims[node_id]),
                )

        # Create combine MLPs for each non-leaf node
        self.combine_mlps = nn.ModuleDict()
        for node_id in node_type_dims.keys():
            if node_id in self.parent_to_children:
                children = self.parent_to_children[node_id]
                # Sum of children's hidden dimensions
                input_dim = sum(self.hidden_dims[child] for child in children)
                self.combine_mlps[str(node_id)] = nn.Sequential(
                    nn.Linear(input_dim, self.hidden_dims[node_id]),
                    nn.ReLU(),
                    nn.Linear(
                        self.hidden_dims[node_id],
                        out_dim if node_id == root_id else self.hidden_dims[node_id],
                    ),
                )

    def _calculate_hidden_dimensions(self) -> Dict[int, int]:
        """
        Calculate adaptive hidden dimensions for each node based on:
        1. For leaf nodes: based on input feature dimensions
        2. For parent nodes: based on the number of genes/features in their descendant nodes

        Similar to DCell: max(min_hidden_dim, ceil(hidden_dim_factor * num_genes))
        """
        hidden_dims = {}

        # First, identify the leaf nodes and their gene counts (input dimensions)
        leaf_gene_counts = {}
        for node_id, dim in self.node_type_dims.items():
            if (
                node_id not in self.parent_to_children
                or not self.parent_to_children[node_id]
            ):
                leaf_gene_counts[node_id] = dim if dim > 0 else 0

        print(f"Leaf gene counts: {leaf_gene_counts}")

        # Propagate gene counts up the hierarchy in reverse level order (bottom-up)
        node_gene_counts = leaf_gene_counts.copy()

        # Process nodes in reverse level order (bottom-up)
        for level in range(self.max_level, -1, -1):
            for node_id in self.nodes_by_level[level]:
                if (
                    node_id in self.parent_to_children
                    and self.parent_to_children[node_id]
                ):
                    # For non-leaf nodes, sum up all gene counts from children
                    total_genes = 0
                    for child in self.parent_to_children[node_id]:
                        if child in node_gene_counts:
                            total_genes += node_gene_counts[child]
                    node_gene_counts[node_id] = total_genes

        print(f"Total gene counts by node: {node_gene_counts}")

        # Calculate hidden dimension for each node using the DCell formula
        for node_id in self.node_type_dims.keys():
            gene_count = node_gene_counts.get(node_id, 0)
            # DCell formula: max(min_hidden_dim, ceil(hidden_dim_factor * num_genes))
            raw_dim = self.hidden_dim_factor * gene_count
            hidden_dims[node_id] = max(self.min_hidden_dim, int(raw_dim + 0.5))  # ceil
            print(
                f"Node {node_id}: genes={gene_count}, raw_dim={raw_dim:.1f}, hidden_dim={hidden_dims[node_id]}"
            )

        return hidden_dims

    def _compute_node_levels(self) -> Tuple[Dict[int, int], int]:
        """Compute level for each node in the DAG (bottom-up)."""
        all_nodes = set(self.node_type_dims.keys())
        node_levels = {node_id: -1 for node_id in all_nodes}

        # Identify leaf nodes (no children)
        leaf_nodes = [
            node_id
            for node_id in all_nodes
            if node_id not in self.parent_to_children
            or not self.parent_to_children[node_id]
        ]

        # Assign level 0 to leaf nodes
        for node_id in leaf_nodes:
            node_levels[node_id] = 0

        # Iteratively assign levels to parents
        max_level = 0
        changed = True

        while changed:
            changed = False

            for node_id in all_nodes:
                # Skip if level already assigned
                if node_levels[node_id] >= 0:
                    continue

                # Check if all children have levels assigned
                if node_id not in self.parent_to_children:
                    continue

                children = self.parent_to_children[node_id]
                if not all(node_levels[child] >= 0 for child in children):
                    continue

                # Assign level as 1 + max level of children
                level = 1 + max(node_levels[child] for child in children)
                node_levels[node_id] = level
                max_level = max(max_level, level)
                changed = True

        return node_levels, max_level

    def _group_nodes_by_level(self) -> Dict[int, List[int]]:
        """Group nodes by their level for level-by-level processing."""
        nodes_by_level = defaultdict(list)
        for node_id, level in self.node_levels.items():
            nodes_by_level[level].append(node_id)
        return nodes_by_level

    def forward(self, x_dict: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Forward pass through the multi-level DAG.

        Args:
            x_dict: Dictionary mapping node IDs to their feature tensors

        Returns:
            Dictionary mapping node IDs to their output features
        """
        # Process input features for all nodes
        node_features = {}
        for node_id, features in x_dict.items():
            if str(node_id) in self.input_mlps:
                node_features[node_id] = self.input_mlps[str(node_id)](features)
            else:
                # Handle nodes without input features
                batch_size = next(iter(x_dict.values())).shape[0]
                device = next(iter(x_dict.values())).device
                node_features[node_id] = torch.zeros(
                    batch_size, self.hidden_dims[node_id], device=device
                )

        # Process nodes level by level (bottom-up)
        output_features = {}

        # Start with leaf nodes (level 0)
        for level in range(self.max_level + 1):
            for node_id in self.nodes_by_level[level]:
                # If node has children, combine their features
                if (
                    node_id in self.parent_to_children
                    and self.parent_to_children[node_id]
                ):
                    children = self.parent_to_children[node_id]
                    # Get all children features and concatenate
                    children_features = [output_features[child] for child in children]
                    concat_features = torch.cat(children_features, dim=1)

                    # Process children features
                    node_out = self.combine_mlps[str(node_id)](concat_features)
                else:
                    # Leaf node, use its processed input features directly
                    node_out = node_features[node_id]

                # Store output features
                output_features[node_id] = node_out

        return output_features


def test_adaptive_dag():
    """Test with a complex DAG structure with grandchildren and great-grandchildren."""
    # Define dimensions for nodes
    node_type_dims = {
        0: 0,  # Root node
        1: 30,  # Level 1: Child of root
        2: 5,  # Level 1: Child of root
        3: 17,  # Level 2: Grandchild (child of 1)
        4: 40,  # Level 2: Grandchild (child of 1)
        5: 6,  # Level 2: Grandchild (child of 2)
        6: 18,  # Level 3: Great-grandchild (child of 3)
        7: 29,  # Level 3: Great-grandchild (child of 5)
        8: 10,  # Level 3: Great-grandchild (child of 5)
    }

    # Edge index representing the multi-level DAG
    # Format: [source, target] where source is the child and target is the parent
    edge_index = torch.tensor(
        [
            # Level 3 → Level 2
            [
                6,
                7,
                8,
                # Level 2 → Level 1
                3,
                4,
                5,
                # Level 1 → Root
                1,
                2,
            ],
            # Parents
            [3, 5, 5, 1, 1, 2, 0, 0],
        ],
        dtype=torch.long,
    )

    out_dim = 32
    root_id = 0

    # Create model with adaptive hidden dimensions
    model = AdaptiveDAGMessagePassing(
        node_type_dims,
        edge_index,
        hidden_dim_factor=1.0,  # Increase from 0.3 to 1.0
        min_hidden_dim=20,  # Decrease from 20 to 10 to see more variation
        out_dim=out_dim,
        root_id=root_id,
    )

    # Print the DAG structure and hidden dimensions
    print("\nDAG Structure:")
    print("Root Node: 0")
    print("Level 1 (Children of Root): 1, 2")
    print("Level 2 (Grandchildren): 3, 4 (children of 1), 5 (child of 2)")
    print("Level 3 (Great-grandchildren): 6 (child of 3), 7, 8 (children of 5)")

    # Create batch of size 2 for testing
    batch_size = 2

    # Create test data - node features
    x_dict = {
        node_id: torch.randn(batch_size, dim)
        for node_id, dim in node_type_dims.items()
        if dim > 0  # Skip nodes with 0 dimensions
    }

    # Forward pass
    output_features = model(x_dict)

    # Print hidden dimensions and output shapes
    print("\nHidden Dimensions:")
    for node_id, dim in model.hidden_dims.items():
        print(f"Node {node_id}: {dim}")

    print("\nOutput Feature Shapes:")
    for node_id, features in output_features.items():
        print(f"Node {node_id}: {features.shape}")

    # Print the root node output shape
    print(f"\nRoot node output shape: {output_features[root_id].shape}")

    return output_features


if __name__ == "__main__":
    outputs = test_adaptive_dag()
