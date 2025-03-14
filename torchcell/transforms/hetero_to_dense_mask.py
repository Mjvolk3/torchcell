from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData
from typing import Dict, Optional
import torch
from torch import Tensor


@functional_transform("hetero_to_dense_mask")
class HeteroToDenseMask(BaseTransform):
    r"""Converts sparse adjacency matrices in a heterogeneous graph to boolean dense masks
    while preserving edge attributes in their original sparse format for memory efficiency.

    Args:
        num_nodes_dict (Dict[str, int], optional): Dictionary mapping node types to
            their desired number of nodes. If not provided for a node type, will
            use the maximum number of nodes found for that type. (default: None)
    """

    def __init__(self, num_nodes_dict: Optional[Dict[str, int]] = None) -> None:
        self.num_nodes_dict = num_nodes_dict or {}

    def forward(self, data: HeteroData) -> HeteroData:
        # First determine number of nodes for each node type
        num_nodes_dict = {}
        for node_type in data.node_types:
            if node_type in self.num_nodes_dict:
                num_nodes = self.num_nodes_dict[node_type]
                orig_num_nodes = data[node_type].num_nodes
                assert orig_num_nodes <= num_nodes
                num_nodes_dict[node_type] = num_nodes
            else:
                num_nodes_dict[node_type] = data[node_type].num_nodes

        # Process each edge type
        for edge_type in data.edge_types:
            src, rel, dst = edge_type
            store = data[edge_type]

            # Handle regular edge_index (for node-to-node relations)
            if hasattr(store, "edge_index") and store.edge_index is not None:
                src_num_nodes = num_nodes_dict[src]
                dst_num_nodes = num_nodes_dict[dst]
                edge_index = store.edge_index
                
                # Create boolean dense adjacency matrix (8x memory savings)
                adj_mask = torch.zeros(
                    (src_num_nodes, dst_num_nodes), 
                    dtype=torch.bool,
                    device=edge_index.device
                )
                
                # Fill the mask with valid edges
                valid_edges = (edge_index[0] < src_num_nodes) & (edge_index[1] < dst_num_nodes)
                valid_edge_index = edge_index[:, valid_edges]
                
                if valid_edge_index.size(1) > 0:
                    adj_mask[valid_edge_index[0], valid_edge_index[1]] = True
                
                # Store the boolean adjacency mask
                store.adj_mask = adj_mask
                
                # KEEP the original edge_index and edge_attr for attribute reference

            # Handle hyperedge_index (for hypergraph/bipartite relations)
            elif hasattr(store, "hyperedge_index") and store.hyperedge_index is not None:
                src_num_nodes = num_nodes_dict[src]
                dst_num_nodes = num_nodes_dict[dst]
                hyperedge_index = store.hyperedge_index

                # Create a boolean bipartite incidence matrix
                inc_mask = torch.zeros(
                    (src_num_nodes, dst_num_nodes), 
                    dtype=torch.bool,
                    device=hyperedge_index.device
                )

                # Fill in the incidence matrix from the hyperedge_index
                valid_edges = (hyperedge_index[0] < src_num_nodes) & (hyperedge_index[1] < dst_num_nodes)
                valid_he_index = hyperedge_index[:, valid_edges]
                
                if valid_he_index.size(1) > 0:
                    inc_mask[valid_he_index[0], valid_he_index[1]] = True

                # Store the boolean incidence matrix
                store.inc_mask = inc_mask
                
                # KEEP the original hyperedge_index and attributes for reference

        # Handle node features for each node type
        for node_type in data.node_types:
            store = data[node_type]
            num_nodes = num_nodes_dict[node_type]
            orig_num_nodes = store.num_nodes

            # Create mask to indicate original vs padded nodes
            store.mask = torch.zeros(num_nodes, dtype=torch.bool)
            store.mask[:orig_num_nodes] = 1

            # Safely pad node features if they exist
            if hasattr(store, "x") and store.x is not None:
                size = [num_nodes - store.x.size(0)] + list(store.x.size())[1:]
                store.x = torch.cat([store.x, store.x.new_zeros(size)], dim=0)

            # Safely pad node positions if they exist
            if hasattr(store, "pos") and store.pos is not None:
                size = [num_nodes - store.pos.size(0)] + list(store.pos.size())[1:]
                store.pos = torch.cat([store.pos, store.pos.new_zeros(size)], dim=0)

            # Safely pad all tensor attributes with proper dimensions
            for attr in dir(store):
                # Skip special attributes, non-tensor attributes, and already processed attributes
                if attr.startswith("_") or attr in [
                    "x",
                    "pos",
                    "mask",
                    "num_nodes",
                    "node_ids",
                ]:
                    continue

                value = getattr(store, attr)
                if isinstance(value, Tensor) and value.size(0) == orig_num_nodes:
                    size = [num_nodes - value.size(0)] + list(value.size())[1:]
                    padded_value = torch.cat([value, value.new_zeros(size)], dim=0)
                    setattr(store, attr, padded_value)

        return data

    def __repr__(self) -> str:
        if not self.num_nodes_dict:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}(num_nodes_dict={self.num_nodes_dict})"