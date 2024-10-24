from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData, Data
from typing import Dict, Optional
import torch
from torch import Tensor


@functional_transform("hetero_to_dense")
class HeteroToDense(BaseTransform):
    r"""Converts sparse adjacency matrices in a heterogeneous graph to dense format
    with shape :obj:`[num_nodes, num_nodes, *]` for each edge type.

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

            # Check if edge_index exists before accessing
            if not hasattr(store, "edge_index"):
                continue

            if store.edge_index is None:
                continue

            src_num_nodes = num_nodes_dict[src]
            dst_num_nodes = num_nodes_dict[dst]

            # Handle edge attributes - safely check if edge_attr exists
            if hasattr(store, "edge_attr") and store.edge_attr is not None:
                edge_attr = store.edge_attr
            else:
                edge_attr = torch.ones(store.edge_index.size(1), dtype=torch.float)

            # Create dense adjacency matrix
            size = torch.Size(
                [src_num_nodes, dst_num_nodes] + list(edge_attr.size())[1:]
            )
            adj = torch.sparse_coo_tensor(store.edge_index, edge_attr, size)
            store.adj = adj.to_dense()

            # Safely delete attributes if they exist
            if hasattr(store, "edge_index"):
                delattr(store, "edge_index")
            if hasattr(store, "edge_attr"):
                delattr(store, "edge_attr")

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

            # Safely pad node labels if they exist
            if (
                hasattr(store, "y")
                and store.y is not None
                and isinstance(store.y, Tensor)
                and store.y.size(0) == orig_num_nodes
            ):
                size = [num_nodes - store.y.size(0)] + list(store.y.size())[1:]
                store.y = torch.cat([store.y, store.y.new_zeros(size)], dim=0)

        return data

    def __repr__(self) -> str:
        if not self.num_nodes_dict:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}(num_nodes_dict={self.num_nodes_dict})"
