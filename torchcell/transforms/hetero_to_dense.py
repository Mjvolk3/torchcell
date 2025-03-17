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

            # Handle regular edge_index (for node-to-node relations)
            if hasattr(store, "edge_index") and store.edge_index is not None:
                src_num_nodes = num_nodes_dict[src]
                dst_num_nodes = num_nodes_dict[dst]

                # Handle edge attributes
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

                # Safely delete attributes
                delattr(store, "edge_index")
                if hasattr(store, "edge_attr"):
                    delattr(store, "edge_attr")

            # Handle hyperedge_index (for hypergraph/bipartite relations)
            elif (
                hasattr(store, "hyperedge_index") and store.hyperedge_index is not None
            ):
                src_num_nodes = num_nodes_dict[src]
                dst_num_nodes = num_nodes_dict[dst]
                hyperedge_index = store.hyperedge_index

                # For bipartite graphs, we want to create a proper incidence matrix
                # of size [src_num_nodes, dst_num_nodes] instead of [src_num_nodes, num_edges]

                # Create a bipartite incidence matrix (src_nodes x dst_nodes)
                inc_matrix = torch.zeros(
                    (src_num_nodes, dst_num_nodes), dtype=torch.float
                )

                # Fill in the incidence matrix from the hyperedge_index
                for i in range(hyperedge_index.size(1)):
                    src_idx = hyperedge_index[0, i].item()
                    dst_idx = hyperedge_index[1, i].item()

                    # Safety check for indices
                    if src_idx < src_num_nodes and dst_idx < dst_num_nodes:
                        inc_matrix[src_idx, dst_idx] = 1.0

                # Store the proper bipartite incidence matrix
                store.inc_matrix = inc_matrix

                # If we have stoichiometry coefficients, create a weighted incidence matrix
                if hasattr(store, "stoichiometry") and store.stoichiometry is not None:
                    weighted_inc = torch.zeros(
                        (src_num_nodes, dst_num_nodes), dtype=torch.float
                    )

                    for i in range(hyperedge_index.size(1)):
                        src_idx = hyperedge_index[0, i].item()
                        dst_idx = hyperedge_index[1, i].item()

                        # Safety check for indices
                        if src_idx < src_num_nodes and dst_idx < dst_num_nodes:
                            weighted_inc[src_idx, dst_idx] = store.stoichiometry[
                                i
                            ].item()

                    store.weighted_inc_matrix = weighted_inc

                # For RMR edges with edge_type (reactant/product), store that too
                if hasattr(store, "edge_type") and store.edge_type is not None:
                    edge_type_inc = torch.zeros(
                        (src_num_nodes, dst_num_nodes), dtype=torch.long
                    )

                    for i in range(hyperedge_index.size(1)):
                        src_idx = hyperedge_index[0, i].item()
                        dst_idx = hyperedge_index[1, i].item()

                        # Safety check for indices
                        if src_idx < src_num_nodes and dst_idx < dst_num_nodes:
                            edge_type_inc[src_idx, dst_idx] = store.edge_type[i].item()

                    store.edge_type_matrix = edge_type_inc

                # Safely delete attributes
                delattr(store, "hyperedge_index")
                # Keep stoichiometry and edge_type for reference - they may be needed elsewhere

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
