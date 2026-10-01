"""Transform converting sparse hetero adjacencies to dense boolean masks."""

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform("hetero_to_dense_mask")
class HeteroToDenseMask(BaseTransform):  # type: ignore[misc]  # BaseTransform is Any (torch_geometric untyped)
    r"""Convert sparse hetero adjacencies to boolean dense masks.

    Edge attributes are preserved in their original sparse format for memory
    efficiency while adjacency/incidence matrices become boolean dense masks.

    A mask entry is set only when both endpoints are real (original) nodes, so an edge
    into a padding row or past the input's node count is dropped from the mask (it
    stays in ``edge_index``). A node tensor whose first axis equals the original node
    count ``N`` is padded with zero rows. Only the attributes named in ``square_attrs``
    are also padded on their second axis, so a listed per-node ``[N, N]`` matrix
    becomes ``[N_pad, N_pad]``; an unlisted ``[N, d]`` tensor keeps its ``d`` columns
    even when ``d == N`` (the shape cannot tell a feature axis from a node axis).

    Args:
        num_nodes_dict (Dict[str, int], optional): Dictionary mapping node types to
            their desired number of nodes. If not provided for a node type, will
            use the maximum number of nodes found for that type. (default: None)
        square_attrs (Dict[str, List[str]], optional): Per node type, the names of
            the ``[N, N, ...]`` node attributes to pad on both axes. A listed
            attribute that is absent or not ``[N, N, ...]`` raises ``ValueError``.
            (default: None, no square attributes)
    """

    def __init__(
        self,
        num_nodes_dict: dict[str, int] | None = None,
        square_attrs: dict[str, list[str]] | None = None,
    ) -> None:
        """Store the per-node-type target node counts and square attribute names.

        Args:
            num_nodes_dict: Optional mapping of node type to target node count.
            square_attrs: Optional mapping of node type to the per-node square
                attributes padded on both axes.
        """
        self.num_nodes_dict = num_nodes_dict or {}
        self.square_attrs = square_attrs or {}

    def forward(self, data: HeteroData) -> HeteroData:
        """Add dense masks, pad node attributes, and return the transformed data."""
        # First determine number of nodes for each node type
        num_nodes_dict = {}
        orig_num_nodes_dict = {
            node_type: data[node_type].num_nodes for node_type in data.node_types
        }
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
                    device=edge_index.device,
                )

                # Fill the mask with edges between real nodes; padding rows and
                # columns stay False
                valid_edges = (edge_index[0] < orig_num_nodes_dict[src]) & (
                    edge_index[1] < orig_num_nodes_dict[dst]
                )
                valid_edge_index = edge_index[:, valid_edges]

                if valid_edge_index.size(1) > 0:
                    adj_mask[valid_edge_index[0], valid_edge_index[1]] = True

                # Store the boolean adjacency mask
                store.adj_mask = adj_mask

                # KEEP the original edge_index and edge_attr for attribute reference

            # Handle hyperedge_index (for hypergraph/bipartite relations)
            elif (
                hasattr(store, "hyperedge_index") and store.hyperedge_index is not None
            ):
                src_num_nodes = num_nodes_dict[src]
                dst_num_nodes = num_nodes_dict[dst]
                hyperedge_index = store.hyperedge_index

                # Create a boolean bipartite incidence matrix
                inc_mask = torch.zeros(
                    (src_num_nodes, dst_num_nodes),
                    dtype=torch.bool,
                    device=hyperedge_index.device,
                )

                # Fill in the incidence matrix from the hyperedge_index, real
                # nodes only
                valid_edges = (hyperedge_index[0] < orig_num_nodes_dict[src]) & (
                    hyperedge_index[1] < orig_num_nodes_dict[dst]
                )
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

            square = self.square_attrs.get(node_type, [])
            for attr in square:
                value = getattr(store, attr, None)
                if not (
                    isinstance(value, Tensor)
                    and value.dim() >= 2
                    and value.size(0) == orig_num_nodes
                    and value.size(1) == orig_num_nodes
                ):
                    shape = (
                        list(value.size()) if isinstance(value, Tensor) else type(value)
                    )
                    raise ValueError(
                        f"square attribute {node_type}.{attr} must be a tensor of "
                        f"shape [{orig_num_nodes}, {orig_num_nodes}, ...], got {shape}"
                    )

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

            # Safely pad all tensor attributes with proper dimensions. ``store.keys()``
            # lists the stored attributes; ``dir(store)`` did not, so this loop padded
            # nothing until tests/torchcell/transforms/test_hetero_to_dense_mask.py
            # checked a second node tensor (2026.09.26).
            for attr in list(store.keys()):
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
                    # a listed per-node matrix ([N, N, ...]) is padded on its columns too
                    if attr in square:
                        size = [num_nodes, num_nodes - value.size(1)] + list(
                            value.size()
                        )[2:]
                        padded_value = torch.cat(
                            [padded_value, padded_value.new_zeros(size)], dim=1
                        )
                    setattr(store, attr, padded_value)

        return data

    def __repr__(self) -> str:
        """Return a string representation including the non-default arguments."""
        args = []
        if self.num_nodes_dict:
            args.append(f"num_nodes_dict={self.num_nodes_dict}")
        if self.square_attrs:
            args.append(f"square_attrs={self.square_attrs}")
        return f"{self.__class__.__name__}({', '.join(args)})"
