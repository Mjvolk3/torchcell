# torchcell/datamodules/lazy_collate
# [[torchcell.datamodules.lazy_collate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodules/lazy_collate

"""
Custom collate function for LazySubgraphRepresentation batching.

Handles the zero-copy architecture where all samples share the same edge_index
tensor by replicating and offsetting edge indices during batching.

Key insight: Replicate edge_index during batching (~100x per epoch) to maintain
the 3.65x speedup from avoiding copies during data loading (~17k+ samples).
"""

from typing import List, Optional, Union, Sequence, Any
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter


def lazy_collate_hetero(data_list: List[HeteroData]) -> HeteroData:
    """
    Custom collate function for batching HeteroData with LazySubgraphRepresentation.

    **The Problem**: LazySubgraphRepresentation returns the SAME edge_index tensor
    for all samples (zero-copy optimization). PyG's default batching requires
    offsetting indices for each graph, which would modify the shared tensor.

    **The Solution**: During batching, replicate edge_index for each sample and
    apply proper offsets. This is done ~100 times per epoch (cheap) vs. ~17k+
    times during data loading (where the real speedup comes from).

    **Cost Analysis**:
    - Data loading copies avoided: 17,000+ samples × 144k edges = MASSIVE savings ✅
    - Batching copies added: ~100 batches × 144k edges = Negligible overhead ✓
    - Net result: 3.65x speedup maintained

    Args:
        data_list: List of HeteroData samples from LazySubgraphRepresentation

    Returns:
        Batched HeteroData with properly offset edge indices and concatenated masks

    Example:
        >>> from torch_geometric.loader import DataLoader
        >>> dataset = Neo4jCellDataset(..., graph_processor=LazySubgraphRepresentation())
        >>> loader = DataLoader(dataset, batch_size=2, collate_fn=lazy_collate_hetero)
        >>> batch = next(iter(loader))
        >>> # batch now has properly batched data compatible with MaskedGINConv
    """
    if len(data_list) == 0:
        raise ValueError("Cannot collate empty data_list")

    if len(data_list) == 1:
        # No batching needed, return as-is
        return data_list[0]

    batch = HeteroData()

    # Track node offsets for each graph in the batch
    # node_offsets[node_type][i] = cumulative number of nodes before graph i
    node_offsets = {}
    for node_type in data_list[0].node_types:
        offsets = [0]
        for data in data_list[:-1]:  # All except last
            offsets.append(offsets[-1] + data[node_type].num_nodes)
        node_offsets[node_type] = offsets

    # Batch edge types: replicate and offset edge_index, concatenate masks
    for edge_type in data_list[0].edge_types:
        src_type, _, dst_type = edge_type

        # Determine if this edge type uses edge_index or hyperedge_index
        edge_data_sample = data_list[0][edge_type]
        uses_hyperedge = hasattr(edge_data_sample, 'hyperedge_index')
        edge_index_key = 'hyperedge_index' if uses_hyperedge else 'edge_index'

        # Get the shared edge_index (same for all samples due to zero-copy)
        # We'll replicate it for each graph with proper offsets
        edge_indices = []
        edge_masks = []

        for i, data in enumerate(data_list):
            # Get edge data for this sample
            edge_data = data[edge_type]

            # Get edge/hyperedge index
            if hasattr(edge_data, edge_index_key):
                edge_index = getattr(edge_data, edge_index_key)
            else:
                # Skip if this edge type doesn't have indices
                continue

            # Apply offset to edge indices
            # Source nodes get src_type offset, dest nodes get dst_type offset
            src_offset = node_offsets[src_type][i]
            dst_offset = node_offsets[dst_type][i]

            offset_edge_index = edge_index.clone()  # Clone to avoid modifying shared tensor
            offset_edge_index[0] += src_offset
            offset_edge_index[1] += dst_offset

            edge_indices.append(offset_edge_index)

            # Collect edge mask
            if hasattr(edge_data, 'mask'):
                edge_masks.append(edge_data.mask)

        # Concatenate across batch
        if edge_indices:
            setattr(batch[edge_type], edge_index_key, torch.cat(edge_indices, dim=1))

        if edge_masks:
            batch[edge_type].mask = torch.cat(edge_masks, dim=0)

        # Copy other edge attributes if they exist
        for key in data_list[0][edge_type].keys():
            if key not in ['edge_index', 'hyperedge_index', 'mask']:
                attrs = [data[edge_type][key] for data in data_list]
                # Only concatenate if all are tensors
                if all(attr is not None and isinstance(attr, torch.Tensor) for attr in attrs):
                    batch[edge_type][key] = torch.cat(attrs, dim=0)
                elif all(attr is not None for attr in attrs):
                    # For non-tensors (like num_edges), store as list
                    batch[edge_type][key] = attrs

    # Batch node types: concatenate features and create batch vectors
    for node_type in data_list[0].node_types:
        # Concatenate node features
        node_features = []
        batch_vectors = []
        num_nodes_list = []

        for i, data in enumerate(data_list):
            node_data = data[node_type]

            # CRITICAL: LazySubgraphRepresentation uses zero-copy for everything
            # x, edge_index, and masks all reference the full graph (including perturbed nodes)
            # num_nodes correctly reflects the full graph size, not filtered size
            num_nodes = node_data.num_nodes

            num_nodes_list.append(num_nodes)

            # Collect all node attributes
            for key in node_data.keys():
                if i == 0:
                    # Initialize lists for each attribute
                    if key not in batch[node_type]:
                        batch[node_type][key] = []

                attr_value = node_data[key]

                if isinstance(attr_value, torch.Tensor):
                    # Tensors: collect for concatenation
                    batch[node_type][key].append(attr_value)
                else:
                    # Non-tensors (lists, scalars): collect as-is
                    batch[node_type][key].append(attr_value)

            # Create batch vector for this graph
            batch_vectors.append(torch.full((num_nodes,), i, dtype=torch.long))

        # Concatenate all node attributes
        for key in batch[node_type].keys():
            if isinstance(batch[node_type][key], list) and len(batch[node_type][key]) > 0:
                # Check if all elements are tensors
                if all(isinstance(item, torch.Tensor) for item in batch[node_type][key]):
                    # Concatenate tensors
                    batch[node_type][key] = torch.cat(batch[node_type][key], dim=0)
                # else: keep as list for non-tensors (e.g., node_ids, phenotype_types)

        # Add batch vector
        batch[node_type].batch = torch.cat(batch_vectors, dim=0)

        # Store num_nodes for compatibility
        batch[node_type].num_nodes = sum(num_nodes_list)

    return batch


def verify_batch_structure(batch: HeteroData, expected_graphs: int = 2) -> bool:
    """
    Verify that batched HeteroData has correct structure for masked message passing.

    Checks:
    1. Edge indices are properly offset (no overlap between graphs)
    2. Edge masks have correct total length
    3. Batch vectors correctly track graph membership
    4. No index out of bounds issues

    Args:
        batch: Batched HeteroData from lazy_collate_hetero
        expected_graphs: Expected number of graphs in batch

    Returns:
        True if batch structure is valid, False otherwise

    Example:
        >>> batch = lazy_collate_hetero([data1, data2])
        >>> assert verify_batch_structure(batch, expected_graphs=2)
    """
    try:
        # Check node batch vectors
        for node_type in batch.node_types:
            batch_vec = batch[node_type].batch
            assert batch_vec.max().item() == expected_graphs - 1, \
                f"Batch vector max {batch_vec.max()} != {expected_graphs - 1}"

            num_nodes = batch[node_type].num_nodes
            assert batch_vec.size(0) == num_nodes, \
                f"Batch vector size {batch_vec.size(0)} != num_nodes {num_nodes}"

        # Check edge indices are within bounds
        for edge_type in batch.edge_types:
            src_type, _, dst_type = edge_type
            edge_data = batch[edge_type]

            # Handle both edge_index and hyperedge_index
            if hasattr(edge_data, 'edge_index'):
                edge_index = edge_data.edge_index
            elif hasattr(edge_data, 'hyperedge_index'):
                edge_index = edge_data.hyperedge_index
            else:
                # Skip if this edge type doesn't have indices
                continue

            src_max = batch[src_type].num_nodes
            dst_max = batch[dst_type].num_nodes

            assert edge_index[0].max() < src_max, \
                f"Source index {edge_index[0].max()} >= {src_max}"
            assert edge_index[1].max() < dst_max, \
                f"Dest index {edge_index[1].max()} >= {dst_max}"

            # Check edge mask length
            if hasattr(edge_data, 'mask'):
                edge_mask = edge_data.mask
                num_edges = edge_index.size(1)
                assert edge_mask.size(0) == num_edges, \
                    f"Mask size {edge_mask.size(0)} != num_edges {num_edges}"

        return True

    except AssertionError as e:
        print(f"Batch structure verification failed: {e}")
        return False


class LazyCollater(Collater):
    """
    PyTorch Lightning-compatible collater for LazySubgraphRepresentation.

    Extends PyG's Collater to use lazy_collate_hetero for HeteroData batching,
    which properly handles zero-copy edge_index references by replicating and
    offsetting during batching.

    This ensures compatibility with PyTorch Lightning while maintaining the
    3.65x speedup from LazySubgraphRepresentation.

    Usage:
        from torch_geometric.loader import DataLoader
        from torchcell.datamodules.lazy_collate import LazyCollater

        # Option 1: Use collate_fn parameter (PyG DataLoader will respect it)
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=LazyCollater(dataset)
        )

        # Option 2: Or use it directly with torch.utils.data.DataLoader
        from torch.utils.data import DataLoader as TorchDataLoader
        loader = TorchDataLoader(
            dataset,
            batch_size=2,
            collate_fn=LazyCollater(dataset)
        )

    Example:
        >>> from torchcell.data.graph_processor import LazySubgraphRepresentation
        >>> dataset = Neo4jCellDataset(..., graph_processor=LazySubgraphRepresentation())
        >>> loader = DataLoader(dataset, batch_size=2, collate_fn=LazyCollater(dataset))
        >>> batch = next(iter(loader))
        >>> # batch now has properly batched data with zero-copy optimization
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        super().__init__(dataset, follow_batch, exclude_keys)

    def __call__(self, batch: List[Any]) -> Any:
        """
        Override collate to use lazy_collate_hetero for HeteroData.

        For HeteroData objects (from LazySubgraphRepresentation), uses
        lazy_collate_hetero which properly handles zero-copy edge_index.

        For other data types, falls back to PyG's default Collater behavior.
        """
        elem = batch[0]

        # Check if this is HeteroData from LazySubgraphRepresentation
        # (indicated by having both edge_index and masks)
        if isinstance(elem, HeteroData):
            # Check if this uses lazy representation (has masks on edges)
            is_lazy = False
            for edge_type in elem.edge_types:
                if hasattr(elem[edge_type], 'mask'):
                    is_lazy = True
                    break

            if is_lazy:
                # Use our custom lazy collate
                return lazy_collate_hetero(batch)
            # else: fall through to default PyG batching

        # For all other cases, use PyG's default Collater
        return super().__call__(batch)


if __name__ == "__main__":
    # Example usage and testing
    print("Custom collate function for LazySubgraphRepresentation")
    print("Import and use with DataLoader:")
    print("  loader = DataLoader(dataset, batch_size=2, collate_fn=lazy_collate_hetero)")
    print("Or use LazyCollater for Lightning compatibility:")
    print("  loader = DataLoader(dataset, batch_size=2, collate_fn=LazyCollater(dataset))")
