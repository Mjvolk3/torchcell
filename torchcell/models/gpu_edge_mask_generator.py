# torchcell/models/gpu_edge_mask_generator
# [[torchcell.models.gpu_edge_mask_generator]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/gpu_edge_mask_generator

"""
GPU-based edge mask generator for zero-copy lazy subgraph representation.

Keeps incidence cache on GPU to generate edge masks from perturbation indices
without CPU→GPU transfer of large mask tensors.

Key optimization:
- CPU approach: Transfer 65 MB masks per batch
- GPU approach: Transfer 0.0004 MB indices per batch, generate masks on GPU
- Expected speedup: 20-30x in data loading pipeline
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from torch_geometric.data import HeteroData
import logging

log = logging.getLogger(__name__)


class GPUEdgeMaskGenerator(nn.Module):
    """
    Generates edge masks on GPU from perturbation indices.

    Stores incidence cache and base masks on GPU to avoid CPU→GPU transfer
    of large mask tensors during training.

    Architecture:
        1. One-time GPU setup (during __init__):
           - Build incidence cache on GPU (~19 MB)
           - Create base edge masks (all True)

        2. Per-batch (in training_step):
           - Transfer perturbation indices (~448 bytes)
           - Generate masks on GPU using incidence cache
           - Use masks in message passing

    Expected Performance:
        - Data transfer: 65 MB → 0.0004 MB per batch (162,500x reduction)
        - Training speed: ~0.3 it/s → 8-10 it/s (25-33x speedup)
    """

    def __init__(self, cell_graph: HeteroData, device: torch.device):
        """
        Initialize GPU mask generator.

        Args:
            cell_graph: Full cell graph with all edge types
            device: GPU device to use
        """
        super().__init__()
        self.device = device

        log.info(f"Initializing GPUEdgeMaskGenerator on {device}")

        # Build incidence cache on GPU
        self._build_incidence_cache_gpu(cell_graph)

        # Build vectorized incidence tensors for batch operations
        self._build_vectorized_incidence_tensors()

        # Register base masks as buffers (included in model.to(device))
        self._create_base_masks(cell_graph)

        log.info(f"GPUEdgeMaskGenerator initialized: {len(self.incidence_cache)} edge types")

    def _build_incidence_cache_gpu(self, cell_graph: HeteroData) -> None:
        """
        Build node-to-edge incidence mappings on GPU.

        For each gene node, creates tensor of edge positions where that gene
        appears as either source or destination.

        Memory: ~19 MB for typical yeast graph (6607 nodes, 9 edge types)
        """
        num_genes = cell_graph["gene"].num_nodes
        cache = {}

        total_elements = 0

        for edge_type in cell_graph.edge_types:
            if edge_type[0] == "gene" and edge_type[2] == "gene":
                edge_index = cell_graph[edge_type].edge_index
                num_edges = edge_index.size(1)

                # Initialize empty lists for each gene
                node_to_edges = [[] for _ in range(num_genes)]

                # Build incidence mapping: node -> [edge_positions]
                for edge_pos in range(num_edges):
                    src = edge_index[0, edge_pos].item()
                    dst = edge_index[1, edge_pos].item()
                    node_to_edges[src].append(edge_pos)
                    if src != dst:  # Avoid duplicate for self-loops
                        node_to_edges[dst].append(edge_pos)

                # Convert lists to tensors and move to GPU
                gpu_tensors = []
                for edges in node_to_edges:
                    if edges:
                        tensor = torch.tensor(edges, dtype=torch.long, device=self.device)
                        gpu_tensors.append(tensor)
                        total_elements += len(edges)
                    else:
                        gpu_tensors.append(torch.tensor([], dtype=torch.long, device=self.device))

                cache[edge_type] = gpu_tensors

        self.incidence_cache = cache

        # Estimate GPU memory usage
        memory_mb = (total_elements * 8) / (1024 ** 2)  # 8 bytes per int64
        log.info(f"Incidence cache GPU memory: {memory_mb:.2f} MB ({total_elements} elements)")

    def _build_vectorized_incidence_tensors(self) -> None:
        """
        Build padded incidence tensors for vectorized batch operations.

        Converts the list-based incidence cache to padded tensors that allow
        for vectorized lookups without Python loops or .item() calls.
        """
        self.incidence_tensors = {}
        self.incidence_masks = {}  # Track valid vs padded positions

        num_genes = len(list(self.incidence_cache.values())[0]) if self.incidence_cache else 0

        for edge_type in self.incidence_cache.keys():
            node_to_edges = self.incidence_cache[edge_type]

            # Find max edges per node for padding
            max_edges = max(len(edges) for edges in node_to_edges) if node_to_edges else 0

            # Create padded tensor
            incidence_tensor = torch.full(
                (num_genes, max_edges),
                -1,  # Padding value
                dtype=torch.long,
                device=self.device
            )

            # Create mask for valid positions
            incidence_mask = torch.zeros(
                (num_genes, max_edges),
                dtype=torch.bool,
                device=self.device
            )

            # Fill tensor with actual edge indices
            for node_idx, edges in enumerate(node_to_edges):
                num_edges = len(edges)
                if num_edges > 0:
                    incidence_tensor[node_idx, :num_edges] = edges
                    incidence_mask[node_idx, :num_edges] = True

            self.incidence_tensors[edge_type] = incidence_tensor
            self.incidence_masks[edge_type] = incidence_mask

            log.debug(f"Edge type {edge_type}: tensor shape {incidence_tensor.shape}, "
                     f"density {incidence_mask.float().mean().item():.3f}")

    def _create_base_masks(self, cell_graph: HeteroData) -> None:
        """
        Create base edge masks (all True) as model buffers.

        These are cloned and modified per-batch to create sample-specific masks.
        """
        for edge_type in self.incidence_cache.keys():
            num_edges = cell_graph[edge_type].edge_index.size(1)
            base_mask = torch.ones(num_edges, dtype=torch.bool, device=self.device)

            # Register as buffer (automatically moved with model.to(device))
            buffer_name = f'base_mask_{edge_type[0]}__{edge_type[1]}__{edge_type[2]}'
            self.register_buffer(buffer_name, base_mask)

    def generate_batch_masks(
        self,
        batch_perturbation_indices: List[torch.Tensor],
        batch_size: int,
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Generate edge masks for a batch of perturbations.

        For batched graphs in PyG, we need to concatenate masks across samples
        with proper offsets matching the concatenated edge_index.

        Args:
            batch_perturbation_indices: List of perturbation index tensors,
                one per sample in batch. Each tensor contains indices of perturbed genes.
            batch_size: Number of samples in batch

        Returns:
            edge_mask_dict: {edge_type: concatenated_mask} where mask length
                matches the batched edge_index length

        Example:
            >>> # Sample 1: perturb genes [42, 100]
            >>> # Sample 2: perturb genes [7]
            >>> pert_indices = [torch.tensor([42, 100]), torch.tensor([7])]
            >>> masks = generator.generate_batch_masks(pert_indices, batch_size=2)
            >>> # masks[edge_type] has length = 2 * num_edges (concatenated)
        """
        batch_masks = {}

        # Generate mask for each sample, then concatenate
        for edge_type in self.incidence_cache.keys():
            # Get base mask
            buffer_name = f'base_mask_{edge_type[0]}__{edge_type[1]}__{edge_type[2]}'
            base_mask = getattr(self, buffer_name)

            # Generate mask for each sample in batch
            sample_masks = []

            for sample_pert_indices in batch_perturbation_indices:
                # Clone base mask (all True)
                sample_mask = base_mask.clone()

                # Ensure perturbation indices are on GPU
                if sample_pert_indices.device != self.device:
                    sample_pert_indices = sample_pert_indices.to(self.device)

                # For each perturbed gene, mask its affected edges
                node_to_edges = self.incidence_cache[edge_type]
                max_node_idx = len(node_to_edges) - 1

                for pert_idx in sample_pert_indices:
                    idx = pert_idx.item()

                    # Bounds check
                    if idx < 0 or idx > max_node_idx:
                        raise IndexError(
                            f"Perturbation index {idx} out of bounds [0, {max_node_idx}]. "
                            f"Edge type: {edge_type}, "
                            f"Sample perturbation indices: {sample_pert_indices.tolist()}"
                        )

                    affected_edges = node_to_edges[idx]
                    if len(affected_edges) > 0:
                        # Set affected edges to False (masked out)
                        sample_mask[affected_edges] = False

                sample_masks.append(sample_mask)

            # Concatenate masks across batch
            # This matches PyG's batching where edge_index is concatenated
            batch_masks[edge_type] = torch.cat(sample_masks, dim=0)

        return batch_masks

    def generate_batch_masks_vectorized(
        self,
        batch_perturbation_indices: List[torch.Tensor],
        batch_size: int,
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        VECTORIZED: Generate edge masks with zero Python loops and zero .item() calls.

        This is the optimized version that eliminates GPU→CPU synchronizations.
        Expected speedup: 10-20x compared to the loop-based version.

        Args:
            batch_perturbation_indices: List of perturbation index tensors,
                one per sample in batch
            batch_size: Number of samples in batch

        Returns:
            edge_mask_dict: {edge_type: concatenated_mask}
        """
        batch_masks = {}

        # Early return for empty batch
        if batch_size == 0:
            return batch_masks

        # Flatten all perturbation indices and create batch assignments
        # This allows us to process all perturbations in parallel
        pert_lengths = torch.tensor([len(p) for p in batch_perturbation_indices],
                                   dtype=torch.long, device=self.device)

        # Skip if no perturbations
        if pert_lengths.sum() == 0:
            # Return all-True masks
            for edge_type in self.incidence_tensors.keys():
                buffer_name = f'base_mask_{edge_type[0]}__{edge_type[1]}__{edge_type[2]}'
                base_mask = getattr(self, buffer_name)
                batch_masks[edge_type] = base_mask.repeat(batch_size)
            return batch_masks

        # Concatenate all perturbation indices
        # DEFENSIVE: Force contiguous GPU tensors to ensure proper device backing
        all_pert_indices = torch.cat([
            p.to(self.device, non_blocking=False).contiguous()
            for p in batch_perturbation_indices if len(p) > 0
        ])

        # EXTRA DEFENSIVE: Ensure concatenated result is on the right device
        # torch.cat can sometimes return CPU tensor if inputs are mixed
        if all_pert_indices.device != self.device:
            all_pert_indices = all_pert_indices.to(self.device, non_blocking=False).contiguous()

        # Create batch assignment for each perturbation
        batch_assignment = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device),
            pert_lengths
        )

        # Process each edge type
        for edge_type in self.incidence_tensors.keys():
            # Get base mask and incidence data
            buffer_name = f'base_mask_{edge_type[0]}__{edge_type[1]}__{edge_type[2]}'
            base_mask = getattr(self, buffer_name)
            num_edges = len(base_mask)

            # Get precomputed incidence tensor and validity mask
            incidence_tensor = self.incidence_tensors[edge_type]
            incidence_mask = self.incidence_masks[edge_type]

            # CRITICAL FIX: Ensure indices are on the same device as incidence_tensor
            # In DDP mode, tensors can be on different GPUs (cuda:0, cuda:1, etc.)
            if all_pert_indices.device != incidence_tensor.device:
                log.debug(f"Device mismatch detected: all_pert_indices on {all_pert_indices.device}, "
                         f"incidence_tensor on {incidence_tensor.device}. Fixing...")
                all_pert_indices = all_pert_indices.to(incidence_tensor.device, non_blocking=False)

            # Also ensure batch_assignment is on the same device
            if batch_assignment.device != incidence_tensor.device:
                log.debug(f"Device mismatch detected: batch_assignment on {batch_assignment.device}, "
                         f"incidence_tensor on {incidence_tensor.device}. Fixing...")
                batch_assignment = batch_assignment.to(incidence_tensor.device, non_blocking=False)

            # Bounds check (vectorized)
            max_node_idx = incidence_tensor.shape[0] - 1
            if (all_pert_indices < 0).any() or (all_pert_indices > max_node_idx).any():
                invalid_idx = all_pert_indices[(all_pert_indices < 0) | (all_pert_indices > max_node_idx)]
                raise IndexError(
                    f"Perturbation indices {invalid_idx.tolist()} out of bounds [0, {max_node_idx}]. "
                    f"Edge type: {edge_type}"
                )

            # Create batch-replicated mask (all True initially)
            batch_mask = base_mask.repeat(batch_size)

            # Vectorized lookup: get all affected edges for all perturbations
            # Shape: [num_perturbations, max_edges_per_node]
            affected_edges = incidence_tensor[all_pert_indices]
            affected_valid = incidence_mask[all_pert_indices]

            # Compute global edge indices with batch offsets
            # Each sample's edges are offset by sample_idx * num_edges
            batch_offsets = batch_assignment * num_edges

            # Broadcasting: add batch offset to each edge index
            # Shape: [num_perturbations, max_edges_per_node]
            affected_edges_global = affected_edges + batch_offsets.unsqueeze(1)

            # Flatten and filter out invalid (padded) positions
            affected_edges_flat = affected_edges_global[affected_valid]

            # Set affected edges to False (vectorized scatter)
            if affected_edges_flat.numel() > 0:
                batch_mask[affected_edges_flat] = False

            batch_masks[edge_type] = batch_mask

        return batch_masks

    def generate_single_mask(
        self,
        perturbation_indices: torch.Tensor
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Generate edge masks for a single sample.

        Useful for validation or single-sample inference.

        Args:
            perturbation_indices: Tensor of perturbed gene indices

        Returns:
            edge_mask_dict: {edge_type: mask_tensor} for one sample
        """
        edge_mask_dict = {}

        # Ensure indices on GPU
        if perturbation_indices.device != self.device:
            perturbation_indices = perturbation_indices.to(self.device)

        for edge_type in self.incidence_cache.keys():
            # Get base mask (all True)
            buffer_name = f'base_mask_{edge_type[0]}__{edge_type[1]}__{edge_type[2]}'
            base_mask = getattr(self, buffer_name)

            # Clone for this sample
            edge_mask = base_mask.clone()

            # Mask edges affected by perturbations
            node_to_edges = self.incidence_cache[edge_type]
            for pert_idx in perturbation_indices:
                affected_edges = node_to_edges[pert_idx.item()]
                if len(affected_edges) > 0:
                    edge_mask[affected_edges] = False

            edge_mask_dict[edge_type] = edge_mask

        return edge_mask_dict

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get estimated GPU memory usage.

        Returns:
            dict with memory usage in MB:
                - incidence_cache: Memory used by incidence cache
                - base_masks: Memory used by base masks
                - total: Total memory usage
        """
        # Count incidence cache elements
        cache_elements = sum(
            sum(len(tensor) for tensor in node_to_edges)
            for node_to_edges in self.incidence_cache.values()
        )
        cache_mb = (cache_elements * 8) / (1024 ** 2)  # int64 = 8 bytes

        # Count base mask elements
        mask_elements = sum(
            getattr(self, f'base_mask_{et[0]}__{et[1]}__{et[2]}').numel()
            for et in self.incidence_cache.keys()
        )
        mask_mb = (mask_elements * 1) / (1024 ** 2)  # bool = 1 byte

        return {
            "incidence_cache_mb": cache_mb,
            "base_masks_mb": mask_mb,
            "total_mb": cache_mb + mask_mb,
        }
