# torchcell/models/dcell_opt
# [[torchcell.models.dcell_opt]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/dcell_opt
# Test file: tests/torchcell/models/test_dcell_opt.py

"""Optimized DCell model for torch.compile compatibility.

Reduces graph breaks by using ModuleList instead of ModuleDict and tensorized operations.
"""

import math
import time
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


class DCellOpt(nn.Module):
    """Optimized DCell with tensorized operations for torch.compile."""

    # Registered buffers (declared so mypy sees them as Tensor, not Tensor | Module)
    has_subsystem: torch.Tensor
    children_indices: torch.Tensor
    num_children: torch.Tensor
    stratum_masks: torch.Tensor
    term_output_dims_tensor: torch.Tensor
    term_row_indices_tensor: torch.Tensor
    term_num_genes_tensor: torch.Tensor

    def __init__(
        self,
        hetero_data: HeteroData,
        min_subsystem_size: int = 20,
        subsystem_ratio: float = 0.3,
        output_size: int = 1,
    ):
        """Build the tensorized subsystem hierarchy from the GO-annotated graph.

        Args:
            hetero_data: Graph carrying the GO ontology and gene annotations.
            min_subsystem_size: Minimum number of genes for a subsystem.
            subsystem_ratio: Fraction setting each subsystem's hidden width.
            output_size: Dimension of the per-subsystem output.
        """
        super().__init__()

        # Store parameters
        self.min_subsystem_size = min_subsystem_size
        self.subsystem_ratio = subsystem_ratio
        self.output_size = output_size
        self.hetero_data = hetero_data

        # Extract GO ontology information
        self.num_genes = hetero_data["gene"].num_nodes
        self.num_go_terms = hetero_data["gene_ontology"].num_nodes
        self.strata = hetero_data["gene_ontology"].strata
        self.stratum_to_terms = hetero_data["gene_ontology"].stratum_to_terms
        self.term_gene_counts = hetero_data["gene_ontology"].term_gene_counts

        # Build hierarchy and gene mappings (still use dicts for initialization)
        self.child_to_parents = self._build_hierarchy(hetero_data)
        self.parent_to_children = self._build_parent_to_children()
        self.term_to_genes = self._build_term_gene_mapping(hetero_data)

        # Pre-compute input/output dimensions
        self.term_input_dims: dict[int, int] = {}
        self.term_output_dims: dict[int, int] = {}

        # OPTIMIZATION: Use ModuleList instead of ModuleDict
        # None placeholders are filled in later via _build_term_module; cast so
        # mypy accepts the placeholder list while runtime behavior is unchanged.
        self.subsystems = nn.ModuleList(
            cast(list[nn.Module], [None] * self.num_go_terms)
        )
        self.linear_heads = nn.ModuleList(
            cast(list[nn.Module], [None] * self.num_go_terms)
        )

        # Track which indices have modules
        self.register_buffer(
            "has_subsystem", torch.zeros(self.num_go_terms, dtype=torch.bool)
        )

        # Get strata order
        self.max_stratum = max(self.stratum_to_terms.keys())
        self.strata_order = sorted(self.stratum_to_terms.keys(), reverse=True)

        # OPTIMIZATION: Build tensorized hierarchy representations
        self._build_tensorized_hierarchy()
        self._build_stratum_masks()

        # Build modules for each GO term
        for term_idx in range(self.num_go_terms):
            self._build_term_module(term_idx)

        # Pre-compute gene state extraction indices
        self._precompute_gene_indices(hetero_data)

        # Pre-compute max output dimension for pre-allocation
        self.max_output_dim = (
            max(self.term_output_dims.values()) if self.term_output_dims else 256
        )

        # Create tensor version of term_output_dims for compile-friendly access
        self.register_buffer(
            "term_output_dims_tensor", torch.zeros(self.num_go_terms, dtype=torch.long)
        )
        for term_idx, output_dim in self.term_output_dims.items():
            self.term_output_dims_tensor[term_idx] = output_dim

        # Analyze stratum sizes for parallel processing
        stratum_sizes = {}
        for stratum, terms in self.stratum_to_terms.items():
            active_terms = sum(1 for t in terms if self.has_subsystem[t])
            stratum_sizes[stratum] = (len(terms), active_terms)

        max_parallel_terms = (
            max(s[1] for s in stratum_sizes.values()) if stratum_sizes else 0
        )

        print("DCellOpt model initialized with PARALLEL processing:")
        print(f"  GO terms: {self.num_go_terms}")
        print(f"  Genes: {self.num_genes}")
        print(f"  Strata: {len(self.strata_order)} (max: {self.max_stratum})")
        print(f"  Active subsystems: {self.has_subsystem.sum().item()}")
        print(f"  Max output dim: {self.max_output_dim}")
        print(f"  Max parallel terms in single stratum: {max_parallel_terms}")
        print(f"  CUDA streams available: {torch.cuda.is_available()}")

        # Initialize profiling mode (disabled by default)
        self._profile_mode = False

    def enable_profiling(self, enable: bool = True) -> None:
        """Enable or disable performance profiling during forward pass."""
        self._profile_mode = enable
        if enable:
            print("DCellOpt profiling enabled - will log stratum processing times")
        else:
            print("DCellOpt profiling disabled")

    def _build_tensorized_hierarchy(self) -> None:
        """Convert parent-child relationships to tensor representations."""
        # Find max children for any term
        max_children = 0
        for children in self.parent_to_children.values():
            max_children = max(max_children, len(children))
        self.max_children = max_children

        # Create padded children tensor
        self.register_buffer(
            "children_indices",
            torch.full((self.num_go_terms, max_children), -1, dtype=torch.long),
        )
        self.register_buffer(
            "num_children", torch.zeros(self.num_go_terms, dtype=torch.long)
        )

        # Fill children indices
        for parent, children in self.parent_to_children.items():
            num_children = len(children)
            if num_children > 0:
                self.children_indices[parent, :num_children] = torch.tensor(
                    children, dtype=torch.long
                )
                self.num_children[parent] = num_children

    def _build_stratum_masks(self) -> None:
        """Build binary masks for each stratum."""
        # Create stratum masks
        self.register_buffer(
            "stratum_masks",
            torch.zeros(self.max_stratum + 1, self.num_go_terms, dtype=torch.bool),
        )

        for stratum, terms in self.stratum_to_terms.items():
            for term in terms:
                self.stratum_masks[stratum, term] = True

        # Pre-compute term lists per stratum for efficient iteration
        self.stratum_term_lists = []
        for stratum in range(self.max_stratum + 1):
            terms = torch.where(self.stratum_masks[stratum])[0]
            self.stratum_term_lists.append(terms)

    @property
    def num_parameters(self) -> dict[str, int]:
        """Count parameters in different parts of the model."""
        subsystem_params = sum(
            p.numel()
            for module in self.subsystems
            if module is not None
            for p in module.parameters()
        )
        linear_head_params = sum(
            p.numel()
            for module in self.linear_heads
            if module is not None
            for p in module.parameters()
        )
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "subsystems": subsystem_params,
            "dcell_linear": linear_head_params,
            "dcell": subsystem_params + linear_head_params,
            "total": total_params,
            "num_go_terms": self.num_go_terms,
            "num_subsystems": self.has_subsystem.sum().item(),
        }

    def _build_hierarchy(self, hetero_data: HeteroData) -> dict[int, list[int]]:
        """Build child -> parents mapping from edge_index."""
        child_to_parents: dict[int, list[int]] = {}

        if ("gene_ontology", "is_child_of", "gene_ontology") in hetero_data.edge_types:
            edge_index = hetero_data[
                "gene_ontology", "is_child_of", "gene_ontology"
            ].edge_index

            for child, parent in edge_index.t():
                child_idx = child.item()
                parent_idx = parent.item()

                if child_idx not in child_to_parents:
                    child_to_parents[child_idx] = []
                child_to_parents[child_idx].append(parent_idx)

        return child_to_parents

    def _build_parent_to_children(self) -> dict[int, list[int]]:
        """Build parent -> children mapping from child_to_parents."""
        parent_to_children: dict[int, list[int]] = {}

        for child, parents in self.child_to_parents.items():
            for parent in parents:
                if parent not in parent_to_children:
                    parent_to_children[parent] = []
                parent_to_children[parent].append(child)

        return parent_to_children

    def _build_term_gene_mapping(self, hetero_data: HeteroData) -> dict[int, list[int]]:
        """Build term -> genes mapping from go_gene_strata_state."""
        term_to_genes: dict[int, list[int]] = {}

        go_gene_state = hetero_data["gene_ontology"].go_gene_strata_state
        # Columns: [go_idx, gene_idx, stratum, state]

        for row in go_gene_state:
            go_idx = int(row[0].item())
            gene_idx = int(row[1].item())

            if go_idx not in term_to_genes:
                term_to_genes[go_idx] = []
            term_to_genes[go_idx].append(gene_idx)

        return term_to_genes

    def _calculate_input_dim(self, term_idx: int) -> int:
        """Calculate input dimension for a GO term."""
        input_dim = 0

        # Add dimensions from child subsystems
        children = self.parent_to_children.get(term_idx, [])
        for child_idx in children:
            input_dim += self._calculate_output_dim(child_idx)

        # Add dimension for gene perturbation states
        go_gene_state = self.hetero_data["gene_ontology"].go_gene_strata_state
        term_mask = go_gene_state[:, 0] == term_idx
        num_genes_for_term = int(term_mask.sum().item())

        gene_dim = max(num_genes_for_term, 1)
        input_dim += gene_dim

        return max(input_dim, 1)

    def _calculate_output_dim(self, term_idx: int) -> int:
        """Calculate output dimension for a GO term based on DCell paper formula."""
        num_genes = int(self.term_gene_counts[term_idx].item())
        return max(self.min_subsystem_size, math.ceil(self.subsystem_ratio * num_genes))

    def _build_term_module(self, term_idx: int) -> None:
        """Build subsystem and linear head for a specific GO term."""
        input_dim = self._calculate_input_dim(term_idx)
        output_dim = self._calculate_output_dim(term_idx)

        self.term_input_dims[term_idx] = input_dim
        self.term_output_dims[term_idx] = output_dim

        # Create subsystem module - use ModuleList indexing
        subsystem = DCellSubsystem(input_dim, output_dim)
        self.subsystems[term_idx] = subsystem

        # Create linear head for auxiliary supervision
        linear_head = nn.Linear(output_dim, self.output_size)
        self.linear_heads[term_idx] = linear_head

        # Mark as having subsystem
        self.has_subsystem[term_idx] = True

    def _precompute_gene_indices(self, hetero_data: HeteroData) -> None:
        """Pre-compute indices for efficient gene state extraction."""
        go_gene_state = hetero_data["gene_ontology"].go_gene_strata_state

        self.rows_per_sample = len(go_gene_state)

        # Still use dictionaries for initialization
        self.term_row_indices = {}
        self.term_num_genes = {}

        print(f"Pre-computing gene indices for {self.num_go_terms} GO terms...")
        start_time = time.time()

        # Find max number of genes for any term for padding
        max_genes_per_term = 0

        for term_idx in range(self.num_go_terms):
            mask = go_gene_state[:, 0] == term_idx
            row_indices = torch.where(mask)[0]

            self.term_row_indices[term_idx] = row_indices.cpu()
            self.term_num_genes[term_idx] = len(row_indices)
            max_genes_per_term = max(max_genes_per_term, len(row_indices))

        # OPTIMIZATION: Create tensorized versions for compile-friendly access
        self.register_buffer(
            "term_row_indices_tensor",
            torch.full((self.num_go_terms, max_genes_per_term), -1, dtype=torch.long),
        )
        self.register_buffer(
            "term_num_genes_tensor", torch.zeros(self.num_go_terms, dtype=torch.long)
        )

        # Fill the tensors
        for term_idx, indices in self.term_row_indices.items():
            num_genes = len(indices)
            if num_genes > 0:
                self.term_row_indices_tensor[term_idx, :num_genes] = indices
            self.term_num_genes_tensor[term_idx] = num_genes

        elapsed = time.time() - start_time
        print(f"Pre-computed indices in {elapsed:.2f} seconds")
        print(f"  Total rows per sample: {self.rows_per_sample}")
        print(f"  Max genes per term: {max_genes_per_term}")
        print(
            f"  Average genes per GO term: {np.mean(list(self.term_num_genes.values())):.1f}"
        )

    def _extract_gene_states_for_term(
        self, term_idx: torch.Tensor, batch: HeteroData
    ) -> torch.Tensor:
        """Extract gene states for a specific GO term using pre-computed indices."""
        batch_size = batch["gene"].batch.max() + 1
        device = batch["gene"].x.device

        # OPTIMIZATION: Use tensor indexing instead of dictionary lookup
        if not isinstance(term_idx, torch.Tensor):
            term_idx = torch.tensor(term_idx, dtype=torch.long, device=device)
        else:
            term_idx = term_idx.to(device)

        # Ensure scalar tensor for indexing
        if term_idx.dim() > 0:
            term_idx = term_idx.squeeze()

        # Use tensor indexing
        num_genes = self.term_num_genes_tensor[term_idx]

        if num_genes == 0:
            return torch.zeros(batch_size, 1, device=device)

        # Get row indices using tensor indexing
        row_indices_padded = self.term_row_indices_tensor[term_idx].to(device)
        row_indices = row_indices_padded[:num_genes]  # Only take valid indices

        go_gene_state = batch["gene_ontology"].go_gene_strata_state

        # Use reshape instead of view to handle dynamic shapes better
        go_gene_state_batched = go_gene_state.reshape(batch_size, -1, 4)

        # Vectorized extraction
        gene_states_batch = []
        for i in range(batch_size):
            sample_data = go_gene_state_batched[i]
            term_rows = sample_data[row_indices]
            gene_states = term_rows[:, 3].float()
            gene_states_batch.append(gene_states)

        return torch.stack(gene_states_batch)

    def _extract_gene_states_parallel(
        self, term_indices: torch.Tensor, batch: HeteroData
    ) -> torch.Tensor:
        """Extract gene states for multiple GO terms in parallel.

        Args:
            term_indices: Tensor of shape (num_terms,) with GO term indices
            batch: HeteroData batch

        Returns:
            Tensor of shape (batch_size, num_terms, max_genes) with gene states
        """
        batch_size = batch["gene"].batch.max() + 1
        device = batch["gene"].x.device
        num_terms = len(term_indices)

        # Ensure term_indices is on the right device
        term_indices = term_indices.to(self.term_num_genes_tensor.device)

        # Get number of genes for each term
        num_genes_per_term = self.term_num_genes_tensor[term_indices]
        max_genes = (
            int(num_genes_per_term.max().item())
            if num_genes_per_term.numel() > 0
            else 1
        )

        # Pre-allocate output tensor
        gene_states_all = torch.zeros(batch_size, num_terms, max_genes, device=device)

        # Extract row indices for all terms at once
        row_indices_all = self.term_row_indices_tensor[term_indices].to(device)

        # Process batch
        go_gene_state = batch["gene_ontology"].go_gene_strata_state
        go_gene_state_batched = go_gene_state.reshape(batch_size, -1, 4)

        # Extract states for each term
        for term_idx, (global_term_idx, num_genes) in enumerate(
            zip(term_indices, num_genes_per_term)
        ):
            if num_genes > 0:
                row_indices = row_indices_all[term_idx, :num_genes]

                # Extract for all samples in batch
                for batch_idx in range(batch_size):
                    sample_data = go_gene_state_batched[batch_idx]
                    term_rows = sample_data[row_indices]
                    gene_states = term_rows[:, 3].float()
                    gene_states_all[batch_idx, term_idx, :num_genes] = gene_states

        return gene_states_all

    def _group_terms_by_dimensions(
        self, term_indices: torch.Tensor
    ) -> list[torch.Tensor]:
        """Group GO terms by their input/output dimensions for efficient batching.

        Args:
            term_indices: Tensor of GO term indices to group

        Returns:
            List of tensors, each containing indices with same dimensions
        """
        dim_groups: dict[tuple[int, int], list[int]] = {}

        for term_idx in term_indices:
            term_idx_int = term_idx.item()

            # Skip if no subsystem
            if not self.has_subsystem[term_idx_int]:
                continue

            # Get dimensions for this term
            input_dim = self.term_input_dims.get(term_idx_int, 0)
            output_dim = self.term_output_dims.get(term_idx_int, 0)

            # Create dimension key
            dim_key = (input_dim, output_dim)

            if dim_key not in dim_groups:
                dim_groups[dim_key] = []
            dim_groups[dim_key].append(term_idx_int)

        # Convert to list of tensors
        grouped_indices = []
        for dim_key, indices in dim_groups.items():
            if indices:  # Only add non-empty groups
                grouped_indices.append(torch.tensor(indices, dtype=torch.long))

        return grouped_indices

    def _process_stratum_parallel(
        self,
        stratum_terms: torch.Tensor,
        batch: HeteroData,
        all_activations: torch.Tensor,
        activation_mask: torch.Tensor,
        linear_outputs_tensor: torch.Tensor,
    ) -> None:
        """Process all terms in a stratum in parallel.

        This method processes multiple GO terms simultaneously while respecting
        their hierarchical dependencies. Terms within the same stratum are independent
        and can be processed in parallel.

        Args:
            stratum_terms: Tensor of GO term indices in this stratum
            batch: HeteroData batch
            all_activations: Pre-allocated tensor for all term activations
            activation_mask: Mask indicating which terms have been processed
            linear_outputs_tensor: Pre-allocated tensor for linear outputs
        """
        if len(stratum_terms) == 0:
            return

        device = all_activations.device

        # Prepare all inputs first (memory coalescing)
        term_inputs = {}
        valid_terms = []

        # First pass: Prepare all inputs
        for term_idx in stratum_terms:
            term_idx_scalar = term_idx.item() if hasattr(term_idx, "item") else term_idx

            # Skip if no subsystem
            if not self.has_subsystem[term_idx_scalar]:
                continue

            # Prepare input for this term
            term_input = self._prepare_term_input_optimized(
                term_idx, batch, all_activations, activation_mask
            )

            if term_input is not None and term_input.numel() > 0:
                term_inputs[term_idx_scalar] = term_input
                valid_terms.append(term_idx_scalar)

        if not valid_terms:
            return

        # Second pass: Process all terms
        # This allows CUDA to better schedule operations
        outputs = {}

        # Process in batches to improve GPU utilization
        # Use torch.cuda.Stream for true parallel execution if available
        if device.type == "cuda" and torch.cuda.is_available():
            # Create streams for parallel processing
            num_streams = min(4, len(valid_terms))  # Use up to 4 CUDA streams
            streams = [
                torch.cuda.Stream()  # type: ignore[no-untyped-call]  # torch.cuda.Stream is untyped
                for _ in range(num_streams)
            ]

            # Distribute terms across streams
            for idx, term_idx in enumerate(valid_terms):
                stream_idx = idx % num_streams

                with torch.cuda.stream(streams[stream_idx]):
                    # Get this term's specific subsystem and linear head
                    subsystem = self.subsystems[term_idx]
                    linear_head = self.linear_heads[term_idx]

                    # Forward through subsystem
                    term_output = subsystem(term_inputs[term_idx])

                    # Store results temporarily
                    outputs[term_idx] = {
                        "activation": term_output,
                        "linear": linear_head(term_output),
                    }

            # Synchronize all streams
            for stream in streams:
                stream.synchronize()
        else:
            # CPU or fallback: Process sequentially but with prepared inputs
            for term_idx in valid_terms:
                subsystem = self.subsystems[term_idx]
                linear_head = self.linear_heads[term_idx]

                term_output = subsystem(term_inputs[term_idx])
                outputs[term_idx] = {
                    "activation": term_output,
                    "linear": linear_head(term_output),
                }

        # Third pass: Store all results (memory coalescing)
        for term_idx in valid_terms:
            output_dim = self.term_output_dims[term_idx]

            # Store activation
            all_activations[:, term_idx, :output_dim] = outputs[term_idx]["activation"]
            activation_mask[:, term_idx] = True

            # Store linear output
            linear_outputs_tensor[:, term_idx, :] = outputs[term_idx]["linear"]

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass through DCell hierarchy with parallel stratum processing."""
        batch_size = batch["gene"].batch.max() + 1
        device = batch["gene"].x.device

        # OPTIMIZATION: Pre-allocate ALL term activations
        all_activations = torch.zeros(
            batch_size, self.num_go_terms, self.max_output_dim, device=device
        )
        activation_mask = torch.zeros(
            batch_size, self.num_go_terms, dtype=torch.bool, device=device
        )

        # Pre-allocate linear outputs
        linear_outputs_tensor = torch.zeros(
            batch_size, self.num_go_terms, self.output_size, device=device
        )

        # Track timing for performance monitoring (optional)
        stratum_times = []

        # Process each stratum in descending order (leaves to root)
        for stratum_idx in range(self.max_stratum, -1, -1):
            # Get all terms in this stratum using pre-computed tensor
            stratum_terms = self.stratum_term_lists[stratum_idx]

            if len(stratum_terms) == 0:
                continue

            # Time stratum processing (only in debug/profile mode)
            if hasattr(self, "_profile_mode") and self._profile_mode:
                start_time = time.time()

            # PARALLEL PROCESSING: Process all terms in stratum simultaneously
            # This maintains hierarchical dependencies while maximizing parallelism
            self._process_stratum_parallel(
                stratum_terms,
                batch,
                all_activations,
                activation_mask,
                linear_outputs_tensor,
            )

            if hasattr(self, "_profile_mode") and self._profile_mode:
                elapsed = time.time() - start_time
                num_terms = len(
                    [
                        t
                        for t in stratum_terms
                        if self.has_subsystem[t.item() if hasattr(t, "item") else t]
                    ]
                )
                stratum_times.append((stratum_idx, num_terms, elapsed))

        # Log profiling info if enabled
        if hasattr(self, "_profile_mode") and self._profile_mode and stratum_times:
            total_time = sum(t[2] for t in stratum_times)
            print("\nStratum processing times (PARALLEL):")
            for stratum_idx, num_terms, elapsed in stratum_times:
                if num_terms > 0:
                    print(
                        f"  Stratum {stratum_idx}: {num_terms} terms in {elapsed:.4f}s ({elapsed / num_terms:.6f}s per term)"
                    )
            print(f"  Total: {total_time:.4f}s")

        # Extract root prediction (stratum 0)
        root_terms = self.stratum_term_lists[0]
        if len(root_terms) == 0:
            raise ValueError("No root terms found in stratum 0")

        # OPTIMIZATION: Use tensor indexing directly - no .item() needed
        root_term_idx = root_terms[0]  # Already a tensor
        predictions = linear_outputs_tensor[:, root_term_idx, :].squeeze(-1)

        # Convert tensors back to dictionary for compatibility
        # This happens AFTER all computation, minimizing graph breaks
        linear_outputs = {}
        for term_idx in range(self.num_go_terms):
            if activation_mask[0, term_idx]:  # Check if term was processed
                linear_outputs[f"GO:{term_idx}"] = linear_outputs_tensor[
                    :, term_idx, :
                ].squeeze(-1)

        linear_outputs["GO:ROOT"] = predictions

        # Also create term_activations dict for compatibility
        term_activations = {}
        for term_idx in range(self.num_go_terms):
            if activation_mask[0, term_idx]:
                # Get actual output dim for this term
                output_dim = self.term_output_dims.get(term_idx, self.max_output_dim)
                term_activations[term_idx] = all_activations[:, term_idx, :output_dim]

        outputs = {
            "linear_outputs": linear_outputs,
            "root_key": f"GO:{int(root_term_idx)}",
            "term_activations": term_activations,
            "all_activations_tensor": all_activations,  # Keep tensor version
            "activation_mask": activation_mask,
            "stratum_outputs": {},
        }

        return predictions, outputs

    def _prepare_term_input_optimized(
        self,
        term_idx: torch.Tensor,  # Now accepts tensor instead of int
        batch: HeteroData,
        all_activations: torch.Tensor,
        activation_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare input for a term using pre-allocated tensors."""
        batch_size = all_activations.size(0)
        device = all_activations.device
        inputs = []

        # Get children using pre-computed indices - use torch.index_select for compile
        # Ensure term_idx is a tensor on the right device
        if not isinstance(term_idx, torch.Tensor):
            term_idx = torch.tensor(
                term_idx, dtype=torch.long, device=self.num_children.device
            )
        else:
            # Ensure it's on the right device
            term_idx = term_idx.to(self.num_children.device)
            if term_idx.dim() == 0:
                term_idx = term_idx.unsqueeze(0)

        num_children = torch.index_select(self.num_children, 0, term_idx).squeeze()

        # Use tensor comparison instead of .item()
        if num_children > 0:
            # Use index_select for compile-friendly indexing
            children_row = torch.index_select(
                self.children_indices, 0, term_idx
            ).squeeze(0)
            children_indices = children_row[:num_children]

            # Extract child activations from pre-allocated tensor
            for child_idx in children_indices:
                # OPTIMIZATION: Keep as tensor - no .item()
                if activation_mask[0, child_idx]:  # Check if child was processed
                    # Get actual output dim for this child using tensor indexing
                    # Use the pre-computed tensor for compile-friendly access
                    child_output_dim = self.term_output_dims_tensor[child_idx]
                    child_act = all_activations[:, child_idx, :child_output_dim]
                    inputs.append(child_act)

        # Add gene perturbation states
        gene_states = self._extract_gene_states_for_term(term_idx, batch)
        if gene_states.numel() > 0 and gene_states.size(1) > 0:
            inputs.append(gene_states)

        # Concatenate all inputs
        if inputs:
            return torch.cat(inputs, dim=1)
        else:
            # Shouldn't happen with well-formed GO hierarchy
            return torch.zeros(batch_size, 1, device=device)


class DCellSubsystem(nn.Module):
    """Individual subsystem module as described in DCell paper."""

    def __init__(self, input_dim: int, output_dim: int):
        """Build a linear-batchnorm-tanh block with DCell weight initialization."""
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.activation = nn.Tanh()

        # DCell paper weight initialization: uniform random between -0.001 and 0.001
        nn.init.uniform_(self.linear.weight, -0.001, 0.001)
        nn.init.uniform_(self.linear.bias, -0.001, 0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear, batch norm, and tanh activation to the input."""
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


# DCellOpt is a separate implementation optimized for torch.compile
# Import DCell from dcell.py if you need the original implementation


if __name__ == "__main__":
    raise SystemExit(
        "the demo main moved to torchcell/scratch/dcell_opt_demo.py on 2026-10-06"
    )
