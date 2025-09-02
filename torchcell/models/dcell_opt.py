# torchcell/models/dcell_opt
# [[torchcell.models.dcell_opt]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/dcell_opt
# Test file: tests/torchcell/models/test_dcell_opt.py

"""
Optimized DCell model for torch.compile compatibility.
Reduces graph breaks by using ModuleList instead of ModuleDict and tensorized operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any
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
import math


class DCellOpt(nn.Module):
    """Optimized DCell with tensorized operations for torch.compile."""

    def __init__(
        self,
        hetero_data: HeteroData,
        min_subsystem_size: int = 20,
        subsystem_ratio: float = 0.3,
        output_size: int = 1,
    ):
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
        self.term_input_dims = {}
        self.term_output_dims = {}

        # OPTIMIZATION: Use ModuleList instead of ModuleDict
        self.subsystems = nn.ModuleList([None] * self.num_go_terms)
        self.linear_heads = nn.ModuleList([None] * self.num_go_terms)

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
            "term_output_dims_tensor",
            torch.zeros(self.num_go_terms, dtype=torch.long)
        )
        for term_idx, output_dim in self.term_output_dims.items():
            self.term_output_dims_tensor[term_idx] = output_dim

        print(f"DCellOpt model initialized:")
        print(f"  GO terms: {self.num_go_terms}")
        print(f"  Genes: {self.num_genes}")
        print(f"  Strata: {len(self.strata_order)} (max: {self.max_stratum})")
        print(f"  Active subsystems: {self.has_subsystem.sum().item()}")
        print(f"  Max output dim: {self.max_output_dim}")

    def _build_tensorized_hierarchy(self):
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

    def _build_stratum_masks(self):
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
    def num_parameters(self) -> Dict[str, int]:
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

    def _build_hierarchy(self, hetero_data: HeteroData) -> Dict[int, List[int]]:
        """Build child -> parents mapping from edge_index."""
        child_to_parents = {}

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

    def _build_parent_to_children(self) -> Dict[int, List[int]]:
        """Build parent -> children mapping from child_to_parents."""
        parent_to_children = {}

        for child, parents in self.child_to_parents.items():
            for parent in parents:
                if parent not in parent_to_children:
                    parent_to_children[parent] = []
                parent_to_children[parent].append(child)

        return parent_to_children

    def _build_term_gene_mapping(self, hetero_data: HeteroData) -> Dict[int, List[int]]:
        """Build term -> genes mapping from go_gene_strata_state."""
        term_to_genes = {}

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
        num_genes_for_term = term_mask.sum().item()

        gene_dim = max(num_genes_for_term, 1)
        input_dim += gene_dim

        return max(input_dim, 1)

    def _calculate_output_dim(self, term_idx: int) -> int:
        """Calculate output dimension for a GO term based on DCell paper formula."""
        num_genes = self.term_gene_counts[term_idx].item()
        return max(self.min_subsystem_size, math.ceil(self.subsystem_ratio * num_genes))

    def _build_term_module(self, term_idx: int):
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

    def _precompute_gene_indices(self, hetero_data: HeteroData):
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
            torch.full((self.num_go_terms, max_genes_per_term), -1, dtype=torch.long)
        )
        self.register_buffer(
            "term_num_genes_tensor",
            torch.zeros(self.num_go_terms, dtype=torch.long)
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
        ptr = batch["gene_ontology"].go_gene_strata_state_ptr

        # OPTIMIZATION: Handle variable rows_per_sample more robustly
        rows_per_sample = ptr[1] - ptr[0]
        
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

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through DCell hierarchy with optimized tensor operations."""
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

        # Process each stratum in descending order (leaves to root)
        for stratum_idx in range(self.max_stratum, -1, -1):
            # Get all terms in this stratum using pre-computed tensor
            stratum_terms = self.stratum_term_lists[stratum_idx]

            if len(stratum_terms) == 0:
                continue

            # OPTIMIZATION: Process terms in batches to reduce loop overhead
            # Process all terms in this stratum
            for term_idx in stratum_terms:
                # term_idx is already a tensor from stratum_term_lists
                
                # Skip if no subsystem - use tensor indexing
                if not self.has_subsystem[term_idx]:
                    continue

                # Prepare input using optimized method - pass tensor directly
                term_input = self._prepare_term_input_optimized(
                    term_idx, batch, all_activations, activation_mask
                )

                # Direct indexing with tensor
                subsystem = self.subsystems[term_idx]
                linear_head = self.linear_heads[term_idx]

                # Forward through subsystem
                term_output = subsystem(term_input)

                # Store in pre-allocated tensor - use tensor indexing
                output_dim = term_output.size(1)
                all_activations[:, term_idx, :output_dim] = term_output
                activation_mask[:, term_idx] = True

                # Linear output
                linear_output = linear_head(term_output)
                linear_outputs_tensor[:, term_idx, :] = linear_output

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
            term_idx = torch.tensor(term_idx, dtype=torch.long, device=self.num_children.device)
        else:
            # Ensure it's on the right device
            term_idx = term_idx.to(self.num_children.device)
            if term_idx.dim() == 0:
                term_idx = term_idx.unsqueeze(0)
        
        num_children = torch.index_select(self.num_children, 0, term_idx).squeeze()
        
        # Use tensor comparison instead of .item()
        if num_children > 0:
            # Use index_select for compile-friendly indexing
            children_row = torch.index_select(self.children_indices, 0, term_idx).squeeze(0)
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
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.activation = nn.Tanh()

        # DCell paper weight initialization: uniform random between -0.001 and 0.001
        nn.init.uniform_(self.linear.weight, -0.001, 0.001)
        nn.init.uniform_(self.linear.bias, -0.001, 0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


# DCellOpt is a separate implementation optimized for torch.compile
# Import DCell from dcell.py if you need the original implementation


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="dcell_kuzmin2018_tmi",
)
def main(cfg: DictConfig):
    """
    Main function to test the optimized DCell model
    """
    from torchcell.scratch.load_batch_005 import load_sample_data_batch
    from torchcell.losses.dcell import DCellLoss
    import torch.optim as optim
    from datetime import datetime
    from tqdm import tqdm

    load_dotenv()

    # Set device based on config
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() != "cpu"
        else "cpu"
    )
    print(f"Using device: {device}")

    # Setup directories for plots
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    if ASSET_IMAGES_DIR is None:
        ASSET_IMAGES_DIR = "assets/images"

    plot_dir = osp.join(ASSET_IMAGES_DIR, f"dcell_opt_training_{timestamp()}")
    os.makedirs(plot_dir, exist_ok=True)

    def save_intermediate_plot(
        epoch,
        all_losses,
        primary_losses,
        auxiliary_losses,
        weighted_auxiliary_losses,
        learning_rates,
        model,
        batch,
    ):
        """Save intermediate training plot every print interval."""
        plt.figure(figsize=(12, 8))

        # Loss curves
        plt.subplot(2, 3, 1)
        plt.plot(range(1, epoch + 2), all_losses, "b-", label="Total Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")

        # Loss components
        plt.subplot(2, 3, 2)
        plt.plot(range(1, epoch + 2), primary_losses, "r-", label="Primary Loss")
        plt.plot(range(1, epoch + 2), auxiliary_losses, "g-", label="Auxiliary Loss")
        plt.plot(
            range(1, epoch + 2),
            weighted_auxiliary_losses,
            "orange",
            label="Weighted Auxiliary Loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Loss Components")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")

        # Get current model predictions for correlation plot
        model.eval()
        with torch.no_grad():
            current_predictions, current_outputs = model(cell_graph, batch)
            true_scores = batch["gene"].phenotype_values

            # Convert to numpy for plotting
            true_np = true_scores.cpu().numpy()
            pred_np = current_predictions.cpu().numpy()

            # Calculate correlation
            correlation = (
                np.corrcoef(true_np, pred_np)[0, 1] if len(true_np) > 1 else 0.0
            )
            mse = np.mean((pred_np - true_np) ** 2)
        model.train()  # Back to training mode

        # Correlation
        plt.subplot(2, 3, 3)
        plt.scatter(true_np, pred_np, alpha=0.7)
        min_val = min(true_np.min(), pred_np.min())
        max_val = max(true_np.max(), pred_np.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.xlabel("True Phenotype Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Correlation (r={correlation:.4f})")
        plt.grid(True)

        # Error distribution
        plt.subplot(2, 3, 4)
        errors = pred_np - true_np
        plt.hist(errors, bins=10, alpha=0.7)
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title(f"Error Distribution (MSE={mse:.6f})")
        plt.grid(True)

        # Learning rate evolution
        plt.subplot(2, 3, 5)
        plt.plot(range(1, epoch + 2), learning_rates, "purple")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.yscale("log")

        # Subsystem analysis
        plt.subplot(2, 3, 6)
        linear_outputs = current_outputs.get("linear_outputs", {})
        subsystem_activities = {}
        for k, v in linear_outputs.items():
            if k != "GO:ROOT":
                subsystem_activities[k] = v.abs().mean().item()

        if subsystem_activities:
            subsystem_activations = list(subsystem_activities.values())
            plt.hist(
                subsystem_activations,
                bins=min(20, len(subsystem_activations)),
                alpha=0.7,
            )
            plt.xlabel("Mean Absolute Activation")
            plt.ylabel("Number of Subsystems")
            plt.title("Subsystem Activation Distribution")
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            osp.join(plot_dir, f"dcell_opt_epoch_{epoch+1:04d}_{timestamp()}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # Load sample data
    print("Loading sample data with GO ontology...")

    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        config="dcell",
        is_dense=False,
    )

    # Move data to device
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Print batch information
    print(f"Batch size: {batch.num_graphs}")
    print(f"Gene nodes: {batch['gene'].num_nodes}")
    print(f"GO nodes: {batch['gene_ontology'].num_nodes}")
    print(f"Perturbation indices shape: {batch['gene'].perturbation_indices.shape}")
    print(
        f"GO gene strata state shape: {batch['gene_ontology'].go_gene_strata_state.shape}"
    )
    print(f"Phenotype values shape: {batch['gene'].phenotype_values.shape}")

    # Initialize optimized DCell model
    print("\nInitializing Optimized DCell model...")
    model = DCellOpt(
        cell_graph,
        min_subsystem_size=cfg.model.subsystem_output_min,
        subsystem_ratio=cfg.model.subsystem_output_max_mult,
        output_size=cfg.model.output_size,
    ).to(device)

    param_counts = model.num_parameters
    print(f"\nParameter counts:")
    print(f"  Subsystems: {param_counts['subsystems']:,}")
    print(f"  Linear heads: {param_counts['dcell_linear']:,}")
    print(f"  Total: {param_counts['total']:,}")

    # Test torch.compile if requested
    compile_mode = cfg.get("compile_mode", None)
    if compile_mode is not None:
        print(f"\nAttempting torch.compile with mode='{compile_mode}'...")
        try:
            model = torch.compile(model, mode=compile_mode, dynamic=True)
            print("✓ Successfully compiled model")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")
            print("Continuing without compilation")

    # Initialize DCellLoss
    loss_func = DCellLoss(
        alpha=cfg.regression_task.dcell_loss.alpha,
        use_auxiliary_losses=cfg.regression_task.dcell_loss.use_auxiliary_losses,
    )

    # Create optimizer
    if cfg.regression_task.optimizer.type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.regression_task.optimizer.lr,
            weight_decay=cfg.regression_task.optimizer.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.regression_task.optimizer.lr,
            weight_decay=cfg.regression_task.optimizer.weight_decay,
        )

    # Setup learning rate scheduler if specified
    scheduler = None
    if cfg.regression_task.lr_scheduler.type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.regression_task.lr_scheduler.mode,
            factor=cfg.regression_task.lr_scheduler.factor,
            patience=cfg.regression_task.lr_scheduler.patience,
            threshold=cfg.regression_task.lr_scheduler.threshold,
            min_lr=cfg.regression_task.lr_scheduler.min_lr,
        )

    # Training parameters - reduce epochs for testing
    epochs = min(cfg.trainer.max_epochs, 100)  # Limit to 100 for testing
    plot_interval = cfg.regression_task.plot_every_n_epochs

    # Lists to track metrics
    all_losses = []
    primary_losses = []
    auxiliary_losses = []
    weighted_auxiliary_losses = []
    learning_rates = []

    # Training loop
    print(f"\nTraining Optimized DCell for {epochs} epochs...")
    for epoch in tqdm(range(epochs)):
        epoch_start_time = time.time()

        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions, outputs = model(cell_graph, batch)

        # Get targets
        targets = batch["gene"].phenotype_values

        # Compute loss using DCellLoss
        total_loss, loss_components = loss_func(predictions, outputs, targets)

        # Extract individual loss components
        primary_loss = loss_components["primary_loss"]
        auxiliary_loss = loss_components["auxiliary_loss"]
        weighted_auxiliary_loss = loss_components["weighted_auxiliary_loss"]

        # Record losses
        all_losses.append(total_loss.item())
        primary_losses.append(primary_loss.item())
        auxiliary_losses.append(auxiliary_loss.item())
        weighted_auxiliary_losses.append(weighted_auxiliary_loss.item())
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # Backward pass and optimization
        total_loss.backward()

        # Gradient clipping if specified
        if cfg.regression_task.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.regression_task.clip_grad_norm_max_norm
            )

        optimizer.step()

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(total_loss)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Get GPU memory stats if using CUDA
        gpu_memory_str = ""
        if device.type == "cuda":
            allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
            gpu_memory_str = f", GPU: {allocated_gb:.2f}/{reserved_gb:.2f}GB"

        # Print stats every epoch
        current_batch_size = batch["gene"].batch.max().item() + 1
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Total Loss: {total_loss.item():.6f}, "
            f"Primary Loss: {primary_loss.item():.6f}, "
            f"Aux Loss: {auxiliary_loss.item():.6f}, "
            f"Weighted Aux Loss: {weighted_auxiliary_loss.item():.6f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}, "
            f"Time: {epoch_time:.2f}s, "
            f"Time/instance: {epoch_time/current_batch_size:.4f}s"
            f"{gpu_memory_str}"
        )

        # Save intermediate plot at intervals
        if (epoch + 1) % plot_interval == 0 or epoch == epochs - 1:
            save_intermediate_plot(
                epoch,
                all_losses,
                primary_losses,
                auxiliary_losses,
                weighted_auxiliary_losses,
                learning_rates,
                model,
                batch,
            )

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_predictions, final_outputs = model(cell_graph, batch)

        print("\nFinal Optimized DCell training results:")
        if final_predictions.numel() > 0:
            true_scores = batch["gene"].phenotype_values

            print("True Phenotype Values:", true_scores.cpu().numpy()[:5])
            print("Predicted Values:", final_predictions.cpu().numpy()[:5])

            # Calculate final metrics
            mse = F.mse_loss(final_predictions, true_scores).item()
            mae = F.l1_loss(final_predictions, true_scores).item()

            # Calculate correlation coefficient
            true_np = true_scores.cpu().numpy()
            pred_np = final_predictions.cpu().numpy()
            correlation = np.corrcoef(true_np, pred_np)[0, 1] if len(true_np) > 1 else 0

            print(f"Final Mean Squared Error: {mse:.6f}")
            print(f"Final Mean Absolute Error: {mae:.6f}")
            print(f"Correlation Coefficient: {correlation:.6f}")

            print(f"\nResults plot saved to '{plot_dir}'")
        else:
            print("No predictions were generated. Check model and data setup.")

    print("\nOptimized DCell training demonstration complete!")

    return model, (final_predictions, final_outputs)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
