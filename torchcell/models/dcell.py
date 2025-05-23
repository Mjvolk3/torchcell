# torchcell/models/dcell
# [[torchcell.models.dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/dcell
# Test file: tests/torchcell/models/test_dcell.py


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


class DCell(nn.Module):
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
        self.hetero_data = hetero_data  # Store reference for dimension calculations

        # Extract GO ontology information
        self.num_genes = hetero_data["gene"].num_nodes
        self.num_go_terms = hetero_data["gene_ontology"].num_nodes
        self.strata = hetero_data["gene_ontology"].strata
        self.stratum_to_terms = hetero_data["gene_ontology"].stratum_to_terms
        self.term_gene_counts = hetero_data["gene_ontology"].term_gene_counts

        # Build hierarchy and gene mappings
        self.child_to_parents = self._build_hierarchy(hetero_data)
        self.parent_to_children = self._build_parent_to_children()
        self.term_to_genes = self._build_term_gene_mapping(hetero_data)

        # Pre-compute input/output dimensions and create modules
        self.term_input_dims = {}
        self.term_output_dims = {}
        self.subsystems = nn.ModuleDict()
        self.linear_heads = nn.ModuleDict()

        # Get strata in descending order (leaves to root: high strata to low strata)
        # Strata 0 = root, higher numbers = more specific/leaf terms
        self.max_stratum = max(self.stratum_to_terms.keys())
        self.strata_order = sorted(self.stratum_to_terms.keys(), reverse=True)

        # Build modules for each GO term
        for term_idx in range(self.num_go_terms):
            self._build_term_module(term_idx)

        # Pre-compute gene state extraction indices
        self._precompute_gene_indices(hetero_data)

        print(f"DCell model initialized:")
        print(f"  GO terms: {self.num_go_terms}")
        print(f"  Genes: {self.num_genes}")
        print(f"  Strata: {len(self.strata_order)} (max: {self.max_stratum})")
        print(f"  Total subsystems: {len(self.subsystems)}")

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

        # Add dimension for gene perturbation states - must match _extract_gene_states_for_term
        # Count genes associated with this GO term from template go_gene_strata_state
        go_gene_state = self.hetero_data["gene_ontology"].go_gene_strata_state
        term_mask = go_gene_state[:, 0] == term_idx
        num_genes_for_term = term_mask.sum().item()

        # _extract_gene_states_for_term always returns at least dimension 1 (even for terms with no genes)
        # So we must match this behavior during initialization
        gene_dim = max(num_genes_for_term, 1)
        input_dim += gene_dim

        return max(input_dim, 1)  # Ensure at least 1 input

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

        # Create subsystem module
        subsystem = DCellSubsystem(input_dim, output_dim)
        self.subsystems[str(term_idx)] = subsystem

        # Create linear head for auxiliary supervision
        linear_head = nn.Linear(output_dim, self.output_size)
        self.linear_heads[str(term_idx)] = linear_head

    def _precompute_gene_indices(self, hetero_data: HeteroData):
        """Pre-compute indices for efficient gene state extraction."""
        go_gene_state = hetero_data["gene_ontology"].go_gene_strata_state

        # Create a mapping from GO term to gene indices
        self.term_gene_indices = {}
        for term_idx in range(self.num_go_terms):
            mask = go_gene_state[:, 0] == term_idx
            self.term_gene_indices[term_idx] = torch.where(mask)[0]

    def _extract_gene_states_for_term(
        self, term_idx: int, batch: HeteroData
    ) -> torch.Tensor:
        """Extract gene states for a specific GO term across all batches using batch pointers."""
        batch_size = batch["gene"].batch.max().item() + 1
        device = batch["gene"].x.device

        go_gene_state = batch["gene_ontology"].go_gene_strata_state
        ptr = batch["gene_ontology"].go_gene_strata_state_ptr

        # Reshape concatenated tensor to [batch_size, samples_per_batch, 4] format
        # ptr tells us the boundaries: each sample has the same number of rows
        rows_per_sample = ptr[1].item() - ptr[0].item()  # Should be 59986

        # Reshape: [total_rows, 4] -> [batch_size, rows_per_sample, 4]
        go_gene_state_batched = go_gene_state.view(batch_size, rows_per_sample, 4)

        # Extract for each sample in the batch
        term_gene_states_list = []

        for i in range(batch_size):
            # Get this sample's go_gene_strata_state: [rows_per_sample, 4]
            sample_go_gene_state = go_gene_state_batched[i]

            # Find rows for this term in this sample
            term_mask = sample_go_gene_state[:, 0] == term_idx
            term_rows = sample_go_gene_state[term_mask]

            if term_rows.size(0) > 0:
                gene_states = term_rows[:, 3]  # Extract states (column 3)
                term_gene_states_list.append(gene_states)
            else:
                # No genes for this term in this sample - create empty tensor
                term_gene_states_list.append(torch.tensor([], device=device))

        # Handle case where no samples have genes for this term
        if all(len(states) == 0 for states in term_gene_states_list):
            return torch.zeros(batch_size, 1, device=device)

        # Pad to same length and stack
        max_genes = max(
            len(states) for states in term_gene_states_list if len(states) > 0
        )
        if max_genes == 0:
            return torch.zeros(batch_size, 1, device=device)

        padded_states = []
        for states in term_gene_states_list:
            if len(states) == 0:
                # No genes for this term in this sample
                padded_states.append(torch.zeros(max_genes, device=device))
            elif len(states) < max_genes:
                # Pad with zeros to match max length
                padding = torch.zeros(max_genes - len(states), device=device)
                padded_states.append(torch.cat([states, padding]))
            else:
                # Already at max length
                padded_states.append(states)

        return torch.stack(padded_states)  # [batch_size, max_genes_for_term]

    def forward(self, batch: HeteroData) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through DCell hierarchy."""
        batch_size = batch["gene"].batch.max().item() + 1
        device = batch["gene"].x.device

        # Store term activations and linear outputs
        term_activations = {}
        linear_outputs = {}

        # Process each stratum in descending order (leaves to root: high strata → low strata)
        # This processes leaf terms first, then their parents, up to root (stratum 0)
        for stratum in self.strata_order:
            # All strata should exist in stratum_to_terms - if not, it's a bug
            if stratum not in self.stratum_to_terms:
                raise ValueError(
                    f"Stratum {stratum} not found in stratum_to_terms. This indicates a bug in GO graph processing."
                )

            stratum_terms = self.stratum_to_terms[stratum]

            # Process all terms in this stratum
            for term_idx in stratum_terms:
                # Convert to int if it's a tensor, otherwise assume it's already an int
                if torch.is_tensor(term_idx):
                    term_idx_int = term_idx.item()
                else:
                    term_idx_int = int(
                        term_idx
                    )  # Ensure it's an int, not numpy scalar or other type

                # Prepare input for this term
                term_input = self._prepare_term_input(
                    term_idx_int, batch, term_activations
                )

                subsystem = self.subsystems[str(term_idx_int)]

                # Forward through subsystem
                term_output = subsystem(term_input)
                term_activations[term_idx_int] = term_output

                # Get linear output for auxiliary supervision
                linear_head = self.linear_heads[str(term_idx_int)]
                linear_output = linear_head(term_output)
                linear_outputs[f"GO:{term_idx_int}"] = linear_output.squeeze(-1)

        # Main prediction from root terms (stratum 0)
        if 0 not in self.stratum_to_terms:
            raise ValueError(
                "No root terms found in stratum 0. Check GO hierarchy processing."
            )

        root_terms = self.stratum_to_terms[0]
        if len(root_terms) == 0:
            raise ValueError(
                "Root terms tensor is empty. Check GO term filtering - root terms may have been filtered out."
            )

        root_term_idx = root_terms[0].item()
        if f"GO:{root_term_idx}" not in linear_outputs:
            raise ValueError(
                f"Root term {root_term_idx} was not processed. Check strata processing order."
            )

        predictions = linear_outputs[f"GO:{root_term_idx}"]

        # Mark root output for DCellLoss
        linear_outputs["GO:ROOT"] = predictions

        # Prepare outputs dictionary
        outputs = {
            "linear_outputs": linear_outputs,
            "term_activations": term_activations,
            "stratum_outputs": {},
        }

        return predictions, outputs

    def _prepare_term_input(
        self,
        term_idx: int,
        batch: HeteroData,
        term_activations: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Prepare input for a specific GO term."""
        batch_size = batch["gene"].batch.max().item() + 1
        device = batch["gene"].x.device
        inputs = []

        # Add inputs from child subsystems (children are processed first due to leaves->root order)
        children = self.parent_to_children.get(term_idx, [])
        for child_idx in children:
            if child_idx in term_activations:
                inputs.append(term_activations[child_idx])

        # Add gene perturbation states for this GO term
        gene_states = self._extract_gene_states_for_term(term_idx, batch)
        if gene_states.numel() > 0 and gene_states.size(1) > 0:
            inputs.append(gene_states)

        # Concatenate all inputs along feature dimension
        if inputs:
            return torch.cat(inputs, dim=1)
        else:
            # If no inputs, this indicates a bug - every GO term should have either children or genes
            raise ValueError(
                f"GO term {term_idx} has no children and no genes. This should not happen in a well-formed GO hierarchy."
            )


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


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/005-kuzmin2018-tmi/conf"),
    config_name="dcell_kuzmin2018_tmi",
)
def main(cfg: DictConfig):
    """
    Main function to test the DCell model with overfitting on a batch
    """
    from torchcell.scratch.load_batch_005 import load_sample_data_batch
    from torchcell.losses.dcell import DCellLoss
    import torch.optim as optim
    from datetime import datetime

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

    plot_dir = osp.join(ASSET_IMAGES_DIR, f"dcell_training_{timestamp()}")
    os.makedirs(plot_dir, exist_ok=True)
    
    def save_intermediate_plot(epoch, all_losses, primary_losses, auxiliary_losses, weighted_auxiliary_losses, learning_rates, model, batch):
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
        plt.plot(range(1, epoch + 2), weighted_auxiliary_losses, "orange", label="Weighted Auxiliary Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Loss Components")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")
        
        # Get current model predictions for correlation plot
        model.eval()
        with torch.no_grad():
            current_predictions, current_outputs = model(batch)
            true_scores = batch["gene"].phenotype_values
            
            # Convert to numpy for plotting
            true_np = true_scores.cpu().numpy()
            pred_np = current_predictions.cpu().numpy()
            
            # Calculate correlation
            correlation = np.corrcoef(true_np, pred_np)[0, 1] if len(true_np) > 1 else 0.0
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
        plt.savefig(osp.join(plot_dir, f"dcell_epoch_{epoch+1:04d}_{timestamp()}.png"), dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory

    # Load sample data - modified for DCell with GO ontology
    print("Loading sample data with GO ontology...")

    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=32, num_workers=4, config="dcell", is_dense=False
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

    # Initialize DCell model
    print("Initializing DCell model...")
    model = DCell(
        cell_graph,
        min_subsystem_size=cfg.model.subsystem_output_min,
        subsystem_ratio=cfg.model.subsystem_output_max_mult,
        output_size=cfg.model.output_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

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

    # Training parameters
    epochs = cfg.trainer.max_epochs
    plot_interval = cfg.regression_task.plot_every_n_epochs

    # Lists to track metrics
    all_losses = []
    primary_losses = []
    auxiliary_losses = []
    weighted_auxiliary_losses = []
    learning_rates = []

    # Training loop
    print("Training DCell to overfit on batch...")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions, outputs = model(batch)

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

        # Print progress at intervals
        if (epoch + 1) % plot_interval == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Total Loss: {total_loss.item():.6f}, "
                f"Primary Loss: {primary_loss.item():.6f}, "
                f"Aux Loss: {auxiliary_loss.item():.6f}, "
                f"Weighted Aux Loss: {weighted_auxiliary_loss.item():.6f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save intermediate plot
            save_intermediate_plot(epoch, all_losses, primary_losses, auxiliary_losses, weighted_auxiliary_losses, learning_rates, model, batch)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_predictions, final_outputs = model(batch)

        print("\nFinal DCell training results:")
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

            # Create final plot
            plt.figure(figsize=(12, 8))

            # Loss curves
            plt.subplot(2, 3, 1)
            plt.plot(range(1, epochs + 1), all_losses, "b-", label="Total Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Training Loss Curve")
            plt.grid(True)
            plt.legend()
            plt.yscale("log")

            # Loss components
            plt.subplot(2, 3, 2)
            plt.plot(range(1, epochs + 1), primary_losses, "r-", label="Primary Loss")
            plt.plot(
                range(1, epochs + 1), auxiliary_losses, "g-", label="Auxiliary Loss"
            )
            plt.plot(
                range(1, epochs + 1),
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

            # Correlation
            plt.subplot(2, 3, 3)
            plt.scatter(true_np, pred_np, alpha=0.7)
            min_val = min(true_np.min(), pred_np.min())
            max_val = max(true_np.max(), pred_np.max())
            plt.plot([min_val, max_val], [min_val, max_val], "r--")
            plt.xlabel("True Phenotype Values")
            plt.ylabel("Predicted Values")
            plt.title(f"Final Correlation (r={correlation:.4f})")
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
            plt.plot(range(1, epochs + 1), learning_rates, "purple")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True)
            plt.yscale("log")

            # Subsystem analysis
            plt.subplot(2, 3, 6)
            linear_outputs = final_outputs.get("linear_outputs", {})
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
            plt.savefig(osp.join(plot_dir, f"dcell_results_{timestamp()}.png"), dpi=150)
            plt.close()

            print(f"\nResults plot saved to '{plot_dir}'")
        else:
            print("No predictions were generated. Check model and data setup.")

    print("\nDCell training demonstration complete!")

    return model, (final_predictions, final_outputs)


if __name__ == "__main__":
    main()
