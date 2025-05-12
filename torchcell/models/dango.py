# torchcell/models/dango
# [[torchcell.models.dango]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/dango
# Test file: tests/torchcell/models/test_dango.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
from torch_geometric.nn import SAGEConv
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean
import hydra
import os
import os.path as osp
from omegaconf import DictConfig, OmegaConf


class DangoPreTrain(nn.Module):
    """
    GNN pre-training component of DANGO model that learns node embeddings from
    protein-protein interaction networks in S. cerevisiae.

    As described in the paper, this module:
    1. Processes PPI networks from the STRING database
    2. Uses a 2-layer GNN for each network to reconstruct graph structure
    3. Shares an initial embedding layer across all networks
    4. Uses the output embeddings for downstream tasks
    """

    def __init__(
        self, gene_num: int, edge_types: List[str], hidden_channels: int = 64
    ):
        super().__init__()

        # Initialize model parameters
        self.gene_num = gene_num
        self.hidden_channels = hidden_channels
        self.edge_types = edge_types

        # Shared embedding layer across all GNNs (H^(0))
        self.gene_embedding = nn.Embedding(gene_num, hidden_channels)

        # Two layers of GNN for each network
        self.layer1_convs = nn.ModuleDict()
        self.layer2_convs = nn.ModuleDict()

        # Reconstruction layers for each network
        self.recon_layers = nn.ModuleDict()

        # Lambda values for weighted MSE
        self.lambda_values = {}

        # Initialize GNN layers and reconstruction layers for each edge type
        for edge_type in self.edge_types:
            # First layer GNN
            self.layer1_convs[edge_type] = SAGEConv(
                hidden_channels, hidden_channels, normalize=False, project=False
            )

            # Second layer GNN
            self.layer2_convs[edge_type] = SAGEConv(
                hidden_channels, hidden_channels, normalize=False, project=False
            )

            # Reconstruction layer to predict adjacency matrix row
            self.recon_layers[edge_type] = nn.Linear(hidden_channels, gene_num)

            # Set lambda value for weighted MSE based on network type
            # For STRING v9.1 networks (as in the paper)
            if edge_type == "string9_1_neighborhood" or edge_type == "string11_0_neighborhood":
                self.lambda_values[edge_type] = 0.1  # > 1% zeros decreased
            elif edge_type == "string9_1_coexpression" or edge_type == "string11_0_coexpression":
                self.lambda_values[edge_type] = 0.1  # > 1% zeros decreased
            elif edge_type == "string9_1_experimental" or edge_type == "string11_0_experimental":
                self.lambda_values[edge_type] = 0.1  # > 1% zeros decreased
            else:  # fusion, cooccurence, database (≤ 1% zeros decreased)
                self.lambda_values[edge_type] = 1.0

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.normal_(self.gene_embedding.weight, mean=0, std=0.1)

        for edge_type in self.edge_types:
            self.layer1_convs[edge_type].reset_parameters()
            self.layer2_convs[edge_type].reset_parameters()
            nn.init.normal_(self.recon_layers[edge_type].weight, mean=0, std=0.1)
            nn.init.zeros_(self.recon_layers[edge_type].bias)

    def forward(
        self, cell_graph: HeteroData
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """
        Forward pass for the DangoPreTrain model

        Args:
            cell_graph: The cell graph containing multiple edge types

        Returns:
            Dictionary containing:
                - 'embeddings': Node embeddings for each edge type (H_i^(2))
                - 'reconstructions': Reconstructed adjacency matrix rows for each edge type (FC(H_i^(2)))
        """
        device = self.gene_embedding.weight.device

        # Get gene node indices
        gene_data = cell_graph["gene"]
        num_nodes = gene_data.num_nodes
        node_indices = torch.arange(num_nodes, device=device)

        # Get initial node embeddings (H^(0)) - shared across all networks
        x_init = self.gene_embedding(node_indices)

        # Process each network separately
        embeddings = {}
        reconstructions = {}

        for edge_type in self.edge_types:
            edge_key = ("gene", edge_type, "gene")

            if edge_key in cell_graph.edge_types:
                edge_index = cell_graph[edge_key].edge_index

                # First layer (H^(1))
                # SAGEConv internally handles the neighborhood aggregation and concatenation
                h1 = self.layer1_convs[edge_type](x_init, edge_index)
                h1 = F.relu(h1)

                # Second layer (H^(2))
                h2 = self.layer2_convs[edge_type](h1, edge_index)
                h2 = F.relu(h2)

                # Store final embeddings (H^(2))
                embeddings[edge_type] = h2

                # Reconstruction to predict adjacency matrix row
                recon = self.recon_layers[edge_type](h2)
                reconstructions[edge_type] = recon
            else:
                # If edge type not in graph, use zeros
                embeddings[edge_type] = torch.zeros_like(x_init)
                reconstructions[edge_type] = torch.zeros(
                    num_nodes, self.gene_num, device=device
                )

        return {
            "embeddings": embeddings,
            "reconstructions": reconstructions,
            "initial_embeddings": x_init,
        }


class MetaEmbedding(nn.Module):
    """
    Meta-embedding module to integrate embeddings from multiple networks.

    As described in the paper, this module:
    1. Takes embeddings from 6 different PPI networks for each node
    2. Uses an MLP to compute attention weights for each embedding
    3. Combines embeddings using a weighted sum based on learned attention
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        # MLP for attention weights (two fully-connected layers)
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for m in self.attention_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, embeddings_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to integrate embeddings from multiple networks

        Args:
            embeddings_dict: Dictionary mapping edge types to node embeddings
                             Each value has shape [num_nodes, hidden_channels]

        Returns:
            Integrated embeddings with shape [num_nodes, hidden_channels]
        """
        # Get list of embeddings from the dictionary
        embeddings_list = list(embeddings_dict.values())

        # Stack embeddings along new dimension
        # Shape: [num_nodes, num_networks, hidden_channels]
        stacked_embeddings = torch.stack(embeddings_list, dim=1)

        # Compute attention scores for each embedding
        # First, reshape for the MLP
        num_nodes, num_networks, hidden_channels = stacked_embeddings.shape
        reshaped_embeddings = stacked_embeddings.view(-1, hidden_channels)

        # Apply MLP
        attention_scores = self.attention_mlp(reshaped_embeddings)
        attention_scores = attention_scores.view(num_nodes, num_networks)

        # Apply softmax to get normalized weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Expand weights for broadcasting
        # Shape: [num_nodes, num_networks, 1]
        attention_weights = attention_weights.unsqueeze(-1)

        # Compute weighted sum
        # Shape: [num_nodes, hidden_channels]
        meta_embeddings = (stacked_embeddings * attention_weights).sum(dim=1)

        return meta_embeddings


class HyperSAGNN(nn.Module):
    """
    Fully vectorized Hypergraph Self-Attention Graph Neural Network
    that handles all perturbation sets in a single forward pass.
    """

    def __init__(self, hidden_channels: int, num_heads: int = 4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        # Static embedding layer
        self.static_embedding = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU()
        )

        # Attention layer parameters
        # Layer 1
        self.Q1 = nn.Linear(hidden_channels, hidden_channels)
        self.K1 = nn.Linear(hidden_channels, hidden_channels)
        self.V1 = nn.Linear(hidden_channels, hidden_channels)
        self.O1 = nn.Linear(hidden_channels, hidden_channels)
        self.beta1 = nn.Parameter(torch.zeros(1))

        # Layer 2
        self.Q2 = nn.Linear(hidden_channels, hidden_channels)
        self.K2 = nn.Linear(hidden_channels, hidden_channels)
        self.V2 = nn.Linear(hidden_channels, hidden_channels)
        self.O2 = nn.Linear(hidden_channels, hidden_channels)
        self.beta2 = nn.Parameter(torch.zeros(1))

        # Final prediction layer
        self.prediction_layer = nn.Linear(hidden_channels, 1)

    def forward(
        self, embeddings: torch.Tensor, batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass processing all nodes at once with masked attention.

        Args:
            embeddings: Tensor of shape [total_nodes, hidden_channels]
            batch_indices: Tensor of shape [total_nodes] indicating set membership

        Returns:
            Predicted interaction scores with shape [num_batches]
        """
        device = embeddings.device
        total_nodes = embeddings.size(0)

        # Get unique batches for score aggregation
        unique_batches = torch.unique(batch_indices)
        num_batches = len(unique_batches)

        # Compute static embeddings for all nodes
        static_embeddings = self.static_embedding(embeddings)

        # Create attention mask where nodes can only attend to others in same set
        # mask[i,j] = True if nodes i and j are in the same set, False otherwise
        same_set_mask = batch_indices.unsqueeze(-1) == batch_indices.unsqueeze(0)

        # Add self-mask to prevent nodes from attending to themselves
        self_mask = torch.eye(total_nodes, dtype=torch.bool, device=device)
        valid_attention_mask = same_set_mask & ~self_mask

        # Apply first attention layer with masked attention
        dynamic_embeddings = self._global_attention_layer(
            embeddings,
            valid_attention_mask,
            self.Q1,
            self.K1,
            self.V1,
            self.O1,
            self.beta1,
        )

        # Apply second attention layer
        dynamic_embeddings = self._global_attention_layer(
            dynamic_embeddings,
            valid_attention_mask,
            self.Q2,
            self.K2,
            self.V2,
            self.O2,
            self.beta2,
        )

        # Compute element-wise squared differences
        squared_diff = (dynamic_embeddings - static_embeddings) ** 2

        # Compute node scores
        node_scores = self.prediction_layer(squared_diff).squeeze(-1)

        # Aggregate scores for each set using scatter_mean
        interaction_scores = scatter_mean(
            node_scores, batch_indices, dim=0, dim_size=num_batches
        )

        return interaction_scores

    def _global_attention_layer(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        Q_proj: nn.Linear,
        K_proj: nn.Linear,
        V_proj: nn.Linear,
        O_proj: nn.Linear,
        beta: nn.Parameter,
    ) -> torch.Tensor:
        """
        Apply global masked multi-head attention.

        Args:
            x: Input tensor with shape [total_nodes, hidden_dim]
            attention_mask: Binary mask with shape [total_nodes, total_nodes]
                           True where attention is allowed, False elsewhere
            Q_proj, K_proj, V_proj, O_proj: Linear projections
            beta: ReZero parameter

        Returns:
            Output tensor with shape [total_nodes, hidden_dim]
        """
        total_nodes = x.size(0)

        # Linear projections
        Q = Q_proj(x)  # [total_nodes, hidden_dim]
        K = K_proj(x)  # [total_nodes, hidden_dim]
        V = V_proj(x)  # [total_nodes, hidden_dim]

        # Reshape for multi-head attention
        Q = Q.view(total_nodes, self.num_heads, self.head_dim).permute(1, 0, 2)
        K = K.view(total_nodes, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = V.view(total_nodes, self.num_heads, self.head_dim).permute(1, 0, 2)
        # Shape: [num_heads, total_nodes, head_dim]

        # Calculate attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        # Shape: [num_heads, total_nodes, total_nodes]

        # Expand attention_mask for multi-head attention
        expanded_mask = attention_mask.unsqueeze(0).expand(self.num_heads, -1, -1)

        # Apply attention mask - set masked-out values to -inf before softmax
        attention.masked_fill_(~expanded_mask, -float("inf"))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=-1)

        # Handle potential NaNs from empty rows (if a node can't attend to any others)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # Apply attention to values
        out = torch.matmul(attention_weights, V)
        # Shape: [num_heads, total_nodes, head_dim]

        # Reshape back to [total_nodes, hidden_dim]
        out = out.permute(1, 0, 2).contiguous().view(total_nodes, self.hidden_channels)

        # Apply output projection
        out = O_proj(out)

        # Apply ReZero connection
        return beta * out + x


class Dango(nn.Module):
    """
    DANGO model for predicting higher-order genetic interactions

    Implements:
    1. GNN pre-training component
    2. Meta-embedding integration
    3. Hypergraph self-attention for prediction
    """

    def __init__(self, gene_num: int, edge_types: List[str], hidden_channels: int = 64, num_heads: int = 4):
        super().__init__()
        self.hidden_channels = hidden_channels

        # GNN pre-training component
        self.pretrain_model = DangoPreTrain(
            gene_num=gene_num, edge_types=edge_types, hidden_channels=hidden_channels
        )

        # Meta-embedding integration module
        self.meta_embedding = MetaEmbedding(hidden_channels=hidden_channels)

        # Hypergraph self-attention network
        self.hyper_sagnn = HyperSAGNN(
            hidden_channels=hidden_channels, num_heads=num_heads
        )

        # Initialize weights with better defaults
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights for better training stability"""
        # Initialize HyperSAGNN linear layers
        for name, module in self.hyper_sagnn.named_modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(
                    module.weight, gain=nn.init.calculate_gain("relu")
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize ReZero parameters to small positive values instead of zero
        # This can help with better gradient flow early in training
        nn.init.constant_(self.hyper_sagnn.beta1, 0.01)
        nn.init.constant_(self.hyper_sagnn.beta2, 0.01)

    def forward(
        self, cell_graph: HeteroData, batch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for the DANGO model

        Args:
            cell_graph: The cell graph containing multiple edge types
            batch: HeteroDataBatch containing perturbation information

        Returns:
            Tuple containing:
                - predictions: Predicted interaction scores
                - outputs: Dictionary containing node embeddings and intermediate values
        """
        # Get embeddings from pre-training component
        pretrain_outputs = self.pretrain_model(cell_graph)

        # Extract node embeddings for each network
        network_embeddings = pretrain_outputs["embeddings"]

        # Integrate embeddings using meta-embedding module
        integrated_embeddings = self.meta_embedding(network_embeddings)

        # Base outputs dictionary
        outputs = {
            "network_embeddings": network_embeddings,
            "integrated_embeddings": integrated_embeddings,
            "reconstructions": pretrain_outputs["reconstructions"],
            "initial_embeddings": pretrain_outputs["initial_embeddings"],
        }

        # Directly index into integrated_embeddings to get perturbed gene embeddings
        perturbed_embeddings = integrated_embeddings[batch["gene"].perturbation_indices]

        # Pass the perturbed embeddings and batch indices to HyperSAGNN
        interaction_scores = self.hyper_sagnn(
            perturbed_embeddings, batch["gene"].perturbation_indices_batch
        )

        # Store results in the outputs dictionary
        outputs["interaction_scores"] = interaction_scores

        # Return both the predictions and the outputs dictionary
        return interaction_scores, outputs

    @property
    def num_parameters(self) -> Dict[str, int]:
        """
        Count the number of trainable parameters in the model
        """

        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "pretrain_model": count_params(self.pretrain_model),
            "meta_embedding": count_params(self.meta_embedding),
            "hyper_sagnn": count_params(self.hyper_sagnn),
        }

        # Calculate overall total
        counts["total"] = sum(counts.values())

        return counts


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/005-kuzmin2018-tmi/conf"),
    config_name="dango_kuzmin2018_tmi",
)
def main(cfg: DictConfig):
    """
    Main function to test the DANGO model with overfitting on a batch
    """
    import os
    import matplotlib.pyplot as plt
    from torchcell.scratch.load_batch_005 import load_sample_data_batch
    import torch.optim as optim
    import numpy as np
    from datetime import datetime
    from dotenv import load_dotenv

    # Import all scheduler types for easy toggling
    from torchcell.losses.dango import (
        DangoLoss,
        PreThenPost,
        LinearUntilUniform,
        LinearUntilFlipped,
    )

    load_dotenv()

    # Set device based on config
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() != "cpu"
        else "cpu"
    )
    print(f"Using device: {device}")

    # Setup directories for plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = f"dango_training_plots_{timestamp}"
    os.makedirs(plot_dir, exist_ok=True)

    # Create subdirectories for different plot types
    loss_dir = os.path.join(plot_dir, "loss_plots")
    correlation_dir = os.path.join(plot_dir, "correlation_plots")

    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(correlation_dir, exist_ok=True)

    # Load sample data
    print("Loading sample data...")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=32,
        num_workers=4,
        metabolism_graph="metabolism_bipartite",
        is_dense=True,
    )

    # Move data to device
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Print batch information
    print(f"Batch size: {batch.num_graphs}")
    print(f"Perturbation indices shape: {batch['gene'].perturbation_indices.shape}")
    if hasattr(batch["gene"], "perturbation_indices_batch"):
        print(
            f"Perturbation batch indices shape: {batch['gene'].perturbation_indices_batch.shape}"
        )
    print(f"Phenotype values shape: {batch['gene'].phenotype_values.shape}")

    # Initialize model
    print("Initializing DANGO model...")
    # Define default STRING v9.1 edge types for demo
    edge_types = [
        "string9_1_neighborhood",
        "string9_1_fusion",
        "string9_1_cooccurence",
        "string9_1_coexpression",
        "string9_1_experimental",
        "string9_1_database",
    ]
    model = Dango(
        gene_num=max_num_nodes,
        edge_types=edge_types,
        hidden_channels=cfg.model.hidden_channels,
        num_heads=cfg.model.num_heads,
    ).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Num parameters: {model.num_parameters}")
    print(f"Using {model.hyper_sagnn.num_heads} attention heads in HyperSAGNN")

    # Use lambda values directly from the model's pretrain component
    lambda_values = model.pretrain_model.lambda_values.copy()

    # Set up training parameters from config
    epochs = cfg.trainer.max_epochs
    plot_interval = cfg.regression_task.plot_every_n_epochs
    transition_epoch = cfg.regression_task.loss_scheduler.transition_epoch

    # Get scheduler class from scheduler map
    scheduler_type = cfg.regression_task.loss_scheduler.type
    from torchcell.losses.dango import SCHEDULER_MAP

    if scheduler_type not in SCHEDULER_MAP:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Create the scheduler instance with transition_epoch
    scheduler_class = SCHEDULER_MAP[scheduler_type]
    scheduler_kwargs = {"transition_epoch": transition_epoch}

    # No additional parameters needed for LinearUntilFlipped

    scheduler = scheduler_class(**scheduler_kwargs)
    print(f"Using {scheduler_type} scheduler with transition_epoch={transition_epoch}")

    # Initialize the DangoLoss module with the selected scheduler
    loss_func = DangoLoss(
        edge_types=model.pretrain_model.edge_types,
        lambda_values=lambda_values,
        scheduler=scheduler,
        reduction="mean",
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Lists to track metrics
    all_losses = []
    recon_losses = []
    interaction_losses = []
    weighted_recon_losses = []
    weighted_interaction_losses = []

    # Initialize validation metrics
    best_mse = float("inf")
    best_epoch = 0

    # Training loop
    print("Training to overfit on batch...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass - returns tuple of (interaction_scores, outputs)
        interaction_scores, outputs = model(cell_graph, batch)

        # Compute reconstruction loss for each edge type
        adjacency_matrices = {}
        for edge_type in model.pretrain_model.edge_types:
            edge_key = ("gene", edge_type, "gene")
            if edge_key in cell_graph.edge_types:
                adj_size = outputs["reconstructions"][edge_type].shape
                # Convert edge_index to dense adjacency matrix
                adjacency_matrices[edge_type] = torch.sparse_coo_tensor(
                    cell_graph[edge_key].edge_index,
                    torch.ones(cell_graph[edge_key].edge_index.shape[1], device=device),
                    (adj_size[0], adj_size[1]),
                ).to_dense()

        # Use the DangoLoss to compute the combined loss
        if interaction_scores.numel() > 0:
            total_loss, loss_dict = loss_func(
                predictions=interaction_scores,
                targets=batch["gene"].phenotype_values,
                reconstructions=outputs["reconstructions"],
                adjacency_matrices=adjacency_matrices,
                current_epoch=epoch,
            )

            # Extract individual losses from the loss dictionary
            recon_loss = loss_dict["reconstruction_loss"]
            interaction_loss = loss_dict["interaction_loss"]
            weighted_recon_loss = loss_dict["weighted_reconstruction_loss"]
            weighted_interaction_loss = loss_dict["weighted_interaction_loss"]
            alpha = loss_dict["alpha"]
        else:
            # If no interaction scores, just use reconstruction loss
            recon_loss = loss_func.compute_reconstruction_loss(
                outputs["reconstructions"], adjacency_matrices
            )
            total_loss = recon_loss
            interaction_loss = torch.tensor(0.0, device=device)
            alpha = torch.tensor(1.0, device=device)
            weighted_recon_loss = recon_loss
            weighted_interaction_loss = torch.tensor(0.0, device=device)

        # Record losses
        all_losses.append(total_loss.item())
        recon_losses.append(recon_loss.item())
        interaction_losses.append(interaction_loss.item())

        # Record weighted losses
        weighted_recon_losses.append(weighted_recon_loss.item())
        weighted_interaction_losses.append(weighted_interaction_loss.item())

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Print progress and generate plots at intervals
        if (epoch + 1) % plot_interval == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss.item():.4f}, "
                f"Recon Loss: {recon_loss.item():.4f}, Interaction Loss: {interaction_loss.item():.4f}, "
                f"Alpha: {alpha.item():.2f}, Weighted Recon: {weighted_recon_loss.item():.4f}, "
                f"Weighted Interaction: {weighted_interaction_loss.item():.4f}"
            )

            # Plot loss curves
            plt.figure(figsize=(14, 12))  # Increase height slightly
            plt.subplot(3, 2, 1)  # Change to 3x2 grid to match final plot
            plt.plot(range(1, epoch + 2), all_losses, "b-", label="Total Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Training Loss")
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 3)
            plt.plot(
                range(1, epoch + 2), recon_losses, "r-", label="Reconstruction Loss"
            )
            plt.plot(
                range(1, epoch + 2), interaction_losses, "g-", label="Interaction Loss"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Unweighted Component Losses")
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 5)
            plt.plot(
                range(1, epoch + 2),
                weighted_recon_losses,
                "r-",
                label="Weighted Reconstruction Loss",
            )
            plt.plot(
                range(1, epoch + 2),
                weighted_interaction_losses,
                "g-",
                label="Weighted Interaction Loss",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Weighted Component Losses (alpha={:.2f})".format(alpha.item()))
            plt.grid(True)
            plt.legend()

            # The plot saving is now handled after adding the correlation plot

            # Evaluate on training data
            model.eval()
            with torch.no_grad():
                # Updated to unpack tuple
                predicted_scores, eval_outputs = model(cell_graph, batch)

                if predicted_scores.numel() > 0:
                    true_scores = batch["gene"].phenotype_values

                    # Calculate metrics
                    mse = F.mse_loss(predicted_scores, true_scores).item()
                    mae = F.l1_loss(predicted_scores, true_scores).item()

                    # Track best model
                    if mse < best_mse:
                        best_mse = mse
                        best_epoch = epoch + 1

                    # Add correlation plot to same figure in the right column
                    plt.subplot(3, 2, 2)
                    plt.scatter(
                        true_scores.cpu().numpy(),
                        predicted_scores.cpu().numpy(),
                        alpha=0.7,
                    )

                    # Get min/max for plot limits
                    min_val = min(
                        true_scores.min().item(), predicted_scores.min().item()
                    )
                    max_val = max(
                        true_scores.max().item(), predicted_scores.max().item()
                    )
                    plt.plot([min_val, max_val], [min_val, max_val], "r--")

                    plt.xlabel("True Interaction Scores")
                    plt.ylabel("Predicted Interaction Scores")
                    plt.title(f"Epoch {epoch+1}: MSE={mse:.6f}, MAE={mae:.6f}")
                    plt.grid(True)

                    # Add error distribution
                    plt.subplot(3, 2, 4)
                    errors = predicted_scores.cpu().numpy() - true_scores.cpu().numpy()
                    plt.hist(errors, bins=20, alpha=0.7)
                    plt.xlabel("Prediction Error")
                    plt.ylabel("Frequency")
                    plt.title(f"Error Distribution (MSE={mse:.6f})")
                    plt.grid(True)

                    # Save the combined figure
                    plt.tight_layout()
                    plt.savefig(os.path.join(loss_dir, f"loss_epoch_{epoch+1}.png"))
                    plt.close()

                    # Also save separate correlation plot for specific directory
                    plt.figure(figsize=(10, 8))
                    plt.scatter(
                        true_scores.cpu().numpy(),
                        predicted_scores.cpu().numpy(),
                        alpha=0.7,
                    )
                    plt.plot([min_val, max_val], [min_val, max_val], "r--")
                    plt.xlabel("True Interaction Scores")
                    plt.ylabel("Predicted Interaction Scores")
                    plt.title(f"Epoch {epoch+1}: MSE={mse:.6f}, MAE={mae:.6f}")
                    plt.grid(True)
                    plt.savefig(
                        os.path.join(
                            correlation_dir, f"correlation_epoch_{epoch+1}.png"
                        )
                    )
                    plt.close()

                    # Network embedding visualization removed

    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Updated to unpack tuple
        predicted_scores, final_outputs = model(cell_graph, batch)

        print("\nTraining results:")
        if predicted_scores.numel() > 0:
            true_scores = batch["gene"].phenotype_values

            print("True Phenotype Values:", true_scores.cpu().numpy())
            print("Predicted Scores:", predicted_scores.cpu().numpy())

            # Calculate final metrics
            mse = F.mse_loss(predicted_scores, true_scores).item()
            mae = F.l1_loss(predicted_scores, true_scores).item()

            # Calculate correlation coefficient
            true_np = true_scores.cpu().numpy()
            pred_np = predicted_scores.cpu().numpy()
            correlation = np.corrcoef(true_np, pred_np)[0, 1]

            print(f"Final Mean Squared Error: {mse:.6f}")
            print(f"Final Mean Absolute Error: {mae:.6f}")
            print(f"Correlation Coefficient: {correlation:.6f}")
            print(f"Best MSE: {best_mse:.6f} at epoch {best_epoch}")

            # Create a comprehensive final results plot
            plt.figure(figsize=(14, 14))

            # Plot loss curves
            plt.subplot(3, 2, 1)
            plt.plot(range(1, epochs + 1), all_losses, "b-", label="Total Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Training Loss Curve")
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 3)
            plt.plot(
                range(1, epochs + 1), recon_losses, "r-", label="Reconstruction Loss"
            )
            plt.plot(
                range(1, epochs + 1), interaction_losses, "g-", label="Interaction Loss"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Unweighted Component Losses")
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 5)
            plt.plot(
                range(1, epochs + 1),
                weighted_recon_losses,
                "r-",
                label="Weighted Reconstruction Loss",
            )
            plt.plot(
                range(1, epochs + 1),
                weighted_interaction_losses,
                "g-",
                label="Weighted Interaction Loss",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Weighted Component Losses")
            plt.grid(True)
            plt.legend()

            # Plot final correlation
            plt.subplot(3, 2, 2)
            scatter = plt.scatter(true_np, pred_np, alpha=0.7)
            min_val = min(true_np.min(), pred_np.min())
            max_val = max(true_np.max(), pred_np.max())
            plt.plot([min_val, max_val], [min_val, max_val], "r--")
            plt.xlabel("True Interaction Scores")
            plt.ylabel("Predicted Interaction Scores")
            plt.title(f"Final Correlation (r={correlation:.4f})")
            plt.grid(True)

            # Plot error distribution
            plt.subplot(3, 2, 4)
            errors = pred_np - true_np
            plt.hist(errors, bins=20, alpha=0.7)
            plt.xlabel("Prediction Error")
            plt.ylabel("Frequency")
            plt.title(f"Error Distribution (MSE={mse:.6f})")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "final_results.png"))
            plt.close()

            print(
                f"\nFinal results plot saved to '{os.path.join(plot_dir, 'final_results.png')}'"
            )
        else:
            print("No interaction scores were predicted. Check batch format.")

    print("\nDemonstration complete!")
    print(f"All plots saved to directory: {plot_dir}")

    return model, (predicted_scores, final_outputs)


if __name__ == "__main__":
    main()
