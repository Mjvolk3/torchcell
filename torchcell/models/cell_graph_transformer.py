# torchcell/models/cell_graph_transformer
# [[torchcell.models.cell_graph_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/cell_graph_transformer
# Test file: tests/torchcell/models/test_cell_graph_transformer.py

"""
Cell Graph Transformer with graph-regularized attention heads.

Implements the architecture from weekly report 2025.45:
- CLS token for whole-cell representation
- Graph-regularized attention heads (KL loss to adjacency matrices)
- Perturbation head with cross-attention for gene interaction prediction
"""

import os
import os.path as osp
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple


class GraphRegularizedTransformerLayer(nn.Module):
    """
    Transformer layer with graph-regularized attention heads.

    Uses manual attention computation to get both output and attention weights
    for graph regularization loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        adjacency_matrices: Optional[Dict[str, torch.Tensor]] = None,
        regularized_head_config: Optional[Dict[str, Dict]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert (
            hidden_dim % num_heads == 0
        ), f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        # Projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer normalization and feedforward
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

        # Store adjacency matrices and regularization config
        self.adjacency_matrices = adjacency_matrices
        self.regularized_head_config = regularized_head_config

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with manual attention computation.

        Args:
            x: [batch, N+1, d] where N+1 includes CLS token at position 0
            return_attention: Whether to return attention weights for regularization

        Returns:
            output: [batch, N+1, d] transformed features
            gene_attention: [batch, heads, N, N] gene-gene attention weights (if return_attention=True)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Reshape: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Manual attention computation (get both output AND weights)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # [batch, heads, seq_len, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights_dropout = self.dropout(attention_weights)

        # Apply attention to values
        attn_output = torch.matmul(
            attention_weights_dropout, v
        )  # [batch, heads, seq_len, head_dim]

        # Extract gene-gene attention block for regularization (exclude CLS token)
        # attention_weights: [batch, heads, N+1, N+1]
        # gene_attention: [batch, heads, N, N]
        gene_attention = (
            attention_weights[:, :, 1:, 1:] if return_attention else None
        )

        # Reshape back: [batch, seq_len, hidden_dim]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, hidden_dim)
        )
        output = self.out_proj(attn_output)

        # Layer norm + residual connection
        output = self.norm1(x + self.dropout(output))

        # Feedforward network
        ffn_output = self.ffn(output)
        output = self.norm2(output + self.dropout(ffn_output))

        return output, gene_attention


class HyperSAGNN(nn.Module):
    """
    Hypergraph Self-Attention Graph Neural Network for perturbation sets.

    Adapted from DANGO model to compute perturbation representations via
    masked self-attention within perturbation sets.
    """

    def __init__(self, hidden_channels: int, num_heads: int = 4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        assert (
            hidden_channels % num_heads == 0
        ), f"hidden_channels {hidden_channels} must be divisible by num_heads {num_heads}"

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

    def forward(
        self, embeddings: torch.Tensor, batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass processing perturbed genes with masked attention.

        Args:
            embeddings: Tensor of shape [total_pert_genes, hidden_channels]
            batch_indices: Tensor of shape [total_pert_genes] indicating set membership

        Returns:
            Perturbation representations with shape [num_batches, hidden_channels]
        """
        device = embeddings.device
        total_nodes = embeddings.size(0)

        # Get unique batches
        unique_batches = torch.unique(batch_indices)
        num_batches = len(unique_batches)

        # Compute static embeddings for all perturbed genes
        static_embeddings = self.static_embedding(embeddings)

        # Create attention mask where genes can only attend to others in same set
        same_set_mask = batch_indices.unsqueeze(-1) == batch_indices.unsqueeze(0)

        # Add self-mask to prevent genes from attending to themselves
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

        # Aggregate per-set representations using scatter_mean
        from torch_scatter import scatter_mean

        set_representations = scatter_mean(
            squared_diff, batch_indices, dim=0, dim_size=num_batches
        )

        return set_representations  # [num_batches, hidden_channels]

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

        # Handle potential NaNs from empty rows (if a gene can't attend to any others)
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


class PerturbationHead(nn.Module):
    """
    Perturbation head with switchable attention mechanisms.

    Implements g_ψ(h_CLS, H_genes, M(S)) from weekly report using either:
    - Cross-attention (default): perturbation summary attends to all genes
    - HyperSAGNN: within-set self-attention for perturbation genes
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            # Cross-attention: perturbation summary queries all genes
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        else:
            # HyperSAGNN: computes z_S via within-set attention
            self.hypersagnn = HyperSAGNN(hidden_dim, num_heads)

        # Prediction MLP: [h_CLS || z_S] -> scalar
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        H: torch.Tensor,
        perturbation_indices: torch.Tensor,
        batch_assignment: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of perturbation head.

        Args:
            H: [N+1, d] transformer output (CLS at position 0, genes at 1:N+1)
            perturbation_indices: [total_pert_genes] indices of perturbed genes
            batch_assignment: [total_pert_genes] batch index for each perturbed gene

        Returns:
            predictions: [batch_size, 1] gene interaction predictions
        """
        batch_size = int(batch_assignment.max().item()) + 1

        # Extract CLS token and gene embeddings
        h_CLS = H[0]  # [d]
        H_genes = H[1:]  # [N, d]

        # Get perturbed gene embeddings
        h_pert = H_genes[perturbation_indices]  # [total_pert_genes, d]

        if self.use_cross_attention:
            # Cross-attention approach
            # Aggregate perturbed genes per sample (mean pooling)
            q_S_list = []
            for b in range(batch_size):
                mask = batch_assignment == b
                if mask.sum() > 0:
                    q_S_list.append(h_pert[mask].mean(dim=0))  # [d]
                else:
                    # Handle edge case: no perturbed genes in this sample
                    q_S_list.append(torch.zeros_like(h_CLS))

            q_S = torch.stack(q_S_list, dim=0)  # [batch_size, d]

            # Cross-attention: q_S attends to all genes
            z_S, _ = self.cross_attn(
                query=q_S.unsqueeze(1),  # [batch_size, 1, d]
                key=H_genes.unsqueeze(0).expand(
                    batch_size, -1, -1
                ),  # [batch_size, N, d]
                value=H_genes.unsqueeze(0).expand(
                    batch_size, -1, -1
                ),  # [batch_size, N, d]
            )
            z_S = z_S.squeeze(1)  # [batch_size, d]
        else:
            # HyperSAGNN approach
            # This applies masked self-attention within each perturbation set
            z_S = self.hypersagnn(h_pert, batch_assignment)  # [batch_size, d]

        # Concatenate with CLS token: [h_CLS || z_S]
        h_CLS_expanded = h_CLS.unsqueeze(0).expand(batch_size, -1)  # [batch_size, d]
        combined = torch.cat([h_CLS_expanded, z_S], dim=-1)  # [batch_size, 2*d]

        # Predict gene interaction
        predictions = self.mlp(combined)  # [batch_size, 1]

        return predictions


class CellGraphTransformer(nn.Module):
    """
    Cell Graph Transformer model.

    Architecture:
    1. Gene embeddings + CLS token
    2. Transformer encoder with graph-regularized attention
    3. Perturbation head with cross-attention
    """

    def __init__(
        self,
        gene_num: int,
        hidden_channels: int,
        num_transformer_layers: int,
        num_attention_heads: int,
        cell_graph: HeteroData,
        graph_regularization_config: Optional[Dict] = None,
        perturbation_head_config: Optional[Dict] = None,
        dropout: float = 0.1,
        adaptive_loss_weighting: bool = False,
        graph_reg_scale: float = 0.001,  # Global scale factor for graph reg
    ):
        super().__init__()
        self.gene_num = gene_num
        self.hidden_channels = hidden_channels
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.adaptive_loss_weighting = adaptive_loss_weighting
        self.graph_reg_scale = graph_reg_scale

        # Gene embeddings
        self.gene_embedding = nn.Embedding(gene_num, hidden_channels)
        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, hidden_channels) * 0.02)

        # Process graph regularization config
        self.graph_regularization_enabled = (
            graph_regularization_config is not None
            and graph_regularization_config.get("enabled", False)
        )

        if self.graph_regularization_enabled:
            # Normalize adjacency matrices from cell_graph
            self.adjacency_matrices = self._normalize_adjacency_matrices(
                cell_graph
            )
            self.regularized_head_config = graph_regularization_config.get(
                "regularized_heads", {}
            )
            self.row_sampling_rate = graph_regularization_config.get(
                "row_sampling_rate", 1.0
            )
        else:
            self.adjacency_matrices = None
            self.regularized_head_config = None
            self.row_sampling_rate = 1.0

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList(
            [
                GraphRegularizedTransformerLayer(
                    hidden_dim=hidden_channels,
                    num_heads=num_attention_heads,
                    adjacency_matrices=self.adjacency_matrices,
                    regularized_head_config=self.regularized_head_config,
                    dropout=dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Perturbation head
        pert_head_config = perturbation_head_config or {}
        self.perturbation_head = PerturbationHead(
            hidden_dim=hidden_channels,
            num_heads=pert_head_config.get("num_heads", 4),
            dropout=pert_head_config.get("dropout", dropout),
            use_cross_attention=pert_head_config.get("use_cross_attention", True),
        )

        # Adaptive loss weighting (learnable)
        if adaptive_loss_weighting:
            self.log_mse_weight = nn.Parameter(torch.tensor(0.0))
            self.log_reg_weight = nn.Parameter(torch.tensor(-3.0))  # Start very small

    def _normalize_adjacency_matrices(
        self, cell_graph: HeteroData
    ) -> Dict[str, torch.Tensor]:
        """
        Normalize adjacency matrices row-wise: A_tilde[i,:] = A[i,:] / (degree[i] + eps).

        Args:
            cell_graph: HeteroData with (gene, edge_type, gene) edges

        Returns:
            Dictionary of normalized adjacency matrices
        """
        normalized_matrices = {}

        # Extract gene-gene edge types only
        for edge_type in cell_graph.edge_types:
            src, rel, dst = edge_type

            # Only process gene-gene edges
            if src != "gene" or dst != "gene":
                continue

            # Get edge index
            edge_index = cell_graph[edge_type].edge_index  # [2, num_edges]

            # Create dense adjacency matrix
            num_nodes = self.gene_num
            A = torch.zeros(num_nodes, num_nodes)
            A[edge_index[0], edge_index[1]] = 1.0

            # Compute row-wise normalization
            row_sums = A.sum(dim=1, keepdim=True) + 1e-10  # [num_nodes, 1]
            A_tilde = A / row_sums  # [num_nodes, num_nodes]

            # Use the relation name as the key (e.g., "physical", "regulatory")
            normalized_matrices[rel] = A_tilde

        return normalized_matrices

    def compute_graph_regularization_loss(
        self, attention_weights: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """
        Compute graph regularization loss using KL divergence.

        Args:
            attention_weights: [batch, heads, N, N] gene-gene attention weights
            layer_idx: Current transformer layer index

        Returns:
            Total regularization loss for this layer
        """
        if not self.graph_regularization_enabled:
            return torch.tensor(0.0, device=attention_weights.device)

        total_loss = torch.tensor(0.0, device=attention_weights.device)
        batch_size, num_heads, N, _ = attention_weights.shape

        for graph_name, config in self.regularized_head_config.items():
            if config["layer"] != layer_idx:
                continue

            head_idx = config["head"]
            lambda_k = config["lambda"]

            # Get normalized adjacency
            if graph_name not in self.adjacency_matrices:
                continue

            A_tilde = self.adjacency_matrices[graph_name].to(
                attention_weights.device
            )  # [N, N]

            # Sample rows (for efficiency)
            if self.row_sampling_rate < 1.0:
                num_sample = int(self.row_sampling_rate * N)
                # Sample rows with positive degree (has edges)
                positive_rows = (A_tilde.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
                if len(positive_rows) > num_sample:
                    sample_idx = positive_rows[
                        torch.randperm(len(positive_rows), device=A_tilde.device)[
                            :num_sample
                        ]
                    ]
                else:
                    sample_idx = positive_rows
            else:
                sample_idx = torch.arange(N, device=A_tilde.device)

            if len(sample_idx) == 0:
                continue

            # Extract attention for this head: [batch, N, N]
            alpha = attention_weights[:, head_idx, :, :]

            # Compute KL divergence row-wise: KL(A_tilde[i,:] || alpha[i,:])
            # KL = Σ A_tilde[i,j] * log(A_tilde[i,j] / alpha[i,j])
            kl_loss = F.kl_div(
                (alpha[:, sample_idx, :] + 1e-8).log(),  # log predictions with epsilon for numerical stability
                A_tilde[sample_idx, :]
                .unsqueeze(0)
                .expand(batch_size, -1, -1),  # targets
                reduction="batchmean",
                log_target=False,
            )

            total_loss = total_loss + lambda_k * kl_loss

        # Apply global scale factor and normalize by number of edges
        total_edges = sum(len(self.adjacency_matrices[g].nonzero()[0])
                         for g in self.adjacency_matrices.keys())
        if total_edges > 0:
            total_loss = total_loss * self.graph_reg_scale / (total_edges / self.gene_num)

        return total_loss

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of Cell Graph Transformer.

        Args:
            cell_graph: Full wildtype graph structure (not used directly, genes indexed by order)
            batch: Perturbation data with indices and phenotypes

        Returns:
            predictions: [batch_size, 1] gene interaction predictions
            representations: Dict with embeddings, attention weights, and losses
        """
        device = self.gene_embedding.weight.device
        N = self.gene_num

        # 1. Create gene embeddings for all genes
        gene_idx = torch.arange(N, device=device)
        gene_embs = self.gene_embedding(gene_idx)  # [N, d]

        # 2. Prepend CLS token
        cls_token = self.cls_token  # [1, d]
        X = torch.cat([cls_token, gene_embs], dim=0).unsqueeze(0)  # [1, N+1, d]

        # 3. Transformer encoder
        H = X
        all_attention_weights = []
        total_graph_reg_loss = torch.tensor(0.0, device=device)

        for layer_idx, layer in enumerate(self.transformer_layers):
            H, attention_weights = layer(H, return_attention=True)

            if self.training and attention_weights is not None:
                # Compute graph regularization loss
                graph_loss = self.compute_graph_regularization_loss(
                    attention_weights, layer_idx
                )
                total_graph_reg_loss = total_graph_reg_loss + graph_loss
                all_attention_weights.append(attention_weights)

        # 4. Perturbation head
        H_squeezed = H.squeeze(0)  # [N+1, d]

        predictions = self.perturbation_head(
            H_squeezed,
            batch["gene"].perturbation_indices,
            batch["gene"].perturbation_indices_batch,
        )

        return predictions, {
            "h_CLS": H_squeezed[0],
            "H_genes": H_squeezed[1:],
            "attention_weights": all_attention_weights,
            "graph_reg_loss": total_graph_reg_loss,
        }

    @property
    def num_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""

        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "gene_embedding": count_params(self.gene_embedding),
            "cls_token": self.cls_token.numel(),
            "transformer_layers": count_params(self.transformer_layers),
            "perturbation_head": count_params(self.perturbation_head),
        }
        counts["total"] = sum(counts.values())
        return counts


def calculate_weight_l2_norm(model: nn.Module) -> float:
    """Calculate L2 norm of all model weights."""
    l2_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_norm += torch.sum(param**2).item()
    return np.sqrt(l2_norm)


def compute_smoothness(X: torch.Tensor) -> float:
    """
    Compute smoothness of node features (oversmoothing diagnostic).

    Lower values indicate oversmoothing (features collapsing toward mean).
    Higher values indicate feature diversity is preserved.

    Args:
        X: Node feature matrix [N, d]

    Returns:
        Frobenius norm of deviation from mean features
    """
    N = X.shape[0]
    mean_features = X.mean(dim=0)
    diff = X - mean_features.expand(N, -1)
    return torch.norm(diff, p="fro").item()


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="cell_graph_transformer",
)
def main(cfg: DictConfig) -> None:
    """Main training function for overfitting test."""
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    from torchcell.timestamp import timestamp
    from torchcell.scratch.load_batch_006_perturbation import (
        load_perturbation_batch,
    )
    from torchcell.losses.logcosh import LogCoshLoss
    from scipy.stats import gaussian_kde
    from scipy import stats

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    DATA_ROOT = os.getenv("DATA_ROOT")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() == "gpu"
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # Load data
    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)

    dataset, batch, cell_graph, gene_set_size = load_perturbation_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        subset_size=cfg.data_module.perturbation_subset_size,
        device=device,
    )

    cell_graph = cell_graph.to(device)
    batch = batch.to(device)

    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing model...")
    print("=" * 80)

    # Extract gene-gene edge types from cell_graph
    print(f"\nCell graph edge types:")
    for edge_type in cell_graph.edge_types:
        src, rel, dst = edge_type
        if src == "gene" and dst == "gene":
            print(f"  {edge_type}: {cell_graph[edge_type].num_edges} edges")

    model = CellGraphTransformer(
        gene_num=cfg.model.gene_num,
        hidden_channels=cfg.model.hidden_channels,
        num_transformer_layers=cfg.model.num_transformer_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        cell_graph=cell_graph,
        graph_regularization_config=cfg.model.graph_regularization,
        perturbation_head_config=cfg.model.perturbation_head,
        dropout=cfg.model.dropout,
        adaptive_loss_weighting=cfg.model.get("adaptive_loss_weighting", False),
        graph_reg_scale=cfg.model.get("graph_reg_scale", 0.001),
    ).to(device)

    print("\nModel architecture:")
    print(model)
    param_counts = model.num_parameters
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # Setup loss and optimizer
    criterion = LogCoshLoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.regression_task.optimizer.lr,
        weight_decay=cfg.regression_task.optimizer.weight_decay,
    )

    # Learning rate scheduler (optional)
    lr_scheduler = None
    if (
        hasattr(cfg.regression_task, "lr_scheduler")
        and cfg.regression_task.lr_scheduler is not None
    ):
        from torchcell.scheduler.cosine_annealing_warmup import (
            CosineAnnealingWarmupRestarts,
        )

        scheduler_config = cfg.regression_task.lr_scheduler
        if scheduler_config.type == "CosineAnnealingWarmupRestarts":
            lr_scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=scheduler_config.first_cycle_steps,
                cycle_mult=scheduler_config.get("cycle_mult", 1.0),
                max_lr=scheduler_config.max_lr,
                min_lr=scheduler_config.min_lr,
                warmup_steps=scheduler_config.warmup_steps,
                gamma=scheduler_config.get("gamma", 1.0),
            )
            print("Using CosineAnnealingWarmupRestarts scheduler")
        else:
            print(f"Warning: Unknown scheduler type {scheduler_config.type}")
    else:
        print("Using constant learning rate (no scheduler)")

    # Training target
    y = batch["gene"].phenotype_values.to(device)

    # Setup directory for plots
    plot_dir = osp.join(ASSET_IMAGES_DIR, f"cell_graph_transformer_{timestamp()}")
    os.makedirs(plot_dir, exist_ok=True)

    def save_intermediate_plot(
        epoch,
        losses,
        pred_losses,
        graph_reg_losses,
        correlations,
        spearman_correlations,
        mses,
        maes,
        rmses,
        learning_rates,
        weight_l2_norms,
        smoothness_history,
        cfg,
        model,
        cell_graph,
        batch,
        y,
    ):
        """Save intermediate training plot every print interval."""
        fig = plt.figure(figsize=(20, 12))

        # ROW 1: Total Loss, Prediction Loss, Graph Reg Loss
        plt.subplot(3, 4, 1)
        plt.plot(range(1, epoch + 2), losses, "b-", label="Total Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Total Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale("log")

        plt.subplot(3, 4, 2)
        plt.plot(
            range(1, epoch + 2),
            pred_losses,
            "orange",
            label="Prediction Loss",
            linewidth=2,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Prediction Loss (LogCosh)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale("log")

        plt.subplot(3, 4, 3)
        plt.plot(
            range(1, epoch + 2),
            graph_reg_losses,
            "green",
            label="Graph Reg Loss",
            linewidth=2,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Graph Regularization Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale("log")

        # ROW 1 cont: Correlations
        plt.subplot(3, 4, 4)
        plt.plot(
            range(1, epoch + 2),
            correlations,
            "g-",
            label="Pearson",
            linewidth=2,
        )
        if spearman_correlations:
            plt.plot(
                range(1, epoch + 2),
                spearman_correlations,
                "b--",
                label="Spearman",
                linewidth=2,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Correlation")
        plt.title("Correlation Evolution")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)

        # ROW 2: Error Metrics
        plt.subplot(3, 4, 5)
        epochs_range = range(1, epoch + 2)
        ax1 = plt.gca()
        ax1.plot(epochs_range, mses, "r-", label="MSE", linewidth=2)
        ax1.plot(epochs_range, rmses, "b-", label="RMSE", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE / RMSE")
        ax1.set_yscale("log")
        ax1.tick_params(axis="y")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(epochs_range, maes, "orange", label="MAE", linewidth=2)
        ax2.set_ylabel("MAE", color="orange")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="orange")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.set_title("Error Metrics Evolution")

        # Get current predictions for visualization
        model.eval()
        with torch.no_grad():
            current_predictions, _ = model(cell_graph, batch)
            true_np = y.cpu().numpy()
            pred_np = current_predictions.squeeze().cpu().numpy()

            valid_mask = ~np.isnan(true_np)
            if np.sum(valid_mask) > 0:
                pred_std = np.std(pred_np[valid_mask])
                true_std = np.std(true_np[valid_mask])

                if pred_std < 1e-8 or true_std < 1e-8:
                    current_corr = 0.0
                else:
                    try:
                        corr_matrix = np.corrcoef(
                            pred_np[valid_mask], true_np[valid_mask]
                        )
                        current_corr = corr_matrix[0, 1]
                        if np.isnan(current_corr):
                            current_corr = 0.0
                    except:
                        current_corr = 0.0
            else:
                current_corr = 0.0
        model.train()

        # Scatter plot
        plt.subplot(3, 4, 6)
        plt.scatter(pred_np[valid_mask], true_np[valid_mask], alpha=0.7)
        min_val = min(true_np[valid_mask].min(), pred_np[valid_mask].min())
        max_val = max(true_np[valid_mask].max(), pred_np[valid_mask].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Predictions vs Truth (r={current_corr:.4f})")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Distribution comparison with KDE
        plt.subplot(3, 4, 7)
        bins = np.linspace(
            min(true_np[valid_mask].min(), pred_np[valid_mask].min()),
            max(true_np[valid_mask].max(), pred_np[valid_mask].max()),
            30,
        )
        plt.hist(
            true_np[valid_mask],
            bins=bins,
            alpha=0.5,
            label="True",
            color="blue",
            density=True,
        )
        plt.hist(
            pred_np[valid_mask],
            bins=bins,
            alpha=0.5,
            label="Predicted",
            color="red",
            density=True,
        )

        # Add KDE
        if len(true_np[valid_mask]) > 1:
            try:
                kde_true = gaussian_kde(true_np[valid_mask])
                kde_pred = gaussian_kde(pred_np[valid_mask])
                x_range = np.linspace(
                    true_np[valid_mask].min(), true_np[valid_mask].max(), 200
                )
                plt.plot(x_range, kde_true(x_range), "b-", linewidth=2, label="True KDE")
                plt.plot(
                    x_range, kde_pred(x_range), "r-", linewidth=2, label="Pred KDE"
                )
            except:
                pass

        plt.xlabel("Gene Interaction Score")
        plt.ylabel("Density")
        plt.title("Value Distributions")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Learning rate
        plt.subplot(3, 4, 8)
        plt.plot(range(1, epoch + 2), learning_rates, "purple", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        # ROW 3: Model Configuration
        plt.subplot(3, 4, 9)
        plt.title("Model Configuration", pad=20)
        param_counts = model.num_parameters
        total_params = param_counts["total"]

        y_pos = 0.92
        params_text = [
            f"Total Parameters: {total_params:,}",
            f"Hidden Channels: {cfg.model.hidden_channels}",
            f"Transformer Layers: {cfg.model.num_transformer_layers}",
            f"Attention Heads: {cfg.model.num_attention_heads}",
            f"Dropout: {cfg.model.dropout}",
            f"Graph Reg Scale: {cfg.model.graph_reg_scale}",
            f"Weight Decay: {cfg.regression_task.optimizer.weight_decay}",
            f"Learning Rate: {cfg.regression_task.optimizer.lr}",
            f"Batch Size: {cfg.data_module.batch_size}",
        ]

        for i, text in enumerate(params_text):
            plt.text(
                0.05,
                y_pos - i * 0.09,
                text,
                transform=plt.gca().transAxes,
                fontsize=9,
                ha="left",
                va="top",
            )

        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        # L2 Norm with Weighted Losses
        plt.subplot(3, 4, 10)
        epochs_range = range(1, len(weight_l2_norms) + 1)
        if len(weight_l2_norms) >= len(pred_losses):
            l2_norms = weight_l2_norms[: len(pred_losses)]
            weight_decay = cfg.regression_task.optimizer.weight_decay
            l2_penalty = [norm * weight_decay for norm in l2_norms]

            plt.plot(
                epochs_range, pred_losses, "b-", label="Pred Loss", linewidth=2
            )
            plt.plot(
                epochs_range,
                graph_reg_losses,
                "g-",
                label="Graph Reg Loss",
                linewidth=2,
            )
            plt.plot(
                epochs_range,
                l2_penalty,
                "purple",
                label=f"L2 Penalty (wd={weight_decay})",
                linewidth=2,
                linestyle="--",
            )

            plt.xlabel("Epoch")
            plt.ylabel("Loss Component")
            plt.title("Loss Components with L2 Norm")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.yscale("log")

        # Error histogram
        plt.subplot(3, 4, 11)
        errors = pred_np[valid_mask] - true_np[valid_mask]
        plt.hist(errors, bins=30, alpha=0.7, edgecolor="black", color="purple")
        plt.axvline(x=0, color="r", linestyle="--", linewidth=2)
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title(f"Error Distribution (μ={np.mean(errors):.4f}, σ={np.std(errors):.4f})")
        plt.grid(True, alpha=0.3)

        # Smoothness evolution (oversmoothing diagnostic)
        plt.subplot(3, 4, 12)
        if smoothness_history:
            epochs_range = range(1, len(smoothness_history) + 1)
            plt.plot(epochs_range, smoothness_history, "darkorange", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Smoothness (Frobenius Norm)")
            plt.title("Gene Embedding Smoothness\n↓ Lower = Oversmoothing")
            plt.grid(True, alpha=0.3)

            # Add current value annotation
            current_smoothness = smoothness_history[-1]
            plt.text(
                0.95,
                0.95,
                f"Current: {current_smoothness:.2f}",
                transform=plt.gca().transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Add warning zone if smoothness is very low (< 10% of initial)
            if len(smoothness_history) > 1:
                initial_smoothness = smoothness_history[0]
                if current_smoothness < 0.1 * initial_smoothness:
                    plt.axhline(
                        y=0.1 * initial_smoothness,
                        color="red",
                        linestyle="--",
                        linewidth=1,
                        alpha=0.5,
                    )
                    plt.text(
                        0.05,
                        0.15,
                        "⚠ Oversmoothing",
                        transform=plt.gca().transAxes,
                        color="red",
                        fontsize=10,
                        weight="bold",
                    )

        plt.suptitle(
            f"Cell Graph Transformer Training - Epoch {epoch + 1}/{cfg.trainer.max_epochs}",
            fontsize=16,
            y=0.998,
        )

        plt.tight_layout()
        plt.savefig(
            osp.join(plot_dir, f"training_epoch_{epoch+1:04d}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    print(f"Batch size: {y.size(0)}")
    print(f"Max epochs: {cfg.trainer.max_epochs}")
    print(f"Plot directory: {plot_dir}")

    # Training loop - Initialize tracking lists
    losses = []
    pred_losses = []
    graph_reg_losses = []
    correlations = []
    spearman_correlations = []
    mses = []
    maes = []
    rmses = []
    learning_rates = []
    weight_l2_norms = []
    smoothness_history = []

    plot_interval = cfg.regression_task.plot_every_n_epochs

    for epoch in range(cfg.trainer.max_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions, representations = model(cell_graph, batch)

        # Compute smoothness of gene embeddings (oversmoothing diagnostic)
        with torch.no_grad():
            H_genes = representations["H_genes"]  # [N, d]
            smoothness = compute_smoothness(H_genes)
            smoothness_history.append(smoothness)

        # Compute losses
        pred_loss = criterion(predictions.squeeze(), y)
        graph_reg_loss = representations["graph_reg_loss"]

        # Apply adaptive weighting if enabled
        if model.adaptive_loss_weighting:
            mse_weight = model.log_mse_weight.exp()
            reg_weight = model.log_reg_weight.exp()
            # Normalize weights
            weight_sum = mse_weight + reg_weight
            mse_weight = 2 * mse_weight / weight_sum
            reg_weight = 2 * reg_weight / weight_sum
            total_loss = mse_weight * pred_loss + reg_weight * graph_reg_loss
        else:
            total_loss = pred_loss + graph_reg_loss

        # Compute metrics before backward pass
        with torch.no_grad():
            pred_np = predictions.squeeze().cpu().numpy()
            y_np = y.cpu().numpy()
            valid_mask = ~np.isnan(y_np)

            if np.sum(valid_mask) > 0:
                pred_std = np.std(pred_np[valid_mask])
                y_std = np.std(y_np[valid_mask])

                # Pearson correlation
                if pred_std < 1e-8 or y_std < 1e-8:
                    corr = 0.0
                    spearman_corr = 0.0
                else:
                    try:
                        corr = np.corrcoef(pred_np[valid_mask], y_np[valid_mask])[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                    except:
                        corr = 0.0

                    try:
                        spearman_corr, _ = stats.spearmanr(
                            pred_np[valid_mask], y_np[valid_mask]
                        )
                        if np.isnan(spearman_corr):
                            spearman_corr = 0.0
                    except:
                        spearman_corr = 0.0

                # Error metrics
                mse = np.mean((pred_np[valid_mask] - y_np[valid_mask]) ** 2)
                mae = np.mean(np.abs(pred_np[valid_mask] - y_np[valid_mask]))
                rmse = np.sqrt(mse)
            else:
                corr = 0.0
                spearman_corr = 0.0
                mse = float("inf")
                mae = float("inf")
                rmse = float("inf")

        # Track metrics
        losses.append(total_loss.item())
        pred_losses.append(pred_loss.item())
        graph_reg_losses.append(graph_reg_loss.item())
        correlations.append(corr)
        spearman_correlations.append(spearman_corr)
        mses.append(mse)
        maes.append(mae)
        rmses.append(rmse)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # Calculate L2 norm
        l2_norm = calculate_weight_l2_norm(model)
        weight_l2_norms.append(l2_norm)

        # Save intermediate plot
        if epoch % plot_interval == 0 or epoch == cfg.trainer.max_epochs - 1:
            save_intermediate_plot(
                epoch,
                losses,
                pred_losses,
                graph_reg_losses,
                correlations,
                spearman_correlations,
                mses,
                maes,
                rmses,
                learning_rates,
                weight_l2_norms,
                smoothness_history,
                cfg,
                model,
                cell_graph,
                batch,
                y,
            )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if cfg.regression_task.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.regression_task.clip_grad_norm_max_norm
            )

        optimizer.step()

        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Print progress
        if epoch % plot_interval == 0:
            print(
                f"Epoch {epoch:4d}: "
                f"Loss={total_loss.item():.4f}, "
                f"Pred={pred_loss.item():.4f}, "
                f"GraphReg={graph_reg_loss.item():.4f}, "
                f"Corr={corr:.4f}, "
                f"Spearman={spearman_corr:.4f}, "
                f"MSE={mse:.4f}, "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

    # Final evaluation
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Final Pearson correlation: {correlations[-1]:.4f}")
    print(f"Final Spearman correlation: {spearman_correlations[-1]:.4f}")
    print(f"Final total loss: {losses[-1]:.4f}")
    print(f"Final prediction loss: {pred_losses[-1]:.4f}")
    print(f"Final graph reg loss: {graph_reg_losses[-1]:.4f}")
    print(f"Final MSE: {mses[-1]:.4f}")
    print(f"Final MAE: {maes[-1]:.4f}")
    print(f"Final RMSE: {rmses[-1]:.4f}")
    print(f"Final L2 norm: {weight_l2_norms[-1]:.4f}")
    print(f"Final smoothness: {smoothness_history[-1]:.2f}")
    print(f"\nAll training plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
