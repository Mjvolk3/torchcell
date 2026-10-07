# torchcell/models/cell_graph_transformer
# [[torchcell.models.cell_graph_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/cell_graph_transformer
# Test file: tests/torchcell/models/test_cell_graph_transformer.py

"""Cell Graph Transformer with graph-regularized attention heads.

Implements the architecture from weekly report 2025.45:
- CLS token for whole-cell representation
- Graph-regularized attention heads (KL loss to adjacency matrices)
- Perturbation head with cross-attention for gene interaction prediction
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class GraphRegularizedTransformerLayer(nn.Module):
    """Transformer layer with graph-regularized attention heads.

    Uses manual attention computation to get both output and attention weights
    for graph regularization loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        adjacency_matrices: dict[str, torch.Tensor] | None = None,
        regularized_head_config: dict[str, dict[str, Any]] | None = None,
        dropout: float = 0.1,
    ):
        """Build multi-head attention projections and graph-regularized heads."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with manual attention computation.

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
        gene_attention = attention_weights[:, :, 1:, 1:] if return_attention else None

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
    """Hypergraph Self-Attention Graph Neural Network for perturbation sets.

    Adapted from DANGO model to compute perturbation representations via
    masked self-attention within perturbation sets.
    """

    def __init__(self, hidden_channels: int, num_heads: int = 4):
        """Build the hyper self-attention projections for the given head config."""
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        assert hidden_channels % num_heads == 0, (
            f"hidden_channels {hidden_channels} must be divisible by num_heads {num_heads}"
        )

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
        """Forward pass processing perturbed genes with masked attention.

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

        set_representations: torch.Tensor = scatter_mean(
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
        """Apply global masked multi-head attention.

        Args:
            x: Input tensor with shape [total_nodes, hidden_dim]
            attention_mask: Binary mask with shape [total_nodes, total_nodes]
                           True where attention is allowed, False elsewhere
            Q_proj: Linear projection producing queries.
            K_proj: Linear projection producing keys.
            V_proj: Linear projection producing values.
            O_proj: Linear projection applied to the attention output.
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
        out_proj: torch.Tensor = O_proj(out)

        # Apply ReZero connection
        return beta * out_proj + x


class PerturbationHead(nn.Module):
    """Perturbation head with switchable attention mechanisms.

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
        """Build the perturbation head using cross-attention or HyperSAGNN."""
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
        """Forward pass of perturbation head.

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
        predictions: torch.Tensor = self.mlp(combined)  # [batch_size, 1]

        return predictions


class CellGraphTransformer(nn.Module):
    """Cell Graph Transformer model.

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
        graph_regularization_config: dict[str, Any] | None = None,
        perturbation_head_config: dict[str, Any] | None = None,
        dropout: float = 0.1,
        adaptive_loss_weighting: bool = False,
        graph_reg_scale: float = 0.001,  # Global scale factor for graph reg
    ):
        """Build the embedding, graph-regularized transformer stack, and head."""
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
        # Graph regularization is enabled when graph_reg_scale > 0
        self.adjacency_matrices: dict[str, torch.Tensor] | None
        if self.graph_reg_scale > 0.0 and graph_regularization_config is not None:
            # Normalize adjacency matrices from cell_graph
            self.adjacency_matrices = self._normalize_adjacency_matrices(cell_graph)
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
    ) -> dict[str, torch.Tensor]:
        """Normalize adjacency matrices row-wise: A_tilde[i,:] = A[i,:] / (degree[i] + eps).

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
        """Compute graph regularization loss using KL divergence.

        Args:
            attention_weights: [batch, heads, N, N] gene-gene attention weights
            layer_idx: Current transformer layer index

        Returns:
            Total regularization loss for this layer
        """
        # Early return if graph regularization is disabled (scale is 0)
        if self.graph_reg_scale == 0.0:
            return torch.tensor(0.0, device=attention_weights.device)

        # When graph reg is enabled (scale > 0), adjacency matrices are populated.
        assert self.adjacency_matrices is not None

        total_loss = torch.tensor(0.0, device=attention_weights.device)
        batch_size, num_heads, N, _ = attention_weights.shape

        for graph_name, config in self.regularized_head_config.items():
            # Handle both single layer (int) and multiple layers (list)
            layer_config = config["layer"]
            layers = [layer_config] if isinstance(layer_config, int) else layer_config
            if layer_idx not in layers:
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
                (
                    alpha[:, sample_idx, :] + 1e-8
                ).log(),  # log predictions with epsilon for numerical stability
                A_tilde[sample_idx, :]
                .unsqueeze(0)
                .expand(batch_size, -1, -1),  # targets
                reduction="batchmean",
                log_target=False,
            )

            total_loss = total_loss + lambda_k * kl_loss

        # Apply global scale factor and normalize by number of edges
        total_edges = sum(
            len(self.adjacency_matrices[g].nonzero()[0])
            for g in self.adjacency_matrices.keys()
        )
        if total_edges > 0:
            total_loss = (
                total_loss * self.graph_reg_scale / (total_edges / self.gene_num)
            )

        return total_loss

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass of Cell Graph Transformer.

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

            if attention_weights is not None:
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
    def num_parameters(self) -> dict[str, int]:
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
    return float(np.sqrt(l2_norm))


def compute_smoothness(X: torch.Tensor) -> float:
    """Compute smoothness of node features (oversmoothing diagnostic).

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
    return float(torch.norm(diff, p="fro").item())


if __name__ == "__main__":
    raise SystemExit(
        "the demo main moved to torchcell/scratch/cell_graph_transformer_demo.py on 2026-10-06"
    )
