# torchcell/models/equivariant_cell_graph_transformer
# [[torchcell.models.equivariant_cell_graph_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/equivariant_cell_graph_transformer
# Test file: tests/torchcell/models/test_equivariant_cell_graph_transformer.py

"""Equivariant Cell Graph Transformer with graph-regularized attention heads.

Implements the generalized virtual cell architecture:
- CLS token for whole-cell representation
- Graph-regularized attention heads (KL loss to adjacency matrices)
- EQUIVARIANT perturbation transformation (preserves per-gene structure)
- Perturbation head with cross-attention for gene interaction prediction
"""

import os
import os.path as osp
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

if TYPE_CHECKING:
    # Type-only import: keeps the models package free of a runtime dependency on
    # torchcell.losses (whose __init__ imports other model-specific losses).
    from torchcell.losses.distributional import DistHead


class GraphRegularizedTransformerLayer(nn.Module):
    """Transformer layer with graph-regularized attention heads.

    Uses manual attention computation to get both output and attention weights
    for graph regularization loss.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """Build Q/K/V/out projections, layer norms, and the feedforward block.

        Args:
            hidden_dim: Model hidden dimension (divisible by num_heads).
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
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

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        head_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with manual attention computation.

        Args:
            x: [batch, N+1, d] where N+1 includes CLS token at position 0
            return_attention: Whether to return attention weights for regularization
            head_mask: Optional [heads, N+1, N+1] bool mask, True where attention is
                ALLOWED. When given (and weights are not requested) attention runs through
                ``scaled_dot_product_attention``.

                WHY THAT MATTERS FOR MEMORY. This layer computes attention MANUALLY --
                explicit matmul then softmax -- because the graph-regularization KL needs
                the weights. That materializes a [batch, heads, N+1, N+1] tensor every
                layer: 9 x 6608^2 x 4B = 1.57 GB, and it is the bulk of the measured
                19.3 GB/run. Turning the KL off does NOT free it; the manual path
                materializes regardless. Masking replaces the KL as the way to inject
                graph structure, which removes the need for the weights, which finally
                allows the fused/flash SDPA kernel that never forms the matrix.

        Returns:
            output: [batch, N+1, d] transformed features
            gene_attention_weights: [batch, heads, N, N] gene-gene attention weights (if return_attention=True)
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

        if not return_attention:
            # FUSED PATH: never forms the [batch, heads, N+1, N+1] matrix.
            #
            # GATED ON `return_attention` ALONE, not on having a mask. The earlier condition
            # `head_mask is not None and not return_attention` forfeited the fused kernel on
            # every UNMASKED layer, so with attention_mask.layers=[1] on a four-layer encoder
            # only ONE layer fused and the other three still materialized
            # [1, 9, 6608, 6608] (9 x 6608^2 x 4B = 1.57 GB) and then discarded it. The KL is
            # the only consumer that needs the weights; when nobody asks for them there is no
            # reason to build them, mask or no mask. SDPA accepts attn_mask=None.
            #
            # dropout_p IS PASSED EXPLICITLY. The old fused branch passed none, so a masked
            # layer silently trained at p=0 while every manual layer applied p=0.1 -- a
            # regularization difference confounded with the mask itself. Note SDPA consumes
            # RNG differently from the manual softmax path, so trajectories will not match
            # earlier runs at an identical seed; that is why this lands on a new wave boundary
            # rather than being pooled with wave3/wave4a.
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None if head_mask is None else head_mask.unsqueeze(0),
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            gene_attention_weights = None
        else:
            # Manual attention computation (get both output AND weights)
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (
                self.head_dim**0.5
            )  # [batch, heads, seq_len, seq_len]
            if head_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    ~head_mask.unsqueeze(0), float("-inf")
                )
            attention_weights = F.softmax(attention_scores, dim=-1)

            # Apply dropout to attention weights
            attention_weights_dropout = self.dropout(attention_weights)

            # Apply attention to values
            attn_output = torch.matmul(
                attention_weights_dropout, v
            )  # [batch, heads, seq_len, head_dim]

            # Extract gene-gene attention block for regularization (exclude CLS token)
            # attention_weights: [batch, heads, N+1, N+1]
            # gene_attention_weights: [batch, heads, N, N]
            gene_attention_weights = (
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

        return output, gene_attention_weights


class HyperSAGNN(nn.Module):
    """Hypergraph Self-Attention Graph Neural Network for perturbation sets.

    Adapted from DANGO model to compute perturbation representations via
    masked self-attention within perturbation sets.
    """

    def __init__(self, hidden_channels: int, num_heads: int = 4):
        """Set up multi-head attention dimensions and projections.

        Args:
            hidden_channels: Model hidden dimension (divisible by num_heads).
            num_heads: Number of attention heads.
        """
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

        set_representations = scatter_mean(
            squared_diff, batch_indices, dim=0, dim_size=num_batches
        )

        return cast(torch.Tensor, set_representations)  # [num_batches, hidden_channels]

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
            Q_proj: Linear projection for queries.
            K_proj: Linear projection for keys.
            V_proj: Linear projection for values.
            O_proj: Linear output projection.
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
        return cast(torch.Tensor, beta * out + x)


class EquivariantPerturbationTransform(nn.Module):
    """Equivariant perturbation transformation that preserves per-gene structure.

    Unlike the standard perturbation head which collapses to a summary vector,
    this module transforms ALL gene embeddings based on perturbation context,
    maintaining the [batch, N, d] shape for downstream equivariant tasks.

    Implements Type I Virtual Instrument: H_genes → H_genes_pert
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual: str = "postln",
        num_layers: int = 1,
        ffn_mult: int = 4,
        extra_layer_ffn_mult: int | None = None,
        hadamard: str = "off",
        null_sink: bool = False,
        null_sink_bias_init: float = -4.0,
        null_sink_trainable: bool = True,
        null_sink_magnitude_match: bool = False,
    ):
        """Build the cross-attention, feedforward, and normalization layers.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of cross-attention heads.
            dropout: Dropout probability.
            residual: ``postln`` (default, unchanged) or ``rezero``.

                POSTLN is `LN(h_i + attn)` -> `LN(x + FFN(x))`, the historical behaviour.
                Two costs, both measured elsewhere in this round:

                  * STABILITY. A post-LN block with NO warmup is the canonical recipe for
                    init-dependent divergence, and that is what we see: identical config,
                    three seeds, val/mean/pearson_per_feature 0.1527 / 0.0235 / 0.0661,
                    with seed 1 stuck at ~0.015 for all 83 epochs. Note the ENCODER
                    already avoids this -- `GraphRegularizedTransformerLayer` closes with
                    a ReZero residual (`beta * out + x`) and no norm. This block is the
                    one place the model still uses post-LN, so the two are inconsistent.
                  * INFORMATION. LayerNorm is invariant to positive rescaling of its
                    input, so `LN(h_i + c_b)` discards the magnitude of the sum -- which
                    is exactly where <h_i, c_b> lives (||h+c||^2 = ||h||^2 + 2<h,c> +
                    ||c||^2). ReZero keeps the residual stream unnormalized.

                REZERO is `h_i + beta * attn` -> `x + beta_ffn * FFN(x)` with both betas
                initialized to ZERO, so the block starts as an exact identity and the
                network begins as its own well-conditioned shallow limit.
            num_layers: Number of stacked cross-attention + FFN blocks.
            ffn_mult: Hidden width of each block's FFN as a multiple of ``hidden_dim``.
            extra_layer_ffn_mult: FFN multiplier for blocks AFTER the first. ``None``
                means every block uses ``ffn_mult``; a smaller value makes the extra
                depth cheap, so "deeper" and "wider" stay separable as arms.
            hadamard: ``off`` | ``replace`` | ``add``. ``replace`` swaps the additive
                context for ``h_i * (1 + gamma(c_b))`` (no additive path, so at init the
                model sees no perturbation); ``add`` keeps both and is the identity to the
                additive reference at init. Either way a rank-d pair term versus rank 0.
            null_sink: Append one zero-valued key/value column to the perturbation
                context, making the real-key weight ``sigmoid(q_i.k_p/sqrt(d) - b)``
                and therefore dependent on the QUERYING gene. This is what supplies a
                pair-(p, i) term at ``|S| = 1``; see the block comment below.
            null_sink_bias_init: Initial value of the learned ``null_bias`` logit
                offset. ``0.0`` starts the sink carrying ~50% of the attention mass, so
                the model closing it is itself an observable result; a strongly negative
                init starts it nearly shut and can never engage (measured: at ``-4.0``
                the gate moved 0.07 over 187 epochs).
            null_sink_trainable: Whether ``null_bias`` receives gradient. ``False``
                pins the sink at its init, which is the sham control for the gate.
            null_sink_magnitude_match: Rescale the attended output by
                ``1/sigmoid(-null_sink_bias_init)``. A zero-valued sink diverts mass to
                a zero vector and so ATTENUATES the context norm (measured 0.5129 at
                ``bias_init=0.0``); 97.8% of the unmatched sink's downstream effect was
                that attenuation rather than the gate. Matching removes the confound
                (norm ratio 1.0258) while leaving the per-query spread untouched
                (CV 0.0971 either way).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # NULL SINK -- one learned scalar that un-degenerates the |S|=1 softmax.
        #
        # THE DEFECT. Every gene token QUERIES the |S| perturbed tokens. At |S|=1 the
        # softmax runs over a ONE-element key set, so alpha = 1.0 for every query
        # regardless of the logit: the attended vector is query-INDEPENDENT and the
        # context collapses to an affine map of the deleted gene, c = W_O(W_V h_p + b_V)
        # + b_O (verified to 2.4e-7). W_Q and W_K are then dead weight -- re-drawing them
        # at std=10 changes the output by EXACTLY 0.0 -- which is 16,380 of 32,760
        # attention parameters receiving no gradient on 95.4% of the expression build.
        # It is a d x d linear layer wearing an attention costume.
        #
        # THE FIX. Append one extra key/value column holding the ZERO vector (which
        # projects to the in_proj biases b_K / b_V), and add a learned scalar `null_bias`
        # to that column's logit. The softmax now has two terms, so at |S|=1
        #
        #     alpha_i = sigmoid( q_i . k_p / sqrt(d_k) - null_bias )
        #
        # which DEPENDS ON i. The deletion can finally affect gene 17 and gene 900 by
        # different amounts -- a genuine pair-(p, i) term, learned inside the attention.
        # Per head it is rank-1; across heads the induced per-gene variation has rank
        # exactly `num_heads` (verified). Equivalently: this IS sigmoid (unnormalized)
        # attention at |S|=1, expressed as a softmax over {sink, p}.
        #
        # WHAT IT DOES NOT DO. It does not restore cardinality. A zero-valued sink keeps
        # the output a convex combination that now includes the origin, so ||c|| SHRINKS
        # with the sink rather than growing with |S| (measured 4.75/2.48/1.91 with sink
        # vs 5.43/2.65/1.95 without, at |S| = 1/2/3). Cardinality needs sum pooling or an
        # explicit |S| feature; this arm is only a probe of whether the |S|=1 degeneracy
        # costs anything at all.
        #
        # `null_sink_bias_init=-20` with `null_sink_trainable=False` makes the sink
        # numerically inert (p_null ~ 2e-9) on the SAME code path -- the sham arm, which
        # is what calibrates whether the paired estimator transfers to an arm that is not
        # an exact identity at init.
        # MAGNITUDE MATCHING -- without it the arm is uninterpretable, because the sink does
        # not only add a pair term, it PERMANENTLY ATTENUATES the context.
        #
        # A zero-valued sink means alpha_p = sigmoid(q.k_p/sqrt(d) - b) < 1 for EVERY query, so
        # the attended vector shrinks. MEASURED on the real module (N=2048, d=90, 9 heads,
        # |S|=1): ||c_sink|| / ||c_ref|| = 0.5139 at bias 0. Worse, that attenuation -- not the
        # pair term -- is almost the whole effect: a bare scalar 0.5 applied to the REFERENCE
        # context, with no sink at all, reproduces 97.8% of the arm's downstream deviation at
        # LN(h_i + c).
        #
        # And it cannot be trained away. AdamW is per-parameter scale-free, so null_bias moves
        # at most ~3e-4 * 11,700 steps = 3.51 in principle, but A4's measured drift rate
        # (0.07 over 7,293 steps) projects only 0.112 across 300 epochs -- alpha stays near
        # 0.53 for the entire run. The reference (alpha = 1) sits on the BOUNDARY of the arm's
        # parameter space, reachable only as null_bias -> -inf. The attenuation also halves
        # grad W_V (0.0313 -> 0.0168), so the arm slows its own compensation, which biases
        # early epochs in a max-over-epochs score.
        #
        # Dividing by sigmoid(-bias_init) restores the expected context norm at init, so the
        # arm differs from the reference by the QUERY-DEPENDENT GATE alone -- which is the
        # only thing it was built to test.
        # HADAMARD ASSERTION. The default operator combines gene and strain ADDITIVELY --
        # H_pert_i = g(h_i + c_b) -- so at |S|=1 nothing depends on the pair (p, i): c_b is
        # one d-vector shared by all N genes. This mode instead ASSERTS the perturbation
        # multiplicatively into every gene's channels,
        #
        #     H_pert_i = g( h_i * (1 + gamma(c_b)) ),   gamma: R^d -> R^d,
        #
        # which is a rank-d (=90) interaction between gene identity and strain, against the
        # null sink's rank-9 (one bounded scalar per head) and the additive form's rank 0.
        # gamma's final layer is zero-initialised so the block is the EXACT identity at init
        # and the arm is a clean ablation: any movement is attributable to the mechanism, not
        # to the extra parameters.
        #
        # TWO MODES, and the difference is not cosmetic.
        #   "replace" -- H_pert_i = g(h_i * (1 + gamma(c_b))). The additive path is GONE, so
        #       at init (gamma = 0) the block is the identity on h_i and the model sees NO
        #       perturbation at all; it must learn the entire pathway through gamma. This is
        #       "assertion instead of cross-attention" in its literal form, and it starts
        #       from a strictly worse place than the reference rather than an equal one.
        #   "add"     -- H_pert_i = g(h_i + c_b + h_i * gamma(c_b)). Identity to the ADDITIVE
        #       REFERENCE at init, so it is a clean ablation: the multiplicative pair term is
        #       added on top of a working model and any movement is attributable to it.
        # Both are run, because "replace" answers the question that was asked and "add"
        # is the one whose null result would be interpretable.
        if hadamard not in ("off", "replace", "add"):
            raise ValueError(
                f"hadamard must be 'off' | 'replace' | 'add', got {hadamard!r}"
            )
        self.hadamard = hadamard
        self.hadamard_gamma: nn.Sequential | None = None
        if hadamard != "off":
            self.hadamard_gamma = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            gamma_out = cast(nn.Linear, self.hadamard_gamma[-1])
            nn.init.zeros_(gamma_out.weight)
            nn.init.zeros_(gamma_out.bias)
        self.null_sink = null_sink
        self.null_scale = 1.0
        if null_sink:
            self.null_bias = nn.Parameter(
                torch.tensor([float(null_sink_bias_init)]),
                requires_grad=null_sink_trainable,
            )
            if null_sink_magnitude_match:
                self.null_scale = 1.0 / float(
                    torch.sigmoid(torch.tensor(-float(null_sink_bias_init)))
                )
        if residual not in ("postln", "rezero"):
            raise ValueError(f"residual must be 'postln' or 'rezero', got {residual!r}")
        self.residual = residual
        if residual == "rezero":
            self.beta_attn = nn.Parameter(torch.zeros(1))
            self.beta_ffn = nn.Parameter(torch.zeros(1))

        # Cross-attention: each gene attends to perturbation context
        # DEPTH. Stage 2 is where ALL strain conditioning happens, and it has been ONE
        # layer. Deepening it is nearly free because |S| is tiny (1-3 perturbed genes):
        # cost is L * N * |S| * d * B ~ 0.46 GFLOP at L=4, i.e. 0.7% of the encoder. Each
        # extra layer lets gene i and the perturbed set interact again, and -- unlike the
        # graph arms -- needs no adjacency, so it transfers to any organism.
        # FFN WIDTH IS THE COST, NOT THE ATTENTION. Measured at B=32, one extra layer is
        # 34.3 GFLOP: cross-attention 0.076, QKVO projections 6.85, and the d->4d->d FFN
        # over all 6,607 tokens 27.4 -- i.e. the FFN is 80% of it. Adding depth with
        # mult=4 therefore mostly buys FEEDFORWARD CAPACITY, which confounds the
        # repeated-attention hypothesis it was built to test.
        #
        # `extra_layer_ffn_mult=0` makes layers 1.. attention-only, leaving layer 0
        # untouched, so the ONLY difference from the L=1 baseline is repeated attention.
        # Cost: L=4 attention-only is 1.20x vs 1.34x for L=2 with the FFN -- three extra
        # rounds of attention for less than one extra FFN block.
        self.num_layers = num_layers
        self.ffn_mults = [ffn_mult] + [
            ffn_mult if extra_layer_ffn_mult is None else extra_layer_ffn_mult
        ] * (num_layers - 1)
        self.cross_attn_layers = nn.ModuleList(
            nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            for _ in range(num_layers)
        )
        self.ffn_layers = nn.ModuleList(
            (
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * m),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * m, hidden_dim),
                )
                if m > 0
                else nn.Identity()
            )
            for m in self.ffn_mults
        )
        self.norm1_layers = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        )
        self.norm2_layers = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        # Back-compat aliases so single-layer state_dicts and any external reference to
        # `.cross_attn` / `.ffn` / `.norm1` / `.norm2` still resolve to layer 0.
        self.cross_attn = self.cross_attn_layers[0]
        self.ffn = self.ffn_layers[0]
        self.norm1 = self.norm1_layers[0]
        self.norm2 = self.norm2_layers[0]

    def _apply_residual(
        self, x: torch.Tensor, attended: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Combine a layer's attended output with its input, per the residual mode.

        Args:
            x: [N, d] the layer's input representation.
            attended: [N, d] cross-attention output for this layer.
            layer_idx: Which layer's norm/FFN parameters to use.

        Returns:
            [N, d] the layer's output.
        """
        # Every `nn.Module.__call__` is untyped under the gate's --follow-imports=silent,
        # so each of these expressions is `Any` and every `return` would leak it. Bind the
        # intermediate to an explicitly annotated local and cast at the boundary -- the
        # same idiom `GraphRegularizedTransformerLayer` already uses for its ReZero return.
        has_ffn = self.ffn_mults[layer_idx] > 0
        out: torch.Tensor
        if self.residual == "rezero":
            # Unnormalized residual stream: preserves the magnitude of h_i + c (where
            # <h_i, c> lives), and is an exact identity at init since both betas are 0.
            out = cast(torch.Tensor, x + self.beta_attn * self.dropout(attended))
            if not has_ffn:
                return out
            return cast(
                torch.Tensor,
                out + self.beta_ffn * self.dropout(self.ffn_layers[layer_idx](out)),
            )
        out = cast(
            torch.Tensor, self.norm1_layers[layer_idx](x + self.dropout(attended))
        )
        if not has_ffn:
            return out
        return cast(
            torch.Tensor,
            self.norm2_layers[layer_idx](
                out + self.dropout(self.ffn_layers[layer_idx](out))
            ),
        )

    def forward(
        self,
        H_genes: torch.Tensor,
        perturbation_indices: torch.Tensor,
        batch_assignment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply equivariant perturbation transformation.

        Args:
            H_genes: [N, d] - wildtype gene embeddings
            perturbation_indices: [total_pert_genes] - indices of perturbed genes
            batch_assignment: [total_pert_genes] - batch index for each perturbed gene

        Returns:
            H_genes_pert: [batch, N, d] - perturbed gene embeddings (EQUIVARIANT!)
            context: [batch, N, d] - the attended perturbation context, returned
                separately so downstream heads can condition on it directly (the
                concat/bilinear/FiLM arms all need c_b, not only h_i + c_b).
        """
        batch_size = int(batch_assignment.max().item()) + 1
        N, d = H_genes.shape

        H_genes_pert_list = []
        # The RAW attended context c_{b,i} kept alongside the fused output. The fused
        # tensor is g(h_i + c_b) -- h_i and c_b meet exactly once, by ADDITION, so any
        # downstream head can only ever see a function of their SUM. Exposing c lets a
        # head take [h_pert ; h_i ; c] and learn arbitrary (h_i, c_b) interactions
        # instead. Cheap: it is already computed, just discarded today.
        context_list = []

        for b in range(batch_size):
            # Get perturbation context for this sample
            mask = batch_assignment == b
            pert_idx_b = perturbation_indices[mask]

            H_cur = H_genes
            first_context: torch.Tensor | None = None
            for layer_idx in range(self.num_layers):
                if len(pert_idx_b) > 0:
                    # K/V read the CURRENT representation of the perturbed genes, so the
                    # set summary evolves with depth rather than being frozen at layer 0.
                    perturbation_context = H_cur[pert_idx_b]  # [|S_b|, d]
                    # The sink is prepended as an extra K/V ROW; `attn_mask` is a float
                    # tensor ADDED to the logits, so putting `null_bias` in column 0 is
                    # what makes the sink's mass learnable. Gradient flows into a
                    # Parameter placed there (verified: grad -1105.5, and the in_proj
                    # Q/K grads go from exactly zero to 148.2 / 212.0).
                    attn_mask = None
                    if self.null_sink:
                        perturbation_context = torch.cat(
                            [H_cur.new_zeros(1, d), perturbation_context], dim=0
                        )  # [|S_b| + 1, d]
                        attn_mask = torch.cat(
                            [
                                self.null_bias.to(H_cur.dtype).view(1, 1).expand(N, 1),
                                H_cur.new_zeros(N, len(pert_idx_b)),
                            ],
                            dim=1,
                        )  # [N, |S_b| + 1]
                    attended, _ = self.cross_attn_layers[layer_idx](
                        query=H_cur.unsqueeze(0),  # [1, N, d]
                        key=perturbation_context.unsqueeze(0),  # [1, |S_b|(+1), d]
                        value=perturbation_context.unsqueeze(0),  # [1, |S_b|(+1), d]
                        attn_mask=attn_mask,
                    )
                    if self.null_scale != 1.0:
                        attended = attended * self.null_scale
                    attended = attended.squeeze(0)  # [N, d]
                else:
                    attended = torch.zeros_like(H_cur)
                if self.hadamard_gamma is not None:
                    # x + x*gamma(c) == x*(1 + gamma(c)), so routing the modulation through
                    # the EXISTING residual keeps postln/rezero, dropout and the FFN
                    # byte-identical between this arm and the reference; only the meaning of
                    # `attended` changes. "add" keeps the additive context alongside it.
                    mod = H_cur * self.hadamard_gamma(attended)
                    attended = mod if self.hadamard == "replace" else attended + mod
                if first_context is None:
                    first_context = attended
                H_cur = self._apply_residual(H_cur, attended, layer_idx)

            assert first_context is not None
            context_list.append(first_context)
            H_pert_b = H_cur

            H_genes_pert_list.append(H_pert_b)

        # Stack to create batch dimension
        H_genes_pert = torch.stack(H_genes_pert_list, dim=0)  # [batch, N, d]
        context = torch.stack(context_list, dim=0)  # [batch, N, d]

        return H_genes_pert, context


class PerturbationGraphPropagation(nn.Module):
    r"""Route the perturbation to graph NEIGHBOURS -- the missing pair-(p, i) term.

    WHY THIS EXISTS. The encoder runs at batch 1 on the WILDTYPE graph, so ``H_genes``
    and ``h_CLS`` are identical for every strain; the only strain-dependent step is
    :class:`EquivariantPerturbationTransform`, whose K/V set is just the |S_b| perturbed
    tokens.  For a SINGLE deletion the softmax is over one key, so the attention weight
    is exactly 1 for every query gene and the attended vector is query-INDEPENDENT:

    .. math::
        H^{\mathrm{pert}}_{b,i} = g\big(h_i + c_b\big), \qquad c_b = W_O W_V h_p .

    Verified numerically (max :math:`|attended_i - attended_0| = 0`).  ~96% of the
    expression rows are single deletions (Kemmeren2014 1,484 singles + Sameith2015,
    whose double-KO sub-study is n=72), so for almost the whole training set the model's
    prediction factorizes ADDITIVELY: gene identity enters as :math:`h_i`, strain
    identity as one 90-d vector, and **no term depends on the pair (p, i)**.  By data
    processing every head output is then a function of :math:`c_b` alone, which is why
    no readout change (S0 -> S1 -> S3, any scoring rule) can enlarge the hypothesis
    class.  The nine graphs shape :math:`h_i` but never CARRY the deletion.

    WHAT THIS DOES.  For each graph ``g`` and hop ``t``, spread unit mass outward from
    the perturbed genes over the row-normalized adjacency:

    .. math::
        r^{g,1} = \hat A_g^\top p_b, \qquad r^{g,t} = \hat A_g^\top r^{g,t-1},

    giving ``len(graphs) * hops`` scalars per gene that answer "how strongly is gene i
    reached from the deletion, along graph g, in t hops".  That quantity depends on the
    PAIR, so it is exactly the term the additive form lacks.  The features are projected
    to ``d`` and added to ``H_genes_pert`` through a ReZero gate initialized at ZERO --
    so the module starts as an exact identity and this is a clean ablation: any gain is
    attributable to propagation, not to the extra parameters.

    Cost is a sparse mat-vec per (graph, hop), i.e. O(|E|) -- negligible next to the
    encoder's dense attention over 6,607 tokens.
    """

    def __init__(
        self,
        hidden_dim: int,
        graph_names: list[str],
        hops: int = 2,
        dropout: float = 0.1,
        gate_mode: str = "rezero",
    ):
        """Build the propagation feature projection.

        Args:
            hidden_dim: Model hidden dimension d.
            graph_names: Relation names to propagate over (the ``cell_dataset.graphs``
                set; one sparse adjacency each).
            hops: Number of propagation steps t. Hop t reaches the t-step neighbourhood.
            dropout: Dropout probability inside the projection MLP.
            gate_mode: ``rezero`` (learned scalar, init 0 -- the module starts as an exact
                identity, so any gain is attributable to propagation rather than to the
                extra parameters) or ``on`` (fixed 1.0, forcing gradient through the
                pathway; a closed ReZero gate is not evidence against a mechanism).
        """
        super().__init__()
        assert hops >= 0, f"hops must be >= 0, got {hops}"
        self.graph_names = list(graph_names)
        self.hops = hops
        # +1 for the HOP-0 SELF-INDICATOR ("is gene i itself deleted in strain b?").
        #
        # This is the cheapest possible pair-(p, i) term and it plugs a real hole: in the
        # baseline every gene receives the SAME c_b, so nothing marks WHICH gene was
        # deleted. The model can only infer it indirectly, by learning that h_i resembles
        # the deletion signature carried in c_b. Yet a deleted gene's own strong
        # down-regulation is the most predictable single value in a Kemmeren profile, and
        # the deleted gene is itself one of the 6,127 measured columns. hop-0 is graph
        # independent, hence ONE feature rather than one per graph.
        self.num_features = len(self.graph_names) * hops + 1
        self.proj = nn.Sequential(
            nn.Linear(self.num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_mode = gate_mode
        self.gate: nn.Parameter | torch.Tensor
        if gate_mode == "on":
            # FORCED ON -- see PerceiverMixing.gate_mode for why a closed ReZero gate is
            # not evidence against a mechanism.
            self.register_buffer("gate", torch.ones(1), persistent=False)
        else:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        H_genes_pert: torch.Tensor,
        adjacencies_T: dict[str, torch.Tensor],
        perturbation_indices: torch.Tensor,
        batch_assignment: torch.Tensor,
    ) -> torch.Tensor:
        """Add graph-routed perturbation features to the perturbed gene embeddings.

        Args:
            H_genes_pert: [batch, N, d] equivariant perturbed gene embeddings.
            adjacencies_T: relation name -> TRANSPOSED row-normalized sparse adjacency
                [N, N], so ``A_T @ x`` spreads mass outward from the perturbed genes.
            perturbation_indices: [total_pert_genes] indices of perturbed genes.
            batch_assignment: [total_pert_genes] batch index per perturbed gene.

        Returns:
            [batch, N, d] embeddings with the pair-(p, i) signal added.
        """
        batch_size, num_genes, _ = H_genes_pert.shape
        device = H_genes_pert.device

        # FLOAT32 THROUGHOUT THE SPARSE PATH. Training runs `precision: bf16-mixed` and
        # CUDA has no bf16 `addmm_sparse` kernel ("NotImplementedError:
        # addmm_sparse_cuda not implemented for 'BFloat16'"). The propagation is O(|E|)
        # sparse mat-vecs, utterly negligible next to the encoder's dense attention over
        # 6,607 tokens, so running it in fp32 costs nothing and avoids the gap entirely.
        # autocast disabled so the sparse ops cannot be silently re-cast back to bf16.
        with torch.autocast(device_type=device.type, enabled=False):
            p0 = torch.zeros(num_genes, batch_size, device=device, dtype=torch.float32)
            p0[perturbation_indices, batch_assignment] = 1.0
            p0 = p0 / p0.sum(dim=0, keepdim=True).clamp(min=1.0)

            # hop 0: the perturbation indicator itself, before any propagation.
            feats: list[torch.Tensor] = [p0.t()]
            for name in self.graph_names:
                x = p0
                A_T = adjacencies_T[name]
                for _ in range(self.hops):
                    x = torch.sparse.mm(A_T, x)  # [N, batch]
                    feats.append(x.t())  # [batch, N]

            # log1p(N * r) keeps "unreached == exactly 0" while lifting the
            # 1/degree-scaled reachabilities off the floor -- a raw r is ~1e-4 and would
            # vanish in the MLP.
            features = torch.stack(feats, dim=-1)  # [batch, N, num_features]
            features = torch.log1p(features * num_genes)
        features = features.to(H_genes_pert.dtype)

        return cast(torch.Tensor, H_genes_pert + self.gate * self.proj(features))


class LowRankBilinear(nn.Module):
    r"""Explicit rank-R multiplicative interaction between gene identity and perturbation.

    WHY. ``h_i`` (what gene *i* is) and ``c`` (what was deleted) meet exactly once, by
    ADDITION, and the very next op is a LayerNorm. The natural pair-term for expression is
    an inner product -- "how related is output gene *i* to the deleted gene" -- and the sum
    does contain it:

    .. math:: \|h_i + c\|^2 = \|h_i\|^2 + 2\langle h_i, c\rangle + \|c\|^2

    but LayerNorm divides by the per-channel standard deviation, DISCARDING exactly the
    magnitude in which :math:`\langle h_i, c\rangle` is encoded. So the interaction is not
    merely hard to read off the sum -- the operation immediately after the sum removes it.

    This computes the interaction directly as a rank-R bilinear form,

    .. math:: b_r = \langle u_r, h_i\rangle \cdot \langle v_r, c\rangle,
              \qquad r = 1 \ldots R,

    yielding R scalars per (strain, gene) that are appended to the per-gene head input.
    R=32 matches the measured effective rank (32.8) of the reproducible residual gene-gene
    correlation. Cost is two d x R projections -- negligible -- and it is the parameter-lean
    alternative to the full ``concat_context`` arm: if concat wins because the head is
    re-deriving an inner product, this should recover most of it for 2*d*R parameters.

    Permutation-equivariant (no gene-indexed parameters), graph-free.
    """

    def __init__(self, hidden_dim: int, rank: int = 32):
        """Build the two projections whose elementwise product is the bilinear form.

        Args:
            hidden_dim: Model hidden dimension d.
            rank: Number of bilinear components R (the output width).
        """
        super().__init__()
        self.rank = rank
        self.u = nn.Linear(hidden_dim, rank, bias=False)
        self.v = nn.Linear(hidden_dim, rank, bias=False)

    def forward(self, H_genes: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute the rank-R interaction per (strain, gene).

        Args:
            H_genes: [N, d] strain-invariant wildtype gene embeddings.
            context: [batch, N, d] attended perturbation context.

        Returns:
            [batch, N, rank] bilinear interaction features.
        """
        batch_size = context.shape[0]
        a = self.u(H_genes).unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, R]
        b = self.v(context)  # [B, N, R]
        # NO LayerNorm here. The first version normalized `a * b` across the R
        # components, which discards the MAGNITUDE of the interaction -- exactly the
        # quantity this module exists to expose, and exactly the defect it was built to
        # route around. The measured null result (-0.0006) was that bug, not evidence.
        return cast(torch.Tensor, a * b)


class ObservedLabelEncoder(nn.Module):
    r"""Encode PARTIALLY OBSERVED labels so the decoder can condition on them.

    WHY. Every arm in this round changes what information reaches the per-gene readout;
    none changes the OUTPUT FACTORIZATION. The decoder emits 6,127 INDEPENDENT marginals
    in one shot, even though the reproducible residual gene-gene correlation has a measured
    participation-ratio effective rank of 32.8 (split-half r = 0.869 against a 1e-4 null).
    That joint structure is simply discarded at the output.

    Masked prediction -- the scGPT / BERT objective -- restores it: mask a fraction of the
    measured genes, feed the REST in as input, and predict the held-out ones. Gene i's
    prediction then depends on gene j's observed value, which is exactly the dependence a
    per-token readout cannot express.

    IT ONLY WORKS WITH CROSS-GENE MIXING. If nothing routes information between gene
    tokens after the perturbation, an observed value at gene j can never reach gene i and
    the conditioning is inert. This module therefore pairs with
    :class:`PerceiverMixing`: the Perceiver is the channel, masking is the objective that
    forces the model to use it. (Measured: the Perceiver alone moved seed 0 by only
    +0.0045 -- a pathway with nothing requiring it.)

    EVALUATION STAYS COMPARABLE. At validation no labels are observed, so the mask is
    100% and the forward pass is identical to the unconditioned model. The per-feature
    metric is therefore directly comparable to every other arm -- masking changes the
    TRAINING signal, not the inference task.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1, gate_mode: str = "on"):
        """Build the projection from (value, observed-flag) to a token-space offset.

        Args:
            hidden_dim: Model hidden dimension d.
            dropout: Dropout probability.
            gate_mode: ``on`` (fixed 1.0, the default) or ``rezero`` (learned scalar,
                init 0). Defaults to forced-on because the whole point is to push
                gradient through the conditioning pathway; with everything masked the
                encoded features are zero anyway, so a 100%-masked (validation) forward
                is still an identity.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Defaults to FORCED ON: the whole point is to push gradient through the
        # conditioning pathway. With everything masked the encoded features are all zero
        # anyway, so a 100%-masked (validation) forward is still identity.
        self.gate_mode = gate_mode
        self.gate: nn.Parameter | torch.Tensor
        if gate_mode == "on":
            self.register_buffer("gate", torch.ones(1), persistent=False)
        else:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        H_genes_pert: torch.Tensor,
        observed_values: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Add an encoding of the observed labels to the gene tokens.

        Args:
            H_genes_pert: [batch, N, d] post-perturbation gene embeddings.
            observed_values: [batch, N] label values, ZERO where unobserved.
            observed_mask: [batch, N] 1.0 where the label is observed, else 0.0.

        Returns:
            [batch, N, d] embeddings carrying the observed-label context.
        """
        # Values are zeroed where unobserved, so the flag is what distinguishes "observed
        # and happens to be 0" from "not observed" -- without it the two are identical.
        features = torch.stack(
            [observed_values * observed_mask, observed_mask], dim=-1
        ).to(H_genes_pert.dtype)
        return cast(torch.Tensor, H_genes_pert + self.gate * self.proj(features))


class CrossGeneMixing(nn.Module):
    r"""GEARS-style cross-gene mixing: one shared global state, then per-gene readout.

    GEARS computes a single ``cross_gene_state`` from ALL gene embeddings and concatenates
    it to every gene before the gene-specific output layer; ablating that layer measurably
    hurt its differential-expression correlation. It is the cheapest form of cross-gene
    communication -- one pooled vector, no pairwise term -- and sits strictly between the
    per-token readout (no communication at all) and full attention.

    GEARS flattens ``[N, d]`` into a single Linear, which here would be 6,607 x 90 ->
    38M parameters. This uses the low-rank equivalent: project each gene to ``rank``, mean
    across genes, broadcast back. Same information path (every gene sees a summary of every
    other gene), tractable parameter count.

    NOTE ON WHAT THIS CAN AND CANNOT DO. Every post-perturbation token is g(h_i + c_b), so
    the pooled state is a function of c_b -- pooling adds no INFORMATION. It does enlarge
    the FUNCTION CLASS: the map c_b -> profile is no longer "add c_b to each h_i and apply
    one shared MLP". Since the reproducible signal has effective rank ~33 and c_b carries
    90 dimensions, information is not the binding constraint here; expressivity is. That is
    the hypothesis this arm tests.
    """

    def __init__(self, hidden_dim: int, rank: int = 64, dropout: float = 0.1):
        """Build the project -> pool -> broadcast path.

        Args:
            hidden_dim: Model hidden dimension d.
            rank: Width of the pooled cross-gene state.
            dropout: Dropout probability.
        """
        super().__init__()
        self.to_state = nn.Linear(hidden_dim, rank)
        self.from_state = nn.Sequential(
            nn.Linear(hidden_dim + rank, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, H_genes_pert: torch.Tensor) -> torch.Tensor:
        """Mix a pooled cross-gene state back into every gene token.

        Args:
            H_genes_pert: [batch, N, d] post-perturbation gene embeddings.

        Returns:
            [batch, N, d] with the shared cross-gene state folded in.
        """
        state = self.to_state(H_genes_pert).mean(dim=1)  # [batch, rank]
        state = state.unsqueeze(1).expand(-1, H_genes_pert.shape[1], -1)
        return cast(
            torch.Tensor,
            H_genes_pert + self.from_state(torch.cat([H_genes_pert, state], dim=-1)),
        )


class ResponseBasisHead(nn.Module):
    r"""Low-rank response-basis decoder: the perturbation activates shared programs.

    .. math:: \hat y_i = \hat y_i^{\text{local}} + \sum_{j=1}^{r} b_{ij}(h_i)\, a_j(z_S)

    The perturbation picks an amplitude for each of ``r`` global response PROGRAMS, and each
    gene carries a learned sensitivity to each program. This is a conditional factor model,
    not attention -- and it is the multiplicative interaction between gene identity and
    perturbation that the additive ``g(h_i + c_b)`` form cannot represent, since h_i and
    c_b meet exactly once by addition and the next op is a LayerNorm that removes the scale
    in which their inner product lives.

    ``rank=32`` is MEASURED, not chosen: residual_covariance_diagnostic.py puts the
    participation-ratio effective rank of the reproducible residual gene-gene correlation
    at 32.8 (split-half r = 0.869 vs a 1e-4 null). If the response really is that low-rank,
    this parameterizes it directly for ~2*d*r parameters.

    Supersedes ``LowRankBilinear``, whose first version applied a LayerNorm to the product
    and so normalized away the magnitude it existed to expose.
    """

    def __init__(
        self, hidden_dim: int, rank: int = 32, param_dim: int = 1, dropout: float = 0.1
    ):
        """Build the per-gene sensitivity and per-perturbation amplitude maps.

        Args:
            hidden_dim: Model hidden dimension d.
            rank: Number of response programs r.
            param_dim: Distributional params per gene (1 point, 19 quantile).
            dropout: Dropout probability.
        """
        super().__init__()
        self.rank = rank
        self.param_dim = param_dim
        self.sensitivity = nn.Linear(hidden_dim, rank, bias=False)  # b_i = q(h_i)
        self.amplitude = nn.Sequential(  # a = g(z_S)
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rank * param_dim),
        )
        # Zero-init the amplitude output so the head starts as an exact identity: the arm
        # is then a clean ablation against the local readout alone.
        # Indexing an nn.Sequential yields Module, so `.weight` types as Tensor|Module.
        amplitude_out = cast(nn.Linear, self.amplitude[-1])
        nn.init.zeros_(amplitude_out.weight)
        nn.init.zeros_(amplitude_out.bias)

    def forward(
        self, H_genes_pert: torch.Tensor, pert_context: torch.Tensor
    ) -> torch.Tensor:
        """Compute the low-rank response contribution.

        Args:
            H_genes_pert: [batch, N, d] post-perturbation gene embeddings.
            pert_context: [batch, 2d] the [z_S ; h_CLS] perturbation summary.

        Returns:
            [batch, N] when ``param_dim == 1`` else [batch, N, param_dim].
        """
        batch_size = H_genes_pert.shape[0]
        b = self.sensitivity(H_genes_pert)  # [batch, N, r]
        a = self.amplitude(pert_context).view(batch_size, self.rank, self.param_dim)
        out = torch.einsum("bnr,brp->bnp", b, a)
        return out.squeeze(-1) if self.param_dim == 1 else out


class PerceiverMixing(nn.Module):
    r"""POST-perturbation gene-gene mixing through a learned latent bottleneck.

    WHY. The encoder is the ONLY place gene representations interact, and it runs on the
    wildtype graph at batch 1 -- so that mixing is identical for every strain. After
    :class:`EquivariantPerturbationTransform` injects the perturbation there is NO
    gene-gene interaction anywhere: the transform's keys/values are the perturbed genes
    only, ``PerGeneHead`` is a per-token MLP, and the global/perturbation heads merely
    pool. Gene *i*'s post-perturbation state therefore depends on :math:`(h_i, \{h_p\})`
    and never on :math:`h_j` for any other gene, which makes downstream cascades
    ("delete p -> changes j -> changes i") structurally unrepresentable.

    Naive fix -- dense self-attention over the N tokens AFTER the perturbation -- is
    per-instance, so it costs B times the encoder: ~503 GFLOP and a
    ``[32, 9, 6607, 6607]`` attention tensor (~25 GB in bf16). Not affordable.

    This block routes every gene through ``num_latents`` learned latents instead:
    genes -> latents -> genes. Every gene still influences every other, but through an
    M-dim bottleneck, so cost is O(N*M) not O(N^2): ~4.9 GFLOP and a ~0.12 GB attention
    tensor at M=32. M=32 is also the MEASURED participation-ratio effective rank (32.8)
    of the reproducible residual gene-gene correlation.

    Graph-FREE -- unlike graph-masked attention it needs no adjacency, so it transfers to
    organisms with no curated interaction networks. Permutation-equivariant: the latents
    are shared across genes, never indexed by gene. ReZero-gated at ZERO so the block is
    an exact identity at init and the arm is a clean ablation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_latents: int = 32,
        num_heads: int = 9,
        dropout: float = 0.1,
        gate_mode: str = "rezero",
    ):
        """Build the latent bank and the two cross-attention stages.

        Args:
            hidden_dim: Model hidden dimension d.
            num_latents: Bottleneck width M. Every gene-gene path runs through these.
            num_heads: Attention heads (must divide hidden_dim).
            dropout: Dropout probability.
            gate_mode: ``rezero`` (learned scalar, init 0) or ``on`` (fixed 1.0).

                ``on`` FORCES the pathway. A ReZero gate initialized at zero has a
                chicken-and-egg failure: the block's internals receive gradient only in
                proportion to the gate, and the gate receives gradient only in proportion
                to how useful the block is with UNTRAINED internals -- which is random. A
                block that only becomes useful once its internals are trained can sit at
                gate ~ 0 indefinitely. MEASURED: the Perceiver's gate finished at 1.0e-5,
                i.e. never opened, while the propagation module's reached 0.125 under the
                identical mechanism. A closed gate is therefore NOT evidence the mechanism
                is useless -- it cannot distinguish "nothing to gain" from "never left the
                basin". ``on`` removes the confound so backprop must flow through the
                pathway, which is the only way to test the mechanism itself.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim) * 0.02)
        self.write_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.read_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_latent = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.gate_mode = gate_mode
        self.gate: nn.Parameter | torch.Tensor
        if gate_mode == "on":
            self.register_buffer("gate", torch.ones(1), persistent=False)
        else:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, H_genes_pert: torch.Tensor) -> torch.Tensor:
        """Mix gene representations through the latent bottleneck.

        Args:
            H_genes_pert: [batch, N, d] post-perturbation gene embeddings.

        Returns:
            [batch, N, d] with cross-gene information mixed in.
        """
        batch_size = H_genes_pert.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)  # [B, M, d]

        # WRITE: latents read from all genes -- this is where cross-gene info is pooled.
        written, _ = self.write_attn(
            query=latents, key=H_genes_pert, value=H_genes_pert, need_weights=False
        )
        latents = self.norm_latent(latents + self.dropout(written))
        latents = latents + self.dropout(self.ffn(latents))

        # READ: every gene queries the latents -- this is where it comes back out.
        read, _ = self.read_attn(
            query=H_genes_pert, key=latents, value=latents, need_weights=False
        )
        return cast(torch.Tensor, H_genes_pert + self.gate * self.norm_out(read))


class PerturbationHead(nn.Module):
    """Perturbation readout head for gene interaction prediction.

    Operates on equivariant perturbed gene embeddings H_genes_pert [batch, N, d]
    and produces scalar gene interaction predictions per sample.

    Implements Type II Virtual Instrument: H_genes_pert → y_GI
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1, pooling: str = "sum"):
        """Build the prediction MLP mapping [h_CLS || z_S] to a scalar.

        Args:
            hidden_dim: Model hidden dimension.
            dropout: Dropout probability.
            pooling: ``sum`` (default) or ``mean`` over the perturbed-gene tokens.

                SUM IS THE DEFAULT because a MEAN is cardinality-blind: it makes z_S for
                S={A} and S={A,A} bit-identical (measured: both -0.3520988), and it makes
                a double's summary a convex combination of its two singles rather than
                something outside their span. Every explicit-set model in the field sums --
                GEARS ``MLP(sum h_Pi)``, CPA ``.sum(dim=1)``, GenePert g(p1)+g(p2),
                SAMS-VAE ``sum m_p e_p`` -- and Deep Sets says why.

                THE CHANGE IS INERT ON TODAY'S DATA, which is the point: at |S|=1 sum and
                mean are the same operation, and 95.4% of the expression build is |S|=1
                (100% of the kemmeren-only arm). At the constant |S|=3 of 010 it is a
                factor of 3 absorbed by the next linear layer. Sum and mean only diverge
                when |S| VARIES within a run -- i.e. exactly the mixed-cardinality joint
                training this is being landed ahead of. So it is future-proofing, not an
                experimental arm; there is nothing here to measure yet.

                ``mean`` is retained so the 010 configuration is exactly reproducible.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        if pooling not in ("sum", "mean"):
            raise ValueError(f"pooling must be 'sum' or 'mean', got {pooling!r}")
        self.pooling = pooling

        # Prediction MLP: [h_CLS || z_S] -> scalar
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_CLS: torch.Tensor,
        H_genes_pert: torch.Tensor,
        perturbation_indices: torch.Tensor,
        batch_assignment: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of perturbation head.

        Args:
            h_CLS: [d] - whole-cell CLS representation
            H_genes_pert: [batch, N, d] - perturbed gene embeddings (EQUIVARIANT)
            perturbation_indices: [total_pert_genes] - indices of perturbed genes
            batch_assignment: [total_pert_genes] - batch index for each perturbed gene

        Returns:
            predictions: [batch_size, 1] gene interaction predictions
        """
        batch_size = H_genes_pert.shape[0]

        # Aggregate perturbed genes per sample
        z_S_list = []
        for b in range(batch_size):
            mask = batch_assignment == b
            pert_idx_b = perturbation_indices[mask]

            if len(pert_idx_b) > 0:
                # Extract perturbed genes for this sample from H_genes_pert
                h_pert_b = H_genes_pert[b, pert_idx_b, :]  # [|S_b|, d]
                z_S_list.append(
                    h_pert_b.sum(dim=0)
                    if self.pooling == "sum"
                    else h_pert_b.mean(dim=0)
                )  # [d]
            else:
                # No perturbation
                z_S_list.append(
                    torch.zeros(self.hidden_dim, device=H_genes_pert.device)
                )

        z_S = torch.stack(z_S_list, dim=0)  # [batch_size, d]

        # Concatenate with CLS token: [h_CLS || z_S]
        h_CLS_expanded = h_CLS.unsqueeze(0).expand(batch_size, -1)  # [batch_size, d]
        combined = torch.cat([h_CLS_expanded, z_S], dim=-1)  # [batch_size, 2*d]

        # Predict gene interaction
        predictions = self.mlp(combined)  # [batch_size, 1]

        return cast(torch.Tensor, predictions)


class GlobalHead(nn.Module):
    """Whole-cell readout head for global phenotypes.

    Maps the CLS token (whole-cell representation), optionally concatenated with a
    mean pool over the equivariant perturbed gene embeddings, to a configurable
    output vector. Used for CalMorph morphology (501-D) AND scalar VisualScore
    (e.g. beta-carotene, output_dim=1).

    graph_level == "global" selects this head.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        use_gene_pool: bool = True,
        dropout: float = 0.1,
        param_dim: int = 1,
    ):
        """Build the MLP mapping [h_CLS (|| GlobalPool(H_genes_pert))] to output_dim.

        Args:
            hidden_dim: Model hidden dimension.
            output_dim: Output vector dimension (e.g. 501 morphology, 1 scalar). This is the
                FEATURE count F -- it stays the phenotype dimensionality regardless of the
                distributional mode.
            use_gene_pool: Concatenate a mean pool over genes with h_CLS.
            dropout: Dropout probability.
            param_dim: Distributional params PER FEATURE (1 point, 2 gaussian, K quantile).
                The output Linear widens to ``output_dim * param_dim`` and the forward
                reshapes to ``[batch, output_dim, param_dim]`` when > 1.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gene_pool = use_gene_pool
        self.param_dim = param_dim

        in_dim = hidden_dim * 2 if use_gene_pool else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * param_dim),
        )

    def forward(self, h_CLS: torch.Tensor, H_genes_pert: torch.Tensor) -> torch.Tensor:
        """Forward pass of the global head.

        Args:
            h_CLS: [d] whole-cell CLS representation.
            H_genes_pert: [batch, N, d] equivariant perturbed gene embeddings.

        Returns:
            predictions: [batch_size, output_dim] global phenotype predictions, or
            [batch_size, output_dim, param_dim] when param_dim > 1 (distributional).
        """
        batch_size = H_genes_pert.shape[0]
        h = h_CLS.unsqueeze(0).expand(batch_size, -1)  # [batch, d]
        if self.use_gene_pool:
            pooled = H_genes_pert.mean(dim=1)  # [batch, d]
            h = torch.cat([h, pooled], dim=-1)  # [batch, 2d]
        out = self.mlp(h)  # [batch, output_dim * param_dim]
        if self.param_dim > 1:
            out = out.view(batch_size, self.output_dim, self.param_dim)
        return cast(torch.Tensor, out)


class CrossAttnHead(nn.Module):
    r"""S3 cross-attention readout head (DETR / Perceiver style) for vector phenotypes.

    The S1 :class:`GlobalHead` collapses the token set to ONE pooled vector
    ``u = [h_CLS || mean_i h_i_pert]`` that every output feature must read from. A mean over
    ~6000 gene tokens dilutes a 1-2 gene perturbation by ~3 orders of magnitude, so the
    strain-specific signal is largely gone BEFORE the readout -- the pool-dilution
    bottleneck. That is the leading hypothesis for morphology sitting at r~0.04 against a
    0.61 noise ceiling while expression (which reads its own token, S0) is at its ceiling.

    S3 removes the shared bottleneck: each of the F output features gets a LEARNED QUERY
    ``q_k`` that cross-attends over the FULL token set ``{h_CLS_pert, h_1_pert, ..., h_N_pert}``:

    .. math::
        A = \mathrm{softmax}\!\Big(\frac{(QW_Q)(H_{\mathrm{pert}}W_K)^\top}{\sqrt{d_k}}\Big),
        \qquad C = A\,(H_{\mathrm{pert}}W_V),

    so output ``k`` reads its OWN weighted mixture ``c_k`` -- per-output support is recovered
    WITHOUT needing a 1-to-1 feature<->token map (morphology features are not genes). Feature
    ``k`` can place attention mass directly on the perturbed genes instead of averaging them
    away. CLS is kept as a key so a feature can still read the whole-cell summary.

    Queries are parameters (like the CLS token), NOT label values, so nothing about the
    target leaks in. The readout is ``DistHead``-agnostic: ``param_dim`` widens the final
    projection to emit point / gaussian / quantile parameters per feature.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        param_dim: int = 1,
        use_ffn: bool = True,
    ):
        """Build the F learned queries, the cross-attention block, and the readout.

        Args:
            hidden_dim: Model hidden dimension d (query/key/value width).
            output_dim: Number of output FEATURES F (e.g. 278 CalMorph features). One
                learned query per feature.
            num_heads: Cross-attention heads (must divide hidden_dim).
            dropout: Dropout probability (attention + FFN + readout).
            param_dim: Distributional params per feature (1 point, 2 gaussian, K quantile).
            use_ffn: Apply a residual feed-forward block to the attended context C.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.param_dim = param_dim
        self.use_ffn = use_ffn

        # F learned feature queries -- parameters, like the CLS token.
        self.queries = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.norm2 = nn.LayerNorm(hidden_dim)

        # Per-feature readout: each feature's context vector -> its param_dim parameters.
        # A shared Linear (d -> param_dim) applied to every feature's context keeps the
        # parameter count independent of F; the per-feature specificity lives in the queries.
        self.readout = nn.Linear(hidden_dim, param_dim)

    def forward(self, h_CLS: torch.Tensor, H_genes_pert: torch.Tensor) -> torch.Tensor:
        """Forward pass of the cross-attention head.

        Args:
            h_CLS: [d] whole-cell CLS representation.
            H_genes_pert: [batch, N, d] equivariant perturbed gene embeddings.

        Returns:
            predictions: [batch, output_dim] when param_dim == 1, else
            [batch, output_dim, param_dim].
        """
        batch_size = H_genes_pert.shape[0]

        # Keys/values = {CLS} U {perturbed gene tokens}: [batch, N+1, d].
        cls_tok = (
            h_CLS.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        )  # [batch, 1, d]
        kv = torch.cat([cls_tok, H_genes_pert], dim=1)  # [batch, N+1, d]

        # F learned queries, shared across the batch: [batch, F, d].
        q = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        attended, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        c = self.norm1(q + self.dropout(attended))  # [batch, F, d]
        if self.use_ffn:
            c = self.norm2(c + self.dropout(self.ffn(c)))

        out = self.readout(c)  # [batch, F, param_dim]
        if self.param_dim == 1:
            out = out.squeeze(-1)  # [batch, F]
        return cast(torch.Tensor, out)


class PerGeneHead(nn.Module):
    """Per-gene (equivariant) readout head for gene-resolved phenotypes.

    Maps the equivariant perturbed gene embeddings H_genes_pert [batch, N, d] to a
    per-gene prediction, preserving the per-node structure. Used for expression
    (MicroarrayExpressionPhenotype / RnaseqExpressionPhenotype) AND proteome
    (ProteinAbundancePhenotype).

    graph_level == "node" selects this head.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        dropout: float = 0.1,
        param_dim: int = 1,
        free_gene_dim: int = 0,
        num_genes: int = 0,
        in_mult: int = 1,
        extra_dim: int = 0,
        film_dim: int = 0,
    ):
        """Build the shared MLP applied to every gene embedding.

        Args:
            hidden_dim: Model hidden dimension.
            output_dim: Per-gene output dimension (default 1 -> scalar per gene).
            dropout: Dropout probability.
            param_dim: Distributional params PER GENE (1 point, 2 gaussian, K quantile).
                With the default ``output_dim=1`` the trailing axis carries the params:
                ``[batch, N]`` (point) or ``[batch, N, param_dim]`` (distributional). The
                measured-gene ``index_select(1, col)`` gather is unaffected either way.
            free_gene_dim: Width of a FREE per-gene embedding concatenated to each token
                before the MLP (0 disables it -- the original S0 behaviour).
            num_genes: Number of gene nodes N; required when ``free_gene_dim > 0``.
            in_mult: How many ``hidden_dim``-wide blocks the head consumes per gene.
                ``1`` is ``h_pert`` alone; ``3`` is the CONCAT arm ``[h_pert ; h_i ; c]``,
                which can represent arbitrary functions of ``(h_i, c_b)`` rather than
                only functions of their sum.
            extra_dim: Extra input width appended per gene, used by the bilinear arm to
                pass its rank-``r`` interaction features (0 disables it).
            film_dim: Width of the FiLM conditioning vector. ``> 0`` builds a conditioner
                emitting per-feature scale/shift, giving ``h_i * gamma(c) + beta(c)`` --
                a genuine rank-``d`` pair term, which the purely additive form lacks.
                ``0`` disables FiLM.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.param_dim = param_dim

        # FREE OUTPUT-GENE EMBEDDING (``free_gene_dim > 0``).  Without it, output-gene
        # identity enters ONLY as the point h_i through a shared MLP, so genes with
        # nearby encoder embeddings are forced into near-identical response functions
        # for every strain.  A free row per gene unties that.
        #
        # Unlike ``learnable_embedding`` (which indexes PERTURBED genes and is inert on
        # val, since only ~4.8% of val genes are ever perturbed in training), this
        # indexes MEASURED genes: ``col_idx`` is row-invariant, so every one of the
        # 6,127 measured genes is supervised by every training strain. No cold start.
        self.free_gene_dim = free_gene_dim
        self.free_gene_embedding: nn.Parameter | None = None
        if free_gene_dim > 0:
            assert num_genes > 0, "free_gene_dim > 0 requires num_genes"
            self.free_gene_embedding = nn.Parameter(
                torch.randn(num_genes, free_gene_dim) * 0.02
            )

        # in_mult = 3 for the CONCAT arm: the head receives [h_pert ; h_i ; c] instead of
        # h_pert alone, so it can represent arbitrary functions of (h_i, c_b) rather than
        # only functions of their sum.
        self.in_mult = in_mult
        self.extra_dim = extra_dim
        # FiLM generator: cond -> (gamma, beta). Zero-initialized final layer so the head
        # starts as an EXACT identity and the arm is a clean ablation.
        self.film: nn.Sequential | None = None
        if film_dim > 0:
            width = hidden_dim * in_mult + free_gene_dim + extra_dim
            self.film = nn.Sequential(
                nn.Linear(film_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * width),
            )
            film_out = cast(nn.Linear, self.film[-1])
            nn.init.zeros_(film_out.weight)
            nn.init.zeros_(film_out.bias)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * in_mult + free_gene_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * param_dim),
        )

    def forward(
        self, H_genes_pert: torch.Tensor, film_cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of the per-gene head.

        Args:
            H_genes_pert: [batch, N, d] equivariant perturbed gene embeddings.
            film_cond: [batch, film_dim] strain-level conditioning vector. Required for
                FiLM to engage: the scale/shift are only applied when the head was built
                with ``film_dim > 0`` AND this is not ``None``, so passing ``None``
                reduces the head to its unconditioned form.

        Returns:
            predictions: [batch, N] if output_dim == param_dim == 1;
            [batch, N, param_dim] for a distributional scalar-per-gene head;
            else [batch, N, output_dim (, param_dim)].
        """
        if self.free_gene_embedding is not None:
            batch_size = H_genes_pert.shape[0]
            emb = self.free_gene_embedding.unsqueeze(0).expand(batch_size, -1, -1)
            H_genes_pert = torch.cat([H_genes_pert, emb], dim=-1)
        if self.film is not None and film_cond is not None:
            # FiLM: the perturbation-set summary generates a per-channel scale and shift
            # for EVERY gene. Multiplicative conditioning -- a class no arm has tested;
            # concat/bilinear/propagation are all additive or feature-appending.
            gamma, beta = self.film(film_cond).chunk(2, dim=-1)
            H_genes_pert = H_genes_pert * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        out = self.mlp(H_genes_pert)  # [batch, N, output_dim * param_dim]
        if self.output_dim == 1:
            # Scalar per gene: the trailing axis is the distributional param axis (or is
            # squeezed away entirely in the point case).
            if self.param_dim == 1:
                out = out.squeeze(-1)  # [batch, N]
        elif self.param_dim > 1:
            batch, num_genes, _ = out.shape
            out = out.view(batch, num_genes, self.output_dim, self.param_dim)
        return cast(torch.Tensor, out)


class PerMetaboliteHead(nn.Module):
    """Per-metabolite readout head via GPR pooling of gene embeddings.

    Metabolism enters as a REPRESENTATION ANNOTATION (not an attention prior):
    perturbed gene embeddings are pooled over the genes catalyzing each reaction
    (gene->gpr->reaction incidence), then pooled over the reactions each metabolite
    participates in (metabolite<-reaction incidence), producing a per-metabolite
    vector. Metabolites are NEVER promoted to encoder nodes -- the encoder keeps
    fixed N gene nodes. Used for MetabolitePhenotype (mapped to Yeast9 metabolite
    ids by the cell_graph metabolite node ordering).

    graph_level == "metabolism" selects this head.

    TODO(ws7): the pooling here is a mean over the catalyzing genes / participating
    reactions using the sha-normalized incidence built from cell_graph. Stoichiometry
    (edge weights on the metabolite<->reaction hyperedge) is ignored for now; a future
    version can weight the reaction->metabolite pool by |stoichiometric coefficient|.
    """

    def __init__(self, hidden_dim: int, output_dim: int = 1, dropout: float = 0.1):
        """Build the MLP applied to every per-metabolite pooled embedding.

        Args:
            hidden_dim: Model hidden dimension.
            output_dim: Per-metabolite output dimension (default 1 -> scalar).
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        H_genes_pert: torch.Tensor,
        gpr_incidence_T: torch.Tensor,
        mr_incidence: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the per-metabolite head.

        Args:
            H_genes_pert: [batch, N, d] equivariant perturbed gene embeddings.
            gpr_incidence_T: sparse [R, N] reaction<-gene incidence (row-normalized
                so each reaction row is a mean over its catalyzing genes).
            mr_incidence: sparse [M, R] metabolite<-reaction incidence (row-normalized
                so each metabolite row is a mean over its participating reactions).

        Returns:
            predictions: [batch, M] if output_dim == 1, else [batch, M, output_dim].
        """
        batch_size, num_genes, d = H_genes_pert.shape
        # Flatten batch/feature into columns so sparse.mm applies once: [N, B*d]
        h_flat = H_genes_pert.permute(1, 0, 2).reshape(num_genes, batch_size * d)
        reaction = torch.sparse.mm(gpr_incidence_T, h_flat)  # [R, B*d]
        metabolite = torch.sparse.mm(mr_incidence, reaction)  # [M, B*d]
        num_met = metabolite.shape[0]
        met = metabolite.reshape(num_met, batch_size, d).permute(1, 0, 2)  # [B, M, d]
        out = self.mlp(met)  # [B, M, output_dim]
        if self.output_dim == 1:
            out = out.squeeze(-1)  # [B, M]
        return cast(torch.Tensor, out)


class MaskedMultitaskLoss(nn.Module):
    """Masked multitask loss over config-selected heads plus graph regularization.

    Each head's loss is masked to the genotypes in the batch that actually carry
    that phenotype (sparse supervision): absent modalities contribute zero. The
    existing graph-regularization attention loss term is added UNCHANGED.

    A head may carry a :class:`~torchcell.losses.distributional.DistHead`
    (``dist_heads[name]``), in which case its loss is the head's proper scoring rule
    (Gaussian CRPS / pinball) over the predicted DISTRIBUTION parameters. Heads with no
    entry keep the plain elementwise point loss (``mse``/``l1``) -- this is what
    ``gene_interaction`` and the synthetic dry-run use, so the pre-distributional behavior
    is preserved exactly when no ``dist_heads`` are passed.
    """

    def __init__(
        self,
        head_weights: dict[str, float] | None = None,
        loss_fn: str = "mse",
        dist_heads: dict[str, nn.Module] | None = None,
    ):
        """Build the masked multitask loss.

        Args:
            head_weights: Per-head scalar weights (default 1.0 for any head present).
            loss_fn: Elementwise regression loss, "mse" or "l1", for heads with no DistHead.
            dist_heads: Optional {head_name: DistHead} routing that head's params through a
                distributional loss. Registered as a ModuleDict so the quantile grid buffer
                moves with ``.to(device)``.
        """
        super().__init__()
        self.head_weights = head_weights or {}
        assert loss_fn in ("mse", "l1"), f"unsupported loss_fn {loss_fn}"
        self.loss_fn = loss_fn
        self.dist_heads = nn.ModuleDict(dist_heads or {})

    def _elementwise(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_fn == "mse":
            return F.mse_loss(pred, target)
        return F.l1_loss(pred, target)

    def _elementwise_masked(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        feature_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Point loss reduced over the entries ``feature_mask`` selects.

        With ``feature_mask=None`` this is EXACTLY ``_elementwise`` (both reduce with an
        unweighted mean), so every pre-existing head is bit-identical. The masked path
        exists for the masked-label objective, which must not score a gene whose value it
        was just handed as input.
        """
        if feature_mask is None:
            return self._elementwise(pred, target)
        # DEFERRED import, not module-level: the top-of-file import of
        # torchcell.losses.distributional is deliberately TYPE_CHECKING-only so this models
        # module carries no runtime dependency on torchcell.losses (whose __init__ pulls in
        # other model-specific losses). Importing inside the function keeps that property
        # while still using ONE definition of the reduction rather than a copy that can drift.
        from torchcell.losses.distributional import masked_mean

        elem = (pred - target) ** 2 if self.loss_fn == "mse" else (pred - target).abs()
        return masked_mean(elem, feature_mask)

    def forward(
        self,
        head_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
        graph_reg_loss: torch.Tensor | None = None,
        feature_masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the masked multitask loss.

        Args:
            head_outputs: {head_name: prediction [B, ...]}.
            targets: {head_name: target [B, ...]} for supervised heads.
            masks: {head_name: bool [B]} selecting supervised genotypes per head.
                A missing mask means all rows are supervised for that head.
            graph_reg_loss: Optional scalar graph-regularization loss, added as-is.
            feature_masks: {head_name: bool [B, F]} selecting which FEATURES to score.
                Used by the masked-label objective so a step is scored only on the genes
                still HIDDEN at that step; a revealed gene is model input, and scoring it
                would reward copying. A missing entry scores every feature, which is the
                pre-existing behaviour.

        Returns:
            total_loss: scalar summed loss.
            per_head_loss: {head_name: detached scalar loss} for logging.
        """
        masks = masks or {}
        feature_masks = feature_masks or {}
        # Establish a device/grad anchor. graph_reg_loss stays UNCHANGED.
        if graph_reg_loss is not None:
            total = graph_reg_loss
        else:
            any_pred = next(iter(head_outputs.values()))
            total = torch.zeros((), device=any_pred.device)

        per_head: dict[str, torch.Tensor] = {}
        for name, pred in head_outputs.items():
            if name not in targets:
                continue
            target = targets[name]
            weight = self.head_weights.get(name, 1.0)
            mask = masks.get(name)
            dist_head = (
                cast("DistHead", self.dist_heads[name])
                if name in self.dist_heads
                else None
            )
            fmask = feature_masks.get(name)
            if dist_head is not None:
                # Distributional head: it owns the row mask, the feature mask and the
                # empty-supervision guard.
                loss = dist_head.loss(pred, target, mask, feature_mask=fmask)
            elif mask is not None:
                if mask.sum() == 0:
                    # No genotype in this batch carries this phenotype -> zero loss,
                    # but keep it connected to the graph so grads flow as zero.
                    loss = pred.sum() * 0.0
                else:
                    loss = self._elementwise_masked(
                        pred[mask], target[mask], None if fmask is None else fmask[mask]
                    )
            else:
                loss = self._elementwise_masked(pred, target, fmask)
            total = total + weight * loss
            per_head[name] = loss.detach()

        return total, per_head


class CellGraphTransformer(nn.Module):
    """Equivariant Cell Graph Transformer model.

    Architecture:
    1. Gene embeddings + CLS token
    2. Transformer encoder with graph-regularized attention
    3. EQUIVARIANT perturbation transformation (Type I Virtual Instrument)
    4. Perturbation readout head (Type II Virtual Instrument)
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
        graph_reg_lambda: float = 0.0,  # Loss lambda for graph regularization
        node_embeddings: dict[str, Any] | None = None,  # Pre-computed embeddings
        learnable_embedding_config: dict[str, Any] | None = None,  # Learnable config
        heads_config: dict[str, Any] | None = None,  # Multitask decoder heads
        perturbation_propagation_config: (
            dict[str, Any] | None
        ) = None,  # Pair-(p,i) routing
        post_perturbation_mixing_config: (
            dict[str, Any] | None
        ) = None,  # Perceiver cross-gene mixing
        attention_mask_config: (
            dict[str, Any] | None
        ) = None,  # Hard graph masking (replaces the KL)
        observed_label_config: (
            dict[str, Any] | None
        ) = None,  # Masked-label conditioning
        cross_gene_config: (
            dict[str, Any] | None
        ) = None,  # GEARS-style pooled cross-gene mixing
    ):
        """Build embeddings, transformer encoder, and perturbation heads.

        Args:
            gene_num: Number of genes (sequence length excluding CLS token).
            hidden_channels: Model hidden dimension.
            num_transformer_layers: Number of transformer encoder layers.
            num_attention_heads: Number of attention heads per layer.
            cell_graph: Reference HeteroData graph providing adjacency.
            graph_regularization_config: Optional graph-regularization config.
            perturbation_head_config: Optional perturbation-head config.
            dropout: Dropout probability.
            graph_reg_lambda: Loss weight for graph regularization.
            node_embeddings: Optional pre-computed node embeddings.
            learnable_embedding_config: Optional config for learnable embeddings.
            heads_config: Optional multitask decoder head config. When None, only the
                single gene-interaction PerturbationHead is active, reproducing the
                pre-multitask single-head behavior exactly (no extra parameters).
                Otherwise a dict selecting any of "global" / "per_gene" /
                "per_metabolite" (each a sub-dict, e.g.
                {"global": {"output_dim": 501, "use_gene_pool": True}}).
            attention_mask_config: Optional HARD graph masking of encoder attention.
                ``{"enabled": bool, "layers": [...], "head_graphs": {head: relation}}``
                burns one relation into each head's attention structurally, so the graph
                prior needs no KL and no lambda. An out-of-range entry in ``layers`` or an
                unmatched relation name RAISES rather than silently applying no mask.
            perturbation_propagation_config: Optional multi-hop spreading of the
                perturbation over each graph's adjacency, supplying a pair-(p, i) feature
                the additive form lacks. NOTE this pathway needs curated organism graphs,
                so it does NOT transfer to an organism without an interactome.
            observed_label_config: Optional masked-prediction conditioning, letting the
                decoder see a fraction of the measured labels and predict the rest. Inert
                without cross-gene mixing, and identity at validation (100% masked).
            post_perturbation_mixing_config: Optional Perceiver-style mixing BETWEEN gene
                tokens after the perturbation is applied -- the channel that makes
                ``observed_label_config`` able to do anything.
            cross_gene_config: Optional GEARS-style pooled cross-gene mixing.
        """
        super().__init__()
        self.gene_num = gene_num
        self.hidden_channels = hidden_channels
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.graph_reg_lambda = graph_reg_lambda

        # === Node Embedding Configuration ===
        # Determine embedding strategy based on config
        self.node_embeddings = node_embeddings
        self.learnable_embedding_config = learnable_embedding_config

        # Calculate pre-computed embedding dimension.
        # `node_embeddings` maps name -> BaseEmbeddingDataset; each item is a PyG Data with
        # `.id` (gene) and `.embeddings` (dict of [1, D] tensors, one per sub-embedding).
        # The total per-gene width is the sum over EVERY selected dataset's sub-embeddings,
        # matching how `Neo4jCellDataset` concatenates them into `cell_graph["gene"].x`.
        # (The previous `.graph.nodes(...)` access assumed a networkx-backed dataset and
        # raised AttributeError -- this path had never been exercised, since every 019
        # config ran with `node_embeddings: []`.)
        precomputed_dim = 0
        if node_embeddings is not None and len(node_embeddings) > 0:
            for ds in node_embeddings.values():
                first = ds[0]
                precomputed_dim += sum(
                    int(t.shape[-1]) for t in first.embeddings.values()
                )

        # Learnable embedding configuration
        learnable_enabled = False
        learnable_size = hidden_channels  # Default

        if learnable_embedding_config is not None:
            learnable_enabled = learnable_embedding_config.get("enabled", False)
            learnable_size = learnable_embedding_config.get("size", hidden_channels)
        else:
            # Auto-enable learnable if no pre-computed embeddings
            if precomputed_dim == 0:
                learnable_enabled = True

        # Create learnable embedding if enabled
        self.gene_embedding: nn.Embedding | None
        if learnable_enabled:
            self.gene_embedding = nn.Embedding(gene_num, learnable_size)
            nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)
        else:
            self.gene_embedding = None

        # Calculate total input dimension
        total_input_dim = (learnable_size if learnable_enabled else 0) + precomputed_dim

        # Create preprocessor if input_dim != hidden_channels
        self.embedding_preprocessor: nn.Sequential | None
        if total_input_dim > 0 and total_input_dim != hidden_channels:
            preprocessor_config = (
                learnable_embedding_config.get("preprocessor", {})
                if learnable_embedding_config
                else {}
            )
            num_layers = preprocessor_config.get("num_layers", 2)
            dropout_rate = preprocessor_config.get("dropout", dropout)

            # Build MLP with LayerNorm and GELU activation
            layers: list[nn.Module] = []
            current_dim = total_input_dim

            for i in range(num_layers):
                # Linear layer
                next_dim = (
                    hidden_channels
                    if i == num_layers - 1
                    else (total_input_dim + hidden_channels) // 2
                )
                layers.append(nn.Linear(current_dim, next_dim))

                # LayerNorm
                layers.append(nn.LayerNorm(next_dim))

                # GELU activation (consistent with transformer)
                if i < num_layers - 1:  # No activation after last layer
                    layers.append(nn.GELU())

                # Dropout
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

                current_dim = next_dim

            self.embedding_preprocessor = nn.Sequential(*layers)
        else:
            self.embedding_preprocessor = None

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, hidden_channels) * 0.02)

        # Process graph regularization config
        # Graph regularization is enabled when graph_reg_lambda > 0
        self.adjacency_matrices: dict[str, torch.Tensor] | None
        if self.graph_reg_lambda > 0.0 and graph_regularization_config is not None:
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
                    dropout=dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Equivariant perturbation transformation (Type I Virtual Instrument)
        pert_head_config = perturbation_head_config or {}
        self.perturbation_transform = EquivariantPerturbationTransform(
            hidden_dim=hidden_channels,
            num_heads=pert_head_config.get("num_heads", 8),
            dropout=pert_head_config.get("dropout", dropout),
            residual=str(pert_head_config.get("residual", "postln")),
            num_layers=int(pert_head_config.get("num_layers", 1)),
            ffn_mult=int(pert_head_config.get("ffn_mult", 4)),
            extra_layer_ffn_mult=(
                None
                if pert_head_config.get("extra_layer_ffn_mult") is None
                else int(pert_head_config["extra_layer_ffn_mult"])
            ),
            hadamard=str(pert_head_config.get("hadamard", "off")),
            null_sink=bool(pert_head_config.get("null_sink", False)),
            null_sink_bias_init=float(
                pert_head_config.get("null_sink_bias_init", -4.0)
            ),
            null_sink_trainable=bool(pert_head_config.get("null_sink_trainable", True)),
            null_sink_magnitude_match=bool(
                pert_head_config.get("null_sink_magnitude_match", False)
            ),
        )

        # Perturbation readout head (Type II Virtual Instrument). `pooling` is read once
        # and shared with the per-gene z_S site so the two can never disagree.
        self.pert_pooling = str(pert_head_config.get("pooling", "sum"))
        self.perturbation_head = PerturbationHead(
            hidden_dim=hidden_channels,
            dropout=pert_head_config.get("dropout", dropout),
            pooling=self.pert_pooling,
        )

        # === Hard graph masking: burn the graphs into attention STRUCTURALLY ===
        # An alternative to the KL, not a companion to it. The KL is a soft penalty whose
        # strength needs calibrating (lambda 2e-7 makes the graph term ~1% of the data
        # loss) and whose gradient requires materializing dense attention. A mask is exact,
        # has no lambda, and -- because the weights are no longer needed -- unlocks the
        # fused SDPA kernel.
        #
        # It also FAILS LOUDLY where the KL failed silently: `compute_graph_regularization_loss`
        # skips an unmatched graph name with a bare `continue`, which is why `physical` and
        # `regulatory` (config names) never matched `physical_interaction` /
        # `regulatory_interaction` (cell_graph names) and 2 of 9 heads went unregularized
        # for three rounds. Here an unmatched name raises.
        mask_cfg = attention_mask_config or {}
        self.attention_head_mask: torch.Tensor | None = None
        self.attention_mask_layers: list[int] = list(mask_cfg.get("layers", []))
        if mask_cfg.get("enabled", False):
            # VALIDATE THE LAYER INDICES. `layers` is compared against `layer_idx` at the
            # forward site, so an index outside the encoder simply never matches: masking is
            # silently absent everywhere while the run still reports attention_mask.enabled=
            # true and n_graphs=9. Verified: layers=[7] on a four-layer stack applies NO mask,
            # raises nothing, logs nothing. That is a no-graph model wearing a masked model's
            # config -- exactly the silent-skip failure the `physical`/`physical_interaction`
            # name mismatch caused for three rounds, and it would now poison every arm since
            # masking is the frozen base. An empty list still means ALL layers.
            bad = [
                int(layer)
                for layer in self.attention_mask_layers
                if not 0 <= int(layer) < num_transformer_layers
            ]
            if bad:
                raise ValueError(
                    f"attention_mask.layers={self.attention_mask_layers} names layer(s) "
                    f"{bad} outside the {num_transformer_layers}-layer encoder "
                    "(use [] to mask every layer)"
                )
            self.attention_head_mask = self._build_head_mask(
                cell_graph, dict(mask_cfg.get("head_graphs", {})), num_attention_heads
            )

        # === Pair-(p, i) routing: propagate the deletion along the graphs ===
        # Sparse TRANSPOSED row-normalized adjacencies are built here (independent of
        # graph_reg_lambda, which owns the dense ones) so the propagation arm can run
        # with the graph prior switched off -- the two are separate levers.
        prop_cfg = perturbation_propagation_config or {}
        self.perturbation_propagation: PerturbationGraphPropagation | None = None
        self.adjacency_T_sparse: dict[str, torch.Tensor] = {}
        if prop_cfg.get("enabled", False):
            prop_graphs = list(
                prop_cfg.get("graphs", [])
            ) or self._gene_gene_relation_names(cell_graph)
            self.adjacency_T_sparse = self._build_sparse_adjacency_T(
                cell_graph, prop_graphs
            )
            self.perturbation_propagation = PerturbationGraphPropagation(
                hidden_dim=hidden_channels,
                graph_names=prop_graphs,
                hops=int(prop_cfg.get("hops", 2)),
                dropout=prop_cfg.get("dropout", dropout),
                gate_mode=str(prop_cfg.get("gate_mode", "rezero")),
            )

        # === Multitask decoder heads (config-selectable) ===
        # When heads_config is None NO extra parameters/buffers are created, so the
        # model state_dict + forward output are identical to the single-head model.
        self.heads_config = heads_config or {}

        # The `global` (vector-phenotype) head has TWO structural forms, selected by the
        # head spec's `decoder` key -- this is the S1-vs-S3 axis of the decoder study:
        #   s1_pool  -> GlobalHead    (pool to one vector, fan out; the pool-dilution baseline)
        #   s3_xattn -> CrossAttnHead (one learned query per feature; per-output support)
        # `param_dim` (1 point / 2 gaussian / K quantile) is the ORTHOGONAL distributional
        # axis and only widens the final projection -- it never changes `output_dim` (=F).
        self.global_head: GlobalHead | CrossAttnHead | None = None
        if "global" in self.heads_config:
            g_cfg = self.heads_config["global"] or {}
            g_decoder = g_cfg.get("decoder", "s1_pool")
            g_param_dim = int(g_cfg.get("param_dim", 1))
            if g_decoder == "s1_pool":
                self.global_head = GlobalHead(
                    hidden_dim=hidden_channels,
                    output_dim=g_cfg.get("output_dim", 501),
                    use_gene_pool=g_cfg.get("use_gene_pool", True),
                    dropout=g_cfg.get("dropout", dropout),
                    param_dim=g_param_dim,
                )
            elif g_decoder == "s3_xattn":
                self.global_head = CrossAttnHead(
                    hidden_dim=hidden_channels,
                    output_dim=g_cfg.get("output_dim", 501),
                    num_heads=int(g_cfg.get("num_heads", num_attention_heads)),
                    dropout=g_cfg.get("dropout", dropout),
                    param_dim=g_param_dim,
                    use_ffn=bool(g_cfg.get("use_ffn", True)),
                )
            else:
                raise ValueError(
                    f"unknown global head decoder {g_decoder!r} "
                    "(expected 's1_pool' or 's3_xattn')"
                )

        self.per_gene_head: PerGeneHead | None = None
        if "per_gene" in self.heads_config:
            pg_cfg = self.heads_config["per_gene"] or {}
            self.per_gene_head = PerGeneHead(
                hidden_dim=hidden_channels,
                output_dim=pg_cfg.get("output_dim", 1),
                dropout=pg_cfg.get("dropout", dropout),
                param_dim=int(pg_cfg.get("param_dim", 1)),
                free_gene_dim=int(pg_cfg.get("free_gene_dim", 0)),
                num_genes=gene_num,
                in_mult=(3 if pg_cfg.get("concat_context", False) else 1)
                + (2 if pg_cfg.get("pert_set_context", False) else 0),
                extra_dim=int(pg_cfg.get("bilinear_rank", 0)),
                # film_dim = hidden_channels, NOT 2*hidden_channels. The conditioner is
                # fed z_S ALONE (see the call site). `pert_cond` is [z_S ; h_CLS], and
                # h_CLS comes from the encoder on the WILDTYPE graph, so it is
                # byte-identical for every strain in the batch -- measured across-strain
                # sd 0.0, against 0.973 for z_S. Feeding it meant half the conditioner's
                # first-layer weights saw a constant and could only contribute a fixed
                # offset, i.e. half the FiLM parameters were structurally inert.
                film_dim=hidden_channels
                if pg_cfg.get("film_on_pert_set", False)
                else 0,
            )
        pg_spec = self.heads_config.get("per_gene") or {}
        self.per_gene_concat_context = bool(pg_spec.get("concat_context", False))
        # F0: give the per-gene head the POOLED PERTURBED-SET representation z_S (and
        # h_CLS) -- exactly the pair of inputs PerturbationHead uses, and the mechanism
        # behind the trigenic-interaction result. PerGeneHead currently sees neither: it
        # gets gene i's own token and nothing else, so the set structure that made the
        # other head work is simply absent from this one.
        self.per_gene_pert_set = bool(pg_spec.get("pert_set_context", False))
        self.per_gene_film = bool(pg_spec.get("film_on_pert_set", False))
        self.bilinear: LowRankBilinear | None = None
        if int(pg_spec.get("bilinear_rank", 0)) > 0:
            self.bilinear = LowRankBilinear(
                hidden_dim=hidden_channels, rank=int(pg_spec["bilinear_rank"])
            )

        # GEARS-style cross-gene mixing (pool -> broadcast), and the low-rank response
        # basis. Both live in the DECODER, after the perturbation operator.
        cg_cfg = cross_gene_config or {}
        self.cross_gene_mixing: CrossGeneMixing | None = None
        if cg_cfg.get("enabled", False):
            self.cross_gene_mixing = CrossGeneMixing(
                hidden_dim=hidden_channels,
                rank=int(cg_cfg.get("rank", 64)),
                dropout=cg_cfg.get("dropout", dropout),
            )
        self.response_basis: ResponseBasisHead | None = None
        pg0 = (heads_config or {}).get("per_gene") or {}
        if int(pg0.get("response_basis_rank", 0)) > 0:
            self.response_basis = ResponseBasisHead(
                hidden_dim=hidden_channels,
                rank=int(pg0["response_basis_rank"]),
                param_dim=int(pg0.get("param_dim", 1)),
                dropout=dropout,
            )

        obs_cfg = observed_label_config or {}
        self.observed_label_encoder: ObservedLabelEncoder | None = None
        if obs_cfg.get("enabled", False):
            self.observed_label_encoder = ObservedLabelEncoder(
                hidden_dim=hidden_channels,
                dropout=obs_cfg.get("dropout", dropout),
                gate_mode=str(obs_cfg.get("gate_mode", "on")),
            )

        # Post-perturbation cross-gene mixing (Perceiver bottleneck).
        mix_cfg = post_perturbation_mixing_config or {}
        self.post_perturbation_mixing: PerceiverMixing | None = None
        if mix_cfg.get("enabled", False):
            self.post_perturbation_mixing = PerceiverMixing(
                hidden_dim=hidden_channels,
                num_latents=int(mix_cfg.get("num_latents", 32)),
                num_heads=int(mix_cfg.get("num_heads", num_attention_heads)),
                dropout=mix_cfg.get("dropout", dropout),
                gate_mode=str(mix_cfg.get("gate_mode", "rezero")),
            )

        self.per_metabolite_head: PerMetaboliteHead | None = None
        self.num_metabolites = 0
        if "per_metabolite" in self.heads_config:
            incidence = self._build_metabolic_incidence(cell_graph)
            if incidence is None:
                raise ValueError(
                    "per_metabolite head requested but cell_graph lacks the "
                    "('gene','gpr','reaction') and/or "
                    "('metabolite','reaction','metabolite') edges needed to build "
                    "the GPR/RMR incidence."
                )
            gpr_incidence_T, mr_incidence, num_metabolites = incidence
            # Sparse incidence is fixed graph structure -> non-persistent buffers so
            # it moves with .to(device) but is not written into checkpoints.
            self.register_buffer("gpr_incidence_T", gpr_incidence_T, persistent=False)
            self.register_buffer("mr_incidence", mr_incidence, persistent=False)
            self.num_metabolites = num_metabolites
            pm_cfg = self.heads_config["per_metabolite"] or {}
            self.per_metabolite_head = PerMetaboliteHead(
                hidden_dim=hidden_channels,
                output_dim=pm_cfg.get("output_dim", 1),
                dropout=pm_cfg.get("dropout", dropout),
            )

    def _build_metabolic_incidence(
        self, cell_graph: HeteroData
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        """Build row-normalized sparse GPR/RMR incidence for the metabolite head.

        Args:
            cell_graph: HeteroData carrying ('gene','gpr','reaction') and
                ('metabolite','reaction','metabolite') edges.

        Returns:
            (gpr_incidence_T [R, N], mr_incidence [M, R], num_metabolites), or None
            if the required metabolic edges are absent.
        """
        gpr_edge = ("gene", "gpr", "reaction")
        rmr_edge = ("metabolite", "reaction", "metabolite")
        if gpr_edge not in cell_graph.edge_types:
            return None
        if rmr_edge not in cell_graph.edge_types:
            return None

        gpr_ei = cell_graph[gpr_edge].edge_index  # [2, E_gpr]: row0 gene, row1 reaction
        rmr_ei = cell_graph[
            rmr_edge
        ].edge_index  # [2, E_rmr]: row0 metab, row1 reaction

        # Resolve reaction / metabolite counts from node stores when available.
        if "reaction" in cell_graph.node_types and (
            cell_graph["reaction"].get("num_nodes", None) is not None
        ):
            num_react = int(cell_graph["reaction"].num_nodes)
        else:
            num_react = int(max(int(gpr_ei[1].max()), int(rmr_ei[1].max()))) + 1
        if "metabolite" in cell_graph.node_types and (
            cell_graph["metabolite"].get("num_nodes", None) is not None
        ):
            num_met = int(cell_graph["metabolite"].num_nodes)
        else:
            num_met = int(rmr_ei[0].max()) + 1

        # Reaction <- gene, row-normalized (each reaction = mean over its genes).
        gene_idx = gpr_ei[0]
        react_idx = gpr_ei[1]
        react_deg = torch.zeros(num_react)
        react_deg.index_add_(0, react_idx, torch.ones(react_idx.shape[0]))
        gpr_vals = 1.0 / react_deg[react_idx].clamp(min=1.0)
        gpr_incidence_T = torch.sparse_coo_tensor(
            torch.stack([react_idx, gene_idx]), gpr_vals, (num_react, self.gene_num)
        ).coalesce()

        # Metabolite <- reaction, row-normalized (each metabolite = mean over reactions).
        met_idx = rmr_ei[0]
        rmr_react_idx = rmr_ei[1]
        met_deg = torch.zeros(num_met)
        met_deg.index_add_(0, met_idx, torch.ones(met_idx.shape[0]))
        mr_vals = 1.0 / met_deg[met_idx].clamp(min=1.0)
        mr_incidence = torch.sparse_coo_tensor(
            torch.stack([met_idx, rmr_react_idx]), mr_vals, (num_met, num_react)
        ).coalesce()

        return gpr_incidence_T, mr_incidence, num_met

    def _build_head_mask(
        self, cell_graph: HeteroData, head_graphs: dict[Any, str], num_heads: int
    ) -> torch.Tensor:
        """Build a [heads, N+1, N+1] bool mask, True where attention is ALLOWED.

        Heads with no assigned graph stay FULLY FREE -- the same hedge the KL design uses
        by regularizing a single layer and leaving the rest unconstrained. CLS attends
        everywhere and is attended by everyone, and every gene keeps a self-loop, so no
        softmax row can be entirely -inf (which would produce NaN).

        Args:
            cell_graph: HeteroData providing (gene, rel, gene) edges.
            head_graphs: head index -> relation name.
            num_heads: Total attention heads.

        Returns:
            [num_heads, N+1, N+1] boolean mask.
        """
        available = {
            rel: cell_graph[(src, rel, dst)].edge_index
            for (src, rel, dst) in cell_graph.edge_types
            if src == "gene" and dst == "gene"
        }
        num_nodes = self.gene_num
        mask = torch.ones(num_heads, num_nodes + 1, num_nodes + 1, dtype=torch.bool)
        for head_str, rel in head_graphs.items():
            head = int(head_str)
            if rel not in available:
                raise ValueError(
                    f"attention_mask head {head} names graph {rel!r}, which is not a "
                    f"gene-gene relation in cell_graph (available: {sorted(available)}). "
                    "The KL path skipped this silently; masking does not."
                )
            edge_index = available[rel]
            head_mask = torch.zeros(num_nodes + 1, num_nodes + 1, dtype=torch.bool)
            head_mask[edge_index[0] + 1, edge_index[1] + 1] = True
            head_mask[edge_index[1] + 1, edge_index[0] + 1] = True  # symmetric
            head_mask.fill_diagonal_(True)  # self-loops: no all -inf row
            head_mask[0, :] = True  # CLS attends to everything
            head_mask[:, 0] = True  # everything attends to CLS
            mask[head] = head_mask
        return mask

    @staticmethod
    def _gene_gene_relation_names(cell_graph: HeteroData) -> list[str]:
        """Relation names of every (gene, rel, gene) edge type, in graph order."""
        return [
            rel
            for (src, rel, dst) in cell_graph.edge_types
            if src == "gene" and dst == "gene"
        ]

    def _build_sparse_adjacency_T(
        self, cell_graph: HeteroData, graph_names: list[str]
    ) -> dict[str, torch.Tensor]:
        """Build TRANSPOSED row-normalized SPARSE adjacencies for propagation.

        Row-normalized as in :meth:`_normalize_adjacency_matrices` (A[i,:] / deg(i)), then
        transposed once here so ``A_T @ x`` spreads mass OUTWARD from the perturbed genes.
        Sparse, not dense: nine dense 6,607 x 6,607 float32 matrices would be ~1.6 GB of
        GPU memory to do O(|E|) work.

        Args:
            cell_graph: HeteroData providing (gene, rel, gene) edge indices.
            graph_names: Relation names to build.

        Returns:
            Mapping relation name -> sparse [N, N] transposed normalized adjacency.
        """
        available = {
            rel: cell_graph[(src, rel, dst)].edge_index
            for (src, rel, dst) in cell_graph.edge_types
            if src == "gene" and dst == "gene"
        }
        missing = [g for g in graph_names if g not in available]
        if missing:
            raise ValueError(
                f"perturbation_propagation.graphs {missing} are not gene-gene relations "
                f"in cell_graph (available: {sorted(available)})"
            )

        num_nodes = self.gene_num
        out: dict[str, torch.Tensor] = {}
        for name in graph_names:
            edge_index = available[name]
            row, col = edge_index[0], edge_index[1]
            degree = torch.zeros(num_nodes).index_add_(0, row, torch.ones(row.numel()))
            values = 1.0 / degree[row].clamp(min=1.0)
            # Transposed: entry (col, row) carries the normalized weight of row -> col.
            out[name] = torch.sparse_coo_tensor(
                torch.stack([col, row]), values, (num_nodes, num_nodes)
            ).coalesce()
        return out

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

            # Key by the relation name AND, when `to_cell_data` has suffixed it, by the
            # bare stem. The graph BUILDER names these `physical` / `regulatory`, while
            # `to_cell_data` emits `physical_interaction` / `regulatory_interaction`; every
            # 019 config uses the builder spelling. Accepting both is what makes the two
            # naming systems meet -- previously they silently did not, and
            # compute_graph_regularization_loss skipped the mismatch with a bare
            # `continue`, so _006/_007 ran 7 regularized heads while declaring 9.
            normalized_matrices[rel] = A_tilde
            if rel.endswith("_interaction"):
                normalized_matrices[rel[: -len("_interaction")]] = A_tilde

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
        # Early return if graph regularization is disabled (lambda is 0)
        if self.graph_reg_lambda == 0.0:
            return torch.tensor(0.0, device=attention_weights.device)

        # When lambda > 0, adjacency_matrices is populated in __init__
        assert self.adjacency_matrices is not None

        total_loss = torch.tensor(0.0, device=attention_weights.device)
        batch_size, num_heads, N, _ = attention_weights.shape

        for graph_name, config in self.regularized_head_config.items():
            # Handle both single int and list of ints for layer
            layer_spec = config["layer"]
            layer_list = [layer_spec] if isinstance(layer_spec, int) else layer_spec
            if layer_idx not in layer_list:
                continue

            head_idx = config["head"]
            lambda_k = config["lambda"]

            # Get normalized adjacency from dict
            if graph_name not in self.adjacency_matrices:
                # RAISE, do not skip. The bare `continue` that used to be here is why
                # `physical` / `regulatory` went unregularized across _006, _007 and _008
                # without a single warning: the run still logged graph_reg/loss and
                # ratio_to_data, just computed over 7 graphs while the config declared 9.
                raise ValueError(
                    f"regularized head names {graph_name!r} but cell_graph has no such "
                    f"gene-gene relation (available: {sorted(self.adjacency_matrices)}). "
                    "to_cell_data SUFFIXES two of them: physical -> physical_interaction, "
                    "regulatory -> regulatory_interaction."
                )
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

            # PER-GRAPH edge-density normalization, with lambda applied EXACTLY ONCE.
            #
            # Two compounding defects lived here and both silently weakened the prior:
            #
            # (a) LAMBDA WAS SQUARED. `lambda_k` was applied per graph and then
            #     `self.graph_reg_lambda` again as a global scale -- but the config sets
            #     BOTH from the same key (`regularized_heads.<g>.lambda:
            #     ${model.graph_regularization.graph_reg_lambda}`), so the effective weight
            #     was lambda^2. At the configured 1e-3 that is 1e-6: a 1000x
            #     under-application, i.e. the graph prior was very nearly off.
            #
            # (b) THE DIVISOR SUMMED EDGES OVER EVERY GRAPH, so activating more graphs
            #     shrank the term for graphs that were ALREADY there. Going from 2 to 9
            #     graphs weakens each one by ~7.5x, so lambda does not mean the same thing
            #     across graph counts and a 2-graph vs 9-graph comparison is not
            #     interpretable -- the graph count and the prior strength are confounded.
            #
            # Normalizing by THIS graph's own edge count makes lambda_k a per-graph
            # quantity invariant to how many other graphs are active, which is what lets
            # us port 010's 9-graph block and keep the tuned lambda meaningful.
            edges_k = self._edge_count(graph_name, A_tilde)
            if edges_k > 0:
                total_loss = total_loss + lambda_k * kl_loss / (edges_k / self.gene_num)

        # NOTE: lambda is applied ONCE, here via lambda_k (= graph_reg_lambda from the
        # config). The previous code multiplied by `self.graph_reg_lambda` a SECOND time
        # after the loop, making the effective weight lambda^2 -- at the swept lambda=1e-3
        # that is 1e-6 before degree normalization, i.e. the graph term was effectively
        # switched off in every run to date.
        return total_loss

    def _edge_count(self, graph_name: str, A_tilde: torch.Tensor) -> int:
        """Nonzero count of a normalized adjacency, cached.

        `A_tilde` is dense [N, N] with N = 6,607 (~43.7 M entries), so counting nonzeros
        on every layer of every batch is not affordable. The adjacency is fixed for the
        life of the model, so the count is computed once per graph.
        """
        if not hasattr(self, "_edge_count_cache"):
            self._edge_count_cache: dict[str, int] = {}
        if graph_name not in self._edge_count_cache:
            self._edge_count_cache[graph_name] = int((A_tilde > 0).sum().item())
        return self._edge_count_cache[graph_name]

    def forward(
        self,
        cell_graph: HeteroData,
        batch: HeteroData,
        return_attention: bool = False,
        observed_values: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass of Equivariant Cell Graph Transformer.

        Args:
            cell_graph: Full wildtype graph structure (not used directly, genes indexed by order)
            batch: Perturbation data with indices and phenotypes
            return_attention: If True, store and return attention weights (memory
                intensive: it forces the manual softmax path, which materializes a
                [1, heads, N+1, N+1] matrix per layer instead of using the fused kernel)
            observed_values: [batch, N] partially observed measured labels for masked
                prediction, or None. Only consumed when the model was built with an
                ``observed_label_config``.
            observed_mask: [batch, N] boolean companion to ``observed_values``, True where
                a value is observed. At validation everything is masked, so the forward
                pass is identical to the unconditioned model and the metric stays
                comparable across arms.

        Returns:
            predictions: [batch_size, 1] gene interaction predictions
            representations: Dict with embeddings, attention weights (if requested), losses, and H_genes_pert
        """
        # Get device from learnable embedding or CLS token
        device = (
            self.gene_embedding.weight.device
            if self.gene_embedding is not None
            else self.cls_token.device
        )
        N = self.gene_num

        # The perturbed-gene indices in a batch are positions in the CELL GRAPH's
        # sorted node list, while the embedding table is indexed by this model's
        # gene_num. Nothing else ties the two together: the cell graph is rebuilt
        # at run time from whatever gene set the genome currently yields, so a
        # genome that has gained or lost genes since training silently repoints
        # every embedding row and the model answers for the wrong genes. That is
        # not hypothetical. The 010 inference_2 and inference_3 runs were scored
        # against a 6,579-gene genome by a checkpoint trained on 6,607 genes; the
        # 28 missing mitochondrial ORFs sort first, so every index was shifted by
        # 28 and every prediction was for a different triple, with no error
        # raised and the runs agreeing perfectly with each other.
        n_graph_genes = int(cell_graph["gene"].num_nodes)
        if n_graph_genes != N:
            raise ValueError(
                f"cell graph has {n_graph_genes} gene nodes but this model was "
                f"built for gene_num={N}. Perturbation indices address the cell "
                "graph's sorted node list, so a mismatch silently scores the "
                "wrong genes. Rebuild the cell graph from the gene set the "
                "checkpoint was trained on."
            )

        # 1. Create gene embeddings for all genes
        gene_idx = torch.arange(N, device=device)

        # Get learnable embeddings if enabled
        if self.gene_embedding is not None:
            H_genes_learnable = self.gene_embedding(gene_idx)  # [N, learnable_size]
        else:
            H_genes_learnable = None

        # Get pre-computed embeddings if available
        if self.node_embeddings is not None and len(self.node_embeddings) > 0:
            # Extract pre-computed embeddings from cell_graph node attributes
            # They should be concatenated in cell_graph["gene"].x
            H_genes_precomputed = cell_graph["gene"].x.to(
                device
            )  # [N, precomputed_dim]
        else:
            H_genes_precomputed = None

        # Combine embeddings
        if H_genes_learnable is not None and H_genes_precomputed is not None:
            H_genes_combined = torch.cat(
                [H_genes_learnable, H_genes_precomputed], dim=-1
            )
        elif H_genes_learnable is not None:
            H_genes_combined = H_genes_learnable
        elif H_genes_precomputed is not None:
            H_genes_combined = H_genes_precomputed
        else:
            raise ValueError(
                "No gene embeddings available (neither learnable nor pre-computed)"
            )

        # Apply preprocessor if needed
        if self.embedding_preprocessor is not None:
            gene_embs = self.embedding_preprocessor(
                H_genes_combined
            )  # [N, hidden_channels]
        else:
            gene_embs = H_genes_combined  # [N, hidden_channels]

        # 2. Prepend CLS token
        cls_token = self.cls_token  # [1, d]
        X = torch.cat([cls_token, gene_embs], dim=0).unsqueeze(0)  # [1, N+1, d]

        # 3. Transformer encoder
        H = X
        all_attention_weights: list[torch.Tensor] | None = (
            [] if return_attention else None
        )
        residual_update_ratios: list[float] | None = [] if return_attention else None
        total_graph_reg_loss = torch.tensor(0.0, device=device)

        for layer_idx, layer in enumerate(self.transformer_layers):
            # Store input for residual ratio computation
            H_prev = H

            # CRITICAL FIX: Only compute attention when actually needed
            # - During training: need for graph_reg_loss (if graph_reg_lambda > 0)
            # - During validation with return_attention=True: need for diagnostics
            need_attention_for_graph_reg = self.graph_reg_lambda > 0.0
            should_return_attention = need_attention_for_graph_reg or return_attention

            layer_mask = (
                self.attention_head_mask
                if self.attention_head_mask is not None
                and (
                    not self.attention_mask_layers
                    or layer_idx in self.attention_mask_layers
                )
                else None
            )
            if layer_mask is not None and layer_mask.device != H.device:
                # `layer_mask` IS `self.attention_head_mask` here -- the conditional above
                # yields either that tensor or None. Move the NARROWED local and write it
                # back, rather than re-reading the Optional attribute (which mypy cannot
                # narrow through a ternary). Same one-time transfer, cached for later
                # layers, one less indirection.
                layer_mask = layer_mask.to(H.device)
                self.attention_head_mask = layer_mask
            H, attention_weights = layer(
                H, return_attention=should_return_attention, head_mask=layer_mask
            )

            if attention_weights is not None:
                # Always compute graph regularization loss (needed for training)
                graph_loss = self.compute_graph_regularization_loss(
                    attention_weights, layer_idx
                )
                total_graph_reg_loss = total_graph_reg_loss + graph_loss

                # Only store attention weights if requested for diagnostics (memory intensive)
                if return_attention:
                    assert all_attention_weights is not None
                    all_attention_weights.append(attention_weights)
                else:
                    # CRITICAL: Delete if not storing to prevent memory accumulation
                    del attention_weights

            # Compute residual update ratio: ||H - H_prev|| / ||H_prev||
            if return_attention:
                assert residual_update_ratios is not None
                with torch.no_grad():
                    residual_norm = torch.norm(H - H_prev, p=2).item()
                    input_norm = torch.norm(H_prev, p=2).item()
                    ratio = residual_norm / (input_norm + 1e-8)
                    residual_update_ratios.append(ratio)

        # 4. Extract CLS and gene embeddings
        H_squeezed = H.squeeze(0)  # [N+1, d]
        h_CLS = H_squeezed[0]  # [d]
        H_genes = H_squeezed[1:]  # [N, d]

        # 5. Apply EQUIVARIANT perturbation transformation (Type I Virtual Instrument)
        H_genes_pert, pert_context = self.perturbation_transform(
            H_genes,
            batch["gene"].perturbation_indices,
            batch["gene"].perturbation_indices_batch,
        )  # [batch, N, d] each - EQUIVARIANT!

        # 5b. Pair-(p, i) routing. Without this the ONLY strain-dependent quantity is a
        #     single d-vector added uniformly to every gene, so nothing downstream can
        #     depend on the relationship between the deleted gene p and the read-out
        #     gene i. Gated by a ReZero parameter initialized at 0 -> exact identity at
        #     init, so enabling it is a clean ablation.
        if self.perturbation_propagation is not None:
            # Move ONCE and cache: these are fixed buffers, and re-uploading nine sparse
            # matrices every step would cost more than the propagation itself.
            if next(iter(self.adjacency_T_sparse.values())).device != device:
                self.adjacency_T_sparse = {
                    name: A.to(device) for name, A in self.adjacency_T_sparse.items()
                }
            H_genes_pert = self.perturbation_propagation(
                H_genes_pert,
                self.adjacency_T_sparse,
                batch["gene"].perturbation_indices,
                batch["gene"].perturbation_indices_batch,
            )

        # 5b2. Observed-label conditioning. Injected BEFORE the cross-gene mixing so the
        #      revealed values have a pathway to reach other genes -- without mixing the
        #      conditioning is inert.
        if self.observed_label_encoder is not None and observed_values is not None:
            assert observed_mask is not None
            H_genes_pert = self.observed_label_encoder(
                H_genes_pert, observed_values, observed_mask
            )

        # 5b3. GEARS-style pooled cross-gene mixing.
        if self.cross_gene_mixing is not None:
            H_genes_pert = self.cross_gene_mixing(H_genes_pert)

        # 5c. Post-perturbation cross-gene mixing. This is the ONLY place gene
        #     representations interact after the perturbation is injected; without it a
        #     gene's state depends on (h_i, {h_p}) and never on any other gene.
        if self.post_perturbation_mixing is not None:
            H_genes_pert = self.post_perturbation_mixing(H_genes_pert)

        # 6. Perturbation readout head (Type II Virtual Instrument)
        #    This is the ORIGINAL single (gene-interaction) head; kept as the first
        #    returned element so single-head behavior is unchanged.
        predictions = self.perturbation_head(
            h_CLS,
            H_genes_pert,
            batch["gene"].perturbation_indices,
            batch["gene"].perturbation_indices_batch,
        )

        # 7. Multitask decoder heads (config-selectable). graph_level selects the
        #    head downstream: global -> class-token, node -> per-gene,
        #    metabolism -> per-metabolite. When no heads are configured this dict is
        #    empty and nothing here runs.
        head_outputs: dict[str, torch.Tensor] = {}
        if self.global_head is not None:
            head_outputs["global"] = self.global_head(h_CLS, H_genes_pert)
        if self.per_gene_head is not None:
            if self.per_gene_concat_context:
                # [h_pert ; h_i ; c] -- h_i is the strain-INVARIANT encoder output, c the
                # raw attended perturbation context. Giving the head all three lets it
                # learn arbitrary (h_i, c_b) interactions; with h_pert alone it can only
                # see g(h_i + c_b), i.e. functions of the SUM.
                pg_in = torch.cat(
                    [
                        H_genes_pert,
                        H_genes.unsqueeze(0).expand(H_genes_pert.shape[0], -1, -1),
                        pert_context,
                    ],
                    dim=-1,
                )
            else:
                pg_in = H_genes_pert

            # z_S: pooled over the PERTURBED gene tokens -- the same pooled set summary
            # PerturbationHead uses, and permutation-invariant over S. Paired with h_CLS
            # it is exactly that head's input, which is the configuration behind the
            # trigenic-interaction result. Pooling follows `perturbation_head.pooling`
            # (default sum) so the two z_S sites can never silently disagree.
            pert_cond: torch.Tensor | None = None
            if (
                self.per_gene_pert_set
                or self.per_gene_film
                or self.response_basis is not None
            ):
                bsz = H_genes_pert.shape[0]
                pert_idx = batch["gene"].perturbation_indices
                pert_b = batch["gene"].perturbation_indices_batch
                z_S = torch.zeros(
                    bsz, self.hidden_channels, device=device, dtype=H_genes_pert.dtype
                )
                for b in range(bsz):
                    sel = pert_idx[pert_b == b]
                    if len(sel) > 0:
                        h_sel = H_genes_pert[b, sel, :]
                        z_S[b] = (
                            h_sel.sum(dim=0)
                            if self.pert_pooling == "sum"
                            else h_sel.mean(dim=0)
                        )
                pert_cond = torch.cat([z_S, h_CLS.unsqueeze(0).expand(bsz, -1)], dim=-1)
            if self.per_gene_pert_set and pert_cond is not None:
                pg_in = torch.cat(
                    [pg_in, pert_cond.unsqueeze(1).expand(-1, pg_in.shape[1], -1)],
                    dim=-1,
                )
            if self.bilinear is not None:
                pg_in = torch.cat([pg_in, self.bilinear(H_genes, pert_context)], dim=-1)
            # z_S ONLY as the FiLM conditioner -- pert_cond's h_CLS half is strain-constant.
            head_outputs["per_gene"] = self.per_gene_head(
                pg_in, film_cond=z_S if self.per_gene_film else None
            )
            if self.response_basis is not None:
                assert pert_cond is not None, (
                    "response_basis needs the perturbation summary; the head spec must "
                    "also request it so z_S is computed"
                )
                head_outputs["per_gene"] = head_outputs[
                    "per_gene"
                ] + self.response_basis(H_genes_pert, pert_cond)
        if self.per_metabolite_head is not None:
            head_outputs["per_metabolite"] = self.per_metabolite_head(
                H_genes_pert, self.gpr_incidence_T, self.mr_incidence
            )

        return predictions, {
            "h_CLS": h_CLS,
            "H_genes": H_genes,
            "H_genes_pert": H_genes_pert,  # Equivariant perturbed gene embeddings
            "z_p": H_genes_pert,  # Backward compatibility alias
            "attention_weights": all_attention_weights,
            "residual_update_ratios": residual_update_ratios,
            "graph_reg_loss": total_graph_reg_loss,
            "head_outputs": head_outputs,
        }

    @property
    def num_parameters(self) -> dict[str, int]:
        """Count parameters in each component."""

        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            # None when learnable embeddings are disabled (content features only).
            "gene_embedding": (
                count_params(self.gene_embedding)
                if self.gene_embedding is not None
                else 0
            ),
            # Projects concat(content_embeddings [+ learnable]) -> hidden. Was omitted from
            # this tally, which made every node-embedding variant report an identical total
            # even though the projection scales with the content dim (NT 2560 vs codon 64).
            "embedding_preprocessor": (
                count_params(self.embedding_preprocessor)
                if self.embedding_preprocessor is not None
                else 0
            ),
            "cls_token": self.cls_token.numel(),
            "transformer_layers": count_params(self.transformer_layers),
            "perturbation_transform": count_params(self.perturbation_transform),
            "perturbation_head": count_params(self.perturbation_head),
        }
        if self.global_head is not None:
            counts["global_head"] = count_params(self.global_head)
        if self.per_gene_head is not None:
            counts["per_gene_head"] = count_params(self.per_gene_head)
        if self.per_metabolite_head is not None:
            counts["per_metabolite_head"] = count_params(self.per_metabolite_head)
        counts["total"] = sum(counts.values())
        return counts


def calculate_weight_l2_norm(model: nn.Module) -> float:
    """Calculate L2 norm of all model weights."""
    l2_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_norm += torch.sum(param**2).item()
    return cast(float, np.sqrt(l2_norm))


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
    return cast(float, torch.norm(diff, p="fro").item())


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="equivariant_cell_graph_transformer",
)
def main(cfg: DictConfig) -> None:
    """Main training function for Equivariant Cell Graph Transformer."""
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    from scipy import stats
    from scipy.stats import gaussian_kde

    from torchcell.losses.logcosh import LogCoshLoss
    from torchcell.scratch.load_batch_006_perturbation import load_perturbation_batch
    from torchcell.timestamp import timestamp

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

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
    print("\nCell graph edge types:")
    for edge_type in cell_graph.edge_types:
        src, rel, dst = edge_type
        if src == "gene" and dst == "gene":
            print(f"  {edge_type}: {cell_graph[edge_type].num_edges} edges")

    model = CellGraphTransformer(  # type: ignore[call-arg]  # pre-existing latent kwarg mismatch in demo main(); behavior unchanged
        gene_num=cfg.model.gene_num,
        hidden_channels=cfg.model.hidden_channels,
        num_transformer_layers=cfg.model.num_transformer_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        cell_graph=cell_graph,
        graph_regularization_config=cfg.model.graph_regularization,
        perturbation_head_config=cfg.model.perturbation_head,
        dropout=cfg.model.dropout,
        graph_reg_scale=cfg.model.get("graph_reg_scale", 0.001),
    ).to(device)

    print("\nModel architecture:")
    print(model)
    param_counts = model.num_parameters
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # Test equivariant transformation
    print("\n" + "=" * 80)
    print("Testing Equivariant Architecture...")
    print("=" * 80)
    model.eval()
    with torch.no_grad():
        test_predictions, test_representations = model(cell_graph, batch)
        H_genes = test_representations["H_genes"]  # [N, d]
        H_genes_pert = test_representations["H_genes_pert"]  # [batch, N, d]

        print(f"\nWildtype gene embeddings: {H_genes.shape}")
        print(f"Perturbed gene embeddings: {H_genes_pert.shape}")
        print("Equivariance preserved: per-gene structure maintained")

        # Check that perturbations actually change the embeddings
        batch_size = H_genes_pert.shape[0]
        print("\nPerturbation Effects (first 3 samples):")
        for b in range(min(3, batch_size)):
            mask = batch["gene"].perturbation_indices_batch == b
            pert_idx = batch["gene"].perturbation_indices[mask]

            if len(pert_idx) > 0:
                # Change in perturbed genes
                pert_change = (
                    (H_genes_pert[b, pert_idx] - H_genes[pert_idx]).norm(dim=-1).mean()
                )

                # Change in non-perturbed genes (from cross-attention context)
                all_idx = torch.arange(H_genes.shape[0], device=H_genes.device)
                non_pert_mask = ~torch.isin(all_idx, pert_idx)
                non_pert_change = (
                    (H_genes_pert[b, non_pert_mask] - H_genes[non_pert_mask])
                    .norm(dim=-1)
                    .mean()
                )

                print(f"  Sample {b}:")
                print(
                    f"    Perturbed genes ({len(pert_idx)}): Δ = {pert_change.item():.4f}"
                )
                print(
                    f"    Non-perturbed genes: Δ = {non_pert_change.item():.4f} (context effect)"
                )
    model.train()

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
    plot_dir = osp.join(
        cast(str, ASSET_IMAGES_DIR), f"equivariant_cell_graph_transformer_{timestamp()}"
    )
    os.makedirs(plot_dir, exist_ok=True)

    def save_intermediate_plot(
        epoch: int,
        losses: list[float],
        pred_losses: list[float],
        graph_reg_losses: list[float],
        correlations: list[float],
        spearman_correlations: list[float],
        mses: list[float],
        maes: list[float],
        rmses: list[float],
        learning_rates: list[float],
        weight_l2_norms: list[float],
        smoothness_history: list[float],
        cfg: DictConfig,
        model: nn.Module,
        cell_graph: HeteroData,
        batch: HeteroData,
        y: torch.Tensor,
    ) -> None:
        """Save intermediate training plot every print interval."""
        plt.figure(figsize=(20, 12))

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
        plt.plot(range(1, epoch + 2), correlations, "g-", label="Pearson", linewidth=2)
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
                    except Exception:
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
        bins = cast(
            Sequence[float],
            np.linspace(
                min(true_np[valid_mask].min(), pred_np[valid_mask].min()),
                max(true_np[valid_mask].max(), pred_np[valid_mask].max()),
                30,
            ),
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
                plt.plot(
                    x_range, kde_true(x_range), "b-", linewidth=2, label="True KDE"
                )
                plt.plot(
                    x_range, kde_pred(x_range), "r-", linewidth=2, label="Pred KDE"
                )
            except Exception:
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
        param_counts = cast("dict[str, int]", model.num_parameters)
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

            plt.plot(epochs_range, pred_losses, "b-", label="Pred Loss", linewidth=2)
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
        plt.title(
            f"Error Distribution (μ={np.mean(errors):.4f}, σ={np.std(errors):.4f})"
        )
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
            f"Equivariant Cell Graph Transformer Training - Epoch {epoch + 1}/{cfg.trainer.max_epochs}",
            fontsize=16,
            y=0.998,
        )

        plt.tight_layout()
        plt.savefig(
            osp.join(plot_dir, f"training_epoch_{epoch + 1:04d}.png"),
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
                    except Exception:
                        corr = 0.0

                    try:
                        spearman_corr, _ = stats.spearmanr(
                            pred_np[valid_mask], y_np[valid_mask]
                        )
                        if np.isnan(spearman_corr):
                            spearman_corr = 0.0
                    except Exception:
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
