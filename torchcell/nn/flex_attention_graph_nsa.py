import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
from torch_geometric.datasets import StochasticBlockModelDataset
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix


class AttentionBlock(nn.Module):
    """
    Base Attention Block with common components for both MAB and SAB
    """

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Define the common components
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # MLP after attention
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x, adj_matrix=None):
        raise NotImplementedError("Implemented in subclasses")


class MAB(AttentionBlock):
    """
    Masked Attention Block - Uses graph structure to mask attention
    """

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__(hidden_dim, num_heads)
        print(
            f"Initializing MAB with dims: hidden={hidden_dim}, heads={num_heads}, head_dim={self.head_dim}"
        )

    def forward(self, x, adj_matrix):
        device = x.device
        print(
            f"MAB.forward - Input on device: {device}, x shape: {x.shape}, adj shape: {adj_matrix.shape}"
        )

        # First normalization layer
        normed_x = self.norm1(x)

        # Reshape for multi-head attention
        batch_size, num_nodes, _ = normed_x.shape
        q = k = v = normed_x.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_nodes, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Choose implementation based on device
        if device.type == "cuda":
            print("  Using CUDA FlexAttention with graph masking")

            # Use FlexAttention with GPU
            def node_mask_mod(b, h, q_idx, kv_idx):
                return adj_matrix[b, q_idx, kv_idx]

            block_mask = create_block_mask(
                node_mask_mod,
                B=batch_size,
                H=None,
                Q_LEN=num_nodes,
                KV_LEN=num_nodes,
                device=device,
            )

            attn_output = flex_attention(q, k, v, block_mask=block_mask)
        else:
            print("  Using CPU manual masked attention")
            # Manual implementation for CPU
            scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)

            # Expand adjacency matrix to match batch and heads
            mask = adj_matrix.unsqueeze(1).expand(batch_size, self.num_heads, -1, -1)

            # Print sparsity stats of the mask
            mask_sparsity = (mask == 0).float().mean().item()
            print(f"  Mask sparsity: {mask_sparsity:.4f} (fraction of zeros)")

            # Apply mask: -inf where there's no edge
            masked_scores = torch.where(
                mask > 0, scores, torch.tensor(-float("inf"), device=device)
            )

            # Apply softmax and compute weighted values
            attn_weights = torch.nn.functional.softmax(masked_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, num_nodes, self.hidden_dim)

        # First residual connection
        x = x + attn_output

        # Second normalization and MLP
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)

        # Second residual connection
        output = x + mlp_output

        print(f"  MAB output shape: {output.shape}")
        return output


class SAB(AttentionBlock):
    """
    Self Attention Block - Uses standard self-attention with no masking
    """

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__(hidden_dim, num_heads)
        print(
            f"Initializing SAB with dims: hidden={hidden_dim}, heads={num_heads}, head_dim={self.head_dim}"
        )

    def forward(self, x, adj_matrix=None):
        device = x.device
        print(f"SAB.forward - Input on device: {device}, x shape: {x.shape}")

        # First normalization layer
        normed_x = self.norm1(x)

        # Reshape for multi-head attention
        batch_size, num_nodes, _ = normed_x.shape
        q = k = v = normed_x.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_nodes, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Choose implementation based on device
        if device.type == "cuda":
            print("  Using CUDA FlexAttention with no masking")
            # Just use regular attention with no masking
            attn_output = flex_attention(q, k, v)
        else:
            print("  Using CPU manual self-attention (no masking)")
            # Manual implementation for CPU
            scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)

            # Apply softmax and compute weighted values (no masking)
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, num_nodes, self.hidden_dim)

        # First residual connection
        x = x + attn_output

        # Second normalization and MLP
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)

        # Second residual connection
        output = x + mlp_output

        print(f"  SAB output shape: {output.shape}")
        return output


class NSAEncoder(nn.Module):
    """
    Node-Set Attention Encoder with customizable pattern of MAB and SAB blocks
    """

    def __init__(self, input_dim, hidden_dim, pattern=None, num_heads=8):
        """
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden dimension for attention layers
            pattern: List of strings specifying the sequence of attention blocks.
                     Each element should be either 'M' for MAB or 'S' for SAB.
                     Example: ['M', 'S', 'M', 'S', 'S']
                     If None, defaults to alternating MAB and SAB: ['M', 'S', 'M', 'S']
            num_heads: Number of attention heads
        """
        super().__init__()

        # Set default pattern if none provided
        if pattern is None:
            pattern = ["M", "S", "M", "S"]  # Default to alternating MAB/SAB

        # Validate pattern
        for block_type in pattern:
            if block_type not in ["M", "S"]:
                raise ValueError(
                    f"Invalid block type '{block_type}'. Must be 'M' for MAB or 'S' for SAB."
                )

        print(
            f"Initializing NSAEncoder: input_dim={input_dim}, hidden_dim={hidden_dim}, pattern={pattern}, heads={num_heads}"
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Create attention blocks according to the specified pattern
        self.layers = nn.ModuleList()
        for i, block_type in enumerate(pattern):
            if block_type == "M":
                self.layers.append(MAB(hidden_dim, num_heads))
                print(f"  Added layer {i}: Masked Attention Block (MAB)")
            else:  # block_type == 'S'
                self.layers.append(SAB(hidden_dim, num_heads))
                print(f"  Added layer {i}: Self Attention Block (SAB)")

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes] (optional)
        """
        print(
            f"NSAEncoder.forward - inputs: x shape={x.shape}, edge_index shape={edge_index.shape}"
        )
        print(
            f"  Input tensors on device: x={x.device}, edge_index={edge_index.device}"
        )

        # Handle batching - if no batch is provided, assume a single graph
        if batch is None:
            print("  No batch provided, assuming single graph")
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Get number of nodes per graph in the batch
        unique_batch, counts = torch.unique(batch, return_counts=True)
        max_nodes = counts.max().item()
        batch_size = len(unique_batch)
        print(f"  Batch size: {batch_size}, max nodes per graph: {max_nodes}")

        # Create batched dense adjacency matrix
        adj = to_dense_adj(edge_index, batch, max_num_nodes=max_nodes)
        print(f"  Created adjacency matrix: shape={adj.shape}, device={adj.device}")

        # Initialize a padded node feature tensor
        padded_x = torch.zeros(batch_size, max_nodes, x.size(1), device=x.device)

        # Fill in the padded tensor with actual node features
        for b in range(batch_size):
            nodes_in_batch = (batch == b).nonzero().squeeze()
            num_nodes_in_batch = nodes_in_batch.size(0)
            padded_x[b, :num_nodes_in_batch] = x[nodes_in_batch]
            print(f"  Graph {b}: {num_nodes_in_batch} nodes")

        # Project input features
        h = self.input_proj(padded_x)
        print(f"  After projection: shape={h.shape}")

        # Apply attention layers according to the pattern
        for i, layer in enumerate(self.layers):
            print(f"\nProcessing layer {i}:")
            if isinstance(layer, MAB):
                # MAB needs adjacency matrix
                print(f"  Layer {i}: MAB using graph adjacency matrix")
                h = layer(h, adj)
            else:  # isinstance(layer, SAB)
                # SAB doesn't need adjacency
                print(f"  Layer {i}: SAB (no adjacency required)")
                h = layer(h)

        # Return the node embeddings
        print(f"\nFinal output shape: {h.shape}")
        return h


def test_nsa_implementation(data, reordered_adj):
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"CUDA available: {cuda_available}, using device: {device}")

    # Model parameters
    input_dim = data.x.size(1) if hasattr(data, "x") and data.x is not None else 1
    hidden_dim = 64
    num_heads = 4

    # Define a custom pattern of attention blocks
    pattern = ["M", "S", "M", "S", "S"]  # Example pattern

    print(
        f"Model parameters: input_dim={input_dim}, hidden_dim={hidden_dim}, num_heads={num_heads}"
    )
    print(f"Using attention block pattern: {pattern}")

    # If no node features exist, create constant ones
    if not hasattr(data, "x") or data.x is None:
        print("No node features found, creating constant ones")
        data.x = torch.ones(data.num_nodes, 1)

    # Move data to device
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    print(f"Moved data to {device}")

    # Initialize the model and move to device
    model = NSAEncoder(
        input_dim=input_dim, hidden_dim=hidden_dim, pattern=pattern, num_heads=num_heads
    ).to(device)
    print(f"Model moved to {device}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # Forward pass
    print("\nStarting forward pass...")
    output = model(data.x, data.edge_index)
    print("Forward pass complete\n")

    print(f"Input shape: {data.x.shape}")
    print(f"Output shape: {output.shape}")

    return output


def main():
    print("=====================================")
    print("Starting Node-Set Attention Test with Custom MAB/SAB Pattern")
    print("=====================================")

    # Print PyTorch and CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Define dataset parameters
    root = "/tmp/sbm"  # Temporary directory for dataset storage
    block_sizes = [25, 25, 25, 25]  # 4 blocks of 25 nodes each
    edge_probs = [
        [0.7, 0.05, 0.05, 0.05],
        [0.05, 0.7, 0.05, 0.05],
        [0.05, 0.05, 0.7, 0.05],
        [0.05, 0.05, 0.05, 0.7],
    ]  # Higher intra-cluster connection probability

    # Load the dataset
    dataset = StochasticBlockModelDataset(
        root=root, block_sizes=block_sizes, edge_probs=edge_probs
    )
    data = dataset[0]  # Get the first (and only) graph

    # Convert to dense adjacency matrix
    dense_adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]

    # Convert to scipy sparse matrix for RCM reordering
    sparse_adj = csr_matrix(dense_adj.numpy())

    # Compute RCM ordering and fix negative strides issue
    perm = torch.tensor(reverse_cuthill_mckee(sparse_adj).copy(), dtype=torch.long)

    # Apply permutation
    reordered_adj = dense_adj[perm][:, perm]

    # Run the test
    node_embeddings = test_nsa_implementation(data, reordered_adj)

    print("\nTest complete")
    print(f"Node embeddings shape: {node_embeddings.shape}")
    print("=====================================")


if __name__ == "__main__":
    main()
