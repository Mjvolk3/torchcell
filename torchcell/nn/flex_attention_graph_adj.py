import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.utils import to_dense_adj
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix

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

# Plot the adjacency matrix
plt.figure(figsize=(6, 6))
plt.imshow(reordered_adj.numpy(), cmap="gray", interpolation="none")
plt.title("Adjacency Matrix")
plt.show()


# CPU-friendly implementation of graph attention with adjacency matrix masking
def graph_attention(q, k, v, adj_matrix, scale=None):
    """
    Compute attention using the graph adjacency matrix as a mask.

    Args:
        q: Queries tensor [batch_size, num_heads, seq_len, head_dim]
        k: Keys tensor [batch_size, num_heads, seq_len, head_dim]
        v: Values tensor [batch_size, num_heads, seq_len, head_dim]
        adj_matrix: Adjacency matrix [seq_len, seq_len]
        scale: Optional scale factor for attention scores

    Returns:
        Output tensor [batch_size, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    # Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1))

    # Apply scaling
    if scale is None:
        scale = head_dim**-0.5
    scores = scores * scale

    # Apply mask from adjacency matrix
    # Expand adj_matrix to match the shape of scores
    mask = adj_matrix.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

    # Apply the mask: -inf where there's no edge
    masked_scores = torch.where(mask > 0, scores, torch.tensor(-float("inf")))

    # Apply softmax to get attention weights
    attn_weights = torch.nn.functional.softmax(masked_scores, dim=-1)

    # Compute the weighted sum of values
    output = torch.matmul(attn_weights, v)

    return output


# Example attention computation
batch_size = 1
num_heads = 4
seq_len = reordered_adj.shape[0]  # Node count
head_dim = 16

# Create random Q, K, V tensors
Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
K = torch.randn(batch_size, num_heads, seq_len, head_dim)
V = torch.randn(batch_size, num_heads, seq_len, head_dim)

# Compute graph attention
output = graph_attention(Q, K, V, reordered_adj)
print("Output shape:", output.shape)

# Visualize one of the attention weight matrices
head_idx = 0
plt.figure(figsize=(6, 6))
attn_weights = torch.nn.functional.softmax(
    torch.matmul(Q[0, head_idx], K[0, head_idx].transpose(-2, -1)) / (head_dim**0.5),
    dim=-1,
)
attn_weights_masked = torch.where(
    reordered_adj > 0, attn_weights, torch.zeros_like(attn_weights)
)
plt.imshow(attn_weights_masked.detach().numpy(), cmap="viridis")
plt.colorbar()
plt.title(f"Graph-Masked Attention Weights (Head {head_idx})")
plt.show()
