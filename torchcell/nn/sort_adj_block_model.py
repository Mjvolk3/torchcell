import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.utils import to_dense_adj
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

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

# Now define the FlexAttention with the adjacency matrix
def graph_adjacency_mask(score, b, h, q_idx, kv_idx, adj_matrix):
    # Check if there's an edge in the adjacency matrix
    # If no edge exists, mask out the attention
    has_edge = adj_matrix[q_idx, kv_idx]
    return torch.where(has_edge, score, -float("inf"))

# Using the reordered adjacency matrix
adj_matrix = reordered_adj

# Create a score_mod that captures the adjacency matrix
def create_adjacency_score_mod(adj_matrix):
    def score_mod(score, b, h, q_idx, kv_idx):
        has_edge = adj_matrix[q_idx, kv_idx].to(dtype=torch.bool)
        return torch.where(
            has_edge,
            score,
            torch.tensor(-float("inf"), device=score.device, dtype=score.dtype),
        )
    return score_mod

# For better performance, we can also use the mask_mod approach
def adjacency_mask_mod(b, h, q_idx, kv_idx):
    return adj_matrix[q_idx, kv_idx] > 0  # True where an edge exists

# Create a BlockMask for more efficient computation
block_mask = create_block_mask(
    adjacency_mask_mod,
    B=None,
    H=None,
    Q_LEN=adj_matrix.shape[0],
    KV_LEN=adj_matrix.shape[1],
)

# Example attention computation
batch_size = 1
num_heads = 4
seq_len = adj_matrix.shape[0]  # Node count
head_dim = 16

# Create random Q, K, V tensors
Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
K = torch.randn(batch_size, num_heads, seq_len, head_dim)
V = torch.randn(batch_size, num_heads, seq_len, head_dim)

# Approach 1: Using score_mod
adj_score_mod = create_adjacency_score_mod(adj_matrix)
output1 = flex_attention(Q, K, V, score_mod=adj_score_mod)
print("Output shape (score_mod):", output1.shape)

# Approach 2: Using block_mask (more efficient)
output2 = flex_attention(Q, K, V, block_mask=block_mask)
print("Output shape (block_mask):", output2.shape)

# Optional: Plot the adjacency matrix
plt.figure(figsize=(6, 6))
plt.imshow(adj_matrix.numpy(), cmap="gray", interpolation="none")
plt.title("Adjacency Matrix Used for FlexAttention")
plt.show()