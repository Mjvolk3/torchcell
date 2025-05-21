import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data


class DAGMessagePassing(MessagePassing):
    def __init__(
        self, child1_dim: int, child2_dim: int, hidden_dim: int = 64, out_dim: int = 32
    ):
        """Message passing for a 3-node DAG with different node attribute counts."""
        super().__init__(aggr="add")

        # MLPs for processing child nodes
        self.child1_mlp = nn.Sequential(
            nn.Linear(child1_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.child2_mlp = nn.Sequential(
            nn.Linear(child2_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # MLP for root node after concatenation
        self.root_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self, x_dict: dict[int, torch.Tensor], edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with heterogeneous node features."""
        # Process child nodes with respective MLPs
        processed_features = []
        processed_features.append(self.child1_mlp(x_dict[1]))
        processed_features.append(self.child2_mlp(x_dict[2]))

        # Concatenate processed features for the root node
        concatenated = torch.cat(processed_features, dim=1)

        # Process through root MLP
        root_out = self.root_mlp(concatenated)

        return root_out


def test_dag_message_passing():
    """Test with a sample 3-node DAG."""
    # Define dimensions
    child1_dim = 3  # First child has 3 attributes
    child2_dim = 5  # Second child has 5 attributes
    hidden_dim = 10
    out_dim = 8

    # Create model
    model = DAGMessagePassing(child1_dim, child2_dim, hidden_dim, out_dim)

    # Create test data - node features
    x_dict = {
        1: torch.randn(1, child1_dim),  # Child1 features
        2: torch.randn(1, child2_dim),  # Child2 features
    }

    # Edge index not actually used in this simplified implementation
    edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)

    # Forward pass
    output = model(x_dict, edge_index)

    print(f"Child1 features shape: {x_dict[1].shape}")
    print(f"Child2 features shape: {x_dict[2].shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    return output


# Run the test
if __name__ == "__main__":
    output = test_dag_message_passing()
