import torch
import wandb
from typing import Optional

class VisGraphDegen:
    def __init__(self, adj_matrix: torch.Tensor, node_features: torch.Tensor) -> None:
        self.adj_matrix = adj_matrix
        self.node_features = node_features

    @staticmethod
    def compute_smoothness(X: torch.Tensor) -> torch.Tensor:
        # X: node feature matrix (N x d)
        N = X.shape[0]
        mean_features = X.mean(dim=0)
        diff = X - mean_features.expand(N, -1)
        return torch.norm(diff, p='fro')

    @staticmethod
    def local_bottleneck_score(
        adj_matrix: torch.Tensor, node_features: torch.Tensor, k: int = 3
    ) -> torch.Tensor:
        n_nodes = adj_matrix.shape[0]
        curr_features = torch.eye(n_nodes, device=adj_matrix.device)
        feature_diversity = []
        for _ in range(k):
            curr_features = torch.mm(curr_features, adj_matrix)
            neighborhood_features = torch.mm(curr_features, node_features)
            feature_var = torch.var(neighborhood_features, dim=1)
            feature_diversity.append(feature_var.mean())
        bottleneck_score = feature_diversity[-1] / (feature_diversity[0] + 1e-6)
        return bottleneck_score

    def log_metrics(self, wandb_key_prefix: str = "train_sample") -> None:
        smoothness = self.compute_smoothness(self.node_features)
        bottleneck = self.local_bottleneck_score(self.adj_matrix, self.node_features)
        wandb.log({
            f"{wandb_key_prefix}/oversmoothing": smoothness.item(),
            f"{wandb_key_prefix}/oversquashing": bottleneck.item()
        })
