import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import os
from torchcell.losses.multi_dim_nan_tolerant import (
    WeightedDistLoss,
    SupCR,
    WeightedMSELoss,
)
from tqdm import tqdm
from torch import nn
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
import os.path as osp
from scipy import stats
from dotenv import load_dotenv

from torchcell.timestamp import timestamp

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
MPLSTYLE_PATH = os.getenv("MPLSTYLE_PATH")
plt.style.use(MPLSTYLE_PATH)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        z = self.conv2(x, edge_index)
        u = global_add_pool(z, batch)
        y_hat = self.fc(u)
        return y_hat, u


class CellLoss(nn.Module):
    def __init__(
        self, lambda_1: float = 1.0, lambda_2: float = 1.0, device: str = "cuda"
    ):
        super().__init__()
        self.mse_loss = WeightedMSELoss(weights=torch.ones(19, device=device))
        self.div_loss = WeightedDistLoss(weights=torch.ones(19, device=device))
        self.con_loss = SupCR()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        graph_embeddings: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss using graph-level predictions and embeddings.

        Args:
            predictions: Graph-level predictions [batch_size, num_targets]
            targets: Graph-level targets [batch_size, num_targets]
            graph_embeddings: Graph-level embeddings [batch_size, embedding_dim]
        """
        mse_loss, _ = self.mse_loss(predictions, targets)
        div_loss, _ = self.div_loss(predictions, targets)
        con_loss = self.con_loss(graph_embeddings, targets).mean()

        weighted_mse = mse_loss
        weighted_div = self.lambda_1 * div_loss
        weighted_con = self.lambda_2 * con_loss
        total = weighted_mse + weighted_div + weighted_con

        return {
            "total_loss": total,
            "norm_mse": weighted_mse / total,
            "norm_div": weighted_div / total,
            "norm_con": weighted_con / total,
            "raw_mse": weighted_mse,
            "raw_div": weighted_div,
            "raw_con": weighted_con,
        }


class Visualization:
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def plot_correlations(self, predictions, true_values, dim, loss_name):
        predictions_np = predictions.detach().cpu().numpy()
        true_values_np = true_values.detach().cpu().numpy()

        mask = ~np.isnan(true_values_np)
        y = true_values_np[mask]
        x = predictions_np[mask]

        pearson, _ = stats.pearsonr(x, y)
        spearman, _ = stats.spearmanr(x, y)
        mse = np.mean((y - x) ** 2)

        plt.figure(figsize=(7, 6))
        plt.scatter(x, y, alpha=0.6, color="#2971A0")
        plt.xlabel(f"Predicted Target {dim}")
        plt.ylabel(f"True Target {dim}")
        plt.title(
            f"{loss_name} - Target {dim}\nMSE={mse:.3e}, n={len(x)}\n"
            f"Pearson={pearson:.3f}, Spearman={spearman:.3f}"
        )

        min_val = min(plt.xlim()[0], plt.ylim()[0])
        max_val = max(plt.xlim()[1], plt.ylim()[1])
        plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

        save_path = osp.join(
            self.base_dir, f"correlations_{loss_name}_target_{dim}_{timestamp()}.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def plot_distribution(self, true_values, predictions, loss_name, dim):
        true_values_np = true_values.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()

        plt.figure(figsize=(7, 6))
        sns.histplot(true_values_np, color="blue", label="True", kde=True, alpha=0.6)
        sns.histplot(
            predictions_np, color="red", label="Predicted", kde=True, alpha=0.6
        )
        plt.legend()
        plt.title(f"{loss_name} - Distribution Matching Target {dim}")

        save_path = osp.join(
            self.base_dir, f"distribution_{loss_name}_target_{dim}_{timestamp()}.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def plot_umap(self, features, labels, loss_name, dim):
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean")
        embedding = reducer.fit_transform(features.detach().cpu().numpy())

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels.cpu().numpy(),
            cmap="coolwarm",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Target Labels")
        plt.title(f"{loss_name} - UMAP Projection Target {dim}")

        save_path = osp.join(
            self.base_dir, f"umap_{loss_name}_target_{dim}_{timestamp()}.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def train_and_evaluate(
    model, loss_fn, train_loader, val_loader, optimizer, num_epochs=2, device="cuda"
):
    print("Train.")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)  # Move batch to device
            optimizer.zero_grad()
            pred, u = model(batch)
            loss_dict = loss_fn(pred, batch.y, u)
            loss_dict["total_loss"].backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels, all_embeddings = [], [], []
    print("Evaluate.")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = batch.to(device)  # Move batch to device
            pred, u = model(batch)
            # Move predictions back to CPU for collection
            all_preds.append(pred.cpu())
            all_labels.append(batch.y.cpu())
            all_embeddings.append(u.cpu())

    return (
        torch.cat(all_preds, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_embeddings, dim=0),
    )


def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Using device: {device}")

    # Dataset setup remains the same
    dataset = QM9(root="/tmp/QM9")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Loss configurations
    loss_configs = {
        # "MSE": {"lambda_1": 0, "lambda_2": 0},
        "DistLoss_1e-2": {"lambda_1": 1e-2, "lambda_2": 0},
        "DistLoss_1e-1": {"lambda_1": 1e-1, "lambda_2": 0},
        "DistLoss_1e0": {"lambda_1": 1e0, "lambda_2": 0},
        "DistLoss_1e1": {"lambda_1": 1e1, "lambda_2": 0},
        "DistLoss_1e2": {"lambda_1": 1e2, "lambda_2": 0},
        "SupCR_1e-2": {"lambda_1": 0, "lambda_2": 1e-2},
        "SupCR_1e-1": {"lambda_1": 0, "lambda_2": 1e-1},
        "SupCR_1e0": {"lambda_1": 0, "lambda_2": 1e0},
        "SupCR_1e1": {"lambda_1": 0, "lambda_2": 1e1},
        "SupCR_1e2": {"lambda_1": 0, "lambda_2": 1e2},
    }

    for loss_name, config in loss_configs.items():

        print(f"Training with {loss_name}")

        model = GCN(dataset.num_features, 128, dataset[0].y.shape[-1]).to(device)
        loss_fn = CellLoss(**config, device=device)  # Pass device here
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        val_preds, val_labels, val_z = train_and_evaluate(
            model, loss_fn, train_loader, val_loader, optimizer, device=device
        )


if __name__ == "__main__":
    # Set up CUDA if available
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        torch.cuda.empty_cache()
    else:
        print("CUDA is not available. Using CPU.")

    # Run main with automatic device selection
    main()
