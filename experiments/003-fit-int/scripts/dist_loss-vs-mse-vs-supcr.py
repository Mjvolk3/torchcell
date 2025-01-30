import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torchcell.losses.multi_dim_nan_tolerant import (
    WeightedDistLoss,
    WeightedSupCRLoss,
    WeightedMSELoss,
)
from torch.utils.data import random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
import os.path as osp
import torchcell
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


def plot_correlations(predictions, true_values, dim, save_path):

    # Convert to numpy and handle NaN values
    predictions_np = predictions.detach().cpu().numpy()
    true_values_np = true_values.detach().cpu().numpy()

    # Colors for plotting
    color = "#2971A0"
    alpha = 0.6

    # Mask for the current dimension
    mask = ~np.isnan(true_values_np)
    y = true_values_np[mask]
    x = predictions_np[mask]

    pearson, _ = stats.pearsonr(x, y)
    spearman, _ = stats.spearmanr(x, y)
    mse = np.mean((y - x) ** 2)

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, alpha=alpha, color=color)
    plt.xlabel(f"Predicted Target {dim}")
    plt.ylabel(f"True Target {dim}")
    plt.title(
        f"Target {dim}\nMSE={mse:.3e}, n={len(x)}\n"
        + f"Pearson={pearson:.3f}, Spearman={spearman:.3f}"
    )

    # Add diagonal line
    min_val = min(plt.xlim()[0], plt.ylim()[0])
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_distribution(
    true_values, predictions, loss_name: str, dim: str, save_path: str
):
    true_values_np = true_values.detach().cpu().numpy()
    predictions_np = predictions.detach().cpu().numpy()

    plt.figure(figsize=(7, 6))
    sns.histplot(true_values_np, color="blue", label="True", kde=True, alpha=0.6)
    sns.histplot(predictions_np, color="red", label="Predicted", kde=True, alpha=0.6)
    plt.legend()
    plt.title(f"Distribution Matching: {loss_name: str} {dim}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_umap(features, labels, loss_name: str, dim: str, save_path: str):
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
    plt.title("UMAP Projection of Predictions")
    plt.savefig(save_path)
    plt.close()


# Load the QM9 dataset
dataset = QM9(root="/tmp/QM9")
num_features = dataset.num_features
num_targets = dataset[0].y.shape[-1]  # 19 regression targets

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=generator
)

# Define loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define GCN model
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
        u = global_add_pool(z, batch)  # Aggregate to graph level
        y_hat = self.fc(u)  # Final output
        return y_hat, z


# Instantiate model, loss, and optimizer
model = GCN(num_features, 128, num_targets)


# write me the bolierplate for a pytorch loss function
class CellLoss(torch.nn.Module):
    # I want to have one loss function
    # \mathcal{L}=\mathcal{L}_{\mathrm{MSE}}(y, \hat{y})+\lambda_1 \mathcal{L}_{\mathrm{div}}(y, \hat{y})+\lambda_2 \mathcal{L}_{\text {con }}\left(z_P, z_I, y\right)
    def __init__(self, lambda_1: float = 1.0, lambda_2: float = 1.0):
        super(CellLoss, self).__init__()
        self.mse_loss = WeightedMSELoss(weights=torch.ones(19))
        self.div_loss = WeightedDistLoss(weights=torch.ones(19))
        self.con_loss = WeightedSupCRLoss(weights=torch.ones(19))
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, predictions, targets, z):
        # Compute individual losses
        mse = self.mse_loss(predictions, targets)
        div = self.div_loss(predictions, targets)
        con = self.con_loss(z, z, targets)

        # Raw losses with lambdas
        weighted_mse = mse
        weighted_div = self.lambda_1 * div
        weighted_con = self.lambda_2 * con

        # Total loss
        total = weighted_mse + weighted_div + weighted_con

        # Normalized contributions
        norm_mse = weighted_mse / total
        norm_div = weighted_div / total
        norm_con = weighted_con / total

        return {
            "total_loss": total,
            "norm_mse": norm_mse,
            "norm_div": norm_div,
            "norm_con": norm_con,
            "raw_mse": weighted_mse,
            "raw_div": weighted_div,
            "raw_con": weighted_con,
        }


losses = {
    "MSE": CellLoss(lambda_1=0, lambda_2=0),
    "DistLoss_1e-2": CellLoss(lambda_1=1e-2, lambda_2=0),
    "DistLoss_1e-1": CellLoss(lambda_1=1e-1, lambda_2=0),
    "DistLoss_1e0": CellLoss(lambda_1=1e0, lambda_2=0),
    "DistLoss_1e1": CellLoss(lambda_1=1e1, lambda_2=0),
    "DistLoss_1e2": CellLoss(lambda_1=1e2, lambda_2=0),
    "SupCR_1e-2": CellLoss(lambda_1=0, lambda_2=1e-2),
    "SupCR_1e-1": CellLoss(lambda_1=0, lambda_2=1e-1),
    "SupCR_1e0": CellLoss(lambda_1=0, lambda_2=1e0),
    "SupCR_1e1": CellLoss(lambda_1=0, lambda_2=1e1),
    "SupCR_1e2": CellLoss(lambda_1=0, lambda_2=1e2),
}

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Modified evaluation function to collect z vectors
def evaluate(loader):
    model.eval()
    all_preds, all_labels, all_z = [], [], []
    with torch.no_grad():
        for data in loader:
            y_hat, z = model(data)
            all_preds.append(y_hat)
            all_labels.append(data.y)
            all_z.append(z)

    return (
        torch.cat(all_preds, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_z, dim=0),
    )


# Function to plot UMAP of node embeddings
def plot_node_embeddings_umap(z, labels, target_idx, filename):
    # Convert to numpy for UMAP
    z_np = z.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Fit UMAP
    reducer = umap.UMAP(random_state=42)
    z_embedded = reducer.fit_transform(z_np)

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        z_embedded[:, 0], z_embedded[:, 1], c=labels_np[:, target_idx], cmap="viridis"
    )
    plt.colorbar(scatter)
    plt.title(f"UMAP of Node Embeddings (Target {target_idx})")
    plt.savefig(filename)
    plt.close()


# Evaluate and get embeddings
val_preds, val_labels, val_z = evaluate(val_loader)

# Plot distributions for each dimension
for i in range(num_targets):
    plot_distribution(
        val_labels[:, i], val_preds[:, i], f"distribution_matching_target_{i}.png"
    )

for i in range(num_targets):
    plot_node_embeddings_umap(
        val_z, val_labels, i, f"umap_node_embeddings_target_{i}.png"
    )

# Plot correlations for each dimension
for i in range(num_targets):
    plot_correlations(
        val_preds[:, i], val_labels[:, i], i, f"correlations_target_{i}.png"
    )
