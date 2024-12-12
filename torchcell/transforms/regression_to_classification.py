# torchcell/transforms/regression_to_classification
# [[torchcell.transforms.regression_to_classification]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/transforms/regression_to_classification
# Test file: tests/torchcell/transforms/test_regression_to_classification.py

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Literal, Optional
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from abc import ABC, abstractmethod
from scipy.stats import norm
import copy
from enum import Enum, auto
from typing import Literal
import torch
from typing import Dict, Optional, Union, List
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
import numpy as np
import copy
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform, Compose
import copy
from typing import List, Union
from torch_geometric.transforms import Compose, BaseTransform
from torch_geometric.data import HeteroData, Batch
import copy
import torch

### Normalize


class LabelNormalizationTransform(BaseTransform):
    """Transform for normalizing labels with different strategies."""

    def __init__(
        self,
        dataset: "Neo4jCellDataset",
        label_configs: Dict[str, Dict],
        eps: float = 1e-8,
    ):
        """
        Args:
            dataset: Neo4jCellDataset instance
            label_configs: Dictionary mapping label names to their configurations
                Example:
                {
                    'fitness': {
                        'strategy': 'minmax',  # or 'standard' or 'robust'
                    }
                }
            eps: Small constant to avoid division by zero
        """
        super().__init__()
        self.label_configs = label_configs
        self.eps = eps
        self.stats = {}

        # Calculate statistics for each label
        df = dataset.label_df.replace([np.inf, -np.inf], np.nan)
        for label, config in label_configs.items():
            if label not in df.columns:
                raise ValueError(f"Label {label} not found in dataset")

            values = df[label].dropna().values
            stats = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "strategy": config["strategy"],
            }
            self.stats[label] = stats

    def normalize(self, values: torch.Tensor, label: str) -> torch.Tensor:
        """Normalize values based on specified strategy."""
        if torch.all(torch.isnan(values)):
            return values

        stats = self.stats[label]
        strategy = stats["strategy"]

        if strategy == "standard":
            return (values - stats["mean"]) / (stats["std"] + self.eps)
        elif strategy == "minmax":
            return (values - stats["min"]) / (stats["max"] - stats["min"] + self.eps)
        elif strategy == "robust":
            iqr = stats["q75"] - stats["q25"]
            return (values - stats["q25"]) / (iqr + self.eps)
        else:
            raise ValueError(f"Unknown normalization strategy: {strategy}")

    def denormalize(self, values: torch.Tensor, label: str) -> torch.Tensor:
        """Denormalize values based on specified strategy."""
        if torch.all(torch.isnan(values)):
            return values

        stats = self.stats[label]
        strategy = stats["strategy"]

        if strategy == "standard":
            return values * stats["std"] + stats["mean"]
        elif strategy == "minmax":
            return values * (stats["max"] - stats["min"]) + stats["min"]
        elif strategy == "robust":
            iqr = stats["q75"] - stats["q25"]
            return values * iqr + stats["q25"]
        else:
            raise ValueError(f"Unknown normalization strategy: {strategy}")

    def forward(self, data: HeteroData) -> HeteroData:
        """Transform the data by normalizing specified labels."""
        data = copy.copy(data)

        for label, config in self.label_configs.items():
            if label in data["gene"]:
                values = data["gene"][label]
                if not isinstance(values, torch.Tensor):
                    values = torch.tensor(values, dtype=torch.float)

                # Store original values with _original suffix
                data["gene"][f"{label}_original"] = values.clone()

                # Replace original label with normalized values
                data["gene"][label] = self.normalize(values, label)

        return data

    def inverse(self, data: HeteroData) -> HeteroData:
        """Inverse transform to recover original scale."""
        data = copy.copy(data)

        for label in self.label_configs:
            if label in data["gene"]:
                values = data["gene"][label]
                if not isinstance(values, torch.Tensor):
                    values = torch.tensor(values, dtype=torch.float)

                # Denormalize values and overwrite normalized ones
                data["gene"][label] = self.denormalize(values, label)

        return data


### Reg to Class


class BaseBinningStrategy(ABC):
    @abstractmethod
    def compute_bins(
        self, values: np.ndarray, num_bins: int
    ) -> tuple[np.ndarray, dict]:
        """Compute bin edges and metadata for values"""
        pass

    def compute_ordinal_labels(
        self, values: torch.Tensor, bin_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ordinal labels for values.
        For n bins, returns n-1 binary values representing threshold crossings.
        Example for 3 bins: [0,0] -> bin 1, [1,0] -> bin 2, [1,1] -> bin 3
        """
        ordinal_labels = torch.zeros((len(values), len(bin_edges) - 2))
        for i, val in enumerate(values):
            if torch.isnan(val):
                ordinal_labels[i] = torch.nan
            else:
                ordinal_labels[i] = (val > bin_edges[1:-1]).float()
        return ordinal_labels

    def compute_soft_labels(
        self,
        values: torch.Tensor,
        bin_edges: torch.Tensor,
        strategy: str = "equal_width",
        sigma_scale: float = 3,
    ) -> torch.Tensor:
        """
        Compute soft labels with consistent heights and widths.
        """
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        # Use the minimum bin width for consistent gaussian spread
        min_bin_width = torch.min(bin_edges[1:] - bin_edges[:-1])
        sigma = min_bin_width * sigma_scale

        soft_labels = torch.zeros((len(values), len(bin_centers)))

        for i, val in enumerate(values):
            if torch.isnan(val):
                soft_labels[i] = torch.nan
            else:
                # Fixed-width gaussian for all bins
                distances = torch.abs(val - bin_centers)
                soft_labels[i] = torch.exp(-0.5 * (distances / sigma) ** 2)

                # Normalize to sum to 1
                if torch.sum(soft_labels[i]) > 0:
                    soft_labels[i] = soft_labels[i] / torch.sum(soft_labels[i])

        return soft_labels

    def compute_onehot_labels(
        self, values: torch.Tensor, bin_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute one-hot encoded labels.
        For n bins, bin_edges should have n+1 values defining n intervals.
        Returns one-hot encoded tensor of shape (len(values), num_bins).
        """
        num_bins = len(bin_edges) - 1
        onehot = torch.zeros((len(values), num_bins))

        for i, val in enumerate(values):
            if torch.isnan(val):
                onehot[i] = torch.nan
            else:
                # Use searchsorted for more intuitive bin assignment
                bin_idx = torch.searchsorted(bin_edges, val, right=True) - 1
                # Clamp to handle any numerical precision edge cases
                bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)
                onehot[i, bin_idx] = 1.0

        return onehot


class EqualWidthStrategy(BaseBinningStrategy):
    def compute_bins(
        self, values: np.ndarray, num_bins: int
    ) -> tuple[np.ndarray, dict]:
        """Compute equal-width bins"""
        non_nan = values[~np.isnan(values)]
        bin_edges = np.linspace(non_nan.min(), non_nan.max(), num_bins + 1)
        metadata = {
            "min": non_nan.min(),
            "max": non_nan.max(),
            "mean": non_nan.mean(),
            "std": non_nan.std(),
            "bin_edges": bin_edges,
            "bin_widths": np.diff(bin_edges),
            "strategy": "equal_width",
        }
        return bin_edges, metadata


class EqualFrequencyStrategy(BaseBinningStrategy):
    def compute_bins(
        self, values: np.ndarray, num_bins: int
    ) -> tuple[np.ndarray, dict]:
        """Compute equal-frequency (quantile) bins"""
        non_nan = values[~np.isnan(values)]
        bin_edges = np.percentile(non_nan, np.linspace(0, 100, num_bins + 1))
        metadata = {
            "min": non_nan.min(),
            "max": non_nan.max(),
            "mean": non_nan.mean(),
            "std": non_nan.std(),
            "bin_edges": bin_edges,
            "bin_counts": np.histogram(non_nan, bin_edges)[0],
            "strategy": "equal_frequency",
        }
        return bin_edges, metadata


class AutoBinStrategy(BaseBinningStrategy):
    def compute_bins(
        self, values: np.ndarray, num_bins: Optional[int] = None
    ) -> tuple[np.ndarray, dict]:
        """Compute bins based on data std"""
        non_nan = values[~np.isnan(values)]
        std = np.std(non_nan)
        range_width = np.max(non_nan) - np.min(non_nan)
        num_bins = int(range_width / std) if num_bins is None else num_bins
        return EqualWidthStrategy().compute_bins(values, num_bins)


class LabelBinningTransform(BaseTransform):
    def __init__(
        self,
        dataset: "Neo4jCellDataset",
        label_configs: Dict[str, Dict],
        normalizer: Optional[LabelNormalizationTransform] = None,
    ):
        """
        Args:
            dataset: Neo4jCellDataset instance
            label_configs: Dict of configurations
            normalizer: Optional normalization transform applied before binning
        """
        super().__init__()
        self.label_configs = label_configs
        self.normalizer = normalizer
        self.strategies = {
            "equal_width": EqualWidthStrategy(),
            "equal_frequency": EqualFrequencyStrategy(),
            "auto": AutoBinStrategy(),
        }

        # Initialize binning parameters for each label
        self.label_metadata = {}
        df = dataset.label_df.replace([np.inf, -np.inf], np.nan)

        # Create normalized df for binning
        if self.normalizer is not None:
            normalized_df = df.copy()
            for label in label_configs:
                if label in df.columns:
                    temp_data = HeteroData()
                    temp_data["gene"] = {
                        label: torch.tensor(df[label].values, dtype=torch.float)
                    }
                    normalized_data = self.normalizer(temp_data)
                    normalized_df[label] = normalized_data["gene"][label].numpy()
            df = normalized_df

        # Now compute bin edges on the normalized data
        for label, config in label_configs.items():
            if label not in df.columns:
                raise ValueError(f"Label {label} not found in dataset")

            strategy = self.strategies[config["strategy"]]
            bin_edges, metadata = strategy.compute_bins(
                df[label].values, config.get("num_bins", None)
            )
            self.label_metadata[label] = metadata

        # If normalizer is provided, also store denormalized bin edges
        if self.normalizer is not None:
            for label, metadata in self.label_metadata.items():
                bin_edges_normalized = torch.tensor(
                    metadata["bin_edges"], dtype=torch.float
                )
                temp_data = HeteroData()
                # Use the actual label name instead of 'temp_label'
                temp_data["gene"] = {label: bin_edges_normalized}
                temp_data = self.normalizer.inverse(temp_data)
                # Fetch the denormalized bin edges using the correct label
                self.label_metadata[label]["bin_edges_denormalized"] = temp_data["gene"][
                    label
                ].numpy()

    def forward(self, data: HeteroData) -> HeteroData:
        """Transform the data by binning specified labels."""
        data = copy.copy(data)

        for label, config in self.label_configs.items():
            if label in data["gene"]:
                values = data["gene"][label]
                if not isinstance(values, torch.Tensor):
                    values = torch.tensor(values, dtype=torch.float)

                # If we have a normalizer, values are already normalized by this point
                # due to composition order. If not, use raw values.
                bin_edges = torch.tensor(self.label_metadata[label]["bin_edges"])
                strategy = self.strategies[config["strategy"]]

                # Store continuous values if requested
                if config.get("store_continuous", True):
                    data["gene"][f"{label}_continuous"] = values.clone()

                label_type = config.get("label_type", "categorical").lower()
                if label_type == "categorical":
                    onehot_labels = strategy.compute_onehot_labels(values, bin_edges)
                    data["gene"][label] = onehot_labels
                elif label_type == "soft":
                    sigma = config.get("sigma", 3)
                    soft_labels = strategy.compute_soft_labels(
                        values, bin_edges, strategy, sigma
                    )
                    data["gene"][label] = soft_labels
                elif label_type == "ordinal":
                    ordinal_labels = strategy.compute_ordinal_labels(values, bin_edges)
                    data["gene"][label] = ordinal_labels

        return data

    def inverse(self, data: HeteroData, seed: int = 42) -> HeteroData:
        """Inverse transform to recover continuous values using random sampling within bins."""
        torch.manual_seed(seed)
        data = copy.copy(data)

        for label, config in self.label_configs.items():
            if label in data["gene"]:
                device = data["gene"][label].device
                label_type = config.get("label_type", "categorical").lower()
                bin_edges = torch.tensor(
                    self.label_metadata[label]["bin_edges"],
                    device=device,
                    dtype=torch.float32,
                )

                # Convert logits to probabilities first
                values = torch.softmax(data["gene"][label], dim=-1)

                if not isinstance(values, torch.Tensor):
                    values = torch.tensor(values, dtype=torch.float, device=device)

                original_shape = values.shape

                # Check for NaN values
                nan_mask = (
                    torch.isnan(values).any(dim=-1)
                    if len(values.shape) > 1
                    else torch.isnan(values)
                )
                continuous_values = torch.full(
                    original_shape[:-1], float("nan"), dtype=torch.float, device=device
                )

                non_nan_mask = ~nan_mask
                if non_nan_mask.any():
                    valid_values = values[non_nan_mask]

                    if label_type == "soft":
                        window_size = 2
                        # Get bin centers
                        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

                        # Find the highest probability bin for each sample
                        max_prob_bins = torch.argmax(valid_values, dim=-1)

                        expected_values = torch.zeros(
                            len(valid_values), device=device, dtype=torch.float32
                        )

                        for i in range(len(valid_values)):
                            peak_bin = max_prob_bins[i]
                            # Attempt to get window around peak bin
                            start_idx = max(0, peak_bin - window_size)
                            end_idx = min(len(bin_centers), peak_bin + window_size + 1)

                            # If we cannot get a full window (e.g., because the peak bin is at the boundary),
                            # just use the single peak bin center.
                            if (end_idx - start_idx) < (2 * window_size + 1):
                                expected_values[i] = bin_centers[peak_bin]
                            else:
                                window_probs = valid_values[i, start_idx:end_idx]
                                window_centers = bin_centers[start_idx:end_idx]

                                # Normalize probabilities in window
                                window_probs = window_probs / window_probs.sum()

                                # Compute weighted average in window
                                expected_values[i] = (
                                    window_probs * window_centers
                                ).sum()

                        continuous_values[non_nan_mask] = expected_values

                    elif label_type == "categorical":
                        indices = torch.argmax(valid_values, dim=-1)
                        temp_values = torch.zeros_like(
                            indices, dtype=torch.float, device=device
                        )

                        for i in range(len(bin_edges) - 1):
                            mask = indices == i
                            if mask.any():
                                low, high = bin_edges[i], bin_edges[i + 1]
                                size = mask.sum()
                                temp_values[mask] = (
                                    torch.rand(size, device=device) * (high - low) + low
                                )

                        continuous_values[non_nan_mask] = temp_values

                    elif label_type == "ordinal":
                        n_thresholds = len(bin_edges) - 2
                        crossings = torch.sum(valid_values > 0.5, dim=-1).clamp(
                            0, n_thresholds
                        )
                        temp_values = torch.zeros_like(
                            crossings, dtype=torch.float, device=device
                        )

                        for i in range(len(bin_edges) - 1):
                            mask = crossings == i
                            if mask.any():
                                low, high = bin_edges[i], bin_edges[i + 1]
                                size = mask.sum()
                                temp_values[mask] = (
                                    torch.rand(size, device=device) * (high - low) + low
                                )

                        continuous_values[non_nan_mask] = temp_values

                data["gene"][label] = continuous_values.view(original_shape[:-1])

        return data

    def get_bin_info(self, label: str) -> dict:
        """Get binning information for a label."""
        if label not in self.label_metadata:
            raise ValueError(f"No binning metadata found for label {label}")
        return self.label_metadata[label]


class InverseCompose(BaseTransform):
    """A transform that applies the inverse of a sequence of transforms in reverse order."""

    def __init__(self, transforms: Union[Compose, List[BaseTransform]]):
        super().__init__()
        if isinstance(transforms, Compose):
            self.transforms = transforms.transforms
        elif isinstance(transforms, list):
            self.transforms = transforms
        else:
            raise ValueError(
                "transforms must be either a Compose object or a list of transforms"
            )

        # Verify all transforms have inverse method
        for t in self.transforms:
            if not hasattr(t, "inverse"):
                raise ValueError(
                    f"Transform {t.__class__.__name__} does not implement inverse method"
                )

    def forward(self, data: Union[HeteroData, Batch]) -> Union[HeteroData, Batch]:
        """Apply inverse transforms in reverse order."""
        # TODO probably don't need to support batch, just did for quick testing.
        data = copy.deepcopy(data)
        for t in reversed(self.transforms):
            data = t.inverse(data)
        return data

    def __repr__(self) -> str:
        args = [f"\n  {t}" for t in self.transforms]
        return f'{self.__class__.__name__}({"".join(args)}\n)'
