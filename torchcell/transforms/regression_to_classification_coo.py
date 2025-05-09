# torchcell/transforms/regression_to_classification_coo
# [[torchcell.transforms.regression_to_classification_coo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/transforms/regression_to_classification_coo
# Test file: tests/torchcell/transforms/test_regression_to_classification_coo.py


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

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Literal, Optional, Union, Tuple, Any
from torch_geometric.data import HeteroData, Batch
from torch_geometric.transforms import BaseTransform, Compose
from abc import ABC, abstractmethod
import copy


class COOLabelNormalizationTransform(BaseTransform):
    """Transform for normalizing labels in COO format with different strategies."""

    def __init__(self, dataset: Any, label_configs: Dict[str, Dict], eps: float = 1e-8):
        """
        Args:
            dataset: Dataset instance containing COO format phenotype data
            label_configs: Dictionary mapping label names to their configurations
                Example:
                {
                    'gene_interaction': {
                        'strategy': 'standard',  # or 'minmax' or 'robust'
                    }
                }
            eps: Small constant to avoid division by zero
        """
        super().__init__()
        self.label_configs = label_configs
        self.eps = eps
        self.stats = {}

        # Calculate statistics for each label type
        self._calculate_stats_from_dataset(dataset)

    def _calculate_stats_from_dataset(self, dataset):
        """Extract phenotype statistics from the dataset in COO format."""
        # Store all values for each phenotype type across the dataset
        phenotype_values_by_type = {}

        # First pass: collect values for each phenotype type
        for sample_idx in range(len(dataset)):
            data = dataset[sample_idx]

            if "phenotype_values" not in data["gene"]:
                continue

            values = data["gene"]["phenotype_values"]
            type_indices = data["gene"]["phenotype_type_indices"]
            phenotype_types = data["gene"]["phenotype_types"]

            # Convert list of lists to flattened list if needed
            flat_phenotype_types = []
            for types in phenotype_types:
                if isinstance(types, list):
                    flat_phenotype_types.extend(types)
                else:
                    flat_phenotype_types.append(types)

            # Process each phenotype type in the configuration
            for label, config in self.label_configs.items():
                if label not in flat_phenotype_types:
                    continue

                # Find type indices matching this label
                label_type_indices = []
                for i, types in enumerate(phenotype_types):
                    if isinstance(types, list) and label in types:
                        label_type_indices.append(i)
                    elif types == label:
                        label_type_indices.append(i)

                # Extract all values for this label
                for type_idx in label_type_indices:
                    mask = type_indices == type_idx
                    if torch.any(mask):
                        label_values = values[mask].cpu().numpy()

                        if label not in phenotype_values_by_type:
                            phenotype_values_by_type[label] = label_values
                        else:
                            phenotype_values_by_type[label] = np.concatenate(
                                [phenotype_values_by_type[label], label_values]
                            )

        # Second pass: compute statistics for each phenotype type
        for label, config in self.label_configs.items():
            if label in phenotype_values_by_type:
                values = phenotype_values_by_type[label]
                # Remove NaNs for statistics calculation
                non_nan_values = values[~np.isnan(values)]

                if len(non_nan_values) > 0:
                    self.stats[label] = {
                        "mean": float(np.mean(non_nan_values)),
                        "std": float(np.std(non_nan_values)),
                        "min": float(np.min(non_nan_values)),
                        "max": float(np.max(non_nan_values)),
                        "q25": float(np.percentile(non_nan_values, 25)),
                        "q75": float(np.percentile(non_nan_values, 75)),
                        "strategy": config["strategy"],
                    }

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
            range_val = stats["max"] - stats["min"]
            return values * range_val + stats["min"]
        elif strategy == "robust":
            iqr = stats["q75"] - stats["q25"]
            return values * iqr + stats["q25"]
        else:
            raise ValueError(f"Unknown normalization strategy: {strategy}")

    def forward(self, data: HeteroData) -> HeteroData:
        """Transform the data by normalizing phenotype values in COO format."""
        data = copy.copy(data)

        if "phenotype_values" not in data["gene"]:
            return data

        values = data["gene"]["phenotype_values"]
        type_indices = data["gene"]["phenotype_type_indices"]
        phenotype_types = data["gene"]["phenotype_types"]

        # Store original values
        data["gene"]["phenotype_values_original"] = values.clone()

        # Create a new tensor for normalized values
        normalized_values = values.clone()

        # Process each label in the configuration
        for label, config in self.label_configs.items():
            if label not in self.stats:
                continue

            # Find all indices for this label
            label_type_indices = []
            for i, types in enumerate(phenotype_types):
                if isinstance(types, list) and label in types:
                    label_type_indices.append(i)
                elif types == label:
                    label_type_indices.append(i)

            # Normalize values for each type index
            for type_idx in label_type_indices:
                mask = type_indices == type_idx
                if torch.any(mask):
                    normalized_values[mask] = self.normalize(values[mask], label)

        # Update the values in the data
        data["gene"]["phenotype_values"] = normalized_values

        return data

    def inverse(self, data: HeteroData) -> HeteroData:
        """Inverse transform to recover original scale."""
        data = copy.copy(data)

        if "phenotype_values" not in data["gene"]:
            return data

        # If we have stored originals, use them
        if "phenotype_values_original" in data["gene"]:
            data["gene"]["phenotype_values"] = data["gene"][
                "phenotype_values_original"
            ].clone()
            return data

        values = data["gene"]["phenotype_values"]
        type_indices = data["gene"]["phenotype_type_indices"]
        phenotype_types = data["gene"]["phenotype_types"]

        # Create a new tensor for denormalized values
        denormalized_values = values.clone()

        # Process each label in the configuration
        for label, config in self.label_configs.items():
            if label not in self.stats:
                continue

            # Find all indices for this label
            label_type_indices = []
            for i, types in enumerate(phenotype_types):
                if isinstance(types, list) and label in types:
                    label_type_indices.append(i)
                elif types == label:
                    label_type_indices.append(i)

            # Denormalize values for each type index
            for type_idx in label_type_indices:
                mask = type_indices == type_idx
                if torch.any(mask):
                    denormalized_values[mask] = self.denormalize(values[mask], label)

        # Update the values in the data
        data["gene"]["phenotype_values"] = denormalized_values

        return data


# Reuse the existing base binning strategy classes
class BaseBinningStrategy(ABC):
    def clamp_values(
        self, values: torch.Tensor, bin_edges: torch.Tensor
    ) -> torch.Tensor:
        """Clamp values to be within bin edge range."""
        # Move bin_edges to the same device as values
        bin_edges = bin_edges.to(values.device)
        return torch.clamp(values, min=bin_edges[0], max=bin_edges[-1])

    def compute_ordinal_labels(
        self, values: torch.Tensor, bin_edges: torch.Tensor
    ) -> torch.Tensor:
        """Compute ordinal labels with clamping for out-of-bounds values."""
        # Move bin_edges to the same device as values
        bin_edges = bin_edges.to(values.device)
        ordinal_labels = torch.zeros(
            (len(values), len(bin_edges) - 2), device=values.device
        )
        clamped_values = self.clamp_values(values, bin_edges)

        for i, val in enumerate(values):
            if torch.isnan(val):
                ordinal_labels[i] = torch.nan
            else:
                ordinal_labels[i] = (clamped_values[i] > bin_edges[1:-1]).float()
        return ordinal_labels

    def compute_soft_labels(
        self,
        values: torch.Tensor,
        bin_edges: torch.Tensor,
        strategy: str = "equal_width",
        sigma_scale: float = 3,
    ) -> torch.Tensor:
        """Compute soft labels with clamping for out-of-bounds values."""
        # Move bin_edges to the same device as values
        bin_edges = bin_edges.to(values.device)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        min_bin_width = torch.min(bin_edges[1:] - bin_edges[:-1])
        sigma = min_bin_width * sigma_scale

        soft_labels = torch.zeros((len(values), len(bin_centers)), device=values.device)
        clamped_values = self.clamp_values(values, bin_edges)

        for i, val in enumerate(values):
            if torch.isnan(val):
                soft_labels[i] = torch.nan
            else:
                # Use clamped value for gaussian computation
                distances = torch.abs(clamped_values[i] - bin_centers)
                soft_labels[i] = torch.exp(-0.5 * (distances / sigma) ** 2)

                # Normalize to sum to 1
                if torch.sum(soft_labels[i]) > 0:
                    soft_labels[i] = soft_labels[i] / torch.sum(soft_labels[i])

        return soft_labels

    def compute_onehot_labels(
        self, values: torch.Tensor, bin_edges: torch.Tensor
    ) -> torch.Tensor:
        """Compute one-hot labels with clamping for out-of-bounds values."""
        # Move bin_edges to the same device as values
        bin_edges = bin_edges.to(values.device)
        num_bins = len(bin_edges) - 1
        onehot = torch.zeros((len(values), num_bins), device=values.device)
        clamped_values = self.clamp_values(values, bin_edges)

        for i, val in enumerate(values):
            if torch.isnan(val):
                onehot[i] = torch.nan
            else:
                # Use clamped value for bin assignment
                bin_idx = (
                    torch.searchsorted(bin_edges, clamped_values[i], right=True) - 1
                )
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


class COOLabelBinningTransform(BaseTransform):
    """Transform for binning labels in COO format."""

    def __init__(
        self,
        dataset: Any,
        label_configs: Dict[str, Dict],
        normalizer: Optional[COOLabelNormalizationTransform] = None,
    ):
        """
        Args:
            dataset: Dataset instance with phenotype data in COO format
            label_configs: Dict of configurations for each label
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

        # Extract values for each phenotype type from the dataset
        self._compute_bins_from_dataset(dataset)

    def _compute_bins_from_dataset(self, dataset):
        """Compute binning parameters from dataset with COO format phenotypes."""
        # Extract all values for each phenotype type
        phenotype_values_by_type = {}

        for sample_idx in range(len(dataset)):
            data = dataset[sample_idx]

            if "phenotype_values" not in data["gene"]:
                continue

            values = data["gene"]["phenotype_values"]
            type_indices = data["gene"]["phenotype_type_indices"]
            phenotype_types = data["gene"]["phenotype_types"]

            # Convert list of lists to flattened list if needed
            flat_phenotype_types = []
            for types in phenotype_types:
                if isinstance(types, list):
                    flat_phenotype_types.extend(types)
                else:
                    flat_phenotype_types.append(types)

            # Process each label in the configuration
            for label, config in self.label_configs.items():
                if label not in flat_phenotype_types:
                    continue

                # Find type indices matching this label
                label_type_indices = []
                for i, types in enumerate(phenotype_types):
                    if isinstance(types, list) and label in types:
                        label_type_indices.append(i)
                    elif types == label:
                        label_type_indices.append(i)

                # Extract all values for this label
                for type_idx in label_type_indices:
                    mask = type_indices == type_idx
                    if torch.any(mask):
                        label_values = values[mask].cpu().numpy()

                        if label not in phenotype_values_by_type:
                            phenotype_values_by_type[label] = label_values
                        else:
                            phenotype_values_by_type[label] = np.concatenate(
                                [phenotype_values_by_type[label], label_values]
                            )

        # Normalize values if a normalizer is provided
        if self.normalizer is not None:
            for label, values in phenotype_values_by_type.items():
                if label in self.normalizer.stats:
                    temp_values = torch.tensor(values, dtype=torch.float)
                    norm_values = self.normalizer.normalize(temp_values, label).numpy()
                    phenotype_values_by_type[label] = norm_values

        # Compute bins for each label
        for label, config in self.label_configs.items():
            if label in phenotype_values_by_type:
                strategy_name = config["strategy"]
                strategy = self.strategies[strategy_name]
                num_bins = config.get("num_bins", 10)  # Default to 10 bins

                bin_edges, metadata = strategy.compute_bins(
                    phenotype_values_by_type[label], num_bins
                )
                self.label_metadata[label] = metadata

                # Store denormalized bin edges if using a normalizer
                if self.normalizer is not None and label in self.normalizer.stats:
                    bin_edges_tensor = torch.tensor(bin_edges, dtype=torch.float)
                    denorm_edges = self.normalizer.denormalize(
                        bin_edges_tensor, label
                    ).numpy()
                    self.label_metadata[label]["bin_edges_denormalized"] = denorm_edges

    def forward(self, data: HeteroData) -> HeteroData:
        """Transform the data by binning phenotype values in COO format."""
        data = copy.copy(data)

        if "phenotype_values" not in data["gene"]:
            return data

        # Extract original COO data
        values = data["gene"]["phenotype_values"]
        type_indices = data["gene"]["phenotype_type_indices"]
        sample_indices = data["gene"]["phenotype_sample_indices"]
        phenotype_types = data["gene"]["phenotype_types"]

        # Store original values
        data["gene"]["phenotype_values_continuous"] = values.clone()

        # Lists to store the new COO format data
        new_values = []
        new_type_indices = []
        new_sample_indices = []

        # Process each unique combination of sample_idx and type_idx
        unique_samples = torch.unique(sample_indices).tolist()

        for sample_idx in unique_samples:
            sample_mask = sample_indices == sample_idx
            sample_type_indices = type_indices[sample_mask]
            sample_values = values[sample_mask]

            unique_types = torch.unique(sample_type_indices).tolist()

            for type_idx in unique_types:
                type_mask = sample_type_indices == type_idx
                type_values = sample_values[type_mask]

                # Get the phenotype type name
                if type_idx >= len(phenotype_types):
                    continue

                if isinstance(phenotype_types[type_idx], list):
                    type_labels = phenotype_types[type_idx]
                else:
                    type_labels = [phenotype_types[type_idx]]

                # Process each label in this phenotype type that's in our config
                for label in type_labels:
                    if (
                        label not in self.label_configs
                        or label not in self.label_metadata
                    ):
                        continue

                    config = self.label_configs[label]
                    bin_edges = torch.tensor(
                        self.label_metadata[label]["bin_edges"], device=values.device
                    )
                    strategy_obj = self.strategies[config["strategy"]]

                    # Apply binning based on label_type
                    label_type = config.get("label_type", "categorical").lower()

                    if label_type == "categorical":
                        binned = strategy_obj.compute_onehot_labels(
                            type_values, bin_edges
                        )
                    elif label_type == "soft":
                        sigma = config.get("sigma", 3)
                        binned = strategy_obj.compute_soft_labels(
                            type_values, bin_edges, config["strategy"], sigma
                        )
                    elif label_type == "ordinal":
                        binned = strategy_obj.compute_ordinal_labels(
                            type_values, bin_edges
                        )
                    else:
                        raise ValueError(f"Unknown label type: {label_type}")

                    # Store with a composite index that encodes both type_idx and bin_idx
                    num_bins = binned.shape[1]
                    base_type_idx = 1000 * type_idx  # Allow up to 1000 bins per type

                    for bin_idx in range(num_bins):
                        # Add to COO storage
                        new_values.append(binned[:, bin_idx])
                        new_type_indices.append(
                            torch.full_like(type_values, base_type_idx + bin_idx)
                        )
                        new_sample_indices.append(
                            torch.full_like(type_values, sample_idx)
                        )

        # Combine all values into the new COO format
        if new_values:
            data["gene"]["phenotype_values"] = torch.cat(new_values)
            data["gene"]["phenotype_type_indices"] = torch.cat(new_type_indices)
            data["gene"]["phenotype_sample_indices"] = torch.cat(new_sample_indices)

            # Flag and metadata for inverse transform
            data["gene"]["phenotype_binned"] = True
            data["gene"]["phenotype_bin_info"] = {
                label: {
                    "bin_edges": self.label_metadata[label]["bin_edges"].tolist(),
                    "label_type": config.get("label_type", "categorical"),
                    "num_bins": len(self.label_metadata[label]["bin_edges"]) - 1,
                    "base_type_idx": 1000 * type_idx,  # Store the encoding scheme
                }
                for label, config in self.label_configs.items()
                if label in self.label_metadata
                for type_idx in range(len(phenotype_types))
                if (
                    isinstance(phenotype_types[type_idx], list)
                    and label in phenotype_types[type_idx]
                )
                or phenotype_types[type_idx] == label
            }

        return data

    def inverse(self, data: HeteroData, seed: int = 42) -> HeteroData:
        """Inverse transform to recover continuous values from binned data."""
        torch.manual_seed(seed)
        data = copy.copy(data)

        # If we have stored continuous values, use them
        if "phenotype_values_continuous" in data["gene"]:
            data["gene"]["phenotype_values"] = data["gene"][
                "phenotype_values_continuous"
            ].clone()
            data["gene"].pop("phenotype_binned", None)
            return data

        if "phenotype_values" not in data["gene"] or not data["gene"].get(
            "phenotype_binned", False
        ):
            return data

        # Extract binned data
        binned_values = data["gene"]["phenotype_values"]
        binned_type_indices = data["gene"]["phenotype_type_indices"]
        binned_sample_indices = data["gene"]["phenotype_sample_indices"]
        phenotype_types = data["gene"]["phenotype_types"]
        bin_info = data["gene"].get("phenotype_bin_info", {})

        # Results storage for one value per unique (sample, base_type) pair
        cont_values = []
        cont_type_indices = []
        cont_sample_indices = []

        # Get all unique sample indices
        unique_samples = torch.unique(binned_sample_indices).tolist()

        for sample_idx in unique_samples:
            # Get all values for this sample
            sample_mask = binned_sample_indices == sample_idx
            sample_type_indices = binned_type_indices[sample_mask]
            sample_values = binned_values[sample_mask]

            # Get base type indices (actual phenotype type, without bin encoding)
            base_type_indices = sample_type_indices // 1000
            unique_base_types = torch.unique(base_type_indices).tolist()

            for base_type_idx in unique_base_types:
                # Skip invalid types
                if base_type_idx >= len(phenotype_types):
                    continue

                # Get the phenotype type name(s)
                if isinstance(phenotype_types[base_type_idx], list):
                    type_labels = phenotype_types[base_type_idx]
                else:
                    type_labels = [phenotype_types[base_type_idx]]

                # Process each label in this phenotype type
                for label in type_labels:
                    if label not in self.label_configs or label not in bin_info:
                        continue

                    # Get bin information
                    label_info = bin_info[label]
                    bin_edges = torch.tensor(
                        label_info["bin_edges"], device=binned_values.device
                    )
                    label_type = label_info["label_type"]

                    # Get values for this specific base type
                    base_mask = base_type_indices == base_type_idx
                    type_mask = sample_mask & base_mask
                    label_values = sample_values[type_mask]
                    label_bin_indices = binned_type_indices[type_mask] % 1000

                    # Skip if no values found
                    if len(label_values) == 0:
                        continue

                    # Process based on label type (categorical, soft, ordinal)
                    cont_value = None

                    if label_type == "ordinal":
                        # Ordinal: Count the positive thresholds
                        # First sort by bin index to ensure proper order
                        sort_idx = torch.argsort(label_bin_indices)
                        sorted_bin_indices = label_bin_indices[sort_idx].long()
                        sorted_values = label_values[sort_idx]

                        # Count thresholds
                        num_thresholds = len(sorted_values)
                        crossings = sum(v > 0.5 for v in sorted_values)

                        # Sample from appropriate bin
                        if crossings < len(bin_edges) - 1:
                            low, high = bin_edges[crossings], bin_edges[crossings + 1]
                            cont_value = (
                                torch.rand(1, device=binned_values.device)
                                * (high - low)
                                + low
                            )

                    else:  # categorical or soft
                        num_bins = len(bin_edges) - 1
                        full_bins = torch.zeros(num_bins, device=binned_values.device)

                        # Fill in the bin values
                        for i, idx in enumerate(label_bin_indices):
                            idx_val = idx.long().item()  # Convert to long and then item
                            if 0 <= idx_val < num_bins:
                                full_bins[idx_val] = label_values[i]

                        if label_type == "categorical":
                            # Find max bin
                            max_bin = torch.argmax(full_bins).item()
                            low, high = bin_edges[max_bin], bin_edges[max_bin + 1]
                            cont_value = (
                                torch.rand(1, device=binned_values.device)
                                * (high - low)
                                + low
                            )
                        else:  # soft
                            # Weighted average
                            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
                            weights = torch.softmax(full_bins, dim=0)
                            cont_value = torch.sum(weights * bin_centers).reshape(1)

                    # Append the computed value
                    if cont_value is not None:
                        cont_values.append(cont_value)
                        cont_type_indices.append(
                            torch.tensor([base_type_idx], device=binned_values.device)
                        )
                        cont_sample_indices.append(
                            torch.tensor([sample_idx], device=binned_values.device)
                        )

        # Combine all values into the new COO format
        if cont_values:
            data["gene"]["phenotype_values"] = torch.cat(cont_values)
            data["gene"]["phenotype_type_indices"] = torch.cat(cont_type_indices)
            data["gene"]["phenotype_sample_indices"] = torch.cat(cont_sample_indices)
            data["gene"].pop("phenotype_binned", None)

        return data


class COOInverseCompose(BaseTransform):
    """A transform that applies the inverse of a sequence of transforms in reverse order
    for data in COO format."""

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
        data = copy.copy(data)
        for t in reversed(self.transforms):
            data = t.inverse(data)
        return data

    def __repr__(self) -> str:
        args = [f"\n  {t}" for t in self.transforms]
        return f'{self.__class__.__name__}({"".join(args)}\n)'
