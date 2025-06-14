# torchcell/transforms/coo_regression_to_classification
# [[torchcell.transforms.coo_regression_to_classification]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/transforms/coo_regression_to_classification
# Test file: tests/torchcell/transforms/test_coo_regression_to_classification.py

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional, Union
from torch_geometric.data import HeteroData, Batch
from torch_geometric.transforms import BaseTransform
from abc import ABC, abstractmethod
import copy
from torch_geometric.transforms import Compose


class COOLabelNormalizationTransform(BaseTransform):
    """Transform for normalizing labels in COO format with different strategies."""

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

    def forward(self, data: Union[HeteroData, Batch]) -> Union[HeteroData, Batch]:
        """Transform the data by normalizing specified labels in COO format."""
        # Check if we have phenotype data in COO format
        if not hasattr(data["gene"], "phenotype_values"):
            return data

        # Get phenotype types - handle batch case where it might be a list of lists
        phenotype_types = data["gene"].phenotype_types
        if isinstance(phenotype_types, list) and phenotype_types and isinstance(phenotype_types[0], list):
            # In batch mode, all items should have the same phenotype types
            phenotype_types = phenotype_types[0]

        # Clone the phenotype values to ensure we can modify them
        new_phenotype_values = data["gene"].phenotype_values.clone()
        
        # Store original values if not already stored
        if not hasattr(data["gene"], "phenotype_values_original"):
            data["gene"].phenotype_values_original = data["gene"].phenotype_values.clone()

        # Process each configured label
        for label, config in self.label_configs.items():
            if label not in phenotype_types:
                continue

            # Find indices where this phenotype appears
            label_idx = phenotype_types.index(label)
            mask = data["gene"].phenotype_type_indices == label_idx

            if mask.sum() == 0:
                continue

            # Get values for this phenotype
            values = new_phenotype_values[mask]

            # Normalize and update values
            normalized_values = self.normalize(values, label)
            new_phenotype_values[mask] = normalized_values

        # Update the data with normalized values
        data["gene"].phenotype_values = new_phenotype_values

        return data

    def inverse(self, data: Union[HeteroData, Batch]) -> Union[HeteroData, Batch]:
        """Inverse transform to recover original scale."""
        # Check if we have phenotype data in COO format
        if not hasattr(data["gene"], "phenotype_values"):
            return data

        # Get phenotype types - handle batch case where it might be a list of lists
        phenotype_types = data["gene"].phenotype_types
        if isinstance(phenotype_types, list) and phenotype_types and isinstance(phenotype_types[0], list):
            # In batch mode, all items should have the same phenotype types
            phenotype_types = phenotype_types[0]

        # Clone the phenotype values to ensure we can modify them
        new_phenotype_values = data["gene"].phenotype_values.clone()

        # Process each configured label
        for label in self.label_configs:
            if label not in phenotype_types:
                continue

            # Find indices where this phenotype appears
            label_idx = phenotype_types.index(label)
            mask = data["gene"].phenotype_type_indices == label_idx

            if mask.sum() == 0:
                continue

            # Get values for this phenotype
            values = new_phenotype_values[mask]

            # Denormalize and update values
            denormalized_values = self.denormalize(values, label)
            new_phenotype_values[mask] = denormalized_values

        # Update the data with denormalized values
        data["gene"].phenotype_values = new_phenotype_values

        return data


### Binning strategies adapted for COO format

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
    def __init__(
        self,
        dataset: "Neo4jCellDataset",
        label_configs: Dict[str, Dict],
        normalizer: Optional[COOLabelNormalizationTransform] = None,
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
                    # Create a mock data object to use normalizer
                    mock_data = HeteroData()
                    mock_data["gene"].phenotype_values = torch.tensor(df[label].values, dtype=torch.float)
                    mock_data["gene"].phenotype_type_indices = torch.zeros(len(df[label]), dtype=torch.long)
                    mock_data["gene"].phenotype_types = [label]
                    normalized_data = self.normalizer(mock_data)
                    normalized_df[label] = normalized_data["gene"]["phenotype_values"].numpy()
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
                # Create a mock data object
                mock_data = HeteroData()
                mock_data["gene"].phenotype_values = bin_edges_normalized
                mock_data["gene"].phenotype_type_indices = torch.zeros(len(bin_edges_normalized), dtype=torch.long)
                mock_data["gene"].phenotype_types = [label]
                # Apply inverse transform
                temp_data = self.normalizer.inverse(mock_data)
                self.label_metadata[label]["bin_edges_denormalized"] = temp_data[
                    "gene"
                ]["phenotype_values"].numpy()

    def forward(self, data: Union[HeteroData, Batch]) -> Union[HeteroData, Batch]:
        """Transform the data by binning specified labels in COO format."""
        # Check if we have phenotype data in COO format
        if not hasattr(data["gene"], "phenotype_values"):
            return data

        # Get phenotype types - handle batch case where it might be a list of lists
        phenotype_types = data["gene"].phenotype_types
        if isinstance(phenotype_types, list) and phenotype_types and isinstance(phenotype_types[0], list):
            # In batch mode, all items should have the same phenotype types
            phenotype_types = phenotype_types[0]

        # We need to handle the binning differently for COO format
        # Since binning changes the dimensionality, we'll need to reorganize the data
        
        # Collect all phenotype data
        all_values = []
        all_type_indices = []
        all_sample_indices = []
        
        for label, config in self.label_configs.items():
            if label not in phenotype_types:
                continue

            # Find indices where this phenotype appears
            label_idx = phenotype_types.index(label)
            mask = data["gene"].phenotype_type_indices == label_idx

            if mask.sum() == 0:
                continue

            # Get values and indices for this phenotype
            values = data["gene"].phenotype_values[mask]
            sample_indices = data["gene"].phenotype_sample_indices[mask]
            
            # Get binning parameters
            bin_edges = torch.tensor(self.label_metadata[label]["bin_edges"])
            strategy = self.strategies[config["strategy"]]

            # Store continuous values if requested
            if config.get("store_continuous", True):
                if not hasattr(data["gene"], f"{label}_continuous"):
                    data["gene"][f"{label}_continuous"] = values

            label_type = config.get("label_type", "categorical").lower()
            if label_type == "categorical":
                binned_values = strategy.compute_onehot_labels(values, bin_edges)
            elif label_type == "soft":
                sigma = config.get("sigma", 3)
                binned_values = strategy.compute_soft_labels(
                    values, bin_edges, strategy, sigma
                )
            elif label_type == "ordinal":
                binned_values = strategy.compute_ordinal_labels(values, bin_edges)

            # For binned data, we need to expand the indices
            num_bins = binned_values.shape[1]
            for i in range(len(values)):
                for j in range(num_bins):
                    all_values.append(binned_values[i, j])
                    all_type_indices.append(label_idx * num_bins + j)
                    all_sample_indices.append(sample_indices[i])

        # Update the data with binned values
        if all_values:
            data["gene"].phenotype_values = torch.stack(all_values)
            data["gene"].phenotype_type_indices = torch.tensor(all_type_indices, dtype=torch.long)
            data["gene"].phenotype_sample_indices = torch.tensor(all_sample_indices, dtype=torch.long)
            
            # Update phenotype types to include bin information
            new_phenotype_types = []
            for label, config in self.label_configs.items():
                if label in phenotype_types:
                    label_type = config.get("label_type", "categorical").lower()
                    bin_edges = self.label_metadata[label]["bin_edges"]
                    num_bins = len(bin_edges) - 1
                    for i in range(num_bins):
                        new_phenotype_types.append(f"{label}_bin_{i}")
            data["gene"].phenotype_types = new_phenotype_types

        return data

    def inverse(self, data: Union[HeteroData, Batch], seed: int = 42) -> Union[HeteroData, Batch]:
        """Inverse transform to recover continuous values from binned COO format."""
        torch.manual_seed(seed)
        # Check if we have phenotype data in COO format
        if not hasattr(data["gene"], "phenotype_values"):
            return data

        # For inverse transform, we need to reconstruct continuous values from bins
        # This is complex for COO format since we need to aggregate bin information
        
        # First, identify which phenotypes are binned
        phenotype_types = data["gene"].phenotype_types
        if isinstance(phenotype_types, list) and phenotype_types and isinstance(phenotype_types[0], list):
            # In batch mode, all items should have the same phenotype types
            phenotype_types = phenotype_types[0]
        
        # Group by original phenotype (before binning)
        phenotype_groups = {}
        for i, ptype in enumerate(phenotype_types):
            # Extract original phenotype name from binned name
            for label in self.label_configs:
                if ptype.startswith(f"{label}_bin_"):
                    if label not in phenotype_groups:
                        phenotype_groups[label] = []
                    phenotype_groups[label].append(i)
                    break
        
        # Reconstruct continuous values
        new_values = []
        new_type_indices = []
        new_sample_indices = []
        new_phenotype_types = []
        
        for label, config in self.label_configs.items():
            if label not in phenotype_groups:
                continue
                
            label_type = config.get("label_type", "categorical").lower()
            bin_edges = torch.tensor(
                self.label_metadata[label]["bin_edges"],
                device=data["gene"].phenotype_values.device,
                dtype=torch.float32,
            )
            
            # Get unique sample indices for this phenotype
            bin_indices = phenotype_groups[label]
            sample_indices_set = set()
            for bin_idx in bin_indices:
                mask = data["gene"].phenotype_type_indices == bin_idx
                sample_indices_set.update(data["gene"].phenotype_sample_indices[mask].tolist())
            
            # Process each sample
            for sample_idx in sorted(sample_indices_set):
                # Collect bin values for this sample
                bin_values = []
                for bin_idx in bin_indices:
                    mask = (data["gene"].phenotype_type_indices == bin_idx) & \
                           (data["gene"].phenotype_sample_indices == sample_idx)
                    if mask.sum() > 0:
                        bin_values.append(data["gene"].phenotype_values[mask][0])
                    else:
                        bin_values.append(torch.tensor(0.0))
                
                if not bin_values:
                    continue
                    
                bin_values = torch.stack(bin_values)
                
                # Check for NaN
                if torch.isnan(bin_values).any():
                    continuous_value = float('nan')
                else:
                    # Reconstruct continuous value based on label type
                    if label_type == "ordinal":
                        # Count number of 1s to determine bin
                        crossings = torch.sum(bin_values > 0.5)
                        bin_idx = crossings.item()
                        low, high = bin_edges[bin_idx], bin_edges[bin_idx + 1]
                        continuous_value = torch.rand(1, device=bin_edges.device) * (high - low) + low
                        continuous_value = continuous_value.item()
                    
                    elif label_type == "soft":
                        # Use weighted average based on soft labels
                        window_size = 2
                        probs = torch.softmax(bin_values, dim=-1)
                        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
                        max_prob_bin = torch.argmax(probs)
                        
                        start_idx = max(0, max_prob_bin - window_size)
                        end_idx = min(len(bin_centers), max_prob_bin + window_size + 1)
                        
                        if (end_idx - start_idx) < (2 * window_size + 1):
                            continuous_value = bin_centers[max_prob_bin].item()
                        else:
                            window_probs = probs[start_idx:end_idx]
                            window_centers = bin_centers[start_idx:end_idx]
                            window_probs = window_probs / window_probs.sum()
                            continuous_value = (window_probs * window_centers).sum().item()
                    
                    elif label_type == "categorical":
                        # Use argmax to find bin
                        probs = torch.softmax(bin_values, dim=-1)
                        bin_idx = torch.argmax(probs)
                        low, high = bin_edges[bin_idx], bin_edges[bin_idx + 1]
                        continuous_value = torch.rand(1, device=bin_edges.device) * (high - low) + low
                        continuous_value = continuous_value.item()
                
                new_values.append(continuous_value)
                new_type_indices.append(len(new_phenotype_types))
                new_sample_indices.append(sample_idx)
            
            if label not in new_phenotype_types:
                new_phenotype_types.append(label)
        
        # Update data with reconstructed continuous values
        if new_values:
            data["gene"].phenotype_values = torch.tensor(new_values, dtype=torch.float)
            data["gene"].phenotype_type_indices = torch.tensor(new_type_indices, dtype=torch.long)
            data["gene"].phenotype_sample_indices = torch.tensor(new_sample_indices, dtype=torch.long)
            data["gene"].phenotype_types = new_phenotype_types

        return data

    def get_bin_info(self, label: str) -> dict:
        """Get binning information for a label."""
        if label not in self.label_metadata:
            raise ValueError(f"No binning metadata found for label {label}")
        return self.label_metadata[label]


class COOInverseCompose(BaseTransform):
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
        # Apply transforms in reverse order
        for t in reversed(self.transforms):
            data = t.inverse(data)
        return data

    def __repr__(self) -> str:
        args = [f"\n  {t}" for t in self.transforms]
        return f'{self.__class__.__name__}({"".join(args)}\n)'