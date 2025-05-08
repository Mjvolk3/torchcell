# tests/torchcell/transforms/test_regression_to_classification_coo
# [[tests.torchcell.transforms.test_regression_to_classification_coo]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/transforms/test_regression_to_classification_coo

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import Compose

from torchcell.transforms.regression_to_classification_coo import (
    COOInverseCompose,
    COOLabelBinningTransform,
    COOLabelNormalizationTransform,
)


class TestCOOLabelNormalizationTransform:
    @pytest.fixture
    def mock_coo_dataset(self):
        """Create a mock dataset with COO format phenotype data."""

        class MockCOODataset:
            def __init__(self):
                self.data = []

                # Create some test data samples
                for i in range(3):
                    data = HeteroData()
                    data["gene"] = {}

                    # Sample 1: gene_interaction values
                    if i == 0:
                        data["gene"]["phenotype_values"] = torch.tensor(
                            [-1.0, 0.0, 1.0]
                        )
                        data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0])
                        data["gene"]["phenotype_sample_indices"] = torch.tensor(
                            [0, 0, 0]
                        )
                        data["gene"]["phenotype_types"] = ["gene_interaction"]

                    # Sample 2: fitness values
                    elif i == 1:
                        data["gene"]["phenotype_values"] = torch.tensor(
                            [0.0, 0.5, 1.0, 2.0]
                        )
                        data["gene"]["phenotype_type_indices"] = torch.tensor(
                            [0, 0, 0, 0]
                        )
                        data["gene"]["phenotype_sample_indices"] = torch.tensor(
                            [0, 0, 0, 0]
                        )
                        data["gene"]["phenotype_types"] = ["fitness"]

                    # Sample 3: mixed phenotypes
                    else:
                        data["gene"]["phenotype_values"] = torch.tensor(
                            [-0.5, 0.5, 1.5]
                        )
                        data["gene"]["phenotype_type_indices"] = torch.tensor([0, 1, 1])
                        data["gene"]["phenotype_sample_indices"] = torch.tensor(
                            [0, 0, 0]
                        )
                        data["gene"]["phenotype_types"] = [
                            "gene_interaction",
                            "fitness",
                        ]

                    self.data.append(data)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return MockCOODataset()

    @pytest.fixture
    def norm_transform(self, mock_coo_dataset):
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return COOLabelNormalizationTransform(mock_coo_dataset, label_configs)

    def test_coo_minmax_normalization(self, norm_transform, mock_coo_dataset):
        # Get a sample with fitness values
        data = mock_coo_dataset[1]

        # Save original values for comparison
        original_values = data["gene"]["phenotype_values"].clone()

        # Apply normalization
        normalized = norm_transform(data)

        # Get stats directly from transform for verification
        stats = norm_transform.stats["fitness"]
        min_val, max_val = stats["min"], stats["max"]

        # Calculate expected values manually
        expected = (original_values - min_val) / (max_val - min_val)

        # Check that normalization was applied correctly
        assert torch.allclose(
            normalized["gene"]["phenotype_values"], expected, atol=1e-5
        )

        # Test inverse transform
        denormalized = norm_transform.inverse(normalized)
        assert torch.allclose(
            denormalized["gene"]["phenotype_values"], original_values, atol=1e-5
        )

    def test_coo_standard_normalization(self, norm_transform, mock_coo_dataset):
        # Get a sample with gene_interaction values
        data = mock_coo_dataset[0]

        # Save original values for comparison
        original_values = data["gene"]["phenotype_values"].clone()

        # Apply normalization
        normalized = norm_transform(data)

        # Get stats directly from transform for verification
        stats = norm_transform.stats["gene_interaction"]
        mean, std = stats["mean"], stats["std"]

        # Calculate expected values manually
        expected = (original_values - mean) / std

        # Check that normalization was applied correctly - use looser tolerance for float precision
        assert torch.allclose(
            normalized["gene"]["phenotype_values"], expected, atol=1e-4
        )

        # Test inverse transform
        denormalized = norm_transform.inverse(normalized)
        assert torch.allclose(
            denormalized["gene"]["phenotype_values"], original_values, atol=1e-4
        )

    def test_coo_mixed_phenotypes(self, norm_transform, mock_coo_dataset):
        # Get a sample with mixed phenotype types
        data = mock_coo_dataset[2]

        # Save original values for comparison
        original_values = data["gene"]["phenotype_values"].clone()
        original_types = data["gene"]["phenotype_type_indices"].clone()

        # Apply normalization
        normalized = norm_transform(data)

        # Verify gene_interaction value (index 0) using stats
        gi_stats = norm_transform.stats["gene_interaction"]
        gi_value = original_values[original_types == 0]
        expected_gi = (gi_value - gi_stats["mean"]) / gi_stats["std"]
        assert torch.allclose(
            normalized["gene"]["phenotype_values"][original_types == 0],
            expected_gi,
            atol=1e-4,
        )

        # Verify fitness values (indices 1-2) using stats
        fitness_stats = norm_transform.stats["fitness"]
        fitness_values = original_values[original_types == 1]
        expected_fit = (fitness_values - fitness_stats["min"]) / (
            fitness_stats["max"] - fitness_stats["min"]
        )
        assert torch.allclose(
            normalized["gene"]["phenotype_values"][original_types == 1],
            expected_fit,
            atol=1e-4,
        )

        # Test inverse transform
        denormalized = norm_transform.inverse(normalized)
        assert torch.allclose(
            denormalized["gene"]["phenotype_values"], original_values, atol=1e-4
        )


class TestCOOLabelNormalizationInverse:
    @pytest.fixture
    def mock_coo_dataset(self):
        """Create a mock dataset with COO format phenotype data."""

        class MockCOODataset:
            def __init__(self):
                self.data = []

                # Create test data
                data = HeteroData()
                data["gene"] = {}
                data["gene"]["phenotype_values"] = torch.tensor(
                    [-1.0, 0.0, 1.0, 0.0, 0.5, 1.0, 2.0]
                )
                data["gene"]["phenotype_type_indices"] = torch.tensor(
                    [0, 0, 0, 1, 1, 1, 1]
                )
                data["gene"]["phenotype_sample_indices"] = torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6]
                )
                data["gene"]["phenotype_types"] = ["gene_interaction", "fitness"]

                self.data.append(data)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return MockCOODataset()

    @pytest.fixture
    def norm_transform(self, mock_coo_dataset):
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return COOLabelNormalizationTransform(mock_coo_dataset, label_configs)

    def test_coo_inverse_minmax(self, norm_transform):
        # Create test data in COO format
        data = HeteroData()
        data["gene"] = {}
        data["gene"]["phenotype_values"] = torch.tensor([0.0, 0.25, 0.5, 1.0])
        data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0])
        data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3])
        data["gene"]["phenotype_types"] = ["fitness"]

        # Expected denormalized values
        stats = norm_transform.stats["fitness"]
        min_val, max_val = stats["min"], stats["max"]
        expected = data["gene"]["phenotype_values"] * (max_val - min_val) + min_val

        # Apply inverse transform
        denormalized = norm_transform.inverse(data)

        # Compare results
        assert torch.allclose(
            denormalized["gene"]["phenotype_values"], expected, atol=1e-5
        )

    def test_coo_inverse_standard(self, norm_transform):
        # Create test data in COO format
        data = HeteroData()
        data["gene"] = {}

        # Get stats to create normalized values
        stats = norm_transform.stats["gene_interaction"]
        mean, std = stats["mean"], stats["std"]

        # Create already normalized values
        normalized_values = torch.tensor(
            [-1.5, 0.0, 1.5]
        )  # Assume these are (orig - mean) / std

        data["gene"]["phenotype_values"] = normalized_values
        data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0])
        data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2])
        data["gene"]["phenotype_types"] = ["gene_interaction"]

        # Expected denormalized values
        expected = normalized_values * std + mean

        # Apply inverse transform
        denormalized = norm_transform.inverse(data)

        # Compare results
        assert torch.allclose(
            denormalized["gene"]["phenotype_values"], expected, atol=1e-5
        )

    def test_coo_inverse_with_nans(self, norm_transform):
        # Create test data in COO format with NaNs
        data = HeteroData()
        data["gene"] = {}
        data["gene"]["phenotype_values"] = torch.tensor([0.0, 0.25, 0.5, float("nan")])
        data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0])
        data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3])
        data["gene"]["phenotype_types"] = ["fitness"]

        # Apply inverse transform
        denormalized = norm_transform.inverse(data)

        # Check NaN is preserved
        assert torch.isnan(denormalized["gene"]["phenotype_values"][3])

        # Check other values are properly denormalized
        stats = norm_transform.stats["fitness"]
        min_val, max_val = stats["min"], stats["max"]
        expected = data["gene"]["phenotype_values"][:3] * (max_val - min_val) + min_val

        assert torch.allclose(
            denormalized["gene"]["phenotype_values"][:3], expected, atol=1e-5
        )


class TestCOOLabelNormalizationRoundTrip:
    @pytest.fixture
    def mock_coo_dataset(self):
        """Create a mock dataset with COO format phenotype data."""

        class MockCOODataset:
            def __init__(self):
                self.data = []

                # Create test data
                data = HeteroData()
                data["gene"] = {}
                data["gene"]["phenotype_values"] = torch.tensor(
                    [-1.0, 0.0, 1.0, float("nan"), 0.0, 0.5, 1.0, 2.0]
                )
                data["gene"]["phenotype_type_indices"] = torch.tensor(
                    [0, 0, 0, 0, 1, 1, 1, 1]
                )
                data["gene"]["phenotype_sample_indices"] = torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7]
                )
                data["gene"]["phenotype_types"] = ["gene_interaction", "fitness"]

                self.data.append(data)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return MockCOODataset()

    @pytest.fixture
    def norm_transform(self, mock_coo_dataset):
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return COOLabelNormalizationTransform(mock_coo_dataset, label_configs)

    def test_coo_round_trip_minmax(self, norm_transform, mock_coo_dataset):
        # Extract test data
        data = mock_coo_dataset[0]

        # Extract just the fitness values
        fitness_mask = data["gene"]["phenotype_type_indices"] == 1
        fitness_values = data["gene"]["phenotype_values"][fitness_mask].clone()
        fitness_indices = data["gene"]["phenotype_type_indices"][fitness_mask].clone()
        fitness_samples = data["gene"]["phenotype_sample_indices"][fitness_mask].clone()

        # Create a new data object with just fitness values
        fitness_data = HeteroData()
        fitness_data["gene"] = {
            "phenotype_values": fitness_values,
            "phenotype_type_indices": fitness_indices,
            "phenotype_sample_indices": fitness_samples,
            "phenotype_types": ["fitness"],
        }

        # Apply normalization
        normalized = norm_transform(fitness_data)

        # Apply inverse transform
        recovered = norm_transform.inverse(normalized)

        # Check we get back the original values
        assert torch.allclose(
            recovered["gene"]["phenotype_values"], fitness_values, atol=1e-5
        )

    def test_coo_round_trip_standard(self, norm_transform, mock_coo_dataset):
        # Extract test data
        data = mock_coo_dataset[0]

        # Extract just the gene_interaction values
        gi_mask = data["gene"]["phenotype_type_indices"] == 0
        gi_values = data["gene"]["phenotype_values"][gi_mask].clone()
        gi_indices = data["gene"]["phenotype_type_indices"][gi_mask].clone()
        gi_samples = data["gene"]["phenotype_sample_indices"][gi_mask].clone()

        # Create a new data object with just gene_interaction values
        gi_data = HeteroData()
        gi_data["gene"] = {
            "phenotype_values": gi_values,
            "phenotype_type_indices": gi_indices,
            "phenotype_sample_indices": gi_samples,
            "phenotype_types": ["gene_interaction"],
        }

        # Apply normalization
        normalized = norm_transform(gi_data)

        # Apply inverse transform
        recovered = norm_transform.inverse(normalized)

        # Check we get back the original values, allowing for NaN comparisons
        assert torch.allclose(
            recovered["gene"]["phenotype_values"], gi_values, equal_nan=True, atol=1e-5
        )


class TestCOOLabelBinningTransform:
    @pytest.fixture
    def mock_coo_dataset(self):
        """Create a mock dataset with COO format phenotype data."""

        class MockCOODataset:
            def __init__(self):
                self.data = []

                # Create test data with more samples for binning
                # Sample 1: fitness values with good distribution
                data1 = HeteroData()
                data1["gene"] = {}
                data1["gene"]["phenotype_values"] = torch.tensor(
                    list(np.linspace(0, 1, 50)) + list(np.linspace(0, 1, 50))
                )
                data1["gene"]["phenotype_type_indices"] = torch.tensor([0] * 100)
                data1["gene"]["phenotype_sample_indices"] = torch.tensor(
                    list(range(100))
                )
                data1["gene"]["phenotype_types"] = ["fitness"]

                # Sample 2: gene_interaction values
                data2 = HeteroData()
                data2["gene"] = {}
                data2["gene"]["phenotype_values"] = torch.tensor(
                    list(np.linspace(-1, 1, 100))
                )
                data2["gene"]["phenotype_type_indices"] = torch.tensor([0] * 100)
                data2["gene"]["phenotype_sample_indices"] = torch.tensor(
                    list(range(100))
                )
                data2["gene"]["phenotype_types"] = ["gene_interaction"]

                self.data = [data1, data2]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return MockCOODataset()

    @pytest.fixture
    def bin_transform(self, mock_coo_dataset):
        label_configs = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 5,
                "label_type": "categorical",
            },
            "gene_interaction": {
                "strategy": "equal_frequency",
                "num_bins": 5,
                "label_type": "soft",
                "sigma": 0.5,
            },
        }
        return COOLabelBinningTransform(mock_coo_dataset, label_configs)

    def test_coo_categorical_binning(self, bin_transform, mock_coo_dataset):
        # Select a subset of fitness data for testing
        test_data = HeteroData()
        test_data["gene"] = {}
        test_data["gene"]["phenotype_values"] = torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 1.0]
        )
        test_data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0, 0])
        test_data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3, 4])
        test_data["gene"]["phenotype_types"] = ["fitness"]

        # Apply binning
        binned = bin_transform(test_data)

        # We should have 5 bins for each of the 5 values, resulting in 25 values in COO format
        assert "phenotype_values" in binned["gene"]
        assert binned["gene"]["phenotype_values"].shape[0] == 25

        # Test inverse transform
        recovered = bin_transform.inverse(binned)

        # Should get back 5 values
        assert recovered["gene"]["phenotype_values"].shape[0] == 5

        # Each value should be in the same bin as original
        bin_edges = torch.tensor(bin_transform.label_metadata["fitness"]["bin_edges"])

        for i, val in enumerate(test_data["gene"]["phenotype_values"]):
            # Find bin for original value safely
            bin_idx = torch.searchsorted(bin_edges, val, right=True) - 1
            bin_idx = torch.clamp(bin_idx, 0, len(bin_edges) - 2)  # Ensure valid range
            bin_start, bin_end = bin_edges[bin_idx], bin_edges[bin_idx + 1]

            # Get recovered value
            rec_val = recovered["gene"]["phenotype_values"][i]

            # Check recovered value is in same bin
            assert (
                bin_start <= rec_val <= bin_end
            ), f"Value {rec_val} not in [{bin_start}, {bin_end}]"

    def test_coo_soft_binning(self, bin_transform, mock_coo_dataset):
        # Select a subset of gene_interaction data for testing
        test_data = HeteroData()
        test_data["gene"] = {}
        test_data["gene"]["phenotype_values"] = torch.tensor(
            [-1.0, -0.5, 0.0, 0.5, 1.0]
        )
        test_data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0, 0])
        test_data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3, 4])
        test_data["gene"]["phenotype_types"] = ["gene_interaction"]

        # Apply binning
        binned = bin_transform(test_data)

        # We should have 5 bins for each of the 5 values
        assert "phenotype_values" in binned["gene"]
        assert binned["gene"]["phenotype_values"].shape[0] == 25

        # Test inverse transform
        recovered = bin_transform.inverse(binned)

        # Should get back 5 values
        assert recovered["gene"]["phenotype_values"].shape[0] == 5

        # Values should be ordered from smallest to largest, matching original order
        values_sorted = torch.sort(recovered["gene"]["phenotype_values"])[0]
        assert torch.allclose(
            values_sorted, recovered["gene"]["phenotype_values"], atol=1e-4
        )


class TestCOOOrdinalBinning:
    @pytest.fixture
    def mock_coo_dataset(self):
        """Create a mock dataset with COO format phenotype data."""

        class MockCOODataset:
            def __init__(self):
                self.data = []

                # Create test data with evenly distributed fitness values
                data = HeteroData()
                data["gene"] = {}
                data["gene"]["phenotype_values"] = torch.tensor(
                    list(np.linspace(0, 1, 100))
                )
                data["gene"]["phenotype_type_indices"] = torch.tensor([0] * 100)
                data["gene"]["phenotype_sample_indices"] = torch.tensor(
                    list(range(100))
                )
                data["gene"]["phenotype_types"] = ["fitness"]

                self.data = [data]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return MockCOODataset()

    @pytest.fixture
    def bin_transform(self, mock_coo_dataset):
        label_configs = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,  # Using 4 bins for simpler testing
                "label_type": "ordinal",
            }
        }
        return COOLabelBinningTransform(mock_coo_dataset, label_configs)

    def test_coo_ordinal_forward(self, bin_transform):
        # Create test data
        test_data = HeteroData()
        test_data["gene"] = {}
        test_data["gene"]["phenotype_values"] = torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 1.0]
        )
        test_data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0, 0])
        test_data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3, 4])
        test_data["gene"]["phenotype_types"] = ["fitness"]

        # Apply binning
        binned = bin_transform(test_data)

        # We should have 3 thresholds for each value (4 bins = 3 thresholds)
        # This means 15 values total (5 samples * 3 thresholds)
        assert binned["gene"]["phenotype_values"].shape[0] == 15

        # Reconstruct ordinal labels for checking
        num_samples = 5
        num_thresholds = 3
        reconstructed = torch.zeros(num_samples, num_thresholds)

        # Group by sample indices
        unique_samples = torch.unique(
            binned["gene"]["phenotype_sample_indices"]
        ).tolist()

        for i, sample_idx in enumerate(unique_samples):
            sample_mask = binned["gene"]["phenotype_sample_indices"] == sample_idx
            sample_values = binned["gene"]["phenotype_values"][sample_mask]

            # The bit indices are encoded in the type indices
            base_type = binned["gene"]["phenotype_type_indices"][sample_mask] // 1000
            bit_indices = binned["gene"]["phenotype_type_indices"][sample_mask] % 1000

            # Sort by bit index
            sorted_idx = torch.argsort(bit_indices)
            bit_indices = bit_indices[sorted_idx].long()
            sample_values = sample_values[sorted_idx]

            # Populate reconstructed matrix
            for j, bit_idx in enumerate(bit_indices):
                reconstructed[i, bit_idx] = sample_values[j]

        # Check ordinal property: if a higher threshold is 1, all lower ones should be 1
        for i in range(num_samples):
            for j in range(1, num_thresholds):
                if reconstructed[i, j] > 0.5:  # Assuming 0.5 is threshold
                    assert torch.all(
                        reconstructed[i, :j] > 0.5
                    ), f"Ordinal property violated at {i}, {j}"

    def test_coo_ordinal_inverse(self, bin_transform):
        # Create test data with ordinal labels in COO format
        num_samples = 4
        num_thresholds = 3

        # Generate the encoded data directly
        values = []
        type_indices = []
        sample_indices = []

        # Sample 0: [0, 0, 0] - lowest bin
        # Sample 1: [1, 0, 0] - second bin
        # Sample 2: [1, 1, 0] - third bin
        # Sample 3: [1, 1, 1] - highest bin
        sample_bits = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]

        for sample_idx in range(num_samples):
            for bit_idx in range(num_thresholds):
                values.append(float(sample_bits[sample_idx][bit_idx]))
                # Encode both the base type (0) and bit position (bit_idx)
                type_indices.append(0 * 1000 + bit_idx)
                sample_indices.append(sample_idx)

        test_data = HeteroData()
        test_data["gene"] = {
            "phenotype_values": torch.tensor(values),
            "phenotype_type_indices": torch.tensor(type_indices),
            "phenotype_sample_indices": torch.tensor(sample_indices),
            "phenotype_types": ["fitness"],
            "phenotype_binned": True,  # Flag to indicate this is binned data
        }

        # Apply inverse transform
        recovered = bin_transform.inverse(test_data)

        # Should get back 4 values
        assert recovered["gene"]["phenotype_values"].shape[0] == 4

        # Get bin edges
        bin_edges = torch.tensor(bin_transform.label_metadata["fitness"]["bin_edges"])

        # Check each value is in the correct bin
        for i in range(num_samples):
            # Count crossings to determine bin
            crossings = sum(sample_bits[i])
            bin_start, bin_end = bin_edges[crossings], bin_edges[crossings + 1]

            # Get the recovered value
            rec_val = recovered["gene"]["phenotype_values"][i]

            # Check it's in the right bin
            assert (
                bin_start <= rec_val <= bin_end
            ), f"Value {rec_val} not in bin {crossings} range [{bin_start}, {bin_end}]"

    def test_coo_ordinal_with_nans(self, bin_transform):
        # Create test data with NaNs
        test_data = HeteroData()
        test_data["gene"] = {}
        test_data["gene"]["phenotype_values"] = torch.tensor(
            [0.0, float("nan"), 0.5, float("nan"), 1.0]
        )
        test_data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0, 0])
        test_data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3, 4])
        test_data["gene"]["phenotype_types"] = ["fitness"]

        # Apply binning
        binned = bin_transform(test_data)

        # Apply inverse transform
        recovered = bin_transform.inverse(binned)

        # Find which recovered indices correspond to original NaN indices
        sample_indices = recovered["gene"]["phenotype_sample_indices"]

        # Check that we have values for non-NaN samples
        assert torch.any(sample_indices == 0), "Missing value for sample 0"
        assert torch.any(sample_indices == 2), "Missing value for sample 2"
        assert torch.any(sample_indices == 4), "Missing value for sample 4"

        # Check for NaN samples - either they're preserved as NaN or skipped
        for i in [1, 3]:
            matches = (sample_indices == i).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                # If sample exists, check for NaN
                idx = matches[0].item()
                assert torch.isnan(
                    recovered["gene"]["phenotype_values"][idx]
                ), f"NaN not preserved for sample {i}"


class TestCOOOrdinalNormBinning:
    @pytest.fixture
    def mock_coo_dataset(self):
        """Create a mock dataset with COO format phenotype data."""

        class MockCOODataset:
            def __init__(self):
                self.data = []

                # Create test data
                data = HeteroData()
                data["gene"] = {}
                data["gene"]["phenotype_values"] = torch.tensor(
                    [-1.0, 0.0, 0.5, 1.0, 2.0, float("nan"), 1.5, float("nan")]
                )
                data["gene"]["phenotype_type_indices"] = torch.tensor(
                    [0, 0, 0, 0, 0, 0, 0, 0]
                )
                data["gene"]["phenotype_sample_indices"] = torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7]
                )
                data["gene"]["phenotype_types"] = ["fitness"]

                self.data.append(data)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return MockCOODataset()

    @pytest.fixture
    def transforms(self, mock_coo_dataset):
        norm_config = {"fitness": {"strategy": "minmax"}}
        norm_transform = COOLabelNormalizationTransform(mock_coo_dataset, norm_config)

        bin_config = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "ordinal",
            }
        }
        bin_transform = COOLabelBinningTransform(
            mock_coo_dataset, bin_config, normalizer=norm_transform
        )

        return [norm_transform, bin_transform]

    def test_coo_full_round_trip(self, transforms, mock_coo_dataset):
        """Test the full round trip through normalization and binning."""
        # Get transforms
        norm_transform, bin_transform = transforms

        # Get test data
        data = mock_coo_dataset[0]

        # Create composed transforms
        forward_transform = Compose(transforms)
        inverse_transform = COOInverseCompose(forward_transform)

        # Apply forward transform
        transformed = forward_transform(data)

        # Apply inverse transform
        recovered = inverse_transform(transformed)

        # Count NaNs in original and recovered data
        original_nans = torch.isnan(data["gene"]["phenotype_values"]).sum().item()
        recovered_nans = torch.isnan(recovered["gene"]["phenotype_values"]).sum().item()

        # Check NaNs are preserved
        assert original_nans == recovered_nans, "NaN count changed after transforms"

        # Check number of samples is preserved
        assert len(recovered["gene"]["phenotype_values"]) == len(
            data["gene"]["phenotype_values"]
        ), "Number of values changed after transforms"


class TestCOOModelOutputSimulation:
    @pytest.fixture
    def mock_coo_dataset(self):
        """Create a mock dataset with COO format phenotype data."""

        class MockCOODataset:
            def __init__(self):
                self.data = []

                # Create test data with evenly distributed fitness values
                data = HeteroData()
                data["gene"] = {}
                data["gene"]["phenotype_values"] = torch.tensor(
                    list(np.linspace(0, 1, 100))
                )
                data["gene"]["phenotype_type_indices"] = torch.tensor([0] * 100)
                data["gene"]["phenotype_sample_indices"] = torch.tensor(
                    list(range(100))
                )
                data["gene"]["phenotype_types"] = ["fitness"]

                self.data = [data]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return MockCOODataset()

    def test_coo_categorical_logits(self, mock_coo_dataset):
        """Test categorical classification with model-like logit outputs in COO format."""
        # Setup transforms
        norm_config = {"fitness": {"strategy": "minmax"}}
        bin_config = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "categorical",
            }
        }

        norm_transform = COOLabelNormalizationTransform(mock_coo_dataset, norm_config)
        bin_transform = COOLabelBinningTransform(
            mock_coo_dataset, bin_config, normalizer=norm_transform
        )

        # Create the fixture data with bin information for proper inverse
        bin_info = {
            "fitness": {
                "bin_edges": bin_transform.label_metadata["fitness"][
                    "bin_edges"
                ].tolist(),
                "label_type": "categorical",
                "num_bins": 4,
                "base_type_idx": 0,
            }
        }

        inverse_transform = COOInverseCompose([norm_transform, bin_transform])

        # Create model-output-like data in COO format
        batch_size = 4
        num_bins = 4

        # Each row (sample) predicts a different bin with high confidence
        values = []
        type_indices = []
        sample_indices = []

        for sample_idx in range(batch_size):
            for bin_idx in range(num_bins):
                # Put a high value in the correct bin, low elsewhere
                value = 10.0 if bin_idx == sample_idx else 0.0
                values.append(value)
                # Encode both the phenotype type (0) and bin position
                type_indices.append(0 * 1000 + bin_idx)
                sample_indices.append(sample_idx)

        pred_data = HeteroData()
        pred_data["gene"] = {
            "phenotype_values": torch.tensor(values),
            "phenotype_type_indices": torch.tensor(type_indices),
            "phenotype_sample_indices": torch.tensor(sample_indices),
            "phenotype_types": ["fitness"],
            "phenotype_binned": True,
            "phenotype_bin_info": bin_info,  # Add bin info
        }

        # Apply inverse transform
        recovered = inverse_transform(pred_data)

        # Should get back batch_size values
        assert recovered["gene"]["phenotype_values"].shape[0] == batch_size

    def test_coo_ordinal_logits(self, mock_coo_dataset):
        """Test ordinal classification with model-like logit outputs in COO format."""
        # Setup transforms
        norm_config = {"fitness": {"strategy": "minmax"}}
        bin_config = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "ordinal",
            }
        }

        norm_transform = COOLabelNormalizationTransform(mock_coo_dataset, norm_config)
        bin_transform = COOLabelBinningTransform(
            mock_coo_dataset, bin_config, normalizer=norm_transform
        )
        inverse_transform = COOInverseCompose([norm_transform, bin_transform])

        # Create model-output-like data in COO format
        batch_size = 4
        num_thresholds = 3  # 4 bins = 3 thresholds

        # Different threshold patterns for each sample
        threshold_patterns = [
            [-10.0, -10.0, -10.0],  # All negative = bin 0
            [10.0, -10.0, -10.0],  # First positive = bin 1
            [10.0, 10.0, -10.0],  # First two positive = bin 2
            [10.0, 10.0, 10.0],  # All positive = bin 3
        ]

        values = []
        type_indices = []
        sample_indices = []

        for sample_idx in range(batch_size):
            for thresh_idx in range(num_thresholds):
                values.append(threshold_patterns[sample_idx][thresh_idx])
                # Encode both the phenotype type (0) and threshold position
                type_indices.append(0 * 1000 + thresh_idx)
                sample_indices.append(sample_idx)

        pred_data = HeteroData()
        pred_data["gene"] = {
            "phenotype_values": torch.tensor(values),
            "phenotype_type_indices": torch.tensor(type_indices),
            "phenotype_sample_indices": torch.tensor(sample_indices),
            "phenotype_types": ["fitness"],
            "phenotype_binned": True,
        }

        # Apply inverse transform
        recovered = inverse_transform(pred_data)

        # Should get back batch_size values
        assert recovered["gene"]["phenotype_values"].shape[0] == batch_size

        # Get bin edges for verification
        norm_bin_edges = torch.tensor(
            bin_transform.label_metadata["fitness"]["bin_edges"]
        )

        # For each sample, check the value is in the right bin
        for sample_idx in range(batch_size):
            sample_mask = recovered["gene"]["phenotype_sample_indices"] == sample_idx
            recovered_value = recovered["gene"]["phenotype_values"][sample_mask].item()

            # Expected bin range - for ordinal, the bin index equals the number of crossings
            crossings = sum(1 for x in threshold_patterns[sample_idx] if x > 0)
            low, high = norm_bin_edges[crossings], norm_bin_edges[crossings + 1]

            # Check the value is in the right bin
            assert (
                low <= recovered_value <= high
            ), f"Sample {sample_idx}: value {recovered_value} not in range [{low}, {high}]"


class TestCOOInverseComposeWithGrads:
    @pytest.fixture
    def mock_coo_dataset(self):
        """Create a mock dataset with COO format phenotype data."""

        class MockCOODataset:
            def __init__(self):
                self.data = []

                # Create test data
                data = HeteroData()
                data["gene"] = {}
                data["gene"]["phenotype_values"] = torch.tensor(
                    list(np.linspace(0, 1, 100))
                )
                data["gene"]["phenotype_type_indices"] = torch.tensor([0] * 100)
                data["gene"]["phenotype_sample_indices"] = torch.tensor(
                    list(range(100))
                )
                data["gene"]["phenotype_types"] = ["fitness"]

                self.data.append(data)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return MockCOODataset()

    def test_coo_inverse_compose_with_grad_tensors(self, mock_coo_dataset):
        """Test that COOInverseCompose works with tensors that have gradients."""
        # Setup transforms
        norm_config = {"fitness": {"strategy": "minmax"}}

        norm_transform = COOLabelNormalizationTransform(mock_coo_dataset, norm_config)
        inverse_transform = COOInverseCompose([norm_transform])

        # Create data with gradients in COO format
        values = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
        data = HeteroData()
        data["gene"] = {
            "phenotype_values": values,
            "phenotype_type_indices": torch.tensor([0, 0, 0]),
            "phenotype_sample_indices": torch.tensor([0, 1, 2]),
            "phenotype_types": ["fitness"],
        }

        # Apply a simple operation that preserves gradients
        modified_values = values * 0.5 + 0.25

        # Create a new data object with the modified values
        mod_data = HeteroData()
        mod_data["gene"] = {
            "phenotype_values": modified_values,
            "phenotype_type_indices": torch.tensor([0, 0, 0]),
            "phenotype_sample_indices": torch.tensor([0, 1, 2]),
            "phenotype_types": ["fitness"],
        }

        # Apply inverse transform
        recovered = inverse_transform(mod_data)

        # Check shape is preserved
        assert recovered["gene"]["phenotype_values"].shape == values.shape

        # Compute a loss and backpropagate
        loss = recovered["gene"]["phenotype_values"].sum()
        loss.backward()

        # Check gradients were computed
        assert values.grad is not None
        assert not torch.allclose(values.grad, torch.zeros_like(values.grad))


def test_coo_composed_transforms():
    """Test the full transformation pipeline with COO format."""

    # Create a mock dataset
    class MockCOODataset:
        def __init__(self):
            self.data = []

            # Create a single data sample with both phenotypes
            data = HeteroData()
            data["gene"] = {}
            data["gene"]["phenotype_values"] = torch.tensor(
                [-1.0, -0.5, 0.0, 0.5, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0]
            )
            data["gene"]["phenotype_type_indices"] = torch.tensor(
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            )
            data["gene"]["phenotype_sample_indices"] = torch.tensor(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            )
            data["gene"]["phenotype_types"] = ["gene_interaction", "fitness"]

            self.data.append(data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    mock_coo_dataset = MockCOODataset()

    # Create transforms
    norm_config = {
        "fitness": {"strategy": "minmax"},
        "gene_interaction": {"strategy": "standard"},
    }
    normalizer = COOLabelNormalizationTransform(mock_coo_dataset, norm_config)

    bin_config = {
        "fitness": {
            "strategy": "equal_width",
            "num_bins": 4,
            "label_type": "categorical",
        },
        "gene_interaction": {
            "strategy": "equal_width",
            "num_bins": 4,
            "label_type": "ordinal",
        },
    }
    binner = COOLabelBinningTransform(mock_coo_dataset, bin_config, normalizer)

    # Get test data
    data = mock_coo_dataset[0]

    # Store key attributes of the original data
    orig_num_gi = (data["gene"]["phenotype_type_indices"] == 0).sum().item()
    orig_num_fitness = (data["gene"]["phenotype_type_indices"] == 1).sum().item()

    # Create composed transforms
    forward_transform = Compose([normalizer, binner])
    inverse_transform = COOInverseCompose(forward_transform)

    # Apply forward transform
    transformed = forward_transform(data)

    # Check transformed data has expected fields
    assert "phenotype_values" in transformed["gene"]
    assert "phenotype_type_indices" in transformed["gene"]
    assert "phenotype_sample_indices" in transformed["gene"]

    # Apply inverse transform
    recovered = inverse_transform(transformed)

    # Check basic recovery - verify we have phenotype values
    assert "phenotype_values" in recovered["gene"]
    assert "phenotype_type_indices" in recovered["gene"]

    # Count the number of values for each type
    rec_type_indices = recovered["gene"]["phenotype_type_indices"]
    rec_num_gi = (rec_type_indices == 0).sum().item()
    rec_num_fitness = (rec_type_indices == 1).sum().item()

    # Check we have correct number of values for each type
    assert (
        rec_num_gi == orig_num_gi
    ), f"Expected {orig_num_gi} gene_interaction values, got {rec_num_gi}"
    assert (
        rec_num_fitness == orig_num_fitness
    ), f"Expected {orig_num_fitness} fitness values, got {rec_num_fitness}"

    # Verify all values are finite (no inf/nan)
    assert not torch.any(
        torch.isnan(recovered["gene"]["phenotype_values"])
    ), "NaN values in recovered data"
    assert not torch.any(
        torch.isinf(recovered["gene"]["phenotype_values"])
    ), "Infinite values in recovered data"

    # Verify values are within reasonable ranges without using masks
    assert torch.all(recovered["gene"]["phenotype_values"] >= -2), "Values too small"
    assert torch.all(recovered["gene"]["phenotype_values"] <= 2), "Values too large"
