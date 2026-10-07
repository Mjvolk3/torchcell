# tests/torchcell/transforms/test_coo_regression_to_classification.py
# [[tests.torchcell.transforms.test_coo_regression_to_classification]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/transforms/test_coo_regression_to_classification.py
"""Tests for COO regression-to-classification label transforms.

2026.09.30 (Phase 12): closed-form tests for the branches the classes above left open.
Every fixture is a hand-built `label_df` on a stand-in dataset (only `label_df` is read)
and a hand-built COO `HeteroData`; every number is worked here and was checked with a
two-line Python run.

* Normalization statistics use population std (`np.std`, ddof 0) and linear percentiles,
  after +-inf becomes NaN and NaN is dropped. On `index` [0, 10, 20, 30] with fitness
  [1, 2, 3, inf]: mean 2, std sqrt(2/3) = 0.816496580927726, min 1, max 3, q25 1.5,
  q75 2.5. `fit_indices` [10, 20, 20] resolves through the `index` COLUMN (a set, so the
  repeat collapses) to fitness [2, 3]: mean 2.5, std 0.5, q25 2.25, q75 2.75.
* Robust scaling on [0, 1, 2, 3, 4]: q25 1, q75 3, iqr 2, so x -> (x - 1) / (2 + 1e-8)
  maps [0, 1, 4] to [-0.5, 0, 1.5], and the inverse x * 2 + 1 maps back.
* Bin edges: equal width on [0..4, NaN] with 4 bins is [0, 1, 2, 3, 4] (widths 1, mean 2,
  std sqrt 2); equal frequency on 0..8 with 4 bins is the 0/25/50/75/100th percentiles
  [0, 2, 4, 6, 8] with histogram counts [2, 2, 2, 3] (numpy closes the last bin); auto on
  [0..4] picks int(range / std) = int(4 / 1.414) = 2 bins, edges [0, 2, 4].
* Label assignment on edges [0, 1, 2, 3, 4]: one-hot is left-closed ([1, 2) is bin 1)
  after clamping to [0, 4], and the right edge 4 folds into bin 3; ordinal compares
  with `>=` against the interior edges [1, 2, 3], so 1 -> [1, 0, 0] and the count of
  ones equals the one-hot bin on every edge (issue #521).
* Soft labels on edges [0, 1, 2] (centers 0.5, 1.5, sigma = 1 * sigma_scale = 1):
  0.5 -> [1, e^-0.5] / (1 + e^-0.5) = [0.6224593, 0.3775407]; -3 clamps to 0, distances
  [0.5, 1.5] -> [e^-0.125, e^-1.125] normalized = [0.7310586, 0.2689414]. The row is a
  softmax of the log-weights, so it stays a distribution when sigma is narrow.
* Inverse (class to value) draws are seeded: after `torch.manual_seed(42)` the first two
  `torch.rand(1)` are 0.8822692632675171 and 0.9150039553642273, one per sample in
  sorted sample order.
"""

import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.transforms import Compose

from torchcell.transforms.coo_regression_to_classification import (
    AutoBinStrategy,
    COOInverseCompose,
    COOLabelBinningTransform,
    COOLabelNormalizationTransform,
    EqualFrequencyStrategy,
    EqualWidthStrategy,
)


class TestCOOLabelNormalizationTransform:
    """Tests for forward normalization of COO-format phenotype labels."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with a label_df of fitness and gene_interaction."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {
                        "fitness": [0.0, 0.5, 1.0, 2.0],
                        "gene_interaction": [-1.0, 0.0, 1.0, np.nan],
                    }
                )

        return MockDataset()

    @pytest.fixture
    def norm_transform(self, mock_dataset):
        """Return a normalization transform (minmax fitness, standard GI)."""
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return COOLabelNormalizationTransform(mock_dataset, label_configs)

    def test_minmax_normalization_coo(self, norm_transform):
        """Verify minmax scaling, original-value storage, and inverse recovery."""
        data = HeteroData()
        # COO format data
        data["gene"]["phenotype_values"] = torch.tensor([0.0, 0.5, 1.0, 2.0])
        data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0])
        data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3])
        data["gene"]["phenotype_types"] = ["fitness"]

        normalized = norm_transform(data)
        expected = torch.tensor([0.0, 0.25, 0.5, 1.0])
        assert torch.allclose(normalized["gene"]["phenotype_values"], expected)

        # Test that original values are stored
        assert hasattr(normalized["gene"], "phenotype_values_original")
        assert torch.allclose(
            normalized["gene"]["phenotype_values_original"],
            torch.tensor([0.0, 0.5, 1.0, 2.0]),
        )

        denormalized = norm_transform.inverse(normalized)
        assert torch.allclose(
            denormalized["gene"]["phenotype_values"], torch.tensor([0.0, 0.5, 1.0, 2.0])
        )

    def test_standard_normalization_coo(self, norm_transform):
        """Verify standard (z-score) normalization of gene_interaction values."""
        data = HeteroData()
        # COO format data for gene_interaction
        data["gene"]["phenotype_values"] = torch.tensor([-1.0, 0.0, 1.0])
        data["gene"]["phenotype_type_indices"] = torch.tensor(
            [1, 1, 1]
        )  # gene_interaction is second
        data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2])
        data["gene"]["phenotype_types"] = ["fitness", "gene_interaction"]

        normalized = norm_transform(data)
        stats = norm_transform.stats["gene_interaction"]
        mean = stats["mean"]
        std = stats["std"]
        expected = (torch.tensor([-1.0, 0.0, 1.0]) - mean) / std
        assert torch.allclose(normalized["gene"]["phenotype_values"], expected)

    def test_mixed_phenotypes_coo(self, norm_transform):
        """Verify per-type normalization when phenotypes are interleaved in COO."""
        data = HeteroData()
        # Mixed phenotypes in COO format
        data["gene"]["phenotype_values"] = torch.tensor([0.0, -1.0, 1.0, 0.0, 2.0, 1.0])
        data["gene"]["phenotype_type_indices"] = torch.tensor(
            [0, 1, 0, 1, 0, 1]
        )  # alternating
        data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 0, 1, 1, 2, 2])
        data["gene"]["phenotype_types"] = ["fitness", "gene_interaction"]

        normalized = norm_transform(data)

        # Check fitness values (indices 0, 2, 4)
        fitness_mask = data["gene"]["phenotype_type_indices"] == 0
        fitness_vals = data["gene"]["phenotype_values"][fitness_mask]
        expected_fitness = (fitness_vals - norm_transform.stats["fitness"]["min"]) / (
            norm_transform.stats["fitness"]["max"]
            - norm_transform.stats["fitness"]["min"]
        )
        assert torch.allclose(
            normalized["gene"]["phenotype_values"][fitness_mask], expected_fitness
        )

        # Check gene_interaction values (indices 1, 3, 5)
        gi_mask = data["gene"]["phenotype_type_indices"] == 1
        gi_vals = data["gene"]["phenotype_values"][gi_mask]
        expected_gi = (gi_vals - norm_transform.stats["gene_interaction"]["mean"]) / (
            norm_transform.stats["gene_interaction"]["std"]
        )
        assert torch.allclose(
            normalized["gene"]["phenotype_values"][gi_mask], expected_gi
        )


class TestCOOLabelNormalizationInverse:
    """Tests for inverse normalization of COO-format phenotype labels."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with a label_df of fitness and gene_interaction."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {
                        "fitness": [0.0, 0.5, 1.0, 2.0],
                        "gene_interaction": [-1.0, 0.0, 1.0, np.nan],
                    }
                )

        return MockDataset()

    @pytest.fixture
    def norm_transform(self, mock_dataset):
        """Return a normalization transform (minmax fitness, standard GI)."""
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return COOLabelNormalizationTransform(mock_dataset, label_configs)

    def test_inverse_minmax_coo(self, norm_transform):
        """Verify inverse of minmax normalization recovers original values."""
        # Test data in COO format
        normalized_values = torch.tensor([0.0, 0.25, 0.5, 1.0])

        temp_data = HeteroData()
        temp_data["gene"].phenotype_values = normalized_values
        temp_data["gene"].phenotype_type_indices = torch.tensor([0, 0, 0, 0])
        temp_data["gene"].phenotype_sample_indices = torch.tensor([0, 1, 2, 3])
        temp_data["gene"].phenotype_types = ["fitness"]
        denormalized = norm_transform.inverse(temp_data)

        # Validate against the original values
        original_values = torch.tensor([0.0, 0.5, 1.0, 2.0])
        assert torch.allclose(
            denormalized["gene"]["phenotype_values"], original_values, atol=1e-6
        )

    def test_inverse_with_nans_coo(self, norm_transform):
        """Verify inverse normalization preserves NaN entries."""
        # Test data with NaN in COO format
        original_values = torch.tensor([0.0, 0.5, 1.0, float("nan")])
        normalized_values = torch.tensor([0.0, 0.25, 0.5, float("nan")])

        temp_data = HeteroData()
        temp_data["gene"].phenotype_values = normalized_values
        temp_data["gene"].phenotype_type_indices = torch.tensor([0, 0, 0, 0])
        temp_data["gene"].phenotype_sample_indices = torch.tensor([0, 1, 2, 3])
        temp_data["gene"].phenotype_types = ["fitness"]
        denormalized = norm_transform.inverse(temp_data)

        # Validate against the original values, allowing NaN comparisons
        assert torch.allclose(
            denormalized["gene"]["phenotype_values"], original_values, equal_nan=True
        )


class TestCOOLabelNormalizationRoundTrip:
    """Tests that forward then inverse normalization is a round trip."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with a label_df of fitness and gene_interaction."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {
                        "fitness": [0.0, 0.5, 1.0, 2.0],
                        "gene_interaction": [-1.0, 0.0, 1.0, np.nan],
                    }
                )

        return MockDataset()

    @pytest.fixture
    def norm_transform(self, mock_dataset):
        """Return a normalization transform (minmax fitness, standard GI)."""
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return COOLabelNormalizationTransform(mock_dataset, label_configs)

    def test_round_trip_minmax_coo(self, norm_transform, mock_dataset):
        """Verify minmax normalize-then-inverse recovers the fitness values."""
        # Test data from mock_dataset in COO format
        original_df = mock_dataset.label_df
        original_values = torch.tensor(original_df["fitness"].values)

        # Create COO data
        data = HeteroData()
        data["gene"].phenotype_values = original_values
        data["gene"].phenotype_type_indices = torch.zeros(
            len(original_values), dtype=torch.long
        )
        data["gene"].phenotype_sample_indices = torch.arange(len(original_values))
        data["gene"].phenotype_types = ["fitness"]

        # Normalize
        normalized = norm_transform(data)

        # Inverse transform
        recovered = norm_transform.inverse(normalized)

        # Validate round trip
        assert torch.allclose(
            recovered["gene"]["phenotype_values"], original_values, atol=1e-6
        )

    def test_round_trip_mixed_phenotypes_coo(self, norm_transform, mock_dataset):
        """Verify round trip for interleaved fitness and gene_interaction values."""
        # Create mixed phenotype data in COO format
        data = HeteroData()
        # Interleave fitness and gene_interaction values
        values = []
        type_indices = []
        sample_indices = []

        for i in range(4):
            # Add fitness value
            values.append(mock_dataset.label_df["fitness"].iloc[i])
            type_indices.append(0)
            sample_indices.append(i)

            # Add gene_interaction value
            values.append(mock_dataset.label_df["gene_interaction"].iloc[i])
            type_indices.append(1)
            sample_indices.append(i)

        data["gene"].phenotype_values = torch.tensor(values, dtype=torch.float)
        data["gene"].phenotype_type_indices = torch.tensor(
            type_indices, dtype=torch.long
        )
        data["gene"].phenotype_sample_indices = torch.tensor(
            sample_indices, dtype=torch.long
        )
        data["gene"].phenotype_types = ["fitness", "gene_interaction"]

        # Normalize
        normalized = norm_transform(data)

        # Inverse transform
        recovered = norm_transform.inverse(normalized)

        # Validate round trip
        assert torch.allclose(
            recovered["gene"]["phenotype_values"],
            data["gene"]["phenotype_values"],
            equal_nan=True,
            atol=1e-6,
        )


class TestCOOLabelBinningTransform:
    """Tests for categorical and soft binning of COO-format labels."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with fitness and gene_interaction label columns."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {
                        "fitness": np.concatenate(
                            [
                                np.linspace(0, 0.3, 33),
                                np.linspace(0.3, 0.7, 34),
                                np.linspace(0.7, 1.0, 33),
                            ]
                        ),
                        "gene_interaction": np.linspace(-1, 1, 100),
                    }
                )

        return MockDataset()

    @pytest.fixture
    def bin_transform(self, mock_dataset):
        """Return a 4-bin equal-width categorical binning transform for GI."""
        label_configs = {
            "gene_interaction": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "categorical",
            }
        }
        return COOLabelBinningTransform(mock_dataset, label_configs)

    def test_categorical_binning_forward_coo(self, bin_transform):
        """Verify categorical binning yields one-hot bins per sample."""
        data = HeteroData()
        # COO format data
        data["gene"]["phenotype_values"] = torch.tensor([-1.0, -0.3, 0.3, 1.0])
        data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0])
        data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3])
        data["gene"]["phenotype_types"] = ["gene_interaction"]

        binned = bin_transform(data)

        # Check that we now have 4 bins × 4 samples = 16 values
        assert binned["gene"]["phenotype_values"].shape == (16,)

        # Check that each sample has exactly one bin with value 1.0
        for sample_idx in range(4):
            mask = binned["gene"]["phenotype_sample_indices"] == sample_idx
            sample_values = binned["gene"]["phenotype_values"][mask]
            assert torch.sum(sample_values) == 1.0
            assert torch.all((sample_values == 0) | (sample_values == 1))

    def test_soft_binning_forward_coo(self, mock_dataset):
        """Verify soft binning produces per-sample bin weights summing to one."""
        label_configs = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "soft",
                "sigma": 0.5,
            }
        }
        bin_transform = COOLabelBinningTransform(mock_dataset, label_configs)

        data = HeteroData()
        data["gene"]["phenotype_values"] = torch.tensor([0.0, 0.3, 0.7, 1.0])
        data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0])
        data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3])
        data["gene"]["phenotype_types"] = ["fitness"]

        binned = bin_transform(data)

        # Check that we have 4 bins × 4 samples = 16 values
        assert binned["gene"]["phenotype_values"].shape == (16,)

        # Check that each sample's bins sum to 1 (soft labels)
        for sample_idx in range(4):
            mask = binned["gene"]["phenotype_sample_indices"] == sample_idx
            sample_values = binned["gene"]["phenotype_values"][mask]
            assert torch.allclose(torch.sum(sample_values), torch.tensor(1.0))

    def test_inverse_categorical_binning_coo(self, bin_transform):
        """Verify inverse of categorical binning yields values in their bins."""
        data = HeteroData()
        # Create one-hot encoded bins in COO format
        # 4 samples, 4 bins each
        values = []
        type_indices = []
        sample_indices = []

        # Sample 0: bin 0
        for i in range(4):
            values.append(1.0 if i == 0 else 0.0)
            type_indices.append(i)
            sample_indices.append(0)

        # Sample 1: bin 1
        for i in range(4):
            values.append(1.0 if i == 1 else 0.0)
            type_indices.append(i)
            sample_indices.append(1)

        # Sample 2: bin 2
        for i in range(4):
            values.append(1.0 if i == 2 else 0.0)
            type_indices.append(i)
            sample_indices.append(2)

        # Sample 3: bin 3
        for i in range(4):
            values.append(1.0 if i == 3 else 0.0)
            type_indices.append(i)
            sample_indices.append(3)

        data["gene"]["phenotype_values"] = torch.tensor(values)
        data["gene"]["phenotype_type_indices"] = torch.tensor(type_indices)
        data["gene"]["phenotype_sample_indices"] = torch.tensor(sample_indices)
        data["gene"]["phenotype_types"] = [
            "gene_interaction_bin_0",
            "gene_interaction_bin_1",
            "gene_interaction_bin_2",
            "gene_interaction_bin_3",
        ]

        inverted = bin_transform.inverse(data)

        # Check that we get back 4 continuous values
        assert inverted["gene"]["phenotype_values"].shape == (4,)
        assert inverted["gene"]["phenotype_types"] == ["gene_interaction"]

        # Check that values fall within expected bins
        bin_edges = torch.tensor(
            bin_transform.label_metadata["gene_interaction"]["bin_edges"]
        )

        for i in range(4):
            mask = inverted["gene"]["phenotype_sample_indices"] == i
            val = inverted["gene"]["phenotype_values"][mask].item()
            assert bin_edges[i] <= val <= bin_edges[i + 1]


class TestCOOInverseCompose:
    """Tests for composing and inverting a chain of COO transforms."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with a single gene_interaction label column."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {"gene_interaction": np.linspace(-1.0, 1.0, 100)}
                )

        return MockDataset()

    @pytest.fixture
    def transforms(self, mock_dataset):
        """Return a normalization followed by a binning transform."""
        norm_config = {"gene_interaction": {"strategy": "minmax"}}
        bin_config = {
            "gene_interaction": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "categorical",
            }
        }

        norm_transform = COOLabelNormalizationTransform(mock_dataset, norm_config)
        bin_transform = COOLabelBinningTransform(
            mock_dataset, bin_config, norm_transform
        )
        return [norm_transform, bin_transform]

    def test_inverse_compose_coo(self, transforms):
        """Verify COOInverseCompose reverses a forward Compose back to values."""
        data = HeteroData()
        data["gene"]["phenotype_values"] = torch.tensor([-1.0, -0.3, 0.3, 1.0])
        data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0])
        data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3])
        data["gene"]["phenotype_types"] = ["gene_interaction"]

        forward_transform = Compose(transforms)
        inverse_transform = COOInverseCompose(forward_transform)

        transformed = forward_transform(data)
        recovered = inverse_transform(transformed)

        # Check that we get back continuous values
        assert recovered["gene"]["phenotype_types"] == ["gene_interaction"]
        assert recovered["gene"]["phenotype_values"].shape == (4,)

        # Check values are in reasonable range
        assert torch.all(recovered["gene"]["phenotype_values"] >= -1.0)
        assert torch.all(recovered["gene"]["phenotype_values"] <= 1.0)


class TestOrdinalBinningCOO:
    """Tests for ordinal (threshold) binning of COO-format labels."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with a single fitness label column."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {
                        "fitness": np.concatenate(
                            [
                                np.linspace(0, 0.3, 33),
                                np.linspace(0.3, 0.7, 34),
                                np.linspace(0.7, 1.0, 33),
                            ]
                        )
                    }
                )

        return MockDataset()

    @pytest.fixture
    def bin_transform(self, mock_dataset):
        """Return a 4-bin equal-width ordinal binning transform for fitness."""
        label_configs = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "ordinal",
            }
        }
        return COOLabelBinningTransform(mock_dataset, label_configs)

    def test_ordinal_forward_coo(self, bin_transform):
        """Verify ordinal binning yields monotone threshold indicators."""
        data = HeteroData()
        data["gene"]["phenotype_values"] = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0, 0])
        data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3, 4])
        data["gene"]["phenotype_types"] = ["fitness"]

        # Apply forward transform
        transformed = bin_transform(data)

        # With 4 bins, we have 3 thresholds, so 3 values per sample
        # 5 samples × 3 thresholds = 15 values
        assert transformed["gene"]["phenotype_values"].shape == (15,)

        # Check ordinal property for each sample
        for sample_idx in range(5):
            mask = transformed["gene"]["phenotype_sample_indices"] == sample_idx
            sample_values = transformed["gene"]["phenotype_values"][mask]

            # Values should be binary
            assert torch.all((sample_values == 0) | (sample_values == 1))

            # Check ordinal property: if a higher threshold is 1, all lower thresholds should be 1
            for i in range(1, len(sample_values)):
                if sample_values[i] == 1:
                    assert torch.all(sample_values[:i] == 1)

    def test_ordinal_with_nans_coo(self, bin_transform):
        """Verify ordinal binning preserves NaN entries for affected samples."""
        data = HeteroData()
        data["gene"]["phenotype_values"] = torch.tensor(
            [0.0, float("nan"), 0.5, float("nan"), 1.0]
        )
        data["gene"]["phenotype_type_indices"] = torch.tensor([0, 0, 0, 0, 0])
        data["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1, 2, 3, 4])
        data["gene"]["phenotype_types"] = ["fitness"]

        # Forward transform
        transformed = bin_transform(data)

        # Check that NaN values are preserved
        for sample_idx in [1, 3]:  # NaN samples
            mask = transformed["gene"]["phenotype_sample_indices"] == sample_idx
            sample_values = transformed["gene"]["phenotype_values"][mask]
            assert torch.isnan(sample_values).all()


class TestBatchProcessingCOO:
    """Tests that COO transforms work on batched HeteroData."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with random gene_interaction labels."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {"gene_interaction": np.random.uniform(-1, 1, 1000)}
                )

        return MockDataset()

    def test_batch_normalization_coo(self, mock_dataset):
        """Verify normalization applies across a multi-graph batch."""
        norm_config = {"gene_interaction": {"strategy": "standard"}}
        norm_transform = COOLabelNormalizationTransform(mock_dataset, norm_config)

        # Create batch data
        batch1 = HeteroData()
        batch1["gene"]["phenotype_values"] = torch.tensor([-0.5, 0.0])
        batch1["gene"]["phenotype_type_indices"] = torch.tensor([0, 0])
        batch1["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1])
        batch1["gene"]["phenotype_types"] = ["gene_interaction"]

        batch2 = HeteroData()
        batch2["gene"]["phenotype_values"] = torch.tensor([0.5, 1.0])
        batch2["gene"]["phenotype_type_indices"] = torch.tensor([0, 0])
        batch2["gene"]["phenotype_sample_indices"] = torch.tensor([0, 1])
        batch2["gene"]["phenotype_types"] = ["gene_interaction"]

        # Create batch

        batch = Batch.from_data_list([batch1, batch2])

        # Apply transform
        normalized = norm_transform(batch)

        # Check that all values are normalized
        assert normalized["gene"]["phenotype_values"].shape == (4,)
        assert hasattr(normalized["gene"], "phenotype_values_original")


class TestModelOutputSimulationCOO:
    """Tests that inverse transforms work on simulated model outputs."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with a single gene_interaction label column."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {"gene_interaction": np.linspace(-1, 1, 100)}
                )

        return MockDataset()

    def test_model_output_inverse_coo(self, mock_dataset):
        """Test that inverse transform works with model-like outputs in COO format."""
        # Setup transforms
        norm_config = {"gene_interaction": {"strategy": "standard"}}
        norm_transform = COOLabelNormalizationTransform(mock_dataset, norm_config)
        inverse_transform = COOInverseCompose([norm_transform])

        # Simulate model outputs (normalized predictions)
        pred_data = HeteroData()
        pred_data["gene"].phenotype_values = torch.randn(
            5
        )  # Random normalized predictions
        pred_data["gene"].phenotype_type_indices = torch.zeros(5, dtype=torch.long)
        pred_data["gene"].phenotype_sample_indices = torch.arange(5)
        pred_data["gene"].phenotype_types = ["gene_interaction"]

        # Apply inverse transform
        recovered = inverse_transform(pred_data)

        # Check that values are denormalized properly
        stats = norm_transform.stats["gene_interaction"]
        for i in range(5):
            normalized_val = pred_data["gene"]["phenotype_values"][i]
            expected = normalized_val * stats["std"] + stats["mean"]
            assert torch.allclose(
                recovered["gene"]["phenotype_values"][i], expected, atol=1e-6
            )


# ---------------------------------------------------------------------------------------
# 2026.09.30 (Phase 12): closed-form branch tests. Values derived in the module docstring.
# ---------------------------------------------------------------------------------------

SEED42_U0 = 0.8822692632675171
SEED42_U1 = 0.9150039553642273


def _dataset(df: pd.DataFrame) -> Any:
    """A stand-in for `Neo4jCellDataset`: the transforms read only `label_df`."""
    return SimpleNamespace(label_df=df)


def _coo(
    values: list[float] | float,
    type_indices: list[int],
    sample_indices: list[int],
    types: list[str] | list[list[str]],
) -> HeteroData:
    data = HeteroData()
    data["gene"].phenotype_values = torch.tensor(values, dtype=torch.float)
    data["gene"].phenotype_type_indices = torch.tensor(type_indices, dtype=torch.long)
    data["gene"].phenotype_sample_indices = torch.tensor(
        sample_indices, dtype=torch.long
    )
    data["gene"].phenotype_types = types
    return data


def _robust() -> COOLabelNormalizationTransform:
    df = pd.DataFrame({"fitness": [0.0, 1.0, 2.0, 3.0, 4.0]})
    return COOLabelNormalizationTransform(
        _dataset(df), {"fitness": {"strategy": "robust"}}
    )


def _bins(label_type: str, **extra: Any) -> COOLabelBinningTransform:
    """Equal-width, 4 bins over gene_interaction [0..4]: edges [0, 1, 2, 3, 4]."""
    df = pd.DataFrame({"gene_interaction": [0.0, 1.0, 2.0, 3.0, 4.0]})
    config = {
        "gene_interaction": {
            "strategy": "equal_width",
            "num_bins": 4,
            "label_type": label_type,
            **extra,
        }
    }
    return COOLabelBinningTransform(_dataset(df), config)


def test_statistics_drop_infinities_and_use_population_std() -> None:
    """Inf is replaced by NaN and dropped, so fitness [1, 2, 3, inf] gives mean 2 and the
    ddof-0 std sqrt(2/3); percentiles interpolate linearly (q25 1.5, q75 2.5).
    """
    df = pd.DataFrame({"index": [0, 10, 20, 30], "fitness": [1.0, 2.0, 3.0, np.inf]})
    t = COOLabelNormalizationTransform(
        _dataset(df), {"fitness": {"strategy": "robust"}}
    )
    assert t.stats == {
        "fitness": {
            "mean": 2.0,
            "std": pytest.approx(0.816496580927726, abs=1e-15),
            "min": 1.0,
            "max": 3.0,
            "q25": 1.5,
            "q75": 2.5,
            "strategy": "robust",
        }
    }


def test_fit_indices_resolve_through_the_index_column() -> None:
    """Records 10 and 20 sit at rows 1 and 2; a positional read of [10, 20] would fail,
    and a read of rows [1, 2] with the wrong column would give different numbers. The
    repeated 20 collapses in the set, so no record is reported missing.
    """
    df = pd.DataFrame({"index": [0, 10, 20, 30], "fitness": [1.0, 2.0, 3.0, np.inf]})
    t = COOLabelNormalizationTransform(
        _dataset(df), {"fitness": {"strategy": "standard"}}, fit_indices=[10, 20, 20]
    )
    assert t.stats["fitness"] == {
        "mean": 2.5,
        "std": 0.5,
        "min": 2.0,
        "max": 3.0,
        "q25": 2.25,
        "q75": 2.75,
        "strategy": "standard",
    }


def test_fit_indices_absent_from_label_df_are_refused() -> None:
    """Record 99 is not in the `index` column: one missing index, named in the error."""
    df = pd.DataFrame({"index": [0, 10, 20, 30], "fitness": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(
        ValueError, match=r"^fit_indices names 1 record indices absent from label_df$"
    ):
        COOLabelNormalizationTransform(
            _dataset(df), {"fitness": {"strategy": "standard"}}, fit_indices=[10, 99]
        )


def test_normalization_refuses_a_label_the_dataset_lacks() -> None:
    df = pd.DataFrame({"fitness": [1.0, 2.0]})
    with pytest.raises(
        ValueError, match=r"^Label gene_interaction not found in dataset$"
    ):
        COOLabelNormalizationTransform(
            _dataset(df), {"gene_interaction": {"strategy": "standard"}}
        )


def test_robust_normalize_and_denormalize_in_closed_form() -> None:
    """(x - q25) / (iqr + eps) with q25 1, iqr 2: [0, 1, 4] -> [-0.5, 0, 1.5]; the
    inverse x * iqr + q25 (no eps) maps [-0.5, 0, 1.5] back to [0, 1, 4].
    """
    t = _robust()
    torch.testing.assert_close(
        t.normalize(torch.tensor([0.0, 1.0, 4.0]), "fitness"),
        torch.tensor([-0.5, 0.0, 1.5]),
    )
    torch.testing.assert_close(
        t.denormalize(torch.tensor([-0.5, 0.0, 1.5]), "fitness"),
        torch.tensor([0.0, 1.0, 4.0]),
    )


def test_all_nan_values_short_circuit_before_the_strategy_is_read() -> None:
    """An all-NaN tensor comes back all NaN from normalize and denormalize without the
    strategy being read: a transform whose ``stats`` were emptied, which would raise on
    any lookup, still returns it. A partly NaN tensor does reach the strategy: the robust
    transform maps [nan, 4.0] to [nan, 1.5] (median 1, IQR 2).
    """
    values = torch.tensor([float("nan"), float("nan")])
    emptied = _robust()
    emptied.stats = {}
    for transform in (_robust(), emptied):
        for out in (
            transform.normalize(values, "fitness"),
            transform.denormalize(values, "fitness"),
        ):
            assert out.shape == (2,)
            assert torch.isnan(out).all()
    mixed = _robust().normalize(torch.tensor([float("nan"), 4.0]), "fitness")
    assert torch.isnan(mixed[0])
    assert mixed[1].item() == 1.5


def test_unknown_normalization_strategy_is_refused_at_construction() -> None:
    """Contract (issue #521): the strategy is validated when the transform is built, so
    a typo ("zscore") fails there with the bad value and the valid names, not on the
    first normalize. The check runs before the label lookup, so it names the strategy
    even for a label the dataset lacks: the frame has no `fitness` column, so checking
    the label first would raise "Label fitness not found in dataset" instead.
    """
    df = pd.DataFrame({"other": [1.0]})
    message = (
        r"^Unknown normalization strategy 'zscore' for label 'fitness'; "
        r"valid strategies: minmax, robust, standard$"
    )
    with pytest.raises(ValueError, match=message):
        COOLabelNormalizationTransform(
            _dataset(df), {"fitness": {"strategy": "zscore"}}
        )


def test_normalization_passes_through_data_without_phenotype_values() -> None:
    """No `phenotype_values`: `forward` and `inverse` return the object they were given
    (`BaseTransform.__call__` would shallow-copy first) and add no original copy.
    """
    t = _robust()
    data = HeteroData()
    data["gene"].x = torch.tensor([1.0])
    assert t.forward(data) is data
    assert t.inverse(data) is data
    assert list(data["gene"].keys()) == ["x"]


def test_normalization_keeps_a_scalar_label_scalar() -> None:
    """A 0-d value 4.0 is normalized to the 0-d 1.5 and back to the 0-d 4.0; the stored
    original is the 0-d 4.0.
    """
    t = _robust()
    data = _coo(4.0, [0], [0], ["fitness"])
    out = t(data)
    torch.testing.assert_close(out["gene"].phenotype_values, torch.tensor(1.5))
    torch.testing.assert_close(out["gene"].phenotype_values_original, torch.tensor(4.0))
    back = t.inverse(out)
    torch.testing.assert_close(back["gene"].phenotype_values, torch.tensor(4.0))


def test_second_normalization_keeps_the_first_original() -> None:
    """`phenotype_values_original` is written once: applying the transform twice keeps
    the RAW [4.0] as the original while the values are normalized twice,
    4 -> 1.5 -> (1.5 - 1) / 2 = 0.25.
    """
    t = _robust()
    data = t(t(_coo([4.0], [0], [0], ["fitness"])))
    torch.testing.assert_close(data["gene"].phenotype_values, torch.tensor([0.25]))
    torch.testing.assert_close(
        data["gene"].phenotype_values_original, torch.tensor([4.0])
    )


def test_batched_phenotype_types_use_the_first_list() -> None:
    """A collated batch carries `phenotype_types` as a list of per-item lists; the first
    is used, so type index 1 is fitness here and [0, 4] becomes [-0.5, 1.5] while the
    type-0 value 9.0 (gene_interaction, not configured) is untouched. The inverse reads
    the same list and restores [0, 4].
    """
    t = _robust()
    types = [["gene_interaction", "fitness"], ["gene_interaction", "fitness"]]
    data = _coo([0.0, 9.0, 4.0], [1, 0, 1], [0, 0, 1], types)
    out = t(data)
    torch.testing.assert_close(
        out["gene"].phenotype_values, torch.tensor([-0.5, 9.0, 1.5])
    )
    back = t.inverse(out)
    torch.testing.assert_close(
        back["gene"].phenotype_values, torch.tensor([0.0, 9.0, 4.0])
    )


def test_configured_label_with_no_entries_leaves_values_untouched() -> None:
    """Fitness is listed in `phenotype_types` but no entry has its type index (1), so the
    mask is empty and every value passes through, forward and inverse.
    """
    t = _robust()
    data = _coo([4.0, 2.0], [0, 0], [0, 1], ["gene_interaction", "fitness"])
    torch.testing.assert_close(
        t(data)["gene"].phenotype_values, torch.tensor([4.0, 2.0])
    )
    torch.testing.assert_close(
        t.inverse(data)["gene"].phenotype_values, torch.tensor([4.0, 2.0])
    )


def test_equal_width_bins_ignore_nan() -> None:
    edges, meta = EqualWidthStrategy().compute_bins(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, np.nan]), 4
    )
    np.testing.assert_array_equal(edges, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(meta["bin_widths"], [1.0, 1.0, 1.0, 1.0])
    assert (meta["min"], meta["max"], meta["mean"]) == (0.0, 4.0, 2.0)
    assert meta["std"] == pytest.approx(np.sqrt(2.0), abs=1e-15)
    assert meta["strategy"] == "equal_width"


def test_equal_frequency_bins_are_percentiles_and_close_the_last_bin() -> None:
    """0..8 at the 0/25/50/75/100th percentiles is [0, 2, 4, 6, 8]; `np.histogram` puts
    8 in the last (closed) bin, so the counts are [2, 2, 2, 3], not four equal bins.
    """
    edges, meta = EqualFrequencyStrategy().compute_bins(np.arange(9.0), 4)
    np.testing.assert_array_equal(edges, [0.0, 2.0, 4.0, 6.0, 8.0])
    np.testing.assert_array_equal(meta["bin_counts"], [2, 2, 2, 3])
    assert (meta["min"], meta["max"], meta["mean"]) == (0.0, 8.0, 4.0)
    assert meta["strategy"] == "equal_frequency"


def test_auto_bins_truncate_range_over_std() -> None:
    """Auto picks int(4 / sqrt 2) = int(2.83) = 2 bins and, although it delegates the
    edges to equal width, records `strategy` "auto" (issue #521), so saved metadata
    names the rule that chose the bin count. An explicit num_bins 4 overrides the rule
    and is still recorded as "auto".
    """
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    edges, meta = AutoBinStrategy().compute_bins(values)
    np.testing.assert_array_equal(edges, [0.0, 2.0, 4.0])
    assert meta["strategy"] == "auto"
    edges4, meta4 = AutoBinStrategy().compute_bins(values, 4)
    np.testing.assert_array_equal(edges4, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert meta4["strategy"] == "auto"


def test_onehot_labels_are_left_closed_and_clamped() -> None:
    """Edges [0, 1, 2, 3, 4]: -1 clamps to 0 (bin 0); 0 and 0.5 are bin 0; the edge 1
    starts bin 1; 3.999 is bin 3; the right edge 4 and the clamped 5 fold into bin 3;
    NaN gives a NaN row.
    """
    edges = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    values = torch.tensor([-1.0, 0.0, 0.5, 1.0, 3.999, 4.0, 5.0, float("nan")])
    nan = float("nan")
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [nan, nan, nan, nan],
        ]
    )
    torch.testing.assert_close(
        EqualWidthStrategy().compute_onehot_labels(values, edges),
        expected,
        equal_nan=True,
    )


def test_ordinal_labels_count_left_closed_crossings_of_interior_edges() -> None:
    """Contract (issue #521): ordinal and one-hot share the left-closed convention of
    `torch.bucketize(..., right=True) - 1` / `np.digitize`, bin i = [edge_i, edge_i+1).
    Ordinal compares with `>=`, so a value ON an interior edge counts as reaching it:
    1 -> [1, 0, 0] (bin 1, as one-hot), 3 -> [1, 1, 1] (bin 3). Other rows: -1 ->
    [0, 0, 0], 1.5 -> [1, 0, 0], 5 (clamped to 4) -> [1, 1, 1], NaN -> NaN. On every
    edge 0..4 and between them, the number of ones equals the one-hot bin index.
    """
    edges = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    values = torch.tensor([-1.0, 1.0, 1.5, 3.0, 5.0, float("nan")])
    nan = float("nan")
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [nan, nan, nan],
        ]
    )
    strategy = EqualWidthStrategy()
    torch.testing.assert_close(
        strategy.compute_ordinal_labels(values, edges), expected, equal_nan=True
    )
    on_and_between = torch.tensor([0.0, 0.999, 1.0, 1.5, 2.0, 2.999, 3.0, 4.0])
    ordinal_class = strategy.compute_ordinal_labels(on_and_between, edges).sum(dim=1)
    onehot_class = strategy.compute_onehot_labels(on_and_between, edges).argmax(dim=1)
    assert ordinal_class.tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert onehot_class.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_soft_labels_are_normalized_gaussians_on_bin_centers() -> None:
    """Edges [0, 1, 2], sigma 1: 0.5 -> [0.6224593, 0.3775407]; 1.0 sits midway ->
    [0.5, 0.5]; -3 clamps to 0 -> [0.7310586, 0.2689414]; NaN -> NaN row.
    """
    out = EqualWidthStrategy().compute_soft_labels(
        torch.tensor([0.5, 1.0, -3.0, float("nan")]),
        torch.tensor([0.0, 1.0, 2.0]),
        sigma_scale=1,
    )
    nan = float("nan")
    expected = torch.tensor(
        [
            [0.6224593312018546, 0.37754066879814546],
            [0.5, 0.5],
            [0.7310585786300049, 0.2689414213699951],
            [nan, nan],
        ]
    )
    torch.testing.assert_close(out, expected, equal_nan=True)


def test_soft_labels_stay_a_distribution_when_sigma_is_narrow() -> None:
    """Contract (issue #521): with sigma = 1 * 0.01 the raw weights underflow (the value
    0 sits 0.5 and 1.5 from the centers, exp(-0.5 * 50^2) = exp(-1250) and exp(-11250),
    both 0.0 in float32 and float64), but the row is a softmax of the log-weights
    [-1250, -11250], which shifts by the max first: [1, exp(-10000)] = [1, 0]. The
    center 0.5 gives [1, 0]; the midpoint 1.0 has equal log-weights -1250 and gives
    exactly [0.5, 0.5], where the old exp-then-normalize returned [0, 0].
    """
    out = EqualWidthStrategy().compute_soft_labels(
        torch.tensor([0.0, 0.5, 1.0]), torch.tensor([0.0, 1.0, 2.0]), sigma_scale=0.01
    )
    torch.testing.assert_close(
        out, torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]]), rtol=0, atol=0
    )


def test_binning_forward_expands_each_value_into_its_bins() -> None:
    """Contract (issue #521): the type list is rewritten in its own order, fitness
    (not configured) passing through as type 0 and gene_interaction becoming
    `gene_interaction_bin_0..3` = types 1..4; type indices come from that list, so
    none points past it. Each COO entry is replaced in place: fitness 0.9 (sample 7)
    stays one entry of type 0; gene_interaction 0.5 (sample 7) and 2.0 (sample 9)
    become one-hot [1, 0, 0, 0] and [0, 0, 1, 0] on types 1..4, sample indices
    repeated per bin. The pre-binning gene_interaction values are kept as
    `gene_interaction_continuous`.
    """
    t = _bins("categorical")
    data = _coo([0.9, 0.5, 2.0], [0, 1, 1], [7, 7, 9], ["fitness", "gene_interaction"])
    out = t(data)["gene"]
    torch.testing.assert_close(
        out.phenotype_values,
        torch.tensor([0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        rtol=0,
        atol=0,
    )
    assert out.phenotype_type_indices.tolist() == [0, 1, 2, 3, 4, 1, 2, 3, 4]
    assert out.phenotype_sample_indices.tolist() == [7, 7, 7, 7, 7, 9, 9, 9, 9]
    assert out.phenotype_types == [
        "fitness",
        "gene_interaction_bin_0",
        "gene_interaction_bin_1",
        "gene_interaction_bin_2",
        "gene_interaction_bin_3",
    ]
    torch.testing.assert_close(
        out.gene_interaction_continuous, torch.tensor([0.5, 2.0])
    )


def test_binning_inverse_passes_unconfigured_labels_through() -> None:
    """The inverse mirrors the forward: decoding the two-label output above gives back
    the type list ["fitness", "gene_interaction"], the fitness entry 0.9 unchanged as
    type 0 (sample 7), and the gene_interaction samples 7 and 9 decoded in sorted
    order with one seed-42 draw each: bin 0 -> 0 + 0.8822693, bin 2 -> 2 + 0.9150040.
    """
    t = _bins("categorical")
    data = _coo([0.9, 0.5, 2.0], [0, 1, 1], [7, 7, 9], ["fitness", "gene_interaction"])
    out = t.inverse(t(data))["gene"]
    torch.testing.assert_close(
        out.phenotype_values, torch.tensor([0.9, SEED42_U0, 2.0 + SEED42_U1])
    )
    assert out.phenotype_type_indices.tolist() == [0, 1, 1]
    assert out.phenotype_sample_indices.tolist() == [7, 7, 9]
    assert out.phenotype_types == ["fitness", "gene_interaction"]

    # Binned label FIRST: gene_interaction 0.5 (sample 7), fitness 0.9 (sample 7),
    # gene_interaction 2.0 (sample 9). Forward: bins are types 0..3, fitness type 4,
    # each entry expanded in place. Inverse: gene_interaction (type 0) first, samples
    # 7 then 9 with the seed-42 draws, then fitness 0.9 as type 1.
    first = _coo([0.5, 0.9, 2.0], [0, 1, 0], [7, 7, 9], ["gene_interaction", "fitness"])
    forward = t(first)
    assert forward["gene"].phenotype_type_indices.tolist() == [
        0, 1, 2, 3, 4, 0, 1, 2, 3
    ]  # fmt: skip
    assert forward["gene"].phenotype_types == [
        "gene_interaction_bin_0",
        "gene_interaction_bin_1",
        "gene_interaction_bin_2",
        "gene_interaction_bin_3",
        "fitness",
    ]
    back = t.inverse(forward)["gene"]
    torch.testing.assert_close(
        back.phenotype_values, torch.tensor([SEED42_U0, 2.0 + SEED42_U1, 0.9])
    )
    assert back.phenotype_type_indices.tolist() == [0, 0, 1]
    assert back.phenotype_sample_indices.tolist() == [7, 9, 7]
    assert back.phenotype_types == ["gene_interaction", "fitness"]


def test_binning_forward_continuous_copy_is_optional_and_written_once() -> None:
    """`store_continuous: False` writes no copy; with the default an existing
    `gene_interaction_continuous` is not overwritten.
    """
    off = _bins("categorical", store_continuous=False)
    out = off(_coo([0.5], [0], [0], ["gene_interaction"]))["gene"]
    assert "gene_interaction_continuous" not in out.keys()

    on = _bins("categorical")
    data = _coo([0.5], [0], [0], ["gene_interaction"])
    data["gene"].gene_interaction_continuous = torch.tensor([-7.0])
    out = on(data)["gene"]
    torch.testing.assert_close(out.gene_interaction_continuous, torch.tensor([-7.0]))


def test_binning_forward_scalar_and_batched_types() -> None:
    """A 0-d 3.5 with batched types [["gene_interaction"]] is unsqueezed and one-hot
    encoded into bin 3; the ordinal variant gives [1, 1, 1] (3.5 > 1, 2, 3).
    """
    data = _coo(3.5, [0], [0], [["gene_interaction"]])
    out = _bins("categorical")(data)["gene"]
    torch.testing.assert_close(out.phenotype_values, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    ordinal = _bins("ordinal")(_coo(3.5, [0], [0], [["gene_interaction"]]))["gene"]
    torch.testing.assert_close(ordinal.phenotype_values, torch.tensor([1.0, 1.0, 1.0]))
    assert ordinal.phenotype_type_indices.tolist() == [0, 1, 2]


def test_binning_forward_rewrites_the_type_list_only_for_configured_labels() -> None:
    """No phenotype_values: same object back. The configured label absent from the
    types leaves the COO untouched. Present in the types with no entries, its name is
    still rewritten to its bins (the type list depends only on the input type list, so
    every batch of one dataset gets the same list) and the fitness entry passes through
    as type 0.
    """
    t = _bins("categorical")
    bare = HeteroData()
    bare["gene"].x = torch.tensor([1.0])
    assert t.forward(bare) is bare

    absent = t(_coo([0.9], [0], [0], ["fitness"]))["gene"]
    torch.testing.assert_close(absent.phenotype_values, torch.tensor([0.9]))
    assert absent.phenotype_types == ["fitness"]

    empty = t(_coo([0.9], [0], [0], ["fitness", "gene_interaction"]))["gene"]
    torch.testing.assert_close(empty.phenotype_values, torch.tensor([0.9]))
    assert empty.phenotype_type_indices.tolist() == [0]
    assert empty.phenotype_sample_indices.tolist() == [0]
    assert empty.phenotype_types == [
        "fitness",
        "gene_interaction_bin_0",
        "gene_interaction_bin_1",
        "gene_interaction_bin_2",
        "gene_interaction_bin_3",
    ]


def test_unknown_label_type_is_refused_at_construction() -> None:
    """Contract (issue #521): `label_type` is validated (after lowercasing) when the
    transform is built; "hard" raises a ValueError naming the bad value, the label,
    and the valid types. A mixed-case valid one ("Ordinal") is accepted.
    """
    with pytest.raises(
        ValueError,
        match=r"^Unknown label_type 'hard' for label 'gene_interaction'; "
        r"valid label types: categorical, ordinal, soft$",
    ):
        _bins("hard")
    assert _bins("Ordinal").label_width("gene_interaction") == 3


def test_binning_construction_errors() -> None:
    """A label missing from `label_df` is a ValueError (even with a normalizer, which
    skips it rather than failing first); an unknown strategy name is a KeyError.
    """
    df = pd.DataFrame({"fitness": [0.0, 1.0]})
    with pytest.raises(
        ValueError, match=r"^Label gene_interaction not found in dataset$"
    ):
        COOLabelBinningTransform(
            _dataset(df), {"gene_interaction": {"strategy": "equal_width"}}
        )
    norm = COOLabelNormalizationTransform(
        _dataset(df), {"fitness": {"strategy": "minmax"}}
    )
    with pytest.raises(
        ValueError, match=r"^Label gene_interaction not found in dataset$"
    ):
        COOLabelBinningTransform(
            _dataset(df),
            {"gene_interaction": {"strategy": "equal_width"}},
            normalizer=norm,
        )
    with pytest.raises(KeyError, match=r"^'quantile'$"):
        COOLabelBinningTransform(
            _dataset(df), {"fitness": {"strategy": "quantile", "num_bins": 2}}
        )


def test_bins_are_computed_on_normalized_values_and_denormalized_back() -> None:
    """Minmax on [0, 2, 4, 6, 8] maps to [0, 0.25, 0.5, 0.75, 1]; four equal-width bins
    on that are [0, 0.25, 0.5, 0.75, 1], and the inverse normalization gives the raw
    edges [0, 2, 4, 6, 8]. `get_bin_info` returns this metadata. The exact equality on
    the normalized edges holds because line 443 casts to float32 (8 + 1e-8 rounds to 8).
    """
    df = pd.DataFrame({"gene_interaction": [0.0, 2.0, 4.0, 6.0, 8.0]})
    norm = COOLabelNormalizationTransform(
        _dataset(df), {"gene_interaction": {"strategy": "minmax"}}
    )
    t = COOLabelBinningTransform(
        _dataset(df),
        {"gene_interaction": {"strategy": "equal_width", "num_bins": 4}},
        normalizer=norm,
    )
    info = t.get_bin_info("gene_interaction")
    np.testing.assert_array_equal(info["bin_edges"], [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(
        info["bin_edges_denormalized"], [0.0, 2.0, 4.0, 6.0, 8.0]
    )
    with pytest.raises(
        ValueError, match=r"^No binning metadata found for label fitness$"
    ):
        t.get_bin_info("fitness")


def _binned(
    rows: dict[int, list[float]], num_bins: int, label: str = "gene_interaction"
) -> HeteroData:
    """COO of per-sample bin vectors: sample s contributes bin j at type index j."""
    values: list[float] = []
    types: list[int] = []
    samples: list[int] = []
    for sample, vector in rows.items():
        for j, v in enumerate(vector):
            values.append(v)
            types.append(j)
            samples.append(sample)
    names = [f"{label}_bin_{j}" for j in range(num_bins)]
    return _coo(values, types, samples, names)


def test_categorical_inverse_draws_seeded_uniforms_in_the_argmax_bin() -> None:
    """Samples are decoded in sorted order (3 before 5), each consuming one draw of the
    seed-42 stream: sample 3 argmax bin 0 -> 0 + 0.8822693; sample 5 argmax bin 2 ->
    2 + 0.9150040. A NaN in sample 8's vector decodes to NaN. Calling again reseeds, so
    the result repeats exactly.
    """
    t = _bins("categorical")
    data = _binned(
        {
            5: [0.0, 0.0, 1.0, 0.0],
            3: [1.0, 0.0, 0.0, 0.0],
            8: [float("nan"), 0.0, 0.0, 0.0],
        },
        4,
    )
    out = t.inverse(data)["gene"]
    expected = torch.tensor([SEED42_U0, 2.0 + SEED42_U1, float("nan")])
    torch.testing.assert_close(out.phenotype_values, expected, equal_nan=True)
    assert out.phenotype_sample_indices.tolist() == [3, 5, 8]
    assert out.phenotype_type_indices.tolist() == [0, 0, 0]
    assert out.phenotype_types == ["gene_interaction"]

    again = t.inverse(_binned({3: [1.0, 0.0, 0.0, 0.0]}, 4))["gene"]
    torch.testing.assert_close(again.phenotype_values, torch.tensor([SEED42_U0]))


def test_categorical_inverse_fills_missing_bins_with_zero() -> None:
    """Sample 0 carries only its bin-2 entry; bins 0, 1, 3 are filled with 0, so the
    vector is [0, 0, 1, 0] and the draw lands in [2, 3): 2 + 0.8822693.
    """
    t = _bins("categorical")
    data = _coo([1.0], [2], [0], [f"gene_interaction_bin_{j}" for j in range(4)])
    out = t.inverse(data)["gene"]
    torch.testing.assert_close(out.phenotype_values, torch.tensor([2.0 + SEED42_U0]))


def test_ordinal_round_trip_counts_crossings_under_the_given_seed() -> None:
    """Forward: 2.5 crosses thresholds 1 and 2, giving [1, 1, 0].

    Contract (issue #521): the ordinal forward emits num_bins - 1 = 3 values per sample
    (one per interior edge) and names exactly 3 types, `gene_interaction_bin_0..2`,
    on type indices [0, 1, 2]; `label_width` reports 3 for ordinal and 4 for one-hot.

    Inverse of that output: two crossings select bin 2 = [2, 3), so the default seed 42
    gives 2 + 0.8822693 and `seed=0` gives 2 + 0.4962566 (the first `torch.rand(1)`
    after `torch.manual_seed(0)`).
    """
    t = _bins("ordinal")
    forward = t(_coo([2.5], [0], [0], ["gene_interaction"]))["gene"]
    torch.testing.assert_close(forward.phenotype_values, torch.tensor([1.0, 1.0, 0.0]))
    assert forward.phenotype_type_indices.tolist() == [0, 1, 2]
    assert forward.phenotype_types == [
        "gene_interaction_bin_0",
        "gene_interaction_bin_1",
        "gene_interaction_bin_2",
    ]
    assert t.label_width("gene_interaction") == 3
    assert _bins("categorical").label_width("gene_interaction") == 4
    snapshot = forward.phenotype_values.clone()
    seeded = t.inverse(
        _coo(snapshot.tolist(), [0, 1, 2], [0, 0, 0], forward.phenotype_types)
    )
    torch.testing.assert_close(
        seeded["gene"].phenotype_values, torch.tensor([2.0 + SEED42_U0])
    )
    zero = t.inverse(
        _coo(snapshot.tolist(), [0, 1, 2], [0, 0, 0], forward.phenotype_types), seed=0
    )
    torch.testing.assert_close(
        zero["gene"].phenotype_values, torch.tensor([2.0 + 0.49625658988952637])
    )


def test_soft_inverse_averages_a_five_bin_window_of_probabilities() -> None:
    """Eight bins over [0, 8] (centers 0.5..7.5), `label_df` 0..8.

    Contract (issue #521): soft labels are already probabilities, so the decode weights
    the bin centers by them directly (no softmax). For p = [0, 0, 0, 0.2, 0.4, 0.1, 0, 0]
    the argmax is bin 4 and the window is bins 2..6, p = [0, 0.2, 0.4, 0.1, 0] on centers
    [2.5, 3.5, 4.5, 5.5, 6.5], so the decode is
    (0 * 2.5 + 0.2 * 3.5 + 0.4 * 4.5 + 0.1 * 5.5 + 0 * 6.5) / (0.2 + 0.4 + 0.1)
    = (0.7 + 1.8 + 0.55) / 0.7 = 3.05 / 0.7 = 4.357142857 (the old softmax of p gave
    4.480023). Near an end the window is short (argmax bin 1: bins 0..3, 4 < 5) and the
    decode snaps to the center 1.5.
    """
    df = pd.DataFrame({"gene_interaction": np.arange(9.0)})
    t = COOLabelBinningTransform(
        _dataset(df),
        {
            "gene_interaction": {
                "strategy": "equal_width",
                "num_bins": 8,
                "label_type": "soft",
            }
        },
    )
    data = _binned(
        {
            0: [0.0, 0.0, 0.0, 0.2, 0.4, 0.1, 0.0, 0.0],
            1: [0.1, 0.6, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
        },
        8,
    )
    out = t.inverse(data)["gene"]
    torch.testing.assert_close(out.phenotype_values, torch.tensor([3.05 / 0.7, 1.5]))


def test_inverse_ignores_types_that_are_not_bins_of_a_configured_label() -> None:
    """No phenotype_values: same object. Types that are not `<label>_bin_<j>` of a
    configured label decode nothing, so the COO is left as it was.
    """
    t = _bins("categorical")
    bare = HeteroData()
    bare["gene"].x = torch.tensor([1.0])
    assert t.inverse(bare) is bare

    data = _coo([0.3], [0], [0], [["fitness_bin_0"]])
    out = t.inverse(data)["gene"]
    torch.testing.assert_close(out.phenotype_values, torch.tensor([0.3]))
    assert out.phenotype_types == [["fitness_bin_0"]]


class _Recorder:
    """A stand-in transform whose `inverse` logs its name and returns its input."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def inverse(self, data: HeteroData) -> HeteroData:
        self.log.append(self.name)
        return data

    def __repr__(self) -> str:
        return f"_Recorder({self.name})"


def test_inverse_compose_runs_inverses_last_first_and_reprs_its_members() -> None:
    """A list [a, b] is inverted as b then a, each seeing the previous output (here the
    same object, since `forward` is called directly); the repr lists members one per
    line.
    """
    log: list[str] = []
    compose = COOInverseCompose([_Recorder("a", log), _Recorder("b", log)])
    data = HeteroData()
    assert compose.forward(data) is data
    assert log == ["b", "a"]
    assert repr(compose) == "COOInverseCompose(\n  _Recorder(a)\n  _Recorder(b)\n)"


def test_inverse_compose_rejects_bad_inputs() -> None:
    """A tuple is neither a Compose nor a list; a member without `inverse` is named."""
    with pytest.raises(
        ValueError,
        match=r"^transforms must be either a Compose object or a list of transforms$",
    ):
        COOInverseCompose((_robust(),))
    with pytest.raises(
        ValueError, match=r"^Transform Compose does not implement inverse method$"
    ):
        COOInverseCompose([Compose([])])


# ---------------------------------------------------------------------------------------
# Phase 24: a strategy rewritten after construction, a configured label absent from the
# data, and a 0-d value under the inverse (coo_regression_to_classification.py:121, 141,
# 572, 664).
# ---------------------------------------------------------------------------------------


def test_a_strategy_rewritten_after_construction_is_refused_both_ways() -> None:
    """The constructor validates the strategy, so only a mutated ``stats`` entry reaches
    the final ``else`` of ``normalize`` and ``denormalize``; both name it.
    """
    t = _robust()
    t.stats["fitness"]["strategy"] = "bogus"
    values = torch.tensor([1.0])
    with pytest.raises(
        ValueError, match=re.escape("Unknown normalization strategy: bogus")
    ):
        t.normalize(values, "fitness")
    with pytest.raises(
        ValueError, match=re.escape("Unknown normalization strategy: bogus")
    ):
        t.denormalize(values, "fitness")


def test_binning_skips_a_configured_label_the_data_does_not_carry() -> None:
    """Two configured labels, data with gene_interaction only: fitness is skipped
    (line 572) and gene_interaction 2.5 is one-hot encoded in bin 2 of edges
    [0, 1, 2, 3, 4], under the types gene_interaction_bin_0..3.
    """
    df = pd.DataFrame(
        {
            "gene_interaction": [0.0, 1.0, 2.0, 3.0, 4.0],
            "fitness": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
    )
    config = {
        label: {"strategy": "equal_width", "num_bins": 4, "label_type": "categorical"}
        for label in ["fitness", "gene_interaction"]
    }
    t = COOLabelBinningTransform(_dataset(df), config)
    out = t(_coo([2.5], [0], [0], ["gene_interaction"]))["gene"]
    torch.testing.assert_close(out.phenotype_values, torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert out.phenotype_types == [f"gene_interaction_bin_{j}" for j in range(4)]
    assert "fitness_continuous" not in out.keys()


def test_inverse_treats_a_0d_value_as_one_entry() -> None:
    """A 0-d 1.0 with type index [2] (bin 2 of 4), sample 0: unsqueezed to one entry
    (line 664), the other bins are filled with 0, the argmax is bin 2, and the seed-42
    draw gives 2 + 0.8822693.
    """
    t = _bins("categorical")
    data = _coo(1.0, [2], [0], [f"gene_interaction_bin_{j}" for j in range(4)])
    assert data["gene"].phenotype_values.dim() == 0
    out = t.inverse(data)["gene"]
    torch.testing.assert_close(out.phenotype_values, torch.tensor([2.0 + SEED42_U0]))
    assert out.phenotype_sample_indices.tolist() == [0]
    assert out.phenotype_types == ["gene_interaction"]
