# tests/torchcell/transforms/test_regression_to_classification.py
# [[tests.torchcell.transforms.test_regression_to_classification]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/transforms/test_regression_to_classification.py
"""Tests for the regression-to-classification label transforms.

2026.09.30 (Phase 14): closed-form tests for the branches the classes above left open,
on a stand-in dataset whose ``label_df`` fitness is [0, 1, 2, 3, 4, inf] (the inf becomes
NaN and is dropped), each number checked with a two-line Python run.

* Statistics: mean 2, population std sqrt(2), min 0, max 4, q25 1, q75 3. Robust
  scaling x -> (x - 1) / (2 + 1e-8) maps [0, 1, 4] to [-0.5, 0, 1.5]; minmax
  x / (4 + 1e-8) maps the equal-width edges [0..4] to [0, .25, .5, .75, 1].
* Auto bins: int(range / std) = int(4 / 1.414) = 2, edges [0, 2, 4].
* One-hot on edges [0, 1, 2, 3, 4] is left-closed after clamping, the right edge folds
  into bin 3. Soft on edges [0, 1, 2] with sigma 1: 0.5 -> [1, e^-0.5] / (1 + e^-0.5).
* Seeded inverses: after ``torch.manual_seed(42)`` the scalar draws are 0.8822692632675171,
  0.9150039553642273, 0.38286375999450684; bins are visited in ascending order and each
  bin draws one uniform per row in row order, value = low + r * (high - low).
* Soft inverse on six unit bins: probabilities proportional to [1, 1, 3, 2, 1, 1] give
  (0.5 + 1.5 + 7.5 + 7 + 4.5) / 8 = 2.625; a peak within two bins of an edge returns
  its center.
"""

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import Compose

from torchcell.transforms.regression_to_classification import (
    AutoBinStrategy,
    EqualWidthStrategy,
    InverseCompose,
    LabelBinningTransform,
    LabelNormalizationTransform,
)


class TestLabelNormalizationTransform:
    """Tests for the forward direction of LabelNormalizationTransform."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with fitness and gene-interaction label columns."""

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
        """Return a transform using minmax for fitness and standard for interaction."""
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return LabelNormalizationTransform(mock_dataset, label_configs)

    def test_minmax_normalization(self, norm_transform):
        """Verify minmax normalization and its inverse round-trip for fitness."""
        data = HeteroData()
        data["gene"]["fitness"] = torch.tensor([0.0, 0.5, 1.0, 2.0])

        normalized = norm_transform(data)
        expected = torch.tensor([0.0, 0.25, 0.5, 1.0])
        assert torch.allclose(normalized["gene"]["fitness"], expected)

        denormalized = norm_transform.inverse(normalized)
        assert torch.allclose(denormalized["gene"]["fitness"], data["gene"]["fitness"])

    def test_standard_normalization(self, norm_transform):
        """Verify standard normalization matches the manual mean/std computation."""
        data = HeteroData()
        data["gene"]["gene_interaction"] = torch.tensor([-1.0, 0.0, 1.0])

        normalized = norm_transform(data)
        stats = norm_transform.stats["gene_interaction"]
        mean = stats["mean"]
        std = stats["std"]
        expected = (data["gene"]["gene_interaction"] - mean) / std
        assert torch.allclose(normalized["gene"]["gene_interaction"], expected)


class TestLabelNormalizationInverse:
    """Tests for the inverse of LabelNormalizationTransform."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with fitness and gene-interaction label columns."""

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
        """Return a transform using minmax for fitness and standard for interaction."""
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return LabelNormalizationTransform(mock_dataset, label_configs)

    # TestLabelNormalizationInverse::test_inverse_minmax
    def test_inverse_minmax(self, norm_transform):
        """Verify the minmax inverse recovers the original fitness values."""
        # Test data
        original_values = torch.tensor([0.0, 0.5, 1.0, 2.0])
        normalized_values = torch.tensor([0.0, 0.25, 0.5, 1.0])

        # Denormalize using the transform
        temp_data = HeteroData()
        temp_data["gene"] = {"fitness": normalized_values}
        denormalized = norm_transform.inverse(temp_data)

        # Validate against the original values
        assert torch.allclose(
            denormalized["gene"]["fitness"], original_values, atol=1e-6
        )

    def test_inverse_standard(self, norm_transform):
        """Verify the standard inverse recovers the original interaction values."""
        # Retrieve stats from the transform
        stats = norm_transform.stats["gene_interaction"]
        mean = stats["mean"]
        std = stats["std"]

        # Test data
        original_values = torch.tensor([-1.0, 0.0, 1.0])
        normalized_values = (original_values - mean) / std

        # Denormalize using the transform
        temp_data = HeteroData()
        temp_data["gene"] = {"gene_interaction": normalized_values}
        denormalized = norm_transform.inverse(temp_data)

        # Validate against the original values
        assert torch.allclose(
            denormalized["gene"]["gene_interaction"], original_values, atol=1e-6
        )

    def test_inverse_with_nans(self, norm_transform):
        """Verify the inverse preserves NaN entries during denormalization."""
        # Test data with NaN
        original_values = torch.tensor([0.0, 0.5, 1.0, float("nan")])
        normalized_values = torch.tensor([0.0, 0.25, 0.5, float("nan")])

        # Denormalize using the transform
        temp_data = HeteroData()
        temp_data["gene"] = {"fitness": normalized_values}
        denormalized = norm_transform.inverse(temp_data)

        # Validate against the original values, allowing NaN comparisons
        assert torch.allclose(
            denormalized["gene"]["fitness"], original_values, equal_nan=True
        )


class TestLabelNormalizationRoundTrip:
    """Tests for normalize-then-inverse round trips."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with fitness and gene-interaction label columns."""

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
        """Return a transform using minmax for fitness and standard for interaction."""
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return LabelNormalizationTransform(mock_dataset, label_configs)

    # TestLabelNormalizationRoundTrip::test_round_trip_minmax
    def test_round_trip_minmax(self, norm_transform, mock_dataset):
        """Verify a minmax normalize/inverse round trip recovers the input."""
        # Test data from mock_dataset
        original_df = mock_dataset.label_df
        original_values = torch.tensor(original_df["fitness"].values)

        # Normalize
        data = HeteroData()
        data["gene"] = {"fitness": original_values}
        normalized = norm_transform(data)

        # Inverse transform
        recovered = norm_transform.inverse(normalized)

        # Validate round trip
        assert torch.allclose(recovered["gene"]["fitness"], original_values, atol=1e-6)

    def test_round_trip_standard(self, norm_transform, mock_dataset):
        """Verify a standard normalize/inverse round trip recovers the input."""
        # Test data from mock_dataset
        original_df = mock_dataset.label_df
        original_values = torch.tensor(original_df["gene_interaction"].values)

        # Normalize
        data = HeteroData()
        data["gene"] = {"gene_interaction": original_values}
        normalized = norm_transform(data)

        # Inverse transform
        recovered = norm_transform.inverse(normalized)

        # Validate round trip, allowing for NaN comparisons
        assert torch.allclose(
            recovered["gene"]["gene_interaction"], original_values, equal_nan=True
        )


class TestLabelBinningTransform:
    """Tests for the forward and inverse of LabelBinningTransform."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with label columns used to fit bin edges."""

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
        """Return a binning transform configured over the mock dataset."""
        label_configs = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 10,
                "label_type": "soft",
                "sigma": 0.5,
            },
            "gene_interaction": {
                "strategy": "equal_frequency",
                "num_bins": 10,
                "label_type": "categorical",
            },
        }
        return LabelBinningTransform(mock_dataset, label_configs)

    def test_soft_binning_forward(self, bin_transform):
        """Verify soft binning produces a valid probability distribution."""
        data = HeteroData()
        data["gene"]["fitness"] = torch.tensor([0.0, 0.3, 0.7, 1.0])

        binned = bin_transform(data)
        assert binned["gene"]["fitness"].shape == (4, 10)
        assert torch.allclose(binned["gene"]["fitness"].sum(dim=1), torch.ones(4))

    def test_categorical_binning_forward(self, bin_transform):
        """Verify categorical binning produces class indices."""
        data = HeteroData()
        data["gene"]["gene_interaction"] = torch.tensor([-1.0, -0.3, 0.3, 1.0])

        binned = bin_transform(data)
        assert binned["gene"]["gene_interaction"].shape == (4, 10)
        assert torch.all(binned["gene"]["gene_interaction"].sum(dim=1) == 1)

    def test_inverse_soft_binning(self, bin_transform):
        """Verify the inverse of soft binning reconstructs continuous values."""
        data = HeteroData()
        logits = torch.zeros(4, 10)
        logits[0, 0] = 10.0
        logits[1, 3] = 10.0
        logits[2, 7] = 10.0
        logits[3, 9] = 10.0
        data["gene"]["fitness"] = logits

        inverted = bin_transform.inverse(data)
        bin_edges = torch.tensor(bin_transform.label_metadata["fitness"]["bin_edges"])

        # Instead of checking exact closeness to expected bin centers,
        # we only check that inverted values are within their respective bin edges.
        # For a soft label with a clear peak, the recovered value should lie within
        # the bin corresponding to the max logit.

        max_bins = torch.argmax(logits, dim=-1)
        for i, mb in enumerate(max_bins):
            low = bin_edges[mb]
            high = bin_edges[mb + 1]
            val = inverted["gene"]["fitness"][i].float()  # ensure float
            assert low <= val <= high

    def test_inverse_categorical_binning(self, bin_transform):
        """Verify the inverse of categorical binning reconstructs values."""
        data = HeteroData()
        logits = torch.zeros(4, 10)
        # No single peak, but they should still be assigned to bin 0 by default
        logits[0, 0] = 1.0
        logits[1, 5] = 1.0
        logits[2, 9] = 1.0
        logits[3, 2] = 1.0
        data["gene"]["gene_interaction"] = logits

        inverted = bin_transform.inverse(data)
        bin_edges = torch.tensor(
            bin_transform.label_metadata["gene_interaction"]["bin_edges"]
        )

        # Check that inverted values fall within correct bins
        indices = torch.argmax(logits, dim=-1)
        for i, idx in enumerate(indices):
            low = bin_edges[idx]
            high = bin_edges[idx + 1]
            val = inverted["gene"]["gene_interaction"][i].float()
            assert low <= val <= high


class TestInverseCompose:
    """Tests for chaining transforms and inverting the composition."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with label columns for the composed transforms."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {
                        "fitness": np.concatenate(
                            [
                                np.linspace(0, 0.3, 33),
                                np.linspace(0.3, 0.7, 34),
                                np.linspace(0.7, 1.0, 33),
                                np.linspace(1.0, 1.5, 33),
                            ]
                        )
                    }
                )

        return MockDataset()

    @pytest.fixture
    def transforms(self, mock_dataset):
        """Return a composed normalization-then-binning transform."""
        norm_config = {"fitness": {"strategy": "minmax"}}
        bin_config = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 8,
                "label_type": "soft",
                "sigma": 0.1,
            }
        }

        norm_transform = LabelNormalizationTransform(mock_dataset, norm_config)
        bin_transform = LabelBinningTransform(mock_dataset, bin_config, norm_transform)
        return [norm_transform, bin_transform]

    # TestInverseCompose::test_inverse_compose
    def test_inverse_compose(self, transforms):
        """Verify inverting a composed transform recovers the original input."""
        data = HeteroData()
        data["gene"]["fitness"] = torch.tensor(
            [0.0, 0.1, 0.3, 0.7, 0.8, 0.8, 0.9, 1.0, 1.0, 1.1, 1.3, 1.5]
        )

        forward_transform = Compose(transforms)
        inverse_transform = InverseCompose(forward_transform)

        transformed = forward_transform(data)
        recovered = inverse_transform(transformed)

        # Now we can directly access the denormalized bin edges from label_metadata
        bin_edges_denormalized = transforms[1].label_metadata["fitness"][
            "bin_edges_denormalized"
        ]
        bin_edges_denormalized = torch.tensor(bin_edges_denormalized, dtype=torch.float)

        eps = 1e-12
        logits = transformed["gene"]["fitness"]
        max_bins = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)

        for i, mb in enumerate(max_bins):
            low = bin_edges_denormalized[mb] - eps
            high = bin_edges_denormalized[mb + 1] + eps
            val = recovered["gene"]["fitness"][i].float()
            assert low <= val <= high, f"Value {val} not within [{low}, {high}]"


class TestOrdinalBinning:
    """Tests for ordinal binning forward and inverse behavior."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with label columns used to fit ordinal bins."""

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
        """Return an ordinal binning transform over the mock dataset."""
        label_configs = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,  # Using 4 bins for simpler testing
                "label_type": "ordinal",
            }
        }
        return LabelBinningTransform(mock_dataset, label_configs)

    def test_ordinal_forward(self, bin_transform):
        """Verify ordinal binning produces the expected cumulative encoding."""
        data = HeteroData()
        data["gene"]["fitness"] = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        # Apply forward transform
        transformed = bin_transform(data)
        ordinal_labels = transformed["gene"]["fitness"]

        # Check shape: should be (5, 3) since 4 bins means 3 thresholds
        assert ordinal_labels.shape == (5, 3)

        # Check if values are binary (0 or 1)
        assert torch.all((ordinal_labels == 0) | (ordinal_labels == 1))

        # Check ordinal property: if a higher threshold is 1, all lower thresholds should be 1
        for i in range(ordinal_labels.shape[0]):
            for j in range(1, ordinal_labels.shape[1]):
                if ordinal_labels[i, j] == 1:
                    assert torch.all(ordinal_labels[i, :j] == 1)

    def test_ordinal_inverse(self, bin_transform):
        """Verify the ordinal inverse reconstructs continuous values."""
        data = HeteroData()
        ordinal_labels = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # lowest bin
                [1.0, 0.0, 0.0],  # second bin
                [1.0, 1.0, 0.0],  # third bin
                [1.0, 1.0, 1.0],  # highest bin
            ]
        )
        data["gene"]["fitness"] = ordinal_labels

        # Apply inverse transform
        recovered = bin_transform.inverse(data)
        continuous_values = recovered["gene"]["fitness"]

        # Get bin edges from transform
        bin_edges = torch.tensor(bin_transform.label_metadata["fitness"]["bin_edges"])

        # Check that values fall within expected bins
        assert (
            continuous_values[0] >= bin_edges[0]
            and continuous_values[0] <= bin_edges[1]
        )
        assert (
            continuous_values[1] >= bin_edges[1]
            and continuous_values[1] <= bin_edges[2]
        )
        assert (
            continuous_values[2] >= bin_edges[2]
            and continuous_values[2] <= bin_edges[3]
        )
        assert (
            continuous_values[3] >= bin_edges[3]
            and continuous_values[3] <= bin_edges[4]
        )

    # TestOrdinalBinning::test_ordinal_round_trip
    def test_ordinal_round_trip(self, bin_transform):
        """Verify an ordinal forward/inverse round trip recovers the input."""
        # Original data
        data = HeteroData()
        original_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        data["gene"]["fitness"] = original_values

        # Forward transform
        transformed = bin_transform(data)

        # Inverse transform
        recovered = bin_transform.inverse(transformed)
        recovered_values = recovered["gene"]["fitness"]

        # Get bin edges
        bin_edges = torch.tensor(bin_transform.label_metadata["fitness"]["bin_edges"])

        # Function to determine bin index that handles edge cases
        def get_bin_index(value, edges):
            if value <= edges[0]:
                return 0
            if value >= edges[-1]:
                return len(edges) - 2
            idx = torch.searchsorted(edges, value.item()) - 1
            return idx

        # Check that each recovered value falls in the same bin as its original value
        for orig, rec in zip(original_values, recovered_values):
            orig_bin = get_bin_index(orig, bin_edges)
            rec_bin = get_bin_index(rec, bin_edges)
            assert orig_bin == rec_bin, (
                f"Original value {orig} and recovered value {rec} are in different bins"
            )

    def test_ordinal_with_nans(self, bin_transform):
        """Verify ordinal binning preserves NaN entries."""
        data = HeteroData()
        data["gene"]["fitness"] = torch.tensor(
            [0.0, float("nan"), 0.5, float("nan"), 1.0]
        )

        # Forward transform
        transformed = bin_transform(data)
        ordinal_labels = transformed["gene"]["fitness"]

        # Check that NaN values are preserved in forward transform
        assert torch.isnan(ordinal_labels[1]).all()
        assert torch.isnan(ordinal_labels[3]).all()

        # Inverse transform
        recovered = bin_transform.inverse(transformed)
        recovered_values = recovered["gene"]["fitness"]

        # Check that NaN values are preserved in inverse transform
        assert torch.isnan(recovered_values[1])
        assert torch.isnan(recovered_values[3])

        # Check that non-NaN values are handled correctly
        bin_edges = torch.tensor(bin_transform.label_metadata["fitness"]["bin_edges"])
        assert (
            recovered_values[0] >= bin_edges[0] and recovered_values[0] <= bin_edges[1]
        )
        assert (
            recovered_values[2] >= bin_edges[1] and recovered_values[2] <= bin_edges[-1]
        )
        assert (
            recovered_values[4] >= bin_edges[-2]
            and recovered_values[4] <= bin_edges[-1]
        )


class TestOrdinalNormBinning:
    """Tests for normalization composed with ordinal binning."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with label columns for the composed transforms."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {
                        "fitness": np.array(
                            [
                                -1.0,  # Below 0
                                0.0,  # At 0
                                0.5,  # Middle
                                1.0,  # At 1
                                2.0,  # Above 1
                                np.nan,  # NaN value
                                1.5,  # Another above 1
                                np.nan,  # Another NaN
                            ],
                            dtype=np.float32,
                        )
                    }
                )

        return MockDataset()

    @pytest.fixture
    def transforms(self, mock_dataset):
        """Return a composed normalization-then-ordinal-binning transform."""
        norm_config = {"fitness": {"strategy": "minmax"}}
        norm_transform = LabelNormalizationTransform(mock_dataset, norm_config)

        bin_config = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "ordinal",
            }
        }
        bin_transform = LabelBinningTransform(
            mock_dataset, bin_config, normalizer=norm_transform
        )
        return [norm_transform, bin_transform]

    # TestOrdinalNormBinning::test_full_round_trip_with_stages
    def test_full_round_trip_with_stages(self, transforms, mock_dataset):
        """Test the full round trip, verifying intermediate stages"""
        norm_transform, bin_transform = transforms

        # Original data
        data = HeteroData()
        original_values = torch.tensor(
            [-1.0, 0.0, 0.5, 1.0, 2.0, float("nan"), 1.5, float("nan")],
            dtype=torch.float32,
        )
        data["gene"]["fitness"] = original_values

        # Create composed transforms
        forward_transform = Compose(transforms)
        inverse_transform = InverseCompose(forward_transform)

        # Do full forward transform
        transformed = forward_transform(data)
        ordinal_labels = transformed["gene"]["fitness"]

        # Verify ordinal properties
        valid_mask = ~torch.isnan(ordinal_labels).any(dim=1)
        assert torch.all(
            (ordinal_labels[valid_mask] == 0) | (ordinal_labels[valid_mask] == 1)
        )

        # Check ordinal property (if higher threshold is 1, lower ones must be 1)
        for i in range(len(ordinal_labels)):
            if valid_mask[i]:
                for j in range(1, ordinal_labels.shape[1]):
                    if ordinal_labels[i, j] == 1:
                        assert torch.all(ordinal_labels[i, :j] == 1)

        # Verify NaN preservation in transform
        nan_mask = torch.isnan(original_values)
        assert torch.all(torch.isnan(ordinal_labels[nan_mask]).all(dim=1))

        # Do full inverse transform
        recovered = inverse_transform(transformed)
        recovered_values = recovered["gene"]["fitness"]

        # Get stored denormalized bin edges
        bin_edges = torch.tensor(
            bin_transform.label_metadata["fitness"]["bin_edges_denormalized"],
            dtype=torch.float32,
        )

        def get_bin_index(value, edges):
            if torch.isnan(value):
                return -1
            if value <= edges[0]:
                return 0
            if value >= edges[-1]:
                return len(edges) - 2
            idx = torch.searchsorted(edges, value.item()) - 1
            return idx

        # Check bins and NaN preservation
        for orig, rec in zip(original_values, recovered_values):
            if torch.isnan(orig):
                assert torch.isnan(rec), f"NaN value not preserved: got {rec}"
            else:
                orig_bin = get_bin_index(orig, bin_edges)
                rec_bin = get_bin_index(rec, bin_edges)
                assert orig_bin == rec_bin, (
                    f"Original value {orig} (bin {orig_bin}) and "
                    f"recovered value {rec} (bin {rec_bin}) are in different bins"
                )

        # Verify intermediate normalized values are in [0,1]
        normalized = norm_transform(data)
        norm_values = normalized["gene"]["fitness"]
        valid_mask = ~torch.isnan(norm_values)
        assert torch.all(norm_values[valid_mask] >= 0)
        assert torch.all(norm_values[valid_mask] <= 1)

    # TestOrdinalNormBinning::test_ordinal_roundtrip_bin_containment
    def test_ordinal_roundtrip_bin_containment(self, transforms, mock_dataset):
        """Test that inverse transform outputs values within their predicted bin ranges for ordinal labels."""
        norm_transform, bin_transform = transforms

        # Original data
        data = HeteroData()
        original_values = torch.tensor(
            [-1.0, 0.0, 0.5, 1.0, 2.0, float("nan"), 1.5, float("nan")],
            dtype=torch.float32,
        )
        data["gene"]["fitness"] = original_values

        # Create composed transforms
        forward_transform = Compose(transforms)
        inverse_transform = InverseCompose(forward_transform)

        # Do forward transform
        transformed = forward_transform(data)

        # Get the predicted bins from the ordinal encoding
        ordinal_labels = transformed["gene"]["fitness"]
        valid_mask = ~torch.isnan(ordinal_labels).any(dim=1)
        predicted_bins = torch.sum(ordinal_labels[valid_mask] > 0.5, dim=1)

        # Do inverse transform
        recovered = inverse_transform(transformed)
        recovered_values = recovered["gene"]["fitness"]

        # Get bin edges
        bin_edges_denorm = torch.tensor(
            bin_transform.label_metadata["fitness"]["bin_edges_denormalized"],
            dtype=torch.float32,
        )

        # Check that each recovered value falls within its predicted bin
        for i, (rec_val, pred_bin) in enumerate(
            zip(recovered_values[valid_mask], predicted_bins)
        ):
            bin_start = bin_edges_denorm[pred_bin]
            bin_end = bin_edges_denorm[pred_bin + 1]

            assert bin_start <= rec_val <= bin_end, (
                f"Recovered value {rec_val} is outside its predicted bin "
                f"[{bin_start}, {bin_end}] for bin {pred_bin}"
            )

        # Verify NaN values are preserved
        assert torch.all(torch.isnan(recovered_values) == torch.isnan(original_values))


class TestModelOutputSimulation:
    """Tests that inverse transforms recover simulated model outputs."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with label columns for the simulation tests."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {"fitness": np.linspace(0, 1, 100)}  # Nice distribution from 0 to 1
                )

        return MockDataset()

    def test_categorical_logits(self, mock_dataset):
        """Test categorical classification with model-like logit outputs."""
        # Setup transforms
        norm_config = {"fitness": {"strategy": "minmax"}}
        bin_config = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "categorical",
            }
        }

        norm_transform = LabelNormalizationTransform(mock_dataset, norm_config)
        bin_transform = LabelBinningTransform(mock_dataset, bin_config, norm_transform)
        transforms = [norm_transform, bin_transform]
        forward_transform = Compose(transforms)
        inverse_transform = InverseCompose(forward_transform)

        # Simulate model outputs (large logit differences for clear predictions)
        batch_size = 4
        num_bins = 4
        logits = torch.zeros(batch_size, num_bins)
        # Each row strongly predicts a different bin
        for i in range(batch_size):
            logits[i, i] = 10.0  # High confidence prediction

        # Create data object with our "model outputs"
        pred_data = HeteroData()
        pred_data["gene"] = {"fitness": logits}

        # Apply inverse transform
        recovered = inverse_transform(pred_data)
        recovered_values = recovered["gene"]["fitness"]

        # Get bin edges for verification
        bin_edges = torch.tensor(
            bin_transform.label_metadata["fitness"]["bin_edges_denormalized"],
            dtype=torch.float32,
        )

        # Verify each value falls within its predicted bin
        for i in range(batch_size):
            pred_bin = i  # We constructed logits to predict bin i for row i
            bin_start = bin_edges[pred_bin]
            bin_end = bin_edges[pred_bin + 1]

            assert bin_start <= recovered_values[i] <= bin_end, (
                f"Value {recovered_values[i]} not in bin {i} range [{bin_start}, {bin_end}]"
            )

    def test_soft_logits(self, mock_dataset):
        """Test soft classification with model-like logit outputs."""
        # Setup transforms
        norm_config = {"fitness": {"strategy": "minmax"}}
        bin_config = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "soft",
                "sigma": 0.1,
            }
        }

        norm_transform = LabelNormalizationTransform(mock_dataset, norm_config)
        bin_transform = LabelBinningTransform(mock_dataset, bin_config, norm_transform)
        transforms = [norm_transform, bin_transform]
        forward_transform = Compose(transforms)
        inverse_transform = InverseCompose(forward_transform)

        # Simulate model outputs (softer predictions)
        batch_size = 4
        num_bins = 4
        logits = torch.zeros(batch_size, num_bins)
        # Create predictions with some uncertainty
        for i in range(batch_size):
            logits[i, i] = 5.0  # Main prediction
            if i > 0:
                logits[i, i - 1] = 2.0  # Some weight to adjacent bin
            if i < num_bins - 1:
                logits[i, i + 1] = 2.0  # Some weight to adjacent bin

        # Create data object with our "model outputs"
        pred_data = HeteroData()
        pred_data["gene"] = {"fitness": logits}

        # Apply inverse transform
        recovered = inverse_transform(pred_data)
        recovered_values = recovered["gene"]["fitness"]

        # Get bin edges for verification
        bin_edges = torch.tensor(
            bin_transform.label_metadata["fitness"]["bin_edges_denormalized"],
            dtype=torch.float32,
        )

        # Verify each value falls within expected range
        for i in range(batch_size):
            pred_bin = i  # Peak prediction is still bin i
            bin_start = bin_edges[pred_bin]
            bin_end = bin_edges[pred_bin + 1]

            assert bin_start <= recovered_values[i] <= bin_end, (
                f"Value {recovered_values[i]} not in bin {i} range [{bin_start}, {bin_end}]"
            )

    def test_ordinal_logits(self, mock_dataset):
        """Test ordinal classification with model-like logit outputs."""
        # Setup transforms
        norm_config = {"fitness": {"strategy": "minmax"}}
        bin_config = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "ordinal",
            }
        }

        norm_transform = LabelNormalizationTransform(mock_dataset, norm_config)
        bin_transform = LabelBinningTransform(mock_dataset, bin_config, norm_transform)
        transforms = [norm_transform, bin_transform]
        forward_transform = Compose(transforms)
        inverse_transform = InverseCompose(forward_transform)

        # Simulate model outputs for ordinal case (n-1 thresholds for n bins)
        batch_size = 4
        num_thresholds = 3  # 4 bins = 3 thresholds
        logits = torch.zeros(batch_size, num_thresholds)

        # Create different threshold patterns
        # Row 0: All negative = bin 0
        logits[0] = torch.tensor([-10.0, -10.0, -10.0])
        # Row 1: One positive = bin 1
        logits[1] = torch.tensor([10.0, -10.0, -10.0])
        # Row 2: Two positive = bin 2
        logits[2] = torch.tensor([10.0, 10.0, -10.0])
        # Row 3: All positive = bin 3
        logits[3] = torch.tensor([10.0, 10.0, 10.0])

        # Create data object with our "model outputs"
        pred_data = HeteroData()
        pred_data["gene"] = {"fitness": logits}

        # Apply inverse transform
        recovered = inverse_transform(pred_data)
        recovered_values = recovered["gene"]["fitness"]

        # Get bin edges for verification
        bin_edges = torch.tensor(
            bin_transform.label_metadata["fitness"]["bin_edges_denormalized"],
            dtype=torch.float32,
        )

        # Verify each value falls within its predicted bin
        for i in range(batch_size):
            bin_start = bin_edges[i]  # Constructed so row i should be in bin i
            bin_end = bin_edges[i + 1]

            assert bin_start <= recovered_values[i] <= bin_end, (
                f"Value {recovered_values[i]} not in bin {i} range [{bin_start}, {bin_end}]"
            )


class TestInverseComposeWithGrads:
    """Tests that inverse compositions preserve gradient flow."""

    @pytest.fixture
    def mock_dataset(self):
        """Return a dataset with label columns for the gradient tests."""

        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame({"fitness": np.linspace(0, 1, 100)})

        return MockDataset()

    @pytest.fixture
    def transforms(self, mock_dataset):
        """Return a composed transform used to check gradient propagation."""
        norm_config = {"fitness": {"strategy": "minmax"}}
        norm_transform = LabelNormalizationTransform(mock_dataset, norm_config)
        return [norm_transform]

    def test_inverse_compose_with_grad_tensors(self, transforms):
        """Test that InverseCompose works with tensors that have gradients."""
        # Create a tensor with gradients
        data = HeteroData()
        x = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
        data["gene"] = {"fitness": x}

        # Create forward and inverse transforms
        forward_transform = Compose(transforms)
        inverse_transform = InverseCompose(transforms)

        # Apply forward transform
        transformed = forward_transform(data)

        # Create a "model" output by doing a simple operation that preserves gradients
        model_output = transformed["gene"]["fitness"] * 0.5 + 0.25

        # Create a new data object with model output
        pred_data = HeteroData()
        pred_data["gene"] = {"fitness": model_output}

        # This should not raise an error even though model_output has gradients
        recovered = inverse_transform(pred_data)

        # Verify the output is reasonable
        assert recovered["gene"]["fitness"] is not None
        assert recovered["gene"]["fitness"].shape == x.shape

        # Verify we can backpropagate through this operation
        loss = recovered["gene"]["fitness"].sum()
        loss.backward()

        # x should now have gradients
        assert x.grad is not None


# ---------------------------------------------------------------------------
# 2026.09.30 (Phase 14): closed-form tests for the branches left open above.


def _five_point_dataset() -> Any:
    """``label_df`` fitness [0, 1, 2, 3, 4, inf]; the inf becomes NaN and is dropped."""
    return SimpleNamespace(
        label_df=pd.DataFrame({"fitness": [0.0, 1.0, 2.0, 3.0, 4.0, np.inf]})
    )


def _gene(values: Any) -> HeteroData:
    data = HeteroData()
    data["gene"]["fitness"] = values
    return data


def _binning(label_type: str, **extra: Any) -> LabelBinningTransform:
    """Equal-width, 4 bins on [0, 4]: edges exactly [0, 1, 2, 3, 4]."""
    config: dict[str, Any] = {
        "strategy": "equal_width",
        "num_bins": 4,
        "label_type": label_type,
        **extra,
    }
    return LabelBinningTransform(_five_point_dataset(), {"fitness": config})


class TestNormalizationBranches:
    """Robust scaling, the all-NaN pass-through, list inputs and the error paths."""

    def test_robust_scaling_and_its_inverse_on_a_list_input(self) -> None:
        """q25 = 1, q75 = 3 on [0..4] (inf dropped), iqr 2: x -> (x - 1) / (2 + 1e-8).

        [0, 1, 4] -> [-0.5, 0, 1.5]; the inverse x * 2 + 1 maps back. A plain list
        input is converted to a float tensor, and the original is kept alongside.
        """
        norm = LabelNormalizationTransform(
            _five_point_dataset(), {"fitness": {"strategy": "robust"}}
        )
        assert norm.stats["fitness"] == {
            "mean": 2.0,
            "std": math.sqrt(2.0),
            "min": 0.0,
            "max": 4.0,
            "q25": 1.0,
            "q75": 3.0,
            "strategy": "robust",
        }
        out = norm(_gene([0.0, 1.0, 4.0]))
        assert torch.allclose(
            out["gene"]["fitness"], torch.tensor([-0.5, 0.0, 1.5]), atol=1e-7
        )
        assert torch.equal(
            out["gene"]["fitness_original"], torch.tensor([0.0, 1.0, 4.0])
        )
        back = norm.inverse(_gene([-0.5, 0.0, 1.5]))
        assert torch.allclose(
            back["gene"]["fitness"], torch.tensor([0.0, 1.0, 4.0]), atol=1e-7
        )

    def test_an_all_nan_tensor_passes_through_both_directions(self) -> None:
        """An all-NaN input is returned as is; one finite value restores the arithmetic."""
        norm = LabelNormalizationTransform(
            _five_point_dataset(), {"fitness": {"strategy": "minmax"}}
        )
        nan2 = torch.full((2,), float("nan"))
        assert torch.isnan(norm.normalize(nan2, "fitness")).all()
        assert torch.isnan(norm.denormalize(nan2, "fitness")).all()
        # a single finite value switches the arithmetic on: (4 - 0) / (4 + 1e-8)
        mixed = norm.normalize(torch.tensor([float("nan"), 4.0]), "fitness")
        assert torch.isnan(mixed[0])
        assert mixed[1].item() == pytest.approx(1.0, abs=1e-7)

    def test_an_unknown_strategy_is_accepted_at_construction_and_refused_at_use(
        self,
    ) -> None:
        """Finding: the constructor stores any strategy string without checking it.

        ``LabelNormalizationTransform.__init__`` (regression_to_classification.py:63)
        records ``config["strategy"]`` verbatim; only ``normalize``/``denormalize``
        (lines 88, 108) refuse it, so a misspelled config survives until the first
        batch. Pinned until the constructor validates the strategy.
        """
        norm = LabelNormalizationTransform(
            _five_point_dataset(), {"fitness": {"strategy": "zscore"}}
        )
        assert norm.stats["fitness"]["strategy"] == "zscore"
        with pytest.raises(
            ValueError, match=r"^Unknown normalization strategy: zscore$"
        ):
            norm.normalize(torch.tensor([1.0]), "fitness")
        with pytest.raises(
            ValueError, match=r"^Unknown normalization strategy: zscore$"
        ):
            norm.denormalize(torch.tensor([1.0]), "fitness")

    def test_a_label_absent_from_the_dataset_is_refused(self) -> None:
        """A configured label missing from ``label_df`` fails the constructor."""
        with pytest.raises(ValueError, match=r"^Label growth not found in dataset$"):
            LabelNormalizationTransform(
                _five_point_dataset(), {"growth": {"strategy": "minmax"}}
            )


class TestBinningBranches:
    """Exact one-hot, soft and ordinal encodings and their seeded inverses."""

    def test_auto_strategy_picks_int_range_over_std_bins(self) -> None:
        """Std of [0..4] is sqrt(2), range 4, int(4 / 1.4142) = 2 bins: edges [0, 2, 4].

        An explicit ``num_bins`` overrides the inference.
        """
        auto = LabelBinningTransform(
            _five_point_dataset(),
            {"fitness": {"strategy": "auto", "label_type": "categorical"}},
        )
        assert auto.get_bin_info("fitness")["bin_edges"].tolist() == [0.0, 2.0, 4.0]
        edges, meta = AutoBinStrategy().compute_bins(np.array([0.0, 4.0, np.nan]), 4)
        assert edges.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert meta["strategy"] == "equal_width"

    def test_get_bin_info_refuses_an_unbinned_label(self) -> None:
        """A label with no binning config has no metadata to return."""
        with pytest.raises(
            ValueError, match=r"^No binning metadata found for label growth$"
        ):
            _binning("categorical").get_bin_info("growth")

    def test_a_binning_label_absent_from_the_dataset_is_refused_with_a_normalizer(
        self,
    ) -> None:
        """With a normalizer the absent column is skipped while normalizing, then refused."""
        norm = LabelNormalizationTransform(
            _five_point_dataset(), {"fitness": {"strategy": "minmax"}}
        )
        with pytest.raises(ValueError, match=r"^Label growth not found in dataset$"):
            LabelBinningTransform(
                _five_point_dataset(),
                {"growth": {"strategy": "equal_width", "num_bins": 2}},
                norm,
            )

    def test_normalized_edges_and_their_denormalized_copy(self) -> None:
        """Minmax maps [0..4] to [0..1] (divisor 4 + 1e-8); 4 equal bins there have
        edges [0, .25, .5, .75, 1], and the inverse maps them back to [0, 1, 2, 3, 4].
        """
        norm = LabelNormalizationTransform(
            _five_point_dataset(), {"fitness": {"strategy": "minmax"}}
        )
        binning = LabelBinningTransform(
            _five_point_dataset(),
            {"fitness": {"strategy": "equal_width", "num_bins": 4}},
            norm,
        )
        info = binning.get_bin_info("fitness")
        assert info["bin_edges"] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0], abs=1e-7)
        assert info["bin_edges_denormalized"] == pytest.approx(
            [0.0, 1.0, 2.0, 3.0, 4.0], abs=1e-6
        )

    def test_onehot_is_left_closed_clamped_and_nan_propagating(self) -> None:
        """Edges [0, 1, 2, 3, 4]: -5 clamps to 0 (bin 0); 1 opens bin 1; 1.999 stays in
        bin 1; the right edge 4 and the clamped 9 fold into bin 3; NaN gives a NaN row.
        The default label type is categorical, and a list input is converted.
        """
        binning = LabelBinningTransform(
            _five_point_dataset(),
            {"fitness": {"strategy": "equal_width", "num_bins": 4}},
        )
        out = binning(_gene([-5.0, 0.0, 1.0, 1.999, 4.0, 9.0, float("nan")]))
        onehot = out["gene"]["fitness"]
        assert onehot[:6].tolist() == [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        assert torch.isnan(onehot[6]).all()
        # the continuous copy is the converted input, unclamped
        assert out["gene"]["fitness_continuous"][:6].tolist() == pytest.approx(
            [-5.0, 0.0, 1.0, 1.999, 4.0, 9.0]
        )

    def test_soft_labels_nan_row_and_the_closed_form(self) -> None:
        """Edges [0, 1, 2], centers [0.5, 1.5], sigma = 1 * 1: 0.5 -> [1, e^-0.5]
        normalized = [0.6224593, 0.3775407]; NaN gives a NaN row.
        """
        soft = EqualWidthStrategy().compute_soft_labels(
            torch.tensor([0.5, float("nan")]), torch.tensor([0.0, 1.0, 2.0]), "x", 1
        )
        e = math.exp(-0.5)
        assert soft[0].tolist() == pytest.approx([1 / (1 + e), e / (1 + e)], abs=1e-7)
        assert torch.isnan(soft[1]).all()

    def test_soft_labels_that_underflow_stay_all_zero(self) -> None:
        """Finding: an underflowed Gaussian row is left at zero, not normalized.

        Edges [0, 0.001, 100]: min width 0.001, sigma 0.003, centers [0.0005, 50.0005].
        The value 100 sits 99.9995 and 49.9995 from them, 33333 and 16666 sigmas, so both
        ``exp`` terms are 0.0 and the ``sum > 0`` guard (regression_to_classification.py
        :217) skips the division, leaving a row that sums to 0 rather than 1. Pinned
        until the row falls back to a one-hot on the nearest center.
        """
        soft = EqualWidthStrategy().compute_soft_labels(
            torch.tensor([100.0, 0.0005]), torch.tensor([0.0, 1e-3, 100.0]), "x", 3
        )
        assert soft.tolist() == [[0.0, 0.0], [1.0, 0.0]]

    def test_categorical_inverse_draws_seeded_uniforms_bin_by_bin(self) -> None:
        """Edges [0..4]; argmax rows [2, 0, 2, NaN]. Under seed 42 the draws go to bin 0
        first (0.8822692632675171), then bin 2 in row order (0.9150039553642273,
        0.38286375999450684), each as low + r * width; the NaN row stays NaN.
        """
        binning = _binning("categorical")
        logits = torch.tensor(
            [
                [0.0, 0.0, 5.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 3.0, 0.0],
                [0.0, float("nan"), 0.0, 0.0],
            ]
        )
        values = binning.inverse(_gene(logits))["gene"]["fitness"]
        assert values[:3].tolist() == pytest.approx(
            [2.9150039553642273, 0.8822692632675171, 2.38286375999450684], abs=1e-6
        )
        assert torch.isnan(values[3])

    def test_ordinal_inverse_counts_crossings_above_one_half(self) -> None:
        """Crossings (entries > 0.5) per row: [1, 1, 0] -> 2, [0, 0, 0] -> 0 and
        [0.9, 0.6, 0.7] -> 3, even though the last is not monotone. Bins 0, 2, 3 draw in
        that order under seed 42: 0 + 0.88227, 2 + 0.91500, 3 + 0.38286.
        """
        binning = _binning("ordinal")
        labels = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.9, 0.6, 0.7]])
        values = binning.inverse(_gene(labels))["gene"]["fitness"]
        assert values.tolist() == pytest.approx(
            [2.9150039553642273, 0.8822692632675171, 3.38286375999450684], abs=1e-6
        )

    def test_an_all_nan_prediction_inverts_to_all_nan(self) -> None:
        """Every row NaN: the inverse returns one NaN per row, shape (2,)."""
        binning = _binning("ordinal")
        values = binning.inverse(_gene(torch.full((2, 3), float("nan"))))
        assert values["gene"]["fitness"].shape == (2,)
        assert torch.isnan(values["gene"]["fitness"]).all()

    def test_soft_inverse_is_a_windowed_expectation_only_away_from_the_edges(
        self,
    ) -> None:
        """Finding: the soft inverse is deterministic and edge-dependent.

        The ``inverse`` docstring promises "random sampling within bins", but the soft
        branch (regression_to_classification.py:469-500) returns a probability-weighted
        mean over a 5-bin window around the peak, and falls back to the bare peak center
        when that window would cross an edge. On edges 0..6 (centers 0.5..5.5):
        probabilities proportional to [1, 1, 3, 2, 1, 1] peak at bin 2, window bins 0-4,
        (0.5 + 1.5 + 3 * 2.5 + 2 * 3.5 + 4.5) / 8 = 21 / 8 = 2.625; a peak at bin 1 has
        a clipped window and returns its center 1.5; a uniform row peaks at bin 0, 0.5.
        Pinned until the docstring and the edge behavior agree.
        """
        six_points: Any = SimpleNamespace(
            label_df=pd.DataFrame({"fitness": [0.0, 6.0]})
        )
        six = LabelBinningTransform(
            six_points,
            {
                "fitness": {
                    "strategy": "equal_width",
                    "num_bins": 6,
                    "label_type": "soft",
                }
            },
        )
        logits = torch.log(
            torch.tensor(
                [
                    [1.0, 1.0, 3.0, 2.0, 1.0, 1.0],
                    [1.0, 3.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            )
        )
        values = six.inverse(_gene(logits))["gene"]["fitness"]
        assert values.tolist() == pytest.approx([2.625, 1.5, 0.5], abs=1e-6)

    def test_an_unknown_label_type_bins_nothing_and_inverts_to_nan(self) -> None:
        """Finding: an unrecognized ``label_type`` is a silent no-op forward, NaN back.

        ``forward`` (regression_to_classification.py:389-407) has no else branch, so the
        label stays continuous; ``inverse`` (446-523) likewise, so every row becomes NaN.
        With ``store_continuous=False`` no ``fitness_continuous`` copy is written.
        Pinned until an unknown label type raises.
        """
        binning = _binning("bogus", store_continuous=False)
        out = binning(_gene(torch.tensor([0.5, 2.5])))
        assert out["gene"]["fitness"].tolist() == [0.5, 2.5]
        assert "fitness_continuous" not in out["gene"]
        back = binning.inverse(_gene(torch.tensor([[0.0, 1.0, 0.0, 0.0]])))
        assert back["gene"]["fitness"].shape == (1,)
        assert torch.isnan(back["gene"]["fitness"]).all()

    def test_inverse_of_a_list_fails_before_its_own_conversion(self) -> None:
        """Finding: the list-to-tensor conversion in ``inverse`` is unreachable.

        ``inverse`` reads ``data["gene"][label].device`` (regression_to_classification.py
        :418) before the ``isinstance`` conversion at 427-428, so a list prediction raises
        ``AttributeError`` where ``forward`` would have converted it. Pinned until the
        device lookup follows the conversion.
        """
        binning = _binning("ordinal")
        with pytest.raises(
            AttributeError, match=r"^'list' object has no attribute 'device'$"
        ):
            binning.inverse(_gene([[1.0, 0.0, 0.0]]))


class TestInverseComposeBranches:
    """Construction errors and the repr of ``InverseCompose``."""

    def test_a_tuple_of_transforms_is_refused(self) -> None:
        """Only a Compose or a list is accepted."""
        norm = LabelNormalizationTransform(
            _five_point_dataset(), {"fitness": {"strategy": "minmax"}}
        )
        as_tuple: Any = (norm,)
        with pytest.raises(
            ValueError,
            match=r"^transforms must be either a Compose object or a list of transforms$",
        ):
            InverseCompose(as_tuple)

    def test_a_transform_without_inverse_is_named_in_the_refusal(self) -> None:
        """The refusal names the class that lacks ``inverse``."""

        class NoInverse:
            pass

        with pytest.raises(
            ValueError, match=r"^Transform NoInverse does not implement inverse method$"
        ):
            InverseCompose([NoInverse()])

    def test_repr_lists_each_transform_on_its_own_indented_line(self) -> None:
        """Each wrapped transform on its own two-space-indented line."""
        norm = LabelNormalizationTransform(
            _five_point_dataset(), {"fitness": {"strategy": "minmax"}}
        )
        binning = _binning("categorical")
        assert repr(InverseCompose(Compose([norm, binning]))) == (
            "InverseCompose(\n  LabelNormalizationTransform()\n"
            "  LabelBinningTransform()\n)"
        )
