import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import Compose

from torchcell.transforms.regression_to_classification import (
    InverseCompose,
    LabelBinningTransform,
    LabelNormalizationTransform,
)


class TestLabelNormalizationTransform:
    @pytest.fixture
    def mock_dataset(self):
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
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return LabelNormalizationTransform(mock_dataset, label_configs)

    def test_minmax_normalization(self, norm_transform):
        data = HeteroData()
        data["gene"]["fitness"] = torch.tensor([0.0, 0.5, 1.0, 2.0])

        normalized = norm_transform(data)
        expected = torch.tensor([0.0, 0.25, 0.5, 1.0])
        assert torch.allclose(normalized["gene"]["fitness"], expected)

        denormalized = norm_transform.inverse(normalized)
        assert torch.allclose(denormalized["gene"]["fitness"], data["gene"]["fitness"])

    def test_standard_normalization(self, norm_transform):
        data = HeteroData()
        data["gene"]["gene_interaction"] = torch.tensor([-1.0, 0.0, 1.0])

        normalized = norm_transform(data)
        stats = norm_transform.stats["gene_interaction"]
        mean = stats["mean"]
        std = stats["std"]
        expected = (data["gene"]["gene_interaction"] - mean) / std
        assert torch.allclose(normalized["gene"]["gene_interaction"], expected)


class TestLabelNormalizationInverse:
    @pytest.fixture
    def mock_dataset(self):
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
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return LabelNormalizationTransform(mock_dataset, label_configs)

    # TestLabelNormalizationInverse::test_inverse_minmax
    def test_inverse_minmax(self, norm_transform):
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
    @pytest.fixture
    def mock_dataset(self):
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
        label_configs = {
            "fitness": {"strategy": "minmax"},
            "gene_interaction": {"strategy": "standard"},
        }
        return LabelNormalizationTransform(mock_dataset, label_configs)

    # TestLabelNormalizationRoundTrip::test_round_trip_minmax
    def test_round_trip_minmax(self, norm_transform, mock_dataset):
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
    @pytest.fixture
    def mock_dataset(self):
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
        data = HeteroData()
        data["gene"]["fitness"] = torch.tensor([0.0, 0.3, 0.7, 1.0])

        binned = bin_transform(data)
        assert binned["gene"]["fitness"].shape == (4, 10)
        assert torch.allclose(binned["gene"]["fitness"].sum(dim=1), torch.ones(4))

    def test_categorical_binning_forward(self, bin_transform):
        data = HeteroData()
        data["gene"]["gene_interaction"] = torch.tensor([-1.0, -0.3, 0.3, 1.0])

        binned = bin_transform(data)
        assert binned["gene"]["gene_interaction"].shape == (4, 10)
        assert torch.all(binned["gene"]["gene_interaction"].sum(dim=1) == 1)

    def test_inverse_soft_binning(self, bin_transform):
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
    @pytest.fixture
    def mock_dataset(self):
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
    @pytest.fixture
    def mock_dataset(self):
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
        label_configs = {
            "fitness": {
                "strategy": "equal_width",
                "num_bins": 4,  # Using 4 bins for simpler testing
                "label_type": "ordinal",
            }
        }
        return LabelBinningTransform(mock_dataset, label_configs)

    def test_ordinal_forward(self, bin_transform):
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
            assert (
                orig_bin == rec_bin
            ), f"Original value {orig} and recovered value {rec} are in different bins"

    def test_ordinal_with_nans(self, bin_transform):
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
    @pytest.fixture
    def mock_dataset(self):
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
    @pytest.fixture
    def mock_dataset(self):
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

            assert (
                bin_start <= recovered_values[i] <= bin_end
            ), f"Value {recovered_values[i]} not in bin {i} range [{bin_start}, {bin_end}]"

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

            assert (
                bin_start <= recovered_values[i] <= bin_end
            ), f"Value {recovered_values[i]} not in bin {i} range [{bin_start}, {bin_end}]"

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

            assert (
                bin_start <= recovered_values[i] <= bin_end
            ), f"Value {recovered_values[i]} not in bin {i} range [{bin_start}, {bin_end}]"


class TestInverseComposeWithGrads:
    @pytest.fixture
    def mock_dataset(self):
        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame({"fitness": np.linspace(0, 1, 100)})

        return MockDataset()

    @pytest.fixture
    def transforms(self, mock_dataset):
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
