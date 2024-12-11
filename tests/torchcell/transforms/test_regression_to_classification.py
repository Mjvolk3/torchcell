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
