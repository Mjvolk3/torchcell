import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.transforms import Compose

from torchcell.transforms.coo_regression_to_classification import (
    COOInverseCompose,
    COOLabelBinningTransform,
    COOLabelNormalizationTransform,
)


class TestCOOLabelNormalizationTransform:
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
        return COOLabelNormalizationTransform(mock_dataset, label_configs)

    def test_minmax_normalization_coo(self, norm_transform):
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
        return COOLabelNormalizationTransform(mock_dataset, label_configs)

    def test_inverse_minmax_coo(self, norm_transform):
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
        return COOLabelNormalizationTransform(mock_dataset, label_configs)

    def test_round_trip_minmax_coo(self, norm_transform, mock_dataset):
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
            "gene_interaction": {
                "strategy": "equal_width",
                "num_bins": 4,
                "label_type": "categorical",
            }
        }
        return COOLabelBinningTransform(mock_dataset, label_configs)

    def test_categorical_binning_forward_coo(self, bin_transform):
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
    @pytest.fixture
    def mock_dataset(self):
        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {"gene_interaction": np.linspace(-1.0, 1.0, 100)}
                )

        return MockDataset()

    @pytest.fixture
    def transforms(self, mock_dataset):
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
                "num_bins": 4,
                "label_type": "ordinal",
            }
        }
        return COOLabelBinningTransform(mock_dataset, label_configs)

    def test_ordinal_forward_coo(self, bin_transform):
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
    @pytest.fixture
    def mock_dataset(self):
        class MockDataset:
            def __init__(self):
                self.label_df = pd.DataFrame(
                    {"gene_interaction": np.random.uniform(-1, 1, 1000)}
                )

        return MockDataset()

    def test_batch_normalization_coo(self, mock_dataset):
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
        from torch_geometric.data import Batch

        batch = Batch.from_data_list([batch1, batch2])

        # Apply transform
        normalized = norm_transform(batch)

        # Check that all values are normalized
        assert normalized["gene"]["phenotype_values"].shape == (4,)
        assert hasattr(normalized["gene"], "phenotype_values_original")


class TestModelOutputSimulationCOO:
    @pytest.fixture
    def mock_dataset(self):
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
