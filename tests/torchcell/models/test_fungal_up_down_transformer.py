# tests/torchcell/models/test_fungal_up_down_transformer.py
# [[tests.torchcell.models.test_fungal_up_down_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_fungal_up_down_transformer.py

import os.path as osp
from unittest.mock import MagicMock, patch

import pytest
import torch

from torchcell.models.fungal_up_down_transformer import FungalUpDownTransformer


class TestFungalUpDownTransformerUpstream:
    @pytest.fixture
    def model(self):
        return FungalUpDownTransformer(
            model_name="upstream_species_lm", target_layer=(8,)
        )

    def test_init(self, model):
        assert model.target_layer == (8,)
        assert model.model_name == "upstream_species_lm"
        assert model.hugging_model_dir == "gagneurlab/SpeciesLM"

    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_check_and_download_model_exists(self, makedirs_mock, exists_mock, model):
        exists_mock.return_value = True
        model._check_and_download_model()
        makedirs_mock.assert_not_called()

    def test_max_sequence_size_upstream(self, model):
        model.model_name = "upstream_species_lm"
        assert model.max_sequence_size == 1003

    def test_max_sequence_size_downstream(self, model):
        model.model_name = "downstream_species_lm"
        assert model.max_sequence_size == 300

    def test_pad_sequence(self, model):
        # Mock a tokenized_data
        tokenized_data = {
            "input_ids": torch.Tensor([[1]]),
            "token_type_ids": torch.Tensor([[1]]),
            "attention_mask": torch.Tensor([[1]]),
        }
        result, pad_length = model._pad_sequence(tokenized_data, mean_embedding=True)

        assert pad_length == 1000

    def test_embed_raises_value_error_for_upstream(self, model):
        sequences = ["A" * 1004]

        with pytest.raises(ValueError) as excinfo:
            model.embed(sequences)
        assert (
            str(excinfo.value)
            == "Seq len for upstream_species_lm must be <= 1003. Provided: 1004"
        )

    def test_embed_raises_value_error_for_downstream_short(self, model):
        model.model_name = "downstream_species_lm"
        sequences = ["A" * 10]

        with pytest.raises(ValueError) as excinfo:
            model.embed(sequences)
        assert (
            str(excinfo.value)
            == "Seq len for downstream_species_lm must be >  11. Provided: 10"
        )

    def test_embed_raises_value_error_for_downstream_long(self, model):
        model.model_name = "downstream_species_lm"
        sequences = ["A" * 301]

        with pytest.raises(ValueError) as excinfo:
            model.embed(sequences)
        assert (
            str(excinfo.value)
            == "Seq len for downstream_species_lm must be <= 300. Provided: 301"
        )

    def test_embed(self, model):
        # Test with the correct sequence length
        sequences = ["ATTTG" * 200 + "ATG"][:1003]  # Adjusting to be exactly 1003 bp
        embedding = model.embed(sequences, mean_embedding=False)

        # Validate the shape of the embedding and other necessary checks
        assert embedding is not None, "Embedding should not be None"
        assert isinstance(embedding, torch.Tensor), "Embedding should be a torch.Tensor"

        # Test with sequence length that requires padding, and mean_embedding is True
        sequences = [
            "ATTTG" * 100 + "ATG"
        ]  # This sequence will be shorter than 1003 bp
        embedding = model.embed(
            sequences, mean_embedding=True
        )  # Adjust to mean_embedding=True

        # Validate the shape of the embedding and other necessary checks
        assert embedding is not None, "Embedding should not be None"
        assert isinstance(embedding, torch.Tensor), "Embedding should be a torch.Tensor"

    def test_embed_mean_embedding(self, model):
        # Test the behavior of the embed method when mean_embedding is True
        sequences = ["ATTTG" * 200 + "ATG"]  # Example list of sequences
        embedding = model.embed(sequences, mean_embedding=True)

        # Validate the shape of the embedding and other necessary checks
        assert embedding is not None, "Embedding should not be None"
        assert isinstance(embedding, torch.Tensor), "Embedding should be a torch.Tensor"
        # Optionally, you can also check the values in the embedding Tensor

    def test_target_layer_as_int(self, model):
        model.target_layer = 1
        sequences = ["ATTTG" * 200 + "ATG"]  # Adjust as per your needs
        embedding = model.embed(sequences, mean_embedding=True)
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (1, 768)

    def test_target_layer_as_single_element_tuple(self, model):
        model.target_layer = (8, 10)
        sequences = ["ATTTG" * 200 + "ATG"]  # Adjust as per your needs
        embedding = model.embed(sequences, mean_embedding=True)
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (1, 768)

    def test_target_layer_as_single_element_tuple_error(self, model):
        model.target_layer = (8, 14)
        sequences = ["ATTTG" * 200 + "ATG"]  # Adjust as per your needs
        with pytest.raises(ValueError) as excinfo:
            model.embed(sequences, mean_embedding=True)
        assert str(excinfo.value) == "Target layer 14 is out of range. Max layer is 13."
