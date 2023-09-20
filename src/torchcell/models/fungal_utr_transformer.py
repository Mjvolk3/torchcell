# src/torchcell/models/fungal_utr_transformer.py
# [[src.torchcell.models.fungal_utr_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/models/fungal_utr_transformer.py
# Test file: src/torchcell/models/test_fungal_utr_transformer.py


import os
import os.path as osp
import zipfile
from re import T

import requests
import torch
from regex import P
from transformers import AutoModelForMaskedLM, AutoTokenizer

from torchcell.models.llm import NucleotideModel


class FungalUtrTransformer(NucleotideModel):
    VALID_MODEL_NAMES = [
        "downstream_species_lm",
        "downstream_agnostic_lm",
        "upstream_species_lm",
        "upstream_agnostic_lm",
    ]

    def __init__(self, model_name: str = None):
        self.tokenizer = None
        self.model = None
        self.model_name = model_name
        self.hugging_model_dir = "gagneurlab/SpeciesLM"
        self.load_model()

    def _check_and_download_model(self):
        # Define the model name
        model_path = osp.join(self.hugging_model_dir, self.model_name)

        # Define the directory where you want the model to be saved
        script_dir = os.path.dirname(os.path.realpath(__file__))
        target_directory = os.path.join(
            script_dir, "pretrained_LLM", "fungal_utr_transformer"
        )

        # Create the target directory if it doesn't exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        model_directory = os.path.join(target_directory, model_path)

        # Check if the model has already been downloaded
        if os.path.exists(model_directory):
            print(f"{model_directory} model already downloaded.")
        else:
            print(f"Downloading {self.model_name} model to {model_directory}...")
            # tokenizer
            AutoTokenizer.from_pretrained(
                self.hugging_model_dir,
                revision=self.model_name,
                cache_dir=target_directory,
            )
            # model
            AutoModelForMaskedLM.from_pretrained(
                self.hugging_model_dir,
                revision=self.model_name,
                cache_dir=target_directory,
            )
            print("Download finished.")

    @property
    def max_sequence_size(self) -> int:
        if self.model_name.split("_")[0] == "upstream":
            return 1003
        elif self.model_name.split("_")[0] == "upstream":
            return 300

    def load_model(self) -> None:
        # Check and download the model if necessary
        self._check_and_download_model()

        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hugging_model_dir, revision=self.model_name
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.hugging_model_dir, revision=self.model_name
        )

    def embed(self, sequences: list, mean_embedding: bool = False) -> torch.Tensor:
        tokens_ids = self.tokenizer.batch_encode_plus(sequences, return_tensors="pt")[
            "input_ids"
        ]

        # Compute the embeddings
        attention_mask = tokens_ids != self.tokenizer.pad_token_id
        torch_outs = self.model(
            tokens_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )

        embeddings = torch_outs["hidden_states"][-1].detach()

        if mean_embedding:
            # Compute mean embeddings per sequence
            embeddings = torch.sum(
                attention_mask.unsqueeze(-1) * embeddings, axis=-2
            ) / torch.sum(attention_mask, axis=-1).unsqueeze(-1)

        return embeddings


def main():
    # Initialize the FungalUtr class with a specific model name
    fungal_utr_model = FungalUtrTransformer("species_upstream_1000")

    # Create a dummy dna sequence
    sequences = ["ATTCTG" * 9]

    # Get embeddings
    embeddings = fungal_utr_model.embed(sequences)

    # print(f"Mean sequence embeddings: {embeddings}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Get Mean embeddings
    embeddings = fungal_utr_model.embed(sequences, mean_embedding=True)

    # print(f"Mean sequence embeddings: {embeddings}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(fungal_utr_model.max_sequence_size)


if __name__ == "__main__":
    main()
