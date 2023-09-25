# src/torchcell/models/fungal_up_down_transformer.py
# [[src.torchcell.models.fungal_up_down_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/models/fungal_up_down_transformer.py
# Test file: src/torchcell/models/test_fungal_up_down_transformer.py


import os
import os.path as osp
import zipfile
from re import T

import requests
import torch
from regex import P
from transformers import AutoModelForMaskedLM, AutoTokenizer

from torchcell.models.llm import NucleotideModel


class FungalUpDownTransformer(NucleotideModel):
    VALID_MODEL_NAMES = [
        "downstream_species_lm",
        # "downstream_agnostic_lm", Not supported
        "upstream_species_lm",
        # "upstream_agnostic_lm",
    ]

    def __init__(
        self, model_name: str = None, target_layer: int | tuple[int, int] = (8,)
    ):
        self.target_layer = target_layer
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
            script_dir, "pretrained_LLM", "fungal_up_down_transformer"
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

    def load_model(self) -> None:
        self._check_and_download_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hugging_model_dir, revision=self.model_name
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.hugging_model_dir, revision=self.model_name
        )
        self.model.eval()

    @property
    def max_sequence_size(self) -> int:
        if self.model_name.split("_")[0] == "upstream":
            return 1003
        elif self.model_name.split("_")[0] == "downstream":
            return 300

    def _pad_sequence(self, sequence: str, desired_length: int) -> str:
        sequence_length = len(sequence)
        if sequence_length < desired_length:
            pad_length = desired_length - sequence_length
            sequence = sequence + "N" * pad_length
            # padding with "N" give token UNK
            # [k for k, v in self.tokenizer.vocab.items() if v == 1]
            # ['[UNK]']
        return sequence

    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        embeddings = []
        proxy_species = "candida_glabrata"

        def kmers_stride1(seq, k=6):
            return [seq[i : i + k] for i in range(0, len(seq) - k + 1)]

        for sequence in sequences:
            sequence_length = len(sequence)
            desired_length = self.max_sequence_size

            if (
                self.model_name.startswith("upstream")
                and sequence_length > desired_length
            ):
                raise ValueError(
                    f"Seq len for {self.model_name} must be <= {desired_length}."
                    f"Provided: {sequence_length}"
                )

            if self.model_name.startswith("downstream"):
                if sequence_length < 11:
                    raise ValueError(
                        f"Seq len for {self.model_name} must be >  11."
                        "Provided: {sequence_length}"
                    )
                elif sequence_length > desired_length:
                    raise ValueError(
                        f"Seq len for {self.model_name} must be <= {desired_length}."
                        "Provided: {sequence_length}"
                    )

            if sequence_length < desired_length and self.model_name.startswith(
                "upstream"
            ):
                sequence = self._pad_sequence(sequence, desired_length)

            # Tokenizing the sequence with proxy_species and kmers_stride1
            tokenized_data = self.tokenizer(
                proxy_species + " " + " ".join(kmers_stride1(sequence)),
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.model(**tokenized_data, output_hidden_states=True)
                hidden_states = outputs.hidden_states

            # Handling different cases for target_layer as in the original work
            if isinstance(self.target_layer, int):
                embedding = hidden_states[self.target_layer][0]
            elif len(self.target_layer) == 1:
                embedding = torch.mean(
                    torch.stack(hidden_states[self.target_layer[0] :]), dim=0
                )[0]
            else:
                if self.target_layer[1] > len(hidden_states):
                    raise ValueError(
                        f"Target layer {self.target_layer[1]} is out of range."
                        f"Max layer is {len(hidden_states)}."
                    )
                embedding = torch.mean(
                    torch.stack(
                        hidden_states[self.target_layer[0] : self.target_layer[1] + 1]
                    ),
                    dim=0,
                )[0]

            if mean_embedding:
                embedding = embedding.mean(dim=0).cpu()

            embeddings.append(embedding)

        embeddings_tensor = (
            torch.stack(embeddings)
            if not mean_embedding
            else torch.cat(embeddings).view(len(sequences), -1)
        )
        return embeddings_tensor


def main():
    fungal_up_down_model = FungalUpDownTransformer("upstream_species_lm", (8,))
    sequences = [
        "ATTCTG" * 50,
        "ATTTTG" * 50,
        "ATGCTG" * 50,
    ]  # Example list of sequences
    embedding = fungal_up_down_model.embed(sequences, mean_embedding=False)
    print(f"Embeddings shape: {embedding.shape}")
    print(fungal_up_down_model.max_sequence_size)


if __name__ == "__main__":
    main()
