# torchcell/models/esm2.py
# [[torchcell.models.esm2]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/esm2.py
# Test file: torchcell/models/test_esm2.py

import os
import os.path as osp

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from torchcell.models.llm import PeptideModel


class Esm2(PeptideModel):
    VALID_MODEL_NAMES = [
        "esm2_t48_15B_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t6_8M_UR50D",
    ]

    def __init__(self, model_name: str):
        self.tokenizer = None
        self.model_name = osp.join("facebook", model_name)
        super().__init__(self.model_name)

    @staticmethod
    def _check_and_download_model(model_name: str):
        # Define the model name

        # Define the directory where you want the model to be saved
        script_dir = osp.dirname(osp.realpath(__file__))
        target_directory = osp.join(script_dir, "pretrained_LLM", "Esm2")

        # Create the target directory if it doesn't exist
        if not osp.exists(target_directory):
            os.makedirs(target_directory)

        model_directory = osp.join(target_directory, model_name)

        # Check if the model has already been downloaded
        if osp.exists(model_directory):
            print(f"{model_name} model already downloaded.")
        else:
            print(f"Downloading {model_name} model to {model_directory}...")
            # tokenizer
            AutoTokenizer.from_pretrained(model_name, cache_dir=target_directory)
            # model
            AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=target_directory)
            print("Download finished.")

    @property
    def max_sequence_size(self):
        # TODO change.
        return 1026

    def load_model(self, model_name: str):
        # Check and download the model if necessary
        self._check_and_download_model(model_name)

        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)

    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        # Encode sequences and obtain attention masks directly
        encoding = self.tokenizer.batch_encode_plus(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_sequence_size,
        )
        tokens_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Check token lengths
        for token_id in tokens_ids:
            if len(token_id) > self.tokenizer.model_max_length:
                raise ValueError(
                    f"Token length {len(token_id)} exceeds the model's maximum length of {self.tokenizer.model_max_length}."
                )

        # Compute the embeddings
        torch_outs = self.model(
            tokens_ids, attention_mask=attention_mask, output_hidden_states=True
        )

        embeddings = torch_outs["hidden_states"][-1].detach()

        if mean_embedding:
            # Compute mean embeddings per sequence
            embeddings = (attention_mask.unsqueeze(-1) * embeddings).sum(
                dim=1
            ) / attention_mask.sum(dim=1).unsqueeze(-1)

        return embeddings


if __name__ == "__main__":
    model_names = [
        # "esm2_t48_15B_UR50D",
        # "esm2_t36_3B_UR50D",
        # "esm2_t33_650M_UR50D",
        # "esm2_t30_150M_UR50D",
        # "esm2_t12_35M_UR50D",
        "esm2_t6_8M_UR50D"
    ]
    for model_name in model_names:
        model = Esm2(model_name=model_name)
        sample_sequences = ["A" * 1024]
        embeddings = model.embed(sample_sequences, mean_embedding=True)
        print(embeddings.shape)
