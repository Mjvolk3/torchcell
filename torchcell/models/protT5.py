# torchcell/models/protT5.py
# [[torchcell.models.protT5]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/protT5.py
# Test file: torchcell/models/test_protT5.py

import os
import os.path as osp
import re

import torch
from transformers import T5EncoderModel, T5Tokenizer

from torchcell.models.llm import PeptideModel


class ProtT5(PeptideModel):
    VALID_MODEL_NAMES = ["prot_t5_xl_uniref50"]

    def __init__(self, model_name: str):
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = osp.join("Rostlab", model_name)
        super().__init__(self.model_name)

    @property
    def max_sequence_size(self):
        return 40000

    @staticmethod
    def _check_and_download_model(model_name: str):
        # Define the directory where you want the model to be saved
        script_dir = osp.dirname(osp.realpath(__file__))
        target_directory = osp.join(script_dir, "pretrained_LLM", "ProtT5")

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
            T5Tokenizer.from_pretrained(
                model_name, cache_dir=target_directory, legacy=True
            )
            # model
            T5EncoderModel.from_pretrained(model_name, cache_dir=target_directory)
            print("Download finished.")

    def _prepare_sequence(self, sequence: str) -> str:
        # Replace all rare/ambiguous amino acids by X
        # Introduce white-space between all amino acids
        return " ".join(list(re.sub(r"[UZOB]", "X", sequence)))

    def load_model(self, model_name: str):
        # Check and download the model if necessary
        self._check_and_download_model(model_name)

        # Load the tokenizer and the model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)

        # Set to half-precision if on GPU, full-precision if on CPU
        self.model.full() if self.device == "cpu" else self.model.half()

    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        # Pre-process sequences
        sequences = [self._prepare_sequence(seq) for seq in sequences]

        # Tokenize sequences and pad up to the longest sequence in the batch
        ids = self.tokenizer(
            sequences, add_special_tokens=True, padding="longest", return_tensors="pt"
        )
        input_ids = ids["input_ids"].to(self.device)
        attention_mask = ids["attention_mask"].to(self.device)

        # Generate embeddings
        with torch.no_grad():
            embedding_repr = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

        embeddings = embedding_repr.last_hidden_state

        if mean_embedding:
            # Compute mean embeddings per sequence
            embeddings = embeddings.mean(dim=1)

        return embeddings


if __name__ == "__main__":
    model = ProtT5(model_name="prot_t5_xl_uniref50")
    sample_sequences = ["P" * 4911]
    embeddings = model.embed(sample_sequences, mean_embedding=True)
    print(embeddings.shape)
