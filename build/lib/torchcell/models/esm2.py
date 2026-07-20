# torchcell/models/esm2.py
# [[torchcell.models.esm2]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/esm2.py
# Test file: torchcell/models/test_esm2.py

"""ESM-2 protein language model wrapper producing per-residue or mean embeddings."""

import os
import os.path as osp

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from torchcell.models.llm import PeptideModel


class Esm2(PeptideModel):
    """Facebook ESM-2 masked-LM wrapper that embeds protein sequences."""

    VALID_MODEL_NAMES = [
        "esm2_t48_15B_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t6_8M_UR50D",
    ]

    def __init__(self, model_name: str):
        """Resolve the HuggingFace model id, pick a device, and load the model.

        Args:
            model_name: ESM-2 checkpoint name, prefixed with ``facebook/``.
        """
        self.tokenizer = None
        self.model_name = osp.join("facebook", model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(self.model_name)

    @staticmethod
    def _check_and_download_model(model_name: str) -> None:
        script_dir = osp.dirname(osp.realpath(__file__))
        target_directory = osp.join(script_dir, "pretrained_LLM", "Esm2")
        if not osp.exists(target_directory):
            os.makedirs(target_directory)
        model_directory = osp.join(target_directory, model_name)
        if osp.exists(model_directory):
            print(f"{model_name} model already downloaded.")
        else:
            print(f"Downloading {model_name} model to {model_directory}...")
            AutoTokenizer.from_pretrained(model_name, cache_dir=target_directory)  # type: ignore[no-untyped-call]  # transformers from_pretrained is untyped
            AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=target_directory)
            print("Download finished.")

    @property
    def max_sequence_size(self) -> int:
        """Return the maximum supported residue sequence length."""
        return 1022

    def load_model(self, model_name: str) -> None:
        """Download if needed, then load the tokenizer and model onto the device."""
        self._check_and_download_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # type: ignore[no-untyped-call]  # transformers from_pretrained is untyped
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)

    def embed(
        self, sequences: str | list[str], mean_embedding: bool = False
    ) -> torch.Tensor:
        """Return last-hidden-state embeddings for one or more sequences.

        Args:
            sequences: A single sequence or list of sequences to embed.
            mean_embedding: If True, return attention-masked mean-pooled embeddings
                instead of per-residue hidden states.

        Returns:
            Tensor of embeddings on the model device.
        """
        if isinstance(sequences, str):
            sequences = [sequences]  # Convert single string to a list

        encoding = self.tokenizer.batch_encode_plus(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_sequence_size,
        )
        tokens_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        tokens_ids = tokens_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        torch_outs = self.model(
            tokens_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        embeddings: torch.Tensor = torch_outs["hidden_states"][-1].detach()

        if mean_embedding:
            embeddings = (attention_mask.unsqueeze(-1) * embeddings).sum(
                dim=1
            ) / attention_mask.sum(dim=1).unsqueeze(-1)
            embeddings = embeddings.unsqueeze(0)

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
