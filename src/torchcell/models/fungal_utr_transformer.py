# src/torchcell/models/fungal_utr.py
import os
import zipfile
from re import T

import requests
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from torchcell.models.llm import NucleotideModel, pretrained_LLM


class FungalUtrTransformer(NucleotideModel):
    """Transformer for fungal UTR (Untranslated Region) sequences.

    This class facilitates the embedding of UTR sequences using pretrained models
    specified by the model name. The pretrained models should be available locally or
    will be downloaded from Figshare if not present.

    Attributes:
        VALID_MODEL_NAMES (List[str]): List of valid names of the available models.
                                       The valid model names are:
                                       - "downstream_300"
                                       - "species_downstream_300"
                                       - "species_upstream_1000"
                                       - "upstream_1000"
        model_name (str): Name of the chosen model.
        tokenizer (AutoTokenizer): Tokenizer instance for the chosen model.
        model (AutoModelForMaskedLM): Model instance for the chosen model.
    """

    VALID_MODEL_NAMES = [
        "downstream_300",
        "species_downstream_300",
        "species_upstream_1000",
        "upstream_1000",
    ]

    def __init__(self, model_name: str = None):
        """Initializes the transformer with the specified model name.

        Args:
            model_name (str, optional): Name of the model to be used.

        Raises:
            ValueError: If the model_name is not provided or is invalid.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        # Check if model_name is provided
        if self.model_name is None:
            raise ValueError(
                f"Model name is required. Please select from {self.VALID_MODEL_NAMES}"
            )
        self.load_model()

    @staticmethod
    def _check_and_download_model():
        """Checks if the pretrained models are available locally.

        If not, downloads and extracts them from Figshare.
        """
        # The URL provided by the Figshare download button
        url = "https://figshare.com/ndownloader/files/41663388"

        # The local path where you want to save the downloaded zip file
        local_file = "LLM.zip"

        # The directory where you want to extract the contents of the zip file
        script_dir = os.path.dirname(os.path.realpath(__file__))
        target_directory = os.path.join(script_dir, "pretrained_LLM", "fungal_utr")

        # Checking if models are already downloaded
        model_dirs = [
            "upstream_1000",
            "species_upstream_1000",
            "downstream_300",
            "species_downstream_300",
        ]
        if all(
            os.path.exists(os.path.join(target_directory, dir_, "config.json"))
            for dir_ in model_dirs
        ):
            print("Models already downloaded.")
            return

        # Downloading the file
        print("Downloading file...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download finished.")

        # Extracting the contents of the zip file
        print("Extracting file...")
        with zipfile.ZipFile(local_file, "r") as zip_ref:
            zip_ref.extractall(target_directory)
        print("Extraction finished.")

        # Deleting the downloaded zip file
        os.remove(local_file)
        print("Downloaded zip file removed.")

    @property
    def max_sequence_size(self) -> int:
        """Gets the maximum size of the sequence based on the model name.

        Returns:
            int: Maximum size of the sequence.
        """
        if self.model_name.split("_")[-1] == "1000":
            return 1000
        elif self.model_name.split("_")[-1] == "300":
            return 300

    def load_model(self) -> None:
        """Loads the specified model and its tokenizer.

        If the models are not available locally, they are downloaded from Figshare.

        Raises:
            ValueError: If the provided model_name is invalid.
        """
        # CHECK if the provided model_name is valid
        if self.model_name not in self.VALID_MODEL_NAMES:
            raise ValueError(
                f"Invalid model name. Please select from {self.VALID_MODEL_NAMES}"
            )

        # Check and download the model if necessary
        self._check_and_download_model()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_directory = os.path.join(
            script_dir, "pretrained_LLM", "fungal_utr", self.model_name
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        self.model = AutoModelForMaskedLM.from_pretrained(model_directory)
        self._max_size = self.tokenizer.model_max_length

    def embed(self, sequences: list, mean_embedding: bool = False) -> torch.Tensor:
        """Embeds a list of sequences using the chosen model.

        Args:
            sequences (list[str]): List of UTR sequences to be embedded.
            mean_embedding (bool, optional): If True, returns the mean embedding of the sequences. Default is False.

        Returns:
            torch.Tensor: Tensor containing the embeddings of the input sequences.
        """
        tokens_ids = self.tokenizer.batch_encode_plus(sequences, return_tensors="pt")[
            "input_ids"
        ]

        attention_mask = tokens_ids != self.tokenizer.pad_token_id
        torch_outs = self.model(
            tokens_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )

        embeddings = torch_outs["hidden_states"][-1].detach()

        if mean_embedding:
            embeddings = torch.sum(
                attention_mask.unsqueeze(-1) * embeddings, axis=-2
            ) / torch.sum(attention_mask, axis=-1).unsqueeze(-1)

        return embeddings


def main():
    # Initialize the FungalUtr class with a specific model name
    fungal_utr_model = FungalUtrTransformer("upstream_1000")

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
