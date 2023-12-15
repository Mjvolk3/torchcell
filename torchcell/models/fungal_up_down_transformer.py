# torchcell/models/fungal_up_down_transformer.py
# [[torchcell.models.fungal_up_down_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/fungal_up_down_transformer.py
# Test file: torchcell/models/test_fungal_up_down_transformer.py
import os
import os.path as osp

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from torchcell.models.llm import NucleotideModel


class FungalUpDownTransformer(NucleotideModel):
    r"""
    The FungalUpDownTransformer embeds DNA sequences
    upstream the CDS and downstream the CDS.
    It was trained on 1003 bp upstream sequences (including the start codon) and
    300 bp downstream sequences (including the stop codon).
    It handles both upstream and downstream sequences.
    valid models include :obj:`["downstream_species_lm", "upstream_species_lm"]`.
    The original paper discusses two other models
    (:obj:`["downstream_agnostic_lm"`, :obj:`"upstream_agnostic_lm"]`),
    but they are not supported here.

    Attributes:
        VALID_MODEL_NAMES (List[str]): Valid names for the model
    """

    VALID_MODEL_NAMES = ["downstream_species_lm", "upstream_species_lm"]

    def __init__(
        self, model_name: str = None, target_layer: int | tuple[int, int] = (8,)
    ):
        """
        Initializes a new instance of the FungalUpDownTransformer.

        Args:
            model_name (str, optional): The name of the model to be loaded.
            target_layer (int | tuple[int, int], optional):
            The layer(s) of the model to be used for embeddings.
            If tuple, embeddings are averaged over the range of layers.
            Defaults to (8,).
        """
        self.target_layer = target_layer
        self.tokenizer = None
        self.model = None
        self.model_name = model_name
        self.hugging_model_dir = "gagneurlab/SpeciesLM"
        self.load_model()

    def _check_and_download_model(self) -> None:
        r"""
        Verifies whether the model is downloaded, if not, downloads the model.
        """
        # Define the model name
        model_path = osp.join(self.hugging_model_dir, self.model_name)

        # Define the directory where you want the model to be saved
        script_dir = osp.dirname(osp.realpath(__file__))
        target_directory = osp.join(
            script_dir, "pretrained_LLM", "fungal_up_down_transformer"
        )

        # Create the target directory if it doesn't exist
        if not osp.exists(target_directory):
            os.makedirs(target_directory)  # pragma: no cover

        model_directory = osp.join(target_directory, model_path)

        # Check if the model has already been downloaded
        if osp.exists(model_directory):
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
        r"""
        Loads the required model and tokenizer for embedding sequences.
        """
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
        r"""
        Returns the maximum permissible sequence size for the loaded model.

        Returns:
            int: The maximum permissible sequence size.
        """
        if self.model_name.split("_")[0] == "upstream":
            return 1003
        elif self.model_name.split("_")[0] == "downstream":
            return 300

    @staticmethod
    def _pad_sequence(
        tokenized_data: dict[str, torch.tensor], mean_embedding: bool
    ) -> tuple[dict[str, torch.Tensor], int]:
        r"""
        Pads the tokenized sequence to meet the desired length.

        Args:
            tokenized_data (dict[str, torch.tensor]): The tokenized sequence data.
            mean_embedding (bool): Specifies whether to compute the mean embedding.

        Returns:
            tuple[dict[str, torch.Tensor], int]: The padded tokenized data and
             the pad length.

        Raises:
            AssertionError: If not using mean embedding and sequence length is
             below 1001 bp.
        """
        assert (
            mean_embedding
        ), "sequences must be 1003 bp if not using meaning embedding"
        # Find out how much we need to pad
        pad_length = 1001 - tokenized_data["input_ids"].shape[-1]
        pad_start = 2
        pad_end = pad_length + pad_start
        # Padding input_ids, token_type_ids, and attention_mask

        tokenized_data["input_ids"] = torch.cat(
            [
                tokenized_data["input_ids"][0][:2],
                torch.zeros((pad_length), dtype=torch.long),
                tokenized_data["input_ids"][0][2:],
            ]
        ).unsqueeze(0)
        tokenized_data["token_type_ids"] = torch.cat(
            [
                tokenized_data["token_type_ids"][0][:2],
                torch.zeros((pad_length), dtype=torch.long),
                tokenized_data["token_type_ids"][0][2:],
            ]
        ).unsqueeze(0)
        tokenized_data["attention_mask"] = torch.cat(
            [
                tokenized_data["attention_mask"][0][:2],
                torch.zeros((pad_length), dtype=torch.long),
                tokenized_data["attention_mask"][0][2:],
            ]
        ).unsqueeze(0)
        return tokenized_data, pad_start, pad_end

    def embed(
        self,
        sequences: list[str],
        mean_embedding: bool = True,
        proxy_species: str = "candida_glabrata",
    ) -> torch.Tensor:
        r"""
        Embeds the provided sequences using the loaded transformer model
        """
        embeddings = []

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
                    f" Provided: {sequence_length}"
                )

            if self.model_name.startswith("downstream"):
                if sequence_length < 11:
                    raise ValueError(
                        f"Seq len for {self.model_name} must be >  11."
                        f" Provided: {sequence_length}"
                    )
                elif sequence_length > desired_length:
                    raise ValueError(
                        f"Seq len for {self.model_name} must be <= {desired_length}."
                        f" Provided: {sequence_length}"
                    )

            # Tokenizing the sequence with proxy_species and kmers_stride1
            tokenized_data = self.tokenizer(
                proxy_species + " " + " ".join(kmers_stride1(sequence)),
                return_tensors="pt",
            )

            # Keep track of the original seq len if slice needed later
            input_ids_len = tokenized_data["input_ids"].shape[-1]

            if self.model_name.startswith("upstream") and input_ids_len < 1001:
                tokenized_data, pad_start, pad_end = self._pad_sequence(
                    tokenized_data, mean_embedding
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
                        f" Max layer is {len(hidden_states)}."
                    )
                embedding = torch.mean(
                    torch.stack(
                        hidden_states[self.target_layer[0] : self.target_layer[1] + 1]
                    ),
                    dim=0,
                )[0]

            if self.model_name.startswith("upstream") and input_ids_len < 1001:
                embedding = torch.concat(
                    [embedding[:pad_start, :], embedding[pad_end:, :]]
                )

            if mean_embedding:
                embedding = embedding.mean(dim=0).cpu()

            embeddings.append(embedding)

        embeddings_tensor = (
            torch.stack(embeddings)
            if not mean_embedding
            else torch.cat(embeddings).view(len(sequences), -1)
        )
        return embeddings_tensor


if __name__ == "__main__":
    model = FungalUpDownTransformer(model_name="upstream_species_lm")
    model.embed(["A" * 500 + "ATG"], mean_embedding=True)
