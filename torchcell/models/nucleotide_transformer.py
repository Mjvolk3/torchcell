import os

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from torchcell.models.llm import NucleotideModel

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"


class NucleotideTransformer(NucleotideModel):
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.load_model()

    @staticmethod
    def _check_and_download_model():
        # Define the model name

        # Define the directory where you want the model to be saved
        script_dir = os.path.dirname(os.path.realpath(__file__))
        target_directory = os.path.join(
            script_dir, "pretrained_LLM", "nucleotide_transformer"
        )

        # Create the target directory if it doesn't exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        model_directory = os.path.join(target_directory, MODEL_NAME)

        # Check if the model has already been downloaded
        if os.path.exists(model_directory):
            print(f"{MODEL_NAME} model already downloaded.")
        else:
            print(f"Downloading {MODEL_NAME} model to {model_directory}...")
            # tokenizer
            AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=target_directory)
            # model
            AutoModelForMaskedLM.from_pretrained(MODEL_NAME, cache_dir=target_directory)
            print("Download finished.")

    @property
    def max_sequence_size(self):
        # Found empirically... listed as 6kb in paper.
        return 5979

    def load_model(self):
        # Check and download the model if necessary
        self._check_and_download_model()

        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        tokens_ids = self.tokenizer.batch_encode_plus(sequences, return_tensors="pt")[
            "input_ids"
        ]
        # TODO check this error handling... looks wrong with 1000
        for token_id, sequence in zip(tokens_ids, sequences):
            if len(token_id) > 1000:
                raise ValueError(
                    f"Sequence length ({len(sequence)}) is too long for this model. "
                    f"Max sequence size is {self.max_size}."
                    f"Token length is {len(token_id)}."
                    f"Max token length is 1000."
                )
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
    # Initialize the NucleotideTransformer
    transformer = NucleotideTransformer()

    # Create a dummy dna sequence
    sequences = ["ACTACG" * 100, "ACTACG" * 100]
    sequences = [i[:5979] for i in sequences]
    print(len(sequences[0]))
    # Get embeddings
    embeddings = transformer.embed(sequences)

    # print(f"Embeddings per token: {embeddings}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Get Mean embeddings
    embeddings = transformer.embed(sequences, mean_embedding=True)

    # print(f"Embeddings per token: {embeddings}")
    print(f"Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
