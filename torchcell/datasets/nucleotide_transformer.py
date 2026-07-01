"""Dataset of Nucleotide Transformer embeddings for S. cerevisiae genes."""

# torchcell/datasets/nucleotide_transformer.py
# [[torchcell.datasets.nucleotide_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/nucleotide_transformer.py
# Test file: tests/torchcell/datasets/test_nucleotide_transformer.py
import os
import os.path as osp
from collections.abc import Callable
from typing import Any, cast

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.data.embedding import BaseEmbeddingDataset
from torchcell.models.nucleotide_transformer import NucleotideTransformer
from torchcell.sequence import ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class NucleotideTransformerDataset(BaseEmbeddingDataset):
    """Per-gene Nucleotide Transformer sequence embeddings keyed by window model."""

    MODEL_TO_WINDOW = {
        "nt_window_5979_max": ("window", 5979, True),
        "nt_window_5979": ("window", 5979, False),
        "nt_window_three_prime_5979": ("window_three_prime", 5979, True, True),
        "nt_window_five_prime_5979": ("window_five_prime", 5979, True, True),
        "nt_window_three_prime_300": ("window_three_prime", 300, True, True),
        "nt_window_five_prime_1003": ("window_five_prime", 1003, True, True),
    }

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        model_name: str | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
    ) -> None:
        """Set up the dataset, computing or loading embeddings if a model is given.

        Args:
            root: Root directory for raw and processed data.
            genome: Genome providing gene sequences and the gene set.
            model_name: Key into ``MODEL_TO_WINDOW`` selecting the embedding model.
            transform: Optional transform applied at access time.
            pre_transform: Optional transform applied before saving.
        """
        self.genome: SCerevisiaeGenome | ParsedGenome | None = genome
        self.model_name = model_name
        super().__init__(root, self.model_name, transform, pre_transform)

        # Conditionally load the data
        if self.model_name:
            print(self.processed_paths[0])
            if not os.path.exists(self.processed_paths[0]):
                # Initialize the language model
                self.transformer = self.initialize_model()
                self.process()
            # TODO me might consider adding this to others
            # only her bc computed on delta.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.data, self.slices = torch.load(
                self.processed_paths[0], map_location=torch.device(device)
            )

        self.genome = self.parse_genome(genome)
        del genome

    @staticmethod
    def parse_genome(genome: SCerevisiaeGenome | None) -> ParsedGenome | None:
        """Build a ParsedGenome holding the gene set, or None if genome is None."""
        # BUG we have to do this black magic because when you merge datasets with +
        # the genome is None
        if genome is None:
            return None
        else:
            data = {}
            data["gene_set"] = genome.gene_set
            return ParsedGenome(**data)

    def initialize_model(self) -> NucleotideTransformer | None:
        """Instantiate the NucleotideTransformer model, or None if no model name."""
        if self.model_name:
            return NucleotideTransformer()
        return None

    def process(self) -> None:
        """Embed each gene's windowed sequence and save the collated dataset."""
        if self.model_name is None:
            return

        data_list = []
        if "five_prime" in self.model_name or "three_prime" in self.model_name:
            window_method, window_size, has_special_codon, allow_undersize = (
                self.MODEL_TO_WINDOW[self.model_name]
            )
        else:
            window_method, window_size, is_max_size = self.MODEL_TO_WINDOW[
                self.model_name
            ]
        window_method = cast(str, window_method)

        # TODO check that genome gene set is SortedSet
        sequences = []
        gene_ids = []
        genome = cast(SCerevisiaeGenome, self.genome)
        transformer = cast(NucleotideTransformer, self.transformer)
        for gene_id in tqdm(genome.gene_set):
            sequence = genome[gene_id]

            if "three_prime" in window_method or "five_prime" in window_method:
                dna_selection = getattr(sequence, window_method)(
                    window_size, allow_undersize=allow_undersize
                )
            else:
                dna_selection = getattr(sequence, window_method)(
                    window_size, is_max_size=is_max_size
                )

            sequences.append(dna_selection.seq)
            gene_ids.append(gene_id)
        # Compute embeddings in batches
        batch_size = 1  # Adjust the batch size according to your memory constraints
        embeddings_list = []
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i : i + batch_size]
            batch_embeddings = transformer.embed(batch_sequences, mean_embedding=True)
            embeddings_list.append(batch_embeddings)

        embeddings = torch.cat(embeddings_list, dim=0)

        for gene_id, dna_selection, embedding in zip(gene_ids, sequences, embeddings):
            # Create or update the dna_window dictionary
            dna_window_dict = {self.model_name: dna_selection}

            data = Data(id=gene_id, dna_windows=dna_window_dict)
            data.embeddings = {self.model_name: embedding}
            data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])


def main() -> None:
    """Build embedding datasets for all configured window models."""
    import wandb
    from dotenv import load_dotenv

    print("Starting main...")
    wandb.init(mode="online", project="torchcell_embeddings")
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(cast(str, DATA_ROOT), "data/sgd/genome")
    )  # type: ignore[call-arg]  # data_root is a valid Genome param; attrs hides it from mypy
    model_names = [
        "nt_window_5979",
        "nt_window_5979_max",
        "nt_window_three_prime_5979",
        "nt_window_five_prime_5979",
        "nt_window_three_prime_300",
        "nt_window_five_prime_1003",
    ]
    event = 0
    for model_name in model_names:
        print(f"event: {event}")
        print(f"starting model_name: {model_name}")
        wandb.log({"event": event})
        dataset = NucleotideTransformerDataset(
            root=osp.join(
                cast(str, DATA_ROOT), "data/scerevisiae/nucleotide_transformer_embed"
            ),
            genome=genome,
            model_name=model_name,
        )
        print(f"Completed Dataset for {model_name}: {dataset}")
        event += 1


if __name__ == "__main__":
    main()
