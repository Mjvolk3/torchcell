from dotenv import load_dotenv
import wandb
from torchcell.datasets import NucleotideTransformerDataset
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
import os.path as osp
import os

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


def main():
    print("Starting main...")
    wandb.init(mode="online", project="torchcell_embeddings")

    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
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
            root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"),
            genome=genome,
            model_name=model_name,
        )
        print(f"Completed Dataset for {model_name}: {dataset}")
        event += 1


if __name__ == "__main__":
    main()
