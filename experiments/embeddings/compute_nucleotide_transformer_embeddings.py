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
        "nt_window_3utr_5979",
        "nt_window_3utr_5979_undersize",
        "nt_window_5utr_5979",
        "nt_window_5utr_5979_undersize",
        "nt_window_3utr_300",
        "nt_window_3utr_300_undersize",
        "nt_window_5utr_1000",
        "nt_window_5utr_1000_undersize",
    ]
    event = 0
    for model_name in model_names:
        print(f"event: {event}")
        print(f"starting model_name: {model_name}")
        wandb.log({"event": event})
        dataset = NucleotideTransformerDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embed"),
            genome=genome,
            model_name=model_name,
        )
        print(f"Completed Dataset for {model_name}: {dataset}")
        event += 1


if __name__ == "__main__":
    main()
