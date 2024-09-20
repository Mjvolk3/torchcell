from dotenv import load_dotenv
import wandb
from torchcell.datasets.esm2 import Esm2Dataset
from torchcell.sequence.genome.scerevisiae.S288C import SCerevisiaeGenome
import os.path as osp
import os

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

def main():
    print("Starting main...")
    wandb.init(mode="online", project="torchcell_embeddings")

    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))

    model_names = [
        "esm2_t33_650M_UR50D_all",
        "esm2_t33_650M_UR50D_no_dubious",
        # "esm2_t36_3B_UR50D_all",
        # "esm2_t36_3B_UR50D_no_dubious_uncharacterized",
        # "esm2_t36_3B_UR50D_no_dubious",
        # "esm2_t36_3B_UR50D_no_uncharacterized",
        # "esm2_t48_15B_UR50D_all",
        # "esm2_t48_15B_UR50D_no_dubious_uncharacterized",
        # "esm2_t48_15B_UR50D_no_dubious",
        # "esm2_t48_15B_UR50D_no_uncharacterized",
    ]

    event = 0
    for model_name in model_names:
        print(f"event: {event}")
        print(f"starting model_name: {model_name}")
        wandb.log({"event": event})

        dataset = Esm2Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
            genome=genome,
            model_name=model_name,
        )

        print(f"Completed Dataset for {model_name}: {dataset}")
        event += 1

if __name__ == "__main__":
    main()