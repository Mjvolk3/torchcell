"""
Check script to verify the InferenceDataset was created correctly.

Uses the InferenceDataset class with Perturbation graph processor
to test the full pipeline (matching equivariant_cell_graph_transformer.py).
"""

import os
import os.path as osp
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


def main():
    # Import here to avoid slow imports if just checking file
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.data.graph_processor import Perturbation

    # Import from the script that has the InferenceDataset class
    import sys

    sys.path.insert(0, osp.dirname(__file__))
    from inference_dataset_1 import InferenceDataset

    # Paths
    inference_dir = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/inference_1"
    )

    print(f"Checking InferenceDataset at: {inference_dir}")
    print("=" * 80)

    # Initialize genome for gene set (required by InferenceDataset)
    print("\nInitializing genome...")
    genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
    go_root = osp.join(DATA_ROOT, "data/go")
    genome = SCerevisiaeGenome(
        genome_root=genome_root, go_root=go_root, overwrite=False
    )
    genome.drop_empty_go()
    print(f"Genome gene set size: {len(genome.gene_set)}")

    # Create InferenceDataset WITH Perturbation graph processor
    # (same as equivariant_cell_graph_transformer.py line 250, 313)
    print("\nCreating InferenceDataset with Perturbation graph processor...")
    dataset = InferenceDataset(
        root=inference_dir,
        gene_set=genome.gene_set,
        graphs=None,
        node_embeddings=None,
        graph_processor=Perturbation(),  # Same as training script
    )

    # Check length
    print("\n" + "=" * 80)
    print(f"Total experiments in dataset: {len(dataset):,}")
    print("=" * 80)

    # Get example data point - now returns processed CellData graph
    print("\nExample data point (index 0) using dataset[0]:")
    print("-" * 40)

    data = dataset[0]
    if data is not None:
        # With Perturbation processor, data is a CellData/Data object
        print(f"Type: {type(data).__name__}")
        print(f"\nGraph attributes:")
        for key in data.keys():
            val = data[key]
            if hasattr(val, "shape"):
                print(f"  {key}: {type(val).__name__} shape={val.shape}")
            elif hasattr(val, "__len__") and not isinstance(val, str):
                print(f"  {key}: {type(val).__name__} len={len(val)}")
            else:
                print(f"  {key}: {val}")

        # Key attributes for training
        print(f"\nKey training attributes:")
        if hasattr(data, "perturbation_indices"):
            print(f"  perturbation_indices: {data.perturbation_indices}")
        if hasattr(data, "y"):
            print(f"  y (labels): {data.y}")
        if hasattr(data, "x"):
            print(f"  x shape: {data.x.shape if data.x is not None else 'None'}")
    else:
        print("ERROR: Could not retrieve index 0")

    # Spot checks at various indices
    print("\n" + "=" * 80)
    print("Spot checks (perturbation_indices at various dataset indices):")
    print("=" * 80)

    total = len(dataset)
    check_indices = [0, 1000, 100000, 1000000, 10000000, 100000000, total - 1]

    for idx in check_indices:
        if idx < total:
            data = dataset[idx]
            if data is not None and hasattr(data, "perturbation_indices"):
                pert_indices = data.perturbation_indices.tolist()
                print(f"  [{idx:>12,}]: perturbation_indices = {pert_indices}")
            else:
                print(f"  [{idx:>12,}]: ERROR - not found or no perturbation_indices")

    # Close LMDB
    dataset.close_lmdb()

    print("\n" + "=" * 80)
    print(
        "Check complete! InferenceDataset with Perturbation processor works correctly."
    )


if __name__ == "__main__":
    main()
