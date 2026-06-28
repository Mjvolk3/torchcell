# mypy: ignore-errors
# DEAD SCRIPT: legacy smoke-test referencing removed APIs (CellDataset no longer
# exists in torchcell.datasets; SCerevisiaeGenome/NucleotideTransformerDataset no
# longer accept data_root/transformer_model_name). Unimportable at runtime and
# referenced nowhere. Left as-is per mypy-pass policy; suppress file-level so it
# does not block the strict gate. Remove the file or rewrite against current APIs.
"""Smoke-test script that iterates an LMDB-backed CellDataset DataLoader."""

import os
import os.path as osp

from dotenv import load_dotenv
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from torchcell.datasets import CellDataset, NucleotideTransformerDataset
from torchcell.datasets.scerevisiae import DmfCostanzo2016Dataset
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


def main() -> None:
    """Build the cell dataset and iterate its DataLoader to exercise LMDB access."""
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    # Get reference genome
    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()
    genome_gene_set = genome.gene_set

    # Sequence transformers
    nt_dataset = NucleotideTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embed"),
        genome=genome,
        transformer_model_name="nt_window_5979",
    )

    # Experiments
    experiments = DmfCostanzo2016Dataset(
        preprocess="low_dmf_std",
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016"),
    )
    print(len(experiments))

    # Gather into CellDatset
    cell_dataset = CellDataset(
        root=osp.join(osp.join(DATA_ROOT, "data/scerevisiae/cell")),
        genome_gene_set=genome_gene_set,
        seq_embeddings=nt_dataset,
        experiments=experiments,
    )
    print(len(cell_dataset))
    data_loader = DataLoader(cell_dataset, batch_size=32, shuffle=True, num_workers=2)
    print(cell_dataset)
    print(cell_dataset[0])

    for batch in tqdm(data_loader):
        # print(f"Accessing LMDB from process {os.getpid()}")
        pass


if __name__ == "__main__":
    main()
