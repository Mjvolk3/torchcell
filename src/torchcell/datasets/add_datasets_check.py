import os
import os.path as osp

from torchcell.datasets import (
    CodonFrequencyDataset,
    FungalUpDownTransformerDataset,
    OneHotGeneDataset,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))

    fud_downstream = FungalUpDownTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/fungal_up_down_embed"),
        genome=genome,
        model_name="species_downstream",
    )

    fud_upstream = FungalUpDownTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/fungal_up_down_embed"),
        genome=genome,
        model_name="species_upstream",
    )

    codon_freq = CodonFrequencyDataset(
        root="data/scerevisiae/codon_frequency", genome=genome
    )

    one_hot_gene = OneHotGeneDataset(
        root="data/scerevisiae/gene_one_hot_encoded", genome=genome
    )
    dataset = fud_downstream + fud_upstream + codon_freq + one_hot_gene
    print(dataset)
    print(dataset[100])
