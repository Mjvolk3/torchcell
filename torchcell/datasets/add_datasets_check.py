import os
import os.path as osp

from torchcell.datasets import (
    CodonFrequencyDataset,
    FungalUpDownTransformerDataset,
    OneHotGeneDataset,
    ProtT5Dataset,
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
        root="data/scerevisiae/gene_one_hot", genome=genome
    )
    prot_T5_dataset_no_dubious = ProtT5Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embed"),
        genome=genome,
        model_name="prot_t5_xl_uniref50_no_dubious",
    )
    prot_T5_dataset_all = ProtT5Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embed"),
        genome=genome,
        model_name="prot_t5_xl_uniref50_all",
    )
    print()
    # print(fud_downstream)
    # print(fud_downstream[100])
    # print(one_hot_gene)
    # print(one_hot_gene[100])
    # # BUG if the datasets are not the same size they will not be added properly. Should make sure that all datasets are the same size. Add.
    # dataset = fud_downstream + one_hot_gene
    # print(dataset)
    # print(dataset[100])
