from torchcell.data import Neo4jCellDataset, ExperimentDeduplicator
import os.path as osp
from dotenv import load_dotenv
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
import os
import json
from torchcell.sequence.genome.scerevisiae.S288C import SCerevisiaeGenome
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
from tqdm import tqdm


def main():
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    with open("experiments/002-dmi-tmi/queries/dmi-tmi_1e04.cql", "r") as f:
        query = f.read()

    ### Add Embeddings
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    with open("gene_set.json", "w") as f:
        json.dump(list(genome.gene_set), f)

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )

    deduplicator = ExperimentDeduplicator()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/002-dmi-tmi/1e04"
    )
    dataset = Neo4jCellDataset( 
        root=dataset_root,
        query=query,
        genome=genome,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        node_embeddings={
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        deduplicator=deduplicator,
        max_size=None,
    )
    print(len(dataset))
    # Data module testing

    data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        batch_size=8,
        random_seed=42,
        num_workers=4,
        pin_memory=False,
    )
    data_module.setup()
    for batch in tqdm(data_module.all_dataloader()):
        pass
        print()

    print("finished")


if __name__ == "__main__":
    main()
