# experiments/003-fit-int/scripts/metabolism_enriched_perturbation_subset
# [[experiments.003-fit-int.scripts.metabolism_enriched_perturbation_subset]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/003-fit-int/scripts/metabolism_enriched_perturbation_subset
# Test file: experiments/003-fit-int/scripts/test_metabolism_enriched_perturbation_subset.py

import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.data import MeanExperimentDeduplicator
from torchcell.data import GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
import json
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.neo4j_cell import SubgraphRepresentation
from tqdm import tqdm
from torchcell.utils import format_scientific_notation
from torchcell.datasets import CodonFrequencyDataset
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.viz.datamodules import plot_dataset_index_split
from torchcell.datamodules.cell import overlap_dataset_index_split


def main():
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    # Initialize genome and GEM.
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_empty_go()

    # Get metabolism genes from the GEM.
    gem = YeastGEM()
    metabolism_genes = gem.gene_set
    print(f"Number of metabolism genes: {len(metabolism_genes)}")

    # Save metabolism genes for reference.
    with open("metabolism_genes.json", "w") as f:
        json.dump(list(metabolism_genes), f)

    # Setup graph and node embeddings.
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )

    # Load query and initialize the dataset.
    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        node_embeddings={"codon_frequency": codon_frequency},
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    seed = 42
    # Initialize the base CellDataModule.
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=seed,
        num_workers=4,
        pin_memory=False,
    )
    cell_data_module.setup()

    # Process different subset sizes while constraining to metabolism genes.
    # sizes = [5e1, 1e2, 5e2, 1e3, 5e3, 7e3, 1e4, 5e4, 1e5]
    sizes = [2.5e4, 5e4]
    for size in sizes:
        print(f"\nProcessing size {format_scientific_notation(size)}")

        gene_subsets = {"metabolism": metabolism_genes}
        perturbation_subset_data_module = PerturbationSubsetDataModule(
            cell_data_module=cell_data_module,
            size=int(size),
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            prefetch=False,
            seed=seed,
            gene_subsets=gene_subsets,
        )
        perturbation_subset_data_module.setup()

        # Print metabolism gene statistics.
        train_data = [dataset[i] for i in perturbation_subset_data_module.index.train]
        metabolism_count = sum(
            1
            for idx in perturbation_subset_data_module.index.train
            if any(
                gene in metabolism_genes
                for gene, indices in dataset.is_any_perturbed_gene_index.items()
                if idx in indices
            )
        )
        print(
            f"Train samples with metabolism genes: {metabolism_count}/{len(train_data)} "
            f"({metabolism_count/len(train_data)*100:.2f}%)"
        )

        # Test the dataloader.
        for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
            break

        # Save the subset indices.
        save_dir = osp.join(
            dataset_root, f"subset_size_{format_scientific_notation(size)}"
        )
        os.makedirs(save_dir, exist_ok=True)
        with open(osp.join(save_dir, "index.json"), "w") as f:
            json.dump(perturbation_subset_data_module.index.model_dump(), f)

        # Plot the dataset index splits.
        exp_name = "experiments-003"
        query_name = "query-001-small-build"
        dm_name = "perturbation-subset-data-module"
        size_str = format_scientific_notation(size)

        # Plot dataset name index.
        ds_index = "dataset-name-index"
        subset_tag = ",".join(gene_subsets.keys())
        title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}_subset_{subset_tag}"
        split_index = overlap_dataset_index_split(
            dataset_index=dataset.dataset_name_index,
            data_module_index=perturbation_subset_data_module.index,
        )
        plot_dataset_index_split(
            split_index=split_index,
            title=title,
            save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
        )

        # Plot phenotype label index.
        ds_index = "phenotype-label-index"
        title = (
            f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
        )
        split_index = overlap_dataset_index_split(
            dataset_index=dataset.phenotype_label_index,
            data_module_index=perturbation_subset_data_module.index,
        )
        plot_dataset_index_split(
            split_index=split_index,
            title=title,
            save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
        )

        # Plot perturbation count index.
        ds_index = "perturbation-count-index"
        title = (
            f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
        )
        split_index = overlap_dataset_index_split(
            dataset_index=dataset.perturbation_count_index,
            data_module_index=perturbation_subset_data_module.index,
        )
        plot_dataset_index_split(
            split_index=split_index,
            title=title,
            save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
            threshold=0.0,
        )


if __name__ == "__main__":
    main()
