import os
import os.path as osp
import json
from dotenv import load_dotenv
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.data import MeanExperimentDeduplicator
from torchcell.data import GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.neo4j_cell import SubgraphRepresentation
from tqdm import tqdm
from torchcell.utils import format_scientific_notation
from torchcell.datasets import CodonFrequencyDataset
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.viz.datamodules import plot_dataset_index_split
from torchcell.datamodules.cell import overlap_dataset_index_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_label_histograms(
    label_df: pd.DataFrame,
    train_indices: list,
    val_indices: list,
    test_indices: list,
    save_dir: str,
):
    # Exclude the index column from plotting.
    label_cols = [col for col in label_df.columns if col != "index"]
    # Build data subsets from the label dataframe.
    train_df = label_df[label_df["index"].isin(train_indices)]
    val_df = label_df[label_df["index"].isin(val_indices)]
    test_df = label_df[label_df["index"].isin(test_indices)]

    for col in label_cols:
        plt.figure(figsize=(8, 6))
        # Plot full dataset histogram with KDE.
        sns.histplot(
            label_df[col],
            color="gray",
            stat="density",
            kde=True,
            label="All",
            alpha=0.4,
        )
        sns.histplot(
            train_df[col],
            color="blue",
            stat="density",
            kde=True,
            label="Train",
            alpha=0.4,
        )
        sns.histplot(
            val_df[col], color="green", stat="density", kde=True, label="Val", alpha=0.4
        )
        sns.histplot(
            test_df[col], color="red", stat="density", kde=True, label="Test", alpha=0.4
        )
        plt.title(f"Histogram for {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(save_dir, f"hist_{col}.png"))
        plt.close()


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
        print("Finished setting up the perturbation subset data module.")

        # Test the dataloader.
        for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
            break

        # Save the subset indices.
        save_dir = osp.join(
            dataset_root, f"subset_size_{format_scientific_notation(size)}"
        )
        os.makedirs(save_dir, exist_ok=True)
        with open(osp.join(save_dir, "index.json"), "w") as f:
            json.dump(perturbation_subset_data_module.index.model_dump(), f, indent=2)

        # Plot the dataset index splits.
        exp_name = "experiments-003"
        query_name = "query-001-small-build"
        dm_name = "perturbation-subset-data-module"
        size_str = format_scientific_notation(size)
        subset_tag = ",".join(gene_subsets.keys())
        subset_str = f"_subset_{subset_tag}" if subset_tag else ""

        # Plot dataset name index.
        ds_index = "dataset-name-index"
        title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}{subset_str}"
        print("Plotting dataset name index...")
        split_index = overlap_dataset_index_split(
            dataset_index=dataset.dataset_name_index,
            data_module_index=perturbation_subset_data_module.index,
        )
        plot_dataset_index_split(
            split_index=split_index,
            title=title,
            save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
        )
        print("Finished plotting dataset name index.")

        # Plot phenotype label index.
        ds_index = "phenotype-label-index"
        title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}{subset_str}"
        print("Plotting phenotype label index...")
        split_index = overlap_dataset_index_split(
            dataset_index=dataset.phenotype_label_index,
            data_module_index=perturbation_subset_data_module.index,
        )
        plot_dataset_index_split(
            split_index=split_index,
            title=title,
            save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
        )
        print("Finished plotting phenotype label index.")

        # Plot perturbation count index.
        ds_index = "perturbation-count-index"
        title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}{subset_str}"
        print("Plotting perturbation count index...")
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
        print("Finished plotting perturbation count index.")

        # --- Now plot overlapping histograms for each label ---
        # Get the full label dataframe from the dataset.
        label_df = (
            dataset.label_df
        )  # assuming dataset.label_df returns a pandas DataFrame
        # Get the subset indices for train, val, and test.
        train_indices = perturbation_subset_data_module.index.train
        val_indices = perturbation_subset_data_module.index.val
        test_indices = perturbation_subset_data_module.index.test
        print("Plotting label histograms...")
        plot_label_histograms(
            label_df, train_indices, val_indices, test_indices, save_dir
        )
        print("Finished plotting label histograms.")


if __name__ == "__main__":
    main()
