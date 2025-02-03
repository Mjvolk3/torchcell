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
    title_prefix: str,
    save_dir: str,
):
    # Ensure the DataFrame has an "index" column.
    label_df = label_df.copy()
    if "index" not in label_df.columns:
        label_df.reset_index(inplace=True)
    label_cols = [col for col in label_df.columns if col != "index"]
    train_df = label_df[label_df["index"].isin(train_indices)]
    val_df = label_df[label_df["index"].isin(val_indices)]
    test_df = label_df[label_df["index"].isin(test_indices)]

    for col in label_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(
            label_df[col].dropna(),
            color="gray",
            stat="density",
            kde=True,
            label="All",
            alpha=0.4,
        )
        sns.histplot(
            train_df[col].dropna(),
            color="blue",
            stat="density",
            kde=True,
            label="Train",
            alpha=0.4,
        )
        sns.histplot(
            val_df[col].dropna(),
            color="green",
            stat="density",
            kde=True,
            label="Val",
            alpha=0.4,
        )
        sns.histplot(
            test_df[col].dropna(),
            color="red",
            stat="density",
            kde=True,
            label="Test",
            alpha=0.4,
        )
        # Use the base title prefix in the figure title.
        plt.title(f"{title_prefix} - {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        file_name = osp.join(save_dir, f"{title_prefix}_hist_{col}.png")
        plt.savefig(file_name, dpi=600)
        plt.close()


def main():
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    # Initialize genome and GEM.
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_empty_go()
    gem = YeastGEM()
    metabolism_genes = gem.gene_set
    print(f"Number of metabolism genes: {len(metabolism_genes)}")
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

    sizes = [1e5, 4e5]
    # sizes = [2.5e4, 5e4, 1e5, 5e5]
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

        for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
            break

        # Construct a common title prefix.
        exp_name = "experiments-003"
        query_name = "query-001-small-build"
        dm_name = "perturbation-subset-data-module"
        size_str = format_scientific_notation(size)
        subset_tag = ",".join(gene_subsets.keys()) if gene_subsets else ""
        subset_str = f"_subset_{subset_tag}" if subset_tag else ""
        # Base title prefix used for all plots.
        base_title = (
            f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}{subset_str}"
        )

        # Plot dataset name index.
        ds_index = "dataset-name-index"
        title = f"{base_title}_{ds_index}"
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
        title = f"{base_title}_{ds_index}"
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
        title = f"{base_title}_{ds_index}"
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

        # Plot overlapping histograms for each label.
        hist_title_prefix = f"{base_title}"
        print("Plotting label histograms...")
        label_df = dataset.label_df.copy()
        if "index" not in label_df.columns:
            label_df.reset_index(inplace=True)
        train_indices = perturbation_subset_data_module.index.train
        val_indices = perturbation_subset_data_module.index.val
        test_indices = perturbation_subset_data_module.index.test
        plot_label_histograms(
            label_df,
            train_indices,
            val_indices,
            test_indices,
            hist_title_prefix,
            ASSET_IMAGES_DIR,
        )
        print("Finished plotting label histograms.")


if __name__ == "__main__":
    main()
