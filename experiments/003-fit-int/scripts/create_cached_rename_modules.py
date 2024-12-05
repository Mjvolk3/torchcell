import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
from torchcell.data import MeanExperimentDeduplicator
from torchcell.data import GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
import json
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.neo4j_cell import SubgraphRepresentation
from tqdm import tqdm
from torchcell.viz.datamodules import plot_dataset_index_split
from torchcell.datamodules.cell import overlap_dataset_index_split
from torchcell.utils import format_scientific_notation


def main():
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
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
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        node_embeddings={
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    print(len(dataset))
    # Data module testing

    print(dataset[2])
    dataset.close_lmdb()

    seed = 42
    # Base Module
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

    for batch in tqdm(cell_data_module.train_dataloader()):
        break

    exp_name = "experiments-003"
    query_name = "query-001-small-build"
    dm_name = "cell-data-module"

    ## Cell Data Module - Dataset Index Plotting - Start

    ## Dataset Index Plotting - Start
    size_str = str(len(dataset))
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=cell_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=cell_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.perturbation_count_index,
        data_module_index=cell_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
        threshold=0.0,
    )
    # ## Cell Data Module - Dataset Index Plotting - End
    
    ## Subset
    dm_name = "perturbation-subset-data-module"
    # 5e1 Module
    size = 5e1
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
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
    ## Dataset Index Plotting - End
    
    ## Subset
    dm_name = "perturbation-subset-data-module"
    # 1e2 Module
    size = 1e2
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
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
    ## Dataset Index Plotting - End
    
    ## Subset
    dm_name = "perturbation-subset-data-module"
    # 5e2 Module
    size = 5e2
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
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
    ## Dataset Index Plotting - End
    
    ## Subset
    dm_name = "perturbation-subset-data-module"
    # 1e3 Module
    size = 1e3
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
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
    ## Dataset Index Plotting - End

    # 5e3 Module
    size = 5e3
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
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
    ## Dataset Index Plotting - End

    # 5e3 Module
    size = 7e3
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
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
    ## Dataset Index Plotting - End

    # 1e4 Module
    size = 1e4
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
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
    ## Dataset Index Plotting - End

    # 5e4 Module
    size = 5e4
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
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
    ## Dataset Index Plotting - End

    # 1e5 Module
    size = 1e5
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
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
    ## Dataset Index Plotting - End

    # 5e5 Module
    size = 5e5
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
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
    ## Dataset Index Plotting - End

    # 1e6 Module
    size = 1e6
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
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
    ## Dataset Index Plotting - End


if __name__ == "__main__":
    main()
