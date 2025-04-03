# experiments/003-fit-int/scripts/hetero_cell_bipartite_powerset_gi
# [[experiments.003-fit-int.scripts.hetero_cell_bipartite_powerset_gi]]

import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import torch
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import combinations, chain
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils._subgraph import bipartite_subgraph, subgraph
from dotenv import load_dotenv

# Import custom modules
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.models.hetero_cell_bipartite import HeteroCellBipartite
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph import SCerevisiaeGraph
from torchcell.transforms.regression_to_classification import (
    LabelNormalizationTransform,
    InverseCompose,
)
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.trainers.fit_int_hetero_cell import RegressionTask
from torchcell.losses.isomorphic_cell_loss import ICLoss

# Set up logging
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")


def power_set(genes):
    """Generate all possible subsets of the genes."""
    return chain.from_iterable(combinations(genes, r) for r in range(len(genes) + 1))


def get_sample_with_multi_deletions(dataset, target_deletions=2, limit=20):
    """Find samples with multiple gene deletions."""
    samples = []
    for idx in tqdm(range(min(1000, len(dataset))), desc="Finding samples"):
        data = dataset[idx]
        if (
            hasattr(data["gene"], "ids_pert")
            and len(data["gene"].ids_pert)
            == target_deletions  # Specific number of deletions
            and not torch.isnan(data["gene"].fitness).any()
            and not torch.isnan(data["gene"].gene_interaction).any()
        ):
            samples.append(idx)
            if len(samples) >= limit:
                break
    return samples


def compute_analytic_interaction(fitness_values, perturbed_genes):
    """
    Compute gene interaction using the inclusion-exclusion principle.

    Args:
        fitness_values: Dictionary mapping gene sets to fitness values
        perturbed_genes: Set of perturbed genes
    """
    genes = tuple(sorted(perturbed_genes))

    # ε_S = ∑_{T⊆S} (-1)^{|S|-|T|} f_T
    interaction = 0.0
    for t_size in range(len(genes) + 1):
        for subset in combinations(genes, t_size):
            subset_tuple = tuple(sorted(subset)) if subset else tuple()
            if subset_tuple in fitness_values:
                sign = (-1) ** (len(genes) - len(subset))
                interaction += sign * fitness_values[subset_tuple]
            else:
                log.warning(f"Missing fitness value for subset {subset_tuple}")

    return interaction


@hydra.main(
    version_base=None,
    config_path=osp.join(EXPERIMENT_ROOT, "003-fit-int/conf"),
    config_name="hetero_cell_bipartite_bad_gi_analytic_v_direct",
)
def main(cfg: DictConfig) -> None:
    log.info("Starting Gene Interaction Analysis 🧬")

    # Check checkpoint path
    checkpoint_path = cfg.model.checkpoint_path
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    # Load genome and graph
    log.info("Loading genome and graph...")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
    )
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    # Setup graphs and metabolism based on config
    graphs = {}
    if "physical" in cfg.cell_dataset.graphs:
        graphs["physical"] = graph.G_physical
    if "regulatory" in cfg.cell_dataset.graphs:
        graphs["regulatory"] = graph.G_regulatory

    incidence_graphs = {}
    if (
        hasattr(cfg.cell_dataset, "incidence_graphs")
        and cfg.cell_dataset.incidence_graphs
    ):
        yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
        if "metabolism_hypergraph" in cfg.cell_dataset.incidence_graphs:
            incidence_graphs["metabolism_hypergraph"] = yeast_gem.reaction_map
        elif "metabolism_bipartite" in cfg.cell_dataset.incidence_graphs:
            incidence_graphs["metabolism_bipartite"] = yeast_gem.bipartite_graph

    # Load dataset
    log.info("Loading query and creating dataset...")
    with open(
        osp.join(EXPERIMENT_ROOT, "003-fit-int/queries/001-small-build.cql"), "r"
    ) as f:
        query = f.read()

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    # Create dataset with SubgraphRepresentation for inference
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=graphs,
        incidence_graphs=incidence_graphs,
        node_embeddings={},  # Empty because we'll use learnable embeddings
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    # Configure label normalization (same as training)
    log.info("Setting up transforms...")
    norm_configs = {
        "fitness": {"strategy": "standard"},
        "gene_interaction": {"strategy": "standard"},
    }
    normalize_transform = LabelNormalizationTransform(dataset, norm_configs)
    inverse_transform = InverseCompose([normalize_transform])

    # Apply transform to dataset
    dataset.transform = normalize_transform

    # Load model from checkpoint
    log.info(f"Loading model from checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with the same configuration
    model = HeteroCellBipartite(
        gene_num=cfg.model.gene_num,
        reaction_num=cfg.model.reaction_num,
        metabolite_num=cfg.model.metabolite_num,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
        activation=cfg.model.activation,
        gene_encoder_config=cfg.model.gene_encoder_config,
        metabolism_config=cfg.model.metabolism_config,
        prediction_head_config=cfg.model.prediction_head_config,
        gpr_conv_config=cfg.model.gpr_conv_config,
    ).to(device)

    # Create loss function
    if (
        hasattr(cfg.regression_task, "is_weighted_phenotype_loss")
        and cfg.regression_task.is_weighted_phenotype_loss
    ):
        weights = torch.ones(2).to(device)  # Default weights for inference
    else:
        weights = None

    loss_func = ICLoss(
        lambda_dist=cfg.regression_task.lambda_dist,
        lambda_supcr=cfg.regression_task.lambda_supcr,
        weights=weights,
    )

    # Create regression task for inference
    task = RegressionTask(
        model=model,
        cell_graph=dataset.cell_graph,
        optimizer_config=cfg.regression_task.optimizer,
        lr_scheduler_config=cfg.regression_task.lr_scheduler,
        batch_size=cfg.data_module.batch_size,
        device=device,
        loss_func=loss_func,
        inverse_transform=inverse_transform,
        forward_transform=normalize_transform,
    )

    # Load checkpoint weights
    log.info("Loading checkpoint weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        # Filter out problematic keys related to loss function weights
        problematic_keys = [
            "loss_func.mse_loss_fn.weights",
            "loss_func.dist_loss_fn.weights",
            "loss_func.supcr_fn.weights",
        ]

        filtered_state_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k not in problematic_keys
        }

        task.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint)

    # Set model to evaluation mode
    task.eval()
    model.eval()

    # Create output directory
    output_dir = "gene_interaction_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Find samples with multi-gene deletions - focus on doubles for simplicity
    log.info("Finding multi-gene deletion samples...")
    multi_deletion_indices = get_sample_with_multi_deletions(
        dataset, target_deletions=2, limit=5
    )
    log.info(f"Found {len(multi_deletion_indices)} double-gene deletion samples")

    # Process each multi-gene deletion sample
    results = []

    # Wildtype fitness (assume 1.0)
    wildtype_fitness = 1.0

    # Process each sample directly without custom functions
    for idx in tqdm(multi_deletion_indices, desc="Processing samples"):
        # Get the original sample
        data = dataset[idx]
        perturbed_genes = data["gene"].ids_pert
        log.info(f"Processing sample {idx} with genes {perturbed_genes}")

        # Get true values
        true_fitness = (
            data["gene"].fitness_original.item()
            if hasattr(data["gene"], "fitness_original")
            else data["gene"].fitness.item()
        )

        true_gene_interaction = (
            data["gene"].gene_interaction_original.item()
            if hasattr(data["gene"], "gene_interaction_original")
            else data["gene"].gene_interaction.item()
        )

        # Run inference on original sample (with both genes perturbed)
        batch = Batch.from_data_list([data.to(device)])
        with torch.no_grad():
            predictions, _ = task.model(dataset.cell_graph.to(device), batch)

        # Apply inverse transform
        temp_data = HeteroData()
        temp_data["gene"] = {
            "fitness": predictions[:, 0].clone(),
            "gene_interaction": predictions[:, 1].clone(),
        }
        inv_data = inverse_transform(temp_data)
        pred_fitness = inv_data["gene"]["fitness"].item()
        pred_gene_interaction = inv_data["gene"]["gene_interaction"].item()

        # Initialize fitness dictionary
        fitness_dict = {
            tuple(sorted(perturbed_genes)): pred_fitness,
            (): wildtype_fitness,
        }

        # Find and run inference on single gene perturbations
        for gene in perturbed_genes:
            # Find the corresponding sample in the dataset
            for single_idx in range(len(dataset)):
                single_data = dataset[single_idx]
                if (
                    hasattr(single_data["gene"], "ids_pert")
                    and len(single_data["gene"].ids_pert) == 1
                    and single_data["gene"].ids_pert[0] == gene
                ):

                    # Run inference on this single gene deletion
                    single_batch = Batch.from_data_list([single_data.to(device)])
                    with torch.no_grad():
                        single_predictions, _ = task.model(
                            dataset.cell_graph.to(device), single_batch
                        )

                    # Apply inverse transform
                    single_temp_data = HeteroData()
                    single_temp_data["gene"] = {
                        "fitness": single_predictions[:, 0].clone(),
                        "gene_interaction": single_predictions[:, 1].clone(),
                    }
                    single_inv_data = inverse_transform(single_temp_data)
                    single_pred_fitness = single_inv_data["gene"]["fitness"].item()

                    # Store fitness value
                    fitness_dict[(gene,)] = single_pred_fitness
                    break
            else:
                log.warning(f"Could not find single deletion for gene {gene}")

        # Calculate analytical gene interaction
        if all(
            g in [k[0] for k in fitness_dict.keys() if len(k) == 1]
            for g in perturbed_genes
        ):
            analytic_interaction = compute_analytic_interaction(
                fitness_dict, perturbed_genes
            )
        else:
            analytic_interaction = None
            log.warning(
                f"Missing single deletions for some genes in {perturbed_genes}, cannot compute analytic interaction"
            )

        # Store results
        results.append(
            {
                "idx": idx,
                "perturbed_genes": perturbed_genes,
                "num_gene_deletions": len(perturbed_genes),
                "true_fitness": true_fitness,
                "pred_fitness": pred_fitness,
                "true_gene_interaction": true_gene_interaction,
                "gene_interaction_prediction": pred_gene_interaction,
                "gene_interaction_analytic": analytic_interaction,
                "fitness_values": fitness_dict,
            }
        )

        # Print current result for monitoring
        log.info(f"Sample {idx}, Genes: {perturbed_genes}")
        log.info(f"  True interaction: {true_gene_interaction:.6f}")
        log.info(f"  Model predicted:  {pred_gene_interaction:.6f}")
        if analytic_interaction is not None:
            log.info(f"  Analytic method:  {analytic_interaction:.6f}")

        # Print fitness values for subsets
        log.info("  Fitness values:")
        for subset, fitness in sorted(fitness_dict.items(), key=lambda x: len(x[0])):
            subset_str = ", ".join(subset) if subset else "wildtype"
            log.info(f"    {subset_str}: {fitness:.6f}")

    # Convert results to DataFrame (exclude fitness_values dictionary for CSV)
    results_for_df = [
        {k: v for k, v in r.items() if k != "fitness_values"} for r in results
    ]
    df = pd.DataFrame(results_for_df)

    # Save results
    df.to_csv(os.path.join(output_dir, "gene_interaction_results.csv"), index=False)

    # Compute metrics for complete samples only
    df_complete = df.dropna(subset=["gene_interaction_analytic"])

    if len(df_complete) > 0:
        # Compute metrics
        pred_mse = mean_squared_error(
            df_complete["true_gene_interaction"],
            df_complete["gene_interaction_prediction"],
        )
        pred_r2 = r2_score(
            df_complete["true_gene_interaction"],
            df_complete["gene_interaction_prediction"],
        )
        pred_pearson = pearsonr(
            df_complete["true_gene_interaction"],
            df_complete["gene_interaction_prediction"],
        )[0]
        pred_spearman = spearmanr(
            df_complete["true_gene_interaction"],
            df_complete["gene_interaction_prediction"],
        )[0]

        analytic_mse = mean_squared_error(
            df_complete["true_gene_interaction"],
            df_complete["gene_interaction_analytic"],
        )
        analytic_r2 = r2_score(
            df_complete["true_gene_interaction"],
            df_complete["gene_interaction_analytic"],
        )
        analytic_pearson = pearsonr(
            df_complete["true_gene_interaction"],
            df_complete["gene_interaction_analytic"],
        )[0]
        analytic_spearman = spearmanr(
            df_complete["true_gene_interaction"],
            df_complete["gene_interaction_analytic"],
        )[0]

        # Create metrics table
        metrics = {
            "Model Direct": {
                "MSE": pred_mse,
                "R²": pred_r2,
                "Pearson": pred_pearson,
                "Spearman": pred_spearman,
            },
            "Analytic": {
                "MSE": analytic_mse,
                "R²": analytic_r2,
                "Pearson": analytic_pearson,
                "Spearman": analytic_spearman,
            },
        }

        # Print metrics
        print("\nPerformance Metrics:")
        print(f"{'Metric':<10} {'Model Direct':<15} {'Analytic':<15}")
        print("-" * 40)
        for metric in ["MSE", "R²", "Pearson", "Spearman"]:
            print(
                f"{metric:<10} {metrics['Model Direct'][metric]:<15.4f} {metrics['Analytic'][metric]:<15.4f}"
            )

        # Create visualization
        if len(df_complete) > 1:
            plt.figure(figsize=(12, 5))

            # Model prediction vs true
            plt.subplot(1, 2, 1)
            plt.scatter(
                df_complete["true_gene_interaction"],
                df_complete["gene_interaction_prediction"],
                alpha=0.8,
            )
            plt.plot(
                [
                    df_complete["true_gene_interaction"].min(),
                    df_complete["true_gene_interaction"].max(),
                ],
                [
                    df_complete["true_gene_interaction"].min(),
                    df_complete["true_gene_interaction"].max(),
                ],
                "k--",
                alpha=0.5,
            )
            plt.xlabel("True Gene Interaction")
            plt.ylabel("Model Prediction")
            plt.title(f"Model Direct (r={pred_pearson:.3f})")
            plt.grid(alpha=0.3)

            # Analytic vs true
            plt.subplot(1, 2, 2)
            plt.scatter(
                df_complete["true_gene_interaction"],
                df_complete["gene_interaction_analytic"],
                alpha=0.8,
            )
            plt.plot(
                [
                    df_complete["true_gene_interaction"].min(),
                    df_complete["true_gene_interaction"].max(),
                ],
                [
                    df_complete["true_gene_interaction"].min(),
                    df_complete["true_gene_interaction"].max(),
                ],
                "k--",
                alpha=0.5,
            )
            plt.xlabel("True Gene Interaction")
            plt.ylabel("Analytic Interaction")
            plt.title(f"Analytic Calculation (r={analytic_pearson:.3f})")
            plt.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "gene_interaction_comparison.png"), dpi=300
            )
            plt.close()

    log.info(f"Analysis complete. Results saved to {output_dir}")

    # Close dataset
    dataset.close_lmdb()

    return df, results


if __name__ == "__main__":
    main()
