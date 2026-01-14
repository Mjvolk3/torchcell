# experiments/010-kuzmin-tmi/scripts/trigenic_interaction_adh1_adh3_adh5.py
# [[experiments.010-kuzmin-tmi.scripts.trigenic_interaction_adh1_adh3_adh5]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/trigenic_interaction_adh1_adh3_adh5.py
"""
Multi-sample inference for ADH/ALD triple knockouts from Ng et al. 2012.

This script predicts the genetic interaction fitness for three triple knockouts
relevant to 2,3-butanediol production in yeast:

  1. B2C-a1a3a5: ADH1Δ ADH3Δ ADH5Δ (YOL086C, YMR083W, YBR145W)
  2. B2C-a1a3a6: ADH1Δ ADH3Δ ALD6Δ (YOL086C, YMR083W, YPL061W)
  3. B2C-a1a5a6: ADH1Δ ADH5Δ ALD6Δ (YOL086C, YBR145W, YPL061W)

These knockouts redirect metabolic flux from ethanol production but create
fitness defects from acetaldehyde accumulation and redox imbalance.

Reference: Ng et al. 2012 (DOI: 10.1128/mBio.00012-12)

Model: CellGraphTransformer with graph regularization (Pearson=0.4619)
"""

import json
import os
import os.path as osp
import sys
from typing import Optional

import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

# Add script directory to path for local imports
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from inference_dataset_1 import (
    InferenceDataset,
    InferenceExperiment,
    InferencePhenotype,
)
from torchcell.data import (
    GenotypeAggregator,
    MeanExperimentDeduplicator,
    Neo4jCellDataset,
)
from torchcell.data.graph_processor import Perturbation
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp
from torchcell.trainers.int_transformer_cell import RegressionTask
from torchcell.transforms.coo_regression_to_classification import (
    COOInverseCompose,
    COOLabelNormalizationTransform,
)
from torch_geometric.transforms import Compose


# ============================================================================
# Gene Configuration (Ng et al. 2012 nomenclature)
# ============================================================================
# ADH1 (a1): YOL086C - Primary alcohol dehydrogenase for ethanol production
# ADH3 (a3): YMR083W - Mitochondrial alcohol dehydrogenase
# ADH5 (a5): YBR145W - Alcohol dehydrogenase isoenzyme V (paralog of ADH1)
# ALD6 (a6): YPL061W - Cytosolic aldehyde dehydrogenase (acetaldehyde → acetate)
#
# Note: YBR145W is the TRUE ADH5 (not YDL168W/SFA1)
# ============================================================================
GENE_MAP = {
    "a1": {"name": "ADH1", "systematic": "YOL086C", "function": "Primary alcohol dehydrogenase"},
    "a3": {"name": "ADH3", "systematic": "YMR083W", "function": "Mitochondrial alcohol dehydrogenase"},
    "a5": {"name": "ADH5", "systematic": "YBR145W", "function": "Alcohol dehydrogenase isoenzyme V"},
    "a6": {"name": "ALD6", "systematic": "YPL061W", "function": "Cytosolic aldehyde dehydrogenase"},
}

# Triple knockouts from Table 3 (Ng et al. 2012)
# Format: (strain_name, (gene_alias1, gene_alias2, gene_alias3), experimental_data)
TRIPLE_KNOCKOUTS = [
    (
        "B2C-a1a3a5",
        ("a1", "a3", "a5"),
        {
            "fermentation_time_hr": 54,
            "dry_cell_weight_g_L": 0.900,
            "residual_glucose_g_L": 2.41,
            "acetaldehyde_g_L": 1.316,
            "butanediol_yield": 0.093,
            "glycerol_yield": 0.238,
            "ethanol_yield": 0.131,
        },
    ),
    (
        "B2C-a1a3a6",
        ("a1", "a3", "a6"),
        {
            "fermentation_time_hr": 63,
            "dry_cell_weight_g_L": 0.619,
            "residual_glucose_g_L": 0.78,
            "acetaldehyde_g_L": 0.312,
            "butanediol_yield": 0.023,
            "glycerol_yield": 0.125,
            "ethanol_yield": 0.318,
        },
    ),
    (
        "B2C-a1a5a6",
        ("a1", "a5", "a6"),
        {
            "fermentation_time_hr": 54,
            "dry_cell_weight_g_L": 0.842,
            "residual_glucose_g_L": 0,
            "acetaldehyde_g_L": 0.263,
            "butanediol_yield": 0.033,
            "glycerol_yield": 0.247,
            "ethanol_yield": 0.259,
        },
    ),
]

# Wild-type reference (BY4742)
WILDTYPE_DATA = {
    "fermentation_time_hr": 40,
    "dry_cell_weight_g_L": 1.349,
    "residual_glucose_g_L": 0,
    "acetaldehyde_g_L": 0.118,
    "butanediol_yield": 0.002,
    "glycerol_yield": 0.032,
    "ethanol_yield": 0.417,
}


def check_triple_exists_in_dataset(
    dataset: Neo4jCellDataset,
    gene_aliases: tuple[str, str, str],
) -> tuple[bool, list[int], dict[str, set]]:
    """
    Check if a triple knockout exists in the dataset using set intersection.

    Args:
        dataset: Neo4jCellDataset with is_any_perturbed_gene_index
        gene_aliases: Tuple of gene aliases (e.g., ("a1", "a3", "a5"))

    Returns:
        Tuple of:
        - exists: True if triple exists in dataset
        - indices: List of dataset indices where the triple appears
        - gene_indices: Dict mapping gene alias to set of indices
    """
    gene_indices = {}

    for alias in gene_aliases:
        systematic = GENE_MAP[alias]["systematic"]
        if systematic in dataset.is_any_perturbed_gene_index:
            gene_indices[alias] = set(dataset.is_any_perturbed_gene_index[systematic])
        else:
            gene_indices[alias] = set()

    # Find intersection of all three gene indices
    if all(len(indices) > 0 for indices in gene_indices.values()):
        intersection = gene_indices[gene_aliases[0]]
        for alias in gene_aliases[1:]:
            intersection = intersection & gene_indices[alias]
        return len(intersection) > 0, sorted(list(intersection)), gene_indices

    return False, [], gene_indices


def main():
    """Run inference for ADH/ALD triple knockouts from Ng et al. 2012."""
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

    print("=" * 80)
    print("ADH/ALD Triple Knockout Inference (Ng et al. 2012)")
    print("=" * 80)

    print(f"\nGene Reference:")
    for alias, info in GENE_MAP.items():
        print(f"  {alias}: {info['name']} ({info['systematic']}) - {info['function']}")

    print(f"\nTriple knockouts to analyze:")
    for strain_name, aliases, _ in TRIPLE_KNOCKOUTS:
        genes = [f"{GENE_MAP[a]['name']}Δ" for a in aliases]
        systematic = [GENE_MAP[a]['systematic'] for a in aliases]
        print(f"  {strain_name}: {' '.join(genes)} ({', '.join(systematic)})")
    print()

    # ========================================================================
    # Model Configuration (matching training config)
    # ========================================================================
    model_config = {
        "gene_num": 6607,
        "hidden_channels": 180,
        "num_transformer_layers": 8,
        "num_attention_heads": 9,
        "dropout": 0.1,
        "learnable_embedding": {
            "enabled": True,
            "size": 180,
            "preprocessor": {"num_layers": 2, "dropout": 0.1},
        },
        "graph_regularization": {
            "graph_reg_lambda": 0.001,
            "graph_reg_layer": 1,
            "row_sampling_rate": 1.0,
            "regularized_heads": {
                "physical": {"layer": 1, "head": 0, "lambda": 0.001},
                "regulatory": {"layer": 1, "head": 1, "lambda": 0.001},
                "tflink": {"layer": 1, "head": 2, "lambda": 0.001},
                "string12_0_neighborhood": {"layer": 1, "head": 3, "lambda": 0.001},
                "string12_0_fusion": {"layer": 1, "head": 4, "lambda": 0.001},
                "string12_0_cooccurence": {"layer": 1, "head": 5, "lambda": 0.001},
                "string12_0_coexpression": {"layer": 1, "head": 6, "lambda": 0.001},
                "string12_0_experimental": {"layer": 1, "head": 7, "lambda": 0.001},
                "string12_0_database": {"layer": 1, "head": 8, "lambda": 0.001},
            },
        },
        "perturbation_head": {"num_heads": 9, "dropout": 0.1},
    }

    graph_names = [
        "physical",
        "regulatory",
        "tflink",
        "string12_0_neighborhood",
        "string12_0_fusion",
        "string12_0_cooccurence",
        "string12_0_coexpression",
        "string12_0_experimental",
        "string12_0_database",
    ]

    # Checkpoint path (relative to DATA_ROOT)
    checkpoint_rel_path = (
        "models/checkpoints/compute-3-3-2036902_bd9e6c666ea1c0e7d1bbb6321fbc4d3bd5f60f100d6dc0e0288cd97e366fc15e/"
        "c7671wgj-best-pearson-epoch=24-val/gene_interaction/Pearson=0.4619.ckpt"
    )
    checkpoint_path = osp.join(DATA_ROOT, checkpoint_rel_path)

    # ========================================================================
    # Initialize Genome and Graph
    # ========================================================================
    print("Initializing genome and graph...")
    genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
    go_root = osp.join(DATA_ROOT, "data/go")

    genome = SCerevisiaeGenome(genome_root=genome_root, go_root=go_root, overwrite=False)
    genome.drop_empty_go()
    print(f"Gene set size: {len(genome.gene_set)}")

    # Verify all genes are in gene set
    print("\nVerifying genes in gene set:")
    for alias, info in GENE_MAP.items():
        systematic = info["systematic"]
        if systematic in genome.gene_set:
            print(f"  ✓ {info['name']} ({systematic}) in gene set")
        else:
            print(f"  ✗ {info['name']} ({systematic}) NOT in gene set")
            raise ValueError(f"{info['name']} ({systematic}) not found in gene set!")

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Build gene multigraph
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)
    graphs_dict = dict(gene_multigraph.graphs) if gene_multigraph else None

    # No pre-computed embeddings (model uses learnable embeddings)
    node_embeddings = None
    print("\nUsing learnable embeddings (no pre-computed node embeddings)")

    # Graph processor for perturbation representation
    graph_processor = Perturbation()

    # ========================================================================
    # Check if Triples Exist in Training Dataset
    # ========================================================================
    print("\n" + "=" * 80)
    print("Checking Dataset for Existing Triple Knockout Data")
    print("=" * 80)

    original_dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
    )

    with open(
        osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    # Load training dataset to check indices and get normalization statistics
    original_dataset = Neo4jCellDataset(
        root=original_dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=gene_multigraph,
        node_embeddings=node_embeddings,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=graph_processor,
        transform=None,
    )

    print(f"\nTraining dataset size: {len(original_dataset)}")

    # Check each gene's presence in perturbed gene index
    print("\nGene perturbation coverage in dataset:")
    for alias, info in GENE_MAP.items():
        systematic = info["systematic"]
        if systematic in original_dataset.is_any_perturbed_gene_index:
            count = len(original_dataset.is_any_perturbed_gene_index[systematic])
            print(f"  {info['name']} ({systematic}): {count:,} experiments")
        else:
            print(f"  {info['name']} ({systematic}): ✗ NOT in perturbed gene index (0 experiments)")

    # Check each triple using set intersection
    print("\nTriple knockout dataset coverage:")
    triples_needing_inference = []

    for strain_name, aliases, exp_data in TRIPLE_KNOCKOUTS:
        exists, indices, gene_indices = check_triple_exists_in_dataset(original_dataset, aliases)

        genes = [GENE_MAP[a]['systematic'] for a in aliases]
        gene_names = [GENE_MAP[a]['name'] for a in aliases]

        # Show the set intersection logic
        set_sizes = [len(gene_indices[a]) for a in aliases]
        set_notation = " ∩ ".join([f"|{a}|={s}" for a, s in zip(aliases, set_sizes)])

        if exists:
            print(f"\n  {strain_name} ({', '.join(gene_names)}):")
            print(f"    Set intersection: {set_notation} → |intersection|={len(indices)}")
            print(f"    ✓ EXISTS in dataset at indices: {indices[:5]}{'...' if len(indices) > 5 else ''}")
        else:
            print(f"\n  {strain_name} ({', '.join(gene_names)}):")
            print(f"    Set intersection: {set_notation} → |intersection|=0")
            # Identify which gene(s) are missing
            missing = [a for a in aliases if len(gene_indices[a]) == 0]
            if missing:
                missing_names = [GENE_MAP[a]['name'] for a in missing]
                print(f"    ✗ NOT in dataset (missing: {', '.join(missing_names)})")
            else:
                print(f"    ✗ NOT in dataset (no experiments with all 3 genes perturbed together)")
            triples_needing_inference.append((strain_name, aliases, exp_data))

    print(f"\n→ {len(triples_needing_inference)} triples need model inference")

    # ========================================================================
    # Initialize Transforms (using training dataset statistics)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Initializing Transforms")
    print("=" * 80)

    # Create normalization transform
    norm_config = {"gene_interaction": {"strategy": "standard"}}
    norm_transform = COOLabelNormalizationTransform(original_dataset, norm_config)

    print("Normalization parameters for gene_interaction:")
    for key, value in norm_transform.stats["gene_interaction"].items():
        if isinstance(value, (int, float)) and key != "strategy":
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    forward_transform = Compose([norm_transform])
    inverse_transform = COOInverseCompose([norm_transform])

    # Close original dataset LMDB
    original_dataset.close_lmdb()

    # ========================================================================
    # Create Inference Dataset for All Triples
    # ========================================================================
    print("\n" + "=" * 80)
    print("Creating Inference Dataset")
    print("=" * 80)

    # Create a directory for the inference
    inference_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/adh_ald_triple_inference"
    )
    os.makedirs(osp.join(inference_root, "processed"), exist_ok=True)

    # Remove existing LMDB if present (to ensure fresh data)
    lmdb_path = osp.join(inference_root, "processed", "lmdb")
    if osp.exists(lmdb_path):
        import shutil
        shutil.rmtree(lmdb_path)
        print(f"Removed existing LMDB at {lmdb_path}")

    # Create the InferenceDataset
    inference_dataset = InferenceDataset(
        root=inference_root,
        gene_set=genome.gene_set,
        graphs=graphs_dict,
        node_embeddings=node_embeddings,
        graph_processor=graph_processor,
        transform=forward_transform,
    )

    # Create experiments for all triples (even if they exist in training data, we want predictions)
    experiments = []
    experiment_metadata = []  # Track which experiment corresponds to which strain

    for strain_name, aliases, exp_data in TRIPLE_KNOCKOUTS:
        triple = tuple(GENE_MAP[a]["systematic"] for a in aliases)
        experiment = InferenceDataset.create_experiment_from_triple(
            triple=triple,
            dataset_name=f"ng2012_{strain_name}",
        )
        experiments.append(experiment)
        experiment_metadata.append({
            "strain_name": strain_name,
            "aliases": aliases,
            "triple": triple,
            "exp_data": exp_data,
        })

        gene_names = [GENE_MAP[a]["name"] for a in aliases]
        print(f"  Created experiment: {strain_name} ({', '.join(gene_names)})")

    # Load experiments into LMDB
    inference_dataset.load_experiments_to_lmdb(experiments)
    print(f"\nDataset size: {len(inference_dataset)}")

    # ========================================================================
    # Initialize Model
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Get graph regularization lambda
    graph_reg_lambda = 0.0
    graph_reg_config = model_config.get("graph_regularization", {})
    if isinstance(graph_reg_config, dict):
        lambda_val = graph_reg_config.get("graph_reg_lambda", 0.0)
        graph_reg_lambda = 0.0 if lambda_val is None else float(lambda_val)

    print(f"\nInstantiating CellGraphTransformer ({timestamp()})")
    model = CellGraphTransformer(
        gene_num=model_config["gene_num"],
        hidden_channels=model_config["hidden_channels"],
        num_transformer_layers=model_config["num_transformer_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        cell_graph=inference_dataset.cell_graph,
        graph_regularization_config=model_config["graph_regularization"],
        perturbation_head_config=model_config["perturbation_head"],
        dropout=model_config["dropout"],
        graph_reg_lambda=graph_reg_lambda,
        node_embeddings=node_embeddings,
        learnable_embedding_config=model_config.get("learnable_embedding"),
    ).to(device)

    # Loss function (needed for RegressionTask)
    loss_func = PointDistGraphReg(
        point_estimator={"type": "mse", "lambda": 1.0},
        distribution_loss={
            "type": "wasserstein",
            "lambda": 0.1,
            "wasserstein_blur": 0.05,
            "wasserstein_p": 2,
            "wasserstein_scaling": 0.9,
            "min_samples_for_wasserstein": 512,
        },
        graph_regularization={"lambda": 1.0},
        buffer={"use_buffer": False},
        ddp={"use_ddp_gather": False},
    )

    # ========================================================================
    # Load Checkpoint
    # ========================================================================
    print(f"\nLoading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Optimizer and scheduler config (not used during inference but required by load_from_checkpoint)
    optimizer_config = {"type": "AdamW", "lr": 1e-4, "weight_decay": 1e-8}
    lr_scheduler_config = {
        "type": "CosineAnnealingWarmupRestarts",
        "first_cycle_steps": 30,
        "cycle_mult": 1.0,
        "max_lr": 5e-4,
        "min_lr": 1e-7,
        "warmup_steps": 1,
        "gamma": 0.70,
    }

    task = RegressionTask.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        model=model,
        cell_graph=inference_dataset.cell_graph,
        loss_func=loss_func,
        device=device,
        optimizer_config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config,
        batch_size=len(experiments),  # All at once
        clip_grad_norm=True,
        clip_grad_norm_max_norm=10.0,
        inverse_transform=inverse_transform,
        plot_every_n_epochs=1,
        plot_sample_ceiling=10000,
        plot_edge_recovery_every_n_epochs=10,
        plot_transformer_diagnostics_every_n_epochs=10,
        grad_accumulation_schedule=None,
        execution_mode="inference",
        strict=False,
    )
    print("Successfully loaded model checkpoint")

    # Set to evaluation mode
    task.eval()
    model.eval()

    # ========================================================================
    # Run Inference
    # ========================================================================
    print("\n" + "=" * 80)
    print("Running Inference")
    print("=" * 80)

    # Create dataloader
    dataloader = DataLoader(
        inference_dataset,
        batch_size=len(experiments),
        shuffle=False,
        num_workers=0,
        follow_batch=["perturbation_indices"],
    )

    predictions_normalized = []
    predictions_original = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            predictions, representations = task(batch)

            # Store normalized predictions
            pred_norm = predictions.cpu().squeeze()
            if pred_norm.dim() == 0:
                pred_norm = pred_norm.unsqueeze(0)
            predictions_normalized = pred_norm.tolist()

            # Apply inverse transform to get original scale
            if inverse_transform is not None:
                batch_size = predictions.size(0)
                temp_data = HeteroData()
                temp_data["gene"].phenotype_values = predictions.squeeze()
                temp_data["gene"].phenotype_type_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
                temp_data["gene"].phenotype_sample_indices = torch.arange(batch_size, dtype=torch.long, device=device)
                temp_data["gene"].phenotype_types = ["gene_interaction"]

                inv_data = inverse_transform(temp_data)
                pred_orig = inv_data["gene"].phenotype_values.cpu()
                if pred_orig.dim() == 0:
                    pred_orig = pred_orig.unsqueeze(0)
                predictions_original = pred_orig.tolist()
            else:
                predictions_original = predictions_normalized

    # ========================================================================
    # Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS: ADH/ALD Triple Knockout Predictions")
    print("=" * 80)

    # Create results table
    print("\n┌" + "─" * 78 + "┐")
    print(f"│ {'Strain':<15} │ {'Genes':<25} │ {'ε (predicted)':<12} │ {'Interpretation':<18} │")
    print("├" + "─" * 78 + "┤")

    results = []
    for i, meta in enumerate(experiment_metadata):
        strain = meta["strain_name"]
        gene_names = [GENE_MAP[a]["name"] + "Δ" for a in meta["aliases"]]
        genes_str = " ".join(gene_names)
        pred = predictions_original[i]

        # Interpretation
        if pred < -0.08:
            interp = "Strong NEGATIVE"
        elif pred > 0.08:
            interp = "Strong POSITIVE"
        else:
            interp = "Neutral/Weak"

        print(f"│ {strain:<15} │ {genes_str:<25} │ {pred:>+12.4f} │ {interp:<18} │")

        results.append({
            "strain": strain,
            "genes": genes_str,
            "systematic": meta["triple"],
            "prediction": pred,
            "prediction_normalized": predictions_normalized[i],
            "interpretation": interp,
            "exp_data": meta["exp_data"],
        })

    print("└" + "─" * 78 + "┘")

    print(f"\nNote: Kuzmin threshold for 'strong' genetic interactions: |ε| > 0.08")

    # ========================================================================
    # Comparison with Experimental Phenotypes
    # ========================================================================
    print("\n" + "=" * 80)
    print("Comparison with Experimental Phenotypes (Ng et al. 2012)")
    print("=" * 80)

    print("\n┌" + "─" * 100 + "┐")
    print(f"│ {'Strain':<15} │ {'DCW (g/L)':<10} │ {'Time (hr)':<10} │ {'Acetaldehyde':<12} │ {'2,3-BD Yield':<12} │ {'ε (pred)':<10} │")
    print("├" + "─" * 100 + "┤")

    # Wild-type reference
    print(f"│ {'BY4742 (WT)':<15} │ {WILDTYPE_DATA['dry_cell_weight_g_L']:<10.3f} │ {WILDTYPE_DATA['fermentation_time_hr']:<10} │ {WILDTYPE_DATA['acetaldehyde_g_L']:<12.3f} │ {WILDTYPE_DATA['butanediol_yield']:<12.3f} │ {'N/A':<10} │")
    print("├" + "─" * 100 + "┤")

    for r in results:
        exp = r["exp_data"]
        dcw_ratio = exp["dry_cell_weight_g_L"] / WILDTYPE_DATA["dry_cell_weight_g_L"]
        print(f"│ {r['strain']:<15} │ {exp['dry_cell_weight_g_L']:<10.3f} │ {exp['fermentation_time_hr']:<10} │ {exp['acetaldehyde_g_L']:<12.3f} │ {exp['butanediol_yield']:<12.3f} │ {r['prediction']:>+10.4f} │")

    print("└" + "─" * 100 + "┘")

    print("\nKey observations from Ng et al. 2012:")
    print("  • Toxic acetaldehyde threshold: ~0.3 g/L")
    print("  • B2C-a1a3a5: Highest 2,3-BD yield (0.093) but severe acetaldehyde toxicity (1.316 g/L)")
    print("  • B2C-a1a5a6: Lower acetaldehyde (0.263 g/L) due to ALD6 deletion (acetaldehyde → acetate)")
    print("  • B2C-a1a3a6: Lowest DCW (0.619 g/L) suggesting strong fitness cost")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print("\nPredicted genetic interaction scores (ε):")
    for r in results:
        print(f"  {r['strain']}: ε = {r['prediction']:+.4f} ({r['interpretation']})")

    print("\nInterpretation guide:")
    print("  • ε < -0.08: Synthetic sick/lethal (triple worse than expected)")
    print("  • ε > +0.08: Suppression/rescue (triple better than expected)")
    print("  • |ε| ≤ 0.08: Approximately additive (no strong epistasis)")

    # Clean up
    inference_dataset.close_lmdb()

    print(f"\n✅ Inference complete! ({timestamp()})")

    return results


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    results = main()
