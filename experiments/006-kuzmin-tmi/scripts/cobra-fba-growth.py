#!/usr/bin/env python3
"""
COBRA FBA Growth Predictions for Gene Deletions
Computes single, double, and triple gene deletion growth rates using FBA
and calculates genetic interactions.

Usage:
    python cobra-fba-growth.py              # Default: 20 genes
    python cobra-fba-growth.py --n-genes 50 # Analyze 50 genes
    python cobra-fba-growth.py --n-genes all # Analyze ALL genes (warning: 1.56 billion triples!)
    python cobra-fba-growth.py --gene-list my_genes.txt # Use specific gene list
"""

import os
import os.path as osp
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Iterator
import cobra
from cobra.flux_analysis.deletion import single_gene_deletion
from dotenv import load_dotenv
import multiprocessing as mp
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import pyarrow as pa
import pyarrow.parquet as pq


def get_cpu_count() -> int:
    """Get CPU count from SLURM or system."""
    # Try SLURM first
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        num_cpus = int(slurm_cpus)
        print(f"num_cpus: {num_cpus}")
        return num_cpus
    # Fall back to system CPU count

    num_cpus = mp.cpu_count()
    print(f"num_cpus: {num_cpus}")
    return num_cpus


def load_model_and_set_medium(results_dir: str = None) -> cobra.Model:
    """Load yeast GEM model with default medium from Zhang et al. 2024."""
    print("Loading model using YeastGEM class...")

    # Use YeastGEM class which downloads and loads the model properly
    from torchcell.metabolism.yeast_GEM import YeastGEM

    yeast_gem = YeastGEM()
    model = yeast_gem.model

    print(f"Model loaded: {len(model.reactions)} reactions, {len(model.genes)} genes")

    # Use the default Yeast9 medium as validated in Zhang et al. 2024
    # (https://doi.org/10.1038/s44320-024-00060-7)
    medium = model.medium
    print(f"\n=== Default Yeast9 Medium Conditions ===")
    print(f"Number of exchange reactions: {len(medium)}")

    # Create a detailed medium summary
    medium_summary = []
    for rxn_id, flux_bound in sorted(medium.items()):
        rxn = model.reactions.get_by_id(rxn_id)
        # Get metabolites involved
        metabolites = list(rxn.metabolites.keys())
        met_names = [m.name for m in metabolites]
        met_ids = [m.id for m in metabolites]

        medium_summary.append(
            {
                "reaction_id": rxn_id,
                "reaction_name": rxn.name,
                "flux_bound_mmol_gDW_h": flux_bound,
                "metabolite_ids": "; ".join(met_ids),
                "metabolite_names": "; ".join(met_names),
                "reaction_equation": rxn.reaction,
            }
        )

        print(f"  {rxn_id}: {rxn.name}")
        print(f"    Flux bound: {flux_bound} mmol/gDW/h")
        print(f"    Metabolites: {', '.join(met_names)}")
        print(f"    Reaction: {rxn.reaction}")

    # Save medium summary to results directory if provided
    if results_dir:
        import pandas as pd

        medium_df = pd.DataFrame(medium_summary)
        medium_df.to_csv(
            osp.join(results_dir, "yeast9_default_medium.csv"), index=False
        )
        print(
            f"\nMedium summary saved to {osp.join(results_dir, 'yeast9_default_medium.csv')}"
        )

    # Test WT growth
    solution = model.optimize()
    print(f"\nWT growth rate: {solution.objective_value:.4f}")

    return model


def _delete_genes_helper(args):
    """Helper function for parallel processing of gene deletions."""
    gene_combo, model_pickle, method = args
    import pickle

    model = pickle.loads(model_pickle)

    with model as m:
        for gene_id in gene_combo:
            if gene_id in m.genes:
                gene = m.genes.get_by_id(gene_id)
                gene.knock_out()

        if method == "fba":
            solution = m.optimize()
        else:
            raise NotImplementedError(
                f"Method {method} not implemented for triple deletion"
            )

        return {
            "ids": gene_combo,
            "growth": (
                solution.objective_value if solution.status == "optimal" else 0.0
            ),
            "status": solution.status,
        }


def double_gene_deletion_with_progress(
    model: cobra.Model,
    gene_list: List[str],
    output_dir: str,
    wt_growth: float,
    method: str = "fba",
    processes: Optional[int] = None,
    batch_size: int = 18000,
) -> str:
    """
    Perform double gene deletions with progress tracking and iterative saving.

    Returns the path to the output file.
    """
    if processes is None:
        processes = get_cpu_count()

    # Calculate total combinations
    n_genes = len(gene_list)
    total_combinations = n_genes * (n_genes - 1) // 2
    print(f"Total double deletions to perform: {total_combinations:,}")

    # Pickle the model once
    import pickle

    model_pickle = pickle.dumps(model)

    # Determine output format and path
    use_parquet = total_combinations > 10000
    if use_parquet:
        output_file = osp.join(output_dir, "double_deletions.parquet")
        schema = pa.schema(
            [
                ("gene1", pa.string()),
                ("gene2", pa.string()),
                ("growth", pa.float64()),
                ("status", pa.string()),
                ("fitness", pa.float64()),
            ]
        )
        writer = None
    else:
        output_file = osp.join(output_dir, "double_deletions.csv")
        csv_header_written = False

    results_buffer = []
    total_processed = 0

    # Generate combinations in batches
    def batch_generator(gene_list, batch_size):
        """Generate batches of gene pair combinations."""
        batch = []
        for combo in combinations(gene_list, 2):
            batch.append(combo)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # Create a single pool for all batches
    with mp.Pool(processes=processes) as pool:
        # Process batches
        for batch_idx, batch in enumerate(batch_generator(gene_list, batch_size)):
            # Create tasks for this batch
            tasks = [(pair, model_pickle, method) for pair in batch]
            
            # Process batch with progress bar
            for result in tqdm(
                pool.imap_unordered(_delete_genes_helper, tasks, chunksize=100),
                total=len(tasks),
                desc=f"Double deletions batch {batch_idx + 1}",
                position=0,
                leave=True,
            ):
                gene_ids = result["ids"]
                growth = result["growth"]
                results_buffer.append(
                    {
                        "gene1": gene_ids[0],
                        "gene2": gene_ids[1],
                        "growth": growth,
                        "status": result["status"],
                        "fitness": growth / wt_growth,
                    }
                )

            total_processed += len(batch)

            # Save this batch to disk
            batch_df = pd.DataFrame(results_buffer)

            if use_parquet:
                # Save to parquet
                table = pa.Table.from_pandas(batch_df, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_file, schema)
                writer.write_table(table)
            else:
                # Save to CSV
                if not csv_header_written:
                    batch_df.to_csv(output_file, index=False, mode="w")
                    csv_header_written = True
                else:
                    batch_df.to_csv(output_file, index=False, mode="a", header=False)

            # Clear buffer
            results_buffer = []

            print(f"Processed {total_processed:,}/{total_combinations:,} double deletions")

    # Close parquet writer if used
    if use_parquet and writer is not None:
        writer.close()

    print(f"Double deletions saved to {output_file}")

    # Return path for loading later
    return output_file


def triple_gene_deletion_iterative(
    model: cobra.Model,
    gene_list: List[str],
    output_dir: str,
    wt_growth: float,
    method: str = "fba",
    processes: Optional[int] = None,
    batch_size: int = 18000,
) -> str:
    """
    Perform triple gene deletions with iterative saving to parquet.

    Returns the path to the final parquet file.
    """
    if processes is None:
        processes = get_cpu_count()

    print(f"Performing triple gene deletions with {processes} processes...")

    # Calculate total combinations
    n_genes = len(gene_list)
    total_combinations = n_genes * (n_genes - 1) * (n_genes - 2) // 6
    print(f"Total triple deletions to perform: {total_combinations:,}")

    # Pickle the model once for efficiency
    import pickle

    model_pickle = pickle.dumps(model)

    # Create output file path
    output_file = osp.join(output_dir, "triple_deletions.parquet")

    # Initialize parquet writer
    schema = pa.schema(
        [
            ("gene1", pa.string()),
            ("gene2", pa.string()),
            ("gene3", pa.string()),
            ("growth", pa.float64()),
            ("status", pa.string()),
            ("fitness", pa.float64()),
        ]
    )

    writer = None
    results_buffer = []
    total_processed = 0

    # Generate combinations in batches
    def batch_generator(gene_list, batch_size):
        """Generate batches of gene combinations."""
        batch = []
        for combo in combinations(gene_list, 3):
            batch.append(combo)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    try:
        with mp.Pool(processes=processes) as pool:
            # Process in batches
            for batch in tqdm(
                batch_generator(gene_list, batch_size),
                total=total_combinations // batch_size + 1,
                desc="Triple deletion batches",
            ):

                # Prepare arguments for this batch
                args_list = [(combo, model_pickle, method) for combo in batch]

                # Process batch
                batch_results = pool.map(_delete_genes_helper, args_list)

                # Convert results to dataframe format
                for result in batch_results:
                    genes = result["ids"]
                    growth = result["growth"]
                    results_buffer.append(
                        {
                            "gene1": genes[0],
                            "gene2": genes[1],
                            "gene3": genes[2],
                            "growth": growth,
                            "status": result["status"],
                            "fitness": growth / wt_growth,
                        }
                    )

                total_processed += len(batch_results)

                # Save to parquet when buffer is full
                if len(results_buffer) >= batch_size:
                    df_batch = pd.DataFrame(results_buffer)
                    table = pa.Table.from_pandas(df_batch, schema=schema)

                    if writer is None:
                        writer = pq.ParquetWriter(output_file, schema)
                    writer.write_table(table)

                    print(
                        f"  Saved batch: {total_processed:,} / {total_combinations:,} ({100*total_processed/total_combinations:.1f}%)"
                    )
                    results_buffer = []

    finally:
        # Save any remaining results
        if results_buffer:
            df_batch = pd.DataFrame(results_buffer)
            table = pa.Table.from_pandas(df_batch, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_file, schema)
            writer.write_table(table)

        if writer:
            writer.close()

    print(f"Triple deletions saved to {output_file}")
    return output_file


def calculate_genetic_interactions(
    single_df: pd.DataFrame,
    double_df: pd.DataFrame,
    triple_df: pd.DataFrame,
    wt_growth: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate digenic (epsilon) and trigenic (tau) interactions.
    """
    print("Calculating genetic interactions...")

    # Create lookup dictionary for single mutant fitness
    single_fitness = {}
    for _, row in single_df.iterrows():
        # Handle different formats of gene IDs (could be set, tuple, or string)
        gene_ids = row["ids"]
        if isinstance(gene_ids, set):
            gene = list(gene_ids)[0] if len(gene_ids) == 1 else str(gene_ids)
        elif isinstance(gene_ids, (list, tuple)):
            gene = gene_ids[0] if len(gene_ids) == 1 else str(gene_ids)
        else:
            gene = gene_ids
        single_fitness[gene] = row["growth"] / wt_growth if wt_growth > 0 else 0

    # Calculate digenic interactions
    digenic_interactions = []
    for _, row in double_df.iterrows():
        genes = row["ids"]
        # Convert to list if it's a set or tuple
        if isinstance(genes, set):
            genes = list(genes)
        elif isinstance(genes, tuple):
            genes = list(genes)

        if len(genes) >= 2:
            g1, g2 = genes[0], genes[1]
        else:
            continue  # Skip if we don't have 2 genes

        f_i = single_fitness.get(g1, 1.0)
        f_j = single_fitness.get(g2, 1.0)
        f_ij = row["growth"] / wt_growth if wt_growth > 0 else 0

        epsilon_ij = f_ij - (f_i * f_j)

        digenic_interactions.append(
            {
                "gene1": g1,
                "gene2": g2,
                "fitness": f_ij,
                "expected_fitness": f_i * f_j,
                "epsilon": epsilon_ij,
                "growth": row["growth"],
                "status": row["status"],
            }
        )

    # Create lookup for digenic interactions
    digenic_lookup = {}
    for item in digenic_interactions:
        key = tuple(sorted([item["gene1"], item["gene2"]]))
        digenic_lookup[key] = item["epsilon"]

    # Calculate trigenic interactions
    trigenic_interactions = []
    for _, row in triple_df.iterrows():
        genes = row["ids"]
        # Convert to list if it's a set or tuple
        if isinstance(genes, set):
            genes = list(genes)
        elif isinstance(genes, tuple):
            genes = list(genes)

        if len(genes) >= 3:
            g1, g2, g3 = genes[0], genes[1], genes[2]
        else:
            continue  # Skip if we don't have 3 genes

        # Get single mutant fitness values
        f_i = single_fitness.get(g1, 1.0)
        f_j = single_fitness.get(g2, 1.0)
        f_k = single_fitness.get(g3, 1.0)

        # Get digenic interactions
        eps_ij = digenic_lookup.get(tuple(sorted([g1, g2])), 0.0)
        eps_ik = digenic_lookup.get(tuple(sorted([g1, g3])), 0.0)
        eps_jk = digenic_lookup.get(tuple(sorted([g2, g3])), 0.0)

        # Triple mutant fitness
        f_ijk = row["growth"] / wt_growth if wt_growth > 0 else 0

        # Calculate tau
        tau_ijk = (
            f_ijk - (f_i * f_j * f_k) - (eps_ij * f_k) - (eps_ik * f_j) - (eps_jk * f_i)
        )

        trigenic_interactions.append(
            {
                "gene1": g1,
                "gene2": g2,
                "gene3": g3,
                "fitness": f_ijk,
                "tau": tau_ijk,
                "epsilon_12": eps_ij,
                "epsilon_13": eps_ik,
                "epsilon_23": eps_jk,
                "growth": row["growth"],
                "status": row["status"],
            }
        )

    return pd.DataFrame(digenic_interactions), pd.DataFrame(trigenic_interactions)


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="COBRA FBA gene deletion analysis")
    parser.add_argument(
        "--n-genes",
        type=str,
        default="20",
        help='Number of genes to analyze (default: 20, use "all" for all genes)',
    )
    parser.add_argument(
        "--gene-list", type=str, help="Path to file with gene list (one per line)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=18000,
        help="Batch size for iterative saving (default: 18000)",
    )
    args = parser.parse_args()

    load_dotenv()
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
    # Paths
    results_dir = osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/results/cobra-fba-growth")
    os.makedirs(results_dir, exist_ok=True)

    # Load model with default Yeast9 medium (Zhang et al. 2024)
    model = load_model_and_set_medium(results_dir)

    # Get WT growth
    wt_solution = model.optimize()
    wt_growth = wt_solution.objective_value
    print(f"Wild-type growth rate: {wt_growth:.4f}")

    # Gene selection based on arguments
    all_genes = [g.id for g in model.genes]
    print(f"\nTotal genes in model: {len(all_genes)}")

    if args.gene_list:
        # Load genes from file
        with open(args.gene_list, "r") as f:
            requested_genes = [line.strip() for line in f if line.strip()]
        # Filter to only genes that exist in model
        gene_subset = [g for g in requested_genes if g in all_genes]
        print(f"Loaded {len(gene_subset)}/{len(requested_genes)} genes from file")
    elif args.n_genes.lower() == "all":
        # Use all genes
        gene_subset = all_genes
        print(f"Using ALL {len(all_genes)} genes")
    else:
        # Use first n genes
        n_genes = int(args.n_genes)
        gene_subset = all_genes[:n_genes]
        print(f"Using first {n_genes} genes")

    # Calculate combinations
    n_singles = len(gene_subset)
    n_doubles = len(gene_subset) * (len(gene_subset) - 1) // 2
    n_triples = len(gene_subset) * (len(gene_subset) - 1) * (len(gene_subset) - 2) // 6

    print(f"\nAnalysis scope:")
    print(f"  Single deletions: {n_singles:,}")
    print(f"  Double deletions: {n_doubles:,}")
    print(f"  Triple deletions: {n_triples:,}")

    if n_triples > 1000000:
        print(f"  WARNING: {n_triples:,} triple deletions will take significant time!")

    print(f"First 5 genes: {gene_subset[:5]}...")

    # Get CPU count for parallel processing
    n_processes = get_cpu_count()
    print(f"Using {n_processes} processes for parallel computation")

    # Single gene deletions
    print("\n=== Single Gene Deletions ===")
    single_results = single_gene_deletion(
        model, gene_list=gene_subset, method="fba", processes=n_processes
    )
    print(f"Completed {len(single_results)} single deletions")

    # Double gene deletions with iterative saving
    print("\n=== Double Gene Deletions ===")
    double_file = double_gene_deletion_with_progress(
        model,
        gene_list=gene_subset,
        output_dir=results_dir,
        wt_growth=wt_growth,
        method="fba",
        processes=n_processes,
        batch_size=18000,
    )

    # Load double results for analysis (only if needed for summary)
    if double_file.endswith(".parquet"):
        double_results = pd.read_parquet(double_file)
    else:
        double_results = pd.read_csv(double_file)

    print(f"Completed {len(double_results)} double deletions")

    # Triple gene deletions - use iterative saving for large sets
    print("\n=== Triple Gene Deletions ===")
    if n_triples > 100000:
        # Use iterative saving for large datasets
        triple_file = triple_gene_deletion_iterative(
            model,
            gene_list=gene_subset,
            output_dir=results_dir,
            wt_growth=wt_growth,
            method="fba",
            processes=n_processes,
            batch_size=args.batch_size,
        )
        # Load back for interaction calculations (may need sampling for huge datasets)
        print("Loading triple deletions for interaction calculations...")
        triple_results = pd.read_parquet(triple_file)
        # Convert to expected format
        triple_results["ids"] = triple_results.apply(
            lambda row: (row["gene1"], row["gene2"], row["gene3"]), axis=1
        )
    else:
        # Small enough to do in memory
        from itertools import combinations

        gene_triples = list(combinations(gene_subset, 3))
        import pickle

        model_pickle = pickle.dumps(model)
        args_list = [(combo, model_pickle, "fba") for combo in gene_triples]

        with mp.Pool(processes=n_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(_delete_genes_helper, args_list),
                    total=len(gene_triples),
                    desc="Triple deletions",
                )
            )
        triple_results = pd.DataFrame(results)

    print(f"Completed {len(triple_results)} triple deletions")

    # Calculate genetic interactions
    digenic_df, trigenic_df = calculate_genetic_interactions(
        single_results, double_results, triple_results, wt_growth
    )

    # Save results
    print("\n=== Saving Results ===")

    # Add fitness column to single deletion results (double/triple already have it)
    single_results["fitness"] = single_results["growth"] / wt_growth

    # Save raw deletion results with fitness (use parquet for large files)
    if len(single_results) > 10000:
        single_results.to_parquet(
            osp.join(results_dir, "single_deletions.parquet"), index=False
        )
    else:
        single_results.to_csv(
            osp.join(results_dir, "single_deletions.csv"), index=False
        )

    # Double results already saved iteratively by double_gene_deletion_with_progress

    # Triple results already saved iteratively if large (>100000)
    if n_triples <= 100000:
        # Add fitness if not already present
        if "fitness" not in triple_results.columns:
            triple_results["fitness"] = triple_results["growth"] / wt_growth
        if len(triple_results) > 10000:
            triple_results.to_parquet(
                osp.join(results_dir, "triple_deletions.parquet"), index=False
            )
        else:
            triple_results.to_csv(
                osp.join(results_dir, "triple_deletions.csv"), index=False
            )

    # Save WT growth as a separate file
    wt_data = pd.DataFrame(
        {
            "genotype": ["WT"],
            "growth": [wt_growth],
            "fitness": [1.0],
            "description": ["Wild-type with no gene deletions"],
        }
    )
    wt_data.to_csv(osp.join(results_dir, "wt_growth.csv"), index=False)

    # Save interaction results (use parquet for large files)
    if len(digenic_df) > 10000:
        digenic_df.to_parquet(
            osp.join(results_dir, "digenic_interactions.parquet"), index=False
        )
    else:
        digenic_df.to_csv(
            osp.join(results_dir, "digenic_interactions.csv"), index=False
        )

    if len(trigenic_df) > 10000:
        trigenic_df.to_parquet(
            osp.join(results_dir, "trigenic_interactions.parquet"), index=False
        )
    else:
        trigenic_df.to_csv(
            osp.join(results_dir, "trigenic_interactions.csv"), index=False
        )

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_source": "YeastGEM class (downloads from GitHub)",
        "wt_growth": wt_growth,
        "wt_fitness": 1.0,
        "n_genes_tested": len(gene_subset),
        "genes_tested": gene_subset,
        "n_single_deletions": len(single_results),
        "n_double_deletions": len(double_results),
        "n_triple_deletions": len(triple_results),
        "n_processes": n_processes,
    }

    with open(osp.join(results_dir, "fba_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to {results_dir}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"WT growth: {wt_growth:.4f}")
    print(f"Single deletions - Mean growth: {single_results['growth'].mean():.4f}")
    print(f"Double deletions - Mean growth: {double_results['growth'].mean():.4f}")
    print(f"Triple deletions - Mean growth: {triple_results['growth'].mean():.4f}")

    # Check for lethal deletions
    single_lethal = (single_results["growth"] < 0.01).sum()
    double_lethal = (double_results["growth"] < 0.01).sum()
    triple_lethal = (triple_results["growth"] < 0.01).sum()

    print(f"\nLethal deletions (growth < 0.01):")
    print(f"  Single: {single_lethal}/{len(single_results)}")
    print(f"  Double: {double_lethal}/{len(double_results)}")
    print(f"  Triple: {triple_lethal}/{len(triple_results)}")

    print("\n=== Interaction Statistics ===")
    print(f"Mean digenic interaction (epsilon): {digenic_df['epsilon'].mean():.4f}")
    print(f"Mean trigenic interaction (tau): {trigenic_df['tau'].mean():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
