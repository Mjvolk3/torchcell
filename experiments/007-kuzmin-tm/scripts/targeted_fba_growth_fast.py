"""
Fast targeted FBA growth predictions using efficient multiprocessing.
Uses pickle for fast serialization and optimized I/O.
"""

import os
import os.path as osp
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import cobra
from dotenv import load_dotenv
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
import gc


def get_cpu_count() -> int:
    """Get CPU count from SLURM or system."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        num_cpus = int(slurm_cpus)
        print(f"Using SLURM CPUs: {num_cpus}")
        return num_cpus
    num_cpus = mp.cpu_count()
    print(f"Using system CPUs: {num_cpus}")
    return num_cpus


def load_model_and_set_medium(results_dir: str = None) -> cobra.Model:
    """Load yeast GEM model with default medium from Zhang et al. 2024."""
    print("Loading model using YeastGEM class...")
    
    from torchcell.metabolism.yeast_GEM import YeastGEM
    
    yeast_gem = YeastGEM()
    model = yeast_gem.model
    
    print(f"Model loaded: {len(model.reactions)} reactions, {len(model.genes)} genes")
    
    # Use default Yeast9 medium
    medium = model.medium
    print(f"\n=== Default Yeast9 Medium Conditions ===")
    print(f"Number of exchange reactions: {len(medium)}")
    
    # Save medium summary if results_dir provided
    if results_dir:
        medium_summary = []
        for rxn_id, flux_bound in sorted(medium.items())[:5]:  # Just show first 5
            rxn = model.reactions.get_by_id(rxn_id)
            print(f"  {rxn_id}: {rxn.name} (bound: {flux_bound})")
    
    # Test WT growth
    solution = model.optimize()
    print(f"\nWT growth rate: {solution.objective_value:.4f}")
    
    return model


def _init_worker(model_pickle):
    """Initialize worker process with model."""
    global _model
    _model = pickle.loads(model_pickle)


def _delete_genes_worker(gene_combo):
    """Worker function for gene deletion with solver timeout."""
    global _model
    
    with _model as m:
        for gene_id in gene_combo:
            if gene_id in m.genes:
                gene = m.genes.get_by_id(gene_id)
                gene.knock_out()
        
        # Set solver timeout to 60 seconds to prevent hanging
        m.solver.configuration.timeout = 60
        
        solution = m.optimize()
        
        # Handle timeout status
        status = solution.status
        if status == 'time_limit':
            # Solver timed out - treat as no growth
            growth = 0.0
        else:
            growth = solution.objective_value if status == "optimal" else 0.0
        
        return {
            "ids": gene_combo,
            "growth": growth,
            "status": status,
        }


def perform_targeted_fba_fast(
    model: cobra.Model,
    perturbations: Dict[str, List],
    output_dir: str,
    wt_growth: float,
    processes: Optional[int] = None,
    write_buffer_size: int = 10000,
) -> Dict[str, str]:
    """
    Perform FBA with fast multiprocessing and efficient I/O.
    """
    if processes is None:
        processes = get_cpu_count()
    
    # Pickle model once
    model_pickle = pickle.dumps(model)
    output_files = {}
    
    # Process each perturbation type
    for pert_type, pert_list in perturbations.items():
        if not pert_list:
            continue
        
        print(f"\n=== Processing {pert_type} ===")
        print(f"Total perturbations: {len(pert_list)}")
        
        # Prepare output file
        output_file = osp.join(output_dir, f"{pert_type}_deletions.parquet")
        
        # Define schema based on perturbation type
        if pert_type == "singles":
            schema = pa.schema([
                ("gene", pa.string()),
                ("growth", pa.float64()),
                ("status", pa.string()),
                ("fitness", pa.float64()),
            ])
        elif pert_type == "doubles":
            schema = pa.schema([
                ("gene1", pa.string()),
                ("gene2", pa.string()),
                ("growth", pa.float64()),
                ("status", pa.string()),
                ("fitness", pa.float64()),
            ])
        elif pert_type == "triples":
            schema = pa.schema([
                ("gene1", pa.string()),
                ("gene2", pa.string()),
                ("gene3", pa.string()),
                ("growth", pa.float64()),
                ("status", pa.string()),
                ("fitness", pa.float64()),
            ])
        
        # Convert perturbations to tuples
        if pert_type == "singles":
            tasks = [(gene,) for gene in pert_list]
        else:
            tasks = [tuple(genes) for genes in pert_list]
        
        # Process with multiprocessing
        results_buffer = []
        writer = None
        timeout_count = 0
        processed_count = 0
        
        # Calculate optimal chunksize
        chunksize = max(1, min(100, len(tasks) // (processes * 10)))
        
        # Create pool with initializer
        with mp.Pool(
            processes=processes,
            initializer=_init_worker,
            initargs=(model_pickle,)
        ) as pool:
            # Process all tasks with progress bar
            for result in tqdm(
                pool.imap_unordered(_delete_genes_worker, tasks, chunksize=chunksize),
                total=len(tasks),
                desc=f"Processing {pert_type}",
                smoothing=0.1
            ):
                processed_count += 1
                
                # Track timeouts
                if result["status"] == "time_limit":
                    timeout_count += 1
                    if timeout_count % 100 == 0:
                        print(f"  Warning: {timeout_count} timeouts so far")
                
                genes = result["ids"]
                growth = result["growth"]
                fitness = growth / wt_growth
                
                if pert_type == "singles":
                    results_buffer.append({
                        "gene": genes[0],
                        "growth": growth,
                        "status": result["status"],
                        "fitness": fitness,
                    })
                elif pert_type == "doubles":
                    results_buffer.append({
                        "gene1": genes[0],
                        "gene2": genes[1],
                        "growth": growth,
                        "status": result["status"],
                        "fitness": fitness,
                    })
                elif pert_type == "triples":
                    results_buffer.append({
                        "gene1": genes[0],
                        "gene2": genes[1],
                        "gene3": genes[2],
                        "growth": growth,
                        "status": result["status"],
                        "fitness": fitness,
                    })
                
                # Write buffer to disk when full
                if len(results_buffer) >= write_buffer_size:
                    df_batch = pd.DataFrame(results_buffer)
                    table = pa.Table.from_pandas(df_batch, schema=schema)
                    
                    if writer is None:
                        writer = pq.ParquetWriter(output_file, schema)
                    writer.write_table(table)
                    
                    results_buffer = []
        
        # Write remaining results
        if results_buffer:
            df_batch = pd.DataFrame(results_buffer)
            table = pa.Table.from_pandas(df_batch, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_file, schema)
            writer.write_table(table)
        
        if writer:
            writer.close()
        
        output_files[pert_type] = output_file
        print(f"Saved {pert_type} results to {output_file}")
        
        # Report timeout statistics
        if timeout_count > 0:
            print(f"  Total timeouts: {timeout_count}/{processed_count} ({100*timeout_count/processed_count:.1f}%)")
        else:
            print(f"  All {processed_count} completed without timeouts")
        
        # Force garbage collection
        gc.collect()
    
    return output_files


def calculate_genetic_interactions(
    single_df: pd.DataFrame,
    double_df: pd.DataFrame,
    triple_df: pd.DataFrame,
    wt_growth: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate digenic and trigenic interactions.
    
    Formulas:
    - Digenic: ε_ij = f_ij - f_i * f_j
    - Trigenic: τ_ijk = f_ijk - f_i * f_j * f_k - ε_ij * f_k - ε_ik * f_j - ε_jk * f_i
    """
    print("\n=== Calculating Genetic Interactions ===")
    
    # Create lookup dictionaries for fast access
    single_fitness = {row['gene']: row['fitness'] for _, row in single_df.iterrows()}
    double_fitness = {
        (row['gene1'], row['gene2']): row['fitness'] 
        for _, row in double_df.iterrows()
    }
    
    # Calculate digenic interactions
    digenic_interactions = []
    for _, row in tqdm(double_df.iterrows(), total=len(double_df), desc="Calculating digenic"):
        gene1, gene2 = row['gene1'], row['gene2']
        f_ij = row['fitness']
        f_i = single_fitness.get(gene1, 1.0)
        f_j = single_fitness.get(gene2, 1.0)
        
        epsilon_ij = f_ij - (f_i * f_j)
        
        digenic_interactions.append({
            'gene1': gene1,
            'gene2': gene2,
            'fitness_observed': f_ij,
            'fitness_expected': f_i * f_j,
            'epsilon': epsilon_ij,
            'fitness_gene1': f_i,
            'fitness_gene2': f_j,
        })
    
    # Calculate trigenic interactions with progress bar
    trigenic_interactions = []
    for _, row in tqdm(triple_df.iterrows(), total=len(triple_df), desc="Calculating trigenic"):
        gene1, gene2, gene3 = row['gene1'], row['gene2'], row['gene3']
        f_ijk = row['fitness']
        
        # Get single fitnesses
        f_i = single_fitness.get(gene1, 1.0)
        f_j = single_fitness.get(gene2, 1.0)
        f_k = single_fitness.get(gene3, 1.0)
        
        # Get pairwise fitnesses and calculate epsilons
        f_ij = double_fitness.get((gene1, gene2), f_i * f_j)
        f_ik = double_fitness.get((gene1, gene3), f_i * f_k)
        f_jk = double_fitness.get((gene2, gene3), f_j * f_k)
        
        epsilon_ij = f_ij - (f_i * f_j)
        epsilon_ik = f_ik - (f_i * f_k)
        epsilon_jk = f_jk - (f_j * f_k)
        
        # Calculate trigenic interaction
        tau_ijk = f_ijk - (f_i * f_j * f_k) - (epsilon_ij * f_k) - (epsilon_ik * f_j) - (epsilon_jk * f_i)
        
        trigenic_interactions.append({
            'gene1': gene1,
            'gene2': gene2,
            'gene3': gene3,
            'fitness_observed': f_ijk,
            'fitness_expected_no_interaction': f_i * f_j * f_k,
            'tau': tau_ijk,
            'epsilon_12': epsilon_ij,
            'epsilon_13': epsilon_ik,
            'epsilon_23': epsilon_jk,
            'fitness_gene1': f_i,
            'fitness_gene2': f_j,
            'fitness_gene3': f_k,
        })
    
    digenic_df = pd.DataFrame(digenic_interactions)
    trigenic_df = pd.DataFrame(trigenic_interactions)
    
    print(f"Calculated {len(digenic_df)} digenic interactions")
    print(f"Calculated {len(trigenic_df)} trigenic interactions")
    
    return digenic_df, trigenic_df


def main():
    """Main execution function."""
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
    
    # Paths
    results_dir = osp.join(EXPERIMENT_ROOT, "007-kuzmin-tm/results/cobra-fba-growth")
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if FBA analysis has already been completed
    required_outputs = [
        "singles_deletions.parquet",
        "doubles_deletions.parquet", 
        "triples_deletions.parquet",
        "digenic_interactions.parquet",
        "trigenic_interactions.parquet",
        "fba_metadata.json"
    ]
    
    all_exist = all(osp.exists(osp.join(results_dir, f)) for f in required_outputs)
    
    if all_exist:
        print("FBA analysis already completed. Loading existing results...")
        with open(osp.join(results_dir, "fba_metadata.json"), "r") as f:
            metadata = json.load(f)
        print(f"Previous run completed at: {metadata['timestamp']}")
        print(f"WT growth: {metadata['wt_growth']:.4f}")
        print(f"Singles: {metadata['n_singles']}")
        print(f"Doubles: {metadata['n_doubles']}")
        print(f"Triples: {metadata['n_triples']}")
        print(f"Runtime: {metadata['runtime_seconds']:.1f} seconds")
        print("\nSkipping FBA analysis. Delete output files to rerun.")
        return
    
    # Load perturbations
    perturbations_file = osp.join(results_dir, "unique_perturbations.json")
    if not osp.exists(perturbations_file):
        raise FileNotFoundError(
            f"Perturbations file not found: {perturbations_file}\n"
            "Please run extract_perturbations.py first."
        )
    
    with open(perturbations_file, "r") as f:
        perturbations = json.load(f)
    
    print("=== Loaded Perturbations ===")
    print(f"Singles: {len(perturbations['singles'])}")
    print(f"Doubles: {len(perturbations['doubles'])}")
    print(f"Triples: {len(perturbations['triples'])}")
    
    # Load model
    model = load_model_and_set_medium(results_dir)
    
    # Get WT growth
    wt_solution = model.optimize()
    wt_growth = wt_solution.objective_value
    print(f"Wild-type growth rate: {wt_growth:.4f}")
    
    # Save WT growth
    wt_data = pd.DataFrame({
        "genotype": ["WT"],
        "growth": [wt_growth],
        "fitness": [1.0],
        "description": ["Wild-type with no gene deletions"],
    })
    wt_data.to_csv(osp.join(results_dir, "wt_growth.csv"), index=False)
    
    # Get CPU count
    n_processes = get_cpu_count()
    
    # Perform targeted FBA
    print(f"\n=== Starting Fast FBA Analysis ===")
    print(f"Using {n_processes} processes")
    
    start_time = datetime.now()
    
    output_files = perform_targeted_fba_fast(
        model=model,
        perturbations=perturbations,
        output_dir=results_dir,
        wt_growth=wt_growth,
        processes=n_processes,
        write_buffer_size=10000,
    )
    
    # Load results for interaction calculations
    print("\n=== Loading results for interaction calculations ===")
    single_df = pd.read_parquet(output_files.get("singles"))
    double_df = pd.read_parquet(output_files.get("doubles"))
    triple_df = pd.read_parquet(output_files.get("triples"))
    
    # Calculate genetic interactions
    digenic_df, trigenic_df = calculate_genetic_interactions(
        single_df, double_df, triple_df, wt_growth
    )
    
    # Save interaction results
    digenic_df.to_parquet(osp.join(results_dir, "digenic_interactions.parquet"), index=False)
    trigenic_df.to_parquet(osp.join(results_dir, "trigenic_interactions.parquet"), index=False)
    
    # Save smaller CSV versions for inspection if small enough
    if len(digenic_df) < 100000:
        digenic_df.to_csv(osp.join(results_dir, "digenic_interactions.csv"), index=False)
    if len(trigenic_df) < 100000:
        trigenic_df.to_csv(osp.join(results_dir, "trigenic_interactions.csv"), index=False)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"WT growth: {wt_growth:.4f}")
    print(f"Single deletions - Mean fitness: {single_df['fitness'].mean():.4f}")
    print(f"Double deletions - Mean fitness: {double_df['fitness'].mean():.4f}")
    print(f"Triple deletions - Mean fitness: {triple_df['fitness'].mean():.4f}")
    
    # Lethal deletions (fitness < 0.01)
    single_lethal = (single_df["fitness"] < 0.01).sum()
    double_lethal = (double_df["fitness"] < 0.01).sum()
    triple_lethal = (triple_df["fitness"] < 0.01).sum()
    
    print(f"\nLethal deletions (fitness < 0.01):")
    print(f"  Single: {single_lethal}/{len(single_df)}")
    print(f"  Double: {double_lethal}/{len(double_df)}")
    print(f"  Triple: {triple_lethal}/{len(triple_df)}")
    
    # Interaction statistics
    print(f"\nInteraction statistics:")
    print(f"  Digenic - Mean ε: {digenic_df['epsilon'].mean():.4f}")
    print(f"  Digenic - Std ε: {digenic_df['epsilon'].std():.4f}")
    print(f"  Trigenic - Mean τ: {trigenic_df['tau'].mean():.4f}")
    print(f"  Trigenic - Std τ: {trigenic_df['tau'].std():.4f}")
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "wt_growth": wt_growth,
        "n_singles": len(single_df),
        "n_doubles": len(double_df),
        "n_triples": len(triple_df),
        "n_processes": n_processes,
        "runtime_seconds": (datetime.now() - start_time).total_seconds(),
    }
    
    with open(osp.join(results_dir, "fba_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTotal runtime: {datetime.now() - start_time}")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()