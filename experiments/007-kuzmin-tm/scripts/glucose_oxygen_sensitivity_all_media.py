#!/usr/bin/env python3
"""
Glucose and oxygen sensitivity analysis across all media conditions.
Tests Vikas Upadhyay's recommendation: run sensitivity analysis on uptake limits
to check if discrete fitness bands are constraint-driven or model-intrinsic.

Tests 48 conditions:
- 3 media types: minimal, YNB, YPD
- 4 glucose levels: 2, 5, 10, 20 mmol/gDW/h
- 4 O2 levels: unlimited (1000), 20, 10, 5 mmol/gDW/h
"""

import os
import os.path as osp
import sys
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns
from scipy import stats
from dotenv import load_dotenv
from torchcell.timestamp import timestamp

# Use spawn instead of fork to avoid memory issues with many workers
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add torchcell to path
sys.path.append('/home/michaelvolk/Documents/projects/torchcell')
from torchcell.metabolism.yeast_GEM import YeastGEM

# Add scripts directory for media setup
sys.path.append('/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts')
from setup_media_conditions import setup_ypd_media, setup_ynb_media, setup_minimal_media

# Neo4j dataset imports
from torchcell.data import Neo4jCellDataset, MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

# Use the torchcell style
mplstyle.use('/home/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle')

# Global model for multiprocessing - EXACT PATTERN from targeted_fba_with_media_yeastgem.py
_model = None
_media_type = None
_glucose_rate = None
_oxygen_rate = None


def init_worker(model_pickle, media_type, glucose_rate, oxygen_rate):
    """Initialize worker with pickled model and constraint settings.

    COPIED from targeted_fba_with_media_yeastgem.py with glucose/O2 additions.
    """
    import pickle
    import logging
    import os
    global _model, _media_type, _glucose_rate, _oxygen_rate

    # Suppress COBRA INFO messages to avoid millions of "Compartment e" logs
    logging.getLogger('cobra').setLevel(logging.WARNING)

    print(f"Worker {os.getpid()}: Loading model...")
    _model = pickle.loads(model_pickle)
    _media_type = media_type
    _glucose_rate = glucose_rate
    _oxygen_rate = oxygen_rate
    print(f"Worker {os.getpid()}: Model loaded with {media_type} media, glucose={glucose_rate}, O2={oxygen_rate}")


def _delete_genes_worker(gene_combo):
    """Worker function for gene deletion with specific glucose/O2 constraints.

    COPIED from targeted_fba_with_media_yeastgem.py with glucose/O2 additions.
    """
    global _model, _media_type, _glucose_rate, _oxygen_rate

    with _model as m:
        # Set up media conditions
        if _media_type == 'YPD':
            m, *_ = setup_ypd_media(m, glucose_rate=_glucose_rate)
        elif _media_type == 'YNB':
            m, *_ = setup_ynb_media(m, glucose_rate=_glucose_rate)
        else:  # minimal
            setup_minimal_media(m, glucose_rate=_glucose_rate)

        # Set oxygen constraint (additional perturbation beyond media)
        oxygen_rxn = m.reactions.get_by_id('r_1992')
        oxygen_rxn.lower_bound = -_oxygen_rate

        # Knock out genes
        for gene_id in gene_combo:
            if gene_id in m.genes:
                gene = m.genes.get_by_id(gene_id)
                gene.knock_out()

        # Set solver timeout
        m.solver.configuration.timeout = 60

        # Optimize
        solution = m.optimize()

        # Handle results
        status = solution.status
        if status == 'time_limit':
            growth = 0.0  # Timeout = no growth
        else:
            growth = solution.objective_value if status == "optimal" else 0.0

    return growth


def run_fba_for_condition(perturbations, model, media_type, glucose_rate, oxygen_rate, n_cpus=None):
    """Run FBA for a specific media/glucose/O2 condition.

    COPIED from targeted_fba_with_media_yeastgem.py with glucose/O2 additions.
    """
    import pickle

    if n_cpus is None:
        # Using spawn method, so can safely use all CPUs
        n_cpus = min(cpu_count(), 128)

    print(f"\n{'='*70}")
    print(f"Running Targeted FBA: {media_type}, glucose={glucose_rate}, O2={oxygen_rate}")
    print(f"{'='*70}")

    # Get wild-type growth rate
    print(f"\nCalculating wild-type growth rate...")
    wt_model = model.copy()

    # Set up media for wild-type
    if media_type == 'YPD':
        wt_model, *_ = setup_ypd_media(wt_model, glucose_rate=glucose_rate)
    elif media_type == 'YNB':
        wt_model, *_ = setup_ynb_media(wt_model, glucose_rate=glucose_rate)
    else:
        setup_minimal_media(wt_model, glucose_rate=glucose_rate)

    # Set oxygen constraint
    oxygen_rxn = wt_model.reactions.get_by_id('r_1992')
    oxygen_rxn.lower_bound = -oxygen_rate

    wt_solution = wt_model.optimize()
    wt_growth = wt_solution.objective_value if wt_solution.status == "optimal" else 0.0
    print(f"Wild-type growth rate: {wt_growth:.6f}")

    # Pickle model for multiprocessing
    model_pickle = pickle.dumps(model)

    # Initialize process pool
    print(f"\nInitializing {n_cpus} workers...")

    results = {'singles': [], 'doubles': [], 'triples': []}

    with Pool(n_cpus, initializer=init_worker, initargs=(model_pickle, media_type, glucose_rate, oxygen_rate)) as pool:

        # Process each perturbation type - EXACT PATTERN from working script
        for pert_type in ['singles', 'doubles', 'triples']:
            if pert_type not in perturbations or not perturbations[pert_type]:
                continue

            pert_list = perturbations[pert_type]
            print(f"\n=== Processing {pert_type} ({len(pert_list)} perturbations) ===")

            # Prepare tasks - EXACT PATTERN from working script
            if pert_type == 'singles':
                tasks = [(gene,) for gene in pert_list]
            else:
                tasks = [tuple(genes) for genes in pert_list]

            # Run FBA with progress bar
            timeouts = 0

            for gene_combo, growth in tqdm(
                zip(tasks, pool.map(_delete_genes_worker, tasks, chunksize=100)),
                total=len(tasks),
                desc=f"{pert_type}"
            ):
                if growth == 0.0:
                    timeouts += 1

                fitness = growth / wt_growth if wt_growth > 1e-6 else 0.0

                # Build result with additional metadata
                result = {
                    'genes': gene_combo,
                    'growth': growth,
                    'fitness': fitness,
                    'media': media_type,
                    'glucose': glucose_rate,
                    'oxygen': oxygen_rate,
                    'wt_growth': wt_growth
                }

                # Add gene columns based on perturbation type
                if pert_type == 'singles':
                    result['gene'] = gene_combo[0]
                elif pert_type == 'doubles':
                    result['gene1'] = gene_combo[0]
                    result['gene2'] = gene_combo[1]
                else:  # triples
                    result['gene1'] = gene_combo[0]
                    result['gene2'] = gene_combo[1]
                    result['gene3'] = gene_combo[2]

                results[pert_type].append(result)

            print(f"  Timeouts/no-growth: {timeouts}/{len(tasks)}")

    print(f"\n{'='*70}")
    print(f"FBA analysis complete")
    print(f"{'='*70}")

    return results, wt_growth


def main():
    """Main execution function."""
    import argparse

    # Configuration: Define all conditions to test
    CONDITIONS = {
        'media_types': ['minimal', 'YNB', 'YPD'],
        'glucose_levels': [2, 5, 10, 20],  # mmol/gDW/h
        'oxygen_levels': [1000, 20, 10, 5],  # mmol/gDW/h (1000 = unlimited)
    }

    # Parse command line arguments (index-based to support SLURM array jobs)
    parser = argparse.ArgumentParser(description='Run FBA with specific glucose/O2 conditions')
    parser.add_argument('--media', type=str, required=True,
                        choices=CONDITIONS['media_types'],
                        help='Media type')
    parser.add_argument('--glucose', type=float, required=True,
                        help='Glucose uptake rate (mmol/gDW/h)')
    parser.add_argument('--oxygen', type=float, required=True,
                        help='Oxygen uptake rate (mmol/gDW/h)')
    args = parser.parse_args()

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = osp.join(DATA_ROOT, "data/torchcell/experiments/007-kuzmin-tm")

    # Set up paths
    dataset_root = osp.join(EXPERIMENT_ROOT, "001-small-build")
    results_dir = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth"

    print("=== Glucose/O2 Sensitivity Analysis ===")
    print(f"Condition: {args.media}, glucose={args.glucose}, O2={args.oxygen}")
    print(f"Total conditions in config: {len(CONDITIONS['media_types']) * len(CONDITIONS['glucose_levels']) * len(CONDITIONS['oxygen_levels'])}")

    # Load perturbations from pre-extracted file (matching targeted_fba_with_media_yeastgem.py pattern)
    print("\n=== Loading Perturbations ===")
    perturbations_file = osp.join(results_dir, "unique_perturbations.json")

    if not osp.exists(perturbations_file):
        print(f"Error: Perturbations file not found: {perturbations_file}")
        print("Run extract_perturbations.py first or use run_glucose_oxygen_sweep.py wrapper")
        sys.exit(1)

    with open(perturbations_file, 'r') as f:
        perturbations = json.load(f)

    print(f"Loaded perturbations:")
    print(f"  Singles: {len(perturbations['singles'])}")
    print(f"  Doubles: {len(perturbations['doubles'])}")
    print(f"  Triples: {len(perturbations['triples'])}")

    # Load model
    print("\n=== Loading YeastGEM Model ===")
    yeastgem = YeastGEM()
    model = yeastgem.model

    # Run FBA for this specific condition (matching working script pattern)
    print(f"\n{'='*70}")
    print(f"Running FBA: {args.media}, glucose={args.glucose}, O2={args.oxygen}")
    print(f"{'='*70}")

    # Run FBA
    results, wt_growth = run_fba_for_condition(
        perturbations, model, args.media, args.glucose, args.oxygen
    )

    # Convert to dataframes
    singles_df = pd.DataFrame(results['singles'])
    doubles_df = pd.DataFrame(results['doubles'])
    triples_df = pd.DataFrame(results['triples'])

    # Save individual results (matching working script pattern from targeted_fba_with_media_yeastgem.py)
    singles_df.to_parquet(osp.join(results_dir, f'singles_deletions_{args.media}_glc{args.glucose}_o2{args.oxygen}.parquet'))
    doubles_df.to_parquet(osp.join(results_dir, f'doubles_deletions_{args.media}_glc{args.glucose}_o2{args.oxygen}.parquet'))
    triples_df.to_parquet(osp.join(results_dir, f'triples_deletions_{args.media}_glc{args.glucose}_o2{args.oxygen}.parquet'))

    print("\n=== Condition Complete ===")
    print(f"Singles: {len(singles_df)}")
    print(f"Doubles: {len(doubles_df)}")
    print(f"Triples: {len(triples_df)}")
    print(f"Wild-type growth: {wt_growth:.6f}")
    print("\nNote: Experimental data matching will be done in postprocess_glucose_oxygen_sweep.py")


if __name__ == "__main__":
    main()
