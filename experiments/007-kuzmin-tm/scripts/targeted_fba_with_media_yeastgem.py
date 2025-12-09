#!/usr/bin/env python3
"""
Targeted FBA growth predictions with configurable media conditions.
Uses YeastGEM class to load model properly.
Based on Suthers et al., 2020 approach for YNB/YPD media setup.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
from datetime import datetime
import argparse

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

# Global model for multiprocessing
_model = None
_media_type = None

def init_worker(model_pickle, media_type='minimal'):
    """Initialize worker with pickled model and media setup."""
    import pickle
    global _model, _media_type

    print(f"Worker {os.getpid()}: Loading model...")
    _model = pickle.loads(model_pickle)
    _media_type = media_type
    print(f"Worker {os.getpid()}: Model loaded with {media_type} media")

def _delete_genes_worker(gene_combo):
    """Worker function for gene deletion with media setup and timeout."""
    global _model, _media_type

    with _model as m:
        # Set up media conditions
        if _media_type == 'YPD':
            m, *_ = setup_ypd_media(m, glucose_rate=10.0)
        elif _media_type == 'YNB':
            m, *_ = setup_ynb_media(m, glucose_rate=10.0)
        else:  # minimal
            setup_minimal_media(m, glucose_rate=10.0)

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

def run_targeted_fba(perturbations, output_dir, model, n_cpus=None, media_type='minimal'):
    """Run FBA analysis with specified media conditions using YeastGEM model."""

    import pickle

    if n_cpus is None:
        # Using spawn method, so can safely use all CPUs
        n_cpus = min(cpu_count(), 128)

    print(f"\n{'='*70}")
    print(f"Running Targeted FBA with {media_type} media")
    print(f"{'='*70}")

    # Get wild-type growth rate
    print(f"\nCalculating wild-type growth rate in {media_type} media...")

    # Create a copy for wild-type calculation
    wt_model = model.copy()

    # Set up media for wild-type
    if media_type == 'YPD':
        wt_model, ynb_added, ynb_missing, aa_added, aa_missing = setup_ypd_media(wt_model, glucose_rate=10.0)
        print(f"YPD media: {len(ynb_added)} YNB components, {len(aa_added)} amino acids")
        if aa_missing:
            print(f"Warning: Missing amino acids: {', '.join(aa_missing[:5])}...")
    elif media_type == 'YNB':
        wt_model, ynb_added, ynb_missing = setup_ynb_media(wt_model, glucose_rate=10.0)
        print(f"YNB media: {len(ynb_added)} vitamin/cofactor components")
        if ynb_missing:
            print(f"Warning: Missing YNB components: {', '.join(ynb_missing)}")
    else:
        setup_minimal_media(wt_model, glucose_rate=10.0)
        print("Using minimal media (glucose, NH4, O2, inorganics)")

    wt_solution = wt_model.optimize()
    wt_growth = wt_solution.objective_value if wt_solution.status == "optimal" else 0.0
    print(f"Wild-type growth rate: {wt_growth:.4f}")

    # Pickle model for multiprocessing
    model_pickle = pickle.dumps(model)

    # Initialize process pool
    print(f"\nInitializing {n_cpus} workers with {media_type} media...")

    with Pool(n_cpus, initializer=init_worker, initargs=(model_pickle, media_type)) as pool:

        # Process each perturbation type
        for pert_type in ['singles', 'doubles', 'triples']:
            if pert_type not in perturbations or not perturbations[pert_type]:
                continue

            pert_list = perturbations[pert_type]
            print(f"\n=== Processing {pert_type} ({len(pert_list)} perturbations) ===")

            # Prepare tasks
            if pert_type == 'singles':
                tasks = [(gene,) for gene in pert_list]
            else:
                tasks = [tuple(genes) for genes in pert_list]

            # Run FBA with progress bar
            results = []
            timeouts = 0

            for gene_combo, growth in tqdm(
                zip(tasks, pool.map(_delete_genes_worker, tasks, chunksize=100)),
                total=len(tasks),
                desc=f"{pert_type}"
            ):
                if growth == 0.0:
                    timeouts += 1

                fitness = growth / wt_growth if wt_growth > 1e-6 else 0.0

                if pert_type == 'singles':
                    results.append({
                        'gene': gene_combo[0],
                        'growth': growth,
                        'fitness': fitness
                    })
                elif pert_type == 'doubles':
                    results.append({
                        'gene1': gene_combo[0],
                        'gene2': gene_combo[1],
                        'growth': growth,
                        'fitness': fitness
                    })
                else:  # triples
                    results.append({
                        'gene1': gene_combo[0],
                        'gene2': gene_combo[1],
                        'gene3': gene_combo[2],
                        'growth': growth,
                        'fitness': fitness
                    })

            # Save results
            df = pd.DataFrame(results)
            output_file = os.path.join(output_dir, f"{pert_type}_deletions_{media_type}.parquet")
            df.to_parquet(output_file)
            print(f"  Saved: {output_file}")
            print(f"  Timeouts/no-growth: {timeouts}/{len(tasks)}")

            # Show fitness distribution
            if len(df) > 0:
                unique_fitness = len(df['fitness'].round(3).unique())
                print(f"  Unique fitness values: {unique_fitness}")

                # Check for discrete bands
                fitness_counts = df['fitness'].round(2).value_counts()
                major_bands = fitness_counts[fitness_counts > len(df) * 0.05].index.tolist()
                if major_bands:
                    print(f"  Major fitness bands (>5%): {sorted(major_bands)}")

    print(f"\n{'='*70}")
    print(f"FBA analysis complete with {media_type} media")
    print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(
        description='Run targeted FBA with configurable media conditions using YeastGEM'
    )
    parser.add_argument(
        '--media',
        type=str,
        default='minimal',
        choices=['minimal', 'YNB', 'YPD'],
        help='Media type: minimal, YNB (vitamins), or YPD (full rich media)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run with small test dataset'
    )
    args = parser.parse_args()

    # Configuration
    BASE_DIR = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm"
    RESULTS_DIR = f"{BASE_DIR}/results/cobra-fba-growth"

    # Load model using YeastGEM class
    print("Loading Yeast9 model using YeastGEM class...")
    yeast_gem = YeastGEM()
    model = yeast_gem.model
    print(f"Model loaded: {model.id}")
    print(f"Reactions: {len(model.reactions)}")
    print(f"Genes: {len(model.genes)}")

    # Load perturbations
    perturbations_file = f"{RESULTS_DIR}/unique_perturbations.json"

    if not os.path.exists(perturbations_file):
        print(f"Error: Perturbations file not found: {perturbations_file}")
        print("Run extract_perturbations.py first")
        return

    with open(perturbations_file, 'r') as f:
        all_perturbations = json.load(f)

    print(f"\nLoaded perturbations:")
    for key, value in all_perturbations.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)}")

    # If test mode, use subset
    if args.test:
        print("\n*** TEST MODE - Using subset of data ***")
        test_perturbations = {}

        if all_perturbations.get('singles'):
            test_perturbations['singles'] = all_perturbations['singles'][:100]

        if all_perturbations.get('doubles'):
            test_perturbations['doubles'] = all_perturbations['doubles'][:200]

        if all_perturbations.get('triples'):
            test_perturbations['triples'] = all_perturbations['triples'][:100]

        all_perturbations = test_perturbations
        print(f"Using test subset: {sum(len(v) for v in all_perturbations.values())} total")

    # Run FBA analysis with specified media
    run_targeted_fba(all_perturbations, RESULTS_DIR, model, media_type=args.media)

if __name__ == "__main__":
    main()