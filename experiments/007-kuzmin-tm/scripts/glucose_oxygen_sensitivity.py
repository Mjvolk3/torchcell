#!/usr/bin/env python3
"""
Glucose and oxygen sensitivity analysis for FBA.
As recommended by Vikas Upadhyay - tests if discrete growth bands are due to media constraints.
"""

import cobra
import pandas as pd
import numpy as np
from cobra.io import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from multiprocessing import Pool, cpu_count

# Global model for multiprocessing
_model = None

def init_worker(model_path):
    """Initialize worker with model."""
    global _model
    _model = load_model(model_path)

def run_fba_condition(params):
    """Run FBA for a specific glucose/oxygen condition."""
    glucose_bound, oxygen_bound, gene_combos = params
    global _model

    results = []

    # Set media conditions
    with _model as model:
        # Set glucose uptake
        glucose_rxn = model.reactions.get_by_id('r_1714')  # D-glucose exchange
        glucose_rxn.lower_bound = -glucose_bound

        # Set oxygen uptake
        oxygen_rxn = model.reactions.get_by_id('r_1992')  # oxygen exchange
        oxygen_rxn.lower_bound = -oxygen_bound

        # Get wild-type growth
        wt_solution = model.optimize()
        wt_growth = wt_solution.objective_value if wt_solution.status == "optimal" else 0.0

        # Test gene deletions
        for genes in gene_combos:
            with model as m:
                # Apply same media conditions
                m.reactions.get_by_id('r_1714').lower_bound = -glucose_bound
                m.reactions.get_by_id('r_1992').lower_bound = -oxygen_bound

                # Knock out genes
                for gene_id in genes:
                    if gene_id in m.genes:
                        gene = m.genes.get_by_id(gene_id)
                        gene.knock_out()

                # Optimize with timeout
                m.solver.configuration.timeout = 60
                solution = m.optimize()

                if solution.status == "optimal":
                    growth = solution.objective_value
                elif solution.status == "time_limit":
                    growth = 0.0  # Timeout = no growth
                else:
                    growth = 0.0

                # Calculate fitness
                fitness = growth / wt_growth if wt_growth > 1e-6 else 0.0

                results.append({
                    'glucose': glucose_bound,
                    'oxygen': oxygen_bound,
                    'genes': '_'.join(genes),
                    'num_genes': len(genes),
                    'wt_growth': wt_growth,
                    'mutant_growth': growth,
                    'fitness': fitness
                })

    return results

def analyze_growth_distributions(results_df):
    """Analyze growth rate distributions to detect discrete bands."""

    analysis = {}

    for condition in results_df[['glucose', 'oxygen']].drop_duplicates().itertuples():
        glucose, oxygen = condition.glucose, condition.oxygen
        subset = results_df[(results_df['glucose'] == glucose) &
                          (results_df['oxygen'] == oxygen)]

        # Get unique fitness values
        fitness_values = subset['fitness'].values
        unique_fitness = np.unique(np.round(fitness_values, 4))

        # Detect bands (values that appear frequently)
        fitness_counts = pd.Series(np.round(fitness_values, 2)).value_counts()
        major_bands = fitness_counts[fitness_counts > len(subset) * 0.01].index.tolist()

        analysis[f"glucose_{glucose}_oxygen_{oxygen}"] = {
            'total_mutants': len(subset),
            'unique_fitness_values': len(unique_fitness),
            'major_bands': sorted(major_bands),
            'band_frequencies': fitness_counts[major_bands].to_dict() if major_bands else {},
            'fitness_range': (fitness_values.min(), fitness_values.max()),
            'fitness_std': fitness_values.std()
        }

    return analysis

def plot_sensitivity_results(results_df, output_dir):
    """Create visualization of sensitivity analysis."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Glucose/Oxygen Sensitivity Analysis - Growth Rate Distributions', fontsize=14)

    # Get unique conditions
    conditions = results_df[['glucose', 'oxygen']].drop_duplicates().values[:6]

    for idx, (glucose, oxygen) in enumerate(conditions):
        ax = axes.flat[idx]

        # Filter data for this condition
        subset = results_df[(results_df['glucose'] == glucose) &
                          (results_df['oxygen'] == oxygen)]

        # Plot histogram
        ax.hist(subset['fitness'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Fitness (mutant/WT)')
        ax.set_ylabel('Count')
        ax.set_title(f'Glucose={glucose}, O₂={oxygen}')
        ax.grid(True, alpha=0.3)

        # Add text with statistics
        unique_vals = len(np.unique(np.round(subset['fitness'], 4)))
        ax.text(0.02, 0.98, f'n={len(subset)}\nunique={unique_vals}',
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"glucose_oxygen_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    # Create heatmap of band positions
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for heatmap
    heatmap_data = []
    labels = []

    for condition in results_df[['glucose', 'oxygen']].drop_duplicates().itertuples():
        glucose, oxygen = condition.glucose, condition.oxygen
        subset = results_df[(results_df['glucose'] == glucose) &
                          (results_df['oxygen'] == oxygen)]

        # Count fitness values in bins
        bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
        hist, _ = np.histogram(subset['fitness'], bins=bins)
        heatmap_data.append(hist)
        labels.append(f'Glu={glucose}, O₂={oxygen}')

    heatmap_data = np.array(heatmap_data)

    # Create heatmap
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(np.arange(20))
    ax.set_xticklabels([f'{i*0.05:.2f}' for i in range(20)], rotation=45)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Fitness bins')
    ax.set_ylabel('Media conditions')
    ax.set_title('Distribution of Fitness Values Across Media Conditions')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20)

    plt.tight_layout()
    heatmap_file = os.path.join(output_dir, f"fitness_bands_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file, heatmap_file

def main():
    print("=" * 70)
    print("Glucose/Oxygen Sensitivity Analysis")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load model
    print("Loading Yeast9 model...")
    model_path = "/home/michaelvolk/Documents/yeast-GEM/model/yeast-GEM.xml"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please ensure yeast-GEM is available")
        return

    model = load_model(model_path)
    print(f"Model loaded: {model.id}")

    # Define test conditions
    glucose_levels = [2, 5, 10, 20]  # mmol/gDW/h
    oxygen_levels = [0, 2, 1000]  # 0=anaerobic, 2=limited, 1000=unlimited

    print(f"\nTest conditions:")
    print(f"  Glucose levels: {glucose_levels} mmol/gDW/h")
    print(f"  Oxygen levels: {oxygen_levels} mmol/gDW/h")

    # Select representative gene combinations to test
    print("\nSelecting test gene combinations...")

    # Load perturbations if available
    results_dir = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth"
    perturbations_file = os.path.join(results_dir, "perturbations_all.json")

    if os.path.exists(perturbations_file):
        with open(perturbations_file, 'r') as f:
            perturbations = json.load(f)

        # Sample gene combinations
        np.random.seed(42)  # For reproducibility
        n_samples = 100  # Test with 100 combinations per condition

        test_combos = []

        # Sample singles
        if perturbations['singles']:
            singles_sample = np.random.choice(perturbations['singles'],
                                            min(30, len(perturbations['singles'])),
                                            replace=False)
            test_combos.extend([[g] for g in singles_sample])

        # Sample doubles
        if perturbations['doubles']:
            doubles_sample = np.random.choice(len(perturbations['doubles']),
                                            min(40, len(perturbations['doubles'])),
                                            replace=False)
            test_combos.extend([perturbations['doubles'][i] for i in doubles_sample])

        # Sample triples
        if perturbations['triples']:
            triples_sample = np.random.choice(len(perturbations['triples']),
                                            min(30, len(perturbations['triples'])),
                                            replace=False)
            test_combos.extend([perturbations['triples'][i] for i in triples_sample])

        print(f"  Testing {len(test_combos)} gene combinations")
        print(f"    Singles: {sum(1 for c in test_combos if len(c) == 1)}")
        print(f"    Doubles: {sum(1 for c in test_combos if len(c) == 2)}")
        print(f"    Triples: {sum(1 for c in test_combos if len(c) == 3)}")
    else:
        # Use default test genes if perturbations not available
        print("  Using default test genes...")
        test_genes = ['YAL001C', 'YAL002W', 'YAL003W', 'YBR001C', 'YBR002C']
        test_combos = [[g] for g in test_genes[:5]]  # Just singles

    # Run sensitivity analysis
    print("\nRunning FBA across conditions...")
    all_results = []

    # Prepare tasks for multiprocessing
    tasks = []
    for glucose in glucose_levels:
        for oxygen in oxygen_levels:
            tasks.append((glucose, oxygen, test_combos))

    # Run with multiprocessing
    n_cpus = min(cpu_count(), 32)
    print(f"Using {n_cpus} CPUs")

    with Pool(n_cpus, initializer=init_worker, initargs=(model_path,)) as pool:
        results_nested = pool.map(run_fba_condition, tasks)

    # Flatten results
    for result_list in results_nested:
        all_results.extend(result_list)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    print(f"\nCompleted {len(results_df)} FBA runs")

    # Analyze distributions
    print("\nAnalyzing growth distributions...")
    distribution_analysis = analyze_growth_distributions(results_df)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for condition, stats in distribution_analysis.items():
        print(f"\n{condition}:")
        print(f"  Unique fitness values: {stats['unique_fitness_values']}")
        print(f"  Major bands: {stats['major_bands']}")
        print(f"  Fitness range: [{stats['fitness_range'][0]:.4f}, {stats['fitness_range'][1]:.4f}]")
        print(f"  Fitness std dev: {stats['fitness_std']:.4f}")

        if stats['band_frequencies']:
            print("  Band frequencies:")
            for band, freq in sorted(stats['band_frequencies'].items()):
                print(f"    {band:.2f}: {freq} mutants ({freq/stats['total_mutants']*100:.1f}%)")

    # Check if bands shift with conditions
    print("\n" + "=" * 70)
    print("BAND SHIFT ANALYSIS")
    print("=" * 70)

    # Compare band positions across glucose levels (fixed oxygen)
    for oxygen in oxygen_levels:
        print(f"\nOxygen = {oxygen} mmol/gDW/h:")
        bands_by_glucose = {}

        for glucose in glucose_levels:
            key = f"glucose_{glucose}_oxygen_{oxygen}"
            if key in distribution_analysis:
                bands_by_glucose[glucose] = distribution_analysis[key]['major_bands']

        # Check if bands shift
        all_bands = set()
        for bands in bands_by_glucose.values():
            all_bands.update(bands)

        if len(all_bands) > 0:
            print(f"  All observed bands: {sorted(all_bands)}")

            # Check consistency
            band_shifts = []
            for g1, g2 in zip(glucose_levels[:-1], glucose_levels[1:]):
                if g1 in bands_by_glucose and g2 in bands_by_glucose:
                    common = set(bands_by_glucose[g1]) & set(bands_by_glucose[g2])
                    different = (set(bands_by_glucose[g1]) | set(bands_by_glucose[g2])) - common
                    if different:
                        band_shifts.append(f"Glucose {g1}→{g2}: bands shifted")

            if band_shifts:
                print("  ⚠ Band positions SHIFT with glucose levels:")
                for shift in band_shifts:
                    print(f"    {shift}")
                print("  → Suggests constraint-driven clustering")
            else:
                print("  ✓ Band positions consistent across glucose levels")
                print("  → Suggests model structure-driven clustering")

    # Save results
    os.makedirs(results_dir, exist_ok=True)

    # Save detailed results
    results_file = os.path.join(results_dir, "glucose_oxygen_sensitivity_results.parquet")
    results_df.to_parquet(results_file)
    print(f"\nDetailed results saved to: {results_file}")

    # Save analysis summary
    summary_file = os.path.join(results_dir, "sensitivity_analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'conditions_tested': {
                'glucose_levels': glucose_levels,
                'oxygen_levels': oxygen_levels
            },
            'num_gene_combinations': len(test_combos),
            'total_fba_runs': len(results_df),
            'distribution_analysis': distribution_analysis
        }, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    # Create visualizations
    print("\nGenerating plots...")
    plot_file, heatmap_file = plot_sensitivity_results(results_df, results_dir)
    print(f"Plots saved to:")
    print(f"  {plot_file}")
    print(f"  {heatmap_file}")

    # Final recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Check if we see discrete bands
    all_unique_counts = [stats['unique_fitness_values'] for stats in distribution_analysis.values()]
    avg_unique = np.mean(all_unique_counts)

    if avg_unique < len(test_combos) * 0.2:  # Less than 20% unique values
        print("\n⚠ DISCRETE GROWTH BANDS DETECTED")
        print("The model produces discrete fitness values rather than continuous distributions.")
        print("\nPossible causes:")
        print("1. Media definition is too restrictive")
        print("2. Model has limited metabolic flexibility")
        print("3. Biomass composition forces specific growth modes")
        print("\nSuggested next steps:")
        print("1. Try more complex media (add amino acids, nucleotides)")
        print("2. Review biomass composition for overly strict requirements")
        print("3. Check for bottleneck reactions that limit metabolic flexibility")
    else:
        print("\n✓ CONTINUOUS GROWTH DISTRIBUTION")
        print("The model produces a reasonable range of fitness values.")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()