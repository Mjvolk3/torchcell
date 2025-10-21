#!/usr/bin/env python3
"""
Test FBA with YNB and YPD media conditions following Suthers et al., 2020.
Based on: "The undefined composition of yeast extract in Yeast-Peptone-Dextrose (YPD)
media was assumed to be that of YNB media plus 20 amino acids and d-glucose."
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

def setup_ynb_media(model, glucose_rate=10.0):
    """
    Set up YNB media following Suthers et al., 2020.
    YNB includes: thiamine, riboflavin, nicotinate, pyridoxin,
    folate, (R)-pantothenate, 4-aminobenzoate, and myo-inositol
    """

    # Reset all exchange reactions first
    for rxn in model.exchanges:
        rxn.lower_bound = 0

    # Basic nutrients (always available)
    # Glucose
    if 'r_1714' in model.reactions:
        model.reactions.r_1714.lower_bound = -glucose_rate  # D-glucose exchange

    # Oxygen
    if 'r_1992' in model.reactions:
        model.reactions.r_1992.lower_bound = -1000  # Unlimited oxygen

    # Ammonium
    if 'r_1654' in model.reactions:
        model.reactions.r_1654.lower_bound = -1000  # NH4+

    # Inorganic nutrients
    if 'r_2100' in model.reactions:
        model.reactions.r_2100.lower_bound = -1000  # H2O
    if 'r_2005' in model.reactions:
        model.reactions.r_2005.lower_bound = -1000  # phosphate
    if 'r_2060' in model.reactions:
        model.reactions.r_2060.lower_bound = -1000  # sulfate
    if 'r_1861' in model.reactions:
        model.reactions.r_1861.lower_bound = -1000  # H+

    # YNB components (supplementary rate = 5% of glucose)
    supplement_rate = glucose_rate * 0.05  # 0.165 mmol/gDW/h when glucose=3.3

    ynb_components = {
        'r_2067': 'thiamine',  # thiamine exchange
        'r_2043': 'riboflavin',  # riboflavin exchange
        'r_1975': 'nicotinate',  # nicotinate exchange
        'r_2029': 'pyridoxine',  # pyridoxine exchange
        'r_1818': 'folate',  # folate exchange
        'r_1996': 'pantothenate',  # (R)-pantothenate exchange
        'r_2101': '4-aminobenzoate',  # 4-aminobenzoate exchange
        'r_1972': 'myo-inositol'  # myo-inositol exchange
    }

    added_nutrients = []
    missing_nutrients = []

    for rxn_id, name in ynb_components.items():
        if rxn_id in model.reactions:
            model.reactions.get_by_id(rxn_id).lower_bound = -supplement_rate
            added_nutrients.append(name)
        else:
            # Try to find alternative reaction IDs
            found = False
            for rxn in model.exchanges:
                if name.lower() in rxn.name.lower() or name.lower() in rxn.id.lower():
                    rxn.lower_bound = -supplement_rate
                    added_nutrients.append(f"{name} (via {rxn.id})")
                    found = True
                    break
            if not found:
                missing_nutrients.append(name)

    return added_nutrients, missing_nutrients

def setup_ypd_media(model, glucose_rate=10.0):
    """
    Set up YPD media following Suthers et al., 2020.
    YPD = YNB + 20 amino acids + d-glucose
    """

    # First set up YNB
    added_ynb, missing_ynb = setup_ynb_media(model, glucose_rate)

    # Add 20 amino acids (supplementary rate = 5% of glucose)
    supplement_rate = glucose_rate * 0.05

    # Standard 20 amino acids
    amino_acids = {
        'r_1654': 'L-alanine',
        'r_1658': 'L-arginine',
        'r_1660': 'L-asparagine',
        'r_1661': 'L-aspartate',
        'r_1672': 'L-cysteine',
        'r_1810': 'L-glutamate',
        'r_1812': 'L-glutamine',
        'r_1813': 'glycine',
        'r_1873': 'L-histidine',
        'r_1893': 'L-isoleucine',
        'r_1902': 'L-leucine',
        'r_1903': 'L-lysine',
        'r_1912': 'L-methionine',
        'r_2002': 'L-phenylalanine',
        'r_2014': 'L-proline',
        'r_2056': 'L-serine',
        'r_2068': 'L-threonine',
        'r_2084': 'L-tryptophan',
        'r_2088': 'L-tyrosine',
        'r_2090': 'L-valine'
    }

    added_aa = []
    missing_aa = []

    for rxn_id, name in amino_acids.items():
        if rxn_id in model.reactions:
            model.reactions.get_by_id(rxn_id).lower_bound = -supplement_rate
            added_aa.append(name)
        else:
            # Try to find by searching exchange reactions
            found = False
            for rxn in model.exchanges:
                # Match amino acid names in reaction name or ID
                aa_simple = name.replace('L-', '').lower()
                if aa_simple in rxn.name.lower() or aa_simple in rxn.id.lower():
                    if 'exchange' in rxn.name.lower() or rxn.id.startswith('r_'):
                        rxn.lower_bound = -supplement_rate
                        added_aa.append(f"{name} (via {rxn.id})")
                        found = True
                        break
            if not found:
                missing_aa.append(name)

    return added_ynb, missing_ynb, added_aa, missing_aa

def test_gene_deletions(model, gene_combos, media_type='YPD'):
    """Test gene deletions under specified media conditions."""

    results = []

    # Get wild-type growth
    wt_solution = model.optimize()
    wt_growth = wt_solution.objective_value if wt_solution.status == "optimal" else 0.0

    print(f"Wild-type growth rate: {wt_growth:.4f}")

    # Test each gene combination
    for i, genes in enumerate(gene_combos):
        if i % 100 == 0:
            print(f"  Testing {i}/{len(gene_combos)} combinations...")

        with model as m:
            # Reapply media conditions (important for context manager)
            if media_type == 'YPD':
                setup_ypd_media(m)
            else:
                setup_ynb_media(m)

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
            else:
                growth = 0.0

            fitness = growth / wt_growth if wt_growth > 1e-6 else 0.0

            results.append({
                'genes': '_'.join(genes),
                'num_genes': len(genes),
                'growth': growth,
                'fitness': fitness,
                'media': media_type
            })

    return pd.DataFrame(results), wt_growth

def analyze_fitness_distribution(df, title="Fitness Distribution"):
    """Analyze and plot fitness distribution."""

    # Get unique fitness values
    unique_fitness = len(df['fitness'].unique())

    # Find major bands (values appearing frequently)
    fitness_rounded = df['fitness'].round(2)
    value_counts = fitness_rounded.value_counts()
    major_bands = value_counts[value_counts > len(df) * 0.01].index.tolist()

    print(f"\n{title}:")
    print(f"  Total mutants: {len(df)}")
    print(f"  Unique fitness values: {unique_fitness}")
    print(f"  Major fitness bands: {sorted(major_bands)}")

    # Calculate band frequencies
    if major_bands:
        print("  Band frequencies:")
        for band in sorted(major_bands):
            count = value_counts[band]
            print(f"    {band:.2f}: {count} mutants ({count/len(df)*100:.1f}%)")

    return major_bands

def plot_comparison(ynb_df, ypd_df, output_dir):
    """Create comparison plots for YNB vs YPD media."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Histograms
    ax = axes[0]
    ax.hist(ynb_df['fitness'], bins=50, alpha=0.5, label='YNB', color='blue', edgecolor='black')
    ax.hist(ypd_df['fitness'], bins=50, alpha=0.5, label='YPD', color='orange', edgecolor='black')
    ax.set_xlabel('Fitness (mutant/WT)')
    ax.set_ylabel('Count')
    ax.set_title('Fitness Distributions: YNB vs YPD')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Scatter plot comparing same mutants
    ax = axes[1]
    # Merge on gene combinations
    merged = ynb_df.merge(ypd_df, on='genes', suffixes=('_ynb', '_ypd'))
    ax.scatter(merged['fitness_ynb'], merged['fitness_ypd'], alpha=0.5, s=10)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # y=x line
    ax.set_xlabel('YNB Fitness')
    ax.set_ylabel('YPD Fitness')
    ax.set_title('Fitness Correlation: YNB vs YPD')
    ax.grid(True, alpha=0.3)

    # Add correlation coefficient
    correlation = merged['fitness_ynb'].corr(merged['fitness_ypd'])
    ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Box plots by deletion order
    ax = axes[2]
    data_to_plot = []
    labels = []
    for num_genes in [1, 2, 3]:
        if num_genes in ynb_df['num_genes'].values:
            data_to_plot.append(ynb_df[ynb_df['num_genes'] == num_genes]['fitness'])
            labels.append(f'YNB-{num_genes}')
        if num_genes in ypd_df['num_genes'].values:
            data_to_plot.append(ypd_df[ypd_df['num_genes'] == num_genes]['fitness'])
            labels.append(f'YPD-{num_genes}')

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness by Deletion Order')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"ynb_ypd_comparison_{timestamp}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file

def main():
    print("=" * 70)
    print("YNB vs YPD Media Comparison (Suthers et al., 2020 approach)")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load model
    print("Loading Yeast9 model...")
    model_path = "/home/michaelvolk/Documents/yeast-GEM/model/yeast-GEM.xml"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = load_model(model_path)
    print(f"Model loaded: {model.id}")
    print(f"Reactions: {len(model.reactions)}")
    print(f"Genes: {len(model.genes)}\n")

    # Load test gene combinations
    print("Loading test gene combinations...")
    results_dir = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth"
    os.makedirs(results_dir, exist_ok=True)

    perturbations_file = os.path.join(results_dir, "perturbations_all.json")

    if os.path.exists(perturbations_file):
        with open(perturbations_file, 'r') as f:
            perturbations = json.load(f)

        # Sample for testing
        np.random.seed(42)
        test_combos = []

        # Sample singles
        if perturbations['singles']:
            singles_sample = np.random.choice(perturbations['singles'],
                                            min(50, len(perturbations['singles'])),
                                            replace=False)
            test_combos.extend([[g] for g in singles_sample])

        # Sample doubles
        if perturbations['doubles']:
            doubles_sample = np.random.choice(len(perturbations['doubles']),
                                            min(100, len(perturbations['doubles'])),
                                            replace=False)
            test_combos.extend([perturbations['doubles'][i] for i in doubles_sample])

        # Sample triples
        if perturbations['triples']:
            triples_sample = np.random.choice(len(perturbations['triples']),
                                            min(50, len(perturbations['triples'])),
                                            replace=False)
            test_combos.extend([perturbations['triples'][i] for i in triples_sample])

        print(f"Testing {len(test_combos)} gene combinations")
        print(f"  Singles: {sum(1 for c in test_combos if len(c) == 1)}")
        print(f"  Doubles: {sum(1 for c in test_combos if len(c) == 2)}")
        print(f"  Triples: {sum(1 for c in test_combos if len(c) == 3)}")
    else:
        # Default test set
        print("Using small default test set...")
        test_combos = [
            ['YAL001C'], ['YAL002W'], ['YBR001C'],
            ['YAL001C', 'YAL002W'], ['YBR001C', 'YBR002C'],
            ['YAL001C', 'YAL002W', 'YAL003W']
        ]

    # Test YNB media
    print("\n" + "=" * 50)
    print("Testing YNB Media (minimal + vitamins)")
    print("=" * 50)

    with model as m:
        added, missing = setup_ynb_media(m)
        print(f"Added nutrients: {', '.join(added[:5])}...")
        if missing:
            print(f"Missing nutrients: {', '.join(missing)}")

        print("\nRunning FBA with YNB media...")
        ynb_results, ynb_wt = test_gene_deletions(m, test_combos, media_type='YNB')

    # Analyze YNB results
    ynb_bands = analyze_fitness_distribution(ynb_results, "YNB Media Results")

    # Test YPD media
    print("\n" + "=" * 50)
    print("Testing YPD Media (YNB + 20 amino acids)")
    print("=" * 50)

    with model as m:
        added_ynb, missing_ynb, added_aa, missing_aa = setup_ypd_media(m)
        print(f"Added YNB nutrients: {len(added_ynb)}")
        print(f"Added amino acids: {len(added_aa)}")
        if missing_aa:
            print(f"Missing amino acids: {', '.join(missing_aa[:5])}...")

        print("\nRunning FBA with YPD media...")
        ypd_results, ypd_wt = test_gene_deletions(m, test_combos, media_type='YPD')

    # Analyze YPD results
    ypd_bands = analyze_fitness_distribution(ypd_results, "YPD Media Results")

    # Compare results
    print("\n" + "=" * 70)
    print("MEDIA COMPARISON")
    print("=" * 70)

    print(f"\nWild-type growth rates:")
    print(f"  YNB: {ynb_wt:.4f} /h")
    print(f"  YPD: {ypd_wt:.4f} /h")
    print(f"  YPD/YNB ratio: {ypd_wt/ynb_wt:.2f}")

    # Check if bands shift between media
    print(f"\nFitness bands comparison:")
    print(f"  YNB bands: {sorted(ynb_bands)}")
    print(f"  YPD bands: {sorted(ypd_bands)}")

    common_bands = set(ynb_bands) & set(ypd_bands)
    if common_bands:
        print(f"  Common bands: {sorted(common_bands)}")

    ynb_only = set(ynb_bands) - set(ypd_bands)
    if ynb_only:
        print(f"  YNB-specific bands: {sorted(ynb_only)}")

    ypd_only = set(ypd_bands) - set(ynb_bands)
    if ypd_only:
        print(f"  YPD-specific bands: {sorted(ypd_only)}")

    # Statistical comparison
    print(f"\nStatistical summary:")
    print(f"  YNB mean fitness: {ynb_results['fitness'].mean():.3f} ± {ynb_results['fitness'].std():.3f}")
    print(f"  YPD mean fitness: {ypd_results['fitness'].mean():.3f} ± {ypd_results['fitness'].std():.3f}")

    # Count rescued mutants
    merged = ynb_results.merge(ypd_results, on='genes', suffixes=('_ynb', '_ypd'))
    rescued = merged[(merged['fitness_ynb'] < 0.1) & (merged['fitness_ypd'] > 0.5)]
    print(f"\nRescued mutants (fitness<0.1 in YNB, >0.5 in YPD): {len(rescued)}")

    if len(rescued) > 0:
        print("  Examples:")
        for _, row in rescued.head(3).iterrows():
            print(f"    {row['genes']}: YNB={row['fitness_ynb']:.3f}, YPD={row['fitness_ypd']:.3f}")

    # Create comparison plots
    print("\nGenerating comparison plots...")
    plot_file = plot_comparison(ynb_results, ypd_results, results_dir)
    print(f"Plot saved to: {plot_file}")

    # Save results
    ynb_file = os.path.join(results_dir, "ynb_test_results.parquet")
    ynb_results.to_parquet(ynb_file)

    ypd_file = os.path.join(results_dir, "ypd_test_results.parquet")
    ypd_results.to_parquet(ypd_file)

    print(f"\nResults saved to:")
    print(f"  {ynb_file}")
    print(f"  {ypd_file}")

    # Final interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if abs(len(ynb_bands) - len(ypd_bands)) > 2:
        print("\n⚠ Different number of fitness bands between media conditions")
        print("This suggests media composition significantly affects the discrete clustering.")
    else:
        print("\n✓ Similar number of fitness bands between media conditions")
        print("The discrete clustering appears to be model-intrinsic rather than media-driven.")

    if ypd_wt > ynb_wt * 1.2:
        print("\n✓ YPD shows higher growth rate as expected (amino acids provide benefit)")
    else:
        print("\n⚠ YPD growth rate not significantly higher than YNB")
        print("Check if amino acid uptake reactions are properly configured.")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()