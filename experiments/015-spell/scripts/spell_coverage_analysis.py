# experiments/015-spell/scripts/spell_coverage_analysis
# [[experiments.015-spell.scripts.spell_coverage_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/015-spell/scripts/spell_coverage_analysis

"""
SPELL Coverage Analysis - Phase 1.2

This module analyzes the enhanced SPELL condition metadata to:
1. Generate frequency distributions by condition category
2. Create co-occurrence matrices (which categories appear together)
3. Calculate parameter ranges (temperature, time, pH, concentrations)
4. Identify missing data gaps
5. Find unique category combinations
6. Generate prioritization recommendations for Environment subclasses

Usage:
    python experiments/015-spell/scripts/spell_coverage_analysis.py
"""

import os
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations
from dotenv import load_dotenv
from torchcell.timestamp import timestamp

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT", osp.expanduser(osp.join("~", "Documents", "projects", "torchcell")))
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", osp.expanduser(osp.join("~", "Documents", "projects", "torchcell", "notes", "assets", "images")))


def load_enhanced_metadata(csv_path=None):
    """Load the enhanced SPELL condition metadata CSV."""
    if csv_path is None:
        csv_path = osp.join(DATA_ROOT, "data/sgd/spell", "spell_conditions_metadata_enhanced.csv")

    if not osp.exists(csv_path):
        raise FileNotFoundError(
            f"Enhanced metadata CSV not found at: {csv_path}\n"
            f"Please run spell.py main() first to generate the enhanced metadata."
        )

    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} condition records from: {csv_path}")
    return df


def generate_frequency_distribution(df):
    """Generate frequency distribution by condition category."""
    print("\n" + "=" * 70)
    print("FREQUENCY DISTRIBUTION BY CATEGORY")
    print("=" * 70)

    # Primary category counts
    category_counts = df['primary_category'].value_counts()

    results = []
    for category, count in category_counts.items():
        pct = 100 * count / len(df)
        results.append({
            'category': category,
            'count': count,
            'percentage': pct
        })
        print(f"  {category:25s}: {count:6,} ({pct:5.1f}%)")

    return pd.DataFrame(results)


def generate_cooccurrence_matrix(df):
    """Generate co-occurrence matrix showing which categories appear together."""
    print("\n" + "=" * 70)
    print("CATEGORY CO-OCCURRENCE ANALYSIS")
    print("=" * 70)

    # Get all unique categories from both primary and secondary
    all_categories = set(df['primary_category'].unique())

    # Add secondary categories
    for cats in df['secondary_categories'].dropna():
        if cats:  # Not empty string
            all_categories.update(cats.split('|'))

    all_categories = sorted(list(all_categories))

    # Create co-occurrence matrix
    cooccur = pd.DataFrame(0, index=all_categories, columns=all_categories)

    for idx, row in df.iterrows():
        # Get all categories for this condition
        cats = [row['primary_category']]
        if pd.notna(row['secondary_categories']) and row['secondary_categories']:
            cats.extend(row['secondary_categories'].split('|'))

        # Count co-occurrences
        for cat1, cat2 in combinations(cats, 2):
            if cat1 in cooccur.index and cat2 in cooccur.columns:
                cooccur.loc[cat1, cat2] += 1
                cooccur.loc[cat2, cat1] += 1

    # Print top co-occurrences
    print("\nTop 20 Category Co-occurrences:")
    cooccur_pairs = []
    for i in range(len(cooccur)):
        for j in range(i+1, len(cooccur)):
            count = cooccur.iloc[i, j]
            if count > 0:
                cooccur_pairs.append({
                    'category1': cooccur.index[i],
                    'category2': cooccur.columns[j],
                    'count': count
                })

    cooccur_df = pd.DataFrame(cooccur_pairs).sort_values('count', ascending=False)
    for idx, row in cooccur_df.head(20).iterrows():
        print(f"  {row['category1']:20s} + {row['category2']:20s}: {row['count']:5,}")

    return cooccur, cooccur_df


def analyze_parameter_ranges(df):
    """Analyze ranges of extracted parameters."""
    print("\n" + "=" * 70)
    print("PARAMETER RANGES")
    print("=" * 70)

    results = {}

    # Temperature
    temp_data = df['temperature_c'].dropna()
    if len(temp_data) > 0:
        print(f"\nTemperature (°C):")
        print(f"  Count:   {len(temp_data):6,}")
        print(f"  Min:     {temp_data.min():6.1f}")
        print(f"  Max:     {temp_data.max():6.1f}")
        print(f"  Mean:    {temp_data.mean():6.1f}")
        print(f"  Median:  {temp_data.median():6.1f}")
        results['temperature'] = {
            'count': len(temp_data),
            'min': temp_data.min(),
            'max': temp_data.max(),
            'mean': temp_data.mean(),
            'median': temp_data.median()
        }

    # Time
    time_data = df['time_min'].dropna()
    if len(time_data) > 0:
        print(f"\nTime (minutes):")
        print(f"  Count:   {len(time_data):6,}")
        print(f"  Min:     {time_data.min():6.1f}")
        print(f"  Max:     {time_data.max():6.1f}")
        print(f"  Mean:    {time_data.mean():6.1f}")
        print(f"  Median:  {time_data.median():6.1f}")
        results['time'] = {
            'count': len(time_data),
            'min': time_data.min(),
            'max': time_data.max(),
            'mean': time_data.mean(),
            'median': time_data.median()
        }

    # pH
    ph_data = df['ph'].dropna()
    if len(ph_data) > 0:
        print(f"\npH:")
        print(f"  Count:   {len(ph_data):6,}")
        print(f"  Min:     {ph_data.min():6.1f}")
        print(f"  Max:     {ph_data.max():6.1f}")
        print(f"  Mean:    {ph_data.mean():6.1f}")
        print(f"  Median:  {ph_data.median():6.1f}")
        results['ph'] = {
            'count': len(ph_data),
            'min': ph_data.min(),
            'max': ph_data.max(),
            'mean': ph_data.mean(),
            'median': ph_data.median()
        }

    # Concentration (group by unit)
    conc_data = df[df['concentration'].notna()]
    if len(conc_data) > 0:
        print(f"\nConcentrations by unit:")
        for unit in conc_data['concentration_unit'].unique():
            unit_data = conc_data[conc_data['concentration_unit'] == unit]['concentration']
            print(f"  {unit:10s}: {len(unit_data):4,} conditions, range: {unit_data.min():.2f} - {unit_data.max():.2f}")

    # Chemical compounds
    chem_data = df['chemical_name'].dropna()
    if len(chem_data) > 0:
        print(f"\nTop 15 Chemical Compounds:")
        for chem, count in chem_data.value_counts().head(15).items():
            print(f"  {chem:30s}: {count:5,}")

    # Carbon sources
    carbon_data = df['carbon_source'].dropna()
    if len(carbon_data) > 0:
        print(f"\nCarbon Sources:")
        for source, count in carbon_data.value_counts().items():
            print(f"  {source:20s}: {count:5,}")

    # Oxygen levels
    oxygen_data = df['oxygen_level'].dropna()
    if len(oxygen_data) > 0:
        print(f"\nOxygen Levels:")
        for level, count in oxygen_data.value_counts().items():
            print(f"  {level:20s}: {count:5,}")

    return results


def analyze_missing_data(df):
    """Analyze missing data patterns."""
    print("\n" + "=" * 70)
    print("MISSING DATA ANALYSIS")
    print("=" * 70)

    # Calculate percentage missing for each column
    missing_analysis = []

    structured_columns = [
        'time_min', 'temperature_c', 'chemical_name', 'concentration',
        'carbon_source', 'nitrogen_source', 'limitation_type',
        'ph', 'oxygen_level', 'stress_type',
        'cell_cycle_phase', 'synchronization_method',
        'replicate_number', 'replicate_type'
    ]

    print("\nMissing Data by Field:")
    for col in structured_columns:
        missing_count = df[col].isna().sum()
        missing_pct = 100 * missing_count / len(df)
        present_count = len(df) - missing_count
        present_pct = 100 - missing_pct

        missing_analysis.append({
            'field': col,
            'present_count': present_count,
            'present_pct': present_pct,
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })

        print(f"  {col:25s}: {present_count:6,} present ({present_pct:5.1f}%), {missing_count:6,} missing ({missing_pct:5.1f}%)")

    return pd.DataFrame(missing_analysis)


def find_unique_combinations(df, top_n=30):
    """Find unique category combinations and their frequencies."""
    print("\n" + "=" * 70)
    print(f"TOP {top_n} UNIQUE CATEGORY COMBINATIONS")
    print("=" * 70)

    # Create full category strings
    df['full_categories'] = df.apply(
        lambda row: row['primary_category'] if pd.isna(row['secondary_categories']) or not row['secondary_categories']
        else row['primary_category'] + '|' + row['secondary_categories'],
        axis=1
    )

    # Count combinations
    combo_counts = df['full_categories'].value_counts().head(top_n)

    results = []
    for combo, count in combo_counts.items():
        pct = 100 * count / len(df)
        results.append({
            'combination': combo,
            'count': count,
            'percentage': pct
        })
        # Format for display
        categories = combo.split('|')
        display = ' + '.join(categories)
        print(f"  {display:50s}: {count:5,} ({pct:4.1f}%)")

    return pd.DataFrame(results)


def generate_environment_prioritization(df, freq_dist, cooccur_df):
    """
    Generate prioritization ranking for Environment subclasses based on:
    - Frequency (how many conditions)
    - Extraction completeness (how well we can extract structured data)
    - Scientific importance (common experimental types)
    """
    print("\n" + "=" * 70)
    print("ENVIRONMENT SUBCLASS PRIORITIZATION")
    print("=" * 70)

    # Define potential Environment subclasses and their mapping to categories
    env_classes = {
        'TimeSeriesEnvironment': {
            'primary_categories': ['time_series'],
            'required_fields': ['time_min', 'is_timeseries'],
            'importance': 9,  # 1-10 scale
        },
        'HeatShockEnvironment': {
            'primary_categories': ['heat_shock'],
            'required_fields': ['temperature_c'],
            'importance': 8,
        },
        'OxidativeStressEnvironment': {
            'primary_categories': ['oxidative_stress'],
            'required_fields': ['chemical_name', 'concentration'],
            'importance': 8,
        },
        'OsmoticStressEnvironment': {
            'primary_categories': ['osmotic_stress'],
            'required_fields': ['chemical_name', 'concentration'],
            'importance': 7,
        },
        'NutrientEnvironment': {
            'primary_categories': ['nutrient_limitation'],
            'required_fields': ['carbon_source', 'nitrogen_source', 'limitation_type'],
            'importance': 9,
        },
        'DrugTreatmentEnvironment': {
            'primary_categories': ['drug_treatment'],
            'required_fields': ['chemical_name', 'concentration'],
            'importance': 7,
        },
        'CellCycleEnvironment': {
            'primary_categories': ['cell_cycle'],
            'required_fields': ['cell_cycle_phase', 'synchronization_method'],
            'importance': 7,
        },
        'DNADamageEnvironment': {
            'primary_categories': ['dna_damage'],
            'required_fields': ['chemical_name'],
            'importance': 5,  # Lower: DNA damage causes genotypic changes, hard to model without sequencing
        },
        'AnaerobicEnvironment': {
            'primary_categories': ['anaerobic'],
            'required_fields': ['oxygen_level'],
            'importance': 7,  # Medium: well-defined environmental condition
        },
        'ChemicalStressEnvironment': {
            'primary_categories': ['chemical_stress'],
            'required_fields': ['chemical_name'],
            'importance': 7,  # Medium: includes ER stress (DTT, tunicamycin) which are important
        },
    }

    priorities = []

    for env_class, config in env_classes.items():
        # Count conditions matching this class
        mask = df['primary_category'].isin(config['primary_categories'])
        matching_conditions = df[mask]
        count = len(matching_conditions)

        if count == 0:
            continue

        # Calculate completeness (what % of conditions have required fields)
        completeness_scores = []
        for field in config['required_fields']:
            field_present = matching_conditions[field].notna().sum()
            completeness_scores.append(field_present / count if count > 0 else 0)

        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0

        # Calculate priority score
        # Weighted: 40% frequency, 30% completeness, 30% importance
        freq_score = count / len(df)  # Normalize to 0-1
        importance_score = config['importance'] / 10  # Normalize to 0-1

        priority_score = (0.4 * freq_score * 100) + (0.3 * avg_completeness * 100) + (0.3 * importance_score * 100)

        priorities.append({
            'environment_class': env_class,
            'condition_count': count,
            'coverage_pct': 100 * count / len(df),
            'avg_completeness': avg_completeness,
            'importance': config['importance'],
            'priority_score': priority_score,
            'primary_categories': ', '.join(config['primary_categories']),
        })

    # Sort by priority score
    priority_df = pd.DataFrame(priorities).sort_values('priority_score', ascending=False)

    print("\nRecommended Implementation Order:")
    print(f"{'Rank':<6}{'Environment Class':<35}{'Conditions':<12}{'Coverage':<10}{'Complete':<10}{'Priority':<10}")
    print("-" * 90)

    for idx, row in priority_df.iterrows():
        rank = list(priority_df.index).index(idx) + 1
        print(f"{rank:<6}{row['environment_class']:<35}{row['condition_count']:<12,}{row['coverage_pct']:<10.1f}%{row['avg_completeness']:<10.1%}{row['priority_score']:<10.1f}")

    return priority_df


def plot_coverage_visualizations(df, freq_dist, missing_df, output_dir=None):
    """Generate visualization plots for coverage analysis."""
    if output_dir is None:
        output_dir = ASSET_IMAGES_DIR

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1. Category frequency bar chart
    ax = axes[0, 0]
    top_categories = freq_dist.head(20)
    n_categories = len(top_categories)
    bars = ax.barh(top_categories['category'], top_categories['count'], color='steelblue')
    ax.set_xlabel('Number of Conditions')
    ax.set_title(f'All {n_categories} Condition Categories')
    ax.set_xlim(0, 8000)
    ax.invert_yaxis()

    # Add count labels on bars
    for bar, count in zip(bars, top_categories['count']):
        ax.text(bar.get_width() + 150, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=9)

    # 2. Extraction completeness
    ax = axes[0, 1]
    top_fields = missing_df.sort_values('present_pct', ascending=True).tail(15)
    ax.barh(top_fields['field'], top_fields['present_pct'], color='steelblue')
    ax.set_xlabel('Percentage of Conditions with Data')
    ax.set_title('Data Extraction Completeness by Field\n(What % of conditions have this field extracted)')
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax.invert_yaxis()

    # 3. Confidence score distribution
    ax = axes[1, 0]
    ax.hist(df['extraction_confidence'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Extraction Confidence Score')
    ax.set_ylabel('Number of Conditions')
    ax.set_title('Distribution of Extraction Confidence Scores')
    ax.axvline(x=df['extraction_confidence'].mean(), color='red', linestyle='--',
               label=f"Mean: {df['extraction_confidence'].mean():.3f}")
    ax.legend()

    # 4. Parameter ranges (temperature distribution)
    ax = axes[1, 1]
    temp_data = df['temperature_c'].dropna()
    if len(temp_data) > 0:
        # Filter to biologically relevant range (0-100°C)
        temp_filtered = temp_data[(temp_data >= 0) & (temp_data <= 100)]
        if len(temp_filtered) > 0:
            ax.hist(temp_filtered, bins=range(0, 105, 5), edgecolor='black', alpha=0.7, color='coral')
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Number of Conditions')
            ax.set_title(f'Temperature Distribution ({len(temp_filtered):,} conditions, 0-100°C)')
            ax.set_xticks(range(0, 101, 10))
            ax.axvline(x=temp_filtered.median(), color='red', linestyle='--',
                       label=f"Median: {temp_filtered.median():.1f}°C")
            ax.axvline(x=30, color='blue', linestyle=':', alpha=0.6,
                       label='30°C (standard)')
            ax.legend()
            if len(temp_data) > len(temp_filtered):
                ax.text(0.98, 0.98, f'Excluded {len(temp_data) - len(temp_filtered)} outliers',
                        transform=ax.transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8),
                        fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No temperature data in 0-100°C range', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No temperature data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    save_path = osp.join(output_dir, "015-spell", "spell_coverage_analysis.png")
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")
    plt.close()


def generate_markdown_report(df, freq_dist, priority_df, missing_df, output_path=None):
    """Generate a comprehensive markdown report."""
    if output_path is None:
        output_path = osp.join(DATA_ROOT, "data/sgd/spell", "spell_coverage_report.md")

    report = []
    report.append("# SPELL Coverage Analysis Report")
    report.append(f"\nGenerated: {timestamp()}")
    report.append(f"\nTotal Conditions Analyzed: {len(df):,}\n")

    report.append("## Executive Summary\n")
    report.append(f"- **Total conditions**: {len(df):,}")
    report.append(f"- **Mean extraction confidence**: {df['extraction_confidence'].mean():.3f}")
    report.append(f"- **Conditions needing manual review**: {df['needs_manual_review'].sum():,} ({100*df['needs_manual_review'].sum()/len(df):.1f}%)")
    report.append(f"- **High confidence conditions (>0.5)**: {(df['extraction_confidence'] > 0.5).sum():,} ({100*(df['extraction_confidence'] > 0.5).sum()/len(df):.1f}%)\n")

    report.append("## Recommended Environment Subclass Implementation Order\n")
    report.append("| Rank | Environment Class | Conditions | Coverage | Completeness | Priority Score |")
    report.append("|------|-------------------|------------|----------|--------------|----------------|")
    for idx, row in priority_df.iterrows():
        rank = list(priority_df.index).index(idx) + 1
        report.append(f"| {rank} | {row['environment_class']} | {row['condition_count']:,} | {row['coverage_pct']:.1f}% | {row['avg_completeness']:.1%} | {row['priority_score']:.1f} |")

    report.append("\n## Category Frequency Distribution\n")
    report.append("| Category | Count | Percentage |")
    report.append("|----------|-------|------------|")
    for _, row in freq_dist.head(20).iterrows():
        report.append(f"| {row['category']} | {row['count']:,} | {row['percentage']:.1f}% |")

    report.append("\n## Data Extraction Completeness\n")
    report.append("| Field | Present | Missing |")
    report.append("|-------|---------|---------|")
    for _, row in missing_df.sort_values('present_pct', ascending=False).iterrows():
        report.append(f"| {row['field']} | {row['present_count']:,} ({row['present_pct']:.1f}%) | {row['missing_count']:,} ({row['missing_pct']:.1f}%) |")

    report.append("\n## Next Steps\n")
    report.append("1. Begin implementation with top 3-5 priority Environment subclasses")
    report.append("2. Review conditions flagged for manual review")
    report.append("3. Set up LLM extraction pipeline for paper-based metadata (Phase 3)")
    report.append("4. Iterate on schema design based on extraction results\n")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Generated markdown report: {output_path}")


def main():
    """Run full coverage analysis pipeline."""
    print("=" * 70)
    print("SPELL COVERAGE ANALYSIS - PHASE 1.2")
    print("=" * 70)

    # Load data
    df = load_enhanced_metadata()

    # Run analyses
    freq_dist = generate_frequency_distribution(df)
    cooccur, cooccur_df = generate_cooccurrence_matrix(df)
    param_ranges = analyze_parameter_ranges(df)
    missing_df = analyze_missing_data(df)
    unique_combos = find_unique_combinations(df)
    priority_df = generate_environment_prioritization(df, freq_dist, cooccur_df)

    # Generate visualizations
    plot_coverage_visualizations(df, freq_dist, missing_df)

    # Generate markdown report
    generate_markdown_report(df, freq_dist, priority_df, missing_df)

    print("\n" + "=" * 70)
    print("COVERAGE ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nOutputs:")
    print(f"  - Markdown report: data/sgd/spell/spell_coverage_report.md")
    print(f"  - Visualizations: {ASSET_IMAGES_DIR}/spell_coverage_analysis_*.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
