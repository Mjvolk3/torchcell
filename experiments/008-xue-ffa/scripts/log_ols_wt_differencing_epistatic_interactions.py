#!/usr/bin/env python3

"""
Log-OLS with WT-differencing epistatic interaction analysis for FFA production data.
Implements primary model using log-transformed fitness with WT-differencing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log
from statsmodels.stats.multitest import multipletests
import warnings
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
from dotenv import load_dotenv
import json
from torchcell.timestamp import timestamp

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Define paths
BASE_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "glm_models"
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", str(BASE_DIR / "figures"))

# Create output directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
Path(ASSET_IMAGES_DIR).mkdir(parents=True, exist_ok=True)

# FFA columns
FFA_COLUMNS = ["C140", "C160", "C180", "C161", "C181", "Total Titer"]

# TF genes and their single-letter abbreviations
# Based on the abbreviations sheet in the Excel file
TF_GENES = {
    "F": "FKH1",
    "G": "GCN5",
    "M": "MED4",
    "O": "OPI1",
    "X": "RFX1",
    "R": "RGR1",
    "P": "RPD3",
    "S": "SPT3",
    "Y": "YAP6",
    "T": "TFC7"
}

# Pseudocount for log transformations
DELTA = 1e-6


def load_ffa_data_with_replicates(file_path):
    """Load FFA data with proper strain identification."""
    # Read abbreviations
    abbrev_df = pd.read_excel(file_path, sheet_name='Abbreviations')
    abbreviations = dict(zip(abbrev_df.iloc[:, 0], abbrev_df.iloc[:, 1]))

    # Read raw titer data
    raw_df = pd.read_excel(file_path, sheet_name='raw-titer (mg-L)')

    # Create long-format dataframe
    data_list = []

    for idx in range(len(raw_df)):
        genotype = str(raw_df.iloc[idx, 0])

        # Process each replicate
        for rep_idx in range(3):
            row_data = {
                'genotype': genotype,
                'replicate': rep_idx + 1
            }

            # Get FFA values
            row_data['C140'] = raw_df.iloc[idx, 1 + rep_idx]
            row_data['C160'] = raw_df.iloc[idx, 4 + rep_idx]
            row_data['C180'] = raw_df.iloc[idx, 7 + rep_idx]
            row_data['C161'] = raw_df.iloc[idx, 10 + rep_idx]
            row_data['C181'] = raw_df.iloc[idx, 13 + rep_idx]

            # Calculate total titer
            row_data['Total Titer'] = (
                row_data['C140'] + row_data['C160'] + row_data['C180'] +
                row_data['C161'] + row_data['C181']
            )

            # Only add if not NaN
            if not np.isnan(row_data['C140']):
                data_list.append(row_data)

    df = pd.DataFrame(data_list)

    # Parse genotype to identify knockouts
    # Pattern: "X-Y-Z 6Δ" means X, Y, Z are knocked out (plus 3Δ background)
    # Pattern: "X 4Δ" means X is knocked out (plus 3Δ background)
    # "wt BY4741" is wild type (use as reference since no pure 3Δ)

    # Initialize KO columns
    for abbr, gene in TF_GENES.items():
        df[f'ko_{gene}'] = 0

    # Parse genotypes
    for idx, row in df.iterrows():
        genotype = row['genotype']

        if 'wt' in genotype.lower():
            # Wild type - no KOs
            continue
        elif '4Δ' in genotype:
            # Single KO format: "X 4Δ"
            parts = genotype.split()
            if len(parts) > 0:
                ko_letter = parts[0].strip()
                if ko_letter in TF_GENES:
                    gene = TF_GENES[ko_letter]
                    df.at[idx, f'ko_{gene}'] = 1
        elif '5Δ' in genotype:
            # Double KO format: "X-Y 5Δ"
            parts = genotype.split()
            if len(parts) > 0 and '-' in parts[0]:
                ko_letters = parts[0].split('-')
                for letter in ko_letters:
                    if letter in TF_GENES:
                        gene = TF_GENES[letter]
                        df.at[idx, f'ko_{gene}'] = 1
        elif '6Δ' in genotype:
            # Triple KO format: "X-Y-Z 6Δ"
            parts = genotype.split()
            if len(parts) > 0 and '-' in parts[0]:
                ko_letters = parts[0].split('-')
                for letter in ko_letters:
                    if letter in TF_GENES:
                        gene = TF_GENES[letter]
                        df.at[idx, f'ko_{gene}'] = 1

    # Identify reference strain
    # Use wild type as reference since no pure 3Δ control
    df['is_reference'] = df['genotype'].str.contains('wt', case=False, na=False)

    print(f"Reference strain: wild type (wt BY4741)")
    print(f"Reference observations: {df['is_reference'].sum()}")

    return df


def create_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create design matrix with main effects and interactions."""
    X = pd.DataFrame(index=df.index)

    # Get KO columns
    ko_cols = [col for col in df.columns if col.startswith("ko_")]
    ko_cols = [col for col in ko_cols if df[col].sum() > 0]  # Only use KOs present in data

    # Add intercept
    X['intercept'] = 1

    # Add main effects
    for col in ko_cols:
        X[col] = df[col]

    # Add pairwise interactions (digenic)
    for col1, col2 in combinations(ko_cols, 2):
        # Only add if both KOs co-occur
        if (df[col1] * df[col2]).sum() > 0:
            X[f"{col1}:{col2}"] = df[col1] * df[col2]

    # Add three-way interactions (trigenic)
    for col1, col2, col3 in combinations(ko_cols, 3):
        # Only add if all three KOs co-occur
        if (df[col1] * df[col2] * df[col3]).sum() > 0:
            X[f"{col1}:{col2}:{col3}"] = df[col1] * df[col2] * df[col3]

    return X


def model_a_log_ols(df: pd.DataFrame, trait_col: str) -> Dict:
    """
    Model A: Primary model - Product-null hypothesis on log scale.
    Uses WT-differencing: s_{g,r} = log(y_{g,r} + δ) - log(y_{ref,r} + δ)
    """
    results = {}

    # Get reference values per replicate
    ref_df = df[df['is_reference']].copy()
    if len(ref_df) == 0:
        print(f"Warning: No reference strain found for trait {trait_col}")
        return {}

    # Calculate reference mean per replicate
    ref_means = ref_df.groupby('replicate')[trait_col].mean()

    # Prepare data with WT-differencing
    model_df = df[~df['is_reference']].copy()

    # Apply WT-differencing
    model_df['log_response'] = np.nan
    for rep in ref_means.index:
        mask = model_df['replicate'] == rep
        if mask.sum() > 0:
            model_df.loc[mask, 'log_response'] = (
                np.log(model_df.loc[mask, trait_col] + DELTA) -
                np.log(ref_means[rep] + DELTA)
            )

    # Remove NaN values
    model_df = model_df.dropna(subset=['log_response'])

    if len(model_df) == 0:
        print(f"Warning: No valid data after WT-differencing for {trait_col}")
        return {}

    # Create design matrix
    X = create_design_matrix(model_df)

    # Add replicate effects (even with WT-differencing, can capture remaining variation)
    for rep in model_df['replicate'].unique():
        if rep != 1:  # Use rep 1 as baseline
            X[f'rep_{rep}'] = (model_df['replicate'] == rep).astype(int)

    y = model_df['log_response']

    # Fit OLS model
    try:
        model = sm.OLS(y, X)
        fit = model.fit()
    except Exception as e:
        print(f"Error fitting model for {trait_col}: {e}")
        return {}

    # Store results
    results['model'] = fit
    results['coefficients'] = fit.params
    results['std_errors'] = fit.bse
    results['pvalues'] = fit.pvalues
    results['conf_intervals'] = fit.conf_int()
    results['r_squared'] = fit.rsquared
    results['adj_r_squared'] = fit.rsquared_adj

    # Extract epistatic interactions
    epistasis = {}

    for param in fit.params.index:
        if ':' in param and 'rep_' not in param:  # Interaction term (not replicate)
            n_interactions = param.count(':')

            # Clean gene names
            clean_name = param.replace('ko_', '')

            epistasis[clean_name] = {
                'E': fit.params[param],  # Log scale epistasis coefficient
                'phi': np.exp(fit.params[param]),  # Epistatic fold
                'se': fit.bse[param],
                'pvalue': fit.pvalues[param],
                'ci_lower': fit.conf_int().loc[param, 0],
                'ci_upper': fit.conf_int().loc[param, 1],
                'phi_ci_lower': np.exp(fit.conf_int().loc[param, 0]),
                'phi_ci_upper': np.exp(fit.conf_int().loc[param, 1]),
                'order': 'digenic' if n_interactions == 1 else 'trigenic'
            }

    results['epistasis'] = epistasis
    results['trait'] = trait_col

    # Count interactions
    results['n_digenic'] = sum(1 for v in epistasis.values() if v['order'] == 'digenic')
    results['n_trigenic'] = sum(1 for v in epistasis.values() if v['order'] == 'trigenic')
    results['n_sig_digenic'] = sum(1 for v in epistasis.values()
                                   if v['order'] == 'digenic' and v['pvalue'] < 0.05)
    results['n_sig_trigenic'] = sum(1 for v in epistasis.values()
                                    if v['order'] == 'trigenic' and v['pvalue'] < 0.05)

    return results


def save_results(results: Dict, output_dir: Path = RESULTS_DIR):
    """Save results to files including CSV format matching multiplicative model."""
    # Build comprehensive summary JSON matching standardized format
    summary = {}

    # Create overall summary
    overall_dig = 0
    overall_tri = 0
    overall_sig_dig = 0
    overall_sig_tri = 0

    for trait in results:
        if results[trait]:
            overall_dig += results[trait].get('n_digenic', 0)
            overall_tri += results[trait].get('n_trigenic', 0)
            overall_sig_dig += results[trait].get('n_sig_digenic', 0)
            overall_sig_tri += results[trait].get('n_sig_trigenic', 0)

            # Extract effect sizes from epistasis dict
            epistasis = results[trait].get('epistasis', {})
            dig_effects = [abs(v['E']) for v in epistasis.values() if v['order'] == 'digenic']
            tri_effects = [abs(v['E']) for v in epistasis.values() if v['order'] == 'trigenic']

            summary[trait] = {
                'n_digenic': results[trait].get('n_digenic', 0),
                'n_trigenic': results[trait].get('n_trigenic', 0),
                'n_sig_digenic': results[trait].get('n_sig_digenic', 0),
                'n_sig_trigenic': results[trait].get('n_sig_trigenic', 0),
                'n_sig_digenic_fdr': 0,  # FDR is in CSV
                'n_sig_trigenic_fdr': 0,
                'r_squared': results[trait].get('r_squared', None),
                'effect_size_digenic': {
                    'mean': float(np.mean(dig_effects)) if dig_effects else 0.0,
                    'median': float(np.median(dig_effects)) if dig_effects else 0.0,
                    'max': float(np.max(dig_effects)) if dig_effects else 0.0
                },
                'effect_size_trigenic': {
                    'mean': float(np.mean(tri_effects)) if tri_effects else 0.0,
                    'median': float(np.median(tri_effects)) if tri_effects else 0.0,
                    'max': float(np.max(tri_effects)) if tri_effects else 0.0
                }
            }

    # Add overall summary
    summary['_overall'] = {
        'total_interactions': overall_dig + overall_tri,
        'n_digenic': overall_dig,
        'n_trigenic': overall_tri,
        'n_sig_digenic': overall_sig_dig,
        'n_sig_trigenic': overall_sig_tri
    }

    with open(output_dir / 'log_ols_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Create CSV output matching multiplicative model format
    csv_rows = []

    # Collect all p-values for FDR correction
    all_pvalues = []
    pvalue_keys = []  # (trait, gene_set) tuples

    for trait, result in results.items():
        if result and 'epistasis' in result:
            for gene_set, epi_data in result['epistasis'].items():
                all_pvalues.append(epi_data['pvalue'])
                pvalue_keys.append((trait, gene_set))

    # Apply FDR correction
    if all_pvalues:
        _, fdr_pvals, _, _ = multipletests(all_pvalues, method='fdr_bh', alpha=0.05)
        fdr_dict = dict(zip(pvalue_keys, fdr_pvals))
    else:
        fdr_dict = {}

    # Create rows for each interaction
    for trait, result in results.items():
        if not result or 'epistasis' not in result:
            continue

        for gene_set, epi_data in result['epistasis'].items():
            # Convert trait name to match multiplicative format
            trait_name = trait.replace('Total Titer', 'Total Titer')

            csv_rows.append({
                'gene_set': gene_set,
                'interaction_type': epi_data['order'],
                'ffa_type': trait_name,
                'interaction_score': epi_data['E'],  # Log-scale epistasis coefficient
                'epistatic_fold': epi_data['phi'],  # exp(E)
                'standard_error': epi_data['se'],
                'p_value': epi_data['pvalue'],
                'fdr_corrected_p': fdr_dict.get((trait, gene_set), np.nan),
                'effect_size': abs(epi_data['E']),
                'significant_p05': epi_data['pvalue'] < 0.05,
                'significant_fdr05': fdr_dict.get((trait, gene_set), 1.0) < 0.05,
                'ci_lower': epi_data['ci_lower'],
                'ci_upper': epi_data['ci_upper']
            })

    # Save as CSV
    if csv_rows:
        df = pd.DataFrame(csv_rows)

        # Save combined results
        csv_path = output_dir / "log_ols_all_interactions.csv"
        df.to_csv(csv_path, index=False)

        # Save digenic and trigenic separately
        digenic_df = df[df['interaction_type'] == 'digenic']
        trigenic_df = df[df['interaction_type'] == 'trigenic']

        digenic_path = output_dir / "log_ols_digenic_interactions.csv"
        trigenic_path = output_dir / "log_ols_trigenic_interactions.csv"

        digenic_df.to_csv(digenic_path, index=False)
        trigenic_df.to_csv(trigenic_path, index=False)

        print(f"CSV results saved to {output_dir}")
        print(f"  - {csv_path.name}")
        print(f"  - {digenic_path.name}")
        print(f"  - {trigenic_path.name}")

    print(f"Results saved to {output_dir}")


def main():
    """Main execution function."""
    print("Loading FFA data...")

    file_path = "/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/ffa_xue2025/raw/Supplementary Data 1_Raw titers.xlsx"

    if not Path(file_path).exists():
        print(f"Error: Data file not found at {file_path}")
        return

    df = load_ffa_data_with_replicates(file_path)

    print(f"\nLoaded {len(df)} observations")
    print(f"Unique genotypes: {df['genotype'].nunique()}")

    # Check KO distribution
    ko_cols = [col for col in df.columns if col.startswith('ko_')]
    for col in ko_cols:
        count = df[col].sum()
        if count > 0:
            print(f"  {col}: {count} observations")

    # Initialize results
    results = {}

    # Run Log-OLS model
    print("\n" + "="*60)
    print("Log-OLS with WT-Differencing Epistatic Interaction Analysis")
    print("="*60)

    for trait_col in FFA_COLUMNS:
        print(f"\nProcessing {trait_col}...")
        result = model_a_log_ols(df, trait_col)
        if result:
            results[trait_col] = result
            print(f"  Total: {result['n_digenic']} digenic, {result['n_trigenic']} trigenic")
            print(f"  Significant: {result['n_sig_digenic']} digenic, {result['n_sig_trigenic']} trigenic")
            if 'r_squared' in result:
                print(f"  R-squared: {result['r_squared']:.3f}")

    # Save results
    print("\nSaving results...")
    save_results(results)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_dig = sum(r.get('n_digenic', 0) for r in results.values())
    total_tri = sum(r.get('n_trigenic', 0) for r in results.values())
    sig_dig = sum(r.get('n_sig_digenic', 0) for r in results.values())
    sig_tri = sum(r.get('n_sig_trigenic', 0) for r in results.values())

    print(f"\nTotal: {total_dig} digenic, {total_tri} trigenic")
    if total_dig > 0 or total_tri > 0:
        print(f"Significant: {sig_dig} digenic ({100*sig_dig/max(total_dig,1):.1f}%), "
              f"{sig_tri} trigenic ({100*sig_tri/max(total_tri,1):.1f}%)")

    print("\nAnalysis complete!")
    return results


if __name__ == "__main__":
    results = main()