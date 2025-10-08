#!/usr/bin/env python3

"""
Final corrected GLM-based epistatic interaction analysis for FFA production data.
Properly implements Model A as the primary model according to specification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log
import warnings
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
from dotenv import load_dotenv
import pickle
import json

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


def model_b_glm(df: pd.DataFrame, trait_col: str) -> Dict:
    """Model B: GLM with Gamma family and log link (robustness check)."""
    results = {}

    # Prepare data - EXCLUDE reference strain (WT)
    # The reference level is captured by the intercept
    model_df = df[~df['is_reference']].copy()
    model_df = model_df.dropna(subset=[trait_col])

    if len(model_df) == 0:
        return {}

    # Create design matrix
    X = create_design_matrix(model_df)

    # Add replicate effects
    for rep in model_df['replicate'].unique():
        if rep != 1:
            X[f'rep_{rep}'] = (model_df['replicate'] == rep).astype(int)

    y = model_df[trait_col]

    # Fit GLM
    try:
        glm_family = Gamma(link=Log())
        model = sm.GLM(y, X, family=glm_family)
        fit = model.fit()
    except Exception as e:
        print(f"Error fitting GLM for {trait_col}: {e}")
        return {}

    # Store results
    results['model'] = fit
    results['coefficients'] = fit.params
    results['std_errors'] = fit.bse
    results['pvalues'] = fit.pvalues

    # Calculate pseudo-R² (McFadden's R²)
    # McFadden's R² = 1 - (deviance / null deviance)
    results['pseudo_r_squared'] = 1 - (fit.deviance / fit.null_deviance)
    results['deviance'] = fit.deviance
    results['null_deviance'] = fit.null_deviance

    # Extract epistatic interactions
    epistasis = {}

    for param in fit.params.index:
        if ':' in param and 'rep_' not in param:
            n_interactions = param.count(':')
            clean_name = param.replace('ko_', '')

            epistasis[clean_name] = {
                'gamma': fit.params[param],
                'phi': np.exp(fit.params[param]),
                'se': fit.bse[param],
                'pvalue': fit.pvalues[param],
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
    """Save results to files."""
    # Save pickle
    with open(output_dir / 'glm_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Save summary JSON
    summary = {}
    for model_name in ['model_a', 'model_b']:
        if model_name not in results:
            continue

        summary[model_name] = {}
        for trait in results[model_name]:
            if results[model_name][trait]:
                summary[model_name][trait] = {
                    'n_digenic': results[model_name][trait].get('n_digenic', 0),
                    'n_trigenic': results[model_name][trait].get('n_trigenic', 0),
                    'n_sig_digenic': results[model_name][trait].get('n_sig_digenic', 0),
                    'n_sig_trigenic': results[model_name][trait].get('n_sig_trigenic', 0),
                    'r_squared': results[model_name][trait].get('r_squared', None)
                }

    with open(output_dir / 'glm_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

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
    all_results = {'model_a': {}, 'model_b': {}}

    # Run Model A (primary)
    print("\n" + "="*60)
    print("MODEL A - Primary Model (Log-OLS with WT-differencing)")
    print("="*60)

    for trait_col in FFA_COLUMNS:
        print(f"\nProcessing {trait_col}...")
        result = model_a_log_ols(df, trait_col)
        if result:
            all_results['model_a'][trait_col] = result
            print(f"  Total: {result['n_digenic']} digenic, {result['n_trigenic']} trigenic")
            print(f"  Significant: {result['n_sig_digenic']} digenic, {result['n_sig_trigenic']} trigenic")
            if 'r_squared' in result:
                print(f"  R-squared: {result['r_squared']:.3f}")

    # Run Model B (robustness)
    print("\n" + "="*60)
    print("MODEL B - Robustness Check (GLM with log link)")
    print("="*60)

    for trait_col in FFA_COLUMNS:
        print(f"\nProcessing {trait_col}...")
        result = model_b_glm(df, trait_col)
        if result:
            all_results['model_b'][trait_col] = result
            print(f"  Total: {result['n_digenic']} digenic, {result['n_trigenic']} trigenic")
            print(f"  Significant: {result['n_sig_digenic']} digenic, {result['n_sig_trigenic']} trigenic")

    # Save results
    print("\nSaving results...")
    save_results(all_results)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for model_name in ['model_a', 'model_b']:
        print(f"\n{model_name.upper()}:")
        total_dig = sum(r.get('n_digenic', 0) for r in all_results[model_name].values())
        total_tri = sum(r.get('n_trigenic', 0) for r in all_results[model_name].values())
        sig_dig = sum(r.get('n_sig_digenic', 0) for r in all_results[model_name].values())
        sig_tri = sum(r.get('n_sig_trigenic', 0) for r in all_results[model_name].values())

        print(f"  Total: {total_dig} digenic, {total_tri} trigenic")
        if total_dig > 0 or total_tri > 0:
            print(f"  Significant: {sig_dig} digenic ({100*sig_dig/max(total_dig,1):.1f}%), "
                  f"{sig_tri} trigenic ({100*sig_tri/max(total_tri,1):.1f}%)")

    print("\nAnalysis complete!")
    return all_results


if __name__ == "__main__":
    results = main()