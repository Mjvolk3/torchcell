#!/usr/bin/env python3

"""
Model C: Composition (CLR) analysis for FFA production data.
Separates capacity (total production) vs mixture (which FFAs are produced).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
from dotenv import load_dotenv
import pickle

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

# Individual FFA columns (5 chains, not including total)
FFA_CHAINS = ["C140", "C160", "C180", "C161", "C181"]

# TF genes and their single-letter abbreviations
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

            # Only add if not NaN (all FFAs present or all absent for a replicate)
            if not np.isnan(row_data['C140']):
                data_list.append(row_data)

    df = pd.DataFrame(data_list)

    # Parse genotype to identify knockouts
    for abbr, gene in TF_GENES.items():
        df[f'ko_{gene}'] = 0

    # Parse genotypes
    for idx, row in df.iterrows():
        genotype = row['genotype']

        if 'wt' in genotype.lower():
            continue
        elif '4Δ' in genotype:
            parts = genotype.split()
            if len(parts) > 0:
                ko_letter = parts[0].strip()
                if ko_letter in TF_GENES:
                    gene = TF_GENES[ko_letter]
                    df.at[idx, f'ko_{gene}'] = 1
        elif '5Δ' in genotype:
            parts = genotype.split()
            if len(parts) > 0 and '-' in parts[0]:
                ko_letters = parts[0].split('-')
                for letter in ko_letters:
                    if letter in TF_GENES:
                        gene = TF_GENES[letter]
                        df.at[idx, f'ko_{gene}'] = 1
        elif '6Δ' in genotype:
            parts = genotype.split()
            if len(parts) > 0 and '-' in parts[0]:
                ko_letters = parts[0].split('-')
                for letter in ko_letters:
                    if letter in TF_GENES:
                        gene = TF_GENES[letter]
                        df.at[idx, f'ko_{gene}'] = 1

    # Identify reference strain
    df['is_reference'] = df['genotype'].str.contains('wt', case=False, na=False)

    print(f"Reference strain: wild type (wt BY4741)")
    print(f"Reference observations: {df['is_reference'].sum()}")

    return df


def create_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create design matrix with main effects and interactions."""
    X = pd.DataFrame(index=df.index)

    # Get KO columns
    ko_cols = [col for col in df.columns if col.startswith("ko_")]
    ko_cols = [col for col in ko_cols if df[col].sum() > 0]

    # Add intercept
    X['intercept'] = 1

    # Add main effects
    for col in ko_cols:
        X[col] = df[col]

    # Add pairwise interactions (digenic)
    for col1, col2 in combinations(ko_cols, 2):
        if (df[col1] * df[col2]).sum() > 0:
            X[f"{col1}:{col2}"] = df[col1] * df[col2]

    # Add three-way interactions (trigenic)
    for col1, col2, col3 in combinations(ko_cols, 3):
        if (df[col1] * df[col2] * df[col3]).sum() > 0:
            X[f"{col1}:{col2}:{col3}"] = df[col1] * df[col2] * df[col3]

    return X


def compute_clr_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CLR (Centered Log-Ratio) transformation for compositional data.

    For each FFA chain i:
    c^(i)_{g,r} = log(p^(i)_{g,r}) - (1/5) * sum_{k=1}^5 log(p^(k)_{g,r})

    where p^(i)_{g,r} = y^(i)_{g,r} / y^(tot)_{g,r}
    """
    clr_df = df.copy()

    # Calculate proportions for each FFA
    for ffa in FFA_CHAINS:
        clr_df[f'{ffa}_prop'] = (df[ffa] + DELTA) / (df['Total Titer'] + 5 * DELTA)

    # Calculate geometric mean of proportions (in log space)
    log_props = []
    for ffa in FFA_CHAINS:
        log_props.append(np.log(clr_df[f'{ffa}_prop']))

    log_geometric_mean = sum(log_props) / len(FFA_CHAINS)

    # Calculate CLR components
    for ffa in FFA_CHAINS:
        clr_df[f'{ffa}_clr'] = np.log(clr_df[f'{ffa}_prop']) - log_geometric_mean

    return clr_df


def model_c_clr_single_chain(df: pd.DataFrame, chain: str) -> Dict:
    """
    Model C for a single FFA chain using CLR transformation.
    Analyzes composition-only epistasis.
    """
    results = {}

    # Compute CLR transformation
    clr_df = compute_clr_transform(df)

    # Get reference values per replicate
    ref_df = clr_df[clr_df['is_reference']].copy()
    if len(ref_df) == 0:
        print(f"Warning: No reference strain found for chain {chain}")
        return {}

    ref_means = ref_df.groupby('replicate')[f'{chain}_clr'].mean()

    # Prepare model data
    model_df = clr_df[~clr_df['is_reference']].copy()

    # Apply differencing from reference CLR
    model_df['clr_response'] = np.nan
    for rep in ref_means.index:
        mask = model_df['replicate'] == rep
        if mask.sum() > 0:
            model_df.loc[mask, 'clr_response'] = (
                model_df.loc[mask, f'{chain}_clr'] - ref_means[rep]
            )

    model_df = model_df.dropna(subset=['clr_response'])

    if len(model_df) == 0:
        print(f"Warning: No valid data after CLR differencing for {chain}")
        return {}

    # Create design matrix
    X = create_design_matrix(model_df)

    # Add replicate effects
    for rep in model_df['replicate'].unique():
        if rep != 1:
            X[f'rep_{rep}'] = (model_df['replicate'] == rep).astype(int)

    y = model_df['clr_response']

    # Fit OLS model on CLR components
    try:
        model = sm.OLS(y, X)
        fit = model.fit()
    except Exception as e:
        print(f"Error fitting CLR model for {chain}: {e}")
        return {}

    # Store results
    results['model'] = fit
    results['coefficients'] = fit.params
    results['std_errors'] = fit.bse
    results['pvalues'] = fit.pvalues
    results['r_squared'] = fit.rsquared
    results['adj_r_squared'] = fit.rsquared_adj

    # Extract composition-only epistatic interactions
    epistasis = {}

    for param in fit.params.index:
        if ':' in param and 'rep_' not in param:
            n_interactions = param.count(':')
            clean_name = param.replace('ko_', '')

            epistasis[clean_name] = {
                'E_mix': fit.params[param],  # Composition-only epistasis
                'phi_mix': np.exp(fit.params[param]),  # Composition epistatic fold
                'se': fit.bse[param],
                'pvalue': fit.pvalues[param],
                'ci_lower': fit.conf_int().loc[param, 0],
                'ci_upper': fit.conf_int().loc[param, 1],
                'order': 'digenic' if n_interactions == 1 else 'trigenic'
            }

    results['epistasis'] = epistasis
    results['chain'] = chain

    # Count interactions
    results['n_digenic'] = sum(1 for v in epistasis.values() if v['order'] == 'digenic')
    results['n_trigenic'] = sum(1 for v in epistasis.values() if v['order'] == 'trigenic')
    results['n_sig_digenic'] = sum(1 for v in epistasis.values()
                                   if v['order'] == 'digenic' and v['pvalue'] < 0.05)
    results['n_sig_trigenic'] = sum(1 for v in epistasis.values()
                                    if v['order'] == 'trigenic' and v['pvalue'] < 0.05)

    return results


def model_c_total_capacity(df: pd.DataFrame) -> Dict:
    """
    Model C for total titer (capacity analysis).
    Same as Model A but specifically for total titer.
    """
    results = {}
    trait_col = 'Total Titer'

    # Get reference values per replicate
    ref_df = df[df['is_reference']].copy()
    if len(ref_df) == 0:
        print("Warning: No reference strain found for total titer")
        return {}

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

    model_df = model_df.dropna(subset=['log_response'])

    if len(model_df) == 0:
        print("Warning: No valid data after WT-differencing for total titer")
        return {}

    # Create design matrix
    X = create_design_matrix(model_df)

    # Add replicate effects
    for rep in model_df['replicate'].unique():
        if rep != 1:
            X[f'rep_{rep}'] = (model_df['replicate'] == rep).astype(int)

    y = model_df['log_response']

    # Fit OLS model
    try:
        model = sm.OLS(y, X)
        fit = model.fit()
    except Exception as e:
        print(f"Error fitting model for total titer: {e}")
        return {}

    # Store results
    results['model'] = fit
    results['coefficients'] = fit.params
    results['std_errors'] = fit.bse
    results['pvalues'] = fit.pvalues
    results['r_squared'] = fit.rsquared
    results['adj_r_squared'] = fit.rsquared_adj

    # Extract capacity epistatic interactions
    epistasis = {}

    for param in fit.params.index:
        if ':' in param and 'rep_' not in param:
            n_interactions = param.count(':')
            clean_name = param.replace('ko_', '')

            epistasis[clean_name] = {
                'E_tot': fit.params[param],  # Total capacity epistasis
                'phi_tot': np.exp(fit.params[param]),  # Capacity epistatic fold
                'se': fit.bse[param],
                'pvalue': fit.pvalues[param],
                'ci_lower': fit.conf_int().loc[param, 0],
                'ci_upper': fit.conf_int().loc[param, 1],
                'order': 'digenic' if n_interactions == 1 else 'trigenic'
            }

    results['epistasis'] = epistasis

    # Count interactions
    results['n_digenic'] = sum(1 for v in epistasis.values() if v['order'] == 'digenic')
    results['n_trigenic'] = sum(1 for v in epistasis.values() if v['order'] == 'trigenic')
    results['n_sig_digenic'] = sum(1 for v in epistasis.values()
                                   if v['order'] == 'digenic' and v['pvalue'] < 0.05)
    results['n_sig_trigenic'] = sum(1 for v in epistasis.values()
                                    if v['order'] == 'trigenic' and v['pvalue'] < 0.05)

    return results


def combine_capacity_and_composition(total_results: Dict, chain_results: Dict, chain: str) -> Dict:
    """
    Combine capacity (total) and composition (CLR) results.

    According to the mathematical relationship:
    E_S(log y^(i)) = E^(tot)_S + E^(mix,i)_S

    Where:
    - E^(tot)_S: capacity-only epistasis (from total titer)
    - E^(mix,i)_S: composition-only epistasis (from CLR)
    - E_S(log y^(i)): combined effect on individual chain i
    """
    combined = {}

    # Get all interaction terms
    all_terms = set()
    if 'epistasis' in total_results:
        all_terms.update(total_results['epistasis'].keys())
    if 'epistasis' in chain_results:
        all_terms.update(chain_results['epistasis'].keys())

    for term in all_terms:
        combined[term] = {'chain': chain}

        # Get capacity component (E_tot)
        if term in total_results.get('epistasis', {}):
            tot_data = total_results['epistasis'][term]
            combined[term]['E_tot'] = tot_data['E_tot']
            combined[term]['phi_tot'] = tot_data['phi_tot']
            combined[term]['pvalue_tot'] = tot_data['pvalue']
        else:
            combined[term]['E_tot'] = 0
            combined[term]['phi_tot'] = 1
            combined[term]['pvalue_tot'] = 1

        # Get composition component (E_mix)
        if term in chain_results.get('epistasis', {}):
            mix_data = chain_results['epistasis'][term]
            combined[term]['E_mix'] = mix_data['E_mix']
            combined[term]['phi_mix'] = mix_data['phi_mix']
            combined[term]['pvalue_mix'] = mix_data['pvalue']
            combined[term]['order'] = mix_data['order']
        else:
            combined[term]['E_mix'] = 0
            combined[term]['phi_mix'] = 1
            combined[term]['pvalue_mix'] = 1
            # Determine order from term structure
            n_interactions = term.count(':')
            combined[term]['order'] = 'digenic' if n_interactions == 1 else 'trigenic'

        # Combined effect
        combined[term]['E_combined'] = combined[term]['E_tot'] + combined[term]['E_mix']
        combined[term]['phi_combined'] = np.exp(combined[term]['E_combined'])

        # Classification (using thresholds)
        E_tot_abs = abs(combined[term]['E_tot'])
        E_mix_abs = abs(combined[term]['E_mix'])

        if E_mix_abs < 0.1 and E_tot_abs > 0.1:
            combined[term]['type'] = 'capacity-only'
        elif E_tot_abs < 0.1 and E_mix_abs > 0.1:
            combined[term]['type'] = 'composition-only'
        elif E_tot_abs > 0.1 and E_mix_abs > 0.1:
            combined[term]['type'] = 'both'
        else:
            combined[term]['type'] = 'neither'

    return combined


def main():
    """Main execution function for Model C (CLR)."""
    print("Loading FFA data...")

    # Use the correct data file location
    file_path = Path("/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/ffa_xue2025/raw/Supplementary Data 1_Raw titers.xlsx")

    if not file_path.exists():
        print(f"Error: Could not find data file at {file_path}")
        return

    df = load_ffa_data_with_replicates(file_path)

    print(f"\nLoaded {len(df)} observations")
    print(f"Unique genotypes: {df['genotype'].nunique()}")

    # Check KO columns
    ko_cols = [col for col in df.columns if col.startswith("ko_")]
    for col in ko_cols:
        n_ko = df[col].sum()
        if n_ko > 0:
            print(f"  {col}: {n_ko} observations")

    # Run Model C
    print("\n" + "="*60)
    print("MODEL C - Composition (CLR) Analysis")
    print("="*60)

    # First, analyze total capacity
    print("\nAnalyzing total capacity...")
    total_results = model_c_total_capacity(df)

    if total_results:
        print(f"  Total capacity interactions: {total_results['n_digenic']} digenic, "
              f"{total_results['n_trigenic']} trigenic")
        print(f"  Significant: {total_results['n_sig_digenic']} digenic, "
              f"{total_results['n_sig_trigenic']} trigenic")
        print(f"  R-squared: {total_results['r_squared']:.3f}")

    # Then, analyze composition for each chain
    model_c_results = {'total': total_results}

    for chain in FFA_CHAINS:
        print(f"\nAnalyzing composition for {chain}...")
        chain_results = model_c_clr_single_chain(df, chain)

        if chain_results:
            model_c_results[chain] = chain_results
            print(f"  Composition interactions: {chain_results['n_digenic']} digenic, "
                  f"{chain_results['n_trigenic']} trigenic")
            print(f"  Significant: {chain_results['n_sig_digenic']} digenic, "
                  f"{chain_results['n_sig_trigenic']} trigenic")
            print(f"  R-squared: {chain_results['r_squared']:.3f}")

            # Combine capacity and composition
            combined = combine_capacity_and_composition(total_results, chain_results, chain)
            model_c_results[f'{chain}_combined'] = combined

            # Count by type
            capacity_only = sum(1 for v in combined.values() if v['type'] == 'capacity-only')
            composition_only = sum(1 for v in combined.values() if v['type'] == 'composition-only')
            both = sum(1 for v in combined.values() if v['type'] == 'both')

            print(f"  Classification: {capacity_only} capacity-only, "
                  f"{composition_only} composition-only, {both} both")

    # Save results
    print("\nSaving Model C results...")
    with open(RESULTS_DIR / 'model_c_clr_results.pkl', 'wb') as f:
        pickle.dump({'model_c': model_c_results}, f)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - MODEL C (CLR)")
    print("="*60)

    print("\nTotal Capacity:")
    print(f"  Significant interactions: {total_results['n_sig_digenic']} digenic, "
          f"{total_results['n_sig_trigenic']} trigenic")

    for chain in FFA_CHAINS:
        if chain in model_c_results:
            result = model_c_results[chain]
            print(f"\n{chain} Composition:")
            print(f"  Significant effects: {result['n_sig_digenic']} digenic, "
                  f"{result['n_sig_trigenic']} trigenic")

    print(f"\nResults saved to {RESULTS_DIR}")
    print("Model C (CLR) analysis complete!")


if __name__ == "__main__":
    main()