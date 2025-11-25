# experiments/008-xue-ffa/scripts/additive_free_fatty_acid_interactions
# [[experiments.008-xue-ffa.scripts.additive_free_fatty_acid_interactions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/additive_free_fatty_acid_interactions
# Test file: experiments/008-xue-ffa/scripts/test_additive_free_fatty_acid_interactions.py


import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
if not ASSET_IMAGES_DIR:
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
    ASSET_IMAGES_DIR = osp.join(PROJECT_ROOT, "notes/assets/images")
os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

# Results directory
RESULTS_DIR = "/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Apply torchcell style
STYLE_PATH = "/Users/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle"
if osp.exists(STYLE_PATH):
    plt.style.use(STYLE_PATH)
else:
    # Fallback style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")


def load_ffa_data(file_path):
    """Load FFA data from Excel file, returning both averaged and replicate data."""
    # Read abbreviations
    abbrev_df = pd.read_excel(file_path, sheet_name='Abbreviations')
    abbreviations = dict(zip(abbrev_df.iloc[:, 0], abbrev_df.iloc[:, 1]))

    # Read raw titer data - get all replicates
    raw_df = pd.read_excel(file_path, sheet_name='raw-titer (mg-L)')

    # Get replicate columns for each FFA (columns 1-15 for raw replicates)
    replicate_data = {}
    replicate_data['C14:0'] = raw_df.iloc[:, [0, 1, 2, 3]].copy()  # Genotype + 3 replicates
    replicate_data['C16:0'] = raw_df.iloc[:, [0, 4, 5, 6]].copy()
    replicate_data['C18:0'] = raw_df.iloc[:, [0, 7, 8, 9]].copy()
    replicate_data['C16:1'] = raw_df.iloc[:, [0, 10, 11, 12]].copy()
    replicate_data['C18:1'] = raw_df.iloc[:, [0, 13, 14, 15]].copy()

    # Compute means from replicates for individual FFAs
    averaged_df = raw_df.iloc[:, [0]].copy()  # Start with genotype column

    for ffa in ['C14:0', 'C16:0', 'C18:0', 'C16:1', 'C18:1']:
        rep_df = replicate_data[ffa]
        # Compute mean of replicates (columns 1-3 are the data)
        averaged_df[ffa] = rep_df.iloc[:, 1:4].mean(axis=1)

    # Calculate Total Titer as sum of individual FFA means
    averaged_df['Total Titer'] = (
        averaged_df['C14:0'] + averaged_df['C16:0'] + averaged_df['C18:0'] +
        averaged_df['C16:1'] + averaged_df['C18:1']
    )

    # For replicate data, calculate Total Titer for each replicate
    total_titer_reps = np.zeros((len(raw_df), 4))  # genotype + 3 replicates
    total_titer_reps[:, 0] = np.arange(len(raw_df))  # Row indices
    for rep_idx in range(3):
        total_titer_reps[:, rep_idx + 1] = (
            replicate_data['C14:0'].iloc[:, rep_idx + 1] +
            replicate_data['C16:0'].iloc[:, rep_idx + 1] +
            replicate_data['C18:0'].iloc[:, rep_idx + 1] +
            replicate_data['C16:1'].iloc[:, rep_idx + 1] +
            replicate_data['C18:1'].iloc[:, rep_idx + 1]
        )
    replicate_data['Total Titer'] = pd.DataFrame(total_titer_reps)
    replicate_data['Total Titer'].iloc[:, 0] = raw_df.iloc[:, 0]  # Set genotype column

    # Rename first column
    averaged_df.columns = ['Genotype'] + list(averaged_df.columns[1:])

    # Store replicate data in a structured format for error propagation
    # Each entry: genotype -> {ffa_type -> [rep1, rep2, rep3]}
    replicate_dict = {}
    for idx in range(len(raw_df)):
        genotype = raw_df.iloc[idx, 0]
        replicate_dict[genotype] = {}
        for ffa in ['C14:0', 'C16:0', 'C18:0', 'C16:1', 'C18:1']:
            rep_df = replicate_data[ffa]
            replicate_dict[genotype][ffa] = rep_df.iloc[idx, 1:4].values.astype(float)
        # Calculate Total Titer replicates as sum
        replicate_dict[genotype]['Total Titer'] = replicate_data['Total Titer'].iloc[idx, 1:4].values.astype(float)

    return averaged_df, abbreviations, replicate_dict


def normalize_by_reference(df, replicate_dict=None):
    """Normalize all values by positive control strain (+ve Ctrl) mean.

    This computes fitness as f = strain_value / reference_mean for each FFA type.
    The +ve Ctrl is the positive control: POX1-FAA1-FAA4 (3Δ metabolic genes).
    All other strains have this baseline plus additional TF deletions.
    Also normalizes replicate data if provided.
    """
    # Find the +ve Ctrl row
    ctrl_mask = df.iloc[:, 0].str.contains('+ve Ctrl', na=False, regex=False)
    if not ctrl_mask.any():
        # Fallback: check for variations
        ctrl_mask = df.iloc[:, 0].str.contains('ve Ctrl', na=False, regex=False)

    if not ctrl_mask.any():
        raise ValueError("Could not find +ve Ctrl row in the data")

    ref_idx = ctrl_mask.idxmax()
    reference_strain = df.iloc[ref_idx, 0]  # Get the actual genotype name
    ref_values = df.iloc[ref_idx, 1:].values.astype(float)

    # Create normalized dataframe
    normalized_df = df.copy()

    # Normalize each column by its reference value
    for col_idx in range(1, len(df.columns)):
        if ref_values[col_idx - 1] != 0:
            normalized_df.iloc[:, col_idx] = df.iloc[:, col_idx].astype(float) / ref_values[col_idx - 1]
        else:
            normalized_df.iloc[:, col_idx] = np.nan

    # Normalize replicate data if provided
    normalized_replicates = None
    if replicate_dict:
        normalized_replicates = {}
        # Get the reference genotype from first row
        ref_replicates = replicate_dict[reference_strain]

        for genotype, ffa_data in replicate_dict.items():
            normalized_replicates[genotype] = {}
            for ffa_idx, (ffa_name, replicates) in enumerate(ffa_data.items()):
                ref_mean = ref_replicates[ffa_name].mean()
                if ref_mean != 0:
                    normalized_replicates[genotype][ffa_name] = replicates / ref_mean
                else:
                    normalized_replicates[genotype][ffa_name] = np.array([np.nan, np.nan, np.nan])

    print(f"Positive control normalization - {reference_strain} values (means): {ref_values}")
    print(f"After normalization, positive control should be all 1.0s: {normalized_df.iloc[ref_idx, 1:].values}")

    return normalized_df, normalized_replicates


def parse_genotype(genotype_str, abbreviations=None, reference_strain=None):
    """Parse genotype string to extract TF deletions beyond the metabolic baseline.

    The reference strain (POX1-FAA1-FAA4) is the positive control baseline.
    Returns list of additional transcription factor deletions.
    """
    if pd.isna(genotype_str) or genotype_str == 'wt BY4741' or genotype_str == 'BY4741':
        return []

    # Check if this is the positive control (+ve)
    if '+ve' in str(genotype_str):
        return []

    # Check if this is the reference strain
    genotype_clean = str(genotype_str).strip()
    if reference_strain and genotype_clean == str(reference_strain).strip():
        return []

    genotype_str = str(genotype_str)
    genes = []

    # Check if genotype contains abbreviations (like P-S-Y-A 4Δ or P-S-Y-B-C 5Δ)
    if 'Δ' in genotype_str:
        # Extract the letter part before the number and delta
        parts = genotype_str.split()
        if len(parts) >= 2:
            letter_part = parts[0]  # Get 'P-S-Y-A' from 'P-S-Y-A 4Δ'

            # Split by dash to get all letters
            letters = letter_part.split('-')

            # ALL letters represent TFs - no filtering needed!
            # The metabolic genes (POX1-FAA1-FAA4) are implicit in ALL strains

            # Convert abbreviations to TF gene names if available
            if abbreviations and letters:
                # Create reverse mapping from letter to TF name
                letter_to_gene = {v: k for k, v in abbreviations.items()}
                # Add F = PKH1 if missing from abbreviations
                if 'F' not in letter_to_gene:
                    letter_to_gene['F'] = 'PKH1'
                genes = [letter_to_gene.get(letter, letter) for letter in letters]
            else:
                genes = letters
    else:
        # Handle other formats if they exist
        parts = genotype_str.split('\u0394')
        for part in parts:
            part = part.strip()
            if part and not part.startswith('#'):
                gene = part.split()[0] if ' ' in part else part
                if gene:
                    genes.append(gene)

    return genes


def compute_se_pvalue(observed_interaction, se_interaction, df=2):
    """
    Compute p-value using standard error and t-distribution.

    Parameters:
    -----------
    observed_interaction : float
        The observed interaction value
    se_interaction : float
        Standard error of the interaction
    df : int
        Degrees of freedom (default 2 for 3 replicates - 1)

    Returns:
    --------
    p_value : float
        Two-tailed p-value from t-test
    """
    if se_interaction == 0 or np.isnan(se_interaction):
        return np.nan

    # t-statistic under null hypothesis that interaction = 0
    t_stat = observed_interaction / se_interaction

    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return p_value


def compute_additive_interactions_with_error_propagation(normalized_df, normalized_replicates, abbreviations=None):
    """
    Compute TF interactions using STANDARD ADDITIVE NULL model with p-values using error propagation from replicate standard errors.

    STANDARD ADDITIVE NULL MODEL (sum-of-deviations with inclusion-exclusion):
    - Digenic: ε_ij = f_ij - f_i - f_j + 1
    - Trigenic: E_ijk = f_ijk - (f_ij + f_ik + f_jk) + (f_i + f_j + f_k) - 1

    Rationale: Each mutation contributes an absolute change from baseline (WT=1).
    Combinations are the sum of those changes plus the baseline.
    The "+1" and "-1" terms come from the baseline contribution.

    Note: All interactions are computed relative to positive control (POX1-FAA1-FAA4).
    Single mutants: positive control + 1 TF deletion (4Δ total)
    Double mutants: positive control + 2 TF deletions (5Δ total)
    Triple mutants: positive control + 3 TF deletions (6Δ total)
    """
    # Results dictionaries
    digenic_interactions = {}
    digenic_sd = {}  # Standard deviations
    digenic_se = {}  # Standard errors
    digenic_pvalues = {}
    trigenic_interactions = {}
    trigenic_sd = {}  # Standard deviations
    trigenic_se = {}  # Standard errors
    trigenic_pvalues = {}

    # Get single, double, and triple mutants with their SDs, SEs and replicate counts
    single_mutants = {}
    single_sd = {}  # Standard deviations
    single_se = {}  # Standard errors
    single_n_reps = {}  # Track number of valid replicates
    double_mutants = {}
    double_sd = {}  # Standard deviations
    double_se = {}  # Standard errors
    double_n_reps = {}
    triple_mutants = {}
    triple_sd = {}  # Standard deviations
    triple_se = {}  # Standard errors
    triple_n_reps = {}

    # Process all genotypes
    # Get reference strain from first row for parse_genotype
    reference_strain = list(normalized_replicates.keys())[0]
    for genotype, ffa_data in normalized_replicates.items():
        genes = parse_genotype(genotype, abbreviations, reference_strain=reference_strain)

        # Calculate means, standard deviations, standard errors, and replicate counts for each FFA
        means = []
        sds = []  # Standard deviations
        ses = []  # Standard errors
        n_reps = []
        for ffa_name in ['C14:0', 'C16:0', 'C18:0', 'C16:1', 'C18:1', 'Total Titer']:
            reps = ffa_data[ffa_name]
            # Filter out NaN values to handle missing replicates
            valid_reps = reps[~np.isnan(reps)]
            n_valid = len(valid_reps)

            if n_valid > 1:
                mean_val = np.mean(valid_reps)
                sd_val = np.std(valid_reps, ddof=1)  # Standard deviation
                se_val = sd_val / np.sqrt(n_valid)  # Standard error
            elif n_valid == 1:
                mean_val = valid_reps[0]
                sd_val = np.nan  # Cannot calculate SD with only 1 replicate
                se_val = np.nan  # Cannot calculate SE with only 1 replicate
            else:
                mean_val = np.nan
                sd_val = np.nan
                se_val = np.nan
            means.append(mean_val)
            sds.append(sd_val)
            ses.append(se_val)
            n_reps.append(n_valid)

        means = np.array(means)
        sds = np.array(sds)
        ses = np.array(ses)
        n_reps = np.array(n_reps)

        if len(genes) == 1:
            single_mutants[genes[0]] = means
            single_sd[genes[0]] = sds
            single_se[genes[0]] = ses
            single_n_reps[genes[0]] = n_reps
        elif len(genes) == 2:
            double_mutants[tuple(sorted(genes))] = means
            double_sd[tuple(sorted(genes))] = sds
            double_se[tuple(sorted(genes))] = ses
            double_n_reps[tuple(sorted(genes))] = n_reps
        elif len(genes) == 3:
            triple_mutants[tuple(sorted(genes))] = means
            triple_sd[tuple(sorted(genes))] = sds
            triple_se[tuple(sorted(genes))] = ses
            triple_n_reps[tuple(sorted(genes))] = n_reps

    # Compute digenic interactions with error propagation (ADDITIVE MODEL)
    print("Computing digenic interactions with ADDITIVE model and error propagation...")
    for (gene1, gene2), f_ij in tqdm(double_mutants.items(), desc="Digenic interactions (additive)"):
        if gene1 in single_mutants and gene2 in single_mutants:
            f_i = single_mutants[gene1]
            f_j = single_mutants[gene2]

            # Get standard deviations and errors
            sd_i = single_sd[gene1]
            sd_j = single_sd[gene2]
            sd_ij = double_sd[(gene1, gene2)]
            se_i = single_se[gene1]
            se_j = single_se[gene2]
            se_ij = double_se[(gene1, gene2)]

            # STANDARD ADDITIVE NULL: ε_ij = f_ij - f_i - f_j + 1
            # The "+1" is the baseline (WT) contribution
            epsilon_ij = f_ij - f_i - f_j + 1
            digenic_interactions[(gene1, gene2)] = epsilon_ij

            # SD propagation for ε_ij = f_ij - f_i - f_j + 1
            # SD(ε_ij) = sqrt(SD(f_ij)^2 + SD(f_i)^2 + SD(f_j)^2 + SD(1)^2)
            # Since 1 is the normalized WT value (constant), SD(1) = 0
            sd_epsilon_ij = np.sqrt(
                sd_ij**2 + sd_i**2 + sd_j**2
            )
            digenic_sd[(gene1, gene2)] = sd_epsilon_ij

            # SE propagation for ε_ij = f_ij - f_i - f_j + 1
            # SE(ε_ij) = sqrt(SE(f_ij)^2 + SE(f_i)^2 + SE(f_j)^2 + SE(1)^2)
            # Since 1 is the normalized WT value (constant), SE(1) = 0
            se_epsilon_ij = np.sqrt(
                se_ij**2 + se_i**2 + se_j**2
            )
            digenic_se[(gene1, gene2)] = se_epsilon_ij

            # Compute p-values using t-test with appropriate degrees of freedom
            pvalues = []
            n_ij = double_n_reps[(gene1, gene2)]
            n_i = single_n_reps[gene1]
            n_j = single_n_reps[gene2]

            for idx in range(len(epsilon_ij)):
                if not np.isnan(epsilon_ij[idx]) and not np.isnan(se_epsilon_ij[idx]):
                    # Use minimum number of replicates minus 1 for df
                    # This is conservative but appropriate for error propagation
                    min_reps = min(n_i[idx], n_j[idx], n_ij[idx])
                    df = max(1, min_reps - 1)  # Ensure df is at least 1
                    pval = compute_se_pvalue(epsilon_ij[idx], se_epsilon_ij[idx], df=df)
                    pvalues.append(pval)
                else:
                    pvalues.append(np.nan)
            digenic_pvalues[(gene1, gene2)] = np.array(pvalues)

    # Compute trigenic interactions with error propagation (ADDITIVE MODEL)
    print("Computing trigenic interactions with ADDITIVE model and error propagation...")
    for (gene1, gene2, gene3), f_ijk in tqdm(triple_mutants.items(), desc="Trigenic interactions (additive)"):
        # Check if all components exist
        if not all(g in single_mutants for g in [gene1, gene2, gene3]):
            continue

        pair1 = tuple(sorted([gene1, gene2]))
        pair2 = tuple(sorted([gene1, gene3]))
        pair3 = tuple(sorted([gene2, gene3]))

        if not all(pair in double_mutants for pair in [pair1, pair2, pair3]):
            continue

        f_i = single_mutants[gene1]
        f_j = single_mutants[gene2]
        f_k = single_mutants[gene3]
        f_ij = double_mutants[pair1]
        f_ik = double_mutants[pair2]
        f_jk = double_mutants[pair3]

        # Get standard deviations
        sd_i = single_sd[gene1]
        sd_j = single_sd[gene2]
        sd_k = single_sd[gene3]
        sd_ij = double_sd[pair1]
        sd_ik = double_sd[pair2]
        sd_jk = double_sd[pair3]
        sd_ijk = triple_sd[(gene1, gene2, gene3)]

        # Get standard errors
        se_i = single_se[gene1]
        se_j = single_se[gene2]
        se_k = single_se[gene3]
        se_ij = double_se[pair1]
        se_ik = double_se[pair2]
        se_jk = double_se[pair3]
        se_ijk = triple_se[(gene1, gene2, gene3)]

        # STANDARD ADDITIVE NULL WITH INCLUSION-EXCLUSION:
        # E_ijk = f_ijk - (f_ij + f_ik + f_jk) + (f_i + f_j + f_k) - 1
        # This cleanly removes lower-order terms (pairs, singles, baseline)
        E_ijk = f_ijk - (f_ij + f_ik + f_jk) + (f_i + f_j + f_k) - 1
        trigenic_interactions[(gene1, gene2, gene3)] = E_ijk

        # SD propagation for E_ijk
        # Coefficients: (+1) for f_ijk, (-1) for each pair, (+1) for each single, (-1) for baseline
        # SD(E_ijk) = sqrt(SD(f_ijk)^2 + SD(f_ij)^2 + SD(f_ik)^2 + SD(f_jk)^2 + SD(f_i)^2 + SD(f_j)^2 + SD(f_k)^2 + SD(1)^2)
        # Since 1 is the normalized WT value (constant), SD(1) = 0
        sd_E_ijk = np.sqrt(
            sd_ijk**2 + sd_ij**2 + sd_ik**2 + sd_jk**2 + sd_i**2 + sd_j**2 + sd_k**2
        )
        trigenic_sd[(gene1, gene2, gene3)] = sd_E_ijk

        # SE propagation for E_ijk
        # SE(E_ijk) = sqrt(SE(f_ijk)^2 + SE(f_ij)^2 + SE(f_ik)^2 + SE(f_jk)^2 + SE(f_i)^2 + SE(f_j)^2 + SE(f_k)^2 + SE(1)^2)
        # Since 1 is the normalized WT value (constant), SE(1) = 0
        se_E_ijk = np.sqrt(
            se_ijk**2 + se_ij**2 + se_ik**2 + se_jk**2 + se_i**2 + se_j**2 + se_k**2
        )
        trigenic_se[(gene1, gene2, gene3)] = se_E_ijk

        # Compute p-values with appropriate degrees of freedom
        pvalues = []
        n_ijk = triple_n_reps[(gene1, gene2, gene3)]
        n_ij = double_n_reps[pair1]
        n_ik = double_n_reps[pair2]
        n_jk = double_n_reps[pair3]
        n_i = single_n_reps[gene1]
        n_j = single_n_reps[gene2]
        n_k = single_n_reps[gene3]

        for idx in range(len(E_ijk)):
            if not np.isnan(E_ijk[idx]) and not np.isnan(se_E_ijk[idx]):
                # Use minimum number of replicates minus 1 for df
                # Include single mutant replicates since they contribute to E_ijk
                min_reps = min(n_i[idx], n_j[idx], n_k[idx], n_ij[idx], n_ik[idx], n_jk[idx], n_ijk[idx])
                df = max(1, min_reps - 1)
                pval = compute_se_pvalue(E_ijk[idx], se_E_ijk[idx], df=df)
                pvalues.append(pval)
            else:
                pvalues.append(np.nan)
        trigenic_pvalues[(gene1, gene2, gene3)] = np.array(pvalues)

    print(f"Found {len(digenic_interactions)} digenic TF interactions (additive model)")
    print(f"Found {len(trigenic_interactions)} trigenic TF interactions (additive model)")

    return (digenic_interactions, digenic_sd, digenic_se, digenic_pvalues,
            trigenic_interactions, trigenic_sd, trigenic_se, trigenic_pvalues,
            single_mutants, double_mutants, triple_mutants,
            single_sd, single_se, double_sd, double_se, triple_sd, triple_se)


def apply_fdr_correction(p_values_dict, method='fdr_bh'):
    """Apply FDR correction to a dictionary of p-values."""
    # Collect all p-values
    all_pvals = []
    keys = []
    indices = []

    for key, pval_array in p_values_dict.items():
        for idx, pval in enumerate(pval_array):
            if not np.isnan(pval):
                all_pvals.append(pval)
                keys.append(key)
                indices.append(idx)

    # Apply FDR correction
    if len(all_pvals) > 0:
        _, fdr_pvals, _, _ = multipletests(all_pvals, method=method, alpha=0.05)

        # Map back to original structure
        fdr_dict = {}
        for key in p_values_dict.keys():
            fdr_dict[key] = np.full_like(p_values_dict[key], np.nan)

        for i, (key, idx) in enumerate(zip(keys, indices)):
            fdr_dict[key][idx] = fdr_pvals[i]
    else:
        fdr_dict = {key: np.full_like(val, np.nan) for key, val in p_values_dict.items()}

    return fdr_dict


def save_interaction_results(digenic_interactions, digenic_sd, digenic_se, digenic_pvalues, digenic_fdr,
                            trigenic_interactions, trigenic_sd, trigenic_se, trigenic_pvalues, trigenic_fdr,
                            columns, results_dir):
    """Save ADDITIVE interaction results to CSV files with SD, SE, effect sizes and FDR-corrected p-values."""

    # Prepare digenic results
    digenic_rows = []
    for (gene1, gene2), interactions in digenic_interactions.items():
        pvalues = digenic_pvalues[(gene1, gene2)]
        fdr_pvals = digenic_fdr[(gene1, gene2)]
        sds = digenic_sd[(gene1, gene2)]
        ses = digenic_se[(gene1, gene2)]

        for idx, col_name in enumerate(columns):
            if idx < len(interactions) and not np.isnan(interactions[idx]):
                digenic_rows.append({
                    'gene_set': f"{gene1}_{gene2}",
                    'interaction_type': 'digenic',
                    'ffa_type': col_name,
                    'interaction_score': interactions[idx],
                    'standard_deviation': sds[idx],
                    'standard_error': ses[idx],
                    'p_value': pvalues[idx],
                    'fdr_corrected_p': fdr_pvals[idx],
                    'effect_size': abs(interactions[idx]),  # Absolute value for effect size
                    'significant_p05': pvalues[idx] < 0.05 if not np.isnan(pvalues[idx]) else False,
                    'significant_fdr05': fdr_pvals[idx] < 0.05 if not np.isnan(fdr_pvals[idx]) else False
                })

    # Prepare trigenic results
    trigenic_rows = []
    for (gene1, gene2, gene3), interactions in trigenic_interactions.items():
        pvalues = trigenic_pvalues[(gene1, gene2, gene3)]
        fdr_pvals = trigenic_fdr[(gene1, gene2, gene3)]
        sds = trigenic_sd[(gene1, gene2, gene3)]
        ses = trigenic_se[(gene1, gene2, gene3)]

        for idx, col_name in enumerate(columns):
            if idx < len(interactions) and not np.isnan(interactions[idx]):
                trigenic_rows.append({
                    'gene_set': f"{gene1}_{gene2}_{gene3}",
                    'interaction_type': 'trigenic',
                    'ffa_type': col_name,
                    'interaction_score': interactions[idx],
                    'standard_deviation': sds[idx],
                    'standard_error': ses[idx],
                    'p_value': pvalues[idx],
                    'fdr_corrected_p': fdr_pvals[idx],
                    'effect_size': abs(interactions[idx]),
                    'significant_p05': pvalues[idx] < 0.05 if not np.isnan(pvalues[idx]) else False,
                    'significant_fdr05': fdr_pvals[idx] < 0.05 if not np.isnan(fdr_pvals[idx]) else False
                })

    # Create DataFrames
    digenic_df = pd.DataFrame(digenic_rows)
    trigenic_df = pd.DataFrame(trigenic_rows)
    combined_df = pd.concat([digenic_df, trigenic_df], ignore_index=True)

    # Save to CSV with ADDITIVE prefix
    ts = timestamp()
    digenic_df.to_csv(osp.join(results_dir, f"additive_digenic_interactions_3_delta_normalized_{ts}.csv"), index=False)
    trigenic_df.to_csv(osp.join(results_dir, f"additive_trigenic_interactions_3_delta_normalized_{ts}.csv"), index=False)
    combined_df.to_csv(osp.join(results_dir, f"additive_all_interactions_3_delta_normalized_{ts}.csv"), index=False)

    # Save summary statistics
    summary_file = osp.join(results_dir, f"additive_interaction_summary_3_delta_normalized_{ts}.txt")
    with open(summary_file, 'w') as f:
        f.write("=== ADDITIVE Interaction Model Summary Statistics ===\n\n")
        f.write(f"Total interactions tested: {len(combined_df)}\n")
        f.write(f"  - Digenic: {len(digenic_df)}\n")
        f.write(f"  - Trigenic: {len(trigenic_df)}\n\n")

        f.write("Significant interactions (p < 0.05):\n")
        sig_di = digenic_df['significant_p05'].sum()
        sig_tri = trigenic_df['significant_p05'].sum()
        f.write(f"  - Digenic: {sig_di}/{len(digenic_df)} ({100*sig_di/len(digenic_df):.1f}%)\n")
        f.write(f"  - Trigenic: {sig_tri}/{len(trigenic_df)} ({100*sig_tri/len(trigenic_df):.1f}%)\n\n")

        f.write("Significant after FDR correction (q < 0.05):\n")
        sig_di_fdr = digenic_df['significant_fdr05'].sum()
        sig_tri_fdr = trigenic_df['significant_fdr05'].sum()
        f.write(f"  - Digenic: {sig_di_fdr}/{len(digenic_df)} ({100*sig_di_fdr/len(digenic_df):.1f}%)\n")
        f.write(f"  - Trigenic: {sig_tri_fdr}/{len(trigenic_df)} ({100*sig_tri_fdr/len(trigenic_df):.1f}%)\n\n")

        f.write("Effect size statistics:\n")
        f.write(f"  Digenic effect sizes: mean={digenic_df['effect_size'].mean():.2f}, "
                f"median={digenic_df['effect_size'].median():.2f}, "
                f"max={digenic_df['effect_size'].max():.2f}\n")
        f.write(f"  Trigenic effect sizes: mean={trigenic_df['effect_size'].mean():.2f}, "
                f"median={trigenic_df['effect_size'].median():.2f}, "
                f"max={trigenic_df['effect_size'].max():.2f}\n\n")

        # By FFA type
        f.write("By FFA type (p < 0.05 / total):\n")
        for ffa in columns:
            ffa_df = combined_df[combined_df['ffa_type'] == ffa]
            sig = ffa_df['significant_p05'].sum()
            f.write(f"  {ffa}: {sig}/{len(ffa_df)} ({100*sig/len(ffa_df):.1f}%)\n")

    print(f"\nResults saved to {results_dir}")
    print(f"Summary saved to {summary_file}")

    return combined_df


def plot_interaction_distributions_and_volcano(digenic_dict, digenic_se, digenic_pvalues,
                                              trigenic_dict, trigenic_se, trigenic_pvalues, columns):
    """Create combined distribution and volcano plots for ADDITIVE interactions by FFA identity."""

    # Use all FFA types including Total Titer
    ffa_types = columns  # All 6 columns

    # Create figure with 2 rows, 6 columns for all FFAs including Total Titer
    fig, axes = plt.subplots(2, 6, figsize=(26, 10))

    # Colors from torchcell.mplstyle palette
    digenic_color = '#7191A9'  # Light blue from palette
    trigenic_color = '#B73C39'  # Red from palette

    # Plot distributions (top row) and volcano plots (bottom row) for each FFA
    for idx, ffa in enumerate(ffa_types):
        ax_dist = axes[0, idx]  # Distribution plot
        ax_volc = axes[1, idx]  # Volcano plot

        # Collect digenic data for this FFA
        digenic_values = []
        digenic_errors = []
        digenic_pvals = []
        for (genes, interaction_values), se_values, pval_values in zip(
            digenic_dict.items(), digenic_se.values(), digenic_pvalues.values()):
            if idx < len(interaction_values):
                val = interaction_values[idx]
                se = se_values[idx] if genes in digenic_se else np.nan
                pval = pval_values[idx] if genes in digenic_pvalues else np.nan
                if not np.isnan(val):
                    digenic_values.append(val)
                    digenic_errors.append(se)
                    digenic_pvals.append(pval)

        # Collect trigenic data for this FFA
        trigenic_values = []
        trigenic_errors = []
        trigenic_pvals = []
        for (genes, interaction_values), se_values, pval_values in zip(
            trigenic_dict.items(), trigenic_se.values(), trigenic_pvalues.values()):
            if idx < len(interaction_values):
                val = interaction_values[idx]
                se = se_values[idx] if genes in trigenic_se else np.nan
                pval = pval_values[idx] if genes in trigenic_pvalues else np.nan
                if not np.isnan(val):
                    trigenic_values.append(val)
                    trigenic_errors.append(se)
                    trigenic_pvals.append(pval)

        # TOP ROW: Distribution plots
        if digenic_values or trigenic_values:
            # Determine common bins for both distributions
            all_values = digenic_values + trigenic_values
            if all_values:
                bins = np.histogram_bin_edges(all_values, bins=25)

                if digenic_values:
                    ax_dist.hist(digenic_values, bins=bins, alpha=0.7, color=digenic_color,
                           label=f'Digenic ($\\delta$) (n={len(digenic_values)})', density=True,
                           edgecolor='black', linewidth=0.5)

                if trigenic_values:
                    ax_dist.hist(trigenic_values, bins=bins, alpha=0.7, color=trigenic_color,
                           label=f'Trigenic ($\\sigma$) (n={len(trigenic_values)})', density=True,
                           edgecolor='black', linewidth=0.5)

        # Remove x-label from top row since bottom row has same label
        # ax_dist.set_xlabel('Interaction Score')
        if idx == 0:
            ax_dist.set_ylabel('Density', fontsize=14)
        ax_dist.set_title(f'{ffa}', fontsize=14, fontweight='bold')
        ax_dist.legend(loc='upper right', fontsize=11)
        ax_dist.grid(True, alpha=0.3)
        ax_dist.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

        # Add statistics text box
        stats_text = []
        if digenic_values:
            mean_di = np.mean(digenic_values)
            std_di = np.std(digenic_values)
            mean_se_di = np.nanmean(digenic_errors)
            stats_text.append(f'Digenic:\n$\\mu$={mean_di:.2f}\n$\\sigma$={std_di:.2f}\nSE={mean_se_di:.3f}')

        if trigenic_values:
            mean_tri = np.mean(trigenic_values)
            std_tri = np.std(trigenic_values)
            mean_se_tri = np.nanmean(trigenic_errors)
            stats_text.append(f'Trigenic:\n$\\mu$={mean_tri:.2f}\n$\\sigma$={std_tri:.2f}\nSE={mean_se_tri:.3f}')

        if stats_text:
            ax_dist.text(0.98, 0.5, '\n\n'.join(stats_text),
                   transform=ax_dist.transAxes, fontsize=10,
                   verticalalignment='center', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))

        # BOTTOM ROW: Volcano plots
        if digenic_values and digenic_pvals:
            valid_di = [(v, p) for v, p in zip(digenic_values, digenic_pvals) if not np.isnan(p)]
            if valid_di:
                di_vals, di_pvals = zip(*valid_di)
                ax_volc.scatter(di_vals, [-np.log10(p) for p in di_pvals],
                              alpha=0.6, s=20, label=r'Digenic ($\delta$)', color=digenic_color)

        if trigenic_values and trigenic_pvals:
            valid_tri = [(v, p) for v, p in zip(trigenic_values, trigenic_pvals) if not np.isnan(p)]
            if valid_tri:
                tri_vals, tri_pvals = zip(*valid_tri)
                ax_volc.scatter(tri_vals, [-np.log10(p) for p in tri_pvals],
                              alpha=0.6, s=20, label=r'Trigenic ($\sigma$)', color=trigenic_color)

        # Add significance lines with labels for legend
        line1 = ax_volc.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, linewidth=1, label='p = 0.05')
        line2 = ax_volc.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.5, linewidth=1, label='p = 0.01')
        ax_volc.axvline(x=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)

        # Labels with consistent font size
        ax_volc.set_xlabel(r'Interaction Score ($\delta$ or $\sigma$)', fontsize=14)
        if idx == 0:
            ax_volc.set_ylabel('-log₁₀(p-value)', fontsize=14)

        # Legend in top right for each plot with line definitions included
        ax_volc.legend(loc='upper right', fontsize=11)

        ax_volc.grid(True, alpha=0.3)
        ax_volc.set_ylim(-0.1, max(3.5, ax_volc.get_ylim()[1]))

    plt.suptitle('TF Interaction Distributions and Volcano Plots by FFA Type (ADDITIVE MODEL)', fontsize=16)
    plt.tight_layout()

    return fig


def plot_significance_summary(combined_df, columns):
    """Create plots showing statistical significance summary for ADDITIVE model."""

    # Create figure with 1 row, 3 columns for summary statistics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: P-value distribution
    ax1 = axes[0]
    ax1.hist(combined_df['p_value'].dropna(), bins=50, alpha=0.7, color='#34699D', edgecolor='black')
    ax1.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
    ax1.set_xlabel('P-value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('P-value Distribution (Additive Model)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Percentage of significant interactions split by type
    ax2 = axes[1]

    # Calculate percentages for each FFA and interaction type
    ffa_labels = []
    digenic_percentages = []
    trigenic_percentages = []

    for ffa in columns:  # Include all FFAs here
        ffa_df = combined_df[combined_df['ffa_type'] == ffa]

        # Digenic
        digenic_ffa = ffa_df[ffa_df['interaction_type'] == 'digenic']
        if len(digenic_ffa) > 0:
            digenic_pct = 100 * digenic_ffa['significant_p05'].sum() / len(digenic_ffa)
        else:
            digenic_pct = 0

        # Trigenic
        trigenic_ffa = ffa_df[ffa_df['interaction_type'] == 'trigenic']
        if len(trigenic_ffa) > 0:
            trigenic_pct = 100 * trigenic_ffa['significant_p05'].sum() / len(trigenic_ffa)
        else:
            trigenic_pct = 0

        ffa_labels.append(ffa.replace('Total ', ''))  # Shorten label
        digenic_percentages.append(digenic_pct)
        trigenic_percentages.append(trigenic_pct)

    x_pos = np.arange(len(ffa_labels))
    width = 0.35

    bars1 = ax2.bar(x_pos - width/2, digenic_percentages, width,
                    label='Digenic', alpha=0.7, color='#7191A9')
    bars2 = ax2.bar(x_pos + width/2, trigenic_percentages, width,
                    label='Trigenic', alpha=0.7, color='#B73C39')

    # Add percentage labels on bars with larger font and offset for trigenic bars
    for i, bars in enumerate([bars1, bars2]):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                # Shift trigenic (red bar) labels slightly right to avoid overlap
                x_offset = bar.get_width()/2. if i == 0 else bar.get_width()/2. + 0.05
                ax2.text(bar.get_x() + x_offset, height + 1,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_xlabel('FFA Type')
    ax2.set_ylabel('% Significant (p < 0.05)')
    ax2.set_title('Significant Interactions by Type (Additive Model)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ffa_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)

    # Plot 3: QQ plot for p-values
    ax3 = axes[2]
    observed_p = np.sort(combined_df['p_value'].dropna())
    expected_p = np.linspace(0, 1, len(observed_p))
    ax3.scatter(expected_p, observed_p, alpha=0.5, s=10, color='#6B8D3A')
    ax3.plot([0, 1], [0, 1], '--', color='#CC8250', label='Expected uniform', linewidth=2)
    ax3.set_xlabel('Expected p-value (uniform)')
    ax3.set_ylabel('Observed p-value')
    ax3.set_title('Q-Q Plot for P-values (Additive Model)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle('TF Interaction Statistical Significance Summary (ADDITIVE MODEL)', fontsize=16)
    plt.tight_layout()

    return fig


def main():
    """Main function to compute ADDITIVE interactions and generate plots."""
    print("="*60)
    print("ADDITIVE INTERACTION MODEL ANALYSIS")
    print("="*60)
    print("\nLoading free fatty acid data...")

    # Load data with replicates
    file_path = "/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/ffa_xue2025/raw/Supplementary Data 1_Raw titers.xlsx"
    raw_df, abbreviations, replicate_dict = load_ffa_data(file_path)

    print(f"Loaded data with {len(raw_df)} strains")
    print(f"Columns: {list(raw_df.columns[1:])}")

    # Normalize by positive control (POX1-FAA1-FAA4)
    print("\nNormalizing by positive control (POX1-FAA1-FAA4)...")
    normalized_df, normalized_replicates = normalize_by_reference(raw_df, replicate_dict)

    # Compute interactions with error propagation using ADDITIVE MODEL
    print("\n=== Computing TF Interactions with ADDITIVE MODEL and Error Propagation ===")
    print("Digenic: ε^A_ij = f_ij - (f_i + f_j - 1)")
    print("Trigenic: τ^A_ijk = f_ijk - (f_ij + f_ik + f_jk) + (f_i + f_j + f_k) - 1")
    print("")

    (digenic_interactions, digenic_sd, digenic_se, digenic_pvalues,
     trigenic_interactions, trigenic_sd, trigenic_se, trigenic_pvalues,
     single_mutants, double_mutants, triple_mutants,
     single_sd, single_se, double_sd, double_se, triple_sd, triple_se) = compute_additive_interactions_with_error_propagation(
        normalized_df, normalized_replicates, abbreviations)

    # Apply FDR correction
    print("\nApplying FDR correction...")
    digenic_fdr = apply_fdr_correction(digenic_pvalues, method='fdr_bh')
    trigenic_fdr = apply_fdr_correction(trigenic_pvalues, method='fdr_bh')

    # Get column names (FFA types)
    columns = list(raw_df.columns[1:])

    # Save interaction results
    print("\nSaving ADDITIVE interaction results with effect sizes and FDR corrections...")
    combined_df = save_interaction_results(
        digenic_interactions, digenic_sd, digenic_se, digenic_pvalues, digenic_fdr,
        trigenic_interactions, trigenic_sd, trigenic_se, trigenic_pvalues, trigenic_fdr,
        columns, RESULTS_DIR
    )

    # Create plots
    print("\nCreating publication-quality plots for ADDITIVE model...")

    # 1. Combined distribution and volcano plots by FFA
    fig1 = plot_interaction_distributions_and_volcano(
        digenic_interactions, digenic_se, digenic_pvalues,
        trigenic_interactions, trigenic_se, trigenic_pvalues,
        columns
    )
    filename1 = f"additive_ffa_distributions_and_volcano_3_delta_normalized_{timestamp()}.png"
    filepath1 = osp.join(ASSET_IMAGES_DIR, filename1)
    fig1.savefig(filepath1, dpi=300)
    plt.close()
    print(f"Distribution and volcano plot saved to:")
    print(f"  {filepath1}")

    # 2. Statistical significance summary
    fig2 = plot_significance_summary(combined_df, columns)
    filename2 = f"additive_ffa_significance_summary_3_delta_normalized_{timestamp()}.png"
    filepath2 = osp.join(ASSET_IMAGES_DIR, filename2)
    fig2.savefig(filepath2, dpi=300)
    plt.close()
    print(f"Significance summary plot saved to:")
    print(f"  {filepath2}")

    print(f"\nAll FFA interaction plots complete!")
    print(f"Results saved to {RESULTS_DIR}")

    # Print final summary
    print("\n=== Final Summary (ADDITIVE MODEL) ===")
    print(f"Total interactions: {len(combined_df)}")
    print(f"Significant (p<0.05): {combined_df['significant_p05'].sum()}")
    print(f"Significant after FDR (q<0.05): {combined_df['significant_fdr05'].sum()}")
    print(f"Mean effect size: {combined_df['effect_size'].mean():.2f}")
    print(f"Median effect size: {combined_df['effect_size'].median():.2f}")

    return combined_df


if __name__ == "__main__":
    results = main()