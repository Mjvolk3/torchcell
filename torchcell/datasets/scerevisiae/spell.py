# torchcell/datasets/scerevisiae/spell
# [[torchcell.datasets.scerevisiae.spell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/spell
# Test file: tests/torchcell/datasets/scerevisiae/test_spell.py

# File needs to be down loaded from here http://sgd-archive.yeastgenome.org/expression/microarray/all_spell_datasets.tar.gz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from dotenv import load_dotenv
from torchcell.timestamp import timestamp

load_dotenv()

ASSET_IMAGES_DIR = osp.expanduser(osp.join("~", "Documents", "projects", "torchcell", "notes", "assets", "images"))
DATA_ROOT = osp.expanduser(osp.join("~", "Documents", "projects", "torchcell"))


def read_pcl_file(pcl_path):
    """
    Parse a PCL (Platform for Clustering and Linkage) file from SPELL database.

    PCL format:
    - Tab-delimited text file
    - Row 1: Header (YORF, NAME, GWEIGHT, then condition names)
    - Row 2: EWEIGHT row with experiment weights
    - Remaining rows: Gene data (ORF, name, weight, expression values)

    Args:
        pcl_path: Path to the .pcl file

    Returns:
        pd.DataFrame: Expression data with genes as rows and conditions as columns
        dict: Metadata including condition names and weights
    """
    with open(pcl_path, 'r') as f:
        # Read header line
        header = f.readline().strip().split('\t')
        # Read EWEIGHT line
        eweight_line = f.readline().strip().split('\t')

    # Parse the file with pandas, using the header we extracted and skipping first 2 rows
    df = pd.read_csv(pcl_path, sep='\t', skiprows=2, names=header)

    # Extract metadata
    metadata = {
        'conditions': header[3:],  # Skip YORF, NAME, GWEIGHT
        'eweights': [float(w) if w else 1.0 for w in eweight_line[3:]],
        'n_genes': len(df),
        'n_conditions': len(header) - 3
    }

    # Set index to ORF names
    df = df.set_index('YORF')

    return df, metadata


def plot_expression_histograms(df, metadata, gene_list=None, title_prefix="SPELL", save_path=None):
    """
    Plot histograms of log2 expression ratios for selected genes across all conditions.

    Args:
        df: DataFrame with expression data (from read_pcl_file)
        metadata: Metadata dict (from read_pcl_file)
        gene_list: List of gene ORF names to plot (default: first 5 genes)
        title_prefix: Prefix for plot title
        save_path: Path to save the figure (if None, displays instead)
    """
    if gene_list is None:
        # Use first 5 genes as default
        gene_list = df.index[:5].tolist()

    # Filter for available genes
    available_genes = [g for g in gene_list if g in df.index]

    if not available_genes:
        print(f"None of the specified genes found in dataset")
        return

    # Get expression columns (skip NAME and GWEIGHT)
    expr_cols = metadata['conditions']

    # Create subplots
    n_genes = len(available_genes)
    fig, axes = plt.subplots(n_genes, 1, figsize=(10, 3 * n_genes))

    if n_genes == 1:
        axes = [axes]

    for idx, gene in enumerate(available_genes):
        # Get expression values for this gene
        expr_values = df.loc[gene, expr_cols].values.astype(float)

        # Remove NaN values
        expr_values = expr_values[~np.isnan(expr_values)]

        # Plot histogram
        axes[idx].hist(expr_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_xlabel('Log2 Expression Ratio')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{gene} ({df.loc[gene, "NAME"]}) - {len(expr_values)} conditions')
        axes[idx].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='No change')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)

    plt.suptitle(f'{title_prefix} - Gene Expression Distributions', fontsize=14, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def plot_global_expression_distribution(df, metadata, title_prefix="SPELL", save_path=None):
    """
    Plot histogram of all log2 expression ratios across all genes and conditions.

    Args:
        df: DataFrame with expression data
        metadata: Metadata dict
        title_prefix: Prefix for plot title
        save_path: Path to save the figure
    """
    # Get all expression columns
    expr_cols = metadata['conditions']

    # Flatten all expression values
    all_values = df[expr_cols].values.flatten()
    all_values = all_values[~np.isnan(all_values)].astype(float)

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_values, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Log2 Expression Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title_prefix} - Global Expression Distribution\n'
                 f'{metadata["n_genes"]} genes × {metadata["n_conditions"]} conditions = '
                 f'{len(all_values):,} measurements')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='No change')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: {np.mean(all_values):.3f}\n' \
                 f'Median: {np.median(all_values):.3f}\n' \
                 f'Std: {np.std(all_values):.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def extract_and_load_all_spell_studies(spell_root_dir, studies_to_load=None, max_studies=None):
    """
    Extract and load PCL files from multiple SPELL studies.

    Args:
        spell_root_dir: Root directory containing .zip study files
        studies_to_load: List of study names (e.g., ['Gasch_2000_PMID_11102521']) or None for all
        max_studies: Maximum number of studies to load (None = unlimited)

    Returns:
        dict: Dictionary mapping (study_name, dataset_name) -> (df, metadata)
    """
    import glob
    import zipfile

    all_data = {}
    studies_loaded = 0

    # Find all zip files
    zip_files = sorted(glob.glob(osp.join(spell_root_dir, "*.zip")))

    # Filter by studies_to_load if specified
    if studies_to_load:
        zip_files = [z for z in zip_files if any(study in z for study in studies_to_load)]

    print(f"Found {len(zip_files)} study archives")
    total_studies_to_load = max_studies if max_studies else len(zip_files)

    for idx, zip_path in enumerate(zip_files, 1):
        if max_studies and studies_loaded >= max_studies:
            print(f"Reached max_studies limit ({max_studies})")
            break

        study_name = osp.basename(zip_path).replace('.zip', '')
        study_dir = osp.join(spell_root_dir, study_name)

        # Progress indicator
        print(f"\n[{idx}/{total_studies_to_load}] Processing: {study_name}")

        # Extract if not already extracted
        if not osp.exists(study_dir):
            try:
                print(f"  Extracting...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(spell_root_dir)
            except Exception as e:
                print(f"  ✗ Error extracting: {e}")
                continue

        # Load all PCL files from this study
        pcl_files = glob.glob(osp.join(study_dir, "*.pcl"))

        if not pcl_files:
            print(f"  ⚠ No PCL files found")
            continue

        datasets_loaded = 0
        for pcl_file in sorted(pcl_files):
            dataset_name = osp.basename(pcl_file).replace('.pcl', '')
            try:
                df, metadata = read_pcl_file(pcl_file)
                all_data[(study_name, dataset_name)] = (df, metadata)
                datasets_loaded += 1
            except Exception as e:
                print(f"  ✗ {dataset_name}: {e}")

        print(f"  ✓ Loaded {datasets_loaded} dataset(s)")
        studies_loaded += 1

    return all_data


def plot_genes_across_all_studies(all_data, gene_list, save_path=None):
    """
    Plot histograms showing expression distributions for specific genes
    across ALL SPELL studies and datasets.

    Args:
        all_data: Dictionary from extract_and_load_all_spell_studies()
                 Maps (study_name, dataset_name) -> (df, metadata)
        gene_list: List of gene ORF names to plot
        save_path: Path to save the figure
    """
    # Collect expression data for each gene across all studies and datasets
    gene_data = {gene: [] for gene in gene_list}
    gene_names = {}
    total_conditions = 0
    total_datasets = len(all_data)
    total_studies = len(set(key[0] for key in all_data.keys()))

    for (study_name, dataset_name), (df, metadata) in all_data.items():
        expr_cols = metadata['conditions']
        total_conditions += len(expr_cols)

        for gene in gene_list:
            if gene in df.index:
                # Store gene name
                if gene not in gene_names:
                    gene_names[gene] = df.loc[gene, 'NAME']

                # Get expression values for this gene in this dataset
                expr_values = df.loc[gene, expr_cols].values.astype(float)
                expr_values = expr_values[~np.isnan(expr_values)]
                gene_data[gene].extend(expr_values.tolist())

    # Create subplots
    n_genes = len(gene_list)
    fig, axes = plt.subplots(n_genes, 1, figsize=(12, 4 * n_genes))

    if n_genes == 1:
        axes = [axes]

    for idx, gene in enumerate(gene_list):
        values = np.array(gene_data[gene])

        if len(values) == 0:
            axes[idx].text(0.5, 0.5, f'No data found for {gene}',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{gene} - No expression data available')
            continue

        # Plot histogram
        axes[idx].hist(values, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_xlabel('Log2 Expression Ratio')
        axes[idx].set_ylabel('Frequency')

        gene_name = gene_names.get(gene, gene)
        axes[idx].set_title(f'{gene} ({gene_name}) - {len(values):,} measurements '
                           f'across {total_studies} studies')
        axes[idx].axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='No change')
        axes[idx].grid(axis='y', alpha=0.3)

        # Add statistics
        stats_text = f'Mean: {np.mean(values):.3f}\nMedian: {np.median(values):.3f}\nStd: {np.std(values):.3f}'
        axes[idx].text(0.98, 0.98, stats_text, transform=axes[idx].transAxes,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[idx].legend()

    plt.suptitle(f'SPELL Database - Gene Expression Across Multiple Studies\n'
                 f'{total_studies} studies, {total_datasets} datasets, {total_conditions:,} total conditions',
                 fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved figure to: {save_path}")
    else:
        plt.show()


def extract_time_info(condition_name):
    """Extract time information from condition name."""
    import re
    result = {'time_min': None, 'is_timeseries': False}

    # Match patterns like "30 min", "2 hours", "120min", "1.5 hr"
    time_patterns = [
        (r'(\d+\.?\d*)\s*min', 1),  # minutes
        (r'(\d+\.?\d*)\s*(?:hr|hours?)', 60),  # hours to minutes (non-capturing group for alternation)
        (r'(\d+\.?\d*)\s*sec', 1/60),  # seconds to minutes
    ]

    for pattern, multiplier in time_patterns:
        match = re.search(pattern, condition_name, re.IGNORECASE)
        if match:
            result['time_min'] = float(match.group(1)) * multiplier
            break

    # Check for time series indicators
    timeseries_keywords = ['time course', 'time series', 'time zero', 'timepoint', 'time point', 't=']
    result['is_timeseries'] = any(kw in condition_name.lower() for kw in timeseries_keywords)

    return result


def extract_temperature_info(condition_name):
    """Extract temperature information from condition name."""
    import re
    result = {'temperature_c': None}

    # Match patterns like "37°C", "37C", "37 degrees", "30 deg"
    temp_patterns = [
        r'(\d+\.?\d*)\s*°C',
        r'(\d+\.?\d*)\s*C(?:\s|$|,)',  # C followed by space, end, or comma
        r'(\d+\.?\d*)\s*deg(?:rees)?(?:\s*C)?',
    ]

    for pattern in temp_patterns:
        match = re.search(pattern, condition_name, re.IGNORECASE)
        if match:
            result['temperature_c'] = float(match.group(1))
            break

    return result


def extract_chemical_info(condition_name):
    """Extract chemical compound information from condition name."""
    import re
    result = {
        'chemical_name': None,
        'concentration': None,
        'concentration_unit': None
    }

    # Common chemicals in yeast experiments
    chemicals = [
        'H2O2', 'hydrogen peroxide', 'peroxide',
        'menadione', 'diamide', 'paraquat',
        'sorbitol', 'NaCl', 'KCl', 'salt',
        'rapamycin', 'tunicamycin', 'brefeldin', 'cycloheximide',
        'DTT', 'cadmium', 'arsenite', 'formaldehyde',
        'MMS', 'methyl methanesulfonate',
        'alpha factor', 'alpha-factor',
        'glucose', 'galactose', 'raffinose', 'glycerol', 'ethanol',
        'acetate', 'benzoate', 'propionate', 'sorbate',
        'lactic acid', 'acetic acid',
        'zeocin', 'hygromycin', 'G418',
    ]

    # Find chemical name
    for chem in chemicals:
        if chem.lower() in condition_name.lower():
            result['chemical_name'] = chem
            break

    # Extract concentration - patterns like "500 mM", "1.5 M", "10 µM", "2%"
    conc_patterns = [
        (r'(\d+\.?\d*)\s*(mM|µM|uM|nM|M|mg/ml|ug/ml|%)', r'\2'),
    ]

    for pattern, unit_group in conc_patterns:
        match = re.search(pattern, condition_name, re.IGNORECASE)
        if match:
            result['concentration'] = float(match.group(1))
            result['concentration_unit'] = match.group(2)
            break

    return result


def extract_nutrient_info(condition_name):
    """Extract nutrient composition information."""
    result = {
        'carbon_source': None,
        'nitrogen_source': None,
        'limitation_type': None,
    }

    # Carbon sources
    carbon_sources = {
        'glucose': ['glucose', 'dextrose'],
        'galactose': ['galactose'],
        'raffinose': ['raffinose'],
        'glycerol': ['glycerol'],
        'ethanol': ['ethanol'],
        'acetate': ['acetate'],
        'lactate': ['lactate', 'lactic acid'],
    }

    for source, keywords in carbon_sources.items():
        if any(kw in condition_name.lower() for kw in keywords):
            result['carbon_source'] = source
            break

    # Nitrogen sources
    nitrogen_sources = {
        'ammonium': ['ammonium', 'nh4'],
        'proline': ['proline'],
        'glutamine': ['glutamine'],
        'urea': ['urea'],
    }

    for source, keywords in nitrogen_sources.items():
        if any(kw in condition_name.lower() for kw in keywords):
            result['nitrogen_source'] = source
            break

    # Limitation type
    limitation_patterns = {
        'carbon_limited': ['c-lim', 'carbon limit', 'carbon-lim'],
        'nitrogen_limited': ['n-lim', 'nitrogen limit', 'nitrogen-lim'],
        'phosphate_limited': ['p-lim', 'phosphate limit', 'phosphate-lim'],
        'sulfur_limited': ['s-lim', 'sulfur limit', 'sulfur-lim'],
    }

    for lim_type, keywords in limitation_patterns.items():
        if any(kw in condition_name.lower() for kw in keywords):
            result['limitation_type'] = lim_type
            break

    return result


def extract_physical_params(condition_name):
    """Extract physical parameters like pH, oxygen level."""
    import re
    result = {
        'ph': None,
        'oxygen_level': None,
    }

    # pH patterns: "pH 5", "pH=5.5", "(pH 3)"
    ph_patterns = [
        r'pH\s*[=:]?\s*(\d+\.?\d*)',
        r'\(pH\s+(\d+\.?\d*)\)',
    ]

    for pattern in ph_patterns:
        match = re.search(pattern, condition_name, re.IGNORECASE)
        if match:
            result['ph'] = float(match.group(1))
            break

    # Oxygen level
    oxygen_keywords = {
        'anaerobic': ['anaerobic', 'anoxic'],
        'aerobic': ['aerobic'],
        'hypoxic': ['hypoxic', 'low oxygen'],
        'hyperoxic': ['hyperoxic', 'high oxygen'],
    }

    for level, keywords in oxygen_keywords.items():
        if any(kw in condition_name.lower() for kw in keywords):
            result['oxygen_level'] = level
            break

    return result


def extract_stress_info(condition_name):
    """Extract stress type information."""
    result = {'stress_type': None}

    stress_types = {
        'heat_shock': ['heat shock', 'heat stress', 'thermal stress', 'temperature shift'],
        'oxidative_stress': ['oxidative', 'h2o2', 'peroxide', 'menadione'],
        'osmotic_stress': ['osmotic', 'sorbitol', 'salt stress'],
        'dna_damage': ['dna damage', 'uv', 'mms', 'radiation'],
        'er_stress': ['er stress', 'tunicamycin', 'dtt'],
        'cell_wall_stress': ['cell wall', 'calcofluor', 'congo red'],
    }

    for stress, keywords in stress_types.items():
        if any(kw in condition_name.lower() for kw in keywords):
            result['stress_type'] = stress
            break

    return result


def extract_cell_cycle_info(condition_name):
    """Extract cell cycle information."""
    result = {
        'cell_cycle_phase': None,
        'synchronization_method': None,
    }

    # Cell cycle phases
    phases = {
        'G1': ['g1', 'g1 phase'],
        'S': ['s phase', ' s '],
        'G2': ['g2', 'g2 phase'],
        'M': ['m phase', 'mitosis', 'metaphase'],
    }

    for phase, keywords in phases.items():
        if any(kw in condition_name.lower() for kw in keywords):
            result['cell_cycle_phase'] = phase
            break

    # Synchronization methods
    sync_methods = {
        'alpha_factor': ['alpha factor', 'alpha-factor', 'α-factor'],
        'elutriation': ['elutriation'],
        'cdc_arrest': ['cdc15', 'cdc28', 'arrest'],
    }

    for method, keywords in sync_methods.items():
        if any(kw in condition_name.lower() for kw in keywords):
            result['synchronization_method'] = method
            break

    return result


def extract_replicate_info(condition_name):
    """Extract replicate information."""
    import re
    result = {
        'replicate_number': None,
        'replicate_type': None,
    }

    # Match patterns like "#1", "rep 1", "replicate 2", "bio rep 1", "tech rep 2"
    rep_patterns = [
        r'#(\d+)',
        r'rep(?:licate)?\s+(\d+)',
        r'(\d+)$',  # Number at end of string
    ]

    for pattern in rep_patterns:
        match = re.search(pattern, condition_name, re.IGNORECASE)
        if match:
            result['replicate_number'] = int(match.group(1))
            break

    # Replicate type
    if 'biological' in condition_name.lower() or 'bio rep' in condition_name.lower():
        result['replicate_type'] = 'biological'
    elif 'technical' in condition_name.lower() or 'tech rep' in condition_name.lower():
        result['replicate_type'] = 'technical'
    elif result['replicate_number'] is not None:
        result['replicate_type'] = 'unknown'  # Has replicate number but type not specified

    return result


def export_condition_metadata(all_data, output_path=None):
    """
    Export detailed condition metadata for analysis and visualization.

    Creates an enhanced CSV with structured columns for environmental parameters:
    - study_name, dataset_name, condition_name
    - primary_category, secondary_categories
    - time_min, is_timeseries
    - temperature_c
    - chemical_name, concentration, concentration_unit
    - carbon_source, nitrogen_source, limitation_type
    - ph, oxygen_level
    - stress_type
    - cell_cycle_phase, synchronization_method
    - replicate_number, replicate_type
    - extraction_confidence, needs_manual_review

    Args:
        all_data: Dictionary from extract_and_load_all_spell_studies()
        output_path: Path to save CSV (default: DATA_ROOT/spell_conditions_metadata.csv)

    Returns:
        pd.DataFrame: Enhanced condition metadata
    """
    import re

    if output_path is None:
        output_path = osp.join(DATA_ROOT, "data/sgd/spell", "spell_conditions_metadata_enhanced.csv")

    # Parse condition metadata
    condition_records = []

    # Define categories based on keywords
    categories = {
        'heat_shock': ['heat', 'temperature', 'thermal', 'hs'],
        'oxidative_stress': ['oxidative', 'peroxide', 'h2o2', 'menadione', 'diamide', 'paraquat'],
        'osmotic_stress': ['osmotic', 'sorbitol', 'nacl', 'kcl', 'salt', 'hyper', 'hypo'],
        'nutrient_limitation': ['nitrogen', 'carbon', 'phosphate', 'sulfur', 'starvation', 'limited', 'lim'],
        'drug_treatment': ['drug', 'rapamycin', 'tunicamycin', 'brefeldin', 'cycloheximide'],
        'cell_cycle': ['cell cycle', 'g1', 'g2', 's phase', 'm phase', 'mitosis', 'alpha factor'],
        'dna_damage': ['dna', 'uv', 'radiation', 'mms', 'damage', 'repair'],
        'chemical_stress': ['chemical', 'dtt', 'cadmium', 'arsenite', 'formaldehyde'],
        'anaerobic': ['anaerobic', 'hypoxia', 'oxygen'],
        'time_series': ['min', 'hour', 'time'],
        'mutant_strain': ['mutant', 'deletion', 'knockout', 'overexpression'],
        'wild_type_control': ['wild type', 'wt', 'control', 'reference', 'untreated'],
    }

    print("Extracting enhanced metadata from conditions...")
    total_conditions = sum(metadata['n_conditions'] for _, (_, metadata) in all_data.items())
    processed = 0

    for (study_name, dataset_name), (df, metadata) in all_data.items():
        for idx, condition in enumerate(metadata['conditions']):
            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed:,}/{total_conditions:,} conditions...")

            # Categorize condition
            condition_lower = condition.lower()
            matched_categories = []

            for category, keywords in categories.items():
                if any(kw in condition_lower for kw in keywords):
                    matched_categories.append(category)

            # If no match, try to infer from structure
            if not matched_categories:
                if re.match(r'^\d+$', condition.strip()):
                    matched_categories.append('numeric_id')
                elif 'chip' in condition_lower or 'array' in condition_lower:
                    matched_categories.append('array_id')
                else:
                    matched_categories.append('uncategorized')

            # Extract structured environmental parameters
            time_info = extract_time_info(condition)
            temp_info = extract_temperature_info(condition)
            chem_info = extract_chemical_info(condition)
            nutrient_info = extract_nutrient_info(condition)
            physical_info = extract_physical_params(condition)
            stress_info = extract_stress_info(condition)
            cell_cycle_info = extract_cell_cycle_info(condition)
            replicate_info = extract_replicate_info(condition)

            # Calculate extraction confidence (0-1 scale)
            # Based on how many structured parameters we successfully extracted
            extracted_fields = 0
            total_attempted = 14  # Number of extractable fields

            if time_info['time_min'] is not None: extracted_fields += 1
            if time_info['is_timeseries']: extracted_fields += 1
            if temp_info['temperature_c'] is not None: extracted_fields += 1
            if chem_info['chemical_name'] is not None: extracted_fields += 1
            if chem_info['concentration'] is not None: extracted_fields += 1
            if nutrient_info['carbon_source'] is not None: extracted_fields += 1
            if nutrient_info['nitrogen_source'] is not None: extracted_fields += 1
            if nutrient_info['limitation_type'] is not None: extracted_fields += 1
            if physical_info['ph'] is not None: extracted_fields += 1
            if physical_info['oxygen_level'] is not None: extracted_fields += 1
            if stress_info['stress_type'] is not None: extracted_fields += 1
            if cell_cycle_info['cell_cycle_phase'] is not None: extracted_fields += 1
            if cell_cycle_info['synchronization_method'] is not None: extracted_fields += 1
            if replicate_info['replicate_number'] is not None: extracted_fields += 1

            # Confidence: higher when more fields extracted
            # 0 fields = 0.1, 5 fields = 0.5, 10+ fields = 0.9+
            extraction_confidence = min(0.9, 0.1 + (extracted_fields / total_attempted) * 0.8)

            # Flag for manual review if:
            # - Very low confidence (< 0.2)
            # - Uncategorized
            # - Has concentration but no chemical name (ambiguous)
            needs_manual_review = (
                extraction_confidence < 0.2 or
                'uncategorized' in matched_categories or
                (chem_info['concentration'] is not None and chem_info['chemical_name'] is None)
            )

            condition_records.append({
                'study_name': study_name,
                'dataset_name': dataset_name,
                'condition_name': condition,
                'condition_index': idx,
                'n_genes': metadata['n_genes'],
                'categories': '|'.join(matched_categories),
                'primary_category': matched_categories[0],
                'secondary_categories': '|'.join(matched_categories[1:]) if len(matched_categories) > 1 else '',
                # Time info
                'time_min': time_info['time_min'],
                'is_timeseries': time_info['is_timeseries'],
                # Temperature
                'temperature_c': temp_info['temperature_c'],
                # Chemical info
                'chemical_name': chem_info['chemical_name'],
                'concentration': chem_info['concentration'],
                'concentration_unit': chem_info['concentration_unit'],
                # Nutrient info
                'carbon_source': nutrient_info['carbon_source'],
                'nitrogen_source': nutrient_info['nitrogen_source'],
                'limitation_type': nutrient_info['limitation_type'],
                # Physical parameters
                'ph': physical_info['ph'],
                'oxygen_level': physical_info['oxygen_level'],
                # Stress info
                'stress_type': stress_info['stress_type'],
                # Cell cycle
                'cell_cycle_phase': cell_cycle_info['cell_cycle_phase'],
                'synchronization_method': cell_cycle_info['synchronization_method'],
                # Replicate info
                'replicate_number': replicate_info['replicate_number'],
                'replicate_type': replicate_info['replicate_type'],
                # Metadata
                'extraction_confidence': round(extraction_confidence, 3),
                'needs_manual_review': needs_manual_review,
            })

    # Create DataFrame
    df_conditions = pd.DataFrame(condition_records)

    # Save to CSV
    df_conditions.to_csv(output_path, index=False)
    print(f"\n✓ Exported {len(df_conditions):,} condition records to: {output_path}")

    # Print enhanced statistics
    print("\n" + "=" * 70)
    print("ENHANCED EXTRACTION SUMMARY")
    print("=" * 70)

    # Category breakdown
    print("\nCondition Category Breakdown:")
    category_counts = df_conditions['primary_category'].value_counts()
    for category, count in category_counts.items():
        pct = 100 * count / len(df_conditions)
        print(f"  {category:25s}: {count:6,} ({pct:5.1f}%)")

    # Extraction statistics
    print(f"\nStructured Parameter Extraction:")
    print(f"  Time information:         {df_conditions['time_min'].notna().sum():6,} conditions ({100*df_conditions['time_min'].notna().sum()/len(df_conditions):5.1f}%)")
    print(f"  Time series flags:        {df_conditions['is_timeseries'].sum():6,} conditions ({100*df_conditions['is_timeseries'].sum()/len(df_conditions):5.1f}%)")
    print(f"  Temperature:              {df_conditions['temperature_c'].notna().sum():6,} conditions ({100*df_conditions['temperature_c'].notna().sum()/len(df_conditions):5.1f}%)")
    print(f"  Chemical compounds:       {df_conditions['chemical_name'].notna().sum():6,} conditions ({100*df_conditions['chemical_name'].notna().sum()/len(df_conditions):5.1f}%)")
    print(f"  Concentrations:           {df_conditions['concentration'].notna().sum():6,} conditions ({100*df_conditions['concentration'].notna().sum()/len(df_conditions):5.1f}%)")
    print(f"  Carbon sources:           {df_conditions['carbon_source'].notna().sum():6,} conditions ({100*df_conditions['carbon_source'].notna().sum()/len(df_conditions):5.1f}%)")
    print(f"  Nitrogen sources:         {df_conditions['nitrogen_source'].notna().sum():6,} conditions ({100*df_conditions['nitrogen_source'].notna().sum()/len(df_conditions):5.1f}%)")
    print(f"  Nutrient limitations:     {df_conditions['limitation_type'].notna().sum():6,} conditions ({100*df_conditions['limitation_type'].notna().sum()/len(df_conditions):5.1f}%)")
    print(f"  pH values:                {df_conditions['ph'].notna().sum():6,} conditions ({100*df_conditions['ph'].notna().sum()/len(df_conditions):5.1f}%)")
    print(f"  Oxygen levels:            {df_conditions['oxygen_level'].notna().sum():6,} conditions ({100*df_conditions['oxygen_level'].notna().sum()/len(df_conditions):5.1f}%)")
    print(f"  Stress types:             {df_conditions['stress_type'].notna().sum():6,} conditions ({100*df_conditions['stress_type'].notna().sum()/len(df_conditions):5.1f}%)")
    print(f"  Cell cycle phases:        {df_conditions['cell_cycle_phase'].notna().sum():6,} conditions ({100*df_conditions['cell_cycle_phase'].notna().sum()/len(df_conditions):5.1f}%)")
    print(f"  Replicate numbers:        {df_conditions['replicate_number'].notna().sum():6,} conditions ({100*df_conditions['replicate_number'].notna().sum()/len(df_conditions):5.1f}%)")

    # Confidence and review statistics
    print(f"\nExtraction Quality:")
    print(f"  Mean confidence:          {df_conditions['extraction_confidence'].mean():.3f}")
    print(f"  Median confidence:        {df_conditions['extraction_confidence'].median():.3f}")
    print(f"  High confidence (>0.5):   {(df_conditions['extraction_confidence'] > 0.5).sum():6,} conditions ({100*(df_conditions['extraction_confidence'] > 0.5).sum()/len(df_conditions):5.1f}%)")
    print(f"  Needs manual review:      {df_conditions['needs_manual_review'].sum():6,} conditions ({100*df_conditions['needs_manual_review'].sum()/len(df_conditions):5.1f}%)")

    print("=" * 70)

    return df_conditions


def check_condition_metadata_quality(all_data):
    """
    Check if all expression data has proper condition labels.

    Returns:
        dict: Statistics about condition metadata quality
    """
    stats = {
        'total_conditions': 0,
        'empty_conditions': 0,
        'generic_conditions': 0,
        'numeric_only_conditions': 0,
        'descriptive_conditions': 0,
        'examples': {
            'empty': [],
            'generic': [],
            'numeric': [],
            'descriptive': []
        }
    }

    generic_patterns = ['condition', 'sample', 'array', 'chip', 'experiment']

    for (study_name, dataset_name), (df, metadata) in all_data.items():
        for cond in metadata['conditions']:
            stats['total_conditions'] += 1

            # Check for empty
            if not cond or cond.strip() == '':
                stats['empty_conditions'] += 1
                if len(stats['examples']['empty']) < 5:
                    stats['examples']['empty'].append(f"{study_name}/{dataset_name}: '{cond}'")

            # Check for generic labels
            elif any(pattern in cond.lower() for pattern in generic_patterns) and len(cond.split()) <= 2:
                stats['generic_conditions'] += 1
                if len(stats['examples']['generic']) < 5:
                    stats['examples']['generic'].append(f"{cond}")

            # Check for numeric-only (like "1", "2", "3")
            elif cond.strip().replace('.', '').replace('-', '').isdigit():
                stats['numeric_only_conditions'] += 1
                if len(stats['examples']['numeric']) < 5:
                    stats['examples']['numeric'].append(f"{cond}")

            # Otherwise it's descriptive
            else:
                stats['descriptive_conditions'] += 1
                if len(stats['examples']['descriptive']) < 5:
                    stats['examples']['descriptive'].append(f"{cond[:60]}")

    print("\n" + "=" * 70)
    print("CONDITION METADATA QUALITY CHECK")
    print("=" * 70)
    print(f"Total conditions: {stats['total_conditions']:,}")
    print(f"  ✓ Descriptive labels: {stats['descriptive_conditions']:,} "
          f"({100*stats['descriptive_conditions']/stats['total_conditions']:.1f}%)")
    print(f"  ⚠ Generic labels: {stats['generic_conditions']:,} "
          f"({100*stats['generic_conditions']/stats['total_conditions']:.1f}%)")
    print(f"  ⚠ Numeric-only labels: {stats['numeric_only_conditions']:,} "
          f"({100*stats['numeric_only_conditions']/stats['total_conditions']:.1f}%)")
    print(f"  ✗ Empty labels: {stats['empty_conditions']:,} "
          f"({100*stats['empty_conditions']/stats['total_conditions']:.1f}%)")

    if stats['examples']['empty']:
        print("\nExample empty conditions:")
        for ex in stats['examples']['empty']:
            print(f"  {ex}")

    if stats['examples']['generic']:
        print("\nExample generic conditions:")
        for ex in stats['examples']['generic']:
            print(f"  '{ex}'")

    if stats['examples']['numeric']:
        print("\nExample numeric-only conditions:")
        for ex in stats['examples']['numeric']:
            print(f"  '{ex}'")

    print("\nExample descriptive conditions:")
    for ex in stats['examples']['descriptive'][:5]:
        print(f"  '{ex}'")

    print("=" * 70 + "\n")

    return stats


def main():
    """
    Load SPELL expression data from multiple studies and create histograms
    for specific genes across all datasets and conditions.
    """
    spell_root_dir = osp.join(DATA_ROOT, "data/sgd/spell")

    # Check if data exists
    if not osp.exists(spell_root_dir):
        print(f"ERROR: SPELL data directory not found: {spell_root_dir}")
        print("\nTo download SPELL data, run these commands:")
        print(f"  mkdir -p {spell_root_dir}")
        print(f"  cd {spell_root_dir}")
        print("  curl -O http://sgd-archive.yeastgenome.org/expression/microarray/all_spell_datasets.tar.gz")
        print("  tar -xzf all_spell_datasets.tar.gz")
        return

    # Check if tar.gz exists but hasn't been extracted
    import glob
    tarfile = osp.join(spell_root_dir, "all_spell_datasets.tar.gz")
    if osp.exists(tarfile):
        zip_files = glob.glob(osp.join(spell_root_dir, "*.zip"))
        if not zip_files:
            print(f"Found {tarfile} but it hasn't been extracted yet.")
            print(f"Extracting to {spell_root_dir}...")
            import tarfile as tf
            with tf.open(tarfile, 'r:gz') as tar:
                tar.extractall(spell_root_dir)
            print("✓ Extraction complete!")

    # Example 1: Load just a few studies for testing
    # studies_to_load = ['Gasch_2000_PMID_11102521', 'Gasch_2001_PMID_11598186']
    # all_data = extract_and_load_all_spell_studies(spell_root_dir, studies_to_load=studies_to_load)

    # Example 2: Load first N studies (faster for exploration)
    # all_data = extract_and_load_all_spell_studies(spell_root_dir, max_studies=10)

    # Example 3: Load ALL studies (comprehensive - will take several minutes!)
    print("=" * 70)
    print("Loading SPELL expression data from ALL studies...")
    print("This will take several minutes - loading ~600 studies...")
    print("=" * 70)
    all_data = extract_and_load_all_spell_studies(spell_root_dir)

    # Calculate total expression measurements
    total_measurements = 0
    total_conditions = 0
    total_genes_measured = set()

    for (study_name, dataset_name), (df, metadata) in all_data.items():
        n_genes = metadata['n_genes']
        n_conds = metadata['n_conditions']
        total_measurements += n_genes * n_conds
        total_conditions += n_conds
        total_genes_measured.update(df.index.tolist())

    print(f"\n{'=' * 70}")
    print(f"SPELL DATABASE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Studies loaded:           {len(set(k[0] for k in all_data.keys())):,}")
    print(f"  Datasets loaded:          {len(all_data):,}")
    print(f"  Total conditions:         {total_conditions:,}")
    print(f"  Unique genes measured:    {len(total_genes_measured):,}")
    print(f"")
    print(f"  TOTAL EXPRESSION VALUES:  {total_measurements:,}")
    print(f"  (genes × conditions across all datasets)")
    print(f"{'=' * 70}\n")

    # Export condition metadata to CSV for analysis
    df_conditions = export_condition_metadata(all_data)

    # Check condition metadata quality
    metadata_stats = check_condition_metadata_quality(all_data)

    # Plot histograms for 3 genes across ALL loaded studies and datasets
    genes_of_interest = ['YDL025C', 'YJL166W', 'YMR027W']

    save_path = osp.join(ASSET_IMAGES_DIR, f"spell_genes_all_studies_{timestamp()}.png")
    plot_genes_across_all_studies(all_data, genes_of_interest, save_path=save_path)

    print(f"\n{'=' * 70}")
    print(f"Images saved to: {ASSET_IMAGES_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()