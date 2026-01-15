---
id: 5z7dipbt0yr33lnp1d10vce
title: Spell
desc: ''
updated: 1768428149817
created: 1768428149817
---

## Overview

Data loading and metadata extraction module for the SPELL (Saccharomyces Genome Database Platform for Expression data archival and Laboratory analysis) microarray database. This module provides functions to parse PCL files, extract structured environmental metadata from free-text condition descriptions, and prepare SPELL data for integration into torchcell datasets.

## Data Source

### SPELL Database

- **Repository**: SGD (Saccharomyces Genome Database)
- **URL**: <http://sgd-archive.yeastgenome.org/expression/microarray/all_spell_datasets.tar.gz>
- **Format**: ZIP archives containing PCL (Platform for Clustering and Linkage) files
- **Content**: ~200+ studies with gene expression microarray data
- **Organism**: *Saccharomyces cerevisiae* (baker's yeast)
- **Data Type**: Log2 expression ratios (treatment vs. reference)

### PCL File Format

Tab-delimited text files with specific structure:

- **Row 1 (Header)**: `YORF`, `NAME`, `GWEIGHT`, condition_1, condition_2, ..., condition_N
- **Row 2 (EWEIGHT)**: Empty, Empty, Empty, weight_1, weight_2, ..., weight_N
- **Row 3+ (Data)**: ORF_ID, Gene_Name, Gene_Weight, expr_1, expr_2, ..., expr_N

**Example**:

```
YORF    NAME    GWEIGHT    WT_30C_10min    WT_37C_10min    WT_42C_10min
EWEIGHT                    1.0             1.0             1.0
YAL001C TFC3    1.0        0.12            0.45            1.23
YAL002W VPS8    1.0        -0.34           0.11            0.89
```

## Core Functions

### read_pcl_file(pcl_path)

Parses a single PCL file into a DataFrame and metadata dictionary.

**Args**:

- `pcl_path`: Path to `.pcl` file

**Returns**:

- `df`: pandas DataFrame with genes as rows, conditions as columns
  - Index: ORF names (e.g., `YAL001C`)
  - Columns: `NAME`, `GWEIGHT`, condition columns
  - Values: Log2 expression ratios
- `metadata`: Dictionary containing:
  - `conditions`: List of condition names (strings)
  - `eweights`: List of experimental weights (floats)
  - `n_genes`: Number of genes (int)
  - `n_conditions`: Number of conditions (int)

**Example**:

```python
df, metadata = read_pcl_file("gasch_2000_environmental.pcl")
print(f"Loaded {metadata['n_genes']} genes × {metadata['n_conditions']} conditions")
print(f"First condition: {metadata['conditions'][0]}")
```

### extract_and_load_all_spell_studies(spell_root_dir, studies_to_load=None, max_studies=None)

Batch loader for multiple SPELL studies. Extracts ZIP archives and loads all PCL files.

**Args**:

- `spell_root_dir`: Root directory containing `.zip` study archives
- `studies_to_load`: Optional list of study names to load (None = all studies)
- `max_studies`: Maximum number of studies to load (None = unlimited)

**Returns**:

- `all_data`: Dictionary mapping `(study_name, dataset_name)` → `(df, metadata)`

**Process**:

1. Scans directory for `.zip` files
2. Extracts each archive (if not already extracted)
3. Finds all `.pcl` files within each study directory
4. Loads each PCL file using `read_pcl_file()`
5. Stores in dictionary with composite key

**Example**:

```python
# Load all studies
all_data = extract_and_load_all_spell_studies("/path/to/spell")

# Load only first 10 studies (for testing)
all_data = extract_and_load_all_spell_studies("/path/to/spell", max_studies=10)

# Load specific studies
all_data = extract_and_load_all_spell_studies(
    "/path/to/spell",
    studies_to_load=['Gasch_2000_PMID_11102521']
)
```

**Output Format**:

```python
{
    ('Gasch_2000_PMID_11102521', 'gasch_2000_environmental'): (df1, metadata1),
    ('Spellman_1998_PMID_9843569', 'spellman_1998_alpha'): (df2, metadata2),
    ...
}
```

### export_condition_metadata(all_data, output_path=None)

**The Core Metadata Extraction Function**: Parses free-text condition descriptions into structured environmental parameters using regex patterns.

**Args**:

- `all_data`: Dictionary from `extract_and_load_all_spell_studies()`
- `output_path`: Path to save CSV (default: `DATA_ROOT/data/sgd/spell/spell_conditions_metadata_enhanced.csv`)

**Returns**:

- `df_conditions`: pandas DataFrame with enhanced metadata (~20 columns × ~14,000 rows)

**Extracted Fields**:

**Study Identification**:

- `study_name`: SPELL study identifier
- `dataset_name`: Dataset within study
- `condition_name`: Original free-text description
- `condition_index`: Index within study
- `n_genes`: Number of genes measured

**Categorization**:

- `categories`: All matched categories (pipe-separated)
- `primary_category`: Main category (first match)
- `secondary_categories`: Additional categories

**Temporal Information**:

- `time_min`: Extracted time in minutes (float)
- `is_timeseries`: Boolean flag for time-series experiments

**Temperature**:

- `temperature_c`: Extracted temperature in Celsius (float)

**Chemical Treatment**:

- `chemical_name`: Compound name (string)
- `concentration`: Numerical concentration value (float)
- `concentration_unit`: Unit (mM, μM, %, mg/mL, etc.)

**Nutrient Information**:

- `carbon_source`: Carbon source (glucose, galactose, raffinose, etc.)
- `nitrogen_source`: Nitrogen source (ammonia, proline, etc.)
- `limitation_type`: Type of nutrient limitation

**Physical Parameters**:

- `ph`: pH value (float, 0-14)
- `oxygen_level`: Oxygen availability (aerobic, anaerobic, hypoxic)

**Stress Information**:

- `stress_type`: Type of stress applied

**Cell Cycle**:

- `cell_cycle_phase`: Phase (G1, S, G2, M)
- `synchronization_method`: Synchronization method (alpha factor, etc.)

**Replication**:

- `replicate_number`: Replicate number (int)
- `replicate_type`: Replicate type (biological, technical)

**Quality Metrics**:

- `extraction_confidence`: Score 0.0-1.0 (how many fields extracted)
- `needs_manual_review`: Boolean flag (low confidence or ambiguous)

**Category Keywords**:

The function uses keyword matching to assign categories:

- **heat_shock**: heat, temperature, thermal, hs
- **oxidative_stress**: oxidative, peroxide, h2o2, menadione, diamide
- **osmotic_stress**: osmotic, sorbitol, nacl, salt
- **nutrient_limitation**: nitrogen, carbon, phosphate, starvation, limited
- **drug_treatment**: drug, rapamycin, tunicamycin, cycloheximide
- **cell_cycle**: cell cycle, g1, g2, s phase, alpha factor
- **dna_damage**: dna, uv, radiation, mms, damage
- **chemical_stress**: chemical, dtt, cadmium, arsenite
- **anaerobic**: anaerobic, hypoxia, oxygen
- **time_series**: min, hour, time
- **mutant_strain**: mutant, deletion, knockout, overexpression
- **wild_type_control**: wild type, wt, control, untreated
- **uncategorized**: No matches found
- **numeric_id**: Pure numeric identifier
- **array_id**: Array/chip identifier

**Extraction Confidence Scoring**:

Confidence calculated based on number of fields successfully extracted:

- **Score = 0.1 + (extracted_fields / 14) × 0.8**
- Range: 0.1 (no fields) to 0.9 (all fields)
- 0 fields → 0.1 confidence
- 5 fields → ~0.4 confidence
- 10+ fields → 0.7-0.9 confidence

**Manual Review Flags**:

Conditions flagged for manual review if:

- Confidence < 0.2 (very few fields extracted)
- Categorized as "uncategorized"
- Has concentration but no chemical name (ambiguous)

**Example Usage**:

```python
# Load all SPELL studies
all_data = extract_and_load_all_spell_studies("/path/to/spell")

# Extract metadata
df_conditions = export_condition_metadata(all_data)

# Filter for heat shock experiments
heat_shock = df_conditions[df_conditions['primary_category'] == 'heat_shock']

# Filter for high-confidence extractions
high_conf = df_conditions[df_conditions['extraction_confidence'] > 0.5]

# Filter conditions needing review
needs_review = df_conditions[df_conditions['needs_manual_review'] == True]
```

## Helper Extraction Functions

The module includes specialized regex-based extraction functions (not typically called directly):

### extract_time_info(condition_str)

Extracts temporal information:

- Patterns: `10min`, `2 hours`, `4h`, `time series`, `20 min`
- Returns: `{'time_min': float, 'is_timeseries': bool}`

### extract_temperature_info(condition_str)

Extracts temperature:

- Patterns: `37C`, `30°C`, `42 degrees`
- Returns: `{'temperature_c': float}`

### extract_chemical_info(condition_str)

Extracts chemical compounds and concentrations:

- Patterns: `0.5mM H2O2`, `10% ethanol`, `200mM NaCl`
- Returns: `{'chemical_name': str, 'concentration': float, 'concentration_unit': str}`

### extract_nutrient_info(condition_str)

Extracts nutrient sources and limitations:

- Patterns: `glucose`, `galactose`, `nitrogen limitation`, `carbon starvation`
- Returns: `{'carbon_source': str, 'nitrogen_source': str, 'limitation_type': str}`

### extract_physical_params(condition_str)

Extracts physical parameters:

- Patterns: `pH 7.5`, `anaerobic`, `hypoxic`
- Returns: `{'ph': float, 'oxygen_level': str}`

### extract_stress_info(condition_str)

Extracts stress type information:

- Returns: `{'stress_type': str}`

### extract_cell_cycle_info(condition_str)

Extracts cell cycle information:

- Patterns: `G1 phase`, `alpha factor arrest`, `M phase`
- Returns: `{'cell_cycle_phase': str, 'synchronization_method': str}`

### extract_replicate_info(condition_str)

Extracts replicate information:

- Patterns: `rep1`, `replicate 2`, `biological replicate`
- Returns: `{'replicate_number': int, 'replicate_type': str}`

## Visualization Functions

### plot_expression_histograms(df, metadata, gene_list=None, title_prefix="SPELL", save_path=None)

Plots log2 expression ratio distributions for specific genes across all conditions.

**Args**:

- `df`, `metadata`: From `read_pcl_file()`
- `gene_list`: List of ORF names (default: first 5 genes)
- `title_prefix`: Plot title prefix
- `save_path`: Path to save figure (None = display)

**Output**: Multi-panel histogram figure (one panel per gene)

### plot_global_expression_distribution(df, metadata, title_prefix="SPELL", save_path=None)

Plots global distribution of all expression values across all genes and conditions.

**Args**:

- `df`, `metadata`: From `read_pcl_file()`
- `title_prefix`: Plot title prefix
- `save_path`: Path to save figure

**Output**: Single histogram with statistics (mean, median, std)

### plot_genes_across_all_studies(all_data, gene_list, save_path=None)

Plots expression distributions for specific genes aggregated across ALL SPELL studies.

**Args**:

- `all_data`: From `extract_and_load_all_spell_studies()`
- `gene_list`: List of ORF names to plot
- `save_path`: Path to save figure

**Output**: Multi-panel histogram (one panel per gene, aggregated across all studies)

## Usage Examples

### Basic Loading

```python
from torchcell.datasets.scerevisiae.spell import (
    read_pcl_file,
    extract_and_load_all_spell_studies,
    export_condition_metadata
)

# Load single PCL file
df, metadata = read_pcl_file("/path/to/study.pcl")
print(f"Genes: {metadata['n_genes']}, Conditions: {metadata['n_conditions']}")

# Load all studies
all_data = extract_and_load_all_spell_studies("/path/to/spell")
print(f"Loaded {len(all_data)} datasets")

# Extract metadata
df_conditions = export_condition_metadata(all_data)
print(f"Extracted metadata for {len(df_conditions)} conditions")
```

### Phase 1 Pipeline

See [[experiments.015-spell.scripts.run_phase1_spell_analysis]] for the complete Phase 1 pipeline that uses this module.

```python
# Step 1: Load all SPELL studies
all_data = extract_and_load_all_spell_studies(spell_root_dir)

# Step 2: Extract enhanced metadata
df_conditions = export_condition_metadata(
    all_data,
    output_path="spell_conditions_metadata_enhanced.csv"
)

# Step 3: Analyze (see spell_coverage_analysis module)
# ...
```

### Filtering and Analysis

```python
# Load metadata CSV
df = pd.read_csv("spell_conditions_metadata_enhanced.csv")

# Filter by category
heat_shock_conditions = df[df['primary_category'] == 'heat_shock']

# Filter by extracted parameters
high_temp = df[df['temperature_c'] > 37]
time_series = df[df['is_timeseries'] == True]
h2o2_treatment = df[df['chemical_name'].str.contains('H2O2', na=False)]

# Filter by quality
high_quality = df[
    (df['extraction_confidence'] > 0.5) &
    (df['needs_manual_review'] == False)
]

# Multi-factor experiments
multi_category = df[df['secondary_categories'].notna() & (df['secondary_categories'] != '')]
```

## Implementation Notes

### Regex Pattern Design

Extraction functions use carefully designed regex patterns:

**Time Extraction**:

- Pattern: `(\d+)\s*(min|minute|minutes|h|hour|hours)`
- Handles: `10min`, `2 hours`, `30 min`, `4h`
- Converts to minutes for standardization

**Temperature Extraction**:

- Pattern: `(\d+)\s*(C|°C|degrees|deg)`
- Handles: `37C`, `30°C`, `42 degrees`
- Returns float in Celsius

**Concentration Extraction**:

- Pattern: `(\d+\.?\d*)\s*(mM|μM|uM|%|mg/mL|M)`
- Handles: `0.5mM`, `10%`, `200μM`
- Returns value + unit separately

### Limitations and Challenges

**Low Overall Confidence (~0.176 mean)**:

- Many conditions use non-standard descriptions
- Abbreviations and lab-specific terminology
- Complex multi-factor experiments hard to parse
- Implicit information (e.g., "WT" without stating temperature)

**Parsing Errors**:

- Numbers from unrelated contexts (e.g., "25000 rpm" → "25000°C")
- Ambiguous chemical names (common names vs. systematic)
- Multiple chemicals in one condition (currently extracts first match only)

**Category Overlap**:

- Some conditions match multiple categories (by design)
- Order of keyword matching affects primary category assignment
- "Uncategorized" dominates (~46% of conditions) → opportunity for improvement

### Future Enhancements

**Phase 2**: Implement Environment subclasses based on priority scores

**Phase 3**: LLM-based extraction for complex conditions

- Use GPT-4 or Claude to parse low-confidence conditions
- Extract from linked publications for missing metadata
- Validate/correct regex-based extractions

**Schema Integration**:

- Map extracted metadata to Environment schema fields
- Create Environment-annotated SPELL datasets
- Enable environment-aware model training

## Related Documentation

- **Pipeline Runner**: [[experiments.015-spell.scripts.run_phase1_spell_analysis]]
- **Coverage Analysis**: [[experiments.015-spell.scripts.spell_coverage_analysis]]
- **Main Pipeline**: [[experiments.015-spell.scripts.spell_analysis]]

## Data Quality Summary

From Phase 1 analysis (~14,000 conditions):

- **Mean extraction confidence**: 0.176 (relatively low)
- **High confidence (>0.5)**: ~5% of conditions
- **Needs manual review**: ~60% of conditions
- **Top extracted fields**: replicate info (>50%), concentration (25%), time (20%)
- **Poorly extracted fields**: pH (<5%), nitrogen source (<1%), temperature (5%)

**Conclusion**: Regex-based extraction provides good baseline for common experimental patterns, but LLM-based extraction needed for comprehensive coverage.
