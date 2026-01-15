---
id: jis8r41wyvknl19zqwdr0nq
title: Spell_analysis
desc: ''
updated: 1768427888115
created: 1768423514140
---

## Overview

Executes the complete SPELL (Saccharomyces Genome Database Platform for Expression data archival and Laboratory analysis) condition coverage analysis pipeline. This Phase 1 analysis loads all SPELL microarray studies, extracts structured metadata from condition descriptions, and generates comprehensive coverage reports to prioritize Environment subclass implementation.

## Data Source

### SPELL Database

- **Source**: SGD Expression Archive
- **URL**: <http://sgd-archive.yeastgenome.org/expression/microarray/all_spell_datasets.tar.gz>
- **Format**: PCL (Platform for Clustering and Linkage) files
- **Content**: Gene expression microarray data with condition metadata
- **Organism**: *Saccharomyces cerevisiae* (baker's yeast)

## Pipeline Components

### 1. run_phase1_spell_analysis.py

Main orchestrator script that executes the complete Phase 1 pipeline:

**Step 1: Load SPELL Studies**

- Reads all PCL files from SPELL data directory
- Parses expression matrices (genes × conditions)
- Extracts condition names and experimental weights
- Optional: `--max-studies N` for testing with subset
- Optional: `--spell-dir PATH` for custom SPELL location

**Step 2: Extract Enhanced Metadata**

- Processes condition descriptions using regex patterns
- Extracts structured parameters:
  - **Time**: time_min, is_timeseries
  - **Temperature**: temperature_c
  - **Chemicals**: chemical_name, concentration, concentration_unit
  - **Nutrients**: carbon_source, nitrogen_source, limitation_type
  - **Stress**: stress_type, oxygen_level
  - **Cell cycle**: cell_cycle_phase, synchronization_method
  - **Replication**: replicate_number, replicate_type
- Assigns primary and secondary category tags
- Calculates extraction confidence score
- Flags conditions needing manual review
- Outputs: `spell_conditions_metadata_enhanced.csv`

**Step 3: Run Coverage Analysis**

- Calls `spell_coverage_analysis.py` to analyze extracted metadata

### 2. spell_coverage_analysis.py

Comprehensive analysis module that generates:

**Frequency Distributions**

- Condition counts by primary category
- Top 20 categories with percentages
- Unique category combinations

**Co-occurrence Analysis**

- Matrix of which categories appear together
- Top 20 category pairs
- Identifies common experimental patterns

**Parameter Range Analysis**

- Temperature: min/max/mean/median distributions
- Time: temporal ranges for time-series experiments
- pH: pH value distributions
- Concentrations: grouped by unit (mM, μM, %, etc.)
- Top chemical compounds used
- Carbon/nitrogen sources enumerated
- Oxygen level categories

**Missing Data Analysis**

- Field-by-field completeness percentages
- Identifies extraction gaps
- Guides manual review priorities

**Environment Subclass Prioritization**

Ranks potential Environment subclasses by:

- **Frequency score (40%)**: How many conditions match
- **Completeness score (30%)**: How well required fields are extracted
- **Importance score (30%)**: Scientific relevance (manually assigned)

Prioritized classes include:

1. TimeSeriesEnvironment
2. NutrientEnvironment
3. HeatShockEnvironment
4. OxidativeStressEnvironment
5. OsmoticStressEnvironment
6. AnaerobicEnvironment
7. ChemicalStressEnvironment
8. DrugTreatmentEnvironment
9. CellCycleEnvironment
10. DNADamageEnvironment (lower priority - causes genotypic changes)

## Outputs

### CSV Files (in `DATA_ROOT/data/sgd/spell/`)

1. **`spell_conditions_metadata_enhanced.csv`** - Enhanced metadata
   - Columns: study_id, condition_name, primary_category, secondary_categories, time_min, temperature_c, chemical_name, concentration, carbon_source, nitrogen_source, ph, oxygen_level, stress_type, cell_cycle_phase, replicate_number, extraction_confidence, needs_manual_review

### Markdown Report (in `DATA_ROOT/data/sgd/spell/`)

1. **`spell_coverage_report.md`** - Comprehensive analysis report
   - Executive summary with key statistics
   - Environment subclass implementation order (ranked table)
   - Category frequency distribution
   - Data extraction completeness by field
   - Recommended next steps

### PNG Visualizations (in `notes/assets/images/`)

1. **`spell_coverage_analysis_<timestamp>.png`** - 4-panel figure:
   - **Top-left**: Top 15 condition categories (bar chart)
   - **Top-right**: Data extraction completeness by field (horizontal bar, color-coded)
   - **Bottom-left**: Extraction confidence score distribution (histogram)
   - **Bottom-right**: Temperature distribution (histogram with median line)

All plots saved at 300 DPI with timestamps for iterative refinement.

## Usage

### From Bash Script (Recommended)

```bash
# Run full pipeline
bash experiments/015-spell/scripts/spell_analysis.sh

# Test with small dataset
python experiments/015-spell/scripts/run_phase1_spell_analysis.py --max-studies 10

# Use existing metadata (skip extraction)
python experiments/015-spell/scripts/run_phase1_spell_analysis.py --skip-extraction

# Custom SPELL directory
python experiments/015-spell/scripts/run_phase1_spell_analysis.py --spell-dir /custom/path
```

### From Python Directly

```bash
# Full pipeline
python experiments/015-spell/scripts/run_phase1_spell_analysis.py

# Just coverage analysis (requires existing CSV)
python experiments/015-spell/scripts/spell_coverage_analysis.py
```

## Key Metrics Reported

1. Total conditions analyzed
2. Mean extraction confidence score
3. Conditions needing manual review (count + percentage)
4. High confidence conditions (>0.5 score)
5. Category frequency distribution
6. Field-by-field data completeness
7. Priority scores for Environment subclasses
8. Top category co-occurrences
9. Parameter ranges (temp, time, pH, concentrations)

## Implementation Notes

### Path Handling

- Uses `DATA_ROOT` from `.env` for SPELL data location
- Uses `ASSET_IMAGES_DIR` from `.env` for visualization output
- Falls back to hardcoded paths if env vars missing
- Must run from torchcell root directory

### Import Structure

- `run_phase1_spell_analysis.py` imports from:
  - `torchcell.datasets.scerevisiae.spell` (data loading/extraction)
  - `spell_coverage_analysis` (local import from same directory)
- Removed `torchcell/analysis/` module - all code in experiment dir

### Metadata Extraction Strategy

Uses regex patterns to parse free-text condition descriptions:

- **Time patterns**: "10min", "2 hours", "time series"
- **Temperature**: "37C", "30°C", "heat shock"
- **Chemicals**: "0.5mM H2O2", "10% ethanol"
- **Categories**: heat_shock, oxidative_stress, nutrient_limitation, etc.

Confidence scoring based on:

- Number of fields successfully extracted
- Presence of known category keywords
- Secondary category tags identified

### Visualization Styling

- Matplotlib/Seaborn with 'whitegrid' style
- Color coding for completeness: green (>50%), orange (20-50%), red (<20%)
- Timestamps in filenames for version control during iteration
- 300 DPI for publication quality

## Next Steps (Phase 2)

1. Implement top 3-5 priority Environment subclasses in `torchcell.datamodels.schema`
2. Review conditions flagged for manual review
3. Set up LLM extraction pipeline for paper-based metadata (Phase 3)
4. Iterate on schema design based on extraction results
5. Integrate SPELL data into training datasets with Environment annotations
