---
id: 0g1e0wm78avfr056it3j9u7
title: Run_phase1_spell_analysis
desc: ''
updated: 1768427884589
created: 1768427144314
---

## Overview

Main orchestrator script that executes the complete SPELL Phase 1 pipeline. This script coordinates three major steps: loading SPELL studies, extracting structured metadata, and running coverage analysis. It serves as the primary entry point for the entire SPELL condition coverage analysis workflow.

## Pipeline Architecture

### Three-Step Sequential Process

**Step 1: Load SPELL Studies** → **Step 2: Extract Metadata** → **Step 3: Coverage Analysis**

Each step can be controlled independently via command-line arguments.

## Functionality

### Step 1: Load SPELL Studies

**Module**: `torchcell.datasets.scerevisiae.spell.extract_and_load_all_spell_studies()`

**Inputs**:

- SPELL root directory (PCL files)
- Optional: `--max-studies N` for testing subset
- Optional: `--spell-dir PATH` for custom location

**Process**:

1. Scans directory for all `.pcl` files
2. Parses each PCL file (tab-delimited format):
   - Header row: condition names
   - EWEIGHT row: experimental weights
   - Data rows: gene expression values
3. Extracts condition metadata from filenames and headers
4. Loads expression matrices (genes × conditions)

**Outputs**:

- `all_data` dictionary: study metadata + expression DataFrames
- Console: Study count, condition count, gene count

**Time**: Several minutes for full SPELL database (~200+ studies)

### Step 2: Extract Enhanced Metadata

**Module**: `torchcell.datasets.scerevisiae.spell.export_condition_metadata()`

**Inputs**:

- `all_data` from Step 1
- Output path for CSV

**Process**:

1. Iterates through all conditions across all studies
2. Parses free-text condition descriptions with regex patterns:
   - **Time**: `10min`, `2 hours`, `time series`, `20 min`, `4h`
   - **Temperature**: `37C`, `30°C`, `42 degrees`, `heat shock`
   - **Chemicals**: `0.5mM H2O2`, `10% ethanol`, `200mM NaCl`
   - **Categories**: nutrient_limitation, oxidative_stress, heat_shock, etc.
3. Extracts structured parameters into fields:
   - Numerical: time_min, temperature_c, concentration, ph
   - Categorical: chemical_name, carbon_source, oxygen_level, stress_type
   - Flags: is_timeseries, needs_manual_review
4. Assigns primary_category and secondary_categories tags
5. Calculates extraction_confidence score (0.0-1.0)

**Outputs**:

- **CSV**: `DATA_ROOT/data/sgd/spell/spell_conditions_metadata_enhanced.csv`
  - Columns: ~20 fields (study_id, condition_name, all extracted parameters)
  - Rows: ~14,000 conditions
- Console: Extraction statistics, condition count

**Extraction Confidence Scoring**:

- Based on number of fields extracted
- Based on category match quality
- Higher score = more information successfully parsed
- Mean score ~0.176 (relatively low → many conditions need manual review)

### Step 3: Run Coverage Analysis

**Module**: `spell_coverage_analysis.main()`

**Inputs**:

- Enhanced metadata CSV from Step 2

**Process**:
See [[experiments.015-spell.scripts.spell_coverage_analysis]] for details.

**Outputs**:

- Markdown report: `DATA_ROOT/data/sgd/spell/spell_coverage_report.md`
- Visualization: `notes/assets/images/015-spell/spell_coverage_analysis.png`
- Console: Comprehensive statistics

## Command-Line Arguments

### Required Arguments

None - all arguments are optional with sensible defaults.

### Optional Arguments

**`--max-studies N`**

- Load only first N studies (for testing)
- Default: None (load all studies)
- Example: `--max-studies 10` for quick test run
- Use case: Development, debugging regex patterns

**`--skip-extraction`**

- Skip metadata extraction if CSV already exists
- Default: False (always extract)
- Use case: Re-run coverage analysis without re-extraction
- Saves time if you only modified analysis code

**`--spell-dir PATH`**

- Custom path to SPELL data directory
- Default: `DATA_ROOT/data/sgd/spell`
- Use case: SPELL data stored in non-standard location

## Usage Examples

### Full Pipeline (Production)

```bash
# From torchcell root directory
python experiments/015-spell/scripts/run_phase1_spell_analysis.py
```

**Expected Runtime**: 5-10 minutes
**Output**: All files generated

### Quick Test (Development)

```bash
# Load only 10 studies for testing
python experiments/015-spell/scripts/run_phase1_spell_analysis.py --max-studies 10
```

**Expected Runtime**: 30-60 seconds
**Output**: Partial data, good for testing regex patterns

### Re-run Analysis Only

```bash
# Use existing CSV, skip extraction
python experiments/015-spell/scripts/run_phase1_spell_analysis.py --skip-extraction
```

**Expected Runtime**: 1-2 minutes
**Output**: Updated report and visualization (CSV unchanged)
**Use case**: Modified plotting code or analysis logic

### Custom SPELL Location

```bash
# SPELL data in non-standard location
python experiments/015-spell/scripts/run_phase1_spell_analysis.py --spell-dir /custom/path/to/spell
```

## Outputs

### CSV Files

**Location**: `DATA_ROOT/data/sgd/spell/`

**File**: `spell_conditions_metadata_enhanced.csv`

**Size**: ~14,000 rows × ~20 columns

**Key Columns**:

- `study_id`: SPELL study identifier
- `condition_name`: Original free-text description
- `primary_category`: Main experimental category
- `secondary_categories`: Pipe-separated additional categories
- `time_min`: Extracted time value
- `temperature_c`: Extracted temperature
- `chemical_name`: Extracted chemical compound
- `concentration`: Numerical concentration value
- `concentration_unit`: Unit (mM, μM, %, mg/mL)
- `carbon_source`, `nitrogen_source`, `limitation_type`
- `ph`, `oxygen_level`, `stress_type`
- `cell_cycle_phase`, `synchronization_method`
- `replicate_number`, `replicate_type`
- `is_timeseries`: Boolean flag
- `extraction_confidence`: Score 0.0-1.0
- `needs_manual_review`: Boolean flag

### Markdown Report

**Location**: `DATA_ROOT/data/sgd/spell/`

**File**: `spell_coverage_report.md`

**Sections**:

1. Executive Summary
   - Total conditions
   - Mean extraction confidence
   - Conditions needing manual review
   - High-confidence condition count
2. Environment Subclass Priority Ranking
   - Ranked table with scores
3. Category Frequency Distribution (top 20)
4. Data Extraction Completeness by Field
5. Next Steps

### Visualization

**Location**: `notes/assets/images/015-spell/`

**File**: `spell_coverage_analysis.png`

**Format**: PNG, 300 DPI, 18" × 14"

**Content**: See [[experiments.015-spell.scripts.spell_coverage_analysis#visualizations]]

## Error Handling

### SPELL Data Not Found

If SPELL directory doesn't exist, script provides download instructions:

```
ERROR: SPELL data directory not found: /path/to/spell

To download SPELL data, run these commands:
  mkdir -p /path/to/spell
  cd /path/to/spell
  curl -O http://sgd-archive.yeastgenome.org/expression/microarray/all_spell_datasets.tar.gz
  tar -xzf all_spell_datasets.tar.gz
```

### CSV Already Exists (with --skip-extraction)

Prints confirmation and skips to Step 3:

```
✓ Skipping extraction - using existing metadata:
  /path/to/spell_conditions_metadata_enhanced.csv
```

## Console Output Structure

### Step 1 Output

```
======================================================================
STEP 1: LOADING SPELL DATA
======================================================================

Loading ALL SPELL studies (this will take several minutes)...
✓ Loaded study 1/214: spellman_1998_alpha.pcl (18 conditions)
✓ Loaded study 2/214: gasch_2000_environmental.pcl (173 conditions)
...
```

### Step 2 Output

```
======================================================================
STEP 2: EXTRACTING ENHANCED METADATA
======================================================================

Processing conditions...
✓ Enhanced metadata extraction complete!
  Output: /path/to/spell_conditions_metadata_enhanced.csv
  Total conditions: 13,984
```

### Step 3 Output

```
======================================================================
STEP 3: RUNNING COVERAGE ANALYSIS
======================================================================

[Detailed analysis output - see spell_coverage_analysis.md]
```

### Final Summary

```
======================================================================
PHASE 1 COMPLETE
======================================================================

Next steps:
  1. Review coverage report: data/sgd/spell/spell_coverage_report.md
  2. Examine visualizations in notes/assets/images/015-spell/
  3. Proceed to Phase 2: Design Environment hierarchy based on priorities
======================================================================
```

## Implementation Notes

### Path Configuration

- **DATA_ROOT**: Read from `.env` with fallback
- **SPELL directory**: `DATA_ROOT/data/sgd/spell` by default
- **CSV output**: Same directory as SPELL data
- **Image output**: `notes/assets/images/015-spell/`

### Import Strategy

**External Imports**:

- `torchcell.datasets.scerevisiae.spell`: Data loading and extraction functions

**Local Imports**:

- `spell_coverage_analysis`: Analysis module in same directory
- Changed from module import (`torchcell.analysis.*`) to local import after directory restructure

### Argument Parsing

Uses `argparse` for clean CLI interface:

- All arguments optional with defaults
- Help text via `--help`
- Type checking for `--max-studies` (must be integer)

### Progress Tracking

Each step clearly delineated with:

- 70-character separator lines
- Step numbers and descriptions
- Success checkmarks (✓)
- File paths for outputs
- Condition/study counts

## Integration with Other Scripts

### Called by Bash Runner

`experiments/015-spell/scripts/spell_analysis.sh`:

```bash
python experiments/015-spell/scripts/run_phase1_spell_analysis.py
```

Simple wrapper for convenience.

### Calls Analysis Module

This script orchestrates, `spell_coverage_analysis.py` performs analysis:

```python
from spell_coverage_analysis import main as run_coverage_analysis
run_coverage_analysis()
```

### Data Flow

```
SPELL PCL files
    ↓ (extract_and_load_all_spell_studies)
all_data dictionary
    ↓ (export_condition_metadata)
spell_conditions_metadata_enhanced.csv
    ↓ (spell_coverage_analysis.main)
Markdown report + Visualization
```

## Next Steps (Phase 2)

After running Phase 1:

1. **Review Outputs**
   - Open `spell_coverage_report.md`
   - Examine `spell_coverage_analysis.png`
   - Understand priority rankings

2. **Design Environment Schema**
   - Implement top 3-5 Environment subclasses
   - Use priority scores to guide order
   - Reference parameter ranges for field definitions

3. **Manual Review**
   - Filter CSV for `needs_manual_review == True`
   - Identify systematic extraction errors
   - Improve regex patterns in `spell.py`

4. **LLM Extraction (Phase 3)**
   - Set up LLM pipeline for low-confidence conditions
   - Focus on fields with <10% completeness
   - Especially for complex multi-factor experiments

5. **Iteration**
   - Re-run pipeline with improved extraction
   - Compare before/after completeness scores
   - Validate Environment schema with real data
