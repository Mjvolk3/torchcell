---
id: q7v2ffyvirsrdz101bm386g
title: FBA-interaction-experiments
desc: ''
updated: 1761088725238
created: 1761088676695
---
## Overview and Objectives

We implemented a comprehensive COBRA FBA (Constraint-Based Reconstruction and Analysis - Flux Balance Analysis) pipeline to predict fitness and genetic interactions for yeast gene deletions, comparing these predictions against experimental trigenic interaction data from Kuzmin et al., 2018. The pipeline processes approximately 1 million gene combinations (4K singles, 651K doubles, 332K triples) through metabolic simulations using the Yeast9 GEM model.

## Data Extraction and Preparation

### Neo4j Dataset Integration

We began by extracting unique gene perturbations from a Neo4j database containing experimental trigenic interaction measurements. The script `extract_perturbations.py` queries the Neo4jCellDataset to identify all unique single, double, and triple gene combinations present in the experimental data. This extraction occurs once and produces `unique_perturbations.json`, a file containing:

- 4,036 single gene deletions
- 651,181 double gene combinations
- 332,313 triple gene combinations

The extraction uses the query defined in `experiments/007-kuzmin-tm/queries/001_small_build.cql` to retrieve genotypes with measured fitness and gene interaction values from Kuzmin et al.'s systematic analysis.

### Gene Set Reconciliation

A critical challenge emerged: the experimental dataset contains ~6,000 genes while the Yeast9 metabolic model only represents ~1,000 genes involved in metabolism. We handle this discrepancy by assigning wild-type fitness (1.0) to genes absent from the metabolic model, representing the biological reality that non-metabolic genes have no effect in FBA simulations. This approach prevents NaN propagation in interaction calculations while maintaining mathematical consistency.

## Media Condition Development

### Research Phase: Finding Appropriate Media Formulations

Initially, we ran FBA simulations with minimal media conditions, achieving poor correlation with experimental data (Pearson r < 0.1). The experimental data was collected in YPD (Yeast Peptone Dextrose) media, a nutrient-rich condition that differs significantly from minimal media.

We researched YPD media formulations for FBA, examining:

- Duarte et al., 2004 - foundational YPD modeling paper
- SysBioChalmers/yeast-GEM repository - community consensus model
- Suthers et al., 2020 - recent media composition for metabolic modeling

### Implementation: Three-Tier Media System

Based on Suthers et al., 2020, we implemented three media conditions in `setup_media_conditions.py`:

**Minimal Media:**

- Glucose at variable rates (2-20 mmol/gDW/h)
- Ammonium as nitrogen source
- Oxygen at variable rates (5-1000 mmol/gDW/h)
- Essential inorganic salts (phosphate, sulfate, trace elements)
- Total: ~17 open exchange reactions

**YNB Media (Yeast Nitrogen Base):**

- Minimal media base plus vitamins/cofactors
- 9 components added at 5% of glucose uptake rate
- Includes: thiamine, riboflavin, nicotinate, pyridoxine, folate, pantothenate, 4-aminobenzoate, myo-inositol, biotin
- Approximates vitamin-supplemented minimal media
- Total: ~26 open exchange reactions

**YPD Media:**

- YNB media plus 20 standard amino acids
- Each amino acid supplied at 5% of glucose uptake rate
- Closest approximation to experimental YPD conditions
- Total: ~46 open exchange reactions

The Suthers et al. model (`experiments/007-kuzmin-tm/1-s2.0-S2214030120300481-mmc8/iIsor850.json`) for *Issatchenkia orientalis* served as our reference for media composition, adapted for *S. cerevisiae*.

## FBA Pipeline Architecture

### Initial Single-Media Pipeline

Our first complete pipeline (`gh_cobra-fba-growth.slurm`) performed FBA analysis with a single media condition:

1. **Verification scripts** validated model parameters:
   - `verify_gpr_knockout_logic.py` - checked gene-protein-reaction logic
   - `verify_biomass_maintenance.py` - validated growth parameters
   - `glucose_oxygen_sensitivity.py` - tested constraint sensitivity

2. **FBA execution** (`targeted_fba_growth_fast.py`):
   - Loaded Yeast9 model
   - Applied gene deletions using COBRA's gene knockout methods
   - Used GLPK solver with 60-second timeout
   - Parallelized across 128 CPUs

3. **Post-processing** (`match_fba_to_experiments.py`):
   - Matched FBA predictions to experimental data
   - Initial interaction calculation (later found incorrect)

4. **Visualization** (`plot_fba_comparison.py`):
   - Generated scatter plots comparing predicted vs experimental

### Evolution: Multi-Media Comparison Pipeline

Recognizing media's importance, we developed an enhanced pipeline (`gh_cobra-fba-growth-all-media.slurm`) that systematically compared all three media conditions:

**Wrapper Script** (`run_fba_all_media.py`):

- Orchestrates the entire pipeline
- Calls `verify_media_differences.py` to validate media setup
- Sequentially runs FBA for minimal, YNB, and YPD media
- Each media condition calls `targeted_fba_with_media_yeastgem.py`

**Core FBA Script** (`targeted_fba_with_media_yeastgem.py`):

- Replaced the original `targeted_fba_growth_fast.py`
- Integrated media setup functions from `setup_media_conditions.py`
- Implemented spawn-based multiprocessing (replacing fork) to avoid memory issues
- Pickles the model once, then spawns 128 workers
- Each worker:
  - Loads pickled model
  - Applies media constraints
  - Performs gene knockouts
  - Runs FBA optimization
  - Returns growth rate

**Multiprocessing Architecture:**

```python
# Use spawn instead of fork (line 18-22)
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Worker initialization
def init_worker(model_pickle, media_type):
    global _model, _media_type
    _model = pickle.loads(model_pickle)
    _media_type = media_type
```

The spawn method prevents memory corruption issues encountered with fork when using 128 workers processing large models.

### Post-Processing and Interaction Calculations

**Initial Formula Error:**

Originally, we calculated trigenic interactions using only the triple mutant and single mutant fitness values:

```python
τ_ijk = f_ijk - f_i * f_j * f_k  # INCORRECT
```

**Corrected Formula Implementation:**

Following the complete formula from Kuzmin et al., we implemented in `postprocess_fba_all_media.py`:

```python
def calculate_fba_triple_interactions(singles_df, doubles_df, triples_df):
    # Get all component fitnesses
    f_i, f_j, f_k = singles_fitness lookups
    f_ij, f_ik, f_jk = doubles_fitness lookups

    # Calculate digenic interactions
    epsilon_ij = f_ij - f_i * f_j
    epsilon_ik = f_ik - f_i * f_k
    epsilon_jk = f_jk - f_j * f_k

    # Calculate trigenic interaction
    tau_ijk = f_ijk - f_i*f_j*f_k - epsilon_ij*f_k - epsilon_ik*f_j - epsilon_jk*f_i
```

This correction significantly improved correlation with experimental interaction values.

**Data Matching Process:**

The post-processing script:

1. Loads experimental data from Neo4j (once for all media)
2. For each media condition:
   - Loads FBA results (singles, doubles, triples parquet files)
   - Calculates digenic and trigenic interactions
   - Matches to experimental data by genotype
   - Saves matched results with both fitness and interaction comparisons

### Visualization Pipeline

`plot_fba_all_media.py` generates a comprehensive 3×2 grid visualization:

- Rows: minimal, YNB, YPD media
- Columns: fitness predictions, interaction predictions
- Each panel shows scatter plot with Pearson correlation
- Results saved to `ASSET_IMAGES_DIR` with timestamp

## Glucose-Oxygen Sensitivity Analysis

### Rationale: Addressing Discrete Fitness Bands

FBA predictions showed discrete fitness bands (0.0, 0.15, 0.4, 1.0) rather than continuous distributions. Following Vikas Upadhyay's recommendation, we tested whether these bands were:

1. **Constraint-driven** - would shift with different nutrient availability
2. **Model-intrinsic** - persist regardless of conditions

### Implementation: 48-Condition Sweep

The sensitivity analysis pipeline (`gh_glucose_oxygen_sensitivity_all_media.slurm`) tests:

- 3 media types (minimal, YNB, YPD)
- 4 glucose levels (2, 5, 10, 20 mmol/gDW/h)
- 4 oxygen levels (5, 10, 20, 1000 mmol/gDW/h)

**Wrapper Script** (`run_glucose_oxygen_sweep.py`):

- Iterates through all 48 conditions
- Calls `glucose_oxygen_sensitivity_all_media.py` for each
- Total runtime: ~53 hours for complete sweep

**Core Script** (`glucose_oxygen_sensitivity_all_media.py`):

- Initially problematic: loaded Neo4j dataset on every call
- Fixed version: loads `unique_perturbations.json` directly
- Applies specific glucose/oxygen constraints per condition
- Saves results as `*_deletions_{media}_glc{glucose}_o2{oxygen}.parquet`

**Critical Bug and Fix:**

The original implementation loaded the entire Neo4j dataset (332K records) on each of 48 calls, causing severe delays. We identified this through job monitoring when worker initialization messages repeated endlessly. The fix involved removing lines 720-790, replacing the Neo4j loading with simple JSON file reading, matching the pattern from `targeted_fba_with_media_yeastgem.py`.

**Post-Processing** (`postprocess_glucose_oxygen_sweep.py`):

- Aggregates results from all 48 conditions
- Calculates interactions for each condition
- Matches to experimental data (loaded once)
- Enables correlation analysis across constraint space

**Visualization** (`plot_glucose_oxygen_sweep.py`):

- Generates heatmaps showing Pearson correlation vs glucose/oxygen levels
- Creates distribution plots showing fitness band shifts
- Produces correlation matrices for systematic analysis

## Testing and Validation Scripts

### Media Comparison Test

`ynb_media_test.slurm` provides rapid testing of media effects:

- Runs subset of data (100 singles, 200 doubles, 100 triples)
- Compares fitness distributions across three media
- Inline Python analysis calculates band statistics
- 1-hour runtime for quick validation

### Analysis Scripts

Several analysis scripts support the pipeline:

**`analyze_suthers_ypd.py`**:

- Loads Suthers et al. I. orientalis model
- Analyzes YPD media composition
- Validates exchange reaction mappings

**`inspect_suthers_media.py`**:

- Examines metabolite availability in different media
- Compares uptake rates across conditions
- Identifies missing components in Yeast9 model

**`test_cobra_gpr_logic.py`**:

- Validates gene-protein-reaction associations
- Tests complex vs isoenzyme logic
- Ensures correct knockout propagation

**`verify_media_differences.py`**:

- Compares growth rates across media conditions
- Validates that media setup creates expected differences
- Checks exchange reaction configurations

## Data Management and Organization

### Results Directory Structure

All outputs are organized in `experiments/007-kuzmin-tm/results/cobra-fba-growth/`:

```
unique_perturbations.json                    # Extracted gene combinations
singles_deletions_{media}.parquet           # FBA results by perturbation type
doubles_deletions_{media}.parquet
triples_deletions_{media}.parquet
matched_fba_experimental_{media}.parquet    # Matched with experimental data
*_glc{glucose}_o2{oxygen}.parquet          # Sensitivity analysis results
```

### File Format Decisions

We chose Parquet format for results storage due to:

- Efficient columnar compression (1M records → ~50MB files)
- Fast read/write with pandas
- Schema preservation
- Better than CSV for large numerical datasets

### Code Cleanup and Maintenance

Through the project, we accumulated multiple versions of scripts. We performed systematic cleanup:

**Removed duplicate plotting scripts:**

- `plot_media_comparison_results.py`
- `plot_fba_comparison_by_media_simple.py`
- `plot_fba_media_fast.py`
- `plot_media_comparison_optimized.py`
- `plot_fba_vs_experimental_by_media.py`

**Removed old versions:**

- `targeted_fba_with_media.py` (replaced by `targeted_fba_with_media_yeastgem.py`)
- `glucose_oxygen_postprocess.py` (replaced by `postprocess_glucose_oxygen_sweep.py`)
- `fba_media_postprocess.py` (replaced by `postprocess_fba_all_media.py`)

**Retained analysis utilities:**

- `tmi_tmf_correlation.py` - TMI/TMF correlation analysis
- `query.py` - Neo4j query utilities
- `plot_fba_distributions.py` - Distribution analysis tools
- `extract_config.py` - Configuration extraction utility

## Performance Optimization

### Multiprocessing Strategy

The shift from fork to spawn method was crucial for stability:

- Fork caused memory corruption with 128 workers
- Spawn ensures clean process initialization
- Model pickling reduces serialization overhead
- Chunksize=100 balances work distribution

### Computational Resources

Each full pipeline run utilizes:

- 128 CPUs for parallel FBA computations
- 500GB RAM allocation
- 24-hour runtime for 3-media comparison
- 365-day allocation for 48-condition sensitivity analysis

### Solver Configuration

GLPK solver parameters:

- 60-second timeout per optimization
- Timeouts treated as zero growth (biologically meaningful)
- No optimality tolerance setting (GLPK limitation)

## Key Findings and Outcomes

The complete pipeline successfully:

1. Processes ~1M gene combinations per condition
2. Calculates both digenic and trigenic interactions
3. Achieves improved correlation with YPD media (r ~0.2-0.3)
4. Demonstrates media-dependent fitness predictions
5. Reveals both constraint-driven and model-intrinsic limitations

The sensitivity analysis showed that while some fitness bands shift with nutrient availability (constraint-driven), others persist across all conditions (model-intrinsic), suggesting fundamental limitations in the metabolic model's ability to capture the full complexity of genetic interactions.

## Technical Stack

- **COBRA/COBRApy**: Constraint-based modeling framework
- **Yeast9 GEM**: Genome-scale metabolic model (~4,000 reactions, ~1,000 genes)
- **Python multiprocessing**: Parallel computation with spawn method
- **Neo4j**: Graph database for experimental data
- **Pandas/Parquet**: Data manipulation and storage
- **SLURM**: High-performance computing job scheduler
- **Matplotlib/Seaborn**: Visualization libraries

This comprehensive pipeline provides a systematic approach to comparing metabolic model predictions with experimental trigenic interaction data, revealing both the power and limitations of constraint-based modeling for predicting complex genetic interactions.
