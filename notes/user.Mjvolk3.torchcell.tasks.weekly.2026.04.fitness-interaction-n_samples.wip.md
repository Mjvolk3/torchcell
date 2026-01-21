---
id: yyug5m463vtw663guqglqfk
title: wip
desc: ''
updated: 1769027455895
created: 1768945278400
---

## 2026.01.20 - Implementation Plan: Adding n_samples and fitness_se to Fitness Phenotypes

### Overview

Add sample size tracking (`n_samples`) and standard error (`fitness_se`) to fitness measurements across three datasets: Costanzo 2016, Kuzmin 2018, and Kuzmin 2020. This enables proper statistical comparison of fitness values after deduplication/aggregation and provides the foundation for validating gene interaction recomputation.

**Related:** [[nlp-data-enhancement|user.Mjvolk3.torchcell.tasks.weekly.2026.04.nlp-data-enhancement]] for methodology

### Motivation

Current issue: When `Neo4jCellDataset` applies `MeanExperimentDeduplicator` and `GenotypeAggregator`, we may combine fitness measurements differently than the original data used to compute gene interactions (ε = f_ij - f_i·f_j). Without tracking sample sizes, we cannot:

1. Compute standard errors to assess if Δε is statistically meaningful
2. Validate whether aggregated data yields consistent interaction estimates
3. Properly propagate uncertainty through derived calculations

### Schema Changes

**File:** `torchcell/datamodels/schema.py`

**FitnessPhenotype modifications:**

```python
class FitnessPhenotype(Phenotype, ModelStrict):
    graph_level: str = "global"
    label_name: str = "fitness"
    label_statistic_name: str = "fitness_se"  # CHANGED from "fitness_std"
    fitness: float = Field(description="wt_growth_rate/ko_growth_rate")
    fitness_std: float | None = Field(
        description="fitness standard deviation (raw data from publication)"
    )
    fitness_se: float | None = Field(  # NEW
        default=None,
        description="fitness standard error (primary uncertainty statistic)"
    )
    n_samples: int | None = Field(  # NEW
        default=None,
        description="""Number of replicate measurements of the fitness ratio.
        For experiment: n independent measurements of strain_of_interest/wt.
        For reference: n independent measurements of wt control.
        Note: numerator and denominator may have different sample sizes;
        this tracks the complete ratio measurement."""
    )

    @field_validator("fitness")
    def validate_fitness(cls, v):
        if math.isnan(v):
            raise ValueError("Fitness cannot be NaN")
        if v <= 0:
            return 0.0
        return v

    @model_validator(mode="after")
    def validate_label_fields(cls, values):
        if values.label_name not in cls.__annotations__:
            raise ValueError(
                f"label_name '{values.label_name}' must be a class attribute"
            )

        if (
            values.label_statistic_name is not None
            and values.label_statistic_name not in cls.__annotations__
        ):
            raise ValueError(
                f"""label_statistic_name '{values.label_statistic_name}'
                must be a class attribute"""
            )

        return values
```

**GeneInteractionPhenotype - NO CHANGES:**

- Keep `gene_interaction_p_value` as the statistic
- Do NOT add `n_samples` (see rationale below)
- Interaction validation via constituent fitness measurements (future work)

**Rationale for not tracking n_samples in GeneInteractionPhenotype:**

- Gene interactions are composite statistics (ε = f_ij - f_i·f_j for doubles)
- Each component fitness has its own sample size
- For doubles: Would need n_i, n_j, n_ij (3 values)
- For triples: Would need n_i, n_j, n_k, n_ij, n_ik, n_jk, n_ijk (7 values!)
- The p-value already accounts for error propagation across all components
- Validation approach: Query constituent fitness measurements and recompute

### Dataset Changes: Costanzo 2016

**File:** `torchcell/datasets/scerevisiae/costanzo2016.py`

Three dataset classes to update:

1. `SmfCostanzo2016Dataset` - Single mutant fitness
2. `DmfCostanzo2016Dataset` - Double mutant fitness
3. `DmiCostanzo2016Dataset` - Double mutant interaction

#### Step 1: Extract n_samples from Papers

**Papers to analyze:**

- `/Users/michaelvolk/Documents/projects/torchcell/papers/costanzoGlobalGeneticInteraction2016/costanzoGlobalGeneticInteraction2016.mmd`
- `/Users/michaelvolk/Documents/projects/torchcell/papers/costanzoGlobalGeneticInteraction2016/SI-costanzoGlobalGeneticInteraction2016.mmd`
- Corresponding PDF files for verification

**Extraction methodology:**
See [[nlp-data-enhancement.sop|user.Mjvolk3.torchcell.tasks.weekly.2026.04.nlp-data-enhancement.sop]]

**Search targets:**

- Keywords: "replicate", "independent", "measurement", "n=", "duplicate", "triplicate", "quadruplicate"
- Sections: Methods, Supplementary Methods, figure legends, table captions
- Specific focus: Sample sizes by experimental condition (temperature, perturbation type, control measurements)

**Expected patterns:**

```python
# Sample size determination from Costanzo et al. 2016
# Quote: "All fitness measurements represent the mean of at least
#         2 independent measurements"
# Source: costanzoGlobalGeneticInteraction2016.mmd, Lines XXX-YYY
# Verified: costanzoGlobalGeneticInteraction2016.pdf, Page N, Methods section
# Date extracted: 2026-01-20
N_SAMPLES_DEFAULT = 2

# For wild-type reference measurements at 26°C:
# Quote: "Wild-type control measurements were performed in quadruplicate
#         for each array plate"
# Source: SI-costanzoGlobalGeneticInteraction2016.mmd, Lines XXX-YYY
# Date extracted: 2026-01-20
N_SAMPLES_WT_26C = 4

# For wild-type reference measurements at 30°C:
# Quote: [to be determined from paper analysis]
# Source: [to be determined]
# Date extracted: 2026-01-20
N_SAMPLES_WT_30C = 4  # placeholder, needs verification

# For temperature-sensitive allele measurements:
# Quote: [to be determined]
# Source: [to be determined]
# Date extracted: 2026-01-20
N_SAMPLES_TSA = 2  # placeholder, needs verification
```

#### Step 2: Update SmfCostanzo2016Dataset

**Location:** `costanzo2016.py:~L47-400`

**Modifications needed:**

1. Add global n_samples constants at top of file (after paper analysis)
2. Update `_process_data_item()` method:

```python
def _process_data_item(self, row: pd.Series) -> dict:
    # ... existing perturbation/genotype/environment code ...

    # Phenotype with n_samples and SE
    smf_key = "Single mutant fitness"
    smf_std_key = "Single mutant fitness stddev"

    # Determine n_samples based on experimental conditions
    # (after paper analysis, implement proper logic)
    if row["Temperature"] == 26:
        n_samples = N_SAMPLES_DEFAULT  # or condition-specific
    elif row["Temperature"] == 30:
        n_samples = N_SAMPLES_DEFAULT
    else:
        n_samples = None

    # Compute SE
    fitness_std_val = row[smf_std_key]
    if fitness_std_val is not None and n_samples is not None:
        fitness_se_val = fitness_std_val / math.sqrt(n_samples)
    else:
        fitness_se_val = None

    phenotype = FitnessPhenotype(
        fitness=row[smf_key],
        fitness_std=fitness_std_val,
        fitness_se=fitness_se_val,  # NEW
        n_samples=n_samples  # NEW
    )

    # Reference phenotype
    if row["Temperature"] == 26:
        phenotype_reference_std = phenotype_reference_std_26
        n_samples_ref = N_SAMPLES_WT_26C
    elif row["Temperature"] == 30:
        phenotype_reference_std = phenotype_reference_std_30
        n_samples_ref = N_SAMPLES_WT_30C
    else:
        n_samples_ref = None

    if phenotype_reference_std is not None and n_samples_ref is not None:
        phenotype_reference_se = phenotype_reference_std / math.sqrt(n_samples_ref)
    else:
        phenotype_reference_se = None

    phenotype_reference = FitnessPhenotype(
        fitness=1.0,
        fitness_std=phenotype_reference_std,
        fitness_se=phenotype_reference_se,  # NEW
        n_samples=n_samples_ref  # NEW
    )

    # ... rest of method ...
```

#### Step 3: Update DmfCostanzo2016Dataset

**Location:** `costanzo2016.py:~L403-700`

Apply same pattern as SmfCostanzo2016Dataset:

- Use global n_samples constants
- Compute fitness_se in `_process_data_item()`
- Handle both experiment and reference phenotypes

#### Step 4: Update DmiCostanzo2016Dataset

**Location:** `costanzo2016.py:~L703-end`

**Important:** This dataset creates `GeneInteractionPhenotype`, NOT `FitnessPhenotype`

- NO changes to gene interaction schema
- NO n_samples tracking for interactions
- Interactions already have p-values (proper statistic)

### Dataset Changes: Kuzmin 2018

**File:** `torchcell/datasets/scerevisiae/kuzmin2018.py`

**Papers to analyze:**

- `/Users/michaelvolk/Documents/projects/torchcell/papers/kuzminSystematicAnalysisComplex2018/kuzminSystematicAnalysisComplex2018.mmd`
- `/Users/michaelvolk/Documents/projects/torchcell/papers/kuzminSystematicAnalysisComplex2018/SI-kuzminSystematicAnalysisComplex2018.mmd`

**Strategy:**

1. Apply same paper analysis methodology as Costanzo 2016
2. Extract condition-specific n_samples
3. Add global constants with citations
4. Update all fitness-related dataset classes
5. No changes to interaction datasets

### Dataset Changes: Kuzmin 2020

**File:** `torchcell/datasets/scerevisiae/kuzmin2020.py`

**Papers to analyze:**

- `/Users/michaelvolk/Documents/projects/torchcell/papers/kuzminExploringWholegenomeDuplicate2020/kuzminExploringWholegenomeDuplicate2020.mmd`
- `/Users/michaelvolk/Documents/projects/torchcell/papers/kuzminExploringWholegenomeDuplicate2020/SI-kuzminExploringWholegenomeDuplicate2020.mmd`

**Strategy:**
Same as Kuzmin 2018

### Testing Strategy

**1. Schema validation:**

```python
# Test FitnessPhenotype with new fields
phenotype = FitnessPhenotype(
    fitness=0.8,
    fitness_std=0.1,
    fitness_se=0.05,
    n_samples=4
)
assert phenotype.label_statistic_name == "fitness_se"
```

**2. SE computation verification:**

```python
# Verify SE = SD / sqrt(n)
import math
fitness_std = 0.1
n_samples = 4
expected_se = fitness_std / math.sqrt(n_samples)  # 0.05
```

**3. Dataset consistency checks:**

```python
# Load dataset and verify fields present
dataset = SmfCostanzo2016Dataset(root="...", subset_n=100)
item = dataset[0]
assert "n_samples" in item["experiment"]["phenotype"]
assert "fitness_se" in item["experiment"]["phenotype"]
assert item["experiment"]["phenotype"]["label_statistic_name"] == "fitness_se"
```

**4. Edge case testing:**

- Missing fitness_std → fitness_se should be None
- Missing n_samples → fitness_se should be None
- Zero n_samples → handle gracefully

### Implementation Order

**Phase 1: Schema + Costanzo 2016**

1. ✓ Read and analyze Costanzo papers for n_samples
2. ✓ Update `FitnessPhenotype` in `schema.py`
3. ✓ Add global n_samples constants to `costanzo2016.py`
4. ✓ Update `SmfCostanzo2016Dataset`
5. ✓ Update `DmfCostanzo2016Dataset`
6. ✓ Test schema and dataset loading
7. ✓ Verify SE computations

**Phase 2: Kuzmin 2018**

1. ✓ Read and analyze Kuzmin 2018 papers
2. ✓ Add global n_samples constants to `kuzmin2018.py`
3. ✓ Update all fitness dataset classes
4. ✓ Test and verify

**Phase 3: Kuzmin 2020**

1. ✓ Read and analyze Kuzmin 2020 papers
2. ✓ Add global n_samples constants to `kuzmin2020.py`
3. ✓ Update all fitness dataset classes
4. ✓ Test and verify

**Phase 4: Validation**

1. ✓ Test deduplication with new fields
2. ✓ Verify Neo4jCellDataset compatibility
3. ✓ Check that old datasets without n_samples still load (backward compatibility)
4. ✓ Document changes in dataset docstrings

### Future Work (Out of Scope)

1. **MeanExperimentDeduplicator updates:**
   - Properly combine n_samples when averaging duplicates (sum)
   - Propagate SE correctly (not simple averaging)

2. **Gene interaction validation:**
   - Query constituent fitness measurements for an interaction
   - Recompute ε with proper error propagation
   - Compare to reported values
   - Assess impact of deduplication/aggregation

3. **Automated paper extraction:**
   - LLM-based extraction with reproducible citations
   - Validation against multiple runs
   - Schema-aware extraction hints
   - See [[nlp-data-enhancement.sop|user.Mjvolk3.torchcell.tasks.weekly.2026.04.nlp-data-enhancement.sop]]

### Critical Path Items

**Must verify from papers:**

- Default n_samples for single/double mutant measurements
- Wild-type reference n_samples (may differ by temperature)
- Whether n_samples varies by perturbation type (KanMX vs NatMX vs TSA vs DAmP)
- Whether n_samples varies by screen type (different array plates)
- Edge cases: essential genes, synthetic lethal combinations

**Code patterns to follow:**

```python
# Good: Explicit condition-based assignment
if temperature == 26 and perturbation_type == "TSA":
    n_samples = N_SAMPLES_TSA_26C
elif temperature == 30:
    n_samples = N_SAMPLES_DEFAULT_30C
else:
    n_samples = None  # Unknown condition

# Good: Safe SE computation
if fitness_std is not None and n_samples is not None and n_samples > 0:
    fitness_se = fitness_std / math.sqrt(n_samples)
else:
    fitness_se = None

# Bad: Assuming a default without justification
n_samples = 2  # Don't do this - must have citation
```

### Questions to Resolve During Paper Analysis

1. Are all measurements done with the same number of replicates?
2. Do reference (WT) measurements have higher replication?
3. Are there condition-specific differences (temperature, media)?
4. How are technical vs biological replicates defined?
5. Are array measurements (columns) treated differently from query measurements (rows)?
6. What happens for missing data or failed measurements?

### Success Criteria

- [ ] FitnessPhenotype schema updated and validated
- [ ] All Costanzo 2016 fitness datasets produce valid n_samples and fitness_se
- [ ] All Kuzmin 2018 fitness datasets produce valid n_samples and fitness_se
- [ ] All Kuzmin 2020 fitness datasets produce valid n_samples and fitness_se
- [ ] Global constants have proper citations traceable to paper line numbers
- [ ] SE values correctly computed as SD/sqrt(n)
- [ ] Reference measurements have appropriate n_samples
- [ ] Backward compatibility maintained (datasets without n_samples still load)
- [ ] Documentation updated in docstrings and comments
