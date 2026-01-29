---
id: 73lqcuy3xojaojkcjhavpfd
title: wip
desc: ''
updated: 1769636887698
created: 1769636875371
---
## 2026.01.27 - Primary Statistic Selection Guidelines

### Core Principle: Purpose-Driven Statistics

The choice of `label_statistic_name` depends on **what you're measuring uncertainty about**:

1. **Uncertainty about the TRUE MEAN** → Use **SE** (Standard Error)
2. **Variability in the DATA** → Use **SD** (Standard Deviation)
3. **Scale-independent comparison** → Use **CV** (Coefficient of Variation)

### Decision Tree for Primary Statistics

#### For Fitness Phenotypes (Costanzo2016)

**Current implementation**: `fitness_std`

**Analysis**:

- Raw data: fitness ratios from n=4 technical replicates (double mutants) or n=68 biological replicates (single mutants)
- Purpose: Compare fitness across different genetic backgrounds
- Question: "How precise is our estimate of the mean fitness?"

**Recommendation**:

- **Primary statistic**: `fitness_se` (SE = SD/√n)
- **Keep for QC**: `fitness_std` (optional field)
- **Rationale**:
  - SE measures precision of the mean estimate
  - Critical for model training: uncertainties should reflect confidence in the mean, not raw data spread
  - Enables proper deduplication: experiments with more replicates get appropriately weighted
  - For meta-analysis/aggregation: variance (SD²) can be reconstructed from SE and n

**Code pattern**:

```python
fitness_se = fitness_std / math.sqrt(n_samples)

phenotype = FitnessPhenotype(
    fitness=row[smf_key],
    fitness_std=fitness_std_val,  # Keep for QC/reproducibility
    fitness_se=fitness_se_val,    # PRIMARY statistic
    n_samples=n_samples,
)
```

#### For Expression Phenotypes (Kemmeren2014)

**Current implementation**: `expression_log2_ratio_se`

**Analysis**:

- Raw data: Multiple technical replicates (dye-swaps) per gene deletion
- Purpose: Identify differentially expressed genes across deletions
- Question: "How confident are we in the log2 fold change estimate?"

**Recommendation**:

- **Primary statistic**: `expression_log2_ratio_se` ✓ (already correct!)
- **Keep for meta-analysis**: `expression_log2_ratio_variance` (optional)
- **Rationale**:
  - Statistics computed **on log2 scale** (critical for expression data!)
  - SE correctly measures uncertainty in the mean log2 ratio
  - CONVENTION: log2(sample/reference) where positive = upregulated
  - Variance useful for combining studies with different sample sizes

**Critical insight from your implementation**:

```python
# CORRECT: Compute log2 per replicate, THEN average
log2_ratios_per_replicate = [
    np.log2(rep_value / refpool_expression[gene])
    for rep_value in replicate_values
]
mean_log2 = np.mean(log2_ratios_per_replicate)
se_log2 = np.std(log2_ratios_per_replicate, ddof=1) / np.sqrt(n)

# WRONG: Average first, then log2
# mean_log2 = np.log2(np.mean(replicate_values) / refpool)  # ❌
```

**Why SE > SD for expression**:

- Comparing 1000s of genes across deletions
- Need to know: "Which genes have RELIABLY different expression?"
- SE captures: "How much would the mean change if we repeated the experiment?"
- SD captures: "How noisy are individual measurements?" (less relevant for gene ranking)

#### For Morphology Phenotypes (Ohya2005)

**Current implementation**: `calmorph_coefficient_of_variation`

**Analysis**:

- Raw data: 281 morphological parameters, each with CV pre-computed by authors
- Purpose: Compare morphology across different cell types/conditions
- Challenge: Parameters on vastly different scales (μm, ratios, angles)

**Recommendation**:

- **Primary statistic**: `calmorph_coefficient_of_variation` ✓ (CV = SD/mean)
- **Rationale**:
  - CV is scale-independent: enables cross-parameter comparison
  - Ohya et al. explicitly chose CV for this reason
  - For morphology: relative variability often more biologically meaningful than absolute
  - Example: 10% variation in cell size more interpretable than "±2.3 μm"

**When CV is appropriate**:

- ✓ Comparing features with different units
- ✓ Mean is far from zero (μ >> 0)
- ✓ Measurements are ratio-scale (not interval-scale)

**When CV is problematic**:

- ❌ Mean near zero (CV → ∞)
- ❌ Negative values possible (e.g., log2 fold changes)
- ❌ Temperature, pH (interval scales where zero is arbitrary)

### Storage Requirements by Purpose

Based on your BioCypher + Neo4j constraints:

#### Minimal Storage (Model Training Only)

```python
# Store only what's needed for ML
phenotype = FitnessPhenotype(
    fitness=mean_fitness,
    fitness_se=se_fitness,     # PRIMARY statistic in Neo4j
    n_samples=n_samples,       # Required for interpreting SE
)
```

#### Full Storage (Reproducibility + Meta-analysis)

```python
# Enable reconstruction of all statistics
phenotype = FitnessPhenotype(
    fitness=mean_fitness,
    fitness_se=se_fitness,           # PRIMARY - in Neo4j
    fitness_std=std_fitness,         # OPTIONAL - can compute from SE×√n
    fitness_variance=var_fitness,    # OPTIONAL - for meta-analysis
    n_samples=n_samples,             # REQUIRED
)
```

**Trade-off**:

- SE alone is sufficient if you have n_samples (can compute SD = SE × √n)
- Storing both SE and SD is redundant BUT aids human interpretation
- Variance useful for inverse-variance weighting in meta-analysis

### Framework-Wide Recommendations

#### 1. Primary Statistic Hierarchy

**Default order of preference**:

1. **SE** - for comparing mean estimates (most common use case)
2. **CI** - when you want to emphasize uncertainty range
3. **CV** - for scale-independent metrics (morphology, ratios)
4. **SD** - only when raw data variability is the primary question

#### 2. Required Fields for All Phenotypes

```python
class Phenotype(ModelStrict):
    # ... existing fields ...
    n_samples: int | None = Field(
        description="Number of independent measurements"
    )
```

**Why n_samples is critical**:

- SE without n is uninterpretable (is SE=0.1 good or bad?)
- Enables reconstruction of SD from SE
- Required for proper deduplication weighting
- Needed for meta-analysis (inverse-variance weighting)

#### 3. Log-Scale Data Special Handling

For any ratio/fold-change phenotype:

```python
# ✓ CORRECT pattern
log2_values = [np.log2(x) for x in raw_values]
mean_log2 = np.mean(log2_values)
se_log2 = np.std(log2_values, ddof=1) / np.sqrt(len(log2_values))

# ❌ WRONG pattern
mean_raw = np.mean(raw_values)
se_raw = np.std(raw_values) / np.sqrt(len(raw_values))
log2_mean = np.log2(mean_raw)  # This is NOT the mean of log2 values!
```

**Rationale**:

- log2 is non-linear: mean(log2(x)) ≠ log2(mean(x))
- Errors are more symmetric on log scale
- Biological interpretation: log2 fold changes are additive

#### 4. Deduplication Strategy

When combining experiments with different sample sizes:

```python
# Inverse-variance weighted mean
weights = [1 / (se**2) for se in standard_errors]
weighted_mean = np.average(values, weights=weights)

# Combined SE (for independent measurements)
combined_se = 1 / np.sqrt(sum(weights))
```

**Why SE-based weighting is superior**:

- Automatically accounts for both sample size AND variability
- Experiment with n=100 but high SE contributes less than n=10 with low SE
- Statistically optimal under normality assumption

### Current Dataset Analysis

#### Costanzo2016 (Fitness)

**Current**:

```python
phenotype = FitnessPhenotype(
    fitness=row[smf_key],
    fitness_std=row[smf_std_key],
    n_samples=N_SAMPLES_QUERY_SMF_TOTAL,  # 68 for query strains
)
```

**Recommended change**:

```python
fitness_se = fitness_std / math.sqrt(n_samples)

phenotype = FitnessPhenotype(
    fitness=row[smf_key],
    fitness_std=fitness_std,      # Keep for transparency
    fitness_se=fitness_se,         # Add as PRIMARY
    n_samples=n_samples,
)

# Update schema
class FitnessPhenotype(Phenotype):
    label_statistic_name: str = "fitness_se"  # Changed from "fitness_std"
```

#### Kemmeren2014 (Expression)

**Current**: ✓ Already optimal!

```python
phenotype = MicroarrayExpressionPhenotype(
    expression_log2_ratio=mean_log2_ratios,
    expression_log2_ratio_se=log2_se,           # PRIMARY ✓
    expression_log2_ratio_variance=log2_var,    # Optional for meta-analysis ✓
    n_samples=n_samples_dict,                   # Required ✓
)
```

**No changes needed** - this follows best practices:

- Statistics on log2 scale ✓
- SE as primary statistic ✓
- Variance stored for aggregation ✓
- n_samples tracked per gene ✓

#### Ohya2005 (Morphology)

**Current**: ✓ Appropriate for morphology

```python
phenotype = CalMorphPhenotype(
    calmorph=base_measurements,
    calmorph_coefficient_of_variation=cv_measurements,  # PRIMARY ✓
)
```

**Consider adding** (if raw data available):

```python
phenotype = CalMorphPhenotype(
    calmorph=base_measurements,
    calmorph_coefficient_of_variation=cv_measurements,  # PRIMARY
    calmorph_n_samples=n_samples_dict,                  # If available
)
```

### Summary Table

| Dataset | Phenotype Type | Current Stat | Recommended Primary | Keep Secondary | Rationale |
|---------|---------------|--------------|-------------------|----------------|-----------|
| Costanzo2016 | Fitness | `fitness_std` | `fitness_se` | `fitness_std`, `n_samples` | Precision of mean estimate matters for comparison |
| Kemmeren2014 | Expression | `expression_log2_ratio_se` | ✓ Keep as-is | `expression_log2_ratio_variance`, `n_samples` | Already optimal - SE on log2 scale |
| Ohya2005 | Morphology | `calmorph_cv` | ✓ Keep as-is | `n_samples` (if available) | Scale-independent comparison needed |

### Implementation Priority

**High Priority** (Costanzo2016):

1. Add `fitness_se` field to `FitnessPhenotype`
2. Change `label_statistic_name = "fitness_se"`
3. Keep `fitness_std` for backward compatibility/QC
4. Update BioCypher YAML to expose `fitness_se`

**Medium Priority** (All datasets):

1. Ensure `n_samples` is always populated
2. Add validation: `n_samples` must be > 0 when SE is provided
3. Document SE vs SD distinction in schema docstrings

**Low Priority** (Future work):

1. Add `variance` fields for meta-analysis
2. Implement inverse-variance weighting in deduplication
3. Add CI fields for human-interpretable uncertainty ranges

### References

- Your PDF: SE measures precision of sample mean, SD measures spread of observations
- SE = SD/√n: Larger samples → more precise mean estimate
- CV = SD/mean: Removes scale dependence, useful for morphology
- Log-scale statistics: Essential for multiplicative processes (expression, fitness ratios)
