---
id: 898b8n5swaj3isnqs49btca
title: wip
desc: ''
updated: 1769097964822
created: 1768845735724
---

## 2026.01.19 - Expression Schema Redesign for ML Applications

### Terminology

**`n_samples` (not `n_replicates`):**

- We use `n_samples` to match standard statistical terminology
- Refers to the number of independent biological measurements
- Required field (not Optional) - SE cannot be interpreted without knowing n

### Problem Statement

Our current `MicroarrayExpressionPhenotype` schema stores standard deviation (SD) as the primary uncertainty measure, but this is problematic for ML applications:

1. **SD tracks technical/biological variability**, not precision of the mean estimate
2. **SD is not comparable** across experiments with different replicate counts (n)
3. **Standard Error (SE) is the proper measure** for ML because it quantifies uncertainty in the log2 fold change estimator
4. **BioCypher/Neo4j constraints** require us to choose PRIMARY fields carefully (can't store everything)

### Why SE is Better Than SD for ML

**Example comparison:**

| Gene  | n  | SD  | SE   | Interpretation                            |
|-------|----|-----|------|-------------------------------------------|
| GeneA | 2  | 0.5 | 0.35 | Low precision (wide confidence bounds)    |
| GeneB | 10 | 0.5 | 0.16 | High precision (narrow confidence bounds) |

- Both have same technical variability (SD = 0.5)
- But GeneB's mean estimate is more reliable (SE = 0.16)
- **For ML**: GeneB should have higher weight/confidence in training
- **SD alone cannot distinguish** between these two scenarios

**SE = SD / √n** accounts for sample size, making it comparable across experiments.

### Final Design Decision (SIMPLIFIED)

**PRIMARY fields (in BioCypher YAML - queryable in Neo4j):**

1. **`expression_log2_ratio: map`** - Mean log2 fold change (label for ML training)
2. **`expression_log2_ratio_se: map`** - Standard error on log2 scale (uncertainty for weighting/filtering)

**SECONDARY fields (in serialized_data only):**

3. **`expression: Dict[str, float]`** - Raw LINEAR values (QC/reproducibility)
4. **`expression_log2_ratio_variance: Dict[str, float]`** - Variance on log2 scale (meta-analysis)
5. **`n_samples: Dict[str, int]`** - Sample size (available but not cluttering YAML)
6. **`expression_se: Dict[str, float]`** - LINEAR scale SE (reference phenotype only)

**REMOVED fields (no backwards compatibility):**

- ~~`expression_technical_std`~~ - Not needed
- ~~`expression_log2_ratio_std`~~ - Replaced by SE and variance

**Key design rationale:**

- Only 2 PRIMARY fields in YAML (minimal complexity)
- SE on log2 scale (matches the transformed data used for ML)
- Variance (not technical_std) for proper meta-analysis
- n_samples in serialized_data (SE already captures precision)

### BioCypher YAML Design

**Add to `biocypher/config/torchcell_schema_config.yaml`:**

```yaml
microarray expression phenotype:
    is_a: phenotypic feature
    represented_as: node
    properties:
        graph_level: str
        label_name: str
        label_statistic_name: str
        expression_log2_ratio: map  # Dict[str, float] - PRIMARY label (log2 scale)
        expression_log2_ratio_se: map  # Dict[str, float] - PRIMARY uncertainty (log2 scale)
        serialized_data: str  # Full phenotype JSON (includes n_samples, variance, raw expression)
```

**If `map` type is not supported by BioCypher, fallback to:**

```yaml
        expression_log2_ratio: str  # JSON string
        expression_log2_ratio_se: str  # JSON string
```

And deserialize in application code as needed.

### Implementation Roadmap

#### Phase 1: Update Schema Definition

**File:** `torchcell/datamodels/schema.py`

**Changes to `MicroarrayExpressionPhenotype`:**

1. **Keep PRIMARY fields:**
   - `expression_log2_ratio: Dict[str, float]` (already exists)
   - `expression_log2_ratio_se: Dict[str, float]` (NEW - add this field)

2. **Keep SECONDARY fields:**
   - `expression: Dict[str, float]` (already exists)
   - `expression_log2_ratio_variance: Dict[str, float] | None` (NEW - add this field)
   - `n_samples: Dict[str, int] | None` (NEW - add this field)
   - `expression_se: Dict[str, float] | None` (NEW - for reference phenotype only)

3. **REMOVE deprecated fields:**
   - `expression_technical_std: Dict[str, float] | None` (delete)
   - `expression_log2_ratio_std: Dict[str, float] | None` (delete if exists)

4. **Update metadata:**
   - Ensure `label_statistic_name = "expression_log2_ratio_se"`

#### Phase 2: Update Dataset Loaders

**File:** `torchcell/datasets/scerevisiae/kemmeren2014.py`

**Critical fix - Order of operations for common reference pool design:**

Kemmeren2014 uses a **common reference pool** design where:

- Reference pool has many replicates (20-200 measurements) → very stable estimate
- Each mutant strain has 1-4 biological replicates
- Log2 ratios are computed per-sample against the mean reference pool

**Current implementation (WRONG):**

```python
# Step 1: Extract LINEAR expression per replicate
all_expressions[gene] = [rep1, rep2, rep3]

# Step 2: Average LINEAR expression FIRST
averaged_expression[gene] = np.mean(all_expressions[gene])

# Step 3: Compute ONE log2 ratio from averages
log2_ratio = log2(averaged_expression / mean_refpool)

# Result: n_samples = 1, SE = undefined
```

**Why this is mathematically wrong:**

```python
# Example with real numbers
mutant_reps = [100, 400]
refpool_mean = 200

# WRONG: log2(mean(x))
mean_mutant = mean([100, 400]) = 250
log2_ratio = log2(250/200) = 0.32

# CORRECT: mean(log2(x))
log2_rep1 = log2(100/200) = -1.0
log2_rep2 = log2(400/200) = 1.0
mean_log2 = mean([-1.0, 1.0]) = 0.0

# These are NOT equal! log2(mean(x)) ≠ mean(log2(x))
```

**Correct approach (mean refpool justified by large n):**

```python
# Step 1: Compute mean reference pool (20-200 reps → highly stable)
mean_refpool[gene] = np.mean(refpool_reps)
# Note: With 200 replicates, refpool SE is ~7x smaller than biological SE
# So we can treat mean_refpool as "ground truth" reference

# Step 2: Extract LINEAR expression per sample replicate
sample_expressions[gene] = [rep1, rep2, rep3]

# Step 3: Compute log2 ratio PER REPLICATE
log2_ratios[gene] = []
for rep_i in sample_expressions[gene]:
    log2_ratios[gene].append(log2(rep_i / mean_refpool[gene]))
# Result: log2_ratios[gene] = [log2_rep1, log2_rep2, log2_rep3]

# Step 4: Compute statistics on log2 scale
mean_log2_ratio[gene] = np.mean(log2_ratios[gene])
SD_log2 = np.std(log2_ratios[gene], ddof=1)
SE_log2[gene] = SD_log2 / np.sqrt(len(log2_ratios[gene]))
variance_log2[gene] = SD_log2 ** 2
n_samples[gene] = len(log2_ratios[gene])  # 2, 3, 4, etc.

# Result: Proper SE and n_samples for each gene!
```

**Changes needed:**

1. Modify `_average_dye_swaps()` to:
   - Keep replicate-level data (don't average yet)
   - Return: `(all_expressions_per_replicate, n_samples)`
2. Modify `create_expression_experiment()` to:
   - Compute log2 ratios PER REPLICATE
   - Average log2 ratios (not linear values)
   - Return: `(mean_log2, SE_log2, variance_log2, n_samples)`
3. Update phenotype instantiation with new fields

**File:** `torchcell/datasets/scerevisiae/sameith2015.py`

**Verify order of operations, then add SE/variance:**

1. **Audit existing code:** Check if log2 is computed before or after averaging
   - If after averaging → apply same fix as Kemmeren2014
   - If already correct → proceed to step 2
2. Change SD → SE calculation: `SE = SD / sqrt(n_samples)`
3. Add variance: `variance = SD ** 2`
4. Add `n_samples` field to track replicate count
5. Update all field names (`expression_technical_std` → `expression_se`)
6. Update phenotype instantiation with new schema fields

#### Phase 3: Update BioCypher YAML

**File:** `biocypher/config/torchcell_schema_config.yaml`

**Add new phenotype type:**

```yaml
# After line 148 (after calmorph phenotype)
microarray expression phenotype:
    is_a: phenotypic feature
    represented_as: node
    properties:
        graph_level: str
        label_name: str
        label_statistic_name: str
        expression_log2_ratio: map  # Try map first, fallback to str if unsupported
        expression_log2_ratio_se: map
        serialized_data: str
```

**Update phenotype member of relationship (line 171-184):**

```yaml
phenotype member of:
    is_a: participates in
    represented_as: edge
    input_label: phenotype
    source:
        [
            fitness phenotype,
            gene essentiality phenotype,
            synthetic lethality phenotype,
            synthetic rescue phenotype,
            gene interaction phenotype,
            calmorph phenotype,
            microarray expression phenotype,  # ADD THIS LINE
        ]
    target: [experiment, experiment reference]
```

#### Phase 4: Update CellAdapter

**File:** `torchcell/adapters/cell_adapter.py`

**Add to node_methods list (around line 88):**

```python
(
    "microarray expression phenotype (chunked)",
    self._microarray_expression_phenotype_node,
),
(
    "microarray expression phenotype reference",
    self._get_microarray_expression_phenotype_reference_nodes,
),
```

**Add new method (after other phenotype methods):**

```python
@data_chunker
def _microarray_expression_phenotype_node(
    self, data: dict, method_name: str
) -> BioCypherNode:
    phenotype = data["experiment"].phenotype
    phenotype_id = hashlib.sha256(
        json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
    ).hexdigest()

    graph_level = phenotype.graph_level
    label_name = phenotype.label_name
    label_statistic_name = phenotype.label_statistic_name
    expression_log2_ratio = phenotype.expression_log2_ratio
    expression_log2_ratio_se = phenotype.expression_log2_ratio_se

    properties = {
        "graph_level": graph_level,
        "label_name": label_name,
        "label_statistic_name": label_statistic_name,
        "expression_log2_ratio": expression_log2_ratio,
        "expression_log2_ratio_se": expression_log2_ratio_se,
        "serialized_data": json.dumps(phenotype.model_dump()),
    }

    return BioCypherNode(
        node_id=phenotype_id,
        preferred_id=f"phenotype_{phenotype_id}",
        node_label="microarray expression phenotype",
        properties=properties,
    )

def _get_microarray_expression_phenotype_reference_nodes(self):
    # Similar to _get_fitness_phenotype_reference_nodes
    # Extract reference phenotype from dataset reference experiments
    pass
```

#### Phase 5: Testing & Validation

**Tests to create/update:**

1. **Schema validation:**
   - Test that SE is NaN when n=1
   - Test that SE is finite when n>1
   - Test relationship: variance ≈ SE² × n

2. **Dataset loader tests:**
   - Verify log2 computed BEFORE averaging (not after)
   - Verify SE on log2 scale (not LINEAR)
   - Compare old vs new values for known genes

3. **BioCypher integration:**
   - Test that `map` type works in Neo4j
   - If not, implement fallback to `str` (JSON)
   - Query test: Filter by expression_log2_ratio_se

4. **End-to-end:**
   - Load Kemmeren2014 dataset
   - Write to Neo4j via BioCypher
   - Query for high-confidence upregulated genes
   - Verify results match expectations

#### Phase 6: Migration

**Breaking changes:**

- Removed `expression_technical_std` field
- Removed `expression_log2_ratio_std` field (replaced by SE and variance)

**Migration path:**

1. Update all code that references removed fields
2. Re-generate datasets with new schema
3. Re-build Neo4j database with new YAML
4. Update documentation

### Key Mathematical Insight: log2(mean) ≠ mean(log2)

**Why order of operations matters:**

Log2 transformation is **non-linear**, so applying it before vs after averaging gives different results:

```python
# Example: Gene has 2 replicates at 100 and 400 intensity
# Reference pool at 200 intensity

# WRONG: Average first, then log2
mean_linear = (100 + 400) / 2 = 250
log2_ratio = log2(250/200) = 0.32  # Appears upregulated

# CORRECT: Log2 first, then average
log2_rep1 = log2(100/200) = -1.0   # Rep 1 is downregulated
log2_rep2 = log2(400/200) = 1.0    # Rep 2 is upregulated
mean_log2 = (-1.0 + 1.0) / 2 = 0.0 # Average is no change!

# Difference: 0.32 vs 0.0 - completely different biological interpretation!
```

**Impact on downstream ML:**

- Wrong approach biases toward upregulation (geometric mean effect)
- Loses biological variability information (n_samples = 1, SE undefined)
- Cannot weight samples by confidence in training

**Current Kemmeren2014 status:**

- ❌ Averages linear values first (`averaged_expression = np.mean(replicates)`)
- ❌ Computes single log2 ratio from averaged values
- ❌ Results in n_samples = 1 for all genes
- ❌ SE is undefined/unusable for ML weighting

**Why using mean refpool is statistically valid:**

In common reference pool designs (Kemmeren, Sameith), the reference has MANY more replicates than samples:

- Kemmeren: 20-200 refpool replicates vs 1-4 sample replicates
- Reference pool SE is ~7-14x smaller than biological sample SE

```python
# Example: Reference pool with n=100
refpool_mean = 200
refpool_SD = 20
refpool_SE = 20 / sqrt(100) = 2.0  # Very precise!

# Sample with n=2
sample_SE = SD / sqrt(2) ≈ 15.0   # Much less precise

# Ratio: sample_SE / refpool_SE = 15/2 = 7.5x
# Conclusion: Refpool uncertainty is negligible compared to sample uncertainty
```

Therefore: **Treating mean refpool as fixed/known** introduces negligible error compared to biological variance, and enables proper per-sample SE estimation.

### Implementation Checklist

- [x] Phase 1: Update `schema.py` with new fields (committed: f8e6880)
- [x] Phase 2a: Fix Kemmeren2014 order of operations (committed: THIS COMMIT)
- [ ] Phase 2b: Update Sameith2015 to compute SE
- [ ] Phase 3: Add microarray expression phenotype to BioCypher YAML
- [ ] Phase 4: Add CellAdapter methods
- [ ] Phase 5: Write/update tests
- [ ] Phase 6: Migrate existing data
- [ ] Documentation: Update CLAUDE.md with new schema design
- [ ] Validation: Compare old vs new log2 values for sample genes

### Key Advantages

- ✅ SE is comparable across different replicate counts
- ✅ SE measures precision of mean estimate (relevant for ML)
- ✅ Enables proper uncertainty-weighted training
- ✅ Supports meta-analysis across datasets
- ✅ Compatible with BioCypher/Neo4j constraints
- ✅ Enables confidence-based data filtering
- ✅ Minimal YAML complexity (only 2 PRIMARY fields)
- ✅ Streamlined main path for querying

## 2026.01.21 - Phase 2a Complete: Fixed Kemmeren2014 Order of Operations

### What We Fixed

**Critical bug:** Kemmeren2014 was averaging LINEAR expression values, then computing a single log2 ratio. This resulted in:

- All genes having `n_samples = 1` (replicate info lost)
- `expression_log2_ratio_se = NaN` for all genes (SE undefined when n=1)
- Incorrect log2 values due to `log2(mean(x)) ≠ mean(log2(x))`

**Solution implemented:**

1. **Renamed method** for clarity:
   - `_average_dye_swaps()` → `_collect_replicate_expressions()`
   - Now returns `Dict[gene, List[float]]` (replicate-level data)

2. **Fixed order of operations** in `create_expression_experiment()`:

   ```python
   # For each gene, for each replicate:
   for rep_value in replicate_values:
       log2_ratio = -np.log2(rep_value / refpool_expression[gene])
       log2_ratios_per_replicate.append(log2_ratio)

   # Then compute statistics on log2 scale:
   mean_log2 = np.mean(log2_ratios_per_replicate)
   sd_log2 = np.std(log2_ratios_per_replicate, ddof=1)
   se_log2 = sd_log2 / np.sqrt(n)
   variance_log2 = sd_log2 ** 2
   ```

3. **Updated phenotype instantiation** with new schema fields:

   ```python
   phenotype = MicroarrayExpressionPhenotype(
       expression=mean_expression,  # Mean LINEAR (for QC)
       expression_log2_ratio=mean_log2_ratios,  # Mean log2 ratios
       expression_log2_ratio_se=log2_se,  # SE on log2 scale
       expression_log2_ratio_variance=log2_variance,  # Variance on log2 scale
       n_samples=n_samples_dict,  # Number of replicates per gene
   )
   ```

4. **Removed deprecated CV-scaled std logic** - replaced by proper replicate-based SE

### Results

After rebuilding the dataset:

- **1,484 gene deletions** processed
- **~7.1 million data points** with valid p-values (vs ~9.2 million total)
- **Variable n_samples**: 2, 3, or 4 replicates per gene (no longer all n=1!)
- **Non-NaN SE values** for genes with n>1

### Volcano Plot Validation

Created `experiments/012-sameith-kemmeren/scripts/kemmeren_volcano.py` to visualize the new data:

- Uses `expression_log2_ratio` (x-axis) and `expression_log2_ratio_se` (for p-value on y-axis)
- Computes t-statistics: `t = log2_fold_change / SE`
- Filters for significance: `|log2FC| > 1.0` and `p < 0.01`
- **3,598 upregulated** and **1,286 downregulated** genes identified

### Files Changed

**`torchcell/datasets/scerevisiae/kemmeren2014.py`:**

- Renamed `_average_dye_swaps` → `_collect_replicate_expressions` (lines 1892-1920, 749-781)
- Updated signature: removed `technical_std`, `refpool_std` parameters
- Fixed `create_expression_experiment()` to compute log2 before averaging (lines 1957-2010)
- Updated both sequential and parallel processing paths
- Removed CV-scaled std calculation logic

**`experiments/012-sameith-kemmeren/scripts/kemmeren_volcano.py`:** (NEW)

- Volcano plot using new SE-based p-values
- Demonstrates proper statistical significance testing

### Worktree Infrastructure Improvements

Also updated `scripts/setup-worktree.sh` to handle worktree-specific env vars:

- `.env` is now COPIED (not symlinked) with worktree-specific path overrides
- Added `.env.vscode` to `.gitignore`
- Keeps `DATA_ROOT` shared (expensive datasets)
- Makes `ASSET_IMAGES_DIR`, `EXPERIMENT_ROOT`, etc. worktree-specific

### Next Steps

Phase 2b: Update Sameith2015 dataset to use the same approach
