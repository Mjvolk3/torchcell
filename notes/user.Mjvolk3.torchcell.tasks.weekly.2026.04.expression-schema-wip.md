---
id: 898b8n5swaj3isnqs49btca
title: expression-schema-wip
desc: ''
updated: 1768845735724
created: 1768845735724
---

## 2026.01.19 - Expression Schema Redesign for ML Applications

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
5. **`n_replicates: Dict[str, int]`** - Sample size (available but not cluttering YAML)
6. **`expression_se: Dict[str, float]`** - LINEAR scale SE (reference phenotype only)

**REMOVED fields (no backwards compatibility):**

- ~~`expression_technical_std`~~ - Not needed
- ~~`expression_log2_ratio_std`~~ - Replaced by SE and variance

**Key design rationale:**

- Only 2 PRIMARY fields in YAML (minimal complexity)
- SE on log2 scale (matches the transformed data used for ML)
- Variance (not technical_std) for proper meta-analysis
- n_replicates in serialized_data (SE already captures precision)

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
        serialized_data: str  # Full phenotype JSON (includes n_replicates, variance, raw expression)
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
   - `n_replicates: Dict[str, int] | None` (NEW - add this field)
   - `expression_se: Dict[str, float] | None` (NEW - for reference phenotype only)

3. **REMOVE deprecated fields:**
   - `expression_technical_std: Dict[str, float] | None` (delete)
   - `expression_log2_ratio_std: Dict[str, float] | None` (delete if exists)

4. **Update metadata:**
   - Ensure `label_statistic_name = "expression_log2_ratio_se"`

#### Phase 2: Update Dataset Loaders

**File:** `torchcell/datasets/scerevisiae/kemmeren2014.py`

**Critical fix - Order of operations:**

Current (WRONG):

```python
# Step 1: Extract LINEAR expression per replicate
# Step 2: Average LINEAR expression
# Step 3: Compute log2(mean_linear / ref)
```

Correct:

```python
# Step 1: Extract LINEAR expression per replicate
# Step 2: Compute log2(replicate_i / ref) for each replicate
# Step 3: Average log2 ratios
# Step 4: Compute SE = SD(log2_ratios) / sqrt(n)
# Step 5: Compute variance = SD(log2_ratios)²
```

**Changes needed:**

1. Modify `_extract_expression_from_gsm()` to return both LINEAR expression AND log2 ratios
2. Modify `_average_dye_swaps()` to:
   - Compute statistics on log2 ratios (not LINEAR)
   - Return: (mean_log2, se_log2, variance_log2, n_replicates)
3. Update `create_expression_experiment()` to populate new fields

**File:** `torchcell/datasets/scerevisiae/sameith2015.py`

**Already mostly correct** - just needs SE instead of SD:

1. Change `_calculate_mean_std()` to `_calculate_statistics_for_ml()`
2. Return SE instead of SD: `se = std / sqrt(n)`
3. Add variance calculation: `variance = std ** 2`
4. Add n_replicates tracking
5. Update phenotype instantiation with new fields

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

### Implementation Checklist

- [ ] Phase 1: Update `schema.py` with new fields
- [ ] Phase 2a: Fix Kemmeren2014 order of operations
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
