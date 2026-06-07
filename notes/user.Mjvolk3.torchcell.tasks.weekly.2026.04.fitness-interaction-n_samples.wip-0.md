---
id: 9l1ptr27ouvjguvewd3lf80
title: wip-0
desc: ''
updated: 1769656449401
created: 1769636388693
---

## 2026.01.28 - Simplified n_replicates Approach

### Primary Goal

**Record the number of measurements used to compute the reported mean fitness value**, regardless of the statistical model used for variance estimation.

### What We Need to Store

```python
class FitnessPhenotype(Phenotype, ModelStrict):
    fitness: float  # Mean fitness value
    fitness_std: float | None  # Reported standard deviation
    fitness_se: float | None  # Standard error (computed or reported)
    n_replicates: int | None  # Number of measurements used for mean
```

**Key principle:** `n_replicates` = **whatever n was used to compute the mean**, even if variance was computed differently (e.g., bootstrap).

### For Costanzo2016

Based on analysis in [[costanzo2016.noise-computation|torchcell.datasets.scerevisiae.costanzo2016.noise-computation]]:

| Measurement | n_replicates | What it represents                          | Variance method          |
|-------------|--------------|---------------------------------------------|--------------------------|
| Query SMF   | 17           | Number of control screens used in bootstrap | Bootstrap across screens |
| Array SMF   | 350          | Number of control screens used in bootstrap | Bootstrap across screens |
| DMF         | 4            | Number of colony measurements               | Raw sample SD            |

**Implementation:**

```python
# In costanzo2016.py constants
N_REPLICATES_QUERY_SMF = 17    # Screens used in bootstrap
N_REPLICATES_ARRAY_SMF = 350   # Screens used in bootstrap
N_REPLICATES_DMF = 4           # Colony measurements
```

### Why This is Sufficient

1. **Users can compute approximate SE:**

   ```python
   SE_approx = fitness_std / sqrt(n_replicates)
   ```

2. **Users understand measurement quality:** Higher n = more precise

3. **Deduplication/aggregation can combine n values:** When averaging across replicates, sum the n values

4. **Error propagation works** (approximately):

   ```python
   Var(ε) = Var(f_ij) + f_j² Var(f_i) + f_i² Var(f_j)
   where Var(f) ≈ (fitness_std / sqrt(n_replicates))²
   ```

### Important Caveats to Document

1. **Bootstrap vs parametric SD:**
   - For Costanzo Query/Array SMF: `fitness_std` is from bootstrap, already SE-like
   - For Costanzo DMF: `fitness_std` is raw sample SD
   - Users computing `SE = SD / sqrt(n)` may over-correct for bootstrap data

2. **Cannot reproduce exact p-values:**
   - Costanzo uses complex model (bootstrap + batch correction + log-normal)
   - Simple error propagation gives approximate uncertainty
   - See [[sga-statistics-notepad|scratch.2026.01.27.161613-sga-statistics-notepad]] for attempted reconstruction
[[scratch.2026.01.27.161613-sga-statistics-notepad]]
3. **Pseudo-replication:**
   - DMF with n=4 represents 4 colonies from ONE genetic cross
   - Not 4 independent biological replicates
   - True biological replication = 1 (one screen performed)

### Deferred for Future

**Detailed replicate tracking** (biological vs technical, replicate type enum):

- Too complex for current need
- Different papers report replication differently
- Can add later if needed (e.g., `replicate_metadata` field)

For now: **Keep it simple, just track n used for mean**

### Next Steps

1. ✅ Document variance computation in [[costanzo2016.noise-computation|torchcell.datasets.scerevisiae.costanzo2016.noise-computation]]
2. ⏳ Update schema with simple `n_replicates` field
3. ⏳ Update Costanzo2016 constants with n values
4. ⏳ Update Costanzo2016 dataset classes to populate n_replicates
5. ⏳ Test and verify SE calculations
6. ⏳ Repeat for Kuzmin2018 and Kuzmin2020

### Final Decision: Option 1 - SD as Primary Statistic

**Core Issue:**

We were trying to standardize on SE across datasets, but:

- Costanzo2016 SMF reports bootstrap SD (already SE-like)
- Costanzo2016 DMF reports raw sample SD (needs √n conversion)
- This mixing creates confusion and forces awkward transformations

**What Actually Matters for ML:**

- **Primary label (fitness):** The mean value - what models train on
- **Uncertainty:** Secondary - useful for weighting/confidence
- **n_replicates:** Helps users understand measurement quality

**Implementation:**

```python
class FitnessPhenotype(Phenotype):
    label_name: str = "fitness"
    label_statistic_name: str = "fitness_std"  # Changed from fitness_se

    fitness: float
    fitness_std: float | None  # Report what's actually in source data
    n_replicates: int | None   # Number used for mean computation
```

**For Costanzo2016:**

| Measurement | fitness_std | n_replicates | What it represents |
|-------------|-------------|--------------|---------------------|
| Query SMF   | 0.0061      | 17           | Bootstrap SD across 17 screens |
| Array SMF   | ~similar    | 350          | Bootstrap SD across 350 screens |
| DMF         | 0.0079      | 4            | Sample SD from 4 colonies |

**Advantages:**

- Reports what's actually in the data (honest, no transformation)
- No ambiguity about computation method
- Simple and straightforward
- Users can approximate SE: `SE ≈ std / √n` (with caveats for bootstrap)

**Trade-offs:**

- Different datasets may report different statistics (SD vs SE vs bootstrap SD)
- Not perfectly "standardized" (but true standardization may be impossible given different variance methods)
- Users need to understand bootstrap vs parametric SD distinction (documented in [[torchcell.datasets.scerevisiae.costanzo2016.noise-computation]])

### Related Notes

- [[fitness-interaction-n_samples.wip|user.Mjvolk3.torchcell.tasks.weekly.2026.04.fitness-interaction-n_samples.wip]] - Original (more complex) plan
- [[costanzo2016.noise-computation|torchcell.datasets.scerevisiae.costanzo2016.noise-computation]] - Technical details on variance computation
- [[sga-statistics-notepad|scratch.2026.01.27.161613-sga-statistics-notepad]] - P-value reconstruction attempts
