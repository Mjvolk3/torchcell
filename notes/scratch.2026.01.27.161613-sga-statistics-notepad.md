---
id: 1nmq9elay4ia4s1ekvmh3o0
title: 161613 Sga Statistics Notepad
desc: ''
updated: 1769637768807
created: 1769552181165
---

## 2026.01.27 - Taking Notes to Get SGA DataModel Correct

- I don't get definition of pseudo replication.
- explain segregates and stochastic segregation
- we don't need to compute n_replicates_total that can be done later at any time...
- for se for this paper i don't think we have any bio replicates since they come from same ascus right?
- I do like option 2.
- I need to verify what is the mathematical formula use for say computing the p value of some interaction?

If do a reconstruction of data for an instance...

### Double Mutant Fitness

```python
dataset.df.iloc[10]
Query Strain ID                             YOR265W_sn1433
Query allele name                                     rbl2
Array Strain ID                             YDL124W_dma699
Array allele name                                  ydl124w
Arraytype/Temp                                       DMA30
Genetic interaction score (ε)                      -0.0285
P-value                                            0.00672
Query single mutant fitness (SMF)                   0.9835
Array SMF                                           1.0128
Double mutant fitness                               0.9676
Double mutant fitness standard deviation            0.0079
Query Systematic Name                              YOR265W
Array Systematic Name                              YDL124W
Temperature                                             30
query_perturbation_type                     NatMX_deletion
array_perturbation_type                     KanMX_deletion
Name: 10, dtype: object
```

### Single Mutant Fitness

```python
dataset.df.iloc[19416]
Strain ID                       YOR265W_sn1433
Systematic gene name                   YOR265W
Allele/Gene name                          rbl2
Single mutant fitness                   0.9835
Single mutant fitness stddev            0.0061
perturbation_type               NatMX_deletion
Temperature                                 30
Name: 19416, dtype: object
```

```python
dataset.df.iloc[11670]
Strain ID                       YDL124W_dma699
Systematic gene name                   YDL124W
Allele/Gene name                       ydl124w
Single mutant fitness                   1.0128
Single mutant fitness stddev            0.0795
perturbation_type               KanMX_deletion
Temperature                                 30
Name: 11670, dtype: object
```

### Double Mutant Interaction

```python
Query Strain ID                             YOR265W_sn1433
Query allele name                                     rbl2
Array Strain ID                             YDL124W_dma699
Array allele name                                  ydl124w
Arraytype/Temp                                       DMA30
Genetic interaction score (ε)                      -0.0285
P-value                                            0.00672
Query single mutant fitness (SMF)                   0.9835
Array SMF                                           1.0128
Double mutant fitness                               0.9676
Double mutant fitness standard deviation            0.0079
Query Systematic Name                              YOR265W
Array Systematic Name                              YDL124W
Temperature                                             30
query_perturbation_type                     NatMX_deletion
array_perturbation_type                     KanMX_deletion
Name: 19494117, dtype: object
```

### Extracting Data

$i$ - YOR265W
$j$ - YDL124W

$f_i = 0.9835$
$std(f_i) = 0.0061$
$f_j = 1.0128$
$std(f_j) = 0.0795$
$f_{ij} = 0.9676$
$std(f_{ij}) = 0.0079$
$\epsilon_{ij} = -0.0285$
$\text{p-value+{\epsilon_{ij}} = 0.00672$

## Looking at Different Replicates For Reproduction of P-Value

```python
# Try different n values to see what would match the reported p-value
print("\nSearching for n that matches reported p-value...")

for test_n in [4, 10, 17, 50, 68, 100, 200, 350, 1400]:
    var_i_test = (sd_i ** 2) / test_n
    var_j_test = (sd_j ** 2) / test_n
    var_ij_test = (sd_ij ** 2) / test_n

    var_epsilon_test = var_ij_test + (f_j ** 2) * var_i_test + (f_i ** 2) * var_j_test
    se_epsilon_test = np.sqrt(var_epsilon_test)
    z_test = abs(epsilon) / se_epsilon_test
    p_value_test = 2 * (1 - norm.cdf(z_test))

    print(f"n={test_n:4d}: p-value={p_value_test:.6f} (z={z_test:.2f})")

Output:
Searching for n that matches reported p-value...
n=   4: p-value=0.469582 (z=0.72)
n=  10: p-value=0.226990 (z=1.21)
n=  17: p-value=0.131673 (z=1.51)
n=  50: p-value=0.042073 (z=2.03)
n=  68: p-value=0.030079 (z=2.17)
n= 100: p-value=0.019599 (z=2.33)
n= 200: p-value=0.009456 (z=2.59)  ← Close to 0.00672!
n= 350: p-value=0.006332 (z=2.73)  ← Very close!
n=1400: p-value=0.003063 (z=2.96)
```

### Scenario 1: Using biological replicates for SMF, technical for DMF

```python
# Query = YOR265W (query strain, NatMX deletion)
# Array = YDL124W (array strain, KanMX deletion)

# Replicate counts from Costanzo2016
n_query = 17    # Query SMF: 17 biological screens
n_array = 350   # Array SMF: 350 biological screens
n_dmf = 4       # DMF: 4 technical replicates (colonies)

# Given data
f_i = 0.9835
sd_i = 0.0061
f_j = 1.0128
sd_j = 0.0795
f_ij = 0.9676
sd_ij = 0.0079
epsilon = -0.0285
reported_p_value = 0.00672

# Convert SD to Variance
var_i = (sd_i ** 2) / n_query
var_j = (sd_j ** 2) / n_array
var_ij = (sd_ij ** 2) / n_dmf

print(f"Var(f_i) with n={n_query} = {var_i:.10f}")
print(f"Var(f_j) with n={n_array} = {var_j:.10f}")
print(f"Var(f_ij) with n={n_dmf} = {var_ij:.10f}")

# Error propagation
var_epsilon = var_ij + (f_j ** 2) * var_i + (f_i ** 2) * var_j
se_epsilon = np.sqrt(var_epsilon)

print(f"\nVar(ε) = {var_epsilon:.10f}")
print(f"SE(ε) = {se_epsilon:.6f}")

# Z-score and p-value
z = abs(epsilon) / se_epsilon
p_value = 2 * (1 - norm.cdf(z))

print(f"\nZ-score = {z:.4f}")
print(f"Computed p-value = {p_value:.6f}")
print(f"Reported p-value = {reported_p_value:.6f}")
print(f"Difference: {abs(p_value - reported_p_value):.6f}")
print(f"Ratio (computed/reported): {p_value / reported_p_value:.2f}x")

Output:
Var(f_i) with n=17 = 0.0000021885
Var(f_j) with n=350 = 0.0000180729
Var(f_ij) with n=4 = 0.0000156025

Var(ε) = 0.0000374933

SE(ε) = 0.006123

Z-score = 4.6542

Computed p-value = 0.000003
Reported p-value = 0.006720

Difference: 0.006717
Ratio (computed/reported): 0.0005 x
```

Result: p-value way too small (1000x too small)!

Scenario 2: Using total measurements (screens × colonies)

```python
# Use total measurements: biological screens × technical colonies
n_query_total = 68     # 17 screens × 4 colonies
n_array_total = 1400   # 350 screens × 4 colonies
n_dmf = 4              # 4 technical replicates

# Convert SD to Variance
var_i = (sd_i ** 2) / n_query_total
var_j = (sd_j ** 2) / n_array_total
var_ij = (sd_ij ** 2) / n_dmf

print(f"Var(f_i) with n={n_query_total} = {var_i:.10f}")
print(f"Var(f_j) with n={n_array_total} = {var_j:.10f}")
print(f"Var(f_ij) with n={n_dmf} = {var_ij:.10f}")

# Error propagation
var_epsilon = var_ij + (f_j ** 2) * var_i + (f_i ** 2) * var_j
se_epsilon = np.sqrt(var_epsilon)

print(f"\nVar(ε) = {var_epsilon:.10f}")
print(f"SE(ε) = {se_epsilon:.6f}")

# Z-score and p-value
z = abs(epsilon) / se_epsilon
p_value = 2 * (1 - norm.cdf(z))

print(f"\nZ-score = {z:.4f}")
print(f"Computed p-value = {p_value:.6f}")
print(f"Reported p-value = {reported_p_value:.6f}")
print(f"Difference: {abs(p_value - reported_p_value):.6f}")
print(f"Ratio (computed/reported): {p_value / reported_p_value:.2f}x")

Output:
Var(f_i) with n=68 = 0.0000005471
Var(f_j) with n=1400 = 0.0000045182
Var(f_ij) with n=4 = 0.0000156025

Var(ε) = 0.0000207159

SE(ε) = 0.004551

Z-score = 6.2622

Computed p-value = 0.000000
Reported p-value = 0.006720

Difference: 0.006720
Ratio (computed/reported): 0.00x
```

Result: p-value even smaller (essentially 0)!

Scenario 3: Use n=4 for everything (treating as pseudo-replicates)

```python
# Assume all measurements use n=4 (technical replicates only)
n_all = 4

var_i = (sd_i ** 2) / n_all
var_j = (sd_j ** 2) / n_all
var_ij = (sd_ij ** 2) / n_all

var_epsilon = var_ij + (f_j ** 2) * var_i + (f_i ** 2) * var_j
se_epsilon = np.sqrt(var_epsilon)
z = abs(epsilon) / se_epsilon
p_value = 2 * (1 - norm.cdf(z))

print(f"Using n=4 for all:")
print(f"Computed p-value = {p_value:.6f}")
print(f"Reported p-value = {reported_p_value:.6f}")

Output:
Using n=4 for all:
Computed p-value = 0.469582
Reported p-value = 0.006720
```

## Conclusions

### Summary of P-Value Reconstruction Attempts

We tested three different replicate count scenarios to reproduce the reported p-value (0.00672):

| Scenario | n_query | n_array | n_dmf | Computed p | Reported p | Ratio |
|----------|---------|---------|-------|------------|------------|-------|
| **Biological replicates** | 17 | 350 | 4 | 0.000003 | 0.00672 | **0.0005x** (1000x too small) |
| **Total measurements** | 68 | 1400 | 4 | ~0 | 0.00672 | **~0x** (too small) |
| **Pseudo-replicates** | 4 | 4 | 4 | 0.470 | 0.00672 | **70x** (too large) |

**Result:** None of the simple error propagation methods match the reported p-value!

### Why Simple Error Propagation Fails

**Key insight from Costanzo SI (Line 88):**
> "Colony size measurements were used to estimate single mutant fitness... with the exception that **bootstrapped means, instead of medians, across replicates were used in variance estimation**"

Costanzo et al. do NOT use simple SD²/n variance estimation. Instead:

1. **Bootstrap resampling** of the 17 or 350 control screens
2. **Batch effect correction** via Linear Discriminant Analysis
3. **Log-normal error model** (multiplicative fitness, not additive)
4. **Systematic factors** explicitly modeled in the fitness calculation

This complex statistical pipeline produces variance estimates that differ significantly from SD²/n.

### Implications for Schema Design

**Decision: Use Option 2 - Split replicate fields**

```python
class FitnessPhenotype(Phenotype):
    n_biological_replicates: int | None = Field(
        default=None,
        description="Number of independent biological units (e.g., screens, cultures)"
    )
    n_technical_replicates: int | None = Field(
        default=None,
        description="Number of technical measurements per biological replicate"
    )
```

**For Costanzo2016:**

- **Query SMF**: `n_biological=17, n_technical=4`
- **Array SMF**: `n_biological=350, n_technical=4`
- **DMF**: `n_biological=1, n_technical=4` (pseudo-replication!)

**Important caveats to document:**

1. ✅ Users CAN compute SE for basic error propagation: `SE = SD / sqrt(n_biological)`
2. ❌ Users CANNOT reproduce exact p-values without the bootstrap model
3. ⚠️ When `n_biological=1`, the measurement has **no true biological replication** (pseudo-replication)
4. 📊 Reported SD may be from bootstrap distribution, not raw sample SD

### Recommendation for KG Queries

When reconstructing interactions from KG queries:

1. **Use n_biological for SE calculation** (conservative)
2. **Document uncertainty propagation formula** used
3. **Warn users** that p-values are approximate (cannot exactly reproduce Costanzo's bootstrap model)
4. **Flag pseudo-replicated data** (n_biological=1) as lower confidence

### Open Question

**What is the "correct" n_replicates for Costanzo2016 DMF?**

- **Statistical perspective**: n=1 (only 1 biological replicate = 1 screen)
- **Pragmatic perspective**: n=4 (4 colonies measured, acknowledging pseudo-replication)
- **Conservative approach**: Use n=1, SE=undefined (or SD itself as rough SE)

**My recommendation:** Store `n_biological=1, n_technical=4` and compute SE conservatively, documenting the pseudo-replication limitation.
