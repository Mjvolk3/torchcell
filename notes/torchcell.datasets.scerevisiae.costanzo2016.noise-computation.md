---
id: dxmgc0lnxcxvme4j7k70v6f
title: Noise Computation
desc: ''
updated: 1769712520862
created: 1769636529805
---

## Costanzo2016 Fitness and Variance Computation

### Critical Quotes from SI

**Line 74 (Double mutant):**

> "All screens were conducted a single time with **4 replicate colonies per double mutant**, unless otherwise indicated"

**Line 88 (Single mutant):**

> "Colony size measurements of SGA deletion and TS array mutant strains were based on an average of **350 replicate control screens**... Colony size measurements of SGA deletion and TS query mutant strains were based on an average of **17 replicate control screens**... Colony size measurements were used to estimate single mutant fitness as described previously (5) with the exception that **bootstrapped means, instead of medians, across replicates were used in variance estimation and final fitness values**."

**Key:** "Replicates" for SMF = **screens** (17 or 350), NOT individual colonies!

### Measurement Summary

| Measurement | n_screens | n_colonies/screen | Total measurements | Variance method          | Reported "std" | n_replicates |
|-------------|-----------|-------------------|--------------------|--------------------------|----------------|--------------|
| Query SMF   | 17        | 4                 | 68                 | Bootstrap across screens | ≈ SE           | **17**       |
| Array SMF   | 350       | 4                 | 1400               | Bootstrap across screens | ≈ SE           | **350**      |
| DMF         | 1         | 4                 | 4                  | Raw sample SD            | Raw SD         | **4**        |

### Single Mutant Fitness (SMF) - Bootstrap Procedure

**Hierarchical structure:**

```
Screen 1: [col1, col2, col3, col4] → mean → screen1_value
Screen 2: [col1, col2, col3, col4] → mean → screen2_value
...
Screen N: [col1, col2, col3, col4] → mean → screenN_value
  (N = 17 for query, N = 350 for array)

Bootstrap: Resample with replacement from [screen1, ..., screenN]
  → Compute mean of resampled screens
  → Repeat B times (e.g., B=1000)
  → Report: mean of bootstrap means, SD of bootstrap means
```

**Key insight:** Bootstrap SD is already SE-like because it's the SD of the **sampling distribution of the mean**.

$$\sigma_{bootstrap}(f) \approx SE(f) = \frac{\sigma_{among\_screens}}{\sqrt{N}}$$

where N = 17 (query) or 350 (array).

### Double Mutant Fitness (DMF) - Raw Sample Statistics

**Computation (not explicitly stated, but likely):**

$$\bar{f}_{ij} = \frac{1}{4}\sum_{k=1}^{4} f_{ij,k}$$

$$\sigma(f_{ij}) = \sqrt{\frac{1}{3}\sum_{k=1}^{4}(f_{ij,k} - \bar{f}_{ij})^2}$$

**No bootstrap** - just raw sample SD across 4 colonies from one screen.

**To get SE:**
$$SE(f_{ij}) = \frac{\sigma(f_{ij})}{\sqrt{4}} = \frac{\sigma(f_{ij})}{2}$$

### Why n_replicates = 17/350, NOT 68/1400?

**Wrong interpretation:** Total measurements = 17×4 = 68 or 350×4 = 1400

**Correct interpretation:**

- Bootstrap resamples **screens as units**, not individual colonies
- 4 colonies per screen are **averaged before** bootstrap resampling
- Colonies on same plate = technical replicates (same environment, same day)
- Screens on different days = biological replicates (capture day-to-day variation)
- **Bootstrap needs biological variation** → resamples screens, not colonies

**Therefore:** n_replicates = number of screens (17 or 350)

### Implications for Error Propagation

For gene interactions: $\varepsilon = f_{ij} - f_i \cdot f_j$

$$Var(\varepsilon) = Var(f_{ij}) + f_j^2 Var(f_i) + f_i^2 Var(f_j)$$

**The mixing problem:**

| Term          | Variance                 | Interpretation                   |
|---------------|--------------------------|----------------------------------|
| $Var(f_i)$    | $\sigma_i^2$ (bootstrap) | Already SE² (accounts for n=17)  |
| $Var(f_j)$    | $\sigma_j^2$ (bootstrap) | Already SE² (accounts for n=350) |
| $Var(f_{ij})$ | $\sigma_{ij}^2$ (raw SD) | Must divide by n=4 to get SE²    |

**Why $SE = \sigma / \sqrt{n}$ fails:**

- For SMF: Over-corrects (bootstrap SD is already ≈ SE)
- For DMF: Correct (raw SD needs division by √n)

**This explains why we couldn't reproduce p-values** - we're mixing two different variance estimation procedures!

### Schema Implementation

**Store n_replicates as:**

- Query SMF: `n_replicates = 17` (screens)
- Array SMF: `n_replicates = 350` (screens)
- DMF: `n_replicates = 4` (colonies)

**Document clearly:**

```python
n_replicates: int | None = Field(
    default=None,
    description="""Number of independent units used to compute mean.

    WARNING: Interpretation of fitness_std differs by measurement type:
    - Costanzo SMF: Bootstrap SD (already SE-like, n=screens)
    - Costanzo DMF: Raw sample SD (divide by sqrt(n) for SE, n=colonies)

    See: torchcell.datasets.scerevisiae.costanzo2016.noise-computation
    """
)
```

### References

- SI Line 74: Double mutant screening (4 colonies per screen)
- SI Line 88: Single mutant fitness bootstrap (17 or 350 screens)
- SI Lines 572-575: Data file descriptions
