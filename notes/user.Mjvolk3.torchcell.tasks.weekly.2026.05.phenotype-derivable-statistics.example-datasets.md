---
id: s9dti4u74ahylximlwel08a
title: example-datasets
desc: ''
updated: 1769647703458
created: 1769638879541
---

## 2026.01.28 - Dataset-Specific Statistics Computation

This document shows what statistics are computable from the actual data available in each phenotype type. **Red boxes** indicate statistics that **cannot be computed** due to missing input data.

## 1. FitnessPhenotype (Costanzo2016) - CURRENT

**Available Data:**

- `fitness` (mean fitness ratio)
- `fitness_std` (standard deviation)
- ⚠️ **NO n_samples** (not currently stored in schema)

**Computation Graph:**

```mermaid
graph TB
    %% Available Inputs
    subgraph AVAILABLE["AVAILABLE DATA"]
        A1[fitness mean]
        A2[fitness_std SD]
    end

    %% Missing Input
    subgraph MISSING["MISSING DATA"]
        A3[n_samples<br/>NOT STORED]
    end

    %% Computable Statistics
    B1[Variance<br/>from SD]
    C1[Standard Deviation<br/>Data Noise]
    C2[Variance<br/>Meta-analysis]
    E1[Coefficient of Variation<br/>Relative Variability]

    %% Unreachable Statistics
    D1[Standard Error<br/>UNREACHABLE]
    D2[95% Confidence Interval<br/>UNREACHABLE]
    D3[t-based CI<br/>UNREACHABLE]
    E2[CV of Mean Estimate<br/>UNREACHABLE]

    %% Connections
    A2 --> C1
    A2 --> B1
    B1 --> C2
    A1 --> E1
    A2 --> E1

    %% Blocked connections
    A3 -.->|BLOCKED| D1
    A2 -.->|BLOCKED| D1
    D1 -.->|BLOCKED| D2
    D1 -.->|BLOCKED| D3
    D1 -.->|BLOCKED| E2

    %% Styling
    classDef available fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    classDef missing fill:#fff9c4,stroke:#f9a825,stroke-width:3px,stroke-dasharray: 5 5
    classDef computable fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef unreachable fill:#ffcdd2,stroke:#c62828,stroke-width:3px

    class AVAILABLE,A1,A2 available
    class MISSING,A3 missing
    class B1,C1,C2,E1 computable
    class D1,D2,D3,E2 unreachable
```

**Result:** ⚠️ **Partial computation** - Can compute SD, Variance, and CV, but **cannot compute SE or CI** without sample size.

**What can be computed:**

- $\sigma^2 = \sigma \times \sigma$ ✅
- $CV = \frac{\text{fitness\_std}}{\text{fitness}} \times 100\%$ ✅

**What CANNOT be computed:**

- $SE = \frac{\text{fitness\_std}}{\sqrt{n}}$ ❌ (missing n)
- $CI = \text{fitness} \pm 1.96 \times SE$ ❌ (needs SE, which needs n)
- $CV_{SE} = \frac{SE}{\mu}$ ❌ (needs SE, which needs n)

**Problem:** Cannot switch `label_statistic_name` from `"fitness_std"` to `"fitness_se"` without n_samples!

## 2. CalMorphPhenotype (Ohya2005)

**Available Data:**

- `calmorph` (base morphological measurements - means)
- `calmorph_coefficient_of_variation` (CV values)
- ⚠️ **NO n_samples**

**Computation Graph:**

```mermaid
graph TB
    %% Available Inputs
    subgraph AVAILABLE["AVAILABLE DATA"]
        A1[calmorph mean]
        A2[calmorph_cv]
    end

    %% Missing Input
    subgraph MISSING["MISSING DATA"]
        A3[n_samples<br/>NOT AVAILABLE]
    end

    %% Computable Statistics
    B1[Standard Deviation<br/>from CV and mean]
    C1[Standard Deviation<br/>Data Noise]
    C2[Variance<br/>Meta-analysis]
    E1[Coefficient of Variation<br/>Relative Variability]

    %% Unreachable Statistics
    D1[Standard Error<br/>UNREACHABLE]
    D2[95% Confidence Interval<br/>UNREACHABLE]
    D3[t-based CI<br/>UNREACHABLE]
    E2[CV of Mean Estimate<br/>UNREACHABLE]

    %% Connections
    A2 --> E1
    A1 --> B1
    A2 --> B1
    B1 --> C1
    B1 --> C2

    %% Blocked connections
    A3 -.->|BLOCKED| D1
    C1 -.->|BLOCKED| D1
    D1 -.->|BLOCKED| D2
    D1 -.->|BLOCKED| D3
    D1 -.->|BLOCKED| E2

    %% Styling
    classDef available fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    classDef missing fill:#fff9c4,stroke:#f9a825,stroke-width:3px,stroke-dasharray: 5 5
    classDef computable fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef unreachable fill:#ffcdd2,stroke:#c62828,stroke-width:3px

    class AVAILABLE,A1,A2 available
    class MISSING,A3 missing
    class B1,C1,C2,E1 computable
    class D1,D2,D3,E2 unreachable
```

**Result:** ⚠️ **Partial computation** - Can derive SD and Variance from CV, but **cannot compute SE, CI, or CV_SE** without sample size.

**What can be computed:**

- $\sigma = \frac{CV \times \mu}{100}$ ✅
- $\sigma^2 = \sigma^2$ ✅
- $CV$ ✅ (already provided)

**What CANNOT be computed:**

- $SE = \frac{\sigma}{\sqrt{n}}$ ❌ (missing n)
- $CI = \mu \pm 1.96 \times SE$ ❌ (needs SE, which needs n)
- $CV_{SE} = \frac{SE}{\mu}$ ❌ (needs SE, which needs n)

**Recommendation:** If Ohya2005 paper reports sample sizes, add `calmorph_n_samples` field to enable SE computation.

## 3. MicroarrayExpressionPhenotype (Kemmeren2014)

**Available Data:**

- `expression_log2_ratio` (mean log2 fold change)
- `expression_log2_ratio_se` (standard error on log2 scale)
- `expression_log2_ratio_variance` (variance on log2 scale)
- `n_samples` (per-gene replicate counts)

**Computation Graph:**

```mermaid
graph TB
    %% Available Inputs
    subgraph AVAILABLE["AVAILABLE DATA - LOG2 SCALE"]
        A1[expression_log2_ratio<br/>mean]
        A2[expression_log2_ratio_se<br/>SE]
        A3[expression_log2_ratio_variance]
        A4[n_samples per gene]
    end

    %% Computable Statistics
    B1[Standard Deviation<br/>from SE and n]
    C1[Standard Deviation<br/>Data Noise]
    C2[Variance<br/>Meta-analysis]
    D1[Standard Error<br/>Mean Precision]
    D2[95% Confidence Interval]
    D3[t-based CI]
    E1[Coefficient of Variation<br/>on log2 scale]

    %% Note box
    NOTE["⚠️ NOTE: CV not typically<br/>used for log2 data<br/>Can be negative!"]

    %% Connections
    A2 --> D1
    A4 --> B1
    A2 --> B1
    B1 --> C1
    A3 --> C2
    D1 --> D2
    D1 --> D3
    A4 --> D3
    A1 --> E1
    C1 --> E1

    E1 -.->|Caution| NOTE

    %% Styling
    classDef available fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    classDef computable fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef note fill:#fff9c4,stroke:#f9a825,stroke-width:2px

    class AVAILABLE,A1,A2,A3,A4 available
    class B1,C1,C2,D1,D2,D3,E1 computable
    class NOTE note
```

**Result:** ✅ **All statistics computable** - SE and n_samples enable full reconstruction of SD, Variance, and CI.

**Key formulas (on log2 scale):**

- $\sigma = SE \times \sqrt{n}$
- $\sigma^2$ ✅ (already provided as `expression_log2_ratio_variance`)
- $CI = \bar{y} \pm 1.96 \times SE$ where $\bar{y}$ is mean log2 ratio

**Important note:** CV is typically **not used** for log2 expression data because:

- Log2 values can be negative (downregulated genes)
- Zero mean is possible (no change)
- CV = σ/μ becomes undefined or misleading when μ ≈ 0

**Alternative:** Use SE directly as uncertainty measure, or confidence intervals for hypothesis testing.

## Summary Table - Current State

| Dataset | Phenotype | Available Stats | Can Compute | Cannot Compute | Reason |
|---------|-----------|----------------|-------------|----------------|--------|
| Costanzo2016 | FitnessPhenotype | mean, SD | Variance, CV | SE, CI, CV_SE | ❌ Missing n_samples |
| Ohya2005 | CalMorphPhenotype | mean, CV | SD, Variance | SE, CI, CV_SE | ❌ Missing n_samples |
| Kemmeren2014 | MicroarrayExpressionPhenotype | mean, SE, Variance, n | SD, CI | None | ✅ Complete data (log2 scale) |

**Key finding:** Both Costanzo2016 and Ohya2005 are limited by missing `n_samples`.

## Recommendations - Current State

### For Costanzo2016 (FitnessPhenotype)

⚠️ **Cannot switch to SE** - Currently using `"fitness_std"` but cannot switch `label_statistic_name` to `"fitness_se"` without n_samples field.

### For Ohya2005 (CalMorphPhenotype)

⚠️ **Consider adding n_samples** if available from paper:

- Would enable SE computation for uncertainty quantification
- Would enable confidence intervals for hypothesis testing
- Currently limited to scale-independent CV for variability

### For Kemmeren2014 (MicroarrayExpressionPhenotype)

✅ **Already optimal** - Using SE as primary statistic on log2 scale is correct approach for expression data.

---

## PROPOSED: FitnessPhenotype with n_samples

### Proposed Schema Addition

Add `n_samples` field to `FitnessPhenotype`:

```python
class FitnessPhenotype(Phenotype, ModelStrict):
    graph_level: str = "global"
    label_name: str = "fitness"
    label_statistic_name: str = "fitness_se"  # CHANGED from fitness_std
    fitness: float = Field(description="wt_growth_rate/ko_growth_rate")
    fitness_se: float | None = Field(
        default=None,
        description="fitness standard error (primary uncertainty statistic)"
    )
    fitness_std: float | None = Field(
        default=None,
        description="fitness standard deviation (raw data from publication)"
    )
    n_samples: int | None = Field(  # NEW FIELD
        default=None,
        description="Number of replicate measurements"
    )
```

### Proposed Data Availability

- `fitness` (mean fitness ratio) ✅
- `fitness_std` (standard deviation) ✅
- `fitness_se` (standard error) ✅ (computed from SD and n)
- `n_samples` (number of replicates) ✅ **NEW**

### Proposed Computation Graph

```mermaid
graph TB
    %% Available Inputs
    subgraph AVAILABLE["AVAILABLE DATA - PROPOSED"]
        A1[fitness mean]
        A2[fitness_std SD]
        A3[n_samples]
    end

    %% Computable Statistics
    B1[Variance<br/>from SD]
    C1[Standard Deviation<br/>Data Noise]
    C2[Variance<br/>Meta-analysis]
    D1[Standard Error<br/>Mean Precision]
    D2[95% Confidence Interval]
    D3[t-based CI]
    E1[Coefficient of Variation<br/>Relative Variability]
    E2[CV of Mean Estimate<br/>Relative Precision]

    %% Connections
    A2 --> C1
    A2 --> B1
    B1 --> C2
    A2 --> D1
    A3 --> D1
    D1 --> D2
    D1 --> D3
    A3 --> D3
    A1 --> E1
    A2 --> E1
    A1 --> E2
    D1 --> E2

    %% Styling
    classDef available fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    classDef computable fill:#e1f5fe,stroke:#0288d1,stroke-width:2px

    class AVAILABLE,A1,A2,A3 available
    class B1,C1,C2,D1,D2,D3,E1,E2 computable
```

**Result:** ✅ **All statistics computable** - fitness_std and n_samples enable computation of SE, CI, and CV metrics.

### Benefits of Adding n_samples

1. **Enable SE computation:**
   - $SE = \frac{\text{fitness\_std}}{\sqrt{\text{n\_samples}}}$
   - Use SE as primary statistic (`label_statistic_name = "fitness_se"`)

2. **Enable confidence intervals:**
   - $CI_{95\%} = \text{fitness} \pm 1.96 \times SE$
   - $CI_t = \text{fitness} \pm t_{\alpha/2, n-1} \times SE$

3. **Enable deduplication with inverse-variance weighting:**
   - Weight experiments by precision: $w = \frac{1}{SE^2}$
   - Experiments with more replicates automatically weighted higher

4. **Enable proper uncertainty quantification:**
   - SE measures precision of mean estimate (better for ML training)
   - SD measures data variability (useful for QC)
   - Both available for different purposes

5. **Consistency with Kemmeren2014:**
   - Both use SE as primary statistic
   - Both store n_samples for reconstruction
   - Unified approach across datasets

### Data Source

Costanzo2016 paper reports:

- Query strains: **n = 68** biological replicates
- Array strains: **n = 4** technical replicates
- Double mutants: **n = 4** technical replicates (typically)

This information is available and can be stored in the schema.

### Implementation

```python
# In Costanzo2016 dataset loader
fitness_se = fitness_std / math.sqrt(n_samples)

phenotype = FitnessPhenotype(
    fitness=row[smf_key],
    fitness_std=fitness_std,      # Keep for reproducibility
    fitness_se=fitness_se,         # PRIMARY statistic
    n_samples=n_samples,           # Enable reconstruction
)
```

### Summary Table - Proposed State

| Dataset | Phenotype | Available Stats | Can Compute | Cannot Compute | Status |
|---------|-----------|----------------|-------------|----------------|--------|
| Costanzo2016 | FitnessPhenotype | mean, SD, **n** | SE, Variance, CV, CI | None | ✅ Complete with proposal |
| Ohya2005 | CalMorphPhenotype | mean, CV | SD, Variance | SE, CI, CV_SE | ❌ Still missing n |
| Kemmeren2014 | MicroarrayExpressionPhenotype | mean, SE, Variance, n | SD, CI | None | ✅ Already complete |

**Impact:** Adding `n_samples` to FitnessPhenotype brings it to parity with MicroarrayExpressionPhenotype for statistical rigor.
