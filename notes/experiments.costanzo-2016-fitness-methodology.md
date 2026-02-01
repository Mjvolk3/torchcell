---
id: wwddho9slgllkqnc346ouz3
title: Costanzo 2016 Fitness Methodology
desc: ''
updated: 1769894179831
created: 1769894179831
---

## 2026.01.31 - Costanzo 2016 Fitness Measurement Methodology and TorchCell Adaptation

### Overview

This document details the Costanzo et al. 2016 fitness measurement methodology and our adaptation for TorchCell experiments using CRISPR-based mutant construction and echo spot plating instead of traditional SGA stamping.

**References:**

- Costanzo et al. (2016) Science 353, aaf1420
- Baryshnikova et al. (2010) Nat Methods 7, 1017-1024 (detailed algorithm)

---

## Costanzo 2016 Original Methodology

### The Colony Size Model

Fitness is **NOT** computed as simple mutant/WT ratio. Instead, they use a **multiplicative model**:

```
C_ij = f_ij × t × s_ij × e
```

**Where:**

- `C_ij` = colony size (pixels, measured from images)
- `f_ij` = double mutant fitness (parameter to estimate)
- `t` = incubation time
- `s_ij` = systematic factors (batch, plate position, temperature, etc.)
- `e` = log-normally distributed random noise

**Fitness decomposition:**

```
f_ij = f_i × f_j + ε_ij
```

Where:

- `f_i` = single mutant fitness (query)
- `f_j` = single mutant fitness (array)
- `ε_ij` = genetic interaction score

### Single Mutant Fitness (SMF) Estimation

**Control screens for SMF measurement:**

1. **Array mutants (DMA/TSA):**
   - Query: natMX marker at neutral locus (WT-like)
   - Array: kanMX-marked deletion/TS alleles
   - **Replicates: ~350 control screens** per temperature

2. **Query mutants:**
   - Query: natMX-marked deletion/TS alleles
   - Array: kanMX marker at neutral locus (WT-like)
   - **Replicates: ~17 control screens** per temperature

**Key processing steps:**

1. Measure colony size from images
2. Apply batch effect correction (Linear Discriminant Analysis)
3. Calibrate between arrays using overlapping strains
4. **Bootstrap resampling** of replicate measurements
5. Compute bootstrapped mean and SD for fitness

**Critical detail from paper:**

> "Colony size measurements were used to estimate single mutant fitness as described previously with the exception that **bootstrapped means, instead of medians, across replicates were used in variance estimation and final fitness values**."

### Double Mutant Fitness and Variance

**Experimental design:**

- Each double mutant: **4 replicate colonies** per screen
- High-confidence screens: **5 independent replicate screens** (40 query strains for reproducibility analysis = ~120,000 double mutants)

**Outputs reported in data files:**

- Double mutant fitness (μ)
- Double mutant fitness standard deviation (σ)
- Genetic interaction score (ε)
- P-value

---

## What Needs to be Tracked - Required Measurements

### 1. **Colony Size Measurements**

**Per colony:**

- Colony size (pixels or area)
- Plate ID
- Row position (1-16 for 384-well)
- Column position (1-24 for 384-well)
- Replicate number (1-4 standard, more for high confidence)
- Genotype (query × array)

**Image metadata:**

- Imaging timestamp
- Incubation time (from plating to imaging)
- Temperature during growth
- Image file path

### 2. **Experimental Factors to Control/Track**

**Systematic factors (s_ij in model):**

- **Batch ID**: Which day/batch cultures were prepared
- **Plate ID**: Individual plate identifier
- **Plate position effects**: Row/column systematic biases
- **Temperature**: Growth temperature (26°C or 30°C in Costanzo)
- **Incubation time**: Time from plating to imaging
- **Media batch**: Which batch of media was used
- **OD at plating**: Starting OD for dilution series

**For normalization/calibration:**

- **WT control colonies**: On every plate for normalization
- **Border effects**: Track if colonies are on plate edges
- **Reference strains**: Overlapping strains across plates for calibration

### 3. **Strain Information**

**Per strain:**

- Gene name
- Systematic name (e.g., YAL001C)
- Mutation type (deletion, TS allele, DAmP, etc.)
- Allele ID/number
- Selection marker (in your case, CRISPR guide info)
- Whether it's a query or array strain
- Temperature sensitivity (for TS alleles)

### 4. **Quality Control Metrics**

**Per colony:**

- Successful growth (yes/no)
- Contamination flag
- Edge effect flag (too close to plate border)
- Neighboring colony interference flag

**Per plate:**

- Number of colonies successfully measured
- Spatial bias metrics (row/column effects)
- WT control colony size distribution

---

## TorchCell Experimental Adaptation

### Key Differences from Costanzo 2016

| Aspect | Costanzo 2016 | TorchCell Adaptation |
|--------|---------------|---------------------|
| **Mutant construction** | SGA (mating + selection) | CRISPR editing |
| **Plating method** | Robotic stamping | Echo spot plating |
| **Colony formation** | Pinning transfers | Dilution to single colonies |
| **Starting material** | Pre-grown colonies | OD-standardized liquid cultures |
| **Dilution control** | Fixed pinning volume | Tunable dilution from OD reading |
| **Plate format** | 384-well (16×24) | 384-well (16×24) - **same** |
| **Image analysis** | Custom pipeline | sgatools.ccbr.utoronto.ca/imageanalysis |
| **Genotype** | Same (deletion/conditional) | Same achievable via CRISPR |

### Advantages of TorchCell Approach

1. **More precise dilution control**: Echo acoustic dispensing + OD standardization
2. **Better single colony formation**: Tunable dilution vs. fixed pinning
3. **Flexible mutant construction**: CRISPR allows any mutation type
4. **Standardized starting culture**: OD normalization before plating

### Critical Adaptations Needed

1. **Dilution rate tuning**: Must empirically determine dilution that gives single colonies from OD-standardized culture
2. **Echo volume calibration**: Determine volume range for optimal colony formation
3. **Growth time optimization**: May differ from stamping due to different starting cell numbers

---

## Pilot Study Design - Statistical Robustness Assessment

### Experimental Design

**Strains:**

- **WT** (reference)
- **3 SMF mutants** (single mutant fitness mutants - select with known fitness defects)

**Goals:**

1. Determine statistical robustness (how many replicates needed)
2. Determine optimal dilution rate from OD-standardized cultures
3. Validate compatibility with sgatools image analysis
4. Estimate variance structure for power analysis

### Recommended Pilot Structure

#### Phase 1: Dilution Rate Optimization (1 week)

**Design:**

- **WT only**
- **OD starting point**: 0.5 (standardized)
- **Dilution series**: 10^-3, 10^-4, 10^-5, 10^-6
- **Replicates**: 8 technical replicates per dilution
- **Plates**: 1 plate per dilution (32 WT colonies total per dilution)

**Measure:**

- Fraction of wells with single colonies
- Colony size distribution
- Edge effects
- Time to visible colony formation

**Success criteria:**

- ≥80% single colony formation
- <5% empty wells
- CV of colony size <30%

#### Phase 2: Strain Comparison and Replicate Determination (2 weeks)

**Design:**

- **Strains**: WT + 3 SMF mutants
- **Optimal dilution**: From Phase 1
- **Replicates per strain**:
  - 4 replicate colonies (Costanzo standard)
  - 8 replicate colonies (2× Costanzo)
  - 16 replicate colonies (4× Costanzo)
  - 32 replicate colonies (8× Costanzo - for variance estimation)
- **Plates**: Multiple plates to assess plate effects
- **Independent batches**: 3 independent culture batches to assess batch effects

**Measure:**

- Colony size (μ, σ) per strain
- Fitness relative to WT
- Variance components:
  - Within-plate variance
  - Between-plate variance
  - Between-batch variance
- Coefficient of variation (CV)

**Analysis:**

- Power analysis: How many replicates needed to detect 10%, 20%, 30% fitness differences?
- Variance decomposition: What fraction of variance is technical vs. biological?
- Compare to Costanzo 2016 reproducibility (Fig. S2)

### Critical Measurements to Track

**Every colony:**

```
colony_id, strain_name, genotype, plate_id, row, col,
batch_id, replicate_num, od_at_plating, dilution_factor,
echo_volume_uL, incubation_time_hrs, temperature_C,
colony_size_pixels, colony_size_area, edge_flag,
contamination_flag, timestamp_plated, timestamp_imaged
```

**Per plate:**

```
plate_id, media_batch, temperature, imaging_conditions,
num_colonies_measured, wt_control_mean, wt_control_sd,
row_effect_pvalue, col_effect_pvalue
```

**Per batch:**

```
batch_id, culture_date, od_measurement, dilution_used,
strains_plated, num_plates
```

### Expected Outcomes

1. **Optimal dilution rate**: e.g., "10^-5 dilution from OD=0.5 gives single colonies"
2. **Replicate requirements**: e.g., "16 replicates needed for 20% fitness difference, 80% power"
3. **Variance structure**: e.g., "Within-plate SD = 0.05, between-batch SD = 0.08"
4. **Protocol validation**: Confirm sgatools compatibility and image analysis quality

### sgatools Image Analysis Integration

**Platform**: <http://sgatools.ccbr.utoronto.ca/imageanalysis>

**Format**: 384-well (16×24) - **compatible with Costanzo pipeline**

**Required image metadata:**

- Plate barcode or ID
- Row/column grid overlay
- Growth condition (temperature, time)

**Output from sgatools:**

- Colony size measurements (pixels)
- Quality flags
- Grid alignment confidence

**Post-processing needed:**

- Link sgatools output to strain metadata
- Apply batch correction (LDA or similar)
- Bootstrap resampling for variance estimation
- Fitness calculation using multiplicative model

---

## Statistical Analysis Plan

### Variance Estimation via Bootstrapping

**Algorithm** (simplified from Baryshnikova 2010):

1. **Input**: Colony size measurements C = [c_1, c_2, ..., c_n] for n replicates
2. **Bootstrap iterations**: B = 1000 (typical)
3. **For each iteration b = 1 to B:**
   - Resample with replacement: C_b = random sample of n colonies from C
   - Fit multiplicative model to C_b → estimate fitness f_b
4. **Compute:**
   - Fitness μ = mean(f_1, f_2, ..., f_B)
   - Fitness σ = std(f_1, f_2, ..., f_B)
   - 95% CI = [2.5th percentile, 97.5th percentile]

### Fitness Calculation

**From colony sizes to fitness:**

```python
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

# Import ModelStrict from torchcell
# from torchcell.datamodels.pydant import ModelStrict

# Define ModelStrict here for completeness
class ModelStrict(BaseModel):
    class Config:
        extra = "forbid"
        frozen = True


class ColonyMeasurement(ModelStrict):
    """Single colony measurement from image analysis (sgatools output)"""
    colony_id: str = Field(..., description="Unique identifier for this colony")
    plate_id: str = Field(..., description="Plate identifier")
    row: int = Field(..., ge=1, le=16, description="Row position (1-16)")
    col: int = Field(..., ge=1, le=24, description="Column position (1-24)")
    colony_size_pixels: float = Field(..., gt=0, description="Colony size in pixels")
    strain_name: str = Field(..., description="Strain identifier")
    genotype: str = Field(..., description="Genotype (e.g., 'ACT1Δ', 'WT')")
    batch_id: str = Field(..., description="Culture batch identifier")
    replicate_num: int = Field(..., ge=1, description="Replicate number")

    # Experimental metadata
    od_at_plating: float = Field(..., gt=0, description="OD600 at plating time")
    dilution_factor: float = Field(..., gt=0, description="Dilution factor (e.g., 1e-5)")
    echo_volume_uL: float = Field(..., gt=0, description="Echo dispense volume in µL")
    incubation_time_hrs: float = Field(..., gt=0, description="Hours from plating to imaging")
    temperature_C: float = Field(..., description="Growth temperature in Celsius")

    # Timestamps
    timestamp_plated: datetime = Field(..., description="When colony was plated")
    timestamp_imaged: datetime = Field(..., description="When plate was imaged")

    # Quality flags
    edge_flag: bool = Field(False, description="True if colony is on plate edge")
    contamination_flag: bool = Field(False, description="True if contamination detected")

    @validator('row', 'col')
    def check_384_well_format(cls, v, field):
        """Ensure 384-well format (16×24)"""
        if field.name == 'row' and not (1 <= v <= 16):
            raise ValueError(f"Row must be 1-16 for 384-well, got {v}")
        if field.name == 'col' and not (1 <= v <= 24):
            raise ValueError(f"Column must be 1-24 for 384-well, got {v}")
        return v


class ColonySizeData(ModelStrict):
    """Collection of colony size measurements for a single strain"""
    strain_name: str = Field(..., description="Strain identifier")
    genotype: str = Field(..., description="Genotype")
    measurements: List[ColonyMeasurement] = Field(..., min_items=1, description="Individual colony measurements")

    @property
    def colony_sizes_array(self) -> np.ndarray:
        """Extract colony sizes as numpy array for analysis"""
        return np.array([m.colony_size_pixels for m in self.measurements])

    @property
    def n_replicates(self) -> int:
        """Number of replicate measurements"""
        return len(self.measurements)

    @property
    def mean_colony_size(self) -> float:
        """Mean colony size"""
        return float(np.mean(self.colony_sizes_array))

    @property
    def std_colony_size(self) -> float:
        """Standard deviation of colony size"""
        return float(np.std(self.colony_sizes_array, ddof=1))

    @property
    def cv_colony_size(self) -> float:
        """Coefficient of variation"""
        return self.std_colony_size / self.mean_colony_size if self.mean_colony_size > 0 else np.nan


class FitnessEstimate(ModelStrict):
    """Bootstrap fitness estimate results"""
    strain_name: str = Field(..., description="Strain identifier")
    genotype: str = Field(..., description="Genotype")
    fitness_mean: float = Field(..., description="Mean fitness from bootstrap")
    fitness_std: float = Field(..., description="Standard deviation from bootstrap")
    fitness_ci_lower: float = Field(..., description="2.5th percentile (lower 95% CI)")
    fitness_ci_upper: float = Field(..., description="97.5th percentile (upper 95% CI)")
    n_replicates: int = Field(..., ge=1, description="Number of colony replicates used")
    n_bootstrap: int = Field(1000, description="Number of bootstrap iterations")
    reference_strain: str = Field("WT", description="Reference strain for fitness calculation")


def estimate_fitness_bootstrap(
    colony_data: ColonySizeData,
    wt_colony_data: ColonySizeData,
    n_boot: int = 1000
) -> FitnessEstimate:
    """
    Estimate fitness using bootstrap resampling

    Args:
        colony_data: Colony measurements for mutant strain
        wt_colony_data: Colony measurements for WT reference from same plates
        n_boot: Number of bootstrap iterations

    Returns:
        FitnessEstimate with mean, std, and confidence intervals
    """
    # Extract colony size arrays
    colony_sizes = colony_data.colony_sizes_array
    wt_colony_sizes = wt_colony_data.colony_sizes_array

    # Bootstrap resampling
    fitness_boot = []

    for b in range(n_boot):
        # Resample with replacement
        mut_sample = np.random.choice(colony_sizes, size=len(colony_sizes), replace=True)
        wt_sample = np.random.choice(wt_colony_sizes, size=len(wt_colony_sizes), replace=True)

        # Ratio of means (simplified - full Costanzo uses model fitting with batch correction)
        fitness_b = np.mean(mut_sample) / np.mean(wt_sample)
        fitness_boot.append(fitness_b)

    # Convert to array for percentile calculation
    fitness_boot = np.array(fitness_boot)

    # Calculate statistics
    fitness_mu = float(np.mean(fitness_boot))
    fitness_sd = float(np.std(fitness_boot))
    fitness_ci_lower = float(np.percentile(fitness_boot, 2.5))
    fitness_ci_upper = float(np.percentile(fitness_boot, 97.5))

    return FitnessEstimate(
        strain_name=colony_data.strain_name,
        genotype=colony_data.genotype,
        fitness_mean=fitness_mu,
        fitness_std=fitness_sd,
        fitness_ci_lower=fitness_ci_lower,
        fitness_ci_upper=fitness_ci_upper,
        n_replicates=colony_data.n_replicates,
        n_bootstrap=n_boot,
        reference_strain=wt_colony_data.strain_name
    )


# Example usage:
"""
# Load sgatools output and create measurements
measurements = [
    ColonyMeasurement(
        colony_id="plate1_A01_rep1",
        plate_id="plate1",
        row=1, col=1,
        colony_size_pixels=1250.5,
        strain_name="ACT1Δ",
        genotype="ACT1Δ",
        batch_id="batch_2026_01_31",
        replicate_num=1,
        od_at_plating=0.5,
        dilution_factor=1e-5,
        echo_volume_uL=2.5,
        incubation_time_hrs=48.0,
        temperature_C=30.0,
        timestamp_plated=datetime.now(),
        timestamp_imaged=datetime.now(),
        edge_flag=False,
        contamination_flag=False
    ),
    # ... more measurements
]

# Create colony size data
act1_data = ColonySizeData(
    strain_name="ACT1Δ",
    genotype="ACT1Δ",
    measurements=measurements
)

wt_data = ColonySizeData(
    strain_name="WT",
    genotype="WT",
    measurements=wt_measurements
)

# Estimate fitness
fitness_result = estimate_fitness_bootstrap(act1_data, wt_data, n_boot=1000)

print(f"Fitness: {fitness_result.fitness_mean:.3f} ± {fitness_result.fitness_std:.3f}")
print(f"95% CI: [{fitness_result.fitness_ci_lower:.3f}, {fitness_result.fitness_ci_upper:.3f}]")
"""
```

**Note**: This is simplified. The full Costanzo method:

1. Corrects for systematic factors (batch, position) before ratio calculation
2. Uses log-transform given log-normal noise model
3. Fits the multiplicative model explicitly
4. Calibrates across plates using reference strains

### Power Analysis for Pilot - Detailed Explanation

**Central Question**: How many replicates needed to reliably detect a fitness difference Δf between mutant and WT?

#### What is Statistical Power?

**Power** = Probability of correctly detecting a real effect (avoiding false negative)

Think of it as: "If there really IS a 20% fitness difference, what's the chance my experiment will detect it as statistically significant?"

- **Power = 0.80 (80%)**: Standard in biology - you have 80% chance to detect a real effect
- **Power = 0.90 (90%)**: More conservative - better chance to detect, but requires more replicates
- **Power = 0.50 (50%)**: Coin flip - bad! You'll miss half of real effects

**Complementary concept - β (beta):**

- β = probability of **false negative** (missing a real effect)
- Power = 1 - β
- So Power = 0.80 means β = 0.20 (20% chance of missing a real effect)

#### Understanding the Parameters

**α (alpha) - Significance level:**

- Probability of **false positive** (claiming effect when there isn't one)
- α = 0.05 is standard → 5% false positive rate
- z_α/2 = 1.96 is the z-score for α = 0.05 (two-tailed test)

**β (beta) - False negative rate:**

- Probability of missing a real effect
- For Power = 0.80: β = 0.20
- z_β = 0.84 is the z-score for β = 0.20

**z-scores and where they come from:**

```
z_α/2 = 1.96  → This is the critical value where 2.5% of normal distribution is in each tail
                 (total 5% for two-tailed test at α=0.05)

z_β = 0.84    → This is the critical value where 20% of normal distribution is in the left tail
                 (for Power = 0.80, β = 0.20)
```

For different power levels:

- Power = 0.70 (β = 0.30): z_β = 0.52
- Power = 0.80 (β = 0.20): z_β = 0.84  ← **Standard choice**
- Power = 0.90 (β = 0.10): z_β = 1.28
- Power = 0.95 (β = 0.05): z_β = 1.64

#### Sample Size Formula Breakdown

**Formula:**

```
n = 2 × (z_α/2 + z_β)² × (σ / Δf)²
```

**What each term means:**

1. **n**: Number of replicates needed **per group** (mutant and WT each need n)
2. **2**: Accounts for comparing two groups (mutant vs WT)
3. **(z_α/2 + z_β)²**: Accounts for both false positive rate (α) and false negative rate (β)
4. **σ**: Standard deviation of fitness measurements (from your pilot data)
5. **Δf**: Minimum effect size you want to detect (e.g., 0.2 = 20% fitness difference)

**Interpretation:**

- **Larger σ** (more noise) → need **more replicates**
- **Smaller Δf** (smaller effect to detect) → need **more replicates**
- **Higher power** (lower β) → need **more replicates**

#### Worked Example with Interpretation

**Scenario**: Your pilot shows σ = 0.15 for fitness measurements

**Calculate n for different effect sizes:**

```python
import numpy as np

# Parameters
alpha = 0.05
power = 0.80
sigma = 0.15  # From pilot data

z_alpha_2 = 1.96  # For α = 0.05 (two-tailed)
z_beta = 0.84     # For power = 0.80

# Effect sizes to test
delta_f_values = [0.10, 0.15, 0.20, 0.30]

for delta_f in delta_f_values:
    n = 2 * ((z_alpha_2 + z_beta) ** 2) * ((sigma / delta_f) ** 2)
    print(f"To detect Δf = {delta_f:.2f} ({delta_f*100:.0f}% difference): n ≈ {np.ceil(n):.0f} replicates per strain")
```

**Results:**

- **Δf = 0.30 (30% difference)**: n ≈ 7 replicates
  - *Interpretation*: With 7 replicates each of mutant and WT, you have 80% chance to detect a 30% fitness defect
- **Δf = 0.20 (20% difference)**: n ≈ 16 replicates
  - *Interpretation*: Need 16 replicates to reliably detect more subtle 20% fitness defects
- **Δf = 0.15 (15% difference)**: n ≈ 28 replicates
  - *Interpretation*: Small effects require many replicates
- **Δf = 0.10 (10% difference)**: n ≈ 63 replicates
  - *Interpretation*: Very small effects are impractical to measure with colony sizes

**Reality check - Costanzo uses 4 replicates:**

- With n=4 and σ=0.15, they can detect Δf ≈ 0.42 (42%) at 80% power
- For smaller effects, they rely on:
  1. **Many genes** → statistical power from multiple comparisons
  2. **Reproducibility** → 5 independent screens for high confidence
  3. **Lower variance** → their σ is likely <0.15 due to optimized protocol

#### Key Insight: What High n in Pilot Tells You

**Your question**: "Will high n at set dilution help us estimate proper n for future experiments?"

**Answer: YES! High n in pilot is extremely valuable:**

1. **Precise σ estimate**:
   - With n=32, you get a precise estimate of σ
   - σ estimate uncertainty decreases as √(1/n)
   - n=32 gives you ~18% precision on σ
   - n=4 gives you ~35% precision on σ

2. **Variance decomposition**:
   - High n lets you separate:
     - **Technical variance** (measurement noise)
     - **Biological variance** (true variation in fitness)
   - This tells you whether adding replicates helps or if you're limited by biology

3. **Power curve generation**:
   - With precise σ, you can plot: n needed vs. Δf you want to detect
   - Gives you a decision tool for future experiments

4. **Protocol optimization**:
   - If σ is high, you know to optimize protocol before scaling up
   - If σ is low, you're good to proceed with fewer replicates

**Example:**

```python
# Pilot: n=32 replicates → σ = 0.12 ± 0.015 (precise!)
# Now you know: to detect 20% difference, need n=13

# If you had done n=4: σ = 0.12 ± 0.03 (uncertain!)
# Could be anywhere from σ=0.09 to σ=0.15
# n needed could be 8 to 19 replicates - big uncertainty!
```

#### Practical Recommendation for Your Pilot

**Given σ is unknown, aim for high n in pilot:**

- **Batch 1**: Test multiple dilutions, n=16-32 per dilution × WT
- **Batch 2**: Optimal dilution, n=32 each for WT + 3 mutants

This gives you:

1. Precise σ estimate for each strain
2. Ability to calculate exact n for future experiments
3. Variance decomposition (within-batch vs between-batch)
4. Protocol validation at scale

**Cost-benefit:**

- Upfront: More work in pilot (2 batches × 384-well plates)
- Future: Know exactly how many replicates needed → save time and resources
- Confidence: Know if your assay can detect the effects you care about

---

## 2-Batch Optimized Pilot Design (Practical Constraint)

### Challenge: Limited to 2 Batches

**Constraint**: Hard to motivate colleagues for >2 batches on different days

**Key Question**: Should we do dilution optimization across batches OR within one batch then apply optimal dilution to batch 2?

### Recommended Strategy: Combined Within-Batch + Between-Batch Design

**Answer**: Do dilution optimization **within Batch 1**, then validate **between batches** with Batch 2

**Rationale:**

1. **Dilution optimization is mostly technical** - doesn't require biological replicates
2. **Between-batch variance is biological** - need to measure it to know if assay is robust
3. **High n within batches** helps you estimate n needed despite limited batch replicates

#### Batch 1: Dilution Optimization + High-n Baseline (Day 1)

**Goal**: Find optimal dilution AND get precise within-batch variance estimate

**Design:**

- **Strains**: WT only (simplifies analysis)
- **Dilution series**: 10^-3, 10^-4, 10^-5, 10^-6
- **Layout**: 384-well plate (16×24 = 384 colonies)
  - Divide into 4 regions (96 colonies each)
  - Each region = one dilution
  - **n = 96 replicates per dilution**

**Plate layout example:**

```
Rows 1-8, Cols 1-12:   10^-3 dilution (96 colonies)
Rows 1-8, Cols 13-24:  10^-4 dilution (96 colonies)
Rows 9-16, Cols 1-12:  10^-5 dilution (96 colonies)
Rows 9-16, Cols 13-24: 10^-6 dilution (96 colonies)
```

**Measurements:**

- Single colony formation rate (% wells with 1 colony, 0 colonies, >1 colony)
- Colony size distribution per dilution
- Within-plate spatial effects (row/column)
- Coefficient of variation (CV) per dilution

**Success criteria for optimal dilution:**

- ≥85% single colony formation
- <3% empty wells
- <10% multi-colony wells
- CV < 0.25 (25%)

**Analysis output from Batch 1:**

1. **Optimal dilution** (e.g., 10^-5)
2. **Precise within-batch σ** at optimal dilution (n=96 gives ~10% precision on σ)
3. **Spatial effect model** (row, column, edge effects)

#### Batch 2: Multi-Strain + Between-Batch Variance (Day 2, 1-2 weeks later)

**Goal**: Measure between-batch variance and strain differentiation

**Design:**

- **Strains**: WT + 3 SMF mutants (4 strains total)
- **Dilution**: **Fixed at optimal from Batch 1**
- **Layout**: 384-well plate
  - Each strain: n = 96 replicates (4 strains × 96 = 384 colonies)

**Plate layout example:**

```
Rows 1-4, Cols 1-24:   WT (96 colonies)
Rows 5-8, Cols 1-24:   Mutant 1 (96 colonies)
Rows 9-12, Cols 1-24:  Mutant 2 (96 colonies)
Rows 13-16, Cols 1-24: Mutant 3 (96 colonies)
```

**Why high n=96 per strain is powerful:**

1. **Precise fitness estimates** for each strain in each batch
2. **Decompose variance**:

   ```
   Total variance = Within-batch variance + Between-batch variance
   σ²_total = σ²_within + σ²_between
   ```

3. **Calculate between-batch variance** even with only 2 batches:
   - Batch 1 WT: fitness_1, σ²_within_1 (from n=96)
   - Batch 2 WT: fitness_2, σ²_within_2 (from n=96)
   - Between-batch variance: σ²_between ≈ (fitness_1 - fitness_2)² / 2
   - Total variance for future: σ²_total = σ²_within + σ²_between

4. **Power calculation with both variance components**:
   - If σ²_between >> σ²_within → Need biological replicates (more batches)
   - If σ²_within >> σ²_between → Technical replicates sufficient (within batch)

#### Analysis Plan: Extracting Maximum Information from 2 Batches

**Step 1: Estimate variance components**

```python
# From Batch 1 (WT at optimal dilution, n=96)
wt_batch1_mean = np.mean(wt_batch1_fitness)
wt_batch1_var = np.var(wt_batch1_fitness, ddof=1)  # Within-batch variance

# From Batch 2 (WT at same dilution, n=96)
wt_batch2_mean = np.mean(wt_batch2_fitness)
wt_batch2_var = np.var(wt_batch2_fitness, ddof=1)  # Within-batch variance

# Pooled within-batch variance
sigma_within = np.sqrt((wt_batch1_var + wt_batch2_var) / 2)

# Between-batch variance (from 2 batch means)
sigma_between = np.abs(wt_batch1_mean - wt_batch2_mean) / np.sqrt(2)

# Total variance for future experiments
sigma_total = np.sqrt(sigma_within**2 + sigma_between**2)

print(f"Within-batch SD: {sigma_within:.4f}")
print(f"Between-batch SD: {sigma_between:.4f}")
print(f"Total SD: {sigma_total:.4f}")
print(f"Fraction from batch effects: {sigma_between**2 / sigma_total**2:.2%}")
```

**Step 2: Calculate n needed for future experiments**

```python
# Use sigma_total for power calculation
def calculate_n_needed(sigma_total, delta_f, alpha=0.05, power=0.80):
    z_alpha_2 = 1.96
    z_beta = 0.84
    n = 2 * ((z_alpha_2 + z_beta) ** 2) * ((sigma_total / delta_f) ** 2)
    return np.ceil(n)

# For different effect sizes
for delta in [0.10, 0.15, 0.20, 0.30]:
    n = calculate_n_needed(sigma_total, delta)
    print(f"To detect {delta:.0%} difference: n = {n:.0f} replicates per strain")
```

**Step 3: Assess if you need biological replicates (multiple batches)**

**Decision rule:**

- If σ²_between < 0.25 × σ²_total → **Technical replicates within batch are sufficient**
- If σ²_between > 0.5 × σ²_total → **Need multiple batches (biological replicates)**

**Interpretation:**

```python
batch_fraction = sigma_between**2 / sigma_total**2

if batch_fraction < 0.25:
    print("GOOD NEWS: Batch effects are small (<25% of variance)")
    print("Future experiments can use single batch with high n within batch")
    print(f"Recommend: n={calculate_n_needed(sigma_within, 0.20):.0f} per strain in one batch")

elif batch_fraction > 0.5:
    print("CAUTION: Batch effects are large (>50% of variance)")
    print("Future experiments need multiple batches to average out batch variance")
    print(f"Recommend: 3-5 batches with n={calculate_n_needed(sigma_within, 0.20):.0f} per strain each")

else:
    print("MODERATE: Batch effects contribute 25-50% of variance")
    print("Consider 2-3 batches for important measurements")
```

#### Example Scenario: What You Might Find

**Scenario 1: Low batch effects (ideal case)**

```
Batch 1 WT fitness: 1.000 ± 0.08 (n=96)
Batch 2 WT fitness: 1.005 ± 0.09 (n=96)

σ_within = 0.085
σ_between = 0.004
σ_total = 0.085

Batch effects: 0.2% of total variance ← EXCELLENT!

→ Future experiments: Single batch, n=16 per strain sufficient
```

**Scenario 2: High batch effects (need more batches)**

```
Batch 1 WT fitness: 1.000 ± 0.08 (n=96)
Batch 2 WT fitness: 0.920 ± 0.09 (n=96)  ← 8% difference between batches!

σ_within = 0.085
σ_between = 0.057
σ_total = 0.103

Batch effects: 31% of total variance ← CONCERNING

→ Future experiments: Need 3-5 batches with n=8-16 per strain each
→ OR improve protocol to reduce batch-to-batch variation
```

#### Why High n=96 is Worth It

**With n=96 per strain, you get:**

1. **Variance precision**: ±10% on σ estimate (vs ±35% with n=4)
2. **Reliable fitness ranking**: Can confidently order strains by fitness
3. **Batch effect quantification**: Even with 2 batches only
4. **Future experiment planning**: Know exactly n needed and whether batches matter
5. **Protocol validation**: CV < 25% proves assay is robust

**Cost comparison:**

- **2 batches × 4 strains × 96 replicates = 768 colonies = 2 plates**
- Same info from low-n design would require 5-10 batches (10-20 plates)
- High n **SAVES** time and resources by front-loading investment

#### Plate Utilization

**Batch 1**: 1× 384-well plate

- Dilution optimization (4 dilutions × 96 = 384 colonies)

**Batch 2**: 1× 384-well plate

- 4 strains × 96 replicates = 384 colonies

**Total**: 2 plates, 2 batches, complete pilot study

#### Critical Insight: SE to Distinguish Strains

**Your key question**: "Will high n at set dilution help estimate proper n for SE to distinguish strains?"

**Answer: Absolutely! Here's why:**

**Standard Error (SE) for distinguishing strains:**

```
SE_difference = √(σ²_mut/n_mut + σ²_wt/n_wt)
```

For equal n:

```
SE_difference = √(2σ²/n) = σ × √(2/n)
```

**To distinguish strains** (e.g., mutant fitness = 0.80 vs WT fitness = 1.00):

- Difference = 0.20 (20%)
- Need: Difference > 2 × SE for p < 0.05 (roughly)
- Therefore: 0.20 > 2 × σ × √(2/n)
- Solve for n: n > 8 × σ² / (0.20)²

**With high n=96 pilot, you get precise σ:**

- If σ = 0.10: n > 8 × 0.01 / 0.04 = **2 replicates** (easy!)
- If σ = 0.15: n > 8 × 0.0225 / 0.04 = **5 replicates** (doable)
- If σ = 0.25: n > 8 × 0.0625 / 0.04 = **13 replicates** (need more)

**Without high n, σ is uncertain:**

- With n=4: σ could be 0.10 ± 0.04 (huge uncertainty!)
- You might plan for n=5 but actually need n=13
- Wasted experiments

### sgatools Integration - Required Data Inputs

Based on <http://sgatools.ccbr.utoronto.ca/imageanalysis> interface:

#### File Naming Convention

**Critical**: For downstream scoring, files must follow pattern:

```
username_platetype_queryname_platenumber_...jpg
```

**Examples:**

- Control plate (WT): `mvolk_ctrl_WT_01_batch1_2026-01-31.jpg`
- Mutant plate: `mvolk_dm_ACT1delta_01_batch1_2026-01-31.jpg`

**Parsed fields:**

- `platetype`: `ctrl` (control/WT) or `dm` (double mutant)
- `queryname`: Strain identifier (e.g., `WT`, `ACT1delta`)
- `platenumber`: Plate number (e.g., `01`, `02`)

#### Required Settings

**Plate format:**

- ☑ 384 colonies (16 × 24) ← **Select this**

**Options:**

- ☑ Autorotate: Automatically adjust rotation
  - Recommended: Check this unless you have perfectly aligned images

**Colonies appearance:**

- ⚫ Bright colonies compared to plate background ← **Select this for standard YPD agar**
- ⚪ Dark colonies (for specific media types)

#### Image Requirements

**Format**: JPG (JPEG)

**Quality:**

- High resolution (at least 1200×1600 pixels recommended)
- Good contrast between colonies and background
- Even illumination (avoid shadows, glare)
- Focus: Colonies sharp and clear

**Timing:**

- Consistent time post-plating (e.g., always 48h ± 2h)
- Record exact imaging time for each plate

#### Notification

**Email**: Provide email for job completion notification

- Recommended for >5 images
- You'll receive link to download results

#### Output from sgatools

**What you get:**

- Colony size measurements (pixels) for each position
- Grid alignment confidence
- Quality flags (failed colonies, edge effects)
- CSV file with: plate_id, row, col, colony_size_pixels

**What you need to add post-sgatools:**

- Link to strain metadata (strain_name, genotype)
- Experimental metadata (batch, OD, dilution, temperature, timestamps)
- Use pydantic `ColonyMeasurement` model to structure data

#### Post-Processing Pipeline

```python
# 1. Load sgatools CSV output
sgatools_data = pd.read_csv("sgatools_output_plate1.csv")

# 2. Load metadata (your experimental records)
metadata = pd.read_csv("plate1_metadata.csv")

# 3. Merge on plate_id, row, col
merged = sgatools_data.merge(metadata, on=['plate_id', 'row', 'col'])

# 4. Create pydantic ColonyMeasurement objects
measurements = [
    ColonyMeasurement(
        colony_id=f"{row['plate_id']}_R{row['row']:02d}_C{row['col']:02d}",
        plate_id=row['plate_id'],
        row=row['row'],
        col=row['col'],
        colony_size_pixels=row['colony_size_pixels'],
        strain_name=row['strain_name'],
        genotype=row['genotype'],
        batch_id=row['batch_id'],
        replicate_num=row['replicate_num'],
        od_at_plating=row['od_at_plating'],
        dilution_factor=row['dilution_factor'],
        echo_volume_uL=row['echo_volume_uL'],
        incubation_time_hrs=row['incubation_time_hrs'],
        temperature_C=row['temperature_C'],
        timestamp_plated=pd.to_datetime(row['timestamp_plated']),
        timestamp_imaged=pd.to_datetime(row['timestamp_imaged']),
        edge_flag=row['edge_flag'],
        contamination_flag=False
    )
    for _, row in merged.iterrows()
]

# 5. Create ColonySizeData per strain
strains = merged['strain_name'].unique()
colony_data = {
    strain: ColonySizeData(
        strain_name=strain,
        genotype=merged[merged['strain_name']==strain]['genotype'].iloc[0],
        measurements=[m for m in measurements if m.strain_name == strain]
    )
    for strain in strains
}

# 6. Estimate fitness
for strain in strains:
    if strain != 'WT':
        fitness = estimate_fitness_bootstrap(
            colony_data[strain],
            colony_data['WT'],
            n_boot=1000
        )
        print(f"{strain}: {fitness.fitness_mean:.3f} ± {fitness.fitness_std:.3f}")
```

### Comparison to Costanzo Reproducibility

**Costanzo 2016 reproducibility metrics** (Fig. S2):

- 4 replicates standard
- 5 independent screens for high-confidence
- Precision vs. Recall trade-off based on threshold

**Your pilot should measure:**

1. **Reproducibility across technical replicates**: Same culture, same plate (n=96 within batch)
2. **Reproducibility across batches**: Different culture preparations (2 batches, high n each)
3. **Variance decomposition**: σ_within vs σ_between
4. **Comparison to Costanzo gold standard**: If possible, include a strain they measured

**Key differences:**

- Costanzo: Many batches, low n per batch (n=4)
- Your pilot: Few batches (n=2), high n per batch (n=96)
- **Both valid** - you're trading batch replicates for within-batch precision
- Your approach **better for variance estimation** with limited batch budget

---

## Implementation Checklist

### Before Starting

- [ ] Select 3 SMF mutants with known fitness defects (ideally from Costanzo data)
- [ ] Verify CRISPR mutants have correct genotype (sequencing)
- [ ] Calibrate Echo acoustic dispenser for agar plate spotting
- [ ] Test OD measurement protocol (standardize spectrophotometer, wavelength)
- [ ] Set up sgatools account and test image upload

### Phase 1: Dilution Optimization

- [ ] Prepare WT cultures, measure OD
- [ ] Prepare dilution series (10^-3 to 10^-6)
- [ ] Echo spot onto 4× 384-well plates
- [ ] Incubate at standard temperature
- [ ] Image at 24h, 48h, 72h
- [ ] Analyze with sgatools
- [ ] Calculate single colony formation rate per dilution
- [ ] Select optimal dilution

### Phase 2: Replicate Determination

- [ ] Prepare WT + 3 mutant cultures
- [ ] Standardize to OD = 0.5
- [ ] Apply optimal dilution from Phase 1
- [ ] Echo spot with replicate structure (4, 8, 16, 32 per strain)
- [ ] Repeat across 3 independent batches
- [ ] Image all plates
- [ ] Analyze with sgatools
- [ ] Export data with full metadata
- [ ] Perform variance decomposition analysis
- [ ] Calculate power for different replicate numbers
- [ ] Write up findings and recommend replicate structure

### Data Organization

**Recommended directory structure:**

```
experiments/
├── 001-smf-pilot/
│   ├── conf/
│   │   └── dilution_optimization.yaml
│   │   └── replicate_determination.yaml
│   ├── data/
│   │   ├── raw_images/
│   │   ├── sgatools_output/
│   │   └── metadata/
│   │       ├── strain_info.csv
│   │       ├── plate_layout.csv
│   │       └── batch_info.csv
│   ├── results/
│   │   ├── phase1_dilution_analysis.csv
│   │   ├── phase2_fitness_estimates.csv
│   │   └── power_analysis.csv
│   └── scripts/
│       ├── fitness_estimation.py
│       └── variance_analysis.py
```

---

## Key References and Resources

**Primary methodology:**

- Baryshnikova et al. (2010) Nat Methods 7:1017-1024 - **Full algorithm details**
- Costanzo et al. (2016) Science 353:aaf1420 - **2016 dataset and modifications**

**Data and tools:**

- Costanzo 2016 data: <http://thecellmap.org/costanzo2016/>
- sgatools: <http://sgatools.ccbr.utoronto.ca/imageanalysis>
- SGAtools documentation: Check if available from CCBR

**Statistical methods:**

- Bootstrap resampling: Efron & Tibshirani (1993)
- Power analysis: Cohen (1988)
- Variance component analysis: Mixed effects models

---

## Notes and Considerations

### Critical Success Factors

1. **OD standardization**: Most critical for reproducibility
2. **Echo calibration**: Verify volume delivery to agar
3. **Single colony formation**: Must tune dilution carefully
4. **Batch effects**: Track and correct systematically
5. **Temperature control**: Maintain constant growth temperature
6. **Imaging consistency**: Same settings, timing across all plates

### Potential Challenges

1. **Echo to agar**: May need optimization compared to liquid dispensing
2. **Drying time**: Spots must dry before incubation starts
3. **Growth time**: May differ from stamping (different starting cell #)
4. **sgatools compatibility**: Verify image format requirements
5. **Variance structure**: May differ from Costanzo due to different plating method

### Future Extensions

- Scale to double mutant measurements
- Implement full batch correction pipeline
- Develop automated data processing pipeline
- Compare to published Costanzo fitness values
- Extend to different growth conditions (temperature, media)
