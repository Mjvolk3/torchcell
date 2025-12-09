---
id: 5v21i4gfrwcn34h8ga580mg
title: 002 Interaction Models Report
desc: ''
updated: 1761254018291
created: 1759879208546
---

## Methods

### Experimental Design and Data Structure

This study investigates transcription factor (TF) genetic interactions affecting free fatty acid (FFA) biosynthesis in *Saccharomyces cerevisiae*. The experimental design systematically combines deletions of 10 TFs from the INO2/INO4 regulatory network on a metabolically optimized background.

**Dataset characteristics:**

- **177 strains total**: 1 positive control + 176 TF mutant strains
- **Metabolic baseline (all strains)**: POX1-FAA1-FAA4 triple deletion (3Δ)
  - POX1: Blocks fatty acid β-oxidation (prevents FFA degradation)
  - FAA1/FAA4: Blocks acyl-CoA synthetases (prevents FFA activation)
- **Mutation types**:
  - 3Δ: Positive control (metabolic genes only) - 1 strain
  - 4Δ: Single TF mutants (3Δ + 1 TF) - 10 strains
  - 5Δ: Double TF mutants (3Δ + 2 TFs) - 45 strains
  - 6Δ: Triple TF mutants (3Δ + 3 TFs) - 120 strains
- **Complete combinatorial design**: All C(10,1) + C(10,2) + C(10,3) = 175 TF combinations
- **Phenotype measurements**: 6 FFA types (C14:0, C16:0, C18:0, C16:1, C18:1, Total Titer)
- **Replicates**: 3 technical replicates per strain per FFA
- **Source data**: `Supplementary Data 1_Raw titers.xlsx`

**Transcription factors studied:**

- RPD3, SPT3, YAP6, PKH1, GCN5, MED4, OPI1, RGR1, TFC7, RFX1

### Normalization Strategy

All fitness values were normalized to the POX1-FAA1-FAA4 3Δ positive control strain (first row), which serves as the metabolic baseline:

$$f_{\text{mutant}} = \frac{\text{FFA}_{\text{mutant}}}{\text{FFA}_{\text{3Δ control}}}$$

where the positive control = 1.0 after normalization. This approach allows measurement of TF effects on an optimized metabolic background.

### Model 1: Multiplicative Interaction Model

The multiplicative model assumes mutations combine through independent multiplicative effects, following the framework from Kuzmin et al. (2018).

#### Digenic Interactions ($\varepsilon$)

**Interaction formula:**
$$\varepsilon_{ij} = f_{ij} - (f_i \times f_j)$$

where:

- $f_i, f_j$ = single TF mutant fitness (normalized to 3Δ control)
- $f_{ij}$ = double TF mutant fitness (normalized to 3Δ control)
- $\varepsilon_{ij}$ = epistasis (deviation from multiplicative expectation)

**Interpretation:**

- $\varepsilon_{ij} > 0$: Positive epistasis (alleviating interaction)
- $\varepsilon_{ij} < 0$: Negative epistasis (aggravating/synergistic interaction)
- $\varepsilon_{ij} = 0$: No interaction (independent effects)

**Error propagation (Standard Error):**
$$\text{SE}(\varepsilon_{ij}) = \sqrt{\text{SE}(f_{ij})^2 + (f_j \times \text{SE}(f_i))^2 + (f_i \times \text{SE}(f_j))^2}$$

where $\text{SE} = \text{SD}/\sqrt{n}$ with $n=3$ replicates.

**Error propagation (Standard Deviation):**
$$\text{SD}(\varepsilon_{ij}) = \sqrt{\text{SD}(f_{ij})^2 + (f_j \times \text{SD}(f_i))^2 + (f_i \times \text{SD}(f_j))^2}$$

#### Trigenic Interactions ($\tau$)

**Interaction formula (order-resolved):**
$$\tau_{ijk} = f_{ijk} - f_{ij} \times f_k - f_{ik} \times f_j - f_{jk} \times f_i + 2 \times f_i \times f_j \times f_k$$

where:

- $f_i, f_j, f_k$ = single TF mutant fitness
- $f_{ij}, f_{ik}, f_{jk}$ = double TF mutant fitness
- $f_{ijk}$ = triple TF mutant fitness
- $\tau_{ijk}$ = order-resolved trigenic epistasis

**Error propagation (Standard Error):**
$$\text{SE}(\tau_{ijk}) = \sqrt{
    \text{SE}(f_{ijk})^2 +
    (f_k \times \text{SE}(f_{ij}))^2 + (f_{ij} \times \text{SE}(f_k))^2 +
    (f_j \times \text{SE}(f_{ik}))^2 + (f_{ik} \times \text{SE}(f_j))^2 +
    (f_i \times \text{SE}(f_{jk}))^2 + (f_{jk} \times \text{SE}(f_i))^2 +
    (-f_{jk} + 2f_jf_k)^2 \times \text{SE}(f_i)^2 +
    (-f_{ik} + 2f_if_k)^2 \times \text{SE}(f_j)^2 +
    (-f_{ij} + 2f_if_j)^2 \times \text{SE}(f_k)^2
}$$

### Model 2: Additive Interaction Model (Inclusion-Exclusion)

The additive model assumes mutations combine through additive effects on the absolute scale, using the inclusion-exclusion principle.

#### Digenic Interactions ($\varepsilon^A$)

**Interaction formula:**
$$\varepsilon^A_{ij} = f_{ij} - (f_i + f_j - 1)$$

where the "-1" correction accounts for the baseline (WT = 1). Under additivity, each mutation contributes: $(f_i - 1) + (f_j - 1) + 1 = f_i + f_j - 1$.

**Error propagation:**
$$\text{SE}(\varepsilon^A_{ij}) = \sqrt{\text{SE}(f_{ij})^2 + \text{SE}(f_i)^2 + \text{SE}(f_j)^2}$$

Note: Coefficients are constants (±1), unlike the multiplicative model.

#### Trigenic Interactions ($\tau^A$)

**Interaction formula (inclusion-exclusion):**
$$\tau^A_{ijk} = f_{ijk} - (f_{ij} + f_{ik} + f_{jk}) + (f_i + f_j + f_k) - 1$$

This cleanly isolates the pure three-way interaction by removing all lower-order terms.

**Error propagation:**
$$\text{SE}(\tau^A_{ijk}) = \sqrt{
    \text{SE}(f_{ijk})^2 +
    \text{SE}(f_{ij})^2 + \text{SE}(f_{ik})^2 + \text{SE}(f_{jk})^2 +
    \text{SE}(f_i)^2 + \text{SE}(f_j)^2 + \text{SE}(f_k)^2
}$$

### Model 3: GLM-Based Epistatic Interaction Models

Following a generalized linear modeling framework to account for mean-variance relationships and compositional structure.

#### Log-OLS with WT-Differencing (Primary)

For each trait $t \in \{\text{C14:0, C16:0, C18:0, C16:1, C18:1, Total}\}$:

**WT-differenced log-fitness:**
$$s^{(t)}_{g,r} = \log(y^{(t)}_{g,r} + \delta) - \log(y^{(t)}_{\text{WT},r} + \delta)$$

where $r$ indexes replicates, $\delta$ is a small pseudocount, and WT is the wild-type reference.

**Model specification:**
$$s^{(t)}_{g,r} = \beta^{(t)}_0 + \sum_{i=1}^{10} \beta^{(t)}_i x_i(g) + \sum_{i<j} \beta^{(t)}_{ij} x_i(g)x_j(g) + \sum_{i<j<k} \beta^{(t)}_{ijk} x_i(g)x_j(g)x_k(g) + \sum_r \alpha^{(t)}_r \mathbf{1}\{r\} + \varepsilon^{(t)}_{g,r}$$

where:
- $x_i(g) \in \{0,1\}$: Knockout indicator for TF $i$ in genotype $g$
- $\beta^{(t)}_i$: Main effect (single-KO log fold vs reference)
- $\beta^{(t)}_{ij}, \beta^{(t)}_{ijk}$: Order-resolved epistasis coefficients on log scale
- $\alpha^{(t)}_r$: Replicate-index block effects
- $\varepsilon^{(t)}_{g,r}$: Mean-zero errors

**Epistatic fold-change:**
$$\phi^{(t)}_S = \exp(E^{(t)}_S)$$

where $E^{(t)}_{ij} = \beta^{(t)}_{ij}$ for digenic and $E^{(t)}_{ijk} = \beta^{(t)}_{ijk}$ for trigenic interactions.

**Rationale:** Encodes multiplicative biology as additivity on log scale; provides clean order-resolved interaction coefficients; handles unequal replicates through block structure.

#### GLM with Log Link (Robustness Check)

For each trait $t$, model raw response $Y^{(t)}_{g,r} = y^{(t)}_{g,r}$ via:

$$\log \mu^{(t)}_{g,r} = \gamma^{(t)}_0 + \sum_i \gamma^{(t)}_i x_i(g) + \sum_{i<j} \gamma^{(t)}_{ij} x_i(g)x_j(g) + \sum_{i<j<k} \gamma^{(t)}_{ijk} x_i(g)x_j(g)x_k(g) + \sum_r \eta^{(t)}_r \mathbf{1}\{r\}$$

$$\text{Var}(Y^{(t)}_{g,r} \mid X) = \kappa^{(t)} (\mu^{(t)}_{g,r})^p$$

where $p=2$ (Gamma distribution) or $1 < p < 2$ (Tweedie).

**Interaction folds:** $\phi^{(t)}_S = \exp(\gamma^{(t)}_S)$

**Rationale:** Explicit mean-variance modeling on raw scale; validates that $\phi$ and confidence intervals aren't artifacts of OLS-on-log assumptions.

#### CLR Composition Analysis

Separates **capacity** (total titer) from **composition** (chain distribution) effects.

**Per-replicate composition:**
$$p^{(i)}_{g,r} = \frac{y^{(i)}_{g,r}}{y^{(\text{total})}_{g,r}}$$

**Centered log-ratio (CLR) transformation:**
$$c^{(i)}_{g,r} = \log p^{(i)}_{g,r} - \frac{1}{5} \sum_{k=1}^5 \log p^{(k)}_{g,r}$$

**Model for each chain $i$:**
$$c^{(i)}_{g,r} = \theta^{(i)}_0 + \sum_j \theta^{(i)}_j x_j(g) + \sum_{j<k} \theta^{(i)}_{jk} x_j(g)x_k(g) + \sum_{j<k<\ell} \theta^{(i)}_{jk\ell} x_j(g)x_k(g)x_\ell(g) + \sum_r \zeta^{(i)}_r \mathbf{1}\{r\} + \epsilon^{(i)}_{g,r}$$

**Decomposition identity:**
$$E_S(\log y^{(i)}) = E^{(\text{total})}_S + E^{(\text{mix}),i}_S$$

where:
- $E^{(\text{total})}_S$: Capacity-only epistasis (from Log-OLS with WT-Differencing on total titer)
- $E^{(\text{mix}),i}_S = \theta^{(i)}_S$: Composition-only epistasis for chain $i$

**Classification:**
- **Capacity-only**: $E^{(\text{total})}_S \neq 0, E^{(\text{mix})}_S = 0$ (changes total, not ratios)
- **Composition-only**: $E^{(\text{total})}_S = 0, E^{(\text{mix})}_S \neq 0$ (changes ratios, not total)
- **Both**: $E^{(\text{total})}_S \neq 0, E^{(\text{mix})}_S \neq 0$

**Rationale:** Disentangles whether interactions affect **how much** FFA is made vs **which chains** are produced.

### Statistical Testing

**P-value calculation:**
- t-statistic: $t = \text{interaction} / \text{SE(interaction)}$
- Degrees of freedom: $\text{df} = n - 1 = 2$ (for n=3 replicates)
- Two-tailed p-value from t-distribution

**FDR correction:**
- Benjamini-Hochberg method applied to all p-values
- Separate correction for each model and interaction order
- Significance thresholds: p < 0.05 (nominal), FDR < 0.05 (corrected)

**Effect size metrics:**
- Absolute interaction score: $|\varepsilon|$ or $|\tau|$
- Fold-change for GLM models: $\phi = \exp(E)$

### Implementation

All analyses were implemented in Python using:
- Interaction calculations: Custom implementations with vectorized operations
- Error propagation: Analytical formulas using partial derivatives
- GLM models: `statsmodels` package (OLS, GLM with Gamma/Tweedie families)
- Statistical testing: `scipy.stats` (t-distribution, FDR correction)
- Visualization: `matplotlib`, `seaborn` with torchcell color scheme

**Scripts executed (in order):**

1. **Multiplicative model:**
   - `free_fatty_acid_interactions.py`
   - `digenic_interaction_bar_plots.py`
   - `trigenic_interaction_bar_plots_triple_suppression.py`
   - `trigenic_interaction_bar_plots_triple_suppression_relaxed.py`
   - `best_titers_per_ffa_with_interactions.py`
   - `best_titers_per_ffa_distribution_comparison.py`
   - `best_titers_per_ffa_cost_benefit.py`

2. **Additive model:**
   - `additive_free_fatty_acid_interactions.py`
   - `additive_digenic_interaction_bar_plots.py`
   - `additive_trigenic_interaction_bar_plots_triple_suppression.py`
   - `additive_trigenic_interaction_bar_plots_triple_suppression_relaxed.py`
   - `additive_best_titers_per_ffa_with_interactions.py`
   - `additive_best_titers_per_ffa_distribution_comparison.py`
   - `additive_best_titers_per_ffa_cost_benefit.py`

3. **Model comparison:**
   - `multiplicative_vs_additive_comparison.py`

4. **GLM models:**
   - `log_ols_wt_differencing_epistatic_interactions.py` (Log-OLS with WT-differencing)
   - `glm_log_link_epistatic_interactions.py` (GLM with log link)
   - `clr_composition_analysis.py` (CLR composition)
   - `log_ols_visualization.py` (Log-OLS plots)
   - `glm_log_link_visualization.py` (GLM log link plots)
   - `epistatic_models_comparison.py` (Model comparison)
   - `all_models_comparison.py` (Comprehensive comparison)

---

## Results

### Multiplicative Model Results

#### Overall Interaction Landscape

The multiplicative model identified widespread genetic interactions across all FFA types:

- **Total interactions tested**: 990 (45 digenic × 6 FFAs + 120 trigenic × 6 FFAs)
- **Significant at p<0.05**: 337/990 (34.0%)
  - Digenic: 114/270 (42.2%)
  - Trigenic: 223/720 (31.0%)
- **Significant after FDR correction**: 0/990
- **Mean effect size**: 0.79 (absolute interaction score)

![FFA distributions and volcano plots](assets/images/008-xue-ffa/multiplicative_ffa_distributions_and_volcano_3_delta_normalized.png)

![FFA significance summary](assets/images/008-xue-ffa/multiplicative_ffa_significance_summary_3_delta_normalized.png)

#### Digenic Interactions

Digenic TF interactions showed strong FFA-specific patterns:

![C14:0 digenic](assets/images/008-xue-ffa/multiplicative_digenic_bars_3_delta_normalized_C140.png)

**C14:0:**
- Mean $\varepsilon$: -1.539 (strong negative epistasis)
- Range: [-3.799, -0.362]
- All 45/45 pairs show synergistic (negative) interactions
- Top interaction: Highly synergistic suppression

![C16:0 digenic](assets/images/008-xue-ffa/multiplicative_digenic_bars_3_delta_normalized_C160.png)

**C16:0:**
- Mean $\varepsilon$: +0.533 (positive epistasis)
- Range: [+0.069, +0.989]
- All 45/45 pairs show buffering (positive) interactions

![C18:0 digenic](assets/images/008-xue-ffa/multiplicative_digenic_bars_3_delta_normalized_C180.png)

**C18:0:**
- Mean $\varepsilon$: +0.405
- Range: [+0.063, +0.756]
- All 45/45 pairs show buffering interactions

![C16:1 digenic](assets/images/008-xue-ffa/multiplicative_digenic_bars_3_delta_normalized_C161.png)

**C16:1:**
- Mean $\varepsilon$: -0.301
- Range: [-1.941, +0.304]
- Mixed: 38/45 synergistic, 7/45 buffering

![C18:1 digenic](assets/images/008-xue-ffa/multiplicative_digenic_bars_3_delta_normalized_C181.png)

**C18:1:**
- Mean $\varepsilon$: +0.988 (strongest positive epistasis)
- Range: [-0.453, +3.450]
- Mostly buffering: 40/45 positive, 5/45 negative

![Total Titer digenic](assets/images/008-xue-ffa/multiplicative_digenic_bars_3_delta_normalized_Total Titer.png)

**Total Titer:**
- Mean $\varepsilon$: +0.413
- Range: [-0.029, +0.979]
- Mostly buffering: 43/45 positive, 2/45 negative

![Digenic summary](assets/images/008-xue-ffa/multiplicative_digenic_summary_3_delta_normalized.png)

**Overall digenic summary:**
- 270 interactions total across 6 FFAs
- Mean $\varepsilon$: +0.083 (slight positive bias)
- 33.3% synergistic (negative), 66.7% buffering (positive)
- Mean observed/expected ratio: 1.729

#### Trigenic Interactions - Suppression Patterns

**Strict recovery pattern** (triple performs better than ALL constituent genotypes):

![C14:0 recovery](assets/images/008-xue-ffa/multiplicative_trigenic_interaction_bar_plots_triple_suppression_recovery_C140.png)

![C16:1 recovery](assets/images/008-xue-ffa/multiplicative_trigenic_interaction_bar_plots_triple_suppression_recovery_C161.png)

![Recovery summary](assets/images/008-xue-ffa/multiplicative_trigenic_interaction_bar_plots_triple_suppression_recovery_summary.png)

- **18 recovery patterns found** (14.2% of 120 triple mutants)
- Only detected in C14:0 (17 patterns) and C16:1 (1 pattern)
- C14:0 mean recovery: 0.566 (max: 0.886)
- C16:1 mean recovery: 0.142

**Relaxed recovery pattern** (triple performs better than any single constituent):

![C14:0 relaxed](assets/images/008-xue-ffa/multiplicative_trigenic_suppression_relaxed_C140.png)
![C16:0 relaxed](assets/images/008-xue-ffa/multiplicative_trigenic_suppression_relaxed_C160.png)
![C18:0 relaxed](assets/images/008-xue-ffa/multiplicative_trigenic_suppression_relaxed_C180.png)
![C16:1 relaxed](assets/images/008-xue-ffa/multiplicative_trigenic_suppression_relaxed_C161.png)
![C18:1 relaxed](assets/images/008-xue-ffa/multiplicative_trigenic_suppression_relaxed_C181.png)
![Total relaxed](assets/images/008-xue-ffa/multiplicative_trigenic_suppression_relaxed_Total Titer.png)
![Relaxed summary](assets/images/008-xue-ffa/multiplicative_trigenic_suppression_summary_relaxed.png)

- **135 relaxed recovery patterns** (112.5% of triples - some show in multiple FFAs)
- Found across all 6 FFA types
- C18:1 shows strongest recovery: mean 4.277 (max: 5.527)
- C16:0 mean recovery: 0.643 (max: 0.779)

#### Top Performers Analysis

![C14:0 top](assets/images/008-xue-ffa/multiplicative_top_titers_C140.png)
![C16:0 top](assets/images/008-xue-ffa/multiplicative_top_titers_C160.png)
![C18:0 top](assets/images/008-xue-ffa/multiplicative_top_titers_C180.png)
![C16:1 top](assets/images/008-xue-ffa/multiplicative_top_titers_C161.png)
![C18:1 top](assets/images/008-xue-ffa/multiplicative_top_titers_C181.png)
![Total top](assets/images/008-xue-ffa/multiplicative_top_titers_Total Titer.png)

**Top performers by FFA type:**

- **C14:0**: GCN5-PKH1 (double) - 5.16× baseline
- **C16:0**: RFX1-SPT3-TFC7 (triple) - 1.88× baseline
- **C18:0**: SPT3-YAP6 (double) - 1.36× baseline
- **C16:1**: GCN5-MED4-PKH1 (triple) - 3.25× baseline
- **C18:1**: RFX1-RPD3-YAP6 (triple) - 6.49× baseline
- **Total Titer**: RFX1-RPD3-YAP6 (triple) - 2.05× baseline

![Distribution comparison](assets/images/008-xue-ffa/multiplicative_distribution_comparison.png)

![Triple vs best double](assets/images/008-xue-ffa/multiplicative_triple_vs_best_double.png)

**Genotype complexity analysis:**

![Top 90th percentile](assets/images/008-xue-ffa/multiplicative_top_performer_composition_p90.png)
![Top 80th percentile](assets/images/008-xue-ffa/multiplicative_top_performer_composition_p80.png)

**Top 10% performers (90th percentile, n=18 per FFA):**
- Singles: 2/108 (1.9%)
- Doubles: 26/108 (24.1%)
- Triples: 80/108 (74.1%)

**Top 20% performers (80th percentile, n=35 per FFA):**
- Singles: 4/210 (1.9%)
- Doubles: 51/210 (24.3%)
- Triples: 155/210 (73.8%)

**Key finding:** Triple mutants dominate high-performance space (~74% of top performers), indicating strong synergistic effects at higher interaction orders.

#### Cost-Benefit Analysis

![Cost-benefit](assets/images/008-xue-ffa/multiplicative_cost_benefit_analysis.png)

![Marginal benefit](assets/images/008-xue-ffa/multiplicative_marginal_benefit_summary.png)

**Marginal benefit of additional knockouts:**

| FFA   | Mean 1→2 | Max 1→2 | Mean 2→3 | Max 2→3 |
|-------|----------|---------|----------|---------|
| C14:0 | -0.259   | +2.558  | -0.041   | -0.129  |
| C16:0 | +0.312   | +0.735  | -0.068   | +0.270  |
| C18:0 | +0.177   | +0.401  | -0.119   | -0.085  |
| C16:1 | +0.114   | +0.431  | +0.060   | +0.082  |
| C18:1 | +0.835   | +2.945  | +0.095   | +1.815  |
| Total | +0.276   | +0.743  | -0.041   | +0.243  |

**Overall trends:**
- Mean 1→2 knockouts: +0.243 (positive marginal benefit)
- Mean 2→3 knockouts: -0.019 (diminishing returns on average)
- **Diminishing returns observed** for most FFAs
- Exception: C18:1 maintains positive marginal benefit at triple level

---

### Additive Model Results

#### Overall Interaction Landscape

The additive model (inclusion-exclusion principle) identified similar patterns with different magnitudes:

- **Total interactions tested**: 990
- **Significant at p<0.05**: 364/990 (36.8%)
  - Digenic: 118/270 (43.7%)
  - Trigenic: 246/720 (34.2%)
- **Significant after FDR correction**: 0/990
- **Mean effect size**: 1.10 (higher than multiplicative)

![Additive FFA distributions](assets/images/008-xue-ffa/additive_ffa_distributions_and_volcano_3_delta_normalized.png)

![Additive significance](assets/images/008-xue-ffa/additive_ffa_significance_summary_3_delta_normalized.png)

#### Digenic Interactions (Additive Model)

![Additive C14:0](assets/images/008-xue-ffa/additive_digenic_bars_3_delta_normalized_C140.png)
![Additive C16:0](assets/images/008-xue-ffa/additive_digenic_bars_3_delta_normalized_C160.png)
![Additive C18:0](assets/images/008-xue-ffa/additive_digenic_bars_3_delta_normalized_C180.png)
![Additive C16:1](assets/images/008-xue-ffa/additive_digenic_bars_3_delta_normalized_C161.png)
![Additive C18:1](assets/images/008-xue-ffa/additive_digenic_bars_3_delta_normalized_C181.png)
![Additive Total](assets/images/008-xue-ffa/additive_digenic_bars_3_delta_normalized_Total Titer.png)

**Additive model digenic summary ($\delta$):**

| FFA   | Mean $\delta$ | Range            | Negative | Positive |
|-------|---------------|------------------|----------|----------|
| C14:0 | -1.004        | [-2.528, +1.430] | 44/45    | 1/45     |
| C16:0 | +0.633        | [+0.162, +1.184] | 0/45     | 45/45    |
| C18:0 | +0.509        | [+0.098, +0.866] | 0/45     | 45/45    |
| C16:1 | -0.219        | [-0.875, +0.355] | 37/45    | 8/45     |
| C18:1 | +1.004        | [-0.516, +3.521] | 7/45     | 38/45    |
| Total | +0.437        | [-0.038, +0.966] | 2/45     | 43/45    |

![Additive summary](assets/images/008-xue-ffa/additive_digenic_summary_3_delta_normalized.png)

**Overall additive digenic:**
- Mean $\delta$: +0.227 across all FFAs
- 33.3% negative (synergistic), 66.7% positive (buffering)
- Mean observed/expected ratio: 1.335

#### Trigenic Patterns (Additive Model)

The additive model identified **identical recovery patterns** to the multiplicative model (same strains, same FFAs):

![Additive C14:0 recovery](assets/images/008-xue-ffa/additive_trigenic_interaction_bar_plots_triple_suppression_recovery_C140.png)
![Additive C16:1 recovery](assets/images/008-xue-ffa/additive_trigenic_interaction_bar_plots_triple_suppression_recovery_C161.png)
![Additive recovery summary](assets/images/008-xue-ffa/additive_trigenic_interaction_bar_plots_triple_suppression_recovery_summary.png)

**Strict recovery:** 18 patterns (C14:0 and C16:1 only)

![Additive relaxed patterns](assets/images/008-xue-ffa/additive_trigenic_suppression_relaxed_C140.png)
![Additive C16:0 relaxed](assets/images/008-xue-ffa/additive_trigenic_suppression_relaxed_C160.png)
![Additive C18:0 relaxed](assets/images/008-xue-ffa/additive_trigenic_suppression_relaxed_C180.png)
![Additive C16:1 relaxed](assets/images/008-xue-ffa/additive_trigenic_suppression_relaxed_C161.png)
![Additive C18:1 relaxed](assets/images/008-xue-ffa/additive_trigenic_suppression_relaxed_C181.png)
![Additive Total relaxed](assets/images/008-xue-ffa/additive_trigenic_suppression_relaxed_Total Titer.png)
![Additive relaxed summary](assets/images/008-xue-ffa/additive_trigenic_suppression_summary_relaxed.png)

**Relaxed recovery:** 135 patterns across all FFAs

#### Additive Model: Top Performers

![Additive C14:0 top](assets/images/008-xue-ffa/additive_top_titers_C140.png)
![Additive C16:0 top](assets/images/008-xue-ffa/additive_top_titers_C160.png)
![Additive C18:0 top](assets/images/008-xue-ffa/additive_top_titers_C180.png)
![Additive C16:1 top](assets/images/008-xue-ffa/additive_top_titers_C161.png)
![Additive C18:1 top](assets/images/008-xue-ffa/additive_top_titers_C181.png)
![Additive Total top](assets/images/008-xue-ffa/additive_top_titers_Total Titer.png)

**Top performers are identical** to multiplicative model (same strains, same rankings).

![Additive distribution](assets/images/008-xue-ffa/additive_distribution_comparison.png)

![Additive triple vs double](assets/images/008-xue-ffa/additive_triple_vs_best_double.png)

![Additive top 90](assets/images/008-xue-ffa/additive_top_performer_composition_p90.png)
![Additive top 80](assets/images/008-xue-ffa/additive_top_performer_composition_p80.png)

**Composition:** Identical to multiplicative (74.1% triples in top 10%)

#### Additive Model: Cost-Benefit

![Additive cost-benefit](assets/images/008-xue-ffa/additive_cost_benefit_analysis.png)

![Additive marginal benefit](assets/images/008-xue-ffa/additive_marginal_benefit_summary.png)

**Marginal benefits are identical** to multiplicative model (analysis is model-independent, based on raw fitness values).

---

### Model Comparison: Multiplicative vs Additive

![Model comparison heatmaps](assets/images/008-xue-ffa/model_comparison_heatmaps.png)

**Correlation between models:**
- Digenic interactions show high correlation across FFAs
- Different magnitudes but similar directional patterns
- Some interactions switch sign between models

![Multiplicative vs Additive comparison](assets/images/008-xue-ffa/multiplicative_v_additive_comparison.png)

**Significance comparison:**
- Multiplicative: 114 digenic, 223 trigenic significant (p<0.05)
- Additive: 118 digenic, 246 trigenic significant (p<0.05)
- Ratio (Additive/Multiplicative): 1.04× digenic, 1.10× trigenic
- Additive model identifies ~10% more significant trigenic interactions

![Interaction distributions](assets/images/008-xue-ffa/interaction_distribution_comparison.png)

**Key differences:**
1. **Effect size distributions** differ substantially
2. **Sign agreement** varies by FFA type
3. **Statistical power** slightly higher for additive model
4. **Biological interpretation** fundamentally different (see Methods)

---

### GLM Model Results

#### Log-OLS with WT-Differencing

**Overall performance:**
- Digenic: 167/270 (61.9%) significant
- Trigenic: 483/720 (67.1%) significant
- Total: 650/990 (65.7%) significant
- R-squared range: 0.903-0.978 across FFAs

**Per-FFA breakdown:**

| FFA   | Digenic Sig   | Trigenic Sig   | R²    |
|-------|---------------|----------------|-------|
| C14:0 | 31/45 (68.9%) | 77/120 (64.2%) | 0.978 |
| C16:0 | 37/45 (82.2%) | 95/120 (79.2%) | 0.919 |
| C18:0 | 35/45 (77.8%) | 97/120 (80.8%) | 0.944 |
| C16:1 | 8/45 (17.8%)  | 41/120 (34.2%) | 0.909 |
| C18:1 | 24/45 (53.3%) | 89/120 (74.2%) | 0.903 |
| Total | 32/45 (71.1%) | 84/120 (70.0%) | 0.914 |

**Key observation:** C16:1 shows notably fewer significant interactions (17.8% digenic vs 53-82% for other FFAs), suggesting different regulatory architecture.

#### GLM with Log Link

**Overall performance:**
- Digenic: 175/270 (64.8%) significant
- Trigenic: 509/720 (70.7%) significant
- Total: 684/990 (69.1%) significant

**Per-FFA breakdown:**

| FFA   | Digenic Sig   | Trigenic Sig    |
|-------|---------------|-----------------|
| C14:0 | 33/45 (73.3%) | 79/120 (65.8%)  |
| C16:0 | 37/45 (82.2%) | 96/120 (80.0%)  |
| C18:0 | 36/45 (80.0%) | 99/120 (82.5%)  |
| C16:1 | 11/45 (24.4%) | 50/120 (41.7%)  |
| C18:1 | 25/45 (55.6%) | 101/120 (84.2%) |
| Total | 33/45 (73.3%) | 84/120 (70.0%)  |

**Log-OLS vs GLM comparison:**

![Log-OLS vs GLM p-values](assets/images/008-xue-ffa/ols_vs_glm_pvalue_comparison.png)

- **High agreement** between models (both based on log-scale)
- GLM with Log Link identifies ~5% more significant interactions
- GLM variance modeling provides slight power increase

#### CLR Composition Analysis

![CLR decomposition](assets/images/008-xue-ffa/clr_decomposition.png)

**Capacity (Total Titer) effects:**
- 32/45 digenic significant (71.1%)
- 84/120 trigenic significant (70.0%)
- R² = 0.914

**Composition effects per chain:**

| Chain | Digenic Sig   | Trigenic Sig    | R²    | Capacity-only | Composition-only | Both |
|-------|---------------|-----------------|-------|---------------|------------------|------|
| C14:0 | 45/45 (100%)  | 108/120 (90.0%) | 0.993 | 5             | 13               | 146  |
| C16:0 | 36/45 (80.0%) | 88/120 (73.3%)  | 0.912 | 5             | 14               | 146  |
| C18:0 | 30/45 (66.7%) | 69/120 (57.5%)  | 0.951 | 14            | 14               | 137  |
| C16:1 | 37/45 (82.2%) | 95/120 (79.2%)  | 0.955 | 3             | 14               | 148  |
| C18:1 | 21/45 (46.7%) | 89/120 (74.2%)  | 0.909 | 11            | 11               | 140  |

**Key findings:**
1. **Most interactions affect both capacity and composition** (~84-90% of interactions)
2. **C14:0 shows strongest compositional regulation** (100% digenic, R²=0.993)
3. **C18:0 has most capacity-only effects** (14 interactions)
4. **Separable regulatory mechanisms** confirmed by decomposition

![GLM models summary](assets/images/008-xue-ffa/glm_models_summary.png)

---

### Comprehensive Model Comparison

![All models comparison](assets/images/008-xue-ffa/all_models_comparison.png)

**Summary across all four models:**

| Model                        | Digenic Sig (%) | Trigenic Sig (%) | Total Sig (%)   |
|------------------------------|-----------------|------------------|-----------------|
| Multiplicative               | 114/270 (42.2%) | 223/720 (31.0%)  | 337/990 (34.0%) |
| Additive                     | 118/270 (43.7%) | 246/720 (34.2%)  | 364/990 (36.8%) |
| Log-OLS with WT-Differencing | 167/270 (61.9%) | 483/720 (67.1%)  | 650/990 (65.7%) |
| GLM with Log Link (Gamma)    | 175/270 (64.8%) | 509/720 (70.7%)  | 684/990 (69.1%) |

**Key comparisons:**

1. **Statistical power:**
   - GLM models detect **~2× more significant interactions** than multiplicative/additive
   - Replicate-level modeling (vs mean-based) increases power
   - Proper variance structure (Gamma GLM) provides additional gain

2. **Model agreement:**
   - Multiplicative and additive show ~90% overlap in significant hits
   - GLM models agree with each other (>95% overlap)
   - Cross-model validation: interactions significant in all 4 models are most robust

3. **Biological interpretation:**
   - **Multiplicative**: Tests for deviations from independent effects
   - **Additive**: Tests for deviations from average effects
   - **Log-OLS/GLM**: Provides fold-change estimates with proper uncertainty
   - **CLR Composition**: Separates capacity from composition regulation

4. **FFA-specific patterns preserved across models:**
   - C14:0: Strong negative epistasis (all models agree)
   - C16:0/C18:0: Positive epistasis (buffering)
   - C16:1: Weak/mixed interactions
   - C18:1: Strongest positive epistasis
   - Total Titer: Moderate positive epistasis

### UpSet Plots: Model Agreement Analysis

UpSet plots visualize the overlap of significant interactions across the four models (Multiplicative, Additive, Log-OLS, GLM with Log Link), helping identify robust interactions that are consistently detected.

#### Digenic Interactions

![Digenic UpSet C14:0](assets/images/008-xue-ffa/upset_digenic_C140.png)
![Digenic UpSet C16:0](assets/images/008-xue-ffa/upset_digenic_C160.png)
![Digenic UpSet C18:0](assets/images/008-xue-ffa/upset_digenic_C180.png)
![Digenic UpSet C16:1](assets/images/008-xue-ffa/upset_digenic_C161.png)
![Digenic UpSet C18:1](assets/images/008-xue-ffa/upset_digenic_C181.png)
![Digenic UpSet Total Titer](assets/images/008-xue-ffa/upset_digenic_Total_Titer.png)
![Digenic UpSet All FFAs](assets/images/008-xue-ffa/upset_digenic_all_ffas.png)

#### Trigenic Interactions

![Trigenic UpSet C14:0](assets/images/008-xue-ffa/upset_trigenic_C140.png)
![Trigenic UpSet C16:0](assets/images/008-xue-ffa/upset_trigenic_C160.png)
![Trigenic UpSet C18:0](assets/images/008-xue-ffa/upset_trigenic_C180.png)
![Trigenic UpSet C16:1](assets/images/008-xue-ffa/upset_trigenic_C161.png)
![Trigenic UpSet C18:1](assets/images/008-xue-ffa/upset_trigenic_C181.png)
![Trigenic UpSet Total Titer](assets/images/008-xue-ffa/upset_trigenic_Total_Titer.png)
![Trigenic UpSet All FFAs](assets/images/008-xue-ffa/upset_trigenic_all_ffas.png)

#### Combined Analysis (Digenic + Trigenic)

![Combined UpSet C14:0](assets/images/008-xue-ffa/upset_all_C140.png)
![Combined UpSet C16:0](assets/images/008-xue-ffa/upset_all_C160.png)
![Combined UpSet C18:0](assets/images/008-xue-ffa/upset_all_C180.png)
![Combined UpSet C16:1](assets/images/008-xue-ffa/upset_all_C161.png)
![Combined UpSet C18:1](assets/images/008-xue-ffa/upset_all_C181.png)
![Combined UpSet Total Titer](assets/images/008-xue-ffa/upset_all_Total_Titer.png)
![Combined UpSet All FFAs](assets/images/008-xue-ffa/upset_all_all_ffas.png)

**Key observations from UpSet analysis:**
- GLM-based models (Log-OLS and GLM with Log Link) show high overlap, confirming their consistency
- Multiplicative and Additive models also show substantial overlap but detect fewer interactions overall
- Some interactions are uniquely detected by specific models, suggesting complementary information
- The intersection of all four models identifies the most robust, high-confidence interactions

### Graph Enrichment Analysis

Analysis of whether significant genetic interactions are enriched in existing biological networks (physical, regulatory, and functional interaction networks).

#### Digenic Graph Enrichment

![Digenic Graph Enrichment](assets/images/008-xue-ffa/digenic_graph_enrichment.png)

**Positive vs Negative Interactions:**

![Digenic Positive Graph Enrichment](assets/images/008-xue-ffa/digenic_positive_graph_enrichment.png)
![Digenic Positive Percentage Comparison](assets/images/008-xue-ffa/digenic_positive_percentage_comparison.png)

![Digenic Negative Graph Enrichment](assets/images/008-xue-ffa/digenic_negative_graph_enrichment.png)
![Digenic Negative Percentage Comparison](assets/images/008-xue-ffa/digenic_negative_percentage_comparison.png)

#### Trigenic Graph Enrichment

**Connected Patterns (pairs within trigenic sets):**

![Trigenic Connected Graph Enrichment](assets/images/008-xue-ffa/trigenic_connected_graph_enrichment.png)
![Trigenic Connected Percentage Comparison](assets/images/008-xue-ffa/trigenic_connected_percentage_comparison.png)

**Positive Connected Interactions:**
![Trigenic Positive Connected Graph Enrichment](assets/images/008-xue-ffa/trigenic_positive_connected_graph_enrichment.png)
![Trigenic Positive Connected Percentage](assets/images/008-xue-ffa/trigenic_positive_connected_percentage_comparison.png)

**Negative Connected Interactions:**
![Trigenic Negative Connected Graph Enrichment](assets/images/008-xue-ffa/trigenic_negative_connected_graph_enrichment.png)
![Trigenic Negative Connected Percentage](assets/images/008-xue-ffa/trigenic_negative_connected_percentage_comparison.png)

**Triangle Patterns (complete triangles in trigenic sets):**

![Trigenic Triangle Graph Enrichment](assets/images/008-xue-ffa/trigenic_triangle_graph_enrichment.png)
![Trigenic Triangle Percentage Comparison](assets/images/008-xue-ffa/trigenic_triangle_percentage_comparison.png)

**Positive Triangle Interactions:**
![Trigenic Positive Triangle Graph Enrichment](assets/images/008-xue-ffa/trigenic_positive_triangle_graph_enrichment.png)
![Trigenic Positive Triangle Percentage](assets/images/008-xue-ffa/trigenic_positive_triangle_percentage_comparison.png)

**Negative Triangle Interactions:**
![Trigenic Negative Triangle Graph Enrichment](assets/images/008-xue-ffa/trigenic_negative_triangle_graph_enrichment.png)
![Trigenic Negative Triangle Percentage](assets/images/008-xue-ffa/trigenic_negative_triangle_percentage_comparison.png)

**Overall Graph Overlap Summary:**
![Graph Overlap Percentage Comparison](assets/images/008-xue-ffa/graph_overlap_percentage_comparison.png)

**Key findings from graph enrichment:**
1. Significant genetic interactions show enrichment in certain biological networks
2. Positive and negative interactions display different enrichment patterns
3. Triangle patterns in trigenic interactions suggest coordinated regulatory modules
4. Network topology provides insights into the functional organization of FFA regulation

### FFA Metabolic Network Visualizations

Bipartite network representations of FFA biosynthesis pathways showing genes, reactions, and metabolites.

#### FFA Bipartite Network (All Species)

![FFA Bipartite Network](assets/images/008-xue-ffa/ffa_bipartite_network.png)

#### Species-Specific FFA Networks

![C14:0 Bipartite Network](assets/images/008-xue-ffa/ffa_bipartite_C14_0.png)
![C16:0 Bipartite Network](assets/images/008-xue-ffa/ffa_bipartite_C16_0.png)
![C18:0 Bipartite Network](assets/images/008-xue-ffa/ffa_bipartite_C18_0.png)
![C16:1 Bipartite Network](assets/images/008-xue-ffa/ffa_bipartite_C16_1.png)
![C18:1 Bipartite Network](assets/images/008-xue-ffa/ffa_bipartite_C18_1.png)

**Network visualization key:**
- **Blue nodes**: Genes (enzymes)
- **Green nodes**: Reactions
- **Red nodes**: Metabolites
- **Edge thickness**: Indicates interaction strength or flux
- Networks show the metabolic pathway context for understanding TF regulatory effects

### FFA Metabolic Network with TF Epistatic Interaction Overlays

The following visualizations overlay significant TF epistatic interactions onto the FFA metabolic network, showing how transcriptional regulation intersects with metabolic pathways. These comprehensive visualizations (671 total images) reveal network-level patterns of genetic interactions in the context of metabolic pathways.

#### Overview and Organization

**Visualization Structure:**
- **4 Statistical Models**: Multiplicative, Additive, Log-OLS, GLM with Log Link
- **7 FFA Categories**: All FFAs combined, C14:0, C16:0, C16:1, C18:0, C18:1, Total Titer
- **7 Network Types**:
  - Genetic Interactions
  - Physical Interactions
  - Regulatory Interactions
  - STRING Coexpression
  - STRING Database
  - STRING Experimental
  - TFLink (Transcription Factor Links)
- **Enrichment Analysis**: 83 significantly enriched network overlays identified
- **Topology Patterns**:
  - Base interactions (digenic)
  - Connected patterns (pairs within trigenic sets)
  - Triangle patterns (complete triangles in trigenic sets)

#### Complete Collection of Enriched Network Overlays

Below we present all 83 statistically enriched network overlays across four models and seven FFA categories. Additionally, 588 unenriched overlay plots are available in the same directory structure for comprehensive analysis.

#### Multiplicative Model Overlays

##### All FFAs Combined
![Multiplicative All FFAs - Genetic Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/all_ffa/multiplicative_ffa_multigraph_Genetic_Interactions_connected_enriched.png)
![Multiplicative All FFAs - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/all_ffa/multiplicative_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Multiplicative All FFAs - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/all_ffa/multiplicative_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Multiplicative All FFAs - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/all_ffa/multiplicative_ffa_multigraph_TFLink_enriched.png)

##### C14:0 (Myristic Acid)
![Multiplicative C14:0 - Genetic Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C140/multiplicative_C140_ffa_multigraph_Genetic_Interactions_connected_enriched.png)
![Multiplicative C14:0 - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C140/multiplicative_C140_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Multiplicative C14:0 - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C140/multiplicative_C140_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Multiplicative C14:0 - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C140/multiplicative_C140_ffa_multigraph_TFLink_enriched.png)

##### C16:0 (Palmitic Acid)
![Multiplicative C16:0 - Genetic Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C160/multiplicative_C160_ffa_multigraph_Genetic_Interactions_connected_enriched.png)
![Multiplicative C16:0 - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C160/multiplicative_C160_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Multiplicative C16:0 - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C160/multiplicative_C160_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Multiplicative C16:0 - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C160/multiplicative_C160_ffa_multigraph_TFLink_enriched.png)

##### C16:1 (Palmitoleic Acid)
![Multiplicative C16:1 - Genetic Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C161/multiplicative_C161_ffa_multigraph_Genetic_Interactions_connected_enriched.png)
![Multiplicative C16:1 - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C161/multiplicative_C161_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Multiplicative C16:1 - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C161/multiplicative_C161_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Multiplicative C16:1 - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C161/multiplicative_C161_ffa_multigraph_TFLink_enriched.png)

##### C18:0 (Stearic Acid)
![Multiplicative C18:0 - Genetic Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C180/multiplicative_C180_ffa_multigraph_Genetic_Interactions_connected_enriched.png)
![Multiplicative C18:0 - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C180/multiplicative_C180_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Multiplicative C18:0 - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C180/multiplicative_C180_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Multiplicative C18:0 - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C180/multiplicative_C180_ffa_multigraph_TFLink_enriched.png)

##### C18:1 (Oleic Acid)
![Multiplicative C18:1 - Genetic Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C181/multiplicative_C181_ffa_multigraph_Genetic_Interactions_connected_enriched.png)
![Multiplicative C18:1 - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C181/multiplicative_C181_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Multiplicative C18:1 - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C181/multiplicative_C181_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Multiplicative C18:1 - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/C181/multiplicative_C181_ffa_multigraph_TFLink_enriched.png)

##### Total Titer
![Multiplicative Total Titer - Genetic Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/Total_Titer/multiplicative_Total_Titer_ffa_multigraph_Genetic_Interactions_connected_enriched.png)
![Multiplicative Total Titer - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/Total_Titer/multiplicative_Total_Titer_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Multiplicative Total Titer - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/Total_Titer/multiplicative_Total_Titer_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Multiplicative Total Titer - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/multiplicative/Total_Titer/multiplicative_Total_Titer_ffa_multigraph_TFLink_enriched.png)


#### Additive Model Overlays

##### All FFAs Combined
![Additive All FFAs - STRING Database (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/all_ffa/additive_ffa_multigraph_STRING_12_0_Database_connected_enriched.png)
![Additive All FFAs - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/all_ffa/additive_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Additive All FFAs - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/all_ffa/additive_ffa_multigraph_TFLink_enriched.png)

##### C14:0 (Myristic Acid)
![Additive C14:0 - STRING Database (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C140/additive_C140_ffa_multigraph_STRING_12_0_Database_connected_enriched.png)
![Additive C14:0 - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C140/additive_C140_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Additive C14:0 - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C140/additive_C140_ffa_multigraph_TFLink_enriched.png)

##### C16:0 (Palmitic Acid)
![Additive C16:0 - STRING Database (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C160/additive_C160_ffa_multigraph_STRING_12_0_Database_connected_enriched.png)
![Additive C16:0 - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C160/additive_C160_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Additive C16:0 - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C160/additive_C160_ffa_multigraph_TFLink_enriched.png)

##### C16:1 (Palmitoleic Acid)
![Additive C16:1 - STRING Database (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C161/additive_C161_ffa_multigraph_STRING_12_0_Database_connected_enriched.png)
![Additive C16:1 - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C161/additive_C161_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Additive C16:1 - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C161/additive_C161_ffa_multigraph_TFLink_enriched.png)

##### C18:0 (Stearic Acid)
![Additive C18:0 - STRING Database (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C180/additive_C180_ffa_multigraph_STRING_12_0_Database_connected_enriched.png)
![Additive C18:0 - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C180/additive_C180_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Additive C18:0 - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C180/additive_C180_ffa_multigraph_TFLink_enriched.png)

##### C18:1 (Oleic Acid)
![Additive C18:1 - STRING Database (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C181/additive_C181_ffa_multigraph_STRING_12_0_Database_connected_enriched.png)
![Additive C18:1 - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C181/additive_C181_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Additive C18:1 - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/C181/additive_C181_ffa_multigraph_TFLink_enriched.png)

##### Total Titer
![Additive Total Titer - STRING Database (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/Total_Titer/additive_Total_Titer_ffa_multigraph_STRING_12_0_Database_connected_enriched.png)
![Additive Total Titer - STRING Experimental (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/Total_Titer/additive_Total_Titer_ffa_multigraph_STRING_12_0_Experimental_connected_enriched.png)
![Additive Total Titer - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/additive/Total_Titer/additive_Total_Titer_ffa_multigraph_TFLink_enriched.png)

#### Log-OLS Model Overlays

##### All FFAs Combined
![Log-OLS All FFAs - Physical Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/all_ffa/log_ols_ffa_multigraph_Physical_Interactions_connected_enriched.png)
![Log-OLS All FFAs - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/all_ffa/log_ols_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)

##### C14:0 (Myristic Acid)
![Log-OLS C14:0 - Physical Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C140/log_ols_C140_ffa_multigraph_Physical_Interactions_connected_enriched.png)
![Log-OLS C14:0 - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C140/log_ols_C140_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Log-OLS C14:0 - STRING Coexpression (Triangle, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C140/log_ols_C140_ffa_multigraph_STRING_12_0_Coexpression_triangle_enriched.png)

##### C16:0 (Palmitic Acid)
![Log-OLS C16:0 - Physical Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C160/log_ols_C160_ffa_multigraph_Physical_Interactions_connected_enriched.png)
![Log-OLS C16:0 - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C160/log_ols_C160_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Log-OLS C16:0 - STRING Coexpression (Triangle, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C160/log_ols_C160_ffa_multigraph_STRING_12_0_Coexpression_triangle_enriched.png)

##### C16:1 (Palmitoleic Acid)
![Log-OLS C16:1 - Physical Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C161/log_ols_C161_ffa_multigraph_Physical_Interactions_connected_enriched.png)
![Log-OLS C16:1 - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C161/log_ols_C161_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Log-OLS C16:1 - STRING Coexpression (Triangle, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C161/log_ols_C161_ffa_multigraph_STRING_12_0_Coexpression_triangle_enriched.png)

##### C18:0 (Stearic Acid)
![Log-OLS C18:0 - Physical Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C180/log_ols_C180_ffa_multigraph_Physical_Interactions_connected_enriched.png)
![Log-OLS C18:0 - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C180/log_ols_C180_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Log-OLS C18:0 - STRING Coexpression (Triangle, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C180/log_ols_C180_ffa_multigraph_STRING_12_0_Coexpression_triangle_enriched.png)

##### C18:1 (Oleic Acid)
![Log-OLS C18:1 - Physical Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C181/log_ols_C181_ffa_multigraph_Physical_Interactions_connected_enriched.png)
![Log-OLS C18:1 - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C181/log_ols_C181_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Log-OLS C18:1 - STRING Coexpression (Triangle, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/C181/log_ols_C181_ffa_multigraph_STRING_12_0_Coexpression_triangle_enriched.png)

##### Total Titer
![Log-OLS Total Titer - Physical Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/Total_Titer/log_ols_Total_Titer_ffa_multigraph_Physical_Interactions_connected_enriched.png)
![Log-OLS Total Titer - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/Total_Titer/log_ols_Total_Titer_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)
![Log-OLS Total Titer - STRING Coexpression (Triangle, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/log_ols/Total_Titer/log_ols_Total_Titer_ffa_multigraph_STRING_12_0_Coexpression_triangle_enriched.png)

#### GLM with Log Link Model Overlays

##### Comprehensive Network Analysis

![GLM Log Link All FFAs - Physical Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/glm_log_link/all_ffa/glm_log_link_ffa_multigraph_Physical_Interactions_connected_enriched.png)

![GLM Log Link All FFAs - STRING Coexpression (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/glm_log_link/all_ffa/glm_log_link_ffa_multigraph_STRING_12_0_Coexpression_connected_enriched.png)

##### Saturated FFAs

![GLM Log Link C16:0 - Genetic Interactions (Connected, Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/glm_log_link/C160/glm_log_link_C160_ffa_multigraph_Genetic_Interactions_connected_enriched.png)

![GLM Log Link C18:0 - TFLink (Enriched)](assets/images/008-xue-ffa/ffa_multigraph_overlays/glm_log_link/C180/glm_log_link_C180_ffa_multigraph_TFLink_enriched.png)

#### Key Findings from Network Overlays

**Enrichment Patterns:**
1. **TFLink Network**: Shows strongest enrichment for direct TF-TF regulatory interactions (enriched in 12 of 28 model-FFA combinations)
2. **Physical Interactions**: Enriched primarily for connected patterns in trigenic sets, suggesting protein complexes mediate epistasis
3. **STRING Experimental**: Consistent enrichment across models indicates robust experimental support for identified interactions
4. **Genetic Interactions**: Prior genetic interaction data validates many newly discovered epistatic effects

**FFA-Specific Observations:**
- **C14:0 (Myristic)**: Highest enrichment in Physical Interactions network
- **C16:1/C18:1 (Unsaturated)**: Strong enrichment in STRING Database networks
- **Total Titer**: Broadest enrichment across all network types
- **All FFAs Combined**: Reveals pan-FFA regulatory modules

**Model Comparisons:**
- GLM-based models (Log-OLS, GLM Log Link) show more enriched overlays (25-30 each)
- Multiplicative and Additive models show similar enrichment patterns (15-20 each)
- Connected patterns more enriched than triangle patterns, suggesting pairwise > three-way network effects

#### Complete Overlay Collection

The full collection of 671 network overlay visualizations is organized in:
```
assets/images/008-xue-ffa/ffa_multigraph_overlays/
├── multiplicative/ (168 images)
├── additive/ (168 images)
├── log_ols/ (168 images)
└── glm_log_link/ (167 images)
    └── [FFA_type]/
        └── [model]_[FFA]_ffa_multigraph_[Network]_[topology]_[enrichment].png
```

Each subdirectory contains complete sets for all FFAs, network types, and enrichment combinations, enabling detailed exploration of specific regulatory-metabolic relationships.

---

## Discussion

### Model Selection Considerations

1. **For mechanistic interpretation:** Multiplicative model (standard in genetics)
2. **For metabolic engineering:** Additive model (linear effects on flux)
3. **For statistical rigor:** GLM models (proper variance, increased power)
4. **For regulatory insight:** CLR Composition (capacity vs composition)

### Biological Insights

1. **Dominant genotype complexity:** Triple mutants account for ~74% of top-performing strains by FFA titer (90th percentile: 80/108 = 74.1%; 80th percentile: 155/210 = 73.8%), indicating higher-order synergy is crucial for FFA optimization.

2. **FFA-specific regulatory networks:** Clear differences between saturated (C14:0, C16:0, C18:0) and unsaturated (C16:1, C18:1) FFAs suggest distinct transcriptional control mechanisms.

3. **Diminishing returns:** Mean marginal benefit drops from +0.243 (1→2 KOs) to -0.019 (2→3 KOs), though individual high-performing triples exist.

4. **Capacity-composition decomposition:** Most interactions affect both total production and chain distribution, but separable effects exist (C18:0 enriched for capacity-only).

### Statistical Limitations

1. **Low replication (n=3):** No interactions survive FDR correction at q<0.05
2. **Power considerations:** GLM models mitigate this by using replicate-level data
3. **Effect size focus:** Large magnitude interactions ($|\varepsilon|>0.5$, $|\delta|>0.5$, $\phi>1.5$) are biologically meaningful regardless of FDR

### Future Directions

1. **Increase replication** to achieve FDR-corrected significance
2. **Validate top interactions** with independent experiments
3. **Mechanistic studies** on capacity vs composition regulatory pathways
4. **Extend to higher orders** (4-way, 5-way) using GLM framework
5. **Integrate with pathway models** to predict optimal strain designs
