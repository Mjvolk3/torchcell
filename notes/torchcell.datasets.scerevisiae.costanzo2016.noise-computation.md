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

## 2026.07.03 - EXACT method sourced from Baryshnikova 2010 SI (corrects earlier assumptions)

Sourced the actual scoring method (Costanzo 2016 uses the Baryshnikova 2010
pipeline). Provenance:

- citation_key: `baryshnikovaQuantitativeAnalysisFitness2010` (DOI 10.1038/nmeth.1534)
- SI PDF retrieved via scriptable Springer ESM URL:
  `https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1534/MediaObjects/41592_2010_BFnmeth1534_MOESM167_ESM.pdf`
- sha256: `b7ec4d0603aed5346fdd5043738358fdc5cc09084854391dc62ed6a35354f3ad`
- Method in Supplementary Note 1 (pp. 28-30). Exact p-value test statistic is in
  "Supplementary Software 1" (Matlab), NOT yet retrieved (in Zotero).

Exact equations (verbatim):

- Eq. 1 / model: `f_ij = f_i*f_j + eps_ij`  (eps = observed - expected DMF, multiplicative).
- Eq. 13 interaction estimate: `eps_ij = I_ij = (1/N_ij) * sum_k R_ijk`, averaged
  over `N_ij` replicate colonies = **4-8** ("typically four per screen with up to
  two screens") and VARIES per interaction.
- Eq. 14 LOCAL s.d. (= our "Double mutant fitness standard deviation" column):
  `sigma_Iij = sqrt( sum_k (R_ijk - I_ij)^2 / (N_ij - 1) )`.
  Authors EXPLICITLY warn this "can dramatically underestimate the true variance"
  because adjacent replicate colonies are NOT independent.
- Eq. 16 EXPECTED (pooled) variance used for confidence:
  `C_ij ~ alpha*f_i*f_j*e^X`, `X ~ N(0, sigma_i^2 + sigma_j^2)`;
  `sigma_ij_expected = exp( sqrt(sigma_i^2 + sigma_j^2) )`, where sigma_i^2 =
  array-strain variance from WT control screens, sigma_j^2 = query-strain variance
  pooled within-array.
- SMF variance (Eq. 11 area): "bootstrap sampling of the median" (n=800) -> a
  bootstrapped SEM. Our SMF file's `stddev` column is the SD, NOT this SEM.

Consequences (corrections to the 2026.01 analysis above):

- The interaction p-value is NOT computed from the local DMF colony s.d.
  (`sigma_Iij`); it uses the POOLED expected variance `sigma_i^2 + sigma_j^2`
  estimated from control screens. That value is NOT in our summary data files, so
  the exact p-value CANNOT be reproduced from our columns alone (empirical
  best-fit with the local s.d. plateaus at corr 0.95, no clean scalar).
- `68/1400` (17x4, 350x4) is doubly wrong: (1) the p-value doesn't use colony
  counts that way, and (2) SMF error is a bootstrapped SEM, small, not the file SD.
- Ontology mapping: DMF `fitness_uncertainty` = `sigma_Iij` -> type `sample_sd`,
  n_samples = N_ij (4-8, varies, NOT in our data), unit `colony` -- BUT this
  underestimates; the ML-honest uncertainty is `sigma_ij_expected` which we lack.
  SMF -> type `standard_error` (bootstrap SEM ~ SD/sqrt(n_screens)).

## 2026.07.03 - EXACT p-value formula from the Matlab source (Supplementary Software 1)

Retrieved Supplementary Software 1 (Matlab) -- manually downloaded by user
(nature.com auth-gated), stored at
`$DATA_ROOT/torchcell-library/baryshnikovaQuantitativeAnalysisFitness2010/software/41592_2010_BFnmeth1534_MOESM171_ESM.zip`
sha256 `c667d4f5b56f23e1ee59aabaeb1712bc6d2ae8a92ed8458ef3b866d89b1e496e`.

Exact p-value (`IO/output_interaction_data.m:15`), verbatim:

```matlab
pvals = sqrt( normcdf(-abs(escores./escores_std)) .* ...
              normcdf(-abs(log((background_mean + escores)./background_mean) ./ log(background_std))) );
```

Decode (variables from compute_sgascore.m):

- `escores` = eps (interaction score I_ij); `escores_std` = sigma_Iij (LOCAL s.d.,
  Eq 14) = our published "Double mutant fitness standard deviation" column.
- `background_mean` = exp(amat_mean) = expected DMF; `background_mean + escores` =
  actual DMF. So the ratio = actual/expected.
- `background_std` = exp(sqrt(amat + qmat)) = sigma_ij_expected (Eq 16); amat =
  array-strain variance, qmat = query-strain variance, both from WT control screens.
  So log(background_std) = sqrt(amat + qmat).

Therefore:

```
p = sqrt( Phi(-|eps / sigma_local|) * Phi(-|log(actual/expected) / sqrt(amat+qmat)|) )
```

= geometric mean of two one-sided normal tail probabilities: (A) eps vs its local
colony s.d. (linear), (B) log-fold actual-vs-expected DMF vs the pooled expected
s.d. (geometric/log).

CONCLUSION: `background_std` (pooled expected s.d.) is NOT written to the output
file (output columns: escore, escore_std, pval, smfit i/j +std, dm_expected,
dm_actual, dm_actual_std). It is an internal quantity requiring the raw colony data

- control screens. **Costanzo's exact p-value is therefore NOT reproducible from
the released summary data** -- only the raw SGA pipeline reproduces it. This closes
the p-value chase definitively.

## 2026.07.04 - SMF stddev IS a bootstrap SE (not a sample SD) -- resolves loader

**Question:** the fitness-uncertainty ontology needs to know what the Costanzo
"Single mutant fitness stddev" column IS statistically -- a sample SD (divide by
sqrt(n)) or an already-a-SE quantity (use as-is)? Pulled directly from the SOM.

**Provenance.** Source = Costanzo 2016 *Science* SOM, `SOM.pdf`
sha256 `79f38885e88495a83beee4e755237a8ce0d9eb29fc00c92134754a0640348c32`
(born-digital Word->PDF; read via `pdftotext -layout`, NOT MinerU -- there is no
Costanzo MinerU capture and pdftotext is the correct first-line tool for a
born-digital text layer). Section "Single mutant fitness standard [deviation]".
Reproducibly retrievable from the Boone lab mirror (Science.org 403s) --
see [[costanzo2016-som-retrieval-provenance]].

**Verbatim (SOM, SMF methods):**
> "Colony size measurements of SGA deletion and TS **array** mutant strains were
> based on an average of **350 replicate control screens** performed at 26degC or
> 30degC. Colony size measurements of SGA deletion and TS **query** mutant strains
> were based on an average of **17 replicate control screens** ... single mutant
> fitness [was estimated] as described previously (5) with the exception that
> **bootstrapped means, instead of medians, across replicates were used in
> variance estimation and final fitness values**."

**Answer: `bootstrap_se`.** The SMF stddev is the SD of the *bootstrapped mean*
fitness estimate -> it is already a standard error of the estimate. Per the
ontology `derive_se`: `bootstrap_se` is used AS-IS; you must NOT divide it by
sqrt(anything).

**The current loader is wrong on two counts** (`costanzo2016.py:370-373`):

```python
n_samples = N_SAMPLES_QUERY_SMF_TOTAL          # = 68 (17 screens x 4 colonies)
fitness_se_val = fitness_std_val / math.sqrt(n_samples)   # WRONG
```

1. It divides a bootstrap SE by sqrt(n) -- double-shrinking an already-SE quantity.
2. n=68 conflates 17 *screens* (the bootstrap resampling unit) with 4 colonies;
   colonies are pseudoreplicates. The resampling unit is the **screen** (17 query,
   350 array), `sample_unit = screen`.

**Loader fix (WS2 / propagation step 1):**

- **SMF:** `fitness_uncertainty = <stddev col>`, `fitness_uncertainty_type =
  bootstrap_se`, `fitness_se = fitness_uncertainty` (as-is). Record for provenance
  `n_samples = 17` (query) / `350` (array), `sample_unit = screen` -- these do NOT
  enter the SE for `bootstrap_se`.
- **DMF** is a DIFFERENT statistic: "Double mutant fitness standard deviation"
  (Data File S1) is a sample SD over the ~4 array colony replicates ->
  `uncertainty_type = sample_sd`, `sample_unit = colony`, `n_samples = <colony n>`,
  SE = SD/sqrt(n). This is why the ontology's per-record `uncertainty_type` is
  essential: SMF and DMF stddevs are not the same kind of number.

Note: Data File S1's 11 columns list SMF + Array SMF + DMF + "Double mutant fitness
standard deviation" -- there is NO SMF-stddev column in S1; the SMF stddev lives in
the strain-level SMF file, and it is the bootstrap SE described above.

## 2026.07.04 - Kuzmin 2018/2020 n_samples (same SGA pipeline)

Kuzmin uses the same Boone-lab pipeline. Sourced from the Kuzmin 2018 MinerU SI
`kuzminSystematicAnalysisComplex2018/si/si1.md` (WS2 Kuzmin migration, PR #23):

- The loader stores the **"Combined mutant fitness standard deviation"** (Data S1
  col. 12) = the double/triple mutant measured in the interaction screen. Kuzmin's
  SI does NOT restate its type and defers to Baryshnikova 2010 (ref 8) -> it is a
  colony **sample_sd** (Eq. 14), `sample_unit = colony`.
- **n_samples = 4** (revised from an initial 8). The exact per-record colony count
  is NOT in Data S1 (12 columns, no count/SE column -- verified by inspecting the
  raw TSV), so it is fixed by three converging lines:
  1. **Empirical back-solve against the reported P-value** (the "other provided
     statistic"): the single-term normal model `2*Phi(-|eps|/(sd/sqrt(n)))` over
     ~410k digenic records matches the reported P-value **median (0.358) best at
     n=4 (0.377)**; n=8 overshoots badly (0.211), n=6 (0.279). Spearman(p_pred,
     p_reported) = **0.985** confirms the col-12 SD drives the p-value ranking.
     Exact recovery is precluded by the unpublished pooled-background term in the
     Baryshnikova p-value (frac(p<0.05) can't be matched by any single-term n), so
     this fixes the central estimate, not an exact per-record n.
  2. **Conservative lower-end** of the Baryshnikova 4-8 range (typically 4/screen).
  3. **Consistency with Costanzo DMF (n=4).**
- The "12-24 colony measurements" in si1.md L59 is the QUERY fitness (col 9,
  bootstrap-derived), a DIFFERENT column whose std the loader does not store -- do
  not confuse it with the stored col-12 combined-mutant SD.
- SMF (single mutant): no reported std -> no uncertainty.

Policy precedent (unexplained range -> value): back-solve from another provided
statistic first; if precluded, take the conservative lower-end. Here both agree on
4. See [[torchcell.datasets.scerevisiae.kuzmin2018]].

## 2026.07.22 - Kuzmin SMF scoring: bootstrapped MEANS (verbatim), array SMF = Costanzo

Confirmed from the Kuzmin 2018 SI (`kuzminSystematicAnalysisComplex2018/si/si1.md`),
sourced for the 019 CRISPR-fitness-assay benchmark
([[experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay]]):

> "The quantitative scoring method employed for single and double mutant fitness
> estimation was described previously (Baryshnikova 2010), with the exception that
> **bootstrapped means, instead of medians, across replicates were used in variance
> estimation and final fitness values.**"

So Kuzmin -- like Costanzo 2016 -- bootstraps **means**, deviating from Baryshnikova
2010's bootstrapped **medians**. All three use the same bootstrap-over-replicate-screens
machinery; only mean-vs-median differs. Provenance facts:

- **Kuzmin QUERY SMF:** high-density array screened in triplicate -> 6 replicate
  screens, 12-24 colony measurements per estimate. Resampling unit = the screen.
- **Kuzmin ARRAY SMF:** "Estimates of the mean single mutant fitness of each array
  strain were taken from a previous study (7)" = **Costanzo 2016** -> for array genes
  Kuzmin's SMF value is NOT independent of Costanzo's (it *is* Costanzo's number).
- **No per-strain Kuzmin SMF SD** is released (only the combined-mutant col-12 sample
  SD, Baryshnikova Eq. 14, n~4) -- so a Kuzmin SMF-SD cannot be compared to ours.
- Kuzmin-vs-Costanzo SMF agreement r ~ 0.63 (their Fig. S1C, n=331); the field's
  cross-lab SMF-vs-SMF correlation is ~0.5-0.63.

Bottom line for the SD ontology: Kuzmin SMF uncertainty, where it exists, is a
bootstrap SE (same as Costanzo SMF), not a colony sample SD.
