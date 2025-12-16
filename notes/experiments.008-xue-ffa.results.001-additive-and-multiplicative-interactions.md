---
id: bq0ewlit51ex8rkjvdan2nz
title: 001 Additive and Multiplicative Interactions
desc: ''
updated: 1759878843420
created: 1758337131967
---

## Additive vs Multiplicative Interaction Models: Conceptual Differences

### Overview

Genetic interaction models quantify how combinations of mutations deviate from expected phenotypes. The two primary models—additive and multiplicative—differ fundamentally in their null expectations and biological interpretations.

### Mathematical Formulations

#### Additive Model

**Digenic Interactions:**
$$\delta_{ij} = f_{ij} - \frac{f_i + f_j}{2}$$

**Trigenic Interactions:**
$$\sigma_{ijk} = f_{ijk} - \frac{f_{ij} + f_{ik} + f_{jk}}{3}$$

**General k-way:**
$$\zeta_S = f_S - \frac{1}{k}\sum_{i \in S} f_{S \setminus \{i\}}$$

#### Multiplicative Model

**Digenic Interactions:**
$$\varepsilon_{ij} = f_{ij} - f_i \times f_j$$

**Trigenic Interactions:**
$$\tau_{ijk} = f_{ijk} - f_{ij} \times f_k - f_{ik} \times f_j - f_{jk} \times f_i + 2 \times f_i \times f_j \times f_k$$

### Key Conceptual Differences

#### 1. Null Expectation

**Additive Model:**

- Expects the combined phenotype to equal the **average** of constituent phenotypes
- Assumes mutations contribute equally to the phenotype
- Natural for traits where effects combine linearly

**Multiplicative Model:**

- Expects the combined phenotype to equal the **product** of constituent phenotypes
- Assumes mutations act independently on different steps of a pathway
- Natural for fitness traits where effects compound

#### 2. Biological Interpretation

**Additive Model Questions:**

- "Does the double mutant perform better or worse than the average of the two single mutants?"
- "Does the triple mutant exceed the average performance of its constituent doubles?"
- Appropriate for metabolic flux, where pathways may have redundancy

**Multiplicative Model Questions:**

- "Do the mutations interact synergistically or antagonistically?"
- "Are the mutations in the same pathway (epistatic) or independent?"
- Standard for growth fitness studies in genetics

#### 3. Error Propagation Differences

**Additive Model:**

- Coefficients are **constants** (1/2 for digenic, 1/3 for trigenic)
- Error propagation: $\text{SE}(\delta_{ij}) = \sqrt{\text{SE}(f_{ij})^2 + \left(\frac{\text{SE}(f_i)}{2}\right)^2 + \left(\frac{\text{SE}(f_j)}{2}\right)^2}$
- Simpler error structure due to linear relationships

**Multiplicative Model:**

- Coefficients **depend on fitness values** (e.g., $f_j$ multiplies $\text{SE}(f_i)$)
- Error propagation: $\text{SE}(\varepsilon_{ij}) = \sqrt{\text{SE}(f_{ij})^2 + (f_j \times \text{SE}(f_i))^2 + (f_i \times \text{SE}(f_j))^2}$
- More complex error structure due to product terms

#### 4. Interaction Sign Interpretation

**Positive Interactions ($\delta > 0$ or $\varepsilon > 0$):**

- **Additive:** Combined effect exceeds average → Synergy or suppression
- **Multiplicative:** Combined effect exceeds product → Alleviating interaction

**Negative Interactions ($\delta < 0$ or $\varepsilon < 0$):**

- **Additive:** Combined effect below average → Interference
- **Multiplicative:** Combined effect below product → Synthetic sick/lethal

#### 5. Model Selection Criteria

**Choose Additive When:**

- Effects are expected to combine linearly
- Studying metabolic flux or enzyme activity
- Mutations affect parallel pathways
- Baseline performance varies widely

**Choose Multiplicative When:**

- Effects are expected to compound
- Studying organism fitness or growth
- Mutations affect serial pathway steps
- Following established genetics conventions

### Application to FFA Production Study

In the context of free fatty acid (FFA) production with transcription factor deletions:

#### Why Both Models Matter

1. **Metabolic Context (Favors Additive):**
   - FFA production is a metabolic flux measurement
   - TFs may regulate parallel biosynthetic pathways
   - Effects on enzyme expression may combine linearly

2. **Genetic Context (Favors Multiplicative):**
   - Standard in genetic interaction studies
   - TFs may act in regulatory cascades
   - Allows comparison with growth fitness studies

#### Model Crossover Behavior

The models can give opposite conclusions:

- At low fitness (f < 1): Multiplicative model predicts stronger negative interactions
- At high fitness (f > 1): Additive model may show negative interaction where multiplicative shows positive
- Crossover point depends on the specific fitness values

#### Statistical Considerations

1. **Sample Size Requirements:**
   - Multiplicative model: Higher variance at extreme fitness values
   - Additive model: More uniform variance across fitness range

2. **Multiple Testing:**
   - Both models test same number of interactions
   - FDR correction equally stringent
   - Biological effect sizes may differ between models

### Recommendations

1. **Report Both Models:** Given the metabolic nature of FFA production but genetic nature of TF deletions, both models provide valuable insights.

2. **Focus on Consistency:** Interactions significant in both models are most robust.

3. **Consider Magnitude:** Large effect sizes (|δ| > 0.2 or |ε| > 0.2) are biologically meaningful regardless of statistical significance.

4. **Validate Biologically:** Key interactions should be validated experimentally regardless of model choice.

### Conclusion

The additive and multiplicative models represent different biological assumptions about how genetic perturbations combine. The additive model asks whether combinations perform better than their average, while the multiplicative model tests for deviations from independent effects. For metabolic engineering applications like FFA production, both perspectives offer complementary insights into the genetic architecture of the trait.
