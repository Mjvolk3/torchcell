---
id: s9srau7td604t3p7fob9tgp
title: case-study-ngProduction23butanediolSaccharomyces2012-slide
desc: ''
updated: 1768410564014
created: 1768410415284
---
## Case Study: Predicting Triple Knockout Fitness

**Reference:** Ng et al. 2012, *Microbial Cell Factories* — ADH/ALD knockouts for 2,3-butanediol production

***

### 1. The Puzzle

| Genotype                                 | Biomass (% WT)       |
|------------------------------------------|----------------------|
| $adh1\Delta$                             | 58%                  |
| $adh1\Delta \; adh5\Delta$               | **47%** $\downarrow$ |
| $adh1\Delta \; adh3\Delta \; adh5\Delta$ | **67%** $\uparrow$   |

**Question:** Why does adding a third deletion *increase* fitness?

***

### 2. The Pairwise Interactions (Costanzo2016)

| Pair      | $\varepsilon$ (epistasis)           |
|-----------|-------------------------------------|
| ADH1-ADH5 | $\mathbf{-0.093}$ (strong negative) |
| ADH1-ADH3 | $-0.015$ (neutral)                  |
| ADH3-ADH5 | $\mathbf{+0.020}$ (slight positive) |

***

### 3. The Answer

t
**$adh1\Delta \; adh5\Delta$ is the worst** because ADH1-ADH5 has strong negative epistasis ($\varepsilon = -0.093$). These are paralogs — deleting both removes redundancy.

**Adding $adh3\Delta$ introduces two new pairwise terms:**

| New interaction | $\varepsilon$     |
|-----------------|-------------------|
| ADH1-ADH3       | $-0.015$          |
| ADH3-ADH5       | $+0.020$          |
| **Net**         | $\mathbf{+0.005}$ |

The **positive** ADH3-ADH5 epistasis partially compensates for the existing damage.

***

### 4. The Quantitative Prediction

**Trigenic interaction definition:**

$$\tau_{ijk} = f_{ijk} - \underbrace{(f_{ij}f_k + f_{ik}f_j + f_{jk}f_i)}_{\text{pairwise contributions}} + \underbrace{2f_i f_j f_k}_{\text{overcounting correction}}$$

**Rearranging to predict $f_{ijk}$ (assuming $\tau = 0$):**

$$f_{ijk}^{\text{exp}} = \underbrace{f_{ij}f_k}_{\substack{\text{double 1-3} \\ \times \text{ single 5}}} + \underbrace{f_{ik}f_j}_{\substack{\text{double 1-5} \\ \times \text{ single 3}}} + \underbrace{f_{jk}f_i}_{\substack{\text{double 3-5} \\ \times \text{ single 1}}} - \underbrace{2f_i f_j f_k}_{\text{correction}}$$

**Input data (Costanzo2016):**

| Singles       | Doubles          |
|---------------|------------------|
| $f_1 = 1.005$ | $f_{13} = 0.433$ |
| $f_3 = 0.445$ | $f_{15} = 0.970$ |
| $f_5 = 1.058$ | $f_{35} = 0.491$ |

**Calculation:**

| Term                   | Calculation                                | Value            |
|------------------------|--------------------------------------------|------------------|
| $f_{13} \times f_5$    | $0.433 \times 1.058$                       | $0.458$          |
| $f_{15} \times f_3$    | $0.970 \times 0.445$                       | $0.432$          |
| $f_{35} \times f_1$    | $0.491 \times 1.005$                       | $0.493$          |
| $2 f_1 f_3 f_5$        | $2 \times 1.005 \times 0.445 \times 1.058$ | $0.947$          |
| **Expected $f_{135}$** | $0.458 + 0.432 + 0.493 - 0.947$            | $\mathbf{0.436}$ |

**Model prediction:**

|                        | Value            |
|------------------------|------------------|
| From singles + doubles | $0.436$          |
| Model $\tau$           | $+0.009$         |
| **Predicted**          | $\mathbf{0.445}$ |
| Observed (Ng et al.)   | $0.667$          |

***

### 5. Key Insights

**Model direction is correct:** $\tau > 0$ predicts triple fitness exceeds expectation — consistent with observed recovery.

**Magnitude gap explained by ADH3's mitochondrial role:**

| Gene | Location         | Costanzo (aerobic)     | Ng (microaerobic) |
|------|------------------|------------------------|-------------------|
| ADH1 | Cytosol          | $f_1 = 1.005$          | Relevant          |
| ADH5 | Cytosol          | $f_5 = 1.058$          | Relevant          |
| ADH3 | **Mitochondria** | $f_3 = 0.445$ (severe) | **Less relevant** |

Under **microaerobic** conditions, mitochondrial respiration is suppressed $\Rightarrow$ ADH3's fitness penalty ($f_3 = 0.445$) is **overstated** by aerobic SGA data.

$$\underbrace{f_{3}^{\text{SGA}} = 0.445}_{\text{aerobic: mito matters}} \not\approx \underbrace{f_{3}^{\text{ferm}}}_{\text{microaerobic: mito suppressed}}$$

This explains why observed ($0.667$) $>$ predicted ($0.445$): the ADH3 penalty doesn't fully apply.
