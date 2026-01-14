---
id: w5l3fvagp3gvrd43uvdg2jg
title: Adh_ald_costanzo2016_lookup
desc: ''
updated: 1768336804247
created: 1768336804247
---

## Costanzo2016 Lookup for ADH/ALD Genes

Retrieves single mutant fitness (SMF) and digenic interaction (ε) values for genes relevant to 2,3-butanediol production case study.

### Single Mutant Fitness (30°C)

| Gene | Systematic | SMF   | Interpretation |
|------|------------|-------|----------------|
| ADH1 | YOL086C    | 1.005 | ~neutral       |
| ADH3 | YMR083W    | 0.445 | **severe**     |
| ADH5 | YBR145W    | 1.058 | ~neutral       |
| ALD6 | YPL061W    | 0.932 | mild cost      |

### Digenic Interactions (30°C)

| Pair      | ε         | Interpretation |
|-----------|-----------|----------------|
| ADH1-ADH3 | -0.015    | Neutral        |
| ADH1-ADH5 | **-0.093**| Strong NEG     |
| ADH1-ALD6 | N/A       | NOT FOUND      |
| ADH3-ADH5 | +0.020    | Neutral        |
| ADH3-ALD6 | -0.059    | Neutral        |
| ADH5-ALD6 | -0.058    | Neutral        |

**Key:** ADH1-ADH5 is the **only strong pairwise interaction** — paralogs from whole-genome duplication.

### Related

- Inference script: [[experiments.010-kuzmin-tmi.scripts.trigenic_interaction_adh1_adh3_adh5]]
