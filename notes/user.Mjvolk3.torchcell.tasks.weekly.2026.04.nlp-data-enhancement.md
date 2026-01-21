---
id: k3ees7my5acbkvuhnyc9g1o
title: nlp-data-enhancement
desc: ''
updated: 1768946915344
created: 1768945362675
---

## 2026.01.20 - Why Extract Dataset Fields from Paper Text?

### The Problem: Incomplete Data Tables

Scientific datasets often provide summary statistics (means, standard deviations) but omit crucial metadata required for proper statistical inference. This metadata exists in the paper's methods sections but is not encoded in the downloadable data files.

**Example: Standard Error vs Standard Deviation**

Consider the Costanzo 2016 dataset ([[costanzo2016|torchcell.datasets.scerevisiae.costanzo2016]]):

**What's in the data table:**

```
Query Strain ID    | Double mutant fitness | Double mutant fitness standard deviation
YBR188C_sn1898     | 0.5779               | 0.2388
```

**What's missing:** Sample size (n)

**Why it matters:**

- Standard Deviation (SD): Measures spread of data → `SD = 0.2388`
- Standard Error (SE): Measures uncertainty of mean → `SE = SD / √n`
- Without n, we cannot compute SE
- SE is essential for statistical comparisons and hypothesis testing

**Where n is found:** Paper methods section

> "All fitness measurements represent the mean of at least 2 independent measurements"

This single sentence transforms our dataset from having only SD to having both SD and SE, enabling proper statistical inference.

### When Text Extraction is Necessary

**Required when:**

1. **Derived statistics need computation:** SE, confidence intervals, effect sizes
2. **Metadata defines data semantics:** What does "replicate" mean? Technical or biological?
3. **Experimental conditions vary:** Do all measurements have same n, or does it vary by condition?
4. **Data validation requires context:** How were outliers handled? What thresholds were applied?

**Schema implications:**

In [[schema.py|torchcell.datamodels.schema#fitnessphenotype]]:

```python
class FitnessPhenotype(Phenotype, ModelStrict):
    # From data table:
    fitness: float
    fitness_std: float | None

    # From paper text:
    n_samples: int | None  # ← Extracted from methods

    # Computed from both:
    fitness_se: float | None  # ← fitness_std / sqrt(n_samples)
```

### Related Files

- **Implementation plan:** [[fitness-interaction-n_samples.wip|user.Mjvolk3.torchcell.tasks.weekly.2026.04.fitness-interaction-n_samples.wip]]
- **Extraction methodology:** [[nlp-data-enhancement.sop|user.Mjvolk3.torchcell.tasks.weekly.2026.04.nlp-data-enhancement.sop]]
- **Target datasets:**
  - [[costanzo2016|torchcell.datasets.scerevisiae.costanzo2016]]
  - [[kuzmin2018|torchcell.datasets.scerevisiae.kuzmin2018]]
  - [[kuzmin2020|torchcell.datasets.scerevisiae.kuzmin2020]]

### Why This Matters for TorchCell

**Current issue:** When [[Neo4jCellDataset|torchcell.datasets.neo4j_cell#neo4jcelldataset]] applies deduplication and aggregation, we combine measurements that may differ from the original data used to compute gene interactions. Without SE, we cannot determine if differences are statistically meaningful.

**Future capability:** With n_samples tracked, we can:

- Validate that aggregated fitness values still produce consistent gene interactions
- Quantify uncertainty propagation through the analysis pipeline
- Make statistically principled decisions about when to trust derived metrics

**See:** [[fitness-interaction-n_samples|user.Mjvolk3.torchcell.tasks.weekly.2026.03.fitness-interaction-n_samples]] for full context
