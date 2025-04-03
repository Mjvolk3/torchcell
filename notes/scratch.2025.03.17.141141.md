---
id: o9avjx8kzhragp419x8xw35
title: '141141'
desc: ''
updated: 1742238920779
created: 1742238704653
---
Here’s a focused **comparison of conventional Perturb-seq, cell-pooling, and guide-pooling**:

---

### 🧬 **Comparison Table**

| Feature                     | **Conventional Perturb-seq**                | **Cell-pooling**               | **Guide-pooling**                           |
|-----------------------------|---------------------------------------------|--------------------------------|---------------------------------------------|
| **Perturbation/cell**       | 1 (low MOI)                                 | 1                              | >1 (high MOI)                               |
| **Droplet contents**        | 1 cell per droplet                          | Multiple cells per droplet     | 1 cell with multiple guides                 |
| **Phenotype**               | Individual cell transcriptome               | Average transcriptome of cells | Composite perturbation in one cell          |
| **Infers interactions?**    | ❌                                           | ❌                              | ✅                                           |
| **Assay complexity**        | Baseline                                    | Requires droplet overloading   | Requires high MOI / sequential transduction |
| **Toxicity risk**           | Low (KO or KD)                              | Low                            | Moderate–High (esp. KO)                     |
| **Signal per droplet/cell** | Baseline                                    | High (more non-empty droplets) | High (more perturbations/cell)              |
| **Compression gain**        | —                                           | ~8× cost/channel               | ~10× cost/cell                              |
| **Scaling with design**     | Linear in $n$ (number unique perturbations) | Sublinear in $n$ via sparsity  | Sublinear in $n$, better with more guides   |
| **Best use case**           | High-fidelity single effects                | Cost-efficient large screens   | Learning interactions                       |

---

### 🔍 Conceptual Differences

|                      | Conventional        | Cell-pooling          | Guide-pooling                     |
|----------------------|---------------------|-----------------------|-----------------------------------|
| Design space         | One guide, one cell | One guide, many cells | Many guides, one cell             |
| Readout model        | Direct effect       | Mean of cell effects  | Sum in log-space of guide effects |
| Modeling requirement | None                | Sparse averaging      | Additive + sparse recovery        |

---

### 🧠 Experimental & Computational Implications

- **Conventional**: Gold standard, but costly; good for reference datasets.
- **Cell-pooling**: Maximizes droplet occupancy → best for maximizing information per sequencing dollar without needing interaction modeling.
- **Guide-pooling**: Best for uncovering genetic interactions; richer signal per cell, but requires careful modeling (e.g., FR-Perturb), and controls for non-additivity.

---

Let me know if you want a decision tree for choosing among them based on screen size, cost, and scientific goals.
