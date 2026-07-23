---
id: gyeiwy7g4xt6udclcoe3v78
title: Abstract
desc: ''
updated: 1784768208466
created: 1778445248212
---
## Title

**Accelerating Metabolic Engineering through Multimodal Phenotype Prediction with TorchCell and the Cell Graph Transformer**

## Abstract

While deep learning has driven major advances in predicting protein structure and function, comparable progress in systems biology and metabolic engineering has been limited by phenotypic data scattered across heterogeneous studies, whose data structures cannot be easily unified without a shared experimental ontology.
To bridge this gap, we developed TorchCell, an open-source framework that aggregates heterogeneous metabolic-engineering data and predicts multimodal cellular phenotypes using deep learning.
Literature and public datasets are annotated using TorchCell's shared experimental ontology and ingested into a Neo4j knowledge graph hosted at NCSA, unifying genotype, phenotype, and environmental records.
From this knowledge graph, dataloaders construct deep-learning-ready datasets in which each engineered strain is encoded as a perturbation operator over a shared wildtype reference cell that combines genome sequence, gene networks, metabolism, and environment, enabling millions of strains to be represented at scale.
On these datasets we developed the Cell Graph Transformer (CGT), a virtual cell architecture in which biological interaction graphs constrain multi-head attention through a loss that aligns attention weights to known network priors, and an equivariant perturbation operator transforms the cell embedding into the perturbed strain state before multitask heads decode phenotype predictions.
In Saccharomyces cerevisiae, CGT predicts trigenic gene-gene interactions with Pearson r = 0.454 ± 0.004 and Spearman ρ = 0.421 ± 0.003, outperforming the deep-learning epistatic interaction models DANGO (r = 0.367 ± 0.0004) and DCell (r = 0.157 ± 0.009), and the mechanistic Yeast9 flux balance analysis, which recovers no epistatic signal (r = 0.0006).
The same architecture also predicts expression under single and double knockouts (Pearson r = 0.543 ± 0.023) and cell morphology under single knockouts (Pearson r = 0.619 ± 0.037), demonstrating that the same latent cell embedding generalizes across qualitatively different phenotype classes relevant to industrial strain design.
We have applied CGT to recommend gene deletions for improved production of β-carotene and betaxanthin in S. cerevisiae.
Unlike most DBTL recommendation tools, which do not represent the strain itself, TorchCell pairs with autonomous platforms such as UIUC's iBioFoundry to enable iterative AI-guided strain engineering for renewable chemicals, fuels, and pigments.

## 2026.06.04 - Error-Bar Provenance Caveat

The trigenic τ comparison reported above (CGT 0.454 ± 0.006; DANGO 0.367 ± 0.0004; DCell 0.157 ± 0.009; Yeast9 0.0006) reuses the values plotted by `experiments/010-kuzmin-tmi/scripts/trigenic_tau_model_comparison.py`. Full provenance and the bar chart are documented in [[experiments.010-kuzmin-tmi.scripts.trigenic_tau_model_comparison]].

**Before submitting**, note that the four uncertainties are not the same statistical quantity:

- **DANGO (± 0.0004)** and **DCell (± 0.009)** are **SEM** computed from 3 replicate Pearson values each.
- **TorchCell (± 0.006)** is a **reported SE** taken at face value — there is no raw replicate array; the plot reconstructs synthetic values around the mean.
- **Yeast9 (0.0006)** is deterministic (FBA), so it has no error by nature.

Confirm whether the TorchCell `± 0.006` is an SE/SEM (comparable to DANGO/DCell) or an SD (~√3× too large relative to them) so all reported error bars represent the same quantity.

## 2026.07.21 - Error-Bar Caveat RESOLVED (WS14)

The caveat above is resolved; the main-text line now reads **CGT r = 0.454 ± 0.004, ρ = 0.421 ± 0.003**. All four Pearson error bars are now the **same statistic — SEM** (`std(ddof=1)/√n`), computed the same way for every model.

- **TorchCell was NOT missing replicates.** It has three real replicate Pearson values from the wandb `inf_1` runs (0.462, 0.452, 0.447; see [[experiments.010-kuzmin-tmi.performance-diff-010-009]]). The old `± 0.006` was the **population SD** (`np.std(ddof=0)`) of those three values — a *different* statistic than DANGO/DCell's SEM, not an SE and not the sample SD. Its SEM is 0.0044, so `± 0.006 → ± 0.004`.
- The plot's earlier synthetic reconstruction is removed; `trigenic_tau_model_comparison.py` now reads the three real replicates and computes SEM uniformly (emitting both `.png` and `.svg`).
- **Only the CGT whisker changed** (Pearson `± 0.006 → ± 0.004`; Spearman `± 0.004 → ± 0.003`). Every mean is unchanged — CGT still rounds to r = 0.454, ρ = 0.421. DANGO (± 0.0004), DCell (± 0.009), and Yeast9 (0.0006, legitimately zero) are unchanged.
