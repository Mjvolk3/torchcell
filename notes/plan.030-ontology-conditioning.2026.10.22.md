---
id: fe9qpwmcwl29xk4imuu41aa
title: '22'
desc: ''
updated: 1792688088376
created: 1792688088377
---

## 2026.10.22 - Conditioning on the experiment's ontology, not its identity

Status: plan, nothing below has run. Supersedes the identity token of [[plan.030-per-entry-dataset-token.2026.09.25]] as the conditioning design; keeps its per-entry training rows and its metrics.

### What is wrong with the current encoding

Every entry row carries the index of its source dataset's NAME in a 15-name vocabulary, and both readouts receive a free 8-vector per name (`CellGraphTransformer.dataset_token`, mode readout). Measured on the Q validation set (2026.10.11): Kuzmin 2018 triples average tau -0.046 and 2020 triples +0.004, a per-screen-mean predictor scores Pearson 0.34 to 0.38, the token learns that mean in epoch 0, and swapping the two screens' tokens on the current runs gives Pearson -0.30 (`val/crosstoken`). The affine mode built on 2026.10.21 (`y = scale[d] * y + shift[d]`, branch `feat/030-token-affine`) constrains HOW the name acts, not WHETHER the model knows it; its shift is the per-screen mean by construction. Neither answers the four balances below.

The model reads a genotype as gene indices only (`perturbation_indices`; no marker, no allele class, no query or array role), so the screen is readable from the genotype only through which genes a screen used. That route is accepted.

### The four balances (owner, 2026.10.22)

1. Generality. Inference on a new experiment must not require picking the most similar past experiment. The user states what differs through the ontology annotation, and the model conditions on that.
2. Shared decoders. One fitness decoder and one interaction decoder for every source, conditioned on where and what the experiment is; how much conditioning they need is open.
3. One scale. Fitness is relative to wild type and interactions are defined on fitness, across every source. Per-dataset normalization would erase that; the normalization constants stay global.
4. All the data. Several sources can measure the same genotype under the same ontology (two Dmf screens, two Tmf screens). Those rows must all train, and the model needs a stated rule for what one prediction means against several measurements. The mean is not loved; the 025 label merge showed that averaging across sources breaks the interaction identity (tau = f_ijk - f_ij f_k - eps_ik - eps_jk is exact within a screen, 0.985, and not across the merge, 0.538).

### Design

**D1. Experiment profile in place of the name.** A pydantic `ExperimentProfile` built from the record, never from the dataset class: `measurement` (fitness, interaction, essentiality, synthetic lethality), `interaction_order` (0 for fitness, 2 or 3), `assay` (SGA colony size, curated literature, SGD annotation), `media` (the schema's `Media` name: SGA double-mutant selection, SGA triple-mutant selection, YPD, ...), `temperature_c` (number), `selection_markers` present in the environment (what the media selects for, not which strain carries which). The build's 15 sources collapse to the DISTINCT profiles; Kuzmin 2018 and 2020 rows of one measurement type share a profile by construction, Costanzo rows differ from them by media and temperature, the curated sources by assay. A script `experiment_profiles_030.py` writes `results/experiment_profiles_030.json` (profile table, the source-to-profile map, the row counts per profile) and the dataset emits `phenotype_profile_indices` beside `phenotype_dataset_indices`, which stays for metrics only (per-source Pearson, cross-token) and never reaches the model.

**D2. Compositional encoding.** Not a one-hot over profiles (that is a name again, one level up). Each field is encoded on its own: categorical fields as one-hot over a controlled vocabulary that the schema already defines, temperature as a number, then one small linear map to the token. A new experiment with a new combination of known field values gets a token without retraining; a new field value is the only thing that needs a vocabulary extension. `mode: readout` and `mode: affine` both take this token (affine: scale and shift become linear functions of the token instead of a table), so the decoder question (balance 2) is an ablation, not a design decision: none, affine, readout, in that order of capacity.

**D3. Duplicates stay as rows; the loss states the rule.** Every stored measurement remains a training row (the per-entry path). With identical profiles the model has one prediction for the genotype, and the loss decides what it estimates: squared error gives the mean, absolute error the median, a Gaussian head (`param_dim 2`, already in the readouts) the mean with a per-genotype spread that absorbs replicate noise. The arms are the three losses on the same rows. The score that decides is not pooled Pearson: it is the model's single value against the replicate sets the data already holds, the 483 cross-year triples (reliability 0.89) and the 11,013 same-strain pairs across the two 2020 screens (reliability 0.44), read as "is the prediction inside the replicate spread".

**D4. Headline metric within screen.** `val/token/<source>/gene_interaction/Pearson` per source and a pooled Pearson after per-source centering become the headline; the raw pooled number is reported beside them with the floor stated. This is the open item from the weekly note and is independent of D1 to D3.

### Arms and order

| arm | conditioning | loss | question |
|---|---|---|---|
| A0 | none (token off) | MSE | the floor without any experiment information |
| A1 | profile, affine | MSE | does level and gain per experiment class recover what the name gave |
| A2 | profile, readout | MSE | does a free per-class readout add anything over A1 |
| A3 | profile, best of A1/A2 | Gaussian NLL | the duplicate rule as a distribution |
| A4 | profile, best of A1/A2 | L1 | the duplicate rule as a median |

A0 to A2 first, on the Q split (the between-screen floor is largest there), seed 1, one A100 each on mmli at batch 512, or 4 x A100 per arm on Delta when the queue allows. Pass condition for D1 before any arm is read: at epoch 0 the pooled val Pearson must sit at the within-screen level, not at the 0.37 floor, and `val/crosstoken` must be exactly the own-token read for the two Kuzmin screens (same profile, the swap is the identity). A3 and A4 follow on the winner.

### What is NOT in scope

Per-screen centering of labels (rejected: balance 3). Telling the model the screen by any route the user does not control at inference. Collapsing duplicate rows at the data level.

### Running jobs touched

IGB 2430655 (Q affine, identity) continues until A1 is ready, then is replaced; 2429088 (Q readout, identity) is the reference for the floor. GilaHyper stays a smoke lane.
