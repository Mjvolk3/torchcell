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

**D1. Experiment profile in place of the name.** A pydantic `ExperimentProfile` built from the record, never from the dataset class: `measurement` (fitness, interaction, essentiality, synthetic lethality), `interaction_order` (0 for fitness, 2 or 3), `assay` (SGA colony size, curated literature, SGD annotation), `media` (the schema's `Media` name: SGA double-mutant selection, SGA triple-mutant selection, YPD, ...), `temperature_c` (number), `selection_markers` present in the environment (what the media selects for, not which strain carries which). The build's 15 sources collapse to the DISTINCT profiles; Kuzmin 2018 and 2020 rows of one measurement type share a profile by construction. Costanzo rows differ from Kuzmin rows in the record by exactly one media component, the uracil dropout the Kuzmin plates need to select the KlURA3-marked third mutation (same SD-MSG base, same four drugs, same His/Arg/Lys dropouts), and by temperature for the Costanzo rows scored at 30 C (Kuzmin stores 26 C throughout). Whether the missing uracil changes a colony-size fitness is not measured and is not assumed; the dropout is kept as a true property of the plate and noted as the one design-linked field that survives (it says three markers were selected, not which screen). The curated sources differ by assay. A script `experiment_profiles_030.py` writes `results/experiment_profiles_030.json` (profile table, the source-to-profile map, the row counts per profile) and the dataset emits `phenotype_profile_indices` beside `phenotype_dataset_indices`, which stays for metrics only (per-source Pearson, cross-token) and never reaches the model.

**D2. Compositional encoding.** Not a one-hot over profiles (that is a name again, one level up). Each field is encoded on its own: categorical fields as one-hot over a controlled vocabulary that the schema already defines, media by COMPOSITION (base medium, the set of dropouts, the set of additives, each a bit) rather than by media name, so the two SGA media differ in one bit and not as two unrelated categories, temperature as a number, then one small linear map to the token. A new experiment with a new combination of known field values gets a token without retraining; a new field value is the only thing that needs a vocabulary extension. `mode: readout` and `mode: affine` both take this token (affine: scale and shift become linear functions of the token instead of a table), so the decoder question (balance 2) is an ablation, not a design decision: none, affine, readout, in that order of capacity.

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

## 2026.10.22 - D2 revised: propagate the record up the schema DAG (DCell for the experiment)

Owner's direction: the annotation of an experiment to the ontology is a DAG; encode the experiment the way DCell encodes a genotype, leaves annotated and states propagated up to the root, so the encoder is general over anything the ontology can capture; and the phenotype leaf dictates the size of the final linear projection.

**The DAG is the schema, read live.** `torchcell.paper.ontology_graph.build_ontology_graph` introspects the pydantic models into classes with their inheritance parent, their fields and their composition references (the pre-commit ontology figure regenerates from it, so it cannot drift). Lanes: genotype (GenePerturbation, four inheritance levels down to SgaKanMxDeletion / SgaTsAllele / SgaDamp), environment (Environment, Media, MediaComponent, Temperature, EnvironmentPerturbation and its three subclasses, the compound identity records), phenotype (Phenotype and fifteen leaves: Fitness, GeneInteraction, GeneEssentiality, SyntheticLethality, SyntheticRescue, CalMorph, three expression leaves, VisualScore, Metabolite, ProteinAbundance, EnvironmentResponse), experiment (Experiment and its fifteen leaves, ExperimentReference and its fifteen leaves). The inheritance side is shallow (one to four levels); the composition side is what gives the experiment its structure: Experiment has Genotype, Environment and Phenotype; Environment has Media and Temperature; Media has MediaComponents, each a compound with a role.

**Nodes and leaves.** Nodes = the schema classes plus value nodes: enum members (MediaComponentRole, TemperatureUnit), the controlled compound vocabulary (ChEBI-grounded through `compound_identity`), and numeric leaves (temperature value, concentrations) carried as values rather than bits. A record's annotation is the set of nodes it instantiates: its Experiment leaf, its Phenotype leaf, Environment, Media, one node per media component with its role (a dropout is a component node with the dropout role, so "lacks uracil" is a leaf, not the absence of one), Temperature with its value, and its ExperimentReference subtree (the reference genotype, environment and phenotype the measurement is relative to; that is the null model of the screen, part of what the experiment IS and not its identity). Excluded by construction: every provenance field (dataset class, citation key, publication, any id). The genotype lane is NOT part of the experiment token; genes go through the gene encoder as now.

**Propagation.** Bottom-up in topological order along inheritance and composition edges, DCell style: each node keeps a small state `s_v = tanh(W_v [ concat(s_children) ; x_v ])` with `x_v` the node's own annotation features (present bit, value), parameters per node (the graph is small, under two hundred nodes, so this is cheap) or shared per class family where a node has many instances (media components). The root Experiment state `e` is the experiment token. It replaces the flat field-wise encoder of D2; everything downstream (affine, readout, the arms A0 to A4, the pass condition) is unchanged. Because the leaves are schema values, an experiment never seen in training is placed by its leaves; the generality claim is then TESTABLE: leave one profile out (every Costanzo 30 C row, or every curated-assay row), train, score the held-out profile.

**Phenotype leaf picks the projection.** The decoder is shared and conditioned on `e`; the Phenotype leaf class selects the final linear projection and its size: scalar for Fitness, GeneInteraction, GeneEssentiality (a logit) and SyntheticLethality, 501 for CalMorph, per gene for the expression leaves, per metabolite for Metabolite. That is the existing `heads` configuration keyed by phenotype class with the output dimension per key, which is already how the morphology and expression heads differ from the scalar heads; what changes is that the conditioning on the experiment becomes one shared input to all of them instead of a name.

**Also opened, separately (D5).** The same propagation on the genotype lane encodes HOW each gene is perturbed (deletion vs ts allele vs DAmP, marker) into the perturbation operator; the model currently reads gene indices only and drops the perturbation class. That is a genotype-side change and a separate arm.

**Caveat stated.** The experiment DAG is three or four levels deep against GO's dozens, so the DCell semantics of subsystems carry less here; what the structure buys is principled parameter sharing across experiments by subtree, and a representation per subtree (an Environment embedding shared by a fitness and an interaction row from the same screen).
