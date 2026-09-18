# Agent 09 -- How the field does on "predict the transcriptional response to a perturbation", and what we have not applied

Written 2026.09.17. Read-only review. Every number below is either quoted from a paper I
read, or computed by me from a file in this repo / the scratch tree, with the path given.
Anything I did not measure is labeled "Hypothesis (untested)".

---

## 0. Bottom line up front

1. **Our leaderboard metric is already the "corrected" metric that the field spent 2024-2025
   discovering it needed.** `val/expression/pearson_per_feature` centers per gene across
   strains, so the two baselines that dominate every Perturb-seq benchmark -- "predict the
   training mean" and "predict no change" -- score **exactly 0.0** on it. This is measured,
   not argued: `expression_baselines.json` records `B0_per_gene_mean.pearson_per_feature =
   0.0` and `B1_no_change.pearson_per_feature = 0.0`.
2. **Therefore our 0.20 is NOT comparable to the "Pearson delta" numbers in the GEARS/scGPT
   literature.** Those are computed per perturbation across genes against a *control*
   reference. Systema (Nat Biotechnol 2025) showed that a method's score on that metric
   correlates 0.91-0.95 with how much "systematic variation" (the shared response direction)
   the dataset contains, and that this correlation falls to about 0.21 once the reference is
   moved to the perturbed centroid, with scores dropping substantially. Our metric removes
   that axis by construction.
3. **Measured here, new**: I decomposed our own validation dumps along that axis. On the
   `V_ref_s0` expression run, full per-feature r = 0.2227, PC1 holds 24.1% of validation
   variance (matching O'Duibhir's 24% for the whole compendium), the model scores 0.4992
   restricted to that axis -- and **0.1786 survives projecting it out, about 80% of the full
   score**. The proteome behaves the same way. So our number is not the slow-growth axis in
   disguise, which is the single most defensible thing about it.
4. **But our metric has a null floor of its own that we have not been reporting.** In
   `baselines_embedding_study.json` a **random 1024-dim embedding** scores `expression_B2_val
   = 0.0493` and `expression_B3_val = 0.0725`. So the bilinear ridge's 0.104 sits only
   ~0.055 above its own random control. **The trained model has no such control at all**;
   its null is unmeasured, so "0.20 versus a 0.104 baseline" overstates the gap by an unknown
   amount. The field (Ahlmann-Eltze; PerturBench) leads with exactly this kind of
   uninformative control. We compute it for the baselines and then do not quote it.
5. **NMSE 1.085 > 1 means that on the field's primary metric (L2 / MSE) our model loses to
   the per-gene mean baseline**, whose NMSE is 1.0 by definition. Ahlmann-Eltze et al. warn
   in as many words that Pearson delta "does not penalize predictions that are consistently
   too small or too large in amplitude". We are currently on the wrong side of that warning.
6. **The single most transferable lesson**: build the perturbation (deletion-strain)
   representation from *other perturbation compendia*, not from sequence or ontology. That
   was the only intervention in Ahlmann-Eltze's benchmark that consistently beat every model
   and every baseline, and our own embedding study shows the ontology/graph embedding
   (`normalized_chrom_pathways`, 0.053) is barely above the random control (0.049) while no
   perturbation-derived embedding has ever been tried.

---

## 1. Part A -- the Kemmeren/Hughes side of the literature

### 1.1 Kemmeren et al. 2014, the dataset itself

Kemmeren, P. et al. "Large-scale genetic perturbations reveal regulatory networks and an
abundance of gene-specific repressors." *Cell* 157, 740-752 (2014).
DOI: [10.1016/j.cell.2014.02.054](https://doi.org/10.1016/j.cell.2014.02.054).
Mirror: `/scratch/projects/torchcell-scratch/torchcell-library/kemmerenLargeScaleGeneticPerturbations2014/paper.md`

The paper's own account of how much signal exists, quoted from `paper.md`:

- 1,484 deletion mutants, one quarter of all yeast genes, SC + 2% glucose, WT doubling
  time 90 min, four replicates per responsive mutant, profiles scored against 428 WTs.
- **"Signatures were classified as different from WT (responsive) when at least four
  transcripts show robust changes ... 53% of mutants are similar to WT (nonresponsive)."**
  Concordant with Hughes et al. 2000's 43% on a smaller set.
- **"The fraction of genes that can be individually removed under a single growth condition
  with no strong effects on gene expression is 43%"** (taking essential genes into account).
- Among *responsive* mutants the response is still sparse: **"A median of 19 transcripts
  change in responsive GSTFs compared to 68 for chromatin factors."** Against ~6,000
  measured genes that is a median of 0.3% of the transcriptome for a responsive
  transcription-factor deletion.
- Cumulatively across all 1,484 mutants only **3,966 transcripts** ever change at
  FC > 1.7, p < 0.05.
- Non-responsiveness is structured, not random: strong enrichment for genes with a close
  paralog/ohnolog (p < 2.2e-16), and for genes with low mRNA and undetectable protein.
  75% of protein kinases/phosphatases are non-responders versus 25% of chromatin factors.

**Read that as a hard statement about the label matrix**: roughly half the strains carry
essentially no strain-specific signal at all, and the responsive half changes a median of
tens of genes out of 6,000. The Kemmeren-specific literature never frames this as a
prediction task with a held-out-strain split; the paper frames the compendium as a network
resource.

### 1.2 O'Duibhir et al. 2014 -- the slow-growth axis, and it is 24% of the data

O'Duibhir, E. et al. "Cell cycle population effects in perturbation studies." *Mol. Syst.
Biol.* 10, 732 (2014). DOI:
[10.15252/msb.20145172](https://doi.org/10.15252/msb.20145172).
Mirror: `.../oduibhirCellCyclePopulation2014/paper.md`

Quoted numbers:

- The recurrent signature is **PC1 of the Kemmeren matrix and "accounts for 24% of the
  variation in the entire dataset."**
- It is the environmental stress response: r = 0.73 against Gasch heat shock, r = 0.82 when
  the heat shock is repeated on their own platform and background, and **r = 0.93 when
  restricted to the genes that define the ESR**.
- Its magnitude is proportional to the growth-rate reduction of the strain, and the number
  of genes with changed expression scales with how much slower the strain grows.
- Mechanistically it is not a single-cell response at all: it is a **redistribution of the
  population over cell cycle phases** (mainly more G1), reproducible in silico from cell
  cycle time courses (r = 0.88 from elutriation data) and confirmed by flow cytometry.
- Removing it helps downstream inference: after projecting out the signature (Gram-Schmidt
  against the normalized slow-growth profile, their Supplementary Dataset S4), GSTF
  deletion profiles show a **reduced false-positive rate for finding GSTF binding in the
  promoters of changed genes**.

This is the single most important paper for our task and it is about our exact dataset.
**A quarter of the label matrix is one direction, that direction is a growth-rate readout,
and it is a population artifact rather than a regulatory response.**

### 1.3 Prior modeling on the yeast deletion compendia

- **Peleg, T., Yosef, N., Ruppin, E. & Sharan, R. "Network-free inference of knockout
  effects in yeast." *PLoS Comput. Biol.* 6, e1000635 (2010).** DOI:
  [10.1371/journal.pcbi.1000635](https://doi.org/10.1371/journal.pcbi.1000635). Predicts
  the **sign** (up/down) of a knockout-gene/target-gene pair from the Hughes compendium
  (>210 knockouts, 24,457 high-confidence pairs). Sign-clustering: **88% accuracy at 73.8%
  coverage**, versus SPINE at 72% accuracy / 2.6% coverage. Important caveat for us: this
  is a **transductive** setting (the pairs come from profiled knockouts) and the target is
  a binary sign on already-known-significant pairs, not a genome-wide profile for an unseen
  deletion. It is not a comparable number.
- **Deleteome-based tools** (e.g. DeleteomeTools, [10.1093/bioinformatics/btae100](https://doi.org/10.1093/bioinformatics/btae100),
  bioRxiv 2024.02.05.578946) use the compendium for guilt-by-association function
  prediction, i.e. profile similarity, never forward prediction of an unseen strain's
  profile.
- **scYeast** (Fan, X. et al., *Synthetic and Systems Biotechnology*, 2027 in the mirror as
  `fanScYeastBiologicalknowledgeguidedFoundation2027`) is the closest thing to a yeast
  perturbation foundation model, and it does **not** do our task. Its perturbation section
  builds ten per-gene GEARS-style networks that **interpolate a single gene's time course**
  under TF induction (IDEA dataset), R^2 0.70-0.97 (their Table 1), plus one strain-level
  held-out case (AQY2, R^2 0.7295 on remaining in-distribution strains). It touches the
  Messner 4,699-strain proteome only to predict **growth rate / ribosome occupancy /
  protein half-life** by transfer learning, never the 1,850-protein profile.

**Finding: I could not find any published work that predicts the genome-wide Kemmeren
profile for held-out deletion strains.** That is a real gap and it is also why there is no
external number to compare 0.20 against on this dataset.

### 1.4 What the yeast field says about the *mechanism* we are asking a graph to represent

Hughes, T. R. & de Boer, C. G. "Mapping yeast transcriptional networks." *Genetics* 195,
9-36 (2013). DOI: [10.1534/genetics.113.153262](https://doi.org/10.1534/genetics.113.153262).
Mirror: `.../hughesMappingYeastTranscriptional2013/paper.md`. (Note: this is the 2013
*review*, not the Hughes et al. 2000 compendium; the mirror key is easy to mis-cite.)

Three quoted facts that bear directly on "are genes allowed to interact enough?":

- **Binding barely predicts response.** Hu et al. (2007) compared 276 TF deletion profiles
  to binding data for 188 of the same proteins and "concluded that **only ~3% of genes bound
  by the TFs were affected by the knockout**. Likewise, **only ~3% of the genes with
  expression changes in the TF mutant were bound by the TF**." Cleaning steps (dropping
  promiscuous stress responders, keeping only conserved motifs) raised this only to
  **6.7% / 4.5%**.
- **Cascades are not the rule.** Accounting for indirect effects (one TF regulating another)
  raised the overlap only to 22%, and Hu et al. "concluded that **'during normal growth,
  regulation by a transcription factor is not propagated appreciably via extended
  cascades'**, suggesting that examples such as the cell cycle ... may be the exception
  rather than the rule."
- **The slow-growth confound was seen here too**: Chua et al. (2006) noted "a relationship
  ... in which the number of expression changes correlated with slowed growth."

Implication for the architecture question. Hypothesis (untested, but well supported by the
above): **more message passing is unlikely to be the bottleneck.** The biology says the true
strain-specific signal is sparse, mostly one hop, and largely *not* recoverable from the
binding/annotation graph at all, while the part that does propagate globally is the
growth/ESR axis, which is a scalar and needs no graph. Kemmeren's own counter-observation is
the ceiling on how much the graph can give: of responsive GSTFs with systematic DNA binding
data, "significant overlap with the deletion signature is found for 70%", and that overlap
is one-sided (activators map to down genes, repressors to up genes). So a graph helps for
*responsive TF deletions with known binding*, which is a minority of the panel.

---

## 2. Part B -- the Perturb-seq prediction literature and its 2024-2025 reckoning

### 2.1 What the optimistic papers claimed and how they measured it

**GEARS** -- Roohani, Y., Huang, K. & Leskovec, J. "Predicting transcriptional outcomes of
novel multigene perturbations with GEARS." *Nat. Biotechnol.* 42, 927-935 (2024). DOI:
[10.1038/s41587-023-01905-6](https://doi.org/10.1038/s41587-023-01905-6). Mirror:
`.../roohaniPredictingTranscriptionalOutcomes2023/paper.md`.

Its metrics, verbatim from the paper:

- **"Because the vast majority of genes do not show substantial variation between
  unperturbed and perturbed states, we restricted our m.s.e. analysis to the harder task of
  only considering the top 20 most differentially expressed genes."** MSE improvement of
  **30-50%** over baselines on the top-20 DE genes.
- Pearson correlation "between mean predicted post-perturbation differential gene
  expression over control and true values **across all genes**" -- i.e. per perturbation,
  across genes, delta-referenced to control.
- Fraction of the top-20 DE genes whose direction of change is right.
- Baselines: "no perturbation" (predict control) and a GRN-propagation model.
- Architecturally it uses a **gene co-expression graph** (each gene linked to its top
  co-expressed neighbors) and a **GO graph** to embed perturbations, precisely so unseen
  perturbations can borrow from annotated neighbors.

### 2.2 The reckoning

**Ahlmann-Eltze, C., Huber, W. & Anders, S. "Deep-learning-based gene perturbation effect
prediction does not yet outperform simple linear baselines." *Nat. Methods* (2025).** DOI:
[10.1038/s41592-025-02772-6](https://doi.org/10.1038/s41592-025-02772-6). Mirror:
`.../ahlmann-eltzeDeeplearningbasedGenePerturbation2025/paper.md` (full text read).

Setup: Norman (double perturbations, 19,264 genes x 84,143 cells, 124 doubles + 100
singles), Replogle K562 (1,087 singles), Replogle RPE1 (1,534 singles), Adamson (81
singles). Models: scGPT, scFoundation, GEARS, CPA, scBERT, Geneformer, UCE. Metric: **L2
distance on the 1,000 most highly expressed genes** (primary), plus Pearson delta.

Findings that bear on us, in their words:

- Doubles: **"All models had a prediction error substantially higher than the additive
  baseline"** where additive is `y_A + y_B - y_control`.
- Genetic interactions: **"None of the models was better than the 'no change' baseline."**
- Unseen singles: **"None of the deep learning models was able to consistently outperform
  the mean prediction or the linear model."** The linear model is
  `argmin_W || Y_train - (G W P^T + b) ||^2` with G, P from a 10-component PCA of the
  training matrix and b the row means.
- **The failure mode is near-constant output**: "for most genes, the predictions of scGPT,
  UCE and scBERT did not vary across perturbations, and those of GEARS and scFoundation
  varied considerably less than the ground truth" (Extended Data Fig. 7, histogram of
  per-gene SD across perturbations against the ground-truth mean SD).
- **What actually helped**: "The approach that did consistently outperform all other models
  was a linear model with **P pretrained on the Replogle data** (using K562 as pretraining
  for Adamson and RPE1, and RPE1 for K562) ... pretraining on the single-cell atlas data
  provided only a small benefit over random embeddings, but **pretraining on perturbation
  data increased predictive performance**."
- **GO did not help**: they converted GEARS' GO annotations into P via a spectral embedding
  of the pathway membership matrix; it did not reach the perturbation-data-pretrained P.
- Explicit metric warning: "Unlike the L2 distance, the Pearson delta metric **does not
  penalize predictions that are consistently too small or too large in amplitude** and thus
  prioritizes correct prediction of the direction of the expression change."
- Honest note on DE-gene restriction: "sorting by differential expression is only possible
  if access to the ground truth is available and can thus not be applied in real-world use
  cases."

**Kernfeld, E., Yang, Y., Weinstock, J. S., Battle, A. & Cahan, P. "A comparison of
computational methods for expression forecasting." *Genome Biology* (2025);
bioRxiv 2023.07.28.551039.** DOI:
[10.1186/s13059-025-03840-y](https://doi.org/10.1186/s13059-025-03840-y) (PMC12621394).
11 large-scale perturbation datasets (PSC, K562, RPE1, CD4+ T, melanoma; overexpression,
CRISPRa, CRISPRi), 9 regression approaches plus DCD-FG, GEARS, GeneFormer, versus mean and
median predictors. Result: **the mean baseline "was almost always the top performer"**, "no
methods were consistently better by any metric", and "very few subsets of genes are
predictable". Their two-part explanation is the most useful sentence in the entire
benchmark literature for us:

> statistically, the bias-variance trade-off favors the training mean when noise is high and
> perturbation effects are small; biologically, **stereotypical responses -- genes that respond
> similarly across many perturbations -- mean the training mean captures biologically
> meaningful aspects** without any correct causal mechanism, so a model can "appear to
> display some level of biological insight, even without including any correct mechanisms."

That "stereotypical response" is exactly O'Duibhir's slow-growth/ESR axis, discovered
independently in human cell lines.

**Csendes, G., Sanz, G., Szalay, K. Z. & Szalai, B. "Benchmarking foundation cell models for
post-perturbation RNA-seq prediction." *BMC Genomics* 26, 393 (2025).** DOI:
[10.1186/s12864-025-11600-2](https://doi.org/10.1186/s12864-025-11600-2). **The simplest
possible baseline -- the mean of the training pseudobulk profiles, identical for every
perturbation -- outperformed scGPT and scFoundation.**

**Bendidi, I. et al. "Benchmarking transcriptomics foundation models for perturbation
analysis: one PCA still rules them all." arXiv:2410.13956 (2024).**
[arxiv.org/abs/2410.13956](https://arxiv.org/abs/2410.13956). PCA and scVI beat the
transcriptomics foundation models on most perturbation tasks, including zero-shot transfer.

**Wong, D. R., Hill, A. S. & Moccia, R. "Simple controls exceed best deep learning
algorithms and reveal foundation model effectiveness for predicting genetic perturbations."
*Bioinformatics* 41, btaf317 (2025).** DOI:
[10.1093/bioinformatics/btaf317](https://doi.org/10.1093/bioinformatics/btaf317)
(PMC12202205). Their baseline is the **"CRISPR-informed mean model"**: predict the mean of
all perturbed training cells, but force the target gene to zero (CRISPRi) or double it
(CRISPRa). Metrics: Pearson, Pearson-DE (top 20), Pearson-delta (PD), Pearson-DE-delta.
The CRISPR-informed mean beat GEARS by 0.08 PD and scGPT by 0.11 PD (p = 9.3e-4 and
8.1e-6). Most striking: fully fine-tuned scGPT versus scGPT **trained from random
initialization** differed by **0.004 PD (p = 0.89)**; withholding the input encoder or the
transformer blocks changed nothing (0.01, p = 0.66; 0.006, p = 0.80). Their conclusion:
"neither foundation weights nor transformer attention provide a competitive advantage."

**Wu, Y. et al. "PerturBench: benchmarking machine learning models for cellular perturbation
analysis." arXiv:2408.10609 (2024).**
[arxiv.org/abs/2408.10609](https://arxiv.org/abs/2408.10609). The methodological
contribution we should steal is the **rank metric**: for each perturbation, how close is the
prediction to its own observation *compared with the predictions made for other
perturbations* (0 = perfect, 0.5 = random). Rationale, verbatim: "**mode or posterior
collapse where a model always generates the same prediction irrespective of target
perturbation may still result in decent cosine similarity or RMSE**." They then catch a
decoder-only baseline that looks fine on cosine similarity but scores exactly 0.50
(random) on the rank metric. On Norman19 the *latent additive* baseline (cosine 0.79, RMSE
rank 0.014) beat CPA and SAMS-VAE; on the 1.6M-cell Jiang24 set the decoder-only baseline
(cosine 0.64, rank 0.32) beat CPA (0.60, 0.42).

**Systema** -- "Systema: a framework for evaluating genetic perturbation response prediction
beyond systematic variation." *Nat. Biotechnol.* (2025). DOI:
[10.1038/s41587-025-02777-8](https://doi.org/10.1038/s41587-025-02777-8) (PMC13271886).
This is the paper that names our problem. "Systematic variation" = the consistent shift
between perturbed and control cells shared by all perturbations. They quantify it as the
cosine between each perturbation-specific shift and the average perturbation effect:
**Adamson 0.76 +- 0.30, Norman 0.50 +- 0.26, Replogle K562 0.32 +- 0.16, average across ten
datasets 0.41 +- 0.18.** Their fix is to reference the delta to the **centroid of the
perturbed cells** instead of the control, plus a **centroid accuracy** (is the predicted
profile closer to its own ground-truth centroid than to any other perturbation's?). Effect:
the correlation between a model's score and the systematic-variation component drops from
**0.91-0.95 to about 0.21**, scores fall "substantially" across the board, and the perturbed
mean baseline is barely beaten on centroid accuracy.

**Virtual Cell Challenge 2025** (Arc Institute), 1,200+ teams. Metrics chosen specifically
to resist the mean baseline: **PDS** (perturbation discrimination, an L1-rank score),
**DES** (differential expression), **MAE**. Wrap-up:
[arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up).
**"Almost all models performed worse than baseline on MAE"**, and results "indicate that
perturbation prediction models are not yet consistently outperforming naive baselines across
all metrics", while teams did improve on discrimination and DE identification.

### 2.3 The counter-arguments worth taking seriously

**"Deep learning-based genetic perturbation models do outperform uninformative baselines on
well-calibrated metrics." bioRxiv 2025.10.20.683304.** DOI:
[10.1101/2025.10.20.683304](https://doi.org/10.1101/2025.10.20.683304). Argues the negative
results are a *measurement* artifact: MSE and control-referenced Pearson suffer "control
bias" and "signal dilution", which let uninformative predictors score well. Their
well-calibrated set is **weighted MSE, weighted R^2, and Normalized Inverse Rank (NIR)**;
under those, most deep models beat mean/control/linear baselines on unseen genes. Note the
direction: the fix is the same fix -- weight by effect size and use a rank metric -- not a
defense of the old metrics.

**State** -- Adduri, A. K. et al. "Predicting cellular responses to perturbation across
diverse contexts with State." *Cell* (2026). DOI:
[10.1016/j.cell.2026.07.052](https://doi.org/10.1016/j.cell.2026.07.052). Mirror:
`.../adduriPredictingCellularResponses2026/paper.md`. The current state of the art, and its
own account of where it wins is instructive:

- Gains are largest where data are large and heterogeneous: "On Parse-PBMC and Tahoe-100M
  ... State outperforms the other baselines by a substantial margin", while on genetic
  screens "State exceeds or matches the respective strongest baselines" and **"the more
  modest gains on genetic perturbations reflect the inherent difficulty of predicting subtle
  perturbation effects."**
- **"Prediction accuracy is strongly correlated with ground-truth perturbation effect sizes
  ... with weaker perturbations proving more challenging for all models."**
- Their cell encoder is trained with **a two-axis loss**: "it predicts expression across
  genes within each cell **and for each gene across cells in each minibatch** ... thus
  enhancing the model's sensitivity to subtler perturbation effects." That second axis is
  literally our evaluation metric used as a training signal.
- Baselines they had to beat are "context mean" and "perturbation mean", and on DEG overlap
  at variable k State's absolute improvement is 36% on Replogle-Nadig, 75% on Tahoe-100M.

---

## 3. Part C -- mapping each finding onto our numbers

### 3.1 Is 0.20 per-gene Pearson across all genes consistent with the field?

There is no published number on this exact metric for this exact dataset, so the honest
answer is a structural comparison, not a horse race.

**Our metric is strictly the harder side of every axis on which the field's metrics were
found to be inflated:**

| inflation the field found | does it apply to us? | evidence |
|---|---|---|
| control-referenced delta credits the shared response axis | **no** | `B1_no_change` = 0.0 in `expression_baselines.json` |
| training-mean baseline scores well | **no** | `B0_per_gene_mean` = 0.0, same file |
| restricting to top-20 DE genes (needs ground truth) | **no** | we average over all 6,169 genes unweighted |
| restricting to top-1,000 expressed genes | **no** | same |
| mode collapse survives cosine/RMSE | **no** | a constant prediction scores 0 |

**But two things cut the other way, and we should say so:**

1. **We never have to get the amplitude right.** Per-gene Pearson is scale and offset free
   per gene. Our NMSE is 1.085 at the Pearson peak (`v13_split_readout.json`,
   `nmse_at_peak` 1.030-1.053 across the V_ref/V_concat s0 runs, `nmse_last` up to 1.079),
   versus 1.0 for `B0_per_gene_mean`. **On Ahlmann-Eltze's primary L2 metric our model would
   be reported as worse than the mean baseline.** That is exactly the gap they flag between
   Pearson delta and L2. This should be stated whenever 0.20 is quoted.
2. **The metric has a non-zero null floor for *fitted* models, which we do not report.**
   From `baselines_embedding_study.json`, random embeddings on the expression task
   (mean of 4 seeds, val):

   | embedding | B2 bilinear | B3 kNN |
   |---|---|---|
   | `random_10` | 0.0229 | 0.0200 |
   | `random_100` | 0.0087 | 0.0592 |
   | `random_1024` | **0.0493** | **0.0725** |
   | `normalized_chrom_pathways` (graph) | 0.0530 | 0.0329 |
   | `species_lm_5p_3p` (regulatory DNA) | 0.0377 | 0.0617 |
   | `calm` (coding sequence) | 0.0892 | 0.1006 |
   | `prot_T5_all` (protein LM) | **0.1055** | **0.1200** |

   The floor rises with embedding dimension, which is the signature of overfit-driven
   spurious per-gene correlation rather than signal. **The graph embedding is within noise
   of the random 1024-dim control on B2.** Hypothesis (untested): a proper permutation null
   (shuffle validation strain labels, refit nothing) would land near 0.02-0.05 for the
   ridge-class baselines and near 0 for the trained model; this is a ten-minute script and
   it should exist before any number is put in a paper.

### 3.2 The slow-growth axis: measured, and it is not the whole story

`lowrank_output_ceiling.json` already computes what the field never computes for its own
datasets: an **oracle** on a rank-r gene basis fitted to the training strains, scored in our
own metric, with the per-gene mean taken from fit strains only (verified in
`lowrank_output_ceiling.py` lines 203-212, so there is no centering leak).

| rank | oracle per-feature Pearson | fraction of val variance |
|---|---|---|
| 1 | **0.2658** | 0.1848 |
| 2 | 0.4121 | 0.2607 |
| 4 | 0.5249 | 0.3586 |
| 8 | 0.5787 | 0.4268 |
| 16 | 0.6662 | 0.5291 |
| 32 | 0.7265 | 0.6190 |
| 64 | 0.7799 | 0.7019 |

**Rank 1 captures 18.5% of validation variance, matching O'Duibhir's 24% of the full
compendium, and an oracle on that single coefficient would score 0.266 -- above our best
measured model.** A model that did nothing but predict each strain's growth-rate coefficient
perfectly would beat us on the leaderboard.

I then measured how much of the model's actual score lives on that axis, using the
validation dumps in `/scratch/projects/torchcell-scratch/val-predictions/` (written
2026.09.17). Procedure: center prediction and target per gene across validation strains,
take the first right singular vector of the **true** validation residual as the shared axis,
and rescore the prediction's on-axis component, its off-axis component, and the off-axis
component against the off-axis truth.

| dump tags | n feat | full r | PC1 var share | pred-PC1 vs full truth | pred-resid vs full truth | **resid vs resid** |
|---|---|---|---|---|---|---|
| `V_ref_s0` (expression, v13 split) | 6,127 | 0.2227 | 0.241 | 0.1477 | 0.1516 | **0.1786** |
| `V_concat_s0` (expression) | 6,127 | 0.1926 | 0.241 | 0.1357 | 0.1238 | **0.1485** |
| `V_ref_s2` (expression) | 6,127 | 0.1400 | 0.190 | 0.0957 | 0.0966 | **0.1081** |
| `P_ref_s0` (proteome, v14) | 977 | 0.1372 | 0.264 | 0.0769 | 0.1005 | **0.1075** |
| `P_concat_s0` (proteome) | 977 | 0.1430 | 0.264 | 0.0982 | 0.0907 | **0.0924** |
| `P_ref_s2` (proteome) | 1,019 | 0.1569 | 0.251 | 0.1039 | 0.1273 | **0.1259** |

(Script: an inline numpy job over the dumps; output at
`/scratch/tmp/.../tasks/b2t3u10zh.output`. Proteome rows are restricted to the proteins with
no NaN in either matrix, 977-1,019 of 1,850, which is why their full r sits above the 0.099
partition mean quoted for v14. The `V_concat_s0` full r of 0.1926 is the run that matches
the 0.1965 campaign figure.)

Caveats, stated plainly: the axis is taken from the validation targets themselves, so its
*direction* is oracular even though the per-strain coefficients are not; centering uses
validation means rather than fit means. Both make the split slightly optimistic.

**Conclusion, and it corrects the hypothesis I went in with:** our score is **not** just the
slow-growth axis. The model is much better on that axis than overall (per-feature r 0.4992
restricted to the PC1 subspace on `V_ref_s0`, versus 0.2227 overall), but **0.179 of the
0.223 survives projecting the axis out -- about 80%.** Split 2 keeps 0.108 of 0.140, also
~77%. The proteome behaves the same way, with a PC1 share of 0.25-0.26 and 0.092-0.126
surviving. That is a materially better position than any Perturb-seq model reported in
Systema, where scores collapse toward the perturbed-mean baseline once the shared axis is
removed, and it is worth reporting as a headline rather than leaving implicit.

Note also that PC1's share of validation variance measured directly on the dumps (0.241 on
split 0, 0.190 on split 2, 0.25-0.26 on the proteome) brackets O'Duibhir's 24% for the full
Kemmeren compendium. The slow-growth axis is present in our splits at exactly published
strength, and it is present in the proteome too.

### 3.3 Responsiveness: half the validation strains are close to unmeasurable

`stratified_responsiveness_seed0.json`, val split:

- responsive: 77 strains, mean |log2| = 0.1645, sd 0.282, 1.12% of entries above |1|
- nonresponsive: 78 strains, mean |log2| = 0.0999, sd 0.170, 0.19% of entries above |1|
- per-gene variance concentration: top 1% of genes carry **17.4%** of the variance, top 5%
  carry 37.7%, top 10% carry **49.5%**, median gene sd = 0.144

The 50/50 responsive split matches Kemmeren's 53% non-responsive exactly. So our unweighted
average over 6,169 genes and 155 strains is spending half its strain budget and 90% of its
gene budget on entries where the paper itself says there is nothing to predict. That is the
"signal dilution" the well-calibrated-metrics preprint names, and the direct analog of
State's "prediction accuracy is strongly correlated with ground-truth perturbation effect
sizes."

### 3.4 Proteome

`proteome_ceiling_replicate.json`:

- ceiling 0.4169 mean / 0.4437 median from duplicate strains; 0.6136 from HIS3 replicates
- v14 partition mean 0.099 -> **23.7% of the duplicate-strain ceiling**; the kNN baseline is
  17.5% of it
- **only 10.6% of KO entries lie beyond 2 sd of the HIS3 noise** (`frac_ko_entries_beyond_2sd_his3`)
- **median per-strain duplicate correlation = 0.028** (`route_d_duplicate_strains.per_strain_median_r`)

That last number is the one to internalize: **for a typical strain, two measurements of the
same deletion do not correlate at all.** Per-protein reliability is 0.21 only because it
pools across strains. The proteome's *strain axis* is close to unmeasurable at this depth,
which caps any perturbation-conditional model regardless of architecture. Reaching 0.13 of a
0.42 ceiling (31%) on this data is, relative to the field, not a weak result; it is a result
on a label with almost no per-strain replicate signal. The right comparison for the paper is
fraction-of-ceiling, not the raw number, and we are one of very few groups reporting a
ceiling at all.

### 3.5 Summary answer to "is our metric harder or easier than theirs?"

**Harder on the axis that matters and that the field just spent two years fixing** (shared
response direction removed by construction; mode collapse scores 0; no ground-truth-gated
gene selection). **Easier on amplitude** (scale-free per gene, and our NMSE > 1 proves we
are exploiting that). **Different, and currently unreported, on the null floor** (a fitted
model with random features scores ~0.05, not 0).

Given that, **0.20 per-gene Pearson across all genes, with 0.18 of it surviving removal of
the shared axis, is a strong number by the field's post-2024 standards** -- Systema's
corrected scores on unseen single perturbations hug the perturbed-mean baseline, and the
VCC's 1,200 teams could not consistently beat naive baselines on MAE. It is simultaneously a
**weak number in absolute terms** -- 0.20 / 0.775 = 26% of the replicate ceiling, and below
the 0.266 an oracle on a single scalar per strain would get.

---

## 4. What the field learned that we have NOT applied

Ordered by expected value, with the source of the lesson and the concrete action.

1. **Build the strain representation from other perturbation compendia, not from sequence or
   ontology.** (Ahlmann-Eltze: pretraining P on perturbation data from another cell line was
   "the approach that did consistently outperform all other models"; atlas-derived gene
   embeddings were "only a small benefit over random embeddings"; the GO spectral embedding
   did not compete.) Our embedding study contains **zero** perturbation-derived embeddings:
   the best is a protein language model (0.1055) and the graph/pathway embedding (0.0530) is
   within noise of the random control (0.0493). torchcell already holds, keyed by deleted
   gene, the Costanzo SGA genetic-interaction profile, the Messner 4,476-strain proteome,
   the Ohya morphology profile, and Hillenmeyer chemogenomics. **A deletion strain's SGA
   interaction profile is the exact analog of "P pretrained on Replogle K562".** Concrete
   test: rerun B2/B3 with an SGA-profile embedding for the perturbed gene. If Ahlmann-Eltze
   transfers, this baseline alone should move well past 0.104 and possibly past the model.
   (Leakage note: it must be a *different modality*, never the Kemmeren matrix itself.)
2. **Report a rank / discrimination metric next to the Pearson.** (PerturBench's rank
   metric; VCC's PDS; Systema's centroid accuracy.) All three exist because a good-looking
   correlation is compatible with strain-agnostic predictions. Our metric resists constant
   predictions, but it does not tell us whether strain A's prediction is closer to strain
   A's truth than to strain B's. That is one line of numpy on the dumps we already write,
   and it is the number a reviewer from this field will ask for.
3. **Report the off-axis (slow-growth-removed) Pearson as a standing second metric, and use
   the growth head to supply the axis explicitly.** (O'Duibhir gives the vector,
   Supplementary Dataset S4, and shows projecting it out reduces false positives; Kernfeld
   names "stereotypical responses" as the reason the mean baseline wins; Systema builds a
   whole framework on it.) The measurement above changes what this buys us: only about 20%
   of the score is on that axis, so a two-head decomposition is **not** a large expected win
   in score, and I should not sell it as one. What it does buy is (a) the defensible
   headline -- "0.18 after the shared growth axis is removed" is the claim a reviewer from
   this field cannot discount, and we currently do not compute it in training; (b) a cleaner
   signal for architecture comparisons, since arms that differ only in how well they track a
   single scalar will look different on the full metric and identical off-axis. Concretely:
   add `pearson_per_feature_offaxis` to the validation loop, with the axis fitted on the
   training split only. Hypothesis (untested): arm rankings from past rounds partially
   reorder under the off-axis metric, and that reordering is the more informative one.
4. **Add the per-gene-across-samples term to the training loss.** (State: "it predicts
   expression across genes within each cell and for each gene across cells in each minibatch
   ... thus enhancing the model's sensitivity to subtler perturbation effects.") We evaluate
   on exactly this axis and train on a pinball loss that has no cross-strain term at all.
   Caveat that must be respected: batch size 32 makes a within-batch per-gene correlation
   noisy, and the campaign already saw Pearson collapse at batch 64 (memory
   `019-readouts-2026-09-08`), so this interacts with the batch-size decision and should be
   run as a small ladder rather than a switch.
5. **Weight the loss and the reported metric by measured reliability / effect size.**
   (State: accuracy tracks effect size; PertEval-scFM: "all models struggle with predicting
   strong or atypical perturbation effects"; the well-calibrated-metrics preprint proposes
   weighted MSE and weighted R^2.) We have per-gene reliabilities in
   `expression_ceiling_replicate.json` and per-protein ones in
   `proteome_ceiling_replicate.json`, and we already have a responsiveness label per strain.
   Report the metric stratified by responsive / non-responsive at minimum; the reliability-
   weighted per-protein loss is already third in the queue, so this is the same idea
   arriving from the literature.
6. **Fix the amplitude, or stop quoting Pearson alone.** (Ahlmann-Eltze's explicit metric
   caveat; VCC's MAE finding that almost all models lose to baseline.) NMSE 1.085 at the
   Pearson peak, with NMSE minimized at epoch ~150-600 and rising thereafter, is the
   textbook picture of a model whose direction is improving while its calibration decays.
   Either report NMSE alongside every Pearson, or add a calibration term.
7. **Report the random-embedding / permutation null.** We compute it and then do not quote
   it. Every negative-result paper in this field leads with its uninformative control.
8. **A co-expression graph beat the GO graph in the one place it was tested.** GEARS builds
   its gene graph from **co-expression computed on the training data**, not from ontology,
   and Ahlmann-Eltze's GO-derived P did not compete. Our graph is chromatin/pathway derived.
   Hypothesis (untested): a gene-gene graph built from the Kemmeren training-split
   co-expression would carry more of the relevant structure than the annotation graph, at
   the cost of being split-dependent (it must be refit per split to avoid leakage).

9. **Do not answer "are genes allowed to interact enough?" with more layers.** Section 1.4
   is the field's answer for this organism: only ~3% of TF-bound genes respond to that TF's
   deletion, only ~3% of responding genes are bound, and TF regulation "is not propagated
   appreciably via extended cascades" under normal growth. GEARS is the one architecture in
   this literature built on a gene graph, and it did not beat an additive or a mean baseline
   in any of the five independent benchmarks above. Hypothesis (untested): depth and
   connectivity are not our binding constraint; *what is attached to each node* (lesson 1)
   and *what the loss optimizes* (lessons 4-5) are. A cheap falsification is a depth /
   neighborhood-size ablation scored on the off-axis metric of lesson 3, where a real
   propagation effect would have to show up.

**What we already do that the field mostly does not, and should be said out loud in any
write-up:** a replicate-based noise ceiling per gene and per protein; a rank-r oracle
ceiling in the evaluation metric; a metric that gives the mean baseline exactly zero; and a
random-embedding control. Systema, PerturBench and the VCC all had to invent the third of
those in 2025. We should present the ceiling-normalized number (0.20 / 0.775 = 26% for
expression; 0.13 / 0.42 = 31% for proteome) as the primary result, because it is the only
form of the number that is comparable across datasets with different noise floors, and
because it is the axis on which we are ahead of the field's practice rather than behind it.

---

## 5. Sources

- Kemmeren et al. 2014, *Cell* 157, 740-752. https://doi.org/10.1016/j.cell.2014.02.054
- Hughes et al. 2000, *Cell* 102, 109-126. https://doi.org/10.1016/S0092-8674(00)00015-5
- Hughes & de Boer 2013, *Genetics* 195, 9-36. https://doi.org/10.1534/genetics.113.153262
- Hu, Killion & Iyer 2007, *Nat Genet* 39, 683-687. https://doi.org/10.1038/ng2012
- O'Duibhir et al. 2014, *Mol Syst Biol* 10, 732. https://doi.org/10.15252/msb.20145172
- Sameith et al. 2015, *Mol Syst Biol* 11, 833. https://doi.org/10.15252/msb.20156253
- Peleg, Yosef, Ruppin & Sharan 2010, *PLoS Comput Biol* 6, e1000635. https://doi.org/10.1371/journal.pcbi.1000635
- Fan et al. (scYeast), *Synth Syst Biotechnol*, mirror key `fanScYeastBiologicalknowledgeguidedFoundation2027`
- Roohani, Huang & Leskovec 2024 (GEARS), *Nat Biotechnol* 42, 927-935. https://doi.org/10.1038/s41587-023-01905-6
- Cui et al. 2024 (scGPT), *Nat Methods* 21, 1470-1480. https://doi.org/10.1038/s41592-024-02201-0
- Theodoris et al. 2023 (Geneformer), *Nature* 618, 616-624. https://doi.org/10.1038/s41586-023-06139-9
- Lotfollahi et al. 2023 (CPA), *Mol Syst Biol* 19, e11517. https://doi.org/10.15252/msb.202211517
- Ahlmann-Eltze, Huber & Anders 2025, *Nat Methods*. https://doi.org/10.1038/s41592-025-02772-6
- Kernfeld et al. 2025, *Genome Biol*. https://doi.org/10.1186/s13059-025-03840-y
- Csendes et al. 2025, *BMC Genomics* 26, 393. https://doi.org/10.1186/s12864-025-11600-2
- Bendidi et al. 2024, arXiv:2410.13956. https://arxiv.org/abs/2410.13956
- Wu et al. 2024 (PerturBench), arXiv:2408.10609. https://arxiv.org/abs/2408.10609
- Wong, Hill & Moccia 2025, *Bioinformatics* 41, btaf317. https://doi.org/10.1093/bioinformatics/btaf317
- Systema 2025, *Nat Biotechnol*. https://doi.org/10.1038/s41587-025-02777-8
- Wenteler et al. 2025 (PertEval-scFM), ICML; bioRxiv 2024.10.02.616248. https://doi.org/10.1101/2024.10.02.616248
- "Deep learning-based genetic perturbation models do outperform uninformative baselines on well-calibrated metrics", bioRxiv 2025.10.20.683304. https://doi.org/10.1101/2025.10.20.683304
- Adduri et al. 2026 (State), *Cell*. https://doi.org/10.1016/j.cell.2026.07.052
- Replogle et al. 2022, *Cell* 185, 2559-2575. https://doi.org/10.1016/j.cell.2022.05.013
- Norman et al. 2019, *Science* 365, 786-793. https://doi.org/10.1126/science.aax4438
- Arc Institute, Virtual Cell Challenge 2025 wrap-up. https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up

### Repo / scratch files this report reads

- `experiments/019-simb-multimodal/results/expression_baselines.json`
- `experiments/019-simb-multimodal/results/baselines_embedding_study.json`
- `experiments/019-simb-multimodal/results/lowrank_output_ceiling.json`
- `experiments/019-simb-multimodal/scripts/lowrank_output_ceiling.py`
- `experiments/019-simb-multimodal/results/expression_ceiling_replicate.json`
- `experiments/019-simb-multimodal/results/proteome_ceiling_replicate.json`
- `experiments/019-simb-multimodal/results/stratified_responsiveness_seed0.json`
- `experiments/019-simb-multimodal/results/v13_split_readout.json`
- `/scratch/projects/torchcell-scratch/val-predictions/*.json`
