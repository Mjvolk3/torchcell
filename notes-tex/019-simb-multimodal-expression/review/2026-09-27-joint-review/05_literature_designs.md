# Reviewer 5: what the literature and our own documents say about proteome vs transcriptome in yeast deletions, how synergy has been demonstrated elsewhere, and the designs that follow

Read in full: mirror papers Messner 2023, Kemmeren 2014, Sameith 2015, Qian 2026 (each sha256-verified
against its own `manifest.json`, all four OK); `notes-tex/028-knockout-expression-metabolic/sections/*.tex`;
`notes-tex/019-simb-multimodal/sections/{2-expression,7-common,8-directions}.tex`;
`notes-tex/019-simb-multimodal-expression/sections/{1-findings,3-baselines,3-next}.tex` plus the proteome
subsection of `2-readouts.tex`; the proteome-expression EDA, covariation, ceiling, responsiveness and
variance-stratified notes; `notes/paper.north-star.md`; `results/v16_joint_{readout,expr_readout}.json`;
`conf/cgt_expr_v16_joint.yaml` and the `J_*` block of `scripts/gh_expr_008_arm.sh`. Nothing edited.
Note for the next reader: `TC_LIT_URL` / `TC_LIT_API_KEY` are NOT in the worktree `.env` (only
`TC_LIT_KEYS_FILE`), so the mirror was read from disk at `$DATA_ROOT/torchcell-library/` with manifest
verification, which is the same bytes `tc-lit` streams. Addendum of 02:07 taken as given throughout.

---

## 1. Provenance defect to fix before any of this is written up

Three of our documents give three different DOIs for Messner 2023. The mirrored paper states
`10.1016/j.cell.2023.03.026` (`messnerProteomicLandscapeGenomewide2023/paper.md:47`), which
`019-simb-multimodal-expression/sections/2-readouts.tex:758` has right;
`019-simb-multimodal/sections/7-common.tex:275` says `10.1038/s41586-023-06739-5` and
`028-knockout-expression-metabolic/sections/0-question.tex` says `10.1016/j.cels.2022.12.003` (a Cell
Systems DOI, and Cell Systems is Zelezniak 2018's journal, cited correctly two paragraphs later). Two of
the three are wrong. Flagged, not fixed, per the no-edit rule.

---

## 2. How correlated are deletion-induced mRNA and protein changes

### 2.1 The raw numbers, and why the per-deletion one carries no information

Measured, ours, on the 1,350 deletions Messner and Kemmeren share
(`experiments/028-knockout-expression/results/proteome_expression_covariation.json`, script
`proteome_expression_covariation.py`; reproduces the July EDA from the served graph):

| statistic | value |
|---|---|
| per-deletion Pearson, protein vs mRNA log2 ratio over 1,812 shared genes (median, IQR) | **0.036** (-0.020 to 0.120) |
| per-gene Pearson across the 1,350 deletions (median, IQR) | **0.075** (0.033 to 0.134) |
| fraction of genes with per-gene r > 0.3 | 1.43 % |
| Messner vs Nadal-Ribelles A, per deletion / per gene (2,038 shared) | 0.011 / 0.012 |
| ridge proteome -> expression, held-out R^2 (uniform avg) | 0.035 |
| ridge expression -> proteome, held-out R^2 | 0.025 (trivial baseline -0.004) |

**The per-deletion 0.036 is not evidence of biology.** Within Messner itself, the 149 duplicate-strain
pairs (145 ORFs deleted two or three times, 99 % on different plates) correlate at a median **0.03**
across proteins per strain (`proteome_ceiling_replicate.json`, route D). Zelezniak 2018 against
Messner on 89 shared kinase deletions reads 0.08 per deletion. So a knockout's whole-profile shape does
not reproduce *within one study, one lab, one medium* any better than it agrees across the two
modalities. Any claim of the form "a deletion's protein response is decoupled from its mRNA response"
is unidentifiable from this statistic: the floor and the observation coincide.

### 2.2 The per-gene number, disattenuated

The per-gene statistic survives because it averages over 1,350 strains. Attenuation correction with our
own measured reliabilities (derivation mine): observed r is attenuated by `sqrt(rho_prot * rho_mRNA)`,
`rho_mRNA = 0.611` (Kemmeren vs Sameith per-gene test-retest on 82 deletions) and `rho_prot` either 0.21
(route D, duplicate strains) or 0.43 (route W, 388 HIS3 replicates). That gives attenuation factors 0.358
and 0.513 and a **disattenuated per-gene r of 0.21 or 0.147** respectively, i.e. the raw 0.075 overstates
the decoupling by two to three times. It is still small: even at 0.21, `r^2 = 0.044`, so **about 96 % of
the reproducible per-gene protein-change variance is not shared with the reproducible mRNA-change
variance.** The SM-vs-SC medium difference sits inside that 96 % and cannot be separated here, so the
honest statement is "post-transcriptional plus condition-specific". The same factors give the **ceiling on
the observed per-gene r, 0.36 to 0.51**, which is the number any cross-modal readout should be scored
against rather than 1.

### 2.3 Where the shared signal actually lives: gene modules, not deletions

Same JSON, gene-by-gene co-variation (two genes get one Pearson across a panel's own strains; two
panels compared by the Spearman between the upper triangles; no strain aligned):

Kemmeren vs Sameith 0.68; Kemmeren vs Caudal isolates 0.481 (top-1 % pairs 0.606 vs 0.070 for the rest);
Messner protein vs Caudal 0.356 (0.482 vs 0.070); **Messner protein vs Kemmeren mRNA 0.310** (0.381 vs
0.125); Messner vs Sameith 0.24; anything vs Nadal-Ribelles A 0.03 to 0.11. Abundance level agrees as
expected (protein in the HIS3 reference vs mean isolate log2 TPM, Spearman 0.676 over 1,823 genes), which
is the classic across-gene steady-state correlation and says nothing about perturbation response.

**This is the single most important fact for the joint-training question.** Four panels differing in
platform, perturbation type and medium agree at 0.24 to 0.68 on *which gene pairs co-vary*, while the two
that share strains agree at 0.04 on *what a given deletion did*. A shared trunk can transfer the first and
has almost nothing of the second to transfer. The corollary is testable and is the basis of E3: if joint
training helps, it should help just as much when the strain-to-strain pairing is destroyed, because the
module structure survives permutation of the strain index.

### 2.4 Fraction post-transcriptional, from the literature

- **Jakobson et al. 2025, Science 390:eadu3198** (Ralser lab, same lab as Messner; read from the bioRxiv
  preprint, `10.1101/2024.10.18.619054`): in 851 meiotic progeny, "only 30 of 127 proteins with a cis-pQTL
  had a significant mRNA allelic imbalance, even though we were well-powered to detect allele-specific
  expression of these mRNAs (117 of the cis-pQTLs had a tag SNP in the associated ORF; median depth 183
  read counts)". **97 of 127 (76 %) of well-powered cis-pQTLs have no detectable mRNA effect.** Their map
  explains a median 22.8 % of protein-abundance variance, "comparable to mapping of mRNA abundance in yeast
  (median 21.9 % variance explained)": genotype-to-protein and genotype-to-mRNA are comparably learnable in
  yeast, they are just not learnable *through each other*. Complex members co-vary at mean Pearson 0.224
  against 0.038 for all protein pairs (p < 1e-195), and they attribute trans attenuation to "buffering of
  the proteome against gene expression noise".
- **Messner 2023 itself never compares to a deletion transcriptome.** Verified by grep: it cites Kemmeren
  2014 only for guilt-by-association precedent and claims novelty on exactly this ground, "neither
  transcriptome nor metabolome captures the post-transcriptional regulation of protein expression"
  (`paper.md:228`). Our Messner-vs-Kemmeren per-deletion comparison is unpublished; reviewer-2's helper
  gives Ozturk 2022 in *S. pombe* (strain-level R 0.51) as the closest published analogue.
- Messner's own class-level statements: ohnolog protein pairs correlate at median Spearman 0.19 vs 0.01;
  in 22 % of 51 complexes a subunit deletion decreases the other subunits and in 18 % it *increases* them
  (feedback, e.g. the proteasome via Rpn4); long-half-life proteins are *more* likely to be differentially
  expressed and tend to decrease; 92 strains carry a whole-chromosome aneuploidy, transmitted to the
  proteome "with a minimum amount of gene-dosage buffering"; a random forest on the proteome predicts
  growth rate at R^2 = 0.68. And 53 % of Kemmeren deletions are NON-responsive (fewer than four robustly
  changed transcripts), enriched for genes with a close paralog, so roughly half of each panel's
  perturbation axis carries almost no signal to share.

### 2.5 Which classes decouple, and the one that does not

Three sources converge (Messner's complex/paralog/half-life results, Jakobson's complex co-variation and
structural pQTN analysis, reviewer-2's helper): **protein-complex subunits, ribosomal proteins and
long-half-life proteins decouple** from mRNA, while **aneuploid gene dosage does not**, passing to protein
essentially unbuffered. The last is an exploitable positive-control stratum: 92 Messner strains carry a
chromosomal duplication whose proteomic and transcriptional signature is large and gene-position-indexed.

### 2.6 What this implies about whether a shared representation can help each head

A bound, with the assumption named. The trunk reaches 0.099 to 0.116 per-feature Pearson on the proteome
(v14), and a *fully measured* proteome maps to the transcriptome at held-out R^2 = 0.035, roughly 0.19 in
correlation units. Under a linear attenuation-chain assumption (heuristic, not a theorem) the transferable
signal through a proteome bottleneck is of order `0.11 x 0.19 ~ 0.02`, the same size as the metric's noise
floor (strain-bootstrap sd 0.026 val, 0.018 test; paired-difference sd 0.0167 per the addendum).
**Hypothesis (untested): the per-deletion pairing channel cannot produce a resolvable joint gain at this
data scale, and the only channel with room is the gene-module channel of 2.3.** E1 measures the bottleneck
directly at no GPU cost; E3 tests which channel any observed gain uses.

---

## 3. How synergy has been demonstrated elsewhere, and how weakly

Surveyed by me plus a helper that read 18 papers directly.

**The field's own negative-control paper is the most useful single input.** Ahlmann-Eltze et al. 2025
(mirror `ahlmann-eltzeDeeplearningbasedGenePerturbation2025/paper.md`), "Deep-learning-based gene
perturbation effect prediction does not yet outperform simple linear baselines". Five foundation models
plus GEARS and CPA against "no change" and "additive" on doubles, and against a "mean" baseline and a
bilinear `Y ~ G W P^T + b` linear model on unseen singles; splits repeated 5 times (doubles) and twice
(singles); model-versus-baseline reported as a **bootstrapped mean ratio with 95 % CI restricted to the
perturbations every arm can predict**. Line 61: "None of the deep learning models was able to consistently
outperform the mean prediction or the linear model." Line 65: **"The approach that did consistently
outperform all other models was a linear model with P pretrained on the Replogle data"**, and elsewhere
"pretraining on the single-cell atlas data provided only a small benefit over random embeddings, but
pretraining on perturbation data increased predictive performance". So the one thing that worked is a
**perturbation embedding transferred from another perturbation dataset, inside a linear model**: a transfer
design, not a joint multi-task head. Our own embedding study contains zero perturbation-derived embeddings
(`3-baselines.tex`; report 09 of the 2026-09-17 review).

**Joint multi-omics perturbation models: the controls are weak, across the board.** MultiPert (PLOS Comput
Biol 2026) reports PCC 0.78 transcriptome / 0.79 proteome but ablates *architecture* (no adversarial
alignment, shared instead of modality-specific encoders), not the second modality; its baselines (scGen,
CPA, scPRAM, scGPT) are different models chosen because they "lack multi-omics data integration
capabilities"; the protein side is **4 proteins**. MultiVCDiff (morphology + transcriptome diffusion,
Spearman 0.754 +/- 0.107 on a strict OOD split) has no morphology-only or transcriptome-only version of
itself. STATE (Adduri et al., *Cell* 2026) is RNA-only: its "cross-modality" means genetic versus chemical
perturbation, its viability readout is an SVM on true deltas applied post hoc, and its one auxiliary
objective (dataset classification on a `[DS]` token) is **never ablated**; the modality extension to
proteomics is stated as aspiration. scGPT's multi-omic claim is scored entirely on clustering metrics
(a 9 % AvgBIO gain) with every layer re-initialized except embeddings and every task fine-tuned separately,
so no single model is ever multi-task. Perturb-Multi (Cell 2025) measures RNA and imaging in the same cells
and runs **no joint model at all**, reporting only a post-hoc Pearson R = 0.5 between the two modalities'
perturbation effects, which is a useful empirical ceiling.

**Cross-modal predictors are the honest comparator, and they are one-directional.** Qian 2026, the
perspective we position against, cites exactly this class: "Cross-modal predictors, including SPIDER and
scLinear, enable transcriptome-to-proteome or modality-to-modality inference, extending constraint-aware
metabolome-proteome translation frameworks" (`paper.md:106`, refs 53-55, ref 55 being Zelezniak 2018).
Nowhere does the consensus roadmap claim that joint training improves each modality's own prediction; its
multi-omics argument is representation alignment plus one-directional inference. The PI's target claim is
stronger than the consensus asserts. Note also that Zelezniak 2018, the proteome-to-metabolome exemplar,
won with **ridge regression plus greedy variable selection** (10-fold CV median R^2 = 0.549) and contains a
clean NULL of its own: KEGG and Reactome annotations "were not more predictive about enzyme co-expression
as random networks".

**Negative transfer is the documented default in multi-task learning**, and we have our own instance:
the 023 betaxanthin-plus-amino-acid joint arm cost `-0.0279 +/- 0.0169` on the primary target, and
`8-directions.tex` calls it "the one auxiliary-head experiment run to completion here, which came out
negative". `decoder-distributional-plan.md:194` already uses the phrase "kills the negative transfer"
about a joint expression arm. The standard MTL diagnostics (task-pair screening in the Standley-2020
style, gradient-conflict cosine in the PCGrad style) have never been run here. **The cheapest of them
costs nothing and should be added to E2: log the cosine similarity between the two heads' gradients on
the shared trunk each epoch.** A joint arm that helps with persistently negative gradient cosine, or
hurts with positive cosine, is a result either way, and it is the mechanism evidence a reviewer will ask
for once a difference is claimed.

**The shuffled-pairing null HAS been run once, and it says the pairing is worth almost nothing.**
Correcting my first draft: Ryu, Bunne, Pinello, Regev and Lopez, arXiv:2405.00838v3, measured paired RNA
and protein in the *same* T cells under 11 kinase inhibitors at 3 doses, so they have a ground-truth
cell-to-cell pairing and can build the full ladder. Predicting RNA from protein on held-out treatments
(nested 5-fold CV over treatments): **perfect pairing Rv = 0.107; pairing scrambled within perturbation
label ("uniform per label") Rv = 0.0794; by-dosage 0.0812**; feature-matching enrichment 6.95 / 1.85
respectively. **About three quarters of the achievable cross-modal signal is carried by the perturbation
label alone, and the per-sample pairing is worth ~0.028 Pearson** even with a perfect pairing and both
modalities measured on the same cells. This is direct precedent for section 2.3's prediction and for E3
below, and it is a far stronger citation than novelty would have been. Means over outer folds only; no
variance reported. Separately, `FOSCTTM` (fraction of samples closer than the true match, random
expectation 0.5) is the field's implicit random-pairing null, but it scores *alignment*, not whether
joint training improved either modality's own prediction.

**The one same-model joint-versus-single test in this literature is NULL.** totalVI (Gayoso 2021, RNA +
protein) against scVI (RNA only) on held-out RNA from the same cells: *"On the held-out RNA data, totalVI
and scVI were largely comparable and outperformed FA."* Adding the protein modality did not improve
held-out RNA. Its protein-side win is against Seurat v3, a different method; MultiVI's only
single-modality comparator is PCA on raw data. Everything else in the family (MOFA+, Cobolt, scArches,
Seurat WNN, scGPT's multi-omic task) is scored on clustering and batch metrics, which cannot test
per-modality prediction at all.

**The sign of the MTL answer flips on the capacity control, so we must declare ours in advance.** Standley
et al. 2020 (ICML) trained all 31 subsets of two 5-task sets. Against single-task networks of equal
per-task capacity, multi-task training is **worse at every group size** (-7.56 % at 2 tasks, -10.69 % at 4,
-19.00 % at 5); against single-task networks sharing an equal *total* compute budget it **wins** for 3 to 5
tasks (+4.23 %, +4.86 %, +0.34 %). Same networks, opposite conclusion. They also find task affinity is not
predictable from transfer-learning similarity (r = -0.12, p = 0.74). PCGrad (Yu et al. 2020) defines
conflict as `cos(g_i, g_j) < 0` and shows the *direction* correction, not magnitude rescaling, is what
works. Neither diagnostic has been run here.

**The right baseline for cross-modal protein prediction is the protein's own mRNA, and almost nobody uses
it.** The one paper that does (bioRxiv `10.1101/2024.12.11.627925`, 205 protein-mRNA pairs on two CITE-seq
atlases) finds **127 of 205 own-mRNA correlations in [0, 0.4) and 28 negative**; ML beats own-mRNA for 204
of 205 proteins, yet **70 of 205 stay below r = 0.4 under both**. Crucially, *accuracy drops pronouncedly
within individual cell types versus across all cells*: most of the apparent cross-modal signal is cell-type
identity, not per-cell coupling. **Our exact analogue is the slow-growth axis**, 24 % of the
validation-target variance, on which the model already scores 0.50 (report 09). Any cross-modal claim must
be re-scored with that axis projected out or a reviewer will read it as the growth axis rediscovered.

---

## 4. What the v16 evidence actually licenses today

Read from `v16_joint_readout.json` and `v16_joint_expr_readout.json` directly, not from the summary:

- **Proteome head, `roll_max`: joint is WORSE in 6 of 6 pairs**, mean `-0.0161`, sd 0.0112, t = -3.53.
  The brief's `-0.008` (t -1.4) is the pre-registered fixed-window statistic, which softens it to 3 of
  6 positive. Both are in the file. So the two scoring rules disagree in *significance* and agree in
  *sign*: nothing supports joint >= single on the proteome, and the order statistic supports the
  opposite. One representative pair, split 0 seed 1 (`J_ref` 0.1116, `J_joint` 0.0872, diff -0.0244),
  reference then joint:

  https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/dnjggnwl

  https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/qbpt86xb

- **Expression head: the +0.056 (t 2.93) is a launch-failure artifact.** Excluding the four J_expr runs
  that never left the plateau (`hyasw3nx`, `m13meldl`, `c1n1jmp3`, `o3zk5y0i`), the contrast is `-0.0010`
  (t -0.10) on the two clean pairs. One never-launched run and one that did launch:

  https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/hyasw3nx

  https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/hzca60i6

- **And the two expression arms are not the same head.** `gh_expr_008_arm.sh` points the MASKED
  `per_gene` head (reveal schedule `[0,10,100,1000]`, its own z-score) at the expression label for
  `J_expr`, while the joint arm's expression head is `per_gene_aux`, which the config states is "never
  revealed". Add the addendum's step-count confound (117 steps/epoch on the union store, ~11 labeled
  rows per batch of 32) and the expression contrast differs in store, steps, masking and head, not in
  "the active heads" as the config header asserts. The proteome contrast has its own smaller version:
  `J_ref` uses `require_modalities [protein_abundance]` (3,581 train rows, 112 steps) and `J_joint`
  uses `[]` (3,726 rows per the addendum, 117 steps), so 4 % more rows and steps, unlabeled for the
  proteome.

- **A floor neither arm clears.** On TEST at the best-validation checkpoint, all three v14 proteome
  checkpoints read 0.067 to 0.082 against the parameter-free B3 kNN on ProtT5 at 0.083 to 0.092 on the
  same strains (`variance_stratified_pearson.md`, 2026-09-17). A joint-vs-single contrast between two
  arms that both lose to a free baseline on test is not a publishable synergy claim whichever way it
  goes. Every design below carries the B3 floor as a gate.

- **The two panels have opposite variance slopes**, which changes what readout to pre-register.
  Expression: per-gene Pearson rises with train variance, 0.183 on the quiet half to 0.329 on the top
  percentile, and its ceiling rises too (0.755 to 0.84). Proteome: it *falls*, 0.136 quiet to 0.073 to
  0.090 loud, and the duplicate-strain ceiling falls with it (0.50 to 0.27) because the proteins that
  move most replicate least. A pooled unweighted mean over features is the wrong instrument on the
  proteome, and the per-protein reliability weights to fix it already exist.

**Summary of the state of the claim: nothing about synergy is established, one thing is directionally
established against it (proteome side, 6 of 6 negative on `roll_max`), and the one positive number is
an artifact of a control arm that failed to launch 4 times in 6 for a reason the addendum traces to
batch composition rather than to the absence of a second label.** Prior odds are unfavorable: the only
same-model joint-versus-single test in the literature is null (totalVI), the only measured pairing ladder
values the pairing at ~0.028 (Ryu), and the MTL default against an equal-capacity control is negative
(Standley).

---

## Proposed experiments

Ranked by evidence per GPU-day inside 14 days. Power throughout uses the addendum's pooled
paired-difference sd 0.0167 (MDE at 80 % power, alpha 0.05: 0.019 at 6 pairs, 0.0135 at 12, 0.0110 at
18, 0.0096 at 24). Throughput: the joint store runs ~117 steps/epoch, and v16 did 500 epochs in ~30 h
at 3 runs/card, so **1,200 epochs is ~3 card-days for 3 runs = ~1 run-day per run**. All GPU rounds use
the adopted template (`expression-fit-review.md`, report 10): 4 split seeds x 3 init seeds, 1,200 fixed
epochs, decision statistic the **mean of `val/<pheno>/pearson_per_feature` over epochs 1,000 to 1,200**
(`roll_max` reported as a descriptive column only), paired t on the 12 (split, seed) differences,
closing test read at each arm's best-val checkpoint. Launch failures are excluded by the addendum's
`pred_sd_ratio` rule and **reported as a separate count**, never absorbed into a mean.

### E0 (CPU, ~1 day, zero GPU) -- pre-register the readout, and fix the instrument

Three readouts, defined before any GPU run, all scored on the existing prediction dumps:
(a) pooled per-feature Pearson, as now; (b) **reliability-weighted** per-feature Pearson with per-protein
weights from route W and per-gene weights from the Kemmeren-Sameith test-retest (both already in
`proteome_ceiling_replicate.json` and `expression_ceiling_replicate.json`) -- this is the readout the
opposite variance slopes demand; (c) **stratified** readouts on strata that already exist on disk:
Kemmeren responsiveness (700 responsive / 783 non-responsive, `results/kemmeren_responsiveness.json`),
per-gene train-variance decile, and the biology strata whose decoupling the literature predicts. Of
those, two are derivable in-repo from the GO DAG the graph builder already loads
(`$DATA_ROOT/data/go/go.obo` plus SGD annotations): **GO "protein-containing complex" membership** and
**"structural constituent of ribosome"**, which is the exact term Messner used for its paralog analysis.
A **protein-half-life** stratum would need an external table (Messner's ref 57) that is not in the
mirror; source it with a provenance record or drop the stratum, do not approximate it. Decision rule:
(b) is the primary statistic for the proteome head and (a) for the expression head; (c) is reported for
both and is where a class-specific effect is allowed to be claimed. **Why first: it costs nothing, and
pre-registering it after seeing a joint result would be indefensible.**

### E1 (CPU, ~1 to 2 days, zero GPU) -- measure the transferable ceiling, and gate everything on it

**E1a, cross-modal conditional-mean oracle.** Extend `masked_conditioning_oracle.py` across modalities:
on the 1,350 shared strains, reveal `m in {1,2,5,10,20,50,100,500,all}` measured *proteins* of a
held-out strain and predict its transcriptome by the Gaussian conditional mean with the ridge fit on
train strains only; and the reverse direction. Score with `pearson_per_feature`, against the
within-modality oracle at the same `m` and the `m = 0` floor of exactly 0.
**Decision rule:** if the `m = all` cross-modal value is below ~0.20, the per-deletion pairing channel
is bounded at that, and a genotype-only trunk that reaches 0.11 on the proteome can transfer at most a
fraction of it. Publish the curve either way; it is the honest upper bound on every joint claim and it
is the same instrument the strand already uses for within-modality conditioning.

**E1b, cross-modal perturbation embedding in the CPU baselines -- the Ahlmann-Eltze intervention.**
Add two perturbation representations to `expression_baselines_split.py` beside ProtT5, CaLM and the
random-1024 control, on the same four partitions: the deleted gene's **Messner proteome response vector**
as `p_b` for the expression task, and its **Kemmeren mRNA response vector** as `p_b` for the proteome task
(coverage 1,350/1,484 and 1,350/4,476). Plus a leakage-free variant in which `p_b` is the
*neighbor-averaged* response of the k nearest OTHER deleted genes in that modality, so no measurement of
strain `b` enters. **Framing discipline, load-bearing:** the direct variant conditions on a measured
phenotype of the same strain and is therefore an *oracle / conditioning* result in the `sec:oracle`
category, not genotype-only; only the neighbor-averaged variant is genotype-only. Separate columns,
separate names.
**Decision rule:** if the neighbor-averaged cross-modal embedding beats ProtT5 by more than the study's
0.02 seed resolution under B2 or B3, the cross-modal information is present at first order and the GPU
round is justified; if it sits at the random-1024 floor (0.049 B2 / 0.073 B3 on expression), joint training
is being asked to find what a linear map on the same information cannot see, and that is worth stating as
the field's own diagnostic applied to ourselves.

### E2 (GPU, 48 runs, ~48 run-days = 16 card-days; `gpu` %3 plus cabbi, ~5 to 7 calendar days) -- the deconfounded four-arm round

The v16 contrast cannot be repaired by more seeds; the arms differ in store, steps/epoch, masking and
head. Rebuild it so they differ only in the active labels.

- **Arms (4):** `K_prot` proteome head only; `K_expr` expression head only; `K_joint` both;
  `K_prot_rows` proteome head only *with the 205 expression-only rows present and the expression head
  at weight 0*. Every arm on the `fig3_proteome` store with `require_modalities []`, so all four see
  the same 3,726 rows and 117 steps/epoch. Both single-label arms use the **same head form** as the
  joint arm's corresponding head (unmasked `per_gene_aux`-style, same z-score); mask schedule `[0]` for
  every arm, so masking is not a variable.
- **Partitions x seeds:** split seeds 0 to 3 x 3 init seeds = 12 pairs per contrast.
- **Epochs / packing / wall / partition:** 1,200 fixed, 3 runs per 48 GB card, ~3 card-days per
  card-triple; `gpu --array=..%3` (unlimited wall) for the bulk, cabbi for overflow since 1,200 epochs is
  ~3 days and inside its 5-day cap. Drop `K_prot_rows` to 2 seeds if cards are short, giving 40 runs.
- **Pre-registered statistic:** paired difference of the E0 primary readout, `K_joint - K_prot` on the
  proteome head and `K_joint - K_expr` on the expression head, on the 12 (split, seed) pairs, plus the
  `K_prot_rows - K_prot` row-control difference. Bonferroni over the two primary contrasts.
- **Decision rule (adopt the joint model):** paired mean > +0.02 with a Bonferroni-corrected CI
  excluding 0, at least 3 of 4 partitions positive, the test-side sign agreeing at the best-val
  checkpoint, **and** the joint arm clearing the B3 kNN test floor on the head in question. MDE at 12
  pairs is 0.0135, so +0.02 is detectable and a null is reportable as "measured, not resolved,
  MDE 0.014" rather than as "did not help".
- **Declare the capacity control in advance (Standley).** The joint arm and the single arms share one
  trunk at identical width and depth and take the same number of optimizer steps, so this round is the
  **equal-total-compute** control, the one under which MTL wins in Standley's Table 1. The
  **equal-per-task-capacity** control, under which MTL loses at every group size there, is a single arm
  trained at 2x the step budget; add `K_prot_2x` at 2,400 epochs on 2 seeds (8 extra runs) if cards allow,
  and if they do not, say in the write-up which control was used and which was not. Declaring this after
  seeing the result is how the same runs get reported with either sign.
- **Free mechanism instrumentation:** log `cos(g_prot, g_expr)` on the shared trunk each epoch (PCGrad's
  conflict definition) and `pred_sd_ratio` per head (Ahlmann-Eltze's collapse diagnostic, already logged).
- **What lets the PI say it provably:** both head contrasts positive past +0.02 with CIs excluding 0 and
  the row-control at zero. That is the only configuration in which "joint >= single on each label's own
  prediction" is a measurement rather than a confound.

### E3 (GPU, 12 runs, ~12 run-days = 4 card-days; ~2 calendar days) -- the shuffled-pairing null

The decisive control, with published precedent (Ryu et al., arXiv:2405.00838v3) and a published
expectation: there, scrambling the pairing within perturbation label cost only 0.107 -> 0.0794 of the
cross-modal signal. Cheap here because it is one extra arm on the E2 round.

- **Arm:** `K_joint_shuf`, identical to `K_joint` except the proteome label vector is **permuted across
  the TRAIN strains that carry both labels**, within responsiveness stratum so the marginal
  distribution of each protein and the per-strain response magnitude are preserved and only the
  strain-to-strain correspondence is destroyed. Validation and test labels are never permuted.
  A second permutation seed per (split, init) pair if cards allow.
- **Partitions x seeds:** the same 4 x 3, paired against the same `K_prot` and `K_expr` runs.
- **Epochs / packing / wall / partition:** as E2.
- **Pre-registered statistic:** the *interaction*, `(K_joint - K_prot) - (K_joint_shuf - K_prot)`, on
  the 12 pairs; and separately whether `K_joint_shuf - K_prot` itself differs from zero.
- **Decision rule, three outcomes, all publishable:** (i) joint gain present and shuffling removes it
  (interaction > +0.02, CI excluding 0) -> the model uses the paired measurement of the same strain,
  which is the strong claim; (ii) joint gain present and **survives** shuffling -> the second label is a
  structural regularizer on the gene representation, consistent with section 2.3, and the paper must say
  that instead of claiming pairing; (iii) no gain in either -> the measured null, stated with its MDE.
  **Outcome (ii) is the one Ryu's ladder predicts, and it is why this arm must run alongside E2 rather
  than after it**: without it, an observed gain will be written up as the strong claim and a reviewer who
  knows that paper will ask for exactly this control.

### E4 (GPU, 24 runs, ~24 run-days = 8 card-days; ~3 calendar days) -- cross-modal held-out prediction and modality dropout

Tests the capability a reviewer actually finds convincing: does the shared trunk let one modality's
supervision reach the other modality's held-out labels?

- **Arms (2, on top of E2's `K_joint` and the two single arms):** `K_drop`, joint training with
  **per-strain random modality dropout** at p = 0.5 on each head's loss mask, which is the standard
  missing-modality regularizer and is the arm that makes the next readout possible; and
  `K_joint_holdout`, joint training in which the proteome label is withheld from a pre-chosen 300 of the
  1,099 both-labeled train strains, so those strains are expression-only at train time.
- **Pre-registered statistic, cross-modal transfer:** for `K_joint_holdout`, the per-feature Pearson of
  the proteome head on those 300 *withheld-label* strains, against three baselines: (a) the same strains
  scored by `K_prot`, (b) the B3 kNN floor, and (c) **the own-modality proxy: that strain's own measured
  mRNA for the same gene**, which is the baseline the CITE-seq protein-prediction literature shows is the
  only honest one and which almost nobody uses. A positive difference over all three is a direct
  demonstration that expression supervision improved the proteome prediction of strains whose proteome was
  never supervised. **This is the one statistic in the whole program that is a clean cross-modal claim
  rather than a joint-vs-single difference**, and it does not depend on the E2 contrast resolving.
- **Decision rule:** paired over 4 splits x 3 seeds, adopt the cross-modal claim at mean > +0.02 with CI
  excluding 0, at least 3 of 4 splits positive, clearing both the B3 and own-mRNA floors, **and surviving
  projection of the slow-growth axis out of both prediction and target** (the analogue of the CITE-seq
  finding that most cross-modal signal is cell-type identity). Report the before and after.
- **Secondary:** `K_drop` vs `K_joint` on both heads, which prices modality dropout as a regularizer and
  is the arm that makes the model usable on the 3,127 proteome-only and 205 expression-only strains.

### E5 (GPU, 24 runs, ~24 run-days = 8 card-days; ~3 calendar days) -- transfer versus joint

The design the field's one positive result (Ahlmann-Eltze) actually supports, and the alternative mechanism
if E2 comes back null.

- **Arms (2):** `T_pre`, trunk pretrained on the proteome alone for 600 epochs then fine-tuned on
  expression alone for 600, total budget matched to the 1,200-epoch arms; and the reverse `T_pre_rev`.
  Compared against the matched single-label arms of E2. 4 splits x 3 seeds, paired; statistic and decision
  rule as E2, on the fine-tuned head's own label.
- **Why separately worth running:** sequential transfer cannot suffer the gradient-interference failure
  mode joint training can, so a positive `T_pre` beside a null `K_joint` is coherent and reportable ("the
  second modality helps as an initialization, not as a simultaneous objective"), while a null in both
  closes the question. Gate on E1b.

### E6 (CPU, ~0.5 day, zero GPU) -- gene-class stratified re-read of every existing arm

No new runs. Score the existing v14, v16 and v13 best-validation prediction dumps under the E0 stratified
readouts, adding the 92 aneuploid Messner strains as a positive-control stratum (dosage passes to protein
unbuffered, so a model that gets nothing there is failing on the easiest available signal). Report
per-stratum joint-minus-single differences with the same paired test, labeled as a re-read of an
underpowered round. **Value: the cheapest way to find out whether the v16 null is a pooled-average null
hiding a class-specific effect, and it runs while E2 is queued.**

### Budget and what is reachable in 14 days

E0 + E1 + E6 are CPU and finish inside the first three days at zero GPU cost. E2 + E3 as one launch is 60
runs (68 with the `K_prot_2x` capacity control), about 20 to 23 card-days, which at 3 concurrent `gpu`
cards plus intermittent cabbi is 6 to 8 calendar days. That leaves room for E4 *or* E5, not both.
**Recommended order: E0, E1, E6 immediately; E2+E3 as one launch; then E4.** E5 only if E1b is positive.

**The claim that is actually reachable in two weeks, stated honestly.** At 12 pairs the MDE is 0.0135, and
every arm effect measured at this trunk across v13 to v18 has been a few hundredths with CIs through zero.
A two-sided "joint >= single on BOTH heads past +0.02" is reachable only if the true effect is larger than
anything this trunk has shown, and the external priors are against it: totalVI's same-model test is null,
Ryu's pairing ladder values the pairing at ~0.028 with a *perfect* pairing, and Standley's equal-capacity
control makes MTL negative at every group size. What IS reachable, and what I would advise the PI to aim
at, is the pair of statements E1a/E1b and E4 produce: **a measured upper bound on what the paired channel
can carry, and a clean cross-modal held-out demonstration on withheld-label strains against the B3, the
own-mRNA and the growth-axis-removed floors.** Those are provable in the time available, they survive the
E3 null, and they are stronger evidence of a shared representation doing real work than a 0.02 difference
between two arms that both lose to a kNN on test.
