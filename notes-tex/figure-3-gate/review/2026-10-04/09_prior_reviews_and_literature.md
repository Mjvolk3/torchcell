# Reviewer 9 of 10: prior internal reviews and the literature (2026-10-04)

Read-only audit by an independent agent; hypotheses are labeled "Hypothesis (untested)", external claims carry (web) or (memory), and repository claims carry absolute paths.

Path legend: WT = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective ; E = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal

## 1. What the prior internal reviews recommended, and what happened to each

The two reviews are /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-fit-review.md (2026-09-17) and /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/review/2026-09-27-joint-review/. Status is checked against /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/, /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh and /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/.

### Done at budget

- **Hop-0 self-indicator and 2-hop propagation** (v17, `L_self` and `L_prop2`, 12 pairs): +0.012 with a CI through 0. Reviewer 04 says that under a corrected window `L_prop2` reads +0.020 (t 3.79). It was never adopted: v18 and v19 inherit from v13.
- **No-mask control** (v18 `Y_k0`): -0.012 (t -2.44). v19 and v20 dropped the mask anyway.
- **Context readout** (v18 `Y_ctx`): +0.005.
- **Graph-retrieval baseline B4** (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/graph_retrieval_baseline.json). On gene-disjoint strains the union of the nine graphs scores 0.177 on test. The CGT dumps score 0.097 and 0.134.
- **Head-matched joint round** (v19, 11 of 12 partitions, /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/joint_checkpoint_readout.json). Expression: K_joint minus K_expr is -0.001 (p 0.55). Proteome: K_joint minus K_prot is -0.017, with 1 of 11 partitions positive, so the non-inferiority test fails.
- **Cross-modal ridge oracle**: proteome to expression 0.226, expression to proteome 0.218 (the `triangle` entry in the same file).

### Done under budget

- **v20 conditioning** is partial at about epoch 159 on 2 splits. C_prot leads its control by +0.081 and +0.088 at matched epochs. C_expr leads by +0.010 and +0.034, but its permuted control also gained +0.019 on split 1. None of this is a result yet.

### Never run (the most valuable list)

- **Perturbation-derived embeddings of the deleted gene**, put through B2/B3. Candidates: the Costanzo SGA interaction profile, chemogenomic profiles, Ohya morphology, and the Messner proteome response. `EMBEDDINGS_FULL` in /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/expression_baselines_split.py contains none of them.
- **The neighbor-averaged cross-modal embedding**, the genotype-only form of the Ahlmann-Eltze intervention (05 E1b).
- **Graph adjacency as a node feature**, or a graph-kNN blend of the existing prediction dumps.
- **Directed TF masks** and a graph-off control.
- **Retiring `E_full`** to calm plus ProtT5. `E_full` is still pinned at /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_expr_v12_head.yaml:68.
- **Warmup plus dropout-0 plus higher learning rate** (`R_pert`), and the launch-hardening ladder (02 E4).
- **Huber or MSE-anchored loss**, and a within-batch per-gene correlation loss term.
- **`M_mag`**, the per-strain magnitude head. The oracle bound is +0.071 and the CPU pre-check was never done.
- **`K_perm`**, the shuffled-label joint null. The joint aux-weight ladder was also never run, and `J_joint05` was never launched.
- **The superset arm and the Standley 2x-compute control.**
- **Transfer versus joint training** (05 E5).
- **Off-axis Pearson and a per-strain discrimination metric.** A grep for either finds nothing. The same grep finds no `LaunchGuard`, so the launch check is still done by hand.
- **Reliability-weighted proteome loss and metric.**

## 2. What the field has established that bears on this task

- **Deep models do not beat simple baselines on unseen single perturbations.** "None of the deep learning models was able to consistently outperform the mean prediction or the linear model." The one intervention that consistently helped was pretraining the perturbation embedding on other perturbation data (Ahlmann-Eltze, Huber, Anders 2025, *Nat Methods*; quoted from the repo, /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/0-strand.tex:274).
- **State** matched the perturbation-mean baseline on genetic perturbation datasets (Adduri et al. 2025; quoted from the repo, /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/0-strand.tex:281).
- **Text and knowledge embeddings help a linear model.** A ridge on GenePT gene embeddings beats published deep models (GenePert, bioRxiv 2024) (web). TxPert reports out-of-distribution gains from combining several biological graphs (Valence Labs, 2025) (web).
- **Control-referenced Pearson credits the shared response axis** (Systema 2025, *Nat Biotechnol*; as summarized in agent-09). The campaign's per-gene metric already gives the mean baseline exactly 0. About 80% of the score survives projecting out the slow-growth axis (agent-09, section 0.3, /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/expression_fit_review_2026_09_17/agent-09-literature.md).
- **Transcription-factor regulation is local in yeast.** Only about 3% of genes a TF binds respond to deleting that TF (Hu, Killion, Iyer 2007, *Nat Genet*) (memory). This fits B4: adjacency through complexes and pathways predicts the profile, while TF-target edges are at chance (09-17 note).
- **No published work predicts the genome-wide Kemmeren profile for held-out deletions** (agent-09, section 1.3).
- **mRNA to protein across genotypes is weak within a gene.** Median Spearman is 0.165 (Teyssonniere 2024, *PNAS*), against about 0.5 across genes (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/review/2026-09-27-joint-review/05a_literature_helper_mrna_protein.md).
- **Jakobson et al. 2025, *Science*, maps natural variants to the proteome across isolates** (web). I found no published transfer from natural variation to engineered deletions (web search; absence not proven).

## 3. Where the documents or manuscript misstate or overstate the literature

- **The manuscript still carries placeholder correlations.** /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/paper/nature-biotech/sections/results.tex:129 (and /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/paper/nature-biotech/sections/outline.tex:49) gives r = 0.543 for expression and 0.619 for morphology. The measured best values are 0.238 and 0.082 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/8-directions.tex:172).
- **The introduction claims joint prediction.** /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/paper/nature-biotech/sections/introduction.tex:40 says CGT "predicts fitness, gene interactions, gene expression, and morphology jointly". v19 measured joint training as a null. Lines 30 to 33 say the structured labels justify a deep model; for expression, B4 contradicts that.
- **The strand document claims the model beats B2.** /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/0-strand.tex:224 says CGT "clears B2 by 0.093, 4.2 sd ... the opposite of what the published benchmark found". That rests on a roll-max, on the luckiest validation split. The gene-disjoint B4 comparison has since overturned it.
- **"Every published operator in this family is additive"** (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/0-strand.tex:238) rests on reading three papers. TxPert (web) and GEARS's cross-gene layer are counterexamples.
- **"Their linear model ... is B2 above"** (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/0-strand.tex:279) is not quite right. As I recall, Ahlmann-Eltze build the gene and perturbation embeddings from a PCA of the training expression (memory). B2 uses ProtT5, so the variant that actually won has never been run here.
- **The "separate capability" claim is overstated.** /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/8-directions.tex:155 says imputation is orthogonal to genotype. That was tested only against a ProtT5 kNN at 0.117, not against B4 at about 0.19 or against the CGT.

## 4. Methods worth trying in this regime, ranked

1. **Bilinear ridge (B2) with perturbation embeddings taken from other perturbation data**: the SGA profile, chemogenomics, the train-split co-expression loading, and the cross-modal neighbor average. This is the field's one consistent winner, it runs on CPU, and the data is already in the repo.
2. **Kernel ridge or multi-kernel regression** over graph diffusion kernels on STRING experimental and database edges, plus a sequence kernel. This is the posterior mean of a multi-task GP whose output covariance comes from training gene PCs (Bonilla et al. 2008, NeurIPS) (memory). It generalizes B4.
3. **Inductive matrix completion with side information on both axes** (Jain and Dhillon 2013, arXiv) (memory). This is B2 with gene-side features set to co-expression loadings.
4. **Fitness times the slow-growth axis, plus a residual.** O'Duibhir 2014 gives the axis vector (memory). The model already scores 0.50 on that axis.
5. **Natural-variation transfer from Caudal 2024.** Not feasible in 20 days: the genotype input drops 133 gene absences per isolate (bug #71, /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.018-natural-isolate-genomics.expression-modeling-setup.md:168), and the bidirectional transfer was never run.

## 5. Top three recommendations for under 20 days

**R1. A ceiling-anchored benchmark as Figure 3.**

- *What*: CGT against B0, B2, B3, B4 and a random-embedding null, on 12 partitions, gene-disjoint test strains, with the 0.775 and 0.42 to 0.61 ceilings drawn in.
- *Metrics*: per-gene Pearson across strains, per-strain Pearson across genes, a discrimination rank, and NMSE.
- *Cost*: CPU plus test reads of the v19 K_* checkpoints, about 3 days.
- *Hypothesis (untested)*: CGT ties B4.
- *Reviewers will ask*: why use a transformer at all. The honest answer is that this is a benchmark result, and the figure should say so.

**R2. Perturbation-embedding gate, then one GPU arm.**

- *What*: B2/B3 with the SGA, chemogenomic and cross-modal-neighbor embeddings against ProtT5. Pre-registered rule: beat ProtT5 on at least 10 of 12 partitions. If one passes, feed it to CGT as the deletion's input, 12 partitions by 1 seed, about 4 card-days.
- *Hypothesis (untested)*: the SGA profile lifts B2 above 0.13.
- *Reviewers will ask*: about leakage through the growth axis. Report the off-axis score.

**R3. Finish v20 as a conditioning panel.**

- *Control*: the permuted arms, plus the ridge references of 0.226 and 0.218.
- *Hypothesis (untested)*: C_prot exceeds K_prot but does not beat the ridge.
- *Reviewers will ask*: whether the network adds anything beyond a linear conditional mean.

**Framings to avoid.** A data-scaling argument will be attacked, because the perturbation axis is fixed at 1,484 points and the baselines would scale too. Any "virtual cell predicts the transcriptome" headline will also be attacked.

Example v20 conditioned run (C_prot, split 0):

https://wandb.ai/zhao-group/torchcell_019_prot_v20/runs/7tiaq7pe

Web sources: [Jakobson 2025, Science](https://www.science.org/doi/10.1126/science.adu3198), [GenePert](https://www.biorxiv.org/content/10.1101/2024.10.27.620513v1), [TxPert](https://www.valencelabs.com/publications/txpert-leveraging-biochemicalrelationships-for-out-of-distributiontranscriptomic-perturbation-prediction/)

Files:

/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-fit-review.md
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/joint_checkpoint_readout.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/graph_retrieval_baseline.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/expression_baselines_split.py
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/0-strand.tex
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/paper/nature-biotech/sections/results.tex
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/paper/nature-biotech/sections/introduction.tex
