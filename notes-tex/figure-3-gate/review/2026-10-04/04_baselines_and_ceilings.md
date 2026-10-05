# Reviewer 4 of 10: baselines, ceilings and oracles (2026-10-04)
Read-only audit by an independent agent; every unmeasured statement is labeled as a hypothesis.

Path prefix used below: `R = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/`.

**1. ESTABLISHED**

**Answer to the deciding question.** On matched partitions, genes, strains and scoring rule, no CGT result on file beats the best baseline.
- **Expression:** CGT is above the ProtT5 kNN but below the graph-keyed kNN.
- **Proteome:** CGT is below both.
- **Morphology:** no matched comparison exists.

Source: `/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/graph_retrieval_baseline.json` (`rounds.{v13,v14}.per_seed.*.keys`, `.cgt`). Statistic is test per-feature Pearson (pf), all strains, about 155 (expression) or 447 (proteome) per seed. The CGT is the best-val-epoch checkpoint (`/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/eval_ckpt_manifest_v13.tsv`, `/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/eval_ckpt_manifest_v14.tsv`). Baseline k was chosen on val.

| split | CGT run: test | best graph key on val: test | physical: test | ProtT5 kNN: test | random key: test | B2 ProtT5: test | own-profile oracle |
|---|---|---|---|---|---|---|---|
| expr s0 | V_ref wq8y8nd5 0.110; V_concat bn37i9vs 0.139 | 0.165 | 0.204 | 0.075 | 0.055 | 0.085 | 0.656 |
| expr s2 | V_ref lp6guytz 0.153 | 0.174 | 0.164 | 0.150 | 0.046 | 0.111 | 0.662 |
| prot s0 | P_ref uc0pm2pv 0.067; P_concat bc8bngdr 0.082 | 0.117 | 0.084 | 0.095 | 0.007 | 0.059 | 0.539 |
| prot s2 | P_ref 25ok3tce 0.082 | 0.111 | 0.092 | 0.090 | -0.011 | 0.064 | 0.548 |

- **Margins, CGT minus best-on-val key:** -0.055, -0.025, -0.021 (expression) and -0.050, -0.035, -0.029 (proteome). That is 6 of 6 negative, with only n=2 split seeds per modality. The paired summary gives union minus CGT V_ref at +0.051 ± 0.022 and P_ref at +0.045 ± 0.015 (`summary.paired`).
- **Strain-level uncertainty is not computable from the file** (no bootstrap). CGT test dumps exist for only 6 of 40 checkpoints.
- **Validation tells the same story.** At the registered window, CGT reads 0.188, 0.148, 0.131 and 0.128 on expression splits 0 to 3, against physical-graph kNN val of 0.189, 0.215, 0.141 and 0.175. On proteome, CGT reads 0.088, 0.076, 0.089 and 0.064 against STRING-experimental kNN val of 0.125, 0.102, 0.121 and 0.097 (`/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v13_split_readout.json` and `/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v14_proteome_readout.json`, `runs[].window_mean`, 4 runs per split).
- **Graph keys carry real signal.** Union minus its rewired graph is +0.150 ± 0.007 on expression and +0.115 ± 0.011 on proteome. Union minus ProtT5 is +0.054 and +0.026. All are n=4 and 4 of 4 positive (`summary.paired`).
- **Sequence representations.** ProtT5 leads. On expression, B3 ProtT5 test is 0.119 ± 0.051 (n=4). The four-embedding stack the model consumes reads 0.091, CaLM 0.100, and the regulatory-DNA representations 0.01 to 0.05. Random-1024 reads 0.073 on val and 0.018 on test. On proteome, B3 ProtT5 test is 0.078, the stack 0.067, and ProtT5+CaLM 0.081 (`/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/expression_baselines_split_full/`, `/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/baselines_split_fig3_proteome_full/`). So the stack adds -0.03 to +0.003 over ProtT5 alone.
- **Ceilings.**
  - Expression: cross-study test-retest gives 0.775 (82 deletions; `/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/expression_ceiling_replicate.json`).
  - Proteome: HIS3 wild-type-replicate route gives 0.614, the duplicate-strain route 0.417 (149 pairs; `/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/proteome_ceiling_replicate.json`).
  - Morphology: wild-type-replicate route only, 0.611 (122 replicates; `/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/morphology_noise_ceiling.json`).
  - Low-rank output space is not limiting: rank 64 reaches 0.78 (`/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/lowrank_output_ceiling.json`).
- **Oracles versus genotype.**
  - Masked conditioning: m=10 gives 0.408, m=100 gives 0.676, m=1000 gives 0.793 (val, 5 draws; `/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/masked_conditioning_oracle.json`).
  - The gain survives a ProtT5 kNN: 97.5 to 100.6% retained (`/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/conditioning_gain_after_genotype.json`).
  - Cross-study retention is 0.51 to 0.62 (`/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/cross_study_conditioning_oracle.json`).
  - Genotype-only best on test is about 0.19 (expression) and 0.12 (proteome), against an own-profile oracle of 0.65 and 0.54.

**2. NOT ESTABLISHED OR CONTRADICTED**

- **The first-pass figures came from mismatched statistics.**
  - "kNN 0.220" is the mean val of the best graph key, selected over 132 cells.
  - "linear 0.113" is the B2 all-sequence val mean, selected over 1,680 cells per seed.
  - "CGT proteome 0.099" is a mean of the per-split `roll_max` values.
  - All of these are validation order statistics. On test the ordering reverses for the proteome.
- **Morphology "CGT 0.082 vs kNN 0.100" is unpaired.** The CGT figure is the `roll_max` of one run (vsceij2v, 1,161 training strains, epoch 27). The kNN figure is the chromatin-pathway key on 3,757 training / 469 val strains, seed 0, chosen among about 114 cells, with no test (`/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/knn_embedding_probe.json`). The chromatin-pathway key collapses on expression test (0.004).
- **No learning curve in training strains exists for any baseline.** The only data point is the fold-90 arm: B2 +0.002, B3 +0.011 on val (`/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/expression_baselines_split/seed0_fold90.json`).
- **Hypothesis (untested):** the STRING coexpression channel may contain Kemmeren-derived profiles, which would leak into the expression graph keys. The physical-graph key is not subject to this and still beats the CGT on expression.

**3. ERRORS AND INCONSISTENCIES**

- **Wrong protein count.** `/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/3-baselines.tex` line 157 says "1,536 proteins". The results files say `n_gene` = 1850.
- **Wrong statistic labelled "matched epoch".** Line 192 gives v14 at 0.114, 0.083, 0.112 and 0.085. These are per-split `roll_max` means; the window means are 0.088, 0.076, 0.089 and 0.064. The text's "margin 0.00 to 0.04" over the floor is contradicted on test.
- **Mismatched comparison in `/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/2-expression.tex` around line 534.** It sets the CGT's 0.196 against a kNN of 0.117 scored on a 140-strain subset, and omits the graph kNN.
- **Search grids are truncated, so baselines are understated.**
  - B2 selected ridge=100, the grid maximum, in 82/112 expression and 81/112 proteome cells.
  - B3 selected k=25, the grid maximum, in 66/112 proteome cells.
  - Graph retrieval with a k grid up to 50 lifts ProtT5 kNN on proteome from 0.078 to 0.090.
- **The "selection mirrors the model's" claim (`3-baselines.tex`) is false.** B3 searches 5 cells; the CGT checkpoint is the maximum over roughly 5,000 epochs. V_ref_s0 reads 0.223 on val and 0.110 on test.
- **Bad entries in the kNN probe.** `/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/knn_embedding_probe.json` has the `*_no_dubious` morphology arms and `one_hot_gene` as NaN.
- **Ceilings computed from a biased numerator.** The proteome and morphology ceiling files compute "fraction realized" from `roll_max`. The expression file still carries route A with a fraction of 1.79.

**4. MISSING BASELINES (ranked for Figure 3)**

1. Test-prediction dumps for all 40 v13/v14 checkpoints, scored in `graph_retrieval_baseline.py`.
2. A morphology baseline set (B2 ridge, ProtT5 kNN, graph kNN) on the CGT's own 1,161-strain partition, and on the full 4,718.
3. A graph-plus-ProtT5 composite key and a ridge on adjacency rows.
4. Baselines on the v19/v20 store that carries both labels.
5. A kNN that conditions on a measured modality, compared against C_prot.
6. A duplicate-strain or cross-study ceiling for morphology.

**5. TOP THREE RECOMMENDATIONS (under 20 days)**

1. **Score every existing checkpoint on test against the graph baselines.**
   - Experiment: dump predictions for all 40 checkpoints (GPU inference only), and fix the score to the registered-window epoch rather than best-val.
   - Control: the graph kNN, ProtT5 kNN and rewired keys on the same strains, plus a strain bootstrap.
   - Cost: about 2 days.
   - Stop: if the CGT is below the graph kNN on at least 3 of 4 splits per modality, Figure 3 reports the CGT as a graph-retrieval-level predictor.
2. **Widen the baseline search grids, then add composite keys.**
   - Experiment: ridge up to 1e4, k up to 200, and a graph+ProtT5 key; also a learning curve at 25/50/75/100% of training strains, 4 splits, CPU only.
   - Hypothesis: the baselines rise by 0.01 to 0.02.
   - Control: random-key and rewired-graph floors.
   - Cost: 2 days.
   - Stop: once the curves are monotone, or plateau at 100%.
3. **Build the morphology triad on matched partitions.**
   - Experiment: B2, ProtT5 kNN and graph kNN on the CGT partition, and on the 4,718-strain build with one CGT run per split.
   - Control: train-mean and random keys.
   - Cost: about 5 days.
   - Stop: morphology test pf for every arm, with n ≥ 4 splits.

Files:
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/graph_retrieval_baseline.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v13_split_readout.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v14_proteome_readout.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/eval_ckpt_manifest_v13.tsv
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/eval_ckpt_manifest_v14.tsv
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/expression_baselines_split_full/
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/baselines_split_fig3_proteome_full/
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/knn_embedding_probe.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/proteome_ceiling_replicate.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/morphology_noise_ceiling.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/graph_retrieval_baseline.py
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/3-baselines.tex
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/2-expression.tex
