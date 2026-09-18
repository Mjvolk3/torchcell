# Would the metabolic module help the KO expression fit?

Agent 07, 2026-09-17. Read-only pass over the flux/metabolic module code, the 026/027/028
record, the 019 metabolism configs, and three new CPU measurements I ran on the Kemmeren
matrix. Every number below traces to a file, a W&B run, or a script I executed; anything
unmeasured is labeled.

**Short answer: no differentiable flux layer. Test the cheap input-feature form, and test
it on CPU in the baseline harness before any GPU run, because the one cheap metabolic
encoding that has already been scored on this exact task reads at the random-control
level.**

---

## 0. What the "metabolic module" actually is, and what 019 already did with it

Three different things get called "the metabolic module" and they must be kept apart.

**(a) The differentiable flux layer (Track B).** `torchcell/metabolism/flux_layer.py`
(219 KB, `class FluxLayer`, line 1613) plus `constraints.py`, `parameters.py`,
`thermo_pipeline.py`, `transport.py`, and the newer `lp_layer.py`. **These files do not
exist on this branch.** They live only in
`/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/torchcell/metabolism/`.
This worktree's `torchcell/metabolism/` holds only `enzyme_kinetics.py` and `yeast_GEM.py`.

**(b) The metabolic READOUT heads (Track A).**
`torchcell/models/cell_graph_transformer_metabolism.py` (572 lines, present here). Its own
docstring says it plainly: "Track A -- the SIMB demonstration -- activates only the readout
heads; no flux layer is involved," and "TRACK B (not implemented here)". It subclasses the
019 `CellGraphTransformer` and adds three heads (betaxanthin scalar, beta-carotene scalar,
mulleder19 vector).

**(c) The FBA SOLVE used as an input feature.** Not a layer at all: cobrapy/HiGHS run
offline, per deletion, and the resulting numbers concatenated onto the model's inputs or
its readout. This is 028 round 5 onward.

**The 019 metabolism configs are (b), not (a).** `experiments/019-simb-multimodal/conf/gh_metabolism_00{0,1,2,3}.yaml`
carry the `[[plan.cgt-metabolism-flux-layer.2026.07.26]]` header comment but contain no flux
anything. `grep -rn flux experiments/019-simb-multimodal/` returns only that plan link, the
ARO4/ARO7 "push tyrosine flux" prose, and one figure-board string. `_000` is the nine-graph
port with `node_embeddings: [prot_T5_all]` and `active_heads: [betaxanthin, mulleder19]`;
`_001` changes checkpoint selection from loss to metric; `_002` swaps point-MSE for Gaussian
CRPS; `_003` is the two-graph ablation. So **metabolism has never been attached to the 019
expression model in any form.** What 019 measured is whether the 019 model can predict
metabolic phenotypes, which is the opposite direction from the question here.

The one 019 result that does bear on the question is the auxiliary-head contrast, and it is
negative. `experiments/019-simb-multimodal/results/bx_aa_paired_summary.json` (12 cells, 46
runs, betaxanthin with vs without the 19-amino-acid metabolome head attached, paired within
(setting, seed)): **fair-exposure mean delta -0.0181, sd 0.0312, se 0.0099, n = 10, 4/10
positive; live subset -0.0279, sd 0.0377, n = 5, 2/5 positive.** Attaching a metabolic
auxiliary head cost prediction on a metabolic target. That matches the 019 phenotype-strand
retrospective (metabolome head costs -0.0265) and 027 wave 6 (amino-acid flux heads -0.039,
sd 0.007, n = 3, all seeds agreeing).

---

## 1. What has the flux module measurably done on any readout?

### 1a. Betaxanthin (026, 027): nothing distinguishable from noise, in either direction

- **026, 98 runs across 4 GPUs (jobs 1576-1579).** 24-run label-permutation null: mean peak
  0.0962, sd 0.0441, 95th pct 0.1545. **No arm clears it.** Best is `flux_anchored`
  0.1151 +/- 0.0095 (p = 0.28); the smallest p across all 22 configurations is 0.24. The null
  MEAN exceeds the previously banked `flux_anchored` 0.0867, `flux_free` 0.0833 and `pooled`
  0.0592, so the old arm ordering was reporting permuted-label noise.
  (memory `026-flux-arm-scores-are-noise`)
- **027 wave 6, box head at parity vs pooled:** paired difference **-0.017, sd 0.046, n = 6,
  t = -0.93**. Pooled itself spans 0.00 to 0.17 across seeds; between-seed spread dominates.
  Wave 7 on the shipped kcat table: **-0.055, sd 0.090, n = 3**.
- **Projection head "2/2 above pooled" did not replicate:** +0.045, +0.035, +0.104, -0.043,
  -0.136 -> n = 5 mean **+0.001, sd 0.093**.
- Live W&B, `torchcell_027_bxfx`, `test_spearman_pinned` at the validation-selected epoch:

  https://wandb.ai/zhao-group/torchcell_027_bxfx/runs/za1p0euu

  https://wandb.ai/zhao-group/torchcell_027_bxfx/runs/vr4t8xef

  (`pooled-s2701` 0.0905 vs `flux_anchored-s2702` 0.0163; also `flux_free-s2702` 0.0843,
  `pooled-s2702` 0.0603, `flux_off-s2701` 0.0962. Null SE on this n is ~0.040, so every one
  of these is inside one null width of every other.)
- **The target itself is null.** FBA baseline on the cassette GEM at YPD reads +0.047
  (permutation p 0.24); FCL's +0.039 is z = 0.98 against null SE 0.040. MDE at n = 3 is
  0.073. A cosine-kNN over `prot_T5_all` beats the whole CGT on it (+0.145 vs pooled 0.115).
  (memory `metabolic-review-2026-09-14-findings`)

### 1b. Amino acids (027 wave 6/7): the flux head COSTS

Box head **-0.039 (sd 0.007, n = 3, 3/3 negative)**; projection head **-0.039 (sd 0.034)**.
Constraints on collapse the head at epoch 8. The one positive, the amino-acid CONCENTRATION
readout, is **+0.0145 (sd 0.0093, n = 3)** and the wave-8 audit found it **never touches the
flux**: `thermo_log_c` is a sigmoid of the pooled context, no projection cell trained its
latent sigma, "the concentration readout reads context only, the flux never enters it."
The explicit context-only control (`ctxconc`, jobs 1899/1930/1931) read paired mean **+0.002**
and met the pre-registered kill rule: **the module is OFF for amino acids.**

### 1c. Essentiality (028): this is where the module's value was located, and it is not the layer

All numbers `test_auroc_all` on the 195 held-out FCL genes, seeds 2700/2701/2702.

| arm | what it is | s2700 | s2701 | s2702 | mean |
|---|---|---|---|---|---|
| `w14b32posw200` | pooled backbone, no module | 0.7801 | 0.7687 | 0.7732 | 0.774 |
| `w15gate_s0` | learned flux layer, gated readout | 0.7661 | 0.7821 | 0.7648 | 0.771 |
| `w15shuf_s0` | **same layer, kcat table SHUFFLED** | 0.7846 | 0.7811 | 0.7722 | 0.779 |
| `w16fba` | offline FBA growth ratios as an affine INPUT | 0.8137 | 0.8055 | 0.8436 | **0.821** |
| `w17gatefba` | learned layer ON TOP of the FBA affine | 0.8098 | 0.8214 | 0.8318 | 0.821 |
| `w20chromprotfba` | FBA affine + chrom_pathways + ProtT5 embeddings | 0.8600 | 0.8662 | 0.8877 | **0.871** |
| `w26lpbio` | forced LP layer on the anchored box | 0.500 | 0.500 | 0.500 | 0.500 |

https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/pcexyjyr

https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/jf3af715

https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/5swszmhc

https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/25w5bino

https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/qk9s45z6

https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/zoh1r1yw

Four readings, all of them load-bearing:

1. **The learned layer is null** (0.771 vs 0.774 pooled).
2. **Shuffling the kcat table does not hurt** (0.779 >= 0.771). The layer's kinetic content is
   unread. This is the cleanest negative control in the whole metabolic line.
3. **The offline solve is worth +0.047**, 3/3 seeds.
4. **The layer adds nothing on top of the solve** (`gatefba` 0.821 == `fba` 0.821), and the
   LP layer on the anchored box is **infeasible for every genotype** (all cells exactly 0.500,
   `flux_active` 0.0): "the anchored box contains no growing flux."

And the 028 doc's own summary line: **THE INPUTS ARE THE NUMBER.** An MLP with no graph at
all, on ProtT5 + chrom_pathways + 4 FBA ratios, reads 0.870/0.872/0.869 **in 6 s of CPU** --
identical to the transformer's 0.871. The reported 0.893 is a 5-fold ensemble, and matched
MLP/GCN ensembles read 0.886 and 0.888. The 1-WL test says structure carries no essentiality
signal at all.

### 1d. Feasibility: the layer has never reached the polytope

- Best flux balance lands **within 8 epochs at every penalty weight** and degrades after;
  the cell with NO physics term in its loss is the most feasible at its selected epoch
  (`penalty_ladder.py`, jobs 1986/2001/2026/2027). Early stopping, not the penalty, is what
  buys feasibility.
- `balance_max_ratio` is **exactly 2.0** in every sign-clamp, boxforce, force-end and box cell
  of waves 6 and 7. Under an honest max or turnover-thresholded rule the Genesis feasibility
  criterion has **never been met in 18 audited cells**.
- Whichever constraint is made exact, the other is grossly violated. 028's conclusion, after
  rounds 7, 10, 15 and 18: "the K-alternation layer never reaches the feasible polytope."
- Prediction spread collapses far more often with the layer than without (betaxanthin:
  context-only spread ratio 0.37-0.46, every flux readout 0.06 down to 0.0000; probe job 2029
  measured the penalized cell's flux varying **2.5e-7 of flux_scale across 933 genotypes**, so
  the head is starved).

**Summary of (1): across betaxanthin, amino acids and essentiality, the differentiable flux
layer has produced no measured gain on any readout, and on amino acids a replicated cost.
The only measured gain in the whole line is the OFFLINE FBA SOLVE used as an input feature on
gene essentiality, +0.047 AUROC.** Note also what "no gain" does NOT mean: 026's own memory is
careful that an experiment that cannot resolve 0.05 cannot resolve a small real effect either.
The honest statement is "measured and null at the achievable power," not "proven harmful"
(except on amino acids, where 3/3 seeds agree on a cost).

---

## 2. The mechanistic argument, the measured correlation, and the causal direction

### 2a. The argument in its strongest form

Hypothesis (this is the case FOR, stated as strongly as I can make it): a deletion perturbs
metabolite pools; pools are sensed by Gcn4/Leu3/Rtg1-3/Hap4/Snf1 and the general amino-acid
control system; those regulators rewrite the transcriptome. So flux state is a compressed,
low-dimensional latent that explains a chunk of the response, and a differentiable flux layer
supplies it as a structured bottleneck the model would otherwise have to learn from 1,244
training strains.

### 2b. The causal direction is the wrong way round for a layer in the loop

The user's framing is right and it is not a technicality. FBA computes a steady-state flux
distribution **given** enzyme bounds, which are set by enzyme abundance, which is set by
expression. In the forward direction expression -> flux, an FBA layer is a deterministic
function of things the model must already predict; it adds no information the encoder does
not have. In the reverse direction flux -> expression, the mechanism is entirely regulatory
(a transcription factor reading a pool), and **no constraint-based model contains any of that
regulation.** Yeast9 has no transcription. So a flux layer in the loop can only inform
expression through a path the model would have to learn anyway, from the same data, with an
extra 4,131-dimensional non-convex optimization in the way.

There is a second, sharper version: the 028 gain works because **essentiality IS the FBA
objective.** FBA's output variable (growth) is literally the label. For expression, FBA's
output variable is not the label, is not a component of the label, and the map from it to the
label is the regulatory network, which the GEM omits.

This is unmeasured as stated -- I am labeling it. **Hypothesis (untested):** no FBA-derived
quantity predicts KO expression beyond what gene identity already predicts. But it is the
reading most consistent with (1) and with 2c.

### 2c. What the measured expression-vs-metabolism correlation actually says

From `experiments/028-knockout-expression/results/expression_metabolic_yko_correlation.json`
(document `notes-tex/028-knockout-expression-metabolic/`), ridge out-of-fold Pearson, whole
profile -> one metabolite, 5 folds, null = 20 label permutations:

| source | Mulleder 19 aa, median | n | betaxanthin | Cooper 17 |
|---|---|---|---|---|
| Messner protein | **0.409** (19/19 > null) | 4,400 | 0.429 | 0.082 |
| Kemmeren mRNA | **0.179** (18/19) | 1,416 | 0.346 | 0.096 |
| Nadal mRNA | 0.097 (17/19) | 2,113 | 0.411 | 0.011 |

Three things follow, and none of them supports a flux layer in the expression model:

1. **Every measured number is in the direction expression -> metabolite.** The agent that read
   the whole document and the 882-line script confirms: `_multivariate(A, y)` fits features ->
   metabolite only; the Mantel read is symmetric by construction. **There is not one number
   anywhere in the repo in the direction metabolite-or-flux -> expression, and not one
   FBA-derived feature anywhere in the 028-knockout-expression line** (no cobra import, no flux
   vector, no pathway covariate, no residual decomposition).
2. **The document's own mechanistic sentence points the same way:** "A strain whose proteome
   carries less of an amino acid's own synthetase or transaminase carries more of the amino
   acid, which is the direction a pool-and-flux picture predicts." That is enzyme abundance
   setting the pool. Expression -> flux.
3. **The mRNA read is half the protein read** (0.179 vs 0.409), and the single features
   carrying it are enzymes (`YHR020W` prolyl-tRNA synthetase vs proline, r = -0.444, n = 4,400;
   ARO2, THR1, ALT1, SHM2, KRS1). So what the metabolic phenotype shares with expression is
   *the abundance of the pathway's own enzymes* -- information the model already has direct
   access to as gene identity, not a flux computation.

A fourth point matters for anyone tempted to read 0.41 as a green light: the document itself
flags that **the plate-aware null has not been run.** Mulleder and Messner are one lab, one
prototrophic collection, both plate-by-plate, and permuting deletion labels permutes plates.
The 0.41 is an upper bound until that is done.

---

## 3. The cheap analog: FBA features as INPUTS. Cost, and the prior for a gain

The user's framing here is the right one, and it is exactly what 028 converged on. But there
are two measurements that bear on it directly, and both are discouraging.

### 3a. The cheap encoding has ALREADY been scored on this exact task, and it reads at control level

`experiments/019-simb-multimodal/results/baselines_embedding_study.json` (written by
`scripts/baselines_embedding_study.py`, 4 split seeds) scores 28 gene representations on the
Kemmeren expression task with B2 (bilinear ridge from the deleted gene's embedding to the
6,127-gene profile) and B3 (kNN in embedding space). `normalized_chrom_pathways` -- the **exact
197-d encoding that carries the 028 round-9 gain** (5 scaled gene properties + 17-way
chromosome one-hot + ~175 pathway multi-hot columns) -- is in the table:

| embedding | dim | B2 val | sd | B3 val | sd | B2 test | B3 test |
|---|---|---|---|---|---|---|---|
| `prot_T5_all` | 1024 | 0.1055 | 0.0258 | 0.1200 | 0.0095 | 0.1101 | 0.1185 |
| `calm` | 768 | 0.0892 | 0.0104 | 0.1006 | 0.0233 | 0.0922 | 0.0998 |
| `codon_frequency` | 64 | 0.0711 | 0.0189 | 0.0787 | 0.0143 | 0.0522 | 0.0620 |
| **`normalized_chrom_pathways`** | **197** | **0.0530** | **0.0364** | **0.0329** | **0.0212** | **0.0199** | **0.0024** |
| `random_1024` (control) | 1024 | 0.0493 | 0.0176 | 0.0725 | 0.0182 | 0.0135 | 0.0179 |
| `random_100` (control) | 100 | 0.0087 | 0.0174 | 0.0592 | 0.0202 | 0.0110 | 0.0373 |

The pathway encoding sits **inside one sd of `random_1024` on B2 val and BELOW it on B3**.
Contrast the same encoding on essentiality, where the ladder reads `chrom_pathways` logistic
**0.830** against one-hot **0.506** and random vectors at chance -- the strongest rung there.

Caveat I will not skip: B2/B3 test the encoding **alone**, and 028 round 14 showed the pathway
columns "carry the round-9 gain, and only with ProtT5 beside them" (`normalized_chrom` alone
0.669; + ProtT5 + FBA 0.811 = ProtT5 alone 0.821). So this is not a proof that pathways cannot
help in combination on expression. But the alone-signal ratios are not comparable: on
essentiality the encoding alone is 0.83 against a 0.5 floor; on expression it is 0.053 against
a 0.049 control.

### 3b. The coverage argument, and it is the decisive one

I measured GEM membership on both sides (script
`/scratch/tmp/.../scratchpad/review/q4_gem_enrich.py`, GEM 9.0.2 gene products parsed from
`$DATA_ROOT/data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.xml`, 1,161 genes):

| set | in yeast-GEM 9.0.2 |
|---|---|
| **028 FCL essentiality labeled genes** | **1,109 / 1,121 = 98.9 %** |
| **Kemmeren deletion strains (019 expression)** | **120 / 1,484 = 8.1 %** |
| Kemmeren reporter genes (the 6,169 targets) | 1,127 / 6,169 = 18.3 % |
| Kemmeren responsive-series deletions (700) | 51 / 700 = 7.3 % |
| Kemmeren non-responsive deletions (783) | 68 / 783 = 8.7 % |

**The FCL yeast essentiality benchmark is a metabolic-model gene set by construction (98.9 %).
The 019 expression benchmark is not (8.1 %).** A per-deleted-gene FBA feature vector would be
the wild-type constant for 91.9 % of the expression training strains. Whatever the 028 affine
did, it cannot do here: it has no instance to act on.

(That 8.1 % also explains the responsiveness null in 4a below, and it is consistent with the
already-recorded fact that 3,991 of 4,930 deleted genes in the betaxanthin build are outside
the GEM.)

### 3c. What it would cost

- **Building the feature table:** `ess_reference_flux_table.py --medium YPD` does ~1,100 genes
  with pFBA + FVA in **8.5 min CPU** and writes per-gene `gene_flux_stats` (8 columns:
  n_reactions, n carrying WT flux, max/sum |WT flux|, max forced |flux|, min/max FVA range,
  deletion growth ratio) plus a 4,131-d per-deletion pFBA vector. For 1,484 Kemmeren deletions
  this is ~10-15 min CPU. Trivial.
- **Scoring it cheaply:** adding an entry to `EMBEDDING_SETS` in
  `experiments/019-simb-multimodal/scripts/expression_baselines_split.py` and re-running B2/B3
  on 4 split seeds. CPU, minutes. Also trivial.
- **Scoring it in the model:** this is where the cost is, and it is not small. v13 expression
  runs at **36 epochs/h on an A40** and is still rising at epoch ~4,000 of 6,000. One arm x 3
  seeds paired against a matched control is roughly **a card-week**. The v11 embedding round
  spent a whole Delta A40x4 allocation to resolve a +0.068 effect.
- **The noise floor to beat:** the 019 nuisance axis is the problem. Between-seed sd 0.0444
  against a within-seed across-arm sd of 0.0058 when one knob drove both; the metabolism grid
  measured replicate sigma 0.030 on identical config + identical seed; the bx_aa paired design
  has se 0.0099 at n = 10. With a pinned split and 3 paired init seeds the SE on a paired
  difference is ~0.02, so the MDE is ~0.05 -- comparable to the whole v11 embedding effect.

### 3d. The prior for a gain

The v11 +0.068 is the honest anchor and it is the strongest argument FOR trying: **embedding
content is the only lever that has ever moved expression beyond the replicate spread.** But
+0.068 was ProtT5 against random at 1,000 epochs, i.e. a 1,024-d protein language model
against noise. The prior transfers to "richer per-gene content helps," not to "metabolic
content helps." Given 3a (the pathway encoding reads at control level on this task) and 3b
(8.1 % coverage), my prior on an FBA-feature gain on expression is **low, well under the 0.05
MDE**, and I would not spend a card-week on it without the CPU gate passing first.

---

## 4. Which Kemmeren responses are metabolic? (measured, this session)

I built the Kemmeren strain-by-gene log2-ratio matrix from the LMDB
(`$DATA_ROOT/data/torchcell/microarray_kemmeren2014/processed/lmdb`, via
`experiments/028-knockout-expression/scripts/cross_study_ko_expression._profiles` and
`cross_study_structure._matrix`): **1,484 deletions x 6,169 reporter genes.** Scripts:
`q4_targets.py` and `q4_go.py` in the scratchpad review dir.

### 4a. Deletion side: NULL. Metabolic deletions do not respond more.

| statistic | GEM deletions (n = 120) | non-GEM (n = 1,364) | AUROC | p (Mann-Whitney) |
|---|---|---|---|---|
| mean absolute log2 ratio | 0.1378 (median 0.1155) | 0.1290 (median 0.1117) | 0.538 | 0.165 |
| per-strain SD of the profile | 0.2235 | 0.2075 | 0.538 | 0.165 |

And on Kemmeren's own responsive/non-responsive GEO-series split
(`results/kemmeren_responsiveness.json`, 700 vs 783): responsive 7.3 % in GEM vs
non-responsive 8.7 %, **Fisher OR 0.826, p 0.34.** Deleting a metabolic gene is not what makes
a strain responsive.

### 4b. Target side: REAL and substantial. The responsive reporters ARE metabolic.

GEM membership by across-strain SD rank (Fisher against the remaining reporters):

| reporter stratum | fraction in GEM | vs rest | OR | p |
|---|---|---|---|---|
| all 6,169 | 0.1827 | -- | -- | -- |
| top 1 % by SD (n = 61) | **0.4262** | 0.1803 | 3.38 | 1.3e-5 |
| top 5 % (n = 308) | 0.3604 | 0.1733 | 2.69 | 3.1e-14 |
| top 10 % (n = 616) | 0.3718 | 0.1617 | 3.07 | 5.7e-32 |
| top 25 % (n = 1,542) | 0.3016 | 0.1431 | 2.59 | 7.2e-41 |

GEM reporters are 18.3 % of reporters and carry **27.1 %** of the across-strain variance.

GO (is_a-propagated via `SCerevisiaeGenome.go_dag` / `.go_genes`, OBO rel 2024-01-17),
top 10 % of reporters by SD:

| GO term | in top 10 % | top 10 % rate | background rate | OR | p |
|---|---|---|---|---|---|
| small molecule metabolic process (GO:0044281) | 134 | 0.218 | 0.089 | **3.42** | 6.5e-25 |
| alpha-amino acid biosynthetic process (GO:1901607) | 24 | 0.039 | 0.012 | **4.37** | 1.2e-7 |
| amino acid biosynthetic process (GO:0008652) | 23 | 0.037 | 0.013 | 3.88 | 1.1e-6 |
| cellular respiration (GO:0045333) | 21 | 0.034 | 0.011 | **4.42** | 6.1e-7 |
| transmembrane transport (GO:0055085) | 42 | 0.068 | 0.039 | 2.00 | 2.3e-4 |
| metabolic process (GO:0008152, the broad term) | 267 | 0.433 | 0.446 | 0.94 | 0.52 |
| translation (GO:0006412) | 8 | 0.013 | 0.035 | **0.34** | 7.3e-4 |
| DNA-templated transcription (GO:0006351) | 1 | 0.002 | 0.017 | **0.08** | 2.6e-4 |

**The user's intuition is correct and it is measured: the genes whose expression moves are
amino-acid biosynthesis, respiration and transport, and they are specifically the small-molecule
end of metabolism, not metabolism broadly (the broad GO:0008152 term is flat at OR 0.94).
Translation and transcription genes are DEPLETED.**

### 4c. But the headroom is small, and this is the number that decides it

Variance share of the full 1,484 x 6,169 matrix:

| gene set | n | share of reporters | share of variance | median SD |
|---|---|---|---|---|
| small molecule metabolic process | 551 | 8.9 % | **15.6 %** | 0.1858 |
| amino acid biosynthesis | 78 | 1.3 % | 2.5 % | 0.1893 |
| cellular respiration | 65 | 1.1 % | 1.5 % | 0.2192 |
| transmembrane transport | 238 | 3.9 % | 5.5 % | 0.1716 |
| all reporters | 6,169 | 100 % | 100 % | 0.1507 |

The headline metric `pearson_per_feature` is an **unweighted mean over ~6,127 genes**. So a
metabolic prior that targeted small-molecule metabolism perfectly touches **8.9 % of the
terms**; amino-acid biosynthesis plus respiration together touch **2.4 %**. Even a heroic
+0.3 Pearson on every small-molecule-metabolism reporter moves the headline by
0.089 x 0.3 = **+0.027**, roughly half the v11 embedding effect and inside the 0.05 MDE.
Targeting amino acids and respiration alone caps out near **+0.007**.

This is the quiet killer. The responsive core is metabolic, but the metric is unweighted, so
being right about the metabolic core buys very little on the number being optimized. (Separately:
the top 10 % of genes carry 49.5 % of the variance while contributing 10 % of the metric's
terms -- `results/stratified_responsiveness_seed0.json`. The metric's dilution problem is a
bigger lever than the metabolic prior, and it is free.)

---

## 5. Recommendation

**Test only in the cheap CPU form. Do NOT put a flux layer in the loop. Do not spend GPU on
this before the CPU gate passes.**

### 5a. Do not do (hard no)

A differentiable flux/FBA/LP layer inside the 019 expression model. Justification, stacked:
the layer has never reached the feasible polytope in three experiments and eighteen audited
cells; its kcat content is provably unread (shuffle == real, 0.779 vs 0.771); it adds nothing
on top of an offline solve (0.821 == 0.821); it costs on the one target where three seeds
agree (amino acids, -0.039); it collapses prediction spread; the causal direction is wrong
(2b); and it is not even on this branch, so it is a port plus a training campaign, not a
config line.

### 5b. Do (cheap, CPU, gated)

**Hypothesis, stated so it can fail:** *a per-deleted-gene metabolic descriptor -- pathway
multi-hot plus the 8 solve-derived reaction statistics plus the 4-medium FBA growth ratios --
carries information about the KO transcriptome that `prot_T5_all` does not already carry.*

Test, in order, and stop at the first failure:

1. **Already done, and it failed.** `normalized_chrom_pathways` B2 0.0530 +/- 0.0364 / B3
   0.0329 against `random_1024` 0.0493 / 0.0725. Record this as the first gate and note that
   it was a gate the 028 ladder would also have used.
2. **Build the FBA table** for the 1,484 Kemmeren deletions
   (`ess_reference_flux_table.py`-style, ~10-15 min CPU). Report the GEM-coverage number in
   the same breath: 8.1 %.
3. **Score it in the baseline harness.** Add `fba_stats` (8-d), `fba_pfba` (4,131-d) and
   `pathways+prot_T5` (a CONCATENATION, since 028 round 14 says pathways only work beside
   ProtT5) to `EMBEDDING_SETS` in `expression_baselines_split.py --embedding-set full`, 4 split
   seeds. Minutes of CPU. This is the direct analogue of `logistic_baselines.py`, which is what
   found the 028 gain before any GPU was spent.
4. **Pre-registered pass rule:** the concatenation must beat `prot_T5_all` alone
   (B2 val 0.1055 +/- 0.0258) on **4/4 split seeds** in both B2 and B3, and the FBA-only
   encodings must beat `random_1024` on 4/4. Anything less is inside the control band and does
   not earn a GPU run.
5. **Only on a pass,** run one paired arm on GPU: incumbent stack
   `[fudt_upstream, calm, prot_T5_all, fudt_downstream]` vs that stack plus the metabolic
   encoding, split seed pinned, 3 init seeds, scored as a paired difference at matched epoch.
   Budget honestly: ~a card-week, MDE ~0.05.

### 5c. Where I would put the budget instead

Given 4c, the metric's own dilution is the bigger and cheaper lever: the top 10 % of reporters
carry 49.5 % of the variance and 10 % of the metric's terms, and
`experiments/019-simb-multimodal/scripts/variance_stratified_pearson.py` already exists to
re-score the prediction dumps by variance stratum and has never been run (its results JSON is
absent). That is a CPU read over `$DATA_ROOT/{val,test}-predictions/*.json` and it tells you
whether the model is already good on the responsive core and being dragged down by 3,000
near-constant genes -- which, if true, reframes the whole campaign and costs nothing.

---

## Appendix: absolute paths

Code and configs

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/models/cell_graph_transformer_metabolism.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/torchcell/metabolism/flux_layer.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/torchcell/metabolism/lp_layer.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/gh_metabolism_000.yaml

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/metabolism_grid_runner.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/scripts/ess_reference_flux_table.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/expression_baselines_split.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/variance_stratified_pearson.py

Results read

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/baselines_embedding_study.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/bx_aa_paired_summary.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/kemmeren_responsiveness.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/stratified_responsiveness_seed0.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/028-knockout-expression/results/expression_metabolic_yko_correlation.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/results/splits/ess_fcl_val0.2_s0.json.gz

Scripts I wrote and ran this session (scratch, not committed)

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/q4_gem_enrich.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/q4_targets.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/q4_go.py
