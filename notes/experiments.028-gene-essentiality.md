---
id: rzuldem67xanozz5ll3aq6c
title: 028 Gene Essentiality
desc: ''
updated: 1789537997385
created: 1789537997385
---

## 2026.09.16 - The essentiality line becomes its own experiment

Gene essentiality began as wave 9 of [[experiments.027-betaxanthin-metabolic]] (`train_bx.py --head essentiality`) and is moved here whole: the trainer (`scripts/train_essentiality.py`, the 027 script with paths repointed and the head fixed to essentiality), the union build (`$DATA_ROOT/data/torchcell/experiments/028-gene-essentiality/essentiality_build`, a copy of the 027 build so the other session's running wave-11 jobs keep theirs), the wave-9 to wave-11 result files, checkpoints (`$DATA_ROOT/checkpoints-028`) and per-gene dumps (`$DATA_ROOT/test-predictions-028`). W&B project: `zhao-group/torchcell_028_gene_essentiality`. The 027 history is in [[experiments.028-gene-essentiality.wave9-essentiality]].

There is no fitness objective anywhere in this experiment. The union build carries single-mutant fitness for 4,540 genotypes only because SGD's essential-gene list has no negative class; every fitness record arrives under a False essentiality mask and contributes no loss. The objective is one logit against a 0/1 label (binary cross entropy), selection is validation AUROC under a centered 5-epoch moving average, and the reported number is the AUROC on the released held-out genes at that epoch.

### What was measured before the move (027 waves 9 to 11)

- Pooled backbone at width 90 (958,501 parameters), released labels, 80 epochs: pinned test AUROC 0.793 / 0.788 / 0.578 at seeds 2700 to 2702 (jobs 2009, 2021, 2022); the third seed collapsed to a constant. Flux arm (biomass column of the projected flux, `FluxBiomassReadout`): 0.540 / 0.758 / 0.512, paired mean minus 0.117, 3 of 3 negative.
- Width 12, one block, one-layer preprocessor (16,874 parameters): 0.8011 / 0.7860 / 0.7958 (jobs 2045 to 2047), against FCL 0.742 and FBA 0.640 on the same 195 genes (`fcl_comparison.py`). Width 24: 0.7429 / 0.7398 / 0.8072.
- The validation curve at width 90 moves by a median of 0.034 to 0.072 between adjacent epochs; at widths 24 and 12 by 0.0067 (`curve_stability.py`).

### Three defects in the wave-9 protocol, and the split file that removes them

The data module drew train/validation at 80/10/10 over the 5,461 non-test records, of which 926 carry a label. Measured consequences: a validation set of 75 labeled genes with about 12 essential, on which one AUROC read has a Hanley-McNeil standard error near 0.07 (hypothesis for the bumpy curves, tested by wave 12); 10 percent of the labeled genes in a random test block nothing reports; 4,535 unlabeled records forwarded per epoch under a False mask, about 6 times the labeled work.

`scripts/build_essentiality_splits.py` writes one file per design under `results/splits/`: the per-gene labels, their provenance, and pinned train/val/test record indices. The trainer's `--split-file` pins all three splits and drops every other record (`CellDataModule(drop_unpinned=True)`, new in `torchcell/datamodules/cell.py`). The test block is always the released held-out genes (195 of 223 resolve in the build).

| split file | labels | train genes (ess) | val genes (ess) | test genes (ess) |
|---|---|---|---|---|
| `ess_fcl_val0.2_s0` | released, 1,121 | 629 (100) | 158 (25) | 195 (31) |
| `ess_genome_val0.2_s0` | released + SGD essential + viable single-mutant fitness, 5,795 | 4,368 (879) | 1,093 (220) | 195 (31) |

The genome source keeps the released label on every disagreement (12 genes, 11 of them SGD-essential that the released split calls nonessential, the same 11 `essentiality_label_agreement.py` found), so the held-out genes are scored on the comparator's labels under either source.

### FBA on four media

`scripts/fba_essentiality_baseline.py` (job 2072): cobrapy `single_gene_deletion` on yeast-GEM 9.0.2, the essentiality score being the negative knockout-to-wild-type growth ratio, rounded to 1e-3 so the ~900 deletions that grow at 0.99 to 1.00 of the wild type rank as the tie they are (without rounding two identical runs differed by 0.03 AUROC on the held-out genes, jobs 2069 and 2070).

| medium | WT growth (1/h) | lethal of 1,121 | AUROC all 1,121 | AUROC held-out 223 | AUROC our 195 [95% CI] |
|---|---|---|---|---|---|
| shipped bounds | 0.0858 | 186 | 0.707 | 0.633 | 0.654 [0.562, 0.748] |
| SM | 0.3138 | 160 | 0.707 | 0.642 | 0.666 [0.571, 0.759] |
| SC | 0.5428 | 91 | 0.719 | 0.653 | 0.666 [0.574, 0.759] |
| YPD approx | 0.5352 | 104 | 0.715 | 0.653 | 0.664 [0.572, 0.756] |

The medium moves the lethal count by a factor of two and the AUROC by 0.01. The published 0.69 on all labeled genes is reproduced to within 0.02 on our GEM copy and our bounds.

### Wave 12, queued 2026-09-16 00:55 (jobs 2071 smoke, 2073 to 2093)

Pooled arm, width 12, one block, one-layer preprocessor, 80 epochs, seeds 2700 to 2702, all scored on the same 195 genes:

| arm tag | change from the wave-11 width-12 cell | decides |
|---|---|---|
| `w12base` | none (wave-11 protocol in the 028 experiment) | the port: same numbers at the same seed |
| `w12fcl` | split file `ess_fcl_val0.2_s0` | whether a 158-gene stratified validation set smooths the curve and changes the selected epoch |
| `w12gen` | split file `ess_genome_val0.2_s0` | whether five times the labels move the held-out AUROC |
| `w12fclwd0`, `w12fclwd1e-2`, `w12fclwd1e-1` | AdamW weight decay 0, 1e-2, 1e-1 (base 1e-4) | the weight-decay dose response |
| `w12fclb32` | batch 32 (base 128) | whether 20 steps per epoch instead of 5 changes the curve |

Launcher: `scripts/gh_train_essentiality.slurm` through `scripts/freeze_and_submit.sh`, one environment per cell. Views: `scripts/wandb_views.py --contrast w12`.

### Wave 13, queued 2026-09-16 01:20 (jobs 2113 to 2130): the prediction routed through the module

Same width-12 cell, same fcl split and base hyperparameters as `w12fcl`, so every cell pairs by seed with a pooled cell. The flux arm is the wave-9 recipe (`flux_projection_anchored`, predicted kcat tier, union thermodynamics, YPD-approx bounds written onto the GEM, 20-epoch constraint warmup). Two readouts, two constraint scales, and the physics-off arm:

| arm tag | arm | readout | constraint scale | what the backbone's gradient passes through |
|---|---|---|---|---|
| `w13fluxbio` | anchored | growth column only | 0.1 | the flux layer only (wave-9 design) |
| `w13fluxbio_s0` | anchored | growth column only | 0 | the flux layer only, no physics penalty |
| `w13fluxctx` | anchored | context head + growth column | 0.1 | both paths |
| `w13fluxctx_s0` | anchored | context head + growth column | 0 | both paths, no physics penalty |
| `w13fluxoff` | stoichiometry only | growth column only | 0.1 | the flux layer with no thermodynamics or capacity |
| `w13fluxoffctx` | stoichiometry only | context head + growth column | 0.1 | both paths |

Per-module gradient norms are logged every epoch (`grad_norm_flux_layer_mean`, `grad_norm_readout_mean` beside the transformer's and the preprocessor's); on the CPU smoke of the context-plus-growth readout the flux layer carried 0.82 against the transformer's 0.68 at epoch 0 (anchored) and 6.6 against 0.45 (stoichiometry only). `--ess-readout` selects the readout (`ContextBiomassReadout` in `train_essentiality.py`).

## 2026.09.16 - Unified in one worktree; rounds 1 and 2 requeued with the 027 levers folded in

The 028 branch built in the other session was cherry-picked onto `feat/kinetics-equilibrator-datasets` (commits 3043dffd to cc60b17f, then 5808b618) and its worktree and branch were retired. The job ids in the two sections above (2071 to 2093, 2113 to 2130) were cancelled before any started, as were the 027 wave-12 cells (2094 to 2148) that duplicated them. The 027 essentiality line (`train_bx.py --head essentiality`, waves 9 to 12) ends here; its wave-11 `_nohead` reruns (jobs 2062 to 2068) are copied into this experiment's results and dumps when they land.

Folded into `train_essentiality.py` from the 027 side: the `gated` readout (`context_biomass` behind a zero-initialized gate, `ess_gate` logged per epoch, so at initialization the cell is the pooled model exactly), `--pos-weight-balanced` (772/126), an `ESS_GRAPH_REG` switch on the launcher, and the model change that removes the unused perturbation readout head (16,381 parameters at width 90, 1,201 at width 24, no gradient on any cell). Cells from here on are therefore not bit-identical at a seed to wave 11.

**Round 1, jobs 2149 to 2181 (33 cells)**, width 12, one block, preprocessor 1, 80 epochs, seeds 2700 to 2702. Beyond the wave-12 table above (base, fcl split, genome labels, weight decay 0 / 1e-2 / 1e-1, batch 32), all on the fcl split: learning rate 3e-4 and 3e-3, the class weight, and graph regularization off.

**Round 2, jobs 2182 to 2208 (27 cells)**, same cell on the fcl split. Beyond the wave-13 table above: the `gated` readout at constraint scale 0.1 and 0, and the shuffled-kcat control (`--flux-shuffle-kcat <seed>`) with the forced readout at scale 0.

Reads, pre-registered: (1) the bumpy-curve hypothesis is decided by `w12fcl` against `w12base`, median absolute validation step on 158 labeled genes against 75; (2) each round-1 lever earns a follow-up only when three of three seeds move the same way against `w12fcl`; (3) "not hurt" is `w13fluxgate*` against `w12fcl` paired by seed with the gate's trajectory beside it; "help" is `w13fluxbio_s0` above `w12fcl` and above `w13fluxshuf_s0`; the collapse's cause is `w13fluxbio` against `w13fluxbio_s0`. W&B project: `https://wandb.ai/zhao-group/torchcell_028_gene_essentiality`.

## 2026.09.16 - Round 1 read: the cells are under-trained, the class weight helps, weight decay and graph regularization do nothing

All 33 round-1 cells completed (jobs 2149 to 2181; `results/w12_readout.json`, figure `notes/assets/images/028-gene-essentiality/w12_readout_2026-09-16-18-15-57.svg`). Held-out AUROC on the 195 genes, seed mean over 2700 to 2702, with the pooled-logit gene bootstrap:

| arm | mean | sd | pooled [95% CI] | selected epochs | median val step |
|---|---:|---:|---|---|---:|
| base, wave-11 protocol, 75-gene validation | 0.759 | 0.020 | 0.760 [0.670, 0.844] | 18, 38, 14 | 0.0056 |
| fcl split, 158-gene validation | 0.725 | 0.019 | 0.738 [0.645, 0.826] | 66, 77, 77 | 0.0026 |
| fcl, weight decay 0 / 1e-2 / 1e-1 | 0.725 / 0.726 / 0.726 | 0.02 | 0.735 each | same | same |
| fcl, graph regularization off | 0.725 | 0.019 | 0.738 | same | same |
| fcl, lr 3e-4 / 3e-3 | 0.658 / 0.686 | 0.09 / 0.04 | 0.732 / 0.713 | 77, 77, 74 / 77, 77, 50 | |
| fcl, batch 32 | 0.765 | 0.005 | 0.764 [0.673, 0.852] | 74, 73, 77 | 0.0032 |
| fcl, class weight 772/126 | 0.759 | 0.016 | 0.769 [0.680, 0.858] | 77, 75, 53 | 0.0034 |
| genome labels, 1,093-gene validation | 0.704 | 0.024 | 0.705 [0.606, 0.802] | 77, 77, 69 | 0.0020 |

- **The bumpy-curve hypothesis holds.** The median epoch-to-epoch validation step halves from 0.0056 on 75 labeled genes to 0.0026 on 158 and 0.0020 on 1,093 (panel b of the figure). The old protocol's curve reads 0.8 on its own 75 genes and the split cells read 0.65 to 0.72 on theirs; the levels are not comparable, only the steps are.
- **The fcl split scores below the old protocol, and the reason is training genes, not validation.** The split trains on 629 labeled genes against the old draw's roughly 740 (80 percent of 926), and every fcl cell selected epoch 66 to 77 of 80 with its validation curve still rising (panel a), so the cells are under-trained at 80 epochs with 5 optimizer steps per epoch. Hypothesis (tested by round 3): the 0.034 gap closes with more optimization.
- **Weight decay does nothing at 0, 1e-2 or 1e-1**, to the third decimal, and neither does removing graph regularization: at 400 optimizer steps with lr 1e-3 a decay of 0.1 shrinks a weight by 4 percent, and the graph term's lambda is 6.5e-5. Both levers are closed on this endpoint.
- **Batch 32 moves three of three seeds up** (fcl 0.7455 / 0.7217 / 0.7085 to 0.7657 / 0.7701 / 0.7594; +0.020, +0.048, +0.051) and collapses the seed spread to 0.005. It is four times the optimizer steps per epoch, which is the same under-training story.
- **The class weight moves three of three seeds up** (to 0.7642 / 0.7722 / 0.7410; +0.019, +0.051, +0.033). Its curve also rises faster (panel a).
- **Genome-wide labels validate at 0.84 and test at 0.70.** The 1,093-gene validation set drawn from the genome-wide labels reads 0.841 / 0.836 / 0.835 at the selected epochs while the released held-out genes read 0.706 / 0.726 / 0.679, the widest validation-to-test gap of any arm; the released held-out genes are a different population from the genome-wide negatives, which is the same point as the bullet below.
- **The learning rate at 3e-4 and 3e-3 is worse than 1e-3 on the seed mean**, but with seed spreads of 0.09 and 0.04 the pooled numbers (0.732, 0.713) sit inside the fcl interval; neither is a lever to keep.
- **Genome-wide labels hurt** (0.704 against 0.725, three of three seeds down). Adding SGD's essential list and the viable single-mutant fitness records as nonessential gives 4,368 training genes but a worse ranking of the released held-out genes. Hypothesis (untested): the viable-fitness negatives are a different population from the released split's nonessential genes.
- **Gradient norms (panel d):** on the fcl cells the preprocessor's gradient starts near 1e-2 and rises about thirty-fold over 80 epochs while the head's stays flat near 0.4; the clip at 10 never binds.

Round 3, jobs 2209 to 2222 (14 cells), all width 12 on the fcl split with graph regularization on: batch 32 with the class weight at 80 and 200 epochs, batch 32 alone at 200 epochs, and the two winners on five-fold cross-validation files (`ess_fcl_fold{0-4}of5_s0`, 629 or 630 train genes and 157 or 158 validation genes each) at 200 epochs, whose fold models are averaged on the held-out genes exactly as FCL averages its five fold models. The ensemble read is `scripts/cv_ensemble_readout.py`.

## 2026.09.16 - The classical ladder reaches 0.80 with FBA as a feature, and the cells were spending most of each epoch spawning workers

`scripts/logistic_baselines.py` on `ess_fcl_val0.2_s0` (fit on 629 train genes, hyperparameter on 158 validation genes, scored once on the 195 held-out genes; `results/logistic_baselines.json`):

| rung | embedding | chosen | val | held-out [95% CI] |
|---|---|---|---:|---|
| B1 FBA growth | none | SC medium | 0.726 | 0.666 [0.57, 0.76] |
| B2 logistic | prot_T5_all | C = 1e-3 | 0.722 | 0.763 [0.67, 0.85] |
| B4 random forest | prot_T5_all | depth 4 | 0.743 | 0.768 [0.68, 0.86] |
| B2 logistic | normalized_chrom_pathways | C = 10 | 0.722 | 0.775 [0.68, 0.86] |
| B5 logistic + FBA | prot_T5_all | C = 0.1 | 0.833 | **0.803** [0.72, 0.88] |
| B5 logistic + FBA | calm | C = 10 | 0.839 | **0.803** [0.72, 0.88] |
| B5 logistic + FBA | normalized_chrom_pathways | C = 10 | 0.820 | 0.796 [0.70, 0.88] |
| B3 cosine kNN | any | | | 0.58 to 0.69 |

- **A logistic regression on a frozen protein embedding plus FBA's four growth ratios scores 0.803 on the held-out genes.** That is above FCL's 0.742 on the same genes and level with the best transformer cell of any round so far (width 12, old protocol, 0.80; batch 32 on the split, 0.765). Neither half reaches it alone: the embedding gives 0.76, FBA 0.67. The transformer is measured against 0.80 from here on.
- **Nearest neighbors do not work on this endpoint** (0.58 to 0.69), unlike betaxanthin where kNN on prot_T5 was the best baseline. Essentiality is not a smooth function of embedding distance at k = 1 to 50.
- **The obvious next transformer cell is FBA's growth ratio as an input feature**, either concatenated to the readout or as a node feature; the ladder says the two sources are complementary.

**Every GPU cell was spending most of each epoch spawning data loader workers.** Three-epoch CPU smoke of the width-12 split cell: 185 s of wall with `NUM_WORKERS=2`, 59 s with `NUM_WORKERS=0`. The context is `spawn`, the loaders are rebuilt each epoch, and each spawned worker re-imports the package, so about 42 s of every epoch was import time on a loader over 787 records. This is why every cell, from 16,874 to 958,501 parameters, took 48 to 60 s per epoch, and why the GPUs read 0 percent utilization with 4 to 5 GB resident. The launcher's default is now `ESS_WORKERS=0`; a GPU smoke confirms the per-epoch time before round 3 is resubmitted under it.

**Confirmed on the GPU (job 2224, width 12, batch 32, no loader workers): 60 epochs in about five minutes, roughly 5 s per epoch against 38 s under two spawned workers.** The 13 pending round-3 cells were cancelled and resubmitted under `ESS_WORKERS=0` (jobs 2224 to 2236); a 200-epoch cell now takes about 17 minutes instead of two hours. Job 2209 (batch 32 with the class weight, seed 2700, 80 epochs) was left running under the old setting and stays in the round as the pair of the resubmitted seeds.

## 2026.09.16 - The ladder redone with cross-validated selection and controls; round 2 read: the gated module does not hurt, the forced module does not help, and the kcat table is not what it reads

**The ladder table above is superseded.** Its hyperparameters were chosen on the validation genes, which makes the validation column a selection statistic. `logistic_baselines.py` now chooses each rung's hyperparameter by five-fold stratified cross-validation inside the 629 training genes and scores the validation and held-out genes once. The ladder is also full: four model classes (logistic, random forest, RBF SVC, cosine kNN) on 24 encodings (the fitness ladder's 17, plus a one-hot gene identity and standard normal vectors of width 1, 10, 100, 1,000, 6,607 and 8,192), each with and without FBA's four growth ratios, 192 fitted rungs plus FBA on four media. Figure and table: `notes-tex/028-gene-essentiality/` section 2; the figure below is every rung.

![](./assets/images/028-gene-essentiality/classical_ladder_2026-09-16-18-15-47.svg)

Held-out AUROC on the 195 genes, the rung per encoding with the highest CV AUROC (`results/logistic_baselines.csv`):

| encoding | best by CV | CV | held-out | with FBA: best by CV | CV | held-out |
|---|---|---|---|---|---|---|
| one_hot_gene | logistic | 0.504 | 0.506 | random forest | 0.775 | 0.679 |
| random_1 to random_8192 | | 0.47 to 0.60 | 0.45 to 0.59 | | 0.73 to 0.81 | 0.56 to 0.69 |
| codon_frequency | SVC | 0.700 | 0.695 | SVC | 0.849 | 0.818 [0.73, 0.90] |
| normalized_chrom_pathways | logistic | 0.705 | 0.775 | logistic | 0.802 | 0.830 [0.75, 0.90] |
| calm | SVC | 0.755 | 0.734 | random forest | 0.850 | 0.806 [0.72, 0.89] |
| prot_T5 | SVC | 0.781 | 0.732 | logistic | 0.842 | 0.821 [0.74, 0.90] |
| esm2 | random forest | 0.731 | 0.708 | random forest | 0.815 | 0.758 |
| mixed | logistic | 0.788 | 0.768 | logistic | 0.833 | 0.792 |
| FBA alone | | | 0.654 to 0.666 | | | |

- **The controls behave.** One-hot scores 0.506 (every held-out column is zero in training) and the random vectors score chance at every width; with FBA appended they score FBA's number. Nothing leaks through the split.
- **The bar is 0.80 to 0.83, and it needs FBA.** No encoding clears 0.78 alone; four clear 0.80 with the four ratios appended. The rung with the highest CV AUROC overall (random forest on CaLM + FBA, 0.850) scores 0.806 held-out; the highest held-out number in the figure (cosine kNN on codon frequency + FBA, 0.841) is a max over 192 rungs, not a selected model.
- **Correction to the previous section:** kNN is not dead on this endpoint once the FBA ratios are appended (0.84 on codon, CaLM); without them it reads 0.63 to 0.74 on the language-model encodings, and the earlier "0.58 to 0.69" was the val-tuned run over four encodings.

**Round 2 (jobs 2182 to 2208), the module on the width-12 cell, paired by seed against the round-1 pooled cell on the same split, batch and budget (`wave12_readout.py --round w13`):**

![](./assets/images/028-gene-essentiality/w13_readout_2026-09-16-18-16-07.svg)

| arm | seeds 2700, 2701, 2702 | mean | pooled [95% CI] |
|---|---|---|---|
| pooled (round-1 fcl) | 0.745, 0.722, 0.708 | 0.725 | 0.738 [0.65, 0.83] |
| forced biomass, scale 0.1 | 0.759, 0.585, 0.774 | 0.706 | 0.765 [0.67, 0.85] |
| forced biomass, scale 0 | 0.770, 0.653, 0.772 | 0.732 | 0.773 [0.68, 0.86] |
| forced biomass, kcat shuffled, scale 0 | 0.771, 0.718, 0.765 | 0.751 | 0.769 [0.68, 0.86] |
| context + growth, scale 0.1 | 0.566, 0.709, 0.722 | 0.666 | 0.717 [0.61, 0.82] |
| context + growth, scale 0 | 0.556, 0.727, 0.718 | 0.667 | 0.724 [0.62, 0.82] |
| gated, scale 0.1 | 0.721, 0.747, 0.690 | 0.719 | 0.738 [0.64, 0.83] |
| gated, scale 0 | 0.714, 0.744, 0.747 | 0.735 | 0.747 [0.65, 0.84] |
| stoichiometry only, forced | 0.524, 0.631, 0.661 | 0.606 | 0.664 [0.56, 0.77] |
| stoichiometry only, context + growth | 0.719, 0.717, 0.678 | 0.705 | 0.755 [0.67, 0.84] |

- **Not hurt: holds.** Gated at scale 0 is +0.02 to +0.04 on two seeds and -0.03 on one against the pooled cell; the gate ends at -0.056 to -0.063 on all six gated runs, and the flux layer's gradient norm is 0.01 to 0.07 against 0.22 to 0.36 on the forced arm's seeds that train. The gated cell is a pooled cell with a small correction.
- **Help: does not hold.** Forced at scale 0 is above the pooled cell on two of three seeds, and the shuffled-kcat control (0.751 mean) matches or beats the real table (0.732). Whatever the growth column carries here, it is not the unit-to-constant correspondence. Pooled bootstrap 0.773 [0.68, 0.86]: above FCL 0.742 and FBA 0.64 to 0.67, level with the pooled cell, below the ladder's 0.80.
- **The collapse is not the penalty's.** Seed 2701 stalls at both scales (0.585 and 0.653) with every module's gradient decaying within ten epochs to a hundredth of the other seeds' and the training loss flat at 0.46 against 0.34. The stoichiometry-only layer collapses on all three seeds (0.52 to 0.66), so on this endpoint the capacity bound and the thermodynamic term keep the growth column informative; on betaxanthin the penalty scale was the collapse dose. Context + growth without a gate costs 0.06 on the mean and puts one seed at 0.56; the same sum behind the gate loses nothing.
- **Readout script fixes in this commit:** the FCL comparator is read from `fcl_comparison.json`'s `scored` block (`ours_scored`), a round can pull another wave's cell in as its pair, and the gate and shuffle arms have labels.

Example runs, gated scale 0 seed 2701 and forced scale 0 seed 2701 (the stalled seed):

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/z2cmvk14>

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/235hi61k>

**Round 4 (9 cells), queued 16:41 as jobs 2237 to 2245, cancelled unstarted and requeued 16:47 as three pack jobs 2246 to 2248.** The three round-2 module readouts at constraint scale 0 on top of the round-3 levers (batch 32, class weight 772/126, 200 epochs, `ESS_WORKERS=0`), seeds 2700 to 2702, paired against the round-3 `b32posw200` pooled cell: `w15gate_s0` (gated), `w15bio_s0` (forced growth column), `w15shuf_s0` (forced, kcat shuffled by the seed). Read with `wave12_readout.py --round w15`. Pre-registered reads are the round-2 ones: not hurt is gated against the pooled cell paired by seed; help is forced above the pooled cell on three of three seeds and above the shuffled control. Round 3 (2224 to 2236) is still draining ahead of it; its partial numbers at 16:40 are b32posw at 80 epochs 0.780 / 0.769 / 0.700, b32posw200 seed 2700 0.780, b32e200 seed 2700 0.766.

**Several cells on one GPU.** Slurm on GilaHyper allocates whole GPUs (gres type `gpu`, no shards), so the one-cell-per-job launcher caps the machine at four concurrent cells while a width-12 cell holds 3.6 GB of a 46 GB card at 10 percent utilization (round 3, `nvidia-smi` at 16:27). `scripts/gh_train_essentiality_pack.slurm` takes one GPU, 12 CPUs and 36 GB and runs every line of a pack file (`conf/packs/<name>.pack`, one cell's `ESS_*` assignments per line) concurrently, each cell to its own log `slurm/output/028-pack_<job>_<stem>.out`; the cell body moved to `scripts/ess_cell.sh`, which both launchers call. Smoked on CPU with two one-epoch pooled cells (both finished, result files written, moved out of `results/`). Round 4 is the first GPU use: one pack per arm, three seeds each, so nine cells occupy three GPUs instead of nine job slots.

## 2026.09.16 - Round 3 read: the two levers at 200 epochs give the tightest cell (0.774 +/- 0.006), the fold ensemble reads 0.795 under FCL's own protocol

Jobs 2209 and 2224 to 2236, all width 12 on the released-label split, `ESS_WORKERS=0` (10.5 to 11 min per 200-epoch cell). `wave12_readout.py --round w14`, `cv_ensemble_readout.py --prefix w14_cv`.

![](./assets/images/028-gene-essentiality/w14_readout_2026-09-16-18-16-12.svg)

| arm | seeds 2700, 2701, 2702 | mean | sd | pooled [95% CI] |
|---|---|---|---|---|
| batch 32 + class weight, 80 epochs | 0.780, 0.769, 0.700 | 0.750 | 0.043 | 0.774 [0.69, 0.86] |
| batch 32 + class weight, 200 epochs | 0.780, 0.769, 0.773 | 0.774 | 0.006 | 0.780 [0.69, 0.87] |
| batch 32, 200 epochs | 0.766, 0.761, 0.755 | 0.761 | 0.005 | 0.764 [0.67, 0.85] |
| 5-fold models (folds 0 to 4, seed 2700) | 0.780, 0.806, 0.797, 0.751, 0.789 | 0.785 | | ensemble 0.795 [0.71, 0.88] |

- **Batch 32 + class weight + 200 epochs is the best pooled cell and the tightest** (sd 0.006 across seeds). The class weight adds +0.014 / +0.008 / +0.018 over batch 32 alone at the same budget, three of three seeds, as at 80 epochs.
- **The longer budget moves one seed only.** The 80- and 200-epoch cells share their trajectory through epoch 80 and select the same epoch on seeds 2700 and 2701 (75, 68; identical held-out numbers); seed 2702 selects epoch 31 at 80 epochs (0.700) and 148 at 200 (0.773).
- **Fold 0 of the 5-fold files is the val0.2 split exactly** (same 629 / 158 / 195 genes, both drawn at seed 0), so `cv0` reproduces `b32posw200` seed 2700 to the digit (0.7801). Not wasted, but not a sixth model either.
- **The fold ensemble, FCL's protocol, reads 0.795 [0.71, 0.88] on the 195 genes** against FCL's 0.742 on the same genes and its published 0.73; fold mean 0.785. Still 0.01 to 0.04 below the ladder's FBA-augmented rungs (0.806 to 0.830).
- The fold curves show sharp single-epoch dips around epochs 20 to 30 on some seeds (fold 4 to 0.3) with the preprocessor gradient dropping to 0.01 at the same epochs and recovering; the smoothed selection rule steps over them.

Example runs, batch 32 + class weight at 200 epochs seed 2702 (selected epoch 148) and fold 1 (0.806):

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/zp335f43>

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/mkrk5vb5>

## 2026.09.16 - Round 4 read: on the best backbone the module neither helps nor hurts, and the shuffled kcat table scores the same as the real one

Pack jobs 2246 to 2248 (three cells per GPU, 29 min per pack; the cards ran at 80 to 98 percent utilization and 11 GB against 10 percent for a single cell). Paired by seed against the round-3 `b32posw200` cell (`wave12_readout.py --round w15`).

![](./assets/images/028-gene-essentiality/w15_readout_2026-09-16-18-16-17.svg)

| arm (all batch 32 + class weight, 200 epochs, scale 0) | seeds 2700, 2701, 2702 | mean | sd | pooled [95% CI] |
|---|---|---|---|---|
| pooled (round 3) | 0.780, 0.769, 0.773 | 0.774 | 0.006 | 0.780 [0.69, 0.87] |
| gated | 0.766, 0.782, 0.765 | 0.771 | 0.010 | 0.768 [0.68, 0.85] |
| forced growth column | 0.772, 0.781, 0.778 | 0.777 | 0.005 | 0.778 [0.69, 0.86] |
| forced, kcat shuffled | 0.785, 0.781, 0.772 | 0.779 | 0.006 | 0.781 [0.69, 0.87] |

- **Every arm is within 0.015 of the pooled cell on every seed.** Not hurt holds again (gated 1 of 3 above, mean -0.003). Help does not: forced 2 of 3 above by 0.005 and 0.012, and the shuffled-kcat control is as good (0.779 vs 0.777).
- **The gate moves now.** It leaves zero after epoch 10 and climbs steadily to -0.17 / +0.20 / +0.11 at epoch 200 (either sign), flux-layer gradient 0.06 to 0.12 against 0.22 to 0.65 in the forced arm. Round 2's stalled seed did not recur under batch 32 with the class weight.
- **Read across rounds 2 and 4:** the forced growth column is as expressive a readout as the MLP head (matches it to 0.01 carrying the whole prediction), but its kcat content is not what it reads (shuffle == real). On this endpoint at width 12 the flux layer acts as a learned nonlinearity over the context, not as a physical prior that ranks genes. The remaining lever for the module question is the module's own inputs (measured kcat tier, thermo source, parameterization), not the readout.
- **Performance ceiling so far:** pooled 0.774 (seed mean) / 0.795 (fold ensemble) against the ladder's 0.81 to 0.83 with FBA appended. Next: the FBA growth ratio as an input to the transformer.

Example runs, gated seed 2701 (gate to +0.20) and shuffled-kcat seed 2700 (0.785):

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/tvbw3cg9>

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/jf3af715>

**Round 5 (w16), packs 2249 and 2250, queued 17:38: FBA's growth ratios as an input to the transformer.** `--ess-readout fba` (new in `train_essentiality.py`, `FbaContextReadout`) adds an affine of the deleted gene's knockout-to-wild-type growth ratio on the four media (`fba_essentiality_per_gene.csv`, the same four numbers the ladder appends) to the pooled arm's context logit; the four weights are logged per epoch as `ess_fba_w_<medium>`. Cells: the round-3 levers (batch 32, class weight, 200 epochs) at seeds 2700 to 2702, paired against `b32posw200`, and the same readout on the five cross-validation folds (seed 2700) for the ensemble number (`cv_ensemble_readout.py --prefix w16_fbacv`). Pre-registered read: the ladder's logistic rung on ProtT5 gains 0.09 from the four ratios (0.732 to 0.821); the transformer gains the same complement if its seed mean moves from 0.774 toward 0.82 on three of three seeds, and the fold ensemble from 0.795 toward the ladder's 0.81 to 0.83. CPU smoke: two epochs already read 0.713 held-out, which is the affine on the ratios alone (FBA scores 0.65 to 0.67).

## 2026.09.16 - Round 5 read: FBA's growth ratios as an input lift the transformer to 0.82, the ladder's level, on three of three seeds

Packs 2249 (three seeds, 27 min) and 2250 (five folds, 27 min), paired against the round-3 `b32posw200` cell. `wave12_readout.py --round w16`, `cv_ensemble_readout.py --prefix w16_fbacv`.

![](./assets/images/028-gene-essentiality/w16_readout_2026-09-16-18-16-21.svg)

| arm (batch 32 + class weight, 200 epochs) | seeds 2700, 2701, 2702 | mean | sd | pooled [95% CI] |
|---|---|---|---|---|
| pooled (round 3) | 0.780, 0.769, 0.773 | 0.774 | 0.006 | 0.780 [0.69, 0.87] |
| context head + FBA ratios | 0.814, 0.805, 0.844 | 0.821 | 0.020 | 0.838 [0.76, 0.91] |
| 5 fold models + FBA ratios | 0.814, 0.824, 0.839, 0.775, 0.853 | 0.821 | | ensemble 0.822 [0.74, 0.90] |

- **Both pre-registered reads hold.** +0.034 / +0.037 / +0.071 on the three seeds (3 of 3), seed mean 0.774 -> 0.821; fold ensemble 0.795 -> 0.822. The transformer with the four solver numbers sits where the ladder's best rungs sit (0.806 to 0.830), 0.08 above FCL on the same genes.
- **The weights end negative on every cell** (a deletion that stops growth is essential), largest on the shipped-bound and YPD columns (-0.5 to -0.7 at the selected epoch), near zero to -0.4 on SC. The cells select at epochs 28 to 70 (vs 68 to 148 without): the affine is learned in the first epochs and the context head fills in around it.
- **Read across rounds 2, 4 and 5:** the learned flux layer's growth column matched the head and ignored its kcat table; the FBA solve's growth ratio adds 0.05. On this endpoint the module's value is in the solve, not in the learned layer. Next: the ratios as a node feature (every gene's, not only the deleted gene's), one medium against four, and a flux arm that receives the ratios beside its own growth column.

Example runs, seed 2702 (0.844) and fold 4 (0.853):

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/k1sfpylr>

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/7ysar8my>

**Round 6 (w17), packs 2251 to 2256, queued 18:22, six arms at three seeds on the round-3 levers, paired against `b32posw200` and the round-5 `fba` cell.** (1) `fbaypd`: the FBA affine on the YPD ratio alone (`--ess-fba-media YPD`), one medium against four. (2) `biofba`: the flux layer's forced growth column plus the FBA affine, no context head (`--ess-readout biomass_fba`, scale 0): the learned layer on top of the solve. (3) `gatefba`: the gated readout plus the affine (`gated_fba`). (4) `codonfba`: the backbone opened on codon frequency (64-d; `--node-embeddings codon_frequency`) plus the affine, since codon + FBA reaches 0.82 on the ladder from 64 numbers. (5) `compfba` and (6) `comp`: the backbone on the species-LM composite the expression line uses (`fudt_upstream fudt_downstream prot_T5_all calm`, 3,332-d) with and without the affine. Trainer additions: `--ess-fba-media`, `biomass_fba`, `gated_fba`, `--node-embeddings` (the dataset is opened on the named embeddings; the preprocessor projects their concatenation), and a guard that refuses a flux readout on a pooled arm (a CPU smoke had run one silently). Read with `wave12_readout.py --round w17`. Pre-registered reads: one medium is enough if `fbaypd` is within seed noise of `fba` (0.821); the layer adds to the solve if `biofba` or `gatefba` is above `fba` on three of three seeds; an embedding is better if its `+fba` cell is above `fba` on three of three.

**Figure fixes (same commit, 6655a4cc):** the ladder takes blue for its fourth model (the palette's yellow read as a lighter orange) and the display names on its axis (`fudt_*` is the species LM 5' and 3' embedding); the readout figures use six hues per tier with a dashed, hollow second tier instead of the palette's darker siblings; panel labels shortened and letters moved clear of them; the two flux diagrams now read left to right.

## 2026.09.16 - Round 6 read: four media beat one, the learned layer adds nothing on top of the solve, the composite embedding equals ProtT5, the codon backbone is poor

Packs 2251 to 2256 (20 to 21 min each, three cells per GPU). Paired against the round-5 `fba` cell (0.814 / 0.805 / 0.844, mean 0.821) and the round-3 pooled cell (0.780 / 0.769 / 0.773). `wave12_readout.py --round w17`.

![](./assets/images/028-gene-essentiality/w17_readout_2026-09-16-19-00-53.svg)

| arm (batch 32 + class weight, 200 epochs) | seeds 2700, 2701, 2702 | mean | sd | pooled [95% CI] |
|---|---|---|---|---|
| context head + FBA ratios, four media (round 5) | 0.814, 0.805, 0.844 | 0.821 | 0.020 | 0.838 [0.76, 0.91] |
| context head + FBA ratio, YPD only | 0.819, 0.775, 0.789 | 0.795 | 0.022 | 0.808 [0.73, 0.89] |
| forced growth column + FBA ratios, no context head | 0.834, 0.811, 0.809 | 0.818 | 0.014 | 0.820 [0.74, 0.90] |
| gated growth column + FBA ratios | 0.810, 0.821, 0.832 | 0.821 | 0.011 | 0.827 [0.75, 0.90] |
| codon-frequency backbone + FBA ratios | 0.747, 0.682, 0.764 | 0.731 | 0.043 | 0.721 [0.62, 0.81] |
| species-LM composite backbone + FBA ratios | 0.802, 0.825, 0.822 | 0.816 | 0.013 | 0.819 [0.73, 0.90] |
| species-LM composite backbone | 0.783, 0.779, 0.773 | 0.778 | 0.005 | 0.776 [0.69, 0.85] |

- **Four media > one.** YPD alone is 0.03 to 0.05 below the four-column cell on two of three seeds (mean 0.795 vs 0.821), still +0.02 over the pooled cell.
- **The learned layer adds nothing on top of the solve.** Forced growth column + affine (no MLP head) 0.818, gated + affine 0.821, each 1 of 3 seeds above the round-5 cell, each within its spread. The flux layer trains (gradient norm 0.4 to 1.1 on the forced cells) but ranks no better than the head. Across rounds 2, 4, 5, 6: the module's ranking is the backbone's; the physics that ranks genes here is the solve's optimum, which the projection layer does not compute.
- **Composite embedding = ProtT5** (0.778 vs 0.774 without FBA; 0.816 vs 0.821 with), at 44k vs 16.5k parameters.
- **Codon backbone is poor**: 0.731 with FBA, below the pooled ProtT5 cell and 0.09 below the ladder's codon + FBA rung (0.818); selected epochs 16 / 172 / 14 and the largest validation step of any cell (0.013): the 12-wide transformer over 64 codon numbers never settles.
- **Next:** the flux-layer probe at initialization (does the growth column rank essential genes at all before training?); a solve-derived gene encoding (per-deletion pFBA vector or the wild-type flux through the gene's reactions) as a node feature; and, if the transformer is to beat the ladder rather than tie it, width or depth on the FBA cell.

Example runs, gated + FBA seed 2702 (0.832) and codon seed 2701 (0.682):

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/zllkcndg>

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/san5c0l6>

## 2026.09.16 - Flux growth probe: the layer's growth column is a signed free coordinate, and the trained forced cell ranks essential genes with it upside down

`scripts/flux_growth_probe.py --result w15_bio_s0_h12l1p1_e200_s2700.json` (CPU, one forward pass over the 982 labeled records at initialization and under the cell's trained weights; `results/flux_growth_probe_w15_bio_s0_h12l1p1_e200_s2700.json`). The score is the AUROC of the NEGATIVE growth column against the label (a deletion that stops growth is essential) and the Spearman of the column against FBA's knockout/wild-type ratio.

| weights | zero-growth deletions | median v_growth | AUROC(-v) train / val / test | Spearman vs FBA (shipped, SM, SC, YPD) |
|---|---|---|---|---|
| initialization | 0 of 982 | -0.27 | 0.618 / 0.555 / 0.547 | +0.12, +0.10, +0.09, +0.11 |
| trained (forced readout, epoch 108) | 0 of 982 | -1.43 | 0.048 / 0.258 / 0.228 | -0.27, -0.26, -0.24, -0.24 |

- **At initialization the column carries a weak signal in the right direction**: 0.55 on the held-out genes, Spearman +0.11 with the solve. Nothing like FBA's 0.65, and no deletion reaches zero growth.
- **Training flips it.** Under the forced cell's weights the negative column reads 0.23 on the held-out genes, i.e. 0.77 with the opposite sign, which is the cell's held-out AUROC (0.772): the affine learned a positive slope, essential genes get HIGHER "growth", and the column is anti-correlated with FBA (-0.25). The median growth is -1.4 mmol/gDW/h against a wild-type FVA range of 0.48 to 0.54 on YPD: the column is negative for most deletions.
- **Why the column can do that**: the arm ends its alternations on the balance projection (`feas_balance_median` 0.000) and leaves the box open, 58 percent of reactions outside their bounds at the selected epoch (`feas_box_violation_frac`, also 0.64 on the round-2 cell and 0.59 on the round-6 cell). The growth reaction's lower bound of zero is not enforced, so `v_growth` is a signed, unbounded coordinate the backbone writes through the abundance head, and the readout uses it as a free latent. This is the mechanism behind "shuffled kcat == real" and "the layer matches the head": the column is not growth.
- **What follows**: the forced readout on a layer whose alternations END on the clip (bounds exact, balance approximate) is the cell that decides whether a bounded growth column carries FBA's signal; the probe then reads whether the trained column correlates positively with the solve. That is round 7.

The same probe on the round-2 forced cell at constraint scale 0.1 (`w13_fluxbio` seed 2700, 80 epochs, selected epoch 77): trained AUROC(-v) 0.091 / 0.266 / 0.241 on train / val / test, Spearman -0.26 to -0.22 against FBA, median growth -2.4. The flip is the forced readout's habit, not one cell's.

**Round 7 (w18), the growth column with the bounds enforced:** the forced readout at scale 0 on the round-3 levers with `--flux-parameterization projection_box` (the alternations end on the clip, so every bound including the growth reaction's zero floor holds exactly and the balance is the soft residual) and with `box` (v = lb + (ub - lb) sigmoid(z); bounds exact by construction, Sv = 0 soft), three seeds each, paired against the round-4 forced cell (balance-ending projection, 0.772 / 0.781 / 0.778) and the pooled cell. Then the probe on each. Reads: a bounded growth column that still matches the head, and whose trained column correlates positively with the solve, says the layer computes something growth-like and the readout was the problem; a bounded column that collapses says the balance-ending projection's free coordinate was the only thing the forced arm had.

## 2026.09.16 - Round 7 read: with the bounds enforced the growth column is dead (clip last) or unbalanced (box), and the module question closes at the layer

Packs 2257 (`projection_box`, 28 min) and 2258 (`box`, 24 min). `wave12_readout.py --round w18`; the probe on all six cells (`results/flux_growth_probe_w18_*.json`).

![](./assets/images/028-gene-essentiality/w18_readout_2026-09-16-19-40-51.svg)

| arm (forced readout, scale 0, round-3 levers) | seeds 2700, 2701, 2702 | mean | box violation | balance residual |
|---|---|---|---|---|
| projection ending on balance (round 4) | 0.772, 0.781, 0.778 | 0.777 | 0.58 | 0.000 |
| projection ending on the box | 0.500, 0.500, 0.500 | 0.500 | 0.00 | 0.85 to 1.0 |
| box form, v = lb + (ub - lb) sigmoid(z) | 0.767, 0.609, 0.785 | 0.720 | 0.00 | 2.0 |

- **Clip last: the growth column is identically zero** for all 982 deletions at init and after training (the clip pins r_2111 at its floor); the cells score exactly 0.5 and select epoch 2.
- **Box form: growth in [0, ub], starts near ub/2 (median 45 to 53 vs a wild-type 0.5), Sv = 0 never approached** (balance residual 2.0 at every epoch); 0.720 mean, below the pooled cell; the probe flips the sign on 2 of 3 trained seeds (0.23 / 0.22 held-out, Spearman -0.25 / -0.28 vs FBA) and reads 0.61 / +0.10 on the third.
- **Conclusion across rounds 2, 4, 5, 6, 7 and the probes:** whichever constraint is exact, the other is violated by as much as the fluxes themselves, so the K-alternation layer never reaches the polytope FBA optimizes over; its growth column is a free signed coordinate, a constant zero, or an unbalanced sigmoid, never growth. The module's value on essentiality is the solve's, and the layer has to reach feasibility before its column can carry it. That is a layer question (more alternations, Dykstra, a warm start from the wild-type solve), and essentiality is the cheap place to test it: a feasible growth column must reproduce FBA's 0.65 at initialization.

Example runs, box seed 2701 (0.609) and clip-last seed 2700 (0.500):

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/q5etjawa>

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/kthpkk1r>

**Round 8 (w19), packs 2259 to 2262, queued 20:02: width and depth on the FBA cell.** The round-5 cell (width 12, one block, one-layer preprocessor, context head + FBA affine, 0.814 / 0.805 / 0.844) against the same readout at width 24 and 48, at two blocks, and with the two-layer preprocessor, three seeds each on the round-3 levers. Read with `wave12_readout.py --round w19`. Pre-registered read: a cell passes the ladder if its seed mean clears 0.83 with three of three seeds above the round-5 cell; the wave-9 finding that width 90 collapsed one seed to a constant is the failure mode to watch at 48.

**Job 2261 (two blocks) failed at the first step: `IndexError: index 4 is out of bounds for dimension 1 with size 4` in `compute_graph_regularization`.** `GRAPH_REG_CONFIG` pins the nine gene graphs to attention heads 0 to 8 of layer index 1, the SECOND block. Two consequences: (1) every one-block cell in this experiment never evaluated the term, which is why round 1's "graph regularization off" reproduced the base cell to the third decimal (both were the same cell), and the section-3 sentence is corrected accordingly; (2) a two-block cell with four heads dies on head 4. The pack is resubmitted with `ESS_GRAPH_REG=0`. Using the term on this backbone would need nine heads (width 90 at 019's settings) or a remapped config; neither is a lever worth spending on here given (1).

## 2026.09.16 - Round 8 read: width and depth on the FBA cell do not pass the ladder, capacity is not the lever

Packs 2259 (width 24), 2260 (width 48), 2262 (two-layer preprocessor), 2263 (two blocks, graph regularization off; 2261 was the failed first submit), three cells per card, 19 min per pack and 26 for the two-block one. `wave12_readout.py --round w19`; tables regenerated by `notes_tex_tables.py`; section 3 gains the round-8 subsection and the open list drops width and depth.

![](./assets/images/028-gene-essentiality/w19_readout_2026-09-16-20-26-18.svg)

| arm (context head + FBA affine, round-3 levers) | params | seeds 2700, 2701, 2702 | mean | sd | paired vs round 5 |
|---|---|---|---|---|---|
| width 12, one block (round 5) | 16,566 | 0.814, 0.805, 0.844 | 0.821 | 0.020 | |
| width 24 | 40,902 | 0.818, 0.822, 0.812 | 0.817 | 0.005 | +0.004, +0.017, -0.032 |
| width 48 | 112,902 | 0.827, 0.819, 0.826 | 0.824 | 0.004 | +0.013, +0.014, -0.018 |
| two blocks, width 12 | 18,450 | 0.810, 0.820, 0.806 | 0.812 | 0.007 | -0.004, +0.015, -0.038 |
| two-layer preprocessor, width 12 | 542,480 | 0.804, 0.829, 0.828 | 0.821 | 0.014 | -0.010, +0.024, -0.016 |

- **Pre-registered read: pass if seed mean > 0.83 with 3/3 seeds above round 5.** No arm passes. Width 48 is the best seed mean (0.824) and the tightest (sd 0.004), up on two seeds and down on 2702; every arm is below the round-5 cell's 0.844 on seed 2702, which is the highest single seed in the document. Every round-8 mean lies inside round 5's pooled bootstrap 0.84 [0.76, 0.91].
- **Seed spread narrows with width** (0.020 at 12, 0.005 at 24, 0.004 at 48); the wave-9 collapse at width 90 did not recur at 48.
- **Validation curves peak 0.80 to 0.82 between epochs 40 and 90 on every arm and drift down**, most at width 24 (to 0.75 by epoch 200); selected epochs 41 to 144.
- **Read:** capacity is not the lever on this endpoint. The transformer with FBA's four ratios sits at 0.81 to 0.82 whatever its size, level with the ladder's best rungs (0.806 to 0.830). Left on the transformer side: the chromosome-pathway indicator (the ladder's best selected rung with FBA, 0.830) and the 17-encoding concatenation as inputs, neither given to the transformer yet; and a solve-derived encoding.

Example runs, width 48 seed 2700 (0.827) and two blocks seed 2702 (0.806):

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/8ju0l6vd>

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/ftluim1i>

## 2026.09.16 - Rounds 9 and 10 launched: the FBA affine on the ladder's other encodings, and the flux layer warm-started from the deletion's own solve

Round 8 closed capacity as a lever, so the two levers left are the transformer's input and what the layer proposes.

**Round 9 (w20, packs 2280 to 2283, 22:59).** The round-5 cell (width 12, one block, context head + FBA affine, ProtT5 input) opened on the ladder's other encodings through `--node-embeddings`: the 197-d chromosome-pathway indicator (`chromfba`; the ladder's best selected rung with FBA, 0.830), ESM2 (`esmfba`), the seven-encoding `mixed` concatenation (`mixfba`), and chromosome-pathway + ProtT5 (`chromprotfba`). Three seeds each. Read with `wave12_readout.py --round w20`. Pre-registered read as in round 8: a cell passes the ladder if its seed mean clears 0.83 with three of three seeds above the round-5 cell (0.814 / 0.805 / 0.844). The first submit (jobs 2272 to 2279) died in one second on a pack path given relative to the experiment folder instead of the worktree; `ESS_PACK` is worktree-relative.

**Round 10 (w21, packs `w21_warm*.pack`, queued behind round 9).** `ess_reference_flux_table.py` solves one pFBA per labeled gene (1,121; genes outside the GEM get the wild type) on YPD through the same `media_report` the cell's box is built with, and writes the layer's per-genotype reference table (`$DATA_ROOT/data/torchcell/reference_flux/ess_reference_flux_table_YPD.npz`). Under `--flux-parameterization warmstart --flux-reference-table <that>` the layer proposes `v_ref(deletion) + alpha N z`, so the growth column starts at the deletion's own pFBA growth rather than at the free signed coordinate the probe found. Arms on the `flux_projection_anchored` arm at constraint scale 0: forced growth column with alpha frozen at 0.01 (`warmbio`), gated growth column + context head with alpha frozen at 0.01 (`warmgate`), alpha learned (`warmgatealpha`), alpha frozen at 0.1 (`warmgate10`). Pre-registered reads: (1) the probe at initialization must read the layer's growth column at FBA's level on YPD (AUROC of the negative column about 0.66 on the 195 genes) or the warm start did not reach the readout; (2) `warmgate` passes if its seed mean sits at the round-5 affine cell's 0.82 with three of three seeds above the round-4 gated cell (0.766 / 0.782 / 0.765), which would say the module carries the solve through the layer with gradients through it; (3) `warmgatealpha` against `warmgate` says whether training spends the feasibility the warm start bought (alpha growth, as in job 1946).

**Ladder, solve-derived encodings.** The same script writes `ess_solve_encodings_YPD.npz`; `logistic_baselines.py --only-missing` fits three new encodings on the split: the deletion's pFBA flux vector (4,131-d), its difference from the wild-type vector, and eight per-gene statistics of the wild-type solve and its variability on the gene's own reactions (reaction count, reactions carrying flux, max and sum absolute flux, forced flux, min and max FVA range, growth ratio). The question is whether the solve has more to give than four growth ratios.

## 2026.09.16 - Round 9 read: chromosome-pathway + ProtT5 with the FBA affine passes the ladder at 0.871; the solve's own encodings score 0.80 on the ladder with no sequence

Packs 2280 to 2283 (19 min each). `wave12_readout.py --round w20`.

![](./assets/images/028-gene-essentiality/w20_readout_2026-09-16-23-22-13.svg)

| arm (context head + FBA affine, round-3 levers, width 12) | params | seeds 2700, 2701, 2702 | mean | sd | paired vs round 5 |
|---|---|---|---|---|---|
| ProtT5 (round 5) | 16,566 | 0.814, 0.805, 0.844 | 0.821 | 0.020 | |
| chromosome-pathway indicator | 6,642 | 0.810, 0.759, 0.774 | 0.781 | 0.027 | -0.004, -0.046, -0.070 |
| chromosome-pathway + ProtT5 | 18,930 | 0.860, 0.866, 0.888 | 0.871 | 0.015 | +0.046, +0.061, +0.044 |
| ESM2 | 19,638 | 0.734, 0.736, 0.749 | 0.740 | 0.008 | -0.080, -0.069, -0.095 |
| seven-encoding concatenation | 78,066 | 0.822, 0.810, 0.827 | 0.820 | 0.009 | +0.008, +0.005, -0.017 |

- **PASSES the pre-registered read** (seed mean > 0.83, 3/3 above round 5): chromosome-pathway + ProtT5, 0.871, pooled 0.868 [0.80, 0.93]; the interval's lower end is the ladder's best rung. First cell to do so. Selects epochs 38 to 53; validation peaks ~0.85 at epochs 25-50 and drifts to 0.80.
- The indicator ALONE is unstable through a 12-wide preprocessor (two seeds select epoch 2, validation falls from 0.73, preprocessor gradient norm ~5 all run) and scores 0.781; the ladder's logistic rung on it is 0.830. ESM2 0.740 (0.08 below ProtT5). The seven-way concatenation returns to ProtT5's number at 4.7x the parameters.
- **Round 12 (w23, packs 2291 to 2293, queued)** confirms the passing cell: the five fold models (`chromprotcv0-4`, for the ensemble under FCL's protocol via `cv_ensemble_readout.py --prefix w23_chromprotcv`) and seeds 2703 to 2706 on the split file. The number is reported after that, not from one cell's three seeds.

Example runs, chromosome-pathway + ProtT5 seed 2702 (0.888) and the indicator alone seed 2701 (0.759):

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/bcd322yd>

<https://wandb.ai/zhao-group/torchcell_028_gene_essentiality/runs/1zxq5as9>

**Ladder, solve-derived encodings (`logistic_baselines.py --only-missing`, from `ess_solve_encodings_YPD.npz`).** CV-selected rungs: eight reaction statistics (WT pFBA + FVA on the gene's reactions, incl. the growth ratio) random forest 0.797 [0.70, 0.88] alone, 0.817 [0.73, 0.90] + FBA (CV 0.847, highest in the ladder); the 4,131-d deletion pFBA vector RF 0.756 (logistic 0.771 [0.66, 0.87]); its delta from WT the same. The growth ratio alone is 0.645. So the solve carries ~0.15 AUROC beyond its growth ratio, level with ProtT5, from no sequence. Figure `classical_ladder_2026-09-16-23-16-30.svg`; section 2 updated.

**Round 11 (w22, packs 2288 to 2290, queued)**: `--ess-solve-features {stats,pfba,stats_pfba}` appends those columns (standardized) to the fba readout's affine on the ProtT5 cell. Pre-registered: passes if seed mean > 0.83 with 3/3 above round 5. The winners of rounds 9 and 11 combine next if both pass.
