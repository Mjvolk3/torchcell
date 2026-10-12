---
id: wfuviwcnr1xe1zeklbfysnv
title: Fcl_comparison
desc: ''
updated: 1789538029015
created: 1789538029015
---

Copied 2026-09-16 from [[experiments.027-betaxanthin-metabolic.scripts.fcl_comparison]]; the script now lives at experiments/028-gene-essentiality/scripts/fcl_comparison.py and reads results/ and $DATA_ROOT/test-predictions-028.

## 2026.09.16 - Reread: the paper prints three yeast AUROCs on three gene sets

`experiments/027-betaxanthin-metabolic/scripts/fcl_comparison.py` scores every method on one gene set and reproduces every published yeast number from the released files before reporting anything. The reread against the authors' own `figures/fig3/fig3a.ipynb` and Supplementary Table S5 turned up a third published value the script had not been checking, and it is the one printed in the figure.

| printed where | value | gene set | statistic | recomputed |
|---|---|---|---|---|
| Fig. 3a legend, Table S5 "FCL (5-fold CV)" | 0.78 +/- 0.065 | validation genes of five folds inside the 898 training genes; 179 per fold, 590 distinct, a gene in one to four folds | mean and population SD over folds of a sample-level AUROC, 124 flux samples per gene | 0.7770 +/- 0.0650 |
| Table S5 "FCL (test)" | 0.73 | the 223 held-out genes | gene-level AUROC of the score averaged over the five fold models | 0.7308 |
| Fig. 3a legend, Table S5 "FBA" | 0.69 | all labeled genes, 1,121 of the 1,124 released rows | one AUROC of the negated cobrapy deletion growth | 0.6931 (0.6945 on all 1,124 rows) |

So neither published pair is a head-to-head. The figure's 0.78 against 0.69 sets training-side genes against all genes, and Table S5's 0.73 against 0.69 sets the held-out genes against all genes; on the held-out 223 FBA scores 0.6300. A model scored on the held-out genes is compared against the 0.73 line, never the 0.78. On the 195 held-out genes our cell graph resolves, FCL scores 0.7421 and FBA 0.6400, and every one of our cells is scored on exactly those 195.

**Whose labels our model trains on.** The released split's. The trainer takes the label from `yeast_essentiality_test_split.csv`, joined to each single-deletion genotype by its deleted gene; the knowledge-graph build (SGD essential genes united with the Costanzo 2016 and Kuzmin 2018 single deletions) supplies only the genotypes and the cell graph, and every phenotype column it carries is ignored for this head. The 223 held-out genes are pinned as our test split and our validation genes come from the 898 training genes. The SGD list agrees with the released label on 98.8 percent of the 1,121 genes (`results/essentiality_label_agreement.json`, 13 disagreements), so the queried label would not be a different target, but it is not the one used.

**Cells are found through result files.** Each `wave*_ess_*_e80_s*.json` result names its cell; its dump is `TEST_PRED_DIR/<result stem>.json`, must agree with the result on every identity field, and must reproduce the result's own pinned AUROC to 1e-6. A result without such a dump is listed under `cells_without_predictions` and not scored. See [[experiments.027-betaxanthin-metabolic.scripts.archive_essentiality_dumps]] for why six cells are in that list on 2026-09-16.

Scored on the 195 genes at this rerun (`results/fcl_comparison.json`):

| cell | seeds | AUROC by seed | FCL on the same genes |
|---|---|---|---|
| pooled, hidden 12, 1 layer, preprocessor 1 (16,874 parameters) | 2700, 2701, 2702 | 0.8011, 0.7860, 0.7958 | 0.7421 |
| flux biomass, hidden 90, 2 layers (934,022 parameters) | 2700, 2701, 2702 | 0.5398, 0.7579, 0.5115 | 0.7421 |

Three seeds of the width-12 cell sit above FCL by 0.044 to 0.059, with a gene-bootstrap interval about 0.17 wide on each; that is a consistent direction, not a settled margin. The width-90 and width-24 pooled cells are unscored until their reruns land.
