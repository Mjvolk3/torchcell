---
id: 0raebrho5u0wh46knjp5oai
title: Merzbacher Comparison — recipe and their true baseline
desc: 'How to run the betaxanthin head-to-head after training, and what Merzbacher 2025 actually achieved once class imbalance is accounted for'
updated: 1785204665038
created: 1785204665038
---

## 2026.07.27 — Read this before quoting any comparison number

Note crumbs so the head-to-head can be run cold. The protocol is fixed **before** we have a
model, so it is not chosen after seeing our number.

| what | where |
| --- | --- |
| split builder | `experiments/020-cachera-betaxanthin/scripts/build_merzbacher_split.py` |
| split output | `results/merzbacher_nested_split.json` |
| baseline analysis | `experiments/020-cachera-betaxanthin/scripts/analyze_merzbacher_baseline.py` |
| baseline output | `results/merzbacher_baseline_analysis.json` |
| their archive (sha256-pinned) | `$DATA_ROOT/data/merzbacher2025_fcl/` |

### What their deposit gives us — far more than the paper implies

The manuscript reports no split and an inconsistent test N (Fig 4b 659, Table S6 649, 20 % of
811 = 162). The Zenodo **code** deposit (10.5281/zenodo.15518666 → record 15761895) has:

| file | what it gives us |
| --- | --- |
| `data/yeast_production_test_split.csv` | the exact 640-ORF test set |
| `data/yeast_production_validation_split.csv` | same genes + released `label` 0/1/2 + CV `fold` |
| `training/yeast_training_production.py` | the exact binning rule |
| `figures/fig4/fig4b.csv` | per-fold accuracy, macro-F1, **MCC**, class-2 accuracy |
| `figures/fig4/fig4c_*.csv` | **per-gene, per-flux-sample predictions + class scores** |

Only the 49 MB code archive is mirrored; `data.zip` is 23 GB of flux samples we do not need.

### THEIR TRUE BASELINE — from their own shipped predictions

Aggregated to gene level by majority vote over each deletion's flux samples, exactly as their
`tools/knockout_voting.py` does. Best model, `RandomForestClassifier_Resampled`:

| true ↓ / pred → | low | **medium** | high |
| --- | --: | --: | --: |
| low (109) | 6 | **101** | 2 |
| medium (431) | 2 | **424** | 5 |
| high (100) | 0 | **82** | 18 |

- gene-level accuracy **0.700**, majority-class rate **0.673**
- **607 of 640 genes called medium — 94.8 %**
- **18 of 100 high producers found**
- majority predictor gets 431/640; theirs gets 448/640. **The entire margin is 17 genes.**

**They computed MCC and did not report it.** From `fig4b.csv`:

| model | accuracy | MCC | high-producer acc |
| --- | --: | --: | --: |
| RandomForest | 0.698 | **0.232** | 0.181 |
| RandomForest_Balanced | 0.698 | 0.233 | 0.117 |
| LogisticRegression | 0.515 | **0.041** | 0.233 |
| LinearSVC_Resampled | 0.445 | **0.031** | 0.273 |

MCC 0.23 is weak but **not zero** — there is real signal, concentrated in the majority class.
Be fair in any writeup: they claim "promising accuracy", not a significant gain.

**The decisive structural point:** every model that finds more high producers has *worse*
overall accuracy, because the only way to call more highs is to stop calling everything
medium. **Accuracy actively selects against the capability the task is about.**

### The recipe, fixed in advance

1. **Ground truth = THEIR RELEASED LABELS**, never labels we re-derive. Their rule min-max
   scales to [0,1] then cuts at 0.40 / 0.65, so the cuts depend on the observed extremes.
   Applied to our (larger) copy of the same screen it gives 107/476/56 against their
   109/431/100 — 81.2 % agreement, and **we call barely half as many high producers**. Same
   rule, same screen, different classes. Re-deriving would compare different tasks.
2. **Bin OUR predictions with thresholds fitted on the TRAIN split only.** Fitting on test
   leaks the class distribution, which on a 67 %-majority problem is most of the answer.
3. **Primary metrics are imbalance-immune**: Spearman, and top-*k* enrichment on high
   producers. Report **MCC** next to accuracy — they had it, so we should show it.
4. **Also report accuracy + per-class + high-producer recall**, so the comparison lands on
   their terms too.
5. **State the fraction of genes we call medium.** If ours is also ~95 %, we reproduced their
   failure mode rather than beat it, and must say so.

### Reconciliations to carry into any writeup — report, never resolve silently

- **639 of 640** of their test genes are in our test split. The one missing is `YBR011C` =
  **IPP1**, which is **essential** ([SGD S000000215](https://www.yeastgenome.org/locus/S000000215),
  null inviable). No haploid deletion collection contains *ipp1Δ*, so no screen could have
  measured it — **their test set contains a gene with no possible deletion phenotype.**
- **`PPA1` is an alias of both IPP1 and VMA16.** The screen has ONE `PPA1` row; their split
  has BOTH YBR011C and YHR026W as separate test genes. At least one of their test entries
  plausibly carries **VMA16's** value scored as if it were IPP1. Our screen-local
  disambiguation (`PPA1`→VMA16, `FEN1`→ELO2) is documented in the split builder, with the
  reasoning recorded so it can be re-derived if its premises change.
- **640 vs the paper's 811** — a 171-gene gap the Methods do not count ("sampling failed to
  converge").
- **Their validation folds appear drawn from the same 640 genes as their test list**
  (`yeast_training_production.py:124-143`), which would mean model selection touched test
  data. FLAGGED, not claimed — confirming needs `yeast_single_knockouts.npz` from the 23 GB
  archive.
- Our Cachera build is **stale** w.r.t. the name resolver (issue #195). The split is built
  from the RAW screen + resolver so it does not inherit that, but **training data still
  will** until a rebuild.

### What we can report that they cannot

- **Regression at all.** They tried and abandoned it — *"challenging with the limited number
  of knockouts at the high and low ends."* Our betaxanthin noise ceiling is **r = 0.914**
  (reliability 0.836, median 15 replicates), so regression is available to us.
- **The ~3,900 non-metabolic deletions.** FCL has no flux cone for a gene outside Yeast9, so
  it cannot make a prediction at all. We predict all ~4,700. That is a category difference,
  not a tie-break.

### Where our model stood when the sweeps were paused

`020-cachera-betaxanthin`, study `betaxanthin_002`, 33/40 trials before cancellation:
**val Pearson 0.430** — 47 % of the 0.914 ceiling — with `prot_T5_all` / learnable **False** /
`crps` / L=2 / hidden 90 / λ 6.5e-5 / decayed / yeo-johnson. The top-5 configs agreed closely,
so it is a plateau rather than a lucky trial. Companion arms: `021-ozaydin-beta-carotene`
(Spearman 0.223) and `022-mulleder-metabolome` (0.180, up from a historical ≤ 0.025).

Related: [[plan.cgt-metabolism-flux-layer.2026.07.26]] ·
[[plan.simb-2026-multimodal-cgt.2026.07.21]]

## 2026.08.01 - Head-to-head vs Cachera/Merzbacher on their 639 test genes

Two figures from `experiments/020-cachera-betaxanthin/scripts/plot_merzbacher_comparison.py`,
built from the 10 finished Delta `020_v4` cells (`$DATA_ROOT/test-predictions/`) and their
shipped `figures/fig4/fig4c_*.csv`. Scored on the 639 genes carrying their label, a CGT
prediction, and a raw screen value (their 640th, IPP1/YBR011C, is essential -- no deletion
strain exists, so no screen could have measured it).

**Only their RandomForest_Resampled**, their best model (the other three Fig 4b models read
0.56-0.64 gene-level against RF's 0.700). **CGT is one cell**, selected by VALIDATION
(`s09_L6_maskon_lr0.0001_yj`, val pearson 0.3639) and never by its score on their test genes:
across ~10 cells at sigma ~ 0.03, picking on test is worth ~2 sigma of free improvement.

Provenance: our re-derivation of their deployed gene-level majority vote reproduces their
published accuracy (0.7011 vs 0.700), asserted in the script rather than assumed.

### Fig 1 -- the spread, per gene (639 points)

![](assets/images/020-cachera-betaxanthin/merzbacher_fig1_scatter_spread_2026-08-01-22-47-34.svg)

**Both models emit a nearly constant value.** 80 % of their genes fall inside an E[class] band
of 0.22 on a 0-2 scale; 80 % of CGT's inside 0.33 z-units. That is the whole reason the
classification comparison carries so little: there is barely any variation to classify with.

The informative axis is the CONTINUOUS one -- the task they tried and abandoned:

| pair | Pearson | Spearman |
|---|--:|--:|
| measured vs Cachera RF | +0.244 | **+0.039** |
| measured vs CGT | +0.294 | **+0.105** |
| Cachera RF vs CGT | +0.108 | **-0.128** |

CGT's rank correlation with measured production is **~2.7x theirs**. Pearson sits far above
Spearman on both model rows because each gets a handful of extreme genes roughly right and is
otherwise flat, so Pearson flatters both. In RANK the two models are slightly
ANTI-correlated -- they are wrong about different genes.

### Fig 2 -- why the accuracy comparison is degenerate

![](assets/images/020-cachera-betaxanthin/merzbacher_fig2_accuracy_artifact_2026-08-01-22-47-34.svg)

Their RF's 0.701 clears the 0.674 majority rate by 0.027, and it does so by calling 95 % of
genes medium; CGT's absolute binning sits at 0.681 doing the same (94 % medium, and it never
calls a single gene low). Force both to the true class marginal and RF drops to 0.556, CGT to
0.549. **Accuracy above the majority line is the conservative-majority strategy, not
discrimination -- for both models.** This figure is EVIDENCE for that claim, not a result.

### Bottom line

**Neither model solves this task.** "CGT matches their best model on accuracy" is true and
nearly meaningless -- it is a statement about the majority class. The defensible claim is the
Spearman gap on a continuous prediction their classifier cannot produce at all.

### Method notes

**Can their side be ranked?** Yes -- they release more than classes. `fig4c_*.csv` carries
`score0/score1/score2` (P(low)/P(med)/P(high), summing to 1.0) for 124 flux samples per gene
across 640 genes, aggregating to 558 distinct gene-level values via
`E[class] = p_med + 2*p_high`. Two checks: (1) TIES ARE INERT -- 82 genes share a value and tie
order is arbitrary, but over 200 random permutations their accuracy moves 0.5563 +- 0.0008 and
high recall not at all; (2) THE AGGREGATION IS NOT LOAD-BEARING, and the one used is the most
generous to them -- `E[class]` 0.5556/0.290, `p_high - p_low` identical, `p_high` alone
0.5446/0.280.

**The rank-matched numbers use an ORACLE class marginal.** Rank-matching needs counts to cut
at, and these take them from the TRUE TEST labels (108/431/100), so both models are handed the
marginal of the answer. The sibling `evaluate_merzbacher_head_to_head.py` uses the TRAIN-POOL
marginal (107/504/28) and is leak-free. Same marginal on both sides, so the RELATIVE claim
holds; the ABSOLUTE values are not deployable accuracy:

| marginal | RF acc | CGT acc | RF hi-rec | CGT hi-rec |
|---|--:|--:|--:|--:|
| test counts (oracle, used here) | 0.5556 | 0.5493 | 0.290 | 0.290 |
| train-pool counts (leak-free) | 0.6228 | 0.6166 | 0.190 | 0.190 |

RF leads by ~0.006 either way, high recall identical either way -- robust to the choice, only
the level moves. The pool expects 28 high producers where the test set holds 100: their test
genes are enriched for extremes relative to the full screen.

**Diagnostics not carried as figures** (regenerate with `--all`): class distribution, per-class
recall, rank-matched confusion matrices. Rank-matched, CGT reaches 0.29 recall on high
producers vs their published 0.18 and 0.19 on low vs their 0.06, paid for in medium recall
(0.70 vs 0.98); the two rank-binned label vectors agree on only 52 % of genes.

**Caveat on MCC.** The 0.205 quoted elsewhere is their PER-FLUX-SAMPLE MCC; they never
published a gene-level MCC. Accuracy and high-producer recall are gene-level on both sides and
are the clean comparisons.
