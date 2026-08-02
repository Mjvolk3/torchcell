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

All figures from `experiments/020-cachera-betaxanthin/scripts/plot_merzbacher_comparison.py`,
built from the 10 finished Delta `020_v4` cells (`$DATA_ROOT/test-predictions/`) and their
shipped `figures/fig4/fig4c_*.csv`.

- **Scored on 639 genes** carrying their label, a CGT prediction, and a raw screen value.
  Their 640th (IPP1/YBR011C) is essential -- no deletion strain exists, so no screen could
  have measured it.
- **Their model = RandomForest_Resampled**, their best (the other three Fig 4b models read
  0.56-0.64 gene-level against RF's 0.700).
- **CGT = one cell, selected by VALIDATION** (`s09_L6_maskon_lr0.0001_yj`, val pearson
  0.3639), never by its score on their test genes -- across ~10 cells at sigma ~ 0.03,
  picking on test would be worth ~2 sigma of free improvement.
- **Provenance asserted, not assumed:** our re-derivation of their deployed gene-level
  majority vote reproduces their published accuracy (0.7011 vs 0.700) or the script exits.

### TL;DR

**Neither model has usable signal on this task, and the published accuracy says nothing.**

| claim | verdict |
|---|---|
| RF beats the majority baseline meaningfully | **No** -- 0.701 vs 0.674, achieved by calling 95% of genes medium |
| CGT beats RF on accuracy | **No, and the question is empty** -- rank-matched, 0.549 vs 0.556 |
| CGT beats RF on global rank correlation | **Yes** -- Spearman +0.105 vs +0.039 |
| CGT beats RF at finding high producers | **No** -- ROC AUC 0.557 vs 0.570; precision@k comparable |
| Either model separates the three classes | **No** -- median percentile rank ~50 in every class, both models |
| The two models agree | **No** -- rank correlation -0.128; they are wrong about different genes |

The one defensible CGT advantage is the global Spearman, on a continuous prediction their
classifier cannot produce at all. It is a weak advantage on a task neither model solves.

---

### Fig 1 -- per-gene spread (639 points)

![](assets/images/020-cachera-betaxanthin/merzbacher_fig1_scatter_spread_2026-08-01-22-54-23.svg)

**Both models emit a nearly constant value.** 80% of their genes fall inside an E[class] band
of 0.22 on a 0-2 scale; 80% of CGT's inside 0.33 z-units. That is why the classification
comparison carries so little -- there is barely any variation to classify with.

| pair | Pearson | Spearman |
|---|--:|--:|
| measured vs Cachera RF | +0.244 | **+0.039** |
| measured vs CGT | +0.294 | **+0.105** |
| Cachera RF vs CGT | +0.108 | **-0.128** |

Pearson sits far above Spearman on both model rows: each gets a handful of extreme genes
roughly right and is otherwise flat, so Pearson flatters both. In RANK the two models are
slightly ANTI-correlated.

*Colour is Merzbacher's released label; x is OUR copy of the same screen. See Fig 7 for how
tightly those correspond.*

### Fig 2 -- why the accuracy comparison is degenerate

![](assets/images/020-cachera-betaxanthin/merzbacher_fig2_accuracy_artifact_2026-08-01-22-54-23.svg)

RF's 0.701 clears the 0.674 majority rate by 0.027, and does so by calling 95% of genes
medium; CGT's absolute binning sits at 0.681 doing the same (94% medium, and it never calls a
single gene low). Force both to the true class marginal and RF drops to 0.556, CGT to 0.549.
**Accuracy above the majority line is the conservative-majority strategy, not discrimination
-- for both models.** Evidence for the TL;DR, not a result in itself.

### Fig 3 -- precision@k for high producers

![](assets/images/020-cachera-betaxanthin/merzbacher_fig3_precision_at_k_2026-08-01-22-54-23.svg)

The decision-relevant, binning-free view: of the top k genes a model nominates, what fraction
are truly high? Random is 0.156.

**Both models are genuinely enriched at small k** -- around 0.70 at k=25, i.e. ~4.5x random --
and they track each other closely, RF marginally ahead below k~30 and CGT ahead from ~30-100.
By k=150 both approach the base rate.

This CORRECTS a reading taken from Spearman alone. Their RF is not flat everywhere: it is
uninformative in the bulk but its most confident calls are strongly enriched. A global rank
statistic averages that away.

The nine unselected CGT cells span 0.10 to 0.65 at k=50 -- some below random. Cell choice
matters more than the CGT-vs-RF gap.

### Fig 4 -- score by class, as percentile rank (the decisive one)

![](assets/images/020-cachera-betaxanthin/merzbacher_fig4_score_by_class_2026-08-01-22-54-23.svg)

Percentile rank puts two models with incomparable units (E[class] on 0-2, CGT z-scores) on one
axis. A model that separates the classes shows three stepping boxes; a useless one puts all
three on 50.

Median percentile rank by Merzbacher label:

| model | low | medium | high |
|---|--:|--:|--:|
| Cachera RF | 54 | 49 | 56 |
| CGT | 48 | 51 | 52 |

**Every box sits on 50.** Neither model orders the classes in the bulk of the population. Read
with Fig 3, the picture is coherent: whatever signal exists lives in a small number of extreme
calls, not in a population-wide ordering. This is the figure that says the task is unsolved.

### Fig 5 -- ROC for high-producer detection

![](assets/images/020-cachera-betaxanthin/merzbacher_fig5_roc_high_producers_2026-08-01-22-54-23.svg)

Threshold-free, no binning, no marginal, one number per model. **Cachera RF 0.570, CGT
val-selected 0.557** -- RF slightly ahead, both near the 0.500 chance line.

So the honest summary is metric-dependent: CGT wins on global Spearman, RF wins on
high-producer AUC, and precision@k is a wash. None of the gaps is large.

### Fig 6 -- how much is cell selection?

![](assets/images/020-cachera-betaxanthin/merzbacher_fig6_cell_spread_2026-08-01-22-54-23.svg)

Ten grid cells against RF's single value.

- **Spearman:** CGT cells span 0.013-0.158; 8 of 10 beat RF's 0.039. The val-selected cell
  (0.105) is upper-middle, not the max.
- **ROC AUC:** CGT cells span 0.406-0.695; only 3 of 10 beat RF's 0.570, and the val-selected
  (0.557) is NOT one of them.

The spread is wider than the between-model gap on both metrics. Four of the low points are the
lr=1e-3 cells that collapsed to a constant predictor. **Any single-number CGT-vs-RF claim is
dominated by which cell you quote**, which is exactly why the cell here is fixed by validation
in advance.

### Fig 7 -- do their labels track our copy of the screen?

![](assets/images/020-cachera-betaxanthin/merzbacher_fig7_label_provenance_2026-08-01-22-54-23.svg)

The whole comparison is scored against their released labels, so this bounds everything above.

Spearman 0.731 against an **attainable ceiling of 0.827** -- a 3-level ordinal against a
continuous variable cannot reach 1.0 because of its own ties, so the raw number is
uninterpretable alone. The labels reach **88% of the maximum possible**, which is good.

The residual: **18.8% of genes would bin differently** under a pure rank-split of our values,
because their min-max rule scales by extremes that differ between their (smaller) copy and
ours. Their "low" reaches +0.02 on our axis while their "medium" starts at -1.05. A model
perfectly predicting OUR measured value would still be scored wrong on ~19% of genes.

---

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

**Diagnostics not carried as figures** (regenerate with `--all`): class distribution,
per-class recall, rank-matched confusion matrices. Rank-matched, CGT reaches 0.29 recall on
high producers vs their published 0.18 and 0.19 on low vs their 0.06, paid for in medium recall
(0.70 vs 0.98); the two rank-binned label vectors agree on only 52% of genes.

**Caveat on MCC.** The 0.205 quoted elsewhere is their PER-FLUX-SAMPLE MCC; they never
published a gene-level MCC. Accuracy and high-producer recall are gene-level on both sides.

**Status.** 10 of 24 Delta `020_v4` cells have finished; wave C (L=4) has only just started.
Re-running the rsync and both scripts picks up the rest. The CGT points can still move.

**Slide set:** Fig 4 and Fig 2 make the argument most economically (the task is unsolved, and
the published accuracy is a majority artifact). Fig 1 and Fig 6 are the honesty panels.
