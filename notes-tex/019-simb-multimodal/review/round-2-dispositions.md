# Round 2 review dispositions

Comments pulled with `python notes-tex/common/zotero_comments.py 019-simb-multimodal` off
`019-simb-multimodal_2026-08-27-17-14-39_507ee4b6.pdf`. **136 annotations, 127 carrying
written text.** The nine without text are highlights only ([8], [12], [26], [31], [69],
[106], [108], [124], [129]) and need no disposition. Keys are the stable Zotero handles.

## Summary

| outcome | count |
|---|---|
| Fixed in the document | 127 of 127 |
| of which: a claim in revision 2 was **wrong** and is corrected | 9 |
| of which: answered by a **new measurement**, not by rewording | 11 |
| of which: **deferred by the author's own decision**, recorded as such | 6 |
| Not addressed | 0 |
| Left for the author (status promotion) | 1 ([44]) |

## The nine claims revision 2 got wrong

These matter more than the rest, so they are listed first.

| # | key | what revision 2 said | what is true |
|---|---|---|---|
| 7, 23, 89 | `3XL5BYU5` etc | `nmse` "never returns below 1", so the model is "no better than the mean" | At the **raw** Pearson peak `nmse` is 0.9928, **below** 1; at the smoothed peak 1.0070. The two are ~1.8 sigma apart in logging noise. The defensible word is **parity**, and 0.939 once rescaled. New `tab:nmse-parity`. |
| 7 | `3XL5BYU5` | squared error is minimized early and never returns | `mse` falls, rises to a local max near epoch 900, then falls to a **late global** minimum at epoch 9,184, at the Pearson peak |
| 6 | `VTGLESU2` | the 0.4301 betaxanthin number comes "from a study in a queued project" | It is the **same runs under a different scoring rule**: 0.4301 is the Optuna raw per-epoch max, 0.4015 is `roll_max` on the same W&B project. Nothing was queued. |
| 21, 22, 103, 109, 123, 131 | `TEYC2NGS` etc | "the graph prior is at chance; free the nine masked heads" | The probe asked the narrow question. The **pair form** reaches AUC 0.6579 against a 0.4971 control on the same strand, and signal differs by phenotype. **Recommendation withdrawn**; masks stay. |
| 84 | `CVPADY7A` | beta-carotene best is 0.118, peaking late, so it shares expression's under-training | Best is **0.1371**, peaking at **epoch 9 of 49**. It groups with the metabolite strands, not with expression. The old number was third best. |
| 86 | `NEE4PJ9I` | "the 010 fitness and interaction task shows neither behavior, both losses falling monotonically" | 010 trains **gene interaction only**; neither loss is monotone past 30 epochs; and 010 **does** show the loss-metric divergence, with the **opposite sign**. New `tab:010-divergence`. |
| 59 | `I7FJIKCF` | FCL's "validation folds appear to be drawn from the same 640 genes" | Their validation set **is** their test set exactly: 640 of 640, both differences empty. Their training loop's subtraction is a no-op. |
| 43, 112 | `CYMCCMJP`, `88CZBF3E` | `C115_A` is a "whole-cell axis ratio" | SI Table 2 calls it **roundness of the mother cell**, a shape parameter, not a size one |
| 14, 72, 74 | `7TT6KRJH` etc | "roughly three quarters of the profile's power survives regressing out fitness" | On the enlarged fitness union it is **94 percent**. Most of the apparent shrinkage was the gene set the Costanzo-only control selects, not the control. |

## Answered by a new measurement

| # | key | question | measurement |
|---|---|---|---|
| 100, 109, 123, 131 | `S9H9NSSQ` | does the graph prior hold for phenotypes other than expression | Pair-form probe on all six strands, 60,000 pairs per cell. `tab:graph-pair-form`, `fig:graph-probe` (nine panels) |
| 118, 136 | `RCH5WVW9`, `7NHJ2R8A` | is m = 1000 revealed genes too expensive to be interesting | Dense low-m curve: **100 random genes reach 77 percent** of the m = 1000 cross-study value, 10 reach 46 percent. Variance-chosen sets beat random at the cheap end. `tab:gene-budget`, new `conditioning_gene_budget.py` |
| 80, 67 | `AGIPNSZB` | is the amino-acid coupling redundant with what betaxanthin supervision already teaches | **No.** 80 percent survives residualizing out a trained model's own held-out predictions; the two predictions correlate at only 0.22. New `sec:aa-redundancy`, `tab:aa-incremental`, `bx_aa_incremental_over_model.py` |
| 72, 74 | `6J8VNSTR`, `WDPRZLAH` | do other single-mutant fitness screens fill the 724-gene gap | **Kuzmin does not** (56 of 724, same SGA array). Baryshnikova + O'Duibhir + Costanzo NatMX reach 4,214 of 4,432. `tab:aa-fitness-union` |
| 47, 48, 49 | `XXEIM3PQ` etc | can the 537 lost training strains be recovered by a better split | **No** under any scheme; the proposal is numerically the split `020_v4` already ran. What it buys is a **reporting** change, free from existing checkpoints. `tab:bx-resplit` |
| 86 | `NEE4PJ9I` | is the 010 claim true | 20 runs pulled; see above |
| 59 | `I7FJIKCF` | are FCL's val and test the same | Zenodo release read directly; see above |
| 68 | `SDPV8HSE` | report a spread on the ridge table | Three spreads emitted; the table now carries the Fisher SE and the across-shuffle SD, with the difference explained |
| 43, 112 | `CYMCCMJP` | expand the CalMorph table with descriptions | 20 rows, all descriptions sourced verbatim from Ohya SI Table 2 (sha256 pinned), zero unsourced. `tab:morph-scalar` |
| 112 | `88CZBF3E` | does cell size work as a warm-up target | **No.** Ceilings 0.88 to 0.97, robust CV 0.02 to 0.12, ranks 78 to 245 of 275. `tab:morph-size` |
| 66 | `RJK48J4X` | can we say the metabolite runs "converge" | Reworded to what is observed (a maximum inside the budget) with the expression precedent named |

## Deferred by the author's decision, recorded in the document

| # | key | decision |
|---|---|---|
| 111, 122, 134, 135 | `I3AZKZ6B` etc | Morphology at full scale and the scalar warm-up are **deferred behind expression**. `sec:directions` "Deferred, and why"; `sec:campaign` "Morphology, deferred, sized for when it starts" |
| 113, 130 | `R6PNIPQN`, `BJMUV8PD` | The post-hoc rescale is **not** part of this campaign; do it when a figure is drawn |
| 114, 127 | `PDCBMYCY`, `UE3UXKBP` | **Resume from checkpoint rather than restart**, multi-GPU DDP on IGB `mmli`. Now item 1 of "Do now" |
| 22, 109, 131 | `LKDA4IE6` etc | Keep the graph masks, **add** unconstrained heads beside them rather than freeing the masked ones |
| 39, 116, 126 | `ZTBMZNBH` etc | The morphology question is reframed to "does adding fitness improve morphology prediction at all", reported per feature as well as averaged |
| 96 | `EFQKM9NM` | Joint proteome and expression training is **not on the near roadmap**; the overlap measurement is stated as a gate, not a plan |

## Everything else, by section

### Front matter and terms

| # | key | concern | disposition |
|---|---|---|---|
| 1 | `DG2Z43TN` | abstract not concise, may not cover the whole document | Rewritten. One clause per strand, all six covered, plus the graph-prior reversal and the campaign decision |
| 2 | `CZHCDBF8` | "objective fighting itself" unclear, is it MSE vs Pearson | Split into the two separate defects it was conflating, as finding 2(i) and 2(ii) |
| 3 | `QVFJABL8` | make clear whether heads are linear probes | New **Head** term states they are not, and names where a linear probe *is* used |
| 4 | `VLFQU5WH` | "prosecuted" the right word | No. Now "carried" |
| 5 | `RWHXK9ZT` | doesn't `pearson_per_instance` tell us strain trends | Yes, and the term entry now says so, plus why it is still not the objective (mean-collapse still scores ~0.4) |
| 87 | `YLZXLINU` | want a contents page and a terms table; "marginal" used two ways | Contents already present; terms section rewritten alphabetically, 17 entries. **Marginal** entry disambiguates the two senses and retires the colloquial one |
| 90 | `E5J9TIDL` | define under-shrinkage | New term entry with the calibration ratio `s` |
| 95 | `783YQUQX` | define selection inflation | New term entry |
| 91, 92, 93, 94 | `4WX4CS64` etc | are grid cells data splits, which split, is configuration the model config | New **Configuration, grid cell, trial** entry: three names for one object, never a data split, split pinned across all of them |
| 45 | `C4AL6YLC` | difference between reliability and the Pearson ceiling | New term entry plus the full derivation in `sec:morphology` |
| 71, 79 | `WV796MDA`, `KJKAJCUB` | what does "regress out" mean | New term entry with the concrete two-sided residual procedure |
| 32 | `CBXWR2VJ` | explain how the oracle works | New **Oracle** term entry plus a six-step method block in `sec:oracle` |
| 81 | `7LJLMHD9` | what is Thorndike | New **Range restriction** term entry |
| 119 | `4MX2DVSZ` | what is a "defensible product surface" | Phrase removed; replaced with what it concretely is |

### Expression

| # | key | disposition |
|---|---|---|
| 9 `T7APESIP` | New "What the post-hoc rescale is, concretely" block: the scalar, why no correlation moves, why it is legitimate |
| 10, 115 `6ILTSBAH`, `VMGLGAGH` | Elevated to the most consequential structural finding, with the mechanism spelled out and the explicit answer that the operator only becomes expressive with more than one perturbed gene |
| 11 `5HACDECW` | "every feature is exactly zero" explained: the measured values are not zero, the reveal-mask-gated branch that reads them is off |
| 27 `FKRLU2AA` | "we obviously need this" reflected in the finding's framing |
| 28 `769PTZS7` | Stated plainly that `tab:pair-rank` is a **derivation**, not a result; only two rungs trained, and not at matched budgets |
| 29 `PBBAPBK8` | Low-rank check explained as a method: truncated SVD, fit on what it scores, an upper bound, and what a null on it would then mean |
| 30 `KY78HZ9I` | "look at scoring at k" added as a real option with the caveat that it changes the task |
| 33 `G6C3VS7K` | The m = 10 result now called out on its own |
| 34 `9PATB5Y8` | Figure 2 rebuilt: old panel B dropped, re-lettered a-e, every panel states its conclusion in its title, d carries the effect size, e is explicitly linked to d |
| 35 `S8XB7PJW` | "worth keeping how" answered: the ~33 coordinated directions are why the oracle works and what a per-gene readout discards |
| 36 `FVWC79HY` | Residual covariance measurement now stated inline with its split-half r, null, and effective rank |
| 37 `UBIMQ6DY` | Orthogonality explained as an experiment (replace the mean with a genotype kNN predictor and re-run) rather than asserted |
| 38 `EUGL2SIA` | "Cannot say" now cross-references `sec:graph-probe` and adds the unmeasured turn epoch |
| 105, 121, 136 `PESXUDXG` etc | kNN probe and oracle brought into the document with full methods and a new four-panel figure |
| 24, 25 `W6CBTBUZ`, `SUHY7UEP` | Figure 1: every series named in-panel by legend; the `s = r` label now rotated parallel to its line in data space |

### Morphology

| # | key | disposition |
|---|---|---|
| 40 `LNBV4S69` | Output masking named as the documented way around the 1,440 cap, in both `sec:morphology` and the summary |
| 41 `9DV8XK6F` | The intended design (train on all morphology with masking, then add expression) recorded, with the practical reason expression goes first |
| 42 `MDD7F48W` | Table expanded to 20 rows with descriptions |
| 44 `RMTEJGFD` | Recommendation kept. **Status left at `todo` for the author to promote**, per the repo rule that agents do not self-promote a section |
| 46 `HPKMTKA8` | The plateau argument rewritten: what a trial is, the two things that follow, and that they point in opposite directions |

### Betaxanthin

| # | key | disposition |
|---|---|---|
| 13 `3EMFXF56` | The 0.298-vs-model comparison **withdrawn as invalid** (different splits and scoring rules) and replaced by the paired measurement in `sec:aa-redundancy` |
| 50 `S5U68X8J` | Base rate now drawn and labeled in the figure and named in the caption |
| 51, 52 `59YB7KNL`, `7EVZSC46` | The anti-correlation rewritten: what agreeing "through the truth alone" would give, and the operational consequence separated from the unexplained sign |
| 53 `MYIHVZNI` | Figure split into two panels; selected cell distinguished from the envelope; **3 of 10 cells sit below base rate at k = 50** stated from computed values; the k = 10 step-size caveat added |
| 54, 57, 63 `NEWNCH9B` etc | New "How the class labels are derived" and "Why every metric is rank-based or re-binned" blocks |
| 55, 56 `4S8WG8B8`, `CREPJEFM` | "Inside/outside" defined as yeast-GEM membership, not training set; Fisher r-to-z explained |
| 58 `2BJFHRVU` | Section rewritten |
| 60 `8GY22JGM` | Replaced with the concrete asymmetry: their model was selected on its test set, ours on validation |
| 61 `6XG6KXBY` | Reframed as hyperparameter sensitivity with the split pinned, and the protocol stated |
| 62 `VGTT22XL` | Their binning rule, the 18.8 percent disagreement, and what we do about it, all stated |
| 64 `P4J8U482` | Gray `keybox` callouts added at the end of the betaxanthin, morphology, amino-acid, beta-carotene, expression and graph-probe sections |
| 65 `HQDCA39E` | Rank-matching explained, including why it is not available in deployment |

### Amino acid, beta-carotene, common, directions

| # | key | disposition |
|---|---|---|
| 15, 70 `NFJY9IE3`, `3U75MZID` | Why fitness enters at all, stated as the shared-growth confound it controls for |
| 16, 17 `3TSD2V2S`, `4G2CXAF9` | The negative auxiliary-head result and the real coupling explicitly separated; "how to use it" answered by the constraint-based layer |
| 18, 104, 110, 132, 133 `WUU9M7JL` etc | Direction: the mask does not have to be symmetric even though attention is, `head_mask[i,j]` and `[j,i]` are written together today, and `QK^T` is already asymmetric |
| 19 `L6RUGNHY` | Reframed: same split for both methods means the comparison stands, sensitivity reported beside it |
| 20 `KDL9ALDS` | Recorded that the joint amino-acid question is **not** closed; the decisive replication experiment is named |
| 73 `N8XZ3G7Q` | The 0.464 comparison group spelled out |
| 75, 76, 77 `XUR99U6G` etc | Figure 7: panels state their machinery, a legend replaces the table cross-reference, panel c carries its finding, error bars added |
| 78 `BWPPS6IV` | Table repaired (it was rendering `nan` because 33 of 46 runs had no curve fetched) and the section **shortened** as asked, with the naive-experiment framing made explicit |
| 82 `ZYVXISZ7` | Range restriction explained with the intuition and the measured SD narrowing |
| 83 `JE46MC9E` | Pigment figure **split** into a betaxanthin figure in `sec:betaxanthin` and a beta-carotene figure in `sec:carotene` |
| 85 `JE46MC9E` | Duplicate strand-summary table removed from section 7; the coverage table (old table 19) rebuilt as a per-strand count table |
| 88 `N6V9E7TY` | The `1 - r^2` arithmetic derived in the calibration caption; the duplicated sentence removed |
| 97 `NAHSM3CR` | Both datasets named with DOIs and media |
| 98 `QK49AYPM` | Graph-prior section reframed as an investigation with a motivation, not an announcement |
| 99 `N2XAPWD9` | Verified: **all 14 figures land inside their own section** (`placeins` with `[section]` makes every `\section` a float barrier) |
| 101 `IDYRGAWG` | Bars now flush |
| 102 `58T2FWCB` | Open attention maps adopted as the "add unconstrained heads" arm |
| 107 `XXEFLKFD` | "share strains" vs "share genes" defined |
| 117 `JNQZF9RW` | Tyrosine arm kept as a negative control |
| 120 `EI3HV6Z6` | Placeholders come out of the manuscript **regardless** of what the campaign finds |
| 125 `TELTSP8K` | The wider case for a metabolic layer written out, and labeled a design argument rather than a measurement |
| 128 `92ZCIRE9` | The fair objection to `O_corr` answered: why correlation is rarely a training loss, and why `O_zmse` is the better-motivated arm |

## New provenance and defects surfaced while doing this

- **Leaderboard run coverage was 306 of 2,187**, reported as complete. Now **2,187 of 2,187**, in about 25 minutes. The candidate rule and its three biases are documented in `sec:coverage`.
- **A namespace bug in the graph probe**: the Ohya mirror stores ORFs lowercase, which would have given morphology zero graph overlap and a clean fabricated 0.5 on every graph. Fixed, with a hard guard that raises below 50 percent gene resolution.
- **`YBR011C` / `PPA1`**: an ambiguous alias resolving two ways between the split builder and the training build. Flagged in `sec:betaxanthin` as **needing verification before the comparison is published**; not confirmed either way here.
- **The merzbacher figure filenames were orphaning themselves** on every render (a fresh timestamp each run while the `.tex` referenced an old one). Stamp pinned; the real render time moved into `provenance.rendered_at`.
- `proteome_expression_eda.py` cannot reach the served NCSA database (the known cold-read `IOException`); it now takes an environment override and records which endpoint produced the numbers.

## Not addressed

None. Every comment carrying written text has a disposition above.

Two items are **open by decision rather than unaddressed**, and both are named in the
document rather than left implicit:

1. **[44] `RMTEJGFD`** asked for a green check on the morphology scalar section. Section
   status chips are the author's to promote; the section is left at `todo`.
2. **[136] `7NHJ2R8A`** asked for gene selection by ease of measurement. Not run, because no
   measurement-cost annotation exists in the repository and inventing one would have been a
   guess. Building that annotation is named as the prerequisite in `sec:campaign`.
