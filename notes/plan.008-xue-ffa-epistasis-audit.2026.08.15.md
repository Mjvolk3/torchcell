---
id: re647rs4pf1b0mv0phbzwgn
title: '15'
desc: ''
updated: 1786822495471
created: 1786822495471
---

## 2026.08.15 - Audit and Re-Run of the 008 Epistasis Analysis

The 008 FFA epistasis analysis was written against early Opus models and has not been
re-examined since. Building the trajectory panels surfaced three defects, one of which
inverts a scientific conclusion, so the whole analysis is re-derived and re-run here.
Methodology reference is Kuzmin 2018, mirrored at
`$DATA_ROOT/torchcell-library/kuzminSystematicAnalysisComplex2018/`.

### Findings to fix

**F1 (critical, changes conclusions) -- degrees of freedom are 1-2 when they should be ~15.**
`compute_se_pvalue` is called with `df = max(1, min(n_i, n_j, n_k, n_ijk) - 1)`, which is
2 for a full triple and 1 whenever any constituent has 2 replicates (42 of 119 total-titer
triples). But tau is a linear combination of SEVEN independently measured strains
(f_i, f_j, f_k, f_ij, f_ik, f_jk, f_ijk), each with its own replicates, and its SE is the
delta-method combination of all seven. The t reference distribution must therefore use the
**Welch-Satterthwaite effective df** of that combination, not the smallest single input's
df. The current choice is drastically heavy-tailed and understates significance.

Sensitivity on total-titer trigenic interactions (n=119), FDR-BH within readout:

| df | raw P<0.05 | FDR<0.05 | positive AND FDR<0.05 |
| --- | --- | --- | --- |
| 1-2 (current) | 75 | 21 | **0** |
| 4 | 87 | 84 | 10 |
| 8 | 94 | 94 | 14 |
| 14 | 95 | 95 | 15 |

The claim "no positive trigenic interaction survives FDR" is an ARTIFACT of df=2. This must
be corrected before any biological reading of the panels.

Note the df also ignores `n_ij, n_ik, n_jk` entirely even though the doubles enter both tau
and its SE -- a second, smaller inconsistency the Welch-Satterthwaite fix subsumes.

**F2 (data labelling) -- FKH1 is read as the Abbreviations header.**
`load_ffa_data` reads the `Abbreviations` sheet with a default header row, consuming FKH1
and yielding 9 of 10 TFs; all 36 FKH1-containing triples are labelled with the bare letter
`F`. Grouping is consistent so the statistics are unaffected, but every committed artifact
carries the wrong gene name. Fix at the source (`header=None`) and assert 10 TFs.

**F3 (multiple-testing family) -- FDR pools all six FFA readouts.**
`apply_fdr_correction` flattens the whole p-value dict, so BH runs over 714 triple x readout
tests. Report BOTH the pooled correction and the within-readout correction, and make the
family explicit in every artifact rather than implicit.

**F4 (coverage) -- only 1 of 4 models has ever been run here.**
`additive_free_fatty_acid_interactions.py`, `glm_log_link_epistatic_interactions.py`, and
`log_ols_wt_differencing_epistatic_interactions.py` still carry hardcoded
`/Users/michaelvolk` paths and have never run on GilaHyper. No cross-model claim is
currently supported.

### Verified correct (do not change)

- **The multiplicative tau matches Kuzmin 2018.** Kuzmin defines
  `tau_ijk = f_ijk - f_i f_j f_k - eps_ij f_k - eps_ik f_j - eps_jk f_i` with
  `eps_ij = f_ij - f_i f_j`; expanding gives
  `tau_ijk = f_ijk - f_ij f_k - f_ik f_j - f_jk f_i + 2 f_i f_j f_k`, exactly the
  implemented form (and the form used in 010-kuzmin-tmi).
- **The delta-method SE is correct.** All seven partial derivatives in `se_tau` check out
  against the tau expression.
- **Normalization to `+ve Ctrl`** (the base production strain) is the right reference.

### Work items

1. Fix F2 at source; assert 10 TFs. Re-derive affected artifacts.
2. Fix F1: Welch-Satterthwaite effective df over all seven contributing strains, applied
   to digenic (3 strains) and trigenic (7 strains) alike, in BOTH the multiplicative and
   additive scripts. Keep the old df available for a documented before/after comparison.
3. Fix F3: emit pooled and within-readout FDR columns; name the family in every artifact.
4. Fix F4: de-hardcode the three unrun scripts (`EXPERIMENT_ROOT`, `DATA_ROOT`,
   package-relative mplstyle), then run all four models.
5. Re-run every 008 script end to end; regenerate all figures and CSVs.
6. Cross-model comparison via `model_upset_plots.py`: which interactions the four models
   agree on.
7. Rebuild the trajectory panels on corrected statistics and update
   [[experiments.008-xue-ffa.scripts.ffa_epistatic_path_panels]] with a correction section.

### Verification

- `read_abbreviations`-style assertion of 10 TFs passes in every loader.
- Recomputed tau still matches the trajectory script's independent implementation exactly.
- Effective df is > 1 for every testable interaction and is reported per row.
- All four model CSVs exist and carry the same gene naming.
- `ruff check` clean on every touched script.

## 2026.08.15 - Results of the Audit and Re-Run

All four models now run on GilaHyper and every 008 artifact has been regenerated.

### F1 confirmed and fixed -- the df choice was inverting the headline conclusion

`compute_se_pvalue` was referred to `df = max(1, min(n) - 1)`, giving **df = 2 for 77 of
119** total-titer triples and **df = 1 for the other 42** (any triple with a 2-replicate
constituent). Replaced with the Welch-Satterthwaite effective df of the delta-method
combination, over all 7 contributing strains for tau and all 3 for epsilon, weighted by
each term's actual contribution to the combined SE. Realized `df_effective` on total-titer
trigenic: **min 1.71, median 4.31, max 9.77**, and it is now emitted as a column.

Effect on the multiplicative model, total-titer trigenic (n=119):

| | raw P<0.05 | FDR within readout | positive AND FDR |
| --- | --- | --- | --- |
| before (df 1-2) | 56 | 21 | **0** |
| after (Welch-Satterthwaite) | 91 | 86 | **11** |

Across the whole multiplicative run, FDR-significant went from **0 of 990 to 456 of 990**.
The previously reported "no positive trigenic interaction survives FDR" was an **artifact
of the degrees of freedom**, not a property of the data. Two of the six highest-titer
triples -- OPI1-RFX1-YAP6 (tau=+0.500) and RFX1-SPT3-YAP6 (tau=+0.343) -- are now
FDR-significant positive.

The same fix was applied to the additive script (unit coefficients, so contributions are
the component SEs). The GLM log-link and log-OLS models were NOT touched: they fit with
statsmodels, which derives residual df from the design correctly.

### F2 fixed -- FKH1 restored across every loader

`header=None` plus an assertion of exactly 10 TFs in all six scripts that read the
Abbreviations sheet. The regenerated tables carry `FKH1`; the bare letter `F` is gone.
The assertion means a silent regression to 9 TFs is now impossible.

### F3 fixed -- both correction families are reported

`fdr_corrected_p` remains the pooled (all-6-readouts) correction; the panel script also
computes BH within the plotted readout and reports both. Total titer is the only readout
with anything surviving a within-readout correction.

### F4 fixed -- all four models run, and they mostly agree

De-hardcoded **30 scripts** (every `/Users/michaelvolk` literal in 008 is gone; all 37
parse, ruff F821 clean). Trigenic significance on total titer, BH within readout:

| model | n | FDR<0.05 | positive | negative |
| --- | --- | --- | --- | --- |
| multiplicative | 119 | 86 | 11 | 75 |
| additive | 119 | 84 | 6 | 78 |
| GLM log-link | 120 | 83 | 0 | 83 |
| log-OLS WT-differencing | 120 | 84 | 0 | 84 |

**73 trigenic interactions are FDR-significant in all four models, with zero sign
discordance** -- a strong consensus set. But that consensus is **entirely negative**: the
two log-scale models find no positive trigenic interaction at all. So the positive
interactions are a linear-scale phenomenon and are **not** cross-model robust. Any claim
about positive three-way epistasis on FFA titer must name the model.

### New defect found during the audit

Comparing models naively gives ZERO overlap between {multiplicative, additive} and
{GLM, log-OLS} because the two families write different `gene_set` separators (`RPD3_SPT3_YAP6`
vs `FKH1:GCN5:MED4`). This is a presentation inconsistency, not a data error, and the repo's
own `model_upset_plots.py` already normalizes for it -- but any new cross-model analysis must
too. Worth unifying at the writer.

### Verified correct, unchanged

- tau matches Kuzmin 2018 exactly (see the derivation in the plan above); the delta-method
  SE partials are all correct; normalization to the base production strain is right.
- The independent tau in `ffa_epistatic_path_panels.py` still agrees with the pipeline's
  `interaction_score` to 0.0 after the re-run.

## 2026.08.25 - Repo-Wide Sweep for the Same df Pattern: 008 Was the Only Site

The df defect suppressed nearly every interaction in 008, so the obvious question is
whether the same `min(n) - 1` reasoning appears anywhere else that propagates error across
several independently measured quantities. Swept every t and z significance test in
`experiments/` and `torchcell/`. **Result: measured and null. 008 was the only affected
site.** Each other test is correct for the quantity it actually tests.

| Site | What it tests | df used | Verdict |
| --- | --- | --- | --- |
| `010.../inference_dataset_2_setting_fitness_thresholds.py` | DMF vs SMF, a two-sample difference | `compute_welch_df(...)` | Correct, already Welch-Satterthwaite |
| `012.../kemmeren_volcano.py` | one released log2 ratio against zero | `n - 1` from the released `n_samples` | Correct. The script consumes a released estimate with its released SE, it does not form a combination |
| `019.../score_decoder_arms.py`, `019.../wave4b_convergence.py` | mean of n paired seed deltas | `len(deltas) - 1` | Correct by construction for a one-sample mean |
| `W019.../run4_doubles_48h.py` | mean of 3 independent plate-level estimates | `e.size - 1` = 2 | Correct, and the script already documents that three plates is low power and a null there is weak evidence of absence |

One related pattern, deliberately left alone:
`010.../inference_dataset_2_setting_fitness_thresholds_simplest_assumptions.py` uses a
normal approximation rather than a t distribution. On small n that is anti-conservative,
the opposite direction from the 008 bug, but the script's docstring declares it a
simplified analysis ("For simplicity: SD = SE") and it exists as the naive companion to
the rigorous Welch-based sibling above. Changing it would defeat its purpose.

### The one residual, now fixed

`compute_se_pvalue` still carried `df=2` as a default in both 008 scripts. Every call site
passes an explicit Welch df, so the default was unreachable, but it was a trap for the next
caller. `df` is now required with no default. Re-running both models reproduces the landed
numbers exactly (456 and 549 FDR-significant; result CSVs byte-identical), confirming the
change is behavior-preserving.

### The generalizable rule

When almost nothing clears significance in a replicate-poor, propagated-error setting,
check the reference distribution before believing the null. Two independent suppressors bit
this analysis: a df that treated ~21 measurements as 3, and an FDR family that corrected
over six readouts when the claim concerned one.
