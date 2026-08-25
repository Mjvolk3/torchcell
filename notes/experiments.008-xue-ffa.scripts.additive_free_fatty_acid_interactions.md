---
id: a0bkdh6s33x0masszadt5wq
title: Additive_free_fatty_acid_interactions
desc: ''
updated: 1787688711691
created: 1787688711691
---

## 2026.08.25 - Additive Companion to the Multiplicative Epistasis Model

This script fits the additive null (`f_ij = f_i + f_j - 1`) against the same Xue 2025 FFA
panel that [[experiments.008-xue-ffa.scripts.free_fatty_acid_interactions]] fits
multiplicatively. It exists so an interaction claim can be checked for model dependence
rather than being read off one null, which matters here: the two linear-scale models find
positive trigenic interactions on total titer while the two log-scale models find none.

Two corrections landed in the 2026.08.15 audit and are recorded here because they changed
what this script reports:

- **Welch-Satterthwaite degrees of freedom.** The p-values had been referred to
  `max(1, min(n) - 1)`, the smallest single input's df, which is 1 or 2 for these strains.
  An interaction score is a linear combination of several independently measured strains,
  so the reference distribution needs the effective df of that combination. In the additive
  model every coefficient is plus or minus one, so each term's contribution to the combined
  SE is simply its own SE.
- **FKH1 was consumed as the Abbreviations sheet header**, mislabeling all 36
  FKH1-containing genotypes as the bare letter `F`. Fixed with `header=None` plus an
  assertion of exactly 10 TFs.

Current run: 990 interactions, 634 at raw P below 0.05, 549 after FDR.

On 2026.08.25 the `df=2` default was removed from `compute_se_pvalue` here as well, for the
reason given in the multiplicative note.

Full audit: [[plan.008-xue-ffa-epistasis-audit.2026.08.15]].
