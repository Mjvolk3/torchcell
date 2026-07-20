---
id: t1plt4o67xd6vzi11vurvc5
title: Constructed_10_dmf_reference
desc: ''
updated: 1784566132316
created: 1784566132316
---

## 2026.07.20 - Published DMF Baseline So Constructed Doubles Can Be Judged

Exists to answer, at the bench, "is this double mutant's fitness what the literature
says it should be?" When the wet-lab builds the 8 set-cover doubles there is otherwise no
reference to compare a measured fitness against. This pulls published double-mutant
fitness ± SD (plus digenic interaction ε and its p-value) for **every** pair among the 10
properly-constructed genes, and flags which pairs are the 8 construction targets
(`is_optimized_double`), so the comparison is a lookup rather than a re-query.

Coverage: all 45 pairs = C(10,2); Costanzo2016 has DMF for **44/45**, Kuzmin2018 for 4,
Kuzmin2020 for 2 — the Kuzmin sparsity is structural (their SMF/DMF values are byproducts
of trigenic arrays, so only query/array strains appear). The 8 construction doubles sit
~0.97–1.12, i.e. near-neutral, which is what makes a measured departure interpretable;
every low-fitness double contains CBF1/YJR060W, whose single defect dominates its pairs.
Data is filtered from the already-queried panel-12 doubles table, so no LMDB re-query.

Forest figure + table: [[experiments.010-kuzmin-tmi.scripts.topk_triples_from_constructed_10]].

### Is the DMF "standard deviation" a real SD? (sourced — yes)

Asked because a released column named `std` can be a sample SD, a standard error, or a
screen-level spread, and comparing an assay SD against the wrong one is silently wrong.
Checked against the mirrored Costanzo SI
(`$DATA_ROOT/torchcell-library/costanzoGlobalGeneticInteraction2016/si/si1.md`,
sha256 `1828703b0ff739fdf1c0d9232fe4fd81a3ce95a1b111780f55ef63bfa676880e`):

- **Double mutant fitness standard deviation = a REAL sample SD over 4 colony replicates.**
  SI line 80: *"All screens were conducted a single time with 4 replicate colonies per
  double mutant, unless otherwise indicated"*. Data File S1's column list (SI line 563)
  confirms `Double mutant fitness | Double mutant fitness standard deviation` is the column
  the loader reads.
- **Single mutant fitness stddev is NOT an SD — it is a bootstrap SE** over control
  *screens* (17 query / 350 array). SI line 94: *"…with the exception that bootstrapped
  means, instead of medians, across replicates were used in variance estimation and final
  fitness values."*

Consequence for assay development: the **doubles** comparison is apples-to-apples — both
the reference and our droplet assay report a colony-level sample SD — whereas the singles
comparison is not (colony SD vs bootstrap SE), which is why our singles SD looked inflated
against "Costanzo 0.081". Note the reference DMF SD is the *noisier* estimate here (n=4 vs
our ~11 colonies), and a 4-colony SD is itself unstable, so read per-double differences
loosely.

Columns added so this is unambiguous at the point of use (per source):
`*_uncertainty_type` (`sample_sd`), `*_n_samples` (4), `*_sample_unit` (`colony`), and the
derived `*_se` = SD/√n. Values come from the loaders' sourced constants
(`N_SAMPLES_DOUBLE_MUTANT`, Kuzmin `N_SAMPLES_COMBINED_MUTANT`), not re-hardcoded here.
Costanzo DMF SE therefore runs ~0.009–0.035 for the 8 construction doubles.
