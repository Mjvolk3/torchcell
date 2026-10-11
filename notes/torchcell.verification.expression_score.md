---
id: 8qgjphdcg3ztilptec0uhja
title: Expression_score
desc: ''
updated: 1791736574412
created: 1791736574412
---

## 2026.10.11 - The error-model microarray score verifier (#1022)

Module: `torchcell/verification/expression_score.py`. Family: `MicroarrayExpressionScorePhenotype`, a per-gene signed significance score of a strain against a reference with its p-value and no log ratio. First dataset: [[torchcell.datasets.scerevisiae.hu2007]] (GEO GSE4654). Registry: `EXPRESSION_SCORE_DATASETS` in [[torchcell.verification.runners]], family runner `run_expression_score`, which dispatches to the loader's `verify_build` because the loader owns the re-read of its released bytes (the bioproduction registries work the same way). It is not in `EXPRESSION_DATASETS`, whose verifier reads `expression_log2_ratio`.

`verify_expression_score_dataset` (L0-L3):

| level | rule | what it asserts |
|---|---|---|
| L0 | structural | every experiment validates against `ExperimentType` |
| L1 | count | the caller's record-count oracle |
| L2 | score_finite | every stored score is finite |
| L2 | p_value_in_unit_interval | every stored p-value is in [0, 1] |
| L2 | p_value_keys_match_scores | every record's p-value map has exactly its score map's keys |
| L3 | reference_score_zero | every reference score is 0 and no reference carries a p-value |
| L3 | deletion_lowers_score | the median score of the deleted genes in their own records is < 0 (a sign inversion flips it) |
| L3 | p_value_is_two_sided_normal | optional: `p == 2 * (1 - Phi(abs(score)))` within the caller's tolerance (the release's rounding) |

`l4_matched_profile_agreement` (L4) compares each strain's profile with a second source's profile of the SAME single deletion, against that source's profiles of the other shared deletions. Genes are the union of both sources' keys less every shared strain's deleted gene, so no pair is driven by the trivially lowered deleted gene, and a pair is scored over the genes both profiles carry. It passes when the matched Pearson r exceed the mismatched ones by a one-sided Mann-Whitney U test at the caller's alpha and the matched median is the larger. A pair with fewer than three shared genes fails the check rather than being dropped. Two measurements of one strain are two records, so this is an agreement test, not a duplicate test; its threshold is a rank test because a TF deletion with no effect in either medium correlates near 0 with itself, and a fixed-r floor would read that as a disagreement.

Tests: `tests/torchcell/verification/test_expression_score.py` (11), on records dumped from real schema objects; the L4 cases are built so matched r is 1 and the mismatched r are those of six fixed random vectors, and a label shift makes the matched pairs the mismatched ones.
