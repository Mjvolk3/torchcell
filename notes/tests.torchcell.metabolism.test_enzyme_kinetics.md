---
id: y926klgq6xdu6blme14lsd9
title: Test_enzyme_kinetics
desc: ''
updated: 1790762213656
created: 1790762213656
---

## 2026.09.30 - Phase 13: the selection cascade and the OED fetch on a fake subprocess

Nine to nineteen tests (two existing real-mirror tests now carry `@pytest.mark.data`), 76 to 100 percent. The module holds no rate laws; it selects `k_cat`/`K_M` from database rows and fetches them from the OED API. Pinned: the exact resolved record after the cascade (the candidate count includes rows the wildtype filter drops), the rule string without the wildtype step and with a 37.5 C target as `37.5C`, "WildType" matched regardless of case, a zero `k_cat` counted as a value, unknown columns kept; the fetch on a fake `subprocess.run` (paging stops at the reported total, exact curl lines and the reproduction command, incomplete paging raises on an empty page and at the page limit, wrong-shaped responses raise `TypeError`); the mirror bytes and sha256, a `tmp_path` round trip, the tamper message.

Finding: with an even number of tied rows the input row order decides the result ([1.0, 3.0] gives 1.0, reversed gives 3.0), against the docstring's claim that the median tie-break removes row-order dependence (lines 308-311).

## 2026.09.30 - Even-tie Finding retired (issue #525)

Now asserted: [1.0, 3.0] and its reverse both resolve to 1.0 (lower median); [3, 1, 4, 2] in all four rotations resolves to 2.0; two rows sharing the median value resolve to the same row (pmid 111, pH 6.5) in either order. The full-cascade record is unchanged (5.0 at 25 C), now explained as the lower median.
