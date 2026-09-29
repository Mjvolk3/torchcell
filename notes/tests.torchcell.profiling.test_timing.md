---
id: 53bgyql50a0ieitlwii6yv1
title: Test_timing
desc: ''
updated: 1790648935142
created: 1790648935142
---

## 2026.09.28 - The timing profiler under a fake clock (Phase 9)

13 tests, no wall-clock assertions: a scripted clock feeds `perf_counter`, so the wrapper's recorded durations are exact (0.25, 0.5, 0.125), and the whole stdout of `print_timing_summary` and `print_comparison_table` is compared as literal text (the `{:<50} {:>8} {:>12.4f} {:>11.4f}` rows, population SD 1.0 for [1, 3] ms, totals 8.50 and 6.00, ratio `1.42x`, headers truncated to twelve characters, all four indicator branches and both N/A branches). Findings: classes in the comparison table are matched by substring with the baseline first (lines 159 and 167), so with the docstring's own example pair `SubgraphRepresentation` and `LazySubgraphRepresentation` every Lazy timing is filed under the baseline and overwrites it, and the optimized column is always N/A; `get_timing_summary` returns `{}` whenever profiling is disabled even when timings exist (line 120), while `get_timings` and `print_timing_summary` still report them. Coverage 13.6% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]].
