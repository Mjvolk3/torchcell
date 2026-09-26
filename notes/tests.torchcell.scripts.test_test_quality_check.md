---
id: jypd9lwxowx9ilvn8tie6q8
title: Test_test_quality_check
desc: ''
updated: 1790416771426
created: 1790416771426
---

## 2026.09.26 - The anti-padding lint on hand-written test files

Each case is one small file under `tmp_path` with a fixed four-line header, so expected findings are `(rule, test, line)` triples. `no-assert` at the def line; `truthiness-only` only under `parametrize` (the same weak asserts pass unparametrized) and reported at the def, not the decorator; `unasserted-call` when the assertion is about arithmetic; and the taint flowing through assignment, `for` target, `with` target, call argument, subscript store, nested factory and the capture fixtures, so none of those is a finding. Asserting module helpers (transitively), `pytest.raises` around the call, `assert_*` calls and an `if`-guarded `raise AssertionError` all count. Fixtures named `test_*` are skipped; the allow marker works on a decorator line or the def line. `_weak_assert` is tabled over fifteen expressions (`len(x) == 0` is not weak; `len(x) >= 1` is). The CLI prints absolute paths for files outside the repo and repo-relative ones inside, then the summary, exit 1 on findings and 0 otherwise; `collect` walks directories, accepts a single `test_*.py` and ignores other names.

Two lint gaps surfaced here and were closed in the Phase 3 PR ([[scripts.test_quality_check]]): captured output was not derived from the torchcell call, and the condition of an `if` that raises `AssertionError` was not read by the taint check. Phase 4 of [[plan.test-suite-buildout.2026.09.25]].
