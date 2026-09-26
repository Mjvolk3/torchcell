---
id: bdmgkbys44q2kzav1tj2j2j
title: Test_quality_check
desc: ''
updated: 1790409456892
created: 1790409456892
---

## 2026.09.26 - Anti-padding lint for tests/

`scripts/test_quality_check.py` is the mechanical half of the no-cheating evaluation (Decisions 18 and 19 of [[plan.test-suite-buildout.2026.09.25]]); the reviewer's quality audit is the other half. Three `ast` rules on every `test_*` function: `no-assert` (no `assert`, no `pytest.raises`/`warns`/`fail`, no `assert*` call, no `raise AssertionError`, no call to a same-file helper that asserts), `truthiness-only` (a parametrized sweep whose assertions are all a bare name, `bool(x)`, `is not None`, `isinstance`, or `len(x) > 0`), and `unasserted-call` (a torchcell name is called and nothing it touched is asserted; "touched" propagates through assignments, `for` targets, subscript stores, `with ... as`, nested helper functions, and the arguments handed to the call, so `loss.backward(); assert x.grad is not None` counts).

`# test-quality: allow <reason>` on the `def` or a decorator line exempts a test; fixtures are never linted. Calibration on the 129 existing files (2026.09.26): a first version raised 43 findings, 39 of them false positives on legitimate idioms (helpers that assert, `for` over a torchcell iterable, gradient checks on an argument); after the propagation rules above, 4 remained and were annotated: two GPU timing benchmarks and one memory benchmark in `tests/torchcell/nn/`, and `test_mormino2022.py`'s audit whose contract is "does not raise". Wired to `make test-quality`, the `test-quality` pre-commit hook on `^tests/`, and a blocking CI step.

## 2026.09.26 - Capture fixtures seed the taint; a guarded raise counts

Two gaps found while writing Phase 3 and the lint's own tests ([[tests.torchcell.scripts.test_test_quality_check]]). First, `tests/torchcell/ontology/test_tc_ontology.py` calls `print_schema_mappings` and asserts on `capsys.readouterr().out`; nothing there is derived from the torchcell call by assignment, so the rule fired. The capture fixtures in a test's signature (`capsys`, `capfd`, `caplog`, `capsysbinary`, `capfdbinary`, `CAPTURE_FIXTURES`) now seed the derived set: output captured after a torchcell call is an assertion about that call. This is deliberately coarse; any assertion on `caplog` satisfies the rule. Second, `if model.value != 1: raise AssertionError(...)` was recognized as an assertion (no `no-assert`) but the taint check never read the `if` condition, so the test was reported as `unasserted-call`; the condition of an `if` whose body raises `AssertionError` now counts as touched. The gate stays at 147 files clean.
