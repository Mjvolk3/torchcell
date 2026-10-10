---
id: hmkaftwmxu4po8fgbffh6po
title: Test_check_candidate_verdicts
desc: ''
updated: 1791617824973
created: 1791617824973
---

## 2026.10.10 - Tripwire tests

Tables the AST helpers (`registered_classes`, `citation_key`) and runs the real `git diff --cached <merge-base>` in a hermetic temporary repository: a new class in a keyed module and a new class in a keyless module are both named; a verdict that is written but unstaged does not count; staging it (and removing the keyless class) makes the hook pass. A class registered outside `torchcell/datasets/` is ignored.
