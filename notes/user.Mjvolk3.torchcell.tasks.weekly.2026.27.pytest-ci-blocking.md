---
id: diwisj0qj447txen528jlez
title: pytest-ci-blocking
desc: ''
updated: 1782930421422
created: 1782930421422
---

## 2026.07.01

- [ ] Make the pytest CI job blocking (`#16`): harden 4 CI-fragile tests (test_s288c module-level DATA_ROOT guard, wall-clock benchmark `@pytest.mark.gpu`, targeted DATA_ROOT skipif, filelock cleanup rewrite) + CPU-only wheel install (torch/scatter from CPU indexes) + remove `continue-on-error`, then add `pytest-coverage` to main's required checks post-merge [[plan.pytest-ci-blocking.2026.07.01]]
