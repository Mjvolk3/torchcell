---
id: a7c61golra3eg2g2eabfnzs
title: Check_candidate_verdicts
desc: ''
updated: 1791617809290
created: 1791617809290
---

## 2026.10.10 - Candidate-verdict tripwire

`scripts/check_candidate_verdicts.py` is the diff-scoped half of plan decision 8, built on the shape of [[scripts.check_paired_tests]]: index versus the merge base with `origin/main`, files under `torchcell/datasets/` added or modified. For each, the classes decorated with `register_dataset` (bare or dotted) in the staged version minus those in the merge-base version are the new registrations. Each needs a module-level `CITATION_KEY` string and `database/candidates/<key>.json` in the same index (`git cat-file -e :<path>`); an unstaged verdict does not count. Exit 1 names each new class and the command that writes its verdict.

Wired as the `candidate-verdicts` local pre-commit hook (`files: ^torchcell/datasets/.*\.py$`) beside `paired-tests`. Whether the verdict passes is the enforcement test's job ([[tests.torchcell.datasets.test_candidate_verdicts]]). Stdlib only.
