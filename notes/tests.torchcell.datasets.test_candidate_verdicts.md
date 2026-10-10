---
id: ofo7lwyk2sp5amck6oc53t2
title: Test_candidate_verdicts
desc: ''
updated: 1791617817148
created: 1791617817148
---

## 2026.10.10 - Registry-wide enforcement

Imports every module under `torchcell/datasets` (so `dataset_registry` holds every class, not only what package `__init__` files import) and asserts: `GRANDFATHERED` is sorted, unique and fully registered; `enforcement_violations` over the real registry and the real store is empty; the walk finds at least the 125 grandfathered classes. A class registered after 2026-10-10 fails here until its module declares `CITATION_KEY` and `database/candidates/<key>.json` holds an admissible verdict. The tuple shrinks as backfill verdicts land; it never grows silently, because a new name in it would have to be added by hand in review.
