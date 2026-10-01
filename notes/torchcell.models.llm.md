---
id: ynxpm2hqxzz360mejf0t20i
title: Llm
desc: ''
updated: 1695170147134
created: 1695170092049
---

## 2026.10.01 - Abstract bodies raise

The abstract `_check_and_download_model`, `load_model` and `embed` of `NucleotideModel` and `PeptideModel` had `pass` bodies, so `super().embed(...)` returned None. Each now raises `NotImplementedError("<Base>.<method> is abstract; implement it in the subclass")`. The concrete subclasses implement all three and still construct. Issue #541; test `test_the_abstract_bodies_raise_when_a_subclass_calls_super`.
