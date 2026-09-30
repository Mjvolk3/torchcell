---
id: fjqq9s2o2irfnc19yosypg5
title: Test_schema
desc: ''
updated: 1790773217913
created: 1790773217913
---

## 2026.09.30 - Phase 16: every untested validator with an acceptance and a refusal

Nine to thirty-eight tests (12 to 71 cases), 66 to 92 percent alone (99 across the suite). Each untested validator with one accepted value and one exact refusal message; the perturbation union picked by its type tag; JSON round trips for environment and genotype; the stored enum values. No schema class is subclassed.

Findings: `Genotype.__eq__` compares sets, so a genotype listing a perturbation twice equals one listing it once (line 1008); the -273 floor applies to Celsius only, -300 Kelvin validates (1035); the `graph_level` refusal message lists the levels in hash order (1617-1619); validating with `from_attributes` skips the SE derivation, leaving `fitness_se` or `environment_response_se` None (1785, 2990); the `SortedDict` coercion is undone and a plain dict stored (2160-2167); microarray log2-ratio keys are never checked against the expression keys (2240-2262).
