---
id: h4qa16vpcdwgrqoxdholp4c
title: __init__
desc: ''
updated: 1791617746382
created: 1791617746382
---

## 2026.10.10 - Package for the candidate gate

`torchcell/candidates/` is piece 1 of [[plan.dataset-admission-pipeline.2026.10.10]]: a typed verdict on a candidate dataset row, decided before any loader work. It is the FIRST stage of the dataset-admission program; `kg_manifest admit` (`torchcell/knowledge_graphs/kg_manifest.py`) stays the last stage and is unchanged.

| module | role |
|---|---|
| [[torchcell.candidates.verdict]] | frozen pydantic records: `CandidateVerdict`, `GateResult`, `AggregationRecord`, the G2 to G5 sub-records |
| [[torchcell.candidates.gates]] | deterministic evaluators for G1 to G5 and `compose_verdict` (the stop rule) |
| [[torchcell.candidates.findings]] | recorded G1 judgments for five settled rows, built from the settled modules' sourced values |
| [[torchcell.candidates.store]] | `database/candidates/<citation_key>.json`, `GRANDFATHERED`, `enforcement_violations` |
| [[torchcell.candidates.ledger]] | the generated dendron ledger (this module's note IS the ledger) |
| [[torchcell.candidates.cli]] | `candidate-gate` / `python -m torchcell.candidates` |

The package exports `AggregationRecord`, `CandidateVerdict`, `GateResult` and `verdict_json_schema`.
