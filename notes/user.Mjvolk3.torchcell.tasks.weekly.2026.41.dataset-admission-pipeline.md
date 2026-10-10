---
id: jeiuomu3pa57rk1az0pb0ig
title: dataset-admission-pipeline
desc: ''
updated: 1791615067595
created: 1791615067595
---

## 2026.10.10

- [ ] Plan: typed candidate gates (G1 primary measurement, G2 representation of material, G3 deposit inventory, G4 duplication, G5 schema fit) as a CandidateVerdict per citation key, a candidate-gate CLI, an /add-dataset skill on the Agent tool, a re-audit over the fifty plus rows 51-61 and the yeast list, and a set-based registration gate [[plan.dataset-admission-pipeline.2026.10.10]]
- [x] Piece 1: `torchcell/candidates/` (verdict, gates, findings, store, ledger, cli), `candidate-gate`, `database/candidates/`, the enforcement test and the `candidate-verdicts` tripwire, verdict columns in both candidate tables [[torchcell.candidates.__init__]] [[torchcell.candidates.gates]] [[torchcell.candidates.cli]] [[scripts.check_candidate_verdicts]]
- [x] Loop note: Agent-tool orchestration replaces pydantic-ai [[torchcell.knowledge_graphs.dataset-admission-loop]]
