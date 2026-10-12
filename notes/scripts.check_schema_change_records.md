---
id: 12mepuykwz6tpgz6l7g3p20
title: Check_schema_change_records
desc: ''
updated: 1791803419760
created: 1791803419760
---

## 2026.10.12 - Schema-change tripwire

`scripts/check_schema_change_records.py` is the pre-commit entry of `schema-change tripwire` ([[torchcell.provenance.schema_change]]), on the shape of [[scripts.check_candidate_verdicts]]: the staged schema surface against the merge base with `origin/main`. Every symbol whose contract fingerprint moved needs exactly one staged record under `database/schema_changes/` with the same before and after fingerprints, status and reasons; a new symbol no served closure holds is printed for the PR body instead. The script puts its own repository root first on `sys.path`, because a script's `sys.path[0]` is `scripts/` and `import torchcell` would otherwise load the editable install (the primary checkout) and check a worktree with main's code.

Wired as the `schema-change-records` hook, firing on the schema surface, `torchcell/datasets/`, `database/schema_changes/` and `database/releases/`. `TORCHCELL_SCHEMA_RECORD_ACK=1` passes a commit; the enforcement test still blocks the landing. Policy: [[torchcell.datamodels.schema-change-policy]].
