---
id: igrwtno0fc1cqhpefguxhqw
title: Report
desc: ''
updated: 1783563821953
created: 1783563821953
---

## 2026.07.08 - Report models: the shared vocabulary that makes "verified" auditable data

This module holds the pure pydantic data models the whole framework speaks in -- `Level` (the L0-L4 enum), `LevelResult` (one check's outcome), `VerificationReport` (provenance + all results for a dataset), plus `Provenance` and `sha256_file`. It exists so a verification outcome is a serializable, re-checkable artifact (a `verification_report.json` beside each dataset), not an ephemeral console pass/fail. `VerificationReport.passed` is deliberately strict: an EMPTY report is NOT passed, so "we never ran a check" can never masquerade as success.

- `StatDerivation` + `DerivationMethod` live here too: they record how a NON-sourced statistic was fixed (back_solve / conservative_low / median), encoding the CLAUDE.md "Adding Datasets" range-resolution policy as validated data with its diagnostics attached -- the counterpart to [[torchcell.verification.sourced]]'s quote-backed constants.
- The `StatDerivation` validator enforces that each method carries the inputs that justify it (range methods need bounds and value-in-range; back_solve needs the matched statistic) -- a derivation cannot be recorded without its evidence.
- `Level` is an ordered `IntEnum` because the tiers are a gate; the check functions that PRODUCE `LevelResult`s live in [[torchcell.verification.levels]], keeping data models and logic separate.
- Consumed by the runners ([[torchcell.verification.runners]]) which serialize reports to JSON siblings of `experiment_reference_index.json`.
