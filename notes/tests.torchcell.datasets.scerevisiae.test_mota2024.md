---
id: 5qyb8hzm6k35gvdo3ldnfg0
title: Test_mota2024
desc: ''
updated: 1790759620313
created: 1790759620313
---

## 2026.09.30 - Phase 12: a richer synthetic table, the merge ledger, downloads

Seven to twenty-two tests, 72 to 97 percent. Title and blank rows before the header; skipped rows (blank score, blank token, `0`, `+++`, a non-breaking-space token); `model_dump()` equality for the merged record (EFG1 and YGR272C at `++`), the tied merge whose gene has no common name, and the octanoic record; the full `DropLog` (RLM2 dropped in three acids, SBR2 once, 10 = 4 kept + 4 dropped + 2 merged); reference index [[0, 1], [2], [3]]; `download` mirror-first then the ESM URL (180 s timeout); `deposit_raw_mirror` with and without `checked_at`.

Findings: a score symbol other than `0`, `+`, `++` is skipped without being counted or ledgered (line 583); a missing header row raises a bare `StopIteration` (572).

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - Grade-symbol and header findings retired (issue #520)

Retired: `test_unknown_grade_symbol_is_skipped_without_a_ledger_entry` and `test_a_sheet_without_the_header_row_raises_a_bare_stop_iteration`. The `+++` row left the shared edge fixture; a dedicated test now asserts the exact refusal for it, and the missing header asserts the exact `RuntimeError` naming the file.
