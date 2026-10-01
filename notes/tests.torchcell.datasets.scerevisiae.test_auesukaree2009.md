---
id: 4qqfl7a82u03qv9bpyjw0bz
title: Test_auesukaree2009
desc: ''
updated: 1790759612784
created: 1790759612784
---

## 2026.09.30 - Phase 12: the parser on a synthetic pdftotext layer, a full build

Eight to twenty-eight tests, 63 to 97 percent. `_parse_tables` runs on a synthetic `pdftotext -layout` text layer with continuation lines, a page-number line ending a table, a footnote and an ignored `Table 7` caption; a full build of eight records in order with `model_dump()` equality for the ethanol record (VMA2 renamed to YBR127C), the heat record (37 C, no perturbation, 30 C reference), the adjudicated FEN1 record and the H2O2 record; the drop log as a hand-built `DropLog`; refusals with exact messages (per-stress count miss, missing stress table, no genome, missing mirror, sha256 mismatch naming both digests in `download` and `deposit_raw_mirror`); `deposit_raw_mirror` idempotent.

Findings: the class counts the docstring calls "per-class self-checksums" only gate continuation lines, so a class declaring (5) with 2 genes passes (lines 586-601); a second token for an ORF already seen in the same table is collapsed without a ledger entry, so the log reads 11 tokens against 8 kept + 2 dropped (654).

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - Class-count and collapse findings retired (issue #520)

Retired: `test_class_count_gates_continuations_but_is_never_checked` and `test_drop_log_counts_and_the_unledgered_duplicate`. Now asserted: a class short of its declared count refuses at table end and a class over it refuses at the next class row, both with exact messages; the drop log reads 11 listed = 8 kept + 2 dropped + 1 collapsed, with the ethanol YBR127C token ledgered as a `CollapsedToken` keeping VMA2.

Review follow-up (same day): the refusal tests now also assert that no `processed/lmdb` exists after the refusal, and that a second construction refuses with the same exact message.
