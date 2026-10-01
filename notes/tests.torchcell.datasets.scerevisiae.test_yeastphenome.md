---
id: yt6mf3zc7km4nzbt5o98e18
title: Test_yeastphenome
desc: ''
updated: 1790769114789
created: 1790769114789
---

## 2026.09.30 - Phase 15: the shipped screen list, downloads, no file-level skip

Eight to fourteen tests, 25 to 83 percent alone (100 with the siblings); the data gate now sits on the seven mirror tests. The shipped screen list holds 47 screens, 47 PMIDs, 64-hex pins, none of the 16 excluded primaries, and 47 distinct raw names although two PMIDs share the stem `jin_liu_2021`; `download` against a fake `urlopen` (exact URL, user agent, 120 s timeout; a present file skipped; a sha256 mismatch refusing and writing nothing); `transform_item`; the inert hooks; `main`. Note for a later fix: `test_yeastphenome_synthetic.py` says "44 pinned real screens" while the list holds 47.

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
