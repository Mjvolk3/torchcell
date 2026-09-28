---
id: rmt8lv8803ph5cdqp8tvohu
title: Test_kemmeren2014_synthetic
desc: ''
updated: 1790562890516
created: 1790562890516
---

## 2026.09.27 - The Kemmeren 2014 loader on in-memory GEOparse objects

Eight tests on real `GEOparse.GEOTypes` GSE, GSM and GPL objects built in memory and pickled to the paths the loader checks, so the network fallback never runs; `process_workers=1` goes through `_process_parallel` (its forked body is invisible to coverage). Loader coverage 70%. Findings: the `ValueError` for a missing "orf name" column is swallowed by the `except Exception` at line 1700, so a malformed Table S1 silently yields two empty maps (line 1591); the channel rule checks `"-a"` before `-b`, so a dye-swapped array for a gene with `-A` in its name reads the refpool channel as the deletion (line 902). Noticed, not pinned: a hardcoded 2633 divided by the deletion count raises `ZeroDivisionError` with none (line 459). Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
