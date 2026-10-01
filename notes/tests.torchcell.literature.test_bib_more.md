---
id: sx1eyzrsqnf8603edximbj1
title: Test_bib_more
desc: ''
updated: 1790562785842
created: 1790562785842
---

## 2026.09.27 - Bibliography branches the first file missed

Fourteen tests; module coverage 99% with the existing bib tests. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.10.01 - `as_keys` on the paired pull (issue #529)

Added `test_fetch_paired_collection_entries_as_keys_sends_values_as_keys`: with `as_keys=True` both values go as `collection_key` (`RNASEQ01`, `4VNJWJAW`); the default still sends `GROUP001` as a key and `topic-a` as a `collection` name. 11 to 12 tests.
