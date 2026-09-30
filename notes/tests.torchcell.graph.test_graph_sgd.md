---
id: np9o05lsqh2ef29ta3lt1xk
title: Test_graph_sgd
desc: ''
updated: 1790562747636
created: 1790562747636
---

## 2026.09.27 - The SGD fetch and parse helpers

Thirteen tests with `aiohttp` replaced by a fake session, `asyncio.sleep` stubbed and `tempfile.tempdir` redirected into `tmp_path`; exact parsed outputs on hand-written SGD fragments. Module coverage 93%. Findings: `ContentTypeError(message)` is built with one argument and raises `TypeError`, so an HTML response escapes the retry loop on the first attempt (sgd.py line 161); `locus()` writes `locus_schema.json` into the current directory (lines 181 to 182). The download-chunk test carries a `# test-quality: allow` marker because the function returns None and the test asserts its side effects. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 16: shared fetches, retries, the chunked main

Thirteen to seventeen tests, 93 to 100 percent. Two concurrent fetches share one 11-GET download; `max_retries=0` makes no request; the chunks-of-50 split in `main_get_all_genes`. Two tests carry `# test-quality: allow` because their targets return None and the side effects are asserted.

Findings: a total fetch failure is written to disk as 11 nulls and later skipped as already cached (lines 100-113, 265); `SCerevisiaeGenome()` is built with its defaults, a relative root and `overwrite=True` (299).
