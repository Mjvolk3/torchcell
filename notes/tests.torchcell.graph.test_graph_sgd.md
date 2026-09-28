---
id: np9o05lsqh2ef29ta3lt1xk
title: Test_graph_sgd
desc: ''
updated: 1790562747636
created: 1790562747636
---

## 2026.09.27 - The SGD fetch and parse helpers

Thirteen tests with `aiohttp` replaced by a fake session, `asyncio.sleep` stubbed and `tempfile.tempdir` redirected into `tmp_path`; exact parsed outputs on hand-written SGD fragments. Module coverage 93%. Findings: `ContentTypeError(message)` is built with one argument and raises `TypeError`, so an HTML response escapes the retry loop on the first attempt (sgd.py line 161); `locus()` writes `locus_schema.json` into the current directory (lines 181 to 182). The download-chunk test carries a `# test-quality: allow` marker because the function returns None and the test asserts its side effects. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
