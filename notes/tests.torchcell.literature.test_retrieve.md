---
id: 04eso36aun3i6mwxct2106q
title: Test_retrieve
desc: ''
updated: 1791270232824
created: 1791270232824
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): the versioned retrievers over an `httpx.MockTransport`. Pinned: the registry, the User-Agent and client arguments (redirects followed, 120 s, 1800 s for zip containers), a 503 raising, a 302 followed, the zip container sha256 refusal message, `None` skipping the container check, zipfile's own missing-member error, and the PMC OA API: the `ftp://` to `https://` rewrite of the tgz link, the not-open-access refusal, and the no-tgz-link refusal for three record shapes.
