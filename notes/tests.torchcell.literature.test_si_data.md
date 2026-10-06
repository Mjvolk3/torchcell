---
id: r4zkxqjfbpsk3l7asgfd477
title: Test_si_data
desc: ''
updated: 1791270225103
created: 1791270225103
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): Dryad and direct-URL SI fetching over an `httpx.MockTransport` (the `httpx.Client` constructor is replaced by a factory that records its keyword arguments). Pinned: the Dryad walk (dataset `doi%3A10.5061%2Fdryad.tt367` -> version -> files) and absolute download URLs, client timeouts (60 s listing, 600 s download), streamed bytes over more than one 1 MiB chunk, no file on an HTTP error, keep-existing versus refetch-empty versus overwrite, and the exact log lines.

Finding: the local filename is the last `/` segment of the whole URL, so `.../data.csv?download=1` lands as `data.csv?download=1` and a URL ending in `/` lands as `file` (si_data.py:80).
