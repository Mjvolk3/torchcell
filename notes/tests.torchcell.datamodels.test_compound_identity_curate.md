---
id: 70rjzbfk5xiz0sbh8xs21uy
title: Test_compound_identity_curate
desc: ''
updated: 1791270209627
created: 1791270209627
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): the PubChem curator, fully offline. `urllib.request.build_opener` is patched to return a fake opener that records every request (full URL, method, body, headers, timeout, clock time) and answers scripted JSON per URL; the module's `time` is a fake clock starting at 1000.0 s, and `date.today()` is pinned to 2026-10-06.

- Rate limit (`_MIN_INTERVAL_S = 0.25`): departures 1000.0, 1000.25, 1000.5 with sleeps of exactly 0.25; with a 0.125 s latency the wait is 0.125.
- Busy backoff: 5, 10, ... 35 s for attempts 0..6; any other fault raises with the fault dict in the message.
- Batching: 101 CIDs give two POSTs (100 then 1) with `cid=1%2C2...` form bodies; cached CIDs are not re-asked.
- `assemble`: every branch on one 13-directive fixture (merge by CID, canonical, mixture with and without a CID or a hand ChEBI, the three terminal statuses, the no-InChIKey CID), plus the precedence and tie rules of `_disambiguate` and both refusals.
- `curate` end to end on tmp input lists: exact request order, the cache JSON and every serialized record; the 50-lookup cache flush and its progress line.

Findings (each reproduced):

- `read_name_list` strips only full-line comments; an inline `# ...` stays in the label (compound_identity_curate.py:176). Latent: no committed input list has one.
- `_request` sleeps the 35 s backoff after the LAST busy attempt before raising (compound_identity_curate.py:278).
- `properties_by_cids` caches the whole batch payload under the single-CID URL of a CID PubChem did not return (compound_identity_curate.py:311-315); only the `prop.cid == cid` guard keeps the wrong row out.

Left uncovered: `main` (demo CLI, out of scope).

### Audit 2 notes applied

- Audit 2: D1 latent (no inline `#` in any committed compound_identity_inputs/*.txt). `test_read_cid_list` now carries an indented `# note` line; the end-to-end docstring now says the serialized table parses back to the asserted records.
