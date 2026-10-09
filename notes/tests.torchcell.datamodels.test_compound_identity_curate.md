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

## 2026.10.09 - The HTTP 429 throttle path (#676, #726)

Six cases added for PubChem's SECOND throttle, found by the #726 pass and described in
[[torchcell.datamodels.compound_identity_curate]]:

- `test_http_429_with_an_html_body_backs_off_then_succeeds`: two 429s then a property
  table; the sleeps are a flat `[60.0, 60.0]`, not the `ServerBusy` schedule's `[5, 10]`.
- `test_http_429_honors_an_integer_retry_after`: `Retry-After: 7` replaces the minute.
- `test_http_429_with_a_date_retry_after_uses_the_flat_minute`: an HTTP-date value is not
  a seconds count, so the computed wait stands.
- `test_http_429_on_every_attempt_aborts_after_its_own_budget`: 21 tries, 21 flat minutes,
  and the abort message reads `0 busy and 21 rate-limited` (the budgets are separate).
- `test_a_non_json_body_on_another_status_aborts_with_the_status`: HTML on a 503 aborts
  naming the status and the body, with no sleep and no retry.
- `test_a_raised_min_interval_spaces_the_requests_further`: `PubChemClient(None, 1.5)`
  departs at 1000.0, 1001.5, 1003.0.

`test_server_busy_on_every_attempt_aborts` was updated for the new abort message
(`7 busy and 0 rate-limited`), which is what keeps the two throttles distinguishable in a
log. Two test doubles changed: `_Response` takes `raw`, `status` and `headers` (default
200 and `{}`, so every existing case reads unchanged), and `_Net.open` passes a prepared
`_Response` through instead of wrapping it, so a non-200 answer can be scripted. The two
`main` tests' `fake_curate` gained the `min_interval` positional argument the CLI now
passes.
