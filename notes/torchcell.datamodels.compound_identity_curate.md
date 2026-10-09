---
id: bghn67cbwmzmarmqghcvz62
title: Compound_identity_curate
desc: ''
updated: 1791533375950
created: 1791533375950
---

The reproducible curator for the pinned compound-identity table: the one thing in the
compound-identity subsystem that touches the network. It reads the committed input lists
under `torchcell/datamodels/compound_identity_inputs/`, queries the PubChem PUG REST API,
and writes `compound_identity_table.json` with deterministic bytes, printing the sha256 to
re-pin into `compound_identity.py::_TABLE_SHA256`. The resolver then reads that table
offline forever. Subsystem overview: [[torchcell.datamodels.compound_identity]].

## 2026.10.09 - PubChem has two throttles and they answer differently (#676)

Found while running the bacterial pass #726 asks for. The client crashed mid-pass with
`json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` after about 300
name lookups at four requests per second.

**Measured on the wire.** PubChem throttles in two ways:

| throttle | status | body | header |
|---|---|---|---|
| dynamic ("ServerBusy") | HTTP 200 | JSON `{"Fault": {"Code": "PUGREST.ServerBusy", ...}}` | none |
| request-rate | **HTTP 429** | `<!doctype html>...<title>429</title>429 Too Many Requests` | no `Retry-After`, no `X-Throttling-Control` |

`_request` handled only the first, so the HTML body reached `json.loads` and raised. The
`_PassThroughErrorProcessor` is what let the 429 through the opener in the first place: it
exists so a 404's JSON `Fault` body can be read as evidence, and it passes every 4xx body
on, HTML included.

**What changed.** A 429 is now recognized as a throttle, with its OWN schedule and its OWN
counter, because it is a per-window block rather than a momentary busy signal: a flat
`_RATE_LIMIT_BACKOFF_S = 60.0` for up to `_MAX_RATE_LIMIT_RETRIES = 20` tries, measured
after the `ServerBusy` schedule's growing 5, 10, ... 35 s (105 s in total) proved too
short. The two counters are separate so a long block does not spend the busy budget, and
the abort message names both. A numeric `Retry-After` replaces the computed wait when one
is sent; an HTTP-date one does not, because only an all-digit value is a seconds count. A
non-JSON body on any OTHER status aborts naming the status and the first 200 bytes, which
is a server fault rather than a throttle and must not be retried.

**`--min-interval`** was added so a long pass can be run slower than the published five
requests per second without moving that documented ceiling (`_MIN_INTERVAL_S` stays 0.25).
It threads through `curate` into `PubChemClient`.

Six hermetic tests in [[tests.torchcell.datamodels.test_compound_identity_curate]] pin the
429 path: backoff then success, an integer `Retry-After` honored, an HTTP-date one ignored,
the separate budget and its abort message, and a non-JSON body on another status aborting
without a retry. The `_Response` test double grew a `status` and a `headers`, and `_Net`
passes a prepared `_Response` through, so a non-200 answer can be scripted.

### The run this was found by is still blocked

Measured the same day from GilaHyper: PubChem answered 429 to EVERY request, including
`water` and `glucose`, from any User-Agent, with no `Retry-After`, for over an hour. The
new abort message is the measurement: `0 busy and 21 rate-limited retries` for
`tolfenamic acid`, i.e. 21 consecutive minute-spaced attempts, all 429. The block is
IP-based (a browser-like User-Agent, no User-Agent and IPv6 behave identically),
so the fix makes the pass survivable but cannot make it run. Resume with the command in
the module docstring; `--cache` makes an interrupted pass resume without re-querying, so a
block costs time rather than progress.
