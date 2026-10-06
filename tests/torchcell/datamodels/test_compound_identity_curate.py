# tests/torchcell/datamodels/test_compound_identity_curate.py
# [[tests.torchcell.datamodels.test_compound_identity_curate]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_compound_identity_curate.py
"""The PubChem curator for the pinned compound-identity table, fully offline.

Fixtures: ``_Net`` replaces BOTH the opener (``urllib.request.build_opener`` is patched
at its import site, so ``PubChemClient`` builds the fake) and the module's ``time``
(``monotonic``/``sleep`` read a fake clock that starts at 1000.0 s and moves only when
``sleep`` is called or an answer declares a latency). Every request is recorded with its
full URL, method, body, headers, timeout and the clock time it left at; answers are
scripted per URL (a list per URL is consumed in order). ``date`` is patched so
``retrieved_at`` is ``2026-10-06``. No socket is ever opened.

Derivations:

* Rate limit: ``_MIN_INTERVAL_S = 0.25``. With zero latency the first call leaves at
  1000.0 (``1000.0 - 0.0 >= 0.25``, no sleep) and every later call sleeps exactly 0.25,
  so departures are 1000.0, 1000.25, 1000.5, ... With a 0.125 s latency per call the
  remaining wait is ``0.25 - 0.125 = 0.125`` (all binary-exact).
* Busy backoff: ``_BUSY_BACKOFF_S * (attempt + 1)`` = 5, 10, ... 35 for attempts 0..6.
* ``urlencode({"cid": "1,2"})`` is ``cid=1%2C2``; ``quote("sodium chloride", safe="")``
  is ``sodium%20chloride`` and ``quote("a/b", safe="")`` is ``a%2Fb``.
* ``assemble`` row names, synonyms, precedence (2 canonical, 1 any name lookup in the
  group, 0 CID only) and statuses are worked per test from the source's branches.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import torchcell.datamodels.compound_identity_curate as cur
from torchcell.datamodels.compound_identity_curate import (
    CuratedRow,
    CurationDirective,
    PubChemClient,
    PubChemProperty,
    _disambiguate,
    _first_property,
    _PassThroughErrorProcessor,
    _single_chebi,
    assemble,
    cid_property_url,
    curate,
    name_property_url,
    parse_line,
    read_cid_list,
    read_name_list,
    serialize,
    synonyms_url,
)

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
PROPS = "Title,InChIKey,ConnectivitySMILES"
CID_BATCH_URL = f"{BASE}/cid/property/{PROPS}/JSON"
SYN_BATCH_URL = f"{BASE}/cid/synonyms/JSON"
NOT_FOUND = {"Fault": {"Code": "PUGREST.NotFound", "Message": "No CID found"}}
BUSY = {"Fault": {"Code": "PUGREST.ServerBusy", "Message": "Too many requests"}}
TODAY = "2026-10-06"


class _Response:
    def __init__(self, payload: Any) -> None:
        self._bytes = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._bytes


class _Net:
    """Fake opener + fake clock; records every request."""

    def __init__(self, answers: dict[str, list[Any]], latency: float = 0.0) -> None:
        self.answers = {url: list(seq) for url, seq in answers.items()}
        self.latency = latency
        self.now = 1000.0
        self.sleeps: list[float] = []
        self.requests: list[dict[str, Any]] = []
        self.on_request: list[Any] = []
        self.built: list[tuple[Any, ...]] = []

    # time module stand-in
    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    # opener stand-in
    def open(self, request: urllib.request.Request, timeout: float) -> _Response:
        self.requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "body": request.data,
                "headers": dict(request.header_items()),
                "timeout": timeout,
                "at": self.now,
            }
        )
        for hook in self.on_request:
            hook()
        self.now += self.latency
        return _Response(self.answers[request.full_url].pop(0))


@pytest.fixture
def net_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    def make(answers: dict[str, list[Any]], latency: float = 0.0) -> _Net:
        net = _Net(answers, latency)

        def build_opener(*handlers: Any) -> _Net:
            net.built.append(handlers)
            return net

        monkeypatch.setattr(urllib.request, "build_opener", build_opener)
        monkeypatch.setattr(
            cur, "time", SimpleNamespace(monotonic=net.monotonic, sleep=net.sleep)
        )
        monkeypatch.setattr(
            cur,
            "date",
            SimpleNamespace(today=lambda: SimpleNamespace(isoformat=lambda: TODAY)),
        )
        return net

    return make


def _prop(
    cid: int, title: str | None, key: str | None, smiles: str | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {"CID": cid}
    if title is not None:
        row["Title"] = title
    if key is not None:
        row["InChIKey"] = key
    if smiles is not None:
        row["ConnectivitySMILES"] = smiles
    return row


def _table(*rows: dict[str, Any]) -> dict[str, Any]:
    return {"PropertyTable": {"Properties": list(rows)}}


# --------------------------------------------------------------------------- #
# Input grammar
# --------------------------------------------------------------------------- #
def test_parse_line_every_directive() -> None:
    """Each directive maps to its field; ``canonical`` is a bare flag."""
    d = parse_line(
        "actinomycin D | query=dactinomycin | cid=2019 | chebi=CHEBI:27666 | canonical",
        "core",
    )
    assert d.model_dump() == {
        "label": "actinomycin D",
        "source": "core",
        "query": "dactinomycin",
        "cid": 2019,
        "canonical": True,
        "chebi_id": "CHEBI:27666",
        "mixture_reason": None,
        "unresolved_reason": None,
        "proprietary_reason": None,
        "undefined_reason": None,
    }
    assert d.name_query == "dactinomycin"
    assert d.skip_lookup is None


def test_free_text_directive_swallows_the_rest_of_the_line() -> None:
    """A prose reason re-joins the remaining parts with `` | `` and ends parsing, so a
    ``key=value`` after it is reason text, not a directive.
    """
    d = parse_line("SOM row | unresolved=table says a | b=c |  d", "smith2016")
    assert d.unresolved_reason == "table says a | b=c | d"
    assert d.cid is None
    assert d.skip_lookup == "table says a | b=c | d"
    assert d.name_query == "SOM row"


@pytest.mark.parametrize(
    ("line", "field", "reason"),
    [
        ("x | mixture=ten homologues", "mixture_reason", "ten homologues"),
        ("x | proprietary=vendor code", "proprietary_reason", "vendor code"),
        ("x | undefined=yeast extract", "undefined_reason", "yeast extract"),
    ],
)
def test_each_free_text_directive_field(line: str, field: str, reason: str) -> None:
    """``mixture`` is not a skip reason; ``proprietary`` and ``undefined`` are."""
    d = parse_line(line, "s")
    assert getattr(d, field) == reason
    assert d.skip_lookup == (None if field == "mixture_reason" else reason)


def test_skip_lookup_precedence_is_unresolved_then_proprietary_then_undefined() -> None:
    d = CurationDirective(
        label="x", source="s", proprietary_reason="p", undefined_reason="u"
    )
    assert d.skip_lookup == "p"
    d2 = CurationDirective(label="x", source="s", undefined_reason="u")
    assert d2.skip_lookup == "u"


def test_parse_line_refuses_an_unknown_directive() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "hil: unknown directive 'smiles=CCO' in 'ethanol | smiles=CCO'"
        ),
    ):
        parse_line("ethanol | smiles=CCO", "hil")


def test_parse_line_refuses_a_non_integer_cid() -> None:
    with pytest.raises(
        ValueError, match=re.escape("invalid literal for int() with base 10: 'abc'")
    ):
        parse_line("x | cid=abc", "s")


def test_read_name_list_skips_comment_and_blank_lines(tmp_path: Path) -> None:
    """``source`` is the file stem; full-line comments (even indented) and blank lines
    are skipped; surrounding whitespace is stripped.
    """
    path = tmp_path / "hillenmeyer2008.txt"
    path.write_text(
        "# review: Hillenmeyer 2008 table S1\n\n   # indented comment\n  NaCl  \n"
        "glucose | query=D-glucose\n",
        encoding="utf-8",
    )
    out = read_name_list(path)
    assert [(d.label, d.source, d.query) for d in out] == [
        ("NaCl", "hillenmeyer2008", None),
        ("glucose", "hillenmeyer2008", "D-glucose"),
    ]


def test_read_name_list_keeps_an_inline_hash_in_the_label(tmp_path: Path) -> None:
    """Finding: the module docstring says ``#`` starts a comment, but ``read_name_list``
    strips only lines whose first non-blank character is ``#``; an inline comment
    stays in the label (and so in the PubChem query). No committed
    compound_identity_inputs/*.txt carries an inline ``#`` today, so the gap is latent. Pinned until inline comments are stripped
    or the grammar documents full-line comments only (compound_identity_curate.py:176).
    """
    path = tmp_path / "src.txt"
    path.write_text("# review X\nNaCl  # table salt\n", encoding="utf-8")
    assert [d.label for d in read_name_list(path)] == ["NaCl  # table salt"]


def test_read_cid_list(tmp_path: Path) -> None:
    path = tmp_path / "wildenhain2015_cids.txt"
    path.write_text(
        "# review: Wildenhain 2015\n3385\n   # note: indented comment\n\n  5793  \n",
        encoding="utf-8",
    )
    out = read_cid_list(path)  # the indented comment is skipped, not int()-parsed
    assert [(d.label, d.source, d.cid) for d in out] == [
        ("CID 3385", "wildenhain2015_cids", 3385),
        ("CID 5793", "wildenhain2015_cids", 5793),
    ]


def test_read_cid_list_refuses_a_non_integer_line(tmp_path: Path) -> None:
    path = tmp_path / "c.txt"
    path.write_text("12x\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match=re.escape("invalid literal for int() with base 10: '12x'")
    ):
        read_cid_list(path)


# --------------------------------------------------------------------------- #
# URLs and payload readers
# --------------------------------------------------------------------------- #
def test_urls_are_exact_and_quote_every_reserved_character() -> None:
    assert name_property_url("sodium chloride") == (
        f"{BASE}/name/sodium%20chloride/property/{PROPS}/JSON"
    )
    assert name_property_url("a/b") == f"{BASE}/name/a%2Fb/property/{PROPS}/JSON"
    assert cid_property_url(3385) == f"{BASE}/cid/3385/property/{PROPS}/JSON"
    assert synonyms_url(3385) == f"{BASE}/cid/3385/synonyms/JSON"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ([], None),
        ("x", None),
        ({}, None),
        ({"PropertyTable": {}}, None),
        (_table(), None),
        ({"PropertyTable": {"Properties": [None]}}, None),
        (NOT_FOUND, None),
        (_table(_prop(7, "Seven", "K7")), (7, "Seven", "K7", None)),
    ],
)
def test_first_property_cases(payload: Any, expected: tuple[Any, ...] | None) -> None:
    """A non-dict, a missing/empty table, a ``None`` first row and a Fault are no row;
    a table yields its first row's four fields.
    """
    got = _first_property(payload)
    assert (
        None if got is None else (got.cid, got.title, got.inchikey, got.smiles)
    ) == expected


def test_first_property_reads_the_first_row_only() -> None:
    got = _first_property(_table(_prop(5, "A", "K", "C"), _prop(6, "B", "L")))
    assert got is not None
    assert (got.cid, got.title, got.inchikey, got.smiles) == (5, "A", "K", "C")


@pytest.mark.parametrize(
    ("synonyms", "expected"),
    [
        (["acetic acid", "CHEBI:15366", "x CHEBI:47622"], None),
        (["ethanol", "CHEBI:16236", "see CHEBI:16236 again"], "CHEBI:16236"),
        (["no curie here", "CHEBI:12ab"], None),
        ([], None),
        ({"CHEBI:1": 1}, None),
        (None, None),
    ],
)
def test_single_chebi(synonyms: Any, expected: str | None) -> None:
    """Exactly one DISTINCT ChEBI id is adopted; two distinct ids or a non-list is None.
    ``CHEBI:12ab`` has no word boundary after a digit run, so it does not match.
    """
    assert _single_chebi(synonyms) == expected


def test_pass_through_processor_returns_error_responses_unchanged() -> None:
    """A 404 reaches the caller: the base processor would hand it to ``parent.error``."""
    proc = _PassThroughErrorProcessor()
    response = SimpleNamespace(code=404, msg="Not Found", info=lambda: {})
    assert proc.http_response(object(), response) is response
    assert proc.https_response(object(), response) is response


def test_client_builds_its_opener_with_the_pass_through_processor() -> None:
    """The real ``build_opener`` drops its default ``HTTPErrorProcessor`` in favor of the
    subclass, so the pass-through is the ONLY error processor.
    """
    client = PubChemClient(None)
    procs = [
        type(h)
        for h in vars(client._opener)["handlers"]
        if isinstance(h, urllib.request.HTTPErrorProcessor)
    ]
    assert procs == [_PassThroughErrorProcessor]


# --------------------------------------------------------------------------- #
# PubChemClient against the fake opener + clock
# --------------------------------------------------------------------------- #
def test_requests_are_spaced_by_the_rate_limit(net_factory: Any) -> None:
    """Zero latency: departures 1000.0, 1000.25, 1000.5; each later call sleeps 0.25."""
    names = ["a", "b", "c"]
    net = net_factory({name_property_url(n): [NOT_FOUND] for n in names})
    client = PubChemClient(None)
    assert [client.property_by_name(n) for n in names] == [None, None, None]
    assert [r["at"] for r in net.requests] == [1000.0, 1000.25, 1000.5]
    assert net.sleeps == [0.25, 0.25]
    gaps = [b["at"] - a["at"] for a, b in zip(net.requests, net.requests[1:])]
    assert min(gaps) >= 0.25


def test_latency_counts_toward_the_interval(net_factory: Any) -> None:
    """A 0.125 s answer leaves 0.125 s to wait; departures stay 0.25 apart."""
    net = net_factory(
        {name_property_url(n): [NOT_FOUND] for n in ["a", "b"]}, latency=0.125
    )
    client = PubChemClient(None)
    client.property_by_name("a")
    client.property_by_name("b")
    assert net.sleeps == [0.125]
    assert [r["at"] for r in net.requests] == [1000.0, 1000.25]


def test_get_request_shape(net_factory: Any) -> None:
    url = name_property_url("sodium chloride")
    net = net_factory(
        {
            url: [
                _table(
                    _prop(
                        5234,
                        "Sodium Chloride",
                        "FAPWRFPIFSIZLT-UHFFFAOYSA-M",
                        "[Na+].[Cl-]",
                    )
                )
            ]
        }
    )
    client = PubChemClient(None)
    got = client.property_by_name("sodium chloride")
    assert got is not None
    assert (got.cid, got.title, got.inchikey, got.smiles) == (
        5234,
        "Sodium Chloride",
        "FAPWRFPIFSIZLT-UHFFFAOYSA-M",
        "[Na+].[Cl-]",
    )
    assert net.requests == [
        {
            "url": url,
            "method": "GET",
            "body": None,
            "headers": {"User-agent": "torchcell-compound-curator"},
            "timeout": 60,
            "at": 1000.0,
        }
    ]
    assert net.built == [(_PassThroughErrorProcessor,)]


def test_name_lookups_are_cached_in_memory(net_factory: Any) -> None:
    net = net_factory({name_property_url("a"): [NOT_FOUND]})
    client = PubChemClient(None)
    client.property_by_name("a")
    client.property_by_name("a")
    assert len(net.requests) == 1


def test_server_busy_backs_off_then_succeeds(net_factory: Any) -> None:
    """Busy twice: backoffs 5.0 then 10.0; the waits after them need no extra sleep."""
    url = name_property_url("a")
    net = net_factory({url: [BUSY, BUSY, _table(_prop(1, "A", "K"))]})
    assert PubChemClient(None).property_by_name("a") == PubChemProperty.model_validate(
        _prop(1, "A", "K")
    )
    assert net.sleeps == [5.0, 10.0]
    assert [r["at"] for r in net.requests] == [1000.0, 1005.0, 1015.0]


def test_server_busy_on_every_attempt_aborts(net_factory: Any) -> None:
    """Finding: seven attempts (``_MAX_BUSY_RETRIES + 1``), and the loop also sleeps the
    35 s backoff after the LAST attempt before raising, so an abort costs 140 s of
    which the final 35 s waits for nothing. Pinned until the backoff is skipped after
    the final attempt (compound_identity_curate.py:278).
    """
    url = name_property_url("a")
    net = net_factory({url: [BUSY] * 7})
    with pytest.raises(
        RuntimeError, match=re.escape(f"PubChem stayed busy after 6 retries for {url}")
    ):
        PubChemClient(None).property_by_name("a")
    assert len(net.requests) == 7
    assert net.sleeps == [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]


def test_any_other_fault_aborts_with_the_fault(net_factory: Any) -> None:
    url = name_property_url("a")
    fault = {"Fault": {"Code": "PUGREST.BadRequest", "Message": "bad"}}
    net_factory({url: [fault]})
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"PubChem fault PUGREST.BadRequest for {url}: "
            "{'Code': 'PUGREST.BadRequest', 'Message': 'bad'}"
        ),
    ):
        PubChemClient(None).property_by_name("a")


def test_properties_by_cids_batches_at_100_and_posts_form_bodies(
    net_factory: Any,
) -> None:
    """101 CIDs -> two POSTs (100 then 1). Each row is cached under its single-CID URL."""
    cids = list(range(1, 102))
    first = _table(*[_prop(c, f"T{c}", f"K{c}") for c in cids[:100]])
    second = _table(_prop(101, "T101", "K101"))
    net = net_factory({CID_BATCH_URL: [first, second]})
    client = PubChemClient(None)
    out = client.properties_by_cids(cids)
    assert sorted(out) == cids
    assert out[101] == PubChemProperty.model_validate(_prop(101, "T101", "K101"))
    assert [r["method"] for r in net.requests] == ["POST", "POST"]
    assert net.requests[0]["body"] == (
        "cid=" + "%2C".join(str(c) for c in cids[:100])
    ).encode("ascii")
    assert net.requests[1]["body"] == b"cid=101"
    assert net.requests[1]["headers"] == {
        "User-agent": "torchcell-compound-curator",
        "Content-type": "application/x-www-form-urlencoded",
    }
    assert client._cache[cid_property_url(101)] == _table(_prop(101, "T101", "K101"))
    # cached CIDs are not asked again
    assert client.properties_by_cids([5, 101])[5].title == "T5"
    assert len(net.requests) == 2


def test_properties_by_cids_caches_the_whole_batch_under_a_missing_cid(
    net_factory: Any,
) -> None:
    """Finding: a CID PubChem did not return is cached under ITS single-CID URL with the
    WHOLE batch payload, whose first row belongs to another CID; only the
    ``prop.cid == cid`` guard keeps it out of the answer. The cache file therefore
    records a response that URL never gave. Pinned until a missing CID is cached as an
    explicit not-found (compound_identity_curate.py:311-315).
    """
    payload = _table(_prop(5, "Five", "K5"))
    net_factory({CID_BATCH_URL: [payload]})
    client = PubChemClient(None)
    assert client.properties_by_cids([5, 7]) == {
        5: PubChemProperty.model_validate(_prop(5, "Five", "K5"))
    }
    assert client._cache[cid_property_url(7)] == payload
    assert client._cache[cid_property_url(5)] == payload


def test_chebi_by_cids(net_factory: Any) -> None:
    """One POST for both; a CID with two distinct ChEBI ids gets None; a CID absent from
    the answer caches ``[]``.
    """
    answer = {
        "InformationList": {
            "Information": [
                {"CID": 176, "Synonym": ["acetic acid", "CHEBI:15366", "CHEBI:47622"]},
                {"CID": 702, "Synonym": ["ethanol", "CHEBI:16236"]},
            ]
        }
    }
    net = net_factory({SYN_BATCH_URL: [answer]})
    client = PubChemClient(None)
    assert client.chebi_by_cids([176, 702, 9]) == {
        176: None,
        702: "CHEBI:16236",
        9: None,
    }
    assert net.requests[0]["body"] == b"cid=176%2C702%2C9"
    assert client._cache[synonyms_url(9)] == []
    assert client.chebi_by_cids([]) == {}
    assert len(net.requests) == 1


def test_flush_and_reload_skip_the_network(net_factory: Any, tmp_path: Path) -> None:
    """``flush`` writes the cache with sorted keys; a new client loads it and asks nothing."""
    cache = tmp_path / "cache.json"
    url = name_property_url("a")
    net = net_factory({url: [NOT_FOUND]})
    client = PubChemClient(cache)
    client.property_by_name("a")
    client.flush()
    assert cache.read_text() == json.dumps({url: NOT_FOUND}, sort_keys=True)
    again = PubChemClient(cache)
    assert again.property_by_name("a") is None
    assert len(net.requests) == 1
    PubChemClient(None).flush()  # no path: nothing written
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cache.json"]


# --------------------------------------------------------------------------- #
# assemble / _disambiguate
# --------------------------------------------------------------------------- #
def _d(line: str, source: str = "hil") -> CurationDirective:
    return parse_line(line, source)


def _p(
    cid: int, title: str | None, key: str | None, smiles: str | None = None
) -> PubChemProperty:
    return PubChemProperty.model_validate(_prop(cid, title, key, smiles))


def test_assemble_every_branch() -> None:
    """Rows worked by hand from the branches (sorted by lowercased name, then CID)."""
    directives = [
        _d("NaCl"),
        _d("sodium chloride", "nad"),
        _d("actinomycin D | chebi=CHEBI:27666 | canonical", "core"),
        CurationDirective(label="CID 3385", source="w_cids", cid=3385),
        _d("tunicamycin | mixture=ten homologues"),
        _d("zymolyase | chebi=CHEBI:9999 | mixture=enzyme blend"),
        _d("mystery mix | mixture=no structure"),
        _d("foo123"),
        _d("YPD | undefined=yeast extract"),
        _d("ACME-77 | proprietary=vendor code"),
        _d("bar | unresolved=not in pubchem"),
        _d("glucose | query=D-glucose"),
        CurationDirective(label="CID 999", source="w_cids", cid=999),
    ]
    nacl = _p(5234, "Sodium Chloride", "FAPWRFPIFSIZLT-UHFFFAOYSA-M", "[Na+].[Cl-]")
    properties: dict[str, PubChemProperty | None] = {
        "NaCl": nacl,
        "sodium chloride": nacl,
        "actinomycin D": _p(2019, "Dactinomycin", "RJURFGZVJUQBHK-IIXSONLDSA-N", "S1"),
        "CID 3385": _p(3385, "Fluorouracil", "GHASVSINZRGABV-UHFFFAOYSA-N", "S2"),
        "tunicamycin": _p(56927848, "Tunicamycin", "KEY-T", "S3"),
        "zymolyase": None,
        "mystery mix": None,
        "foo123": None,
        "glucose": _p(5793, "D-Glucose", "WQZGKKKJIJFFOK-GASJEMHNSA-N", "S4"),
        "CID 999": _p(999, "Weird", None, "S5"),
    }
    chebi = {
        5234: "CHEBI:26710",
        2019: None,
        3385: "CHEBI:46345",
        56927848: "CHEBI:29699",
        5793: "CHEBI:4167",
        999: None,
    }
    rows = assemble(directives, properties, chebi, TODAY)
    pm = "pubchem_api"
    expected = [
        CuratedRow(
            name="ACME-77",
            resolution_status="PROPRIETARY",
            unresolved_reason="vendor code",
            retrieved_at=TODAY,
            sources=["hil"],
            precedence=1,
        ),
        CuratedRow(
            name="actinomycin D",
            inchikey="RJURFGZVJUQBHK-IIXSONLDSA-N",
            pubchem_cid=2019,
            chebi_id="CHEBI:27666",
            smiles="S1",
            source_url=name_property_url("actinomycin D"),
            retrieval_method=pm,
            retrieved_at=TODAY,
            resolution_status="RESOLVED",
            sources=["core"],
            precedence=2,
        ),
        CuratedRow(
            name="bar",
            resolution_status="UNRESOLVED_PUBLIC",
            unresolved_reason="not in pubchem",
            retrieved_at=TODAY,
            sources=["hil"],
            precedence=1,
        ),
        CuratedRow(
            name="d-glucose",
            inchikey="WQZGKKKJIJFFOK-GASJEMHNSA-N",
            pubchem_cid=5793,
            chebi_id="CHEBI:4167",
            smiles="S4",
            synonyms=["glucose"],
            source_url=name_property_url("D-glucose"),
            retrieval_method=pm,
            retrieved_at=TODAY,
            resolution_status="RESOLVED",
            sources=["hil"],
            precedence=1,
        ),
        CuratedRow(
            name="fluorouracil",
            inchikey="GHASVSINZRGABV-UHFFFAOYSA-N",
            pubchem_cid=3385,
            chebi_id="CHEBI:46345",
            smiles="S2",
            synonyms=["CID 3385"],
            source_url=cid_property_url(3385),
            retrieval_method=pm,
            retrieved_at=TODAY,
            resolution_status="RESOLVED",
            sources=["w_cids"],
            precedence=0,
        ),
        CuratedRow(
            name="foo123",
            source_url=name_property_url("foo123"),
            retrieval_method=pm,
            retrieved_at=TODAY,
            resolution_status="UNRESOLVED_PUBLIC",
            unresolved_reason="PubChem name lookup returned no CID (PUGREST.NotFound)",
            sources=["hil"],
            precedence=1,
        ),
        CuratedRow(
            name="mystery mix",
            source_url=name_property_url("mystery mix"),
            retrieval_method=pm,
            retrieved_at=TODAY,
            resolution_status="UNRESOLVED_PUBLIC",
            unresolved_reason="no structure",
            sources=["hil"],
            precedence=1,
        ),
        CuratedRow(
            name="sodium chloride",
            inchikey="FAPWRFPIFSIZLT-UHFFFAOYSA-M",
            pubchem_cid=5234,
            chebi_id="CHEBI:26710",
            smiles="[Na+].[Cl-]",
            synonyms=["NaCl"],
            source_url=name_property_url("NaCl"),
            retrieval_method=pm,
            retrieved_at=TODAY,
            resolution_status="RESOLVED",
            sources=["hil", "nad"],
            precedence=1,
        ),
        CuratedRow(
            name="tunicamycin",
            pubchem_cid=56927848,
            chebi_id="CHEBI:29699",
            source_url=name_property_url("tunicamycin"),
            retrieval_method=pm,
            retrieved_at=TODAY,
            resolution_status="RESOLVED_MIXTURE",
            unresolved_reason="ten homologues",
            sources=["hil"],
            precedence=1,
        ),
        CuratedRow(
            name="weird",
            pubchem_cid=999,
            smiles="S5",
            synonyms=["CID 999"],
            source_url=cid_property_url(999),
            retrieval_method=pm,
            retrieved_at=TODAY,
            resolution_status="UNRESOLVED_PUBLIC",
            unresolved_reason="PubChem returned a CID with no InChIKey",
            sources=["w_cids"],
            precedence=0,
        ),
        CuratedRow(
            name="YPD",
            resolution_status="UNDEFINED_MIXTURE",
            unresolved_reason="yeast extract",
            retrieved_at=TODAY,
            sources=["hil"],
            precedence=1,
        ),
        CuratedRow(
            name="zymolyase",
            chebi_id="CHEBI:9999",
            source_url=name_property_url("zymolyase"),
            retrieval_method=pm,
            retrieved_at=TODAY,
            resolution_status="RESOLVED_MIXTURE",
            unresolved_reason="enzyme blend",
            sources=["hil"],
            precedence=1,
        ),
    ]
    assert rows == expected


def test_a_mixed_cid_and_name_group_records_the_cid_url() -> None:
    """A CID line and a name line on one compound: precedence 1 (a name lookup is in the
    group), but the recorded source is the single-CID endpoint.
    """
    p = _p(3385, "Fluorouracil", "K", "S")
    rows = assemble(
        [CurationDirective(label="CID 3385", source="w", cid=3385), _d("5-FU")],
        {"CID 3385": p, "5-FU": p},
        {3385: None},
        TODAY,
    )
    assert [
        (r.name, r.synonyms, r.source_url, r.precedence, r.sources) for r in rows
    ] == [
        ("fluorouracil", ["5-FU", "CID 3385"], cid_property_url(3385), 1, ["hil", "w"])
    ]


def test_a_terminal_member_wins_over_a_lookup_in_its_group() -> None:
    """Two lines on one label (``name:`` merge key, case-folded): the first member's
    terminal reason sets the status; the case variant is dropped from synonyms because
    it is the row's own key.
    """
    rows = assemble(
        [_d("Foo | unresolved=r1"), _d("foo | proprietary=r2", "nad")], {}, {}, TODAY
    )
    assert rows == [
        CuratedRow(
            name="Foo",
            resolution_status="UNRESOLVED_PUBLIC",
            unresolved_reason="r1",
            retrieved_at=TODAY,
            sources=["hil", "nad"],
            precedence=1,
        )
    ]


def test_conflicting_canonical_names_are_refused() -> None:
    p = _p(1, "X", "K")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "conflicting canonical names for one compound: ['X1', 'x2'] "
            "(sources ['a', 'b']); exactly one input line may claim `canonical` per compound"
        ),
    ):
        assemble(
            [_d("x2 | canonical", "b"), _d("X1 | canonical", "a")],
            {"x2": p, "X1": p},
            {},
            TODAY,
        )


def test_a_title_tie_qualifies_both_rows_with_their_cid() -> None:
    """Two CIDs titled Cisplatin, both reached by name (precedence 1): the key
    ``cisplatin`` is a tie, so neither keeps it; the unique synonym survives.
    """
    rows = assemble(
        [_d("cisplatin"), _d("cis-diamminedichloroplatinum")],
        {
            "cisplatin": _p(5702198, "Cisplatin", "K1"),
            "cis-diamminedichloroplatinum": _p(84691, "Cisplatin", "K2"),
        },
        {},
        TODAY,
    )
    assert [(r.name, r.synonyms) for r in rows] == [
        ("cisplatin (CID 5702198)", []),
        ("cisplatin (CID 84691)", ["cis-diamminedichloroplatinum"]),
    ]


def test_precedence_awards_a_key_to_the_curated_row() -> None:
    """Curated (2) beats CID-only (0) on the name; a curated name also strips the key
    from a name-index row's synonyms (1 < 2).
    """
    rows = assemble(
        [
            _d("cisplatin | canonical", "core"),
            CurationDirective(label="CID 84691", source="w", cid=84691),
            _d("Platinol"),
            _d("cisplatin2 | query=cisplatin"),
        ],
        {
            "cisplatin": _p(5702198, "Cisplatin", "K1"),
            "CID 84691": _p(84691, "Cisplatin", "K2"),
            "Platinol": _p(777, "Platinol-AQ", "K3"),
            "cisplatin2": _p(5702198, "Cisplatin", "K1"),
        },
        {},
        TODAY,
    )
    assert [(r.name, r.synonyms, r.precedence) for r in rows] == [
        ("cisplatin", ["cisplatin2"], 2),
        ("cisplatin (CID 84691)", ["CID 84691"], 0),
        ("platinol-aq", ["Platinol"], 1),
    ]


def test_disambiguate_strips_a_lower_precedence_synonym() -> None:
    rows = [
        CuratedRow(
            name="NaCl", pubchem_cid=1, resolution_status="RESOLVED", precedence=2
        ),
        CuratedRow(
            name="salt",
            pubchem_cid=2,
            synonyms=["nacl", "rock salt"],
            resolution_status="RESOLVED",
            precedence=1,
        ),
    ]
    assert [(r.name, r.synonyms) for r in _disambiguate(rows)] == [
        ("NaCl", []),
        ("salt", ["rock salt"]),
    ]


def test_a_name_collision_without_a_cid_is_refused() -> None:
    """A terminal row (no CID) tying with a resolved row on its name cannot be qualified."""
    with pytest.raises(
        ValueError, match=re.escape("name collision on 'x' with no CID to qualify it")
    ):
        assemble([_d("x | unresolved=r"), _d("y")], {"y": _p(9, "X", "K")}, {}, TODAY)


def test_serialize_exact_bytes() -> None:
    """Sorted keys, indent 2, ``sources``/``precedence`` dropped, non-ASCII kept, newline."""
    row = CuratedRow(
        name="β-alanine",
        pubchem_cid=239,
        resolution_status="RESOLVED",
        sources=["s"],
        precedence=1,
    )
    assert serialize([row]) == (
        "{\n"
        '  "records": [\n'
        "    {\n"
        '      "chebi_id": null,\n'
        '      "inchikey": null,\n'
        '      "name": "β-alanine",\n'
        '      "pubchem_cid": 239,\n'
        '      "resolution_status": "RESOLVED",\n'
        '      "retrieval_method": null,\n'
        '      "retrieved_at": null,\n'
        '      "smiles": null,\n'
        '      "source_url": null,\n'
        '      "synonyms": [],\n'
        '      "unresolved_reason": null\n'
        "    }\n"
        "  ],\n"
        '  "schema_version": 2\n'
        "}\n"
    )


# --------------------------------------------------------------------------- #
# curate end to end
# --------------------------------------------------------------------------- #
def test_curate_end_to_end(net_factory: Any, tmp_path: Path) -> None:
    """Request order: the CID batch, then each DISTINCT name query in file order (the
    terminal ``YPD`` line is never asked), then one synonyms batch over every CID
    found, sorted. Departures 0.25 s apart. The cache file holds every URL answered,
    and the serialized table parses back to exactly the records below.
    """
    names = tmp_path / "hil.txt"
    names.write_text(
        "# review: Hillenmeyer 2008 SI\nNaCl\nsodium chloride\nfoo123\n"
        "YPD | undefined=yeast extract\nsalt | query=NaCl\n",
        encoding="utf-8",
    )
    cids = tmp_path / "w_cids.txt"
    cids.write_text("# review: Wildenhain 2015\n3385\n", encoding="utf-8")
    nacl = _table(
        _prop(5234, "Sodium Chloride", "FAPWRFPIFSIZLT-UHFFFAOYSA-M", "[Na+].[Cl-]")
    )
    fu = _table(
        _prop(
            3385, "Fluorouracil", "GHASVSINZRGABV-UHFFFAOYSA-N", "C1=C(C(=O)NC(=O)N1)F"
        )
    )
    syn = {
        "InformationList": {
            "Information": [
                {"CID": 3385, "Synonym": ["5-FU", "CHEBI:46345"]},
                {"CID": 5234, "Synonym": ["salt", "CHEBI:26710"]},
            ]
        }
    }
    net = net_factory(
        {
            CID_BATCH_URL: [fu],
            name_property_url("NaCl"): [nacl],
            name_property_url("sodium chloride"): [nacl],
            name_property_url("foo123"): [NOT_FOUND],
            SYN_BATCH_URL: [syn],
        }
    )
    cache = tmp_path / "cache.json"
    rows, directives = curate([names], [cids], cache)

    assert [(r["method"], r["url"], r["body"], r["at"]) for r in net.requests] == [
        ("POST", CID_BATCH_URL, b"cid=3385", 1000.0),
        ("GET", name_property_url("NaCl"), None, 1000.25),
        ("GET", name_property_url("sodium chloride"), None, 1000.5),
        ("GET", name_property_url("foo123"), None, 1000.75),
        ("POST", SYN_BATCH_URL, b"cid=3385%2C5234", 1001.0),
    ]
    assert [d.label for d in directives] == [
        "NaCl",
        "sodium chloride",
        "foo123",
        "YPD",
        "salt",
        "CID 3385",
    ]
    assert json.loads(cache.read_text()) == {
        cid_property_url(3385): fu,
        name_property_url("NaCl"): nacl,
        name_property_url("sodium chloride"): nacl,
        name_property_url("foo123"): NOT_FOUND,
        synonyms_url(3385): ["5-FU", "CHEBI:46345"],
        synonyms_url(5234): ["salt", "CHEBI:26710"],
    }
    payload = json.loads(serialize(rows))
    assert payload == {
        "schema_version": 2,
        "records": [
            {
                "name": "fluorouracil",
                "inchikey": "GHASVSINZRGABV-UHFFFAOYSA-N",
                "pubchem_cid": 3385,
                "chebi_id": "CHEBI:46345",
                "smiles": "C1=C(C(=O)NC(=O)N1)F",
                "synonyms": ["CID 3385"],
                "source_url": cid_property_url(3385),
                "retrieval_method": "pubchem_api",
                "retrieved_at": TODAY,
                "resolution_status": "RESOLVED",
                "unresolved_reason": None,
            },
            {
                "name": "foo123",
                "inchikey": None,
                "pubchem_cid": None,
                "chebi_id": None,
                "smiles": None,
                "synonyms": [],
                "source_url": name_property_url("foo123"),
                "retrieval_method": "pubchem_api",
                "retrieved_at": TODAY,
                "resolution_status": "UNRESOLVED_PUBLIC",
                "unresolved_reason": "PubChem name lookup returned no CID (PUGREST.NotFound)",
            },
            {
                "name": "sodium chloride",
                "inchikey": "FAPWRFPIFSIZLT-UHFFFAOYSA-M",
                "pubchem_cid": 5234,
                "chebi_id": "CHEBI:26710",
                "smiles": "[Na+].[Cl-]",
                "synonyms": ["NaCl", "salt"],
                "source_url": name_property_url("NaCl"),
                "retrieval_method": "pubchem_api",
                "retrieved_at": TODAY,
                "resolution_status": "RESOLVED",
                "unresolved_reason": None,
            },
            {
                "name": "YPD",
                "inchikey": None,
                "pubchem_cid": None,
                "chebi_id": None,
                "smiles": None,
                "synonyms": [],
                "source_url": None,
                "retrieval_method": None,
                "retrieved_at": TODAY,
                "resolution_status": "UNDEFINED_MIXTURE",
                "unresolved_reason": "yeast extract",
            },
        ],
    }


def test_curate_flushes_the_cache_every_50_distinct_lookups(
    net_factory: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """51 names: after the 50th lookup the cache is flushed (the 51st request sees a
    50-entry file on disk) and the progress line is printed once.
    """
    labels = [f"n{i:02d}" for i in range(51)]
    names = tmp_path / "many.txt"
    names.write_text("# review\n" + "\n".join(labels) + "\n", encoding="utf-8")
    net = net_factory({name_property_url(n): [NOT_FOUND] for n in labels})
    cache = tmp_path / "cache.json"
    seen: list[int] = []
    net.on_request.append(
        lambda: seen.append(
            len(json.loads(cache.read_text())) if cache.exists() else -1
        )
    )
    rows, _ = curate([names], [], cache)
    assert seen == [-1] * 50 + [50]
    assert capsys.readouterr().out == "  ... 50/51 name lookups\n"
    assert len(json.loads(cache.read_text())) == 51
    assert [r.resolution_status for r in rows] == ["UNRESOLVED_PUBLIC"] * 51
    assert len(net.requests) == 51  # no synonyms batch: no CID was found
