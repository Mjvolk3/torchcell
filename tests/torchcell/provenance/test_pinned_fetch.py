# tests/torchcell/provenance/test_pinned_fetch.py
# [[tests.torchcell.provenance.test_pinned_fetch]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/provenance/test_pinned_fetch.py
"""Tests for the source-first, mirror-fallback fetch of one pinned file.

The source is an ``httpx.Client`` on a ``MockTransport``, so no test touches the
network. The pinned file is the 9-byte payload ``PIN``; the
mirror is a function that writes given bytes and counts its calls. Expected values: the
origin is ``"source"`` only when the source answers 200 with exactly the pinned bytes;
every other case ends with the mirror's bytes at ``dest``, origin ``"mirror"`` and the
reason the source was passed over; a mirror whose bytes miss the pin raises and leaves
no file.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from torchcell.provenance.pinned_fetch import FetchOutcome, fetch_pinned, sha256_of

PIN = b"ACGTACGT\n"
PIN_SHA = hashlib.sha256(PIN).hexdigest()
URL = "https://source.test/genome.fna"


def _source(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class _Mirror:
    """Writes ``data`` to the destination and counts how often it was asked."""

    def __init__(self, data: bytes = PIN) -> None:
        self.data = data
        self.calls = 0

    def __call__(self, dest: Path) -> None:
        self.calls += 1
        dest.write_bytes(self.data)


def _fetch(
    tmp_path: Path, mirror: _Mirror, source_url: str | None, http: httpx.Client | None
) -> FetchOutcome:
    return fetch_pinned(
        tmp_path / "tier" / "genome.fna",
        sha256=PIN_SHA,
        nbytes=len(PIN),
        source_url=source_url,
        mirror=mirror,
        label="set/genome.fna",
        http=http,
    )


def test_source_with_the_pinned_bytes_is_used_and_the_mirror_is_not_asked(
    tmp_path: Path,
) -> None:
    mirror = _Mirror()
    outcome = _fetch(
        tmp_path, mirror, URL, _source(lambda r: httpx.Response(200, content=PIN))
    )
    dest = tmp_path / "tier" / "genome.fna"
    assert outcome == FetchOutcome(
        path=str(dest),
        sha256=PIN_SHA,
        bytes=9,
        origin="source",
        source_url=URL,
        source_failure=None,
    )
    assert dest.read_bytes() == PIN
    assert mirror.calls == 0
    assert sorted(p.name for p in dest.parent.iterdir()) == ["genome.fna"]


def test_changed_source_bytes_fall_back_to_the_mirror_and_name_both_hashes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    drifted = b"ACGTACGA\n"
    mirror = _Mirror()
    with caplog.at_level("WARNING"):
        outcome = _fetch(
            tmp_path,
            mirror,
            URL,
            _source(lambda r: httpx.Response(200, content=drifted)),
        )
    dest = tmp_path / "tier" / "genome.fna"
    assert (outcome.origin, mirror.calls, dest.read_bytes()) == ("mirror", 1, PIN)
    assert outcome.source_failure == (
        "source bytes differ from the pin: 9 bytes sha256 "
        f"{hashlib.sha256(drifted).hexdigest()}, pinned 9 bytes sha256 {PIN_SHA}"
    )
    assert f"set/genome.fna: source {URL} not used" in caplog.text
    assert sorted(p.name for p in dest.parent.iterdir()) == ["genome.fna"]


def test_source_status_503_falls_back_to_the_mirror(tmp_path: Path) -> None:
    mirror = _Mirror()
    outcome = _fetch(tmp_path, mirror, URL, _source(lambda r: httpx.Response(503)))
    assert (outcome.origin, outcome.source_failure, mirror.calls) == (
        "mirror",
        "HTTP 503",
        1,
    )


def test_source_connection_error_falls_back_to_the_mirror(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    mirror = _Mirror()
    outcome = _fetch(tmp_path, mirror, URL, _source(refuse))
    assert (outcome.origin, outcome.source_failure, mirror.calls) == (
        "mirror",
        "ConnectError: no route",
        1,
    )
    assert sha256_of(tmp_path / "tier" / "genome.fna") == PIN_SHA


def test_no_source_url_goes_to_the_mirror_without_a_request(tmp_path: Path) -> None:
    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the source must not be asked")

    mirror = _Mirror()
    outcome = _fetch(tmp_path, mirror, None, _source(never))
    assert (outcome.origin, outcome.source_url, outcome.source_failure) == (
        "mirror",
        None,
        "no source URL is recorded for this file",
    )


def test_non_http_source_url_goes_to_the_mirror(tmp_path: Path) -> None:
    outcome = _fetch(tmp_path, _Mirror(), "ftp://source.test/genome.fna", None)
    assert outcome.source_failure == (
        "source URL scheme is not http(s): ftp://source.test/genome.fna"
    )


def test_mirror_bytes_that_miss_the_pin_raise_and_leave_no_file(tmp_path: Path) -> None:
    mirror = _Mirror(b"wrong")
    with pytest.raises(RuntimeError, match="mirror bytes differ from the pin"):
        _fetch(tmp_path, mirror, URL, _source(lambda r: httpx.Response(404)))
    assert list((tmp_path / "tier").iterdir()) == []
