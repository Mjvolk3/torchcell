# tests/torchcell/literature/test_retrieve.py
# [[tests.torchcell.literature.test_retrieve]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_retrieve.py
"""The versioned retrievers over an in-memory HTTP transport.

Fixture: ``httpx.Client`` is replaced at its import site
(``torchcell.literature.retrieve.httpx``) by a factory that records its constructor
keyword arguments and builds the real client on an ``httpx.MockTransport``; every
request URL and User-Agent is recorded. The zip containers are built in memory with
``zipfile``; their sha256 is computed with ``hashlib`` as the independent oracle. The
PMC OA answers are hand-written in the OA service's XML shape.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from typing import Any

import httpx
import pytest

import torchcell.literature.retrieve as retrieve

_REAL_CLIENT = httpx.Client
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120 Safari/537.36 torchcell-literature"
)


class _Server:
    def __init__(self, routes: dict[str, httpx.Response]) -> None:
        self.routes = routes
        self.seen: list[tuple[str, str]] = []
        self.client_kwargs: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.seen.append((url, request.headers["user-agent"]))
        return self.routes[url]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Any:
    def make(routes: dict[str, httpx.Response]) -> _Server:
        srv = _Server(routes)

        def factory(**kwargs: Any) -> httpx.Client:
            srv.client_kwargs.append(kwargs)
            return _REAL_CLIENT(transport=httpx.MockTransport(srv.handler), **kwargs)

        monkeypatch.setattr(httpx, "Client", factory)
        return srv

    return make


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_registry_maps_dotted_paths_to_the_functions() -> None:
    assert retrieve.RETRIEVERS == {
        "torchcell.literature.retrieve.springer_esm": retrieve.springer_esm,
        "torchcell.literature.retrieve.direct_url": retrieve.direct_url,
        "torchcell.literature.retrieve.zip_member": retrieve.zip_member,
        "torchcell.literature.retrieve.pmc_oa_api": retrieve.pmc_oa_api,
    }


@pytest.mark.parametrize("fn", [retrieve.springer_esm, retrieve.direct_url])
def test_single_get_retrievers(server: Any, fn: Any) -> None:
    """Both GET once with the torchcell User-Agent, redirects followed, 120 s timeout."""
    url = "https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1534/MediaObjects/x_ESM.pdf"
    srv = server({url: httpx.Response(200, content=b"%PDF-1.5 esm")})
    assert fn(url) == b"%PDF-1.5 esm"
    assert srv.seen == [(url, _UA)]
    assert srv.client_kwargs == [
        {"follow_redirects": True, "timeout": 120.0, "headers": {"User-Agent": _UA}}
    ]


def test_get_raises_on_a_server_error(server: Any) -> None:
    server({"https://h.org/x": httpx.Response(503)})
    with pytest.raises(
        httpx.HTTPStatusError, match=re.escape("Server error '503 Service Unavailable'")
    ):
        retrieve.direct_url("https://h.org/x")


def test_get_follows_a_redirect(server: Any) -> None:
    srv = server(
        {
            "https://h.org/a": httpx.Response(
                302, headers={"Location": "https://h.org/b"}
            ),
            "https://h.org/b": httpx.Response(200, content=b"B"),
        }
    )
    assert retrieve.direct_url("https://h.org/a") == b"B"
    assert [u for u, _ in srv.seen] == ["https://h.org/a", "https://h.org/b"]


def test_zip_member_checks_the_container_sha_then_reads_the_member(server: Any) -> None:
    container = _zip({"supp/Table_S1.csv": b"id,v\n1,2\n", "other.txt": b"x"})
    srv = server({"https://h.org/s.zip": httpx.Response(200, content=container)})
    sha = hashlib.sha256(container).hexdigest()
    assert (
        retrieve.zip_member("https://h.org/s.zip", "supp/Table_S1.csv", sha)
        == b"id,v\n1,2\n"
    )
    assert srv.client_kwargs[0]["timeout"] == 1800.0


def test_zip_member_refuses_a_changed_container(server: Any) -> None:
    container = _zip({"a.csv": b"1"})
    server({"https://h.org/s.zip": httpx.Response(200, content=container)})
    got = hashlib.sha256(container).hexdigest()
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"zip container sha256 mismatch for https://h.org/s.zip: got {got}, expected {'0' * 64}"
        ),
    ):
        retrieve.zip_member("https://h.org/s.zip", "a.csv", "0" * 64)


def test_zip_member_without_a_container_sha_and_a_missing_member(server: Any) -> None:
    """``None`` skips the container check (re-zipping hosts); an absent member raises
    zipfile's own ``KeyError``.
    """
    container = _zip({"a.csv": b"1"})
    server({"https://h.org/s.zip": httpx.Response(200, content=container)})
    assert retrieve.zip_member("https://h.org/s.zip", "a.csv", None) == b"1"
    with pytest.raises(
        KeyError, match=re.escape("There is no item named 'b.csv' in the archive")
    ):
        retrieve.zip_member("https://h.org/s.zip", "b.csv", None)


OA = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id="


def test_pmc_oa_api_rewrites_ftp_and_fetches_the_tgz(server: Any) -> None:
    xml = (
        '<OA><responseDate>2026-10-06 10:00:00</responseDate><request id="PMC3">x</request>'
        '<records returned-count="1" total-count="1"><record id="PMC3" license="CC BY">'
        '<link format="pdf" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/x.pdf"/>'
        '<link format="tgz" updated="2020" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/08/e0/PMC3.tar.gz"/>'
        "</record></records></OA>"
    )
    tgz = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/08/e0/PMC3.tar.gz"
    srv = server(
        {
            OA + "PMC3": httpx.Response(200, content=xml.encode()),
            tgz: httpx.Response(200, content=b"\x1f\x8b"),
        }
    )
    assert retrieve.pmc_oa_api("PMC3") == b"\x1f\x8b"
    assert [u for u, _ in srv.seen] == [OA + "PMC3", tgz]


def test_pmc_oa_api_refuses_a_non_open_access_id(server: Any) -> None:
    xml = '<OA><request id="PMC9"/><error code="idIsNotOpenAccess">not Open Access</error></OA>'
    server({OA + "PMC9": httpx.Response(200, content=xml.encode())})
    with pytest.raises(
        ValueError, match=re.escape("PMC PMC9 not open-access: idIsNotOpenAccess")
    ):
        retrieve.pmc_oa_api("PMC9")


@pytest.mark.parametrize(
    "record",
    ['<link format="pdf" href="ftp://h/x.pdf"/>', '<link format="tgz" href=""/>', ""],
)
def test_pmc_oa_api_refuses_a_record_without_a_tgz_link(
    server: Any, record: str
) -> None:
    xml = f'<OA><records><record id="PMC5">{record}</record></records></OA>'
    srv = server({OA + "PMC5": httpx.Response(200, content=xml.encode())})
    with pytest.raises(
        ValueError, match=re.escape("PMC PMC5: no tgz package link in OA response")
    ):
        retrieve.pmc_oa_api("PMC5")
    assert len(srv.seen) == 1
