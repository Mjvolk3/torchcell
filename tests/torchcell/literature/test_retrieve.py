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
_PLAIN_UA = "torchcell-literature (+https://github.com/Mjvolk3/torchcell)"


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
        "torchcell.literature.retrieve.pmc_cloud_object": retrieve.pmc_cloud_object,
        "torchcell.literature.retrieve.plos_supplementary": retrieve.plos_supplementary,
        "torchcell.literature.retrieve.elsevier_mmc": retrieve.elsevier_mmc,
        "torchcell.literature.retrieve.local_archive": retrieve.local_archive,
    }


def test_local_archive_returns_the_bytes_whose_hash_is_pinned(tmp_path: Any) -> None:
    payload = b"Time,Blank,1,2\r\n00:01:43,0.000,0.131,0.134\r\n"
    path = tmp_path / "MV_ex21.csv"
    path.write_bytes(payload)
    pinned = hashlib.sha256(payload).hexdigest()
    assert retrieve.local_archive(str(path), pinned) == payload


def test_local_archive_refuses_bytes_that_changed(tmp_path: Any) -> None:
    path = tmp_path / "inhibitors.xlsx"
    path.write_bytes(b"edited after the manifest was written")
    pinned = hashlib.sha256(b"the bytes the manifest recorded").hexdigest()
    with pytest.raises(retrieve.ArchiveHashMismatchError, match=pinned):
        retrieve.local_archive(str(path), pinned)


def test_local_archive_raises_on_a_missing_file(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError):
        retrieve.local_archive(str(tmp_path / "gone.bsm"), "0" * 64)


def test_local_archive_is_a_retrieval_method() -> None:
    from torchcell.literature.manifest import RetrievalMethod

    assert RetrievalMethod("local_archive") is RetrievalMethod.local_archive


@pytest.mark.parametrize(
    ("fn", "params", "url"),
    [
        (
            retrieve.pmc_cloud_object,
            {"key": "PMC9.1/Table S1.xlsx"},
            "https://pmc-oa-opendata.s3.amazonaws.com/PMC9.1/Table%20S1.xlsx",
        ),
        (
            retrieve.plos_supplementary,
            {"journal": "plosgenetics", "object_doi": "10.1371/journal.pgen.1.s001"},
            "https://journals.plos.org/plosgenetics/article/file"
            "?id=10.1371/journal.pgen.1.s001&type=supplementary",
        ),
        (
            retrieve.elsevier_mmc,
            {"pii": "S2405471220303665", "filename": "mmc2.xlsx"},
            "https://ars.els-cdn.com/content/image/1-s2.0-S2405471220303665-mmc2.xlsx",
        ),
    ],
)
def test_supplementary_retrievers_get_one_built_url(
    server: Any, fn: Any, params: dict[str, str], url: str
) -> None:
    """Each SI retriever is one GET of the URL its params spell (space -> %20)."""
    srv = server({url: httpx.Response(200, content=b"PK si")})
    assert fn(**params) == b"PK si"
    assert srv.seen == [(url, _UA)]


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


# --------------------------------------------------------------------------- #
# Per-host User-Agent: Zenodo refuses the browser string
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://zenodo.org/api/records/8284223/files/x.zip/content", _PLAIN_UA),
        ("https://ZENODO.ORG/records/8284223", _PLAIN_UA),
        ("https://sandbox.zenodo.org/records/1", _PLAIN_UA),
        ("https://notzenodo.org/x", _UA),
        ("https://ars.els-cdn.com/content/image/1-s2.0-X-mmc2.xlsx", _UA),
        ("https://static-content.springer.com/esm/x_ESM.pdf", _UA),
    ],
)
def test_the_user_agent_is_chosen_by_host(url: str, expected: str) -> None:
    """Zenodo and its subdomains get the plain agent; every other host the browser one.

    The subdomain match is on a dotted boundary, so ``sandbox.zenodo.org`` is covered and
    ``notzenodo.org`` is not.
    """
    assert retrieve.user_agent_for(url) == expected


def test_the_plain_agent_names_the_project_and_is_not_a_browser_string() -> None:
    assert _PLAIN_UA == "torchcell-literature (+https://github.com/Mjvolk3/torchcell)"
    assert "Mozilla" not in _PLAIN_UA and "Chrome" not in _PLAIN_UA
    assert retrieve.PLAIN_UA_HOSTS == frozenset({"zenodo.org"})


def test_a_zenodo_get_sends_the_plain_agent(server: Any) -> None:
    """The fix: the recorded retrieval of a Zenodo archive goes out non-browser."""
    url = "https://zenodo.org/api/records/8284223/files/SBRG/precise1k-v1.0.zip/content"
    srv = server({url: httpx.Response(200, content=b"PK\x03\x04")})
    assert retrieve.direct_url(url) == b"PK\x03\x04"
    assert srv.seen == [(url, _PLAIN_UA)]
    assert srv.client_kwargs == [
        {
            "follow_redirects": True,
            "timeout": 120.0,
            "headers": {"User-Agent": _PLAIN_UA},
        }
    ]


def test_a_zenodo_zip_member_read_sends_the_plain_agent(server: Any) -> None:
    """``zip_member`` is the retriever the PRECISE-1K provenance records."""
    container = _zip({"data/precise1k/metadata_qc.csv": b"sample,x\np1k_00001,1\n"})
    url = "https://zenodo.org/records/8284223/files/precise1k-v1.0.zip/content"
    srv = server({url: httpx.Response(200, content=container)})
    sha = hashlib.sha256(container).hexdigest()
    assert retrieve.zip_member(url, "data/precise1k/metadata_qc.csv", sha) == (
        b"sample,x\np1k_00001,1\n"
    )
    assert srv.seen == [(url, _PLAIN_UA)]


def test_a_browser_agent_host_is_unchanged_by_the_zenodo_route(server: Any) -> None:
    """Regression guard: adding the Zenodo route left every other host on ``_UA``."""
    url = "https://h.org/some.zip"
    srv = server({url: httpx.Response(200, content=b"x")})
    assert retrieve.direct_url(url) == b"x"
    assert srv.seen == [(url, _UA)]
