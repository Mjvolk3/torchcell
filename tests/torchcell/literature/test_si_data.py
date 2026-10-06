# tests/torchcell/literature/test_si_data.py
# [[tests.torchcell.literature.test_si_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_si_data.py
"""SI-data fetching (Dryad + direct URLs) over an in-memory HTTP transport.

Fixture: ``httpx.Client`` is replaced at its import site
(``torchcell.literature.si_data.httpx``) by a factory that records the constructor
keyword arguments and builds the real client on an ``httpx.MockTransport``, whose
handler answers scripted URLs and records every request. Nothing leaves the process.

Derivations: ``quote("doi:10.5061/dryad.tt367", safe="")`` is
``doi%3A10.5061%2Fdryad.tt367``; the Dryad hrefs are API-relative and are joined to
``https://datadryad.org``; a direct URL's local name is the percent-decoded last path
segment (``My%20Table.xlsx`` -> ``My Table.xlsx``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

import torchcell.literature.si_data as si_data

_REAL_CLIENT = httpx.Client


class _Server:
    def __init__(self, routes: dict[str, httpx.Response]) -> None:
        self.routes = routes
        self.seen: list[str] = []
        self.client_kwargs: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.seen.append(url)
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


DS_URL = "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.tt367"
FILES_URL = "https://datadryad.org/api/v2/versions/4242/files"


def _dryad_routes() -> dict[str, httpx.Response]:
    return {
        DS_URL: httpx.Response(
            200, json={"_links": {"stash:version": {"href": "/api/v2/versions/4242"}}}
        ),
        FILES_URL: httpx.Response(
            200,
            json={
                "_embedded": {
                    "stash:files": [
                        {
                            "path": "SGA_data.txt",
                            "_links": {
                                "stash:download": {"href": "/api/v2/files/91/download"}
                            },
                        },
                        {
                            "path": "README.md",
                            "_links": {
                                "stash:download": {"href": "/api/v2/files/92/download"}
                            },
                        },
                    ]
                }
            },
        ),
        "https://datadryad.org/api/v2/files/91/download": httpx.Response(
            200, content=b"gene\tfitness\nYAL001C\t0.91\n"
        ),
        "https://datadryad.org/api/v2/files/92/download": httpx.Response(
            200, content=b"# readme\n"
        ),
    }


def test_dryad_files_walks_dataset_version_files(server: Any) -> None:
    srv = server(_dryad_routes())
    assert si_data.dryad_files("10.5061/dryad.tt367") == [
        ("SGA_data.txt", "https://datadryad.org/api/v2/files/91/download"),
        ("README.md", "https://datadryad.org/api/v2/files/92/download"),
    ]
    assert srv.seen == [DS_URL, FILES_URL]
    assert srv.client_kwargs == [{"timeout": 60.0, "follow_redirects": True}]


def test_dryad_files_raises_on_a_missing_dataset(server: Any) -> None:
    server({DS_URL: httpx.Response(404, json={"error": "not found"})})
    with pytest.raises(
        httpx.HTTPStatusError, match=re.escape("Client error '404 Not Found'")
    ):
        si_data.dryad_files("10.5061/dryad.tt367")


def test_download_file_streams_exact_bytes_and_makes_parents(
    server: Any, tmp_path: Path
) -> None:
    body = bytes(range(256)) * 9000  # 2.3 MB: more than one 1 MiB chunk
    srv = server({"https://x.org/d.bin": httpx.Response(200, content=body)})
    dest = tmp_path / "a" / "b" / "d.bin"
    assert si_data.download_file("https://x.org/d.bin", dest) == dest
    assert dest.read_bytes() == body
    assert srv.client_kwargs == [{"timeout": 600.0, "follow_redirects": True}]


def test_download_file_error_writes_nothing(server: Any, tmp_path: Path) -> None:
    server({"https://x.org/gone": httpx.Response(410)})
    dest = tmp_path / "out" / "gone"
    with pytest.raises(
        httpx.HTTPStatusError, match=re.escape("Client error '410 Gone'")
    ):
        si_data.download_file("https://x.org/gone", dest)
    assert not dest.exists()
    assert dest.parent.is_dir()  # the parent is made before the request


def test_fetch_si_data_combines_dryad_and_extra_urls(
    server: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    routes = _dryad_routes()
    routes["https://lab.org/files/My%20Table.xlsx"] = httpx.Response(
        200, content=b"PK-xlsx"
    )
    srv = server(routes)
    caplog.set_level(logging.INFO, logger="torchcell.literature.si_data")
    out = si_data.fetch_si_data(
        tmp_path,
        dryad_doi="10.5061/dryad.tt367",
        extra_urls=["https://lab.org/files/My%20Table.xlsx"],
    )
    d = tmp_path / "si" / "si_data"
    assert out == [
        (d / "SGA_data.txt", "https://datadryad.org/api/v2/files/91/download"),
        (d / "README.md", "https://datadryad.org/api/v2/files/92/download"),
        (d / "My Table.xlsx", "https://lab.org/files/My%20Table.xlsx"),
    ]
    assert (d / "My Table.xlsx").read_bytes() == b"PK-xlsx"
    assert (d / "SGA_data.txt").read_bytes() == b"gene\tfitness\nYAL001C\t0.91\n"
    assert [r.getMessage() for r in caplog.records if r.name == si_data.__name__] == [
        f"SI data: wrote {d / 'SGA_data.txt'} (26 bytes) from https://datadryad.org/api/v2/files/91/download",
        f"SI data: wrote {d / 'README.md'} (9 bytes) from https://datadryad.org/api/v2/files/92/download",
        f"SI data: wrote {d / 'My Table.xlsx'} (7 bytes) from https://lab.org/files/My%20Table.xlsx",
    ]
    assert len(srv.seen) == 5


def test_fetch_si_data_keeps_a_nonempty_file_and_refetches_an_empty_one(
    server: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    d = tmp_path / "si" / "si_data"
    d.mkdir(parents=True)
    (d / "kept.csv").write_bytes(b"old")
    (d / "empty.csv").write_bytes(b"")
    srv = server(
        {
            "https://h.org/kept.csv": httpx.Response(200, content=b"new"),
            "https://h.org/empty.csv": httpx.Response(200, content=b"filled"),
        }
    )
    caplog.set_level(logging.INFO, logger="torchcell.literature.si_data")
    out = si_data.fetch_si_data(
        tmp_path, extra_urls=["https://h.org/kept.csv", "https://h.org/empty.csv"]
    )
    assert out == [
        (d / "kept.csv", "https://h.org/kept.csv"),
        (d / "empty.csv", "https://h.org/empty.csv"),
    ]
    assert (d / "kept.csv").read_bytes() == b"old"
    assert (d / "empty.csv").read_bytes() == b"filled"
    assert srv.seen == ["https://h.org/empty.csv"]
    assert [r.getMessage() for r in caplog.records if r.name == si_data.__name__][
        0
    ] == (f"SI data: keeping existing {d / 'kept.csv'} (3 bytes)")


def test_fetch_si_data_overwrite_refetches(server: Any, tmp_path: Path) -> None:
    d = tmp_path / "si" / "si_data"
    d.mkdir(parents=True)
    (d / "kept.csv").write_bytes(b"old")
    server({"https://h.org/kept.csv": httpx.Response(200, content=b"new")})
    si_data.fetch_si_data(
        tmp_path, extra_urls=["https://h.org/kept.csv"], overwrite=True
    )
    assert (d / "kept.csv").read_bytes() == b"new"


def test_fetch_si_data_with_nothing_requested_returns_empty(
    server: Any, tmp_path: Path
) -> None:
    srv = server({})
    assert si_data.fetch_si_data(tmp_path) == []
    assert srv.seen == []
    assert not (tmp_path / "si").exists()


def test_fetch_si_data_keeps_the_query_string_in_the_local_name(
    server: Any, tmp_path: Path
) -> None:
    """Finding: the local filename is the last ``/``-segment of the WHOLE URL, query
    string included, so a Zenodo-style ``.../files/data.csv?download=1`` lands as
    ``data.csv?download=1``, and a URL ending in ``/`` lands as ``file``, where a second
    such URL would overwrite or be skipped as "existing". Pinned until the name is taken
    from the URL path only (si_data.py:80).
    """
    server(
        {
            "https://zenodo.org/records/7/files/data.csv?download=1": httpx.Response(
                200, content=b"a,b\n"
            ),
            "https://lab.org/export/": httpx.Response(200, content=b"x"),
        }
    )
    out = si_data.fetch_si_data(
        tmp_path,
        extra_urls=[
            "https://zenodo.org/records/7/files/data.csv?download=1",
            "https://lab.org/export/",
        ],
    )
    d = tmp_path / "si" / "si_data"
    assert [p.name for p, _ in out] == ["data.csv?download=1", "file"]
    assert sorted(p.name for p in d.iterdir()) == ["data.csv?download=1", "file"]
