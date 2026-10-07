# tests/torchcell/literature/test_capture_si.py
# [[tests.torchcell.literature.test_capture_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_capture_si.py
"""Supplementary-file capture over an in-memory HTTP transport.

Fixture: ``httpx.Client`` is replaced (``monkeypatch.setattr(httpx, "Client", ...)``)
by a factory that builds the real client on an ``httpx.MockTransport``, so both the
discovery client (``capture_si.make_client``) and the retrievers
(``retrieve._get``) talk to one scripted server that answers ``(method, url)`` pairs
and records every request. An unscripted request raises ``KeyError`` in the handler,
which the capture reports as ``failed``; nothing leaves the process.

Mirror keys are built under ``tmp_path`` with a real ``manifest.json``. Expected
sha256 values are computed with ``hashlib`` on the scripted bytes (the independent
oracle); expected URLs are spelled out literally from the publishers' URL patterns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import torchcell.literature.capture_si as cs
from torchcell.literature.manifest import (
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    write_manifest,
)
from torchcell.literature.retrieve import RETRIEVERS

_REAL_CLIENT = httpx.Client
NOW = "2026-10-07T00:00:00+00:00"
S3 = "https://pmc-oa-opendata.s3.amazonaws.com"


class _Server:
    def __init__(self, routes: dict[tuple[str, str], httpx.Response]) -> None:
        self.routes = routes
        self.seen: list[tuple[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, str(request.url))
        self.seen.append(key)
        return self.routes[key]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Any:
    def make(routes: dict[tuple[str, str], httpx.Response]) -> _Server:
        srv = _Server(routes)

        def factory(**kwargs: Any) -> httpx.Client:
            return _REAL_CLIENT(transport=httpx.MockTransport(srv.handler), **kwargs)

        monkeypatch.setattr(httpx, "Client", factory)
        return srv

    return make


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _key(
    root: Path, key: str, doi: str, extra: list[ArtifactRecord] | None = None
) -> Path:
    """A mirrored key: ``paper.pdf`` plus a manifest recording it (and ``extra``)."""
    key_dir = root / key
    key_dir.mkdir(parents=True)
    (key_dir / "paper.pdf").write_bytes(b"%PDF paper")
    files = [
        ArtifactRecord(
            path="paper.pdf",
            role="paper_pdf",
            bytes=10,
            sha256=_sha(b"%PDF paper"),
            source="zotero:attachment:A1",
        ),
        *(extra or []),
    ]
    write_manifest(key_dir, Manifest(citation_key=key, doi=doi, files=files))
    return key_dir


def _idconv(
    doi_quoted: str, pmcid: str | None
) -> tuple[tuple[str, str], httpx.Response]:
    url = (
        "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
        f"?ids={doi_quoted}&format=json&tool=torchcell"
    )
    record: dict[str, Any] = {"requested-id": "x"}
    if pmcid:
        record["pmcid"] = pmcid
    return ("GET", url), httpx.Response(200, json={"records": [record]})


def _listing(prefix: str, keys: list[str]) -> tuple[tuple[str, str], httpx.Response]:
    contents = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>false</IsTruncated>{contents}</ListBucketResult>"
    )
    return ("GET", f"{S3}/?list-type=2&prefix={prefix}"), httpx.Response(200, text=body)


JATS = b"""<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink"><body>
<supplementary-material id="S1"><media xlink:href="tab_s1.xlsx"/>
  <caption><p>See <ext-link xlink:href="GSE1">GSE1</ext-link></p></caption>
</supplementary-material>
<supplementary-material id="S2"><inline-supplementary-material xlink:href="fig_s1.pdf"/>
  <license xlink:href="https://creativecommons.org/licenses/by/4.0/"/>
</supplementary-material>
<supplementary-material id="S3" xlink:href="data_s1.csv"/>
<sec sec-type="supplementary-material"><media xlink:href="movie_s1.mp4"/></sec>
</body></article>"""


def test_jats_supplementary_hrefs_takes_media_not_links() -> None:
    assert cs.jats_supplementary_hrefs(JATS) == [
        "tab_s1.xlsx",
        "fig_s1.pdf",
        "data_s1.csv",
        "movie_s1.mp4",
    ]


def _pmc_routes(
    hosted: dict[str, bytes], listed_xml: bytes
) -> dict[tuple[str, str], httpx.Response]:
    """The ID converter, PMC9.1 listing, XML and objects for DOI 10.1000/abc."""
    names = ["PMC9.1/PMC9.1.xml", *(f"PMC9.1/{n}" for n in hosted)]
    routes = dict([_idconv("10.1000%2Fabc", "PMC9"), _listing("PMC9.", names)])
    routes[("GET", f"{S3}/PMC9.1/PMC9.1.xml")] = httpx.Response(200, content=listed_xml)
    for name, data in hosted.items():
        routes[("GET", f"{S3}/PMC9.1/{name}")] = httpx.Response(200, content=data)
    return routes


TWO_FILES_XML = b"""<article xmlns:xlink="http://www.w3.org/1999/xlink">
<supplementary-material><media xlink:href="mmc1.pdf"/></supplementary-material>
<supplementary-material><media xlink:href="mmc2.xlsx"/></supplementary-material>
<supplementary-material><media xlink:href="mmc3.docx"/></supplementary-material>
</article>"""


def test_capture_continues_numbering_and_records_provenance(
    server: Any, tmp_path: Path
) -> None:
    """mmc1 is already recorded as si1.pdf; a stray si2 is on disk -> si3, si4."""
    recorded = ArtifactRecord(
        path="si/si1.pdf",
        role="si_pdf",
        bytes=4,
        sha256=_sha(b"%PDF"),
        source=f"{S3}/PMC9.1/mmc1.pdf",
        original_filename="mmc1.pdf",
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=f"{S3}/PMC9.1/mmc1.pdf",
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": "PMC9.1/mmc1.pdf"},
            sha256=_sha(b"%PDF"),
            retrieved_at="2026-01-01",
        ),
    )
    key_dir = _key(tmp_path, "smith2020", "10.1000/abc", [recorded])
    (key_dir / "si").mkdir()
    (key_dir / "si" / "si1.pdf").write_bytes(b"%PDF")
    (key_dir / "si" / "si2_middle.json").write_text("{}")
    srv = server(
        _pmc_routes(
            {"mmc1.pdf": b"%PDF", "mmc2.xlsx": b"PK xlsx", "mmc3.docx": b"PK docx"},
            TWO_FILES_XML,
        )
    )
    result = cs.capture_key(
        cs.make_client(), tmp_path, cs.KeyRequest(citation_key="smith2020"), now=NOW
    )
    assert result.outcome == cs.SiOutcome.CAPTURED
    assert result.route == "pmc_cloud"
    assert result.pmcid == "PMC9"
    assert result.already_recorded == 1
    assert [(f.path, f.original_filename, f.sha256) for f in result.files] == [
        ("si/si3.xlsx", "mmc2.xlsx", _sha(b"PK xlsx")),
        ("si/si4.docx", "mmc3.docx", _sha(b"PK docx")),
    ]
    assert (key_dir / "si" / "si3.xlsx").read_bytes() == b"PK xlsx"
    assert (key_dir / "si" / "si4.docx").read_bytes() == b"PK docx"
    assert ("GET", f"{S3}/PMC9.1/mmc1.pdf") not in srv.seen
    manifest = json.loads((key_dir / "manifest.json").read_text())
    assert manifest["files"][-1] == {
        "path": "si/si4.docx",
        "role": "si_data",
        "bytes": 7,
        "sha256": _sha(b"PK docx"),
        "source": f"{S3}/PMC9.1/mmc3.docx",
        "zotero_md5": None,
        "original_filename": "mmc3.docx",
        "retrieval": {
            "method": "pmc_cloud",
            "source_url": f"{S3}/PMC9.1/mmc3.docx",
            "retriever": "torchcell.literature.retrieve.pmc_cloud_object",
            "params": {"key": "PMC9.1/mmc3.docx"},
            "sha256": _sha(b"PK docx"),
            "retrieved_at": NOW,
            "last_check": None,
        },
        "processing": None,
    }
    assert manifest["si_data_sources"] == [
        f"{S3}/PMC9.1/mmc2.xlsx",
        f"{S3}/PMC9.1/mmc3.docx",
    ]
    # the paper record, written without original_filename, still omits the key
    assert "original_filename" not in manifest["files"][0]


def test_rerun_downloads_nothing_and_leaves_manifest_unchanged(
    server: Any, tmp_path: Path
) -> None:
    key_dir = _key(tmp_path, "smith2020", "10.1000/abc")
    srv = server(
        _pmc_routes(
            {"mmc1.pdf": b"%PDF", "mmc2.xlsx": b"PK xlsx", "mmc3.docx": b"PK docx"},
            TWO_FILES_XML,
        )
    )
    request = cs.KeyRequest(citation_key="smith2020")
    first = cs.capture_key(cs.make_client(), tmp_path, request, now=NOW)
    assert first.outcome == cs.SiOutcome.CAPTURED
    assert [f.path for f in first.files] == ["si/si1.pdf", "si/si2.xlsx", "si/si3.docx"]
    before = (key_dir / "manifest.json").read_bytes()
    srv.seen.clear()
    second = cs.capture_key(cs.make_client(), tmp_path, request, now="2027-01-01")
    assert second.outcome == cs.SiOutcome.PRESENT
    assert second.already_recorded == 3
    assert second.files == []
    assert (key_dir / "manifest.json").read_bytes() == before
    object_gets = [u for m, u in srv.seen if u.startswith(f"{S3}/PMC9.1/mmc")]
    assert object_gets == []
    assert sorted(p.name for p in (key_dir / "si").iterdir()) == [
        "si1.pdf",
        "si2.xlsx",
        "si3.docx",
    ]


def test_dry_run_plans_paths_and_writes_nothing(server: Any, tmp_path: Path) -> None:
    key_dir = _key(tmp_path, "smith2020", "10.1000/abc")
    before = (key_dir / "manifest.json").read_bytes()
    srv = server(
        _pmc_routes(
            {"mmc1.pdf": b"%PDF", "mmc2.xlsx": b"PK xlsx", "mmc3.docx": b"PK docx"},
            TWO_FILES_XML,
        )
    )
    result = cs.capture_key(
        cs.make_client(),
        tmp_path,
        cs.KeyRequest(citation_key="smith2020"),
        dry_run=True,
    )
    assert result.outcome == cs.SiOutcome.WOULD_CAPTURE
    assert [(f.path, f.source_url, f.sha256) for f in result.files] == [
        ("si/si1.pdf", f"{S3}/PMC9.1/mmc1.pdf", None),
        ("si/si2.xlsx", f"{S3}/PMC9.1/mmc2.xlsx", None),
        ("si/si3.docx", f"{S3}/PMC9.1/mmc3.docx", None),
    ]
    assert not (key_dir / "si").exists()
    assert (key_dir / "manifest.json").read_bytes() == before
    assert [u for _, u in srv.seen if "/mmc" in u] == []


def _no_pmc(doi_quoted: str) -> dict[tuple[str, str], httpx.Response]:
    return dict([_idconv(doi_quoted, None)])


def _crossref(
    doi: str, message: dict[str, Any]
) -> tuple[tuple[str, str], httpx.Response]:
    return ("GET", f"https://api.crossref.org/works/{doi}"), httpx.Response(
        200, json={"message": message}
    )


def test_blocked_publisher_page_becomes_manual_with_url(
    server: Any, tmp_path: Path
) -> None:
    doi = "10.1128/aem.01665-20"
    key_dir = _key(tmp_path, "thompson2020", doi)
    before = (key_dir / "manifest.json").read_bytes()
    page = f"https://journals.asm.org/doi/suppl/{doi}"
    routes = _no_pmc("10.1128%2Faem.01665-20")
    routes.update(
        [
            _crossref(doi, {"member": "235", "publisher": "ASM"}),
            (("GET", page), httpx.Response(403, text="Just a moment...")),
        ]
    )
    server(routes)
    result = cs.capture_key(
        cs.make_client(), tmp_path, cs.KeyRequest(citation_key="thompson2020")
    )
    assert result.outcome == cs.SiOutcome.MANUAL
    assert result.route == "asm"
    assert result.manual_url == page
    assert result.routes_tried == ["pmc_cloud:not_found", "asm:manual"]
    assert result.detail == f"DOI has no PMCID; HTTP 403 from {page}"
    assert not (key_dir / "si").exists()
    assert (key_dir / "manifest.json").read_bytes() == before


def test_atypon_page_that_loads_is_parsed(server: Any, tmp_path: Path) -> None:
    doi = "10.1021/acssynbio.8b00429"
    _key(tmp_path, "acs2018", doi)
    page = f"https://pubs.acs.org/doi/suppl/{doi}"
    f1 = f"https://pubs.acs.org/doi/suppl/{doi}/suppl_file/sb8b00429_si_001.pdf"
    body = (
        f'<a href="/doi/suppl/{doi}/suppl_file/sb8b00429_si_001.pdf">SI</a>'
        f'<a href="/doi/suppl/{doi}/suppl_file/sb8b00429_si_001.pdf">again</a>'
    )
    routes = _no_pmc("10.1021%2Facssynbio.8b00429")
    routes.update(
        [
            _crossref(doi, {"member": "316", "publisher": "ACS"}),
            (("GET", page), httpx.Response(200, text=body)),
            (("GET", f1), httpx.Response(200, content=b"%PDF acs")),
        ]
    )
    server(routes)
    result = cs.capture_key(
        cs.make_client(), tmp_path, cs.KeyRequest(citation_key="acs2018"), now=NOW
    )
    assert result.outcome == cs.SiOutcome.CAPTURED
    assert [(f.path, f.original_filename, f.retriever) for f in result.files] == [
        (
            "si/si1.pdf",
            "sb8b00429_si_001.pdf",
            "torchcell.literature.retrieve.direct_url",
        )
    ]


def _springer_page(doi: str, names: list[str]) -> str:
    art = "art%3A" + doi.replace("/", "%2F")
    links = "".join(
        f'<a href="https://media.springernature.com/original/springer-static/esm/'
        f'{art}/MediaObjects/{n}">x</a>'
        for n in names
    )
    return f'<meta name="citation_doi" content="{doi}"/>{links}'


def test_pmc_lists_none_then_springer_supplies_files(
    server: Any, tmp_path: Path
) -> None:
    """Measured case (sameith 2015): PMC JATS lists no SI, the ESM page lists 4."""
    doi = "10.1186/s1"
    _key(tmp_path, "bmc2015", doi)
    esm = "https://static-content.springer.com/esm/art%3A10.1186%2Fs1/MediaObjects/"
    routes = dict(
        [
            _idconv("10.1186%2Fs1", "PMC5"),
            _listing("PMC5.", ["PMC5.1/PMC5.1.xml"]),
            _crossref(doi, {"member": "297", "publisher": "Springer"}),
        ]
    )
    routes[("GET", f"{S3}/PMC5.1/PMC5.1.xml")] = httpx.Response(
        200, content=b"<article/>"
    )
    routes[("GET", f"https://link.springer.com/article/{doi}")] = httpx.Response(
        200, text=_springer_page(doi, ["1_MOESM1_ESM.xlsx", "1_MOESM2_ESM.pdf"])
    )
    routes[("GET", esm + "1_MOESM1_ESM.xlsx")] = httpx.Response(200, content=b"PK")
    routes[("GET", esm + "1_MOESM2_ESM.pdf")] = httpx.Response(403)
    server(routes)
    result = cs.capture_key(
        cs.make_client(), tmp_path, cs.KeyRequest(citation_key="bmc2015"), now=NOW
    )
    assert result.routes_tried == ["pmc_cloud:none_listed", "springer:files"]
    # the 403 on one ESM download leaves the key partial, named with its URL
    assert result.outcome == cs.SiOutcome.PARTIAL
    assert [(f.path, f.source_url) for f in result.files] == [
        ("si/si1.xlsx", esm + "1_MOESM1_ESM.xlsx")
    ]
    assert result.unretrieved == [f"1_MOESM2_ESM.pdf ({esm}1_MOESM2_ESM.pdf)"]
    assert result.manual_url == esm + "1_MOESM2_ESM.pdf"


def test_springer_challenge_page_is_manual(server: Any, tmp_path: Path) -> None:
    doi = "10.1038/nmeth.1"
    _key(tmp_path, "nat2008", doi)
    page = f"https://link.springer.com/article/{doi}"
    routes = _no_pmc("10.1038%2Fnmeth.1")
    routes.update(
        [
            _crossref(doi, {"member": "297"}),
            (("GET", page), httpx.Response(200, text="<html>_fs-ch challenge</html>")),
        ]
    )
    server(routes)
    result = cs.capture_key(
        cs.make_client(), tmp_path, cs.KeyRequest(citation_key="nat2008")
    )
    assert result.outcome == cs.SiOutcome.MANUAL
    assert result.manual_url == page
    assert result.detail == f"DOI has no PMCID; HTTP 200 from {page}"


def test_elsevier_probe_stops_after_two_misses_and_names_the_hole(server: Any) -> None:
    pii = "S0000000000000001"
    base = f"https://ars.els-cdn.com/content/image/1-s2.0-{pii}-"
    present = {base + "mmc1.pdf", base + "mmc3.xlsx"}

    srv = server({})

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        srv.seen.append((request.method, url))
        assert request.method == "HEAD"
        return httpx.Response(200 if url in present else 404)

    srv.handler = handler
    route = cs.route_elsevier(
        cs.make_client(), "10.1016/x", {"alternative-id": ["S0000-0000(00)00000-1"]}
    )
    assert route.status == cs.RouteStatus.FILES
    assert [(c.original_filename, c.params) for c in route.candidates] == [
        ("mmc1.pdf", {"pii": pii, "filename": "mmc1.pdf"}),
        ("mmc3.xlsx", {"pii": pii, "filename": "mmc3.xlsx"}),
    ]
    assert route.unretrieved == ["mmc2 (no file under a probed extension)"]
    assert (
        route.manual_url == f"https://www.sciencedirect.com/science/article/pii/{pii}"
    )
    n_ext = len(cs.ELSEVIER_EXTENSIONS)
    # mmc1 hits on its first extension, mmc2 misses every one, mmc3 hits on its
    # second (xlsx), then mmc4 and mmc5 miss every one and end the probe.
    assert len(srv.seen) == 1 + n_ext + 2 + n_ext + n_ext


def test_plos_route_names_files_from_the_redirect(server: Any) -> None:
    doi = "10.1371/journal.pgen.1"
    xml = (
        b'<article xmlns:xlink="http://www.w3.org/1999/xlink">'
        b'<supplementary-material mimetype="image" '
        b'xlink:href="info:doi/10.1371/journal.pgen.1.s001"/></article>'
    )
    file_url = (
        "https://journals.plos.org/plosgenetics/article/file"
        "?id=10.1371/journal.pgen.1.s001&type=supplementary"
    )
    server(
        {
            (
                "GET",
                "https://journals.plos.org/plosgenetics/article/file"
                f"?id={doi}&type=manuscript",
            ): httpx.Response(200, content=xml),
            ("HEAD", file_url): httpx.Response(
                302,
                headers={
                    "location": "https://storage.googleapis.com/c/10.1371/"
                    "journal.pgen.1/1/pgen.1.s001.tif?X-Goog-Signature=abc"
                },
            ),
        }
    )
    route = cs.route_plos(cs.make_client(), doi)
    assert route.status == cs.RouteStatus.FILES
    assert [c.model_dump() for c in route.candidates] == [
        {
            "original_filename": "pgen.1.s001.tif",
            "source_url": file_url,
            "method": RetrievalMethod.direct_url,
            "retriever": "torchcell.literature.retrieve.plos_supplementary",
            "params": {
                "journal": "plosgenetics",
                "object_doi": "10.1371/journal.pgen.1.s001",
            },
        }
    ]


def test_publisher_with_no_route_and_unknown_doi_are_manual(server: Any) -> None:
    server({})
    client = cs.make_client()
    oup = cs.route_publisher(
        client, "10.1093/nar/x", {"member": "286", "publisher": "OUP"}
    )
    assert (oup.status, oup.manual_url, oup.detail) == (
        cs.RouteStatus.MANUAL,
        "https://doi.org/10.1093/nar/x",
        "no scripted route for OUP",
    )
    missing = cs.route_publisher(client, "10.9/none", None)
    assert (missing.route, missing.manual_url) == (
        "crossref",
        "https://doi.org/10.9/none",
    )


def test_foreign_si_key_is_present_without_any_request(
    server: Any, tmp_path: Path
) -> None:
    zotero_si = ArtifactRecord(
        path="si/si1.pdf",
        role="si_pdf",
        bytes=3,
        sha256=_sha(b"abc"),
        source="zotero:attachment:A2",
    )
    _key(tmp_path, "ohya2005", "10.1073/pnas.1", [zotero_si])
    srv = server({})
    result = cs.capture_key(
        cs.make_client(), tmp_path, cs.KeyRequest(citation_key="ohya2005")
    )
    assert result.outcome == cs.SiOutcome.PRESENT
    assert result.detail == "holds 1 SI files from another path: ['si/si1.pdf']"
    assert srv.seen == []


def test_missing_key_dir_or_manifest_is_no_mirror_dir(
    server: Any, tmp_path: Path
) -> None:
    (tmp_path / "halfCaptured2026").mkdir()
    srv = server({})
    client = cs.make_client()
    absent = cs.capture_key(client, tmp_path, cs.KeyRequest(citation_key="nope2026"))
    half = cs.capture_key(
        client, tmp_path, cs.KeyRequest(citation_key="halfCaptured2026")
    )
    no_key = cs.capture_key(
        client, tmp_path, cs.KeyRequest(citation_key=None, doi="10.1/x")
    )
    assert [r.outcome for r in (absent, half, no_key)] == [
        cs.SiOutcome.NO_MIRROR_DIR
    ] * 3
    assert (
        half.detail == "halfCaptured2026/manifest.json absent (paper not captured yet)"
    )
    assert no_key.detail == "no mirror directory with this DOI"
    assert srv.seen == []


def test_requests_for_dois_skips_underscore_stores(tmp_path: Path) -> None:
    _key(tmp_path, "smith2020", "10.1000/ABC")
    (tmp_path / "_bib").mkdir()
    (tmp_path / "_bib" / "manifest.json").write_text('{"version": 1, "bibs": []}')
    requests = cs.requests_for_dois(tmp_path, ["10.1000/abc", "10.2/none"])
    assert requests == [
        cs.KeyRequest(citation_key="smith2020", doi="10.1000/abc"),
        cs.KeyRequest(citation_key=None, doi="10.2/none"),
    ]
    assert cs.dedupe([*requests, requests[0]]) == requests


def test_main_dry_run_writes_report_under_report_dir(
    server: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mirror = tmp_path / "torchcell-library"
    _key(mirror, "smith2020", "10.1000/abc")
    server(
        _pmc_routes(
            {"mmc1.pdf": b"%PDF"},
            b'<article xmlns:xlink="http://www.w3.org/1999/xlink"><supplementary-material xlink:href="mmc1.pdf"/></article>',
        )
    )
    reports = tmp_path / "reports"
    code = cs.main(
        [
            "smith2020",
            "absent2026",
            "--dry-run",
            "--mirror-root",
            str(mirror),
            "--report-dir",
            str(reports),
        ]
    )
    assert code == 0
    [path] = list(reports.iterdir())
    assert path.name.startswith("si_dryrun_") and path.suffix == ".json"
    report = cs.SiCaptureReport.model_validate_json(path.read_text())
    assert report.dry_run is True
    assert report.tally() == {"no_mirror_dir": 1, "would_capture": 1}
    assert report.summary() == "2 keys | no_mirror_dir=1 would_capture=1"
    assert not (mirror / "_sync_reports").exists()
    assert "2 keys | no_mirror_dir=1 would_capture=1" in capsys.readouterr().out


def test_route_retrievers_are_registered() -> None:
    for name in (cs.R_PMC_CLOUD, cs.R_SPRINGER, cs.R_PLOS, cs.R_ELSEVIER, cs.R_DIRECT):
        assert RETRIEVERS[name].__module__ == "torchcell.literature.retrieve"
    assert RETRIEVERS[cs.R_PMC_CLOUD].__name__ == "pmc_cloud_object"


def test_original_filename_is_omitted_from_json_when_unset() -> None:
    bare = ArtifactRecord(path="paper.pdf", role="paper_pdf", bytes=1, sha256="0")
    named = bare.model_copy(update={"original_filename": "mmc1.pdf"})
    assert "original_filename" not in json.loads(bare.model_dump_json())
    assert json.loads(named.model_dump_json())["original_filename"] == "mmc1.pdf"


def test_ocr_records_only_the_outputs_of_captured_pdfs(
    server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``do_ocr`` OCRs the new SI PDF and records its MinerU outputs, nothing else.

    The fake ``ocr_pdf`` writes what ``_run_mineru.py`` + ``ocr_pdf`` write beside a
    PDF (markdown, layout JSON, the processing record, a figure under
    ``si/images/<stem>/``). A stray file at the key root must stay unrecorded.
    """
    key_dir = _key(tmp_path, "smith2020", "10.1000/abc")
    (key_dir / "stray.txt").write_text("not ours")
    server(
        _pmc_routes(
            {"mmc1.pdf": b"%PDF si", "mmc2.xlsx": b"PK xlsx", "mmc3.docx": b"PK"},
            TWO_FILES_XML,
        )
    )
    ocr_calls: list[str] = []
    provenance = {
        "processor": "torchcell.literature.ocr.ocr_pdf",
        "tool": "mineru",
        "version": "2.5.4",
        "params": {"dpi": 200},
        "input_sha256": [_sha(b"%PDF si")],
    }

    def fake_ocr(pdf: Path) -> Path:
        ocr_calls.append(pdf.name)
        md = pdf.with_suffix(".md")
        md.write_text("# si\n")
        (pdf.parent / f"{pdf.stem}_content_list.json").write_text("[]")
        (pdf.parent / f"{pdf.stem}_ocr_provenance.json").write_text(
            json.dumps(provenance)
        )
        images = pdf.parent / "images" / pdf.stem
        images.mkdir(parents=True)
        (images / "f1.jpg").write_bytes(b"jpg")
        return md

    monkeypatch.setattr(cs, "ocr_pdf", fake_ocr)
    result = cs.capture_key(
        cs.make_client(),
        tmp_path,
        cs.KeyRequest(citation_key="smith2020"),
        do_ocr=True,
        now=NOW,
    )
    assert result.outcome == cs.SiOutcome.CAPTURED
    assert ocr_calls == ["si1.pdf"]
    assert result.ocr == ["si/si1.md"]
    manifest = Manifest.model_validate_json((key_dir / "manifest.json").read_text())
    roles = {r.path: r.role for r in manifest.files}
    assert roles == {
        "paper.pdf": "paper_pdf",
        "si/si1.pdf": "si_pdf",
        "si/si2.xlsx": "si_data",
        "si/si3.docx": "si_data",
        "si/si1.md": "si_ocr",
        "si/si1_content_list.json": "ocr_layout",
        "si/si1_ocr_provenance.json": "ocr_provenance",
        "si/images/si1/f1.jpg": "ocr_image",
    }
    md = next(r for r in manifest.files if r.path == "si/si1.md")
    assert (md.source, md.sha256) == ("mineru-ocr", _sha(b"# si\n"))
    assert md.processing is not None
    assert md.processing.input_sha256 == [_sha(b"%PDF si")]
    # the OCR'd markdown carries no original_filename, yet a rerun stays ours
    rerun = cs.capture_key(
        cs.make_client(), tmp_path, cs.KeyRequest(citation_key="smith2020")
    )
    assert rerun.outcome == cs.SiOutcome.PRESENT


def test_elife_route_collects_additional_files_and_figure_source_data(
    server: Any,
) -> None:
    cdn = "https://cdn.elifesciences.org/articles/05224/"
    article = {
        "additionalFiles": [
            {
                "uri": cdn + "elife-05224-supp1-v2.xlsx",
                "filename": "elife-05224-supp1-v2.xlsx",
            }
        ],
        "body": [
            {
                "type": "figure",
                "assets": [
                    {
                        "sourceData": [
                            {"uri": cdn + "elife-05224-fig1-data1-v2.csv"},
                            {"uri": cdn + "elife-05224-supp1-v2.xlsx"},
                        ]
                    }
                ],
            }
        ],
    }
    server(
        {
            ("GET", "https://api.elifesciences.org/articles/05224"): httpx.Response(
                200, json=article
            )
        }
    )
    route = cs.route_elife(cs.make_client(), "10.7554/eLife.05224")
    assert route.status == cs.RouteStatus.FILES
    assert [(c.original_filename, c.params) for c in route.candidates] == [
        ("elife-05224-supp1-v2.xlsx", {"url": cdn + "elife-05224-supp1-v2.xlsx"}),
        (
            "elife-05224-fig1-data1-v2.csv",
            {"url": cdn + "elife-05224-fig1-data1-v2.csv"},
        ),
    ]


def test_wiley_route_parses_download_supplement_links(server: Any) -> None:
    doi = "10.1111/1462-2920.70095"
    link = "/action/downloadSupplement?doi=10.1111%2F1462-2920.70095&amp;file=emi70095-sup-0001-Table.xlsx"
    page = f'<html>{doi}<a href="{link}">S1</a><a href="{link}">S1</a></html>'
    server(
        {
            ("GET", f"https://onlinelibrary.wiley.com/doi/full/{doi}"): httpx.Response(
                200, text=page
            )
        }
    )
    route = cs.route_wiley(cs.make_client(), doi)
    url = (
        "https://onlinelibrary.wiley.com/action/downloadSupplement"
        "?doi=10.1111%2F1462-2920.70095&file=emi70095-sup-0001-Table.xlsx"
    )
    assert [(c.original_filename, c.source_url) for c in route.candidates] == [
        ("emi70095-sup-0001-Table.xlsx", url)
    ]
