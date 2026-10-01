# tests/torchcell/graph/test_graph_sgd.py
# [[tests.torchcell.graph.test_graph_sgd]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/graph/test_graph_sgd.py
"""SGD locus fetch/cache on a fake aiohttp session; nothing reaches yeastgenome.org.

``aiohttp.ClientSession`` is replaced at its import site (``torchcell.graph.sgd.aiohttp``)
by :class:`FakeSession`, which answers every GET with a JSON body ``{"url": <url>}`` (or a
scripted failure) and records each URL. ``asyncio.sleep`` is replaced by a recorder so the
exponential backoff (``2 ** retry`` seconds) is asserted without waiting. ``DATA_ROOT``
points at tmp_path wherever the module's default gene directory is read, and the
non-JSON branch's ``NamedTemporaryFile`` is redirected into tmp_path via
``tempfile.tempdir``.

Endpoint URLs are ``osp.join(sgd_url, locusID, endpoint)``, so for ``YAL001C`` the
``go_details`` URL is ``https://www.yeastgenome.org/backend/locus/YAL001C/go_details``.

2026.09.30 (Phase 16). Added: two concurrent ``fetch_data`` calls on one gene share one
download (11 GETs, not 22) and a later call issues none; ``max_retries=0`` returns None
without a GET; a download whose every endpoint fails (with the default 10 retries: 110
GETs and backoff delays ``2**0 .. 2**8`` per endpoint, 9 sleeps each, 99 in all) raises
a ValueError naming the 11 failed keys and writes no file, so ``download_genes`` fetches
the locus again on the next run (issue #538); ``main_get_all_genes`` opens the genome
under ``$DATA_ROOT`` with ``overwrite=False`` and runs one ``download_gene_chunk`` per 50
loci, so 120 loci give chunks of 50, 50 and 20 with ``create_gene`` and
``is_validated=False``.
"""

import asyncio
import json
import os
import os.path as osp
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from pydantic import BaseModel

import torchcell.graph.sgd as sgd
from torchcell.graph.sgd import Gene

BASE = "https://www.yeastgenome.org/backend/locus"
ENDPOINTS = [
    "sequence_details",
    "neighbor_sequence_details",
    "posttranslational_details",
    "protein_experiment_details",
    "protein_domain_details",
    "go_details",
    "phenotype_details",
    "interaction_details",
    "regulation_details",
    "literature_details",
]


class FakeResponse:
    """A response with a fixed content type and body."""

    def __init__(self, content_type: str, body: Any) -> None:
        """Store the header and body."""
        self.headers = {"Content-Type": content_type}
        self._body = body

    async def json(self) -> Any:
        """Return the body as parsed JSON."""
        return self._body

    async def text(self) -> str:
        """Return the body as text."""
        return str(self._body)

    async def __aenter__(self) -> "FakeResponse":
        """Enter the response context."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Leave the context."""
        return None


class FakeSession:
    """Records GETs; ``mode`` selects json, html, a non-container json, or an error."""

    urls: list[str] = []
    headers_seen: list[dict[str, str]] = []
    mode: str = "json"

    async def __aenter__(self) -> "FakeSession":
        """Enter the session context."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Leave the context."""
        return None

    def get(self, url: str, headers: dict[str, str]) -> FakeResponse:
        """Record the GET and answer per ``mode``."""
        FakeSession.urls.append(url)
        FakeSession.headers_seen.append(headers)
        if FakeSession.mode == "error":
            raise aiohttp.ClientError("boom")
        if FakeSession.mode == "html":
            return FakeResponse("text/html", "<html>down</html>")
        if FakeSession.mode == "scalar":
            return FakeResponse("application/json", 7)
        return FakeResponse("application/json; charset=utf-8", {"url": url})


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> type[FakeSession]:
    """Install FakeSession at the sgd import site and a recording ``asyncio.sleep``."""
    FakeSession.urls = []
    FakeSession.headers_seen = []
    FakeSession.mode = "json"
    monkeypatch.setattr(sgd, "aiohttp", SimpleNamespace(ClientSession=FakeSession))
    return FakeSession


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Recorded backoff delays in place of real sleeps."""
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("torchcell.graph.sgd.asyncio.sleep", fake_sleep)
    return delays


def test_data_root_reads_env_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set: the value; unset or empty: a ValueError naming the .env file."""
    monkeypatch.setenv("DATA_ROOT", "/some/root")
    assert sgd.data_root() == "/some/root"
    assert sgd._default_gene_dir() == "/some/root/data/sgd/genome/genes"
    monkeypatch.setenv("DATA_ROOT", "")
    msg = "DATA_ROOT environment variable is not set. Please set it in your .env file."
    with pytest.raises(ValueError, match=re.escape(msg)):
        sgd.data_root()
    monkeypatch.delenv("DATA_ROOT")
    with pytest.raises(ValueError, match=re.escape(msg)):
        sgd.data_root()


def test_new_gene_has_no_data(tmp_path: Path) -> None:
    """A fresh gene creates its directory, sets save_path, and refuses ``data``."""
    gene_dir = tmp_path / "genes"
    gene = Gene(locusID="YAL001C", base_data_dir=str(gene_dir))
    assert gene_dir.is_dir()
    assert gene.save_path == str(gene_dir / "YAL001C.json")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "No data available. Call fetch_data(), e.g., "
            "`asyncio.run(gene.fetch_data())`"
        ),
    ):
        gene.data  # noqa: B018


def test_cached_gene_loads_from_disk(tmp_path: Path) -> None:
    """An existing ``<locus>.json`` is read at construction."""
    (tmp_path / "YAL001C.json").write_text(json.dumps({"locus": {"id": 1}}))
    gene = Gene(locusID="YAL001C", base_data_dir=str(tmp_path))
    assert gene.data == {"locus": {"id": 1}}


def test_read_errors(tmp_path: Path) -> None:
    """``read`` rejects a missing file and a JSON list."""
    gene = Gene(locusID="YAL001C", base_data_dir=str(tmp_path))
    with pytest.raises(
        ValueError, match=re.escape(f"File {gene.save_path} does not exist")
    ):
        gene.read()
    Path(gene.save_path).write_text("[1, 2]")
    with pytest.raises(
        ValueError, match=re.escape(f"File {gene.save_path} is not a dict")
    ):
        gene.read()


def test_fetch_data_downloads_every_endpoint_and_writes(
    tmp_path: Path, session: type[FakeSession]
) -> None:
    """Unvalidated fetch: 11 GETs in order, each stored under its key, then written."""
    gene = Gene(locusID="YAL001C", is_validated=False, base_data_dir=str(tmp_path))
    asyncio.run(gene.fetch_data())
    expected_urls = [f"{BASE}/YAL001C"] + [f"{BASE}/YAL001C/{e}" for e in ENDPOINTS]
    assert session.urls == expected_urls
    assert session.headers_seen == [{"accept": "application/json"}] * 11
    expected = {"locus": {"url": f"{BASE}/YAL001C"}} | {
        e: {"url": f"{BASE}/YAL001C/{e}"} for e in ENDPOINTS
    }
    assert gene.data == expected
    assert json.loads(Path(gene.save_path).read_text()) == expected
    assert Path(gene.save_path).read_text() == json.dumps(expected, indent=4)


def test_fetch_data_raises_when_download_leaves_no_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download that stores nothing surfaces as ``Data fetch failed.``."""

    async def no_op(self: Gene) -> None:
        return None

    monkeypatch.setattr(Gene, "download_data", no_op)
    gene = Gene(locusID="YAL001C", base_data_dir=str(tmp_path))
    with pytest.raises(ValueError, match=re.escape("Data fetch failed.")):
        asyncio.run(gene.fetch_data())


def test_client_error_retries_with_backoff_then_returns_none(
    tmp_path: Path, session: type[FakeSession], sleeps: list[float]
) -> None:
    """Three failing attempts: sleeps 2**0 and 2**1, then None (no sleep after the last)."""
    session.mode = "error"
    gene = Gene(locusID="YAL001C", base_data_dir=str(tmp_path))
    assert asyncio.run(gene._get_data(f"{BASE}/YAL001C", max_retries=3)) is None
    assert session.urls == [f"{BASE}/YAL001C"] * 3
    assert sleeps == [1, 2]


def test_non_container_json_raises(tmp_path: Path, session: type[FakeSession]) -> None:
    """A JSON scalar is rejected with the value in the message."""
    session.mode = "scalar"
    gene = Gene(locusID="YAL001C", base_data_dir=str(tmp_path))
    with pytest.raises(ValueError, match=re.escape("Data is not a dict or list: 7")):
        asyncio.run(gene._get_data(f"{BASE}/YAL001C"))


def test_non_json_response_raises_type_error(
    tmp_path: Path, session: type[FakeSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: sgd.py:161 builds ``ContentTypeError(message)`` with one argument.

    aiohttp's ``ClientResponseError.__init__`` needs ``request_info`` and ``history``,
    so constructing it raises ``TypeError``, which the ``except (ClientError,
    ContentTypeError)`` does not catch: an HTML response escapes the retry loop on the
    first attempt instead of being retried. The body is still saved to a temp file.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    session.mode = "html"
    gene = Gene(locusID="YAL001C", base_data_dir=str(tmp_path / "genes"))
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ClientResponseError.__init__() missing 1 required positional argument: "
            "'history'"
        ),
    ):
        asyncio.run(gene._get_data(f"{BASE}/YAL001C", max_retries=3))
    assert session.urls == [f"{BASE}/YAL001C"]
    saved = [p for p in os.listdir(tmp_path) if p.endswith(".html")]
    assert len(saved) == 1
    assert (tmp_path / saved[0]).read_text() == "<html>down</html>"


def test_validated_locus_writes_schema_into_cwd(
    tmp_path: Path, session: type[FakeSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: sgd.py:181-182 write ``locus_schema.json`` into the CURRENT directory.

    ``validate_data`` is stubbed at its import site with a two-field model so the
    validated branch runs on the fake body; ``locus()`` returns its ``model_dump()``.
    """

    class TinyLocus(BaseModel):
        url: str

    monkeypatch.setattr(sgd, "validate_data", lambda data: TinyLocus(**data))
    monkeypatch.chdir(tmp_path)
    gene = Gene(locusID="YAL001C", base_data_dir=str(tmp_path / "genes"))
    assert asyncio.run(gene.locus()) == {"url": f"{BASE}/YAL001C"}
    assert json.loads((tmp_path / "locus_schema.json").read_text()) == (
        TinyLocus.model_json_schema()
    )


def test_chunks() -> None:
    """Five items in chunks of two: [0, 1], [2, 3], [4]."""
    assert list(sgd.chunks([0, 1, 2, 3, 4], 2)) == [[0, 1], [2, 3], [4]]


def test_create_gene_uses_default_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``create_gene`` builds a Gene under ``$DATA_ROOT/data/sgd/genome/genes``."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    gene = sgd.create_gene("YAL002W", False)
    assert (gene.locusID, gene.is_validated, gene.save_path) == (
        "YAL002W",
        False,
        osp.join(str(tmp_path), "data/sgd/genome/genes/YAL002W.json"),
    )


def test_download_gene_chunk_skips_cached_and_fetches_the_rest(  # test-quality: allow returns None; asserts the files, factory calls, GETs and sleeps it causes
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session: type[FakeSession],
    sleeps: list[float],
) -> None:
    """YAL001C is cached and skipped; YAL002W is fetched (11 GETs) after a 1 s pause
    and written as the locus body plus one body per endpoint, each ``{"url": <url>}``,
    indented 4.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    gene_dir = tmp_path / "data/sgd/genome/genes"
    gene_dir.mkdir(parents=True)
    (gene_dir / "YAL001C.json").write_text("{}")
    made: list[tuple[str, bool]] = []

    def factory(locus: str, validated: bool) -> Gene:
        made.append((locus, validated))
        return sgd.create_gene(locus, validated)

    asyncio.run(sgd.download_gene_chunk(["YAL001C", "YAL002W"], factory, False))
    assert sleeps == [1]
    assert made == [("YAL002W", False)]
    assert session.urls == [f"{BASE}/YAL002W"] + [
        f"{BASE}/YAL002W/{e}" for e in ENDPOINTS
    ]
    assert sorted(os.listdir(gene_dir)) == ["YAL001C.json", "YAL002W.json"]
    assert (gene_dir / "YAL001C.json").read_text() == "{}"
    expected = {"locus": {"url": f"{BASE}/YAL002W"}} | {
        e: {"url": f"{BASE}/YAL002W/{e}"} for e in ENDPOINTS
    }
    assert (gene_dir / "YAL002W.json").read_text() == json.dumps(expected, indent=4)


def test_concurrent_fetches_share_one_download(
    tmp_path: Path, session: type[FakeSession]
) -> None:
    """The second ``fetch_data`` awaits the pending task instead of scheduling another.

    A third call after completion issues no GET at all.
    """
    gene = Gene(locusID="YAL001C", is_validated=False, base_data_dir=str(tmp_path))

    async def twice() -> None:
        await asyncio.gather(gene.fetch_data(), gene.fetch_data())

    asyncio.run(twice())
    assert len(session.urls) == 11
    assert session.urls[0] == f"{BASE}/YAL001C"
    asyncio.run(gene.fetch_data())
    assert len(session.urls) == 11
    assert gene.data["go_details"] == {"url": f"{BASE}/YAL001C/go_details"}


def test_zero_retries_returns_none_without_a_request(
    tmp_path: Path, session: type[FakeSession]
) -> None:
    """``max_retries=0`` skips the loop: None, and the session is never opened."""
    gene = Gene(locusID="YAL001C", base_data_dir=str(tmp_path))
    assert asyncio.run(gene._get_data(f"{BASE}/YAL001C", max_retries=0)) is None
    assert session.urls == []


def test_failed_download_raises_and_caches_nothing(  # test-quality: allow returns None; asserts the error, file, GETs, sleeps and factory calls it causes
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session: type[FakeSession],
    sleeps: list[float],
) -> None:
    """A download where every GET fails raises, writes no file and stores no data.

    It used to store None under each of the 11 keys and write them as JSON nulls, which
    ``download_genes`` then skipped forever as "already exists" (issue #538). Now the
    ValueError names every failed key, and a later ``download_genes`` builds the gene
    again and issues the 110 GETs anew.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    gene_dir = tmp_path / "data/sgd/genome/genes"
    session.mode = "error"
    gene = sgd.create_gene("YAL001C", False)
    with pytest.raises(ValueError) as failed:
        asyncio.run(gene.fetch_data())
    keys = ["locus", *ENDPOINTS]
    assert str(failed.value) == (
        f"SGD fetch failed for YAL001C: {keys} returned no data after every retry; "
        "nothing was cached"
    )
    assert len(session.urls) == 11 * 10
    assert sleeps == [2**i for i in range(9)] * 11
    assert os.listdir(gene_dir) == []
    assert gene._data == {}
    session.urls = []
    made: list[str] = []

    def factory(locus: str, validated: bool) -> Gene:
        made.append(locus)
        return sgd.create_gene(locus, validated)

    with pytest.raises(ValueError, match="SGD fetch failed for YAL001C"):
        asyncio.run(sgd.download_genes(["YAL001C"], factory, False))
    assert (made, len(session.urls)) == (["YAL001C"], 110)


def test_partial_download_failure_names_only_the_failed_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session: type[FakeSession]
) -> None:
    """One failed endpoint is enough to refuse the write; the error names just it."""

    async def no_go(self: Gene) -> None:
        return None

    monkeypatch.setattr(Gene, "go_details", no_go)
    gene = Gene(locusID="YAL001C", is_validated=False, base_data_dir=str(tmp_path))
    with pytest.raises(ValueError) as failed:
        asyncio.run(gene.fetch_data())
    assert str(failed.value) == (
        "SGD fetch failed for YAL001C: ['go_details'] returned no data after every "
        "retry; nothing was cached"
    )
    assert os.listdir(tmp_path) == []
    assert len(session.urls) == 10


class _GenomeStub:
    """Records its constructor arguments; holds 120 locus ids."""

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        _GenomeStub.calls.append((args, kwargs))
        self.gene_set = [f"Y{i:03d}" for i in range(120)]


def test_main_get_all_genes_chunks_by_fifty(  # test-quality: allow returns None; asserts the constructor call and chunks it causes
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The genome opens under ``$DATA_ROOT`` with ``overwrite=False``; 50 loci per chunk.

    It used to be ``SCerevisiaeGenome()``, the relative ``data/sgd/genome`` root with
    ``overwrite=True`` (issue #538; memory note genome-overwrite-true-rebuild-race).
    ``load_dotenv`` is stubbed so the repo ``.env`` is not read. The 120 loci go out as
    chunks of 50, 50, 20, each with ``create_gene`` and validation off.
    """
    import torchcell.sequence.genome.scerevisiae.s288c as s288c

    dotenv_calls: list[object] = []
    monkeypatch.setattr(sgd, "load_dotenv", lambda *a, **k: dotenv_calls.append(a))
    monkeypatch.setenv("DATA_ROOT", "/data-root")
    _GenomeStub.calls = []
    monkeypatch.setattr(s288c, "SCerevisiaeGenome", _GenomeStub)
    seen: list[tuple[list[str], object, bool]] = []

    async def fake_chunk(
        chunk: list[str], create_gene_fn: object, validate_flag: bool
    ) -> None:
        seen.append((chunk, create_gene_fn, validate_flag))

    monkeypatch.setattr(sgd, "download_gene_chunk", fake_chunk)
    sgd.main_get_all_genes()
    assert dotenv_calls == [()]
    assert _GenomeStub.calls == [
        (
            (),
            {
                "genome_root": "/data-root/data/sgd/genome",
                "go_root": "/data-root/data/go",
                "overwrite": False,
            },
        )
    ]
    ids = [f"Y{i:03d}" for i in range(120)]
    assert [chunk for chunk, _, _ in seen] == [ids[:50], ids[50:100], ids[100:]]
    assert {(fn, flag) for _, fn, flag in seen} == {(sgd.create_gene, False)}
