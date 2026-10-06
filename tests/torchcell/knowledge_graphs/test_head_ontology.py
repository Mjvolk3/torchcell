# tests/torchcell/knowledge_graphs/test_head_ontology.py
# [[tests.torchcell.knowledge_graphs.test_head_ontology]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_head_ontology.py
"""Issue #619: the KG build's head ontology is the sha256-pinned local Biolink mirror.

Every BioCypher config in ``biocypher/config/`` must set ``head_ontology.url`` to a local
path (no ``scheme://``) that names the mirrored file: the container path
``/var/lib/neo4j/biocypher/ontology/...`` (the generation container mounts the build
tree's copy of the repo ``biocypher/`` there) or the repo-relative
``biocypher/ontology/...``. The stored file must hash to the sha256 pinned in its
provenance record, which was a3e47f5a...707d1a over 451318 bytes when fetched from
GitHub on 2026-10-02. ``verify_head_ontology`` accepts the mirror and refuses a URL, a
missing file, a missing record, a wrong root node and altered bytes, each with its
exact message.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

import torchcell.knowledge_graphs.head_ontology as head_ontology
from torchcell.knowledge_graphs.head_ontology import (
    BIOLINK_ROOT_NODE,
    BIOLINK_SOURCE_URL,
    CONTAINER_BIOCYPHER_DIR,
    CONTAINER_ONTOLOGY_PATH,
    REPO_ONTOLOGY_PATH,
    HeadOntologyError,
    HeadOntologyMirror,
    build_mirror_record,
    load_mirror_record,
    main,
    record_path,
    verify_head_ontology,
)
from torchcell.literature.manifest import (
    ArtifactRecord,
    RetrievalMethod,
    RetrievalRecord,
    SourceCheck,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[3]
MIRROR = REPO / REPO_ONTOLOGY_PATH
PINNED_SHA256 = "a3e47f5a42e4e0e6b4c715f8f4dbbeef5e77683621fa9ad57d15b7d048707d1a"
PINNED_BYTES = 451318
# Every file under biocypher/config that configures BioCypher itself (the schema
# configs carry no ``biocypher:`` section).
BIOCYPHER_CONFIGS = [
    "biocypher_docker_config.yaml",
    "costanzo_biocypher_config.yaml",
    "costanzo_biocypher_docker_config.yaml",
    "linux-amd_biocypher_config.yaml",
    "linux-arm_biocypher_config.yaml",
]


def test_the_config_list_is_every_biocypher_config_in_the_repo() -> None:
    """A new config variant must be added here, so it cannot skip the check."""
    found = sorted(
        p.name
        for p in (REPO / "biocypher" / "config").glob("*.yaml")
        if "biocypher" in (yaml.safe_load(p.read_text()) or {})
    )
    assert found == BIOCYPHER_CONFIGS


@pytest.mark.parametrize("name", BIOCYPHER_CONFIGS)
def test_every_config_points_at_the_local_mirror(name: str) -> None:
    """The url is a path with no scheme, naming the mirror as the container or the
    repo root sees it, with Biolink's ``entity`` as root node.
    """
    config = yaml.safe_load((REPO / "biocypher" / "config" / name).read_text())
    head = config["biocypher"]["head_ontology"]
    assert "://" not in head["url"]
    assert head["url"] in (CONTAINER_ONTOLOGY_PATH, REPO_ONTOLOGY_PATH)
    assert head["root_node"] == "entity"


def test_linux_amd_names_the_path_inside_the_generation_container() -> None:
    """The live rebuild's container .env names linux-amd; its url is under the mount
    of the build tree's biocypher directory, and maps back onto the repo file.
    """
    config = yaml.safe_load(
        (REPO / "biocypher/config/linux-amd_biocypher_config.yaml").read_text()
    )
    url = config["biocypher"]["head_ontology"]["url"]
    assert url == "/var/lib/neo4j/biocypher/ontology/biolink-model-v3.2.1.owl.ttl"
    assert REPO / "biocypher" / Path(url).relative_to(CONTAINER_BIOCYPHER_DIR) == MIRROR


def test_the_stored_file_matches_its_pinned_sha256() -> None:
    """The record pins the bytes; the stored file hashes to them."""
    record = load_mirror_record(MIRROR)
    assert (record.file.sha256, record.file.bytes) == (PINNED_SHA256, PINNED_BYTES)
    assert sha256_file(MIRROR) == PINNED_SHA256
    assert MIRROR.stat().st_size == PINNED_BYTES
    assert record.file.retrieval is not None
    assert record.file.retrieval.source_url == BIOLINK_SOURCE_URL
    assert record.file.retrieval.sha256 == PINNED_SHA256
    assert record.file.retrieval.retrieved_at == "2026-10-02T21:49:16Z"
    assert record.retrieval_command.endswith(BIOLINK_SOURCE_URL)


def test_the_repo_relative_config_verifies_from_the_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """linux-arm (the host config) resolves against the working directory."""
    monkeypatch.chdir(REPO)
    assert (
        verify_head_ontology("biocypher/config/linux-arm_biocypher_config.yaml")
        == MIRROR
    )


def _write_config(path: Path, url: str, root_node: str = "entity") -> Path:
    path.write_text(
        f"biocypher:\n  head_ontology:\n    url: {url}\n    root_node: {root_node}\n"
    )
    return path


def _copy_mirror(tmp_path: Path) -> Path:
    dest = tmp_path / MIRROR.name
    shutil.copy2(MIRROR, dest)
    shutil.copy2(record_path(MIRROR), record_path(dest))
    return dest


def test_verify_accepts_the_mirror_by_absolute_path(tmp_path: Path) -> None:
    """An absolute path to an intact copy with its record verifies."""
    copy = _copy_mirror(tmp_path)
    assert verify_head_ontology(_write_config(tmp_path / "bc.yaml", str(copy))) == copy


def test_verify_refuses_a_missing_head_ontology(tmp_path: Path) -> None:
    """No ``head_ontology`` means BioCypher's GitHub default: refused."""
    cfg = tmp_path / "bc.yaml"
    cfg.write_text("biocypher:\n  offline: true\n")
    message = (
        f"{cfg}: biocypher.head_ontology.url is not set, so BioCypher would fetch "
        f"{BIOLINK_SOURCE_URL} at run time; point it at the mirrored {REPO_ONTOLOGY_PATH}"
    )
    with pytest.raises(HeadOntologyError, match=f"^{re.escape(message)}$"):
        verify_head_ontology(cfg)


@pytest.mark.parametrize("url", [BIOLINK_SOURCE_URL, f"file://{MIRROR}"])
def test_verify_refuses_any_url(tmp_path: Path, url: str) -> None:
    """A url with a scheme is refused, file:// included: the config names a path."""
    cfg = _write_config(tmp_path / "bc.yaml", url)
    message = (
        f"{cfg}: head_ontology.url {url!r} is a URL, not a local path; the build "
        "must not fetch the ontology at run time"
    )
    with pytest.raises(HeadOntologyError, match=f"^{re.escape(message)}$"):
        verify_head_ontology(cfg)


def test_verify_refuses_a_missing_file(tmp_path: Path) -> None:
    """A path that does not exist is named with its resolved form."""
    missing = tmp_path / "nope.owl.ttl"
    cfg = _write_config(tmp_path / "bc.yaml", str(missing))
    message = f"{cfg}: head ontology {str(missing)!r} not found (resolved to {missing})"
    with pytest.raises(HeadOntologyError, match=f"^{re.escape(message)}$"):
        verify_head_ontology(cfg)


def test_verify_refuses_a_file_without_its_record(tmp_path: Path) -> None:
    """The bytes alone are not enough; the provenance record must sit beside them."""
    copy = tmp_path / MIRROR.name
    shutil.copy2(MIRROR, copy)
    cfg = _write_config(tmp_path / "bc.yaml", str(copy))
    message = f"no provenance record at {record_path(copy)}"
    with pytest.raises(HeadOntologyError, match=f"^{re.escape(message)}$"):
        verify_head_ontology(cfg)


def test_verify_refuses_a_different_root_node(tmp_path: Path) -> None:
    """The root node must be the recorded one."""
    copy = _copy_mirror(tmp_path)
    cfg = _write_config(tmp_path / "bc.yaml", str(copy), root_node="named thing")
    message = f"{cfg}: head_ontology.root_node 'named thing' != recorded 'entity'"
    with pytest.raises(HeadOntologyError, match=f"^{re.escape(message)}$"):
        verify_head_ontology(cfg)


def test_verify_refuses_altered_bytes(tmp_path: Path) -> None:
    """One appended byte changes the sha256 and the size; both are reported."""
    copy = _copy_mirror(tmp_path)
    with copy.open("ab") as fh:
        fh.write(b"\n")
    got = sha256_file(copy)
    cfg = _write_config(tmp_path / "bc.yaml", str(copy))
    message = (
        f"{copy}: sha256 {got} ({PINNED_BYTES + 1} bytes) does not match the pinned "
        f"{PINNED_SHA256} ({PINNED_BYTES} bytes) in {record_path(copy)}"
    )
    with pytest.raises(HeadOntologyError, match=f"^{re.escape(message)}$"):
        verify_head_ontology(cfg)


# --------------------------------------------------------------------------- #
# 2026.10.06 (Phase 21): ``build_mirror_record`` and the ``record`` / ``verify`` CLI on a
# tmp ontology file. The re-check runs the REAL ``check_source`` -> ``direct_url``
# retriever with ``httpx.Client`` replaced at ``torchcell.literature.retrieve.httpx`` by
# a MockTransport client, and ``datetime`` frozen at 2026-10-06T12:00:00Z.
# --------------------------------------------------------------------------- #
_TTL = b"@prefix biolink: <https://w3id.org/biolink/vocab/> .\n"
_REAL_CLIENT = httpx.Client


class _FrozenDatetime:
    @staticmethod
    def now(tz: Any) -> datetime:
        return datetime(2026, 10, 6, 12, 0, 0, tzinfo=tz)


def _serve(monkeypatch: pytest.MonkeyPatch, body: bytes) -> list[str]:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=body)

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setattr(head_ontology, "datetime", _FrozenDatetime)
    return seen


def _expected(path: Path, last_check: SourceCheck | None) -> HeadOntologyMirror:
    sha = hashlib.sha256(_TTL).hexdigest()
    return HeadOntologyMirror(
        ontology="biolink 3.2.1",
        root_node="entity",
        retrieval_command="curl -sL -o o.ttl URL",
        file=ArtifactRecord(
            path=path.name,
            role="ontology",
            bytes=len(_TTL),
            sha256=sha,
            source=BIOLINK_SOURCE_URL,
            retrieval=RetrievalRecord(
                method=RetrievalMethod.direct_url,
                source_url=BIOLINK_SOURCE_URL,
                retriever="torchcell.literature.retrieve.direct_url",
                params={"url": BIOLINK_SOURCE_URL},
                sha256=sha,
                retrieved_at="2026-10-02T00:00:00Z",
                last_check=last_check,
            ),
        ),
    )


def test_build_mirror_record_without_check(tmp_path: Path) -> None:
    path = tmp_path / "o.ttl"
    path.write_bytes(_TTL)
    got = build_mirror_record(
        path, "2026-10-02T00:00:00Z", "curl -sL -o o.ttl URL", False
    )
    assert got == _expected(path, None)


@pytest.mark.parametrize(("served", "matches"), [(_TTL, True), (b"drifted\n", False)])
def test_build_mirror_record_check_reruns_the_retriever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: bytes, matches: bool
) -> None:
    """The check fetches the canonical Biolink URL once; ``matches`` compares the
    produced sha256 with the stored file's, and upstream drift never rewrites ``sha256``.
    """
    seen = _serve(monkeypatch, served)
    path = tmp_path / "o.ttl"
    path.write_bytes(_TTL)
    got = build_mirror_record(
        path, "2026-10-02T00:00:00Z", "curl -sL -o o.ttl URL", True
    )
    check = SourceCheck(
        checked_at="2026-10-06T12:00:00Z",
        produced_sha256=hashlib.sha256(served).hexdigest(),
        matches=matches,
    )
    assert got == _expected(path, check)
    assert seen == [BIOLINK_SOURCE_URL]


def test_main_record_then_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``record`` writes ``<file>.provenance.json`` (model JSON, indent 2, trailing
    newline) and prints its sha256; ``verify`` on a config naming the file prints ``ok``.
    """
    path = tmp_path / "o.ttl"
    path.write_bytes(_TTL)
    main(
        [
            "record",
            "--file",
            str(path),
            "--retrieved-at",
            "2026-10-02T00:00:00Z",
            "--retrieval-command",
            "curl -sL -o o.ttl URL",
        ]
    )
    rec = record_path(path)
    sha = hashlib.sha256(_TTL).hexdigest()
    assert capsys.readouterr().out == f"wrote {rec} sha256={sha}\n"
    assert rec.read_text() == _expected(path, None).model_dump_json(indent=2) + "\n"
    assert json.loads(rec.read_text())["root_node"] == BIOLINK_ROOT_NODE
    config = tmp_path / "cfg.yaml"
    config.write_text(
        yaml.safe_dump(
            {"biocypher": {"head_ontology": {"url": str(path), "root_node": "entity"}}}
        )
    )
    main(["verify", str(config)])
    assert capsys.readouterr().out == f"ok {path.resolve()}\n"


def test_main_record_with_check_writes_the_last_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _serve(monkeypatch, _TTL)
    path = tmp_path / "o.ttl"
    path.write_bytes(_TTL)
    main(
        [
            "record",
            "--file",
            str(path),
            "--retrieved-at",
            "2026-10-02T00:00:00Z",
            "--retrieval-command",
            "c",
            "--check",
        ]
    )
    payload = json.loads(record_path(path).read_text())
    assert payload["file"]["retrieval"]["last_check"] == {
        "checked_at": "2026-10-06T12:00:00Z",
        "produced_sha256": hashlib.sha256(_TTL).hexdigest(),
        "matches": True,
    }
