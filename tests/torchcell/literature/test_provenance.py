"""Tests for torchcell.literature.provenance (retrieval/processing records)."""

import hashlib
from collections.abc import Callable

import pytest

from torchcell.literature.manifest import Manifest, sha256_file
from torchcell.literature.provenance import (
    ArtifactRecord,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
    check_source,
    run_retriever,
    verify_artifact,
)

PAYLOAD = b"genetic interaction score algorithm"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _fake_registry(payload: bytes = PAYLOAD) -> dict[str, Callable[..., bytes]]:
    return {"torchcell.literature.retrieve.fake": lambda url: payload}


def _retrieval(sha: str = PAYLOAD_SHA) -> RetrievalRecord:
    return RetrievalRecord(
        method=RetrievalMethod.springer_esm,
        source_url="https://static-content.springer.com/esm/x.pdf",
        retriever="torchcell.literature.retrieve.fake",
        params={"url": "https://static-content.springer.com/esm/x.pdf"},
        sha256=sha,
        retrieved_at="2026-07-03",
    )


# --------------------------------------------------------------------------- #
# Enum: manual_browser is the typed record for a manual-once retrieval (the
# recipe is the command, the deposited bytes are canonical); radiant_endpoint
# reserved (issue #20).
# --------------------------------------------------------------------------- #
def test_manual_browser_and_radiant_endpoint_methods():
    values = {m.value for m in RetrievalMethod}
    assert "manual_browser" in values
    assert "radiant_endpoint" in values


# --------------------------------------------------------------------------- #
# Records round-trip (pydantic-first).
# --------------------------------------------------------------------------- #
def _artifact_with_provenance() -> ArtifactRecord:
    return ArtifactRecord(
        path="software/sga.zip",
        role="software",
        bytes=190975,
        sha256=PAYLOAD_SHA,
        retrieval=_retrieval(),
        processing=ProcessingRecord(
            processor="torchcell.literature.ocr.ocr_pdf",
            tool="mineru",
            version="2.x",
            params={"dpi": 350},
            input_sha256=[PAYLOAD_SHA],
        ),
    )


def test_records_round_trip():
    art = _artifact_with_provenance()
    assert art == ArtifactRecord(**art.model_dump())
    assert art.retrieval is not None and art.retrieval.retriever.endswith("fake")


def test_manifest_preserves_provenance_on_round_trip():
    """Regression: Manifest.files must serialize/reload ArtifactRecord in full.

    A ``list[FileRecord]`` field would drop retrieval/processing by declared-type
    serialization; ``list[ArtifactRecord]`` keeps them across model_dump_json().
    """
    art = _artifact_with_provenance()
    manifest = Manifest(citation_key="baryshnikova2010", files=[art])
    reloaded = Manifest.model_validate_json(manifest.model_dump_json())
    (got,) = reloaded.files
    assert got.retrieval is not None and got.retrieval.sha256 == PAYLOAD_SHA
    assert got.processing is not None and got.processing.tool == "mineru"
    assert got == art


# --------------------------------------------------------------------------- #
# run_retriever resolves the dotted path via an injectable registry.
# --------------------------------------------------------------------------- #
def test_run_retriever_resolves_registry():
    data = run_retriever(_retrieval(), registry=_fake_registry())
    assert data == PAYLOAD


def test_run_retriever_unknown_raises():
    rec = _retrieval()
    with pytest.raises(KeyError):
        run_retriever(rec, registry={})  # retriever not in registry


# --------------------------------------------------------------------------- #
# verify_artifact: stored file's sha256 vs the record (mirror integrity).
# --------------------------------------------------------------------------- #
def test_verify_artifact(tmp_path):
    (tmp_path / "software").mkdir()
    f = tmp_path / "software" / "sga.zip"
    f.write_bytes(PAYLOAD)
    good = ArtifactRecord(
        path="software/sga.zip",
        role="software",
        bytes=len(PAYLOAD),
        sha256=sha256_file(f),
    )
    assert verify_artifact(good, tmp_path)
    bad = good.model_copy(update={"sha256": "deadbeef"})
    assert not verify_artifact(bad, tmp_path)


# --------------------------------------------------------------------------- #
# check_source: re-run retriever, detect upstream drift (never follow it).
# --------------------------------------------------------------------------- #
def test_check_source_match_and_drift():
    ok = check_source(_retrieval(), now="2026-07-03", registry=_fake_registry())
    assert ok.matches and ok.produced_sha256 == PAYLOAD_SHA

    # upstream now serves different bytes -> detected, matches False.
    drift = check_source(
        _retrieval(), now="2026-07-03", registry=_fake_registry(b"changed upstream")
    )
    assert not drift.matches
    assert drift.produced_sha256 == hashlib.sha256(b"changed upstream").hexdigest()


# --------------------------------------------------------------------------- #
# git_repository: a file of a git repository at a pinned commit. The clone goes
# through gh (network), so the hermetic tests pin the command list and read a file
# out of a local repository's commit tree.
# --------------------------------------------------------------------------- #
def test_git_repository_is_the_last_method_and_a_registered_retriever():
    from torchcell.literature.retrieve import RETRIEVERS, git_repository

    assert list(RetrievalMethod)[-1] == RetrievalMethod.git_repository
    assert RetrievalMethod.git_repository.value == "git_repository"
    assert RETRIEVERS["torchcell.literature.retrieve.git_repository"] is git_repository


def test_git_clone_commands_are_the_recorded_recipe():
    from torchcell.literature.retrieve import git_clone_commands

    assert git_clone_commands("owner/repo", "abc123", "/x/repo") == [
        ["gh", "repo", "clone", "owner/repo", "/x/repo", "--", "--depth", "1"],
        ["git", "-C", "/x/repo", "fetch", "--depth", "1", "origin", "abc123"],
        ["git", "-C", "/x/repo", "checkout", "--detach", "abc123"],
    ]


def test_git_show_reads_the_commit_tree_not_the_working_tree(tmp_path):
    import subprocess

    from torchcell.literature.retrieve import git_show

    repo = tmp_path / "repo"
    (repo / "d").mkdir(parents=True)
    (repo / "d" / "f.txt").write_bytes(PAYLOAD)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("add", "d/f.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    commit = git("rev-parse", "HEAD")
    (repo / "d" / "f.txt").write_bytes(b"edited after the commit")
    assert git_show(str(repo), commit, "d/f.txt") == PAYLOAD
    with pytest.raises(subprocess.CalledProcessError):
        git_show(str(repo), commit, "d/missing.txt")
