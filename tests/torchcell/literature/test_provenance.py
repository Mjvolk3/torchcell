"""Tests for torchcell.literature.provenance (retrieval/processing records)."""

import hashlib
from collections.abc import Callable

import pytest

from torchcell.literature.manifest import sha256_file
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
# Enum: no manual_browser; radiant_endpoint reserved (issue #20).
# --------------------------------------------------------------------------- #
def test_no_manual_browser_method():
    values = {m.value for m in RetrievalMethod}
    assert "manual_browser" not in values
    assert "radiant_endpoint" in values


# --------------------------------------------------------------------------- #
# Records round-trip (pydantic-first).
# --------------------------------------------------------------------------- #
def test_records_round_trip():
    art = ArtifactRecord(
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
    assert art == ArtifactRecord(**art.model_dump())
    assert art.retrieval is not None and art.retrieval.retriever.endswith("fake")


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
