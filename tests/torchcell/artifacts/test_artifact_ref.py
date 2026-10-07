# tests/torchcell/artifacts/test_artifact_ref.py
# [[tests.torchcell.artifacts.test_artifact_ref]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/artifacts/test_artifact_ref.py
"""``ArtifactRef``: the ``tc://`` string round trip and every validator refusal."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from torchcell.artifacts.ref import TIERS, ArtifactRef

SHA = "0123456789abcdef" * 4


def _ref(**overrides: object) -> ArtifactRef:
    fields: dict[str, object] = {
        "tier": "genomes",
        "key": "peter2018_1011_assemblies",
        "path": "allReferenceGenes.tar.gz",
        "sha256": SHA,
    }
    fields.update(overrides)
    return ArtifactRef.model_validate(fields)


def test_tiers_are_the_four_named_tiers() -> None:
    assert TIERS == ("raw", "genomes", "library", "objects")


def test_string_form_without_member() -> None:
    ref = _ref()
    assert str(ref) == "tc://genomes/peter2018_1011_assemblies/allReferenceGenes.tar.gz"
    assert ArtifactRef.parse(str(ref), sha256=SHA) == ref


def test_string_form_with_a_member_that_itself_contains_a_hash() -> None:
    """Caudal's member form ``<gene>.fasta#<token>``: split at the FIRST ``#``."""
    ref = _ref(member="YAL001C.fasta#AAA_1")
    text = "tc://genomes/peter2018_1011_assemblies/allReferenceGenes.tar.gz#YAL001C.fasta#AAA_1"
    assert str(ref) == text
    parsed = ArtifactRef.parse(text, sha256=SHA)
    assert parsed == ref
    assert parsed.member == "YAL001C.fasta#AAA_1"


def test_parse_keeps_nested_paths_and_optional_fields() -> None:
    parsed = ArtifactRef.parse(
        "tc://objects/caudal2024-isolate-esm2/esm2/embeddings.zarr",
        sha256=SHA,
        bytes=12,
        media_type="application/zarr",
    )
    assert parsed == ArtifactRef(
        tier="objects",
        key="caudal2024-isolate-esm2",
        path="esm2/embeddings.zarr",
        sha256=SHA,
        bytes=12,
        media_type="application/zarr",
    )
    assert parsed.member is None


@pytest.mark.parametrize(
    "sha256",
    [SHA.upper(), SHA[:-1], SHA + "0", "g" * 64, ""],
    ids=["uppercase", "63-chars", "65-chars", "non-hex", "empty"],
)
def test_sha256_must_be_64_lowercase_hex(sha256: str) -> None:
    with pytest.raises(ValidationError, match="sha256 must be 64 lowercase hex"):
        _ref(sha256=sha256)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("", "key must be one directory name"),
        ("a/b", "key must be one directory name"),
        (".", "key must be one directory name"),
        ("..", "key must be one directory name"),
        ("_bib", "starts with '_', which names a service directory"),
        ("a#b", "key must not contain '#'"),
    ],
)
def test_key_refusals(key: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _ref(key=key)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("", "path must name a file"),
        (".", "path must name a file"),
        ("/abs/file.fa", "path must be relative"),
        ("../escape.fa", "must not contain '..'"),
        ("a/../b.fa", "must not contain '..'"),
        ("file.fa#x", "path must not contain '#'"),
        ("a//b.fa", "path must be normalized"),
        ("./a.fa", "path must be normalized"),
        ("dir/", "path must be normalized"),
    ],
)
def test_path_refusals(path: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _ref(path=path)


def test_negative_bytes_is_refused() -> None:
    with pytest.raises(ValidationError, match="bytes must be non-negative"):
        _ref(bytes=-1)


def test_an_empty_member_is_refused() -> None:
    with pytest.raises(ValidationError, match="member must be a non-empty name"):
        _ref(member="")
    with pytest.raises(ValidationError, match="member must be a non-empty name"):
        ArtifactRef.parse("tc://raw/k/file.tsv#", sha256=SHA)


def test_unknown_tier_and_extra_fields_are_refused() -> None:
    with pytest.raises(ValidationError):
        _ref(tier="scratch")
    with pytest.raises(ValidationError):
        _ref(url="https://example.org")


def test_ref_is_frozen() -> None:
    ref = _ref()
    with pytest.raises(ValidationError):
        ref.path = "other.fa"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("http://raw/k/f.tsv", "does not start with 'tc://'"),
        ("tc://raw/k", "is not tc://<tier>/<key>/<path>"),
        ("tc://raw", "is not tc://<tier>/<key>/<path>"),
        ("tc://scratch/k/f.tsv", "unknown tier 'scratch'"),
    ],
)
def test_parse_refusals(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ArtifactRef.parse(text, sha256=SHA)


def test_parse_applies_the_field_validators() -> None:
    with pytest.raises(ValidationError, match="path must be normalized"):
        ArtifactRef.parse("tc://raw/k/a//b.tsv", sha256=SHA)
    with pytest.raises(ValidationError, match="sha256 must be 64 lowercase hex"):
        ArtifactRef.parse("tc://raw/k/b.tsv", sha256="abc")
