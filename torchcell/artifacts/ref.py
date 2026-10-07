# torchcell/artifacts/ref.py
# [[torchcell.artifacts.ref]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/artifacts/ref.py
# Test file: tests/torchcell/artifacts/test_artifact_ref.py

"""``ArtifactRef``: the one pointer type from a graph record to bytes kept off the graph.

A ref names a file in one of the four artifact tiers (``raw``, ``genomes``, ``library``,
``objects``) by its key directory and relative path, pins the file's bytes by sha256, and
optionally names a member inside the file (a tar member, a FASTA record token, an h5ad
obs name). The member never changes which bytes are pinned: ``sha256`` is always the
sha256 of the FILE at ``path``.

String form: ``tc://<tier>/<key>/<path>[#<member>]``. The string carries the location
only; the sha256 travels beside it (``ArtifactRef.parse(text, sha256=...)``), because a
location without a hash is not a ref. The member is everything after the FIRST ``#``, so
a member may itself contain ``#`` (Caudal's ``<gene>.fasta#<token>``) while a path may
not.
"""

from __future__ import annotations

import posixpath
import re
from typing import Literal, Self, get_args

from pydantic import field_validator, model_validator

from torchcell.datamodels.pydant import ModelStrict

Tier = Literal["raw", "genomes", "library", "objects"]
TIERS: tuple[str, ...] = get_args(Tier)
URI_SCHEME = "tc://"
MEMBER_SEPARATOR = "#"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ArtifactRef(ModelStrict):
    """A sha256-pinned pointer to one file (and optionally a member) in an artifact tier."""

    tier: Tier
    key: str
    path: str
    member: str | None = None
    sha256: str
    bytes: int | None = None
    media_type: str | None = None

    @field_validator("sha256")
    @classmethod
    def _sha256_is_lowercase_hex(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError(
                f"sha256 must be 64 lowercase hex characters, got {value!r}"
            )
        return value

    @field_validator("key")
    @classmethod
    def _key_is_one_directory_name(cls, value: str) -> str:
        if not value or "/" in value or value in (".", ".."):
            raise ValueError(f"key must be one directory name, got {value!r}")
        if value.startswith("_"):
            raise ValueError(
                f"key {value!r} starts with '_', which names a service directory"
            )
        if MEMBER_SEPARATOR in value:
            raise ValueError(f"key must not contain '#', got {value!r}")
        return value

    @field_validator("path")
    @classmethod
    def _path_is_normalized_and_relative(cls, value: str) -> str:
        if value in ("", "."):
            raise ValueError(f"path must name a file, got {value!r}")
        if value.startswith("/"):
            raise ValueError(f"path must be relative (no leading slash), got {value!r}")
        if ".." in value.split("/"):
            raise ValueError(f"path must not contain '..', got {value!r}")
        if MEMBER_SEPARATOR in value:
            raise ValueError(f"path must not contain '#', got {value!r}")
        if posixpath.normpath(value) != value:
            raise ValueError(
                f"path must be normalized, got {value!r} "
                f"(normalized: {posixpath.normpath(value)!r})"
            )
        return value

    @field_validator("bytes")
    @classmethod
    def _bytes_is_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError(f"bytes must be non-negative, got {value}")
        return value

    @model_validator(mode="after")
    def _member_requires_a_path(self) -> Self:
        if self.member is not None and (not self.member or not self.path):
            raise ValueError("member must be a non-empty name inside the file at path")
        return self

    def __str__(self) -> str:
        """``tc://<tier>/<key>/<path>[#<member>]``."""
        base = f"{URI_SCHEME}{self.tier}/{self.key}/{self.path}"
        if self.member is None:
            return base
        return f"{base}{MEMBER_SEPARATOR}{self.member}"

    @classmethod
    def parse(
        cls,
        text: str,
        *,
        sha256: str,
        bytes: int | None = None,
        media_type: str | None = None,
    ) -> Self:
        """Build a ref from its string form plus the sha256 the string does not carry.

        Raises ``ValueError`` when ``text`` is not ``tc://<tier>/<key>/<path>[#<member>]``
        with a known tier; the field validators then apply as on direct construction.
        """
        if not text.startswith(URI_SCHEME):
            raise ValueError(f"{text!r} does not start with {URI_SCHEME!r}")
        location, sep, member = text[len(URI_SCHEME) :].partition(MEMBER_SEPARATOR)
        parts = location.split("/", 2)
        if len(parts) != 3:
            raise ValueError(f"{text!r} is not tc://<tier>/<key>/<path>[#<member>]")
        tier, key, path = parts
        if tier not in TIERS:
            raise ValueError(f"{text!r}: unknown tier {tier!r}; tiers are {TIERS}")
        return cls.model_validate(
            {
                "tier": tier,
                "key": key,
                "path": path,
                "member": member if sep else None,
                "sha256": sha256,
                "bytes": bytes,
                "media_type": media_type,
            }
        )
