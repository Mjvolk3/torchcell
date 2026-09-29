# torchcell/api_keys.py
# [[torchcell.api_keys]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/api_keys.py
# Test file: tests/torchcell/test_api_keys.py

"""Named API keys stored as sha256 hashes, shared by the ``tc-lit`` and ``tc-data`` servers.

Both endpoints authenticate an ``X-API-Key`` header against a set of named keys. The
key values are never stored or logged: a keys file holds ``{name: sha256hex}``, an inline
``name:key,name2:key2`` spec (quick-start and tests only) is hashed on load, and a
presented key is compared constant-time against every stored hash. Each server binds
its own environment variable names by subclassing :class:`ApiKeys` and overriding
:meth:`ApiKeys.from_env` (``TC_LIT_*`` for the literature endpoint, ``TC_DATA_*`` for
the dataset endpoint); the model and the comparison are identical.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

API_KEY_HEADER = "X-API-Key"


def hash_key(key: str) -> str:
    """sha256 hex digest of an API key (what is stored and compared, never the key)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class ApiKeys(BaseModel):
    """Named API keys, stored as sha256 hashes for constant-time verification."""

    model_config = ConfigDict(frozen=True)

    hashes: dict[str, str] = Field(
        description="Map of key name -> sha256 hex of the key value."
    )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Load ``{name: sha256hex}`` from a JSON keys file."""
        data = json.loads(Path(path).read_text())
        return cls(hashes={str(k): str(v) for k, v in data.items()})

    @classmethod
    def from_pairs(cls, spec: str, env_name: str = "API_KEYS") -> Self:
        """Parse ``name1:key1,name2:key2`` plaintext pairs, hashing each key.

        Convenient for quick-start and tests; the keys file (hashes at rest) is
        preferred for anything real since env values are visible via ``ps``.
        ``env_name`` names the variable in the error for a malformed pair.
        """
        hashes: dict[str, str] = {}
        for pair in spec.split(","):
            pair = pair.strip()
            if not pair:
                continue
            name, _, key = pair.partition(":")
            if not name or not key:
                raise ValueError(f"bad {env_name} pair: {pair!r}")
            hashes[name.strip()] = hash_key(key.strip())
        return cls(hashes=hashes)

    @classmethod
    def from_env_names(cls, file_var: str, inline_var: str) -> Self:
        """Load keys from the keys-file variable (preferred) or the inline-pairs variable.

        Raises ``KeyError`` if neither is set: a server never runs unauthenticated.
        """
        keys_file = os.environ.get(file_var)
        if keys_file:
            return cls.from_file(keys_file)
        inline = os.environ.get(inline_var)
        if inline:
            return cls.from_pairs(inline, env_name=inline_var)
        raise KeyError(f"Set {file_var} or {inline_var} to run the server.")

    def verify(self, presented: str) -> str | None:
        """Return the name of the key matching ``presented``, else None.

        Constant-time over the stored hashes; the presented value is never logged.
        """
        candidate = hash_key(presented)
        for name, stored in self.hashes.items():
            if hmac.compare_digest(candidate, stored):
                return name
        return None


def mint_key(name: str) -> tuple[str, dict[str, str]]:
    """A fresh random key plus the ``{name: sha256hex}`` entry that stores its hash."""
    key = secrets.token_urlsafe(32)
    return key, {name: hash_key(key)}


def print_minted_key(name: str, keys_file_var: str) -> None:
    """Mint a key and print it with the JSON keys-file line (the ``--gen-key`` CLI)."""
    key, entry = mint_key(name)
    print(f"API key for '{name}' (give this to the client, it is NOT stored):\n  {key}")
    print(f"\nAdd this to your {keys_file_var} (JSON of {{name: sha256hex}}):")
    print(f"  {json.dumps(entry)}")
