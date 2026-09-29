# tests/torchcell/test_api_keys.py
# [[tests.torchcell.test_api_keys]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/test_api_keys.py
"""``torchcell.api_keys``: hashed named keys shared by the tc-lit and tc-data servers."""

import hashlib
import json
from pathlib import Path

import pytest

from torchcell.api_keys import (
    API_KEY_HEADER,
    ApiKeys,
    hash_key,
    mint_key,
    print_minted_key,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_header_name_is_x_api_key() -> None:
    assert API_KEY_HEADER == "X-API-Key"


def test_hash_key_is_sha256_of_utf8() -> None:
    assert hash_key("secret") == _sha("secret")


def test_from_pairs_hashes_every_key_and_strips_whitespace() -> None:
    keys = ApiKeys.from_pairs(" mac : key-1 , collab:key-2 ,,")
    assert keys.hashes == {"mac": _sha("key-1"), "collab": _sha("key-2")}


def test_from_pairs_rejects_a_pair_without_a_key_naming_the_variable() -> None:
    with pytest.raises(ValueError, match=r"bad TC_X_API_KEYS pair: 'mac'"):
        ApiKeys.from_pairs("mac", env_name="TC_X_API_KEYS")


def test_from_file_reads_name_to_hash(tmp_path: Path) -> None:
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"mac": _sha("k")}))
    assert ApiKeys.from_file(keys_file).hashes == {"mac": _sha("k")}


def test_verify_returns_the_matching_name_or_none() -> None:
    keys = ApiKeys.from_pairs("mac:key-1,collab:key-2")
    assert keys.verify("key-2") == "collab"
    assert keys.verify("key-3") is None


def test_from_env_names_prefers_the_keys_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"filekey": _sha("from-file")}))
    monkeypatch.setenv("TC_X_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("TC_X_API_KEYS", "inline:from-env")
    keys = ApiKeys.from_env_names("TC_X_KEYS_FILE", "TC_X_API_KEYS")
    assert keys.hashes == {"filekey": _sha("from-file")}


def test_from_env_names_falls_to_inline_pairs_then_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TC_X_KEYS_FILE", raising=False)
    monkeypatch.setenv("TC_X_API_KEYS", "inline:from-env")
    keys = ApiKeys.from_env_names("TC_X_KEYS_FILE", "TC_X_API_KEYS")
    assert keys.hashes == {"inline": _sha("from-env")}
    monkeypatch.delenv("TC_X_API_KEYS")
    with pytest.raises(
        KeyError, match="Set TC_X_KEYS_FILE or TC_X_API_KEYS to run the server."
    ):
        ApiKeys.from_env_names("TC_X_KEYS_FILE", "TC_X_API_KEYS")


def test_mint_key_entry_is_the_hash_of_the_returned_key() -> None:
    key, entry = mint_key("mac")
    assert entry == {"mac": _sha(key)}
    assert ApiKeys(hashes=entry).verify(key) == "mac"


def test_print_minted_key_prints_key_and_keys_file_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_minted_key("mac", "TC_DATA_KEYS_FILE")
    out = capsys.readouterr().out
    key = out.splitlines()[1].strip()
    assert out.startswith("API key for 'mac'")
    assert "Add this to your TC_DATA_KEYS_FILE" in out
    assert json.dumps({"mac": _sha(key)}) in out
