# tests/torchcell/test_endpoint_http.py
# [[tests.torchcell.test_endpoint_http]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/test_endpoint_http.py
"""The leaf HTTP contract: no torchcell imports, the shared names, httpx satisfies it."""

from __future__ import annotations

import ast
from pathlib import Path

import httpx

from torchcell import endpoint_http
from torchcell.endpoint_http import (
    API_KEY_VAR,
    CHUNK,
    DEFAULT_TIMEOUT,
    URL_VAR,
    HttpClient,
)


def test_leaf_module_imports_nothing_from_torchcell() -> None:
    """The point of the module: either client can import it first."""
    tree = ast.parse(Path(endpoint_http.__file__).read_text(encoding="utf-8"))
    imported = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not [name for name in imported if name.startswith("torchcell")]


def test_shared_names_are_the_tc_data_contract() -> None:
    assert (URL_VAR, API_KEY_VAR) == ("TC_DATA_URL", "TC_DATA_API_KEY")
    assert DEFAULT_TIMEOUT == 120.0
    assert CHUNK == 1 << 20


def test_httpx_client_satisfies_the_protocol() -> None:
    """``httpx.Client`` has the two calls; the Protocol is runtime-checkable by shape."""
    client = httpx.Client()
    try:
        assert callable(client.get) and callable(client.stream)
        http: HttpClient = client
        assert http is client
    finally:
        client.close()
