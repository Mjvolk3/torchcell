# torchcell/endpoint_http.py
# [[torchcell.endpoint_http]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/endpoint_http
# Test file: tests/torchcell/test_endpoint_http.py
"""The HTTP contract every tc-data client shares, as a leaf module.

``torchcell.datasets.client`` and ``torchcell.artifacts.resolve`` both talk to tc-data
with the same two ``httpx.Client`` calls, the same environment variable names and the
same transfer constants. They used to live in ``torchcell.datasets.client``, but
importing that module runs ``torchcell/datasets/__init__.py``, which imports every
dataset loader, which imports ``torchcell.data.neo4j_cell``, which imports
``torchcell.artifacts``: a fresh ``import torchcell.artifacts`` therefore failed with a
partially initialized package (2026-10-07). This module imports nothing from torchcell,
so either side can import it first.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

URL_VAR = "TC_DATA_URL"
API_KEY_VAR = "TC_DATA_API_KEY"
DEFAULT_TIMEOUT = 120.0
CHUNK = 1 << 20


class HttpClient(Protocol):
    """The two ``httpx.Client`` calls the clients make; a test client satisfies it too."""

    def get(self, url: str, *, headers: Mapping[str, str]) -> Any:
        """A buffered GET returning a response with ``status_code`` and ``content``."""
        ...

    def stream(self, method: str, url: str, *, headers: Mapping[str, str]) -> Any:
        """A streaming request usable as a context manager yielding the response."""
        ...
