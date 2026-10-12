# torchcell/endpoint_http.py
# [[torchcell.endpoint_http]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/endpoint_http
# Test file: tests/torchcell/test_endpoint_http.py
"""The HTTP contract every tc-data client shares, as a leaf module.

``torchcell.datasets.client`` and ``torchcell.artifacts.resolve`` both talk to tc-data
with the same two ``httpx.Client`` calls, the same endpoint URL and the same transfer
constants. tc-data is a public read-only mirror, so no request carries a key:
``DEFAULT_URL`` is the public endpoint and ``TC_DATA_URL`` overrides it for a mirror or
a dev socket. These names used to live in ``torchcell.datasets.client``, but importing
that module runs ``torchcell/datasets/__init__.py``, which imports every dataset loader,
which imports ``torchcell.data.neo4j_cell``, which imports ``torchcell.artifacts``: a
fresh ``import torchcell.artifacts`` therefore failed with a partially initialized
package (2026-10-07). This module imports nothing from torchcell, so either side can
import it first.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

URL_VAR = "TC_DATA_URL"
DEFAULT_URL = "http://torchcell-database.ncsa.illinois.edu:8724"
DEFAULT_TIMEOUT = 120.0
CHUNK = 1 << 20


class HttpClient(Protocol):
    """The two ``httpx.Client`` calls the clients make; a test client satisfies it too."""

    def get(self, url: str) -> Any:
        """A buffered GET returning a response with ``status_code`` and ``content``."""
        ...

    def stream(
        self, method: str, url: str, *, headers: Mapping[str, str] | None = None
    ) -> Any:
        """A streaming request usable as a context manager yielding the response.

        ``headers`` carries only transfer headers (``Range`` when resuming a partial
        file); no request carries a key.
        """
        ...
