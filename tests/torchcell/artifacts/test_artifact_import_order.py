# tests/torchcell/artifacts/test_artifact_import_order.py
# [[tests.torchcell.artifacts.test_artifact_import_order]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/artifacts/test_artifact_import_order.py
"""A fresh interpreter can import ``torchcell.artifacts`` first.

Import cycles hide behind import order: the test suite imports the datasets package
before the artifacts package, so the 2026-10-07 cycle (``torchcell.artifacts`` ->
``torchcell.datasets.client`` -> every loader -> ``torchcell.data.neo4j_cell`` ->
``torchcell.artifacts``, partially initialized) passed CI and failed the first script
that imported the artifacts package on its own. Each case here runs in a subprocess so
nothing imported by pytest can mask the order.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

FIRST_IMPORTS = [
    "import torchcell.artifacts",
    "from torchcell.artifacts import resolve, check, materialize, ArtifactRef",
    "import torchcell.data.neo4j_cell",
    "import torchcell.data.neo4j_query_raw",
    "import torchcell.endpoint_http",
]


@pytest.mark.parametrize("statement", FIRST_IMPORTS)
def test_module_imports_cleanly_in_a_fresh_interpreter(statement: str) -> None:
    """The statement is the FIRST torchcell import of the process and succeeds."""
    result = subprocess.run(
        [sys.executable, "-c", f"{statement}\nprint('ok')"],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")
