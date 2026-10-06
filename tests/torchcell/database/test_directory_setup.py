# tests/torchcell/database/test_directory_setup.py
# [[tests.torchcell.database.test_directory_setup]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/database/test_directory_setup.py
"""``directory_setup.main`` on a fake workspace and data root under ``tmp_path``.

Fixture: ``dotenv.load_dotenv`` is replaced by a recorder BEFORE the module is imported
fresh (it calls ``load_dotenv()`` at import), so no ``.env`` is read; the module globals
``DATA_ROOT`` and ``WORKSPACE_DIR`` are then pointed at ``tmp_path/data`` and
``tmp_path/ws``. The workspace holds ``database/conf/gh_neo4j.conf``,
``database/database.env`` and ``biocypher/{config/c.yaml,ontology/o.ttl}``.

Contract pinned: the ten directories, ``gh_neo4j.conf`` copied as ``conf/neo4j.conf``,
the env file copied as ``database/.env`` (default ``$WORKSPACE_DIR/database/database.env``
or ``--env-file``), and ``biocypher/`` REPLACED (a stale file in the old copy is gone).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

import torchcell.database


@pytest.fixture
def setup_mod(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[ModuleType, list[tuple[object, ...]]]:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: calls.append((a, k)))
    monkeypatch.delitem(
        sys.modules, "torchcell.database.directory_setup", raising=False
    )
    # the import also binds the module on the package; record the old binding (or its
    # absence) so teardown restores it
    monkeypatch.setattr(torchcell.database, "directory_setup", None, raising=False)
    mod = importlib.import_module("torchcell.database.directory_setup")
    ws = tmp_path / "ws"
    (ws / "database" / "conf").mkdir(parents=True)
    (ws / "database" / "conf" / "gh_neo4j.conf").write_text(
        "server.memory.heap.max_size=8g\n"
    )
    (ws / "database" / "database.env").write_text("NEO4J_AUTH=neo4j/x\n")
    (ws / "biocypher" / "config").mkdir(parents=True)
    (ws / "biocypher" / "ontology").mkdir()
    (ws / "biocypher" / "config" / "c.yaml").write_text("biocypher: {}\n")
    (ws / "biocypher" / "ontology" / "o.ttl").write_text("@prefix x: <y> .\n")
    monkeypatch.setattr(mod, "DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(mod, "WORKSPACE_DIR", str(ws))
    return mod, calls


def _tree(root: Path) -> list[str]:
    return sorted(
        str(p.relative_to(root)) + ("/" if p.is_dir() else "") for p in root.rglob("*")
    )


def test_main_builds_the_tree_and_copies_conf_env_and_biocypher(
    setup_mod: tuple[ModuleType, list[tuple[object, ...]]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod, calls = setup_mod
    assert calls == [((), {})]
    stale = tmp_path / "data" / "database" / "biocypher" / "stale.yaml"
    stale.parent.mkdir(parents=True)
    stale.write_text("old\n")
    mod.main([])
    data = tmp_path / "data"
    assert _tree(data) == [
        "database/",
        "database/.env",
        "database/biocypher/",
        "database/biocypher/config/",
        "database/biocypher/config/c.yaml",
        "database/biocypher/ontology/",
        "database/biocypher/ontology/o.ttl",
        "database/conf/",
        "database/conf/neo4j.conf",
        "database/data/",
        "database/data/torchcell/",
        "database/import/",
        "database/logs/",
        "database/metrics/",
        "database/plugins/",
        "database/slurm/",
    ]
    assert (
        data / "database" / "conf" / "neo4j.conf"
    ).read_text() == "server.memory.heap.max_size=8g\n"
    assert (data / "database" / ".env").read_text() == "NEO4J_AUTH=neo4j/x\n"
    assert capsys.readouterr().out == "Setup completed successfully.\n"


def test_env_file_argument_overrides_the_default(
    setup_mod: tuple[ModuleType, list[tuple[object, ...]]], tmp_path: Path
) -> None:
    mod, _ = setup_mod
    other = tmp_path / "primary.env"
    other.write_text("NEO4J_AUTH=neo4j/primary\n")
    mod.main(["--env-file", str(other)])
    assert (
        tmp_path / "data" / "database" / ".env"
    ).read_text() == "NEO4J_AUTH=neo4j/primary\n"


def test_a_missing_env_file_is_refused_after_the_tree_exists(
    setup_mod: tuple[ModuleType, list[tuple[object, ...]]], tmp_path: Path
) -> None:
    """The directories and neo4j.conf are written before the env copy fails, and the
    biocypher copy never happens.
    """
    mod, _ = setup_mod
    missing = tmp_path / "nope.env"
    with pytest.raises(
        FileNotFoundError, match=re.escape(f"No such file or directory: '{missing}'")
    ):
        mod.main(["--env-file", str(missing)])
    data = tmp_path / "data" / "database"
    assert (data / "conf" / "neo4j.conf").is_file()
    assert not (data / "biocypher" / "config").exists()
