# tests/torchcell/sequence/test_db_connection.py
# [[tests.torchcell.sequence.test_db_connection]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/test_db_connection.py
"""The per-thread database connection manager, with a recording stand-in database.

Fixture: ``_FakeDB`` (module level, so pickle can name it) records the positional and
keyword arguments it was opened with and counts instances. The database "file" is an
empty file under ``tmp_path`` (the manager only checks that it exists). No gffutils
database is opened.

Contract pinned: one connection per thread, created lazily and reused; ``close`` drops
it so the next call reopens; a missing file refuses with the exact message; a pickle
round trip carries the configuration and never a live connection. Finding: keyword
constructor arguments do not survive pickling.
"""

from __future__ import annotations

import pickle
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from torchcell.sequence.db_connection import DatabaseConnectionManager


class _FakeDB:
    opened = 0

    def __init__(self, path: str, *args: Any, **kwargs: Any) -> None:
        type(self).opened += 1
        self.path = path
        self.args = args
        self.kwargs = kwargs


@pytest.fixture
def db_file(tmp_path: Path) -> str:
    path = tmp_path / "genome.db"
    path.write_bytes(b"")
    _FakeDB.opened = 0
    return str(path)


def test_connection_is_lazy_cached_and_reopened_after_close(db_file: str) -> None:
    mgr = DatabaseConnectionManager(db_file, _FakeDB, "pos", keep_order=True)  # type: ignore[type-var, unused-ignore]
    assert _FakeDB.opened == 0
    first = mgr.get_connection()
    assert (first.path, first.args, first.kwargs) == (
        db_file,
        ("pos",),
        {"keep_order": True},
    )
    assert mgr.get_connection() is first
    assert _FakeDB.opened == 1
    mgr.close_connection()
    assert mgr._local.db is None
    second = mgr.get_connection()
    assert second is not first
    assert _FakeDB.opened == 2


def test_close_without_a_connection_is_a_no_op(db_file: str) -> None:
    mgr = DatabaseConnectionManager(db_file, _FakeDB)  # type: ignore[type-var, unused-ignore]
    mgr.close_connection()
    assert not hasattr(mgr._local, "db")


def test_missing_database_is_refused(tmp_path: Path) -> None:
    path = str(tmp_path / "absent.db")
    mgr = DatabaseConnectionManager(path, _FakeDB)  # type: ignore[type-var, unused-ignore]
    with pytest.raises(
        FileNotFoundError, match=re.escape(f"Database not found at {path}")
    ):
        mgr.get_connection()


def test_each_thread_gets_its_own_connection(db_file: str) -> None:
    mgr = DatabaseConnectionManager(db_file, _FakeDB)  # type: ignore[type-var, unused-ignore]
    main = mgr.get_connection()
    seen: list[Any] = []
    worker = threading.Thread(target=lambda: seen.append(mgr.get_connection()))
    worker.start()
    worker.join()
    assert seen[0] is not main
    assert _FakeDB.opened == 2
    assert mgr.get_connection() is main


def test_pickle_carries_configuration_and_no_connection(db_file: str) -> None:
    mgr = DatabaseConnectionManager(db_file, _FakeDB, "a", "b")  # type: ignore[type-var, unused-ignore]
    mgr.get_connection()
    clone = pickle.loads(pickle.dumps(mgr))
    assert (clone.db_path, clone.db_class, clone.db_args, clone.db_kwargs) == (
        db_file,
        _FakeDB,
        ("a", "b"),
        {},
    )
    assert not hasattr(clone._local, "db")
    conn = clone.get_connection()
    assert conn.args == ("a", "b")
    assert _FakeDB.opened == 2


def test_pickle_turns_keyword_arguments_into_attributes(db_file: str) -> None:
    """Finding: ``__reduce__`` returns ``(cls, (path, cls, *args), db_kwargs)``; pickle
    treats the third element as STATE and hands it to ``__setstate__``, so after a round
    trip ``db_kwargs`` is ``{}``, each keyword is set as an attribute of the manager,
    and the worker's connection is opened WITHOUT it. Latent: every live construction
    (``GffutilsConnectionManager(path)`` in s288c.py) passes no keyword. Pinned until
    ``__reduce__`` rebuilds with the kwargs (db_connection.py:116-120).
    """
    mgr = DatabaseConnectionManager(db_file, _FakeDB, keep_order=True)  # type: ignore[type-var, unused-ignore]
    clone = pickle.loads(pickle.dumps(mgr))
    assert clone.db_kwargs == {}
    assert clone.keep_order is True
    assert clone.get_connection().kwargs == {}
    assert mgr.get_connection().kwargs == {"keep_order": True}


def test_getstate_drops_the_thread_local_and_setstate_restores_one(
    db_file: str,
) -> None:
    mgr = DatabaseConnectionManager(db_file, _FakeDB, "x", k=1)  # type: ignore[type-var, unused-ignore]
    mgr.get_connection()
    state = mgr.__getstate__()
    assert state == {
        "db_path": db_file,
        "db_class": _FakeDB,
        "db_args": ("x",),
        "db_kwargs": {"k": 1},
    }
    fresh = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
    fresh.__setstate__(state)
    assert isinstance(fresh._local, threading.local)
    assert fresh.get_connection().kwargs == {"k": 1}
