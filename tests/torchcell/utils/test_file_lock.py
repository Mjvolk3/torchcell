# tests/torchcell/utils/test_file_lock.py
# [[tests.torchcell.utils.test_file_lock]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/utils/test_file_lock.py
"""Tests for torchcell.utils.file_lock.

2026.09.30 (Phase 14). Added contention tests with a second lock holder in a thread
(``filelock`` locks are ``flock`` locks on an open file description, so two
``FileLock`` objects on one path exclude each other inside one process too), each
timed with ``time.monotonic``:

- a held lock blocks ``update_json_with_lock`` until the holder releases it at about
  0.3 s, and the update then sees the holder's write;
- read, write and update each give up after their ``timeout`` of 0.2 s with
  ``filelock.Timeout`` naming the ``.lock`` path and log the exact error line; the
  elapsed time is asserted in [0.2, 1.5) s, the upper bound generous for a loaded CI
  runner;
- Findings: ``timeout=0`` becomes the 60 s default (``timeout or default``), and
  ``retry_delay`` is never passed to ``filelock`` (a 5 s delay still acquires a lock
  released at 0.15 s in under 1 s);
- a stale ``.lock`` file with content and no holder does not block, and is truncated to
  empty while held; the context manager and the update path release on an exception;
- the exact lock and staging paths: ``<name><suffix>.lock`` and, as a Finding, the
  staging file ``<stem>.tmp``, which ``a.json`` and ``a.yaml`` share under two
  different locks; a failed stage leaves the target untouched;
- the exact written text (indent 2, non-ASCII kept), ``FileNotFoundError`` messages,
  and ``cleanup_lock_files`` recursing and counting only what it removed.
"""

import json
import logging
import multiprocessing as mp
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock, Timeout

from torchcell.utils import file_lock as file_lock_module
from torchcell.utils.file_lock import FileLockHelper


def test_read_write_json_basic():
    """Test basic read/write functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.json"
        test_data = {"key": "value", "number": 42}

        # Write data
        FileLockHelper.write_json_with_lock(file_path, test_data)

        # Read data
        read_data = FileLockHelper.read_json_with_lock(file_path)

        assert read_data == test_data


def test_create_if_missing():
    """Test creating file if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "new_file.json"
        default_data = {"default": "data"}

        # Read non-existent file with create_if_missing=True
        data = FileLockHelper.read_json_with_lock(
            file_path, create_if_missing=True, default_data=default_data
        )

        assert data == default_data
        assert file_path.exists()


def test_file_not_found():
    """Test FileNotFoundError when file doesn't exist and create_if_missing=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "non_existent.json"

        with pytest.raises(FileNotFoundError):
            FileLockHelper.read_json_with_lock(file_path)


def test_update_json():
    """Test atomic JSON update functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "update_test.json"
        initial_data = {"counter": 0, "items": []}

        # Create initial file
        FileLockHelper.write_json_with_lock(file_path, initial_data)

        # Update function
        def increment_counter(data):
            data["counter"] += 1
            data["items"].append(f"item_{data['counter']}")
            return data

        # Update the file
        updated_data = FileLockHelper.update_json_with_lock(
            file_path, increment_counter
        )

        assert updated_data["counter"] == 1
        assert updated_data["items"] == ["item_1"]

        # Verify file contents
        read_data = FileLockHelper.read_json_with_lock(file_path)
        assert read_data == updated_data


def _concurrent_writer(file_path, process_id, num_writes):
    """Helper function for concurrent write test."""
    for i in range(num_writes):

        def update_func(data):
            if "writes" not in data:
                data["writes"] = []
            data["writes"].append(f"process_{process_id}_write_{i}")
            return data

        FileLockHelper.update_json_with_lock(
            file_path, update_func, create_if_missing=True, default_data={}
        )
        time.sleep(0.01)  # Small delay to increase chance of contention


def test_concurrent_writes():
    """Test that concurrent writes are properly serialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "concurrent_test.json"
        num_processes = 4
        num_writes_per_process = 5

        # Create processes
        processes = []
        for i in range(num_processes):
            p = mp.Process(
                target=_concurrent_writer,
                args=(str(file_path), i, num_writes_per_process),
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Read final result
        data = FileLockHelper.read_json_with_lock(file_path)

        # Verify all writes were recorded
        assert "writes" in data
        assert len(data["writes"]) == num_processes * num_writes_per_process

        # Verify no writes were lost
        for i in range(num_processes):
            for j in range(num_writes_per_process):
                expected_write = f"process_{i}_write_{j}"
                assert expected_write in data["writes"]


def test_cleanup_lock_files():
    """cleanup_lock_files removes orphaned .lock files and leaves data files intact.

    The precondition (orphan ``.lock`` sidecars) is constructed explicitly rather
    than relying on ``write_json_with_lock`` leaving lock files behind: filelock
    >= 3.21.0 (2026-02-12, py-filelock changelog "delete lock file on release")
    removes the lock file on release on Unix, so that assumption is version- and
    OS-dependent. This asserts what ``cleanup_lock_files`` actually promises --
    removing orphaned ``*.lock`` files while leaving ``*.json`` data untouched.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Data files plus explicit orphan lock sidecars. The ".json.lock" naming
        # matches FileLockHelper._get_lock_path ("<name>.json" -> "<name>.json.lock").
        for i in range(3):
            FileLockHelper.write_json_with_lock(tmp_path / f"file_{i}.json", {"id": i})
            (tmp_path / f"file_{i}.json.lock").touch()

        # Precondition: three orphan lock files exist regardless of filelock version.
        assert len(list(tmp_path.glob("*.lock"))) == 3

        # Clean up lock files.
        removed_count = FileLockHelper.cleanup_lock_files(tmp_path)
        assert removed_count == 3

        # Lock files removed; data files untouched.
        assert len(list(tmp_path.glob("*.lock"))) == 0
        assert len(list(tmp_path.glob("*.json"))) == 3


def test_nested_directories():
    """Test that parent directories are created as needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "nested" / "deep" / "test.json"
        test_data = {"nested": True}

        # Write to nested path
        FileLockHelper.write_json_with_lock(file_path, test_data)

        # Verify file and directories were created
        assert file_path.exists()
        assert file_path.parent.exists()

        # Read back data
        read_data = FileLockHelper.read_json_with_lock(file_path)
        assert read_data == test_data


# --- contention, timeouts and exact paths (Phase 14) ---------------------------- #


@contextmanager
def _held_by_another_thread(lock_path: Path, hold_s: float) -> Iterator[list[float]]:
    """Hold ``lock_path`` in a thread for up to ``hold_s`` seconds; yield once held.

    The holder releases after ``hold_s`` or as soon as the context exits, whichever is
    first; the yielded list receives the monotonic release time. The exit joins the
    thread, so a test never leaks a held lock.
    """
    acquired = threading.Event()
    done = threading.Event()
    released_at: list[float] = []

    def hold() -> None:
        lock = FileLock(lock_path)
        with lock.acquire(timeout=5):
            acquired.set()
            done.wait(timeout=hold_s)
            released_at.append(time.monotonic())

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(timeout=5)
    try:
        yield released_at
    finally:
        done.set()
        thread.join(timeout=10)


def test_lock_path_appends_lock_to_the_full_suffix(tmp_path: Path) -> None:
    """``<name><suffix>.lock``: the whole suffix is kept, and a suffixless name just
    gains ``.lock``.
    """
    assert (
        FileLockHelper._get_lock_path(tmp_path / "a.json") == tmp_path / "a.json.lock"
    )
    assert (
        FileLockHelper._get_lock_path(tmp_path / "x.tar.gz")
        == tmp_path / "x.tar.gz.lock"
    )
    assert FileLockHelper._get_lock_path(str(tmp_path / "noext")) == (
        tmp_path / "noext.lock"
    )


def test_held_lock_blocks_update_until_release(tmp_path: Path) -> None:
    """The file is set to ``{"n": 1}`` while another thread holds the lock, which it
    releases at about 0.3 s; the update, started while the lock is held, cannot run
    before the release and then reads that value, so it returns ``{"n": 2}``. Tolerance: the update must
    finish at or after the recorded release time and within 3 s of starting.
    """
    path = tmp_path / "counter.json"
    lock_path = tmp_path / "counter.json.lock"
    ran_at: list[float] = []

    def bump(data: dict[str, int]) -> dict[str, int]:
        ran_at.append(time.monotonic())
        return {"n": data["n"] + 1}

    with _held_by_another_thread(lock_path, hold_s=0.3) as released_at:
        path.write_text(json.dumps({"n": 1}))
        start = time.monotonic()
        result = FileLockHelper.update_json_with_lock(path, bump, timeout=5)
    assert result == {"n": 2}
    assert json.loads(path.read_text()) == {"n": 2}
    assert ran_at[0] >= released_at[0]
    assert ran_at[0] - start < 3.0


@pytest.mark.parametrize(
    ("verb", "call"),
    [
        ("reading", lambda p: FileLockHelper.read_json_with_lock(p, timeout=0.2)),
        (
            "writing",
            lambda p: FileLockHelper.write_json_with_lock(p, {"a": 1}, timeout=0.2),
        ),
        (
            "updating",
            lambda p: FileLockHelper.update_json_with_lock(p, dict, timeout=0.2),
        ),
    ],
)
def test_timeout_raises_after_the_timeout_and_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, verb: str, call: Any
) -> None:
    """With the lock held (for up to 5 s), each operation gives up after its 0.2 s timeout:
    ``filelock.Timeout`` whose message names the ``.lock`` path, one error record
    ``Failed to acquire lock for <verb> <path> within 0.2s``, and the data file left
    as it was. Elapsed is asserted in [0.2, 1.5) s.
    """
    path = tmp_path / "data.json"
    path.write_text('{"kept": true}')
    lock_path = tmp_path / "data.json.lock"
    caplog.set_level(logging.ERROR, logger=file_lock_module.__name__)
    with _held_by_another_thread(lock_path, hold_s=5.0):
        start = time.monotonic()
        with pytest.raises(Timeout) as excinfo:
            call(path)
        elapsed = time.monotonic() - start
    assert 0.2 <= elapsed < 1.5
    assert str(excinfo.value) == f"The file lock '{lock_path}' could not be acquired."
    assert excinfo.value.lock_file == str(lock_path)
    assert [r.getMessage() for r in caplog.records] == [
        f"Failed to acquire lock for {verb} {path} within 0.2s"
    ]
    assert path.read_text() == '{"kept": true}'


def test_zero_timeout_becomes_the_sixty_second_default(tmp_path: Path) -> None:
    """Finding: every method resolves ``timeout = timeout or cls.default_timeout``, so a
    caller asking for 0 (try once, do not wait) gets 60 s; ``with_file_lock`` exposes
    the resolved value on the returned lock. An explicit 2.5 is kept. Pinned until the
    default is applied only to ``None``.
    """
    lock = FileLockHelper.with_file_lock(tmp_path / "a.json", timeout=0)
    assert lock.timeout == 60.0
    assert lock.lock_file == str(tmp_path / "a.json.lock")
    assert (
        FileLockHelper.with_file_lock(tmp_path / "a.json", timeout=2.5).timeout == 2.5
    )
    assert FileLockHelper.with_file_lock(tmp_path / "a.json").timeout == 60.0


def test_retry_delay_is_not_the_polling_interval(tmp_path: Path) -> None:
    """Finding: ``retry_delay`` ("Delay between lock acquisition retries") is resolved
    and never passed to ``filelock``, which polls at its own 0.05 s default. The holder
    releases at about 0.15 s; with ``retry_delay=5.0`` the read still returns in under
    1 s instead of after the first 5 s retry. Pinned until the argument is forwarded as
    ``poll_interval`` or removed.
    """
    path = tmp_path / "data.json"
    path.write_text('{"a": 1}')
    with _held_by_another_thread(tmp_path / "data.json.lock", hold_s=0.15):
        start = time.monotonic()
        data = FileLockHelper.read_json_with_lock(path, timeout=5, retry_delay=5.0)
        elapsed = time.monotonic() - start
    assert data == {"a": 1}
    assert elapsed < 1.0


def test_stale_lock_file_does_not_block_and_is_truncated_while_held(
    tmp_path: Path,
) -> None:
    """A ``.lock`` file left with content by a process that no longer holds it is not
    a lock: the update acquires within its 0.5 s timeout, and while it holds the lock
    the file exists and is empty (opened with ``O_TRUNC``).
    """
    path = tmp_path / "data.json"
    lock_path = tmp_path / "data.json.lock"
    lock_path.write_text("12345\n")
    seen: list[str] = []

    def peek(data: dict[str, Any]) -> dict[str, Any]:
        seen.append(lock_path.read_text())
        return {**data, "seen": True}

    result = FileLockHelper.update_json_with_lock(path, peek, timeout=0.5)
    assert result == {"seen": True}
    assert seen == [""]


def test_with_file_lock_releases_on_an_exception(tmp_path: Path) -> None:
    """Inside the ``with`` block a second lock object cannot take the lock without
    blocking; after a ``ValueError`` escapes the block the lock is released and the
    second object takes it at once.
    """
    target = tmp_path / "notes.txt"
    other = FileLock(tmp_path / "notes.txt.lock")
    with pytest.raises(ValueError, match="^boom$"):
        with FileLockHelper.with_file_lock(target) as lock:
            assert lock.is_locked
            with pytest.raises(Timeout):
                other.acquire(blocking=False)
            raise ValueError("boom")
    assert not lock.is_locked
    other.acquire(blocking=False)
    assert other.is_locked
    other.release()


def test_update_releases_and_leaves_the_file_when_the_function_raises(
    tmp_path: Path,
) -> None:
    """An ``update_func`` that raises propagates its error; the file keeps its bytes,
    no staging file is left, and the lock is free for the next caller.
    """
    path = tmp_path / "data.json"
    path.write_text('{"a": 1}')

    def explode(data: dict[str, int]) -> dict[str, int]:
        raise KeyError("missing")

    with pytest.raises(KeyError, match="missing"):
        FileLockHelper.update_json_with_lock(path, explode)
    assert path.read_text() == '{"a": 1}'
    assert not (tmp_path / "data.tmp").exists()
    other = FileLock(tmp_path / "data.json.lock")
    other.acquire(blocking=False)
    other.release()


def test_staging_file_is_the_stem_tmp_and_shared_across_suffixes(
    tmp_path: Path,
) -> None:
    """Finding: writes stage through ``file_path.with_suffix(".tmp")``, so ``a.json``
    and ``a.yaml`` both stage through ``a.tmp`` while holding different locks
    (``a.json.lock``, ``a.yaml.lock``); two concurrent writers of the two files can
    overwrite each other's staging bytes. Shown by making ``a.tmp`` a directory: both
    writes fail naming ``a.tmp``, and each target keeps its previous bytes. Pinned
    until the staging name is derived from the full file name.
    """
    staging = tmp_path / "a.tmp"
    staging.mkdir()
    for name in ("a.json", "a.yaml"):
        target = tmp_path / name
        target.write_text("previous")
        with pytest.raises(
            IsADirectoryError,
            match=re.escape(f"[Errno 21] Is a directory: '{staging}'"),
        ):
            FileLockHelper.write_json_with_lock(target, {"new": True})
        assert target.read_text() == "previous"


def test_written_text_is_indented_and_keeps_non_ascii(tmp_path: Path) -> None:
    r"""Defaults ``indent=2, ensure_ascii=False``: the file text is exact and the
    accented character is stored as itself, not ``\u00e9``; ``indent=0`` and
    ``ensure_ascii=True`` change both.
    """
    path = tmp_path / "out.json"
    FileLockHelper.write_json_with_lock(path, {"name": "caf\u00e9", "n": [1]})
    assert path.read_text(encoding="utf-8") == (
        '{\n  "name": "caf\u00e9",\n  "n": [\n    1\n  ]\n}'
    )
    FileLockHelper.write_json_with_lock(
        path, {"name": "caf\u00e9"}, indent=0, ensure_ascii=True
    )
    assert path.read_text() == '{\n"name": "caf\\u00e9"\n}'


def test_missing_file_messages_and_default_data(tmp_path: Path) -> None:
    """``read`` without ``create_if_missing`` and ``update`` with
    ``create_if_missing=False`` both raise ``File not found: <path>`` and create
    nothing; ``update`` with a missing parent directory creates it, passes the given
    ``default_data`` to the function and writes the result; ``read`` with no
    ``default_data`` creates the file as ``{}``.
    """
    missing = tmp_path / "missing.json"
    with pytest.raises(
        FileNotFoundError, match=re.escape(f"File not found: {missing}")
    ):
        FileLockHelper.read_json_with_lock(missing)
    with pytest.raises(
        FileNotFoundError, match=re.escape(f"File not found: {missing}")
    ):
        FileLockHelper.update_json_with_lock(missing, dict, create_if_missing=False)
    assert not missing.exists()

    nested = tmp_path / "new" / "dir" / "list.json"
    result = FileLockHelper.update_json_with_lock(
        nested, lambda data: [*data, 2], default_data=[1]
    )
    assert result == [1, 2]
    assert json.loads(nested.read_text()) == [1, 2]

    created = tmp_path / "created.json"
    assert FileLockHelper.read_json_with_lock(created, create_if_missing=True) == {}
    assert created.read_text() == "{}"


def test_cleanup_recurses_and_counts_only_what_it_removed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Two lock files (one nested two levels down) are removed; a directory named
    ``stuck.lock`` cannot be unlinked, is logged as a warning with the OS error, stays,
    and is not counted, so the return is 2. A ``.json`` data file is untouched.
    """
    (tmp_path / "top.json.lock").write_text("")
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    (tmp_path / "sub" / "deeper" / "x.json.lock").write_text("")
    stuck = tmp_path / "stuck.lock"
    stuck.mkdir()
    (tmp_path / "keep.json").write_text("{}")
    caplog.set_level(logging.WARNING, logger=file_lock_module.__name__)
    assert FileLockHelper.cleanup_lock_files(tmp_path) == 2
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == [
        "keep.json",
        "stuck.lock",
        "sub",
        "sub/deeper",
    ]
    assert [r.getMessage() for r in caplog.records] == [
        f"Failed to remove lock file {stuck}: [Errno 21] Is a directory: '{stuck}'"
    ]


if __name__ == "__main__":
    # Run basic tests
    test_read_write_json_basic()
    test_create_if_missing()
    test_update_json()
    test_concurrent_writes()
    test_cleanup_lock_files()
    test_nested_directories()
    print("All tests passed!")
