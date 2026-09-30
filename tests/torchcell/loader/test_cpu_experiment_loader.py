# tests/torchcell/loader/test_cpu_experiment_loader.py
# [[tests.torchcell.loader.test_cpu_experiment_loader]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/loader/test_cpu_experiment_loader.py
"""Teardown/anti-hang tests for CpuExperimentLoaderMultiprocessing.

The loader spawns worker processes that feed a bounded ``data_queue``. If
``close()`` blocked (e.g. a plain ``join()`` while a worker is stuck on a full
queue), an error in the consumer would surface as a silent hang rather than a
clean crash -- the failure mode these tests guard against.

2026.09.30, Phase 16: the batch arithmetic and the teardown budget on hand-built inputs.
``worker_function`` runs in-process with thread queues and ``_die_with_parent`` stubbed
(calling the real one would arm PR_SET_PDEATHSIG on the pytest process): batch ``i`` of a
7-item list at batch size 3 is ``[3i, 3i + 1, 3i + 2]`` cut at 7, so indices 1, 0, 2 give
``[3, 4, 5]``, ``[0, 1, 2]``, ``[6]``. ``len`` is ``ceil(n / batch_size)``: 10 items at 4
is 3, 8 at 4 is 2, 0 at 4 is 0. With one real worker, 5 items at batch size 2 arrive in
order ``[0, 1], [2, 3], [4]``. ``close`` puts one ``None`` per worker, then loops at most
50 times, each iteration draining ``data_queue`` and joining every worker with timeout
0.1; a worker alive after the 50th join is terminated and joined without a timeout.
``_die_with_parent`` calls ``prctl(1, SIGKILL=9, 0, 0, 0)`` and exits with status 0 only
when the parent pid is already 1.
"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
from typing import Any

import pytest

from torchcell.loader import CpuExperimentLoaderMultiprocessing
from torchcell.loader import cpu_experiment_loader as cel


def _finishes_within(fn, timeout=20.0):
    """Run ``fn()`` in a daemon thread; True iff it returns within ``timeout``."""
    done = threading.Event()

    def target():
        fn()
        done.set()

    threading.Thread(target=target, daemon=True).start()
    done.wait(timeout)
    return done.is_set()


def test_iterates_all_items_then_closes():
    loader = CpuExperimentLoaderMultiprocessing(
        list(range(20)), batch_size=4, num_workers=2
    )
    seen: list[int] = []
    for batch in loader:
        seen.extend(batch)
    assert sorted(seen) == list(range(20))
    assert loader.is_closed
    assert not any(worker.is_alive() for worker in loader.workers)


def test_close_after_partial_consume_is_bounded():
    # Abandon iteration with batches still buffered in data_queue: close() must
    # drain + terminate rather than block forever on a plain join().
    loader = CpuExperimentLoaderMultiprocessing(
        list(range(200)), batch_size=1, num_workers=2
    )
    next(iter(loader))  # consume one batch, leave the rest in flight
    assert _finishes_within(loader.close), "close() hung (teardown regression)"
    assert not any(worker.is_alive() for worker in loader.workers)


def test_close_is_idempotent():
    loader = CpuExperimentLoaderMultiprocessing(
        list(range(20)), batch_size=4, num_workers=2
    )
    assert _finishes_within(loader.close)
    assert loader.is_closed
    loader.close()  # second call must be a harmless no-op
    assert not any(worker.is_alive() for worker in loader.workers)


def test_worker_function_drops_inherited_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A forked worker must drop any LMDB ``env`` inherited from the parent.

    Sharing an LMDB environment across ``fork()`` makes a read in the child raise
    ``MDB_MAP_RESIZED`` once the file has grown past the inherited mapping. The
    worker resets ``dataset.env`` to None before its loop so it lazily opens its
    OWN env. Here the queue carries an immediate stop sentinel, so the reset is
    the only observable effect (the dataset is never indexed).
    """

    class _FakeDataset:
        def __init__(self) -> None:
            self.env: object | None = object()  # env handle inherited via fork

        def __len__(self) -> int:
            return 0

        def __getitem__(self, i: int) -> object:  # pragma: no cover - not reached
            raise AssertionError("worker should stop before indexing the dataset")

    monkeypatch.setattr(cel, "_die_with_parent", lambda: None)
    dataset = _FakeDataset()
    load_q: queue.Queue[int | None] = queue.Queue()
    data_q: queue.Queue[list[object]] = queue.Queue()
    load_q.put(None)  # stop the worker immediately, after the env reset
    CpuExperimentLoaderMultiprocessing.worker_function(
        load_q,  # type: ignore[arg-type]  # thread Queue stands in for mp Queue
        data_q,  # type: ignore[arg-type]
        dataset,  # type: ignore[arg-type]  # minimal dataset stub, not a Sequence
        batch_size=1,
    )
    assert dataset.env is None


def test_worker_function_slices_each_requested_batch_in_request_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Indices 1, 0, 2 over ``range(7)`` at batch size 3 give ``[3, 4, 5]``, ``[0, 1, 2]``
    and the short tail ``[6]``; a plain list has no ``env`` and is left alone.
    """
    armed: list[bool] = []
    monkeypatch.setattr(cel, "_die_with_parent", lambda: armed.append(True))
    load_q: queue.Queue[int | None] = queue.Queue()
    data_q: queue.Queue[list[int]] = queue.Queue()
    for item in (1, 0, 2, None):
        load_q.put(item)
    CpuExperimentLoaderMultiprocessing.worker_function(
        load_q,  # type: ignore[arg-type]  # thread Queue stands in for mp Queue
        data_q,  # type: ignore[arg-type]
        list(range(7)),
        batch_size=3,
    )
    assert armed == [True]
    assert [data_q.get_nowait() for _ in range(3)] == [[3, 4, 5], [0, 1, 2], [6]]
    assert data_q.empty()


class _FakeLibc:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def prctl(self, *args: int) -> int:
        self.calls.append(args)
        return 0


@pytest.mark.parametrize(("ppid", "exits"), [(4242, []), (1, [0])])
def test_die_with_parent_arms_pdeathsig_and_exits_only_when_orphaned(  # test-quality: allow the libc, getppid and _exit recorders carry the assertions
    monkeypatch: pytest.MonkeyPatch, ppid: int, exits: list[int]
) -> None:
    """``prctl(PR_SET_PDEATHSIG=1, SIGKILL, 0, 0, 0)`` is always requested from
    ``libc.so.6``; ``os._exit(0)`` follows only when the parent is already init (pid 1).
    """
    libc = _FakeLibc()
    opened: list[tuple[str, bool]] = []

    def fake_cdll(name: str, use_errno: bool = False) -> _FakeLibc:
        opened.append((name, use_errno))
        return libc

    exit_codes: list[int] = []
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(os, "getppid", lambda: ppid)
    monkeypatch.setattr(os, "_exit", exit_codes.append)
    cel._die_with_parent()
    assert opened == [("libc.so.6", True)]
    assert libc.calls == [(1, 9, 0, 0, 0)]
    assert exit_codes == exits


@pytest.mark.parametrize(
    ("n", "batch_size", "batches"), [(10, 4, 3), (8, 4, 2), (0, 4, 0)]
)
def test_len_is_the_ceiling_of_items_over_batch_size(
    n: int, batch_size: int, batches: int
) -> None:
    """``len`` and ``total_batches`` are both ``ceil(n / batch_size)``; no workers start."""
    loader = CpuExperimentLoaderMultiprocessing(
        list(range(n)), batch_size=batch_size, num_workers=0
    )
    assert len(loader) == batches
    assert loader.total_batches == batches
    loader.close()


def test_one_worker_delivers_batches_in_dataset_order() -> None:
    """With a single worker the queue is FIFO: ``[0, 1], [2, 3], [4]``, then closed."""
    loader = CpuExperimentLoaderMultiprocessing(
        list(range(5)), batch_size=2, num_workers=1
    )
    assert list(loader) == [[0, 1], [2, 3], [4]]
    assert loader.is_closed


def test_an_empty_dataset_yields_nothing_and_closes() -> None:
    """Zero batches: the first ``next`` closes the loader and stops."""
    loader = CpuExperimentLoaderMultiprocessing([], batch_size=3, num_workers=1)
    assert list(loader) == []
    assert loader.is_closed
    assert not any(worker.is_alive() for worker in loader.workers)


class _FakeWorker:
    """A process stand-in that stays alive until it has been joined ``dies_after`` times."""

    def __init__(self, dies_after: int | None) -> None:
        self.dies_after = dies_after
        self.joins: list[float | None] = []
        self.terminated = 0

    def is_alive(self) -> bool:
        if self.terminated:
            return False
        return self.dies_after is None or len(self.joins) < self.dies_after

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)

    def terminate(self) -> None:
        self.terminated += 1


def _closing_loader(worker: _FakeWorker) -> tuple[Any, queue.Queue[Any]]:
    loader = CpuExperimentLoaderMultiprocessing([], batch_size=1, num_workers=0)
    load_q: queue.Queue[Any] = queue.Queue()
    data_q: queue.Queue[Any] = queue.Queue()
    data_q.put([1])
    data_q.put([2])
    loader.workers = [worker]  # type: ignore[list-item]  # recorder stands in for Process
    loader.load_queue = load_q  # type: ignore[assignment]  # thread Queue stands in
    loader.data_queue = data_q  # type: ignore[assignment]
    return loader, load_q


def test_close_terminates_a_worker_still_alive_after_fifty_joins() -> None:
    """A worker that never exits is joined 50 times at 0.1 s, then terminated once and
    joined without a timeout; the stop sentinel was sent and the buffered batches drained.
    """
    worker = _FakeWorker(dies_after=None)
    loader, load_q = _closing_loader(worker)
    loader.close()
    assert worker.joins == [0.1] * 50 + [None]
    assert worker.terminated == 1
    assert load_q.get_nowait() is None
    assert load_q.empty()
    assert loader.data_queue.empty()


def test_close_stops_joining_once_the_worker_exits() -> None:
    """A worker that exits after its third join is never terminated: three joins, no more."""
    worker = _FakeWorker(dies_after=3)
    loader, _ = _closing_loader(worker)
    loader.close()
    assert worker.joins == [0.1, 0.1, 0.1]
    assert worker.terminated == 0
