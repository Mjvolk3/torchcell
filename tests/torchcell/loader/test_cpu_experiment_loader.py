"""Teardown/anti-hang tests for CpuExperimentLoaderMultiprocessing.

The loader spawns worker processes that feed a bounded ``data_queue``. If
``close()`` blocked (e.g. a plain ``join()`` while a worker is stuck on a full
queue), an error in the consumer would surface as a silent hang rather than a
clean crash -- the failure mode these tests guard against.
"""

from __future__ import annotations

import queue
import threading

from torchcell.loader import CpuExperimentLoaderMultiprocessing


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


def test_worker_function_drops_inherited_env() -> None:
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
