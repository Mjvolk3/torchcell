"""Teardown/anti-hang tests for CpuExperimentLoaderMultiprocessing.

The loader spawns worker processes that feed a bounded ``data_queue``. If
``close()`` blocked (e.g. a plain ``join()`` while a worker is stuck on a full
queue), an error in the consumer would surface as a silent hang rather than a
clean crash -- the failure mode these tests guard against.
"""

from __future__ import annotations

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
