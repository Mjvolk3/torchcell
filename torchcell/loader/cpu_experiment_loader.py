"""CPU-side prefetching batch loaders backed by worker threads/processes."""

# torchcell/loader/cpu_experiment_loader
# [[torchcell.loader.cpu_experiment_loader]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/loader/cpu_experiment_loader
# Test file: tests/torchcell/loader/test_cpu_experiment_loader.py

import queue
from collections.abc import Iterator, Sequence
from multiprocessing import Process, Queue
from typing import Any


class CpuExperimentLoaderMultiprocessing:
    """Process-based loader that prefetches dataset batches into a queue."""

    def __init__(
        self, dataset: Sequence[Any], batch_size: int, num_workers: int
    ) -> None:
        """Start worker processes and prime the queue with initial batch indices."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.load_queue: Queue[int | None] = Queue()
        self.data_queue: Queue[list[Any]] = Queue(maxsize=num_workers)
        self.workers: list[Process] = []

        # Calculate how many batches are needed
        self.total_batches = (len(dataset) + batch_size - 1) // batch_size

        # Flag to track if close has been called
        # make close idempotent
        self.is_closed = False

        for _ in range(num_workers):
            worker = Process(
                target=self.worker_function,
                args=(self.load_queue, self.data_queue, self.dataset, self.batch_size),
            )
            worker.start()
            self.workers.append(worker)

        # Initially, load the first few batches depending on the number of workers
        for i in range(min(self.total_batches, num_workers)):
            self.load_queue.put(i)  # Signal with unique batch index

    @staticmethod
    def worker_function(
        load_queue: "Queue[int | None]",
        data_queue: "Queue[list[Any]]",
        dataset: Sequence[Any],
        batch_size: int,
    ) -> None:
        """Pull batch indices, slice the dataset, and push batches until stopped."""
        while True:
            batch_index = load_queue.get()  # Get unique batch index
            if batch_index is None:  # If None signal received, terminate the worker
                break

            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = [dataset[i] for i in range(start_idx, end_idx)]
            data_queue.put(batch)

    def __iter__(self) -> Iterator[Any]:
        """Reset the batch counter and return self as the iterator."""
        self.batch_index = 0  # Reset batch index for new iteration
        return self

    def __next__(self) -> list[Any]:
        """Return the next prefetched batch, scheduling another, or stop iteration."""
        if self.batch_index >= self.total_batches:
            self.close()  # Ensure cleanup when iteration is complete
            raise StopIteration

        batch = self.data_queue.get()

        # Prepare next batch if there are more batches to process
        if self.batch_index + len(self.workers) < self.total_batches:
            self.load_queue.put(self.batch_index + len(self.workers))

        self.batch_index += 1
        return batch

    def __len__(self) -> int:
        """Return the number of batches the loader will yield."""
        # Calculate how many batches are needed
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def close(self) -> None:
        """Tear down workers once (idempotent); never block indefinitely.

        On the normal path workers are parked on ``load_queue.get()`` and exit on
        the ``None`` sentinel. On an error path the consumer may have abandoned
        iteration with ``data_queue`` full and workers blocked on ``put`` -- those
        never reach the sentinel, so a plain ``join()`` would hang forever. We
        drain ``data_queue`` to unblock them within a bounded budget, then
        force-terminate any straggler so ``close()`` itself can never deadlock.
        """
        if self.is_closed:
            return
        self.is_closed = True
        for _ in self.workers:
            self.load_queue.put(None)
        for _ in range(50):  # ~5s graceful budget (50 x 0.1s)
            if not any(worker.is_alive() for worker in self.workers):
                break
            try:
                while True:
                    self.data_queue.get_nowait()
            except queue.Empty:
                pass
            for worker in self.workers:
                worker.join(timeout=0.1)
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
