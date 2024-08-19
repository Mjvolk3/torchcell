# torchcell/loader/cpu_experiment_loader
# [[torchcell.loader.cpu_experiment_loader]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/loader/cpu_experiment_loader
# Test file: tests/torchcell/loader/test_cpu_experiment_loader.py


from threading import Thread
from queue import Queue
from typing import Iterable
from multiprocessing import Process, Queue
import json
import torch
from typing import Optional
import os

class CpuExperimentLoader:
    def __init__(self, dataset, batch_size: int, num_workers: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.load_queue = Queue()
        self.data_queue = Queue(maxsize=num_workers)
        self.workers = []

        # Calculate how many batches are needed
        self.total_batches = (len(dataset) + batch_size - 1) // batch_size

        # Flag to track if close has been called
        # make close idempotent
        self.is_closed = False

        for _ in range(num_workers):
            worker = Thread(
                target=self.worker_function,
                args=(self.load_queue, self.data_queue, self.dataset, self.batch_size),
            )
            worker.start()
            self.workers.append(worker)

        # Initially, load the first few batches depending on the number of workers
        for i in range(min(self.total_batches, num_workers)):
            self.load_queue.put(i)  # Signal with unique batch index

    @staticmethod
    def worker_function(load_queue: Queue, data_queue: Queue, dataset, batch_size: int):
        while True:
            batch_index = load_queue.get()  # Get unique batch index
            if batch_index is None:  # If None signal received, terminate the worker
                break

            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = [dataset[i] for i in range(start_idx, end_idx)]
            data_queue.put(batch)

    def __iter__(self) -> Iterable:
        self.batch_index = 0  # Reset batch index for new iteration
        return self

    def __next__(self):
        if self.batch_index >= self.total_batches:
            self.close()  # Ensure cleanup when iteration is complete
            raise StopIteration

        batch = self.data_queue.get()

        # Prepare next batch if there are more batches to process
        if self.batch_index + len(self.workers) < self.total_batches:
            self.load_queue.put(self.batch_index + len(self.workers))

        self.batch_index += 1
        return batch

    def close(self):
        if not self.is_closed:
            # Send termination signal to each worker
            for _ in self.workers:
                self.load_queue.put(None)
            for worker in self.workers:
                worker.join()  # Wait for all workers to finish
            self.is_closed = True  # Set the flag to prevent repeated cleanup


class CpuExperimentLoaderMultiprocessing:
    def __init__(self, dataset, batch_size: int, num_workers: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.load_queue = Queue()
        self.data_queue = Queue(maxsize=num_workers)
        self.workers = []

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
    def worker_function(load_queue: Queue, data_queue: Queue, dataset, batch_size: int):
        while True:
            batch_index = load_queue.get()  # Get unique batch index
            if batch_index is None:  # If None signal received, terminate the worker
                break

            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = [dataset[i] for i in range(start_idx, end_idx)]
            data_queue.put(batch)

    def __iter__(self) -> Iterable:
        self.batch_index = 0  # Reset batch index for new iteration
        return self

    def __next__(self):
        if self.batch_index >= self.total_batches:
            self.close()  # Ensure cleanup when iteration is complete
            raise StopIteration

        batch = self.data_queue.get()

        # Prepare next batch if there are more batches to process
        if self.batch_index + len(self.workers) < self.total_batches:
            self.load_queue.put(self.batch_index + len(self.workers))

        self.batch_index += 1
        return batch

    def __len__(self):
        # Calculate how many batches are needed
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def close(self):
        if not self.is_closed:
            # Send termination signal to each worker
            for _ in self.workers:
                self.load_queue.put(None)
            for worker in self.workers:
                worker.join()  # Wait for all workers to finish
            self.is_closed = True  # Set the flag to prevent repeated cleanup


class CpuDataModule:
    def __init__(
        self,
        dataset,
        cache_dir: str = "cache",
        batch_size: int = 32,
        random_seed: int = 42,
        num_workers: int = 0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        self.setup()

    def setup(self):
        torch.manual_seed(self.random_seed)
        os.makedirs(self.cache_dir, exist_ok=True)

        cached_indices_file = os.path.join(self.cache_dir, "cached_indices.json")
        if os.path.exists(cached_indices_file):
            with open(cached_indices_file, "r") as f:
                cached_data = json.load(f)
                self.train_indices = cached_data["train_indices"]
                self.val_indices = cached_data["val_indices"]
                self.test_indices = cached_data["test_indices"]
        else:
            self._generate_indices()
            cached_data = {
                "train_indices": self.train_indices,
                "val_indices": self.val_indices,
                "test_indices": self.test_indices,
            }
            with open(cached_indices_file, "w") as f:
                json.dump(cached_data, f)

    def _generate_indices(self):
        total_samples = len(self.dataset)
        torch.manual_seed(self.random_seed)
        shuffled_indices = torch.randperm(total_samples).tolist()

        num_train = int(self.train_ratio * total_samples)
        num_val = int(self.val_ratio * total_samples)

        self.train_indices = shuffled_indices[:num_train]
        self.val_indices = shuffled_indices[num_train : num_train + num_val]
        self.test_indices = shuffled_indices[num_train + num_val :]

    def _create_loader(self, indices: Optional[list[int]] = None):
        if indices is not None:
            subset = torch.utils.data.Subset(self.dataset, indices)
        else:
            subset = self.dataset

        return CpuExperimentLoaderMultiprocessing(
            dataset=subset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def train_dataloader(self):
        return self._create_loader(self.train_indices)

    def val_dataloader(self):
        return self._create_loader(self.val_indices)

    def test_dataloader(self):
        return self._create_loader(self.test_indices)

    def all_dataloader(self):
        return self._create_loader()
