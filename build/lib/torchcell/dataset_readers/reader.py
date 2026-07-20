# torchcell/dataset_readers/reader.py
# [[torchcell.dataset_readers.reader]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/dataset_readers/reader.py
# Test file: tests/torchcell/dataset_readers/test_reader.py
"""Read-only access to a processed dataset stored in an LMDB environment."""

import json
import os.path as osp
import pickle
from collections.abc import Iterator
from typing import Any

import lmdb
import numpy as np

from torchcell.data.data import ExperimentReferenceIndex


class LmdbDatasetReader:
    """Read items and the experiment reference index from an LMDB dataset."""

    def __init__(self, dataset_dir: str) -> None:
        """Open the LMDB environment and load the experiment reference index."""
        self.dataset_dir = dataset_dir
        self.env: lmdb.Environment = None
        self._experiment_reference_index: list[ExperimentReferenceIndex] | None = None
        self.db: object = None  # Add a db attribute
        self._init_db()
        self._load_experiment_reference_index()

    def _init_db(self) -> None:
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            osp.join(self.dataset_dir, "processed/data.lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=20,
        )
        # Open the default database
        with self.env.begin(write=False) as _txn:
            self.db = self.env.open_db(
                None, create=False
            )  # None refers to the default database

    def _load_experiment_reference_index(self) -> None:
        """Load the experiment reference index from its JSON file if present."""
        index_file_path = osp.join(
            self.dataset_dir, "preprocess/experiment_reference_index.json"
        )
        if osp.exists(index_file_path):
            with open(index_file_path) as file:
                index_data = json.load(file)
                # Convert dictionaries to ExperimentReferenceIndex objects
                self._experiment_reference_index = [
                    ExperimentReferenceIndex.from_stored(item) for item in index_data
                ]

    @property
    def experiment_reference_index(self) -> list[ExperimentReferenceIndex] | None:
        """Return the loaded experiment reference index."""
        return self._experiment_reference_index

    def save_experiment_reference_index(
        self, index_data: Any
    ) -> None:  # JSON-serializable
        """Write the experiment reference index to its JSON file."""
        index_file_path = osp.join(
            self.dataset_dir, "preprocess/experiment_reference_index.json"
        )
        with open(index_file_path, "w") as file:
            json.dump(index_data, file, indent=4)

    def close_lmdb(self) -> None:
        """Close the LMDB environment if it is open."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def __getitem__(self, idx: int | list[int] | np.ndarray) -> Any:
        """Return one item, or a list of items for array/list indices."""
        if isinstance(idx, (list, np.ndarray)):
            if isinstance(idx, list):
                idx = np.array(idx)
            if idx.dtype == np.bool_:
                idx = np.where(idx)[0]
            return [self.get_single_item(i) for i in idx]
        else:
            return self.get_single_item(idx)

    def get_single_item(self, idx: int) -> Any:  # pickle.loads returns Any
        """Return the unpickled item at the given integer index, or None."""
        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None
            deserialized_data = pickle.loads(serialized_data)
            return deserialized_data

    def __len__(self) -> int:
        """Return the number of entries in the LMDB database."""
        with self.env.begin() as txn:
            # Use the database handle
            length: int = txn.stat(db=self.db)["entries"]
        return length

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator over the dataset."""
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self) -> str:
        """Return a concise representation showing the dataset length."""
        return f"{self.__class__.__name__}({len(self)})"
