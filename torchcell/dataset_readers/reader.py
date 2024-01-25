# torchcell/dataset_readers/reader.py
# [[torchcell.dataset_readers.reader]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/dataset_readers/reader.py
# Test file: tests/torchcell/dataset_readers/test_reader.py

import os.path as osp
import pickle
import lmdb
import numpy as np
import json

from torchcell.datamodels import BaseGenotype, InterferenceGenotype
from torchcell.data.data import ExperimentReferenceIndex

class LmdbDatasetReader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.env = None
        self._experiment_reference_index = None
        self.db = None  # Add a db attribute
        self._init_db()
        self._load_experiment_reference_index()

    def _init_db(self):
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            osp.join(self.dataset_dir, "processed/data.lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=20
        )
        # Open the default database
        with self.env.begin(write=False) as txn:
            self.db = self.env.open_db(None, create=False)  # None refers to the default database


    def _load_experiment_reference_index(self):
        index_file_path = osp.join(
            self.dataset_dir, "preprocess/experiment_reference_index.json"
        )
        if osp.exists(index_file_path):
            with open(index_file_path, "r") as file:
                index_data = json.load(file)
                # Convert dictionaries to ExperimentReferenceIndex objects
                self._experiment_reference_index = [ExperimentReferenceIndex(**item) for item in index_data]

    @property
    def experiment_reference_index(self):
        return self._experiment_reference_index

    def save_experiment_reference_index(self, index_data):
        index_file_path = osp.join(
            self.dataset_dir, "preprocess/experiment_reference_index.json"
        )
        with open(index_file_path, "w") as file:
            json.dump(index_data, file, indent=4)

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray)):
            if isinstance(idx, list):
                idx = np.array(idx)
            if idx.dtype == np.bool_:
                idx = np.where(idx)[0]
            return [self.get_single_item(i) for i in idx]
        else:
            return self.get_single_item(idx)

    def get_single_item(self, idx):
        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None
            deserialized_data = pickle.loads(serialized_data)
            return deserialized_data

    def __len__(self):
        with self.env.begin() as txn:
            # Use the database handle
            length = txn.stat(db=self.db)["entries"]
        return length

    def __iter__(self):
        """Return an iterator over the dataset."""
        for idx in range(len(self)):
            yield self[idx]


    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"
