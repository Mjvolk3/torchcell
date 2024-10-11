# torchcell/data/deduplicate
# [[torchcell.data.deduplicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/deduplicate
# Test file: tests/torchcell/data/test_deduplicate.py

import hashlib
import numpy as np
from typing import Any
import logging
import json
from torchcell.datamodels import (
    Genotype,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    MeanDeletionPerturbation,
    GeneInteractionPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
)
from abc import ABC, abstractmethod
from scipy.stats import t
import os
import lmdb
import pickle
from abc import ABC, abstractmethod
from typing import Any
from tqdm import tqdm
import json
import os
import os.path as osp
import lmdb
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Any, Union
from torchcell.datamodels.schema import (
    EXPERIMENT_TYPE_MAP,
    EXPERIMENT_REFERENCE_TYPE_MAP,
)

log = logging.getLogger(__name__)


class Deduplicator(ABC):
    def __init__(self, root: str):
        self.root = root
        self.lmdb_dir = os.path.join(self.root, "deduplication", "lmdb")
        self.env = None

    @abstractmethod
    def duplicate_check(self, data: Any) -> dict[str, list[int]]:
        pass

    @abstractmethod
    def create_deduplicate_entry(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        pass

    def _init_lmdb(self, readonly=True):
        """Initialize the LMDB environment."""
        if self.env is not None:
            self.close_lmdb()
        os.makedirs(os.path.dirname(self.lmdb_dir), exist_ok=True)
        if not readonly or osp.exists(self.lmdb_dir):
            self.env = lmdb.open(
                self.lmdb_dir,
                map_size=int(1e12),
                readonly=readonly,
                create=not readonly,
                lock=not readonly,
                readahead=False,
                meminit=False,
            )
        else:
            self.env = None

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def process(self, input_path: str, output_path: str) -> None:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._init_lmdb(readonly=False)  # Initialize LMDB for writing

        env_input = lmdb.open(input_path, readonly=True)

        # Read all data from input LMDB
        data = []
        with env_input.begin() as txn_input:
            cursor = txn_input.cursor()
            for _, value in cursor:
                json_data = json.loads(value.decode("utf-8"))
                # Reconstruct Pydantic objects
                experiment_class = EXPERIMENT_TYPE_MAP[
                    json_data["experiment"]["experiment_type"]
                ]
                experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                    json_data["experiment_reference"]["experiment_reference_type"]
                ]
                reconstructed_data = {
                    "experiment": experiment_class(**json_data["experiment"]),
                    "experiment_reference": experiment_reference_class(
                        **json_data["experiment_reference"]
                    ),
                }
                data.append(reconstructed_data)

        # Perform deduplication
        duplicate_check = self.duplicate_check(data)
        deduplicated_data = []
        deduplicated_count = 0

        with self.env.begin(write=True) as txn_output:
            for idx, (hash_key, indices) in enumerate(
                tqdm(duplicate_check.items(), desc="Deduplicating and writing to LMDB")
            ):
                if len(indices) > 1:
                    # Compute mean entry for duplicate experiments
                    duplicate_experiments = [data[i] for i in indices]
                    mean_entry = self.create_deduplicate_entry(duplicate_experiments)
                    deduplicated_data.append(mean_entry)
                    deduplicated_count += len(indices) - 1
                else:
                    # Keep non-duplicate experiments as is
                    deduplicated_data.append(data[indices[0]])

                # Serialize to JSON and write the deduplicated data to LMDB
                json_data = {
                    "experiment": deduplicated_data[-1]["experiment"].model_dump(),
                    "experiment_reference": deduplicated_data[-1][
                        "experiment_reference"
                    ].model_dump(),
                }
                txn_output.put(f"{idx}".encode(), json.dumps(json_data).encode())

        env_input.close()
        self.close_lmdb()

        log.info(f"Deduplication complete. LMDB database written to {output_path}")
        log.info(f"Number of instances deduplicated: {deduplicated_count}")
        log.info(
            f"Total number of instances after deduplication: {len(deduplicated_data)}"
        )

    def __getitem__(self, index: Union[int, slice, list]):
        self._init_lmdb(readonly=True)  # Initialize LMDB for reading
        if isinstance(index, int):
            return self._get_record_by_index(index)
        elif isinstance(index, slice):
            return self._get_records_by_slice(index)
        elif isinstance(index, list):
            return [self._get_record_by_index(idx) for idx in index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def _get_record_by_index(self, index: int):
        if self.env is None:
            raise ValueError("LMDB environment is not initialized.")
        data_key = f"{index}".encode()
        with self.env.begin() as txn:
            value = txn.get(data_key)
            if value is None:
                raise IndexError(f"No item found at index {index}")
            json_data = json.loads(value.decode("utf-8"))
            experiment_class = EXPERIMENT_TYPE_MAP[
                json_data["experiment"]["experiment_type"]
            ]
            experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                json_data["experiment_reference"]["experiment_reference_type"]
            ]
            return {
                "experiment": experiment_class(**json_data["experiment"]),
                "experiment_reference": experiment_reference_class(
                    **json_data["experiment_reference"]
                ),
            }

    def _get_records_by_slice(self, slice_obj: slice):
        if self.env is None:
            raise ValueError("LMDB environment is not initialized.")
        start, stop, step = slice_obj.indices(len(self))
        data_keys = [f"{i}".encode() for i in range(start, stop, step)]
        with self.env.begin() as txn:
            results = []
            for key in data_keys:
                value = txn.get(key)
                if value is not None:
                    json_data = json.loads(value.decode())
                    experiment_class = EXPERIMENT_TYPE_MAP[
                        json_data["experiment"]["experiment_type"]
                    ]
                    experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                        json_data["experiment_reference"]["experiment_reference_type"]
                    ]
                    reconstructed_data = {
                        "experiment": experiment_class(**json_data["experiment"]),
                        "experiment_reference": experiment_reference_class(
                            **json_data["experiment_reference"]
                        ),
                    }
                    results.append(reconstructed_data)
            return results

    def __len__(self):
        self._init_lmdb(readonly=True)
        if self.env is None:
            return 0  # Return 0 if the LMDB doesn't exist yet
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    def __bool__(self):
        return os.path.exists(self.lmdb_dir)

    def __repr__(self):
        return f"Deduplicator(root={self.root})"


if __name__ == "__main__":
    pass
