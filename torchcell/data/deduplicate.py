# torchcell/data/deduplicate
# [[torchcell.data.deduplicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/deduplicate
# Test file: tests/torchcell/data/test_deduplicate.py
"""Abstract LMDB-backed deduplicator that merges duplicate experiment records."""

import json
import logging
import os
import os.path as osp
from abc import ABC, abstractmethod
from typing import Any, cast

import lmdb
from tqdm import tqdm

from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
)

log = logging.getLogger(__name__)


class Deduplicator(ABC):
    """Base class that deduplicates experiment records into an output LMDB store."""

    def __init__(self, root: str):
        """Set the data root and the deduplication LMDB path; defer opening the env."""
        self.root = root
        self.lmdb_dir = os.path.join(self.root, "deduplication", "lmdb")
        self.env: Any = None

    @abstractmethod
    def duplicate_check(self, data: Any) -> dict[str, list[int]]:
        """Return a mapping of hash key to the record indices sharing that key."""
        pass

    @abstractmethod
    def duplicate_key(self, record: dict[str, Any]) -> str:
        """Return the duplicate-group hash for ONE raw record dict (pre-pydantic).

        Streaming counterpart of :meth:`duplicate_check`: ``process`` calls this
        per record straight off the stored JSON, so grouping never requires the
        whole dataset in memory. Must produce the same key the pydantic-based
        ``duplicate_check`` would for that record.
        """
        pass

    @abstractmethod
    def create_deduplicate_entry(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Merge a group of duplicate experiments into a single representative entry."""
        pass

    def _init_lmdb(self, readonly: bool = True) -> None:
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

    def close_lmdb(self) -> None:
        """Close the LMDB environment if it is open."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def process(self, input_path: str, output_path: str) -> None:
        """Read records from ``input_path``, deduplicate, and write to the output LMDB.

        Two streaming passes, never the whole dataset in memory. The previous
        implementation materialized every record as pydantic objects in one list
        (plus a second full list of outputs) -- at the 025 build's 43.8M records
        that OOM-killed slurm job 1560 immediately after a completed conversion.

        Pass 1 groups input KEYS by ``duplicate_key`` computed off the raw JSON.
        Pass 2 writes groups in first-occurrence order (dict insertion order, the
        same order the old code produced): singleton groups pass the stored bytes
        through untouched -- they were written by the converter's ``model_dump``,
        so a reconstruct-and-redump would be an identity round trip -- and only
        genuine duplicate groups are reconstructed for ``create_deduplicate_entry``.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._init_lmdb(readonly=False)  # Initialize LMDB for writing

        env_input = lmdb.open(input_path, readonly=True, readahead=False)

        # Pass 1: group input keys by the raw-record duplicate hash.
        key_groups: dict[str, list[bytes]] = {}
        with env_input.begin() as txn_input:
            cursor = txn_input.cursor()
            for key, value in tqdm(cursor, desc="Deduplication pass 1: grouping"):
                hash_key = self.duplicate_key(json.loads(value.decode("utf-8")))
                key_groups.setdefault(hash_key, []).append(bytes(key))

        # Pass 2: write one output record per group.
        deduplicated_count = 0
        total_groups = len(key_groups)
        with env_input.begin() as txn_input, self.env.begin(write=True) as txn_output:
            for idx, input_keys in enumerate(
                tqdm(
                    key_groups.values(),
                    total=total_groups,
                    desc="Deduplicating and writing to LMDB",
                )
            ):
                if len(input_keys) == 1:
                    txn_output.put(f"{idx}".encode(), txn_input.get(input_keys[0]))
                    continue
                duplicate_experiments = []
                for input_key in input_keys:
                    json_data = json.loads(txn_input.get(input_key).decode("utf-8"))
                    experiment_class = EXPERIMENT_TYPE_MAP[
                        json_data["experiment"]["experiment_type"]
                    ]
                    experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                        json_data["experiment_reference"]["experiment_reference_type"]
                    ]
                    duplicate_experiments.append(
                        {
                            "experiment": experiment_class(**json_data["experiment"]),
                            "experiment_reference": experiment_reference_class(
                                **json_data["experiment_reference"]
                            ),
                        }
                    )
                mean_entry = self.create_deduplicate_entry(duplicate_experiments)
                deduplicated_count += len(input_keys) - 1
                txn_output.put(
                    f"{idx}".encode(),
                    json.dumps(
                        {
                            "experiment": mean_entry["experiment"].model_dump(),
                            "experiment_reference": mean_entry[
                                "experiment_reference"
                            ].model_dump(),
                        }
                    ).encode(),
                )

        env_input.close()
        self.close_lmdb()

        log.info(f"Deduplication complete. LMDB database written to {output_path}")
        log.info(f"Number of instances deduplicated: {deduplicated_count}")
        log.info(f"Total number of instances after deduplication: {total_groups}")

    def __getitem__(
        self, index: int | slice | list[int]
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Return deduplicated record(s) for an int, slice, or list of indices."""
        self._init_lmdb(readonly=True)  # Initialize LMDB for reading
        if isinstance(index, int):
            return self._get_record_by_index(index)
        elif isinstance(index, slice):
            return self._get_records_by_slice(index)
        elif isinstance(index, list):
            return [self._get_record_by_index(idx) for idx in index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def _get_record_by_index(self, index: int) -> dict[str, Any]:
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

    def _get_records_by_slice(self, slice_obj: slice) -> list[dict[str, Any]]:
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

    def __len__(self) -> int:
        """Return the number of deduplicated records, or 0 if the LMDB is absent."""
        self._init_lmdb(readonly=True)
        if self.env is None:
            return 0  # Return 0 if the LMDB doesn't exist yet
        with self.env.begin() as txn:
            return cast(int, txn.stat()["entries"])

    def __bool__(self) -> bool:
        """Return whether the deduplication LMDB directory exists."""
        return os.path.exists(self.lmdb_dir)

    def __repr__(self) -> str:
        """Return a string identifying the deduplicator and its root."""
        return f"Deduplicator(root={self.root})"


if __name__ == "__main__":
    pass
