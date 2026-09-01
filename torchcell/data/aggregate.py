"""LMDB-backed aggregation of experiment/reference pairs into grouped entries."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, cast

import lmdb
from tqdm import tqdm

from torchcell.datamodels import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
    ExperimentReferenceType,
    ExperimentType,
    ModelStrict,
    Phenotype,
)


class ExperimentInfo(ModelStrict):
    """Type metadata for an experiment/reference pair."""

    experiment_type: str
    experiment_reference_type: str


class Aggregator(ABC):
    """Abstract base that groups experiment pairs into an aggregated LMDB store."""

    def __init__(self, root: str):
        """Set up paths and lazy LMDB/phenotype state under the given root."""
        self.root = root
        self.lmdb_dir = os.path.join(self.root, "aggregation", "lmdb")
        self.env: Any = None
        self._phenotype_info: list[type[Phenotype]] | None = None

    @abstractmethod
    def aggregate_check(
        self, data: dict[str, ExperimentType | ExperimentReferenceType]
    ) -> str:
        """Return a grouping key for the experiment pair.

        The key determines which experiments are aggregated together.
        """
        pass

    @abstractmethod
    def aggregate_key_raw(self, record: dict[str, Any]) -> str:
        """Return the grouping key for ONE raw record dict (pre-pydantic).

        Streaming counterpart of :meth:`aggregate_check`: ``process`` calls this
        per record straight off the stored JSON, so grouping never requires the
        whole dataset in memory. Must produce the same key ``aggregate_check``
        would for that record.
        """
        pass

    def create_aggregate_entry(
        self,
        experiments_to_aggregate: list[
            list[dict[str, ExperimentType | ExperimentReferenceType]]
        ],
    ) -> list[dict[str, ExperimentType | ExperimentReferenceType]]:
        """Flatten nested lists of experiments into a single list."""
        return [exp for exp_list in experiments_to_aggregate for exp in exp_list]

    # TODO now that phenotype info is moved to [[torchcell.data.neo4j_cell]] we can probably remove this
    @property
    def phenotype_info(self) -> list[type[Phenotype]]:
        """Return the phenotype classes present in the aggregated store (cached)."""
        if self._phenotype_info is None:
            self._phenotype_info = self._get_phenotype_info()
        return self._phenotype_info

    def _get_phenotype_info(self) -> list[type[Phenotype]]:
        self._init_lmdb(readonly=True)
        if self.env is None:
            return []

        phenotype_classes = set()
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                experiments_list = json.loads(value.decode("utf-8"))
                for exp_pair in experiments_list:
                    experiment_type = exp_pair["experiment"]["experiment_type"]
                    experiment_class = EXPERIMENT_TYPE_MAP[experiment_type]
                    phenotype_class = experiment_class.__annotations__["phenotype"]
                    phenotype_classes.add(phenotype_class)

        self.close_lmdb()
        return list(phenotype_classes)

    def _init_lmdb(self, readonly: bool = True) -> None:
        """Initialize the LMDB environment."""
        if self.env is not None:
            self.close_lmdb()
        os.makedirs(os.path.dirname(self.lmdb_dir), exist_ok=True)
        if not readonly or os.path.exists(self.lmdb_dir):
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
        """Close the LMDB environment if open and reset the handle."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def process(self, input_path: str, output_path: str) -> None:
        """Read pairs from input LMDB, group by aggregate key, and write groups out.

        Two streaming passes, never the whole dataset in memory. The previous
        implementation held every record as pydantic objects inside
        ``aggregated_data`` -- its own comment conceded "can potentially be huge",
        and at the 025 build's scale that is hundreds of GB.

        Pass 1 groups input KEYS by ``aggregate_key_raw`` computed off the raw
        JSON. Pass 2 assembles each output group as a JSON array by joining the
        STORED record bytes -- they were written by upstream ``model_dump`` calls,
        so a reconstruct-and-redump would be an identity round trip. Group order
        is first-occurrence (dict insertion order), matching the old output.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._init_lmdb(readonly=False)  # Initialize LMDB for writing

        env_input = lmdb.open(input_path, readonly=True, readahead=False)

        # Pass 1: group input keys by the raw-record aggregation hash.
        key_groups: dict[str, list[bytes]] = {}
        with env_input.begin(write=False) as txn_input:
            cursor = txn_input.cursor()
            for key, value in tqdm(cursor, desc="Aggregation pass 1: grouping"):
                agg_key = self.aggregate_key_raw(json.loads(value.decode("utf-8")))
                key_groups.setdefault(agg_key, []).append(bytes(key))

        # Pass 2: write each group as a JSON array of the stored records.
        total_groups = len(key_groups)
        total_experiments = 0
        with (
            env_input.begin(write=False) as txn_input,
            self.env.begin(write=True) as txn_output,
        ):
            for idx, input_keys in enumerate(
                tqdm(key_groups.values(), total=total_groups, desc="Aggregating data")
            ):
                total_experiments += len(input_keys)
                group_bytes = (
                    b"[" + b",".join(txn_input.get(k) for k in input_keys) + b"]"
                )
                txn_output.put(f"{idx}".encode(), group_bytes)

        env_input.close()
        self.close_lmdb()

        # Reset experiment_info to force recomputation on next access
        self._experiment_info = None

        print(f"Aggregation complete. LMDB database written to {output_path}")
        print(f"Total number of aggregated groups: {total_groups}")
        print(f"Total number of experiments after aggregation: {total_experiments}")

    def __getitem__(
        self, index: int | slice | list[int]
    ) -> (
        list[dict[str, ExperimentType | ExperimentReferenceType]]
        | list[list[dict[str, ExperimentType | ExperimentReferenceType]]]
    ):
        """Return aggregated group(s) by int index, slice, or list of indices."""
        self._init_lmdb(readonly=True)  # Initialize LMDB for reading
        if isinstance(index, int):
            return self._get_record_by_index(index)
        elif isinstance(index, slice):
            return self._get_records_by_slice(index)
        elif isinstance(index, list):
            return [item for idx in index for item in self._get_record_by_index(idx)]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def _get_record_by_index(
        self, index: int
    ) -> list[dict[str, ExperimentType | ExperimentReferenceType]]:
        if self.env is None:
            raise ValueError("LMDB environment is not initialized.")
        data_key = f"{index}".encode()
        with self.env.begin() as txn:
            value = txn.get(data_key)
            if value is None:
                raise IndexError(f"No item found at index {index}")
            json_data = json.loads(value.decode("utf-8"))
            return [
                {
                    "experiment": EXPERIMENT_TYPE_MAP[
                        exp["experiment"]["experiment_type"]
                    ](**exp["experiment"]),
                    "experiment_reference": EXPERIMENT_REFERENCE_TYPE_MAP[
                        exp["experiment_reference"]["experiment_reference_type"]
                    ](**exp["experiment_reference"]),
                }
                for exp in json_data
            ]

    def _get_records_by_slice(
        self, slice_obj: slice
    ) -> list[list[dict[str, ExperimentType | ExperimentReferenceType]]]:
        if self.env is None:
            raise ValueError("LMDB environment is not initialized.")
        start, stop, step = slice_obj.indices(len(self))
        data_keys = [f"{i}".encode() for i in range(start, stop, step)]
        results = []
        with self.env.begin() as txn:
            for key in data_keys:
                value = txn.get(key)
                if value is not None:
                    json_data = json.loads(value.decode())
                    result = [
                        {
                            "experiment": EXPERIMENT_TYPE_MAP[
                                exp["experiment"]["experiment_type"]
                            ](**exp["experiment"]),
                            "experiment_reference": EXPERIMENT_REFERENCE_TYPE_MAP[
                                exp["experiment_reference"]["experiment_reference_type"]
                            ](**exp["experiment_reference"]),
                        }
                        for exp in json_data
                    ]
                    results.append(result)
        return results

    def __len__(self) -> int:
        """Return the number of aggregated groups in the store."""
        self._init_lmdb(readonly=True)
        if self.env is None:
            return 0  # Return 0 if the LMDB doesn't exist yet
        with self.env.begin() as txn:
            return cast(int, txn.stat()["entries"])

    def __bool__(self) -> bool:
        """Return whether the aggregated LMDB directory exists."""
        return os.path.exists(self.lmdb_dir)

    def __repr__(self) -> str:
        """Return a string representation showing the root path."""
        return f"Aggregator(root={self.root})"
