"""Convert raw Neo4j experiment records into typed experiments stored in LMDB."""

# torchcell/datamodels/conversion
# [[torchcell.datamodels.conversion]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/conversion
# Test file: tests/torchcell/datamodels/test_conversion.py
import hashlib
import json
import multiprocessing as mp
import os
import os.path as osp
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import lmdb
from tqdm import tqdm

# Import from sibling submodules directly, not the package __init__, so this
# module does not depend on __init__ import order (avoids a circular import that
# surfaces once imports are alphabetically sorted).
from torchcell.datamodels.pydant import ModelStrict
from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
    ExperimentReferenceType,
    ExperimentType,
)
from torchcell.lmdb_map_size import BUILD_LMDB_MAP_SIZE

if TYPE_CHECKING:
    from torchcell.data.neo4j_query_raw import Neo4jQueryRaw
import logging

log = logging.getLogger(__name__)

CONVERSION_BATCH_RECORDS = 2000
"""Records per worker task and per LMDB write transaction in the conversion stage."""

_WORKER_CONVERTER: "Converter | None" = None
"""The converter a forked worker uses; set by ``Converter.process`` before the pool."""


def _convert_batch(
    batch: list[tuple[bytes, bytes]],
) -> tuple[list[tuple[bytes, bytes]], int, int]:
    """Worker: convert a batch of stored rows; ``(rows, n_changed, n_skipped)``."""
    converter = _WORKER_CONVERTER
    if converter is None:
        raise RuntimeError("_convert_batch called outside Converter.process")
    rows: list[tuple[bytes, bytes]] = []
    changed = 0
    skipped = 0
    for key, value in batch:
        row, was_changed = converter._convert_record(key, value)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        changed += int(was_changed)
    return rows, changed, skipped


class ConversionEntry(ModelStrict):
    """One input/output type pair plus its conversion functions."""

    experiment_input_type: type[ExperimentType]
    experiment_conversion_function: Callable[..., ExperimentType | None]
    experiment_output_type: type[ExperimentType]
    experiment_reference_input_type: type[ExperimentReferenceType]
    experiment_reference_conversion_function: Callable[..., ExperimentReferenceType]
    experiment_reference_output_type: type[ExperimentReferenceType]


class ConversionMap(ModelStrict):
    """Collection of conversion entries defining how to map experiment types."""

    entries: list[ConversionEntry]


class Converter(ABC):
    """Abstract base that converts queried records and caches them in LMDB."""

    def __init__(self, root: str, query: "Neo4jQueryRaw"):
        """Store the root path and query, and set up the LMDB directory path."""
        self.root = root
        self.query = query
        self.lmdb_dir = os.path.join(self.root, "conversion", "lmdb")
        self.env: lmdb.Environment | None = None

    @property
    @abstractmethod
    def conversion_map(self) -> ConversionMap:
        """Return the ConversionMap defining type-to-type conversions."""

    def _init_lmdb(self, readonly: bool = True) -> None:
        """Initialize the LMDB environment."""
        if self.env is not None:
            self.close_lmdb()
        os.makedirs(os.path.dirname(self.lmdb_dir), exist_ok=True)
        if not readonly or osp.exists(self.lmdb_dir):
            self.env = lmdb.open(
                self.lmdb_dir,
                map_size=BUILD_LMDB_MAP_SIZE,
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

    def convert(
        self, data: dict[str, ExperimentType | ExperimentReferenceType]
    ) -> dict[str, ExperimentType | ExperimentReferenceType | None]:
        """Convert one experiment/reference pair using the conversion map."""
        if "experiment" not in data or "experiment_reference" not in data:
            raise ValueError(
                "Input data must contain both 'experiment' and "
                "'experiment_reference' keys"
            )

        converted_data: dict[str, ExperimentType | ExperimentReferenceType | None] = {}

        for key in ["experiment", "experiment_reference"]:
            for entry in self.conversion_map.entries:
                input_type = (
                    entry.experiment_input_type
                    if key == "experiment"
                    else entry.experiment_reference_input_type
                )
                conversion_function = (
                    entry.experiment_conversion_function
                    if key == "experiment"
                    else entry.experiment_reference_conversion_function
                )
                output_type = (
                    entry.experiment_output_type
                    if key == "experiment"
                    else entry.experiment_reference_output_type
                )

                if isinstance(data[key], input_type):
                    converted_data[key] = conversion_function(data[key])
                    if not isinstance(converted_data[key], output_type):
                        raise TypeError(
                            f"Conversion function did not return expected type for {key}"
                        )
                    break
            else:
                # If no conversion was found, keep the original data
                converted_data[key] = data[key]

        # Include any other keys from the original data that weren't converted
        for key, value in data.items():
            if key not in ["experiment", "experiment_reference"]:
                converted_data[key] = value

        return converted_data

    @staticmethod
    def _compute_hash(data: dict[str, Any]) -> str:
        """Compute a SHA256 hash of the input data."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _convert_record(
        self, key: bytes, value: bytes
    ) -> tuple[tuple[bytes, bytes] | None, bool]:
        """Convert ONE stored record; ``(row, changed)``, or ``(None, False)`` if skipped.

        Skipping on a decode or conversion error is the behavior the single-threaded
        loop had; it is logged, never silent.
        """
        try:
            data_dict = json.loads(value.decode("utf-8"))
            experiment_class = EXPERIMENT_TYPE_MAP[
                data_dict["experiment"]["experiment_type"]
            ]
            experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                data_dict["experiment_reference"]["experiment_reference_type"]
            ]
            data = {
                "experiment": experiment_class(**data_dict["experiment"]),
                "experiment_reference": experiment_reference_class(
                    **data_dict["experiment_reference"]
                ),
            }
            converted = self.convert(data)
            # convert() never yields None values; the Optional in its return type is
            # conservative, so model_dump() is safe here.
            out = {
                "experiment": converted["experiment"].model_dump(),  # type: ignore[union-attr]
                "experiment_reference": converted["experiment_reference"].model_dump(),  # type: ignore[union-attr]
            }
            changed = self._compute_hash(data_dict) != self._compute_hash(out)
            return (key, json.dumps(out).encode()), changed
        except json.JSONDecodeError:
            log.error(f"Error decoding JSON for entry {key!r}. Skipping this entry.")
        except Exception as e:
            log.error(f"Error processing entry {key!r}: {e}. Skipping this entry.")
        return None, False

    def process(
        self,
        input_path: str,
        output_path: str,
        *,
        batch_records: int = CONVERSION_BATCH_RECORDS,
        num_workers: int | None = None,
        limit: int | None = None,
    ) -> None:
        """Convert every record from the input LMDB and write results to output.

        The input cursor runs on this thread; batches of ``batch_records`` stored rows
        go to ``num_workers`` forked processes (default: the CPUs this process may run
        on, minus two, at most 16), which rebuild the models, convert, and return the
        output rows; each returned batch is committed in ONE write transaction, in
        input order. The previous loop did all of it on one thread inside one
        transaction over the whole store, and on the 030 build (43.8M records, 1 TB
        input) ran at 900 falling to 290 records/s as the input's page cache and the
        transaction's dirty pages fought over the job's memory (slurm 2748, cancelled
        at 43 percent after 15.6 h at 266 GB resident). ``limit`` converts only the
        first ``limit`` records, for tuning ``batch_records`` on a real store.
        """
        global _WORKER_CONVERTER
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if num_workers is None:
            num_workers = max(1, min(16, len(os.sched_getaffinity(0)) - 2))
        # The workers are forked from this process and inherit the converter through
        # the module global; the output env is opened AFTER the pool exists so no
        # child holds a write handle.
        _WORKER_CONVERTER = self
        converted_count = 0
        skipped_count = 0
        total_count = 0
        pending: deque[Future[tuple[list[tuple[bytes, bytes]], int, int]]] = deque()
        max_in_flight = 2 * num_workers
        progress = tqdm(
            desc="Converting and writing to LMDB", unit="rec", unit_scale=True
        )

        def drain_one() -> None:
            nonlocal converted_count, skipped_count, total_count
            rows, changed, skipped = pending.popleft().result()
            with self.env.begin(write=True) as txn_output:  # type: ignore[union-attr]
                for key, out in rows:
                    txn_output.put(key, out)
            converted_count += changed
            skipped_count += skipped
            total_count += len(rows)
            progress.update(len(rows) + skipped)

        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=mp.get_context("fork")
        ) as pool:
            self._init_lmdb(readonly=False)
            env_input = lmdb.open(input_path, readonly=True, readahead=False)
            with env_input.begin() as txn_input:
                batch: list[tuple[bytes, bytes]] = []
                for n, (key, value) in enumerate(txn_input.cursor()):
                    if limit is not None and n >= limit:
                        break
                    batch.append((bytes(key), bytes(value)))
                    if len(batch) == batch_records:
                        pending.append(pool.submit(_convert_batch, batch))
                        batch = []
                        if len(pending) >= max_in_flight:
                            drain_one()
                if batch:
                    pending.append(pool.submit(_convert_batch, batch))
                while pending:
                    drain_one()
        progress.close()
        _WORKER_CONVERTER = None
        if skipped_count:
            log.error(f"Skipped {skipped_count} records that failed to convert")

        env_input.close()
        self.close_lmdb()

        log.info(f"Conversion complete. LMDB database written to {output_path}")
        log.info(f"Number of instances converted: {converted_count}")
        log.info(f"Total number of instances written: {total_count}")
        log.info(f"Total number of instances processed: {total_count}")

    def __getitem__(
        self, index: int | slice | list[int]
    ) -> (
        dict[str, ExperimentType | ExperimentReferenceType]
        | list[dict[str, ExperimentType | ExperimentReferenceType]]
    ):
        """Return converted record(s) by integer index, slice, or list of indices."""
        self._init_lmdb(readonly=True)  # Initialize LMDB for reading
        if isinstance(index, int):
            return self._get_record_by_index(index)
        elif isinstance(index, slice):
            return self._get_records_by_slice(index)
        elif isinstance(index, list):
            return [self._get_record_by_index(idx) for idx in index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def _get_record_by_index(
        self, index: int
    ) -> dict[str, ExperimentType | ExperimentReferenceType]:
        if self.env is None:
            raise ValueError("LMDB environment is not initialized.")
        data_key = f"data_{index}".encode()
        with self.env.begin() as txn:
            data_json = txn.get(data_key)
            if data_json is None:
                raise IndexError(f"Record not found at index: {index}")
            data_dict = json.loads(data_json.decode())

            # Reconstruct Pydantic objects
            experiment_class = EXPERIMENT_TYPE_MAP[
                data_dict["experiment"]["experiment_type"]
            ]
            experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                data_dict["experiment_reference"]["experiment_reference_type"]
            ]

            reconstructed_data = {
                "experiment": experiment_class(**data_dict["experiment"]),
                "experiment_reference": experiment_reference_class(
                    **data_dict["experiment_reference"]
                ),
            }
            return reconstructed_data

    def _get_records_by_slice(
        self, slice_obj: slice
    ) -> list[dict[str, ExperimentType | ExperimentReferenceType]]:
        if self.env is None:
            raise ValueError("LMDB environment is not initialized.")
        start, stop, step = slice_obj.indices(len(self))
        data_keys = [f"data_{i}".encode() for i in range(start, stop, step)]
        with self.env.begin() as txn:
            results = []
            for key in data_keys:
                value = txn.get(key)
                if value is not None:
                    data_dict = json.loads(value.decode())
                    experiment_class = EXPERIMENT_TYPE_MAP[
                        data_dict["experiment"]["experiment_type"]
                    ]
                    experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                        data_dict["experiment_reference"]["experiment_reference_type"]
                    ]
                    reconstructed_data = {
                        "experiment": experiment_class(**data_dict["experiment"]),
                        "experiment_reference": experiment_reference_class(
                            **data_dict["experiment_reference"]
                        ),
                    }
                    results.append(reconstructed_data)
            return results

    def __len__(self) -> int:
        """Return the number of converted records stored in LMDB."""
        self._init_lmdb(readonly=True)
        if self.env is None:
            return 0  # Return 0 if the LMDB doesn't exist yet
        with self.env.begin() as txn:
            return cast(int, txn.stat()["entries"])

    def __bool__(self) -> bool:
        """Return True if the LMDB directory exists."""
        return os.path.exists(self.lmdb_dir)

    def __repr__(self) -> str:
        """Return a string representation showing the root path."""
        return f"Converter(root={self.root})"


if __name__ == "__main__":
    pass
