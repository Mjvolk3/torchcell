import os
import json
import lmdb
from abc import ABC, abstractmethod
from typing import Type
from tqdm import tqdm
from torchcell.datamodels import ModelStrict
from torchcell.datamodels import EXPERIMENT_TYPE_MAP, EXPERIMENT_REFERENCE_TYPE_MAP
from torchcell.datamodels import ExperimentType, ExperimentReferenceType
from torchcell.datamodels import Phenotype


class ExperimentInfo(ModelStrict):
    experiment_type: str
    experiment_reference_type: str


class Aggregator(ABC):
    def __init__(self, root: str):
        self.root = root
        self.lmdb_dir = os.path.join(self.root, "aggregation", "lmdb")
        self.env = None
        self._phenotype_info = None

    @abstractmethod
    def aggregate_check(
        self, data: dict[str, ExperimentType | ExperimentReferenceType]
    ) -> str:
        """
        Check if the experiment should be aggregated based on specific criteria.
        Returns a string key for grouping similar experiments.
        """
        pass

    def create_aggregate_entry(
        self,
        experiments_to_aggregate: list[
            list[dict[str, ExperimentType | ExperimentReferenceType]]
        ],
    ) -> list[dict[str, ExperimentType | ExperimentReferenceType]]:
        return [exp for exp_list in experiments_to_aggregate for exp in exp_list]

    # TODO now that phenotype info is moved to [[torchcell.data.neo4j_cell]] we can probably remove this
    @property
    def phenotype_info(self) -> list[Type[Phenotype]]:
        if self._phenotype_info is None:
            self._phenotype_info = self._get_phenotype_info()
        return self._phenotype_info

    def _get_phenotype_info(self) -> list[Type[Phenotype]]:
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

    def _init_lmdb(self, readonly=True):
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

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def process(self, input_path: str, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._init_lmdb(readonly=False)  # Initialize LMDB for writing

        env_input = lmdb.open(input_path, readonly=True)

        aggregated_data = {}

        with env_input.begin(write=False) as txn_input:
            cursor = txn_input.cursor()
            for _, value in tqdm(cursor, desc="Aggregating data"):
                exp_pair = json.loads(value.decode("utf-8"))
                experiment_class = EXPERIMENT_TYPE_MAP[
                    exp_pair["experiment"]["experiment_type"]
                ]
                experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                    exp_pair["experiment_reference"]["experiment_reference_type"]
                ]
                reconstructed_pair = {
                    "experiment": experiment_class(**exp_pair["experiment"]),
                    "experiment_reference": experiment_reference_class(
                        **exp_pair["experiment_reference"]
                    ),
                }

                agg_key = self.aggregate_check(reconstructed_pair)
                if agg_key not in aggregated_data:
                    aggregated_data[agg_key] = []
                aggregated_data[agg_key].append(reconstructed_pair)

        # aggregated_data can potentially be huge since we have lists of objs.
        # might just want in store index.
        with self.env.begin(write=True) as txn_output:
            for idx, (agg_key, experiments) in enumerate(aggregated_data.items()):
                json_data = [
                    {
                        "experiment": exp["experiment"].model_dump(),
                        "experiment_reference": exp[
                            "experiment_reference"
                        ].model_dump(),
                    }
                    for exp in experiments
                ]
                txn_output.put(f"{idx}".encode(), json.dumps(json_data).encode())

        env_input.close()
        self.close_lmdb()

        # Reset experiment_info to force recomputation on next access
        self._experiment_info = None

        print(f"Aggregation complete. LMDB database written to {output_path}")
        print(f"Total number of aggregated groups: {len(aggregated_data)}")
        print(
            f"Total number of experiments after aggregation: {sum(len(exps) for exps in aggregated_data.values())}"
        )

    def __getitem__(
        self, index: int | slice | list
    ) -> list[dict[str, ExperimentType | ExperimentReferenceType]]:
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

    def __len__(self):
        self._init_lmdb(readonly=True)
        if self.env is None:
            return 0  # Return 0 if the LMDB doesn't exist yet
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    def __bool__(self):
        return os.path.exists(self.lmdb_dir)

    def __repr__(self):
        return f"Aggregator(root={self.root})"
