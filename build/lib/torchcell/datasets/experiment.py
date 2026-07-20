"""Dataset that merges multiple experiment datasets into a single LMDB store."""

# torchcell/datasets/experiment.py
# [[torchcell.datasets.experiment]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/experiment.py
# Test file: torchcell/datasets/test_experiment.py

import json
import logging
import pickle
from collections.abc import Callable
from typing import Any, cast

import lmdb
from tqdm import tqdm

from torchcell.data import Dataset  # type: ignore[attr-defined]
from torchcell.sequence import GeneSet

log = logging.getLogger(__name__)


# IDEA right now experiments: list[Dataset] but the Dataset[0]: Data
# will need to adhere to the same schema
# TODO unify Costanzo2016
class MergedExperiment(Dataset):  # type: ignore[misc]  # Dataset resolves to Any (dead import)
    """Combine several experiment datasets into one deduplicated LMDB-backed dataset."""

    def __init__(
        self,
        root: str = "data/scerevisiae/merged_experiment",
        experiments: list[Dataset] | None = None,
        preprocess: dict[str, Any] | None = None,
        skip_process_file_exist_check: bool = False,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
    ) -> None:
        """Set up the merged dataset and validate the preprocess config.

        Args:
            root: Root directory for raw, processed, and preprocess files.
            experiments: Datasets to merge; each item must share a schema.
            preprocess: Optional preprocessing config (e.g. temperature filter).
            skip_process_file_exist_check: Skip the processed-file existence check.
            transform: Optional transform applied to each item on access.
            pre_transform: Optional transform applied before saving.
        """
        self.experiments = experiments
        self.preprocess = preprocess
        self._skip_process_file_exist = skip_process_file_exist_check
        self.preprocess = preprocess
        self.preprocess_dir = osp.join(root, "preprocess")
        self._length: int | None = None
        self._gene_set: GeneSet | None = None
        self._df: Any = None
        # Check for existing preprocess config
        existing_config = self.load_preprocess_config()
        if existing_config is not None:
            if existing_config != self.preprocess:
                raise ValueError(
                    "New preprocess does not match existing config."
                    "Delete the processed and process dir for a new Dataset."
                    "Or define a new root."
                )
        # Check for existing preprocess config
        super().__init__(root, transform, pre_transform)
        self.env: lmdb.Environment = None

    @property
    def skip_process_file_exist(self) -> bool:
        """Return whether the processed-file existence check is skipped."""
        return self._skip_process_file_exist

    @property
    def raw_file_names(self) -> list[str]:
        """Return the expected raw input file names."""
        return ["strain_ids_and_single_mutant_fitness.xlsx"]

    @property
    def processed_file_names(self) -> str:
        """Return the processed LMDB file name."""
        return "data.lmdb"

    def _init_db(self) -> None:
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            osp.join(self.processed_dir, "data.lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def close_lmdb(self) -> None:
        """Close the open LMDB environment if one exists."""
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def wt(self) -> None:
        """Return the wild-type reference (not yet implemented)."""
        # TODO implement this, must be combination of all experiments
        pass

    # Merge Operations - This is HACK y
    # data_item is a dynamic Data/dict experiment record
    def get_genotype_key(self, data_item: Any) -> Any:
        """Return the genotype used as a merge key for a data item."""
        return data_item["genotype"]

    def get_environment_key(self, data_item: Any) -> Any:
        """Return the environment used as a merge key for a data item."""
        return data_item["phenotype"]["environment"]

    def get_phenotype_key(self, data_item: Any) -> Any:
        """Return the phenotype observation used as a merge key for a data item."""
        return data_item["phenotype"]["observation"]

    def contains_key(self, key: Any, data_list: list[Any]) -> bool:
        """Return whether any item in the list has the given genotype key."""
        return key in [self.get_genotype_key(data_item) for data_item in data_list]

    def contains_environment_key(
        self, genotype_key: Any, environment_key: Any, data_list: list[Any]
    ) -> bool:
        """Return whether the genotype already has the given environment key."""
        return environment_key in [
            self.get_environment_key(data_item)
            for data_item in data_list
            if self.get_genotype_key(data_item) == genotype_key
        ]

    @property
    def experiment_indices(self) -> dict[str, Any]:
        """Return cached per-source-dataset index ranges, or an empty dict."""
        indices_path = osp.join(self.processed_dir, "experiment_indices.json")

        if osp.exists(indices_path):
            with open(indices_path) as f:
                indices: dict[str, Any] = json.load(f)
                return indices

        # If no cached indices found, log a warning and return an empty dict
        log.warning("No cached experiment indices found!")
        return {}

    def process(self) -> None:
        """Merge, dedup, and write all experiment items to the processed LMDB."""
        combined_data_list: list[Any] = []
        source_dataset_list = []  # New list to track the source dataset of each item

        # Extract the desired temperature from the preprocess dictionary, if specified
        desired_temperature = None
        if self.preprocess and "temperature" in self.preprocess:
            desired_temperature = self.preprocess["temperature"]

        for dataset in cast(list[Any], self.experiments):
            for data_item in dataset:
                # If a temperature preprocessing condition exists,
                # skip items not matching it
                if desired_temperature is not None:
                    current_temp = data_item["phenotype"]["environment"]["temperature"]
                    if current_temp != desired_temperature:
                        continue  # Skip this item

                genotype_key = self.get_genotype_key(data_item)
                environment_key = self.get_environment_key(data_item)
                phenotype_key = self.get_phenotype_key(data_item)

                if not self.contains_key(genotype_key, combined_data_list):
                    combined_data_list.append(data_item)
                    source_dataset_list.append(dataset.__class__.__name__)
                else:
                    if not self.contains_environment_key(
                        genotype_key, environment_key, combined_data_list
                    ):
                        combined_data_list.append(data_item)
                        source_dataset_list.append(dataset.__class__.__name__)
                    else:
                        # This checks if the phenotype key exists for the
                        # given genotype and environment.
                        # However, the function "contains_phenotype_key" is not defined
                        # in your provided code.
                        # You might want to implement it similarly to
                        # "contains_environment_key".
                        if not self.contains_phenotype_key(
                            genotype_key,
                            environment_key,
                            phenotype_key,
                            combined_data_list,
                        ):
                            combined_data_list.append(data_item)
                            source_dataset_list.append(dataset.__class__.__name__)
                        else:
                            log.warning(
                                f"Data item with genotype key: {genotype_key}, "
                                f"environment key: {environment_key}, and "
                                f"phenotype key: {phenotype_key} already exists in the merged dataset."
                            )

        # Initialize LMDB environment
        log.info("lmdb begin")
        env = lmdb.open(osp.join(self.processed_dir, "data.lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            # Iterate through each data item
            for idx, item in tqdm(enumerate(combined_data_list)):
                serialized_data = pickle.dumps(item)
                txn.put(f"{idx}".encode(), serialized_data)

        # cache gene property
        self.gene_set = self.compute_gene_set(combined_data_list)

        # Now compute the experiment indices using source_dataset_list
        experiment_indices: dict[str, dict[str, Any]] = {
            dataset.__class__.__name__: {"size": 0, "indices": []}
            for dataset in cast(list[Any], self.experiments)
        }
        for idx, dataset_name in enumerate(source_dataset_list):
            experiment_indices[dataset_name]["indices"].append(idx)
            experiment_indices[dataset_name]["size"] += 1

        # Save the cached indices as a JSON file
        with open(osp.join(self.processed_dir, "experiment_indices.json"), "w") as f:
            json.dump(experiment_indices, f, indent=4)

    def get(self, idx: int) -> Any:
        """Initialize LMDB if it hasn't been initialized yet."""
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None
            data = pickle.loads(serialized_data)
            if self.transform:
                data = self.transform(data)
            return data

    # New method to save preprocess configuration to a JSON file
    def save_preprocess_config(self, preprocess: dict[str, Any]) -> None:
        """Write the preprocess config to JSON in the preprocess directory."""
        if not osp.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)
        with open(osp.join(self.preprocess_dir, "preprocess_config.json"), "w") as f:
            json.dump(preprocess, f)

    # New method to load existing preprocess configuration
    def load_preprocess_config(self) -> Any:  # dynamic JSON config or None
        """Load the saved preprocess config from JSON, or return None if absent."""
        config_path = osp.join(self.preprocess_dir, "preprocess_config.json")

        if osp.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            return config
        else:
            return None

    def len(self) -> int:
        """Return the number of entries in the processed LMDB store."""
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            length: int = txn.stat()["entries"]

        # Must be closed for dataloader num_workers > 0
        self.close_lmdb()

        return length

    @staticmethod
    def compute_gene_set(data_list: list[Any]) -> GeneSet:
        """Return the GeneSet of all genotype ids across the given data list."""
        computed_gene_set = GeneSet()
        for data in data_list:
            for genotype in data.genotype:
                computed_gene_set.add(genotype["id"])
        return computed_gene_set

    # Reading from JSON and setting it to self._gene_set
    @property
    def gene_set(self) -> GeneSet:
        """Return the dataset's GeneSet, loading it from JSON when available."""
        try:
            if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
                with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                    self._gene_set = GeneSet(json.load(f))
            elif self._gene_set is None:
                raise ValueError(
                    "gene_set not written during process. "
                    "Please call compute_gene_set in process."
                )
            return self._gene_set
        # CHECK can probably remove this
        except json.JSONDecodeError:
            raise ValueError("Invalid or empty JSON file found.")

    @gene_set.setter
    def gene_set(self, value: GeneSet) -> None:
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        os.makedirs(self.preprocess_dir, exist_ok=True)
        with open(osp.join(self.preprocess_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

    def __repr__(self) -> str:
        """Return a string with the class name and item count."""
        return f"{self.__class__.__name__}({len(self)})"


if __name__ == "__main__":
    import os
    import os.path as osp

    from dotenv import load_dotenv

    from torchcell.datasets.scerevisiae import (
        DmfCostanzo2016Dataset,
        SmfCostanzo2016Dataset,
    )

    load_dotenv()
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))

    dmf_dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e2"),
        preprocess={"duplicate_resolution": "low_dmf_std"},
        subset_n=100,
    )
    print(dmf_dataset)
    print(dmf_dataset[0])
    print(len(dmf_dataset.gene_set))

    smf_dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_smf"),
        preprocess={"duplicate_resolution": "low_std_both"},
        skip_process_file_exist_check=True,
        subset_n=100,
    )
    print(smf_dataset)
    print(smf_dataset[0])
    print(len(smf_dataset.gene_set))

    # TODO implement this
    fitness = MergedExperiment(
        root=osp.join(DATA_ROOT, "data/scerevisiae/smf_dmf"),
        experiments=[smf_dataset, dmf_dataset],
        preprocess={"temperature": 30},
    )
    print(fitness)
    print(fitness[0])
    print(fitness.experiment_indices)
    print(fitness.gene_set)
