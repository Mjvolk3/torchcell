# torchcell/datamodels/conversion
# [[torchcell.datamodels.conversion]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/conversion
# Test file: tests/torchcell/datamodels/test_conversion.py
# from torchcell.data.neo4j_query_raw import Neo4jQueryRaw
from abc import ABC, abstractmethod
import os
import pickle
import lmdb
from tqdm import tqdm
from typing import Callable, Type
from torchcell.datamodels import ModelStrict
from torchcell.datamodels import ExperimentType, ExperimentReferenceType


class ConversionEntry(ModelStrict):
    experiment_input_type: Type[ExperimentType]
    experiment_conversion_function: Callable
    experiment_output_type: Type[ExperimentType]
    experiment_reference_input_type: Type[ExperimentReferenceType]
    experiment_reference_conversion_function: Callable
    experiment_reference_output_type: Type[ExperimentReferenceType]


class ConversionMap(ModelStrict):
    entries: list[ConversionEntry]


class Converter(ABC):
    def __init__(self, root: str):
        self.root = root

    @property
    @abstractmethod
    def conversion_map(self) -> ConversionMap:
        pass

    def convert(self, data: dict) -> dict:
        if "experiment" not in data or "experiment_reference" not in data:
            raise ValueError(
                "Input data must contain both 'experiment' and "
                "'experiment_reference' keys"
            )

        converted_data = {}

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

    def process(self, input_path: str, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        env_input = lmdb.open(input_path, readonly=True)
        env_output = lmdb.open(output_path, map_size=int(1e12))

        with env_input.begin() as txn_input, env_output.begin(write=True) as txn_output:
            cursor = txn_input.cursor()
            for idx, (_, value) in enumerate(
                tqdm(cursor, desc="Converting and writing to LMDB")
            ):
                data = pickle.loads(value)
                converted_data = self.convert(data)
                txn_output.put(f"{idx}".encode(), pickle.dumps(converted_data))

        env_input.close()
        env_output.close()

        print(f"Conversion complete. LMDB database written to {output_path}")


if __name__ == "__main__":
    pass
