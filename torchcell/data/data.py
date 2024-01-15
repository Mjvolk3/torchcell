# torchcell/data/data.py
# [[torchcell.data.data]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/data.py
# Test file: tests/torchcell/data/test_data.py

import functools
import json
import logging
import os
import os.path as osp
import pickle
import random
import re
import shutil
import zipfile
from abc import ABC, abstractproperty
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Union
import multiprocessing as mp

# import lmdb
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

# import polars as pl
import torch
from attrs import define, field

# from polars import DataFrame, col
from pydantic import Field, field_validator
from torch_geometric.data import (
    Data,
    DataLoader,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from tqdm import tqdm
import json
import logging
import os
import os.path as osp
import pickle
import random
import re
import shutil
import zipfile
from abc import ABC, abstractproperty
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import h5py
import lmdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

# from polars import DataFrame, col
from torch_geometric.data import (
    Data,
    DataLoader,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from tqdm import tqdm

from torchcell.data import Dataset
from torchcell.datamodels import ModelStrict
from torchcell.prof import prof, prof_input
from torchcell.sequence import GeneSet


import hashlib
import json
from torchcell.data import Dataset
from torchcell.datamodels import (
    BaseEnvironment,
    BaseGenotype,
    BasePhenotype,
    BaseExperiment,
    GenePerturbation,
    Media,
    ModelStrict,
    ReferenceGenome,
    Temperature,
    DeletionGenotype,
    DeletionPerturbation,
    FitnessPhenotype,
    FitnessExperimentReference,
    ExperimentReference,
    FitnessExperiment,
    ExperimentReference,
    DampPerturbation,
    TsAllelePerturbation,
    InterferenceGenotype,
    KanMxDeletionPerturbation,
    NatMxDeletionPerturbation,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgdTsAllelePerturbation,
    SgdDampPerturbation,
    SuppressorAllelePerturbation,
    SgdSuppressorAllelePerturbation,
    SuppressorGenotype,
)
from torchcell.prof import prof, prof_input
from torchcell.sequence import GeneSet


class ExperimentReferenceIndex(ModelStrict):
    reference: ExperimentReference
    index: List[bool]

    def __repr__(self):
        if len(self.index) > 5:
            return f"ExperimentReferenceIndex(reference={self.reference}, index={self.index[:5]}...)"
        else:
            return f"ExperimentReferenceIndex(reference={self.reference}, index={self.index})"


class ReferenceIndex(ModelStrict):
    data: List[ExperimentReferenceIndex]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @field_validator("data")
    def validate_data(cls, v):
        summed_indices = sum(
            [
                boolean_value
                for exp_ref_index in v
                for boolean_value in exp_ref_index.index
            ]
        )

        if summed_indices != len(v[0].index):
            raise ValueError("Sum of indices must equal the number of experiments")
        return v


def serialize_for_hashing(obj) -> str:
    """
    Serialize a Pydantic object for hashing.
    """
    return json.dumps(obj.dict(), sort_keys=True)


def compute_md5_hash(content: str) -> str:
    """
    Compute the MD5 hash of a string.
    """
    return hashlib.md5(content.encode()).hexdigest()


def compute_experiment_reference_index(
    dataset: Dataset,
) -> list[ExperimentReferenceIndex]:
    # Hashes for each reference
    print("Computing hashes...")
    reference_hashes = [
        compute_md5_hash(serialize_for_hashing(data["reference"]))
        for data in tqdm(dataset)
    ]

    # Identify unique hashes
    unique_hashes = set(reference_hashes)

    # Initialize ExperimentReferenceIndex list
    reference_indices = []

    print("Finding unique references...")
    for unique_hash in tqdm(unique_hashes):
        # Create a boolean list where True indicates the presence of the unique reference
        index_list = [ref_hash == unique_hash for ref_hash in reference_hashes]

        # Find the corresponding reference object for the unique hash
        ref_index = reference_hashes.index(unique_hash)
        unique_ref = dataset[ref_index]["reference"]

        # Create ExperimentReferenceIndex object
        exp_ref_index = ExperimentReferenceIndex(reference=unique_ref, index=index_list)
        reference_indices.append(exp_ref_index)

    return reference_indices
