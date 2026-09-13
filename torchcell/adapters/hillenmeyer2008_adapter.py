# torchcell/adapters/hillenmeyer2008_adapter.py
# [[torchcell.adapters.hillenmeyer2008_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/hillenmeyer2008_adapter.py
# Test file: tests/torchcell/adapters/test_hillenmeyer2008_adapter.py
"""BioCypher adapters exposing the Hillenmeyer 2008 HIP/HOP chemogenomic compendium as
knowledge-graph nodes and edges.

One record is one (deletion strain, condition, control set) triple. The genotype is
gene-keyed, so it serves a ``genotype`` node plus one ``perturbation`` node per deleted
ORF (an ``EngineeredCopyNumberPerturbation`` for HIP, a ``KanMxDeletionPerturbation`` for
HOP), unlike the Bloom 2019 segregant adapter. The condition is an ``environment`` whose
media, temperature and each small-molecule / physical perturbation are first-class nodes,
and the readout is an ``environment response phenotype``.

HET and HOM are separate datasets because their scores are incomparable (a log-ratio and
a z-score), so they get one adapter class and one conf each; the two confs are identical
in structure and differ only in the comment naming their matrix.
"""

import os.path as osp

import yaml
from omegaconf import DictConfig, OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.hillenmeyer2008 import (
    HetHillenmeyer2008Dataset,
    HomHillenmeyer2008Dataset,
)


def _config(filename: str) -> DictConfig:
    """Load one adapter enable-list from ``torchcell/adapters/conf``."""
    config_path = osp.join(osp.dirname(osp.abspath(__file__)), "conf", filename)
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as file:
        config = OmegaConf.create(yaml.safe_load(file))
    if not isinstance(config, DictConfig):
        raise TypeError(f"{filename} must parse to a mapping, got {type(config)}")
    return config


class HetHillenmeyer2008Adapter(CellAdapter):
    """Cell adapter serving the HIP (heterozygous) log2-ratio dataset to BioCypher."""

    def __init__(
        self,
        dataset: HetHillenmeyer2008Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the HET adapter conf enable-list and initialize the base CellAdapter."""
        super().__init__(
            _config("het_hillenmeyer2008_adapter.yaml"),
            dataset,
            process_workers,
            io_workers,
            chunk_size,
            loader_batch_size,
        )
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class HomHillenmeyer2008Adapter(CellAdapter):
    """Cell adapter serving the HOP (homozygous) z-score dataset to BioCypher."""

    def __init__(
        self,
        dataset: HomHillenmeyer2008Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the HOM adapter conf enable-list and initialize the base CellAdapter."""
        super().__init__(
            _config("hom_hillenmeyer2008_adapter.yaml"),
            dataset,
            process_workers,
            io_workers,
            chunk_size,
            loader_batch_size,
        )
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size
