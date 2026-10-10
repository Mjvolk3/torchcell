# torchcell/adapters/hawkins2020_adapter.py
# [[torchcell.adapters.hawkins2020_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/hawkins2020_adapter.py
# Test file: tests/torchcell/adapters/test_hawkins2020_adapter.py
"""BioCypher adapter exposing the Hawkins 2020 mismatch-CRISPRi screen as
knowledge-graph nodes and edges.

One record per released sgRNA with a measured relative fitness: its single
``BacterialCrisprInterferencePerturbation`` is served as a ``bacterial perturbation``
node whose ``description`` carries the guide's mismatch design, the ``CrisprConstruct``
holding the 20-nt spacer as a ``crispr construct`` node, and the released replicate mean
as an ``EnvironmentResponsePhenotype``. The environment carries the 1 mM IPTG that
induces dCas9 as a ``SmallMoleculePerturbation`` on both the experiment and the
reference, so the environment-perturbation pair is enabled.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.hawkins2020 import (
    MismatchCrispriFitnessHawkins2020Dataset,
)


class MismatchCrispriFitnessHawkins2020Adapter(CellAdapter):
    """Cell adapter that serves MismatchCrispriFitnessHawkins2020Dataset to BioCypher."""

    def __init__(
        self,
        dataset: MismatchCrispriFitnessHawkins2020Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "mismatch_crispri_fitness_hawkins2020_adapter.yaml"
        )
        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as file:
            yaml_config = yaml.safe_load(file)
        config = OmegaConf.create(yaml_config)
        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size
