# torchcell/adapters/rachwalski2024_adapter.py
# [[torchcell.adapters.rachwalski2024_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/rachwalski2024_adapter.py
# Test file: tests/torchcell/adapters/test_rachwalski2024_adapter.py
"""BioCypher adapter exposing the Rachwalski 2024 mobile CRISPRi crosses as
knowledge-graph nodes and edges.

One record per (strain, medium, inducer dose). A record's
``BacterialCrisprInterferencePerturbation`` and ``BacterialDeletionPerturbation`` leaves
are served as ``bacterial perturbation`` nodes, the ``CrisprConstruct`` each knockdown
carries as a ``crispr construct`` node, the anhydrotetracycline of an induced plate as an
``environment perturbation`` node, and the released normalized colony growth as a
``FitnessPhenotype``. The crossed records put two ``bacterial perturbation`` nodes on one
``genotype``, which is the pairing this dataset exists for.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.rachwalski2024 import CrispriCrossRachwalski2024Dataset


class CrispriCrossRachwalski2024Adapter(CellAdapter):
    """Cell adapter that serves CrispriCrossRachwalski2024Dataset to BioCypher."""

    def __init__(
        self,
        dataset: CrispriCrossRachwalski2024Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "crispri_cross_rachwalski2024_adapter.yaml"
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
