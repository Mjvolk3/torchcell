# torchcell/adapters/niu2019_adapter.py
# [[torchcell.adapters.niu2019_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/niu2019_adapter.py
# Test file: tests/torchcell/adapters/test_niu2019_adapter.py
"""BioCypher adapter exposing the Niu 2019 CRISPRa/CRISPRi pinene-tolerance strains as
knowledge-graph nodes and edges.

One record per stored strain: its targeted genes as
``BacterialCrisprActivationPerturbation`` (issue #799) or
``BacterialCrisprInterferencePerturbation`` leaves served as ``bacterial perturbation``
nodes, each carrying the shared ``dCas9*-MCPSoxS`` construct on a ``crispr construct``
node, the released OD600 ratio as an ``EnvironmentResponsePhenotype``, and 0.5% pinene
as an environment perturbation on both the experiment and the reference.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.niu2019 import CrisprPineneToleranceNiu2019Dataset


class CrisprPineneToleranceNiu2019Adapter(CellAdapter):
    """Cell adapter that serves CrisprPineneToleranceNiu2019Dataset to BioCypher."""

    def __init__(
        self,
        dataset: CrisprPineneToleranceNiu2019Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "crispr_pinene_tolerance_niu2019_adapter.yaml"
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
