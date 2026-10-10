# torchcell/adapters/royet2025_adapter.py
# [[torchcell.adapters.royet2025_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/royet2025_adapter.py
# Test file: tests/torchcell/adapters/test_royet2025_adapter.py
"""BioCypher adapter exposing the Royet 2025 KT2440 metal Tn-seq screen as
knowledge-graph nodes and edges.

One record per (gene, metal): a single gene-level ``TransposonInsertionPerturbation``
served as a ``bacterial perturbation`` node, and the TRANSIT log2FC as an
``EnvironmentResponsePhenotype``. Every record's environment carries exactly one
``SmallMoleculePerturbation``, the metal chloride at its Methods dose, on the experiment
and on the reference, so the environment-perturbation pair is enabled with no
per-condition branch. The four conditions share one medium (LB) and one temperature, so
those nodes are single entities; no phage and no CRISPRi construct appears in any
record, so those pairs stay off.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.royet2025 import EnvMetalTnseqRoyet2025Dataset


class EnvMetalTnseqRoyet2025Adapter(CellAdapter):
    """Cell adapter that serves EnvMetalTnseqRoyet2025Dataset to BioCypher."""

    def __init__(
        self,
        dataset: EnvMetalTnseqRoyet2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "pputida_env_metal_tnseq_royet2025_adapter.yaml"
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
