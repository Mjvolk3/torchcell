# torchcell/adapters/girgis2009_adapter.py
# [[torchcell.adapters.girgis2009_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/girgis2009_adapter.py
# Test file: tests/torchcell/adapters/test_girgis2009_adapter.py
"""BioCypher adapter exposing the Girgis 2009 antibiotic transposon selection as
knowledge-graph nodes and edges.

One record per (gene, antibiotic): a single gene-level
``TransposonInsertionPerturbation`` served as a ``bacterial perturbation`` node, and the
combined z-score as an ``EnvironmentResponsePhenotype``. Every record's environment
carries exactly one ``SmallMoleculePerturbation``, the drug at its Table 1 dose, so the
environment-perturbation pair is enabled and needs no per-condition branch. No phage and
no CRISPRi construct appears in any record, so those pairs stay off, and the one medium
all 17 conditions share means the media node is a single entity the whole dataset joins
on.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.girgis2009 import EnvChemgenGirgis2009Dataset


class EnvChemgenGirgis2009Adapter(CellAdapter):
    """Cell adapter that serves EnvChemgenGirgis2009Dataset to BioCypher."""

    def __init__(
        self,
        dataset: EnvChemgenGirgis2009Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "ecoli_env_chemgen_girgis2009_adapter.yaml"
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
