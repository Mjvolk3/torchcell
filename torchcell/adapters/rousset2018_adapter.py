# torchcell/adapters/rousset2018_adapter.py
# [[torchcell.adapters.rousset2018_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/rousset2018_adapter.py
# Test file: tests/torchcell/adapters/test_rousset2018_adapter.py
"""BioCypher adapter exposing the Rousset 2018 CRISPR-dCas9 screens as knowledge-graph
nodes and edges.

One record per (sgRNA, screen): a single ``BacterialCrisprInterferencePerturbation``
served as a ``bacterial perturbation`` node with its guide on a ``crispr construct``
node, and the guide's log2 fold change as an ``EnvironmentResponsePhenotype``.

The environment is where this conf differs from the other bacterial ones. All four
stored screens, the three phage challenges and the lambda transduction assay, carry
exactly one ``PhagePerturbation``, so the conf enables ``phage perturbation`` and NOT
``environment perturbation``: the served ``_environment_perturbation_node`` does not
filter phages out, so a conf enabling both would emit each phage twice, once under each
label, on one content id. The aTc that induces dCas9 is a component of the medium rather
than an environment perturbation (the paper lists it with the maltose and CaCl2), which
is what leaves the phage as the only environment perturbation any record carries. The
released growth screen is Cui 2018's and is not stored, so every one of the 68,400
records carries a phage and a temperature and the enable-list needs no per-screen branch.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.rousset2018 import CrispriScreenRousset2018Dataset


class CrispriScreenRousset2018Adapter(CellAdapter):
    """Cell adapter that serves CrispriScreenRousset2018Dataset to BioCypher."""

    def __init__(
        self,
        dataset: CrispriScreenRousset2018Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "ecoli_crispri_rousset2018_adapter.yaml"
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
