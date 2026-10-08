# torchcell/adapters/babu2014_adapter.py
# [[torchcell.adapters.babu2014_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/babu2014_adapter.py
# Test file: tests/torchcell/adapters/test_babu2014_adapter.py
"""BioCypher adapter exposing the Babu 2014 eSGA digenic interaction map as
knowledge-graph nodes and edges.

One record per released (donor, recipient) pair: a two-leaf genotype of gene-level
``BacterialDeletionPerturbation`` objects served as ``bacterial perturbation`` nodes,
and the signed eSGA S score as a ``GeneInteractionPhenotype``. No new graph node class
is needed: ``gene interaction phenotype`` is already a served class (the yeast
interaction datasets declare it), and the two leaves differ only in their ``collection``
and ``cassette``, which the ``bacterial perturbation`` id rule already covers. The
screen is a single condition, so the medium and the temperature are single entities the
whole dataset joins on and the environment carries no perturbation; the two marker drugs
are ``selection_agent`` components of that medium rather than small-molecule edits, so
the environment-perturbation pair stays off. No phage and no CRISPRi construct appears
in any record.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.babu2014 import GeneInteractionBabu2014Dataset


class GeneInteractionBabu2014Adapter(CellAdapter):
    """Cell adapter that serves GeneInteractionBabu2014Dataset to BioCypher."""

    def __init__(
        self,
        dataset: GeneInteractionBabu2014Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "gene_interaction_babu2014_adapter.yaml"
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
