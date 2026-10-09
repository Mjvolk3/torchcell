# torchcell/adapters/butland2008_adapter.py
# [[torchcell.adapters.butland2008_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/butland2008_adapter.py
# Test file: tests/torchcell/adapters/test_butland2008_adapter.py
"""BioCypher adapter exposing the unfiltered Butland 2008 eSGA interaction matrix as
knowledge-graph nodes and edges.

One record per cell of Supplementary Table 4's S-score sheet: a two-leaf genotype of
gene-level bacterial perturbation objects served as ``bacterial perturbation`` nodes, and
the signed eSGA S score as a ``GeneInteractionPhenotype``. The query leaf is always a
``BacterialDeletionPerturbation``; the recipient is one too for a Keio row and a
``BacterialMarkedAllelePerturbation`` for one of the 149 ``SPA-tag essential`` rows
(issue #792), which is a leaf the base adapter already serves under the same graph class
(``BACTERIAL_PERTURBATION_LEAVES``, PR #837) with the same five declared properties. No
new graph class is needed: ``gene interaction phenotype`` is already a served class
(declared by the yeast interaction datasets and consumed by the sibling Babu 2014
adapter), and the leaves differ in their ``collection``, their ``cassette``, the Keio
isolate carried as the construction ``batch`` and the SPA tag, all of which the
``bacterial perturbation`` id rule already covers. The screen is a single condition, so
the medium and the temperature are single entities the whole dataset joins on and the
environment carries no perturbation; the two marker drugs are ``selection_agent``
components of that medium rather than small-molecule edits, so the
environment-perturbation pair stays off. No phage and no CRISPRi construct appears in
any record.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.butland2008 import GeneInteractionButland2008Dataset


class GeneInteractionButland2008Adapter(CellAdapter):
    """Cell adapter that serves GeneInteractionButland2008Dataset to BioCypher."""

    def __init__(
        self,
        dataset: GeneInteractionButland2008Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "gene_interaction_butland2008_adapter.yaml"
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
