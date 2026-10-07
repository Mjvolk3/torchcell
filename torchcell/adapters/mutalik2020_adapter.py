# torchcell/adapters/mutalik2020_adapter.py
# [[torchcell.adapters.mutalik2020_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/mutalik2020_adapter.py
# Test file: tests/torchcell/adapters/test_mutalik2020_adapter.py
"""BioCypher adapter exposing the Mutalik 2020 phage-resistance RB-TnSeq screen as
knowledge-graph nodes and edges.

One record per (gene, experiment): a single ``TransposonInsertionPerturbation`` served as
a ``bacterial perturbation`` node, and the gene fitness as an
``EnvironmentResponsePhenotype``. The only environment perturbation this row carries is
the challenge's ``PhagePerturbation``, so the conf enables ``phage perturbation`` and NOT
``environment perturbation``: the served ``_environment_perturbation_node`` does not
filter phages out, so enabling both would write every phage twice, under two labels on
one content id (``cell_adapter.py``, the comment above
``_phage_perturbation_node_from``). Nothing is lost by the choice -- the kanamycin of the
solid-agar assays is a plate ingredient on the medium, not a dosed perturbation, and the
release's own metadata carries it as the ``LB_agar`` recipe's ``Condition_2``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.mutalik2020 import PhageRbTnseqMutalik2020Dataset


class PhageRbTnseqMutalik2020Adapter(CellAdapter):
    """Cell adapter that serves PhageRbTnseqMutalik2020Dataset to BioCypher."""

    def __init__(
        self,
        dataset: PhageRbTnseqMutalik2020Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "phage_rbtnseq_mutalik2020_adapter.yaml"
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
