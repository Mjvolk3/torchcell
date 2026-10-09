# torchcell/adapters/lim2025_proteome_fold_change_adapter.py
# [[torchcell.adapters.lim2025_proteome_fold_change_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/lim2025_proteome_fold_change_adapter.py
# Test file: tests/torchcell/adapters/test_lim2025_proteome_fold_change_adapter.py
"""BioCypher adapter exposing Lim 2025's evolved-isolate-over-IPL400 proteome contrasts
as knowledge-graph nodes and edges.

One record per released ``Proteome_<isolate>vsIPL400_<medium>`` sheet: the numerator
strain's seven designed ``BacterialDeletionPerturbation`` leaves served as
``bacterial perturbation``, its called variants served as
``bacterial sequence variant perturbation`` (issue #731), and the per-protein log2 fold
changes with their unadjusted and BH-adjusted p-values as a
``ProteinFoldChangePhenotype``. The phenotype node class is ``protein fold change
phenotype``, NOT ``protein abundance phenotype``: a ratio and an absolute level are
different measurements and must never pool, which is why this dataset is a sibling of
``ProteomeLim2025Dataset`` rather than a mode of it.

Two of the four records carry isoprenol as a ``SmallMoleculePerturbation`` on the
environment and two are the unstressed condition, so the environment-perturbation pair is
enabled and emits nothing for the latter two. Every record gaps temperature, so the
temperature methods emit nothing. No phage and no CRISPRi construct appears in any
record.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.lim2025 import ProteomeFoldChangeLim2025Dataset


class ProteomeFoldChangeLim2025Adapter(CellAdapter):
    """Cell adapter that serves ProteomeFoldChangeLim2025Dataset to BioCypher."""

    def __init__(
        self,
        dataset: ProteomeFoldChangeLim2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "proteome_fold_change_lim2025_adapter.yaml"
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
