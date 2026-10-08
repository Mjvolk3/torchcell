# torchcell/adapters/mohiuddin2022_adapter
# [[torchcell.adapters.mohiuddin2022_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/mohiuddin2022_adapter
# Test file: tests/torchcell/adapters/test_mohiuddin2022_adapter.py
"""BioCypher adapter for the Mohiuddin 2022 promoter-GFP reporter library.

RECORD = one (arm, plate, well, read hour) ``PromoterActivityExperiment``, so the
served nodes are the new ``promoter activity phenotype`` class plus the shared
experiment, genotype, environment, media, temperature, dataset and publication classes.

Both optional pairs are on. The genotype carries one ``HeterologousPathwayPerturbation``
(the episomal promoter-GFP reporter), which is a namespaced bacterial leaf, so
``bacterial perturbation`` and ``perturbation to genotype`` are enabled. The three
treated arms carry the antibiotic as a ``SmallMoleculePerturbation`` on their reads from
hour five on, so the environment-perturbation pair is enabled; the untreated arm and the
pre-dose reads carry none, which the environment-perturbation method simply does not
emit for. No CRISPR construct and no phage.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.mohiuddin2022 import PromoterReporterMohiuddin2022Dataset


class PromoterReporterMohiuddin2022Adapter(CellAdapter):
    """Cell adapter that serves PromoterReporterMohiuddin2022Dataset to BioCypher."""

    def __init__(
        self,
        dataset: PromoterReporterMohiuddin2022Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "promoter_reporter_mohiuddin2022_adapter.yaml"
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
