# torchcell/adapters/choe2019_tf_knockout_adapter.py
# [[torchcell.adapters.choe2019_tf_knockout_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/choe2019_tf_knockout_adapter.py
# Test file: tests/torchcell/adapters/test_choe2019_tf_knockout_adapter.py
"""BioCypher adapter exposing Choe 2019's two Keio transcription-factor knockouts as
knowledge-graph nodes and edges.

One record per strain: a single ``BacterialDeletionPerturbation`` on the BW25113
locus-tag namespace, served as ``bacterial perturbation``, and the growth rate the
Supplementary Fig. 6 legend states as a fraction of wild-type BW25113 in the same
medium. One medium at one temperature with no added compound, so the
environment-perturbation pair is off.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.choe2019_growth_rate import (
    TranscriptionFactorKnockoutChoe2019Dataset,
)


class TranscriptionFactorKnockoutChoe2019Adapter(CellAdapter):
    """Cell adapter that serves TranscriptionFactorKnockoutChoe2019Dataset."""

    def __init__(
        self,
        dataset: TranscriptionFactorKnockoutChoe2019Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "tf_knockout_growth_choe2019_adapter.yaml"
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
