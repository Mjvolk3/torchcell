# torchcell/adapters/kuzmin2020_adapter
# [[torchcell.adapters.kuzmin2020_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/kuzmin2020_adapter

"""BioCypher adapter exposing SGD gene-essentiality data as graph nodes and edges."""

import logging
import os.path as osp

import yaml
from biocypher._logger import get_logger
from omegaconf import OmegaConf

from biocypher import BioCypher  # type: ignore[attr-defined]  # biocypher untyped
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.sgd import GeneEssentialitySgdDataset

# logging
# Get the biocypher logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
logger = get_logger("biocypher")
logger.setLevel(logging.ERROR)


class GeneEssentialitySgdAdapter(CellAdapter):
    """CellAdapter that loads the SGD gene-essentiality dataset into BioCypher."""

    def __init__(
        self,
        dataset: GeneEssentialitySgdDataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter YAML config and initialize the CellAdapter.

        Args:
            dataset: The SGD gene-essentiality dataset to adapt.
            process_workers: Number of worker processes for node/edge processing.
            io_workers: Number of worker processes for I/O.
            chunk_size: Number of records processed per chunk.
            loader_batch_size: Batch size used by the data loader.
        """
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(
            current_dir, "conf", "gene_essentiality_sgd_adapter.yaml"
        )

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"GeneEssentialitySgdAdapter initialized with config: {self.config}"
        )  # Debug print
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


if __name__ == "__main__":
    import math
    import multiprocessing as mp
    import os
    import os.path as osp
    from datetime import datetime
    from typing import cast

    from dotenv import load_dotenv

    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    load_dotenv()
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
    BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
    SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")
    num_workers = mp.cpu_count()
    io_workers = math.ceil(0.2 * num_workers)
    process_workers = num_workers - io_workers

    ## Smf
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    dataset = GeneEssentialitySgdDataset(
        root=osp.join(DATA_ROOT, "data/torchcell/gene_essentiality_sgd"),
        scerevisiae_graph=graph,
    )
    adapter = GeneEssentialitySgdAdapter(
        dataset=dataset,
        process_workers=process_workers,
        io_workers=io_workers,
        chunk_size=int(1e4),
        loader_batch_size=int(1e4),
    )
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
