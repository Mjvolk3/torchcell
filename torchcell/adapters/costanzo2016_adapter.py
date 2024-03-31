# torchcell/adapters/costanzo2016_adapter.py
# [[torchcell.adapters.costanzo2016_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/costanzo2016_adapter.py
# Test file: tests/torchcell/adapters/test_costanzo2016_adapter.py

from tqdm import tqdm
import hashlib
import json
from biocypher import BioCypher
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._logger import get_logger
import logging
from typing import Set
from torchcell.datasets.scerevisiae.costanzo2016 import (
    SmfCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
)
from torchcell.adapters.cell_adapter import CellAdapter

# logging
# Get the biocypher logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
logger = get_logger("biocypher")
logger.setLevel(logging.ERROR)


class SmfCostanzo2016Adapter(CellAdapter):
    def __init__(
        self,
        dataset: SmfCostanzo2016Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        super().__init__(
            dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class DmfCostanzo2016Adapter(CellAdapter):
    def __init__(
        self,
        dataset: DmfCostanzo2016Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        super().__init__(
            dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


# class DmfCostanzo2016Adapter(CellAdapter):
#     def __init__(
#         self,
#         dataset: DmfCostanzo2016Dataset,
#         process_workers: int,
#         io_workers: int,
#         chunk_size: int = int(1e4),
#         loader_batch_size: int = int(1e3),
#     ):
#         super().__init__(
#             dataset, process_workers, io_workers, chunk_size, loader_batch_size
#         )
#         self.dataset = dataset
#         self.process_workers = process_workers
#         self.io_workers = io_workers
#         self.chunk_size = chunk_size
#         self.loader_batch_size = loader_batch_size

#     def get_nodes(self):

#         log.info("Running: self._get_experiment_reference_nodes()")
#         yield from self._get_experiment_reference_nodes()
#         log.info("Running: self._get_genome_nodes()")
#         yield from self._get_genome_nodes()
#         log.info("Running: self.get_data_by_type(self._experiment_node)")
#         yield from self.get_data_by_type(self._experiment_node)
#         log.info("Running: self.get_data_by_type(self._genotype_node)")
#         yield from self.get_data_by_type(self._genotype_node)
#         log.info("Running: self.get_data_by_type(self._perturbation_node)")
#         yield from self.get_data_by_type(self._perturbation_node)
#         log.info("Running: self._get_environment_nodes()")
#         yield from self._get_environment_nodes()
#         log.info("Running: self._get_reference_environment_nodes()")
#         yield from self._get_reference_environment_nodes()
#         # OPTIMIZED
#         # yield from self.get_data_by_type(self._media_node)
#         # OPTIMIZED
#         log.info("Running: self._get_media_nodes()")
#         yield from self._get_media_nodes()
#         log.info("Running: self._get_reference_media_nodes()")
#         yield from self._get_reference_media_nodes()
#         # yield from self.get_data_by_type(self._temperature_node)
#         # OPTIMIZED
#         log.info("Running: self._get_temperature_nodes()")
#         yield from self._get_temperature_nodes()
#         log.info("Running: self._get_reference_temperature_nodes()")
#         yield from self._get_reference_temperature_nodes()
#         log.info("Running: self.get_data_by_type(self._phenotype_node)")
#         yield from self.get_data_by_type(self._phenotype_node)
#         log.info("Running: self._get_reference_phenotype_nodes()")
#         yield from self._get_reference_phenotype_nodes()
#         log.info("Running: self._get_dataset_nodes()")
#         yield from self.get_dataset_nodes()

#     def _get_environment_nodes(self) -> list[BioCypherNode]:
#         # HACK - we know we can loop ref for this node type
#         nodes = []
#         seen_node_ids = set()
#         for data in tqdm(self.dataset.experiment_reference_index):
#             environment_id = hashlib.sha256(
#                 json.dumps(data.reference.reference_environment.model_dump()).encode(
#                     "utf-8"
#                 )
#             ).hexdigest()
#             if environment_id not in seen_node_ids:
#                 seen_node_ids.add(environment_id)
#                 media = json.dumps(
#                     data.reference.reference_environment.media.model_dump()
#                 )
#                 node = BioCypherNode(
#                     node_id=environment_id,
#                     preferred_id="environment",
#                     node_label="environment",
#                     properties={
#                         "temperature": data.reference.reference_environment.temperature.value,
#                         "media": media,
#                         "serialized_data": json.dumps(
#                             data.reference.reference_environment.model_dump()
#                         ),
#                     },
#                 )
#                 nodes.append(node)
#         return nodes

#     def _get_media_nodes(self) -> list[BioCypherNode]:
#         # HACK - we know we can loop ref for this node type
#         seen_node_ids = set()
#         nodes = []
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             media_id = hashlib.sha256(
#                 json.dumps(
#                     data.reference.reference_environment.media.model_dump()
#                 ).encode("utf-8")
#             ).hexdigest()
#             if media_id not in seen_node_ids:
#                 seen_node_ids.add(media_id)
#                 name = data.reference.reference_environment.media.name
#                 state = data.reference.reference_environment.media.state
#                 node = BioCypherNode(
#                     node_id=media_id,
#                     preferred_id="media",
#                     node_label="media",
#                     properties={
#                         "name": name,
#                         "state": state,
#                         "serialized_data": json.dumps(
#                             data.reference.reference_environment.media.model_dump()
#                         ),
#                     },
#                 )
#                 nodes.append(node)
#         return nodes

#     def _get_temperature_nodes(self) -> list[BioCypherNode]:
#         # HACK - we know we can loop ref for this node type
#         seen_node_ids = set()
#         nodes = []
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             temperature_id = hashlib.sha256(
#                 json.dumps(
#                     data.reference.reference_environment.temperature.model_dump()
#                 ).encode("utf-8")
#             ).hexdigest()
#             if temperature_id not in seen_node_ids:
#                 seen_node_ids.add(temperature_id)
#                 node = BioCypherNode(
#                     node_id=temperature_id,
#                     preferred_id="temperature",
#                     node_label="temperature",
#                     properties={
#                         "value": data.reference.reference_environment.temperature.value,
#                         "unit": data.reference.reference_environment.temperature.unit,
#                         "serialized_data": json.dumps(
#                             data.reference.reference_environment.temperature.model_dump()
#                         ),
#                     },
#                 )
#                 nodes.append(node)
#         return nodes

#     def get_edges(self):
#         log.info("Running: self.get_reference_dataset_edges()")
#         yield from self.get_reference_dataset_edges()
#         log.info("Running: self.get_data_by_type(self._experiment_dataset_edge)")
#         yield from self.get_data_by_type(self._experiment_dataset_edge)
#         log.info("Running: self._get_reference_experiment_edges()")
#         yield from self._get_reference_experiment_edges()
#         log.info("Running: self.get_data_by_type(self._genotype_experiment_edge)")
#         yield from self.get_data_by_type(self._genotype_experiment_edge)
#         log.info("Running: self.get_data_by_type(self._perturbation_genotype_edges)")
#         yield from self.get_data_by_type(self._perturbation_genotype_edges)
#         log.info("Running: self.get_data_by_type(self._environment_experiment_edges)")
#         yield from self.get_data_by_type(self._environment_experiment_edges)
#         log.info("Running: self._get_environment_experiment_ref_edges()")
#         yield from self._get_environment_experiment_ref_edges()
#         log.info("Running: self.get_data_by_type(self._phenotype_experiment_edges)")
#         yield from self.get_data_by_type(self._phenotype_experiment_edges)
#         # yield from self.get_data_by_type(self._media_environment_edge)  # OPTIMIZED
#         log.info("Running: self._get_media_environment_edges()")
#         yield from self._get_media_environment_edges()
#         # yield from self.get_data_by_type(self._temperature_environment_edge)
#         # OPTIMIZED
#         log.info("Running: self._get_temperature_environment_edges()")
#         yield from self._get_temperature_environment_edges()
#         log.info("Running: self._get_genome_edges()")
#         yield from self._get_genome_edges()  # KEEP
#         log.info("Finished: get_edges")

#     def _get_media_environment_edges(self) -> list[BioCypherEdge]:
#         # HACK Optimized by using reference
#         # We know reference contains all media and envs
#         edges = []
#         seen_media_environment_pairs: Set[tuple] = set()
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             environment_id = hashlib.sha256(
#                 json.dumps(data.reference.reference_environment.model_dump()).encode(
#                     "utf-8"
#                 )
#             ).hexdigest()
#             media_id = hashlib.sha256(
#                 json.dumps(
#                     data.reference.reference_environment.media.model_dump()
#                 ).encode("utf-8")
#             ).hexdigest()
#             media_environment_pair = (media_id, environment_id)
#             if media_environment_pair not in seen_media_environment_pairs:
#                 seen_media_environment_pairs.add(media_environment_pair)
#                 edge = BioCypherEdge(
#                     source_id=media_id,
#                     target_id=environment_id,
#                     relationship_label="media member of",
#                 )
#                 edges.append(edge)
#         return edges

#     def _get_temperature_environment_edges(self) -> list[BioCypherEdge]:
#         # HACK Optimized by using reference
#         # We know reference contain all envs and temps
#         edges = []
#         seen_temperature_environment_pairs: Set[tuple] = set()
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             environment_id = hashlib.sha256(
#                 json.dumps(data.reference.reference_environment.model_dump()).encode(
#                     "utf-8"
#                 )
#             ).hexdigest()
#             temperature_id = hashlib.sha256(
#                 json.dumps(
#                     data.reference.reference_environment.temperature.model_dump()
#                 ).encode("utf-8")
#             ).hexdigest()
#             temperature_environment_pair = (temperature_id, environment_id)
#             if temperature_environment_pair not in seen_temperature_environment_pairs:
#                 seen_temperature_environment_pairs.add(temperature_environment_pair)
#                 edge = BioCypherEdge(
#                     source_id=temperature_id,
#                     target_id=environment_id,
#                     relationship_label="temperature member of",
#                 )
#                 edges.append(edge)
#         return edges


if __name__ == "__main__":
    import os.path as osp
    from dotenv import load_dotenv
    from datetime import datetime
    import os

    ##
    load_dotenv()
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DATA_ROOT = os.getenv("DATA_ROOT")
    BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
    SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")

    ## SMF
    # bc = BioCypher(
    #     output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
    #     biocypher_config_path=BIOCYPHER_CONFIG_PATH,
    #     schema_config_path=SCHEMA_CONFIG_PATH,
    # )
    # dataset = SmfCostanzo2016Dataset(
    #     osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
    # )
    # num_workers = mp.cpu_count()
    # io_workers = math.ceil(0.2 * num_workers)
    # process_workers = num_workers - io_workers
    # adapter = SmfCostanzo2016Adapter(
    #     dataset=dataset,
    #     process_workers=6,
    #     io_workers=4,
    #     chunk_size=int(1e4),
    #     loader_batch_size=int(1e4),
    # )
    # bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())
    # bc.write_import_call()
    # bc.write_schema_info(as_node=True)
    # bc.summary()

    ## DMF
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    # dataset = DmfCostanzo2016Dataset(
    #     root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016")
    # )
    dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016_1e3"),
        subset_n=int(1e3),
    )
    adapter = DmfCostanzo2016Adapter(
        dataset=dataset,
        process_workers=10,
        io_workers=10,
        chunk_size=100,
        loader_batch_size=10,
    )
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    bc.summary()
