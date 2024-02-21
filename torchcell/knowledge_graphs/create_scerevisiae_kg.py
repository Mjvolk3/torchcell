# torchcell/knowledge_graphs/create_scerevisiae_kg
# [[torchcell.knowledge_graphs.create_scerevisiae_kg]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/create_scerevisiae_kg
# Test file: tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg.py


from biocypher import BioCypher
from torchcell.adapters import (
    SmfCostanzo2016Adapter,
    DmfCostanzo2016Adapter,
    SmfKuzmin2018Adapter,
    DmfKuzmin2018Adapter,
    TmfKuzmin2018Adapter,
)
from torchcell.datasets.scerevisiae import (
    SmfCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
    SmfKuzmin2018Dataset,
    DmfKuzmin2018Dataset,
    TmfKuzmin2018Dataset,
)
import logging
import warnings
import multiprocessing as mp
from datetime import datetime

from biocypher import BioCypher
from dotenv import load_dotenv

from datetime import datetime
import os
import os.path as osp

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")


def main():
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )

    # num_workers = mp.cpu_count()
    num_workers = 10

    adapters = [
        DmfCostanzo2016Adapter(
            dataset=DmfCostanzo2016Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016_sub_10000"),
                subset_n=10000,
            ),
            num_workers=num_workers,
        ),
        SmfKuzmin2018Adapter(
            dataset=SmfKuzmin2018Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/smf_kuzmin2018")
            ),
            num_workers=num_workers,
        ),
        DmfKuzmin2018Adapter(
            dataset=DmfKuzmin2018Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018")
            ),
            num_workers=num_workers,
        ),
        TmfKuzmin2018Adapter(
            dataset=TmfKuzmin2018Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/tmf_kuzmin2018")
            ),
            num_workers=num_workers,
        ),
        SmfCostanzo2016Adapter(
            dataset=SmfCostanzo2016Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
            ),
            num_workers=num_workers,
        ),
    ]

    for adapter in adapters:
        bc.write_nodes(adapter.get_nodes())
        bc.write_edges(adapter.get_edges())

    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    bc.summary()


if __name__ == "__main__":
    main()
