# torchcell/knowledge_graphs/create_scerevisiae_kg_small
# [[torchcell.knowledge_graphs.create_scerevisiae_kg_small]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/create_scerevisiae_kg_small.py
# Test file: tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg_small.py


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
from dotenv import load_dotenv
import os
import os.path as osp
from datetime import datetime
import multiprocessing as mp


def main() -> str:
    # Configure logging
    logging.basicConfig(level=logging.INFO, filename="biocypher_warnings.log")
    logging.captureWarnings(True)
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
    SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")
    BIOCYPHER_OUT_PATH = os.getenv("BIOCYPHER_OUT_PATH")

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, BIOCYPHER_OUT_PATH, time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )

    # Ordered adapters from smallest to largest
    adapters = [
        SmfCostanzo2016Adapter(
            dataset=SmfCostanzo2016Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
            ),
            num_workers=mp.cpu_count(),
        ),
        DmfCostanzo2016Adapter(
            dataset=DmfCostanzo2016Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016")
            ),
            num_workers=mp.cpu_count(),
        ),
        SmfKuzmin2018Adapter(
            dataset=SmfKuzmin2018Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/smf_kuzmin2018")
            ),
            num_workers=mp.cpu_count(),
        ),
        DmfKuzmin2018Adapter(
            dataset=DmfKuzmin2018Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018")
            ),
            num_workers=mp.cpu_count(),
        ),
        TmfKuzmin2018Adapter(
            dataset=TmfKuzmin2018Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/tmf_kuzmin2018")
            ),
            num_workers=mp.cpu_count(),
        ),
    ]

    for adapter in adapters:
        bc.write_nodes(adapter.get_nodes())
        bc.write_edges(adapter.get_edges())

    # Write admin import statement and schema information (for biochatter)
    bc.write_import_call()
    bc.write_schema_info(as_node=True)

    bc.summary()
    # Returns bash script path

    relative_bash_script_path = osp.join(
        "biocypher-out", time, "neo4j-admin-import-call.sh"
    )
    return relative_bash_script_path


if __name__ == "__main__":
    print(main())
