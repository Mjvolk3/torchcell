# torchcell/knowledge_graphs/create_scerevisiae_kg.py
# [[torchcell.knowledge_graphs.create_scerevisiae_kg]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/create_scerevisiae_kg.py
# Test file: tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg.py

from biocypher import BioCypher
from torchcell.adapters import (
    SmfCostanzo2016Adapter,
    # DmfCostanzo2016Adapter,
    # SmfKuzmin2018Adapter,
    # DmfKuzmin2018Adapter,
    # TmfKuzmin2018Adapter,
)
from torchcell.datasets.scerevisiae import (
    SmfCostanzo2016Dataset,
    # DmfCostanzo2016Dataset,
    # SmfKuzmin2018Dataset,
    # DmfKuzmin2018Dataset,
    # TmfKuzmin2018Dataset,
)
import logging
import warnings
from dotenv import load_dotenv
import os
import os.path as osp
from datetime import datetime

def main() -> str:
    # Configure logging
    logging.basicConfig(level=logging.INFO, filename="biocypher_warnings.log")
    logging.captureWarnings(True)
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bc = BioCypher(output_directory=osp.join(DATA_ROOT, "biocypher-out", time))
    # Ordered adapters from smallest to largest
    adapters = [
        SmfCostanzo2016Adapter(
            dataset=SmfCostanzo2016Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
            ),
            num_workers=10,
        )
    ]

    for adapter in adapters:
        bc.write_nodes(adapter.get_nodes())
        # bc.write_edges(adapter.get_edges())

    # Write admin import statement and schema information (for biochatter)
    bc.write_import_call()
    bc.write_schema_info(as_node=True)

    bc.summary()
    # Returns bash script path 
    bash_script_path = osp.join(bc._output_directory, "neo4j-admin-import-call.sh")
    return bash_script_path


if __name__ == "__main__":
    main()
