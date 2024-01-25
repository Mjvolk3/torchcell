# torchcell/knowledge_graphs/create_scerevisiae_kg.py
# [[torchcell.knowledge_graphs.create_scerevisiae_kg]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/create_scerevisiae_kg.py
# Test file: tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg.py

from biocypher import BioCypher
from torchcell.pypy_adapters import (
    # SmfCostanzo2016Adapter,
    # DmfCostanzo2016Adapter,
    SmfKuzmin2018Adapter,
    DmfKuzmin2018Adapter,
    TmfKuzmin2018Adapter,
)
from torchcell.dataset_readers import LmdbDatasetReader

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, filename="biocypher_warnings.log")
logging.captureWarnings(True)


def main():
    # logger.info(f"Started at {datetime.now()}") but use logging
    bc = BioCypher()

    # num_workers = mp.cpu_count()
    num_workers = 6

    logging.info(f"Using {num_workers} workers")
    # Ordered adapters from smallest to largest
    adapters = [
        SmfKuzmin2018Adapter(
            dataset=LmdbDatasetReader("data/torchcell/smf_kuzmin2018"),
        ),
        DmfKuzmin2018Adapter(
            dataset=LmdbDatasetReader("data/torchcell/dmf_kuzmin2018"),
        ),
        TmfKuzmin2018Adapter(
           dataset=LmdbDatasetReader("data/torchcell/tmf_kuzmin2018"),
        ),
    ]

    for adapter in adapters:
        bc.write_nodes(adapter.get_nodes())
        bc.write_edges(adapter.get_edges())

    # Write admin import statement and schema information (for biochatter)
    bc.write_import_call()
    bc.write_schema_info(as_node=True)

    # Print summary
    bc.summary()
    # log the finish time
    logging.info(f"Finished at {datetime.now()}")


if __name__ == "__main__":
    main()
