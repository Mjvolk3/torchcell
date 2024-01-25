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

# Configure logging
logging.basicConfig(level=logging.INFO, filename="biocypher_warnings.log")
logging.captureWarnings(True)


def main():
    # logger.info(f"Started at {datetime.now()}") but use logging
    bc = BioCypher()

    # num_workers = mp.cpu_count()
    num_workers = 4

    # Ordered adapters from smallest to largest
    adapters = [
        DmfCostanzo2016Adapter(
            dataset=DmfCostanzo2016Dataset(), num_workers=num_workers
        ),
        SmfKuzmin2018Adapter(dataset=SmfKuzmin2018Dataset(), num_workers=num_workers),
        DmfKuzmin2018Adapter(dataset=DmfKuzmin2018Dataset(), num_workers=num_workers),
        TmfKuzmin2018Adapter(dataset=TmfKuzmin2018Dataset(), num_workers=num_workers),
        SmfCostanzo2016Adapter(
            dataset=SmfCostanzo2016Dataset(), num_workers=num_workers
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


if __name__ == "__main__":
    main()
