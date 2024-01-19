# torchcell/knowledge_graphs/create_scerevisiae_kg.py
# [[torchcell.knowledge_graphs.create_scerevisiae_kg]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/create_scerevisiae_kg.py
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

# Configure logging
logging.basicConfig(level=logging.INFO, filename="biocypher_warnings.log")
logging.captureWarnings(True)

# Example: Generating a warning
warnings.warn("This is a test warning")

bc = BioCypher()

# Ordered adapters from smallest to largest
adapters = [
    # DmfCostanzo2016Adapter(
    #     dataset=DmfCostanzo2016Dataset(
    #         root="data/torchcell/dmf_costanzo2016_subset_n_100000",
    #         subset_n=100000,
    #         preprocess=None,
    #     )
    # ),
    SmfKuzmin2018Adapter(dataset=SmfKuzmin2018Dataset()),
    # TmfKuzmin2018Adapter(dataset=TmfKuzmin2018Dataset()),
    # DmfKuzmin2018Adapter(dataset=DmfKuzmin2018Dataset()),
    # DmfCostanzo2016Adapter(dataset=DmfCostanzo2016Dataset()),
]

for adapter in adapters:
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())

# Write admin import statement and schema information (for biochatter)
bc.write_import_call()
bc.write_schema_info(as_node=True)

# Print summary
bc.summary()
