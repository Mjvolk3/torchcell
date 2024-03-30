# torchcell/datasets/datasets
# [[torchcell.datasets.datasets]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/datasets
# Test file: tests/torchcell/datasets/test_datasets.py

from torchcell.datasets.scerevisiae import (
    SmfCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
    SmfKuzmin2018Dataset,
    DmfKuzmin2018Dataset,
    TmfKuzmin2018Dataset,
)

datasets_register = {SmfCostanzo2016Dataset.__name__: "",
                     DmfCostanzo2016Dataset.__name__: "",
                     SmfKuzmin2018Dataset.__name__: "",
                     DmfKuzmin2018Dataset.__name__: "",
                     TmfKuzmin2018Dataset.__name__: ""}
