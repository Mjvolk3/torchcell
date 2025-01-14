# torchcell/data/hetero_data
# [[torchcell.data.hetero_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/hetero_data
# Test file: tests/torchcell/data/test_hetero_data.py

from torch_geometric.data import HeteroData
from torch import Tensor
import numpy as np
from torch_geometric.typing import SparseTensor
from typing import Any
from collections.abc import Sequence, Mapping
from torch_geometric.typing import TensorFrame


def custom_size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = " " * indent

    # Special handling for nested dictionaries - show their size instead of content
    if isinstance(value, dict):
        if any(isinstance(v, (list, tuple)) for v in value.values()):
            return f"{pad}{key}=dict(len={len(value)})"
        elif len(value) > 4:
            return f"{pad}{key}=dict(len={len(value)})"

    # For all other cases, use the original size_repr logic
    if isinstance(value, Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, Tensor) and getattr(value, "is_nested", False):
        out = str(list(value.to_padded_tensor(padding=0.0).size()))
    elif isinstance(value, Tensor):
        out = str(list(value.size()))
    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, SparseTensor):
        out = str(value.sizes())[:-1] + f", nnz={value.nnz()}]"
    elif isinstance(value, TensorFrame):
        out = f"{value.__class__.__name__}([{value.num_rows}, {value.num_cols}])"
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = "{}"
    elif (
        isinstance(value, Mapping)
        and len(value) == 1
        and not isinstance(list(value.values())[0], Mapping)
    ):
        lines = [custom_size_repr(k, v, 0) for k, v in value.items()]
        out = "{ " + ", ".join(lines) + " }"
    elif isinstance(value, Mapping):
        lines = [custom_size_repr(k, v, indent + 2) for k, v in value.items()]
        out = "{\n" + ",\n".join(lines) + ",\n" + pad + "}"
    else:
        out = str(value)

    key = str(key).replace("'", "")
    return f"{pad}{key}={out}"


def hetero_repr(self) -> str:
    info1 = [custom_size_repr(k, v, 2) for k, v in self._global_store.items()]
    info2 = [custom_size_repr(k, v, 2) for k, v in self._node_store_dict.items()]
    info3 = [custom_size_repr(k, v, 2) for k, v in self._edge_store_dict.items()]
    info = ",\n".join(info1 + info2 + info3)
    info = f"\n{info}\n" if len(info) > 0 else info
    return f"{self.__class__.__name__}({info})"


# Monkey patch the HeteroData class
HeteroData.__repr__ = hetero_repr
