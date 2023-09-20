# src/torchcell/datamodels/pydantic.py
# [[src.torchcell.datamodels.pydantic]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datamodels/pydantic.py
# Test file: src/torchcell/datamodels/test_pydantic.py

import json
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
from pydantic import BaseModel, ConfigDict, Extra


class ModelStrict(BaseModel):
    class Config:
        extra = Extra.forbid
        frozen = True


class ModelStrictArbitrary(BaseModel):
    class Config:
        extra = Extra.forbid
        frozen = True
        arbitrary_types_allowed = True


if __name__ == "__main__":

    class Model(BaseModelStrict):
        a: str

    try:
        # This will raise an error because of the extra field 'b'
        model = Model(a="a", b="b")
    except Exception as e:
        print(f"Error: {e}")

    try:
        # Create a valid model instance
        model = Model(a="a")
        # Try to modify an attribute (this will raise an error because the model is frozen)
        model.a = "new value"
    except Exception as e:
        print(f"Error: {e}")
