# src/torchcell/data_models.py
# [[src.torchcell.data_models]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/data_models.py
# Test file: tests/torchcell/test_data_models.py

import json
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
from pydantic import BaseModel, ConfigDict, Extra

class BaseModelStrict(BaseModel):
    class Config:
        extra = Extra.forbid
        frozen = True

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
