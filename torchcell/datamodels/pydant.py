# torchcell/datamodels/pydantic.py
# [[torchcell.datamodels.pydantic]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/pydantic.py
# Test file: torchcell/datamodels/test_pydantic.py

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
