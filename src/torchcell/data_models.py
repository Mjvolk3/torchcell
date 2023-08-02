import json
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
from pydantic import BaseModel, ConfigDict, validator


class BaseModelStrict(BaseModel):
    # I am reusing in the library so it could be backed out as a more primitive class type. [[src/torchcell/ncbi/sequence.py]]
    model_config = ConfigDict(frozen=True, extra="forbid")


if __name__ == "__main__":
    pass
