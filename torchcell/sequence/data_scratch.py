# torchcell/sequence/data_scratch.py
# [[torchcell.sequence.data_scratch]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/data_scratch.py
# Test file: torchcell/sequence/test_data_scratch.py


import logging
from abc import ABC, abstractmethod
from turtle import st
from typing import Set

import gffutils
import matplotlib.pyplot as plt
import pandas as pd
from attrs import define, field
from Bio import SeqIO
from Bio.Seq import Seq
from gffutils import Feature, FeatureDB
from gffutils.biopython_integration import to_seqfeature
from matplotlib import pyplot as plt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from sympy import sequence

from torchcell.data_models import BaseModelStrict

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

if __name__ == "__main__":
    pass
