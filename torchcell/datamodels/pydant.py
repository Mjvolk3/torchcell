# torchcell/datamodels/pydantic.py
# [[torchcell.datamodels.pydantic]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/pydantic.py
# Test file: torchcell/datamodels/test_pydantic.py
from pydantic import BaseModel


class ModelStrict(BaseModel):
    class Config:
        extra = "forbid"
        frozen = True


class ModelStrictArbitrary(BaseModel):
    class Config:
        extra = "forbid"
        frozen = True
        arbitrary_types_allowed = True
