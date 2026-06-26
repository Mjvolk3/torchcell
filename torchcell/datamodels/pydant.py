# torchcell/datamodels/pydantic.py
# [[torchcell.datamodels.pydantic]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/pydantic.py
# Test file: torchcell/datamodels/test_pydantic.py
"""Strict Pydantic base models used across torchcell data schemas."""

from pydantic import BaseModel


class ModelStrict(BaseModel):
    """Base model that forbids extra fields and is immutable."""

    class Config:
        """Pydantic config forbidding extras and freezing instances."""

        extra = "forbid"
        frozen = True


class ModelStrictArbitrary(BaseModel):
    """Strict immutable model that also permits arbitrary field types."""

    class Config:
        """Pydantic config forbidding extras, freezing, allowing arbitrary types."""

        extra = "forbid"
        frozen = True
        arbitrary_types_allowed = True
