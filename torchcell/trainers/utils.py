"""Pydantic helpers for indexing models by valid Python identifier names."""

from typing import Any

from pydantic import BaseModel, ConstrainedStr, validator


class VarNameStr(ConstrainedStr):
    """Constrained string that must be a valid Python identifier."""

    @classmethod
    def validate(cls, v: str) -> str:
        """Return the value if it is a valid identifier, else raise ValueError."""
        if not v.isidentifier():
            raise ValueError("String is not a valid Python identifier")
        return v


class ModelIndex(BaseModel):
    """Mapping of identifier-named keys to models, accessible as attributes."""

    __root__: dict[VarNameStr, Any]

    @validator("__root__", pre=True, each_item=True)
    def check_keys(cls, v, field, values, **kwargs):
        """Validate that each root value is a valid Python identifier."""
        if not v.isidentifier():
            raise ValueError(f"Invalid attribute name: {v}")
        return v

    def __getattr__(self, item):
        """Return the root entry stored under the given attribute name."""
        return self.__root__[item]


# Example usage
try:
    model_index = ModelIndex(
        __root__={
            "dcell": "some_value",  # Replace with your actual object
            "dcell_linear": "another_value",
            "1invalid": "this_will_fail",
        }
    )
except ValueError as e:
    print(e)  # Will raise an error for '1invalid'

# Accessing attributes
print(model_index.dcell)
print(model_index.dcell_linear)
