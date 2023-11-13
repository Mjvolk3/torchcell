from typing import Any, Dict

from pydantic import BaseModel, ConstrainedStr, validator


class VarNameStr(ConstrainedStr):
    @classmethod
    def validate(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError("String is not a valid Python identifier")
        return v


class ModelIndex(BaseModel):
    __root__: dict[VarNameStr, Any]

    @validator("__root__", pre=True, each_item=True)
    def check_keys(cls, v, field, values, **kwargs):
        if not v.isidentifier():
            raise ValueError(f"Invalid attribute name: {v}")
        return v

    def __getattr__(self, item):
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
