"""Pydantic helpers for indexing models by valid Python identifier names.

NOTE: This module targets the pydantic v1 API (``ConstrainedStr``, ``validator``,
``__root__``), which was removed in pydantic v2. It has no importers in the
codebase and raises ``PydanticImportError`` on import under the installed
pydantic v2, so it is effectively dead (zero importers). It is dropped from the
strict mypy gate via a ``[tool.mypy] ignore_errors`` override in pyproject.toml; a
proper fix is migration to pydantic v2 or removal (out of type-only scope).
"""

from typing import Any

from pydantic import BaseModel, ConstrainedStr, validator


class VarNameStr(ConstrainedStr):  # pydantic v1 base, removed in v2
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

    @validator(
        "__root__", pre=True, each_item=True
    )  # pydantic v1 validator API, removed in v2
    def check_keys(
        cls, v: str, field: Any, values: dict[str, Any], **kwargs: Any
    ) -> str:
        """Validate that each root value is a valid Python identifier."""
        if not v.isidentifier():
            raise ValueError(f"Invalid attribute name: {v}")
        return v

    def __getattr__(self, item: str) -> Any:
        """Return the root entry stored under the given attribute name."""
        return self.__root__[item]  # VarNameStr is a str subtype (pydantic v1)


# Example usage
try:
    model_index = ModelIndex(
        # VarNameStr is a str subtype (pydantic v1 __root__ pattern, removed in v2).
        __root__={
            "dcell": "some_value",
            "dcell_linear": "another_value",
            "1invalid": "this_will_fail",
        }
    )
except ValueError as e:
    print(e)  # Will raise an error for '1invalid'

# Accessing attributes
print(model_index.dcell)
print(model_index.dcell_linear)
