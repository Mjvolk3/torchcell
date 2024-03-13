---
id: 03ga1ybde0b0dxjhwg48w3z
title: models
desc: ''
updated: 1690661498578
created: 1690658956502
---
[Models](https://docs.pydantic.dev/latest/usage/models/)

## Required Fields

[Required Fields](https://docs.pydantic.dev/latest/usage/models/#required-fields)

### Required Fields - mypy compatibility

> Here a, b and c are all required. However, this use of `b: int = ...` does not work properly with mypy, and as of v1.0 should be avoided in most cases.

```python
from pydantic import BaseModel, Field


class Model(BaseModel):
    a: int
    b: int = ... # Dont use, mypy incompatible
    c: int = Field(...) # Would only use for alias specification
```

### Required Fields - Note

> In Pydantic V1, fields annotated with Optional or Any would be given an implicit default of None even if no default was explicitly specified. This behavior has changed in Pydantic V2, and there are no longer any type annotations that will result in a field having an implicit default value.

## Fields with non-hashable default values

[Fields with non-hashable default values](https://docs.pydantic.dev/latest/usage/models/#fields-with-non-hashable-default-values)

```python
class Model(BaseModel):
    item_counts: List[Dict[str, int]] = [{}] # Can initialize a dict with this

m1 = Model()
m1.item_counts[0]['a'] = 1
print(m1.item_counts)
#> [{'a': 1}]

m2 = Model()
print(m2.item_counts) # Fresh list of dict
#> [{}]
```
