from typing import Any, List, Optional

from pydantic import BaseModel, validator


class BaseModelStrict(BaseModel):
    class Config:
        extra = "forbid"


if __name__ == "__main__":
    pass
