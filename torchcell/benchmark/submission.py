# torchcell/benchmark/submission.py
# [[torchcell.benchmark.submission]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/submission.py
# Test file: tests/torchcell/benchmark/test_submission.py

"""The submission contract: what a benchmark upload must look like.

A submission is two things. The predictions file is a CSV in long format with exactly
the columns :data:`PREDICTION_COLUMNS`, one row per (record, target) pair, covering
every validation and test pair of the dataset's public template. The metadata is a JSON
object in the shape of :class:`SubmissionMetadata`. Both are pydantic models so a
submitter can validate locally with the same code the server runs, and the JSON Schema
served at ``/submission-schema`` is generated from them, never typed by hand.

Predictions are submitted, never scores: the server computes every metric.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

PREDICTION_COLUMNS: tuple[str, ...] = ("record_id", "split", "target", "prediction")
MAX_HYPERPARAMETERS = 64
# A finite but enormous prediction overflows the squared-error sums into inf and nan,
# which are not valid JSON; no label is anywhere near this size.
MAX_ABS_PREDICTION = 1e12


class Split(StrEnum):
    """The three splits of a benchmark dataset."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


SCORED_SPLITS: tuple[Split, ...] = (Split.VAL, Split.TEST)

Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:|+\-]+$"),
]
ShortText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
]
LongText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]
HyperparameterKey = Annotated[str, StringConstraints(min_length=1, max_length=64)]
HyperparameterString = Annotated[str, StringConstraints(max_length=256)]


class PredictionRow(BaseModel):
    """One row of the predictions CSV: a predicted value for one target of one record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: Identifier = Field(
        description="Record id exactly as the dataset's template.csv lists it."
    )
    split: Literal[Split.VAL, Split.TEST] = Field(
        description="The split the template assigns to this record (val or test)."
    )
    target: Identifier = Field(description="Target name, one of the dataset's targets.")
    prediction: float = Field(
        allow_inf_nan=False,
        ge=-MAX_ABS_PREDICTION,
        le=MAX_ABS_PREDICTION,
        description="The predicted value, in label units; finite and at most 1e12 in size.",
    )


DataScope = Literal["split_only", "torchcell_db", "external"]
"""What a method was trained or conditioned on beyond its encodings; see the field."""


class SubmissionMetadata(BaseModel):
    """What the leaderboard shows about a method, sent as JSON beside the predictions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method_name: ShortText = Field(description="Short name shown on the leaderboard.")
    description: LongText = Field(description="What the method does, in plain words.")
    model_family: ShortText = Field(
        description="Model class, for example 'kNN', 'ridge', 'random forest', 'GNN'."
    )
    encoding: ShortText = Field(
        description="Input representation, for example 'one-hot gene' or 'ESM2'."
    )
    code_url: AnyHttpUrl | None = Field(
        default=None,
        description="Public code that reproduces the predictions; needed to be verified.",
    )
    data_scope: DataScope = Field(
        description=(
            "What the method was trained or conditioned on, beyond the encodings. "
            "split_only: the dataset's train split and nothing else (inductive). "
            "torchcell_db: other records of the TorchCell database as well, for "
            "example another dataset's measurements as input (transductive). "
            "external: measurements from outside the TorchCell database. Encoders of "
            "sequence or genes and mechanistic simulators are not data for this field."
        )
    )
    data_description: LongText | None = Field(
        default=None,
        description="What the additional data is; required unless data_scope is split_only.",
    )
    hyperparameters: dict[
        HyperparameterKey, HyperparameterString | int | float | bool
    ] = Field(
        default_factory=dict,
        max_length=MAX_HYPERPARAMETERS,
        description="The settings selected on the validation split.",
    )

    @model_validator(mode="after")
    def _additional_data_is_described(self) -> SubmissionMetadata:
        if self.data_scope != "split_only" and self.data_description is None:
            raise ValueError(
                f"data_description is required when data_scope is {self.data_scope}"
            )
        return self


def submission_json_schema() -> dict[str, Any]:
    """The JSON Schema of the row and metadata models plus the CSV column order."""
    return {
        "columns": list(PREDICTION_COLUMNS),
        "prediction_row": PredictionRow.model_json_schema(),
        "metadata": SubmissionMetadata.model_json_schema(),
    }
