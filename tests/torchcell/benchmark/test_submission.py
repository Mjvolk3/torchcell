# tests/torchcell/benchmark/test_submission.py
# [[tests.torchcell.benchmark.test_submission]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_submission.py
"""``torchcell.benchmark.submission``: the row and metadata contract.

A row is accepted only for the val and test splits, with a finite prediction and an id
made of the allowed characters; lax parsing turns the CSV string ``"0.25"`` into the
float 0.25. Metadata rejects unknown fields, requires the external-data description
when external data is declared, and caps the hyperparameter count at 64.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.benchmark.submission import (
    MAX_HYPERPARAMETERS,
    PREDICTION_COLUMNS,
    SCORED_SPLITS,
    PredictionRow,
    Split,
    SubmissionMetadata,
    submission_json_schema,
)

METADATA: dict[str, Any] = {
    "method_name": "ridge on one-hot",
    "description": "Ridge regression on a one-hot gene encoding.",
    "model_family": "ridge",
    "encoding": "one-hot gene",
    "data_scope": "split_only",
}


def test_columns_and_scored_splits() -> None:
    assert PREDICTION_COLUMNS == ("record_id", "split", "target", "prediction")
    assert SCORED_SPLITS == (Split.VAL, Split.TEST)


def test_row_parses_csv_strings() -> None:
    row = PredictionRow.model_validate(
        {
            "record_id": "YAL001C",
            "split": "val",
            "target": "fitness",
            "prediction": "0.25",
        }
    )
    assert row.split is Split.VAL
    assert row.prediction == 0.25


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "split",
            "train",
            "Input should be <Split.VAL: 'val'> or <Split.TEST: 'test'>",
        ),
        ("prediction", "nan", "Input should be a finite number"),
        ("prediction", "inf", "Input should be a finite number"),
        ("prediction", "", "Input should be a valid number"),
        ("prediction", "1e13", "Input should be less than or equal to 1000000000000"),
        (
            "prediction",
            "-1e300",
            "Input should be greater than or equal to -1000000000000",
        ),
        ("record_id", "", "String should have at least 1 character"),
        ("record_id", "a b", "String should match pattern"),
        ("record_id", "x" * 129, "String should have at most 128 characters"),
    ],
)
def test_row_rejections(field: str, value: str, message: str) -> None:
    data = {
        "record_id": "r1",
        "split": "test",
        "target": "fitness",
        "prediction": "1.0",
    }
    data[field] = value
    with pytest.raises(ValidationError) as caught:
        PredictionRow.model_validate(data)
    errors = caught.value.errors()
    assert [e["loc"] for e in errors] == [(field,)]
    assert message in errors[0]["msg"]


def test_row_forbids_extra_columns() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PredictionRow.model_validate(
            {
                "record_id": "r1",
                "split": "val",
                "target": "fitness",
                "prediction": 1.0,
                "score": 0.9,
            }
        )


def test_metadata_defaults_and_whitespace() -> None:
    metadata = SubmissionMetadata.model_validate(
        {**METADATA, "method_name": "  ridge on one-hot  "}
    )
    assert metadata.method_name == "ridge on one-hot"
    assert metadata.code_url is None
    assert metadata.hyperparameters == {}


def test_metadata_requires_a_description_beyond_split_only() -> None:
    """``torchcell_db`` and ``external`` need ``data_description``; ``split_only`` does
    not; an unknown scope and the boolean the field replaced are refused.
    """
    for scope in ("torchcell_db", "external"):
        with pytest.raises(
            ValidationError,
            match=f"data_description is required when data_scope is {scope}",
        ):
            SubmissionMetadata.model_validate({**METADATA, "data_scope": scope})
    described = SubmissionMetadata.model_validate(
        {
            **METADATA,
            "data_scope": "external",
            "data_description": "STRING v12 protein links.",
        }
    )
    assert described.data_scope == "external"
    assert described.data_description == "STRING v12 protein links."
    assert SubmissionMetadata.model_validate(METADATA).data_description is None
    with pytest.raises(ValidationError, match="data_scope"):
        SubmissionMetadata.model_validate({**METADATA, "data_scope": "all_of_it"})
    with pytest.raises(ValidationError, match="uses_external_data"):
        SubmissionMetadata.model_validate({**METADATA, "uses_external_data": False})


def test_metadata_code_url_must_be_http() -> None:
    ok = SubmissionMetadata.model_validate(
        {**METADATA, "code_url": "https://github.com/someone/method"}
    )
    assert str(ok.code_url) == "https://github.com/someone/method"
    with pytest.raises(ValidationError, match="URL scheme should be 'http' or 'https'"):
        SubmissionMetadata.model_validate({**METADATA, "code_url": "ftp://host/x"})


def test_metadata_hyperparameter_limits() -> None:
    at_limit = {f"k{i}": i for i in range(MAX_HYPERPARAMETERS)}
    assert (
        len(
            SubmissionMetadata.model_validate(
                {**METADATA, "hyperparameters": at_limit}
            ).hyperparameters
        )
        == 64
    )
    with pytest.raises(ValidationError, match="at most 64 items"):
        SubmissionMetadata.model_validate(
            {**METADATA, "hyperparameters": {**at_limit, "one_more": 1}}
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SubmissionMetadata.model_validate({**METADATA, "test_pearson": 0.99})


def test_json_schema_names_the_contract() -> None:
    schema = submission_json_schema()
    assert schema["columns"] == ["record_id", "split", "target", "prediction"]
    assert schema["prediction_row"]["required"] == [
        "record_id",
        "split",
        "target",
        "prediction",
    ]
    assert schema["metadata"]["required"] == [
        "method_name",
        "description",
        "model_family",
        "encoding",
        "data_scope",
    ]
