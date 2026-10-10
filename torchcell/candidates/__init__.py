# torchcell/candidates/__init__.py
# [[torchcell.candidates.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/candidates/__init__.py

"""The candidate gate: typed verdicts on a dataset row before any loader is written.

See [[plan.dataset-admission-pipeline.2026.10.10]]. ``kg_manifest admit`` stays the last
stage of the dataset-admission program and is unchanged; this package is its first.
"""

from torchcell.candidates.verdict import (
    AggregationRecord,
    CandidateVerdict,
    GateResult,
    verdict_json_schema,
)

__all__ = ["AggregationRecord", "CandidateVerdict", "GateResult", "verdict_json_schema"]
