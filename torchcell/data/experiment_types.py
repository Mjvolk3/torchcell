# torchcell/data/experiment_types.py
# [[torchcell.data.experiment_types]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/experiment_types
# Test file: tests/torchcell/data/test_experiment_types.py

"""The one place a stored experiment pair is mapped back to its pydantic classes.

``torchcell.data.aggregate`` and ``torchcell.data.deduplicate`` read every record of a dev
LMDB through ``EXPERIMENT_TYPE_MAP[experiment_type]`` and
``EXPERIMENT_REFERENCE_TYPE_MAP[experiment_reference_type]``. A store written before the
ontology unification (P2) carries one of the 14 retired tags (``bacterial_fitness``,
``yeast_protein_turnover``, ...), which the maps no longer hold: such a store is
UNREADABLE until it is rebuilt, not merely stale. This module makes that failure say so,
naming the retired tag and the tag its records now carry
(``schema.RETIRED_EXPERIMENT_TYPES``), instead of a bare ``KeyError``. It never maps a
retired tag onto its successor for the caller: the record's class, and so its id in the
graph, is decided by a rebuild, never by a reader.
"""

from __future__ import annotations

from typing import Any

from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
    RETIRED_EXPERIMENT_TYPES,
    Experiment,
    ExperimentReference,
)


class RetiredExperimentTypeError(KeyError):
    """A stored record carries an experiment-type tag the schema retired."""


def _lookup(tag: str, table: dict[str, Any], field: str) -> Any:
    try:
        return table[tag]
    except KeyError:
        if tag in RETIRED_EXPERIMENT_TYPES:
            raise RetiredExperimentTypeError(
                f"{field} {tag!r} was retired by the ontology unification (its records "
                f"are now {RETIRED_EXPERIMENT_TYPES[tag]!r}); this store was written "
                "under the old tag and is unreadable until it is rebuilt "
                "(torchcell.database.build_dataset_lmdb)"
            ) from None
        raise KeyError(f"unknown {field} {tag!r}") from None


def experiment_class(tag: str) -> type[Experiment]:
    """The experiment class stored under ``experiment_type`` ``tag``."""
    cls = _lookup(tag, EXPERIMENT_TYPE_MAP, "experiment_type")
    assert isinstance(cls, type) and issubclass(cls, Experiment)
    return cls


def experiment_reference_class(tag: str) -> type[ExperimentReference]:
    """The reference class stored under ``experiment_reference_type`` ``tag``."""
    cls = _lookup(tag, EXPERIMENT_REFERENCE_TYPE_MAP, "experiment_reference_type")
    assert isinstance(cls, type) and issubclass(cls, ExperimentReference)
    return cls


def experiment_pair(
    json_pair: dict[str, Any], *, reference_key: str = "experiment_reference"
) -> dict[str, Experiment | ExperimentReference]:
    """Rebuild one stored ``{experiment, experiment_reference}`` pair as pydantic models."""
    experiment = json_pair["experiment"]
    reference = json_pair[reference_key]
    return {
        "experiment": experiment_class(experiment["experiment_type"])(**experiment),
        "experiment_reference": experiment_reference_class(
            reference["experiment_reference_type"]
        )(**reference),
    }
