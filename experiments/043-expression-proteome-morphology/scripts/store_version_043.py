# experiments/043-expression-proteome-morphology/scripts/store_version_043.py
# [[experiments.043-expression-proteome-morphology.scripts.store_version_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/store_version_043
"""One switch for the 043 store generation: every path that depends on the raw store.

Two store generations exist side by side, and nothing of one overwrites the other:

* ``v1``: ``raw_043`` (job 5856), built with the gene-set clause, which dropped 188
  served records (145 ``ReleasedIdentifierPerturbation`` records of Messner 2023 and
  Nadal-Ribelles 2025, 43 whose ORFs are outside the gene set). Its representation,
  ``global_split_v1``, the linear evals of ``results/linear_evals_043.json`` and
  ``prep_v2`` are the inputs of every run up to 2026-10-23.
* ``v2``: ``raw_043_v2`` (2026-10-23), built with every served record admitted (no
  gene-set clause; the genome never gates a record), and everything downstream of it
  under ``_v2`` names, ``global_split_v2`` and ``prep_v3``.

The experiment scripts read the generation from the ``STORE_043_VERSION`` environment
variable (the slurm scripts of the v2 chain export ``STORE_043_VERSION=v2``); the trainer
reads ``data.store_version`` of ``conf/train_043_default.yaml``. An unknown value raises.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict

StoreVersion = Literal["v1", "v2"]
ENV_VAR = "STORE_043_VERSION"
#: The scripts' default generation when ``STORE_043_VERSION`` is unset: v1, so every
#: committed v1 result regenerates from its script unchanged.
SCRIPT_DEFAULT: StoreVersion = "v1"


class StorePaths043(BaseModel):
    """The names (never absolute paths) one store generation uses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: StoreVersion
    raw_tag: str  # $DATA_ROOT/data/torchcell/experiments/<EXPERIMENT>/<raw_tag>
    representation_tag: str  # sibling of raw_tag
    split_version: str  # conf/splits/<split_version>.json
    prep_version: str  # train_043_prep/<prep_version>_route<R>
    linear_evals_cache: str  # sibling of raw_tag
    suffix: str  # appended to results/<stem>.json names ("" for v1)

    def result_name(self, stem: str, ext: str = "json") -> str:
        """``results/<stem><suffix>.<ext>``, e.g. ``linear_evals_043_v2.json``."""
        return f"{stem}{self.suffix}.{ext}"


STORE_VERSIONS: dict[StoreVersion, StorePaths043] = {
    "v1": StorePaths043(
        version="v1",
        raw_tag="raw_043",
        representation_tag="representation_043",
        split_version="global_split_v1",
        prep_version="prep_v2",
        linear_evals_cache="linear_evals_cache",
        suffix="",
    ),
    "v2": StorePaths043(
        version="v2",
        raw_tag="raw_043_v2",
        representation_tag="representation_043_v2",
        split_version="global_split_v2",
        prep_version="prep_v3",
        linear_evals_cache="linear_evals_cache_v2",
        suffix="_v2",
    ),
}


def store_paths(version: str) -> StorePaths043:
    """The names of one generation; an unknown version raises."""
    if version not in STORE_VERSIONS:
        raise ValueError(
            f"store version {version!r} is not one of {sorted(STORE_VERSIONS)}"
        )
    return STORE_VERSIONS[version]  # type: ignore[index]


def active() -> StorePaths043:
    """The generation named by ``STORE_043_VERSION`` (``SCRIPT_DEFAULT`` when unset)."""
    return store_paths(os.environ.get(ENV_VAR, SCRIPT_DEFAULT))
