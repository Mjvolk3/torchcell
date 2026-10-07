# torchcell/artifacts/ref.py
# [[torchcell.artifacts.ref]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/artifacts/ref.py
# Test file: tests/torchcell/artifacts/test_artifact_ref.py

"""``ArtifactRef``: the one pointer type from a graph record to bytes kept off the graph.

The class is DEFINED in ``torchcell.datamodels.schema`` because record classes carry it
(``SequenceVariantPerturbation.sequence_ref``, ``CrisprConstruct.effector_plasmid_ref``,
...), so its contract belongs to the fingerprinted schema surface, and because defining it
here would make ``schema`` import this package while this package's resolver imports the
dataset loaders that import ``schema`` (a cycle). This module re-exports it under the
names the artifact tier uses. See ``ArtifactRef`` for the string form
``tc://<tier>/<key>/<path>[#<member>]``.
"""

from torchcell.datamodels.schema import (
    ARTIFACT_MEMBER_SEPARATOR,
    ARTIFACT_TIERS,
    ARTIFACT_URI_SCHEME,
    ArtifactRef,
    ArtifactTier,
)

Tier = ArtifactTier
TIERS: tuple[str, ...] = ARTIFACT_TIERS
URI_SCHEME = ARTIFACT_URI_SCHEME
MEMBER_SEPARATOR = ARTIFACT_MEMBER_SEPARATOR

__all__ = ["MEMBER_SEPARATOR", "TIERS", "URI_SCHEME", "ArtifactRef", "Tier"]
