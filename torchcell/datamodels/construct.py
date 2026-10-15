# torchcell/datamodels/construct.py
# [[torchcell.datamodels.construct]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/construct.py
# Test file: tests/torchcell/datamodels/test_construct.py

"""Where added DNA sits: on which construct, at which host site, carrying which variant.

The material layer of a gain-of-function perturbation, as a composed pointer instead of
four loose nullable strings. The classes are DEFINED in
``torchcell.datamodels.schema`` and re-exported here, for the reason ``ArtifactRef`` is
(``torchcell/artifacts/ref.py``): ``GeneAdditionPerturbation.placement`` carries
``ConstructPlacement``, so its contract belongs to the fingerprinted schema surface,
and defining it in a module the construct store reaches would make ``schema`` import
the resolver, which imports the dataset loaders, which import ``schema``. This module
is the name the material layer's own readers use.

What they fix: a ``plasmid_contig_id: str | None`` and a ``locus_tag: str | None`` can
be half-filled and were ``None`` in 26 of 26 served gene-addition stores, while a
``placement`` is either a whole pointer or a typed gap.

Three facts, three objects:

- ``ConstructPlacement`` says WHICH DNA was added: a contig by its sha256 identity, a
  feature of that contig by its key, and the ``ArtifactRef`` that pins the bytes, so a
  stored record verifies itself without the construct index being present.
- ``HostPlacement`` says WHERE in the host it went: the assembly set the coordinates are
  written against, the site's label verbatim, and the three spans an integration has
  (the host interval it landed in, the host interval it replaced, and the interval of
  the CONTIG that was transferred). Absent for an episomal construct, which is a
  replicon of its own and sits at no host site.
- ``ConstructVariantCall`` says WHAT differs from the part the construct was built
  from: a coordinate on the contig, in the shape the resequencing releases already use.
  Cachera 2023's ``ARO4-K229L`` and ``ARO7-G141S`` were free strings
  (``variant="K229L"``), and a free string is not a sequence fact.

Hashability is load-bearing, not incidental. ``Genotype.__eq__`` compares
``set(self.perturbations)``, and a frozen pydantic model hashes the tuple of its field
values, so every object that can hang off a perturbation must hash. Each class here
therefore derives from ``HashableProvenanceGapMixin`` (canonical-JSON hash), which is
also what lets them carry ``provenance_gaps`` at all, and
``GeneAdditionPerturbation`` lists that mixin FIRST in its own bases for the same
reason.

One design choice worth stating, because the obvious alternative is a defect. The
CONTIG-side insert span (the segment between the homology arms that is transferred) is
a single field on ``HostPlacement``, and ``ConstructPlacement.insert_span`` is a
read-only property that reads it. Two fields holding one fact in two classes is what
principle 5 forbids, and the fact belongs with the integration: an episomal construct
has no insert span, because the whole replicon is present.
"""

from torchcell.datamodels.schema import (
    PLACEMENT_TARGET_REFUSED_SO,
    ConstructPlacement,
    ConstructSpan,
    ConstructVariantCall,
    FeatureKey,
    HostPlacement,
    Localization,
    SpanComposition,
    SpanPart,
)

__all__ = [
    "PLACEMENT_TARGET_REFUSED_SO",
    "ConstructPlacement",
    "ConstructSpan",
    "ConstructVariantCall",
    "FeatureKey",
    "HostPlacement",
    "Localization",
    "SpanComposition",
    "SpanPart",
]
