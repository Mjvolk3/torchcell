# tests/torchcell/datamodels/_essentiality_evidence.py
"""A minimal valid ``GeneEssentialityEvidence`` for tests that build essentiality
phenotypes but are not about the evidence itself (#924).

``GeneEssentialityPhenotype.evidence`` is required, so every test that constructs the
phenotype needs one. The values are placeholders bound to a fake source, never a real
quote: tests about a loader's evidence use that loader's own factory.
"""

from __future__ import annotations

from typing import Any

from torchcell.datamodels.schema import (
    EssentialityEvidenceClass,
    EssentialityEvidenceType,
    EssentialityStatement,
    EssentialityStatementRole,
    GeneEssentialityEvidence,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

FAKE_SHA256 = "0" * 64


def statement(value: Any = "rule", quote: str = "a quoted rule") -> SourcedValue:
    """A ``SourcedValue`` on a fake, hash-pinned source."""
    return SourcedValue(
        value=value,
        quote=quote,
        provenance=Provenance(
            source_uri="paper.md", citation_key="testKey2026", sha256=FAKE_SHA256
        ),
    )


def statements(
    *extra: tuple[EssentialityStatementRole, SourcedValue],
) -> list[EssentialityStatement]:
    """One scoring rule and one call definition, plus ``extra`` (role, quote) pairs."""
    pairs = [
        (EssentialityStatementRole.scoring_rule, statement()),
        (
            EssentialityStatementRole.call_definition,
            statement("definition", "a quoted definition"),
        ),
        *extra,
    ]
    return [EssentialityStatement(role=role, provenance=value) for role, value in pairs]


def curated_evidence(experiment_type: str | None = None) -> GeneEssentialityEvidence:
    """Curated-null-phenotype evidence; ``experiment_type`` on an experiment only."""
    return GeneEssentialityEvidence(
        evidence_class=EssentialityEvidenceClass.curated_annotation,
        evidence_type=EssentialityEvidenceType.curated_null_phenotype,
        statements=statements(),
        curated_experiment_type=experiment_type,
    )
