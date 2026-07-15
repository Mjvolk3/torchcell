# tests/torchcell/verification/test_visual_score_verification.py
"""Unit tests for the WS7 visual-score verifier + VisualScorePhenotype (synthetic).

Builds records from the real pydantic models and checks the passing case plus each
failure mode the verifier is responsible for (duplicate ORF, non-zero reference,
missing target_product), and the schema validator's own guards (out-of-scale, bad
scale ordering, n_replicates).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ReferenceGenome,
    Temperature,
    VisualScoreExperiment,
    VisualScoreExperimentReference,
    VisualScorePhenotype,
)
from torchcell.verification.report import Level, Provenance
from torchcell.verification.visual_score import (
    verify_visual_score_dataset,
    visual_score_gene_set,
)

PROV = Provenance(source_uri="test://synthetic", citation_key="test2013")
GENES = ["YBR270C", "YCR106W", "YDR260C"]


def _phenotype(score: float, *, ref: bool = False) -> VisualScorePhenotype:
    return VisualScorePhenotype(
        visual_score=score,
        n_replicates=1 if ref else 2,
        score_scale_min=-5,
        score_scale_max=5,
        score_semantics="higher = more orange = more carotenoid",
        target_product="beta-carotene",
    )


def _record(gene: str, score: float, *, ref_score: float = 0.0) -> dict[str, Any]:
    env = Environment(
        media=Media(name="SC-URA", state="solid", is_synthetic=True),
        temperature=Temperature(value=30),
    )
    experiment = VisualScoreExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=env,
        phenotype=_phenotype(score),
    )
    reference = VisualScoreExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=_phenotype(ref_score, ref=True),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _good_records() -> list[dict[str, Any]]:
    return [_record(g, s) for g, s in zip(GENES, [5.0, -3.0, 2.0])]


# --- schema validator guards ------------------------------------------------- #
def test_out_of_scale_score_rejected():
    with pytest.raises(ValidationError):
        _phenotype(9.0)


def test_bad_scale_ordering_rejected():
    with pytest.raises(ValidationError):
        VisualScorePhenotype(
            visual_score=0.0,
            n_replicates=1,
            score_scale_min=5,
            score_scale_max=-5,
            score_semantics="x",
            target_product="beta-carotene",
        )


def test_zero_replicates_rejected():
    with pytest.raises(ValidationError):
        VisualScorePhenotype(
            visual_score=0.0,
            n_replicates=0,
            score_scale_min=-5,
            score_scale_max=5,
            score_semantics="x",
            target_product="beta-carotene",
        )


# --- verifier ---------------------------------------------------------------- #
def test_good_dataset_passes_all_levels():
    report = verify_visual_score_dataset(
        _good_records(), dataset_name="good", provenance=PROV, expected_count=3
    )
    assert report.passed, report.summary()
    assert {Level.L0, Level.L1, Level.L2, Level.L3} <= report.levels_covered


def test_duplicate_orf_fails_uniqueness():
    records = _good_records()
    records.append(_record("YBR270C", 4.0))  # duplicate ORF
    report = verify_visual_score_dataset(
        records, dataset_name="dup", provenance=PROV, expected_count=4
    )
    u = [r for r in report.results if r.name == "orf_uniqueness"]
    assert u and not u[0].passed


def test_nonzero_reference_fails():
    records = _good_records()
    records.append(_record("YEL060C", 3.0, ref_score=2.0))
    report = verify_visual_score_dataset(
        records, dataset_name="badref", provenance=PROV, expected_count=4
    )
    r = [x for x in report.results if x.name == "reference_zero"]
    assert r and not r[0].passed


def test_wrong_count_fails():
    report = verify_visual_score_dataset(
        _good_records(), dataset_name="miscount", provenance=PROV, expected_count=99
    )
    c = [r for r in report.results if r.level is Level.L1 and r.name == "count"]
    assert c and not c[0].passed


def test_visual_score_gene_set():
    assert visual_score_gene_set(_good_records()) == set(GENES)
