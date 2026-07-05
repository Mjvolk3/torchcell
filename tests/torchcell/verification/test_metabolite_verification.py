# tests/torchcell/verification/test_metabolite_verification.py
"""Unit tests for the WS8 metabolite verifier + MetabolitePhenotype (synthetic)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    ReferenceGenome,
    Temperature,
)
from torchcell.verification.metabolite import (
    metabolite_gene_set,
    verify_metabolite_dataset,
)
from torchcell.verification.report import Level, Provenance

PROV = Provenance(source_uri="test://synthetic", citation_key="test2023")
GENES = ["YMR056C", "YBR085W", "YJR155W"]
MTYPE = "cri_spa_corrected_fluorescence_intensity_24h"


def _phenotype(level: float, *, ref: bool = False) -> MetabolitePhenotype:
    return MetabolitePhenotype(
        metabolite_level={"betaxanthin": level},
        metabolite_level_se=None if ref else {"betaxanthin": 0.1},
        n_replicates={"betaxanthin": 1 if ref else 8},
        measurement_type=MTYPE,
    )


def _record(gene: str, level: float, *, ref_level: float = 0.0) -> dict[str, Any]:
    env = Environment(
        media=Media(name="SC", state="solid"), temperature=Temperature(value=30)
    )
    experiment = MetaboliteExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=env,
        phenotype=_phenotype(level),
    )
    reference = MetaboliteExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=_phenotype(ref_level, ref=True),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _good_records() -> list[dict[str, Any]]:
    return [_record(g, lv) for g, lv in zip(GENES, [1.5, -0.8, 0.3])]


# --- schema guards ----------------------------------------------------------- #
def test_mismatched_replicate_keys_rejected():
    with pytest.raises(ValidationError):
        MetabolitePhenotype(
            metabolite_level={"betaxanthin": 1.0},
            n_replicates={"lycopene": 3},
            measurement_type=MTYPE,
        )


def test_empty_level_rejected():
    with pytest.raises(ValidationError):
        MetabolitePhenotype(
            metabolite_level={}, n_replicates={}, measurement_type=MTYPE
        )


# --- verifier ---------------------------------------------------------------- #
def test_good_dataset_passes_all_levels():
    report = verify_metabolite_dataset(
        _good_records(), dataset_name="good", provenance=PROV, expected_count=3
    )
    assert report.passed, report.summary()
    assert {Level.L0, Level.L1, Level.L2, Level.L3} <= report.levels_covered


def test_duplicate_orf_fails_uniqueness():
    records = _good_records()
    records.append(_record("YMR056C", 2.0))
    report = verify_metabolite_dataset(
        records, dataset_name="dup", provenance=PROV, expected_count=4
    )
    u = [r for r in report.results if r.name == "orf_uniqueness"]
    assert u and not u[0].passed


def test_nonzero_reference_fails():
    records = _good_records()
    records.append(_record("YDR001C", 1.0, ref_level=0.7))
    report = verify_metabolite_dataset(
        records, dataset_name="badref", provenance=PROV, expected_count=4
    )
    r = [x for x in report.results if x.name == "reference_zero"]
    assert r and not r[0].passed


def test_mixed_measurement_type_fails():
    records = _good_records()
    exp = MetaboliteExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name="YDR002W", perturbed_gene_name="YDR002W"
                )
            ]
        ),
        environment=Environment(
            media=Media(name="SC", state="solid"), temperature=Temperature(value=30)
        ),
        phenotype=MetabolitePhenotype(
            metabolite_level={"betaxanthin": 1.0},
            n_replicates={"betaxanthin": 4},
            measurement_type="ms_abundance",  # different assay
        ),
    )
    ref = _record("YDR002W", 1.0)["reference"]
    records.append({"experiment": exp.model_dump(), "reference": ref})
    report = verify_metabolite_dataset(
        records, dataset_name="mixed", provenance=PROV, expected_count=4
    )
    m = [r for r in report.results if r.name == "measurement_type_consistent"]
    assert m and not m[0].passed


def test_metabolite_gene_set():
    assert metabolite_gene_set(_good_records()) == set(GENES)
