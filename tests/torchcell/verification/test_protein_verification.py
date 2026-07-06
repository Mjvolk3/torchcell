# tests/torchcell/verification/test_protein_verification.py
"""Unit tests for the WS9 protein verifier + ProteinAbundancePhenotype (synthetic)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    ReferenceGenome,
    Temperature,
)
from torchcell.verification.protein import protein_gene_set, verify_protein_dataset
from torchcell.verification.report import Level, Provenance

PROV = Provenance(source_uri="test://synthetic", citation_key="test2018")
GENES = ["YMR056C", "YBR085W", "YJR155W"]
PROTS = ["YAL003W", "YAL005C"]
MTYPE = "swath_ms_label_free_log_signal_sva"


def _phenotype(scale: float) -> ProteinAbundancePhenotype:
    return ProteinAbundancePhenotype(
        protein_abundance={p: scale + i for i, p in enumerate(PROTS)},
        protein_abundance_se={p: 0.1 for p in PROTS},
        n_replicates={p: 3 for p in PROTS},
        measurement_type=MTYPE,
    )


def _record(gene: str, scale: float, *, ref_scale: float = 9.0) -> dict[str, Any]:
    env = Environment(
        media=Media(name="SM", state="liquid"), temperature=Temperature(value=30)
    )
    experiment = ProteinAbundanceExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=env,
        phenotype=_phenotype(scale),
    )
    reference = ProteinAbundanceExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=_phenotype(ref_scale),
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _good_records() -> list[dict[str, Any]]:
    return [_record(g, s) for g, s in zip(GENES, [8.0, 10.0, 9.5])]


def test_mismatched_replicate_keys_rejected():
    with pytest.raises(ValidationError):
        ProteinAbundancePhenotype(
            protein_abundance={"YAL003W": 8.0},
            n_replicates={"YAL005C": 3},
            measurement_type=MTYPE,
        )


def test_empty_abundance_rejected():
    with pytest.raises(ValidationError):
        ProteinAbundancePhenotype(
            protein_abundance={}, n_replicates={}, measurement_type=MTYPE
        )


def test_good_dataset_passes_all_levels():
    report = verify_protein_dataset(
        _good_records(), dataset_name="good", provenance=PROV, expected_count=3
    )
    assert report.passed, report.summary()
    assert {Level.L0, Level.L1, Level.L2, Level.L3} <= report.levels_covered


def test_duplicate_orf_fails_uniqueness():
    records = _good_records()
    records.append(_record("YMR056C", 7.0))
    report = verify_protein_dataset(
        records, dataset_name="dup", provenance=PROV, expected_count=4
    )
    u = [r for r in report.results if r.name == "orf_uniqueness"]
    assert u and not u[0].passed


def test_key_mismatched_reference_fails():
    records = _good_records()
    bad = _record("YDR001C", 8.0)
    # drop a protein from the reference so keys no longer match the experiment
    bad["reference"]["phenotype_reference"]["protein_abundance"].pop("YAL005C")
    records.append(bad)
    report = verify_protein_dataset(
        records, dataset_name="badref", provenance=PROV, expected_count=4
    )
    r = [x for x in report.results if x.name == "reference_finite"]
    assert r and not r[0].passed


def test_mixed_measurement_type_fails():
    records = _good_records()
    exp = ProteinAbundanceExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name="YDR002W", perturbed_gene_name="YDR002W"
                )
            ]
        ),
        environment=Environment(
            media=Media(name="SM", state="liquid"), temperature=Temperature(value=30)
        ),
        phenotype=ProteinAbundancePhenotype(
            protein_abundance={p: 8.0 for p in PROTS},
            n_replicates={p: 3 for p in PROTS},
            measurement_type="ibaq_copies_per_cell",  # different assay
        ),
    )
    ref = _record("YDR002W", 8.0)["reference"]
    records.append({"experiment": exp.model_dump(), "reference": ref})
    report = verify_protein_dataset(
        records, dataset_name="mixed", provenance=PROV, expected_count=4
    )
    m = [r for r in report.results if r.name == "measurement_type_consistent"]
    assert m and not m[0].passed


def test_protein_gene_set():
    assert protein_gene_set(_good_records()) == set(GENES)
