# tests/torchcell/verification/test_metabolite_verification.py
"""Unit tests for the WS8 metabolite verifier + MetabolitePhenotype (synthetic).

2026.09.30 (Phase 17): every level's verdict and exact message on the three-strain
good table (levels 1.5, -0.8, 0.3 on one metabolite, SE 0.1, reference 0), then one
table per failure mode, each built by mutating one stored dump so only the level under
test moves. The ``reference_finite`` path (``reference_centered=False``) is pinned on a
proper-subset reference (passes), an empty reference, a reference key the strain did not
measure, and an infinite reference (each fails with ``n_bad`` 1; the infinite value is
counted in ``n_values``, the other two are not). A NaN SE is dropped before the
non-negativity check while a negative SE fails it.

2026.10.15 (lane R): an ADDED gene is part of which strain a record measures, so two
strains differing only in a cassette are two strains, and the L4 gene set holds the host
loci -- every perturbed name except the heterologous ones, keyed on ``source_organism``
rather than on the ``gene_addition`` type string (so a heterologous PATHWAY gene is out
and a native extra copy is in).
"""

from __future__ import annotations

import math
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
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.metabolite import (
    metabolite_gene_set,
    verify_metabolite_dataset,
)
from torchcell.verification.report import Level, Provenance

PROV = Provenance(source_uri="test://synthetic", citation_key="test2023")
GENES = ["YMR056C", "YBR085W", "YJR155W"]
MTYPE = "cri_spa_corrected_hsv_yellowness_24h"


def _phenotype(level: float, *, ref: bool = False) -> MetabolitePhenotype:
    return MetabolitePhenotype(
        metabolite_level={"betaxanthin": level},
        metabolite_level_se=None if ref else {"betaxanthin": 0.1},
        n_replicates={"betaxanthin": 1 if ref else 8},
        measurement_type=MTYPE,
    )


def _record(gene: str, level: float, *, ref_level: float = 0.0) -> dict[str, Any]:
    env = Environment(
        media=Media(name="SC", state="solid", is_synthetic=True),
        temperature=Temperature(value=30),
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
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
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
    u = [r for r in report.results if r.name == "genotype_uniqueness"]
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
            media=Media(name="SC", state="solid", is_synthetic=True),
            temperature=Temperature(value=30),
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


def test_absolute_reference_passes_when_not_centered():
    """Absolute-quantity datasets (reference != 0, e.g. Mulleder mM) pass with
    reference_centered=False via the reference_finite check, not reference_zero.
    """
    records = [_record(g, lv, ref_level=5.0) for g, lv in zip(GENES, [7.5, 5.2, 6.1])]
    report = verify_metabolite_dataset(
        records,
        dataset_name="abs",
        provenance=PROV,
        expected_count=3,
        reference_centered=False,
    )
    assert report.passed, report.summary()
    rf = [r for r in report.results if r.name == "reference_finite"]
    assert rf and rf[0].passed
    assert not any(r.name == "reference_zero" for r in report.results)


# --- Phase 17: exact messages per level and per failure mode ------------------ #
def _result(records: list[dict[str, Any]], name: str, **kwargs: Any) -> Any:
    report = verify_metabolite_dataset(
        records,
        dataset_name="t",
        provenance=PROV,
        expected_count=len(records),
        **kwargs,
    )
    return next(r for r in report.results if r.name == name)


def test_good_dataset_exact_results_in_order() -> None:
    """Seven results, in the order the verifier adds them, each with its exact message."""
    report = verify_metabolite_dataset(
        _good_records(), dataset_name="good", provenance=PROV, expected_count=3
    )
    assert [(r.level, r.name, r.passed, r.message) for r in report.results] == [
        (Level.L0, "structural", True, "3 records validated"),
        (Level.L1, "count", True, "observed 3, expected 3"),
        (
            Level.L1,
            "genotype_uniqueness",
            True,
            "3 unique strains (deletion sets), one record each",
        ),
        (Level.L2, "value_fidelity", True, "3 values checked"),
        (Level.L2, "se_nonnegative", True, "3 values checked"),
        (
            Level.L3,
            "reference_zero",
            True,
            "reference metabolite level == 0 for all 3 values",
        ),
        (
            Level.L3,
            "measurement_type_consistent",
            True,
            f"single measurement_type: {MTYPE!r}",
        ),
    ]


def test_failure_messages_for_duplicate_nonzero_reference_and_mixed_type() -> None:
    """Duplicate strain, a 0.7 reference and a second assay, each with its message."""
    records = _good_records() + [_record("YMR056C", 2.0)]
    dup = _result(records, "genotype_uniqueness")
    assert dup.message == "1 deletion sets appear in multiple records"
    assert dup.details == {"n_strains": 3, "n_duplicated": 1, "n_duplicate_records": 1}

    records = _good_records() + [_record("YDR001C", 1.0, ref_level=-0.7)]
    ref = _result(records, "reference_zero")
    assert ref.message == "reference level not identically 0: max|v|=0.7"
    assert ref.details == {"n_values": 4, "worst_abs": 0.7}

    records = _good_records()
    records[0]["experiment"]["phenotype"]["measurement_type"] = "ms_abundance"
    mixed = _result(records, "measurement_type_consistent")
    # sorted: "cri_spa_..." precedes "ms_abundance"
    assert mixed.message == (
        f"2 distinct measurement_types mixed: [{MTYPE!r}, 'ms_abundance']"
    )


def _absolute_records() -> list[dict[str, Any]]:
    return [_record(g, lv, ref_level=5.0) for g, lv in zip(GENES, [7.5, 5.2, 6.1])]


def test_reference_finite_accepts_a_proper_subset_reference() -> None:
    """A strain measuring a metabolite the WT lacks still has a well-defined reference."""
    records = _absolute_records()
    phenotype = records[0]["experiment"]["phenotype"]
    phenotype["metabolite_level"]["lycopene"] = 2.0
    phenotype["metabolite_level_se"]["lycopene"] = 0.1
    phenotype["n_replicates"]["lycopene"] = 8
    result = _result(records, "reference_finite", reference_centered=False)
    assert result.passed is True
    assert result.message == "reference level finite + key-subset for all 3 values"


@pytest.mark.parametrize(
    ("reference_levels", "n_values"),
    [({}, 2), ({"lycopene": 5.0}, 2), ({"betaxanthin": math.inf}, 3)],
    ids=["empty", "not_a_subset", "infinite"],
)
def test_reference_finite_fails_each_malformed_reference(
    reference_levels: dict[str, float], n_values: int
) -> None:
    """One bad reference out of three; only the infinite one is counted as a value."""
    records = _absolute_records()
    records[1]["reference"]["phenotype_reference"]["metabolite_level"] = (
        reference_levels
    )
    result = _result(records, "reference_finite", reference_centered=False)
    assert result.passed is False
    assert result.message == "1 reference levels non-finite, empty, or not a key-subset"
    assert result.details == {"n_values": n_values, "n_bad": 1}


def test_se_nan_is_dropped_and_negative_se_fails() -> None:
    """NaN SE (single replicate) is skipped; a negative SE is flagged ``< 0.0``."""
    records = _good_records()
    records[0]["experiment"]["phenotype"]["metabolite_level_se"] = {
        "betaxanthin": math.nan
    }
    assert _result(records, "se_nonnegative").message == "2 values checked"

    records = _good_records()
    records[2]["experiment"]["phenotype"]["metabolite_level_se"] = {"betaxanthin": -0.1}
    negative = _result(records, "se_nonnegative")
    assert negative.passed is False
    assert negative.message == "1/3 values invalid"
    assert negative.details["bad"] == [{"index": 2, "value": -0.1, "reason": "< 0.0"}]


def _addition(name: str, source_organism: str | None) -> dict[str, Any]:
    """One addition perturbation as a built store serializes it (the key is present)."""
    return {
        "systematic_gene_name": name,
        "perturbed_gene_name": name.lower(),
        "perturbation_type": "gene_addition",
        "source_organism": source_organism,
    }


def test_an_added_gene_is_part_of_the_strain_it_was_added_to() -> None:
    """Two records differing only by the added cassette are two strains, not a duplicate.

    The signature used to drop every ``gene_addition``, which made a store whose strains
    differ ONLY in what they add read as one strain measured many times -- what every
    production-chassis release is. A store whose addition background is constant has that
    constant in every key, so its grouping is unchanged.
    """
    records = [_record("YMR056C", 1.0), _record("YMR056C", 2.0)]
    for i, record in enumerate(records):
        record["experiment"]["genotype"]["perturbations"].append(
            _addition(f"CASSETTE{i}", None)
        )
    unique = _result(records, "genotype_uniqueness")
    assert unique.passed is True
    assert unique.message == "2 unique strains (deletion sets), one record each"
    constant = [_record("YMR056C", 1.0), _record("YMR056C", 2.0)]
    for record in constant:
        record["experiment"]["genotype"]["perturbations"].append(
            _addition("CASSETTE", None)
        )
    assert _result(constant, "genotype_uniqueness").message == (
        "1 deletion sets appear in multiple records"
    )


def test_the_gene_set_holds_the_host_loci_and_not_the_heterologous_names() -> None:
    """A heterologous addition is out of the L4 universe; a native extra copy is in.

    The rule is ``source_organism`` against the record's own species, the same test the
    bioproduction families apply, and not the ``gene_addition`` type string: the string
    dropped a native copy (a real ORF) and kept a heterologous pathway gene (which is a
    gene of another genome).
    """
    records = [_record("YMR056C", 1.0)]
    records[0]["experiment"]["genotype"]["perturbations"] += [
        _addition("crtE", "Xanthophyllomyces dendrorhous"),
        _addition("YBR085W", "Saccharomyces cerevisiae"),
        _addition("CASSETTE", None),
    ]
    assert metabolite_gene_set(records) == {"YMR056C", "YBR085W"}


# --- issue #595: datasets released under several protocols -------------------- #
PROTOCOLS = frozenset({"protocol_a", "protocol_b"})


def _protocol_records() -> list[dict[str, Any]]:
    """YMR056C on protocols a and b, YBR085W on protocol a; references match."""
    records = []
    for gene, mtype in [
        ("YMR056C", "protocol_a"),
        ("YMR056C", "protocol_b"),
        ("YBR085W", "protocol_a"),
    ]:
        record = _record(gene, 1.0, ref_level=5.0)
        record["experiment"]["phenotype"]["measurement_type"] = mtype
        record["reference"]["phenotype_reference"]["measurement_type"] = mtype
        records.append(record)
    return records


def test_protocol_split_passes_one_record_per_strain_and_protocol() -> None:
    """The same strain on two declared protocols is two records, not a duplicate."""
    kwargs: dict[str, Any] = {
        "reference_centered": False,
        "protocol_measurement_types": PROTOCOLS,
    }
    records = _protocol_records()
    unique = _result(records, "genotype_uniqueness", **kwargs)
    assert unique.passed is True
    assert unique.message == "3 unique (strain, protocol) pairs, one record each"
    mtype = _result(records, "measurement_type_consistent", **kwargs)
    assert mtype.passed is True
    assert mtype.message == (
        "2 declared protocol measurement_types, each record's reference on its own "
        "protocol"
    )
    # Without the declaration the same table fails both checks.
    assert _result(records, "genotype_uniqueness").passed is False
    assert _result(records, "measurement_type_consistent").passed is False


def test_protocol_split_fails_a_cross_protocol_reference_and_an_undeclared_type() -> (
    None
):
    """A reference on another protocol, and a type outside the declaration, both fail."""
    kwargs: dict[str, Any] = {
        "reference_centered": False,
        "protocol_measurement_types": PROTOCOLS,
    }
    records = _protocol_records()
    records[1]["reference"]["phenotype_reference"]["measurement_type"] = "protocol_a"
    records[2]["experiment"]["phenotype"]["measurement_type"] = "protocol_c"
    records[2]["reference"]["phenotype_reference"]["measurement_type"] = "protocol_c"
    result = _result(records, "measurement_type_consistent", **kwargs)
    assert result.passed is False
    assert result.message == (
        "1 undeclared measurement_types, 1 records whose reference is on another "
        "protocol"
    )
    assert result.details == {
        "measurement_types": ["protocol_a", "protocol_b", "protocol_c"],
        "undeclared": ["protocol_c"],
        "n_reference_protocol_differs": 1,
    }
    records = _protocol_records()
    records.append(_protocol_records()[0])
    dup = _result(records, "genotype_uniqueness", **kwargs)
    assert dup.message == "1 (strain, protocol) pairs appear in multiple records"


# --- environment-keyed uniqueness (#753, Ishii 2007's dilution-rate arm) ----- #
def _dilution_rate_records() -> list[dict[str, Any]]:
    """Three WILD-TYPE records at 0.1, 0.2 and 0.7 h-1, one genotype between them."""
    records = []
    for rate in (0.1, 0.2, 0.7):
        record = _record("YMR056C", 1.0, ref_level=5.0)
        record["experiment"]["genotype"]["perturbations"] = []
        record["experiment"]["environment"]["dilution_rate_per_hour"] = rate
        records.append(record)
    return records


def test_environment_keyed_uniqueness_separates_one_strain_across_environments() -> (
    None
):
    """One genotype at three dilution rates is three records, not two duplicates."""
    records = _dilution_rate_records()
    unique = _result(
        records, "genotype_uniqueness", reference_centered=False, environment_keyed=True
    )
    assert unique.passed is True
    assert unique.message == "3 unique (strain, environment) pairs, one record each"
    assert unique.details == {
        "n_strains": 3,
        "n_duplicated": 0,
        "n_duplicate_records": 0,
    }
    # Keyed on the strain alone the same three records are one duplicated key.
    strain_only = _result(records, "genotype_uniqueness", reference_centered=False)
    assert strain_only.passed is False
    assert strain_only.message == "1 deletion sets appear in multiple records"
    assert strain_only.details == {
        "n_strains": 1,
        "n_duplicated": 1,
        "n_duplicate_records": 2,
    }


def test_environment_keyed_uniqueness_still_fails_a_repeated_environment() -> None:
    """Two records at the SAME strain and the SAME rate are a duplicate either way."""
    records = _dilution_rate_records()
    records.append(_dilution_rate_records()[0])
    dup = _result(
        records, "genotype_uniqueness", reference_centered=False, environment_keyed=True
    )
    assert dup.passed is False
    assert dup.message == "1 (strain, environment) pairs appear in multiple records"
    assert dup.details == {"n_strains": 3, "n_duplicated": 1, "n_duplicate_records": 1}


def test_environment_keyed_uniqueness_does_not_change_a_fixed_environment() -> None:
    """With one environment for every record the key is the strain plus a constant."""
    records = _good_records()
    plain = _result(records, "genotype_uniqueness")
    keyed = _result(records, "genotype_uniqueness", environment_keyed=True)
    assert (plain.passed, keyed.passed) == (True, True)
    assert (
        plain.details
        == keyed.details
        == {"n_strains": 3, "n_duplicated": 0, "n_duplicate_records": 0}
    )
    assert keyed.message == "3 unique (strain, environment) pairs, one record each"


def test_allow_duplicate_strains_reports_the_collisions_as_a_finding() -> None:
    """Cooper 2010's replicate rows: two records of one strain, L1 passes with the count."""
    from torchcell.verification.metabolite import _l1_orf_uniqueness

    records = [
        _record("YAL001C", 1.0),
        _record("YAL001C", 2.0),
        _record("YAL002W", 3.0),
    ]
    refused = _l1_orf_uniqueness(records)
    finding = _l1_orf_uniqueness(records, allow_duplicate_strains=True)
    assert refused.passed is False
    assert finding.passed is True
    assert finding.details == {
        "n_strains": 2,
        "n_duplicated": 1,
        "n_duplicate_records": 1,
    }
    assert finding.message.endswith(
        "(released replicate rows with no strain key; reported, not refused)"
    )
