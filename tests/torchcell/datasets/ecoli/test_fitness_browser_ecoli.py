# tests/torchcell/datasets/ecoli/test_fitness_browser_ecoli.py
# [[tests.torchcell.datasets.ecoli.test_fitness_browser_ecoli]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_fitness_browser_ecoli.py
"""``torchcell/datasets/ecoli/fitness_browser_ecoli.py``: fates, records and a full build.

Hermetic: a fixture database with the deposit's layout; the genomes, the ECK placement
and the assembly pin are replaced by fixed values. The ``data`` tests read the real
deposit and the built dev store.
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp
from pathlib import Path
from typing import Any

import pytest

import torchcell.datasets.ecoli.fitness_browser_ecoli as fe
from tests.torchcell.datasets._fitness_browser_fixture import experiment_row, write_feba
from torchcell.data import file_sha256
from torchcell.datamodels.media import RCH2_DEFINED_NOCARBON_PRICE2018
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    PhysicalFactor,
    TransposonInsertionPerturbation,
)
from torchcell.datasets import fitness_browser as fb
from torchcell.datasets.ecoli import price2018
from torchcell.datasets.ecoli import wetmore2015 as wetmore

PRICE21B = fe.SOURCE_STUDIES["Price21b"]


def _exp(name: str = "set52IT047", **cells: Any) -> fb.FbExperiment:
    base = {
        "pubId": "Price21b",
        "media": "RCH2_defined_noCarbon",
        "condition_1": "D-Mannitol",
        "concentration_1": "10",
        "units_1": "mM",
        "mutantLibrary": "Keio_ML9a",
    }
    return fb.experiment_from_row(experiment_row("Keio", name, 1, **{**base, **cells}))


def test_study_table_names_each_keio_publication_once() -> None:
    assert set(fe.SOURCE_STUDIES) == {"Wetmore15", "Price18", "Price21b"}
    assert fe.SOURCE_STUDIES["Wetmore15"].doi == wetmore.PAPER_DOI
    assert fe.SOURCE_STUDIES["Wetmore15"].own_release_dataset == (
        "RbTnseqWetmore2015Dataset"
    )
    price = fe.SOURCE_STUDIES["Price18"]
    assert (price.doi, price.source_kind, price.own_release_dataset) == (
        price2018.PAPER_DOI,
        "compendium_only",
        "RbTnseqPrice2018EcoliDataset",
    )
    assert (PRICE21B.doi, PRICE21B.source_kind, PRICE21B.issues) == (
        "10.1101/2021.11.02.466981",
        "primary",
        (939,),
    )
    assert PRICE21B.own_release_dataset is None


@pytest.mark.parametrize(
    ("cells", "fate", "detail"),
    [
        ({"pubId": ""}, "unattributed_release", None),
        (
            {"pubId": "Price18", "condition_1": "Sucrose"},
            "author_withdrawn",
            wetmore.AUTHOR_WITHDRAWAL.rule,
        ),
        (
            {"pubId": "Wetmore15", "condition_1": "D-Mannitol"},
            "author_withdrawn",
            wetmore.AUTHOR_WITHDRAWAL.rule,
        ),
        (
            {"pubId": "Wetmore15", "condition_1": "D-Glucose"},
            "own_release_loaded",
            "RbTnseqWetmore2015Dataset",
        ),
        (
            {"pubId": "Price18", "condition_1": "D-Glucose"},
            "own_release_loaded",
            "RbTnseqPrice2018EcoliDataset",
        ),
        ({"media": "SDM_noCarbon"}, "medium_not_in_media_library", "SDM_noCarbon"),
        ({"expGroup": "stress"}, "condition_not_representable", "group 'stress'"),
        ({"units_1": "g/L"}, "condition_not_representable", "unit 'g/L'"),
        ({"temperature": ""}, "condition_not_representable", "no temperature"),
        ({}, "stored", None),
    ],
)
def test_decide_takes_the_first_rule_that_applies(
    cells: dict[str, Any], fate: str, detail: str | None
) -> None:
    decision = fe.decide(_exp(**cells))
    assert (decision.fate, decision.detail) == (fate, detail)


def test_price21b_mannitol_is_outside_the_withdrawal_scope() -> None:
    # The 2021 note names "this publication" (Price 2018); GapMind's own mannitol
    # samples are a later publication the browser still serves.
    assert fe.decide(_exp(condition_1="D-Mannitol")).fate == "stored"


def test_the_module_key_is_the_deposit_mirror_key() -> None:
    assert fe.CITATION_KEY == fb.DEPOSIT_CITATION_KEY


def test_decide_refuses_an_unknown_publication() -> None:
    with pytest.raises(KeyError, match="not in SOURCE_STUDIES"):
        fe.decide(_exp(pubId="Somebody99"))


def test_unrepresentable_names_each_reason() -> None:
    assert fe.unrepresentable(_exp(aerobic="Anaerobic")) == (
        "aerobic='Anaerobic', liquid='Liquid'"
    )
    assert fe.unrepresentable(_exp(media="LB")) == (
        "group 'carbon source' on medium 'LB'"
    )
    assert fe.unrepresentable(
        _exp(condition_2="Dimethyl Sulfoxide", units_2="vol%", concentration_2="1")
    ) == ("2 condition columns")
    assert fe.unrepresentable(_exp(concentration_1="ten")) == (
        "condition 'D-Mannitol' at 'ten'"
    )
    assert fe.unrepresentable(_exp()) is None


def test_environment_is_the_medium_the_carbon_source_and_the_temperature() -> None:
    environment = fe.build_environment(_exp())
    assert environment.media == RCH2_DEFINED_NOCARBON_PRICE2018
    assert environment.temperature is not None
    assert environment.temperature.value == 30.0
    (perturbation,) = environment.perturbations
    assert isinstance(perturbation, EnvironmentPhysicalPerturbation)
    assert perturbation.factor is PhysicalFactor.carbon_source
    assert perturbation.magnitude is not None
    assert perturbation.magnitude.value == 10.0
    assert perturbation.magnitude.unit is ConcentrationUnit.millimolar
    assert environment.aerobicity == "aerobic"
    assert environment.duration_generations is None
    assert [g.field for g in environment.provenance_gaps] == [
        "duration_hours",
        "duration_generations",
    ]
    generations = fe.build_environment(_exp(nGenerations=5.2))
    assert generations.duration_generations == 5.2
    assert [g.field for g in generations.provenance_gaps] == ["duration_hours"]
    with pytest.raises(ValueError, match="medium 'LB'"):
        fe.build_environment(_exp(media="LB"))


def test_genotype_phenotype_and_reference() -> None:
    gene = fb.ReleasedGene(locus_id="100", tag="b0002", name="thrA")
    (perturbation,) = fe.build_genotype(gene, "Keio_ML9a").perturbations
    assert isinstance(perturbation, TransposonInsertionPerturbation)
    # the locus tag the deposit gives, in its own namespace; nothing derived
    assert perturbation.systematic_gene_name == "b0002"
    assert perturbation.perturbed_gene_name == "thrA"
    assert perturbation.gene_namespace == "ecoli_k12_mg1655_bnumber"
    assert perturbation.identifier_mapping is None
    assert perturbation.library_pool == "Keio_ML9a"
    assert perturbation.transposon is None

    se = 0.25
    t = -0.8 / math.sqrt(0.01 + se**2)
    phenotype = fe.build_phenotype(-0.8, t, "Keio:set52IT047")
    assert phenotype.environment_response == -0.8
    assert phenotype.environment_response_t_statistic == t
    assert phenotype.environment_response_se == pytest.approx(se, rel=1e-12)
    assert phenotype.screen_id == "Keio:set52IT047"
    assert {g.field for g in phenotype.provenance_gaps} == {
        "environment_response_uncertainty",
        "n_samples",
        "sample_unit",
    }
    reference = fe.build_reference(
        "ds", fe.build_environment(_exp()), _reference(), "x"
    )
    assert reference.phenotype_reference.environment_response == 0.0


def _reference() -> AssemblyReferenceGenome:
    return AssemblyReferenceGenome(
        species="Escherichia coli",
        strain="BW25113",
        ploidy="haploid",
        assembly_set="ecoli_K12_BW25113_ASM75055v1",
        assembly_accession="GCA_000750555.1",
    )


GENES = [
    ("Keio", "100", "b0002", "thrA", 1),
    ("Keio", "101", "b0003", "thrB", 1),
    ("Keio", "102", "b4700", None, 6),
]
PUBLICATIONS = [
    (s.pub_id, s.browser_title, s.browser_url) for s in fe.SOURCE_STUDIES.values()
]


def _write(path: Path) -> Path:
    experiments = [
        experiment_row(
            "Keio", "set1IT003", 1, pubId="Wetmore15", media="M9 minimal media_noCarbon"
        ),
        experiment_row("Keio", "set3IT001", 2, pubId="Price18", media="LB"),
        experiment_row("Keio", "set57IT037", 3, pubId="", media="SDM_noCarbon"),
        experiment_row(
            "Keio",
            "set52IT047",
            4,
            pubId="Price21b",
            media="RCH2_defined_noCarbon",
            condition_1="D-Mannitol",
            concentration_1="10",
            mutantLibrary="Keio_ML9a",
        ),
        experiment_row(
            "Keio",
            "set52IT049",
            5,
            pubId="Price21b",
            media="RCH2_defined_noCarbon",
            condition_1="D-Mannitol",
            concentration_1="2.5",
            mutantLibrary="Keio_ML9a",
        ),
    ]
    names = [str(e["expName"]) for e in experiments]
    fitness = [
        (name, locus, 0.1 * (j + 1) + 0.01 * k, (0.1 * (j + 1) + 0.01 * k) * 2.0)
        for j, name in enumerate(names)
        for k, locus in enumerate(("100", "101", "102"))
    ]
    return write_feba(
        path,
        experiments=experiments,
        publications=PUBLICATIONS,
        genes=GENES,
        fitness={"Keio": fitness},
    )


def test_process_stores_only_the_unloaded_attributed_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "rbtnseq_fitness_browser_ecoli"
    raw = root / "raw"
    raw.mkdir(parents=True)
    path = _write(raw / fe.RAW_FILE)
    monkeypatch.setattr(
        fb, "FEBA", fb.FEBA.model_copy(update={"sha256": file_sha256(path)})
    )
    monkeypatch.setattr(fe, "assembly_reference", lambda strain: _reference())
    # no genome is read anywhere in the loader since #1057: it imports no genome reader
    assert not hasattr(fe, "bacterial_genome")

    dataset = fe.FitnessBrowserEcoliDataset(root=str(root))
    # every released locus of every stored sample, the one without a symbol included
    assert len(dataset) == 2 * 3
    first = dataset.transform_item(dataset[0])
    assert first["experiment"].phenotype.screen_id == "Keio:set52IT047"
    assert first["experiment"].phenotype.environment_response == pytest.approx(0.4)
    assert first["experiment"].phenotype.environment_response_t_statistic == (
        pytest.approx(0.8)
    )
    assert first["publication"].doi == PRICE21B.doi
    (perturbation,) = first["experiment"].genotype.perturbations
    assert perturbation.systematic_gene_name == "b0002"
    assert perturbation.perturbed_gene_name == "thrA"
    tags = {
        dataset[i]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for i in range(len(dataset))
    }
    assert tags == {"b0002", "b0003", "b4700"}
    screens = {
        dataset[i]["experiment"]["phenotype"]["screen_id"] for i in range(len(dataset))
    }
    assert screens == {"Keio:set52IT047", "Keio:set52IT049"}

    preprocess = Path(dataset.preprocess_dir)
    counts = json.loads((preprocess / "dropped_records.json").read_text())
    assert counts["fates"] == {
        "own_release_loaded": 2,
        "unattributed_release": 1,
        "stored": 2,
    }
    assert counts["stored_records"] == 6
    assert counts["n_loci"] == 3
    genes = json.loads((preprocess / "genes.json").read_text())
    assert [(g["tag"], g["name"]) for g in genes] == [
        ("b0002", "thrA"),
        ("b0003", "thrB"),
        ("b4700", "b4700"),
    ]
    samples = json.loads((preprocess / "samples.json").read_text())
    assert [(s["exp_name"], s["fate"]) for s in samples] == [
        ("set1IT003", "own_release_loaded"),
        ("set3IT001", "own_release_loaded"),
        ("set57IT037", "unattributed_release"),
        ("set52IT047", "stored"),
        ("set52IT049", "stored"),
    ]
    dataset.close_lmdb()
    row = fe.stored_publications_match(str(root))
    assert row.passed, row.message


def test_process_refuses_a_deposit_whose_title_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moved = {
        **fe.SOURCE_STUDIES,
        "Price21b": PRICE21B.model_copy(update={"browser_title": "Another title"}),
    }
    monkeypatch.setattr(fe, "SOURCE_STUDIES", moved)
    raw = tmp_path / "ds" / "raw"
    raw.mkdir(parents=True)
    path = _write(raw / fe.RAW_FILE)
    monkeypatch.setattr(
        fb, "FEBA", fb.FEBA.model_copy(update={"sha256": file_sha256(path)})
    )
    with pytest.raises(ValueError, match="the deposit says"):
        fe.FitnessBrowserEcoliDataset(root=str(tmp_path / "ds"))


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


@pytest.mark.data
def test_the_deposit_keio_fates() -> None:
    path = fb.feba_path(_data_root())
    if not path.exists():
        pytest.skip(f"deposit absent: {path}")
    feba = fb.Feba(path, verify=False)
    fb.check_studies(fe.SOURCE_STUDIES, feba.publications())
    decisions = [fe.decide(e) for e in feba.experiments(fe.ORG_ID)]
    feba.close()
    assert fb.fate_counts_by_publication(decisions) == {
        "Price18": {"own_release_loaded": 71},
        "Price21b": {"stored": fe.EXPECTED_SAMPLES},
        "Wetmore15": {"own_release_loaded": 87},
        "no-pubId": {"unattributed_release": 19},
    }


@pytest.mark.data
def test_built_store_passes_its_verification() -> None:
    root = osp.join(_data_root(), fe.DATASET_ROOT_REL)
    path = osp.join(root, "preprocess", "verification_report.json")
    if not osp.exists(path):
        pytest.skip(f"dev store not built and verified: {root}")
    with open(path) as handle:
        report = json.load(handle)
    failed = [r["name"] for r in report["results"] if not r["passed"]]
    assert failed == []
