# tests/torchcell/datasets/pputida/test_fitness_browser_putida.py
# [[tests.torchcell.datasets.pputida.test_fitness_browser_putida]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_fitness_browser_putida.py
"""``torchcell/datasets/pputida/fitness_browser_putida.py``: fates, environments, a build.

Hermetic: a fixture database with the deposit's layout; the genome, the reconciler and
the assembly pin are replaced by fixed values. The ``data`` tests read the real deposit
and the built dev store.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

import pytest

import torchcell.datasets.pputida.fitness_browser_putida as fp
from tests.torchcell.datasets._fitness_browser_fixture import experiment_row, write_feba
from torchcell.data import file_sha256
from torchcell.datamodels.schema import (
    ConcentrationUnit,
    DoseBasis,
    EnvironmentPhysicalPerturbation,
    PhysicalFactor,
    ReferenceGenome,
    SmallMoleculePerturbation,
    TransposonInsertionPerturbation,
)
from torchcell.datasets import fitness_browser as fb
from torchcell.datasets.pputida import borchert2024 as b24


def _exp(name: str = "set1IT001", **cells: Any) -> fb.FbExperiment:
    base = {"pubId": "Schmidt22", "mutantLibrary": "Putida_ML5_JBEI"}
    return fb.experiment_from_row(
        experiment_row("Putida", name, 1, **{**base, **cells})
    )


def test_study_table_reuses_the_borchert_publications() -> None:
    studies = fp.SOURCE_STUDIES
    for key, pub in (
        ("schmidt2022", "Schmidt22"),
        ("thompson2020", "Thompson20"),
        ("rand2017", "Rand17"),
    ):
        assert studies[pub].publication == b24.SOURCE_STUDIES[key].publication  # type: ignore[index]
    assert {s.source_kind for s in studies.values()} == {"primary"}
    assert sorted(i for s in studies.values() for i in s.issues) == list(
        range(932, 941)
    )
    assert studies["Pearson24"].publication.pubmed_id == "38113125"
    assert studies["Thompson19"].doi == "10.1016/j.mec.2019.e00098"


@pytest.mark.parametrize(
    ("cells", "fate", "detail"),
    [
        ({"pubId": ""}, "unattributed_release", None),
        (
            {"pubId": "VillalobosEscobedo23", "media": "R2A"},
            "medium_not_in_media_library",
            "R2A",
        ),
        ({"expGroup": "reactor"}, "condition_not_representable", "group 'reactor'"),
        (
            {"expGroup": "nitrogen source"},
            "condition_not_representable",
            "group 'nitrogen source' on medium 'MOPS minimal media_noCarbon'",
        ),
        (
            {"condition_3": "Butanol", "units_3": "mM", "concentration_3": "1"},
            "condition_not_representable",
            "condition_3 'Butanol'",
        ),
        (
            {
                "condition_2": "20AA_mix_minus_Xyz",
                "units_2": "X",
                "concentration_2": "1",
            },
            "condition_not_representable",
            "amino-acid dropout '20AA_mix_minus_Xyz'",
        ),
        (
            {"condition_2": "Glucose", "units_2": "X", "concentration_2": "1"},
            "condition_not_representable",
            "unit 'X' on 'Glucose'",
        ),
        ({"units_1": "vol%"}, "condition_not_representable", "condition_1 unit 'vol%'"),
        ({}, "stored", None),
    ],
)
def test_decide_takes_the_first_rule_that_applies(
    cells: dict[str, Any], fate: str, detail: str | None
) -> None:
    decision = fp.decide(_exp(**cells))
    assert (decision.fate, decision.detail) == (fate, detail)


def test_the_module_key_is_the_deposit_mirror_key() -> None:
    assert fp.CITATION_KEY == fb.DEPOSIT_CITATION_KEY
    assert fp.DATASET_CLASS_NAME == fp.FitnessBrowserPutidaDataset.__name__
    assert set(fp.SOURCE_STUDIES) - fp.STORED_PUB_IDS == {"VillalobosEscobedo23"}


def test_decide_refuses_an_unknown_publication() -> None:
    with pytest.raises(KeyError, match="not in SOURCE_STUDIES"):
        fp.decide(_exp(pubId="Somebody99"))


def test_carbon_source_with_dmso_and_generations() -> None:
    env = fp.build_environment(
        _exp(
            condition_2="Dimethyl Sulfoxide",
            units_2="vol%",
            concentration_2="0.5",
            nGenerations=6.1,
        )
    )
    assert env.media == b24.MEDIA_BY_LABEL["MOPS minimal media_noCarbon"]
    first, second = env.perturbations
    assert isinstance(first, EnvironmentPhysicalPerturbation)
    assert first.factor is PhysicalFactor.carbon_source
    assert isinstance(second, SmallMoleculePerturbation)
    assert second.concentration.value == 0.5
    assert second.concentration.unit is ConcentrationUnit.percent_v_v
    assert env.duration_generations == 6.1
    assert [g.field for g in env.provenance_gaps] == ["duration_hours"]


def test_amino_acid_mix_and_its_dropout() -> None:
    env = fp.build_environment(
        _exp(condition_2="20AA_mix_minus_Arg", units_2="X", concentration_2="1")
    )
    _, mix, dropout = env.perturbations
    assert isinstance(mix, SmallMoleculePerturbation)
    assert mix.compound.name == b24.AMINO_ACID_MIX
    assert mix.concentration.basis is DoseBasis.fixed
    assert isinstance(dropout, EnvironmentPhysicalPerturbation)
    assert dropout.factor is PhysicalFactor.nutrient_dropout
    assert dropout.agent is not None
    full = fp.build_environment(
        _exp(condition_2="20AA_mix", units_2="X", concentration_2="1")
    )
    assert len(full.perturbations) == 2


def test_nitrogen_source_is_on_the_glucose_mops() -> None:
    env = fp.build_environment(
        _exp(
            expGroup="nitrogen source",
            media="MOPS minimal media_Glucose_noNitrogen",
            condition_1="L-Lysine",
        )
    )
    assert env.media == b24.FB_MOPS_GLUCOSE_NO_NITROGEN
    (first,) = env.perturbations
    assert isinstance(first, EnvironmentPhysicalPerturbation)
    assert first.factor is PhysicalFactor.nitrogen_source


def test_the_environment_matches_borchert_for_the_same_columns() -> None:
    """Same medium and perturbations as the compendium loader builds, so the two stores'
    environments differ only by the duration this deposit adds.
    """
    sample = b24.SampleMetadata(
        exp_name="set1IT001",
        set_name="set1",
        exp_desc="x",
        exp_group="carbon source",
        mutant_library="Putida_ML5_JBEI",
        person="p",
        media="MOPS minimal media_noCarbon",
        temperature=30.0,
        aerobic="Aerobic",
        total_rep=1,
        rep=1,
        condition_1="D-Glucose",
        units_1="mM",
        concentration_1=20.0,
        condition_2="Dimethyl Sulfoxide",
        units_2="vol%",
        concentration_2=0.5,
    )
    theirs = b24.build_environment(sample)
    ours = fp.build_environment(
        _exp(condition_2="Dimethyl Sulfoxide", units_2="vol%", concentration_2="0.5")
    )
    assert ours.media == theirs.media
    assert ours.perturbations == theirs.perturbations
    assert ours.temperature == theirs.temperature


def test_genotype_and_phenotype() -> None:
    (perturbation,) = fp.build_genotype("PP_0001", "parB", "Putida_ML5").perturbations
    assert isinstance(perturbation, TransposonInsertionPerturbation)
    assert perturbation.transposon == b24.TRANSPOSON
    assert perturbation.library_pool == "Putida_ML5"
    phenotype = fp.build_phenotype(-1.0, -2.0, "Putida:set1IT001")
    assert phenotype.environment_response_se == pytest.approx((0.25 - 0.01) ** 0.5)
    assert phenotype.environment_response_t_statistic == -2.0


GENES = ["PP_0001", "PP_0002"]
PUBLICATIONS = [
    (s.pub_id, s.browser_title, s.browser_url) for s in fp.SOURCE_STUDIES.values()
]


def _reference() -> ReferenceGenome:
    return ReferenceGenome(
        species="Pseudomonas putida",
        strain="KT2440",
        ploidy="haploid",
        assembly_set="pputida_KT2440_ASM756v2",
        assembly_accession="GCA_000007565.2",
    )


def _write(path: Path) -> Path:
    experiments = [
        experiment_row("Putida", "set1IT001", 1, pubId="Thompson20"),
        experiment_row(
            "Putida",
            "set27IT001",
            2,
            pubId="Schmidt22",
            expGroup="nitrogen source",
            media="MOPS minimal media_Glucose_noNitrogen",
            condition_1="L-Lysine",
        ),
        experiment_row(
            "Putida",
            "set8IT001",
            3,
            pubId="",
            expGroup="reactor",
            media="M9_1percentGlucose",
            condition_1="",
            units_1="",
            concentration_1="",
        ),
        experiment_row(
            "Putida",
            "set40IT001",
            4,
            pubId="VillalobosEscobedo23",
            media="R2A",
            expGroup="r2a control",
            condition_1="",
            units_1="",
            concentration_1="",
        ),
    ]
    fitness = [
        (str(e["expName"]), locus, 0.5 + j + 0.1 * k, (0.5 + j + 0.1 * k) * 3.0)
        for j, e in enumerate(experiments)
        for k, locus in enumerate(GENES)
    ]
    return write_feba(
        path,
        experiments=experiments,
        publications=PUBLICATIONS,
        genes=[
            ("Putida", "PP_0001", "PP_0001", "parB", 1),
            ("Putida", "PP_0002", "PP_0002", None, 1),
        ],
        fitness={"Putida": fitness},
    )


def test_process_builds_one_record_per_locus_and_stored_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "rbtnseq_fitness_browser_putida"
    raw = root / "raw"
    raw.mkdir(parents=True)
    path = _write(raw / fp.RAW_FILE)
    monkeypatch.setattr(
        fb, "FEBA", fb.FEBA.model_copy(update={"sha256": file_sha256(path)})
    )
    monkeypatch.setattr(fp, "assembly_reference", lambda strain: _reference())
    monkeypatch.setattr(
        fp,
        "reconcile_locus_tags",
        lambda *a, **k: pytest.fail("a genome was read"),
        raising=False,
    )

    dataset = fp.FitnessBrowserPutidaDataset(root=str(root))
    assert len(dataset) == 2 * len(GENES)
    first = dataset.transform_item(dataset[0])
    (perturbation,) = first["experiment"].genotype.perturbations
    assert perturbation.systematic_gene_name == "PP_0001"
    assert perturbation.perturbed_gene_name == "parB"
    assert first["experiment"].phenotype.screen_id == "Putida:set1IT001"
    assert first["experiment"].phenotype.environment_response == pytest.approx(0.5)
    assert first["publication"].doi == fp.SOURCE_STUDIES["Thompson20"].doi
    second = dataset.transform_item(dataset[1])
    (unnamed,) = second["experiment"].genotype.perturbations
    assert (unnamed.systematic_gene_name, unnamed.perturbed_gene_name) == (
        "PP_0002",
        "PP_0002",
    )
    last = dataset.transform_item(dataset[len(dataset) - 1])
    assert last["publication"].doi == fp.SOURCE_STUDIES["Schmidt22"].doi
    assert last["experiment"].environment.media == b24.FB_MOPS_GLUCOSE_NO_NITROGEN

    preprocess = Path(dataset.preprocess_dir)
    counts = json.loads((preprocess / "dropped_records.json").read_text())
    assert counts["fates"] == {
        "stored": 2,
        "unattributed_release": 1,
        "medium_not_in_media_library": 1,
    }
    assert counts["stored_by_publication"] == {"Schmidt22": 1, "Thompson20": 1}
    samples = json.loads((preprocess / "samples.json").read_text())
    dataset.close_lmdb()
    from torchcell.verification.runners import stream_records

    row = fp.stored_publications_match(stream_records(str(root)), samples)
    assert row.passed, row.message


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


@pytest.mark.data
def test_the_deposit_putida_fates() -> None:
    path = fb.feba_path(_data_root())
    if not path.exists():
        pytest.skip(f"deposit absent: {path}")
    feba = fb.Feba(path, verify=False)
    fb.check_studies(fp.SOURCE_STUDIES, feba.publications())
    decisions = [fp.decide(e) for e in feba.experiments(fp.ORG_ID)]
    feba.close()
    counts = fb.fate_counts(decisions)
    assert {d.pub_id for d in decisions if d.fate == "stored"} == fp.STORED_PUB_IDS
    assert counts == {
        "stored": fp.EXPECTED_SAMPLES,
        "unattributed_release": 34,
        "medium_not_in_media_library": 46,
    }


@pytest.mark.data
def test_built_store_passes_its_verification() -> None:
    root = osp.join(_data_root(), fp.DATASET_ROOT_REL)
    path = osp.join(root, "preprocess", "verification_report.json")
    if not osp.exists(path):
        pytest.skip(f"dev store not built and verified: {root}")
    with open(path) as handle:
        report = json.load(handle)
    failed = [r["name"] for r in report["results"] if not r["passed"]]
    assert failed == []
