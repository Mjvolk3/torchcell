# tests/torchcell/datasets/test_fitness_browser.py
# [[tests.torchcell.datasets.test_fitness_browser]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_fitness_browser.py
"""``torchcell/datasets/fitness_browser.py``: the pinned deposit reader and its helpers.

Hermetic: a fixture database with the deposit's table layout
(``_fitness_browser_fixture``). The ``data`` test reads the real deposit's size only.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from tests.torchcell.datasets._fitness_browser_fixture import experiment_row, write_feba
from torchcell.datasets import fitness_browser as fb


@pytest.fixture
def feba(tmp_path: Path) -> fb.Feba:
    path = write_feba(
        tmp_path / "feba.db",
        experiments=[
            experiment_row(
                "Org",
                "set1IT002",
                2,
                pubId="Pub20",
                nGenerations=4.5,
                temperature="25",
                condition_2="Dimethyl Sulfoxide",
                units_2="vol%",
                concentration_2="0.1",
            ),
            experiment_row("Org", "set1IT001", 1, pubId=""),
            experiment_row(
                "Org",
                "set9IT093",
                3,
                condition_1="",
                units_1="",
                concentration_1="",
                condition_2="",
                units_2="mM",
                concentration_2="5",
            ),
            experiment_row("Other", "set1IT001", 1, temperature=""),
        ],
        publications=[("Pub20", "A title", " https://doi.org/10.1/x ")],
        genes=[("Org", "L1", "b0001", "thrL", 1), ("Org", "L2", None, None, 9)],
        fitness={
            "Org": [
                ("set1IT001", "L1", 0.5, 1.0),
                ("set1IT001", "L2", -0.2, -0.5),
                ("set1IT002", "L1", 1.5, 3.0),
                ("set1IT002", "L2", -1.0, -2.0),
                ("set9IT093", "L1", 0.1, 0.4),
            ]
        },
        organisms=[("Org", "Genus", "species", "S1")],
    )
    return fb.Feba(path, verify=False)


def test_experiments_are_typed_in_num_order(feba: fb.Feba) -> None:
    experiments = feba.experiments("Org")
    assert [e.exp_name for e in experiments] == ["set1IT001", "set1IT002", "set9IT093"]
    first, second, third = experiments
    assert first.pub_id is None
    assert first.screen_id == "Org:set1IT001"
    assert first.conditions == (
        fb.FbCondition(
            position=1,
            condition="D-Glucose",
            concentration_text="20",
            concentration=20.0,
            units="mM",
        ),
    )
    assert second.pub_id == "Pub20"
    assert second.temperature_c == 25.0
    assert second.n_generations == 4.5
    assert second.ph is None
    assert [c.position for c in second.conditions] == [1, 2]
    assert second.conditions[1].concentration == 0.1
    assert second.quality.g_med == 120
    # a dose with no condition is kept as released, never dropped or repaired
    assert third.conditions == (
        fb.FbCondition(
            position=2,
            condition=None,
            concentration_text="5",
            concentration=5.0,
            units="mM",
        ),
    )
    assert feba.experiments("Other")[0].temperature_c is None


def test_a_non_numeric_dose_keeps_its_text() -> None:
    row = experiment_row("Org", "x", 1, concentration_1="0.5x")
    (condition,) = fb.experiment_from_row(row).conditions
    assert (condition.concentration_text, condition.concentration) == ("0.5x", None)


def test_publications_organisms_genes_and_counts(feba: fb.Feba) -> None:
    assert feba.publications() == {
        "Pub20": fb.FbPublication(
            pub_id="Pub20", title="A title", url="https://doi.org/10.1/x"
        )
    }
    assert feba.organisms() == (
        fb.FbOrganism(org_id="Org", genus="Genus", species="species", strain="S1"),
    )
    genes = feba.genes("Org")
    assert genes["L1"].sys_name == "b0001"
    assert genes["L2"] == fb.FbGene(
        locus_id="L2", sys_name=None, gene=None, gene_type=9
    )
    assert feba.gene_counts("Org") == {"set1IT001": 2, "set1IT002": 2, "set9IT093": 1}
    with pytest.raises(KeyError, match="no FitByExp table"):
        feba.gene_counts("Missing")


def test_values_is_the_complete_loci_by_experiment_matrix(feba: fb.Feba) -> None:
    values = feba.values("Org", ["set1IT002", "set1IT001"])
    assert values.loci == ("L1", "L2")
    assert values.experiments == ("set1IT002", "set1IT001")
    np.testing.assert_array_equal(values.fitness, [[1.5, 0.5], [-1.0, -0.2]])
    np.testing.assert_array_equal(values.t, [[3.0, 1.0], [-2.0, -0.5]])
    with pytest.raises(ValueError, match="do not share one locus set"):
        feba.values("Org", ["set1IT001", "set9IT093"])
    with pytest.raises(ValueError, match="no values for"):
        feba.values("Org", ["set1IT001", "absent"])
    with pytest.raises(ValueError, match="requested twice"):
        feba.values("Org", ["set1IT001", "set1IT001"])


def test_verify_feba_refuses_other_bytes(tmp_path: Path) -> None:
    path = tmp_path / "feba.db"
    path.write_bytes(b"not the deposit")
    with pytest.raises(fb.FebaPinError, match="bytes, pinned"):
        fb.verify_feba(path)
    with pytest.raises(fb.FebaPinError, match="bytes, pinned"):
        fb.Feba(path)


def test_feba_path_is_under_the_price2018_raw_mirror(tmp_path: Path) -> None:
    assert fb.feba_path(str(tmp_path)) == (
        tmp_path
        / "torchcell-raw/priceMutantPhenotypesThousands2018/data/fitness_browser/feba.db"
    )


def _study(**kwargs: object) -> fb.SourceStudy:
    base: dict[str, object] = {
        "pub_id": "Pub20",
        "browser_title": "A title",
        "browser_url": "https://doi.org/10.1/x",
        "doi": "10.1/x",
        "citation_key": None,
        "source_kind": "primary",
    }
    return fb.SourceStudy.model_validate({**base, **kwargs})


def test_check_studies_compares_title_and_url_verbatim(feba: fb.Feba) -> None:
    publications = feba.publications()
    fb.check_studies({"Pub20": _study()}, publications)
    with pytest.raises(ValueError, match="the deposit says"):
        fb.check_studies({"Pub20": _study(browser_title="A Title")}, publications)
    with pytest.raises(ValueError, match="not in the deposit"):
        fb.check_studies({"Pub21": _study(pub_id="Pub21")}, publications)
    with pytest.raises(ValueError, match="keyed under another pubId"):
        fb.check_studies({"Pub20": _study(pub_id="Pub21")}, publications)


def test_source_study_publication_by_doi_else_pubmed() -> None:
    assert _study().publication.doi == "10.1/x"
    assert _study().publication.doi_url == "https://doi.org/10.1/x"
    pubmed = _study(doi=None, pubmed_id="38113125").publication
    assert (pubmed.doi, pubmed.pubmed_id) == (None, "38113125")
    assert pubmed.pubmed_url == "https://pubmed.ncbi.nlm.nih.gov/38113125/"
    with pytest.raises(ValueError, match="neither a DOI nor a PubMed id"):
        _ = _study(doi=None).publication


def test_derived_se_inverts_the_moderated_t() -> None:
    se = 0.3
    fitness = -1.2
    t = fitness / math.sqrt(fb.T_SIGMA**2 + se**2)
    assert fb.derived_se(fitness, t) == pytest.approx(se, rel=1e-12)
    with pytest.raises(ValueError, match="undefined"):
        fb.derived_se(0.0, 1.0)
    with pytest.raises(ValueError, match="below the identity's floor"):
        fb.derived_se(0.05, 1.0)


def test_fate_counts() -> None:
    decisions = [
        fb.SampleDecision(exp_name="a", pub_id="P", fate="stored"),
        fb.SampleDecision(exp_name="b", pub_id=None, fate="unattributed_release"),
        fb.SampleDecision(exp_name="c", pub_id="P", fate="stored"),
    ]
    assert fb.fate_counts(decisions) == {"stored": 2, "unattributed_release": 1}
    assert fb.fate_counts_by_publication(decisions) == {
        "P": {"stored": 2},
        "no-pubId": {"unattributed_release": 1},
    }
    assert set(fb.FATE_DESCRIPTIONS) == {
        "stored",
        "unattributed_release",
        "own_release_loaded",
        "author_withdrawn",
        "medium_not_in_media_library",
        "condition_not_representable",
    }


def test_compendium_check_pools_and_scores_each_sample() -> None:
    own = {
        "s1": {"g1": 1.0, "g2": 2.0, "g3": 3.0, "g9": 9.0},
        "s2": {"g1": -1.0, "g2": 0.0, "g3": 1.0},
        "s3": {"g1": 0.0},
    }
    served = {
        "s1": {"g1": 1.0, "g2": 2.0, "g3": 3.5},
        "s2": {"g1": -1.0, "g2": 0.0, "g3": 1.0, "g4": 4.0},
    }
    check = fb.compendium_check(
        own, served, {"s3": "not served"}, against="store", min_pearson_r=0.99
    )
    assert check.compared == ("s1", "s2")
    assert check.absent_there == {"s3": "not served"}
    assert check.n_values == 6
    assert check.max_abs_difference == pytest.approx(0.5)
    a1, a2 = check.agreements
    assert (a1.n_genes, a1.n_only_here, a1.n_only_there) == (3, 1, 0)
    assert (a2.n_genes, a2.n_only_here, a2.n_only_there) == (3, 0, 1)
    assert a2.pearson_r == pytest.approx(1.0)
    x = np.array([1.0, 2.0, 3.0, -1.0, 0.0, 1.0])
    y = np.array([1.0, 2.0, 3.5, -1.0, 0.0, 1.0])
    assert check.pooled_pearson_r == pytest.approx(float(np.corrcoef(x, y)[0, 1]))
    assert check.worst_sample_r == pytest.approx(a1.pearson_r)
    assert check.passed
    result = fb.compendium_check_result(check, "row")
    assert (result.name, result.passed) == ("row", True)
    strict = fb.compendium_check(
        own, served, {"s3": "x"}, against="store", min_pearson_r=0.9999
    )
    assert not strict.passed


def test_compendium_check_refuses_an_unexplained_absence() -> None:
    with pytest.raises(ValueError, match="without a reason"):
        fb.compendium_check(
            {"s1": {"g": 1.0}}, {}, {}, against="store", min_pearson_r=0.9
        )
    with pytest.raises(ValueError, match="no gene shared"):
        fb.compendium_check(
            {"s1": {"g": 1.0}}, {"s1": {"h": 1.0}}, {}, against="s", min_pearson_r=0.9
        )


def _record(screen_id: str, tag: str, value: float, n: int = 1) -> dict[str, object]:
    return {
        "experiment": {
            "phenotype": {"screen_id": screen_id, "environment_response": value},
            "genotype": {"perturbations": [{"systematic_gene_name": tag}] * n},
        }
    }


def test_stored_fitness_strips_the_organism_prefix() -> None:
    records = [_record("Keio:s1", "g1", 1.0), _record("s2", "g1", 2.0)]
    assert fb.stored_fitness(records, "Keio") == {"s1": {"g1": 1.0}, "s2": {"g1": 2.0}}
    with pytest.raises(ValueError, match="stored twice"):
        fb.stored_fitness([_record("s", "g", 1.0), _record("s", "g", 2.0)], "Keio")
    with pytest.raises(ValueError, match="one perturbation"):
        fb.stored_fitness([_record("s", "g", 1.0, n=2)], "Keio")


@pytest.mark.data
def test_the_deposit_has_the_pinned_size() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    path = fb.feba_path(os.environ["DATA_ROOT"])
    if not path.exists():
        pytest.skip(f"deposit absent: {path}")
    assert path.stat().st_size == fb.FEBA.n_bytes
