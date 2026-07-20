# tests/torchcell/datasets/scerevisiae/test_yeastphenome
# [[tests.torchcell.datasets.scerevisiae.test_yeastphenome]]
"""Build-smoke test for the curated YeastPhenome growth-phenome loader.

Builds a 3-screen subset (khozoie quinine + berry_gasch multi-assay + pagani haploid)
into a tmp root from the sha256-pinned raw mirror (no network), then asserts: schema
round-trip, the NPV z-score label, the homozygous-diploid genotype, that BOTH haploid
(BY4741) and homozygous-diploid (BY4743) complete-loss-of-function backgrounds are
ingested, that the fields curation does not carry (environment ``temperature``;
phenotype ``n_samples`` / uncertainty / ``sample_unit``) are typed ``ProvenanceGap``s
(never guessed), and -- the multi-screen invariant -- that the SAME (strain, condition)
measured by two assays (microarray vs barseq) yields TWO distinct records (near-replicate
screens, not duplicates). Skipped when the ``$DATA_ROOT`` raw mirror is absent (CI).
"""

import os
import os.path as osp
import shutil

import pytest
from dotenv import load_dotenv

from torchcell.datamodels.schema import EnvironmentResponseExperiment, MeasurementType
from torchcell.datasets.scerevisiae import yeastphenome as yp_mod
from torchcell.datasets.scerevisiae.yeastphenome import YeastPhenomeDataset

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
if DATA_ROOT is None:
    pytest.skip("requires DATA_ROOT data (absent in CI)", allow_module_level=True)

# 3-screen subset: khozoie (hom, single quinine condition) + berry_gasch (hom, same
# conditions run by microarray AND barseq -> near-replicate-screen distinctness) +
# pagani_arino (HAPLOID 'hap a' -> BY4741, the complete-loss-of-function background that
# is NOT homozygous-diploid).
_SUBSET = [
    s for s in yp_mod.SCREENS if s["pmid"] in ("19416971", "22102822", "17630978")
]
_RAW_MIRROR = osp.join(DATA_ROOT, "data/torchcell/yeastphenome/raw")
_RAW_FILES = [f"{s['pmid']}_{s['stem']}_valuez.txt" for s in _SUBSET]
if not all(osp.exists(osp.join(_RAW_MIRROR, f)) for f in _RAW_FILES):
    pytest.skip("requires the YeastPhenome raw mirror", allow_module_level=True)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, monkeypatch_module):
    """Build the 2-screen subset into a tmp root, seeding raw/ from the mirror (no net)."""
    monkeypatch_module.setattr(yp_mod, "SCREENS", _SUBSET)
    root = tmp_path_factory.mktemp("yeastphenome")
    raw = osp.join(str(root), "raw")
    os.makedirs(raw, exist_ok=True)
    for f in _RAW_FILES:
        shutil.copy(osp.join(_RAW_MIRROR, f), osp.join(raw, f))
    return YeastPhenomeDataset(root=str(root))


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (pytest's is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _find(dataset, pmid, compound):
    for i in range(len(dataset)):
        exp = dataset[i]["experiment"]
        if dataset[i]["publication"]["pubmed_id"] == pmid and (
            exp["environment"]["perturbations"][0]["compound"]["name"] == compound
        ):
            return dataset[i]
    raise AssertionError(f"no record for {pmid} / {compound}")


def test_multi_screen_count(dataset):
    # khozoie (1 env) + berry_gasch (7 encodable env), both ~4900 ORFs -> well over one screen.
    assert len(dataset) > 4228


def test_schema_round_trip_and_npv_label(dataset):
    rec = _find(dataset, "19416971", "quinine")
    exp = EnvironmentResponseExperiment(**rec["experiment"])
    assert exp.phenotype.measurement_type is MeasurementType.z_score
    assert exp.phenotype.environment_response is not None
    assert rec["reference"]["phenotype_reference"]["environment_response"] == 0.0


def test_homozygous_diploid_deletion_genotype(dataset):
    rec = _find(dataset, "19416971", "quinine")
    perts = rec["experiment"]["genotype"]["perturbations"]
    assert len(perts) == 1
    assert perts[0]["perturbation_type"] == "kanmx_deletion"
    assert rec["reference"]["genome_reference"]["ploidy"] == "diploid"
    assert rec["reference"]["genome_reference"]["strain"] == "BY4743"


def test_haploid_screens_are_ingested_as_haploid_background(dataset):
    """HAPLOID deletion collections are complete loss-of-function too -- they must be
    ingested (BY4741, ploidy=haploid), NOT excluded. Only heterozygous (dosage) is out.
    """
    haploid = [
        dataset[i]
        for i in range(len(dataset))
        if dataset[i]["publication"]["pubmed_id"] == "17630978"
    ]
    assert haploid, "haploid 'hap a' screen was not ingested"
    ref = haploid[0]["reference"]["genome_reference"]
    assert ref["ploidy"] == "haploid"
    assert ref["strain"] == "BY4741"
    # the deletion itself is the SAME total-absence perturbation as in a diploid screen
    pert = haploid[0]["experiment"]["genotype"]["perturbations"][0]
    assert pert["perturbation_type"] == "kanmx_deletion"
    assert pert["state"] == "absent"


def test_condition_and_media(dataset):
    env = _find(dataset, "19416971", "quinine")["experiment"]["environment"]
    assert env["perturbations"][0]["compound"]["name"] == "quinine"
    assert env["perturbations"][0]["concentration"]["value"] == 2.0
    assert env["media"]["name"] == "YPD"


def test_uncarried_fields_are_typed_gaps_not_guesses(dataset):
    exp = _find(dataset, "19416971", "quinine")["experiment"]
    assert exp["environment"]["temperature"] is None
    assert {g["field"] for g in exp["environment"]["provenance_gaps"]} == {
        "temperature"
    }
    ph = exp["phenotype"]
    ph_gap_fields = {g["field"] for g in ph["provenance_gaps"]}
    assert ph_gap_fields == {
        "n_samples",
        "environment_response_uncertainty",
        "sample_unit",
    }
    for f in ph_gap_fields:
        assert ph[f] is None
    for g in ph["provenance_gaps"] + exp["environment"]["provenance_gaps"]:
        assert g["reason"] == "not_carried_by_curation"
        assert g["looked_in"] is not None


def test_near_replicate_screens_are_distinct_records(dataset):
    """berry_gasch runs H2O2 0.4 mM by microarray AND barseq -> two distinct records for the
    same (gene, condition), differing only by the readout method recorded in units.
    """
    matches = []
    for i in range(len(dataset)):
        exp = dataset[i]["experiment"]
        if dataset[i]["publication"]["pubmed_id"] != "22102822":
            continue
        p = exp["environment"]["perturbations"][0]
        gene = exp["genotype"]["perturbations"][0]["systematic_gene_name"]
        if (
            p["compound"]["name"] == "hydrogen peroxide"
            and p["concentration"]["value"] == 0.4
            and gene == "YAL002W"
        ):
            matches.append(exp["phenotype"]["units"])
    assert len(matches) == 2  # microarray + barseq
    assert "microarray" in " ".join(matches) and "barseq" in " ".join(matches)
    assert matches[0] != matches[1]  # distinct readout -> distinct records
