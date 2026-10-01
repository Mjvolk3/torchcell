# tests/torchcell/datasets/scerevisiae/test_yeastphenome
# [[tests.torchcell.datasets.scerevisiae.test_yeastphenome]]
"""Tests for the curated YeastPhenome growth-phenome loader.

The data-gated tests (``--data``) build a 3-screen subset (khozoie quinine + berry_gasch
multi-assay + pagani haploid) into a tmp root from the sha256-pinned raw mirror (no
network), then assert: schema round-trip, the NPV z-score label, the homozygous-diploid
genotype, that BOTH haploid (BY4741) and homozygous-diploid (BY4743) complete-loss-of-
function backgrounds are ingested, that the fields curation does not carry (environment
``temperature``; phenotype ``n_samples`` / uncertainty / ``sample_unit``) are typed
``ProvenanceGap``s (never guessed), and -- the multi-screen invariant -- that the SAME
(strain, condition) measured by two assays (microarray vs barseq) yields TWO distinct
records (near-replicate screens, not duplicates). They skip when the ``$DATA_ROOT`` raw
mirror is absent (CI).

2026.09.30 (Phase 15): hermetic tests added beside them (the import-time ``load_dotenv()``
and the module-level skips removed, the data gate moved onto the mirror tests). The
record grammar is pinned in ``test_yeastphenome_synthetic.py``; this file pins what is
left of the module:

- the pinned screen list as shipped: 47 screens, 47 distinct PMIDs, a 64-hex ``valuez``
  pin each, none of the 16 already-built primary PMIDs the module comment excludes
  (Costanzo 27708008/33958448, Kemmeren 24766815, Kuzmin 29674565/32586993, Baryshnikova
  21076421, Auesukaree 19638689, Hoepfner 24360837, Wildenhain 27136353, Hillenmeyer
  18420932, Smith 16738555/26956608, Ohya 16365294, Ozaydin 22918085, Vanacloig 35883225,
  Mota 38419072), and 47 distinct raw file names although PMIDs 33924665 and 34944020
  share the stem ``jin_liu_2021`` (the ``<pmid>_`` prefix keeps them apart).
- ``download`` against a fake ``urlopen``: a present file is skipped, a missing one is
  fetched from ``<YP_RAW>/<pmid>/<stem>_valuez.txt`` with a ``Mozilla/5.0`` user agent and
  a 120 s timeout and written byte for byte, and bytes off the pin raise
  "<stem> valuez sha256 mismatch: got <digest>, expected <pin>" with nothing written.
- ``transform_item`` on a one-column synthetic screen (hom, colony size, furfural 10 mM,
  YPD; rows YAL001C 1.5 and Q0250 -0.75), the inert ``preprocess_raw`` and
  ``create_experiment`` hooks, and ``main`` with the class replaced by a recorder.
"""

import hashlib
import os
import os.path as osp
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError, verify_raw_files
from torchcell.datamodels.schema import (
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    MeasurementType,
    Publication,
)
from torchcell.datasets.scerevisiae import yeastphenome as yp_mod
from torchcell.datasets.scerevisiae.yeastphenome import YeastPhenomeDataset

_DATA_ROOT = os.environ["DATA_ROOT"]

# 3-screen subset: khozoie (hom, single quinine condition) + berry_gasch (hom, same
# conditions run by microarray AND barseq -> near-replicate-screen distinctness) +
# pagani_arino (HAPLOID 'hap a' -> BY4741, the complete-loss-of-function background that
# is NOT homozygous-diploid).
_SUBSET = [
    s for s in yp_mod.SCREENS if s["pmid"] in ("19416971", "22102822", "17630978")
]
_RAW_MIRROR = osp.join(_DATA_ROOT, "data/torchcell/yeastphenome/raw")
_RAW_FILES = [f"{s['pmid']}_{s['stem']}_valuez.txt" for s in _SUBSET]
# The shipped screen list, captured at import: the data fixture below replaces
# ``SCREENS`` for the rest of the module once it runs.
_SHIPPED_SCREENS = list(yp_mod.SCREENS)
_needs_mirror = pytest.mark.skipif(
    not all(osp.exists(osp.join(_RAW_MIRROR, f)) for f in _RAW_FILES),
    reason="requires the YeastPhenome raw mirror",
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, monkeypatch_module):
    """Build the 2-screen subset into a tmp root, seeding raw/ from the mirror (no net)."""
    monkeypatch_module.setattr(yp_mod, "SCREENS", _SUBSET)
    root = tmp_path_factory.mktemp("yeastphenome")
    raw = osp.join(str(root), "raw")
    os.makedirs(raw, exist_ok=True)
    for f in _RAW_FILES:
        shutil.copy(osp.join(_RAW_MIRROR, f), osp.join(raw, f))
    # Module scope is set up before the conftest's function-scoped real-pin restore, so
    # this real build puts the real build-time check back itself (issue #561).
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(yp_mod, "verify_raw_files", verify_raw_files)
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


@pytest.mark.data
@_needs_mirror
def test_multi_screen_count(dataset):
    # khozoie (1 env) + berry_gasch (7 encodable env), both ~4900 ORFs -> well over one screen.
    assert len(dataset) > 4228


@pytest.mark.data
@_needs_mirror
def test_schema_round_trip_and_npv_label(dataset):
    rec = _find(dataset, "19416971", "quinine")
    exp = EnvironmentResponseExperiment(**rec["experiment"])
    assert exp.phenotype.measurement_type is MeasurementType.z_score
    assert exp.phenotype.environment_response is not None
    assert rec["reference"]["phenotype_reference"]["environment_response"] == 0.0


@pytest.mark.data
@_needs_mirror
def test_homozygous_diploid_deletion_genotype(dataset):
    rec = _find(dataset, "19416971", "quinine")
    perts = rec["experiment"]["genotype"]["perturbations"]
    assert len(perts) == 1
    assert perts[0]["perturbation_type"] == "kanmx_deletion"
    assert rec["reference"]["genome_reference"]["ploidy"] == "diploid"
    assert rec["reference"]["genome_reference"]["strain"] == "BY4743"


@pytest.mark.data
@_needs_mirror
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


@pytest.mark.data
@_needs_mirror
def test_condition_and_media(dataset):
    env = _find(dataset, "19416971", "quinine")["experiment"]["environment"]
    assert env["perturbations"][0]["compound"]["name"] == "quinine"
    assert env["perturbations"][0]["concentration"]["value"] == 2.0
    assert env["media"]["name"] == "YPD"


@pytest.mark.data
@_needs_mirror
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


@pytest.mark.data
@_needs_mirror
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


def test_parse_condition_defensin_is_peptide_biologic():
    """The four PMID-31451498 plant defensins parse to a peptide BiologicPerturbation
    (sequence-identified, NOT a small molecule) -- the audit DEFECT fix. Pure unit test
    of ``_parse_condition`` (no build/network).
    """
    from torchcell.datamodels.schema import (
        BiologicAgentClass,
        BiologicPerturbation,
        SmallMoleculePerturbation,
    )

    for name in ("DmAMP1", "NaD1", "NbD6", "SBI6"):
        pert = yp_mod._parse_condition(f"{name} [10 uM]")
        assert isinstance(pert, BiologicPerturbation)
        assert pert.agent_class == BiologicAgentClass.peptide
        assert pert.name == name
        assert pert.concentration.value == 10.0

    # a normal dosed compound still routes to a resolver-filled SmallMoleculePerturbation
    small = yp_mod._parse_condition("furfural [10 mM]")
    assert isinstance(small, SmallMoleculePerturbation)
    assert small.compound.name == "furfural"
    assert small.compound.inchikey is not None  # resolved via the pinned table


# Already-built primaries the module comment lists as excluded up front.
_EXCLUDED_PRIMARY_PMIDS = {
    "27708008",
    "33958448",
    "24766815",
    "29674565",
    "32586993",
    "21076421",
    "19638689",
    "24360837",
    "27136353",
    "18420932",
    "16738555",
    "26956608",
    "16365294",
    "22918085",
    "35883225",
    "38419072",
}

_SCREEN = {"pmid": "99999903", "stem": "synthetic_c", "valuez_sha256": "0" * 64}
_HEADER = "orf\thom | growth (colony size) | furfural [10 mM] | YPD | synthetic"
_BODY = f"{_HEADER}\nYAL001C\t1.5\nQ0250\t-0.75\n"


def _raw_name(screen: dict[str, str]) -> str:
    return f"{screen['pmid']}_{screen['stem']}_valuez.txt"


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> YeastPhenomeDataset:
    """One synthetic screen, one encodable column, two ORF rows."""
    monkeypatch.setattr(yp_mod, "SCREENS", [_SCREEN])
    root = tmp_path / "yeastphenome"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / _raw_name(_SCREEN)).write_text(_BODY)
    return YeastPhenomeDataset(root=str(root))


def test_shipped_screens_are_pinned_distinct_and_exclude_built_primaries() -> None:
    """47 screens, one per PMID, each with a 64-hex pin; no PMID of an already-built
    primary (so nothing is double-counted); and 47 distinct raw file names even though
    two PMIDs share the stem ``jin_liu_2021``.
    """
    pmids = [s["pmid"] for s in _SHIPPED_SCREENS]
    assert len(_SHIPPED_SCREENS) == 47
    assert len(set(pmids)) == 47
    assert set(pmids) & _EXCLUDED_PRIMARY_PMIDS == set()
    assert all(
        len(s["valuez_sha256"]) == 64
        and set(s["valuez_sha256"]) <= set("0123456789abcdef")
        for s in _SHIPPED_SCREENS
    )
    assert sorted(
        s["pmid"] for s in _SHIPPED_SCREENS if s["stem"] == "jin_liu_2021"
    ) == ["33924665", "34944020"]
    names = [_raw_name(s) for s in _SHIPPED_SCREENS]
    assert len(set(names)) == 47
    assert "33924665_jin_liu_2021_valuez.txt" in names
    assert "34944020_jin_liu_2021_valuez.txt" in names


class _Response:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.data


def test_download_skips_present_files_and_fetches_missing_ones_from_the_pin(
    synthetic: YeastPhenomeDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With screen C already under ``raw/`` and a second screen D missing, ``download``
    opens exactly one URL, D's file at the v1.0 pin, with the Mozilla user agent and a
    120 s timeout, and writes the returned bytes (whose sha256 is D's pin) verbatim.
    """
    body = b"orf\tcol\nYAL001C\t0.5\n"
    screen_d = {
        "pmid": "99999904",
        "stem": "synthetic_d",
        "valuez_sha256": hashlib.sha256(body).hexdigest(),
    }
    monkeypatch.setattr(yp_mod, "SCREENS", [_SCREEN, screen_d])
    opened: list[tuple[str, dict[str, str], int]] = []

    def fake_urlopen(request: Any, timeout: int) -> _Response:
        opened.append((request.full_url, dict(request.headers), timeout))
        return _Response(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    synthetic.download()
    assert opened == [
        (
            "https://raw.githubusercontent.com/yeastphenome/yp-data/"
            "83e2917bf86955ec6ba66dc70ff2ed0fe24ecbe8/Datasets/99999904/"
            "synthetic_d_valuez.txt",
            {"User-agent": "Mozilla/5.0"},
            120,
        )
    ]
    raw = Path(synthetic.raw_dir)
    assert (raw / _raw_name(screen_d)).read_bytes() == body
    assert (raw / _raw_name(_SCREEN)).read_text() == _BODY


def test_download_refuses_bytes_off_the_pin_and_writes_nothing(
    synthetic: YeastPhenomeDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fetched bytes whose sha256 is not the screen's pin raise
    ``RawSha256MismatchError`` naming the URL and both digests, and neither the
    destination file nor a ``.partial`` is created.
    """
    screen_e = {"pmid": "99999905", "stem": "synthetic_e", "valuez_sha256": "1" * 64}
    monkeypatch.setattr(yp_mod, "SCREENS", [screen_e])
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: _Response(b"drifted")
    )
    got = hashlib.sha256(b"drifted").hexdigest()
    before = sorted(p.name for p in Path(synthetic.raw_dir).iterdir())
    with pytest.raises(RawSha256MismatchError) as err:
        synthetic.download()
    assert str(err.value) == (
        f"sha256 mismatch for {yp_mod.YP_RAW}/99999905/synthetic_e_valuez.txt: "
        f"expected {'1' * 64}, observed {got}"
    )
    assert not (Path(synthetic.raw_dir) / _raw_name(screen_e)).exists()
    assert sorted(p.name for p in Path(synthetic.raw_dir).iterdir()) == before


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): with every screen's NPV file already in ``raw/``
    PyG skips ``download()``, so ``process()`` verifies each against its
    ``valuez_sha256`` first. The first screen off its pin raises
    ``RawSha256MismatchError`` naming it and both digests; no store is written.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(yp_mod, [_raw_name(s) for s in yp_mod.SCREENS])
    raw = staged.root / "raw" / _raw_name(yp_mod.SCREENS[0])
    with pytest.raises(RawSha256MismatchError) as err:
        YeastPhenomeDataset(root=str(staged.root))
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected "
        "264a4f2c6d8da3c66ea376f04554d7e950d825cd853da2a47c7439f3860ab149, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


def test_items_retype_through_the_environment_response_classes(
    synthetic: YeastPhenomeDataset,
) -> None:
    """``transform_item`` rebuilds both stored items through the declared classes (lines
    510 to 518) and each dumps back to the stored dictionary; the label is the NPV as a
    z-score (1.5 and -0.75) against a reference of 0.0, and the publication is PMID
    99999903 with no DOI.
    """
    assert len(synthetic) == 2
    npvs = []
    for index in range(2):
        item = synthetic[index]
        typed = synthetic.transform_item(item)
        assert type(typed["experiment"]) is EnvironmentResponseExperiment
        assert type(typed["reference"]) is EnvironmentResponseExperimentReference
        assert typed["experiment"].model_dump() == item["experiment"]
        assert typed["reference"].model_dump() == item["reference"]
        assert typed["publication"] == Publication(
            pubmed_id="99999903", pubmed_url="https://pubmed.ncbi.nlm.nih.gov/99999903/"
        )
        assert typed["experiment"].phenotype.measurement_type is MeasurementType.z_score
        assert typed["reference"].phenotype_reference.environment_response == 0.0
        npvs.append(typed["experiment"].phenotype.environment_response)
    assert npvs == [1.5, -0.75]


def test_inline_build_leaves_the_generic_hooks_inert(
    synthetic: YeastPhenomeDataset,
) -> None:
    """Records are built inside ``process``: ``preprocess_raw`` returns the frame it was
    given, unchanged, and ``create_experiment`` raises a bare ``NotImplementedError``.
    """
    frame = pd.DataFrame({"orf": ["YAL001C"], "npv": [1.5]})
    returned = synthetic.preprocess_raw(frame)
    assert returned is frame
    assert returned.to_dict("list") == {"orf": ["YAL001C"], "npv": [1.5]}
    with pytest.raises(NotImplementedError) as info:
        synthetic.create_experiment()
    assert str(info.value) == ""


def test_main_builds_under_data_root_and_prints_the_first_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` constructs the dataset at ``$DATA_ROOT/data/torchcell/yeastphenome`` (no
    other argument) and prints its length and first item; the class is a recorder.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    calls: list[dict[str, Any]] = []

    class _Recorder:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def __len__(self) -> int:
            return 5

        def __getitem__(self, index: int) -> str:
            return f"item[{index}]"

    monkeypatch.setattr(yp_mod, "YeastPhenomeDataset", _Recorder)
    yp_mod.main()
    assert calls == [{"root": f"{tmp_path}/data/torchcell/yeastphenome"}]
    assert capsys.readouterr().out == "len = 5\nitem[0]\n"
