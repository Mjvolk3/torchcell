# tests/torchcell/datasets/scerevisiae/test_smith2006_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_smith2006_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_smith2006_synthetic.py
"""Smith 2006 fatty-acid loader: ``process()`` and ``download()`` on a synthetic table.

``process()`` reads the legacy BIFF ``msb4100051-s1.xls`` through
``pd.read_excel(engine="xlrd")``. No library in the env can WRITE a BIFF .xls (xlwt is
not installed, xlrd 2 only reads), so the raw file is a placeholder and
``FattyAcidSmith2006Dataset._read_table`` (the file-loader boundary, one line) is
monkeypatched to return the table as that call would; everything downstream runs for
real. The genome is a duck-typed stub carrying the five attributes the name policy
reads.

Table rows (Systematic Name, Glucose (YEPD) flag, oleate, myristate, acetate):

    0 YAL001C  -    3    4  3.0    kept: 3 records (common name TFC3 from the genome)
    1 yal002w  NG   1    1  1.0    YEPD growth-control failure: dropped
    2 YAL003W  -    3    3  3.0    alias of YAL002W, which is present directly: collision
    3 YBR099W  -    1    2  2.5    alias of YBR002W (not present directly): 3 records
    4 YZZ999W  -    not a systematic pattern: unresolved
    5 YPL999C  -    systematic pattern, no alias: unresolved
    6 GHOST1   -    unresolved
    7 YAL001C  -    second row for YAL001C: duplicate
    8 YCR001W  -    -    1  1.0    blank oleate cell: 2 records

Records: 3 + 3 + 2 = 8 of 9 rows x 3 conditions = 27; the 19 dropped are YEPD 1 x 3,
unresolved 3 x 3, collision 1 x 3, duplicate 1 x 3 and one blank cell. References are
one per condition: oleate [0, 3], myristate [1, 4, 6], acetate [2, 5, 7].
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from torchcell.datamodels.media import YPBA, YPBM, YPBO
from torchcell.datamodels.schema import (
    AssayType,
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    ResponseCategory,
    SampleUnit,
    Temperature,
)
from torchcell.datasets.scerevisiae import smith2006 as s
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

_DATASET = "FattyAcidSmith2006Dataset"
_GENES = {"YAL001C", "YAL002W", "YBR002W", "YCR001W"}
_STANDARD = {"TFC3": ["YAL001C"], "VPS8": ["YAL002W"], "FAKE": ["YCR001W"]}
_NAN = float("nan")


class _Resolution:
    def __init__(self, status: str, systematic: str | None) -> None:
        self.status = status
        self.systematic_name = systematic

    @property
    def is_current_gene(self) -> bool:
        return self.status in ("current", "renamed")


class _FakeGenome:
    """The slice of ``SCerevisiaeGenome`` the Smith name policy reads."""

    gene_set = _GENES
    feature_index = {"standard_to_ids": _STANDARD}
    gene_attribute_table = pd.DataFrame({"ID": sorted(_GENES)})
    alias_to_systematic: dict[str, list[str]] = {
        "YAL003W": ["YAL002W"],
        "YBR099W": ["YBR002W"],
    }

    def resolve_gene_name(self, name: str) -> _Resolution:
        if name in ("TFC3", "VPS8"):
            return _Resolution("renamed", _STANDARD[name][0])
        return _Resolution("retired", None)


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _FakeGenome())


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            s.SYSTEMATIC_COL: [
                "YAL001C",
                "yal002w",
                "YAL003W",
                "YBR099W",
                "YZZ999W",
                "YPL999C",
                "GHOST1",
                "YAL001C",
                "YCR001W",
            ],
            s.YEPD_QC_COL: [_NAN, "NG", _NAN, _NAN, _NAN, _NAN, _NAN, _NAN, _NAN],
            "Oleate (YPBO)": [3, 1, 3, 1, 3, 3, 3, 3, _NAN],
            "Myristae (YPBM)": [4, 1, 3, 2, 3, 3, 3, 3, 1],
            "Acetate (YPBA)": [3.0, 1.0, 3.0, 2.5, 3.0, 3.0, 3.0, 3.0, 1.0],
        }
    )


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> s.FattyAcidSmith2006Dataset:
    monkeypatch.setattr(
        s.FattyAcidSmith2006Dataset, "_read_table", lambda self: _table()
    )
    raw = tmp_path / "smith" / "raw"
    raw.mkdir(parents=True)
    (raw / s.XLS_FILENAME).write_bytes(b"placeholder: _read_table is stubbed")
    return s.FattyAcidSmith2006Dataset(root=str(tmp_path / "smith"), genome=_genome())


_CARBON = {
    "oleate": (
        YPBO,
        0.1,
        Compound(
            name="oleic acid",
            inchikey="ZQPPMHVWECSIRJ-KTKRTIGZSA-N",
            smiles="CCCCCCCCC=CCCCCCCCC(=O)O",
            pubchem_cid=445639,
            chebi_id="CHEBI:16196",
        ),
    ),
    "myristate": (
        YPBM,
        0.125,
        Compound(
            name="myristic acid",
            inchikey="TUNFSRHWOTWDNC-UHFFFAOYSA-N",
            smiles="CCCCCCCCCCCCCC(=O)O",
            pubchem_cid=11005,
            chebi_id="CHEBI:28875",
        ),
    ),
    "acetate": (
        YPBA,
        2.0,
        Compound(
            name="acetic acid",
            inchikey="QTBSBXVTEAMEQO-UHFFFAOYSA-N",
            smiles="CC(=O)O",
            pubchem_cid=176,
            chebi_id="CHEBI:15366",
        ),
    ),
}
_CLEAR = (
    "clear-zone size around the cell patch on turbid fatty-acid agar, scored by visual "
    "inspection: 4=larger than wild type, 3=wild type, 2=less than wild type, 1=small or "
    "not detectable"
)
_GROWTH = (
    "growth of the cell patch on nonfermentable acetate, scored by visual inspection: "
    "3=wild-type, 2=moderate, 1=little/no growth (an undocumented 2.5=intermediate is "
    "kept verbatim from the released table)"
)


def _environment(condition: str) -> Environment:
    media, percent, compound = _CARBON[condition]
    return Environment(
        media=media,
        temperature=Temperature(value=30.0),
        perturbations=[
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source,
                magnitude=Concentration(
                    value=percent, unit=ConcentrationUnit.percent_w_v
                ),
                agent=compound,
            )
        ],
        aerobicity="aerobic",
        duration_hours=72.0,
    )


def _assay(condition: str) -> tuple[AssayType, str]:
    if condition == "acetate":
        return AssayType.colony_size_array, _GROWTH
    return AssayType.halo_zone, _CLEAR


def _experiment(
    orf: str,
    common: str,
    condition: str,
    score: float,
    category: ResponseCategory,
    label: str,
) -> dict[str, object]:
    assay, units = _assay(condition)
    return EnvironmentResponseExperiment(
        dataset_name=_DATASET,
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=common
                )
            ]
        ),
        environment=_environment(condition),
        phenotype=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.ordinal,
            assay_type=assay,
            environment_response=score,
            category=category,
            category_label=label,
            n_samples=3,
            sample_unit=SampleUnit.biological_replicate,
            units=units,
        ),
    ).model_dump()


def _reference(condition: str) -> dict[str, object]:
    assay, units = _assay(condition)
    return EnvironmentResponseExperimentReference(
        dataset_name=_DATASET,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4742"
        ),
        environment_reference=_environment(condition),
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.ordinal,
            assay_type=assay,
            category=ResponseCategory.no_change,
            category_label="wild_type",
            n_samples=3,
            sample_unit=SampleUnit.biological_replicate,
            units=units,
        ),
    ).model_dump()


_C = ResponseCategory
_EXPECTED = [
    _experiment("YAL001C", "TFC3", "oleate", 3.0, _C.no_change, "wild_type"),
    _experiment("YAL001C", "TFC3", "myristate", 4.0, _C.enhanced, "enhanced"),
    _experiment("YAL001C", "TFC3", "acetate", 3.0, _C.no_change, "wild_type"),
    _experiment("YBR002W", "YBR002W", "oleate", 1.0, _C.severely_reduced, "defective"),
    _experiment("YBR002W", "YBR002W", "myristate", 2.0, _C.reduced, "reduced"),
    _experiment(
        "YBR002W", "YBR002W", "acetate", 2.5, _C.mildly_reduced, "intermediate"
    ),
    _experiment(
        "YCR001W", "YCR001W", "myristate", 1.0, _C.severely_reduced, "defective"
    ),
    _experiment("YCR001W", "YCR001W", "acetate", 1.0, _C.severely_reduced, "poor"),
]
_REFERENCES = ["oleate", "myristate", "acetate", "oleate", "myristate", "acetate"]
_REFERENCES += ["myristate", "acetate"]


def test_eight_records_in_strain_then_condition_order(
    dataset: s.FattyAcidSmith2006Dataset,
) -> None:
    assert len(dataset) == 8
    for i, expected in enumerate(_EXPECTED):
        assert dataset[i]["experiment"] == expected
        assert dataset[i]["reference"] == _reference(_REFERENCES[i])
    assert (
        dataset[0]["publication"]
        == Publication(
            pubmed_id="16738555",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/16738555/",
            doi="10.1038/msb4100051",
            doi_url="https://doi.org/10.1038/msb4100051",
        ).model_dump()
    )


def test_drop_log_accounts_for_every_missing_record(
    dataset: s.FattyAcidSmith2006Dataset,
) -> None:
    """27 source records, 8 kept, 19 dropped across the five rules."""
    log = json.loads(
        (Path(dataset.root) / "preprocess" / "dropped_records.json").read_text()
    )
    assert (
        log["dataset"],
        log["source_records"],
        log["kept_records"],
        log["dropped_records"],
    ) == (_DATASET, 27, 8, 19)
    assert [
        (rule["rule"], rule["scope"], rule["n_records"], rule["items"])
        for rule in log["rules"]
    ] == [
        ("strain_failed_the_yepd_growth_control", "strain", 3, ["YAL002W"]),
        (
            "systematic_name_does_not_resolve_to_a_current_orf",
            "strain",
            9,
            ["GHOST1", "YPL999C", "YZZ999W"],
        ),
        (
            "alias_resolution_collides_with_a_directly_present_orf",
            "strain",
            3,
            ["YAL003W"],
        ),
        ("second_row_for_an_already_resolved_orf", "strain", 3, ["YAL001C"]),
        ("condition_cell_is_blank", "cell", 1, []),
    ]


def test_side_files_gene_set_and_reference_index(
    dataset: s.FattyAcidSmith2006Dataset,
) -> None:
    preprocess = Path(dataset.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR002W",
        "YCR001W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert sorted(entry["member_indices"] for entry in index) == [
        [0, 3],
        [1, 4, 6],
        [2, 5, 7],
    ]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert (manifest["dataset_name"], manifest["loader_class"]) == ("smith", _DATASET)
    assert dataset.raw_file_names == ["msb4100051-s1.xls"]
    assert dataset.experiment_class is EnvironmentResponseExperiment
    assert dataset.reference_class is EnvironmentResponseExperimentReference
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()
    frame = {"untouched": True}
    assert dataset.preprocess_raw(frame) is frame


def test_process_refuses_to_run_without_a_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        s.FattyAcidSmith2006Dataset, "_read_table", lambda self: _table()
    )
    raw = tmp_path / "smith" / "raw"
    raw.mkdir(parents=True)
    (raw / s.XLS_FILENAME).write_bytes(b"placeholder")
    with pytest.raises(
        RuntimeError,
        match=r"FattyAcidSmith2006Dataset requires a genome for systematic-name "
        r"resolution; inject SCerevisiaeGenome\(\.\.\.\)",
    ):
        s.FattyAcidSmith2006Dataset(root=str(tmp_path / "smith"))


_XLS_BYTES = b"synthetic Supplementary Table 1 bytes"


def _deposit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Repoint the pin at the synthetic bytes, then deposit them into the mirror."""
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setattr(s, "XLS_SHA256", hashlib.sha256(_XLS_BYTES).hexdigest())
    source = tmp_path / "msb4100051-s1.xls"
    source.write_bytes(_XLS_BYTES)
    s.deposit_raw_mirror(xls_path=source, retrieved_at="2026-09-27")
    return data_root


def test_download_links_the_deposited_mirror_file_then_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deposit_raw_mirror`` records one Europe PMC ``zip_member`` retrieval with no
    container hash; ``download()`` then symlinks the mirror file into ``raw/``.
    """
    data_root = _deposit(tmp_path, monkeypatch)
    manifest = s.load_manifest()
    (record,) = manifest.files
    assert (record.path, record.sha256, record.bytes) == (
        "data/msb4100051-s1.xls",
        hashlib.sha256(_XLS_BYTES).hexdigest(),
        len(_XLS_BYTES),
    )
    assert record.retrieval is not None
    assert record.retrieval.params == {
        "url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1681483/"
        "supplementaryFiles",
        "member": "msb4100051-s1.xls",
        "container_sha256": None,
    }
    monkeypatch.setattr(
        s.FattyAcidSmith2006Dataset, "_read_table", lambda self: _table()
    )
    dataset = s.FattyAcidSmith2006Dataset(
        root=str(tmp_path / "smith"), genome=_genome()
    )
    link = tmp_path / "smith" / "raw" / s.XLS_FILENAME
    assert os.readlink(link) == str(
        data_root / "torchcell-raw" / s.CITATION_KEY / "data" / s.XLS_FILENAME
    )
    assert len(dataset) == 8
    with pytest.raises(KeyError, match="data/nope is not in the raw-mirror manifest"):
        s.manifest_sha256(manifest, "data/nope")


def test_download_refuses_changed_or_missing_mirror_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _deposit(tmp_path, monkeypatch)
    mirror = data_root / "torchcell-raw" / s.CITATION_KEY / "data" / s.XLS_FILENAME
    mirror.write_bytes(b"changed")
    got = hashlib.sha256(b"changed").hexdigest()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"msb4100051-s1.xls sha256 mismatch: got {got}, "
            f"expected {hashlib.sha256(_XLS_BYTES).hexdigest()}"
        ),
    ):
        s.FattyAcidSmith2006Dataset(root=str(tmp_path / "a"), genome=_genome())
    mirror.unlink()
    with pytest.raises(
        RuntimeError,
        match=re.escape(f"required raw artifact missing from mirror: {mirror}"),
    ):
        s.FattyAcidSmith2006Dataset(root=str(tmp_path / "b"), genome=_genome())


def test_deposit_refuses_bytes_that_are_not_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unpinned bytes are refused before anything is copied; a mirror file that later
    differs from the (repointed) pin is refused rather than overwritten.
    """
    source = tmp_path / "other.xls"
    source.write_bytes(b"other")
    got = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"{source} sha256 mismatch: got {got}, expected {s.XLS_SHA256}"
        ),
    ):
        s.deposit_raw_mirror(xls_path=source, data_root=str(tmp_path / "dr"))
    data_root = _deposit(tmp_path, monkeypatch)
    dest = data_root / "torchcell-raw" / s.CITATION_KEY / "data" / s.XLS_FILENAME
    dest.write_bytes(b"drifted")
    with pytest.raises(
        RuntimeError,
        match=re.escape(f"{dest} exists with a different sha256; refusing"),
    ):
        s.deposit_raw_mirror(xls_path=tmp_path / "msb4100051-s1.xls")
