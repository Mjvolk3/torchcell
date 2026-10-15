# tests/torchcell/datasets/scerevisiae/test_hackett2020.py
# [[tests.torchcell.datasets.scerevisiae.test_hackett2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_hackett2020.py
"""Hackett 2020 IDEA loader and its verifier, end to end on synthetic files.

``DATA_ROOT`` points into ``tmp_path``. The raw mirror is written by the loader's own
``deposit_raw_mirror`` from synthetic retrievals (a tall zip, the site page, two GEO
series matrices) and a synthetic library mirror holding Table EV1 and Dataset EV2
under the library manifest's sha256. Module constants stating the REAL release's size
(6,175 genes, 217 courses, 1,693 arrays, 80 validation arrays) and every pin are
monkeypatched to the synthetic one's.

The synthetic release has four genes (AAC1, ACA1, HMX1, OCT1) and three IDEA courses:
ACA1 induced in SMY2049 under phosphate limitation at 0 and 5 min, the GEV control
SMY69 under methionine limitation at 0 and 2.5 min, and CBF1 in SMY22 (no Table EV1
row: refused) at 0, 5 and 15 min. Dataset EV2 holds one validation course, HMX1 at 0
and 5 min, with its OCT1 cell turned into 2019-10-01 the way Excel did it. Every
layer's value at gene ``g`` (index ``i``) and array time ``t`` is a closed form below,
so each stored number is asserted exactly.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from torchcell.datamodels.media import (
    CHEMOSTAT_MET_LIMITED_HACKETT2020,
    CHEMOSTAT_P_LIMITED_HACKETT2020,
)
from torchcell.datamodels.schema import (
    AlleleEdit,
    InduciblePromoterInsertionPerturbation,
    InductionTimecourseExpressionPhenotype,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import hackett2020 as h
from torchcell.literature.manifest import (
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.verification import induction_timecourse as v
from torchcell.verification.report import Provenance
from torchcell.verification.runners import stream_records

GENES = ["AAC1", "ACA1", "HMX1", "OCT1"]
SYSTEMATIC = {
    "AAC1": "YMR056C",
    "ACA1": "YER045C",
    "HMX1": "YLR205C",
    "OCT1": "YKL134C",
    "CBF1": "YJR060W",
}
ZEV_TAIL = "leu2d0::ACT1pr-ZEV-NatMX ybr032wd::LEU2pr(-314 --> -1)-LEU2"
GEV_MET6 = "(Pgal10-1+Gal1)::loxP gal4d::LEU2 leu2d0::pACT1-GEV-NatMX met6d::HphMX4"
STRAIN_ROWS = [
    ["SMY2049", f"KanMX-Z3EVpr-ACA1, {ZEV_TAIL}", "TF", "This manuscript"],
    ["SMY69", GEV_MET6, "Control", "McIsaac et al (2012, MBoC)"],
    ["SMY254", ZEV_TAIL, "Control", "McIsaac et al (2013, NAR)"],
    [
        "SMY257",
        "leu2d0::ACT1pr-4ZEV-NatMX ybr032wd::LEU2pr(-314 --> -1)-LEU2 KanMX-4ZEVpr-GCN4",
        "TF",
        "McIsaac et al (2013, NAR)",
    ],
    [
        "DBY12021",
        "(Pgal10-1+Gal1)::loxP leu2d0::ACT1pr-GEV-NatMX gal4d::LEU2",
        "Control",
        "McIsaac et al (2011, MBoC)",
    ],
    [
        "SMY2047",
        "HAP1+::pACT1-Z3EV-NatMX::eno2term urad0 can1dSte2pr-Sphis5 his3d1 lypd0 "
        "URA3::Z3EVpr-GLN3",
        "TF",
        "This manuscript",
    ],
    [
        "SMY2605",
        "HAP1+::pACT1-Z3EV-NatMX::eno2term ura∆0 can1∆::Ste2pr-Sphis5 his3∆1 lyp∆0 "
        "URA3::Z3EVpr-HMX1",
        "Prediction",
        "This manuscript",
    ],
]
#: (TF, strain, date, restriction, mechanism, times) of the synthetic IDEA table.
TALL_COURSES = [
    ("ACA1", "SMY2049", "02/29/2016", "P", "ZEV", [0.0, 5.0]),
    ("GEV", "SMY69", "01/01/2015", "M", "GEV", [0.0, 2.5]),
    ("CBF1", "SMY22", "01/01/2015", "P", "GEV", [0.0, 5.0, 15.0]),
]


def _cleaned(i: int, t: float) -> float:
    return 0.0 if t == 0 else round(0.25 * (i + 1) + t / 100, 6)


def _tall_row(course: tuple[Any, ...], t: float, i: int, gene: str) -> list[Any]:
    tf, strain, date, restriction, mechanism, _ = course
    cleaned = _cleaned(i, t)
    kept = cleaned if i in (1, 2) else 0.0  # ACA1 and HMX1 carry signal
    shrunk = round(kept * 0.5, 6)
    return [
        tf,
        strain,
        date,
        restriction,
        mechanism,
        t,
        gene,
        100.0 + i,  # green_median
        200.0 + i + t,  # red_median
        0.0 if t == 0 else round(cleaned + 0.01, 6),  # log2_ratio
        cleaned,
        0.1 + i / 100,  # log2_noise_model
        kept,  # zth2d
        kept,  # selected
        shrunk,
    ]


def _tall_zip() -> bytes:
    rows = [
        _tall_row(course, t, i, gene)
        for course in TALL_COURSES
        for gene_index, gene in enumerate(GENES)
        for t in course[5]
        for i in [gene_index]
    ]
    frame = pd.DataFrame(rows, columns=h.TALL_COLUMNS)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(h.TALL_MEMBER, frame.to_csv(sep="\t", index=False))
    return buffer.getvalue()


def _xlsx(frames: dict[str, pd.DataFrame], header: bool = True) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet, index=False, header=header)
    return buffer.getvalue()


def _strain_table() -> bytes:
    cells = [
        ["Table EV1: Strains", None, None, None],
        ["Strain", "Genotype", "Type", "Experiment/Microarray Source"],
        *STRAIN_ROWS,
    ]
    return _xlsx({h.STRAINS_SHEET: pd.DataFrame(cells)}, header=False)


def _validation() -> bytes:
    rows = []
    for t in (0, 5):
        for i, gene in enumerate(GENES):
            cleaned = _cleaned(i, t)
            rows.append(
                [
                    "HMX1",
                    "P",
                    "ZEV",
                    t,
                    datetime(2019, 10, 1) if gene == "OCT1" else gene,
                    50.0 + i,
                    60.0 + i + t,
                    0.0 if t == 0 else round(cleaned + 0.02, 6),
                    cleaned,
                ]
            )
    frame = pd.DataFrame(rows, columns=h.VALIDATION_COLUMNS)
    return _xlsx({"Legend": pd.DataFrame({"x": ["legend"]}), h.VALIDATION_SHEET: frame})


def _ev1() -> bytes:
    """Dataset EV1: the served IDEA arrays' shrunken values, gene x array."""
    data: dict[str, list[Any]] = {
        "GENE": [datetime(2019, 10, 1) if g == "OCT1" else g for g in GENES]
    }
    for course in TALL_COURSES:
        tc = h.timecourse_id(*course[:5])
        for t in course[5]:
            data[f"{tc}_{t}"] = [
                _tall_row(course, t, i, g)[14] for i, g in enumerate(GENES)
            ]
    return _xlsx(
        {"Legend": pd.DataFrame({"x": ["legend"]}), "Dataset EV1": pd.DataFrame(data)}
    )


def _geo(titles: list[str], strains: list[str]) -> bytes:
    lines = [
        '!Series_geo_accession\t"GSE142864"',
        "!Sample_title\t" + "\t".join(f'"{t}"' for t in titles),
        "!Sample_characteristics_ch1\t" + "\t".join(f'"strain: {s}"' for s in strains),
        "!Sample_characteristics_ch1\t"
        + "\t".join('"background: s288c"' for _ in titles),
        "!series_matrix_table_begin",
        '"ID_REF"',
    ]
    return gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0)


GEO_IDEA = _geo(
    ["SMY2049_ACA1_t0", "SMY2049_ACA1_t5", "SMY69_GEV_M_t0", "SMY69_GEV_M_t2.5"],
    ["SMY2049", "SMY2049", "SMY69", "SMY69"],
)
GEO_VALIDATION = _geo(["SMY2605_HMX1_t0", "SMY2605_HMX1_t5"], ["SMY2605", "SMY2605"])
TALL = _tall_zip()
STRAINS = _strain_table()
VALIDATION = _validation()
EV1 = _ev1()
SITE_JS = b"var w='https://storage.googleapis.com/calico-website-pin-public-bucket';"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(root))
    for name, value in {
        "N_GENES": 4,
        "N_TALL_TIMECOURSES": 3,
        "N_TALL_ARRAYS": 7,
        "N_VALIDATION_TIMECOURSES": 1,
        "N_VALIDATION_ARRAYS": 2,
        "TALL_ZIP_SHA256": _sha(TALL),
        "STRAINS_SHA256": _sha(STRAINS),
        "VALIDATION_SHA256": _sha(VALIDATION),
        "SHRUNKEN_EV1_SHA256": _sha(EV1),
        "GEO_GPL16244_SHA256": _sha(GEO_IDEA),
        "GEO_GPL20785_SHA256": _sha(GEO_VALIDATION),
        "TALL_ZIP_GCS_MD5_BASE64": base64.b64encode(
            hashlib.md5(TALL, usedforsecurity=False).digest()
        ).decode(),
    }.items():
        monkeypatch.setattr(h, name, value)
    monkeypatch.setattr(
        h,
        "RAW_SHA256",
        {
            h.TALL_ZIP_NAME: _sha(TALL),
            h.STRAINS_NAME: _sha(STRAINS),
            h.VALIDATION_NAME: _sha(VALIDATION),
        },
    )
    monkeypatch.setattr(h, "RELEASE_TF_TOKENS_WITHOUT_AN_ORF", SYSTEMATIC)
    library = root / "torchcell-library" / h.CITATION_KEY
    (library / "si").mkdir(parents=True)
    files = []
    for rel, data in (
        (h.STRAINS_LIBRARY, STRAINS),
        (h.VALIDATION_LIBRARY, VALIDATION),
        (h.SHRUNKEN_EV1_LIBRARY, EV1),
    ):
        (library / rel).write_bytes(data)
        url = h.PMC_BUCKET + rel
        files.append(
            ArtifactRecord(
                path=rel,
                role="si_data",
                bytes=len(data),
                sha256=_sha(data),
                source=url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.pmc_cloud,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.pmc_cloud_object",
                    sha256=_sha(data),
                    retrieved_at="2026-10-11",
                ),
            )
        )
    (library / "manifest.json").write_text(
        Manifest(citation_key=h.CITATION_KEY, files=files).model_dump_json(indent=2)
    )
    download = tmp_path / "download"
    download.mkdir()
    (download / h.TALL_ZIP_NAME).write_bytes(TALL)
    (download / "idea_main.js").write_bytes(SITE_JS)
    (download / h.GEO_GPL16244_NAME).write_bytes(GEO_IDEA)
    (download / h.GEO_GPL20785_NAME).write_bytes(GEO_VALIDATION)
    h.deposit_raw_mirror(
        download_dir=download, retrieved_at="2026-10-11", data_root=str(root)
    )
    return root


@pytest.fixture
def dataset(data_root: Path) -> h.InductionTimecourseHackett2020Dataset:
    return h.InductionTimecourseHackett2020Dataset(root=str(data_root / "store"))


def _records(dataset: h.InductionTimecourseHackett2020Dataset) -> list[dict[str, Any]]:
    return [dataset[i] for i in range(len(dataset))]


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def test_raw_mirror_records_every_retrieval(data_root: Path) -> None:
    manifest = h.load_manifest(str(data_root))
    rows = [(f.path, f.role, f.retrieval.method.value) for f in manifest.files]  # type: ignore[union-attr]
    assert rows == [
        (h.TALL_REL, "raw_data", "direct_url"),
        (h.SITE_JS_REL, "raw_data", "direct_url"),
        (h.GEO_GPL16244_REL, "raw_data", "direct_url"),
        (h.GEO_GPL20785_REL, "raw_data", "direct_url"),
        (h.STRAINS_REL, "si_data", "pmc_cloud"),
        (h.VALIDATION_REL, "si_data", "pmc_cloud"),
    ]
    tall = manifest.files[0]
    assert tall.sha256 == _sha(TALL)
    assert tall.retrieval is not None
    assert tall.retrieval.params == {
        "url": f"{h.IDEA_BUCKET}/{h.TALL_ZIP_NAME}",
        "gcs_md5_base64": h.TALL_ZIP_GCS_MD5_BASE64,
        "gcs_generation": h.TALL_ZIP_GCS_GENERATION,
    }
    assert h.manifest_sha256(manifest, h.STRAINS_REL) == _sha(STRAINS)
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        h.manifest_sha256(manifest, "data/absent.txt")


def test_deposit_refuses_a_wrong_bucket_md5_and_a_changed_file(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = tmp_path / "download"
    monkeypatch.setattr(h, "TALL_ZIP_GCS_MD5_BASE64", "AAAA")
    with pytest.raises(ValueError, match="bucket's published md5"):
        h.deposit_raw_mirror(
            download_dir=download, retrieved_at="x", data_root=str(data_root)
        )
    monkeypatch.setattr(
        h,
        "TALL_ZIP_GCS_MD5_BASE64",
        base64.b64encode(hashlib.md5(TALL, usedforsecurity=False).digest()).decode(),
    )
    (download / "idea_main.js").write_bytes(SITE_JS + b"changed")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        h.deposit_raw_mirror(
            download_dir=download, retrieved_at="x", data_root=str(data_root)
        )


# --------------------------------------------------------------------------- #
# Strain table
# --------------------------------------------------------------------------- #
def _row(strain: str, genotype: str, kind: str = "TF") -> h.StrainRow:
    return h.StrainRow(
        strain=strain, genotype=genotype, strain_type=kind, source="s", sheet_row=3
    )


@pytest.mark.parametrize(
    ("genotype", "background", "insertion"),
    [
        (
            "Same as DBY12021 with KanMX-Pgal1-ARG80",
            "DBY12021",
            ("ARG80", "Pgal1", "KanMX", "KanMX-Pgal1-ARG80", "GEV"),
        ),
        (
            f"{GEV_MET6} KanMX-Pgal1MET32",
            "SMY69",
            ("MET32", "Pgal1", "KanMX", "KanMX-Pgal1MET32", "GEV"),
        ),
        (
            "KanMX-Z3EVpr-SIR2, leu2d0::ACT1pr-ZEV-NatMX, "
            "ybr032wd::LEU2pr(-314 --> -1)-LEU2",
            "SMY254",
            ("SIR2", "Z3EVpr", "KanMX", "KanMX-Z3EVpr-SIR2", "ZEV"),
        ),
        (
            STRAIN_ROWS[3][1],
            "SMY257_4ZEV",
            ("GCN4", "4ZEVpr", "KanMX", "KanMX-4ZEVpr-GCN4", "4ZEV"),
        ),
        (
            STRAIN_ROWS[6][1],
            "Z3EV_HAP1",
            ("HMX1", "Z3EVpr", "URA3", "URA3::Z3EVpr-HMX1", "Z3EV"),
        ),
    ],
)
def test_parse_strain_reads_each_template(
    genotype: str, background: str, insertion: tuple[str, ...]
) -> None:
    spec = h.parse_strain(_row("SMYX", genotype))
    assert spec.background == background
    assert spec.insertion is not None
    assert (
        spec.insertion.gene_token,
        spec.insertion.promoter_name,
        spec.insertion.marker,
        spec.insertion.cassette,
        spec.insertion.responsive_to,
    ) == insertion


def test_parse_strain_controls_and_refusals() -> None:
    spec = h.parse_strain(_row("SMY254", ZEV_TAIL, "Control"))
    assert (spec.background, spec.insertion) == ("SMY254", None)
    with pytest.raises(
        h.StrainTableError, match="control genotype on a non-control row"
    ):
        h.parse_strain(_row("SMY254", ZEV_TAIL, "TF"))
    with pytest.raises(h.StrainTableError, match="matches 0 templates"):
        h.parse_strain(_row("SMYX", "KanMX-tetO7-ACA1"))


@pytest.mark.parametrize(
    ("token", "strain"),
    [
        ("SMY2049", "SMY2049"),
        ("yRSM92", "SMY92"),
        ("YukoSMY2047", "SMY2047"),
        ("SMY108n", "SMY108"),
        ("SMY2266b", "SMY2266"),
        ("SMY254c", "SMY254"),
    ],
)
def test_table_strain_of_maps_release_tokens(token: str, strain: str) -> None:
    assert h.table_strain_of(token) == strain


def test_table_strain_of_refuses_an_unknown_token() -> None:
    with pytest.raises(h.StrainTableError, match="unrecognized release strain token"):
        h.table_strain_of("DBY12021")


def test_read_strain_table_and_refusals(tmp_path: Path) -> None:
    path = tmp_path / "t.xls"
    path.write_bytes(STRAINS)
    rows = h.read_strain_table(path)
    assert list(rows) == [r[0] for r in STRAIN_ROWS]
    assert rows["SMY69"].sheet_row == 4
    assert rows["SMY69"].genotype == GEV_MET6
    dup = tmp_path / "dup.xls"
    cells = [
        ["Table EV1: Strains", None, None, None],
        ["Strain", "Genotype", "Type", "Experiment/Microarray Source"],
        STRAIN_ROWS[0],
        STRAIN_ROWS[0],
    ]
    dup.write_bytes(_xlsx({h.STRAINS_SHEET: pd.DataFrame(cells)}, header=False))
    with pytest.raises(h.StrainTableError, match="appears twice"):
        h.read_strain_table(dup)


def test_backgrounds_carry_their_alleles_and_activator() -> None:
    rows = {r[0]: _row(r[0], r[1], r[2]) for r in STRAIN_ROWS}
    smy69 = h.build_background("SMY69", rows["SMY69"])
    assert [(a.allele_name, a.systematic_gene_name, a.edit) for a in smy69.alleles] == [
        ("(Pgal10-1+Gal1)::loxP", "YBR020W", AlleleEdit.cassette_replacement),
        ("gal4d::LEU2", "YPL248C", AlleleEdit.cassette_replacement),
        ("leu2Δ0", "YCL018W", AlleleEdit.full_deletion),
        ("met6d::HphMX4", "YER091C", AlleleEdit.cassette_replacement),
    ]
    assert [(c.name, c.elements) for c in smy69.integrations] == [
        ("leu2d0::pACT1-GEV-NatMX", ["pACT1", "GEV", "NatMX"])
    ]
    assert (smy69.mating_type, smy69.ploidy) == (None, "haploid")
    assert [g.field for g in smy69.provenance_gaps] == ["mating_type"]
    hap1 = h.build_background("Z3EV_HAP1", rows["SMY2047"])
    assert [a.allele_name for a in hap1.alleles] == [
        "ura3Δ0",
        "his3Δ1",
        "can1Δ::STE2pr-Sp_his5",
        "lyp1Δ0",
    ]
    assert [c.locus_systematic_gene_name for c in hap1.integrations] == ["YLR256W"]
    assert hap1.name == "HAP1+ Z3EV background (SMY2047, SMY2605-SMY2614)"
    assert h.build_background("SMY257_4ZEV", rows["SMY257"]).integrations[
        0
    ].elements == ["ACT1pr", "4ZEV", "NatMX"]


def test_check_tf_matches_refuses_a_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(h, "RELEASE_TF_TOKENS_WITHOUT_AN_ORF", SYSTEMATIC)
    zev = h.parse_strain(_row("SMY2049", STRAIN_ROWS[0][1]))
    h._check_tf_matches(zev, "ACA1", "ZEV")
    with pytest.raises(h.StrainTableError, match="release system GEV"):
        h._check_tf_matches(zev, "ACA1", "GEV")
    with pytest.raises(h.StrainTableError, match="release induces HMX1"):
        h._check_tf_matches(zev, "HMX1", "ZEV")
    control = h.parse_strain(_row("SMY254", ZEV_TAIL, "Control"))
    with pytest.raises(h.StrainTableError, match="control row induced as ACA1"):
        h._check_tf_matches(control, "ACA1", "ZEV")


# --------------------------------------------------------------------------- #
# Releases and keys
# --------------------------------------------------------------------------- #
def test_read_validation_restores_oct1_and_refuses_other_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(h, "N_VALIDATION_ARRAYS", 2)
    monkeypatch.setattr(h, "N_GENES", 4)
    path = tmp_path / "ev2.xlsx"
    path.write_bytes(VALIDATION)
    frame = h.read_validation(path, GENES)
    assert sorted(set(frame["GeneName"])) == GENES
    with pytest.raises(h.ReleaseShapeError, match="OCT1 dated"):
        h.read_validation(path, ["AAC1", "ACA1", "HMX1", "PHO5"])


def test_induced_orf_is_the_pinned_table_and_refuses_an_unpinned_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The induced gene's ORF comes from the pinned token table, never a genome; the
    release's own names are the expression keys (no table is consulted for them).
    """
    monkeypatch.setattr(h, "RELEASE_TF_TOKENS_WITHOUT_AN_ORF", SYSTEMATIC)
    assert h.induced_orf("ACA1") == "YER045C"
    with pytest.raises(h.StrainTableError, match="pinned table does not carry"):
        h.induced_orf("aca1")
    assert h.RELEASE_TF_TOKENS_WITHOUT_AN_ORF is SYSTEMATIC


def test_pinned_tf_table_is_release_tokens_to_orfs() -> None:
    """Every pinned value is an ORF and the two Table EV1 spellings are present."""
    table = h.RELEASE_TF_TOKENS_WITHOUT_AN_ORF
    assert len(table) == 213
    assert (table["CUP2"], table["ACE1"]) == ("YGL166W", "YGL166W")
    assert table["YGR066c"] == "YGR066C"
    assert all(re.match(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$", v) for v in table.values())


def test_environment_has_the_inducer_only_after_t0() -> None:
    t0, t5 = h.environment("P", 0.0), h.environment("M", 5.0)
    assert (t0.perturbations, t0.duration_hours) == ([], 0.0)
    assert t5.duration_hours == 5.0 / 60.0
    (inducer,) = t5.perturbations
    assert isinstance(inducer, SmallMoleculePerturbation)
    assert (inducer.compound.name, inducer.concentration.value) == ("estradiol", 1.0)
    assert t5.media is CHEMOSTAT_MET_LIMITED_HACKETT2020
    assert t0.media is CHEMOSTAT_P_LIMITED_HACKETT2020
    assert {g.field for g in t0.provenance_gaps} == {
        "dilution_rate_per_hour",
        "auxotroph_supplements",
    }
    assert t0.culture_format is not None
    assert (
        t0.culture_format.vessel,
        t0.culture_format.working_volume_ul,
        t0.culture_format.shaking_rpm,
    ) == (h.VESSEL, 300_000.0, 400.0)


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
def test_build_serves_every_listed_array(
    dataset: h.InductionTimecourseHackett2020Dataset,
) -> None:
    records = _records(dataset)
    keys = [
        (
            r["experiment"]["phenotype"]["timecourse_id"],
            round(r["experiment"]["environment"]["duration_hours"] * 60, 9),
        )
        for r in records
    ]
    assert keys == [
        ("ACA1_SMY2049_02/29/2016_P_ZEV", 0.0),
        ("ACA1_SMY2049_02/29/2016_P_ZEV", 5.0),
        ("GEV_SMY69_01/01/2015_M_GEV", 0.0),
        ("GEV_SMY69_01/01/2015_M_GEV", 2.5),
        ("HMX1_P_ZEV", 0.0),
        ("HMX1_P_ZEV", 5.0),
    ]
    aca1 = records[1]["experiment"]
    (pert,) = aca1["genotype"]["perturbations"]
    assert InduciblePromoterInsertionPerturbation.model_validate(pert).model_dump() == {
        "systematic_gene_name": "YER045C",
        "perturbed_gene_name": "ACA1",
        "gene_namespace": "scerevisiae_s288c_sgd_systematic",
        "identifier_mapping": None,
        "provenance": "engineered",
        "state": "present",
        "mechanism_so_id": "SO:0000667",
        "mechanism_so_name": "insertion",
        "expression_direction": "increased",
        "crispr": None,
        "description": "Inducible synthetic promoter inserted upstream of a present "
        "gene's start codon",
        "perturbation_type": "inducible_promoter_insertion",
        "promoter_name": "Z3EVpr",
        "cassette": "KanMX-Z3EVpr-ACA1",
        "marker": "KanMX",
        "responsive_to": "ZEV",
        "native_promoter_retained": True,
        "strain_id": "SMY2049",
    }
    assert records[2]["experiment"]["genotype"]["perturbations"] == []
    assert records[1]["reference"]["genome_reference"]["strain"] == "SMY254"
    assert records[2]["reference"]["genome_reference"]["strain"] == "SMY69"
    assert records[4]["reference"]["genome_reference"]["strain"] == (
        "HAP1+ Z3EV background (SMY2047, SMY2605-SMY2614)"
    )


def test_build_stores_every_layer_exactly(
    dataset: h.InductionTimecourseHackett2020Dataset,
) -> None:
    records = _records(dataset)
    phen = InductionTimecourseExpressionPhenotype.model_validate(
        records[1]["experiment"]["phenotype"]
    )
    by_key = {g: i for i, g in enumerate(GENES)}  # the release's own names
    assert dict(phen.log2_cleaned) == {k: _cleaned(i, 5.0) for k, i in by_key.items()}
    assert dict(phen.sample_channel_intensity) == {
        k: 205.0 + i for k, i in by_key.items()
    }
    assert dict(phen.reference_channel_intensity) == {
        k: 100.0 + i for k, i in by_key.items()
    }
    assert phen.log2_noise_sd is not None
    assert dict(phen.log2_noise_sd) == {k: 0.1 + i / 100 for k, i in by_key.items()}
    assert phen.log2_shrunken is not None
    assert dict(phen.log2_shrunken) == {
        k: (round(_cleaned(i, 5.0) * 0.5, 6) if i in (1, 2) else 0.0)
        for k, i in by_key.items()
    }
    assert (phen.timecourse_id, phen.strain_label, phen.experiment_date) == (
        "ACA1_SMY2049_02/29/2016_P_ZEV",
        "SMY2049",
        "02/29/2016",
    )
    assert phen.provenance_gaps == []
    ref = records[1]["reference"]["phenotype_reference"]
    assert set(ref["log2_cleaned"].values()) == {0.0}
    assert ref["timecourse_id"] == phen.timecourse_id


def test_validation_arrays_gap_what_ev2_does_not_release(
    dataset: h.InductionTimecourseHackett2020Dataset,
) -> None:
    phen = _records(dataset)[5]["experiment"]["phenotype"]
    assert (phen["log2_noise_sd"], phen["log2_shrunken"], phen["experiment_date"]) == (
        None,
        None,
        None,
    )
    assert sorted(g["field"] for g in phen["provenance_gaps"]) == [
        "experiment_date",
        "log2_noise_sd",
        "log2_selected",
        "log2_shrunken",
        "log2_thresholded",
    ]
    assert phen["log2_cleaned"]["OCT1"] == _cleaned(3, 5.0)  # the restored OCT1
    assert (phen["strain_label"], phen["release_table"]) == (
        "SMY2605",
        h.VALIDATION_NAME,
    )


def test_ledger_accounts_for_the_refused_course(
    dataset: h.InductionTimecourseHackett2020Dataset,
) -> None:
    ledger = json.loads(
        (Path(dataset.preprocess_dir) / "release_ledger.json").read_text()
    )
    assert ledger["refused"] == [
        {
            "timecourse_id": "CBF1_SMY22_01/01/2015_P_GEV",
            "strain_label": "SMY22",
            "n_arrays": 3,
            "reason": "strain_not_in_table_ev1",
        }
    ]
    assert (ledger["n_records"], ledger["n_refused_arrays"]) == (6, 3)
    assert ledger["strains_per_background"] == {"SMY254": 1, "SMY69": 1, "Z3EV_HAP1": 1}
    assert (ledger["n_gene_keys_as_released"], ledger["n_tf_tokens_pinned"]) == (
        4,
        len(SYSTEMATIC),
    )
    assert sorted(dataset.gene_set) == ["YER045C", "YLR205C"]


def test_ledger_refuses_an_unaccounted_array() -> None:
    with pytest.raises(ValueError, match="served \\+ refused arrays"):
        h.ReleaseLedger(
            tall_sha256="a",
            validation_sha256="b",
            strains_sha256="c",
            n_genes=4,
            n_tall_timecourses=3,
            n_tall_arrays=7,
            n_validation_timecourses=1,
            n_validation_arrays=2,
            n_records=5,
            n_served_timecourses=3,
            refused=[],
            n_refused_arrays=0,
            excel_date_gene_restored="OCT1",
            n_gene_keys_as_released=4,
            n_tf_tokens_pinned=0,
            arrays_per_timecourse={},
            strains_per_background={},
            yrsm_identified={},
        )


# --------------------------------------------------------------------------- #
# The verifier on the synthetic store
# --------------------------------------------------------------------------- #
def test_verifier_passes_every_rule_but_the_unmirrored_quotes(
    dataset: h.InductionTimecourseHackett2020Dataset,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v, "PAPER_TIMECOURSES", 3)
    monkeypatch.setattr(v, "PAPER_ARRAYS", 7)
    monkeypatch.setattr(v, "PAPER_GENES", 4)
    monkeypatch.setattr(v, "PAPER_THRESHOLDED_SERIES", 6)
    monkeypatch.setattr(v, "PAPER_SELECTED_SERIES", 6)
    root = Path(dataset.root)
    dataset.close_lmdb()
    report = v.verify_induction_timecourse(
        lambda: stream_records(str(root)),
        dataset_name="synthetic",
        provenance=Provenance(source_uri="synthetic"),
        expected_count=6,
        raw_dir=root / "raw",
        preprocess_dir=root / "preprocess",
        library_root=data_root / "torchcell-library",
        data_root=data_root,
        sgd_genes={"YER045C", "YLR205C"},
    )
    failed = {r.name for r in report.results if not r.passed}
    assert failed == {"sourced_values"}
    by_name = {r.name: r for r in report.results}
    assert by_name["release_fidelity"].details["n_values"] == 6 * 4 * 8 - 2 * 4 * 4
    assert by_name["induction_upregulates"].details["up_fraction"] == 1.0
    assert set(by_name["sourced_values"].details["failed_sources"]) == {
        "paper.md",
        "si/si1.md",
        f"{h.RAW_DIR_REL}/{h.GEO_GPL20785_REL}",
        f"{h.RAW_DIR_REL}/{h.SITE_JS_REL}",
    }
