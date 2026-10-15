# tests/torchcell/datasets/ecoli/test_balakrishnan2022.py
# [[tests.torchcell.datasets.ecoli.test_balakrishnan2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_balakrishnan2022.py
"""Balakrishnan 2022 mRNA number-fraction loader (``torchcell.datasets.ecoli.balakrishnan2022``).

The synthetic tests write the three-file layout the loader consumes: a Table S3
workbook whose description sheet is the 29 declared samples and whose fractions sheet
carries the 29 released headers (``a4_1`` twice) over six rows (a current gene, a
pseudogene locus, a split-fragment synonym, a retired b-number and a locus-``0`` row,
every one stored under its released identifier); the paper text carrying the paper
quotes; and a minimal Mori 2021 Appendix ``.docx`` carrying the strain quotes. The
loader reads no genome; the verifier's L4 containment reads an in-test one, and the
assembly pin is an in-test object, so nothing reads ``$DATA_ROOT``.

The ``@pytest.mark.data`` tests read the real raw mirror and the dev-tree LMDB and pin
the numbers the dendron note states: 28 records, 4,342 keys, 18 of them the locus-0
rows keyed by their released gene cell, two references.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openpyxl
import pandas as pd
import pytest

import torchcell.datasets.ecoli.balakrishnan2022 as bk
import torchcell.datasets.ecoli.mori2021 as mori2021
from torchcell.data import file_sha256
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    DeletionStrainPerturbation,
    EnvironmentPhysicalPerturbation,
    PromoterReplacementPerturbation,
    SmallMoleculePerturbation,
)
from torchcell.literature.manifest import RetrievalMethod
from torchcell.literature.manual_deposit import extend_manifest, verify_deposit
from torchcell.sequence.genome.base import GeneNameStatus

#: ``(gene, locus)`` per synthetic row; every row is a stored key.
ROWS: tuple[tuple[str, Any], ...] = (
    ("keepA", "b0001"),
    ("keepB", "b0002"),
    ("pseudo", "b0003"),
    ("frag_1", "b0004"),
    ("gone", "b3036"),
    ("insJK", 0),
)
#: The keys the six rows are stored under: the b-number, or the gene cell for locus 0.
KEYS = ("b0001", "b0002", "b0003", "b0004", "b3036", "insJK")
GENE_SET = {"b0001", "b0002", "b1101", "b3212"}
LOCI = {*GENE_SET, "b0003"}
RETIRED = {"b3036"}
UNIQUE = list(dict.fromkeys(bk.RELEASED_HEADERS))


class FakeGenome:
    """The verifier's genome: two of the six keys are its genes, one a pseudogene."""

    ASSEMBLY_SET = bk.ASSEMBLY_SET

    def __init__(self) -> None:
        """Two stored keys as genes, one pseudogene locus, one retired name."""
        self.gene_set = GENE_SET
        self.genbank = SimpleNamespace(loci=LOCI)

    def resolve_gene_name(self, name: str) -> Any:
        """Any name as itself, retired or current."""
        status = GeneNameStatus.RETIRED if name in RETIRED else GeneNameStatus.CURRENT
        return SimpleNamespace(systematic_name=name, status=status)


def _value(locus: Any, sample: str) -> float:
    """Each column sums to 1: two kept rows trade 0.001 * column index."""
    d = 0.001 * UNIQUE.index(sample)
    return {"b0001": 0.4 + d, "b0002": 0.3 - d}.get(str(locus), 0.075)


def _description_row(spec: bk.SampleSpec, first: bool) -> list[Any]:
    nitrogen = bk.NITROGEN_FILL_SERIES.get(
        spec.sample, bk.M9_NITROGEN if spec.medium == "M9" else bk.MOPS_NITROGEN
    )
    text = {"NQ1243": bk._Q_PU_PTSG, "NQ1390": bk._Q_PU_PTSG, "NQ393": bk._Q_PLAC_GOGAT}
    return [
        spec.sample,
        spec.series if first else None,
        spec.growth_rate_per_h,
        spec.strain,
        spec.medium,
        bk.CARBON_SOURCE,
        nitrogen,
        spec.supplement,
        text.get(spec.strain, "wild type"),
        f"{spec.sample}.fastq",
    ]


def write_table(path: Path, *, mutate: Any = None) -> None:
    book = openpyxl.Workbook()
    desc = book.active
    assert desc is not None
    desc.title = bk.SHEET_DESCRIPTION
    desc.append(list(bk.DESCRIPTION_HEADER))
    seen: set[str] = set()
    for spec in bk.SAMPLES:
        desc.append(_description_row(spec, spec.series not in seen))
        seen.add(spec.series)
    frac = book.create_sheet(bk.SHEET_FRACTIONS)
    frac.append([*bk.IDENTITY_COLUMNS, *bk.RELEASED_HEADERS])
    for gene, locus in ROWS:
        frac.append(
            [gene, locus, 900, *(_value(locus, s) for s in bk.RELEASED_HEADERS)]
        )
    if mutate is not None:
        mutate(book)
    book.save(path)


def write_docx(path: Path, paragraphs: list[str]) -> None:
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "word/document.xml", f"<w:document><w:body>{body}</w:body></w:document>"
        )


def write_release(raw: Path) -> dict[str, str]:
    raw.mkdir(parents=True, exist_ok=True)
    write_table(raw / bk.TABLE_S3)
    (raw / bk.PAPER_TXT).write_text(
        " ".join(bk.PAPER_QUOTES.values()), encoding="utf-8"
    )
    write_docx(raw / bk.MORI_APPENDIX, list(bk.MORI_QUOTES.values()))
    (raw / bk.SI_TEXT).write_text("\n".join(bk.SI_QUOTES.values()), encoding="utf-8")
    return {name: str(raw / name) for name in bk.RAW_SHA256}


@pytest.fixture
def release(tmp_path: Path) -> dict[str, str]:
    return write_release(tmp_path / "raw")


# --------------------------------------------------------------------------- #
# Declarations
# --------------------------------------------------------------------------- #
def test_twenty_nine_described_samples_become_twenty_eight_records() -> None:
    assert len(bk.SAMPLES) == 29
    assert [s.sample for s in bk.SAMPLES if s.drop is not None] == ["a3_1"]
    assert len(bk.RELEASED_HEADERS) == 29
    assert len(set(bk.RELEASED_HEADERS)) == len(bk.LOADED) == bk.EXPECTED_RECORDS == 28
    assert {s.sample for s in bk.LOADED} == set(bk.RELEASED_HEADERS)


def test_strain_genotypes_type_the_table_s1_edits() -> None:
    assert bk.strain_genotype("NCM3722").perturbations == []
    for strain in ("NQ1243", "NQ1390"):
        [edit] = bk.strain_genotype(strain).perturbations
        assert isinstance(edit, PromoterReplacementPerturbation)
        assert (edit.systematic_gene_name, edit.promoter_name, edit.is_inducible) == (
            "b1101",
            "Pu",
            True,
        )
    gdha, gogat = bk.strain_genotype("NQ393").perturbations
    assert isinstance(gdha, DeletionStrainPerturbation)
    assert (gdha.systematic_gene_name, gdha.perturbed_gene_name) == ("b1761", "gdhA")
    assert gdha.identifier_mapping is not None
    assert gdha.identifier_mapping.route == "gene_symbol"
    assert isinstance(gogat, PromoterReplacementPerturbation)
    assert (gogat.systematic_gene_name, gogat.promoter_name) == ("b3212", "PLlac-01")
    assert gogat.expression_direction == "decreased"
    assert set(bk.UNTYPED_EDITS) == {
        "NQ1243 xylR (Ptet)",
        "NQ1390 xylR (lacIq promoter)",
    }
    with pytest.raises(ValueError, match="not a strain"):
        bk.strain_genotype("MG1655")


def test_the_loader_reads_no_genome() -> None:
    import inspect

    assert (
        "ecoli_genome"
        not in inspect.signature(
            bk.MrnaFractionBalakrishnan2022Dataset.__init__
        ).parameters
    )
    assert not hasattr(bk, "select_genes")
    assert not hasattr(bk, "check_gene_symbols")


def test_environment_carries_glucose_the_supplement_and_37_c() -> None:
    env = bk.build_environment("M9", "300 µM 3MBA")
    assert env.media is bk.M9_BALAKRISHNAN2022
    assert env.temperature is not None and env.temperature.value == 37.0
    assert env.provenance_gaps == []
    assert (
        bk.SOURCED_VALUES["temperature_c"].provenance.source_uri == bk.SI_TEXT_RELPATH
    )
    carbon, dose = env.perturbations
    assert isinstance(carbon, EnvironmentPhysicalPerturbation)
    assert carbon.magnitude is not None and carbon.magnitude.value == 0.2
    assert isinstance(dose, SmallMoleculePerturbation)
    assert (dose.compound.name, dose.concentration.value) == ("3MBA", 300.0)
    assert len(bk.build_environment("MOPS", None).perturbations) == 1
    with pytest.raises(ValueError, match="not '<n> µM"):
        bk.build_environment("M9", "1 mM IPTG")


def test_media_defer_the_base_and_state_the_nitrogen_salt() -> None:
    deferred, nitrogen = bk.M9_BALAKRISHNAN2022.components
    assert nitrogen.concentration is not None
    assert deferred.definition.value == "composition_deferred"
    assert deferred.defers_to == [bk._DEFERS_TO_SI]
    assert "references 7, 10 and 57" in bk._DEFERS_TO_SI
    assert bk.SOURCED_VALUES["media_recipes_deferred"] in deferred.provenance
    assert (nitrogen.compound.name, nitrogen.concentration.value) == (
        "ammonium sulfate",
        11.34,
    )
    assert bk.MOPS_BALAKRISHNAN2022.base_medium == "MOPS_MINIMAL"
    assert bk.MOPS_BALAKRISHNAN2022.components[1].compound.name == "ammonium chloride"


# --------------------------------------------------------------------------- #
# Released-bytes checks
# --------------------------------------------------------------------------- #
def test_check_quotes_reads_all_four_files(release: dict[str, str]) -> None:
    assert bk.check_quotes(release) == {"n_quotes_checked": 21}


@pytest.mark.parametrize(
    "which", [bk.PAPER_TXT, bk.MORI_APPENDIX, bk.TABLE_S3, bk.SI_TEXT]
)
def test_check_quotes_refuses_a_file_missing_a_quote(
    release: dict[str, str], which: str
) -> None:
    raw = Path(release[which])
    if which in (bk.PAPER_TXT, bk.SI_TEXT):
        raw.write_text("nothing", encoding="utf-8")
    elif which == bk.MORI_APPENDIX:
        write_docx(raw, ["nothing"])
    else:
        write_table(raw, mutate=lambda b: b[bk.SHEET_DESCRIPTION].delete_rows(2, 30))
    with pytest.raises(RuntimeError, match="quote"):
        bk.check_quotes(release)


def test_sample_metadata_matches_and_pins_the_fill_series(
    release: dict[str, str],
) -> None:
    out = bk.check_sample_metadata(release[bk.TABLE_S3])
    assert out == {"n_described": 29, "nitrogen_fill_series": bk.NITROGEN_FILL_SERIES}


@pytest.mark.parametrize(
    "cell,value,match",
    [
        ("D2", "MG1655", "release says"),
        ("F2", "0.4% glucose", "carbon source"),
        ("G2", "11.34 mM (NH4)2SO8", "off-recipe nitrogen"),
        ("A1", "Sample", "header"),
    ],
)
def test_sample_metadata_refuses_a_changed_cell(
    tmp_path: Path, cell: str, value: str, match: str
) -> None:
    path = tmp_path / "t.xlsx"

    def change(book: Any) -> None:
        book[bk.SHEET_DESCRIPTION][cell] = value

    write_table(path, mutate=change)
    with pytest.raises(RuntimeError, match=match):
        bk.check_sample_metadata(str(path))


def test_read_fractions_keeps_cells_verbatim_and_checks_the_duplicate(
    release: dict[str, str],
) -> None:
    rows, check = bk.read_fractions(release[bk.TABLE_S3])
    assert [r.locus for r in rows] == ["b0001", "b0002", "b0003", "b0004", "b3036", "0"]
    assert rows[0].values["c5"] == pytest.approx(0.4 + 0.001 * UNIQUE.index("c5"))
    assert set(rows[0].values) == set(UNIQUE)
    assert check["duplicate_columns_1_based"] == [16, 17]
    assert check["duplicate_identical_rows"] == len(ROWS)


@pytest.mark.parametrize(
    "cell,value,match",
    [
        ("A1", "name", "identity block"),
        ("D1", "c9", "sample headers"),
        ("P2", 0.5, "differ in 1 rows"),
        ("D2", 0.9, "sums"),
    ],
)
def test_read_fractions_refuses_a_changed_sheet(
    tmp_path: Path, cell: str, value: Any, match: str
) -> None:
    path = tmp_path / "t.xlsx"

    def change(book: Any) -> None:
        book[bk.SHEET_FRACTIONS][cell] = value

    write_table(path, mutate=change)
    with pytest.raises(RuntimeError, match=match):
        bk.read_fractions(str(path))


def test_gene_keys_store_every_row_under_its_released_identifier(
    release: dict[str, str],
) -> None:
    rows, _ = bk.read_fractions(release[bk.TABLE_S3])
    keyed = bk.gene_keys(rows)
    assert [item.key for item in keyed] == list(KEYS)
    assert [item.keyed_by_gene for item in keyed] == [False] * 5 + [True]
    assert keyed[-1].row.locus == bk.ZERO_LOCUS == "0"
    with pytest.raises(RuntimeError, match="share the key 'b0001'"):
        bk.gene_keys([rows[0], rows[0]])


def test_phenotypes_store_the_column_and_the_reference_mean(
    release: dict[str, str],
) -> None:
    rows, _ = bk.read_fractions(release[bk.TABLE_S3])
    kept = bk.gene_keys(rows[:2])
    phenotype = bk.build_phenotype(kept, "c5")
    assert phenotype.n_libraries == 1
    assert phenotype.mrna_number_fraction["b0001"] == rows[0].values["c5"]
    reference = bk.build_reference_phenotype(kept, ("c5", "c0_1"))
    assert reference.n_libraries == 2
    assert reference.mrna_number_fraction["b0001"] == pytest.approx(
        (rows[0].values["c5"] + rows[0].values["c0_1"]) / 2
    )


def test_publication_and_mirror_paths_follow_the_citation_key(tmp_path: Path) -> None:
    pub = bk.publication()
    assert (pub.pubmed_id, pub.doi) == ("36480614", "10.1126/science.abk2066")
    assert bk.raw_mirror_dir(str(tmp_path)) == tmp_path / bk.RAW_DIR_REL
    assert {r.method.value for r in bk.RETRIEVALS.values()} == {
        "direct_url",
        "pmc_cloud",
    }


def test_the_si_deposit_records_eight_files_five_of_them_zip_members() -> None:
    deposit = bk.SI_DEPOSIT
    assert deposit.issues == (853, 906)
    assert deposit.directories() == ("data", "si")
    members = [f for f in deposit.files if f.container is not None]
    assert [f.name for f in members] == [
        f"science.abk2066_table_s{n}.xlsx" for n in range(3, 8)
    ]
    assert {f.container for f in members} == {f"data/{bk.TABLES_ZIP}"}
    assert len(deposit.files) == 8
    record = bk.si_text_record(165750)
    assert record.processing is not None
    assert record.processing.input_sha256 == [bk.SI_PDF_SHA256]
    assert record.processing.params == {"args": ["-layout", "-enc", "UTF-8"]}


# --------------------------------------------------------------------------- #
# Mirror, download and an end-to-end build
# --------------------------------------------------------------------------- #
def _patch_pins(monkeypatch: pytest.MonkeyPatch, release: dict[str, str]) -> None:
    shas = {name: file_sha256(path) for name, path in release.items()}
    monkeypatch.setattr(
        bk, "DATA_SHA256", {k: shas[k] for k in (bk.TABLE_S3, bk.PAPER_TXT)}
    )
    monkeypatch.setattr(bk, "RAW_SHA256", shas)
    monkeypatch.setattr(bk, "MORI_APPENDIX_SHA256", shas[bk.MORI_APPENDIX])
    monkeypatch.setattr(bk, "SI_TEXT_SHA256", shas[bk.SI_TEXT])


def test_deposit_writes_the_manifest_is_idempotent_and_refuses_bad_bytes(
    release: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_pins(monkeypatch, release)
    source = Path(release[bk.TABLE_S3]).parent
    data_root = tmp_path / "root"
    root = bk.deposit_raw_mirror(source_dir=source, data_root=str(data_root))
    root = bk.deposit_raw_mirror(source_dir=source, data_root=str(data_root))
    manifest = bk.load_manifest(str(data_root))
    assert {f.path for f in manifest.files} == set(bk.MIRROR_RELPATH.values())
    assert manifest.provenance_complete is False
    (root / bk.MIRROR_RELPATH[bk.PAPER_TXT]).write_text("tampered")
    with pytest.raises(RuntimeError, match="different sha256"):
        bk.deposit_raw_mirror(source_dir=source, data_root=str(data_root))
    Path(release[bk.PAPER_TXT]).write_text("tampered")
    with pytest.raises(RuntimeError, match="does not hash"):
        bk.deposit_raw_mirror(source_dir=source, data_root=str(data_root))


def test_download_links_this_mirror_and_the_mori_appendix(
    release: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_pins(monkeypatch, release)
    data_root = tmp_path / "root"
    bk.deposit_raw_mirror(
        source_dir=Path(release[bk.TABLE_S3]).parent, data_root=str(data_root)
    )
    si_text = Path(release[bk.SI_TEXT]).read_bytes()
    mirror = bk.raw_mirror_dir(str(data_root))
    (mirror / "si").mkdir()
    (mirror / bk.SI_TEXT_RELPATH).write_bytes(si_text)
    extend_manifest(
        mirror,
        [bk.si_text_record(len(si_text))],
        citation_key=bk.CITATION_KEY,
        doi=bk.PAPER_DOI,
        title=bk.PAPER_TITLE,
    )
    mori = SimpleNamespace(
        load_manifest=lambda root: "manifest",
        manifest_sha256=lambda manifest, relpath: file_sha256(
            release[bk.MORI_APPENDIX]
        ),
        raw_mirror_dir=lambda root: Path(release[bk.MORI_APPENDIX]).parent,
        appendix_text=mori2021.appendix_text,
    )
    monkeypatch.setattr(bk, "mori2021", mori)
    monkeypatch.setattr(bk, "MORI_APPENDIX_RELPATH", bk.MORI_APPENDIX)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    raw = tmp_path / "build" / "raw"
    dataset = bk.MrnaFractionBalakrishnan2022Dataset.__new__(
        bk.MrnaFractionBalakrishnan2022Dataset
    )
    monkeypatch.setattr(type(dataset), "raw_dir", property(lambda self: str(raw)))
    dataset.download()
    assert sorted(os.listdir(raw)) == sorted(bk.RAW_SHA256)


def _reference_genome(*_: Any, background: Any = None) -> AssemblyReferenceGenome:
    return AssemblyReferenceGenome(
        species="Escherichia coli",
        strain=background.name,
        ploidy="haploid",
        assembly_set=bk.ASSEMBLY_SET,
        assembly_accession="GCA_000005845.2",
        background=background,
    )


def test_process_builds_twenty_eight_records_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mrna_fraction_balakrishnan2022"
    paths = write_release(root / "raw")
    _patch_pins(monkeypatch, paths)
    monkeypatch.setattr(bk, "EXPECTED_SOURCE_ROWS", len(ROWS))
    monkeypatch.setattr(bk, "EXPECTED_GENE_KEYS", len(ROWS))
    monkeypatch.setattr(bk, "EXPECTED_ZERO_LOCUS_ROWS", 1)
    monkeypatch.setattr(bk, "assembly_reference", _reference_genome)

    dataset = bk.MrnaFractionBalakrishnan2022Dataset(root=str(root))
    assert len(dataset) == 28
    assert dataset.gene_set == set(KEYS)
    items = [dataset.transform_item(dataset[i]) for i in range(len(dataset))]
    assert {i["reference"].genome_reference.strain for i in items} == {"NCM3722"}
    assert {i["reference"].phenotype_reference.n_libraries for i in items} == {2}
    # NQ1243 and NQ1390 (8 records) carry one edit each, NQ393 (7 records) two
    assert sum(len(i["experiment"].genotype.perturbations) for i in items) == 22
    assert {i["experiment"].environment.temperature.value for i in items} == {37.0}
    dataset.close_lmdb()

    preprocess = Path(dataset.preprocess_dir)
    ledger = json.loads((preprocess / "dropped_records.json").read_text())
    assert (ledger["records"], ledger["stored_gene_keys"]) == (28, len(ROWS))
    assert ledger["keyed_by_released_gene_cell"]["n"] == 1
    assert ledger["keyed_by_released_gene_cell"]["items"] == ["insJK"]
    assert "refused_rows" not in ledger
    samples = pd.read_csv(preprocess / "samples.csv")
    assert list(samples["sample"]) == [s.sample for s in bk.LOADED]
    assert (
        dict(zip(samples["sample"], samples["nitrogen_cell_verbatim"], strict=True))[
            "a4_1"
        ]
        == "11.34 mM (NH4)2SO7"
    )
    keys = pd.read_csv(preprocess / "gene_keys.csv")
    assert list(keys["key"]) == list(KEYS)
    assert list(keys["keyed_by_gene"]) == [False] * 5 + [True]

    report = bk.verify_build(str(root), genome=FakeGenome())  # type: ignore[arg-type]
    # the L4 containment reads the in-test genome, which holds two of the six keys:
    # a genome-reference finding under the verdict, never a gate
    failed = [r for r in report.results if not r.passed]
    assert [r.name for r in failed if r.finding_class.value != "report"] == []
    assert report.passed


def test_process_refuses_a_count_that_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mrna_fraction_balakrishnan2022"
    paths = write_release(root / "raw")
    _patch_pins(monkeypatch, paths)
    monkeypatch.setattr(bk, "EXPECTED_SOURCE_ROWS", len(ROWS))
    with pytest.raises(RuntimeError, match="keys"):
        bk.MrnaFractionBalakrishnan2022Dataset(root=str(root))


# --------------------------------------------------------------------------- #
# The real mirror and the dev store
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """The real ``DATA_ROOT`` from ``.env``; skip when the mirror is not deposited."""
    from dotenv import dotenv_values, find_dotenv

    root = str(dotenv_values(find_dotenv(usecwd=True))["DATA_ROOT"])
    if not (bk.raw_mirror_dir(root) / "manifest.json").exists():
        pytest.skip(f"raw mirror not deposited under {root}")
    return root


@pytest.mark.data
def test_raw_mirror_matches_every_pin_and_every_quote() -> None:
    data_root = _data_root()
    root = bk.raw_mirror_dir(data_root)
    paths = {name: str(root / rel) for name, rel in bk.MIRROR_RELPATH.items()}
    for name, path in paths.items():
        assert file_sha256(path) == bk.DATA_SHA256[name]
    mori = mori2021.raw_mirror_dir(data_root) / bk.MORI_APPENDIX_RELPATH
    assert file_sha256(str(mori)) == bk.MORI_APPENDIX_SHA256
    si_text = root / bk.SI_TEXT_RELPATH
    assert file_sha256(str(si_text)) == bk.SI_TEXT_SHA256
    assert bk.check_quotes(
        {**paths, bk.MORI_APPENDIX: str(mori), bk.SI_TEXT: str(si_text)}
    ) == {"n_quotes_checked": 21}
    assert bk.si_layout_text(root / bk.SI_PDF_RELPATH) == si_text.read_text()


@pytest.mark.data
def test_the_si_deposit_is_recorded_in_the_raw_mirror() -> None:
    root = bk.raw_mirror_dir(_data_root())
    verify_deposit(bk.SI_DEPOSIT, root)
    held = {r.path: r for r in bk.load_manifest(str(root.parent.parent)).files}
    for f in bk.SI_DEPOSIT.files:
        retrieval = held[f.path].retrieval
        assert retrieval is not None
        assert retrieval.method is RetrievalMethod.manual_browser
        assert held[f.path].sha256 == f.sha256
    assert held[bk.SI_TEXT_RELPATH].sha256 == bk.SI_TEXT_SHA256
    for directory in ("si", "data"):
        assert (root / directory / "DEPOSIT.md").is_file()
        assert (root / directory / "SHA256SUMS.txt").is_file()


@pytest.mark.data
def test_built_lmdb_numbers() -> None:
    root = osp.join(_data_root(), "data", "torchcell", "mrna_fraction_balakrishnan2022")
    ledger = json.loads(Path(root, "preprocess", "dropped_records.json").read_text())
    assert (ledger["records"], ledger["released_rows"], ledger["stored_gene_keys"]) == (
        28,
        4342,
        4342,
    )
    assert ledger["keyed_by_released_gene_cell"]["n"] == 18
    assert ledger["keyed_by_released_gene_cell"]["items"][:2] == ["insAB-1", "insAB-2"]
    samples = pd.read_csv(Path(root, "preprocess", "samples.csv"))
    assert set(samples["n_gene_keys"]) == {4342}
