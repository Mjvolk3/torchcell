# tests/torchcell/datasets/ecoli/test_balakrishnan2022.py
# [[tests.torchcell.datasets.ecoli.test_balakrishnan2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_balakrishnan2022.py
"""Balakrishnan 2022 mRNA number-fraction loader (``torchcell.datasets.ecoli.balakrishnan2022``).

The synthetic tests write the three-file layout the loader consumes: a Table S3
workbook whose description sheet is the 29 declared samples and whose fractions sheet
carries the 29 released headers (``a4_1`` twice) over six rows, one per row rule; the
paper text carrying the paper quotes; and a minimal Mori 2021 Appendix ``.docx``
carrying the strain quotes. The genome and the assembly pin are in-test objects, so
nothing reads ``$DATA_ROOT``.

The ``@pytest.mark.data`` tests read the real raw mirror and the dev-tree LMDB and pin
the numbers the dendron note states: 28 records, 4,176 keys, 148 refused b-numbers
(74 + 69 + 5), 18 locus-0 rows, two references.
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
    EnvironmentPhysicalPerturbation,
    PromoterReplacementPerturbation,
    SmallMoleculePerturbation,
)
from torchcell.sequence.genome.base import GeneNameStatus

#: ``(gene, locus)`` per synthetic row, one per row rule.
ROWS: tuple[tuple[str, Any], ...] = (
    ("keepA", "b0001"),
    ("keepB", "b0002"),
    ("pseudo", "b0003"),
    ("frag_1", "b0004"),
    ("gone", "b3036"),
    ("insJK", 0),
)
GENE_SET = {"b0001", "b0002", "b1101", "b3212"}
LOCI = {*GENE_SET, "b0003"}
RETIRED = {"b3036"}
UNIQUE = list(dict.fromkeys(bk.RELEASED_HEADERS))


class FakeGenome:
    """The kept b-numbers as its genes, one pseudogene locus, one retired name."""

    ASSEMBLY_SET = bk.ASSEMBLY_SET

    def __init__(self, ptsg: str = "b1101") -> None:
        """``ptsg`` is what the ptsG symbol resolves to, to test a moved pin."""
        self.gene_set = GENE_SET
        self.genbank = SimpleNamespace(loci=LOCI)
        self._symbols = {"ptsG": ptsg, "gltB": "b3212"}

    def resolve_gene_name(self, name: str) -> Any:
        """The two promoter targets by symbol; any other name as itself."""
        if name in self._symbols:
            return SimpleNamespace(
                systematic_name=self._symbols[name], status=GeneNameStatus.RENAMED
            )
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


def test_strain_genotypes_type_only_the_sourced_promoter_swaps() -> None:
    assert bk.strain_genotype("NCM3722").perturbations == []
    for strain in ("NQ1243", "NQ1390"):
        [edit] = bk.strain_genotype(strain).perturbations
        assert isinstance(edit, PromoterReplacementPerturbation)
        assert (edit.systematic_gene_name, edit.promoter_name, edit.is_inducible) == (
            "b1101",
            "Pu",
            True,
        )
    [gogat] = bk.strain_genotype("NQ393").perturbations
    assert isinstance(gogat, PromoterReplacementPerturbation)
    assert (gogat.systematic_gene_name, gogat.promoter_name) == ("b3212", "Plac")
    assert gogat.expression_direction == "decreased"
    with pytest.raises(ValueError, match="not a strain"):
        bk.strain_genotype("MG1655")


def test_gene_symbols_must_resolve_to_the_declared_b_numbers() -> None:
    bk.check_gene_symbols(FakeGenome())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="ptsG resolves to b9999"):
        bk.check_gene_symbols(FakeGenome(ptsg="b9999"))  # type: ignore[arg-type]


def test_environment_carries_glucose_the_supplement_and_a_temperature_gap() -> None:
    env = bk.build_environment("M9", "300 µM 3MBA")
    assert env.media is bk.M9_BALAKRISHNAN2022
    assert env.temperature is None
    assert [g.field for g in env.provenance_gaps] == ["temperature"]
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
    assert (nitrogen.compound.name, nitrogen.concentration.value) == (
        "ammonium sulfate",
        11.34,
    )
    assert bk.MOPS_BALAKRISHNAN2022.base_medium == "MOPS_MINIMAL"
    assert bk.MOPS_BALAKRISHNAN2022.components[1].compound.name == "ammonium chloride"


# --------------------------------------------------------------------------- #
# Released-bytes checks
# --------------------------------------------------------------------------- #
def test_check_quotes_reads_all_three_files(release: dict[str, str]) -> None:
    assert bk.check_quotes(release) == {"n_quotes_checked": 13}


@pytest.mark.parametrize("which", [bk.PAPER_TXT, bk.MORI_APPENDIX, bk.TABLE_S3])
def test_check_quotes_refuses_a_file_missing_a_quote(
    release: dict[str, str], which: str
) -> None:
    raw = Path(release[which])
    if which == bk.PAPER_TXT:
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


def test_select_genes_applies_one_rule_per_refused_row(release: dict[str, str]) -> None:
    rows, _ = bk.read_fractions(release[bk.TABLE_S3])
    kept, refused = bk.select_genes(rows, FakeGenome())  # type: ignore[arg-type]
    assert [r.locus for r in kept] == ["b0001", "b0002"]
    assert [(r.locus, r.rule) for r in refused] == [
        ("b0003", bk.REFUSE_NON_GENE_LOCUS),
        ("b0004", bk.REFUSE_FRAGMENT_SYNONYM),
        ("b3036", bk.REFUSE_RETIRED),
        ("0", bk.REFUSE_ZERO_LOCUS),
    ]
    with pytest.raises(RuntimeError, match="share one b-number"):
        bk.select_genes([rows[0], rows[0]], FakeGenome())  # type: ignore[arg-type]


def test_phenotypes_store_the_column_and_the_reference_mean(
    release: dict[str, str],
) -> None:
    rows, _ = bk.read_fractions(release[bk.TABLE_S3])
    kept = rows[:2]
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
    monkeypatch.setattr(bk, "EXPECTED_GENE_KEYS", 2)
    monkeypatch.setattr(bk, "EXPECTED_ZERO_LOCUS_ROWS", 1)
    monkeypatch.setattr(bk, "EXPECTED_REFUSED_B_NUMBERS", 3)
    monkeypatch.setattr(bk, "assembly_reference", _reference_genome)

    dataset = bk.MrnaFractionBalakrishnan2022Dataset(
        root=str(root),
        ecoli_genome=FakeGenome(),  # type: ignore[arg-type]
    )
    assert len(dataset) == 28
    assert dataset.gene_set == {"b0001", "b0002"}
    items = [dataset.transform_item(dataset[i]) for i in range(len(dataset))]
    assert {i["reference"].genome_reference.strain for i in items} == {"NCM3722"}
    assert {i["reference"].phenotype_reference.n_libraries for i in items} == {2}
    assert sum(len(i["experiment"].genotype.perturbations) for i in items) == 15
    dataset.close_lmdb()

    preprocess = Path(dataset.preprocess_dir)
    ledger = json.loads((preprocess / "dropped_records.json").read_text())
    assert (ledger["records"], ledger["stored_gene_keys"]) == (28, 2)
    assert {k: v["n"] for k, v in ledger["refused_rows"].items()} == {
        bk.REFUSE_NON_GENE_LOCUS: 1,
        bk.REFUSE_FRAGMENT_SYNONYM: 1,
        bk.REFUSE_RETIRED: 1,
        bk.REFUSE_ZERO_LOCUS: 1,
    }
    samples = pd.read_csv(preprocess / "samples.csv")
    assert list(samples["sample"]) == [s.sample for s in bk.LOADED]
    assert (
        dict(zip(samples["sample"], samples["nitrogen_cell_verbatim"], strict=True))[
            "a4_1"
        ]
        == "11.34 mM (NH4)2SO7"
    )
    assert len(pd.read_csv(preprocess / "refused_rows.csv")) == 4

    report = bk.verify_build(str(root), genome=FakeGenome())  # type: ignore[arg-type]
    assert [(r.level, r.name) for r in report.results if not r.passed] == []


def test_process_refuses_a_count_that_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mrna_fraction_balakrishnan2022"
    paths = write_release(root / "raw")
    _patch_pins(monkeypatch, paths)
    monkeypatch.setattr(bk, "EXPECTED_SOURCE_ROWS", len(ROWS))
    with pytest.raises(RuntimeError, match="kept"):
        bk.MrnaFractionBalakrishnan2022Dataset(
            root=str(root),
            ecoli_genome=FakeGenome(),  # type: ignore[arg-type]
        )


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
    assert bk.check_quotes({**paths, bk.MORI_APPENDIX: str(mori)}) == {
        "n_quotes_checked": 13
    }


@pytest.mark.data
def test_built_lmdb_numbers() -> None:
    root = osp.join(_data_root(), "data", "torchcell", "mrna_fraction_balakrishnan2022")
    ledger = json.loads(Path(root, "preprocess", "dropped_records.json").read_text())
    assert (ledger["records"], ledger["released_rows"], ledger["stored_gene_keys"]) == (
        28,
        4342,
        4176,
    )
    assert {k: v["n"] for k, v in ledger["refused_rows"].items()} == {
        bk.REFUSE_NON_GENE_LOCUS: 74,
        bk.REFUSE_FRAGMENT_SYNONYM: 69,
        bk.REFUSE_ZERO_LOCUS: 18,
        bk.REFUSE_RETIRED: 5,
    }
    samples = pd.read_csv(Path(root, "preprocess", "samples.csv"))
    assert set(samples["n_gene_keys"]) == {4176}
