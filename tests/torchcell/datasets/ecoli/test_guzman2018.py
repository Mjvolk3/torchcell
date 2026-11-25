# tests/torchcell/datasets/ecoli/test_guzman2018.py
# [[tests.torchcell.datasets.ecoli.test_guzman2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_guzman2018.py
"""Guzmán 2018's own GEO release: the readers, the genotypes and a hermetic build.

Hermetic tests write a three-library TPM matrix, its series matrix and an Additional
file 4 workbook in ``tmp_path``; ``data`` tests read the deposited files and the store.
"""

from __future__ import annotations

import gzip
import json
import os
import os.path as osp
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openpyxl import Workbook

from torchcell.data.experiment_dataset import file_sha256
from torchcell.datamodels.schema import (
    DeletionStrainPerturbation,
    LocusVariantCallPerturbation,
    ReferenceGenome,
    RNASeqExpressionPhenotype,
    SiteVariantCallPerturbation,
    VariantCallMode,
)
from torchcell.datasets.ecoli import guzman2018 as m
from torchcell.verification.report import Level, LevelResult
from torchcell.verification.runners import load_records

COLUMNS = ["wt_R1", "thrA_A1_F5_I1_R1", "thrA_A2_F1_I0_R1"]
TPM_ROWS = [("BW25113_0001", [10.0, 20.0, 30.0]), ("BW25113_0349_1", [5.0, 0.0, 1.5])]
LOCI = {"thrA": "BW25113_0002", "metK": "BW25113_2942"}


def write_tpm(path: Path) -> Path:
    lines = ["," + ",".join(COLUMNS)]
    lines += [g + "," + ",".join(str(v) for v in vals) for g, vals in TPM_ROWS]
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def write_matrix(path: Path) -> Path:
    titles = "\t".join(f'"FPS{i}: {c}"' for i, c in enumerate(COLUMNS, 1))
    sources = [
        '"wild type"',
        '"Adapted - thrA knockout, Experiment1, Flask5, Clone"',
        '"Adapted - thrA knockout, Experiment2, Flask1, Population"',
    ]
    gsm = "\t".join(f'"GSM{i}"' for i in range(len(COLUMNS)))
    text = (
        f"!Sample_title\t{titles}\n!Sample_geo_accession\t{gsm}\n"
        f"!Sample_source_name_ch1\t{chr(9).join(sources)}\n"
    )
    with gzip.open(path, "wt") as handle:
        handle.write(text)
    return path


def write_mutations(path: Path) -> Path:
    workbook = Workbook()
    description = workbook.active
    assert description is not None
    description.title = "Description"
    description.append(["Additional File 4: Mutation summary"])
    description.append([m.FREQUENCY_NOTE])
    sheet = workbook.create_sheet("thrA")
    sheet.append(
        [
            "Position",
            "Mutation Type",
            "Sequence Change",
            "Gene",
            "Details",
            "FPS thrA A1 F5 I1 R1",
            "FPS thrA A2 F1 I0 R1",
        ]
    )
    sheet.append(
        [4118534, "SNP", "T->A", "metJ, metB", "intergenic (-211/-66)", 1, None]
    )
    sheet.append([3080768, "SNP", "G->C", "metK", "G235A (GGC->GCC)", None, 0.46])
    sheet.append(
        [638004, "MOB", "IS5 (+) +4 bp", "metK", "coding (461-464/1239 nt)", None, None]
    )
    workbook.save(path)
    return path


def test_readers_parse_the_release(tmp_path: Path) -> None:
    columns, tpm = m.read_tpm(write_tpm(tmp_path / "tpm.csv.gz"))
    assert columns == COLUMNS
    assert tpm["thrA_A2_F1_I0_R1"] == {"BW25113_0001": 30.0, "BW25113_0349_1": 1.5}
    libs = m.libraries(columns, m.read_series_matrix(write_matrix(tmp_path / "m.gz")))
    assert [(lib.gene, lib.call_mode) for lib in libs] == [
        (None, None),
        ("thrA", VariantCallMode.clone),
        ("thrA", VariantCallMode.population),
    ]
    assert libs[2].dna_sample == "FPS thrA A2 F1 I0 R1"
    calls = m.read_calls(
        write_mutations(tmp_path / "s4.xlsx"),
        ["FPS thrA A1 F5 I1 R1", "FPS thrA A2 F1 I0 R1"],
    )
    assert [c.position for c in calls["FPS thrA A1 F5 I1 R1"]] == [4118534]
    assert [c.frequency_statement for c in calls["FPS thrA A2 F1 I0 R1"]] == ["0.46"]


def test_libraries_refuse_an_unparsed_column() -> None:
    matrix = {
        "!Sample_source_name_ch1": {"odd": "x"},
        "!Sample_geo_accession": {"odd": "GSM1"},
    }
    with pytest.raises(ValueError, match="unparsed library column"):
        m.libraries(["odd"], matrix)


def test_genotype_keys_the_deletion_and_both_call_shapes(tmp_path: Path) -> None:
    calls = m.read_calls(
        write_mutations(tmp_path / "s4.xlsx"),
        ["FPS thrA A1 F5 I1 R1", "FPS thrA A2 F1 I0 R1"],
    )
    clone = m.Library(
        column="thrA_A1_F5_I1_R1",
        gsm="GSM1",
        source_name="x",
        gene="thrA",
        dna_sample="FPS thrA A1 F5 I1 R1",
        call_mode=VariantCallMode.clone,
    )
    perturbations = m.genotype(clone, calls["FPS thrA A1 F5 I1 R1"], LOCI).perturbations
    deletion = next(
        p for p in perturbations if isinstance(p, DeletionStrainPerturbation)
    )
    assert deletion.systematic_gene_name == "BW25113_0002"
    assert deletion.collection == "Keio collection"
    site = next(p for p in perturbations if isinstance(p, SiteVariantCallPerturbation))
    assert site.systematic_gene_name == "CP009273.1:4118534"
    assert site.flanking_gene_statement == "metJ, metB"
    assert site.call.call_mode is VariantCallMode.clone
    population = m.genotype(
        clone.model_copy(update={"call_mode": VariantCallMode.population}),
        calls["FPS thrA A2 F1 I0 R1"],
        LOCI,
    ).perturbations
    variant = next(p for p in population if isinstance(p, LocusVariantCallPerturbation))
    assert (variant.systematic_gene_name, variant.call.frequency) == (
        "BW25113_2942",
        0.46,
    )
    wild = clone.model_copy(update={"gene": None, "call_mode": None})
    assert m.genotype(wild, [], LOCI).perturbations == []


def test_mob_call_is_an_insertion_at_a_zero_length_interval() -> None:
    call = m.Call(
        position=638004,
        mutation_type="MOB",
        sequence_change="IS5 (+) +4 bp",
        gene="ybdR",
        details="coding (461-464/1239 nt)",
        frequency_statement="1",
        frequency=1.0,
    )
    variant = m.variant_call(call, VariantCallMode.clone)
    assert (variant.position_start, variant.position_end) == (638004, 638003)
    assert variant.type_statement == "MOB"


def test_phenotypes_gap_the_counts() -> None:
    phenotype = m.phenotype({"a": 1.0}, "wt_R2")
    assert phenotype.expression_count is None
    assert phenotype.gapped_fields() == {"expression_count"}
    assert phenotype.replicate_id == "wt_R2"
    reference = m.reference_phenotype([{"a": 1.0}, {"a": 3.0}])
    assert reference.expression_tpm == {"a": 2.0}
    assert reference.replicate_id is None
    with pytest.raises(ValueError, match="without a ProvenanceGap"):
        RNASeqExpressionPhenotype(expression_tpm={"a": 1.0}, expression_count=None)


def test_environment_kanamycin_only_for_deletion_strains() -> None:
    names = {c.compound.name for c in m.environment(kanamycin=True).media.components}
    assert "kanamycin" in names
    wild = {c.compound.name for c in m.environment(kanamycin=False).media.components}
    assert "kanamycin" not in wild


class _Genome:
    def resolve_gene_name(self, name: str) -> Any:
        if name not in LOCI:
            return SimpleNamespace(
                status=SimpleNamespace(value="retired"), systematic_name=name
            )
        return SimpleNamespace(
            status=SimpleNamespace(value="renamed"), systematic_name=LOCI[name]
        )


def test_resolve_symbols_refuses_a_symbol_off_the_annotation(tmp_path: Path) -> None:
    lib = m.Library(
        column="c",
        gsm="g",
        source_name="s",
        gene="nope",
        dna_sample=None,
        call_mode=None,
    )
    with pytest.raises(RuntimeError, match="does not reach one BW25113 locus"):
        m.resolve_symbols(_Genome(), [lib], {})  # type: ignore[arg-type]


REFERENCE = ReferenceGenome(
    species="Escherichia coli",
    strain="BW25113",
    assembly_set="ecoli_K12_BW25113_ASM75055v1",
    assembly_accession="GCA_000750555.1",
)


def test_hermetic_deposit_build_and_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    src = tmp_path / "src"
    src.mkdir()
    files = {
        m.TPM.name: write_tpm(src / m.TPM.name),
        m.SERIES_MATRIX.name: write_matrix(src / m.SERIES_MATRIX.name),
        m.MUTATIONS.name: write_mutations(src / m.MUTATIONS.name),
        m.PAPER_TEXT.name: src / m.PAPER_TEXT.name,
    }
    files[m.PAPER_TEXT.name].write_text("paper")
    for name in ("TPM", "SERIES_MATRIX", "MUTATIONS", "PAPER_TEXT"):
        raw = getattr(m, name)
        monkeypatch.setattr(
            m, name, raw.model_copy(update={"sha256": file_sha256(files[raw.name])})
        )
    monkeypatch.setattr(
        m, "RAW_FILES", (m.PAPER_TEXT, m.MUTATIONS, m.TPM, m.SERIES_MATRIX)
    )
    data_root = tmp_path / "data_root"
    m.deposit_raw_mirror(files, str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setattr(m, "EXPECTED_RECORDS", 3)
    monkeypatch.setattr(m, "EXPECTED_GENES", 2)
    monkeypatch.setattr(m, "EXPECTED_CALLS", 2)
    monkeypatch.setattr(m, "assembly_reference", lambda strain: REFERENCE)
    root = data_root / m.DATASET_ROOT_REL
    dataset = m.RnaseqGuzman2018Dataset(root=str(root), ecoli_genome=_Genome())  # type: ignore[arg-type]
    assert len(dataset) == 3
    ledger = json.loads((root / "preprocess" / "libraries.json").read_text())
    assert ledger["loci"] == {"metK": "BW25113_2942", "thrA": "BW25113_0002"}

    stub = LevelResult(level=Level.L3, name="stub", passed=True, message="stub")
    monkeypatch.setattr(m, "audit_sourced_value", lambda value, root: stub)
    monkeypatch.setattr(m, "_l4_crosscheck", lambda tpm, base: stub)
    report = m.verify_build(str(root), str(data_root))
    names = {r.name: r.passed for r in report.results}
    assert names["tpm_fidelity_to_release"] is True
    assert names["replicate_id_is_release_column"] is True
    stored = {
        r["experiment"]["phenotype"]["replicate_id"]: r["experiment"]["phenotype"][
            "expression_tpm"
        ]["BW25113_0001"]
        for r in load_records(str(root))
    }
    assert stored == {"wt_R1": 10.0, "thrA_A1_F5_I1_R1": 20.0, "thrA_A2_F1_I0_R1": 30.0}
    assert names["genotype_shape"] is False  # one wild-type library here, two declared


def _data_root() -> str:
    return os.environ.get("DATA_ROOT", "")


@pytest.mark.data
def test_release_counts() -> None:
    root = m.raw_mirror_dir(_data_root())
    columns, tpm = m.read_tpm(root / m.TPM.relpath)
    libs = m.libraries(columns, m.read_series_matrix(root / m.SERIES_MATRIX.relpath))
    calls = m.read_calls(
        root / m.MUTATIONS.relpath,
        sorted({lib.dna_sample for lib in libs if lib.dna_sample}),
    )
    assert len(libs) == 14
    assert sum(len(v) for v in calls.values()) == 12
    assert {len(v) for v in tpm.values()} == {4329}
    # Seven conditions in biological duplicate, the replicate label on every column.
    assert Counter(c.rsplit("_", 1)[1] for c in columns) == {"R1": 7, "R2": 7}
    conditions = Counter(c.rsplit("_", 1)[0] for c in columns)
    assert len(conditions) == 7
    assert set(conditions.values()) == {2}
    assert len({lib.gsm for lib in libs}) == 14


def _replicate_store_root() -> str:
    return os.environ.get(
        "GUZMAN2018_STORE", osp.join(_data_root(), m.DATASET_ROOT_REL)
    )


@pytest.mark.data
def test_store_holds_one_record_per_released_library() -> None:
    """Each record is one released library; the reference is the mean of the two WT.

    ``GUZMAN2018_STORE`` points at a scratch build; by default the dev store.
    """
    root = _replicate_store_root()
    if not osp.isdir(osp.join(root, "processed")):
        pytest.skip("rnaseq_guzman2018 store not built")
    _, tpm = m.read_tpm(m.raw_mirror_dir(_data_root()) / m.TPM.relpath)
    records = load_records(root)
    ids = [r["experiment"]["phenotype"]["replicate_id"] for r in records]
    assert len(records) == 14
    assert sorted(ids) == sorted(tpm)
    for rec in records:
        phenotype = rec["experiment"]["phenotype"]
        assert dict(phenotype["expression_tpm"]) == tpm[phenotype["replicate_id"]]
    reference = records[0]["reference"]["phenotype_reference"]
    assert reference["replicate_id"] is None
    wild = [tpm["wt_R1"], tpm["wt_R2"]]
    assert (
        max(
            abs(reference["expression_tpm"][g] - (wild[0][g] + wild[1][g]) / 2)
            for g in wild[0]
        )
        < 1e-9
    )


@pytest.mark.data
def test_dev_store_verifies() -> None:
    root = _replicate_store_root()
    if not osp.isdir(osp.join(root, "processed")):
        pytest.skip("rnaseq_guzman2018 store not built")
    report = m.verify_build(root, _data_root())
    assert report.passed, report.summary()
