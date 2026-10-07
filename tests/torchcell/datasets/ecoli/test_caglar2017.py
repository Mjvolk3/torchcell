# tests/torchcell/datasets/ecoli/test_caglar2017.py
# [[tests.torchcell.datasets.ecoli.test_caglar2017]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_caglar2017.py
"""Caglar 2017: the strain gate, the REL606 tier addition, and the raw mirror.

Synthetic tests pin the gate's logic, the retrieval records, the idempotent deposit and
the identifier parsers on hand-built bytes. The ``data``-marked tests read the real
``$DATA_ROOT`` (paper mirror and raw mirror) and run only with ``--data``.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from torchcell.datamodels.schema import BACTERIAL_LOCUS_TAG_PATTERNS
from torchcell.datasets.bacteria_common import bacterial_genome, reconcile_locus_tags
from torchcell.datasets.ecoli import caglar2017 as c
from torchcell.literature.manifest import Manifest, sha256_file
from torchcell.verification.sourced import (
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# The strain gate
# --------------------------------------------------------------------------- #
#: The tier's strain vocabulary before the REL606 set was deposited.
_K12_AND_KT2440 = Literal["MG1655", "BW25113", "KT2440"]


def test_the_gate_is_open_now_that_the_tier_holds_rel606() -> None:
    """REL606 is in ``BacterialReferenceStrain`` (set ecoli_B_REL606_ASM1798v1), so the
    finding is pinnable and carries neither the gap nor the tier addition.
    """
    finding = c.require_pinnable_strain()
    assert finding.strain == "REL606"
    assert finding.lineage == "E. coli B"
    assert finding.tier_strains == ("MG1655", "BW25113", "KT2440", "REL606")
    assert finding.pinnable is True
    assert (finding.gap, finding.tier_addition) == (None, None)


def test_without_rel606_in_the_vocabulary_no_tier_strain_is_its_genome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c, "BacterialReferenceStrain", _K12_AND_KT2440)
    finding = c.strain_pin_finding()
    assert finding.tier_strains == ("MG1655", "BW25113", "KT2440")
    assert finding.pinnable is False
    assert finding.gap == c.STRAIN_GAP
    assert finding.tier_addition == c.REL606_TIER_ADDITION


def test_the_gate_refuses_with_the_typed_gap_on_genome_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c, "BacterialReferenceStrain", _K12_AND_KT2440)
    with pytest.raises(
        c.UnpinnedStrainError, match=r"strain REL606 \(E\. coli B\)"
    ) as e:
        c.require_pinnable_strain()
    gap = e.value.finding.gap
    assert gap is not None
    assert gap.field == "genome_reference"
    assert gap.reason == ProvenanceGapReason.deferred_pending_source_review
    assert gap.looked_in is not None
    assert gap.looked_in.sha256 == c.PAPER_MD_SHA256
    assert gap.resolve_with is not None
    assert gap.resolve_with.source_uri == (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/017/985/"
        "GCA_000017985.1_ASM1798v1/"
    )
    assert gap.resolve_with.sha256 == (
        "aacf2559815f959c9417984ce1632228fd94caeac4b62b7910f714e310542e6b"
    )


def test_the_gate_opens_once_the_vocabulary_holds_the_strain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c, "STRAIN", c.STRAIN.model_copy(update={"value": "MG1655"}))
    finding = c.require_pinnable_strain()
    assert finding.pinnable is True
    assert finding.gap is None
    assert finding.tier_addition is None


# --------------------------------------------------------------------------- #
# The REL606 tier addition
# --------------------------------------------------------------------------- #
def test_the_tier_addition_names_asm1798v1_and_its_nine_ncbi_members() -> None:
    addition = c.REL606_TIER_ADDITION
    assert (addition.genbank_accession, addition.refseq_accession) == (
        "GCA_000017985.1",
        "GCF_000017985.1",
    )
    assert (addition.genbank_replicon, addition.refseq_replicon) == (
        "CP000819.1",
        "NC_012967.1",
    )
    assert addition.replicon_length_bp == 4629812
    assert [(m.path, m.role) for m in addition.members] == [
        ("GCA_000017985.1_ASM1798v1_genomic.gbff.gz", "annotation"),
        ("GCA_000017985.1_ASM1798v1_genomic.fna.gz", "sequence"),
        ("GCA_000017985.1_ASM1798v1_genomic.gff.gz", "annotation"),
        ("GCA_000017985.1_ASM1798v1_protein.faa.gz", "sequence"),
        ("GCA_000017985.1_ASM1798v1_feature_table.txt.gz", "index"),
        ("GCA_000017985.1_ASM1798v1_assembly_report.txt", "index"),
        ("GCF_000017985.1_ASM1798v1_genomic.gbff.gz", "annotation"),
        ("GCF_000017985.1_ASM1798v1_genomic.gff.gz", "annotation"),
        ("GCF_000017985.1_ASM1798v1_gene_ontology.gaf.gz", "annotation"),
    ]
    for member in addition.members:
        directory = (
            addition.genbank_url
            if member.path.startswith("GCA")
            else (addition.refseq_url)
        )
        assert member.url == directory + member.path
        assert re.fullmatch(r"[0-9a-f]{32}", member.md5)
        assert re.fullmatch(r"[0-9a-f]{64}", member.sha256)


def test_the_proposed_rel606_pattern_is_disjoint_from_every_deposited_namespace() -> (
    None
):
    rel606 = re.compile(c.REL606_TIER_ADDITION.locus_tag_pattern)
    rel606_tags = ["ECB_00001", "ECB_t00001", "ECB_r00022", "ECB_04279"]
    other_tags = ["b0001", "BW25113_0001", "PP_0001", "PP_16SA", "YAL001C", "ECB_0001"]
    assert [bool(rel606.match(tag)) for tag in rel606_tags] == [True] * 4
    assert [bool(rel606.match(tag)) for tag in other_tags] == [False] * 6
    namespace = c.REL606_TIER_ADDITION.gene_namespace
    assert BACTERIAL_LOCUS_TAG_PATTERNS[namespace] == rel606.pattern
    for other, pattern in BACTERIAL_LOCUS_TAG_PATTERNS.items():
        if other != namespace:
            assert [bool(re.match(pattern, t)) for t in rel606_tags] == [False] * 4


# --------------------------------------------------------------------------- #
# Retrieval records
# --------------------------------------------------------------------------- #
def test_the_si_tables_are_pmc_bucket_objects_s2_to_s5() -> None:
    specs = c.si_table_specs()
    assert [s.relpath for s in specs] == [
        "data/srep45303-s2.csv",
        "data/srep45303-s3.csv",
        "data/srep45303-s4.csv",
        "data/srep45303-s5.csv",
    ]
    first = specs[0].retrieval
    assert first.retriever == "torchcell.literature.retrieve.pmc_cloud_object"
    assert first.params == {"key": "PMC5394689.1/srep45303-s2.csv"}
    assert first.source_url == (
        "https://pmc-oa-opendata.s3.amazonaws.com/PMC5394689.1/srep45303-s2.csv"
    )
    assert (
        first.sha256
        == specs[0].sha256
        == ("1486290bf6a340ae64ee20c915435c0a00ff5eede489de1f56b62733b66f8940")
    )


def test_yp_batches_cover_table_s3_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(c, "YP_BATCH_SIZE", 2)
    monkeypatch.setattr(c, "YP_BATCH_SHA256", ("a" * 64, "b" * 64))
    specs = c.yp_batch_specs(["YP_1.1", "YP_2.1", "YP_3.1"])
    url0 = c.EFETCH + "?db=protein&rettype=gp&retmode=text&id=YP_1.1,YP_2.1"
    assert [(s.relpath, s.sha256, s.retrieval.source_url) for s in specs] == [
        ("ncbi_protein/yp_batch_00.gp", "a" * 64, url0),
        (
            "ncbi_protein/yp_batch_01.gp",
            "b" * 64,
            c.EFETCH + "?db=protein&rettype=gp&retmode=text&id=YP_3.1",
        ),
    ]
    assert specs[0].retrieval.retriever == "torchcell.literature.retrieve.direct_url"
    assert specs[0].retrieval.params == {"url": url0}
    with pytest.raises(ValueError, match="make 3 batches; 2 are pinned"):
        c.yp_batch_specs(["YP_1.1", "YP_2.1", "YP_3.1", "YP_4.1", "YP_5.1"])


# --------------------------------------------------------------------------- #
# Synthetic raw mirror
# --------------------------------------------------------------------------- #
def _genpept(records: list[tuple[str, list[str]]]) -> str:
    """GenPept text of protein records, each with one CDS carrying ``tags``."""
    out = []
    for accession, tags in records:
        record = SeqRecord(
            Seq("MKR"),
            id=accession,
            name=accession.split(".")[0],
            description="synthetic",
            annotations={"molecule_type": "protein"},
        )
        record.features.append(
            SeqFeature(SimpleLocation(0, 3), type="CDS", qualifiers={"locus_tag": tags})
        )
        out.append(record)
    handle = io.StringIO()
    SeqIO.write(out, handle, "genbank")
    return handle.getvalue()


def _stage(root: Path) -> dict[str, bytes]:
    """A staged raw tree: four tables (two ECB/YP rows) and one GenPept batch."""
    files = {
        "data/srep45303-s2.csv": b"sampleNum,dataSet\n1,MURI_016\n",
        "data/srep45303-s3.csv": b",MURI_016\nECB_00001,4.6\nECB_00002,8.6\n",
        "data/srep45303-s4.csv": b",MURI_016\nYP_1.1,0.9\nYP_2.1,8.2\n",
        "data/srep45303-s5.csv": b'"","Branch"\n"1","OAA from PEP"\n',
        "ncbi_protein/yp_batch_00.gp": _genpept(
            [("YP_1.1", ["ECB_00001"]), ("YP_2.1", ["ECB_00009"])]
        ).encode(),
    }
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return files


def _pin_to(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> None:
    tables = {
        table: (obj, _sha(files[f"data/{obj}"]), desc)
        for table, (obj, _, desc) in c.SI_TABLES.items()
    }
    monkeypatch.setattr(c, "SI_TABLES", tables)
    monkeypatch.setattr(
        c, "YP_BATCH_SHA256", (_sha(files["ncbi_protein/yp_batch_00.gp"]),)
    )


class _FrozenDatetime:
    stamp = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)

    @classmethod
    def now(cls, tz: Any = None) -> datetime:
        return cls.stamp


def test_deposit_raw_mirror_copies_every_file_and_writes_an_exact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _stage(tmp_path / "staging")
    _pin_to(monkeypatch, files)
    monkeypatch.setattr(c, "datetime", _FrozenDatetime)

    root = c.deposit_raw_mirror(
        source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )

    assert (
        root == tmp_path / "dr" / "torchcell-raw" / "caglarColiMolecularPhenotype2017"
    )
    for rel, data in files.items():
        assert (root / rel).read_bytes() == data
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["citation_key"] == "caglarColiMolecularPhenotype2017"
    assert manifest["doi"] == "10.1038/srep45303"
    assert manifest["created_at"] == "2026-10-07T12:00:00+00:00"
    assert [
        (f["path"], f["bytes"], f["sha256"], f["role"]) for f in manifest["files"]
    ] == [(rel, len(data), _sha(data), "raw_data") for rel, data in files.items()]
    batch = manifest["files"][4]["retrieval"]
    assert batch["method"] == "direct_url"
    assert batch["params"] == {
        "url": c.EFETCH + "?db=protein&rettype=gp&retmode=text&id=YP_1.1,YP_2.1"
    }
    assert manifest["files"][0]["retrieval"]["params"] == {
        "key": "PMC5394689.1/srep45303-s2.csv"
    }


def test_a_second_deposit_leaves_the_mirror_and_its_manifest_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _stage(tmp_path / "staging")
    _pin_to(monkeypatch, files)
    monkeypatch.setattr(c, "datetime", _FrozenDatetime)
    root = c.deposit_raw_mirror(
        source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )
    first = (root / "manifest.json").read_bytes()

    monkeypatch.setattr(_FrozenDatetime, "stamp", datetime(2027, 1, 1, tzinfo=UTC))
    c.deposit_raw_mirror(
        source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )

    assert (root / "manifest.json").read_bytes() == first


def test_deposit_refuses_staged_bytes_off_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _stage(tmp_path / "staging")
    _pin_to(monkeypatch, files)
    (tmp_path / "staging" / "data" / "srep45303-s5.csv").write_bytes(b"drifted")
    with pytest.raises(RuntimeError, match=r"srep45303-s5\.csv: sha256 .* pinned"):
        c.deposit_raw_mirror(
            source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
        )


def test_deposit_refuses_a_mirror_file_holding_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _stage(tmp_path / "staging")
    _pin_to(monkeypatch, files)
    existing = c.raw_mirror_dir(str(tmp_path / "dr")) / "data" / "srep45303-s2.csv"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"someone else's bytes")
    with pytest.raises(RuntimeError, match="exists with a different sha256; refusing"):
        c.deposit_raw_mirror(
            source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
        )
    assert existing.read_bytes() == b"someone else's bytes"
    root = c.raw_mirror_dir(str(tmp_path / "dr"))
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == [
        "data",
        "data/srep45303-s2.csv",
    ]


def test_deposit_refuses_a_manifest_recording_other_files_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _stage(tmp_path / "staging")
    _pin_to(monkeypatch, files)
    root = c.raw_mirror_dir(str(tmp_path / "dr"))
    root.mkdir(parents=True)
    other = Manifest(citation_key=c.CITATION_KEY, files=[])
    (root / "manifest.json").write_text(other.model_dump_json())
    with pytest.raises(
        RuntimeError, match="records other files; refusing to overwrite"
    ):
        c.deposit_raw_mirror(
            source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
        )
    assert [p.name for p in root.iterdir()] == ["manifest.json"]


def test_retrieve_raw_files_runs_each_recorded_retriever_and_checks_its_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _stage(tmp_path / "source")
    _pin_to(monkeypatch, files)
    by_url = {
        **{
            spec.retrieval.source_url: files[spec.relpath]
            for spec in c.si_table_specs()
        },
        c.EFETCH + "?db=protein&rettype=gp&retmode=text&id=YP_1.1,YP_2.1": files[
            "ncbi_protein/yp_batch_00.gp"
        ],
    }
    calls: list[str] = []

    def fake_retriever(record: Any) -> bytes:
        calls.append(record.source_url)
        return by_url[record.source_url]

    monkeypatch.setattr(c, "run_retriever", fake_retriever)
    staging = c.retrieve_raw_files(tmp_path / "staging")
    assert calls == list(by_url)
    for rel, data in files.items():
        assert (staging / rel).read_bytes() == data

    c.retrieve_raw_files(tmp_path / "staging")
    assert len(calls) == 5  # every staged file already holds its pin

    monkeypatch.setattr(c, "run_retriever", lambda record: b"upstream changed")
    (staging / "data" / "srep45303-s2.csv").unlink()
    with pytest.raises(RuntimeError, match="differs from the pin"):
        c.retrieve_raw_files(staging)


def test_load_manifest_and_manifest_sha256_read_the_deposit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _stage(tmp_path / "staging")
    _pin_to(monkeypatch, files)
    c.deposit_raw_mirror(
        source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )
    manifest = c.load_manifest(str(tmp_path / "dr"))
    assert c.manifest_sha256(manifest, "data/srep45303-s4.csv") == _sha(
        files["data/srep45303-s4.csv"]
    )
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        c.manifest_sha256(manifest, "data/absent.csv")


# --------------------------------------------------------------------------- #
# Identifier parsers and coverage
# --------------------------------------------------------------------------- #
def test_genpept_locus_tags_reads_one_tag_per_record_and_refuses_otherwise(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.gp"
    good.write_text(_genpept([("YP_1.1", ["ECB_00001"]), ("YP_2.1", ["ECB_00002"])]))
    assert c.genpept_locus_tags(good) == {"YP_1.1": "ECB_00001", "YP_2.1": "ECB_00002"}

    two_tags = tmp_path / "two.gp"
    two_tags.write_text(_genpept([("YP_1.1", ["ECB_00001", "ECB_00002"])]))
    with pytest.raises(ValueError, match="YP_1.1: CDS locus tags"):
        c.genpept_locus_tags(two_tags)

    repeated = tmp_path / "repeated.gp"
    repeated.write_text(
        _genpept([("YP_1.1", ["ECB_00001"]), ("YP_1.1", ["ECB_00001"])])
    )
    with pytest.raises(ValueError, match="YP_1.1 appears twice"):
        c.genpept_locus_tags(repeated)


def _gbff(
    path: Path,
    genes: list[tuple[str, list[str], str]],
    pseudo: frozenset[str] = frozenset(),
) -> None:
    """A gzipped GenBank flat file with one gene + CDS per (tag, old tags, protein id)."""
    record = SeqRecord(
        Seq("ATG" * 10),
        id="CP000819.1",
        name="CP000819",
        description="synthetic",
        annotations={"molecule_type": "DNA"},
    )
    for tag, old, protein in genes:
        gene_q: dict[str, list[str]] = {"locus_tag": [tag]}
        if old:
            gene_q["old_locus_tag"] = old
        if tag in pseudo:
            gene_q["pseudo"] = [""]
        record.features.append(
            SeqFeature(SimpleLocation(0, 3), type="gene", qualifiers=gene_q)
        )
        record.features.append(
            SeqFeature(
                SimpleLocation(0, 3),
                type="CDS",
                qualifiers={"locus_tag": [tag], "protein_id": [protein]},
            )
        )
    handle = io.StringIO()
    SeqIO.write([record], handle, "genbank")
    with gzip.open(path, "wt") as out:
        out.write(handle.getvalue())


def test_identifier_coverage_counts_each_route_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _stage(tmp_path / "raw")
    _pin_to(monkeypatch, files)
    _gbff(
        tmp_path / "gca.gbff.gz",
        [("ECB_00001", [], "ACT1.1"), ("ECB_00002", [], "ACT2.1")],
    )
    _gbff(tmp_path / "gcf.gbff.gz", [("ECB_RS00005", ["ECB_00001"], "WP_1.1")])

    coverage = c.identifier_coverage(
        tmp_path / "raw", tmp_path / "gca.gbff.gz", tmp_path / "gcf.gbff.gz"
    )

    assert coverage.model_dump() == {
        "mrna_ids": 2,
        "mrna_ecb": 2,
        "mrna_in_genbank_locus_tags": 2,
        "mrna_in_refseq_old_locus_tags": 1,
        "protein_ids": 2,
        "protein_yp": 2,
        "protein_in_genbank_protein_ids": 0,
        "protein_in_refseq_protein_ids": 0,
        "protein_resolved_by_ncbi_record": 2,
        "protein_locus_tag_in_genbank": 1,
        "protein_row_aligned_with_mrna": 1,
        "deposited_namespace_matches": {
            "ecoli_k12_mg1655_bnumber": 0,
            "ecoli_k12_bw25113_locus_tag": 0,
            "pputida_kt2440_locus_tag": 0,
            "ecoli_b_rel606_locus_tag": 2,
        },
        "rel606_pattern_matches": 2,
    }


def test_annotation_summary_counts_tag_forms_pseudogenes_and_gaf_rows(
    tmp_path: Path,
) -> None:
    _gbff(
        tmp_path / "gca.gbff.gz",
        [
            ("ECB_00001", [], "ACT1.1"),
            ("ECB_t00001", [], "ACT2.1"),
            ("ECB_0001", [], "A.1"),
        ],
        pseudo=frozenset({"ECB_t00001"}),
    )
    _gbff(
        tmp_path / "gcf.gbff.gz",
        [("ECB_RS00005", ["ECB_00001"], "WP_1.1"), ("ECB_RS00010", [], "WP_2.1")],
    )
    rows = [
        ["RefSeq", "WP_1.1", "x", "", "GO:0005737", "GO_REF:1", "IEA"] + [""] * 10,
        ["RefSeq", "WP_1.1", "x", "", "GO:0003677", "GO_REF:1", "IEA"] + [""] * 10,
        ["RefSeq", "WP_2.1", "y", "", "GO:0003677", "PMID:1", "IDA"] + [""] * 10,
    ]
    with gzip.open(tmp_path / "go.gaf.gz", "wt") as out:
        out.write("!gaf-version: 2.2\n")
        out.writelines("\t".join(row) + "\n" for row in rows)

    summary = c.annotation_summary(
        tmp_path / "gca.gbff.gz", tmp_path / "gcf.gbff.gz", tmp_path / "go.gaf.gz"
    )

    assert summary.model_dump() == {
        "genbank_gene_features": 3,
        "genbank_tag_prefixes": {"ECB_": 2, "ECB_t": 1},
        "genbank_pseudogenes": 1,
        "genbank_pattern_matches": 2,
        "refseq_gene_features": 2,
        "refseq_tag_prefixes": {"ECB_RS": 2},
        "refseq_genes_with_old_locus_tag": 1,
        "gaf_rows": 3,
        "gaf_objects": 2,
        "gaf_evidence": {"IDA": 1, "IEA": 2},
    }


# --------------------------------------------------------------------------- #
# Real data (run with --data)
# --------------------------------------------------------------------------- #
_DATA_ROOT = Path(os.environ.get("DATA_ROOT", ""))
_ON_DISK = pytest.mark.skipif(
    not (c.raw_mirror_dir(str(_DATA_ROOT)) / "manifest.json").is_file()
    or not (_DATA_ROOT / "torchcell-library" / c.CITATION_KEY / c.PAPER_MD).is_file(),
    reason="the Caglar 2017 paper mirror and raw mirror are not under $DATA_ROOT "
    "(export the real DATA_ROOT with --data)",
)


def _data_root() -> Path:
    return _DATA_ROOT


@pytest.mark.data
@_ON_DISK
def test_every_sourced_value_is_backed_by_a_verbatim_quote_in_the_mirror() -> None:
    library = _data_root() / "torchcell-library"
    values = [
        getattr(c, name)
        for name in dir(c)
        if isinstance(getattr(c, name), SourcedValue)
    ]
    assert len(values) == 26
    results = {value.quote: audit_sourced_value(value, library) for value in values}
    assert {quote: r.message for quote, r in results.items() if not r.passed} == {}


@pytest.mark.data
@_ON_DISK
def test_the_raw_mirror_holds_exactly_the_pinned_files() -> None:
    root = c.raw_mirror_dir(str(_data_root()))
    manifest = c.load_manifest(str(_data_root()))
    specs = c.raw_file_specs(c.read_table_ids(root / c.si_table_relpath("S3")))
    assert len(specs) == 46
    assert [(r.path, r.sha256) for r in manifest.files] == [
        (s.relpath, s.sha256) for s in specs
    ]
    for spec in specs:
        assert sha256_file(root / spec.relpath) == spec.sha256


@pytest.mark.data
@_ON_DISK
def test_the_mirror_tables_have_the_shapes_the_si_states() -> None:
    root = c.raw_mirror_dir(str(_data_root()))

    def header_and_rows(table: str) -> tuple[list[str], int]:
        with open(root / c.si_table_relpath(table), newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            return header, sum(1 for _ in reader)

    s1_header, s1_rows = header_and_rows("S1")
    s2_header, s2_rows = header_and_rows("S2")
    s3_header, s3_rows = header_and_rows("S3")
    assert (s2_rows, len(s2_header) - 1) == c.TABLE_S2.value
    assert (s3_rows, len(s3_header) - 1) == c.TABLE_S3.value
    assert s1_rows == 171
    assert {"RNA_Data_Freq", "Protein_Data_Freq"} <= set(s1_header)
    with open(root / c.si_table_relpath("S4"), newline="") as handle:
        branches = {row["Branch"] for row in csv.DictReader(handle)}
    assert len(branches) == c.TABLE_S4.value


@pytest.mark.data
@_ON_DISK
def test_every_table_s3_protein_resolves_to_a_rel606_locus_tag_through_ncbi() -> None:
    root = c.raw_mirror_dir(str(_data_root()))
    mrna = c.read_table_ids(root / c.si_table_relpath("S2"))
    protein = c.read_table_ids(root / c.si_table_relpath("S3"))
    crosswalk: dict[str, str] = {}
    for spec in c.yp_batch_specs(protein):
        crosswalk.update(c.genpept_locus_tags(root / spec.relpath))
    rel606 = re.compile(c.REL606_TIER_ADDITION.locus_tag_pattern)
    assert set(crosswalk) == set(protein)
    assert all(rel606.match(tag) for tag in crosswalk.values())
    assert sum(crosswalk[p] == m for m, p in zip(mrna, protein, strict=True)) == (
        ROW_ALIGNED
    )
    namespace = c.REL606_TIER_ADDITION.gene_namespace
    for other, pattern in BACTERIAL_LOCUS_TAG_PATTERNS.items():
        if other != namespace:
            assert not any(re.match(pattern, name) for name in mrna + protein)
    assert all(re.match(BACTERIAL_LOCUS_TAG_PATTERNS[namespace], m) for m in mrna)


_REL606_CACHE = _DATA_ROOT / "data/ecoli/rel606/genome/data.db"


@pytest.mark.data
@_ON_DISK
@pytest.mark.skipif(
    not _REL606_CACHE.is_file(), reason="the REL606 data.db cache is not built"
)
def test_every_table_s2_id_is_a_current_rel606_gene() -> None:
    """The reconciliation the note predicted, now measured on the REL606 genome: all
    4,196 Table S2 ids resolve CURRENT at the locus-tag layer, none remapped.
    """
    root = c.raw_mirror_dir(str(_data_root()))
    mrna = c.read_table_ids(root / c.si_table_relpath("S2"))
    genome = bacterial_genome("ecoli", "REL606", str(_data_root()))
    stored, report = reconcile_locus_tags(genome, pd.Series(mrna), label="caglar S2")
    assert stored.tolist() == mrna
    assert (report.unique_names, report.resolved, report.remapped) == (4196, 4196, 0)
    assert report.layer_histogram["locus tag"] == 4196
    assert report.outside_namespace == ()


#: Rows of Table S3 whose NCBI locus tag equals the ECB_ id on the same row of Table S2,
#: measured on the deposited mirror.
ROW_ALIGNED = 4196
