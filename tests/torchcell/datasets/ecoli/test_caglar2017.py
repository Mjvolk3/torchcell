# tests/torchcell/datasets/ecoli/test_caglar2017.py
# [[tests.torchcell.datasets.ecoli.test_caglar2017]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_caglar2017.py
"""Caglar 2017: the strain gate, the raw mirror, the log-base back-solve, both loaders.

Synthetic tests pin the gate's logic, the retrieval records, the idempotent deposit, the
identifier parsers, the DESeq2 VST back-solve and the sample-sheet, environment,
reference and phenotype builders on hand-built bytes, then build BOTH loaders end to end:
a synthetic mirror (Table S1 with six data samples and a pilot row, Tables S2 and S3 made
by applying DESeq2's +1 size factors and the base-2 VST to known integer counts, one
GenPept batch) deposited under a temporary ``DATA_ROOT``, over the real
``EcoliBREL606Genome`` class reading the synthetic REL606 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` (ECB_00001 9 bp, ECB_00002
15 bp, ECB_00003 14 bp) with the network refused.

The ``data``-marked tests read the real ``$DATA_ROOT`` and run only with ``--data``:
every ``SourcedValue`` against the mirror, the back-solve of the released tables (every
cell an integer count in base 2, none in base e), the paper's Table 1 sample counts
reproduced from Table S1, and the dev-tree builds (152 RNA-seq and 105 proteome records,
4,196 CURRENT loci each) through the module's own L0-L4 verifiers.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from tests.torchcell.sequence.genome._bacterial_fixtures import (
    REL606_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError
from torchcell.datamodels.media import DAVIS_MINIMAL, DM500
from torchcell.datamodels.schema import (
    BACTERIAL_LOCUS_TAG_PATTERNS,
    AssemblyReferenceGenome,
    Concentration,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    FoldChangeScale,
    PhysicalFactor,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    LocusTagResolutionError,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.ecoli import caglar2017 as c
from torchcell.literature.manifest import Manifest, RetrievalMethod, sha256_file
from torchcell.sequence.genome.ecoli.rel606 import REL606_ASSEMBLY, EcoliBREL606Genome
from torchcell.verification.report import (
    DerivationMethod,
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
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
def test_the_si_tables_are_pmc_bucket_objects_s2_to_s6_and_s9() -> None:
    """Tables S1 to S5 and S8, i.e. PMC objects ``-s2`` to ``-s6`` and ``-s9``: the SI
    numbers the tables one behind the bucket's objects. ``-s6`` (Table S5, the doubling
    times) joined the mirror in the #776 revision and ``-s9`` (Table S8, the DESeq2
    fold changes) in the #770 one, which is why the list is six and not four.
    """
    specs = c.si_table_specs()
    assert [s.relpath for s in specs] == [
        "data/srep45303-s2.csv",
        "data/srep45303-s3.csv",
        "data/srep45303-s4.csv",
        "data/srep45303-s5.csv",
        "data/srep45303-s6.csv",
        "data/srep45303-s9.csv",
    ]
    # Tables S5 and S8 are the doubling-time and fold-change loaders' tables and were
    # fetched for them, after the first deposit, so their records carry their own
    # retrieval date.
    assert specs[4].retrieval.retrieved_at == "2026-10-09"
    table_s8 = specs[5].retrieval
    assert (table_s8.method, table_s8.retrieved_at) == (
        RetrievalMethod.pmc_cloud,
        "2026-10-09",
    )
    assert table_s8.params == {"key": "PMC5394689.1/srep45303-s9.csv"}
    assert table_s8.source_url == (
        "https://pmc-oa-opendata.s3.amazonaws.com/PMC5394689.1/srep45303-s9.csv"
    )
    assert (
        table_s8.sha256
        == specs[5].sha256
        == ("738e1ee3f17e62a76a610741bc1b60ea21ee6eb80bdaa06846587dae791b44f5")
    )
    assert [s.retrieval.retrieved_at for s in specs[:4]] == [c.RAW_RETRIEVED_AT] * 4
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


#: Table S8's 24 columns, in released order (measured on the mirror).
S8_HEADER = [
    "Unnamed: 0",
    "X",
    "id",
    "baseMean",
    "log2FoldChange",
    "lfcSE",
    "stat",
    "pvalue",
    "padj",
    "gene_name",
    "signChange",
    "pick_data",
    "growthPhase.x",
    "test_for",
    "contrast",
    "base",
    "fullFileName",
    "dataType",
    "carbonSource",
    "Mg",
    "Na",
    "growthPhase.y",
    "investigatedEffect",
    "testVSbase",
]


def _s8_group(
    values: Sequence[tuple[str, float]],
    *,
    test_vs_base: str,
    phase: str,
    effect: str,
    test_for: str,
    contrast: str,
    base: str,
    data_type: str = "protein",
    p_values: Sequence[float] | None = None,
    adjusted: Sequence[float] | None = None,
) -> tuple[str, list[list[Any]]]:
    """One Table S8 group's rows and its ``fullFileName``.

    ``p_values`` defaults to the released shape of a tested row and ``adjusted`` to its
    Benjamini-Hochberg value, so a group built from the defaults passes
    ``adjustment_back_solve``. A ``nan`` fold change is a row DESeq2 left blank.
    """
    name = f"resDf_{data_type}_set00_{phase}_{effect}__{test_vs_base}.csv"
    raw = (
        [0.01 * (index + 1) for index in range(len(values))]
        if p_values is None
        else list(p_values)
    )
    adjust = (
        list(c.benjamini_hochberg(np.asarray(raw, dtype=np.float64)))
        if adjusted is None
        else list(adjusted)
    )
    rows = []
    for index, ((identifier, fold), p, q) in enumerate(
        zip(values, raw, adjust, strict=True), start=1
    ):
        rows.append(
            [
                index,
                index,
                identifier,
                10.0 * index,
                fold,
                0.25,
                fold / 0.25,
                p,
                q,
                f"gene{index}",
                1 if fold > 0 else -1,
                data_type,
                phase,
                test_for,
                contrast,
                base,
                name,
                data_type,
                "SYAN",
                "baseMgAllMg",
                "baseNaAllNa",
                phase,
                effect,
                test_vs_base,
            ]
        )
    return name, rows


def _s8_csv(values: Sequence[tuple[str, float]], **group: Any) -> bytes:
    """A one-group Table S8 file."""
    return _s8_file([_s8_group(values, **group)])


def _s8_file(groups: Sequence[tuple[str, list[list[Any]]]]) -> bytes:
    handle = io.StringIO()
    writer = csv.writer(handle)
    writer.writerow(S8_HEADER)
    for _, rows in groups:
        writer.writerows(rows)
    return handle.getvalue().encode()


def _stage(root: Path) -> dict[str, bytes]:
    """A staged raw tree: four tables (two ECB/YP rows) and one GenPept batch."""
    files = {
        "data/srep45303-s2.csv": b"sampleNum,dataSet\n1,MURI_016\n",
        "data/srep45303-s3.csv": b",MURI_016\nECB_00001,4.6\nECB_00002,8.6\n",
        "data/srep45303-s4.csv": b",MURI_016\nYP_1.1,0.9\nYP_2.1,8.2\n",
        "data/srep45303-s5.csv": b'"","Branch"\n"1","OAA from PEP"\n',
        "data/srep45303-s6.csv": (
            b"name,replicate,doubling.time.minutes,doubling.time.minutes.95m,"
            b"doubling.time.minutes.95p,r.squared\nGlucose.tab,1,53.2,49.1,59.2,0.98\n"
        ),
        "data/srep45303-s9.csv": _s8_csv(
            [("YP_1.1", 1.0), ("YP_2.1", -2.0)],
            test_vs_base="lowMgVSbaseMg",
            phase="Exp",
            effect="batchNumberPLUSMg",
            test_for="Mg_mM_Levels",
            contrast="lowMg",
            base="baseMg",
        ),
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
    batch = manifest["files"][5]["retrieval"]
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


# --------------------------------------------------------------------------- #
# deposit_si_table: the additive revision path
# --------------------------------------------------------------------------- #
#: ``SI_TABLES`` as the module releases it, captured before any test patches it.
_RELEASED_SI_TABLES: dict[str, tuple[str, str, str]] = dict(c.SI_TABLES)


def _pin_tables(
    monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes], tables: tuple[str, ...]
) -> None:
    """Pin exactly ``tables`` to the staged bytes, keeping the released table order."""
    monkeypatch.setattr(
        c,
        "SI_TABLES",
        {
            table: (obj, _sha(files[f"data/{obj}"]), desc)
            for table, (obj, _, desc) in _RELEASED_SI_TABLES.items()
            if table in tables
        },
    )


#: The four tables the FIRST deposit knew, before Table S5 was pinned.
_FIRST_DEPOSIT_TABLES = ("S1", "S2", "S3", "S4")


def _deposit_without_s5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, bytes], Path]:
    """The mirror as the first deposit left it: four tables and one GenPept batch."""
    files = _stage(tmp_path / "staging")
    _pin_tables(monkeypatch, files, _FIRST_DEPOSIT_TABLES)
    monkeypatch.setattr(
        c, "YP_BATCH_SHA256", (_sha(files["ncbi_protein/yp_batch_00.gp"]),)
    )
    monkeypatch.setattr(c, "datetime", _FrozenDatetime)
    root = c.deposit_raw_mirror(
        source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )
    assert not (root / "data" / "srep45303-s6.csv").exists()
    _pin_tables(monkeypatch, files, (*_FIRST_DEPOSIT_TABLES, "S5"))
    return files, root


def test_deposit_si_table_inserts_one_record_after_the_last_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, root = _deposit_without_s5(tmp_path, monkeypatch)
    before = json.loads((root / "manifest.json").read_text())
    assert [f["path"] for f in before["files"]] == [
        "data/srep45303-s2.csv",
        "data/srep45303-s3.csv",
        "data/srep45303-s4.csv",
        "data/srep45303-s5.csv",
        "ncbi_protein/yp_batch_00.gp",
    ]

    monkeypatch.setattr(_FrozenDatetime, "stamp", datetime(2027, 3, 3, tzinfo=UTC))
    dest = c.deposit_si_table(
        "S5", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )

    assert dest == root / "data" / "srep45303-s6.csv"
    assert dest.read_bytes() == files["data/srep45303-s6.csv"]
    after = json.loads((root / "manifest.json").read_text())
    assert [f["path"] for f in after["files"]] == [
        "data/srep45303-s2.csv",
        "data/srep45303-s3.csv",
        "data/srep45303-s4.csv",
        "data/srep45303-s5.csv",
        "data/srep45303-s6.csv",
        "ncbi_protein/yp_batch_00.gp",
    ]
    # The first deposit's records and its created_at are left byte-identical: this is a
    # revision of one mirror, not a second deposit of it.
    assert after["files"][:4] == before["files"][:4]
    assert after["files"][5] == before["files"][4]
    assert after["created_at"] == "2026-10-07T12:00:00+00:00"
    record = after["files"][4]
    staged = files["data/srep45303-s6.csv"]
    assert (record["role"], record["bytes"], record["sha256"]) == (
        "raw_data",
        len(staged),
        _sha(staged),
    )
    assert record["retrieval"]["method"] == "pmc_cloud"
    assert record["retrieval"]["params"] == {"key": "PMC5394689.1/srep45303-s6.csv"}
    # Its own retrieval date, because the first deposit did not fetch it.
    assert record["retrieval"]["retrieved_at"] == c.TABLE_RETRIEVED_AT["S5"]
    assert record["source"] == record["retrieval"]["source_url"]


def test_a_second_deposit_of_the_same_table_rewrites_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _deposit_without_s5(tmp_path, monkeypatch)
    c.deposit_si_table(
        "S5", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )
    first = (root / "manifest.json").read_bytes()

    c.deposit_si_table(
        "S5", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )

    assert (root / "manifest.json").read_bytes() == first


def test_deposit_si_table_refuses_a_record_already_present_with_other_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _deposit_without_s5(tmp_path, monkeypatch)
    c.deposit_si_table(
        "S5", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][4]["bytes"] = manifest["files"][4]["bytes"] + 1
    path.write_text(json.dumps(manifest))

    with pytest.raises(
        RuntimeError,
        match=r"records data/srep45303-s6\.csv differently; refusing to overwrite",
    ):
        c.deposit_si_table(
            "S5", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
        )
    assert json.loads(path.read_text()) == manifest


def test_deposit_si_table_refuses_staged_bytes_off_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _deposit_without_s5(tmp_path, monkeypatch)
    (tmp_path / "staging" / "data" / "srep45303-s6.csv").write_bytes(b"drifted")

    with pytest.raises(RuntimeError, match=r"srep45303-s6\.csv: sha256 .* pinned"):
        c.deposit_si_table(
            "S5", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
        )
    assert not (root / "data" / "srep45303-s6.csv").exists()
    assert [
        f["path"] for f in json.loads((root / "manifest.json").read_text())["files"]
    ][-1] == "ncbi_protein/yp_batch_00.gp"


def test_deposit_si_table_refuses_a_mirror_file_holding_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _deposit_without_s5(tmp_path, monkeypatch)
    existing = root / "data" / "srep45303-s6.csv"
    existing.write_bytes(b"someone else's bytes")

    with pytest.raises(RuntimeError, match="exists with a different sha256; refusing"):
        c.deposit_si_table(
            "S5", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
        )
    assert existing.read_bytes() == b"someone else's bytes"


def test_deposit_si_table_refuses_a_mirror_with_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _stage(tmp_path / "staging")
    _pin_tables(monkeypatch, files, (*_FIRST_DEPOSIT_TABLES, "S5"))

    with pytest.raises(
        RuntimeError, match=r"manifest\.json does not exist; run the full deposit first"
    ):
        c.deposit_si_table(
            "S5", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
        )
    assert not (
        c.raw_mirror_dir(str(tmp_path / "dr")) / "data" / "srep45303-s6.csv"
    ).exists()


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
    assert len(calls) == 6  # five SI tables, then the one GenPept batch
    for rel, data in files.items():
        assert (staging / rel).read_bytes() == data

    c.retrieve_raw_files(tmp_path / "staging")
    assert len(calls) == 6  # every staged file already holds its pin

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
# The log base: the DESeq2 VST back-solve
# --------------------------------------------------------------------------- #
#: Table S2's floor on the mirror; any floor works, this one is the real one.
FLOOR = -1.14456265985177


def _vst_table(counts: dict[str, list[int]], genes: list[str]) -> pd.DataFrame:
    """A released-style table: counts -> DESeq2 (+1) size factors -> base-2 VST."""
    matrix = np.array([counts[s] for s in counts], dtype=np.float64).T
    factors = c.deseq2_size_factors(matrix)
    return pd.DataFrame(
        c.vst_forward(matrix / factors, FLOOR), index=genes, columns=list(counts)
    )


def test_vst_inverse_undoes_vst_forward_and_maps_the_floor_to_zero() -> None:
    q = np.array([0.0, 0.5, 1.0, 37.25, 1e5])
    y = c.vst_forward(q, FLOOR)
    assert y[0] == pytest.approx(FLOOR, abs=1e-15)
    assert c.vst_inverse(y, FLOOR) == pytest.approx(q, rel=1e-9, abs=1e-12)
    # For a large count the transform is log2 of it, which is why the base is 2.
    assert y[-1] == pytest.approx(np.log2(1e5), abs=1e-4)


def test_deseq2_size_factors_are_the_median_ratio_to_the_geometric_mean_plus_one() -> (
    None
):
    counts = np.array([[1.0, 3.0], [3.0, 1.0], [7.0, 15.0]])
    # Plus one: [[2, 4], [4, 2], [8, 16]]; geometric means sqrt(8), sqrt(8), sqrt(128);
    # ratios 2**-0.5, 2**0.5, 2**-0.5 in sample one and the reverse in sample two.
    assert c.deseq2_size_factors(counts) == pytest.approx([2**-0.5, 2**0.5])


def test_back_solve_recovers_the_integer_counts_and_the_size_factors() -> None:
    counts = {"A": [0, 1, 7, 2], "B": [1, 0, 12, 4], "C": [3, 1, 0, 9]}
    table = _vst_table(counts, ["g1", "g2", "g3", "g4"])
    result = c.back_solve_counts(table, label="synthetic")
    assert result.counts.to_dict(orient="list") == counts
    assert result.counts.dtypes.unique().tolist() == [np.dtype("int64")]
    matrix = np.array(list(counts.values()), dtype=np.float64).T
    assert result.size_factors.to_numpy() == pytest.approx(
        c.deseq2_size_factors(matrix), rel=1e-12
    )
    evidence = result.evidence
    assert evidence.model_dump(
        exclude={
            "max_count_integer_deviation",
            "max_size_factor_relative_deviation",
            "size_factor_min",
            "size_factor_max",
        }
    ) == {
        "table": "synthetic",
        "n_features": 4,
        "n_samples": 3,
        "log_base": 2.0,
        "floor_value": FLOOR,
        "floor_level": 2.0**FLOOR,
        "zero_cells": 3,
        "samples_with_a_zero": 3,
        "library_size_min": 10,
        "library_size_median": 13.0,
        "library_size_max": 17,
    }
    assert evidence.max_count_integer_deviation < 1e-9
    assert evidence.max_size_factor_relative_deviation < 1e-12


def test_back_solve_refuses_a_natural_log_table() -> None:
    table = _vst_table({"A": [0, 1, 7], "B": [1, 0, 12]}, ["g1", "g2", "g3"])
    as_ln = table * np.log(2.0)  # the same numbers read as a base-e transform
    with pytest.raises(RuntimeError, match="from an integer"):
        c.back_solve_counts(as_ln, label="ln")


def test_back_solve_refuses_a_sample_whose_smallest_count_is_not_one() -> None:
    counts = {"A": [0, 2, 6, 10], "B": [1, 0, 12, 4], "C": [3, 1, 0, 9]}
    table = _vst_table(counts, ["g1", "g2", "g3", "g4"])
    with pytest.raises(RuntimeError, match="size factors differ from DESeq2's"):
        c.back_solve_counts(table, label="no-ones")


def test_back_solve_refuses_a_sample_with_no_nonzero_count() -> None:
    table = _vst_table({"A": [0, 1, 7], "B": [1, 0, 12]}, ["g1", "g2", "g3"])
    table["Z"] = FLOOR
    with pytest.raises(RuntimeError, match="sample Z has no nonzero count"):
        c.back_solve_counts(table, label="empty")


def test_the_log_base_is_recorded_as_a_back_solve_derivation() -> None:
    table = _vst_table({"A": [0, 1, 7], "B": [1, 0, 12]}, ["g1", "g2", "g3"])
    evidence = c.back_solve_counts(table, label="table_s2").evidence
    derivation = c.log_base_derivation([evidence])
    assert (derivation.field, derivation.method.value, derivation.value) == (
        "log_base",
        "back_solve",
        2.0,
    )
    assert sorted(derivation.diagnostics) == [
        "table_s2_max_count_integer_deviation",
        "table_s2_max_size_factor_relative_deviation",
    ]
    assert derivation.provenance == c.LOG_TRANSFORMED.provenance


# --------------------------------------------------------------------------- #
# Table S1, environments, references, phenotypes
# --------------------------------------------------------------------------- #
S1_HEADER = [
    "sampleNum",
    "dataSet",
    "experiment",
    "growthTime_hr",
    "harvestDate",
    "RNA_Data_Freq",
    "Protein_Data_Freq",
    "batchNumber",
    "carbonSource",
    "Mg_mM",
    "Na_mM",
    "growthPhase",
    "Mg_mM_Levels",
    "Na_mM_Levels",
    "uniqueCondition",
]

#: The synthetic sheet: (sample, experiment, hours, RNA, protein, batch, carbon, Mg, Na,
#: phase, Mg level, Na level, condition). MURI_001 is a pilot with no data.
S1_ROWS = [
    ("MURI_001", "pilot_24_hour", "24", "0", "0", "1", "base", "0.8", "5", "NA",
     "baseMg", "baseNa", "unique_condition_35"),
    ("MURI_002", "glucose_time_course", "3", "1", "1", "7", "glucose", "0.8", "5",
     "exponential", "baseMg", "baseNa", "unique_condition_07"),
    ("MURI_003", "glucose_time_course", "4", "1", "2", "8", "glucose", "0.8", "5",
     "exponential", "baseMg", "baseNa", "unique_condition_07"),
    ("MURI_004", "glucose_time_course", "24", "1", "1", "7", "glucose", "0.8", "5",
     "stationary", "baseMg", "baseNa", "unique_condition_25"),
    ("MURI_005", "lactate_growth", "9", "1", "0", "16", "lactate", "0.8", "5",
     "exponential", "baseMg", "baseNa", "unique_condition_16"),
    ("MURI_006", "MgSO4_stress_low", "5", "1", "1", "25", "glucose", "0.005", "5",
     "exponential", "lowMg", "baseNa", "unique_condition_02"),
    ("MURI_007", "NaCl_stress", "29", "0", "1", "14", "glucose", "0.8", "200",
     "stationary", "baseMg", "highNa", "unique_condition_27"),
]  # fmt: skip
GENES = ["ECB_00001", "ECB_00002", "ECB_00003"]
#: Gene-feature spans of the three synthetic loci (REL606_LOCI): 9, 15 and 14 bp.
LENGTHS = np.array([9.0, 15.0, 14.0])
RNA_COUNTS = {
    "MURI_002": [0, 1, 7],
    "MURI_003": [1, 0, 12],
    "MURI_004": [3, 1, 0],
    "MURI_005": [0, 1, 4],
    "MURI_006": [1, 5, 0],
}
PROTEINS = ["YP_1.1", "YP_2.1", "YP_3.1"]
PROTEIN_COUNTS = {
    "MURI_002": [1, 0, 3],
    "MURI_003": [0, 1, 6],
    "MURI_004": [2, 1, 0],
    "MURI_006": [1, 0, 1],
    "MURI_007": [0, 4, 1],
}


def _sheet_csv(rows: Sequence[tuple[str, ...]]) -> bytes:
    handle = io.StringIO()
    writer = csv.writer(handle)
    writer.writerow(S1_HEADER)
    for number, row in enumerate(rows, start=1):
        writer.writerow([number, row[0], row[1], row[2], "1/1/13", *row[3:]])
    return handle.getvalue().encode()


def _rows(path: Path, rows: Sequence[tuple[str, ...]] = S1_ROWS) -> list[c.SampleRow]:
    path.write_bytes(_sheet_csv(rows))
    return c.read_sample_sheet(path)


def test_read_sample_sheet_keeps_rows_with_data_and_types_every_cell(
    tmp_path: Path,
) -> None:
    rows = _rows(tmp_path / "s1.csv")
    assert [r.sample for r in rows] == [f"MURI_00{i}" for i in range(2, 8)]
    assert rows[4].model_dump() == {
        "sample": "MURI_006",
        "experiment": "MgSO4_stress_low",
        "condition": "unique_condition_02",
        "growth_time_hr": 5.0,
        "batch": 25,
        "carbon_source": "glucose",
        "mg_mm": 0.005,
        "mg_level": "lowMg",
        "na_mm": 5.0,
        "na_level": "baseNa",
        "growth_phase": c.GrowthPhase.exponential,
        "rna_technical_replicates": 1,
        "protein_technical_replicates": 1,
    }
    assert [r.reference_condition for r in rows] == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ({10: "baseMg", 7: "0.08"}, r"MURI_002: baseMg at 0.08 mM Mg2\+"),
        ({11: "baseNa", 8: "100"}, r"MURI_002: baseNa at 100.0 mM Na\+"),
        ({9: "log"}, "growth_phase"),
        ({6: "acetate"}, "carbon_source"),
    ],
)
def test_read_sample_sheet_refuses_a_cell_it_cannot_read(
    tmp_path: Path, edit: dict[int, str], message: str
) -> None:
    row = list(S1_ROWS[1])
    for index, value in edit.items():
        row[index] = value
    with pytest.raises(ValueError, match=message):
        _rows(tmp_path / "s1.csv", [tuple(row)])


def test_read_sample_sheet_refuses_a_repeated_sample(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repeats a sample id"):
        _rows(tmp_path / "s1.csv", [S1_ROWS[1], S1_ROWS[1]])


def test_each_condition_is_davis_minimal_with_its_stated_edits(tmp_path: Path) -> None:
    rows = {r.sample: r for r in _rows(tmp_path / "s1.csv")}
    base = c.build_environment(rows["MURI_002"])
    assert base.media is DM500
    assert (base.perturbations, base.duration_hours, base.aerobicity) == (
        [],
        3.0,
        "aerobic",
    )
    assert base.temperature is not None and base.temperature.value == 37.0

    lactate = c.build_environment(rows["MURI_005"])
    assert lactate.media is DAVIS_MINIMAL
    (carbon,) = lactate.perturbations
    assert isinstance(carbon, EnvironmentPhysicalPerturbation)
    assert carbon.factor is PhysicalFactor.carbon_source
    assert carbon.magnitude is not None
    assert (carbon.magnitude.value, carbon.magnitude.unit) == (
        0.5,
        ConcentrationUnit.g_per_l,
    )
    assert carbon.agent is not None and carbon.agent.name == "lactate"

    (magnesium,) = c.build_environment(rows["MURI_006"]).perturbations
    assert isinstance(magnesium, SmallMoleculePerturbation)
    assert magnesium.compound.name == "magnesium sulfate"
    assert (magnesium.concentration.value, magnesium.concentration.unit) == (
        0.005,
        ConcentrationUnit.millimolar,
    )
    assert magnesium.description.startswith("magnesium sulfate set to 0.005 mM final")

    (sodium,) = c.build_environment(rows["MURI_007"]).perturbations
    assert isinstance(sodium, SmallMoleculePerturbation)
    assert sodium.compound.name == "sodium chloride"
    assert sodium.concentration.value == 195.0  # 200 mM Na+ less the ~5 mM base
    with pytest.raises(ValueError, match="is not above the 5.0 mM base"):
        c.sodium_perturbation(5.0)


def test_environment_cache_returns_one_object_per_defining_cells(
    tmp_path: Path,
) -> None:
    rows = _rows(tmp_path / "s1.csv")
    environment_of = c.environment_cache()
    assert environment_of(rows[0]) is environment_of(rows[0])
    assert environment_of(rows[0]) is not environment_of(rows[1])  # 3 h vs 4 h


def test_reference_rows_follow_the_glucose_base_rule_per_phase(tmp_path: Path) -> None:
    rows = _rows(tmp_path / "s1.csv")
    references = c.reference_rows(rows)
    assert {p.value: [r.sample for r in m] for p, m in references.items()} == {
        "exponential": ["MURI_002", "MURI_003"],
        "stationary": ["MURI_004"],
    }
    environment = c.reference_environment(c.GrowthPhase.exponential, [4.0, 3.0, 3.0])
    assert environment.duration_hours is None
    (gap,) = environment.provenance_gaps
    assert gap.field == "duration_hours"
    assert gap.note is not None and "dates at 3, 4 h" in gap.note
    clash = [r.model_copy(update={"condition": "other"}) for r in rows[:1]] + rows[1:]
    with pytest.raises(RuntimeError, match="selects conditions"):
        c.reference_rows(clash)


def test_tpm_is_counts_per_base_scaled_to_a_million() -> None:
    values = c.tpm(np.array([0.0, 1.0, 7.0]), LENGTHS)
    rate = np.array([0.0, 1 / 15, 7 / 14])
    assert values == pytest.approx(rate / rate.sum() * 1e6)
    assert values.sum() == pytest.approx(1e6)


def test_the_rnaseq_phenotype_and_reference_keep_counts_and_tpm() -> None:
    phenotype = c.rnaseq_phenotype(GENES, np.array([0.0, 1.0, 7.0]), LENGTHS)
    assert dict(phenotype.expression_count) == dict(zip(GENES, [0, 1, 7], strict=True))
    assert phenotype.measurement_type == "rnaseq_tpm"
    assert [g.field for g in phenotype.provenance_gaps] == ["n_mapped_reads"]
    block = np.array([[0.0, 1.0], [1.0, 0.0], [7.0, 12.0]])
    reference = c.rnaseq_reference_phenotype(GENES, block, LENGTHS)
    expected = (c.tpm(block[:, 0], LENGTHS) + c.tpm(block[:, 1], LENGTHS)) / 2
    assert list(reference.expression_tpm.values()) == pytest.approx(expected)
    # mean counts 0.5, 0.5, 9.5 round half to even
    assert dict(reference.expression_count) == dict(zip(GENES, [0, 0, 10], strict=True))


def test_the_protein_phenotype_is_the_count_over_the_size_factor() -> None:
    phenotype = c.protein_phenotype(GENES, np.array([2.0, 0.0, 5.0]), 2.0)
    assert phenotype.protein_abundance == dict(zip(GENES, [1.0, 0.0, 2.5], strict=True))
    assert phenotype.n_replicates == dict.fromkeys(GENES, 1)
    assert phenotype.protein_abundance_se is None
    block = np.array([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]])
    reference = c.protein_reference_phenotype(GENES, block)
    assert reference.protein_abundance == dict(zip(GENES, [2.0, 0.0, 2.0], strict=True))
    assert reference.protein_abundance_se == pytest.approx(
        dict(zip(GENES, [1.0, 0.0, 0.0], strict=True))
    )
    assert reference.n_replicates == dict.fromkeys(GENES, 2)
    single = c.protein_reference_phenotype(GENES, block[:, :1])
    assert single.protein_abundance_se is not None
    assert all(math.isnan(v) for v in single.protein_abundance_se.values())


def test_protein_crosswalk_covers_table_s3_exactly(tmp_path: Path) -> None:
    first = tmp_path / "b0.gp"
    first.write_text(_genpept([("YP_1.1", ["ECB_00001"]), ("YP_2.1", ["ECB_00002"])]))
    second = tmp_path / "b1.gp"
    second.write_text(_genpept([("YP_3.1", ["ECB_00003"])]))
    assert c.protein_crosswalk([first, second], PROTEINS) == dict(
        zip(PROTEINS, GENES, strict=True)
    )
    with pytest.raises(RuntimeError, match=r"miss \['YP_3.1'\] and add \[\]"):
        c.protein_crosswalk([first], PROTEINS)
    with pytest.raises(ValueError, match="YP_1.1 appears in two batches"):
        c.protein_crosswalk([first, first], PROTEINS)


def test_read_released_table_requires_the_sheet_samples_in_order(
    tmp_path: Path,
) -> None:
    rows = [r for r in _rows(tmp_path / "s1.csv") if r.rna_technical_replicates > 0]
    table = pd.DataFrame(RNA_COUNTS, index=GENES)
    table.to_csv(tmp_path / "s2.csv")
    assert c.read_released_table(tmp_path / "s2.csv", rows).columns.tolist() == list(
        RNA_COUNTS
    )
    table[list(reversed(RNA_COUNTS))].to_csv(tmp_path / "swapped.csv")
    with pytest.raises(RuntimeError, match="in sheet order"):
        c.read_released_table(tmp_path / "swapped.csv", rows)
    pd.DataFrame(RNA_COUNTS, index=["ECB_00001"] * 3).to_csv(tmp_path / "dup.csv")
    with pytest.raises(RuntimeError, match="repeats an identifier"):
        c.read_released_table(tmp_path / "dup.csv", rows)


def test_build_accounting_refuses_numbers_that_do_not_add_up() -> None:
    accounting = c.BuildAccounting(
        dataset="d",
        table="t",
        sheet_rows_with_data=3,
        candidate_records=3,
        kept_records=2,
        dropped_records=1,
        drops_by_reason={},
        kept_by_growth_phase={},
        kept_by_carbon_source={},
        kept_by_experiment={},
        reference_samples={},
        notes=[],
    )
    with pytest.raises(RuntimeError, match="drop reasons do not add up"):
        accounting.check()
    with pytest.raises(RuntimeError, match="retention does not add up"):
        accounting.model_copy(update={"kept_records": 3}).check()


# --------------------------------------------------------------------------- #
# Both loaders end to end over a synthetic mirror and a synthetic REL606 genome
# --------------------------------------------------------------------------- #
REL606_PIN = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="REL606",
    assembly_set="ecoli_B_REL606_ASM1798v1",
    assembly_accession="GCA_000017985.1",
)


def _synthetic_mirror(staging: Path) -> dict[str, bytes]:
    """Tables S1 to S4 and one GenPept batch, written as the mirror lays them out."""
    staging.mkdir(parents=True, exist_ok=True)
    s2 = _vst_table(RNA_COUNTS, GENES)
    s3 = _vst_table(PROTEIN_COUNTS, PROTEINS)
    s2.to_csv(staging / "s2.csv")
    s3.to_csv(staging / "s3.csv")
    files = {
        "data/srep45303-s2.csv": _sheet_csv(S1_ROWS),
        "data/srep45303-s3.csv": (staging / "s2.csv").read_bytes(),
        "data/srep45303-s4.csv": (staging / "s3.csv").read_bytes(),
        "data/srep45303-s5.csv": b'"","Branch"\n"1","OAA from PEP"\n',
        "data/srep45303-s6.csv": (
            b"name,replicate,doubling.time.minutes,doubling.time.minutes.95m,"
            b"doubling.time.minutes.95p,r.squared\nGlucose.tab,1,53.2,49.1,59.2,0.98\n"
        ),
        "data/srep45303-s9.csv": _s8_file(S8_GROUPS),
        "ncbi_protein/yp_batch_00.gp": _genpept(
            [(p, [g]) for p, g in zip(PROTEINS, GENES, strict=True)]
        ).encode(),
    }
    for rel, data in files.items():
        (staging / rel).parent.mkdir(parents=True, exist_ok=True)
        (staging / rel).write_bytes(data)
    return files


@pytest.fixture
def mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The synthetic raw mirror deposited under a temporary DATA_ROOT, pins patched."""
    files = _synthetic_mirror(tmp_path / "staging")
    _pin_to(monkeypatch, files)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    return c.deposit_raw_mirror(source_dir=tmp_path / "staging")


@pytest.fixture
def genome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliBREL606Genome:
    """The synthetic REL606 genome; the network refuses and the pin is patched."""
    tier = write_assembly(tmp_path / "tier", REL606_ASSEMBLY, REL606_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, tier)
    monkeypatch.setattr(c, "assembly_reference", lambda strain: REL606_PIN)
    (tmp_path / "rel606").mkdir()
    return EcoliBREL606Genome(genome_root=str(tmp_path / "rel606"), overwrite=False)


@pytest.fixture
def rnaseq(
    tmp_path: Path, mirror: Path, genome: EcoliBREL606Genome
) -> c.RnaseqCaglar2017Dataset:
    return c.RnaseqCaglar2017Dataset(
        root=str(tmp_path / "rnaseq_caglar2017"), ecoli_genome=genome
    )


@pytest.fixture
def proteome(
    tmp_path: Path, mirror: Path, genome: EcoliBREL606Genome
) -> c.ProteomeCaglar2017Dataset:
    return c.ProteomeCaglar2017Dataset(
        root=str(tmp_path / "proteome_caglar2017"), ecoli_genome=genome
    )


def _ledger(dataset: Any, name: str) -> Any:
    return json.loads(Path(dataset.preprocess_dir, name).read_text())


def _record(dataset: Any, index: int) -> dict[str, Any]:
    record = dataset.get_single_item(index)
    assert record is not None
    return dict(record)


def test_download_links_each_pinned_mirror_file_and_refuses_an_off_pin_manifest(
    tmp_path: Path, mirror: Path
) -> None:
    dataset = c.ProteomeCaglar2017Dataset.__new__(c.ProteomeCaglar2017Dataset)
    dataset.root = str(tmp_path / "linked")
    dataset.download()
    raw = tmp_path / "linked" / "raw"
    assert sorted(os.listdir(raw)) == [
        "srep45303-s2.csv",
        "srep45303-s4.csv",
        "yp_batch_00.gp",
    ]
    assert os.readlink(raw / "yp_batch_00.gp") == str(
        mirror / "ncbi_protein" / "yp_batch_00.gp"
    )
    manifest = json.loads((mirror / "manifest.json").read_text())
    manifest["files"][0]["sha256"] = "ab" * 32
    (mirror / "manifest.json").write_text(json.dumps(manifest))
    other = c.RnaseqCaglar2017Dataset.__new__(c.RnaseqCaglar2017Dataset)
    other.root = str(tmp_path / "linked2")
    with pytest.raises(ManifestPinMismatchError):
        other.download()


def test_the_rnaseq_build_writes_one_record_per_library(
    rnaseq: c.RnaseqCaglar2017Dataset,
) -> None:
    assert len(rnaseq) == 5
    assert sorted(rnaseq.gene_set) == GENES
    samples = _ledger(rnaseq, "record_samples.json")
    assert [(s["index"], s["sample"]) for s in samples] == list(enumerate(RNA_COUNTS))
    for index, (sample, counts) in enumerate(RNA_COUNTS.items()):
        phenotype = _record(rnaseq, index)["experiment"]["phenotype"]
        assert dict(phenotype["expression_count"]) == dict(
            zip(GENES, counts, strict=True)
        )
        assert list(phenotype["expression_tpm"].values()) == pytest.approx(
            c.tpm(np.array(counts, dtype=float), LENGTHS)
        )
        assert samples[index]["library_size"] == sum(counts)
    lactate = _record(rnaseq, 3)["experiment"]
    assert lactate["genotype"]["perturbations"] == []
    assert lactate["environment"]["duration_hours"] == 9.0
    assert lactate["environment"]["perturbations"][0]["agent"]["name"] == "lactate"


def test_each_rnaseq_record_points_at_its_phase_s_reference(
    rnaseq: c.RnaseqCaglar2017Dataset,
) -> None:
    exponential = _record(rnaseq, 0)["reference"]
    stationary = _record(rnaseq, 2)["reference"]
    assert _record(rnaseq, 4)["reference"] == exponential
    assert exponential["genome_reference"] == REL606_PIN.model_dump()
    block = np.array([RNA_COUNTS["MURI_002"], RNA_COUNTS["MURI_003"]], dtype=float).T
    assert list(
        exponential["phenotype_reference"]["expression_tpm"].values()
    ) == pytest.approx(
        list(
            c.rnaseq_reference_phenotype(GENES, block, LENGTHS).expression_tpm.values()
        )
    )
    assert dict(stationary["phenotype_reference"]["expression_count"]) == dict(
        zip(GENES, RNA_COUNTS["MURI_004"], strict=True)
    )
    assert rnaseq.experiment_reference_index is not None
    assert len(rnaseq.experiment_reference_index) == 2


def test_the_rnaseq_build_writes_its_ledgers(rnaseq: c.RnaseqCaglar2017Dataset) -> None:
    evidence = c.VstBackSolve.model_validate(_ledger(rnaseq, "vst_back_solve.json"))
    assert (evidence.table, evidence.n_samples, evidence.n_features) == (
        "table_s2",
        5,
        3,
    )
    assert _ledger(rnaseq, "log_base_derivation.json")["value"] == 2.0
    assert _ledger(rnaseq, "gene_lengths.json") == {
        "ECB_00001": 9,
        "ECB_00002": 15,
        "ECB_00003": 14,
    }
    reconciliation = _ledger(rnaseq, "locus_tag_reconciliation.json")
    assert reconciliation["status_histogram"]["current"] == 3
    accounting = c.BuildAccounting.model_validate(
        _ledger(rnaseq, "build_accounting.json")
    )
    assert accounting.model_dump(include={
        "sheet_rows_with_data", "candidate_records", "kept_records", "dropped_records",
        "drops_by_reason", "kept_by_growth_phase", "kept_by_carbon_source",
        "reference_samples",
    }) == {
        "sheet_rows_with_data": 6,
        "candidate_records": 5,
        "kept_records": 5,
        "dropped_records": 0,
        "drops_by_reason": {},
        "kept_by_growth_phase": {"exponential": 4, "stationary": 1},
        "kept_by_carbon_source": {"glucose": 4, "lactate": 1},
        "reference_samples": {
            "exponential": ["MURI_002", "MURI_003"],
            "stationary": ["MURI_004"],
        },
    }  # fmt: skip
    groups = {g["condition"]: g for g in _ledger(rnaseq, "replicate_groups.json")}
    assert groups["unique_condition_07"]["samples"] == ["MURI_002", "MURI_003"]
    assert groups["unique_condition_07"]["growth_times_hr"] == [3.0, 4.0]


def test_the_proteome_build_keys_table_s3_by_rel606_locus(
    proteome: c.ProteomeCaglar2017Dataset,
) -> None:
    assert len(proteome) == 5
    assert sorted(proteome.gene_set) == GENES
    assert _ledger(proteome, "protein_crosswalk.json") == dict(
        zip(PROTEINS, GENES, strict=True)
    )
    matrix = np.array(list(PROTEIN_COUNTS.values()), dtype=float).T
    factors = c.deseq2_size_factors(matrix)
    samples = _ledger(proteome, "record_samples.json")
    assert [s["technical_replicates"] for s in samples] == [1, 2, 1, 1, 1]
    for index, counts in enumerate(PROTEIN_COUNTS.values()):
        phenotype = _record(proteome, index)["experiment"]["phenotype"]
        assert list(phenotype["protein_abundance"].values()) == pytest.approx(
            np.array(counts) / factors[index]
        )
        assert set(phenotype["n_replicates"].values()) == {1}
        assert samples[index]["size_factor"] == pytest.approx(factors[index])
    reference = _record(proteome, 4)["reference"]["phenotype_reference"]  # stationary
    block = np.array([PROTEIN_COUNTS["MURI_004"]], dtype=float).T / factors[2]
    assert list(reference["protein_abundance"].values()) == pytest.approx(block[:, 0])
    assert set(reference["n_replicates"].values()) == {1}
    sodium = _record(proteome, 4)["experiment"]["environment"]["perturbations"]
    assert [p["compound"]["name"] for p in sodium] == ["sodium chloride"]


# --------------------------------------------------------------------------- #
# #771: per-record attribution to the study that first reported the sample
#
# 27 of the 152 real mRNA samples and 27 of the 105 real protein samples are Houser
# 2015's glucose time course. The synthetic sheet carries the same shape in miniature:
# MURI_002, MURI_003 and MURI_004 are `glucose_time_course`, so 3 of the 5 records of
# each family are Houser's and 2 are this paper's.
# --------------------------------------------------------------------------- #
HOUSER_PUBLICATION = {
    "doi": "10.1371/journal.pcbi.1004400",
    "doi_url": "https://doi.org/10.1371/journal.pcbi.1004400",
    "pubmed_id": "26275208",
    "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/26275208/",
}


def test_attribute_sample_splits_on_table_s1_s_experiment_column(
    tmp_path: Path,
) -> None:
    """The glucose time course is Houser 2015's; every other condition is Caglar's."""
    rows = _rows(tmp_path / "s1.csv")
    attributions = {row.sample: c.attribute_sample(row) for row in rows}
    assert {s: a.study for s, a in attributions.items()} == {
        "MURI_002": "houser2015",
        "MURI_003": "houser2015",
        "MURI_004": "houser2015",
        "MURI_005": "caglar2017",
        "MURI_006": "caglar2017",
        "MURI_007": "caglar2017",
    }
    houser = attributions["MURI_002"]
    assert houser.experiment == c.HOUSER2015_EXPERIMENT == "glucose_time_course"
    assert houser.doi == c.HOUSER2015_DOI
    assert houser.evidence == (
        "HOUSER2015_DEFERRAL",
        "HOUSER2015_CITATION",
        "HOUSER2015_DEPOSITS",
    )
    assert attributions["MURI_005"].evidence == ()
    assert attributions["MURI_005"].doi == c.PAPER_DOI


def test_houser_2015_is_an_unmirrored_source_study_named_by_caglar_only() -> None:
    """The attribution is sourced from Caglar's citation; the paper itself is unread.

    No mirror key, so nothing re-reads Houser. The DOI is not in Caglar's citation, so
    the article number in the DOI must be the article id the citation prints, which is
    what ties the resolved record to the cited one.
    """
    houser = c.SOURCE_STUDIES["houser2015"]
    assert houser.is_mirrored is c.HOUSER2015_IS_MIRRORED is False
    assert houser.citation_key is None
    assert c.SOURCE_STUDIES["caglar2017"].citation_key == c.CITATION_KEY
    assert c.SOURCE_STUDIES["caglar2017"].is_mirrored is True
    assert houser.title is not None and houser.title in c.HOUSER2015_CITATION.quote
    assert "e1004400" in c.HOUSER2015_CITATION.quote
    assert c.HOUSER2015_DOI.rsplit(".", 1)[1] == "1004400"
    assert houser.publication.model_dump(include=set(HOUSER_PUBLICATION)) == (
        HOUSER_PUBLICATION
    )


def test_the_three_attribution_quotes_come_from_the_pinned_paper_ocr() -> None:
    """Each quote is anchored to the same sha256 the rest of the loader's values are."""
    for value in (c.HOUSER2015_DEFERRAL, c.HOUSER2015_CITATION, c.HOUSER2015_DEPOSITS):
        assert value.provenance.citation_key == c.CITATION_KEY
        assert value.provenance.sha256 == c.PAPER_MD_SHA256
    assert "presented previously10" in c.HOUSER2015_DEFERRAL.quote
    assert c.HOUSER2015_DEPOSITS.value == ("GSE67402", "PXD002140")
    assert "GSE67402 for the glucose time-course" in c.HOUSER2015_DEPOSITS.quote


@pytest.mark.parametrize("family", ["rnaseq", "proteome"])
def test_each_record_cites_the_study_that_first_reported_its_sample(
    family: str,
    rnaseq: c.RnaseqCaglar2017Dataset,
    proteome: c.ProteomeCaglar2017Dataset,
) -> None:
    """The three glucose-time-course records cite Houser 2015, the other two Caglar."""
    dataset: Any = rnaseq if family == "rnaseq" else proteome
    samples = [s["sample"] for s in _ledger(dataset, "record_samples.json")]
    by_sample = {
        sample: _record(dataset, index)["publication"]
        for index, sample in enumerate(samples)
    }
    houser = {s: p for s, p in by_sample.items() if p["doi"] == c.HOUSER2015_DOI}
    caglar = {s: p for s, p in by_sample.items() if p["doi"] == c.PAPER_DOI}
    assert sorted(houser) == ["MURI_002", "MURI_003", "MURI_004"]
    assert len(caglar) == 2
    assert set(houser) | set(caglar) == set(samples)
    one = next(iter(houser.values()))
    assert {k: one[k] for k in HOUSER_PUBLICATION} == HOUSER_PUBLICATION
    assert next(iter(caglar.values()))["pubmed_id"] is None


@pytest.mark.parametrize("family", ["rnaseq", "proteome"])
def test_the_build_writes_the_source_study_ledger(
    family: str,
    rnaseq: c.RnaseqCaglar2017Dataset,
    proteome: c.ProteomeCaglar2017Dataset,
) -> None:
    """The ledger carries the rule, both studies, and one row per sample."""
    dataset: Any = rnaseq if family == "rnaseq" else proteome
    ledger = _ledger(dataset, "source_study_attribution.json")
    assert ledger["issue"] == "771"
    assert "glucose_time_course" in ledger["rule"]
    assert ledger["samples_by_study"] == {"caglar2017": 2, "houser2015": 3}
    assert ledger["studies"]["houser2015"]["is_mirrored"] is False
    assert ledger["studies"]["houser2015"]["doi"] == c.HOUSER2015_DOI
    assert ledger["studies"]["caglar2017"]["citation_key"] == c.CITATION_KEY
    rows = {row["sample"]: row for row in ledger["samples"]}
    assert len(rows) == 5
    assert rows["MURI_002"]["study"] == "houser2015"
    assert rows["MURI_002"]["experiment"] == "glucose_time_course"
    assert rows["MURI_006"]["study"] == "caglar2017"


def test_a_released_id_outside_rel606_stops_the_build(
    genome: EcoliBREL606Genome,
) -> None:
    with pytest.raises(LocusTagResolutionError, match="1 of 2 names"):
        c.reconcile_rel606(genome, ["ECB_00001", "ECB_99999"], label="probe")
    assert c.reconcile_rel606(genome, GENES, label="probe")[0] == GENES


def test_the_verifiers_pass_the_built_records_and_catch_a_broken_scale(
    rnaseq: c.RnaseqCaglar2017Dataset, proteome: c.ProteomeCaglar2017Dataset
) -> None:
    rna = [_record(rnaseq, i) for i in range(len(rnaseq))]
    evidence = c.VstBackSolve.model_validate(_ledger(rnaseq, "vst_back_solve.json"))
    report = c.verify_rnaseq_records(
        rna, expected_count=5, gene_universe=GENES, back_solve=evidence
    )
    assert report.passed, report.summary()
    assert [r.name for r in report.results] == [
        "structural",
        "count",
        "sample_uniqueness",
        "tpm_value_fidelity",
        "count_value_fidelity",
        "tpm_scale",
        "measurement_type_consistent",
        "reference_tpm",
        "assembly_pin",
        "log_base_back_solve",
        "gene_containment_rel606",
    ]
    rna[0]["experiment"]["phenotype"]["expression_tpm"]["ECB_00001"] += 1.0
    broken = c.verify_rnaseq_records(
        rna, expected_count=5, gene_universe=GENES[:2], back_solve=evidence
    )
    failed = {r.name for r in broken.results if not r.passed}
    assert failed == {"tpm_scale", "gene_containment_rel606"}

    proteins = [_record(proteome, i) for i in range(len(proteome))]
    protein_evidence = c.VstBackSolve.model_validate(
        _ledger(proteome, "vst_back_solve.json")
    )
    protein_report = c.verify_proteome_records(
        proteins, expected_count=5, gene_universe=GENES, back_solve=protein_evidence
    )
    assert protein_report.passed, protein_report.summary()
    off = protein_evidence.model_copy(update={"max_count_integer_deviation": 0.4})
    assert not c._l3_back_solve(off).passed


def test_run_verification_writes_the_report_of_a_built_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rnaseq: c.RnaseqCaglar2017Dataset
) -> None:
    import torchcell.verification.runners as runners

    data_root = tmp_path / "vr"
    built = data_root / "data" / "torchcell"
    built.mkdir(parents=True)
    os.symlink(rnaseq.root, built / "rnaseq_caglar2017")
    monkeypatch.setattr(
        runners, "_gene_set_for_reference", lambda ref, base: set(GENES)
    )
    monkeypatch.setattr(c, "sourced_values", lambda: {})
    report = c.run_verification("rnaseq", str(data_root))
    assert report.passed, report.summary()
    written = json.loads(
        Path(rnaseq.preprocess_dir, "verification_report.json").read_text()
    )
    assert written["dataset_name"] == "rnaseq_caglar2017"


def test_main_dispatches_build_and_verify(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built: list[str] = []

    class Stub:
        def __init__(self, root: str) -> None:
            built.append(root)

        def __len__(self) -> int:
            return 7

    monkeypatch.setattr(c, "DATASET_CLASSES", {"rnaseq": Stub, "proteome": Stub})
    monkeypatch.setattr(c, "_data_root", lambda: "/dr")
    assert c.main(["build", "--family", "proteome"]) == 0
    assert built == ["/dr/data/torchcell/proteome_caglar2017"]
    assert "Stub: len = 7" in capsys.readouterr().out

    report = VerificationReport(
        dataset_name="rnaseq_caglar2017", provenance=Provenance(source_uri="x")
    )
    report.add(LevelResult(level=Level.L0, name="n", passed=False, message="m"))
    monkeypatch.setattr(c, "run_verification", lambda family: report)
    assert c.main(["verify", "--family", "rnaseq"]) == 1


#: Module-level SourcedValues: 26 from the raw-mirror branch, 19 the loaders add, the
#: 3 Houser 2015 attribution quotes (#771), and the 8 the Table S8 protein fold-change
#: loader adds (#770).
SOURCED_VALUE_COUNT = 26 + 19 + 3 + 8


def test_every_sourced_value_is_collected_by_name() -> None:
    values = c.sourced_values()
    assert values["LOG_TRANSFORMED"] is c.LOG_TRANSFORMED
    assert values["SIZE_FACTOR_PSEUDOCOUNT"].value == 1
    assert len(values) == SOURCED_VALUE_COUNT


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
    assert len(values) == SOURCED_VALUE_COUNT
    results = {value.quote: audit_sourced_value(value, library) for value in values}
    assert {quote: r.message for quote, r in results.items() if not r.passed} == {}


@pytest.mark.data
@_ON_DISK
def test_the_raw_mirror_holds_every_pinned_file_intact() -> None:
    """Every file this module pins is in the manifest at its pin and on disk at its pin.

    CONTAINMENT, not whole-mirror equality: one citation key's raw mirror is written by
    several loaders (``deposit_si_table``), and a table a sibling module deposited for a
    phenotype this module does not build is a legitimate extension, not drift. What
    would be drift is a pinned record missing, or present with other bytes, and that is
    what this asserts. The manifest's own order is checked for the pinned files only.
    """
    root = c.raw_mirror_dir(str(_data_root()))
    manifest = c.load_manifest(str(_data_root()))
    specs = c.raw_file_specs(c.read_table_ids(root / c.si_table_relpath("S3")))
    assert len(specs) == 47
    recorded = {r.path: r.sha256 for r in manifest.files}
    assert {s.relpath: s.sha256 for s in specs}.items() <= recorded.items()
    pinned_order = [
        r.path for r in manifest.files if r.path in {s.relpath for s in specs}
    ]
    assert pinned_order == [s.relpath for s in specs]
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


def _sheet_rows(assay: str) -> list[c.SampleRow]:
    root = c.raw_mirror_dir(str(_data_root()))
    sheet = c.read_sample_sheet(root / c.si_table_relpath("S1"))
    if assay == "S2":
        return [r for r in sheet if r.rna_technical_replicates > 0]
    return [r for r in sheet if r.protein_technical_replicates > 0]


@pytest.mark.data
@_ON_DISK
@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            "S2",
            {
                "n_samples": 152,
                "floor_value": -1.14456265985177,
                "zero_cells": 185331,
                "samples_with_a_zero": 152,
                "library_size_min": 3271,
                "library_size_median": 177377.0,
                "library_size_max": 5005764,
            },
        ),
        (
            "S3",
            {
                "n_samples": 105,
                "floor_value": 0.968021139095309,
                "zero_cells": 203214,
                "samples_with_a_zero": 105,
                "library_size_min": 1107,
                "library_size_median": 64241.0,
                "library_size_max": 198696,
            },
        ),
    ],
)
def test_the_released_tables_invert_to_integer_counts_in_base_two(
    table: str, expected: dict[str, Any]
) -> None:
    """The measured back-solve: every cell of Tables S2 and S3 is the base-2 DESeq2 VST
    of an integer count, and the per-sample factors are DESeq2's +1 size factors.
    """
    root = c.raw_mirror_dir(str(_data_root()))
    rows = _sheet_rows(table)
    released = c.read_released_table(root / c.si_table_relpath(table), rows)
    evidence = c.back_solve_counts(released, label=table).evidence
    assert evidence.model_dump(include=set(expected)) == expected
    assert evidence.n_features == 4196
    assert evidence.max_count_integer_deviation < 1e-8
    assert evidence.max_size_factor_relative_deviation < 1e-12
    as_ln = released * np.log(2.0)
    with pytest.raises(RuntimeError, match="from an integer"):
        c.back_solve_counts(as_ln, label=f"{table} read as ln")


@pytest.mark.data
@_ON_DISK
@pytest.mark.parametrize(
    ("table", "phase", "carbon", "mg", "na"),
    [
        (
            "S2",
            {"exponential": 79, "stationary": 63, "late_stationary": 10},
            {"glucose": 115, "glycerol": 25, "lactate": 6, "gluconate": 6},
            {"lowMg": 36, "baseMg": 92, "highMg": 24},
            {"baseNa": 136, "highNa": 16},
        ),
        (
            "S3",
            {"exponential": 56, "stationary": 37, "late_stationary": 12},
            {"glucose": 66, "glycerol": 27, "lactate": 6, "gluconate": 6},
            {"lowMg": 6, "baseMg": 87, "highMg": 12},
            {"baseNa": 94, "highNa": 11},
        ),
    ],
)
def test_the_sample_sheet_reproduces_the_paper_s_table_1_sample_counts(
    table: str,
    phase: dict[str, int],
    carbon: dict[str, int],
    mg: dict[str, int],
    na: dict[str, int],
) -> None:
    """The '# samples' column of the paper's Table 1, mRNA and protein halves."""
    rows = _sheet_rows(table)
    assert dict(Counter(r.growth_phase.value for r in rows)) == phase
    assert dict(Counter(r.carbon_source for r in rows)) == carbon
    assert dict(Counter(r.mg_level for r in rows)) == mg
    assert dict(Counter(r.na_level for r in rows)) == na


_BUILT = pytest.mark.skipif(
    not all(
        (_DATA_ROOT / "data/torchcell" / slug / "processed/lmdb").is_dir()
        for slug in c.DATASET_SLUGS.values()
    ),
    reason="the Caglar 2017 dev-tree LMDBs are not built under $DATA_ROOT",
)


@pytest.mark.data
@_BUILT
@pytest.mark.parametrize(
    ("family", "records", "references"),
    [
        ("rnaseq", 152, {"exponential": 21, "stationary": 12, "late_stationary": 6}),
        ("proteome", 105, {"exponential": 20, "stationary": 11, "late_stationary": 6}),
    ],
)
def test_the_dev_tree_builds_pin_the_measured_counts_and_pass_l0_to_l4(
    family: c.Family, records: int, references: dict[str, int]
) -> None:
    from torchcell.verification.runners import _gene_set_for_reference, load_records

    root = _data_root() / "data/torchcell" / c.DATASET_SLUGS[family]
    accounting = c.BuildAccounting.model_validate_json(
        (root / "preprocess/build_accounting.json").read_text()
    )
    assert (accounting.kept_records, accounting.dropped_records) == (records, 0)
    assert {k: len(v) for k, v in accounting.reference_samples.items()} == references
    reconciliation = json.loads(
        (root / "preprocess/locus_tag_reconciliation.json").read_text()
    )
    assert reconciliation["status_histogram"]["current"] == 4196
    assert reconciliation["remapped"] == 0
    built = load_records(str(root))
    universe = _gene_set_for_reference(
        built[0]["reference"]["genome_reference"], str(_data_root())
    )
    evidence = c.VstBackSolve.model_validate_json(
        (root / "preprocess/vst_back_solve.json").read_text()
    )
    verify = (
        c.verify_rnaseq_records if family == "rnaseq" else c.verify_proteome_records
    )
    report = verify(
        built, expected_count=records, gene_universe=universe, back_solve=evidence
    )
    assert report.passed, report.summary()


# --------------------------------------------------------------------------- #
# Table S8: the protein fold changes
# --------------------------------------------------------------------------- #
#: The protein fold changes of the synthetic mirror, by Table S8 group. ``YP_3.1`` is
#: blank in the lowMg group (DESeq2 writes no fold change where the base mean is 0).
S8_LOW_MG = [("YP_1.1", 1.5), ("YP_2.1", -0.75), ("YP_3.1", float("nan"))]
S8_HIGH_NA = [("YP_1.1", -2.0), ("YP_2.1", 0.5), ("YP_3.1", 3.25)]
#: Table S8 carries both data types, so the synthetic table does too: one gene-level
#: group, which the loader counts and refuses as out of scope.
S8_MRNA_GROUP = _s8_group(
    [(gene, 1.0) for gene in GENES],
    test_vs_base="lowMgVSbaseMg",
    phase="Exp",
    effect="batchNumberPLUSMg",
    test_for="Mg_mM_Levels",
    contrast="lowMg",
    base="baseMg",
    data_type="mrna",
)
#: Four groups: the lowMg contrast under both control models, the highNa contrast under
#: the primary one, and the gene-level group. The doubling-time group is a refusal.
S8_GROUPS = [
    _s8_group(
        S8_LOW_MG,
        test_vs_base="lowMgVSbaseMg",
        phase="Exp",
        effect="batchNumberPLUSMg",
        test_for="Mg_mM_Levels",
        contrast="lowMg",
        base="baseMg",
    ),
    _s8_group(
        S8_LOW_MG,
        test_vs_base="lowMgVSbaseMg",
        phase="Exp",
        effect="batchNumberPLUSMgPLUSdoublingTimeMinutes",
        test_for="Mg_mM_Levels",
        contrast="lowMg",
        base="baseMg",
    ),
    _s8_group(
        S8_HIGH_NA,
        test_vs_base="highNaVSbaseNa",
        phase="Sta",
        effect="batchNumberPLUSNa",
        test_for="Na_mM_Levels",
        contrast="highNa",
        base="baseNa",
    ),
    S8_MRNA_GROUP,
]


def test_benjamini_hochberg_is_the_step_up_adjustment_capped_at_one() -> None:
    # n = 4: p * 4 / rank is [0.04, 0.04, 0.0533..., 1.6], made monotone from the
    # right and capped, so the first two share the running minimum 0.04.
    adjusted = c.benjamini_hochberg(np.array([0.01, 0.02, 0.04, 0.40]))
    assert list(adjusted) == pytest.approx([0.04, 0.04, 0.05333333333333334, 0.4])
    # n = 2: [1.8, 0.95] made monotone from the right is [0.95, 0.95]; the cap at 1
    # never bites here, which is why the larger raw p-value sets both.
    assert list(c.benjamini_hochberg(np.array([0.9, 0.95]))) == pytest.approx(
        [0.95, 0.95]
    )
    assert list(c.benjamini_hochberg(np.array([1.0, 1.0]))) == pytest.approx([1.0, 1.0])
    # Order does not matter: the adjustment travels back to its own row, so the same
    # four p-values shuffled give the same four adjusted values, shuffled with them.
    assert list(c.benjamini_hochberg(np.array([0.04, 0.40, 0.01, 0.02]))) == (
        pytest.approx([0.05333333333333334, 0.4, 0.04, 0.04])
    )


def test_adjustment_back_solve_identifies_the_correction_and_refuses_another(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s8.csv"
    path.write_bytes(_s8_file(S8_GROUPS))
    protein = c.read_table_s8_protein(path)
    evidence = c.adjustment_back_solve(protein)
    assert evidence.method == "benjamini_hochberg"
    assert (evidence.groups, evidence.rows_with_both) == (3, 9)
    assert evidence.max_abs_deviation < 1e-15

    bonferroni = _s8_group(
        S8_HIGH_NA,
        test_vs_base="highNaVSbaseNa",
        phase="Sta",
        effect="batchNumberPLUSNa",
        test_for="Na_mM_Levels",
        contrast="highNa",
        base="baseNa",
        p_values=[0.01, 0.02, 0.04],
        adjusted=[0.03, 0.06, 0.12],
    )
    path.write_bytes(_s8_file([bonferroni, S8_MRNA_GROUP]))
    with pytest.raises(
        RuntimeError, match="the released correction is not Benjamini-Hochberg"
    ):
        c.adjustment_back_solve(c.read_table_s8_protein(path))


def test_read_table_s8_protein_keeps_the_protein_rows_and_refuses_an_unknown_cell(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s8.csv"
    path.write_bytes(_s8_file(S8_GROUPS))
    protein = c.read_table_s8_protein(path)
    assert len(protein) == 9
    assert set(protein["dataType"]) == {"protein"}
    assert sorted(set(protein.columns)) == sorted(c.S8_COLUMNS)

    only_protein = tmp_path / "one_type.csv"
    only_protein.write_bytes(_s8_file(S8_GROUPS[:3]))
    with pytest.raises(RuntimeError, match=r"dataType values \['protein'\]"):
        c.read_table_s8_protein(only_protein)

    unknown_phase = tmp_path / "phase.csv"
    unknown_phase.write_bytes(
        _s8_file(
            [
                S8_MRNA_GROUP,
                _s8_group(
                    S8_HIGH_NA,
                    test_vs_base="highNaVSbaseNa",
                    phase="LateSta",
                    effect="batchNumberPLUSNa",
                    test_for="Na_mM_Levels",
                    contrast="highNa",
                    base="baseNa",
                ),
            ]
        )
    )
    with pytest.raises(RuntimeError, match=r"growth phases \['LateSta'\]"):
        c.read_table_s8_protein(unknown_phase)


def test_contrast_groups_read_the_released_descriptors_and_the_control_model(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s8.csv"
    path.write_bytes(_s8_file(S8_GROUPS))
    groups = c.contrast_groups(c.read_table_s8_protein(path))
    assert [g.file_name for g in groups] == sorted(name for name, _ in S8_GROUPS[:3])
    # Sorted by file name, and "PLUSdoublingTimeMinutes__" sorts before "__" ('P' <
    # '_'), so the secondary control model of the lowMg contrast comes first.
    assert [g.primary_control_model for g in groups] == [False, True, True]
    primary = next(g for g in groups if g.test_vs_base == "highNaVSbaseNa")
    assert primary.model_dump() == {
        "file_name": "resDf_protein_set00_Sta_batchNumberPLUSNa__highNaVSbaseNa.csv",
        "test_vs_base": "highNaVSbaseNa",
        "growth_phase": c.GrowthPhase.stationary,
        "test_for": "Na_mM_Levels",
        "contrast": "highNa",
        "base": "baseNa",
        "investigated_effect": "batchNumberPLUSNa",
        "rows": 3,
    }


def test_contrast_groups_refuse_a_group_whose_descriptors_vary(tmp_path: Path) -> None:
    name, rows = S8_GROUPS[0]
    mutated = [list(row) for row in rows]
    mutated[1][S8_HEADER.index("contrast")] = "highMg"
    path = tmp_path / "s8.csv"
    path.write_bytes(_s8_file([(name, mutated), S8_MRNA_GROUP]))
    with pytest.raises(RuntimeError, match=r"descriptor columns vary \{'contrast'"):
        c.contrast_groups(c.read_table_s8_protein(path))


def test_table_s8_protein_order_refuses_a_group_listing_the_ids_otherwise(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s8.csv"
    path.write_bytes(_s8_file(S8_GROUPS))
    protein = c.read_table_s8_protein(path)
    groups = c.contrast_groups(protein)
    assert c.table_s8_protein_order(protein, groups) == PROTEINS

    reversed_group = _s8_group(
        list(reversed(S8_HIGH_NA)),
        test_vs_base="highNaVSbaseNa",
        phase="Sta",
        effect="batchNumberPLUSNa",
        test_for="Na_mM_Levels",
        contrast="highNa",
        base="baseNa",
    )
    path.write_bytes(_s8_file([S8_GROUPS[0], reversed_group, S8_MRNA_GROUP]))
    protein = c.read_table_s8_protein(path)
    with pytest.raises(RuntimeError, match="list the ids in another order"):
        c.table_s8_protein_order(protein, c.contrast_groups(protein))


def _group(**overrides: Any) -> c.ContrastGroup:
    cells: dict[str, Any] = {
        "file_name": "probe.csv",
        "test_vs_base": "lowMgVSbaseMg",
        "growth_phase": c.GrowthPhase.exponential,
        "test_for": "Mg_mM_Levels",
        "contrast": "lowMg",
        "base": "baseMg",
        "investigated_effect": "batchNumberPLUSMg",
        "rows": 3,
    }
    cells.update(overrides)
    return c.ContrastGroup(**cells)


def test_contrast_samples_select_the_two_groups_table_s1_states(tmp_path: Path) -> None:
    rows = [r for r in _rows(tmp_path / "s1.csv") if r.protein_technical_replicates > 0]
    low_mg = c.contrast_samples(rows, _group())
    assert low_mg.model_dump() == {
        "test_samples": ("MURI_006",),
        "base_samples": ("MURI_002", "MURI_003"),
        "test_times_hr": (5.0,),
        "base_times_hr": (3.0, 4.0),
        "doses": (0.005,),
    }
    high_na = c.contrast_samples(
        rows,
        _group(
            test_vs_base="highNaVSbaseNa",
            growth_phase=c.GrowthPhase.stationary,
            test_for="Na_mM_Levels",
            contrast="highNa",
            base="baseNa",
        ),
    )
    assert (high_na.test_samples, high_na.base_samples, high_na.doses) == (
        ("MURI_007",),
        ("MURI_004",),
        (200.0,),
    )
    # The carbon-source contrast has RNA data only: lactate's one sample is not a
    # protein sample, so the contrast has no test group and the build stops.
    with pytest.raises(RuntimeError, match="puts 0 samples in the lactate group"):
        c.contrast_samples(
            rows,
            _group(
                test_vs_base="lactateVSglucose",
                test_for="carbonSource",
                contrast="lactate",
                base="glucose",
            ),
        )


def test_a_level_holding_several_doses_is_refused_with_the_doses_it_measured(
    tmp_path: Path,
) -> None:
    sheet = [
        *S1_ROWS,
        ("MURI_008", "NaCl_stress", "30", "0", "1", "14", "glucose", "0.8", "300",
         "stationary", "baseMg", "highNa", "unique_condition_28"),
    ]  # fmt: skip
    rows = [
        r
        for r in _rows(tmp_path / "s1.csv", sheet)
        if r.protein_technical_replicates > 0
    ]
    group = _group(
        test_vs_base="highNaVSbaseNa",
        growth_phase=c.GrowthPhase.stationary,
        test_for="Na_mM_Levels",
        contrast="highNa",
        base="baseNa",
        rows=3,
    )
    samples = c.contrast_samples(rows, group)
    assert samples.doses == (200.0, 300.0)
    refusal = c.refuse_pooled_dose(group, samples)
    assert refusal.reason == "level pools several doses"
    assert refusal.rows == 3
    assert refusal.measurement == (
        "Table S1 puts 2 protein samples in the highNa group at na_mm ['200', '300']; "
        "SmallMoleculePerturbation requires one Concentration and the released cell is "
        "a level, not a dose"
    )
    with pytest.raises(RuntimeError, match=r"probe\.csv: na_mm \[200\.0, 300\.0\]"):
        c.contrast_environment(group, samples)


def test_refuse_control_model_names_the_primary_design_it_duplicates() -> None:
    refusal = c.refuse_control_model(
        _group(
            investigated_effect="batchNumberPLUSMgPLUSdoublingTimeMinutes", rows=4196
        )
    )
    assert (refusal.reason, refusal.rows) == ("secondary control model", 4196)
    assert "equal those of batchNumberPLUSMg," in refusal.measurement


def test_contrast_environment_is_the_test_level_with_no_single_duration(
    tmp_path: Path,
) -> None:
    rows = [r for r in _rows(tmp_path / "s1.csv") if r.protein_technical_replicates > 0]
    group = _group()
    environment = c.contrast_environment(group, c.contrast_samples(rows, group))
    assert environment.media == DM500
    assert environment.temperature == Temperature(value=37.0)
    assert environment.duration_hours is None
    magnesium = environment.perturbations[0]
    assert isinstance(magnesium, SmallMoleculePerturbation)
    assert magnesium.compound.name == "magnesium sulfate"
    assert magnesium.concentration.value == 0.005
    gap = environment.provenance_gaps[0]
    assert gap.field == "duration_hours"
    assert gap.note == (
        "the exponential lowMg group of lowMgVSbaseMg pools samples Table S1 dates at "
        "5 h; the paper sets the phase by optical density, so no single duration "
        "describes the group"
    )
    carbon = _group(
        test_vs_base="glycerolVSglucose",
        test_for="carbonSource",
        contrast="glycerol",
        base="glucose",
    )
    swapped = c.contrast_environment(
        carbon,
        c.ContrastSamples(
            test_samples=("MURI_002",),
            base_samples=("MURI_003",),
            test_times_hr=(3.0,),
            base_times_hr=(4.0,),
            doses=(),
        ),
    )
    assert swapped.media == DAVIS_MINIMAL
    glycerol = swapped.perturbations[0]
    assert isinstance(glycerol, EnvironmentPhysicalPerturbation)
    assert glycerol.factor is PhysicalFactor.carbon_source
    assert glycerol.magnitude == Concentration(
        value=0.5, unit=ConcentrationUnit.g_per_l
    )


def _values(
    fold: Sequence[tuple[str, float]], **columns: Sequence[float]
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "log2FoldChange": [value for _, value in fold],
            "lfcSE": [0.25] * len(fold),
            "pvalue": [0.01 * (i + 1) for i in range(len(fold))],
            "padj": [0.04 * (i + 1) for i in range(len(fold))],
        },
        index=[name for name, _ in fold],
    )
    for name, column in columns.items():
        frame[name] = list(column)
    return frame


def test_a_phenotype_drops_the_blank_rows_and_nests_the_statistic_maps() -> None:
    nan = float("nan")
    values = _values(
        [("a", 1.5), ("b", nan), ("c", -0.75)],
        pvalue=[0.01, 0.02, nan],
        padj=[0.04, 0.05, nan],
    )
    phenotype = c.protein_fold_change_phenotype(GENES, values, 3)
    assert phenotype.protein_fold_change == {"ECB_00001": 1.5, "ECB_00003": -0.75}
    assert phenotype.protein_fold_change_se == {"ECB_00001": 0.25, "ECB_00003": 0.25}
    # ECB_00002 is blank, so it is not a key anywhere; ECB_00003 has a fold change but
    # no test result, so it is a key in neither p-value map.
    assert phenotype.protein_fold_change_p_value == {"ECB_00001": 0.01}
    assert phenotype.protein_fold_change_p_value_adjusted == {"ECB_00001": 0.04}
    assert phenotype.p_value_adjustment_method == "benjamini_hochberg"
    assert phenotype.fold_change_scale is FoldChangeScale.log2
    assert phenotype.n_replicates == {"ECB_00001": 3, "ECB_00003": 3}
    assert phenotype.measurement_type == (
        "deseq2_wald_log2_fold_change_design_batch_plus_condition"
    )
    assert phenotype.reference_basis == (
        "the base level reference condition of the same growth phase: glucose with "
        "5 mM Na+ and 0.8 mM Mg2+ on Davis Minimal medium (DM500)"
    )
    with pytest.raises(RuntimeError, match="the group carries no fold change"):
        c.protein_fold_change_phenotype(
            GENES, _values([("a", nan), ("b", nan), ("c", nan)]), 3
        )


def test_the_reference_phenotype_is_the_log2_scales_neutral_value() -> None:
    phenotype = c.protein_fold_change_phenotype(
        GENES, _values([("a", 1.5), ("b", -2.0), ("c", 0.5)]), 3
    )
    reference = c.fold_change_reference_phenotype(phenotype, 11)
    assert reference.protein_fold_change == dict.fromkeys(GENES, 0.0)
    assert reference.fold_change_scale is FoldChangeScale.log2
    assert reference.n_replicates == dict.fromkeys(GENES, 11)
    assert reference.protein_fold_change_se is None
    assert reference.protein_fold_change_p_value is None
    assert reference.p_value_adjustment_method is None
    assert reference.measurement_type == phenotype.measurement_type


def test_replicate_derivations_take_the_conservative_low_end_of_the_two_groups() -> (
    None
):
    derivations = c.replicate_derivations(
        [
            {"file_name": "a.csv", "n_test_samples": 3, "n_base_samples": 20},
            {"file_name": "b.csv", "n_test_samples": 15, "n_base_samples": 20},
        ]
    )
    assert [d.field for d in derivations] == [
        "n_replicates[a.csv]",
        "n_replicates[b.csv]",
    ]
    assert [(d.value, d.range_low, d.range_high) for d in derivations] == [
        (3.0, 3.0, 23.0),
        (15.0, 15.0, 35.0),
    ]
    assert {d.method for d in derivations} == {DerivationMethod.conservative_low}
    assert derivations[0].diagnostics == {"n_test_samples": 3.0, "n_base_samples": 20.0}
    assert derivations[0].provenance is not None
    assert derivations[0].provenance.sha256 == c.SI_TABLES["S8"][1]


def test_deposit_si_table_appends_one_record_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _stage(tmp_path / "staging")
    _pin_to(monkeypatch, files)
    monkeypatch.setattr(c, "datetime", _FrozenDatetime)
    root = c.deposit_raw_mirror(
        source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )
    manifest = tmp_path / "dr" / "torchcell-raw" / c.CITATION_KEY / "manifest.json"
    # Start from a mirror that does NOT hold Table S8, as the first deposit left it.
    trimmed = Manifest.model_validate_json(manifest.read_text())
    trimmed.files = [f for f in trimmed.files if f.path != "data/srep45303-s9.csv"]
    manifest.write_text(trimmed.model_dump_json())
    (root / "data" / "srep45303-s9.csv").unlink()

    dest = c.deposit_si_table(
        "S8", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )
    assert dest == root / "data" / "srep45303-s9.csv"
    assert dest.read_bytes() == files["data/srep45303-s9.csv"]
    written = Manifest.model_validate_json(manifest.read_text())
    assert written.created_at == "2026-10-07T12:00:00+00:00"
    assert [f.path for f in written.files] == [
        "data/srep45303-s2.csv",
        "data/srep45303-s3.csv",
        "data/srep45303-s4.csv",
        "data/srep45303-s5.csv",
        "data/srep45303-s9.csv",
        "ncbi_protein/yp_batch_00.gp",
    ]
    record = next(f for f in written.files if f.path == "data/srep45303-s9.csv")
    assert record.bytes == len(files["data/srep45303-s9.csv"])
    assert record.retrieval is not None
    assert record.retrieval.params == {"key": "PMC5394689.1/srep45303-s9.csv"}

    frozen = manifest.read_bytes()
    c.deposit_si_table(
        "S8", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )
    assert manifest.read_bytes() == frozen


def test_deposit_si_table_refuses_a_record_of_other_content_and_a_missing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _stage(tmp_path / "staging")
    _pin_to(monkeypatch, files)
    with pytest.raises(
        RuntimeError, match="does not exist; run the full deposit first"
    ):
        c.deposit_si_table(
            "S8", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
        )
    monkeypatch.setattr(c, "datetime", _FrozenDatetime)
    c.deposit_raw_mirror(
        source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
    )
    manifest = tmp_path / "dr" / "torchcell-raw" / c.CITATION_KEY / "manifest.json"
    drifted = Manifest.model_validate_json(manifest.read_text())
    for record in drifted.files:
        if record.path == "data/srep45303-s9.csv":
            record.bytes += 1
    manifest.write_text(drifted.model_dump_json())
    with pytest.raises(
        RuntimeError,
        match=r"records data/srep45303-s9\.csv differently; refusing to overwrite",
    ):
        c.deposit_si_table(
            "S8", source_dir=tmp_path / "staging", data_root=str(tmp_path / "dr")
        )


@pytest.fixture
def fold_change(
    tmp_path: Path, mirror: Path, genome: EcoliBREL606Genome
) -> c.ProteinFoldChangeCaglar2017Dataset:
    return c.ProteinFoldChangeCaglar2017Dataset(
        root=str(tmp_path / "protein_fold_change_caglar2017"), ecoli_genome=genome
    )


def test_the_fold_change_build_keeps_the_primary_control_model_only(
    fold_change: c.ProteinFoldChangeCaglar2017Dataset,
) -> None:
    assert len(fold_change) == 2
    assert sorted(fold_change.gene_set) == GENES
    accounting = _ledger(fold_change, "build_accounting.json")
    assert accounting["candidate_groups"] == 3
    assert accounting["kept_records"] == 2
    assert accounting["refused_groups"] == 1
    assert accounting["refused_rows"] == 3
    assert accounting["refusals_by_reason"] == {"secondary control model": 1}
    assert accounting["kept_by_growth_phase"] == {"exponential": 1, "stationary": 1}
    assert accounting["kept_by_test_for"] == {"Mg_mM_Levels": 1, "Na_mM_Levels": 1}
    assert accounting["table_rows"] == 12
    assert accounting["mrna_rows"] == 3
    assert accounting["protein_rows"] == 9
    assert [r["file_name"] for r in accounting["refused"]] == [
        "resDf_protein_set00_Exp_batchNumberPLUSMgPLUSdoublingTimeMinutes__"
        "lowMgVSbaseMg.csv"
    ]

    records = _ledger(fold_change, "fold_change_records.json")
    assert [(r["contrast"], r["growth_phase"]) for r in records] == [
        ("lowMg", "exponential"),
        ("highNa", "stationary"),
    ]
    assert [(r["n_test_samples"], r["n_base_samples"]) for r in records] == [
        (1, 2),
        (1, 1),
    ]
    # #771 attributes a SAMPLE; a contrast is not a sample, so the record cites THIS
    # paper and the attribution is measured per group instead. The fixture reproduces
    # the release's pattern: no test sample is Houser's, every base sample here is.
    assert [r["houser2015_test_samples"] for r in records] == [[], []]
    assert [r["houser2015_base_samples"] for r in records] == [
        ["MURI_002", "MURI_003"],
        ["MURI_004"],
    ]
    assert [_record(fold_change, i)["publication"]["doi"] for i in (0, 1)] == [
        c.PAPER_DOI,
        c.PAPER_DOI,
    ]
    # The lowMg group leaves YP_3.1 blank, so its record carries two of three loci.
    assert [r["n_fold_change"] for r in records] == [2, 3]
    assert [r["n_p_value_adjusted"] for r in records] == [2, 3]

    low_mg = _record(fold_change, 0)
    phenotype = low_mg["experiment"]["phenotype"]
    assert phenotype["protein_fold_change"] == {"ECB_00001": 1.5, "ECB_00002": -0.75}
    assert phenotype["fold_change_scale"] == "log2"
    assert phenotype["n_replicates"] == {"ECB_00001": 1, "ECB_00002": 1}
    assert low_mg["reference"]["phenotype_reference"]["protein_fold_change"] == {
        "ECB_00001": 0.0,
        "ECB_00002": 0.0,
    }
    assert low_mg["reference"]["phenotype_reference"]["n_replicates"] == {
        "ECB_00001": 2,
        "ECB_00002": 2,
    }
    assert low_mg["experiment"]["genotype"]["perturbations"] == []
    assert [
        p["compound"]["name"]
        for p in low_mg["experiment"]["environment"]["perturbations"]
    ] == ["magnesium sulfate"]
    assert low_mg["reference"]["environment_reference"]["media"]["name"] == DM500.name

    high_na = _record(fold_change, 1)["experiment"]
    assert high_na["phenotype"]["protein_fold_change"] == {
        "ECB_00001": -2.0,
        "ECB_00002": 0.5,
        "ECB_00003": 3.25,
    }
    assert [p["compound"]["name"] for p in high_na["environment"]["perturbations"]] == [
        "sodium chloride"
    ]
    assert high_na["environment"]["perturbations"][0]["concentration"]["value"] == 195.0
    assert _ledger(fold_change, "adjustment_back_solve.json")["groups"] == 3
    assert [
        d["method"] for d in _ledger(fold_change, "n_replicates_derivation.json")
    ] == ["conservative_low", "conservative_low"]


def test_the_fold_change_verifier_passes_the_records_and_catches_a_broken_reference(
    fold_change: c.ProteinFoldChangeCaglar2017Dataset,
) -> None:
    records = [_record(fold_change, i) for i in range(len(fold_change))]
    evidence = c.AdjustmentBackSolve.model_validate(
        _ledger(fold_change, "adjustment_back_solve.json")
    )
    report = c.verify_protein_fold_change_records(
        records, expected_count=2, gene_universe=GENES, adjustment=evidence
    )
    assert report.passed, report.summary()
    assert [r.name for r in report.results] == [
        "structural",
        "count",
        "contrast_uniqueness",
        "value_fidelity",
        "p_values_are_probabilities",
        "reference_is_the_scales_neutral_value",
        "fold_change_scale_consistent",
        "measurement_type_consistent",
        "contrast_environment_uniqueness",
        "sample_uniqueness",
        "fold_change_se_nonnegative",
        "adjusted_p_values_are_probabilities",
        "statistic_keys_are_nested",
        "p_value_adjustment_method_consistent",
        "assembly_pin",
        "p_value_adjustment_back_solve",
        "gene_containment_rel606",
    ]

    records[0]["reference"]["phenotype_reference"]["protein_fold_change"][
        "ECB_00001"
    ] = 1.0
    records[1]["experiment"]["phenotype"]["p_value_adjustment_method"] = "bonferroni"
    broken = c.verify_protein_fold_change_records(
        records,
        expected_count=2,
        gene_universe=GENES[:1],
        adjustment=evidence.model_copy(update={"max_abs_deviation": 0.5}),
    )
    assert {r.name for r in broken.results if not r.passed} == {
        "reference_is_the_scales_neutral_value",
        "p_value_adjustment_method_consistent",
        "p_value_adjustment_back_solve",
        "gene_containment_rel606",
    }
