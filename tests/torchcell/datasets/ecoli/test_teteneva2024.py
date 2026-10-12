# tests/torchcell/datasets/ecoli/test_teteneva2024.py
# [[tests.torchcell.datasets.ecoli.test_teteneva2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_teteneva2024.py

"""The two gates of the Teteneva 2024 row, pinned where they stand.

Gate 1 (the host) is closed: the hermetic tests check the deposited W3110 set's record
against the registry id and the schema's vocabularies, and the data-gated tests re-hash
all ten members and re-derive the annotation finding off the deposited bytes, so a
member that changes, or a parser that starts accepting a flat file with no ``locus_tag``,
fails by name.

Gate 2 (the paper) is half open: the owner's SI deposit is in the raw mirror and is
recorded and measured here (hermetically on a synthetic Table S4, and on the deposited
bytes behind ``--data``), while the paper text is not yet in the library mirror. One
data-gated test is a TRIPWIRE that fails the night ``lit_sync`` mirrors the filed paper,
with the message naming the work that then becomes possible.
"""

import gzip
import os
import os.path as osp
import re
from pathlib import Path

import openpyxl
import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from torchcell.datamodels.schema import ASSEMBLY_SET_ACCESSIONS, BACTERIAL_ASSEMBLY_SETS
from torchcell.datasets.ecoli import teteneva2024 as te
from torchcell.literature.manual_deposit import verify_deposit
from torchcell.sequence.genome.registry import (
    ECOLI_K12_W3110,
    resolve,
    verify_assembly_set,
)
from torchcell.verification.sourced import ProvenanceGapReason

DATA_ROOT = os.environ.get("DATA_ROOT", "")
TIER_MANIFEST = osp.join(
    DATA_ROOT, "torchcell-genomes", ECOLI_K12_W3110, "manifest.json"
)
requires_tier = pytest.mark.skipif(
    not osp.isfile(TIER_MANIFEST),
    reason=f"requires {ECOLI_K12_W3110} under $DATA_ROOT/torchcell-genomes",
)
requires_mirrors = pytest.mark.skipif(
    not osp.isdir(osp.join(DATA_ROOT, "torchcell-library")),
    reason="requires $DATA_ROOT/torchcell-library and $DATA_ROOT/torchcell-raw",
)

SHA256 = re.compile(r"^[0-9a-f]{64}$")
MD5 = re.compile(r"^[0-9a-f]{32}$")


# --------------------------------------------------------------------------------------
# Gate 1: the deposited set, hermetically
# --------------------------------------------------------------------------------------
def test_the_deposit_record_names_the_registry_set_and_the_accession_pair() -> None:
    """The record's set id is the registry constant and its pair is GCA .1 / GCF .2."""
    deposit = te.W3110_TIER_DEPOSIT
    assert deposit.assembly_set == ECOLI_K12_W3110 == "ecoli_K12_W3110_ASM1024v1"
    assert deposit.genbank_accession == "GCA_000010245.1"
    assert deposit.refseq_accession == "GCF_000010245.2"
    assert deposit.genbank_replicon == "AP009048.1"
    assert deposit.refseq_replicon == "NC_007779.1"
    assert deposit.replicon_length_bp == 4_646_332
    assert deposit.assembly_name == "ASM1024v1"
    assert deposit.deposited_at == "2026-10-09"


def test_the_deposit_lists_ten_distinct_members_with_both_digests() -> None:
    """Six GCA members, four GCF members, every md5 and sha256 well formed."""
    members = te.W3110_TIER_DEPOSIT.members
    assert len(members) == 10
    assert len({m.path for m in members}) == 10
    assert [m.path.split("_")[0] for m in members] == ["GCA"] * 6 + ["GCF"] * 4
    assert [m.role for m in members] == [
        "annotation",
        "sequence",
        "annotation",
        "sequence",
        "index",
        "index",
        "annotation",
        "annotation",
        "annotation",
        "index",
    ]
    for member in members:
        assert MD5.match(member.md5), member.path
        assert SHA256.match(member.sha256), member.path
        assert member.bytes > 0


def test_the_gcf_assembly_report_is_a_member_because_the_gca_one_is_stale() -> None:
    """Only this set deposits the GCF report, and the record says why."""
    paths = {m.path for m in te.W3110_TIER_DEPOSIT.members}
    assert "GCF_000010245.2_ASM1024v1_assembly_report.txt" in paths
    assert "GCA_000010245.1_ASM1024v1_assembly_report.txt" in paths
    assert te.STALE_GCA_REFSEQ_ACCESSION == "GCF_000010245.1"
    assert te.STALE_GCA_REFSEQ_REPLICON == "AC_000091.1"
    assert te.STALE_GCA_REFSEQ_ACCESSION != te.W3110_TIER_DEPOSIT.refseq_accession


def test_the_set_is_in_the_schema_vocabulary_with_the_gcf_report_s_pair() -> None:
    """W3110 is a reference strain (#910) whose set pairs GCA .1 with GCF .2."""
    assert BACTERIAL_ASSEMBLY_SETS["W3110"] == ECOLI_K12_W3110
    assert ASSEMBLY_SET_ACCESSIONS[ECOLI_K12_W3110] == (
        te.W3110_TIER_DEPOSIT.genbank_accession,
        te.W3110_TIER_DEPOSIT.refseq_accession,
    )


# --------------------------------------------------------------------------------------
# Gate 2: the refusal, hermetically
# --------------------------------------------------------------------------------------
def test_every_loader_value_is_a_recoverable_gap_naming_where_to_look() -> None:
    """Six gaps, all deferred_pending_source_review, each with a resolve_with target."""
    assert [gap.field for gap in te.LOADER_GAPS] == [
        "n_samples",
        "phenotype_statistic",
        "time_zero",
        "media",
        "gene_namespace",
        "genotype",
    ]
    for gap in te.LOADER_GAPS:
        assert gap.reason is ProvenanceGapReason.deferred_pending_source_review
        assert gap.resolve_with is not None
        assert gap.resolve_with.citation_key is None
        assert gap.resolve_with.sha256 is None
        assert gap.note


def test_the_pmc_deposit_holds_the_workbook_the_row_was_counted_off() -> None:
    """Sixteen objects were listed; Table S4 is one of them and is the loader's input."""
    keys = [f.key for f in te.PMC_OA_DEPOSIT]
    assert len(keys) == len(set(keys)) == 16
    assert te.TABLE_S4_OA_KEY in keys
    assert te.TABLE_S4_OA_KEY == ("PMC11188689.1/supplementary_table_s4_wrae096.xlsx")
    assert [f.role for f in te.PMC_OA_DEPOSIT].count("si_data") == 2
    assert all(key.startswith(f"{te.PAPER_PMCID}.1/") for key in keys)


def test_the_schedule_counts_are_attributed_to_the_generator_not_restated() -> None:
    """The two counts carry the script that measured them, and the row number."""
    assert te.SCHEDULE_ROW == 41
    assert te.SCHEDULE_FITNESS_VALUES == 66_162
    assert te.SCHEDULE_GENE_BY_SAMPLE_ROWS == 11_027
    assert te.SCHEDULE_SOURCE == (
        "experiments/database/scripts/build_bacteria_candidate_datasets_table.py"
    )


def test_the_module_registers_no_dataset_and_lists_the_work_in_order() -> None:
    """No dataset class here; the first edit is the curation decision everything waits on."""
    assert not [
        name
        for name in dir(te)
        if name.endswith("Dataset") and isinstance(getattr(te, name), type)
    ]
    assert len(te.EDITS_NEEDED) == 7
    assert te.EDITS_NEEDED[0].startswith(
        "let scripts/lit_sync.py mirror the filed paper"
    )
    assert "gene symbols only" in te.EDITS_NEEDED[1]
    assert te.EDITS_NEEDED[-1].startswith("the loader itself")


def test_the_si_deposit_is_five_files_each_equal_to_a_listed_bucket_object() -> None:
    """Every deposited file names the PMC object it was measured equal to."""
    deposit = te.SI_DEPOSIT
    assert (deposit.citation_key, deposit.doi, deposit.issues) == (
        te.EXPECTED_CITATION_KEY,
        te.PAPER_DOI,
        (910,),
    )
    listed = {f.key for f in te.PMC_OA_DEPOSIT}
    assert deposit.directories() == ("si",)
    assert len(deposit.files) == 5
    for f in deposit.files:
        assert f.scripted_equivalent is not None
        assert f.scripted_equivalent.params["key"] == f"PMC11188689.1/{f.name}"
        assert f.scripted_equivalent.params["key"] in listed
    assert {f.path for f in deposit.files} >= {te.TABLE_S3_RELPATH, te.TABLE_S4_RELPATH}
    assert sum(te.TABLE_S4_ROWS_PER_WATER_SAMPLE.values()) == (
        te.TABLE_S4_GENE_BY_SAMPLE_ROWS
    )
    assert te.TABLE_S4_FITNESS_VALUES == 6 * te.TABLE_S4_GENE_BY_SAMPLE_ROWS
    assert te.TABLE_S4_FITNESS_VALUES == te.SCHEDULE_FITNESS_VALUES
    for counts in te.SYMBOL_ROUTES.values():
        assert sum(counts.values()) == te.TABLE_S4_DISTINCT_SYMBOLS


def _table_s4(path: Path, *, symbol: str | None = "thrL") -> None:
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "DATA"
    sheet.append(
        ["Lake water sample", "gene fitness values", None, None, None, None, None]
        + ["Gene name", "Principal component", None, "Euclidean distance"]
        + ["Protein description"]
    )
    sheet.append([None, "filtered", None, None, "non-filtered"])
    sheet.append(
        [None, "day 2", "day 4", "day 8", "day 2", "day 4", "day 8", None, 1, 2]
    )
    sheet.append([1, 0.1, 0.2, 0.3, 0.4, 0.5, None, symbol, 1.0, 2.0, 0.5, "p"])
    sheet.append([2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, "thrA", 1.0, 2.0, 0.5, "p"])
    sheet.append([2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, "thrL", 1.0, 2.0, 0.5, "p"])
    book.save(path)


def test_table_s4_inventory_names_the_columns_and_counts_values_and_symbols(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s4.xlsx"
    _table_s4(path)
    inventory = te.table_s4_inventory(path)
    assert inventory.gene_by_sample_rows == 3
    assert inventory.rows_per_water_sample == {1: 1, 2: 2}
    assert inventory.fitness_columns == te.TABLE_S4_FITNESS_COLUMNS
    assert inventory.fitness_values == 17
    assert inventory.identifier_column == "Gene name"
    assert inventory.distinct_symbols == ("thrA", "thrL")
    assert inventory.other_columns == (
        "Principal component",
        "Euclidean distance",
        "Protein description",
    )


def test_table_s4_inventory_refuses_another_layout_and_a_row_without_a_symbol(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s4.xlsx"
    _table_s4(path, symbol=None)
    with pytest.raises(ValueError, match="no gene symbol"):
        te.table_s4_inventory(path)
    book = openpyxl.load_workbook(path)
    book["DATA"]["H1"] = "Locus tag"
    book.save(path)
    with pytest.raises(ValueError, match="Table S4 header"):
        te.table_s4_inventory(path)


# --------------------------------------------------------------------------------------
# Gate 1: the deposited bytes
# --------------------------------------------------------------------------------------
@pytest.mark.data
@requires_tier
def test_every_deposited_member_hashes_to_the_record() -> None:
    """``verify_assembly_set`` re-hashes the tier; the ten digests are the record's."""
    assert verify_assembly_set(ECOLI_K12_W3110) == {
        m.path: m.sha256 for m in te.W3110_TIER_DEPOSIT.members
    }


@pytest.mark.data
@requires_tier
def test_the_genbank_member_is_refused_and_the_refseq_member_parses() -> None:
    """The GenBank-first ingest does not reach W3110, re-derived off the bytes."""
    genbank, refseq = te.annotation_routes()
    assert genbank.member == te.GENBANK_MEMBER
    assert genbank.gene_features == te.GENBANK_GENE_FEATURES == 4_444
    assert genbank.gene_features_with_locus_tag == 0
    assert genbank.read_genbank_error == te.GENBANK_READ_ERROR
    assert genbank.loci is None
    assert refseq.member == te.REFSEQ_MEMBER
    assert refseq.gene_features == te.REFSEQ_GENE_FEATURES == 4_531
    assert refseq.gene_features_with_locus_tag == te.REFSEQ_GENE_FEATURES
    assert refseq.read_genbank_error is None
    assert refseq.loci == te.REFSEQ_LOCI
    assert refseq.replicon == "NC_007779.1"


@pytest.mark.data
@requires_tier
def test_the_genbank_crosswalk_is_a_note_of_eck_jw_and_b_numbers() -> None:
    """3,730 triples, 3,730 distinct JW numbers, 3,726 distinct b-numbers."""
    triples = te.eck_jw_b_notes()
    assert len(triples) == te.ECK_JW_B_TRIPLES == 3_730
    assert len({jw for _, jw, _ in triples}) == te.ECK_JW_B_DISTINCT_JW
    assert len({b for _, _, b in triples}) == te.ECK_JW_B_DISTINCT_BNUMBER == 3_726
    eck, jw, bnum = triples[0]
    assert (eck, jw, bnum) == ("ECK0001", "JW4367", "b0001")


@pytest.mark.data
@requires_tier
def test_the_two_assembly_reports_disagree_about_the_refseq_release() -> None:
    """The GCA report names the 2006 release; only the GCF report names the pair."""
    gca = open(
        resolve(ECOLI_K12_W3110, "GCA_000010245.1_ASM1024v1_assembly_report.txt")
    ).read()
    gcf = open(
        resolve(ECOLI_K12_W3110, "GCF_000010245.2_ASM1024v1_assembly_report.txt")
    ).read()
    assert "# RefSeq assembly accession: GCF_000010245.1\n" in gca
    assert "# RefSeq assembly accession: GCF_000010245.2\n" in gcf
    assert "# GenBank assembly accession: GCA_000010245.1\n" in gca
    assert "# GenBank assembly accession: GCA_000010245.1\n" in gcf
    assert "\tAC_000091.1\t" in gca
    assert "\tNC_007779.1\t" in gcf


# --------------------------------------------------------------------------------------
# Gate 2: the tripwire
# --------------------------------------------------------------------------------------
@pytest.mark.data
@requires_mirrors
def test_the_paper_is_still_in_neither_mirror_so_the_loader_stays_blocked() -> None:
    """A TRIPWIRE: it fails the night lit_sync mirrors the paper filed on 2026-10-12.

    The raw mirror holds the owner's SI deposit (one key, this paper's); the library
    mirror holds nothing until lit_sync runs. When this fails, the Methods are
    readable: follow ``EDITS_NEEDED`` from item 2.
    """
    found = {
        root: sorted(
            key
            for key in os.listdir(osp.join(DATA_ROOT, root))
            if "teteneva" in key.lower()
        )
        for root in ("torchcell-library", "torchcell-raw")
    }
    assert found == {
        "torchcell-library": [],
        "torchcell-raw": [te.EXPECTED_CITATION_KEY],
    }, (
        "the Teteneva 2024 paper is now in the library mirror, so its Methods can be "
        f"read: {found}; follow torchcell.datasets.ecoli.teteneva2024.EDITS_NEEDED"
    )
    assert te.LIBRARY_MIRROR_KEYS_MATCHING_TETENEVA == 0
    assert te.RAW_MIRROR_KEYS_MATCHING_TETENEVA == 1


@pytest.mark.data
@requires_mirrors
@requires_tier
def test_the_deposited_table_s4_and_its_symbol_routes_are_the_measured_ones() -> None:
    """Re-derive the inventory and the per-member symbol landing off the pinned bytes."""
    key_root = Path(DATA_ROOT) / "torchcell-raw" / te.EXPECTED_CITATION_KEY
    verify_deposit(te.SI_DEPOSIT, key_root)
    inventory = te.table_s4_inventory(key_root / te.TABLE_S4_RELPATH)
    assert inventory.gene_by_sample_rows == te.TABLE_S4_GENE_BY_SAMPLE_ROWS
    assert inventory.rows_per_water_sample == te.TABLE_S4_ROWS_PER_WATER_SAMPLE
    assert inventory.fitness_values == te.TABLE_S4_FITNESS_VALUES
    assert inventory.fitness_columns == te.TABLE_S4_FITNESS_COLUMNS
    assert len(inventory.distinct_symbols) == te.TABLE_S4_DISTINCT_SYMBOLS
    assert te.symbol_routes(inventory.distinct_symbols) == te.SYMBOL_ROUTES


# --------------------------------------------------------------------------------------
# Gate 1a: the measurement functions, on synthetic flat files
# --------------------------------------------------------------------------------------
def _write_gbff(path: Path, *, tagged: bool, notes: tuple[str, ...] = ()) -> None:
    """A one-locus GenBank flat file, gzipped, with or without a ``locus_tag``."""
    record = SeqRecord(
        Seq("ATGAAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTTAAACCCTAA"),
        id="XX000001.1",
        name="XX000001",
        annotations={"molecule_type": "DNA", "topology": "circular"},
    )
    gene_qualifiers: dict[str, list[str]] = {"gene": ["thrL"]}
    cds_qualifiers: dict[str, list[str]] = {
        "gene": ["thrL"],
        "product": ["thr operon leader peptide"],
    }
    if tagged:
        gene_qualifiers["locus_tag"] = ["Y75_RS00005"]
        cds_qualifiers["locus_tag"] = ["Y75_RS00005"]
    if notes:
        cds_qualifiers["note"] = list(notes)
    location = SimpleLocation(0, 60, strand=1)
    record.features.append(
        SeqFeature(location, type="gene", qualifiers=gene_qualifiers)
    )
    record.features.append(SeqFeature(location, type="CDS", qualifiers=cds_qualifiers))
    with gzip.open(path, "wt") as handle:
        SeqIO.write(record, handle, "genbank")


def _serve(
    monkeypatch: pytest.MonkeyPatch, members: dict[str, Path]
) -> list[tuple[str, str]]:
    """Stub ``resolve`` in the module under test; record every ``(set, member)`` asked."""
    calls: list[tuple[str, str]] = []

    def serve(assembly_set: str, filename: str, **_: object) -> str:
        calls.append((assembly_set, filename))
        return str(members[filename])

    monkeypatch.setattr(te, "resolve", serve)
    return calls


def test_gene_feature_counts_separate_tagged_from_untagged_features(
    tmp_path: Path,
) -> None:
    """One gene feature each way: the tagged file counts 1 of 1, the other 0 of 1."""
    tagged, untagged = tmp_path / "tagged.gbff.gz", tmp_path / "untagged.gbff.gz"
    _write_gbff(tagged, tagged=True)
    _write_gbff(untagged, tagged=False)
    assert te._gene_feature_counts(str(tagged)) == (1, 1)
    assert te._gene_feature_counts(str(untagged)) == (1, 0)


def test_annotation_routes_refuses_the_untagged_member_and_parses_the_tagged_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route pair is built from whichever bytes the tier serves, GenBank first."""
    members = {
        te.GENBANK_MEMBER: tmp_path / "gca.gbff.gz",
        te.REFSEQ_MEMBER: tmp_path / "gcf.gbff.gz",
    }
    _write_gbff(members[te.GENBANK_MEMBER], tagged=False)
    _write_gbff(members[te.REFSEQ_MEMBER], tagged=True)
    calls = _serve(monkeypatch, members)
    genbank, refseq = te.annotation_routes()
    assert calls == [
        (ECOLI_K12_W3110, te.GENBANK_MEMBER),
        (ECOLI_K12_W3110, te.REFSEQ_MEMBER),
    ]
    assert genbank.gene_features == 1
    assert genbank.gene_features_with_locus_tag == 0
    assert genbank.read_genbank_error == (
        f"{te.GENBANK_MEMBER}: a gene feature at [0:60](+) has no locus_tag"
    )
    assert genbank.loci is None
    assert genbank.replicon is None
    assert refseq.gene_features_with_locus_tag == 1
    assert refseq.read_genbank_error is None
    assert refseq.loci == 1
    assert refseq.replicon == "XX000001.1"


def test_symbol_routes_count_unique_ambiguous_and_absent_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ThrL is on one feature of each member; an unknown symbol lands on none."""
    members = {
        te.GENBANK_MEMBER: tmp_path / "gca.gbff.gz",
        te.REFSEQ_MEMBER: tmp_path / "gcf.gbff.gz",
    }
    _write_gbff(members[te.GENBANK_MEMBER], tagged=False)
    _write_gbff(members[te.REFSEQ_MEMBER], tagged=True)
    _serve(monkeypatch, members)
    assert te.symbol_index(te.REFSEQ_MEMBER) == {"thrL": {"Y75_RS00005"}}
    assert te.symbol_routes(("thrL", "zzzZ")) == {
        te.GENBANK_MEMBER: {"unique": 1, "ambiguous": 0, "none": 1},
        te.REFSEQ_MEMBER: {"unique": 1, "ambiguous": 0, "none": 1},
    }


def test_eck_jw_b_notes_reads_only_the_crosswalk_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The triple note is read apart; any other note on the same CDS is not a triple."""
    member = tmp_path / "gca.gbff.gz"
    _write_gbff(member, tagged=False, notes=("ECK0001:JW4367:b0001", "not a crosswalk"))
    _serve(monkeypatch, {te.GENBANK_MEMBER: member})
    assert te.eck_jw_b_notes() == (("ECK0001", "JW4367", "b0001"),)
