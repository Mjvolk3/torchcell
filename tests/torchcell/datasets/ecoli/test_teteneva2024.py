# tests/torchcell/datasets/ecoli/test_teteneva2024.py
# [[tests.torchcell.datasets.ecoli.test_teteneva2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_teteneva2024.py

"""The two gates of the Teteneva 2024 row, pinned where they stand.

Gate 1 (the host) is closed: the hermetic tests check the deposited W3110 set's record
against the registry id and the schema's vocabularies, and the data-gated tests re-hash
all ten members and re-derive the annotation finding off the deposited bytes, so a
member that changes, or a parser that starts accepting a flat file with no ``locus_tag``,
fails by name.

Gate 2 (the paper) is blocked: one data-gated test asserts the paper is still in neither
mirror. It is a TRIPWIRE and it is meant to fail the day the paper is curated, with the
message naming the work that then becomes possible.
"""

import os
import os.path as osp
import re

import pytest

from torchcell.datamodels.schema import ASSEMBLY_SET_ACCESSIONS, BACTERIAL_ASSEMBLY_SETS
from torchcell.datasets.ecoli import teteneva2024 as te
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


def test_the_set_is_in_no_schema_vocabulary_so_nothing_served_changes() -> None:
    """W3110 is not a reference strain and its set id is not an assembly-set member."""
    assert "W3110" not in BACTERIAL_ASSEMBLY_SETS
    assert ECOLI_K12_W3110 not in set(BACTERIAL_ASSEMBLY_SETS.values())
    assert ECOLI_K12_W3110 not in ASSEMBLY_SET_ACCESSIONS


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
    assert len(te.EDITS_NEEDED) == 8
    assert te.EDITS_NEEDED[0].startswith("file the paper in Zotero")
    assert te.EDITS_NEEDED[-1].startswith("the loader itself")


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
    """A TRIPWIRE: it fails the day the paper is curated, which is the signal to build.

    When this fails, the row is unblocked: follow ``EDITS_NEEDED`` from item 2.
    """
    found = {
        root: sorted(
            key
            for key in os.listdir(osp.join(DATA_ROOT, root))
            if "teteneva" in key.lower()
        )
        for root in ("torchcell-library", "torchcell-raw")
    }
    assert found == {"torchcell-library": [], "torchcell-raw": []}, (
        "the Teteneva 2024 paper is now mirrored, so the loader is no longer blocked: "
        f"{found}; follow torchcell.datasets.ecoli.teteneva2024.EDITS_NEEDED"
    )
    assert te.LIBRARY_MIRROR_KEYS_MATCHING_TETENEVA == 0
    assert te.RAW_MIRROR_KEYS_MATCHING_TETENEVA == 0
