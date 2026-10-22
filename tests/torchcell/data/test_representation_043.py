# tests/torchcell/data/test_representation_043.py
"""Exact behavior of the 043 representation resolver on hand-built contexts."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from torchcell.data.representation_043 import (
    AccessoryRow,
    AlleleVectorStore,
    Block,
    ContextTables,
    GapKind,
    GenePositions,
    KeyVocabulary,
    PositionGap,
    ReporterKeys,
    RepresentationConfig,
    RepresentationResolver,
    SegregantReason,
    assign_segregant_genes,
    chromosome_copy_vector,
    resolve_change_types,
)
from torchcell.models.genotype_tokens_043 import CHANGE_TYPE_INDEX, ChangeType

REF = CHANGE_TYPE_INDEX[ChangeType.reference]
VAR = CHANGE_TYPE_INDEX[ChangeType.variant]
LOF = CHANGE_TYPE_INDEX[ChangeType.loss_of_function]
ABS = CHANGE_TYPE_INDEX[ChangeType.absent]
DEL = CHANGE_TYPE_INDEX[ChangeType.deletion]


def sha(i: int) -> str:
    """A fake 64-hex key."""
    return f"{i:064x}"


def test_key_vocabulary_ids_and_unknown_key() -> None:
    vocab = KeyVocabulary([sha(5), sha(2), sha(9), sha(2)])
    assert len(vocab) == 3
    assert vocab.ids([sha(2), None, sha(9), sha(5)]).tolist() == [0, -1, 2, 1]
    assert vocab.key(1) == sha(5)
    fast = np.array([sha(9).encode(), b"", sha(2).encode()], dtype="S64")
    assert vocab.ids(fast).tolist() == [2, -1, 0]
    with pytest.raises(KeyError, match="1 keys not in the vocabulary"):
        vocab.ids([sha(3)])


def test_index_gap_maps_ambiguous_sequence() -> None:
    """The isolate index's ``ambiguous_sequence`` gap string resolves to its own kind."""
    from torchcell.data.representation_043 import INDEX_GAP

    assert INDEX_GAP["ambiguous_sequence"] is GapKind.AMBIGUOUS_SEQUENCE
    assert int(GapKind.AMBIGUOUS_SEQUENCE) == 11
    assert len(set(INDEX_GAP.values())) == len(INDEX_GAP)


def test_change_type_precedence() -> None:
    marks = [
        (0, ChangeType.variant),
        (0, ChangeType.loss_of_function),
        (1, ChangeType.deletion),
        (1, ChangeType.absent),
        (2, ChangeType.variant),
    ]
    assert resolve_change_types(marks, 4).tolist() == [LOF, ABS, VAR, REF]


def test_segregant_rule_runs_single_block_and_reasons() -> None:
    blocks = [
        Block(chromosome="chrI", start=100, end=200, parent=1, posterior=0.95),
        Block(chromosome="chrI", start=210, end=400, parent=1, posterior=1.0),
        Block(chromosome="chrI", start=410, end=500, parent=2, posterior=0.6),
        Block(chromosome="chrI", start=510, end=900, parent=2, posterior=0.9),
        Block(chromosome="chrI", start=905, end=1000, parent=1, posterior=0.99),
    ]
    chrom = ["chrI", "chrI", "chrI", "chrI", "chrI", "chrII"]
    start = np.array([120, 150, 450, 600, 880, 10], dtype=np.int64)
    end = np.array([180, 350, 460, 700, 950, 20], dtype=np.int64)
    parent, reason = assign_segregant_genes(chrom, start, end, blocks, 0.9)
    # gene 1 spans the posterior-level split 200|210 of one parent: one confident run
    assert parent.tolist() == [1, 1, 0, 2, 0, 0]
    assert reason.tolist() == [
        SegregantReason.ASSIGNED,
        SegregantReason.ASSIGNED,
        SegregantReason.LOW_POSTERIOR,
        SegregantReason.ASSIGNED,
        SegregantReason.STRADDLE,
        SegregantReason.NO_BLOCK,
    ]
    parent_s, reason_s = assign_segregant_genes(
        chrom, start, end, blocks, 0.9, rule="single_block"
    )
    assert parent_s.tolist() == [1, 0, 0, 2, 0, 0]
    assert int(reason_s[1]) == SegregantReason.STRADDLE


def test_chromosome_copy_vector() -> None:
    copy, unknown = chromosome_copy_vector(2, [{"chromosome": 8, "change": 1}])
    assert copy[8] == pytest.approx(1.5)
    assert float(copy.sum()) == pytest.approx(16 + 1.5)
    assert not unknown.any()
    copy, unknown = chromosome_copy_vector(None, [{"chromosome": 3, "change": -1}])
    assert np.isnan(copy[3]) and unknown.tolist() == [i == 3 for i in range(17)]
    copy, unknown = chromosome_copy_vector(None, [])
    assert copy.tolist() == [1.0] * 17 and not unknown.any()


def _resolver(route: str) -> RepresentationResolver:
    # 4 genes; contexts S288C (keys 0-3), ISO (keys 10-13 per gene, gene 3 unreleased),
    # PAR (a segregant parent, keys 20-23). Key id = the integer itself.
    g = 4
    key_ids = np.zeros((3, g, 4), dtype=np.int32)
    for c, base in enumerate((0, 10, 20)):
        for i in range(g):
            key_ids[c, i, :] = base + i
    key_gap = np.zeros((3, g, 4), dtype=np.int8)
    key_ids[1, 3, :] = -1
    key_gap[1, 3, :] = GapKind.NO_ISOLATE_ALLELE
    key_ids[1, 2, 0] = -1
    key_gap[1, 2, 0] = GapKind.FLANK_SHORT
    gene_copy = np.array(
        [[1, 1, 1, 1], [2, 1, 0.5, np.nan], [1, 3, 1, 1]], dtype=np.float32
    )
    tables = ContextTables(
        context_ids=("S288C", "ISO", "PAR"),
        key_ids=key_ids,
        key_gap=key_gap,
        gene_copy=gene_copy,
        state=np.zeros((3, g), dtype=np.int8),
        accessory={
            "ISO": {
                "orf1": AccessoryRow(
                    key_ids=(30, 31, 32, 33), gaps=(0, 0, 0, 0), copy_number=2.0
                )
            }
        },
    )
    positions = GenePositions.build(
        [("chrI", 10, 20), ("chrI", 30, 40), ("chrVIII", 5, 15), ("chrmt", 1, 9)],
        {"chrI": 100, "chrVIII": 50, "chrmt": 10},
    )
    reporter = ReporterKeys(
        feature_id="yeGFP",
        upstream_by_variant={"v1": 40},
        downstream=41,
        protein=42,
        calm=43,
    )
    return RepresentationResolver(
        RepresentationConfig(route=route),  # type: ignore[arg-type]
        positions,
        tables,
        reporter,
        ["BY", "RM"],
    )


def test_isolate_route_a_and_b() -> None:
    marks = [
        (0, ChangeType.variant),
        (0, ChangeType.loss_of_function),
        (1, ChangeType.absent),
        (2, ChangeType.deletion),
    ]
    a = _resolver("A").isolate(
        "ISO",
        marks,
        2,
        [{"chromosome": 8, "change": 1}],
        [("orf1", 2.0), ("orf9", 1.0)],
    )
    assert a.context_id == "ISO"
    assert a.change_type.tolist() == [LOF, ABS, DEL, REF]
    assert a.key_ids[:, 2].tolist() == [10, -1, 12, -1]
    assert a.key_ids[:, 0].tolist() == [10, -1, -1, -1]
    assert a.key_gap[:, 0].tolist() == [
        0,
        GapKind.ABSENT_IN_CONTEXT,
        GapKind.FLANK_SHORT,
        GapKind.NO_ISOLATE_ALLELE,
    ]
    assert a.operator_mask.tolist() == [False, False, True, False]
    assert a.chrom_copy.tolist() == [1.0, 1.0, 1.5, 1.0]
    assert a.position_gap.tolist() == [0, 0, 0, PositionGap.COPY_NUMBER_UNRELEASED]
    assert [t.key_ids for t in a.extra_tokens] == [(30, 31, 32, 33), (-1, -1, -1, -1)]
    assert a.extra_tokens[1].gaps == (GapKind.ACCESSORY_WITHOUT_ROW,) * 4

    b = _resolver("B").isolate("ISO", marks, 2, [], [])
    assert b.context_id == "S288C"
    # the edit list is the variant/LOF genes only: gene 0 carries the isolate's keys
    assert b.key_ids[:, 2].tolist() == [10, 1, 2, 3]
    assert b.operator_mask.tolist() == [True, True, True, False]
    assert b.chrom_copy.tolist() == [1.0] * 4


def test_segregant_keys_parents_and_gaps() -> None:
    blocks = [
        Block(chromosome="chrI", start=1, end=25, parent=1, posterior=1.0),
        Block(chromosome="chrI", start=26, end=60, parent=2, posterior=0.92),
        Block(chromosome="chrVIII", start=1, end=50, parent=2, posterior=0.5),
    ]
    r = _resolver("A").segregant(blocks, (("BY", None), ("RM", "PAR")))
    assert r.key_ids[:, 3].tolist() == [0, 21, -1, -1]
    assert r.change_type.tolist() == [REF, VAR, REF, REF]
    assert r.allele_parent.tolist() == [-1, 1, -1, -1]
    assert r.key_gap[:, 0].tolist() == [
        0,
        0,
        GapKind.SEGREGANT_UNASSIGNED,
        GapKind.SEGREGANT_UNASSIGNED,
    ]
    assert r.gene_copy[:2].tolist() == [1.0, 3.0]
    assert r.segregant_reason is not None
    assert r.segregant_reason.tolist() == [
        SegregantReason.ASSIGNED,
        SegregantReason.ASSIGNED,
        SegregantReason.LOW_POSTERIOR,
        SegregantReason.NO_BLOCK,
    ]
    with pytest.raises(KeyError, match="parent vocabulary"):
        _resolver("A").segregant(blocks, (("BY", None), ("YJM145", "PAR")))


def test_promoter_and_deletion_records() -> None:
    res = _resolver("A")
    p = res.promoter("v1")
    assert p.extra_tokens[0].key_ids == (40, 41, 42, 43)
    assert p.key_ids[:, 0].tolist() == [0, 1, 2, 3]
    missing = res.promoter("v2")
    assert missing.extra_tokens[0].gaps[0] == GapKind.MISSING_REPORTER_WINDOW
    d = res.deletion([(1, ChangeType.deletion)])
    assert d.marked.tolist() == [1] and d.operator_mask.tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert d.rel_midpoint.tolist() == pytest.approx([0.15, 0.35, 0.2, 0.5])
    assert d.chrom_index.tolist() == [1, 1, 8, 0]
    assert d.genotype_class == "deletion"
    empty = res.deletion([])
    assert empty.genotype_class == "reference" and empty.marked.tolist() == []


def test_allele_vector_store_reads_and_raises(tmp_path: Path) -> None:
    vocab = KeyVocabulary([sha(1), sha(2), sha(3)])
    path = tmp_path / "enc.h5"
    with h5py.File(path, "w") as f:
        f["sha256"] = np.array([sha(3).encode(), sha(1).encode()], dtype="S64")
        f["embedding"] = np.array([[3, 3], [1, 1]], dtype=np.float16)
    store = AlleleVectorStore(str(path), vocab)
    ids = vocab.ids([sha(1), sha(3), sha(1)])
    assert store.vectors(ids).tolist() == [[1, 1], [3, 3], [1, 1]]
    with pytest.raises(KeyError, match="1 keys have no vector"):
        store.vectors(vocab.ids([sha(2)]))
    with pytest.raises(KeyError, match="gap id"):
        store.vectors(np.array([-1]))
