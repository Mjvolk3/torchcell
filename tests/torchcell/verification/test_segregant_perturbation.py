# tests/torchcell/verification/test_segregant_perturbation.py
# [[tests.torchcell.verification.test_segregant_perturbation]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_segregant_perturbation.py
"""The perturbation-on-segregant verifier on the synthetic Hale 2024 store.

The store is built by the real loader from ``_hale2024_synthetic`` (14 records, three
served backgrounds, two guides, three ledgered control gRNAs). Every level passes on it;
each failure branch is then driven by one targeted edit: a stored value that no longer
equals its released cell, a background that no longer re-expands to its column, a
Supplemental Data 2 cell that disagrees with Supplemental Data 12, a ledger that does
not balance, a parent file whose bytes moved, and a quote that left its source.

Hand-derived values: the normalization offsets ATC1 0.3025, ATC2 0.3225, CON 0.2525
from 2 retained control gRNAs; the variant table holds 2 SNVs, 1 deletion and 1
insertion, and the 3 anchored rows carry the synthetic FASTA's base (chrI 100 'T',
chrI 250 'C', chrII 60 'A').
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import lmdb
import pandas as pd
import pytest

from tests.torchcell.datasets.scerevisiae import _hale2024_synthetic as syn
from torchcell.datasets.scerevisiae import hale2024 as h
from torchcell.sequence.genome.registry import HALE2024_3S_VARIANTS
from torchcell.verification import segregant_perturbation as sp
from torchcell.verification.report import Level, Provenance, VerificationReport
from torchcell.verification.runners import stream_records
from torchcell.verification.sourced import SourcedValue

_PROVENANCE = Provenance(source_uri="synthetic", citation_key=h.CITATION_KEY)
_SGD = {"YAL001C", "YBL001C", "YCL001W"}


@pytest.fixture
def store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[h.CrispriSegregantHale2024Dataset, syn.SyntheticRelease]:
    release = syn.write_release(tmp_path)
    syn.patch(monkeypatch, release)
    dataset = h.CrispriSegregantHale2024Dataset(
        root=str(tmp_path / "store"),
        genome=syn.StubGenome(),  # type: ignore[arg-type]
    )
    dataset.close_lmdb()
    return dataset, release


def _verify(
    dataset: h.CrispriSegregantHale2024Dataset, release: syn.SyntheticRelease
) -> VerificationReport:
    genome = syn.StubGenome()
    return sp.verify_segregant_perturbation_streaming(
        stream_records(dataset.root),
        dataset_name="crispri_segregant_hale2024",
        provenance=_PROVENANCE,
        expected_count=14,
        raw_dir=dataset.raw_dir,
        preprocess_dir=dataset.preprocess_dir,
        s288c_fasta=release.s288c,
        variant_table=release.variants,
        parent_files={h.S288C_ASSEMBLY_FSA: release.s288c, h.SD13: release.variants},
        library_root=None,
        raw_root=None,
        resolve_gene_name=genome.resolve_gene_name,
        sgd_genes=_SGD,
    )


def _by_name(report: VerificationReport) -> dict[str, Any]:
    return {r.name: r for r in report.results}


def test_every_level_passes_on_the_synthetic_store(
    store: tuple[h.CrispriSegregantHale2024Dataset, syn.SyntheticRelease],
) -> None:
    report = _verify(*store)
    failing = [(r.name, r.message) for r in report.results if not r.passed]
    assert failing == []
    assert report.levels_covered == {Level.L0, Level.L1, Level.L2, Level.L3, Level.L4}
    results = _by_name(report)
    assert results["structural"].details["n_distinct_backgrounds"] == 3
    assert results["screen_partition"].details["per_screen"] == {
        "ATC1": 5,
        "ATC2": 4,
        "CON": 5,
    }
    assert results["ledger_balance"].details["balance"] == {
        "ATC1": 0,
        "ATC2": 0,
        "CON": 0,
    }
    norm = results["normalization_reproduced"].details
    assert norm["n_controls_kept"] == 2
    for screen, offset in syn.offsets().items():
        assert norm["recomputed"][screen] == pytest.approx(offset, abs=1e-15)
        assert norm["released"][screen] == pytest.approx(offset, abs=1e-15)
    assert results["mosaic_round_trip"].details["n_blocks"] == {
        "segA": 4,
        "segB": 4,
        "x_BY_BYx": 3,
    }
    assert results["environment_documented"].details["per_screen"] == {
        "ATC1": ["anhydrotetracycline@0.25ug/mL"],
        "ATC2": ["anhydrotetracycline@0.25ug/mL"],
        "CON": [""],
    }
    variant = results["variant_reference"].details
    assert variant["kinds"] == {"snv": 2, "deletion": 1, "insertion": 1}
    assert variant["n_match"] == 3
    # 3 shared markers x 3 columns, every cell called
    assert results["matrix_subset_agreement"].details["compared"] == 9
    assert "sourced_values" not in results


def test_a_stored_value_that_left_its_cell_fails_value_fidelity(
    store: tuple[h.CrispriSegregantHale2024Dataset, syn.SyntheticRelease],
) -> None:
    dataset, release = store
    env = lmdb.open(str(Path(dataset.processed_dir) / "lmdb"))
    with env.begin(write=True) as txn:
        stored = txn.get(b"3")
        assert stored is not None
        record = pickle.loads(stored)
        record["experiment"]["phenotype"]["environment_response"] = 9.0
        txn.put(b"3", pickle.dumps(record))
    env.close()
    result = _by_name(_verify(dataset, release))["value_fidelity"]
    assert result.passed is False
    assert result.details["n_mismatch"] == 1
    (example,) = result.details["examples"]
    assert (example["screen"], example["lineage"]) == ("ATC1", "GB1_b4")
    assert example["stored"][0] == 9.0


def test_mosaics_ledger_and_matrix_failures(
    store: tuple[h.CrispriSegregantHale2024Dataset, syn.SyntheticRelease],
) -> None:
    dataset, release = store
    matrix = h.read_genotype_matrix(Path(dataset.raw_dir) / h.SD12)
    background = next(
        r["experiment"]["genotype"]["background"]
        for r in stream_records(dataset.root)
        if r["experiment"]["genotype"]["background"]["segregant_id"] == "segB"
    )
    flipped = {
        **background,
        "blocks": [{**b, "parent": 3 - b["parent"]} for b in background["blocks"]],
    }
    result = sp._l2_mosaics({"segB": flipped}, {"segB": {"parent_1"}}, matrix)
    assert result.passed is False
    assert result.details["failures"] == [
        "segB: calls differ",
        "segB: roles ['parent_1'] != {'segregant'}",
    ]

    assays, _ = sp.read_released_tables(dataset.raw_dir)
    ledger = Path(dataset.preprocess_dir) / "dropped_records.json"
    rules = syn.read_json(ledger)
    rules[0]["per_assay"]["CON"] = 2
    ledger.write_text(json.dumps(rules))
    balance = sp._l1_ledger_balance(dataset.preprocess_dir, assays)
    assert balance.passed is False
    assert balance.details["balance"] == {"ATC1": 0, "ATC2": 0, "CON": -1}

    sd2 = Path(dataset.raw_dir) / h.SD2
    frame = pd.read_csv(sd2, sep="\t")
    frame.loc[0, "segA"] = 1  # chrI 100: SD12 calls segA BY (0) there
    sd2.unlink()  # a symlink into the synthetic mirror
    frame.to_csv(sd2, sep="\t", index=False, na_rep="NA")
    subset = sp._l4_matrix_subset(dataset.raw_dir, matrix)
    assert subset.passed is False
    assert subset.details["disagree"] == 1


def test_parents_and_sources_that_moved_fail(
    store: tuple[h.CrispriSegregantHale2024Dataset, syn.SyntheticRelease],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, release = store
    backgrounds = {
        r["experiment"]["genotype"]["background"]["segregant_id"]: r["experiment"][
            "genotype"
        ]["background"]
        for r in stream_records(dataset.root)
    }
    other = tmp_path / "other.fsa"
    other.write_text(">x\nA\n")
    moved = sp._l4_parents_pinned(
        backgrounds, {h.S288C_ASSEMBLY_FSA: other, h.SD13: release.variants}
    )
    assert moved.passed is False
    assert moved.details["failures"] == [
        f"BY: {h.S288C_ASSEMBLY_FSA} does not hash to the pinned sha256"
    ]
    assert sp._l4_parents_pinned(backgrounds, {}).details["failures"] == [
        f"3S: no file for {HALE2024_3S_VARIANTS}/{h.SD13}",
        f"BY: no file for sgd_S288C_R64-4-1_20230830/{h.S288C_ASSEMBLY_FSA}",
    ]

    library = tmp_path / "library" / h.CITATION_KEY
    library.mkdir(parents=True)
    (library / "paper.md").write_text("the assays were run at 21 C")
    sha = hashlib.sha256(b"the assays were run at 21 C").hexdigest()
    present = SourcedValue(
        value=21.0,
        quote="run at 21 C",
        provenance=Provenance(
            source_uri="paper.md", citation_key=h.CITATION_KEY, sha256=sha
        ),
    )
    absent = present.model_copy(update={"quote": "run at 30 C"})
    monkeypatch.setattr(h, "SOURCED_VALUES", [present, absent])
    monkeypatch.setattr(h, "CODE_SOURCED_VALUES", [])
    audit = sp._l3_sourced_values(tmp_path / "library", tmp_path / "raw")
    assert audit.passed is False
    assert audit.details["failures"] == [
        {"quote": "run at 30 C", "message": "quote no longer found in paper.md"}
    ]
