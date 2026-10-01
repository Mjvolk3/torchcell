# tests/torchcell/datasets/scerevisiae/test_nadal_ribelles2025.py
# [[tests.torchcell.datasets.scerevisiae.test_nadal_ribelles2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_nadal_ribelles2025.py
"""Nadal-Ribelles 2025 pseudobulk loader: what a record stores per batch, the resolver,
the ledger, a stale raw file and ``main``.

2026.09.30 (Phase 17). The end-to-end build on the synthetic ``.Rdata`` pair is pinned in
``test_nadal_ribelles2025_synthetic.py``; its fixture helpers (the stub genome, the
``fcs``/``ptbs`` tables, the expected environments) are reused here. The files are
written with ``rdata.write_rda`` as there; no R, no mirror, no ``$DATA_ROOT``.

Per batch. The memory note ``nadal-ribelles-assignment-impure`` records that the same
genotype profiled in two cartridge batches does not replicate itself (median r 0.043
against 0.022 for different genotypes, experiment 028). The loader has no batch key: a
record is one ``fcs`` table per (condition, label), and its two single-cell scalars come
from ``ptbs`` by label. The batch fixture gives ``bc-YAL012W`` two control rows (batch
c1: 60 cells, sd 1.5; batch c2: 40 cells, sd 0.9) and WT two control rows (c1: 300 cells,
sd 1.0; c2: 200, sd 1.2). Contract (issue #541, fixed 2026.10.01): a label repeated within
one condition raises ``RepeatedPtbsLabelError`` naming the condition and the labels, since
no field says which row belongs to the logFC vector; the released ``ptbs`` repeats no
label (0 of 3,207 control and 3,204 nacl rows). The lookup fixture has one row per label,
with NaCl listing WT (250, 1.05) BEFORE ``bc-YAL012W`` (45, 0.8): the record stores
0.8 / 45 and the reference 1.05 / 250 (looked up by label, not position).

A table ``DEG_Heat_bc_YAL012W.csv`` names a condition the paper did not profile: the
build raises ``UnknownConditionError`` before any store is written, instead of storing it
with the CONTROL environment.

Resolver (stub genome from the synthetic file plus the alias ``YPL998W`` and ``RETIRED``
both pointing at ``YZZ000W``, which is not a genome ID): ``yal012w`` -> YAL012W,
``str1`` -> YAL012W (Alias column), ``YPL998W`` and ``RETIRED`` -> None (candidate not in
the genome), ``YPL997W`` (systematic form, no alias) -> None.

Ledger on the synthetic build: kept = 3 (record 0: MET14, mup1, STR1) + 2 + 1 + 1 = 7;
dropped = 2 (15S_RRNA twice); collisions = 1 (YKL001C after MET14); unparsed = 1
(NOTANORF). ``main`` builds under ``$DATA_ROOT/data/torchcell/nadal_ribelles_perturbseq2025``
with ``load_dotenv`` stubbed and ``SCerevisiaeGenome`` a recorder returning the stub.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import rdata

from tests.torchcell.datasets.scerevisiae.test_nadal_ribelles2025_synthetic import (
    _CONTROL,
    _NACL,
    _deg,
    _experiment,
    _genome,
    _reference,
    _StubGenome,
    _write_files,
)
from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.schema import MarkerDeletionPerturbation
from torchcell.datasets.scerevisiae import nadal_ribelles2025 as m
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

_FCS = {
    "DEG_Control_bc_YAL012W.csv": _deg(["MET14", "MUP1"], [1.0, -0.5]),
    "DEG_NaCl_bc_YAL012W.csv": _deg(["MET14"], [2.0]),
}
_BATCH_PTBS = {
    "control": pd.DataFrame(
        {
            "assignment_consensus2": ["bc-YAL012W", "bc-YAL012W", "WT", "WT"],
            "batch": ["c1", "c2", "c1", "c2"],
            "cell_number": [60.0, 40.0, 300.0, 200.0],
            "sd_lvscore_scaledFU2": [1.5, 0.9, 1.0, 1.2],
        }
    ),
    "NaCl": pd.DataFrame(
        {
            "assignment_consensus2": ["WT", "bc-YAL012W"],
            "cell_number": [250.0, 45.0],
            "sd_lvscore_scaledFU2": [1.05, 0.8],
        }
    ),
}
_PTBS = {
    "control": pd.DataFrame(
        {
            "assignment_consensus2": ["bc-YAL012W", "WT"],
            "cell_number": [60.0, 300.0],
            "sd_lvscore_scaledFU2": [1.5, 1.0],
        }
    ),
    "NaCl": _BATCH_PTBS["NaCl"],
}


def _write(directory: Path, fcs: dict[str, Any], ptbs: dict[str, Any]) -> None:
    directory.mkdir(parents=True)
    rdata.write_rda(str(directory / m.FC_NAME), {"fcs": fcs})
    rdata.write_rda(str(directory / m.PTB_NAME), {"ptbs": ptbs})
    (directory / m.README_NAME).write_bytes(b"readme\n")


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fcs: dict[str, Any],
    ptbs: dict[str, Any],
) -> m.NadalRibellesPerturbSeq2025Dataset:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    _write(tmp_path / "nadal" / "raw", fcs, ptbs)
    return m.NadalRibellesPerturbSeq2025Dataset(
        root=str(tmp_path / "nadal"), genome=_genome()
    )


def test_a_label_repeated_within_a_condition_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two control rows for ``bc-YAL012W`` (batches c1, c2) and for WT: refused, naming
    the condition and both normalized labels, before any store is written; the first
    row's scalars are no longer stored silently.
    """
    with pytest.raises(m.RepeatedPtbsLabelError) as err:
        _build(tmp_path, monkeypatch, _FCS, _BATCH_PTBS)
    assert str(err.value) == (
        "ptbs condition 'control' lists genotype label(s) ['WT', 'bc_YAL012W'] on more "
        "than one row; one record takes one row's dispersion and n_cells, refusing to "
        "keep the first"
    )
    assert list((tmp_path / "nadal" / "processed").iterdir()) == []


def test_ptbs_lookup_is_by_label_not_row_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control stores its one row (1.5, 60) and WT (1.0, 300); NaCl lists WT first, the
    mutant still gets its own row (0.8, 45) and the reference the WT row (1.05, 250).
    """
    dataset = _build(tmp_path, monkeypatch, _FCS, _PTBS)
    assert len(dataset) == 2
    assert dataset[0]["experiment"] == _experiment(
        "YAL012W",
        "CYS3",
        "bc_YAL012W",
        _CONTROL,
        {"YKL001C": 1.0, "YGR055W": -0.5},
        1.5,
        60,
    )
    assert dataset[0]["reference"] == _reference(
        _CONTROL, ["YKL001C", "YGR055W"], 1.0, 300
    )
    assert dataset[1]["experiment"] == _experiment(
        "YAL012W", "CYS3", "bc_YAL012W", _NACL, {"YKL001C": 2.0}, 0.8, 45
    )
    assert dataset[1]["reference"] == _reference(_NACL, ["YKL001C"], 1.05, 250)


def test_an_unknown_condition_is_refused_before_any_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``Heat`` table is refused with ``UnknownConditionError`` instead of being stored
    under the control environment; nothing is written to ``processed/``.
    """
    fcs = {**_FCS, "DEG_Heat_bc_YAL012W.csv": _deg(["MUP1"], [0.7])}
    with pytest.raises(m.UnknownConditionError) as err:
        _build(tmp_path, monkeypatch, fcs, _PTBS)
    assert str(err.value) == (
        "condition 'heat' is neither of the profiled conditions ['control', 'nacl']; "
        "refusing to store it with the control environment"
    )
    assert list((tmp_path / "nadal" / "processed").iterdir()) == []


class _AliasGenome(_StubGenome):
    """The synthetic stub plus two aliases whose candidate is not a genome ID."""

    alias_to_systematic: dict[str, list[str]] = {
        **_StubGenome.alias_to_systematic,
        "YPL998W": ["YZZ000W"],
        "RETIRED": ["YZZ000W"],
    }


def test_resolver_uppercases_and_drops_aliases_that_leave_the_genome() -> None:
    """Each branch of ``resolve`` (lines 246-261): an ID in any case, the Alias column,
    and the two alias-map lookups that reject a candidate outside the genome.
    """
    holder = SimpleNamespace(genome=cast(SCerevisiaeGenome, _AliasGenome()))
    resolve, sys_to_common = m.NadalRibellesPerturbSeq2025Dataset._resolvers(
        cast(m.NadalRibellesPerturbSeq2025Dataset, holder)
    )
    tokens = ["yal012w", "str1", "YPL999W", "oldsym", "YPL998W", "RETIRED", "YPL997W"]
    assert [resolve(t) for t in tokens] == [
        "YAL012W",
        "YAL012W",
        "YKL001C",
        "YGR055W",
        None,
        None,
        None,
    ]
    assert sys_to_common["YKL001C"] == "MET14"


def test_build_logs_the_gene_ledger_and_every_skipped_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """On the synthetic tables: 4 records, 7 genes kept, 2 dropped, 1 collision, 1
    unparsed label, with the unparseable label and the all-unresolvable table named.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    _write_files(tmp_path / "nadal" / "raw")
    with caplog.at_level(logging.INFO, logger=m.log.name):
        m.NadalRibellesPerturbSeq2025Dataset(
            root=str(tmp_path / "nadal"), genome=_genome()
        )
    messages = [r.getMessage() for r in caplog.records if r.name == m.log.name]
    assert messages == [
        "Loading FC_genotype.Rdata (426 MB; ~6 min) ...",
        "Dropping record with unparseable ORF label: NOTANORF",
        "Record Control/YGR055W has no resolvable genes; skipping",
        "Resolved 4 records; genes kept=7 dropped=2 (unresolvable), collisions=1, "
        "ORF-labels unparsed=1",
        "Wrote 4 pseudobulk records to LMDB",
    ]


def test_a_stale_raw_file_is_refused_at_build_time_though_the_mirror_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): ``download`` verifies every MIRROR file and links
    only what ``raw/`` lacks, so a leftover ``raw/ptb_summary.Rdata`` (WT row 999 cells)
    stays in place while the mirror copy (500 cells) passes. ``process`` then verifies
    ``raw/`` itself and refuses the stale file with ``RawSha256MismatchError`` naming it,
    the pin and its digest; no store is written.
    """
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = data_root / m.RAW_DIR_REL
    files = _write_files(mirror)
    for name, data in files.items():
        monkeypatch.setitem(m.SHA256_EXPECTED, name, hashlib.sha256(data).hexdigest())
    raw = tmp_path / "nadal" / "raw"
    raw.mkdir(parents=True)
    stale = {
        "control": pd.DataFrame(
            {
                "assignment_consensus2": ["WT"],
                "cell_number": [999.0],
                "sd_lvscore_scaledFU2": [0.5],
            }
        )
    }
    rdata.write_rda(str(raw / m.PTB_NAME), {"ptbs": stale})
    stale_bytes = (raw / m.PTB_NAME).read_bytes()
    with pytest.raises(RawSha256MismatchError) as err:
        m.NadalRibellesPerturbSeq2025Dataset(
            root=str(tmp_path / "nadal"), genome=_genome()
        )
    assert str(err.value) == (
        f"sha256 mismatch for {raw / m.PTB_NAME}: expected "
        f"{hashlib.sha256(files[m.PTB_NAME]).hexdigest()}, "
        f"observed {hashlib.sha256(stale_bytes).hexdigest()}"
    )
    assert not os.path.islink(raw / m.PTB_NAME)
    assert (raw / m.PTB_NAME).read_bytes() == stale_bytes
    assert [os.readlink(raw / n) for n in (m.FC_NAME, m.README_NAME)] == [
        str(mirror / m.FC_NAME),
        str(mirror / m.README_NAME),
    ]
    assert list((tmp_path / "nadal" / "processed").iterdir()) == []


def test_main_builds_under_data_root_with_a_read_only_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` opens the genome at ``$DATA_ROOT/data/sgd/genome`` with ``overwrite=False``,
    builds the dataset under ``data/torchcell/nadal_ribelles_perturbseq2025`` and prints
    the length and record 0 (3 genes, dispersion 1.25, 120 cells, control environment),
    after the base class's streaming line from the build.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    genome_calls: list[dict[str, Any]] = []

    def genome(**kwargs: Any) -> SCerevisiaeGenome:
        genome_calls.append(kwargs)
        return _genome()

    monkeypatch.setattr(m, "SCerevisiaeGenome", genome)
    root = tmp_path / "data" / "torchcell" / "nadal_ribelles_perturbseq2025"
    _write_files(root / "raw")
    m.main()
    assert genome_calls == [
        {
            "genome_root": f"{tmp_path}/data/sgd/genome",
            "go_root": f"{tmp_path}/data/go",
            "overwrite": False,
        }
    ]
    perturbation = MarkerDeletionPerturbation(
        systematic_gene_name="YAL012W",
        perturbed_gene_name="CYS3",
        marker="URA3",
        strain_id="bc_YAL012W",
    ).model_dump()
    assert capsys.readouterr().out == (
        # printed by the base class's reference-index pass during the build
        "Computing experiment_reference_index (streaming)...\n"
        "len = 4\n"
        f"record[0] perturbations: {[perturbation]}\n"
        "record[0] n phenotype genes: 3\n"
        "record[0] dispersion: 1.25 n_cells: 120\n"
        f"record[0] environment: {_CONTROL.model_dump()}\n"
    )
    assert (root / "processed" / "lmdb" / "data.mdb").is_file()
