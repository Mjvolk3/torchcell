# tests/torchcell/paper/test_signal.py
# [[tests.torchcell.paper.test_signal]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/paper/test_signal.py
"""``torchcell.paper.signal``: the path resolver and the CLI front of the gzip signal.

``resolve_lmdb`` runs on directories under ``tmp_path`` holding an empty ``data.mdb``
(the resolver checks only that file's existence). ``main`` runs through a monkeypatched
``sys.argv`` with ``load_dotenv`` stubbed, ``DATA_ROOT`` set to ``tmp_path``, and the two
LMDB readers plus ``phenotype_descriptor`` replaced by recorders in the ``signal``
namespace; the two formatters (``human_bytes``, ``scientific``) stay real, so the printed
signal line is worked by hand: 1,234,567 bytes -> 1234.567 KB >= 1000 -> 1.234567 MB ->
``1.2 MB``; scientific: floor(log10) = 6, mantissa 1.2 -> ``1.2×10⁶``. 584 bytes ->
below 1 KB -> ``584 B``; floor(log10 584) = 2, mantissa 5.84 -> ``5.8×10²``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

from torchcell.paper import signal


def _make_lmdb(dir_: Path) -> Path:
    dir_.mkdir(parents=True)
    (dir_ / "data.mdb").write_bytes(b"")
    return dir_


# --- resolve_lmdb -------------------------------------------------------------


def test_relative_arg_joins_data_root_and_appends_processed_lmdb(
    tmp_path: Path,
) -> None:
    """``data/torchcell/x`` -> ``<DATA_ROOT>/data/torchcell/x/processed/lmdb``."""
    want = _make_lmdb(tmp_path / "data/torchcell/x/processed/lmdb")
    assert signal.resolve_lmdb("data/torchcell/x", str(tmp_path), False) == want


def test_lmdb_flag_takes_the_arg_as_the_lmdb_dir(tmp_path: Path) -> None:
    """With ``--lmdb`` nothing is appended, relative or absolute. The absolute case
    passes a nonexistent ``DATA_ROOT``; it would resolve identically without the
    ``os.path.isabs`` branch, because ``Path(root) / "/abs"`` is ``Path("/abs")``, so that
    branch has no separately testable behavior.
    """
    want = _make_lmdb(tmp_path / "ds/processed/lmdb")
    assert signal.resolve_lmdb("ds/processed/lmdb", str(tmp_path), True) == want
    assert signal.resolve_lmdb(str(want), "/nonexistent-root", True) == want


def test_a_dir_without_data_mdb_is_refused_with_the_resolved_path(
    tmp_path: Path,
) -> None:
    """The directory exists but has no ``data.mdb``; with ``--lmdb`` pointing at the
    dataset root instead of its LMDB the message names the root that was checked.
    """
    (tmp_path / "ds/processed/lmdb").mkdir(parents=True)
    with pytest.raises(
        FileNotFoundError,
        match=f"^{re.escape(f'No data.mdb under {tmp_path}/ds/processed/lmdb')}$",
    ):
        signal.resolve_lmdb("ds", str(tmp_path), False)
    _make_lmdb(tmp_path / "ds2/processed/lmdb")
    with pytest.raises(
        FileNotFoundError, match=f"^{re.escape(f'No data.mdb under {tmp_path}/ds2')}$"
    ):
        signal.resolve_lmdb("ds2", str(tmp_path), True)


# --- main ---------------------------------------------------------------------------


def _patch_readers(
    monkeypatch: pytest.MonkeyPatch, n: int, nbytes: int
) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []
    record = {"experiment": {"phenotype": {"graph_level": "global"}}}

    def read_first_record(lmdb_dir: Path) -> dict[str, Any]:
        calls.append(("read_first_record", lmdb_dir))
        return record

    def phenotype_descriptor(rec: dict[str, Any]) -> tuple[str, str]:
        calls.append(("phenotype_descriptor", rec is record))
        return "vector (3)", "edge"

    def stream_gzip_signal(lmdb_dir: Path, *, label: str) -> tuple[int, int]:
        calls.append(("stream_gzip_signal", (lmdb_dir, label)))
        return n, nbytes

    def no_dotenv() -> bool:
        calls.append(("load_dotenv", None))
        return True

    monkeypatch.setattr(signal, "read_first_record", read_first_record)
    monkeypatch.setattr(signal, "phenotype_descriptor", phenotype_descriptor)
    monkeypatch.setattr(signal, "stream_gzip_signal", stream_gzip_signal)
    monkeypatch.setattr(signal, "load_dotenv", no_dotenv)
    return calls


def test_main_prints_the_five_lines_for_a_relative_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dotenv first, then the resolved dir goes to both readers, the label is the raw
    argument, and the signal line reads ``1.2 MB  (1.2×10⁶ bytes, 1,234,567)``.
    """
    lmdb_dir = _make_lmdb(tmp_path / "data/torchcell/ds/processed/lmdb")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["signal", "data/torchcell/ds"])
    calls = _patch_readers(monkeypatch, n=20705612, nbytes=1234567)
    signal.main()
    assert calls == [
        ("load_dotenv", None),
        ("read_first_record", lmdb_dir),
        ("phenotype_descriptor", True),
        ("stream_gzip_signal", (lmdb_dir, "data/torchcell/ds")),
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"lmdb        {lmdb_dir}",
        "records     20,705,612",
        "shape       vector (3)",
        "graph role  edge",
        "signal      1.2 MB  (1.2×10⁶ bytes, 1,234,567)",
    ]


def test_main_with_lmdb_flag_and_a_sub_kilobyte_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--lmdb`` on an absolute LMDB dir; 584 bytes prints ``584 B  (5.8×10² bytes, 584)``."""
    lmdb_dir = _make_lmdb(tmp_path / "x/processed/lmdb")
    monkeypatch.setenv("DATA_ROOT", "/nonexistent-root")
    monkeypatch.setattr(sys, "argv", ["signal", str(lmdb_dir), "--lmdb"])
    calls = _patch_readers(monkeypatch, n=17, nbytes=584)
    signal.main()
    assert calls == [
        ("load_dotenv", None),
        ("read_first_record", lmdb_dir),
        ("phenotype_descriptor", True),
        ("stream_gzip_signal", (lmdb_dir, str(lmdb_dir))),
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"lmdb        {lmdb_dir}",
        "records     17",
        "shape       vector (3)",
        "graph role  edge",
        "signal      584 B  (5.8×10² bytes, 584)",
    ]


def test_main_without_data_root_raises_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DATA_ROOT`` is read with ``os.environ[...]``: unset (and no .env), a KeyError,
    and no reader is called.
    """
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", ["signal", "ds"])
    calls = _patch_readers(monkeypatch, n=0, nbytes=0)
    with pytest.raises(KeyError, match=r"^'DATA_ROOT'$"):
        signal.main()
    assert calls == [("load_dotenv", None)]


def test_main_without_a_dataset_argument_exits_with_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Argparse exits 2 naming the missing positional; nothing else runs."""
    monkeypatch.setattr(sys, "argv", ["signal"])
    calls = _patch_readers(monkeypatch, n=0, nbytes=0)
    with pytest.raises(SystemExit) as exc:
        signal.main()
    assert exc.value.code == 2
    assert capsys.readouterr().err.splitlines()[-1] == (
        "signal: error: the following arguments are required: dataset"
    )
    assert calls == []
