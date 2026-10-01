# tests/scripts/test_lit_reocr_si.py
# [[tests.scripts.test_lit_reocr_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/scripts/test_lit_reocr_si.py
"""``scripts/lit_reocr_si.py`` on a synthetic mirror: order, refusals, retirement.

The script is loaded from its file. ``ocr_pdf`` is replaced by a fake that does what
the runner does for an SI PDF (writes ``<stem>.md`` referencing
``images/<stem>/<stem>.jpg`` and that figure), and the Zotero side
(``ZoteroLibrary.from_env``, ``build_citation_index``, ``backfill_key``) by recorders,
so no MinerU, network or Zotero is touched. ``retire_flat_figures`` runs the real
``scripts/deprecate.sh`` into a graveyard under ``tmp_path``.

Contract: SI PDFs run in natural order across every key before anything else; a key
with no SI PDF refuses before any OCR; an unresolved figure reference after the OCR
refuses with nothing retired and no manifest written; the flat pre-#579
``si/images/<file>`` figures go to the graveyard under a per-key name; then each key
is backfilled with ``force=True``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from torchcell.literature.backfill import KeyBackfillResult

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lit_reocr_si.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lit_reocr_si", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reocr = _load()


def _key(root: Path, name: str, si_names: list[str], flat: list[str]) -> Path:
    key = root / name
    (key / "si" / "images").mkdir(parents=True)
    (key / "paper.pdf").write_bytes(b"%PDF paper")
    for si in si_names:
        (key / "si" / si).write_bytes(b"%PDF si")
    for fig in flat:
        (key / "si" / "images" / fig).write_bytes(b"old")
    return key


class _Recorder:
    def __init__(self, fail_on: str | None = None, write_figure: bool = True) -> None:
        self.events: list[str] = []
        self.fail_on = fail_on
        self.write_figure = write_figure

    def ocr_pdf(self, pdf: Path, *, device_mode: str) -> Path:
        self.events.append(f"ocr {pdf.parent.parent.name}/{pdf.name} {device_mode}")
        if pdf.name == self.fail_on:
            raise RuntimeError(f"MinerU failed on {pdf.name}")
        figure = pdf.parent / "images" / pdf.stem / f"{pdf.stem}.jpg"
        if self.write_figure:
            figure.parent.mkdir(parents=True, exist_ok=True)
            figure.write_bytes(b"new")
        md = pdf.with_suffix(".md")
        md.write_text(f"![](images/{pdf.stem}/{pdf.stem}.jpg)\n")
        return md

    def backfill_key(self, key_dir: Path, **kwargs: Any) -> Any:
        self.events.append(
            f"backfill {key_dir.name} force={kwargs['force']} index={kwargs['citation_index']}"
        )
        return KeyBackfillResult(citation_key=key_dir.name, mode="enriched", n_files=1)


def _install(monkeypatch: pytest.MonkeyPatch, rec: _Recorder) -> None:
    monkeypatch.setattr(reocr, "ocr_pdf", rec.ocr_pdf)
    monkeypatch.setattr(reocr, "backfill_key", rec.backfill_key)
    monkeypatch.setattr(reocr.ZoteroLibrary, "from_env", classmethod(lambda cls: "LIB"))
    monkeypatch.setattr(reocr, "build_citation_index", lambda lib: {"lib": lib})


def test_si_pdfs_are_natural_and_a_key_without_any_refuses(tmp_path: Path) -> None:
    key = _key(tmp_path, "k", ["si10.pdf", "si2.pdf", "si1.pdf", "table.pdf"], [])
    assert [p.name for p in reocr.si_pdfs(key)] == ["si1.pdf", "si2.pdf", "si10.pdf"]
    empty = _key(tmp_path, "empty", [], [])
    with pytest.raises(reocr.NoSiPdfError) as refused:
        reocr.si_pdfs(empty)
    assert str(refused.value) == f"no si/si*.pdf under {empty}"


def test_flat_figures_are_only_files_directly_in_si_images(tmp_path: Path) -> None:
    key = _key(tmp_path, "k", ["si1.pdf"], ["b.jpg", "a.jpg"])
    (key / "si" / "images" / "si1").mkdir()
    (key / "si" / "images" / "si1" / "c.jpg").write_bytes(b"x")
    assert [p.name for p in reocr.flat_si_figures(key)] == ["a.jpg", "b.jpg"]
    assert reocr.flat_si_figures(_key(tmp_path, "none", [], [])) == []


def test_unresolved_refs_name_the_markdown_and_the_reference(tmp_path: Path) -> None:
    """Markdown, HTML and content-list forms are read from ``si/*.md`` only; a prose
    URL is not a reference.
    """
    key = _key(tmp_path, "k", [], [])
    (key / "si" / "images" / "si1").mkdir()
    (key / "si" / "images" / "si1" / "ok.jpg").write_bytes(b"x")
    (key / "si" / "si1.md").write_text(
        "![](images/si1/ok.jpg)\n![](images/si1/gone.jpg)\n"
        "see https://example.org/images/logo.png\n"
    )
    (key / "si" / "si10.md").write_text('<img src="images/si10/x.png">\n')
    (key / "si" / "si2.md").write_text("no figures\n")
    assert reocr.unresolved_si_figure_refs(key) == [
        "si1.md: images/si1/gone.jpg",
        "si10.md: images/si10/x.png",
    ]


def test_full_run_order_retirement_and_backfill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two keys: every SI PDF of both is OCR'd (natural order, device passed) before
    the flat figures of ``lee`` are retired to the graveyard under a per-key name with
    a DEPRECATION.txt naming the staging path; ``ohya`` has no flat figure and
    nothing is retired for it; both are backfilled with ``force=True``.
    """
    root = tmp_path / "torchcell-library"
    lee = _key(root, "lee", ["si1.pdf", "si2.pdf"], ["f1.jpg", "f2.jpg"])
    _key(root, "ohya", ["si10.pdf", "si2.pdf"], [])
    graveyard = tmp_path / "graveyard"
    rec = _Recorder()
    _install(monkeypatch, rec)

    reocr.reocr_keys(root, ["lee", "ohya"], graveyard=str(graveyard), device_mode="cpu")

    assert rec.events == [
        "ocr lee/si1.pdf cpu",
        "ocr lee/si2.pdf cpu",
        "ocr ohya/si2.pdf cpu",
        "ocr ohya/si10.pdf cpu",
        "backfill lee force=True index={'lib': 'LIB'}",
        "backfill ohya force=True index={'lib': 'LIB'}",
    ]
    assert sorted(p.name for p in (lee / "si" / "images").iterdir()) == ["si1", "si2"]
    (entry,) = list(graveyard.iterdir())
    assert entry.name.endswith("__lee__si-images-flat-pre579")
    moved = entry / "lee__si-images-flat-pre579"
    assert sorted(p.name for p in moved.iterdir()) == ["f1.jpg", "f2.jpg"]
    record = (entry / "DEPRECATION.txt").read_text().splitlines()
    assert record[0] == f"original_path: {lee}/si/lee__si-images-flat-pre579"
    assert record[-1] == f"reason: {reocr.RETIRE_REASON}"
    assert not (lee / "si" / "lee__si-images-flat-pre579").exists()


def test_an_ocr_failure_retires_nothing_and_writes_no_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "torchcell-library"
    lee = _key(root, "lee", ["si1.pdf", "si2.pdf"], ["f1.jpg"])
    rec = _Recorder(fail_on="si2.pdf")
    _install(monkeypatch, rec)
    with pytest.raises(RuntimeError, match="MinerU failed on si2.pdf"):
        reocr.reocr_keys(
            root, ["lee"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert rec.events == ["ocr lee/si1.pdf cuda", "ocr lee/si2.pdf cuda"]
    assert (lee / "si" / "images" / "f1.jpg").read_bytes() == b"old"
    assert not (tmp_path / "g").exists()


def test_an_unresolved_reference_refuses_before_retiring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "torchcell-library"
    lee = _key(root, "lee", ["si1.pdf"], ["f1.jpg"])
    rec = _Recorder(write_figure=False)
    _install(monkeypatch, rec)
    with pytest.raises(reocr.UnresolvedFigureError) as refused:
        reocr.reocr_keys(
            root, ["lee"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert str(refused.value) == (
        "1 SI figure reference(s) unresolved after re-OCR; nothing retired, no "
        "manifest written: lee/si/si1.md: images/si1/si1.jpg"
    )
    assert rec.events == ["ocr lee/si1.pdf cuda"]
    assert (lee / "si" / "images" / "f1.jpg").read_bytes() == b"old"


def test_main_reads_keys_and_root_from_the_command_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, list[str], str, str]] = []
    monkeypatch.setattr(reocr, "load_dotenv", lambda path: None)
    monkeypatch.setattr(
        reocr,
        "reocr_keys",
        lambda root, keys, *, graveyard, device_mode: calls.append(
            (root, keys, graveyard, device_mode)
        ),
    )
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert reocr.main(["--key", "a", "--key", "b"]) == 0
    assert reocr.main(["--key", "c", "--root", "/m", "--device-mode", "cpu"]) == 0
    assert calls == [
        (tmp_path / "torchcell-library", ["a", "b"], reocr.DEFAULT_GRAVEYARD, "cuda"),
        (Path("/m"), ["c"], reocr.DEFAULT_GRAVEYARD, "cpu"),
    ]
