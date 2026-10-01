# tests/torchcell/literature/test_run_mineru.py
# [[tests.torchcell.literature.test_run_mineru]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_run_mineru.py
"""The standalone MinerU runner with MinerU faked: per-PDF figures (issue #579).

Fixture: ``mineru``, ``mineru.cli.common``, ``mineru.utils.pdf_image_tools`` and
``mineru.version`` are fake modules in ``sys.modules``. The fake ``do_parse`` writes the
tree MinerU writes, ``<output_dir>/<stem>/auto/`` holding ``<stem>.md``,
``<stem>_content_list.json``, ``<stem>_middle.json`` and ``images/<file>``, from a
scripted figure list per stem (no ``images/`` at all for an empty list), and records the DPI its page loader is called with
(``load_images_from_pdf`` defaults to 200, as MinerU's does). Its markdown references
each figure as ``![](images/<file>)`` and its content list as
``"img_path": "images/<file>"``; the middle JSON names the bare file. ``HF_HOME`` points
into ``tmp_path``; no model, GPU or network is touched.

Contract: an SI PDF run with ``--images-dir images/<stem>`` writes its figures to
``<out-dir>/images/<stem>/`` and rewrites the markdown and content-list references to
that directory; two SI PDFs sharing ``si/`` both keep their figures. A re-run of one PDF
leaves exactly its new figures in its own directory and touches no sibling (another PDF's
directory, or the flat pre-#579 ``si/images/<file>``). ``paper.pdf`` keeps the flat
``images/`` and its references unchanged. A reference to a figure MinerU did not write
exits 5 before anything is written. The runner prints ``MINERU_VERSION`` and the
effective ``MINERU_DPI``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from torchcell.literature import _run_mineru as runner


class _FakeMinerU:
    """Scripted MinerU: ``figures[stem]`` is the list of figure files to write."""

    def __init__(self, figures: dict[str, list[str]], extra_refs: list[str]) -> None:
        self.figures = figures
        self.extra_refs = extra_refs
        self.dpis: list[int] = []

    def load_images_from_pdf(self, pdf_bytes: bytes, dpi: int = 200) -> list[Any]:
        self.dpis.append(dpi)
        return []

    def do_parse(
        self,
        output_dir: str,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        backend: str,
        parse_method: str,
    ) -> None:
        (stem,) = pdf_file_names
        pit = sys.modules["mineru.utils.pdf_image_tools"]
        pit.load_images_from_pdf(pdf_bytes_list[0])
        auto = Path(output_dir) / stem / "auto"
        auto.mkdir(parents=True)
        names = self.figures[stem]
        if names:
            (auto / "images").mkdir()
        for name in names:
            (auto / "images" / name).write_bytes(f"{stem}:{name}".encode())
        refs = names + self.extra_refs
        (auto / f"{stem}.md").write_text(
            f"# {stem}\n" + "".join(f"![](images/{n})\n" for n in refs)
        )
        (auto / f"{stem}_content_list.json").write_text(
            json.dumps([{"type": "image", "img_path": f"images/{n}"} for n in refs])
        )
        (auto / f"{stem}_middle.json").write_text(
            json.dumps([{"image_path": n} for n in refs])
        )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    figures: dict[str, list[str]],
    extra_refs: list[str] | None = None,
) -> _FakeMinerU:
    fake = _FakeMinerU(figures, extra_refs or [])
    modules: dict[str, ModuleType] = {
        name: ModuleType(name)
        for name in (
            "mineru",
            "mineru.cli",
            "mineru.cli.common",
            "mineru.utils",
            "mineru.utils.pdf_image_tools",
            "mineru.version",
        )
    }
    modules["mineru.cli.common"].do_parse = fake.do_parse  # type: ignore[attr-defined]
    pit = modules["mineru.utils.pdf_image_tools"]
    pit.load_images_from_pdf = fake.load_images_from_pdf  # type: ignore[attr-defined]
    modules["mineru.version"].__version__ = "2.7.6"  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    return fake


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "huggingface")


def _run(monkeypatch: pytest.MonkeyPatch, pdf: Path, *extra: str) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "_run_mineru.py",
            "--pdf-path",
            str(pdf),
            "--out-dir",
            str(pdf.parent),
            *extra,
        ],
    )
    return runner.main()


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def _pdf(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"%PDF-1.4\n")
    return path


def test_two_si_pdfs_both_keep_their_figures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """si1 (two figures) then si2 (one figure) into the same ``si/``: every figure of
    both is on disk under its own directory, each markdown and content list points at
    it, the middle JSON is copied unchanged, and no scratch directory is left.
    """
    fake = _install(monkeypatch, {"si1": ["a1.jpg", "b1.png"], "si2": ["c2.jpg"]})
    si = tmp_path / "ck" / "si"
    si1, si2 = _pdf(si, "si1.pdf"), _pdf(si, "si2.pdf")

    assert _run(monkeypatch, si1, "--images-dir", "images/si1") == 0
    assert _run(monkeypatch, si2, "--images-dir", "images/si2", "--dpi", "350") == 0

    assert _tree(si) == [
        "images/si1/a1.jpg",
        "images/si1/b1.png",
        "images/si2/c2.jpg",
        "si1.md",
        "si1.pdf",
        "si1_content_list.json",
        "si1_middle.json",
        "si2.md",
        "si2.pdf",
        "si2_content_list.json",
        "si2_middle.json",
    ]
    assert (si / "si1.md").read_text() == (
        "# si1\n![](images/si1/a1.jpg)\n![](images/si1/b1.png)\n"
    )
    assert (si / "si2.md").read_text() == "# si2\n![](images/si2/c2.jpg)\n"
    assert json.loads((si / "si1_content_list.json").read_text()) == [
        {"type": "image", "img_path": "images/si1/a1.jpg"},
        {"type": "image", "img_path": "images/si1/b1.png"},
    ]
    assert json.loads((si / "si2_middle.json").read_text()) == [
        {"image_path": "c2.jpg"}
    ]
    assert (si / "images" / "si1" / "a1.jpg").read_bytes() == b"si1:a1.jpg"
    assert (si / "images" / "si2" / "c2.jpg").read_bytes() == b"si2:c2.jpg"
    assert fake.dpis == [200, 350]
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "MINERU_VERSION=2.7.6",
        "MINERU_DPI=200",
        f"OK: si1.pdf -> {si}/si1.md",
        "[mineru] page rasterization DPI -> 350",
        "MINERU_VERSION=2.7.6",
        "MINERU_DPI=350",
        f"OK: si2.pdf -> {si}/si2.md",
    ]


def test_a_rerun_replaces_only_its_own_figures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """si1 is re-run and now yields ``z1.jpg`` only: ``images/si1`` holds exactly that
    file, si2's directory and the flat pre-#579 ``images/legacy.jpg`` are byte for byte
    untouched. A further re-run in which MinerU writes no ``images/`` removes
    ``images/si1`` and still touches neither sibling.
    """
    figures = {"si1": ["a1.jpg", "b1.png"], "si2": ["c2.jpg"]}
    _install(monkeypatch, figures)
    si = tmp_path / "ck" / "si"
    si1, si2 = _pdf(si, "si1.pdf"), _pdf(si, "si2.pdf")
    (si / "images").mkdir(parents=True)
    (si / "images" / "legacy.jpg").write_bytes(b"legacy")
    assert _run(monkeypatch, si1, "--images-dir", "images/si1") == 0
    assert _run(monkeypatch, si2, "--images-dir", "images/si2") == 0

    figures["si1"] = ["z1.jpg"]
    assert _run(monkeypatch, si1, "--images-dir", "images/si1") == 0
    assert _tree(si / "images") == ["legacy.jpg", "si1/z1.jpg", "si2/c2.jpg"]
    assert (si / "si1.md").read_text() == "# si1\n![](images/si1/z1.jpg)\n"
    assert (si / "images" / "si2" / "c2.jpg").read_bytes() == b"si2:c2.jpg"
    assert (si / "images" / "legacy.jpg").read_bytes() == b"legacy"

    figures["si1"] = []
    assert _run(monkeypatch, si1, "--images-dir", "images/si1") == 0
    assert _tree(si / "images") == ["legacy.jpg", "si2/c2.jpg"]
    assert not (si / "images" / "si1").exists()
    assert (si / "si1.md").read_text() == "# si1\n"
    assert sorted(p.name for p in si.iterdir() if p.name.startswith(".")) == []


def test_the_paper_keeps_the_flat_images_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``paper.pdf`` with the default ``--images-dir images``: figures directly under
    ``images/`` and the references exactly as MinerU wrote them. A re-run replaces
    the old figure.
    """
    figures = {"paper": ["p1.jpg"]}
    _install(monkeypatch, figures)
    root = tmp_path / "ck"
    paper = _pdf(root, "paper.pdf")
    assert _run(monkeypatch, paper) == 0
    figures["paper"] = ["p2.jpg"]
    assert _run(monkeypatch, paper) == 0

    assert _tree(root) == [
        "images/p2.jpg",
        "paper.md",
        "paper.pdf",
        "paper_content_list.json",
        "paper_middle.json",
    ]
    assert (root / "paper.md").read_text() == "# paper\n![](images/p2.jpg)\n"
    assert json.loads((root / "paper_content_list.json").read_text()) == [
        {"type": "image", "img_path": "images/p2.jpg"}
    ]


def test_a_reference_to_an_unwritten_figure_exits_5_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The markdown references ``ghost.jpg``, which MinerU did not write: exit 5 with
    the exact stderr line; the PDF's directory gains no markdown, JSON or figure.
    """
    _install(monkeypatch, {"si1": ["a1.jpg"]}, extra_refs=["ghost.jpg"])
    si = tmp_path / "ck" / "si"
    si1 = _pdf(si, "si1.pdf")
    assert _run(monkeypatch, si1, "--images-dir", "images/si1") == 5
    assert capsys.readouterr().err == (
        "ERROR: si1.md references images/ghost.jpg, not produced\n"
    )
    assert sorted(p.name for p in si.iterdir() if not p.name.startswith(".")) == [
        "si1.pdf"
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("![](images/a.jpg) and images/b.jpeg", "![](x/y/a.jpg) and x/y/b.jpeg"),
        ("prose images/figures stay", "prose images/figures stay"),
        ("![](images/a.gif)", "![](images/a.gif)"),
    ],
)
def test_rewrite_image_refs_touches_only_figure_files(text: str, expected: str) -> None:
    """Only ``images/<file>.(jpg|jpeg|png)`` is a figure reference; prose and other
    extensions pass through.
    """
    assert runner._rewrite_image_refs(text, {"a.jpg", "b.jpeg"}, "x/y") == expected


def test_rewrite_image_refs_refuses_an_unknown_figure() -> None:
    rewrite: Callable[[str, set[str], str], str] = runner._rewrite_image_refs
    with pytest.raises(runner.UnresolvedImageRefError) as refused:
        rewrite("![](images/a.jpg)", {"b.jpg"}, "images/si1")
    assert str(refused.value) == "references images/a.jpg, not produced"
