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
``images/`` and its references unchanged, and its re-run never touches another root
PDF's ``images/<stem>/``. A reference to a figure MinerU did not write exits 5 before
anything is written and leaves no scratch. The runner prints ``MINERU_VERSION`` and
the effective ``MINERU_DPI``.

2026.10.01 (PR #585 review): every run starts from a fresh scratch and writes the
markdown only after the figures are in place, so a crash in the copy or between the
moves leaves the previous markdown with its figures, and the next run succeeds.
References are rewritten only in MinerU's three forms (``](images/``,
``<img src="images/``, ``"img_path": "images/``); prose URLs pass through.
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
    the exact stderr line; the PDF's directory gains no markdown, JSON, figure or
    scratch directory.
    """
    _install(monkeypatch, {"si1": ["a1.jpg"]}, extra_refs=["ghost.jpg"])
    si = tmp_path / "ck" / "si"
    si1 = _pdf(si, "si1.pdf")
    assert _run(monkeypatch, si1, "--images-dir", "images/si1") == 5
    assert capsys.readouterr().err == (
        "ERROR: si1.md references images/ghost.jpg, not produced\n"
    )
    assert sorted(p.name for p in si.iterdir()) == ["si1.pdf"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("![](images/a.jpg)", "![](x/y/a.jpg)"),
        ('<td><img src="images/b.jpeg"/></td>', '<td><img src="x/y/b.jpeg"/></td>'),
        ('{"img_path": "images/a.jpg"}', '{"img_path": "x/y/a.jpg"}'),
        ("![](images/A B.PNG)", "![](x/y/A B.PNG)"),
        (
            "see https://example.org/images/logo.png and images/a.jpg",
            "see https://example.org/images/logo.png and images/a.jpg",
        ),
        ("prose images/figures stay", "prose images/figures stay"),
    ],
)
def test_rewrite_image_refs_touches_only_mineru_reference_forms(
    text: str, expected: str
) -> None:
    """Only MinerU's markdown image, HTML ``<img src=`` and content-list
    ``"img_path"`` forms are references, whatever the name's case, spaces or
    extension; a prose URL or bare ``images/...`` passes through unchecked.
    """
    produced = {"a.jpg", "b.jpeg", "A B.PNG"}
    assert runner._rewrite_image_refs(text, produced, "x/y") == expected


def test_rewrite_image_refs_refuses_an_unknown_figure() -> None:
    rewrite: Callable[[str, set[str], str], str] = runner._rewrite_image_refs
    with pytest.raises(runner.UnresolvedImageRefError) as refused:
        rewrite("![](images/a.jpg)", {"b.jpg"}, "images/si1")
    assert str(refused.value) == "references images/a.jpg, not produced"


def test_dpi_patch_reads_module_dicts_and_never_triggers_lazy_attributes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``_patch_dpi`` finds the modules that imported ``load_images_from_pdf`` by
    name through their ``__dict__``: a lazy module in ``sys.modules`` whose
    ``__getattr__`` would import heavy optional dependencies (transformers does this,
    reaching torchvision; CI failure on PR #585) is never asked for the attribute,
    and a ``None`` entry (an import blocked in ``sys.modules``) is skipped.
    An importer that bound the function by name is patched to force the DPI.
    """
    fake = _install(monkeypatch, {})

    class _Lazy(ModuleType):
        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError("No module named 'torchvision'")

    monkeypatch.setitem(sys.modules, "lazy_heavy", _Lazy("lazy_heavy"))
    monkeypatch.setitem(sys.modules, "blocked_import", None)
    importer = ModuleType("mineru_importer")
    original = sys.modules["mineru.utils.pdf_image_tools"].load_images_from_pdf
    importer.load_images_from_pdf = original  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mineru_importer", importer)

    runner._patch_dpi(350)

    assert capsys.readouterr().out == "[mineru] page rasterization DPI -> 350\n"
    patched = vars(importer)["load_images_from_pdf"]
    assert patched is not original
    assert sys.modules["mineru.utils.pdf_image_tools"].load_images_from_pdf is patched
    patched(b"")
    assert fake.dpis == [350]


def test_a_rerun_after_a_failed_copy_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The second run of si1 dies in the figure copy (``OSError``): ``si1.md`` still
    references the old figure, which is still in place. The third run succeeds with
    exactly the new figures and matching references, and no scratch is left.
    """
    figures = {"si1": ["old.jpg"]}
    _install(monkeypatch, figures)
    si = tmp_path / "ck" / "si"
    si1 = _pdf(si, "si1.pdf")
    assert _run(monkeypatch, si1, "--images-dir", "images/si1") == 0

    figures["si1"] = ["new1.jpg", "new2.jpg"]

    def broken_copytree(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr("shutil.copytree", broken_copytree)
        with pytest.raises(OSError, match="disk full"):
            _run(monkeypatch, si1, "--images-dir", "images/si1")
    assert (si / "si1.md").read_text() == "# si1\n![](images/si1/old.jpg)\n"
    assert _tree(si / "images") == ["si1/old.jpg"]

    assert _run(monkeypatch, si1, "--images-dir", "images/si1") == 0
    assert _tree(si / "images") == ["si1/new1.jpg", "si1/new2.jpg"]
    assert (si / "si1.md").read_text() == (
        "# si1\n![](images/si1/new1.jpg)\n![](images/si1/new2.jpg)\n"
    )
    assert sorted(p.name for p in si.iterdir()) == [
        "images",
        "si1.md",
        "si1.pdf",
        "si1_content_list.json",
        "si1_middle.json",
    ]


def test_a_rerun_after_a_kill_between_the_moves_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The state a kill between the two moves leaves: the old figure moved into the
    scratch's ``.images.old``, the new one staged in ``.images.new``, nothing in
    ``images/si1``. The next run discards that scratch and ends with exactly its
    figures, references and no scratch.
    """
    figures = {"si1": ["old.jpg"]}
    _install(monkeypatch, figures)
    si = tmp_path / "ck" / "si"
    si1 = _pdf(si, "si1.pdf")
    assert _run(monkeypatch, si1, "--images-dir", "images/si1") == 0
    scratch = si / ".mineru_scratch_si1"
    (scratch / ".images.old").mkdir(parents=True)
    (scratch / ".images.new").mkdir()
    (si / "images" / "si1" / "old.jpg").rename(scratch / ".images.old" / "old.jpg")
    (scratch / ".images.new" / "half.jpg").write_bytes(b"half")

    figures["si1"] = ["new.jpg"]
    assert _run(monkeypatch, si1, "--images-dir", "images/si1") == 0
    assert _tree(si / "images") == ["si1/new.jpg"]
    assert (si / "si1.md").read_text() == "# si1\n![](images/si1/new.jpg)\n"
    assert not scratch.exists()


def test_a_paper_rerun_keeps_another_root_pdfs_figures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Costanzo 2016 shape: ``SOM.pdf`` at the key root writes ``images/SOM/``
    inside the paper's flat ``images/``. A ``paper.pdf`` run then a re-run replace
    only the paper's flat files; ``images/SOM/`` is byte for byte untouched.
    """
    figures = {"SOM": ["s1.jpg"], "paper": ["p1.jpg"]}
    _install(monkeypatch, figures)
    root = tmp_path / "ck"
    som, paper = _pdf(root, "SOM.pdf"), _pdf(root, "paper.pdf")
    assert _run(monkeypatch, paper) == 0
    assert _run(monkeypatch, som, "--images-dir", "images/SOM") == 0
    figures["paper"] = ["p2.jpg"]
    assert _run(monkeypatch, paper) == 0

    assert _tree(root / "images") == ["SOM/s1.jpg", "p2.jpg"]
    assert (root / "images" / "SOM" / "s1.jpg").read_bytes() == b"SOM:s1.jpg"
    assert (root / "SOM.md").read_text() == "# SOM\n![](images/SOM/s1.jpg)\n"
    assert (root / "paper.md").read_text() == "# paper\n![](images/p2.jpg)\n"

    figures["paper"] = []
    assert _run(monkeypatch, paper) == 0
    assert _tree(root / "images") == ["SOM/s1.jpg"]
