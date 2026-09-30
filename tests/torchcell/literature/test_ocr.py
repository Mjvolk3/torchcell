# tests/torchcell/literature/test_ocr.py
# [[tests.torchcell.literature.test_ocr]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_ocr.py
"""The MinerU OCR driver with the subprocess faked: exact command lines and refusals.

Fixture: ``ocr.subprocess`` is replaced by a namespace whose ``run`` records every call
(the command list and its keyword arguments, including a snapshot of the ``env`` it was
handed) and returns a real ``subprocess.CompletedProcess`` with a scripted return code
and stderr. On a zero exit the fake writes ``<stem>.md`` next to the PDF, as the
standalone runner ``_run_mineru.py`` does. MinerU, its conda env, HuggingFace and the
GPU are never touched; the PDFs are empty files in ``tmp_path``. Every MinerU and HF
variable is cleared first so the developer's shell cannot leak in.

Expected command (``ocr.py`` lines 93 to 108), for ``<dir>/paper.pdf``::

    [<mineru python>, <ocr.py dir>/_run_mineru.py,
     --pdf-path <dir>/paper.pdf, --out-dir <dir>,
     --backend pipeline, --lang en, --method auto, --dpi 0]

with ``env`` = the parent environment plus ``MINERU_MODEL_SOURCE=huggingface``,
``MINERU_DEVICE_MODE`` (argument, else ``$MINERU_DEVICE_MODE``, else ``cuda``) and
``HF_HOME`` (``$HF_HOME``, else ``$DATA_ROOT/models/mineru/hf_cache``, else unset), and
``capture_output=True, text=True, timeout=3600``. DPI: argument, else
``$TORCHCELL_MINERU_DPI``, else 0 (MinerU's own 200).

Findings pinned here: no local check that the PDF exists (the refusal is the runner's
exit 2); nothing records the MinerU version, arguments or DPI (no sidecar, a bare path
back), against the provenance rule; ``ocr_artifact`` orders SI files lexicographically
(``si10`` before ``si2``) and returns an empty list for a directory with no PDFs; the
processor that Mormino 2022's OCR provenance names does not exist in this module.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import torchcell.literature.ocr as ocr

RUNNER = str(Path(ocr.__file__).resolve().parent / "_run_mineru.py")
_MINERU_VARS = (
    "HF_HOME",
    "MINERU_DEVICE_MODE",
    "MINERU_MODEL_SOURCE",
    "TORCHCELL_MINERU_DPI",
    "TORCHCELL_MINERU_PYTHON",
)


class _FakeRun:
    """Records each ``subprocess.run`` call and plays back scripted outcomes.

    ``outcomes`` maps a PDF name to ``(returncode, stderr)``; a name not listed
    succeeds. A zero exit writes ``<stem>.md`` with ``markdown`` unless ``write_md``
    is False.
    """

    def __init__(
        self,
        outcomes: dict[str, tuple[int, str]] | None = None,
        write_md: bool = True,
        markdown: str = "# Title\nabc",
    ) -> None:
        self.outcomes = outcomes or {}
        self.write_md = write_md
        self.markdown = markdown
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(
        self, cmd: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        kwargs = {**kwargs, "env": dict(kwargs["env"])}
        self.calls.append((list(cmd), kwargs))
        pdf = Path(cmd[cmd.index("--pdf-path") + 1])
        returncode, stderr = self.outcomes.get(pdf.name, (0, ""))
        if returncode == 0 and self.write_md:
            pdf.with_suffix(".md").write_text(self.markdown)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in _MINERU_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data-root"))


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakeRun) -> _FakeRun:
    monkeypatch.setattr(ocr, "subprocess", SimpleNamespace(run=fake))
    return fake


def _pdf(directory: Path, name: str = "paper.pdf") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"%PDF-1.4\n")
    return path


def _tail(pdf: Path, backend: str, lang: str, method: str, dpi: str) -> list[str]:
    return [
        RUNNER,
        "--pdf-path",
        str(pdf),
        "--out-dir",
        str(pdf.parent),
        "--backend",
        backend,
        "--lang",
        lang,
        "--method",
        method,
        "--dpi",
        dpi,
    ]


# ------------------------------------------------------------------ ocr_pdf


def test_default_call_is_the_exact_command_env_and_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No arguments and no MinerU variables: the default interpreter, pipeline / en /
    auto, DPI "0", device cuda, HF_HOME derived from DATA_ROOT, one hour timeout. The
    parent environment is copied, not mutated: a marker variable reaches the child and
    ``MINERU_MODEL_SOURCE`` never appears in ``os.environ``.
    """
    fake = _install(monkeypatch, _FakeRun())
    monkeypatch.setenv("TC_OCR_MARKER", "kept")
    pdf = _pdf(tmp_path / "ck")

    md = ocr.ocr_pdf(pdf)

    assert md == tmp_path / "ck" / "paper.md"
    ((cmd, kwargs),) = fake.calls
    assert cmd == [
        ocr.DEFAULT_MINERU_PYTHON,
        *_tail(pdf, "pipeline", "en", "auto", "0"),
    ]
    env = kwargs.pop("env")
    assert kwargs == {"capture_output": True, "text": True, "timeout": 3600}
    assert env["HF_HOME"] == f"{tmp_path}/data-root/models/mineru/hf_cache"
    assert env["MINERU_MODEL_SOURCE"] == "huggingface"
    assert env["MINERU_DEVICE_MODE"] == "cuda"
    assert env["TC_OCR_MARKER"] == "kept"
    assert "MINERU_MODEL_SOURCE" not in os.environ
    assert "HF_HOME" not in os.environ


def test_arguments_reach_the_command_line_and_beat_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every keyword lands in its flag. An explicit ``dpi=0`` beats
    ``$TORCHCELL_MINERU_DPI=275`` (``is not None``, not truthiness), and
    ``device_mode="cpu"`` beats ``$MINERU_DEVICE_MODE=cuda:1``.
    """
    fake = _install(monkeypatch, _FakeRun())
    monkeypatch.setenv("TORCHCELL_MINERU_PYTHON", "/opt/mineru/bin/python")
    monkeypatch.setenv("TORCHCELL_MINERU_DPI", "275")
    monkeypatch.setenv("MINERU_DEVICE_MODE", "cuda:1")
    pdf = _pdf(tmp_path / "ck", "si1.pdf")

    ocr.ocr_pdf(
        str(pdf),
        backend="vlm-auto-engine",
        lang="ch",
        method="ocr",
        device_mode="cpu",
        dpi=0,
        timeout=60,
    )

    ((cmd, kwargs),) = fake.calls
    assert cmd == [
        "/opt/mineru/bin/python",
        *_tail(pdf, "vlm-auto-engine", "ch", "ocr", "0"),
    ]
    assert kwargs["timeout"] == 60
    assert kwargs["env"]["MINERU_DEVICE_MODE"] == "cpu"


def test_environment_defaults_apply_when_arguments_are_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``$TORCHCELL_MINERU_DPI=350`` gives ``--dpi 350``; ``$MINERU_DEVICE_MODE=cpu``
    gives device cpu; an explicit ``$HF_HOME`` wins over the DATA_ROOT derivation.
    """
    fake = _install(monkeypatch, _FakeRun())
    monkeypatch.setenv("TORCHCELL_MINERU_DPI", "350")
    monkeypatch.setenv("MINERU_DEVICE_MODE", "cpu")
    monkeypatch.setenv("HF_HOME", "/models/hf")
    pdf = _pdf(tmp_path / "ck")

    assert ocr.ocr_pdf(pdf) == tmp_path / "ck" / "paper.md"

    ((cmd, kwargs),) = fake.calls
    assert cmd[-2:] == ["--dpi", "350"]
    assert kwargs["env"]["MINERU_DEVICE_MODE"] == "cpu"
    assert kwargs["env"]["HF_HOME"] == "/models/hf"


@pytest.mark.parametrize(
    ("hf_home", "data_root", "expected"),
    [
        ("/models/hf", "/d", "/models/hf"),
        ("", "/d", "/d/models/mineru/hf_cache"),  # empty HF_HOME counts as unset
        (None, "/d/", "/d/models/mineru/hf_cache"),  # Path drops the trailing slash
        (None, None, None),
    ],
)
def test_hf_home_resolution_order(
    monkeypatch: pytest.MonkeyPatch,
    hf_home: str | None,
    data_root: str | None,
    expected: str | None,
) -> None:
    for name, value in (("HF_HOME", hf_home), ("DATA_ROOT", data_root)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert ocr._hf_home() == expected


def test_with_no_hf_home_and_no_data_root_the_child_gets_no_hf_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The driver does not refuse; it leaves ``HF_HOME`` out of the child's
    environment, and the runner's own ``_ensure_hf_home`` is what exits 4.
    """
    fake = _install(monkeypatch, _FakeRun())
    monkeypatch.delenv("DATA_ROOT")
    assert ocr.ocr_pdf(_pdf(tmp_path / "ck")) == tmp_path / "ck" / "paper.md"
    ((_, kwargs),) = fake.calls
    assert "HF_HOME" not in kwargs["env"]


def test_nonzero_exit_raises_with_the_last_2000_characters_of_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stderr is 1000 "A" then 2000 "B": the message keeps only the B tail."""
    stderr = "A" * 1000 + "B" * 2000
    _install(monkeypatch, _FakeRun(outcomes={"paper.pdf": (3, stderr)}))
    with pytest.raises(RuntimeError) as refused:
        ocr.ocr_pdf(_pdf(tmp_path / "ck"))
    assert str(refused.value) == "MinerU failed (exit 3) on paper.pdf:\n" + "B" * 2000


def test_a_missing_pdf_is_not_checked_before_the_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding: ``ocr_pdf`` never checks that the PDF exists (ocr.py lines 82 to 110),
    so a missing file still costs a MinerU-env interpreter start; the refusal is the
    runner's exit 2 and its stderr, relayed. Pinned until the driver checks first.
    """
    stderr = f"ERROR: PDF not found: {tmp_path}/ck/absent.pdf\n"
    fake = _install(monkeypatch, _FakeRun(outcomes={"absent.pdf": (2, stderr)}))
    absent = tmp_path / "ck" / "absent.pdf"
    with pytest.raises(RuntimeError) as refused:
        ocr.ocr_pdf(absent)
    assert str(refused.value) == f"MinerU failed (exit 2) on absent.pdf:\n{stderr}"
    assert [cmd[3] for cmd, _ in fake.calls] == [str(absent)]


def test_success_without_the_markdown_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, _FakeRun(write_md=False))
    pdf = _pdf(tmp_path / "ck")
    with pytest.raises(RuntimeError) as refused:
        ocr.ocr_pdf(pdf)
    assert str(refused.value) == (
        f"MinerU reported success but {tmp_path}/ck/paper.md is missing"
    )


def test_a_non_integer_dpi_variable_refuses_before_the_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _install(monkeypatch, _FakeRun())
    monkeypatch.setenv("TORCHCELL_MINERU_DPI", "350dpi")
    with pytest.raises(ValueError) as refused:
        ocr.ocr_pdf(_pdf(tmp_path / "ck"))
    assert str(refused.value) == "invalid literal for int() with base 10: '350dpi'"
    assert fake.calls == []


def test_a_timeout_propagates_unwrapped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hung MinerU is not turned into a skip or a RuntimeError: the
    ``TimeoutExpired`` from ``subprocess.run`` reaches the caller with its timeout.
    """

    def hang(cmd: list[str], **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(ocr, "subprocess", SimpleNamespace(run=hang))
    with pytest.raises(subprocess.TimeoutExpired) as expired:
        ocr.ocr_pdf(_pdf(tmp_path / "ck"), timeout=5)
    assert expired.value.timeout == 5


def test_log_lines_name_the_device_and_the_markdown_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The markdown is "# Title", a newline, "abc": 7 + 1 + 3 = 11 bytes."""
    _install(monkeypatch, _FakeRun(markdown="# Title\nabc"))
    pdf = _pdf(tmp_path / "ck")
    with caplog.at_level(logging.INFO, logger=ocr.log.name):
        ocr.ocr_pdf(pdf, device_mode="cpu")
    assert [r.getMessage() for r in caplog.records if r.name == ocr.log.name] == [
        "MinerU: OCR paper.pdf (device=cpu)",
        f"MinerU: wrote {tmp_path}/ck/paper.md (11 bytes)",
    ]


def test_no_version_or_dpi_record_is_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding: CLAUDE.md's provenance rule asks every OCR artifact to record
    ``mineru_version`` plus the arguments and DPI, but ``ocr_pdf`` returns a bare
    ``Path`` and writes nothing besides what the runner writes: the directory holds
    only the PDF and the markdown, and the command carries no version (the MinerU env
    is whatever the interpreter path resolves to). Pinned until the driver records a
    ``ProcessingRecord`` itself.
    """
    _install(monkeypatch, _FakeRun())
    pdf = _pdf(tmp_path / "ck")
    assert ocr.ocr_pdf(pdf, dpi=350) == tmp_path / "ck" / "paper.md"
    assert sorted(p.name for p in (tmp_path / "ck").iterdir()) == [
        "paper.md",
        "paper.pdf",
    ]


def test_the_processor_named_by_mormino_2022_does_not_exist() -> None:
    """Finding: Mormino 2022's OCR ``ProcessingRecord`` (mormino2022.py line 434)
    names ``torchcell.literature.ocr.run_mineru``; the module's entry points are
    ``ocr_pdf`` and ``ocr_artifact``, so that dotted path does not resolve. Pinned until
    the record names ``ocr_pdf``.
    """
    import torchcell.datasets.scerevisiae.mormino2022 as mormino

    source = Path(mormino.__file__).read_text()
    (processor,) = re.findall(
        r'processor="(torchcell\.literature\.ocr\.[^"]+)"', source
    )
    module_name, _, attribute = processor.rpartition(".")
    with pytest.raises(AttributeError) as missing:
        getattr(importlib.import_module(module_name), attribute)
    assert str(missing.value) == (
        "module 'torchcell.literature.ocr' has no attribute 'run_mineru'"
    )


# ------------------------------------------------------------- ocr_artifact


def _artifact(root: Path) -> Path:
    """paper.pdf, five SI-shaped files and three decoys."""
    _pdf(root, "paper.pdf")
    for name in ("si1.pdf", "si2.pdf", "si10.pdf", "table.pdf", "SI3.pdf"):
        _pdf(root / "si", name)
    _pdf(root / "si" / "nested", "si4.pdf")
    _pdf(root, "si5.pdf")  # top-level si file: not under si/
    (root / "si" / "si6.md").write_text("not a pdf")
    return root


def test_ocr_artifact_runs_paper_then_si_pdfs_in_lexicographic_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding: ``sorted(si_dir.glob("si*.pdf"))`` (ocr.py line 135) is lexicographic,
    so ``si10`` runs and is returned before ``si2``; the docstring's "si1.md..." reads
    as numeric. Pinned until the sort is natural. The glob is case-sensitive and not
    recursive (``SI3.pdf``, ``nested/si4.pdf``, ``table.pdf`` and the top-level
    ``si5.pdf`` are skipped), and the keywords reach every call (``--dpi 350``).
    """
    fake = _install(monkeypatch, _FakeRun())
    root = _artifact(tmp_path / "ck")

    produced = ocr.ocr_artifact(root, dpi=350, backend="vlm-auto-engine")

    assert produced == [
        root / "paper.md",
        root / "si" / "si1.md",
        root / "si" / "si10.md",
        root / "si" / "si2.md",
    ]
    assert [(cmd[3], cmd[5], cmd[7], cmd[-1]) for cmd, _ in fake.calls] == [
        (str(root / "paper.pdf"), str(root), "vlm-auto-engine", "350"),
        (str(root / "si" / "si1.pdf"), str(root / "si"), "vlm-auto-engine", "350"),
        (str(root / "si" / "si10.pdf"), str(root / "si"), "vlm-auto-engine", "350"),
        (str(root / "si" / "si2.pdf"), str(root / "si"), "vlm-auto-engine", "350"),
    ]


def test_ocr_artifact_without_a_paper_is_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding: a directory with no ``paper.pdf`` is not refused: an SI-only artifact
    returns just the SI markdown, and an empty directory (or one whose ``si`` is a
    file) returns ``[]`` without starting MinerU. Pinned until a missing paper
    refuses.
    """
    fake = _install(monkeypatch, _FakeRun())
    si_only = tmp_path / "si-only"
    _pdf(si_only / "si", "si1.pdf")
    assert ocr.ocr_artifact(si_only) == [si_only / "si" / "si1.md"]

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "si").write_text("a file, not a directory")
    assert ocr.ocr_artifact(empty) == []
    assert [cmd[3] for cmd, _ in fake.calls] == [str(si_only / "si" / "si1.pdf")]


def test_ocr_artifact_stops_at_the_first_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """si1 exits 1: the paper already ran (its markdown is on disk), si1's error is
    raised, si10 and si2 never start, and the list of what was produced is lost.
    """
    fake = _install(monkeypatch, _FakeRun(outcomes={"si1.pdf": (1, "boom")}))
    root = _artifact(tmp_path / "ck")
    with pytest.raises(RuntimeError) as refused:
        ocr.ocr_artifact(root)
    assert str(refused.value) == "MinerU failed (exit 1) on si1.pdf:\nboom"
    assert [Path(cmd[3]).name for cmd, _ in fake.calls] == ["paper.pdf", "si1.pdf"]
    assert (root / "paper.md").read_text() == "# Title\nabc"
