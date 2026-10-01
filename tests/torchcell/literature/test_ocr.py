# tests/torchcell/literature/test_ocr.py
# [[tests.torchcell.literature.test_ocr]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_ocr.py
"""The MinerU OCR driver with the subprocess faked: exact command lines and refusals.

Fixture: ``ocr.subprocess`` is replaced by a namespace whose ``run`` records every call
(the command list and its keyword arguments, including a snapshot of the ``env`` it was
handed) and returns a real ``subprocess.CompletedProcess`` with a scripted return code
and stderr. On a zero exit the fake writes ``<stem>.md`` next to the PDF and prints the
runner's two facts on stdout (``MINERU_VERSION=2.7.6``, ``MINERU_DPI=<n>``), as the
standalone runner ``_run_mineru.py`` does. MinerU, its conda env, HuggingFace and the
GPU are never touched; the PDFs are empty files in ``tmp_path``. Every MinerU and HF
variable is cleared first so the developer's shell cannot leak in.

Expected command, for ``<dir>/paper.pdf``::

    [<mineru python>, <ocr.py dir>/_run_mineru.py,
     --pdf-path <dir>/paper.pdf, --out-dir <dir>,
     --backend pipeline, --lang en, --method auto, --dpi 0, --images-dir images]

and ``--images-dir images/<stem>`` for any other PDF (``si/si1.pdf`` ->
``images/si1``, issue #579).

with ``env`` = the parent environment plus ``MINERU_MODEL_SOURCE=huggingface``,
``MINERU_DEVICE_MODE`` (argument, else ``$MINERU_DEVICE_MODE``, else ``cuda``) and
``HF_HOME`` (``$HF_HOME``, else ``$DATA_ROOT/models/mineru/hf_cache``, else unset), and
``capture_output=True, text=True, timeout=3600``. DPI: argument, else
``$TORCHCELL_MINERU_DPI``, else 0 (MinerU's own 200).

2026.10.01 (issue #546): the five findings pinned here are retired. A missing PDF
raises ``PdfNotFoundError`` before MinerU starts; each OCR writes its
``ProcessingRecord`` (MinerU version and effective DPI as the runner reported them, the
arguments, the exact command, the PDF's sha256) to ``<stem>_ocr_provenance.json``, and
a runner that does not report both facts is refused; ``ocr_artifact`` runs SI files in
natural order (``si2`` before ``si10``) and refuses a directory with no ``paper.pdf``;
Mormino 2022's OCR record names ``ocr_pdf``, which resolves.
"""

from __future__ import annotations

import hashlib
import importlib
import json
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
        stdout: str | None = None,
    ) -> None:
        self.outcomes = outcomes or {}
        self.write_md = write_md
        self.markdown = markdown
        self.stdout = stdout
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
        dpi = int(cmd[cmd.index("--dpi") + 1])
        stdout = (
            self.stdout
            if self.stdout is not None
            else f"MINERU_VERSION=2.7.6\nMINERU_DPI={dpi or 200}\nOK\n"
        )
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=stdout, stderr=stderr
        )


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


def _tail(
    pdf: Path, backend: str, lang: str, method: str, dpi: str, images: str
) -> list[str]:
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
        "--images-dir",
        images,
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
        *_tail(pdf, "pipeline", "en", "auto", "0", "images"),
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
        *_tail(pdf, "vlm-auto-engine", "ch", "ocr", "0", "images/si1"),
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
    assert cmd[-4:] == ["--dpi", "350", "--images-dir", "images"]
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


def test_a_missing_pdf_is_refused_before_the_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A path that is not a file (absent, or a directory) raises ``PdfNotFoundError``
    naming it, and MinerU's interpreter is never started.
    """
    fake = _install(monkeypatch, _FakeRun())
    absent = tmp_path / "ck" / "absent.pdf"
    with pytest.raises(ocr.PdfNotFoundError) as refused:
        ocr.ocr_pdf(absent)
    assert str(refused.value) == f"PDF not found: {absent}"
    directory = tmp_path / "ck" / "dir.pdf"
    directory.mkdir(parents=True)
    with pytest.raises(ocr.PdfNotFoundError) as refused:
        ocr.ocr_pdf(directory)
    assert str(refused.value) == f"PDF not found: {directory}"
    assert fake.calls == []


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


def test_each_ocr_writes_its_processing_record_beside_the_markdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLAUDE.md's provenance rule: every OCR artifact records ``mineru_version``, the
    arguments and the DPI. ``si/si2.pdf`` at ``dpi=350`` writes
    ``si/si2_ocr_provenance.json``, the whole ``ProcessingRecord`` asserted: version
    2.7.6 and DPI 350 as the runner printed them, the requested DPI, backend, lang,
    method, device, figures directory ``images/si2``, the exact command list, and the
    sha256 of the fixture PDF bytes. With ``dpi`` omitted the requested DPI
    is 0 and the recorded effective DPI is the runner's 200.
    """
    fake = _install(monkeypatch, _FakeRun())
    pdf = _pdf(tmp_path / "ck" / "si", "si2.pdf")

    assert ocr.ocr_pdf(pdf, dpi=350, device_mode="cpu") == pdf.with_suffix(".md")

    assert sorted(p.name for p in pdf.parent.iterdir()) == [
        "si2.md",
        "si2.pdf",
        "si2_ocr_provenance.json",
    ]
    ((cmd, _),) = fake.calls
    record = json.loads((pdf.parent / "si2_ocr_provenance.json").read_text())
    assert record == {
        "processor": "torchcell.literature.ocr.ocr_pdf",
        "tool": "mineru",
        "version": "2.7.6",
        "params": {
            "backend": "pipeline",
            "lang": "en",
            "method": "auto",
            "device_mode": "cpu",
            "dpi_requested": 350,
            "dpi": 350,
            "images_dir": "images/si2",
            "command": [
                ocr.DEFAULT_MINERU_PYTHON,
                *_tail(pdf, "pipeline", "en", "auto", "350", "images/si2"),
            ],
        },
        "input_sha256": [hashlib.sha256(b"%PDF-1.4\n").hexdigest()],
    }
    assert cmd == record["params"]["command"]

    ocr.ocr_pdf(_pdf(tmp_path / "ck"))
    paper_record = json.loads(
        (tmp_path / "ck" / "paper_ocr_provenance.json").read_text()
    )
    assert (paper_record["params"]["dpi_requested"], paper_record["params"]["dpi"]) == (
        0,
        200,
    )
    assert paper_record["params"]["images_dir"] == "images"


@pytest.mark.parametrize(
    ("stdout", "wrong"),
    [
        ("OK\n", "MINERU_VERSION, MINERU_DPI"),
        ("MINERU_VERSION=2.7.6\n", "MINERU_DPI"),
        (
            "MINERU_VERSION=2.7.6\nMINERU_VERSION=2.7.7\nMINERU_DPI=200\n",
            "MINERU_VERSION",
        ),
    ],
)
def test_a_runner_that_does_not_report_its_facts_once_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdout: str, wrong: str
) -> None:
    """Exit 0 without exactly one ``MINERU_VERSION`` and one ``MINERU_DPI`` line is a
    ``RunnerReportError`` naming the facts with the wrong count, and no record file is
    written (the markdown the runner wrote stays).
    """
    _install(monkeypatch, _FakeRun(stdout=stdout))
    pdf = _pdf(tmp_path / "ck")
    with pytest.raises(ocr.RunnerReportError) as refused:
        ocr.ocr_pdf(pdf)
    assert str(refused.value) == (
        "MinerU runner on paper.pdf must print each of MINERU_VERSION, MINERU_DPI "
        f"exactly once; wrong count for {wrong}"
    )
    assert sorted(p.name for p in (tmp_path / "ck").iterdir()) == [
        "paper.md",
        "paper.pdf",
    ]


def test_the_processor_named_by_mormino_2022_resolves_to_ocr_pdf() -> None:
    """Mormino 2022's OCR ``ProcessingRecord`` names the function that OCRs one PDF,
    ``torchcell.literature.ocr.ocr_pdf``, and that dotted path resolves to it.
    """
    import torchcell.datasets.scerevisiae.mormino2022 as mormino

    source = Path(mormino.__file__).read_text()
    (processor,) = re.findall(
        r'processor="(torchcell\.literature\.ocr\.[^"]+)"', source
    )
    assert processor == "torchcell.literature.ocr.ocr_pdf"
    module_name, _, attribute = processor.rpartition(".")
    assert getattr(importlib.import_module(module_name), attribute) is ocr.ocr_pdf


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


def test_ocr_artifact_runs_paper_then_si_pdfs_in_natural_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SI files run and are returned in natural order, ``si1``, ``si2``, ``si10``.
    The glob is case-sensitive and not recursive (``SI3.pdf``, ``nested/si4.pdf``,
    ``table.pdf`` and the top-level ``si5.pdf`` are skipped), the keywords reach every
    call (``--dpi 350``), and each SI PDF gets its own figures directory.
    """
    fake = _install(monkeypatch, _FakeRun())
    root = _artifact(tmp_path / "ck")

    produced = ocr.ocr_artifact(root, dpi=350, backend="vlm-auto-engine")

    assert produced == [
        root / "paper.md",
        root / "si" / "si1.md",
        root / "si" / "si2.md",
        root / "si" / "si10.md",
    ]
    si = root / "si"
    assert [(cmd[3], cmd[5], cmd[7], cmd[13], cmd[15]) for cmd, _ in fake.calls] == [
        (str(root / "paper.pdf"), str(root), "vlm-auto-engine", "350", "images"),
        (str(si / "si1.pdf"), str(si), "vlm-auto-engine", "350", "images/si1"),
        (str(si / "si2.pdf"), str(si), "vlm-auto-engine", "350", "images/si2"),
        (str(si / "si10.pdf"), str(si), "vlm-auto-engine", "350", "images/si10"),
    ]


def test_ocr_artifact_without_a_paper_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A directory with no ``paper.pdf`` raises ``MissingPaperPdfError`` naming it,
    before any OCR: an SI-only artifact does not OCR its SI, and an empty directory (or
    one whose ``si`` is a file) does not return ``[]``. The only caller,
    ``capture.capture_by_doi``, downloads the article as ``paper.pdf`` first, so a
    missing one is a broken capture.
    """
    fake = _install(monkeypatch, _FakeRun())
    si_only = tmp_path / "si-only"
    _pdf(si_only / "si", "si1.pdf")
    with pytest.raises(ocr.MissingPaperPdfError) as refused:
        ocr.ocr_artifact(si_only)
    assert str(refused.value) == f"no paper.pdf in artifact directory {si_only}"

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "si").write_text("a file, not a directory")
    with pytest.raises(ocr.MissingPaperPdfError) as refused:
        ocr.ocr_artifact(empty)
    assert str(refused.value) == f"no paper.pdf in artifact directory {empty}"
    assert fake.calls == []
    assert not (si_only / "si" / "si1.md").exists()


def test_ocr_artifact_stops_at_the_first_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """si1 exits 1: the paper already ran (its markdown and processing record are on
    disk), si1's error is raised, si2 and si10 never start, and the list of what was
    produced is lost.
    """
    fake = _install(monkeypatch, _FakeRun(outcomes={"si1.pdf": (1, "boom")}))
    root = _artifact(tmp_path / "ck")
    with pytest.raises(RuntimeError) as refused:
        ocr.ocr_artifact(root)
    assert str(refused.value) == "MinerU failed (exit 1) on si1.pdf:\nboom"
    assert [Path(cmd[3]).name for cmd, _ in fake.calls] == ["paper.pdf", "si1.pdf"]
    assert (root / "paper.md").read_text() == "# Title\nabc"
    assert (root / "paper_ocr_provenance.json").is_file()
    assert not (root / "si" / "si1_ocr_provenance.json").exists()
