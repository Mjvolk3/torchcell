# tests/torchcell/test_readme_pypi.py
# [[tests.torchcell.test_readme_pypi]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/test_readme_pypi.py
"""The README is the PyPI project description, so its references must be absolute.

``pyproject.toml`` ships ``README.md`` as the long description. PyPI renders it away from
the repository, where a relative path resolves to nothing: release 1.6.0 showed the
overview figure as a broken image because it was written ``./notes/assets/images/...``.
Images therefore point at ``raw.githubusercontent.com/Mjvolk3/torchcell/main/<path>``,
and each such ``<path>`` must exist in the checkout, so a moved asset fails here instead
of on the published page.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW_PREFIX = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/"
_HTML_REF = re.compile(r'(?:src|href)="([^"]+)"')
_MD_REF = re.compile(r"\]\(([^)\s]+)\)")


def _references(text: str) -> list[str]:
    """Every HTML ``src``/``href`` value and every markdown link target, in order."""
    return _HTML_REF.findall(text) + _MD_REF.findall(text)


def test_reference_scan_reads_html_and_markdown_forms() -> None:
    """The scan returns both forms, so the absolute-only test below cannot pass by
    missing a reference.
    """
    text = '<img src="./a.png" /> <a href="https://x.org/b">b</a> [c](docs/c.md)'
    assert _references(text) == ["./a.png", "https://x.org/b", "docs/c.md"]


def test_every_readme_reference_is_absolute() -> None:
    """No image or link in the README is a repository-relative path."""
    references = _references((REPO / "README.md").read_text())
    relative = [r for r in references if not r.startswith(("https://", "http://"))]
    assert relative == []


def test_raw_main_references_name_files_in_the_checkout() -> None:
    """The two repository-hosted images are the logo and the overview figure, and both
    files exist at the paths their URLs name.
    """
    references = _references((REPO / "README.md").read_text())
    raw_paths = [
        r.removeprefix(RAW_PREFIX) for r in references if r.startswith(RAW_PREFIX)
    ]
    assert raw_paths == [
        "notes/assets/drawio/torchcell-logo.drawio.png",
        "notes/assets/images/Fig1-torchcell-overview-abc.png",
    ]
    assert [(REPO / path).is_file() for path in raw_paths] == [True, True]
