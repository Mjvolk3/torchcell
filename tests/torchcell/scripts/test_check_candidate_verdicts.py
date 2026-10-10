# tests/torchcell/scripts/test_check_candidate_verdicts.py
# [[tests.torchcell.scripts.test_check_candidate_verdicts]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_check_candidate_verdicts.py
"""``scripts/check_candidate_verdicts.py`` against a temporary git repository.

The AST pieces are tabled; the git piece runs the real index-vs-merge-base diff in
``tmp_path``: ``main`` registers one class, ``feature`` registers a new one in the same
module and adds a module whose new class has no ``CITATION_KEY``. The hook fails naming
both, passes once the verdict JSON is staged and the key is declared, and ignores an
unstaged verdict. Git is hermetic: ``HOME`` and the global config point into ``tmp_path``.
"""

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check_candidate_verdicts as ccv  # type: ignore[import-not-found]  # noqa: E402

KEYED = """\
from torchcell.datasets.dataset_registry import register_dataset
from torchcell import datasets

CITATION_KEY = "fixtureKey2020"


@register_dataset
class OldDataset:
    pass
"""


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("@register_dataset\nclass A:\n    pass\n", {"A"}),
        ("@registry.register_dataset\nclass B:\n    pass\n", {"B"}),
        ("@other\nclass C:\n    pass\n\nclass D:\n    pass\n", set()),
        ("def f():\n    @register_dataset\n    class E:\n        pass\n", set()),
    ],
)
def test_registered_classes(source: str, expected: set[str]) -> None:
    """Top-level classes with a bare or dotted register_dataset decorator."""
    assert ccv.registered_classes(source) == expected


@pytest.mark.parametrize(
    ("source", "key"),
    [
        ('CITATION_KEY = "k1"\n', "k1"),
        ('CITATION_KEY: Final = "k2"\n', "k2"),
        ("CITATION_KEY = OTHER\n", None),
        ('KEY = "k3"\n', None),
    ],
)
def test_citation_key(source: str, key: str | None) -> None:
    """Only a module-level string constant named CITATION_KEY counts."""
    assert ccv.citation_key(source) == key


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True
    )
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture(autouse=True)
def _hermetic_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "t")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@t")
    yield


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """main: one keyed module with OldDataset; feature: NewDataset and a keyless module."""
    repo = tmp_path / "repo"
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    _write(repo, "torchcell/datasets/ecoli/keyed.py", KEYED)
    _write(
        repo, "torchcell/other.py", "@register_dataset\nclass Elsewhere:\n    pass\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(
        repo,
        "torchcell/datasets/ecoli/keyed.py",
        KEYED + "\n\n@register_dataset\nclass NewDataset:\n    pass\n",
    )
    _write(
        repo,
        "torchcell/datasets/ecoli/bare.py",
        "@register_dataset\nclass BareDataset:\n    pass\n",
    )
    _write(
        repo, "torchcell/other.py", "@register_dataset\nclass Elsewhere2:\n    pass\n"
    )
    _git(repo, "add", "-A")
    return repo


def test_new_registrations_against_the_merge_base(repo: Path) -> None:
    """Only classes new under torchcell/datasets/, with their module's key."""
    assert ccv.new_registrations("main", repo) == [
        ("torchcell/datasets/ecoli/bare.py", "BareDataset", None),
        ("torchcell/datasets/ecoli/keyed.py", "NewDataset", "fixtureKey2020"),
    ]


def test_hook_fails_until_the_verdict_is_staged(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both problems named; an unstaged verdict does not count; staging it clears one."""
    _write(repo, "database/candidates/fixtureKey2020.json", "{}")
    assert ccv.main(["--base", "main", "--repo", str(repo)]) == 1
    assert capsys.readouterr().out == (
        "candidate-verdicts: torchcell/datasets/ecoli/bare.py: BareDataset is new and "
        "the module has no CITATION_KEY\n"
        "candidate-verdicts: torchcell/datasets/ecoli/keyed.py: NewDataset is new and "
        "database/candidates/fixtureKey2020.json is not in the index (python -m "
        "torchcell.candidates gate --citation-key fixtureKey2020 --write)\n"
        "candidate-verdicts: 2 of 2 new registration(s) lack a candidate verdict\n"
    )
    _git(repo, "add", "database/candidates/fixtureKey2020.json")
    _write(repo, "torchcell/datasets/ecoli/bare.py", "")
    _git(repo, "add", "-A")
    assert ccv.main(["--base", "main", "--repo", str(repo)]) == 0
    assert capsys.readouterr().out == (
        "candidate-verdicts: 1 new registration(s) vs main, all carry a verdict\n"
    )
