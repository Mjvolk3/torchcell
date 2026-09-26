# tests/torchcell/scripts/test_check_paired_tests.py
# [[tests.torchcell.scripts.test_check_paired_tests]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_check_paired_tests.py
"""``scripts/check_paired_tests.py`` against a temporary git repository.

The pure pieces (``is_excepted``, ``expected_test``, ``load_exceptions``) are tabled; the
git piece runs the real ``git diff --cached <merge-base>`` inside ``tmp_path``: a ``main``
with one module and its test, a ``feature`` branch adding three modules (one under an
excepted directory, one with a ``pairs`` entry) and modifying the old one. Default mode
sees the additions only; ``--strict`` adds the modification; a staged-but-uncommitted
file counts (that is the pre-commit path) and an unstaged one does not. Git is hermetic:
``HOME`` and the global config point into ``tmp_path`` and the ``GIT_*`` variables are
removed.
"""

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check_paired_tests as cpt  # type: ignore[import-not-found]  # noqa: E402

PYPROJECT = """\
[tool.torchcell.test_exceptions]
paths = ["torchcell/scratch/", "*_legacy.py"]

[tool.torchcell.test_exceptions.pairs]
"torchcell/losses/dcell.py" = "tests/torchcell/losses/test_losses_dcell.py"
"""


@pytest.mark.parametrize(
    ("path", "patterns", "excepted"),
    [
        ("torchcell/scratch/x.py", ["torchcell/scratch/"], True),
        ("torchcell/scratchy.py", ["torchcell/scratch/"], False),
        ("torchcell/models/thing_legacy.py", ["*_legacy.py"], True),
        ("torchcell/models/legacy_thing.py", ["*_legacy.py"], False),
        ("torchcell/models/a/b.py", ["torchcell/models/*/b.py"], True),
        ("torchcell/x.py", ["torchcell/x.py"], True),
        ("torchcell/x.py", ["x.py"], False),
        ("torchcell/x.py", [], False),
    ],
)
def test_is_excepted(path: str, patterns: list[str], excepted: bool) -> None:
    """Prefix (trailing slash), glob on the path or the basename, or an exact path."""
    assert cpt.is_excepted(path, patterns) is excepted


@pytest.mark.parametrize(
    ("module", "test"),
    [
        ("torchcell/models/dcell.py", "tests/torchcell/models/test_dcell.py"),
        ("torchcell/x.py", "tests/torchcell/test_x.py"),
        ("torchcell/a/b/c.py", "tests/torchcell/a/b/test_c.py"),
    ],
)
def test_expected_test_mirrors_the_directory(module: str, test: str) -> None:
    """``torchcell/a/b.py`` -> ``tests/torchcell/a/test_b.py``."""
    assert cpt.expected_test(module) == test


def test_load_exceptions_reads_the_table_or_returns_empty(tmp_path: Path) -> None:
    """Both keys when present; empty lists when the table (or the tool section) is absent."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PYPROJECT)
    assert cpt.load_exceptions(pyproject) == (
        ["torchcell/scratch/", "*_legacy.py"],
        {"torchcell/losses/dcell.py": "tests/torchcell/losses/test_losses_dcell.py"},
    )
    pyproject.write_text('[project]\nname = "x"\n')
    assert cpt.load_exceptions(pyproject) == ([], {})


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


def _write(repo: Path, rel: str, text: str = "x = 1\n") -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """main: one module + its test; feature: three added modules and one modified."""
    repo = tmp_path / "repo"
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    _write(repo, "pyproject.toml", PYPROJECT)
    _write(repo, "torchcell/__init__.py", "")
    _write(repo, "torchcell/old.py")
    _write(repo, "tests/torchcell/test_old.py", "def test_old():\n    assert True\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "torchcell/new.py")
    _write(repo, "torchcell/scratch/junk.py")
    _write(repo, "torchcell/losses/dcell.py")
    _write(repo, "torchcell/losses/__init__.py", "")
    _write(repo, "torchcell/__main__.py")
    _write(repo, "torchcell/old.py", "x = 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feature")
    return repo


def test_changed_modules_added_only_by_default_and_modified_under_strict(
    repo: Path,
) -> None:
    """``__init__``/``__main__`` are exempt; the modified module appears only with strict."""
    added = [
        "torchcell/losses/dcell.py",
        "torchcell/new.py",
        "torchcell/scratch/junk.py",
    ]
    assert cpt.changed_modules("main", strict=False, repo=repo) == added
    assert cpt.changed_modules("main", strict=True, repo=repo) == sorted(
        added + ["torchcell/old.py"]
    )


def test_changed_modules_sees_staged_but_not_unstaged_files(repo: Path) -> None:
    """Index vs merge-base: a staged new module counts, an unstaged one does not."""
    _write(repo, "torchcell/staged.py")
    _write(repo, "torchcell/unstaged.py")
    _git(repo, "add", "torchcell/staged.py")
    assert cpt.changed_modules("main", strict=False, repo=repo) == [
        "torchcell/losses/dcell.py",
        "torchcell/new.py",
        "torchcell/scratch/junk.py",
        "torchcell/staged.py",
    ]


def test_missing_pairs_honors_paths_and_pairs(repo: Path) -> None:
    """The scratch module is excepted; dcell.py looks for its ``pairs`` target; new.py the mirror."""
    patterns, pairs = cpt.load_exceptions(repo / "pyproject.toml")
    modules = cpt.changed_modules("main", strict=False, repo=repo)
    assert cpt.missing_pairs(modules, patterns, pairs, repo) == [
        ("torchcell/losses/dcell.py", "tests/torchcell/losses/test_losses_dcell.py"),
        ("torchcell/new.py", "tests/torchcell/test_new.py"),
    ]


def test_main_names_every_missing_pair_then_passes_once_the_tests_exist(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 1 with one line per module plus the summary; exit 0 once both files exist
    (untracked is enough, the gate checks the working tree).
    """
    assert cpt.main(["--base", "main", "--repo", str(repo)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        "paired-tests: torchcell/losses/dcell.py has no test file "
        "tests/torchcell/losses/test_losses_dcell.py",
        "paired-tests: torchcell/new.py has no test file tests/torchcell/test_new.py",
        "paired-tests: 2 of 3 added module(s) lack a paired test; add the test or an "
        "entry under [tool.torchcell.test_exceptions] in pyproject.toml",
    ]
    _write(
        repo,
        "tests/torchcell/losses/test_losses_dcell.py",
        "def test_a():\n    assert True\n",
    )
    _write(repo, "tests/torchcell/test_new.py", "def test_b():\n    assert True\n")
    assert cpt.main(["--base", "main", "--repo", str(repo)]) == 0
    assert (
        capsys.readouterr().out.strip()
        == "paired-tests: 3 added module(s) vs main, all paired"
    )
    assert cpt.main(["--base", "main", "--repo", str(repo), "--strict"]) == 0
    assert capsys.readouterr().out.strip() == (
        "paired-tests: 4 added or modified module(s) vs main, all paired"
    )
