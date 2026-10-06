# tests/torchcell/scripts/test_legacy_partition.py
# [[tests.torchcell.scripts.test_legacy_partition]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_legacy_partition.py
"""``scripts/legacy_partition.py`` on an eleven-module repository under ``tmp_path``.

``legacy_partition.REPO`` is monkeypatched to the temporary tree, so every function that
reads the module global (``package_modules``, ``root_files``, the entry-point scan, the
git lookup) sees it. The tree is built so every category and every kind of root edge
appears once: a relative import, a package ``__init__`` re-export (the init-only case), a
path-style reference in a shell script, a ``[project.scripts]`` entry point, the setuptools
version attribute, an experiment below the live threshold (ignored) and one above it with
a letter prefix (a root), a docstring mention (not a reference), a string constant in the
one file exempt from the string scan, a carve-out module and an already-legacy package.
The expected row for every module is written out in ``EXPECTED``.
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import legacy_partition as lp  # type: ignore[import-not-found]  # noqa: E402

FILES: dict[str, str] = {
    "torchcell/__init__.py": '__version__ = "0.0"\n',
    "torchcell/live.py": "from .helper import help_me\n",
    "torchcell/helper.py": "def help_me():\n    return 1\n",
    "torchcell/cli.py": "def main():\n    return 0\n",
    "torchcell/dead.py": "import torchcell.deadb\n",
    "torchcell/deadb.py": "X = 1\n",
    "torchcell/pkg/__init__.py": "from .init_only import thing\n",
    "torchcell/pkg/init_only.py": "thing = 1\n",
    "torchcell/scratch/junk.py": "import torchcell.dead\n",
    "torchcell/legacy/__init__.py": "",
    "torchcell/legacy/old.py": "OLD = 1\n",
    "tests/test_doc.py": '"""Mentions torchcell.dead in a docstring only."""\n',
    "tests/test_live.py": "from torchcell.live import help_me\n",
    "tests/test_pkg.py": "import torchcell.pkg\n",
    "tests/torchcell/test_import_all.py": 'NAMES = ["torchcell.deadb"]\n',
    "scripts/run.sh": "python torchcell/helper.py\n",
    "database/README": "",
    "experiments/015-old/x.py": "import torchcell.dead\n",
    "experiments/W019-new/y.py": "from torchcell.helper import help_me\n",
    "Makefile": "",
    "pyproject.toml": (
        '[project]\nname = "torchcell"\ndynamic = ["version"]\n\n'
        '[project.scripts]\ntc = "torchcell.cli:main"\n\n'
        '[tool.setuptools.dynamic]\nversion = {attr = "torchcell.__version__"}\n'
    ),
}

# module -> (category, via, live_critical, already_legacy)
EXPECTED: dict[str, tuple[str, str, bool, bool]] = {
    "torchcell": ("live", "package has live members", True, False),
    "torchcell.cli": ("live", "pyproject.toml#project.scripts", True, False),
    "torchcell.dead": ("legacy", "unreachable from every root", False, False),
    "torchcell.deadb": ("legacy", "unreachable from every root", False, False),
    "torchcell.helper": ("live", "scripts/run.sh", True, False),
    "torchcell.legacy": ("legacy", "every package member is legacy", False, True),
    "torchcell.legacy.old": ("legacy", "unreachable from every root", False, True),
    "torchcell.live": ("live", "tests/test_live.py", False, False),
    "torchcell.pkg": ("live", "package has live members", False, False),
    "torchcell.pkg.init_only": ("init-only", "torchcell.pkg", False, False),
    "torchcell.scratch.junk": ("carve-out", "", False, False),
}


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str]
) -> Path:
    repo = tmp_path / "repo"
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(lp, "REPO", repo)
    return repo


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _build(tmp_path, monkeypatch, FILES)


def test_partition_assigns_every_module_its_category_and_proof(repo: Path) -> None:
    """One row per module with the category, the root or ``__init__`` that reached it,
    live-critical from the production roots only, and the already-legacy flag.
    """
    rows = lp.partition(lp.ImporterGraph(), with_git=False)
    got = {
        str(r["module"]): (
            r["category"],
            r["via"],
            r["live_critical"],
            r["already_legacy"],
        )
        for r in rows
    }
    assert got == EXPECTED
    by_module = {str(r["module"]): r for r in rows}
    assert by_module["torchcell.helper"]["lines"] == 2
    assert by_module["torchcell.legacy"]["lines"] == 0
    assert by_module["torchcell.live"]["path"] == "torchcell/live.py"
    assert all(r["last_commit"] == "" for r in rows)


def test_root_files_take_only_live_experiments_and_known_roots(repo: Path) -> None:
    """The experiment at 015 is not a root; W019 is; the three root dirs, the Makefile
    and pyproject's scripts table are; nothing under torchcell/ is.
    """
    rels = [lp._rel(p) for p in lp.root_files()]
    assert rels == [
        "tests/test_doc.py",
        "tests/test_live.py",
        "tests/test_pkg.py",
        "tests/torchcell/test_import_all.py",
        "scripts/run.sh",
        "database/README",
        "experiments/W019-new/y.py",
        "Makefile",
    ]
    graph = lp.ImporterGraph()
    assert graph.root_edges["tests/test_doc.py"] == set()
    assert graph.root_edges["tests/torchcell/test_import_all.py"] == set()
    assert graph.root_edges["scripts/run.sh"] == {"torchcell", "torchcell.helper"}
    assert graph.root_edges["pyproject.toml#project.scripts"] == {
        "torchcell",
        "torchcell.cli",
    }
    assert graph.edges["torchcell.live"] == {"torchcell", "torchcell.helper"}
    assert graph.edges["torchcell.pkg"] == {"torchcell", "torchcell.pkg.init_only"}
    assert graph.production_roots() == [
        "scripts/run.sh",
        "database/README",
        "experiments/W019-new/y.py",
        "Makefile",
        "pyproject.toml#project.scripts",
    ]


def test_a_string_constant_in_a_scanned_root_makes_a_module_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same list literal that is ignored in test_import_all.py counts elsewhere."""
    files = dict(FILES)
    files["tests/test_strings.py"] = 'MODULES = ["torchcell.deadb"]\n'
    _build(tmp_path, monkeypatch, files)
    rows = {
        str(r["module"]): r for r in lp.partition(lp.ImporterGraph(), with_git=False)
    }
    assert (rows["torchcell.deadb"]["category"], rows["torchcell.deadb"]["via"]) == (
        "live",
        "tests/test_strings.py",
    )
    assert rows["torchcell.dead"]["category"] == "legacy"


def test_check_names_legacy_and_init_only_modules_outside_the_legacy_tree(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three violations; the already-legacy package is not one of them."""
    graph = lp.ImporterGraph()
    assert lp.check(graph, lp.partition(graph, with_git=False)) == 1
    assert capsys.readouterr().out.splitlines() == [
        "legacy-check: torchcell/dead.py: legacy (unreachable from every root)",
        "legacy-check: torchcell/deadb.py: legacy (unreachable from every root)",
        "legacy-check: torchcell/pkg/init_only.py: init-only (torchcell.pkg)",
        "legacy-check: 3 violation(s)",
    ]


def test_check_names_a_root_that_imports_the_legacy_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A test importing torchcell.legacy.old makes it live (no module violation for it,
    it is already under legacy/) but the root itself is a violation.
    """
    files = dict(FILES)
    files["tests/test_old.py"] = "import torchcell.legacy.old\n"
    _build(tmp_path, monkeypatch, files)
    graph = lp.ImporterGraph()
    rows = lp.partition(graph, with_git=False)
    by_module = {str(r["module"]): r for r in rows}
    assert by_module["torchcell.legacy.old"]["category"] == "live"
    assert by_module["torchcell.legacy"]["via"] == "package has live members"
    assert lp.check(graph, rows) == 1
    assert capsys.readouterr().out.splitlines()[-2:] == [
        "legacy-check: tests/test_old.py: imports torchcell.legacy, torchcell.legacy.old",
        "legacy-check: 4 violation(s)",
    ]


def test_check_passes_once_the_violations_are_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Move the dead pair under legacy/ and import init_only from a test: exit 0."""
    files = {
        k: v
        for k, v in FILES.items()
        if k not in ("torchcell/dead.py", "torchcell/deadb.py")
    }
    files["torchcell/legacy/dead.py"] = "import torchcell.legacy.deadb\n"
    files["torchcell/legacy/deadb.py"] = "X = 1\n"
    files["torchcell/scratch/junk.py"] = "import torchcell.legacy.dead\n"
    files["tests/test_pkg.py"] = "from torchcell.pkg.init_only import thing\n"
    _build(tmp_path, monkeypatch, files)
    graph = lp.ImporterGraph()
    assert lp.check(graph, lp.partition(graph, with_git=False)) == 0
    assert capsys.readouterr().out.strip() == (
        "legacy-check: the live tree is closed under imports; no root touches legacy"
    )


def test_main_prints_the_table_writes_json_and_summarizes_counts(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--table`` (default categories legacy + init-only), ``--output`` JSON, and the
    per-category module and line counts.
    """
    out = tmp_path / "out" / "partition.json"
    assert lp.main(["--no-git", "--table", "--output", str(out)]) == 0
    lines_by = {
        c: sum(
            _line_count(FILES[path])
            for path, (category, *_rest) in (
                (str(lp.package_modules()[m].relative_to(lp.REPO)), EXPECTED[m])
                for m in EXPECTED
            )
            if category == c
        )
        for c in ("live", "init-only", "legacy", "carve-out", "package-data")
    }
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == f"wrote {out}"
    assert printed[1:8] == [
        "| Module | Lines | Last commit | Category | Reached via |",
        "|---|---|---|---|---|",
        "| `torchcell/dead.py` | 1 |  | legacy | unreachable from every root |",
        "| `torchcell/deadb.py` | 1 |  | legacy | unreachable from every root |",
        "| `torchcell/legacy/__init__.py` | 0 |  | legacy | every package member is legacy |",
        "| `torchcell/legacy/old.py` | 1 |  | legacy | unreachable from every root |",
        "| `torchcell/pkg/init_only.py` | 1 |  | init-only | torchcell.pkg |",
    ]
    assert printed[8] == ""
    assert printed[9] == (
        f"modules: live 5 ({lines_by['live']} lines), init-only 1 ({lines_by['init-only']} lines), "
        f"legacy 4 ({lines_by['legacy']} lines), carve-out 1 ({lines_by['carve-out']} lines), "
        "package-data 0 (0 lines)"
    )
    payload = json.loads(out.read_text())
    assert payload["counts"] == {
        "live": 5,
        "init-only": 1,
        "legacy": 4,
        "carve-out": 1,
        "package-data": 0,
    }
    assert payload["lines"] == lines_by
    assert [r["module"] for r in payload["rows"]] == sorted(EXPECTED)


def test_colliding_module_spellings_are_kept_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``torchcell/cell.py`` beside ``torchcell/cell/__init__.py``: the directory keeps the
    name, the file is reported as ``torchcell.cell#file`` with a warning on stderr.
    """
    files = {
        "torchcell/__init__.py": "",
        "torchcell/cell.py": "A = 1\n",
        "torchcell/cell/__init__.py": "",
        "tests/.keep": "",
        "scripts/.keep": "",
        "database/.keep": "",
        "Makefile": "",
        "pyproject.toml": "[project]\nname = 'torchcell'\n",
    }
    repo = _build(tmp_path, monkeypatch, files)
    (repo / "experiments").mkdir()
    modules = lp.package_modules()
    assert modules["torchcell.cell"] == repo / "torchcell" / "cell" / "__init__.py"
    assert modules["torchcell.cell#file"] == repo / "torchcell" / "cell.py"
    assert capsys.readouterr().err.strip() == (
        "warning: torchcell.cell is spelled by torchcell/cell/__init__.py and "
        "torchcell/cell.py; reporting torchcell/cell.py as torchcell.cell#file"
    )


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("see torchcell/models/dcell.py", {"torchcell.models.dcell"}),
        ("python -m torchcell.data.cell_data", {"torchcell.data.cell_data"}),
        ("torchcell alone, and torchcellx.y", set()),
        ("a torchcell/x/y.pyc file", set()),
    ],
)
def test_string_references(text: str, names: set[str]) -> None:
    """Dotted names and ``.py`` paths, nothing else."""
    assert lp.string_references(text) == names


def test_git_last_commit_is_untracked_outside_a_repository(repo: Path) -> None:
    """No git history under tmp_path gives the sentinel rather than an error."""
    assert lp._git_last_commit(repo / "torchcell" / "live.py") == "untracked"


# --- package-data (2026.10.06) ------------------------------------------------------

PACKAGE_DATA_TOML = (
    '\n[tool.setuptools.package-data]\ntorchcell = ["py.typed", "conf/*.yaml"]\n'
)


def _package_data_files(*, with_entry: bool) -> dict[str, str]:
    """FILES plus an unimported ``torchcell/conf`` package that ships ``a.yaml``."""
    files = dict(FILES)
    files["torchcell/conf/__init__.py"] = '"""Ships YAML."""\n'
    files["torchcell/conf/a.yaml"] = "k: 1\n"
    if with_entry:
        files["pyproject.toml"] = FILES["pyproject.toml"] + PACKAGE_DATA_TOML
    return files


def _rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str]
) -> dict[str, dict[str, object]]:
    _build(tmp_path, monkeypatch, files)
    return {
        str(r["module"]): r for r in lp.partition(lp.ImporterGraph(), with_git=False)
    }


def test_an_unimported_package_listed_as_package_data_is_package_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``conf/*.yaml`` under the ``torchcell`` key matches ``torchcell/conf/a.yaml``, so
    the never-imported ``torchcell/conf/__init__.py`` is ``package-data`` (via names the
    pattern) and ``--check`` reports only the three violations of the base tree.
    """
    _build(tmp_path, monkeypatch, _package_data_files(with_entry=True))
    graph = lp.ImporterGraph()
    rows = lp.partition(graph, with_git=False)
    by_module = {str(r["module"]): r for r in rows}
    assert (
        by_module["torchcell.conf"]["category"],
        by_module["torchcell.conf"]["via"],
    ) == ("package-data", "ships torchcell:conf/*.yaml")
    assert by_module["torchcell.conf"]["live_critical"] is False
    assert {
        m: r["category"] for m, r in by_module.items() if m != "torchcell.conf"
    } == {m: e[0] for m, e in EXPECTED.items()}
    assert lp.check(graph, rows) == 1
    assert capsys.readouterr().out.splitlines() == [
        "legacy-check: torchcell/dead.py: legacy (unreachable from every root)",
        "legacy-check: torchcell/deadb.py: legacy (unreachable from every root)",
        "legacy-check: torchcell/pkg/init_only.py: init-only (torchcell.pkg)",
        "legacy-check: 3 violation(s)",
    ]


def test_the_same_package_without_the_pyproject_entry_is_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The YAML alone keeps nothing: no package-data table, the ``__init__`` is legacy
    and ``--check`` names it.
    """
    _build(tmp_path, monkeypatch, _package_data_files(with_entry=False))
    graph = lp.ImporterGraph()
    rows = lp.partition(graph, with_git=False)
    by_module = {str(r["module"]): r for r in rows}
    assert (
        by_module["torchcell.conf"]["category"],
        by_module["torchcell.conf"]["via"],
    ) == ("legacy", "unreachable from every root")
    assert lp.check(graph, rows) == 1
    assert capsys.readouterr().out.splitlines() == [
        "legacy-check: torchcell/conf/__init__.py: legacy (unreachable from every root)",
        "legacy-check: torchcell/dead.py: legacy (unreachable from every root)",
        "legacy-check: torchcell/deadb.py: legacy (unreachable from every root)",
        "legacy-check: torchcell/pkg/init_only.py: init-only (torchcell.pkg)",
        "legacy-check: 4 violation(s)",
    ]


def test_a_module_beside_the_package_data_is_classified_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``torchcell/conf/loader.py`` is not an ``__init__``: unreached, it is legacy and a
    violation, while the ``__init__`` keeps ``package-data`` (its only member is legacy,
    which would otherwise make it ``legacy`` by the package rule).
    """
    files = _package_data_files(with_entry=True)
    files["torchcell/conf/loader.py"] = "X = 1\n"
    _build(tmp_path, monkeypatch, files)
    graph = lp.ImporterGraph()
    rows = lp.partition(graph, with_git=False)
    by_module = {str(r["module"]): r for r in rows}
    assert (
        by_module["torchcell.conf.loader"]["category"],
        by_module["torchcell.conf.loader"]["via"],
    ) == ("legacy", "unreachable from every root")
    assert by_module["torchcell.conf"]["category"] == "package-data"
    assert lp.check(graph, rows) == 1
    assert capsys.readouterr().out.splitlines() == [
        "legacy-check: torchcell/conf/loader.py: legacy (unreachable from every root)",
        "legacy-check: torchcell/dead.py: legacy (unreachable from every root)",
        "legacy-check: torchcell/deadb.py: legacy (unreachable from every root)",
        "legacy-check: torchcell/pkg/init_only.py: init-only (torchcell.pkg)",
        "legacy-check: 4 violation(s)",
    ]


def test_a_python_file_matched_by_a_pattern_is_not_package_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``conf/*`` matches ``__init__.py``, ``loader.py`` and the subpackage directory
    ``conf/sub`` in a YAML-free package: a ``.py`` match ships code, not data, and a
    matched directory is not a file, so nothing is marked and the ``__init__`` stays
    ``legacy``.
    """
    files = dict(FILES)
    files["torchcell/conf/__init__.py"] = ""
    files["torchcell/conf/loader.py"] = "X = 1\n"
    files["torchcell/conf/sub/__init__.py"] = ""
    files["pyproject.toml"] = (
        FILES["pyproject.toml"]
        + '\n[tool.setuptools.package-data]\ntorchcell = ["conf/*"]\n'
    )
    rows = _rows(tmp_path, monkeypatch, files)
    assert rows["torchcell.conf"]["category"] == "legacy"
    assert lp.package_data_dirs() == {}


def test_package_data_keys_are_package_names_and_foreign_keys_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dotted key globs from its own directory (``torchcell.kg.conf`` + ``*.yaml``,
    two YAMLs, one directory entry); ``torchcell = ["conf/*.yaml"]`` matches only the
    top-level ``torchcell/conf`` (a glob, not a recursive search, so the nested
    ``torchcell/kg/conf`` keeps its own key); a key that only shares the prefix
    (``torchcellx``) is not this package and is ignored although
    ``torchcellx/conf/a.yaml`` matches.
    """
    files = dict(FILES)
    files["torchcell/kg/__init__.py"] = ""
    files["torchcell/kg/conf/__init__.py"] = ""
    files["torchcell/kg/conf/x.yaml"] = "a: 1\n"
    files["torchcell/kg/conf/y.yaml"] = "b: 2\n"
    files["torchcell/conf/__init__.py"] = ""
    files["torchcell/conf/a.yaml"] = "k: 1\n"
    files["torchcellx/conf/a.yaml"] = "k: 1\n"
    files["pyproject.toml"] = FILES["pyproject.toml"] + (
        "\n[tool.setuptools.package-data]\n"
        'torchcell = ["conf/*.yaml"]\n'
        '"torchcell.kg.conf" = ["*.yaml"]\n'
        'torchcellx = ["conf/*.yaml"]\n'
    )
    repo = _build(tmp_path, monkeypatch, files)
    assert lp.package_data_dirs() == {
        repo / "torchcell" / "conf": "torchcell:conf/*.yaml",
        repo / "torchcell" / "kg" / "conf": "torchcell.kg.conf:*.yaml",
    }
    rows = {
        str(r["module"]): r for r in lp.partition(lp.ImporterGraph(), with_git=False)
    }
    assert (
        rows["torchcell.kg.conf"]["category"],
        rows["torchcell.kg.conf"]["via"],
    ) == ("package-data", "ships torchcell.kg.conf:*.yaml")
    # the parent package counts a package-data member as non-legacy
    assert (rows["torchcell.kg"]["category"], rows["torchcell.kg"]["via"]) == (
        "live",
        "package has live members",
    )
    assert rows["torchcell.conf"]["category"] == "package-data"


def test_main_counts_package_data_in_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Line counts per category on the base tree plus the docstring-only conf
    ``__init__``: live torchcell 1 + cli 2 + helper 2 + live 1 + pkg 1 = 7; init-only 1;
    legacy dead 1 + deadb 1 + legacy/__init__ 0 + legacy/old 1 = 3; carve-out 1;
    package-data 1.
    """
    _build(tmp_path, monkeypatch, _package_data_files(with_entry=True))
    assert lp.main(["--no-git", "--check"]) == 1
    assert capsys.readouterr().out.splitlines()[0] == (
        "modules: live 5 (7 lines), init-only 1 (1 lines), legacy 4 (3 lines), "
        "carve-out 1 (1 lines), package-data 1 (1 lines)"
    )


def test_the_first_pattern_that_marks_a_directory_is_the_one_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``conf/*.yaml`` then ``conf/*.txt`` both match files in ``torchcell/conf``; the
    reported pattern is the first in pyproject order, not the last.
    """
    files = _package_data_files(with_entry=False)
    files["torchcell/conf/genes.txt"] = "YAL001C\n"
    files["pyproject.toml"] = (
        FILES["pyproject.toml"]
        + '\n[tool.setuptools.package-data]\ntorchcell = ["conf/*.yaml", "conf/*.txt"]\n'
    )
    rows = _rows(tmp_path, monkeypatch, files)
    assert rows["torchcell.conf"]["via"] == "ships torchcell:conf/*.yaml"


def test_a_package_data_init_that_a_root_imports_stays_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reachability wins: a test importing ``torchcell.conf`` makes its ``__init__``
    ``live`` via that test, although its directory ships listed YAML.
    """
    files = _package_data_files(with_entry=True)
    files["tests/test_conf.py"] = "import torchcell.conf\n"
    rows = _rows(tmp_path, monkeypatch, files)
    assert (rows["torchcell.conf"]["category"], rows["torchcell.conf"]["via"]) == (
        "live",
        "tests/test_conf.py",
    )


@pytest.mark.parametrize(
    ("init_text", "category", "via"),
    [
        ("", "package-data", "ships torchcell:conf/*.yaml"),
        ('"""Ships YAML."""\n', "package-data", "ships torchcell:conf/*.yaml"),
        ("X = 1\n", "legacy", "unreachable from every root"),
        ('"""Ships YAML."""\nimport os\n', "legacy", "unreachable from every root"),
    ],
)
def test_only_an_empty_or_docstring_only_init_is_package_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    init_text: str,
    category: str,
    via: str,
) -> None:
    """An ``__init__`` holding code beside listed YAML is classified as before (legacy
    when unreached) and is a ``--check`` violation, so the category cannot hide code.
    """
    files = _package_data_files(with_entry=True)
    files["torchcell/conf/__init__.py"] = init_text
    _build(tmp_path, monkeypatch, files)
    graph = lp.ImporterGraph()
    rows = lp.partition(graph, with_git=False)
    by_module = {str(r["module"]): r for r in rows}
    assert (
        by_module["torchcell.conf"]["category"],
        by_module["torchcell.conf"]["via"],
    ) == (category, via)
    lp.check(graph, rows)
    flagged = (
        "legacy-check: torchcell/conf/__init__.py: legacy (unreachable from every root)"
        in capsys.readouterr().out.splitlines()
    )
    assert flagged is (category == "legacy")


def test_a_docstring_only_module_beside_package_data_is_still_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an ``__init__.py`` can be ``package-data``: a docstring-only
    ``torchcell/conf/notes.py`` beside listed YAML is unreached and stays ``legacy``.
    """
    files = _package_data_files(with_entry=True)
    files["torchcell/conf/notes.py"] = '"""Only a docstring."""\n'
    rows = _rows(tmp_path, monkeypatch, files)
    assert (
        rows["torchcell.conf.notes"]["category"],
        rows["torchcell.conf.notes"]["via"],
    ) == ("legacy", "unreachable from every root")
    assert rows["torchcell.conf"]["category"] == "package-data"
