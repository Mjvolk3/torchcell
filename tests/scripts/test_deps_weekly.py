# tests/scripts/test_deps_weekly.py
# [[tests.scripts.test_deps_weekly]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/scripts/test_deps_weekly.py
"""The pure parts of the weekly dependency job, with exact expected outputs.

Requirement-line parsing and floor raising (extras, inline and block comments, upper
bounds, ``[hold]``, never lowering, PEP 503 names such as ``torch_geometric`` and
``GEOparse``), cap stripping and lifting, the freeze diff, the torch pick from fixture
PyPI JSON and PyG pages, the commit subject, the land decision, and the consistency of
``env/deps_policy.toml`` with ``env/requirements.txt``. No network, no subprocess.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    path = REPO / "scripts" / "deps_weekly.py"
    spec = importlib.util.spec_from_file_location("deps_weekly", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deps_weekly"] = module
    spec.loader.exec_module(module)
    return module


dw = _load()

POLICY_TOML = """\
python = "3.13.0"

[hold]
torch_geometric = "cluster builds"
numpy = "cluster builds"

[caps.biocypher]
reason = "exact pin"
probe = false

[caps.neo4j]
reason = "no recorded reason"
probe = true
comment_marker = "no recorded reason"

[caps.bibtexparser]
reason = "v1 modules"
probe = true
"""

REQS = """\
# header comment
numpy>=2.0
torch-geometric>=2.8.0

# bio
GEOparse>=2.0.4
matplotlib>=3.9  # boxplot(tick_labels=)
uvicorn[standard]>=0.30
scipy>=1.11.4
biocypher==0.15.2
# Driver 4.4 line. The cap has no recorded reason;
# the job tries the suite without it.
neo4j>=4.4.11,<5.0
# v1 modules of bibtexparser
bibtexparser>=1.4,<2
"""

FREEZE = {
    "numpy": "2.3.5",
    "torch-geometric": "2.9.1",
    "geoparse": "2.0.4",
    "matplotlib": "3.10.7",
    "uvicorn": "0.38.0",
    "scipy": "1.10.0",
    "biocypher": "0.15.2",
    "neo4j": "4.4.13",
    "bibtexparser": "1.4.3",
}


@pytest.fixture()
def policy(tmp_path: Path) -> Any:
    path = tmp_path / "deps_policy.toml"
    path.write_text(POLICY_TOML)
    return dw.load_policy(path)


# ── requirement lines ─────────────────────────────────────────────────────────


def test_parse_requirement_line_splits_name_extras_specs_and_comment() -> None:
    req = dw.parse_requirement_line("uvicorn[standard]>=0.30,<1  # server")
    assert (req.name, req.extras, req.specs, req.marker, req.rest) == (
        "uvicorn",
        "[standard]",
        (">=0.30", "<1"),
        "",
        "  # server",
    )
    assert req.canonical == "uvicorn"
    assert req.render() == "uvicorn[standard]>=0.30,<1  # server"


def test_parse_requirement_line_blank_and_comment_are_none() -> None:
    assert dw.parse_requirement_line("") is None
    assert dw.parse_requirement_line("   ") is None
    assert dw.parse_requirement_line("# torch stack") is None


def test_parse_requirement_line_canonical_names() -> None:
    assert dw.parse_requirement_line("torch_geometric>=2.8.0").canonical == (
        "torch-geometric"
    )
    assert dw.parse_requirement_line("GEOparse>=2.0.4").canonical == "geoparse"


@pytest.mark.parametrize(
    "line", ["-r other.txt", "--index-url https://x", "neo4j>=4.4, <5", "neo4j=>4"]
)
def test_parse_requirement_line_rejects_unsupported(line: str) -> None:
    with pytest.raises(ValueError):
        dw.parse_requirement_line(line)


def test_every_real_requirement_line_renders_byte_for_byte() -> None:
    text = (REPO / "env" / "requirements.txt").read_text()
    lines = text.split("\n")
    rendered = [
        line if (req := dw.parse_requirement_line(line)) is None else req.render()
        for line in lines
    ]
    assert rendered == lines


def test_is_bounded() -> None:
    assert dw.is_bounded(dw.parse_requirement_line("neo4j>=4.4.11,<5.0")) is True
    assert dw.is_bounded(dw.parse_requirement_line("biocypher==0.15.2")) is True
    assert dw.is_bounded(dw.parse_requirement_line("scipy>=1.11.4")) is False
    assert dw.is_bounded(dw.parse_requirement_line("tqdm")) is False


@pytest.mark.parametrize(
    ("line", "tested", "expected"),
    [
        ("scipy>=1.11.4", "1.16.3", "scipy>=1.16.3"),
        ("scipy>=1.11.4", "1.10.0", "scipy>=1.11.4"),  # never lowered
        ("scipy>=1.11.4", "1.11.4", "scipy>=1.11.4"),
        ("omegaconf>=2.3", "2.3.0", "omegaconf>=2.3"),  # 2.3.0 == 2.3: unchanged
        ("torch>=2.11.0", "2.12.1+cpu", "torch>=2.12.1"),  # local label dropped
        ("neo4j>=4.4.11,<5.0", "4.4.13", "neo4j>=4.4.13,<5.0"),  # bound kept
        ("biocypher==0.15.2", "0.16.0", "biocypher==0.15.2"),  # a pin has no floor
        ("matplotlib>=3.9  # why", "3.10.7", "matplotlib>=3.10.7  # why"),
        ("uvicorn[standard]>=0.30", "0.38.0", "uvicorn[standard]>=0.38.0"),
    ],
)
def test_raise_floor(line: str, tested: str, expected: str) -> None:
    assert dw.raise_floor(dw.parse_requirement_line(line), tested).render() == expected


def test_strip_cap_keeps_only_floors() -> None:
    assert dw.strip_cap(dw.parse_requirement_line("neo4j>=4.4.11,<5.0")).render() == (
        "neo4j>=4.4.11"
    )
    assert dw.strip_cap(dw.parse_requirement_line("biocypher==0.15.2")).render() == (
        "biocypher"
    )
    assert dw.strip_cap(dw.parse_requirement_line("x>1,!=1.5,<=3  # c")).render() == (
        "x>1  # c"
    )


def test_strip_one_cap_touches_only_that_line() -> None:
    assert dw.strip_one_cap(REQS, "neo4j") == REQS.replace(
        "neo4j>=4.4.11,<5.0", "neo4j>=4.4.11"
    )
    assert dw.strip_one_cap(REQS, "BibTexParser") == REQS.replace(
        "bibtexparser>=1.4,<2", "bibtexparser>=1.4"
    )
    with pytest.raises(ValueError):
        dw.strip_one_cap(REQS, "pandas")


# ── rewriting env/requirements.txt ────────────────────────────────────────────


def test_rewrite_requirements_raises_floors_except_held(policy: object) -> None:
    result = dw.rewrite_requirements(REQS, FREEZE, policy, frozenset())
    assert (
        result.text
        == """\
# header comment
numpy>=2.0
torch-geometric>=2.8.0

# bio
GEOparse>=2.0.4
matplotlib>=3.10.7  # boxplot(tick_labels=)
uvicorn[standard]>=0.38.0
scipy>=1.11.4
biocypher==0.15.2
# Driver 4.4 line. The cap has no recorded reason;
# the job tries the suite without it.
neo4j>=4.4.13,<5.0
# v1 modules of bibtexparser
bibtexparser>=1.4.3,<2
"""
    )
    assert [(r.name, r.old, r.new) for r in result.raises] == [
        ("matplotlib", "3.9", "3.10.7"),
        ("uvicorn", "0.30", "0.38.0"),
        ("neo4j", "4.4.11", "4.4.13"),
        ("bibtexparser", "1.4", "1.4.3"),
    ]
    assert result.stale_comments == []


def test_rewrite_requirements_lifts_caps(policy: object) -> None:
    freeze = {**FREEZE, "neo4j": "6.0.2", "bibtexparser": "2.0.0"}
    result = dw.rewrite_requirements(
        REQS, freeze, policy, frozenset({"neo4j", "bibtexparser"})
    )
    # neo4j has a comment_marker: its comment block goes with the cap. bibtexparser has
    # none: its comment stays and is flagged stale.
    assert (
        result.text
        == """\
# header comment
numpy>=2.0
torch-geometric>=2.8.0

# bio
GEOparse>=2.0.4
matplotlib>=3.10.7  # boxplot(tick_labels=)
uvicorn[standard]>=0.38.0
scipy>=1.11.4
biocypher==0.15.2
neo4j>=6.0.2
# v1 modules of bibtexparser
bibtexparser>=2.0.0
"""
    )
    assert result.stale_comments == ["bibtexparser"]


def test_rewrite_requirements_missing_marker_is_an_error(policy: object) -> None:
    text = REQS.replace("The cap has no recorded reason;", "Driver line;")
    with pytest.raises(ValueError, match="comment_marker"):
        dw.rewrite_requirements(text, FREEZE, policy, frozenset({"neo4j"}))


def test_rewrite_requirements_uninstalled_direct_dependency_is_an_error(
    policy: object,
) -> None:
    freeze = {k: v for k, v in FREEZE.items() if k != "geoparse"}
    with pytest.raises(ValueError, match="GEOparse is a direct dependency"):
        dw.rewrite_requirements(REQS, freeze, policy, frozenset())


def test_drop_cap_table() -> None:
    assert dw.drop_cap_table(POLICY_TOML, "neo4j") == POLICY_TOML.replace(
        '[caps.neo4j]\nreason = "no recorded reason"\nprobe = true\n'
        'comment_marker = "no recorded reason"\n\n',
        "",
    )
    assert dw.drop_cap_table(POLICY_TOML, "bibtexparser").endswith(
        'comment_marker = "no recorded reason"\n'
    )
    with pytest.raises(ValueError):
        dw.drop_cap_table(POLICY_TOML, "pandas")


# ── policy file vs requirements ──────────────────────────────────────────────


def test_policy_caps_match_bounded_requirement_lines() -> None:
    policy = dw.load_policy(REPO / "env" / "deps_policy.toml")
    reqs = dw.requirement_lines((REPO / "env" / "requirements.txt").read_text())
    bounded = {r.canonical for r in reqs if dw.is_bounded(r)}
    assert bounded == {canonicalize_name(n) for n in policy.caps}
    assert bounded == {"biocypher", "neo4j", "bibtexparser"}


def test_policy_seed_values() -> None:
    policy = dw.load_policy(REPO / "env" / "deps_policy.toml")
    reqs = {
        r.canonical
        for r in dw.requirement_lines((REPO / "env" / "requirements.txt").read_text())
    }
    assert policy.python == "3.13.0"
    assert policy.held() == {
        "torch",
        "torch-geometric",
        "lightning",
        "torchmetrics",
        "transformers",
        "numpy",
        "pandas",
    }
    assert policy.held() <= reqs
    assert {n: c.probe for n, c in policy.caps.items()} == {
        "biocypher": False,
        "neo4j": False,
        "bibtexparser": True,
    }


def test_policy_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "p.toml"
    path.write_text(POLICY_TOML + "\n[caps.x]\nreason = 'r'\nprobe = true\nextra = 1\n")
    with pytest.raises(ValueError):
        dw.load_policy(path)


# ── freeze, lock, diff ──────────────────────────────────────────────────────


def test_parse_freeze_normalizes_and_drops_tools() -> None:
    text = (
        "GEOparse==2.0.4\ntorch==2.12.1+cpu\n-e file:///repo\npip==25.3\nuv==0.9.0\n"
        "torchcell==1.8.5\nTorch_Scatter==2.1.2+pt212cpu\n\n"
    )
    assert dw.parse_freeze(text) == {
        "geoparse": "2.0.4",
        "torch": "2.12.1+cpu",
        "torch-scatter": "2.1.2+pt212cpu",
    }
    with pytest.raises(ValueError):
        dw.parse_freeze("pkg @ file:///x.whl\n")


def test_render_lock_is_sorted_with_header() -> None:
    assert dw.render_lock({"scipy": "1.16.3", "attrs": "25.4.0"}) == (
        dw.LOCK_HEADER + "attrs==25.4.0\nscipy==1.16.3\n"
    )
    assert dw.parse_freeze(dw.render_lock({"scipy": "1.16.3"})) == {"scipy": "1.16.3"}


def test_diff_freeze() -> None:
    diff = dw.diff_freeze(
        {"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "2.1", "d": "4"}
    )
    assert diff.added == {"d": "4"}
    assert diff.removed == {"c": "3"}
    assert {k: (v.old, v.new) for k, v in diff.changed.items()} == {"b": ("2", "2.1")}
    assert diff.count() == 3


# ── torch pick ──────────────────────────────────────────────────────────────

_FILE = [{"filename": "x.whl", "yanked": False}]
PYPI = {
    "releases": {
        "2.15.0": _FILE,
        "2.14.1": _FILE,
        "2.14.0": [{"filename": "x.whl", "yanked": True}],
        "2.13.0": _FILE,
        "2.13.0rc1": _FILE,
        "2.12.1": _FILE,
        "2.12.0": _FILE,
        "2.11.0": _FILE,
        "2.10.0": [],
    }
}
PAGE_WITH = (
    '<a href="torch-2.12.0%2Bcpu/torch_scatter-2.1.2%2Bpt212cpu-cp313-cp313-'
    'linux_x86_64.whl">torch_scatter-2.1.2+pt212cpu-cp313-cp313-linux_x86_64.whl</a>'
)
PAGE_WITHOUT = (
    '<a href="torch-2.13.0%2Bcpu/pyg_lib-0.8.0%2Bpt213cpu-cp310-abi3-manylinux_2_28_'
    'x86_64.whl">pyg_lib</a>\n<a href="x/torch_scatter-2.1.2%2Bpt213cpu-cp313-cp313-'
    'linux_aarch64.whl">aarch64 only</a>'
)


def test_final_releases_skip_pre_yanked_and_empty() -> None:
    assert [str(v) for v in dw.final_releases(PYPI)] == [
        "2.15.0",
        "2.14.1",
        "2.13.0",
        "2.12.1",
        "2.12.0",
        "2.11.0",
    ]


def test_scatter_wheel_listed() -> None:
    assert dw.scatter_wheel_listed(PAGE_WITH) is True
    assert dw.scatter_wheel_listed(PAGE_WITHOUT) is False


def test_select_torch_walks_newest_first() -> None:
    pages = {
        dw.PYG_PAGE.format(version="2.15.0"): None,
        dw.PYG_PAGE.format(version="2.14.1"): None,
        dw.PYG_PAGE.format(version="2.13.0"): PAGE_WITHOUT,
        dw.PYG_PAGE.format(version="2.12.1"): PAGE_WITH,
    }
    fetched: list[str] = []

    def fetch(url: str) -> str | None:
        fetched.append(url)
        return pages[url]

    choice = dw.select_torch(PYPI, Version("2.11.0"), "2.11.0", fetch)
    assert (choice.pypi_latest, choice.testable, choice.ci_before) == (
        "2.15.0",
        "2.12.1",
        "2.11.0",
    )
    assert [(c.version, c.verdict) for c in choice.checked] == [
        ("2.15.0", "no page"),
        ("2.14.1", "no page"),
        ("2.13.0", "no torch_scatter wheel"),
        ("2.12.1", "torch_scatter wheel listed"),
    ]
    assert fetched == list(pages)


def test_select_torch_stops_at_the_floor() -> None:
    def fetch(url: str) -> str | None:
        return PAGE_WITHOUT

    with pytest.raises(dw.CommandFailed, match="no torch >= 2.12.0"):
        dw.select_torch(PYPI, Version("2.12.0"), "2.11.0", fetch)


# ── decisions ───────────────────────────────────────────────────────────────


def test_commit_subject() -> None:
    assert dw.commit_subject(True, "2026.11.22") == (
        "PATCH(deps): weekly dependency update 2026.11.22"
    )
    assert dw.commit_subject(False, "2026.11.22") == (
        "build(deps): weekly tested-versions snapshot 2026.11.22"
    )


def _checks(*pairs: tuple[str, str]) -> list[Any]:
    return [dw.CheckState(name=n, bucket=b) for n, b in pairs]


@pytest.mark.parametrize(
    ("checks", "timed_out", "loop_live", "enqueue", "reason"),
    [
        (
            [("pytest", "pass"), ("docs", "skipping")],
            False,
            True,
            True,
            "all checks passed; enqueued",
        ),
        (
            [("pytest", "pass")],
            False,
            False,
            False,
            "merge-queue loop is not live; PR left open",
        ),
        (
            [("pytest", "fail"), ("ruff", "cancel"), ("mypy", "pass")],
            False,
            True,
            False,
            "checks failed: pytest, ruff",
        ),
        ([("pytest", "pending")], False, True, False, "checks not done: pytest"),
        ([("pytest", "pass")], True, True, False, "checks did not finish in time"),
        ([], False, True, False, "no checks reported on the PR"),
    ],
)
def test_decide_land(
    checks: list[tuple[str, str]],
    timed_out: bool,
    loop_live: bool,
    enqueue: bool,
    reason: str,
) -> None:
    decision = dw.decide_land(_checks(*checks), timed_out, loop_live)
    assert (decision.enqueue, decision.reason) == (enqueue, reason)


def _probes(neo4j: tuple[str, int], bib: tuple[str, int]) -> dict[str, Any]:
    return {
        "neo4j": _arm("uncapped-neo4j", {**FREEZE, "neo4j": neo4j[0]}, neo4j[1]),
        "bibtexparser": _arm(
            "uncapped-bibtexparser", {**FREEZE, "bibtexparser": bib[0]}, bib[1]
        ),
    }


def test_cap_outcomes(policy: object) -> None:
    out = dw.cap_outcomes(REQS, policy, FREEZE, _probes(("6.0.2", 0), ("1.4.3", 0)))
    assert [
        (c.name, c.capped_version, c.uncapped_version, c.verdict, c.stale_comment)
        for c in out
    ] == [
        ("biocypher", "0.15.2", None, "not probed", False),
        ("neo4j", "4.4.13", "6.0.2", "lifted", False),
        ("bibtexparser", "1.4.3", "1.4.3", "not binding", False),
    ]
    # The 2026-11-17 dry run: bibtexparser 2 breaks the import, neo4j alone is judged
    # on its own arm.
    failed = dw.cap_outcomes(REQS, policy, FREEZE, _probes(("6.4.1", 1), ("2.1.0", 4)))
    assert [c.verdict for c in failed] == [
        "not probed",
        "kept: the suite fails without it",
        "kept: the suite fails without it",
    ]
    # Both pass alone: only the first is lifted, the other waits a run.
    both = dw.cap_outcomes(REQS, policy, FREEZE, _probes(("6.4.1", 0), ("2.1.0", 0)))
    assert [(c.verdict, c.stale_comment) for c in both] == [
        ("not probed", False),
        ("lifted", False),
        ("deferred: passes alone; one cap is lifted per run", False),
    ]
    not_run = dw.cap_outcomes(REQS, policy, FREEZE, {})
    assert [c.verdict for c in not_run] == [
        "not probed",
        "kept: not probed in this run",
        "kept: not probed in this run",
    ]
    lifted_bib = dw.cap_outcomes(
        REQS, policy, FREEZE, _probes(("4.4.13", 0), ("2.1.0", 0))
    )
    assert [(c.verdict, c.stale_comment) for c in lifted_bib] == [
        ("not probed", False),
        ("not binding", False),
        ("lifted", True),  # no comment_marker: its comment is flagged
    ]


def test_pytest_summary() -> None:
    log = (
        "....\nFAILED tests/a.py::t - assert 1 == 2\n"
        "1 failed, 4100 passed, 220 skipped, 3 warnings in 812.33s (0:13:32)\n"
    )
    assert dw.pytest_summary(log) == (
        "1 failed, 4100 passed, 220 skipped, 3 warnings in 812.33s (0:13:32)"
    )
    assert dw.pytest_summary("== 12 passed in 1.02s ==\n") == "12 passed in 1.02s"
    assert dw.pytest_summary("ImportError while loading conftest\n") == (
        "no pytest summary line (see the log)"
    )


def test_pr_number_from_url() -> None:
    assert (
        dw.pr_number_from_url("https://github.com/Mjvolk3/torchcell/pull/1170") == 1170
    )
    with pytest.raises(dw.CommandFailed):
        dw.pr_number_from_url("Warning: 1 uncommitted change")


def _arm(name: str, freeze: dict[str, str], exit_code: int) -> Any:
    return dw.ArmResult(
        name=name,
        requirements="env/requirements.txt",
        stripped_caps=[],
        install_seconds=1.0,
        freeze=freeze,
        pytest_exit=exit_code,
        pytest_summary="1 passed in 1.00s",
        pytest_seconds=1.0,
        log="/x.log",
    )


def _report(**kw: object) -> Any:
    base: dict[str, object] = {
        "date": "2026.11.22",
        "started_at": "2026-11-22T05:00:00",
        "host": "h",
        "repo": "/r",
        "head": "abc",
        "python": "3.13.0",
        "dry_run": False,
        "land_requested": False,
        "work_dir": "/w",
    }
    return dw.DepsReport(**{**base, **kw})


def test_effective_freeze_and_exit_codes(policy: object) -> None:
    probes = _probes(("6.0.2", 0), ("1.4.3", 0))
    caps = dw.cap_outcomes(REQS, policy, FREEZE, probes)
    lifted = _report(capped=_arm("capped", FREEZE, 0), uncapped=probes, caps=caps)
    assert dw.effective_freeze(lifted) == {**FREEZE, "neo4j": "6.0.2"}
    two = [c.model_copy(update={"verdict": "lifted"}) for c in caps[1:]]
    with pytest.raises(ValueError, match="more than one cap lifted"):
        dw.effective_freeze(
            _report(capped=_arm("capped", FREEZE, 0), uncapped=probes, caps=two)
        )
    kept = _report(capped=_arm("capped", FREEZE, 0), caps=[])
    assert dw.effective_freeze(kept) == FREEZE
    red = _report(capped=_arm("capped", FREEZE, 1), status="suite-failed")
    with pytest.raises(ValueError):
        dw.effective_freeze(red)
    assert red.exit_code() == 1
    assert _report(status="job-failed").exit_code() == 2
    assert _report(status="no-change").exit_code() == 0
    assert _report(status="changes").exit_code() == 0
