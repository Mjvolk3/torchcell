# scripts/deps_weekly.py
# [[scripts.deps_weekly]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/deps_weekly.py
"""Weekly dependency update: install the newest allowed versions, test, raise floors.

``run`` (meant for a fresh detached worktree of ``origin/main``, see
``scripts/deps_weekly_cron.sh``):

1. picks the newest torch whose PyG wheel page lists a cp313 linux x86_64
   ``torch_scatter`` wheel (PyPI's latest torch is often ahead of PyG);
2. arm ``capped``: a new venv, installed in the CI order (torch from the CPU index,
   ``env/requirements.txt``, torch-scatter from the PyG page, the test tools, then the
   checkout), and ``pytest tests/torchcell`` run hermetically;
3. diffs the resolution against ``env/tested/requirements.lock.txt`` (which packages
   moved);
4. one arm ``uncapped-<name>`` per cap with ``probe = true`` in ``env/deps_policy.toml``
   has ``probe = true``): the same with the probed caps removed; a cap the resolver then
   exceeds, with the suite still passing, is lifted;
5. when ``capped`` passed: rewrites the lock, ``env/torch-ci-version.txt``, the floors in
   ``env/requirements.txt`` (never lowered, never for ``[hold]``) and lifted caps, then
   commits on ``deps/weekly-<date>``, pushes and opens a PR. The subject is
   ``PATCH(deps): ...`` (a PyPI release once landed) when ``env/requirements.txt``
   changed, else ``build(deps): ...``. ``--land`` waits for the PR's checks and enqueues
   it in the merge queue;
6. when ``capped`` failed: changes nothing and files (or comments on) one issue.

Exit codes: 0 done (no change, changes made, or a dry run), 1 the suite failed on the
newest dependencies, 2 the job itself failed (an install, git or gh command, a refusal).
A JSON report is always written to
``${XDG_STATE_HOME:-~/.local/state}/torchcell/deps-weekly/<date>.json``; ``report``
prints the newest one, ``write-lock`` writes the lock from a report through the same
code path ``run`` uses.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import tomllib
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict

REPO_SLUG = "Mjvolk3/torchcell"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYG_PAGE = "https://data.pyg.org/whl/torch-{version}+cpu.html"
PYPI_TORCH_JSON = "https://pypi.org/pypi/torch/json"

REQUIREMENTS = Path("env/requirements.txt")
REQUIREMENTS_DEPENDENT = Path("env/requirements_dependent.txt")
REQUIREMENTS_TEST = Path("env/requirements_test.txt")
POLICY = Path("env/deps_policy.toml")
LOCK = Path("env/tested/requirements.lock.txt")
TORCH_CI = Path("env/torch-ci-version.txt")

DEFAULT_WORK_ROOT = Path("/scratch/tmp/torchcell-deps-weekly")
DEPRECATED_DIR = "/scratch/projects/torchcell-deprecated"
FAILURE_ISSUE_TITLE = "deps-weekly: test suite fails on the newest dependencies"
ISSUE_LABEL = "torchcell-lib"
LAND_TIMEOUT_S = 7200
CHECKS_REGISTER_POLL_S = 30
# Tooling of the job's own venvs, not part of the resolution being tested.
TOOL_PACKAGES = frozenset({"torchcell", "pip", "uv"})
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
)
CO_AUTHOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PR_FOOTER = "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
LOCK_HEADER = (
    "# env/tested/requirements.lock.txt\n"
    "# Written by scripts/deps_weekly.py, never by hand: the full resolution of the last\n"
    "# weekly run whose test suite passed (the last known good). See env/tested/README.md.\n"
)

# "capped", or "uncapped-<canonical name of the probed cap>".
ArmName = str
Status = Literal["running", "no-change", "changes", "suite-failed", "job-failed"]
CapVerdict = Literal[
    "not probed",
    "not binding",
    "lifted",
    "kept: the suite fails without it",
    "kept: not probed in this run",
    "deferred: passes alone; one cap is lifted per run",
]


class Refusal(RuntimeError):
    """A precondition of ``run`` does not hold; nothing was done."""


class CommandFailed(RuntimeError):
    """An external command of the job failed (an install, git, gh): exit 2."""


# ── Policy ────────────────────────────────────────────────────────────────────


class CapPolicy(BaseModel):
    """One upper bound or exact pin in ``env/requirements.txt``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: str
    probe: bool
    comment_marker: str | None = None


class DepsPolicy(BaseModel):
    """``env/deps_policy.toml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    python: str
    hold: dict[str, str]
    caps: dict[str, CapPolicy]

    def held(self) -> frozenset[str]:
        """Canonical names whose floor is never raised."""
        return frozenset(canonicalize_name(n) for n in self.hold)

    def cap(self, name: str) -> CapPolicy | None:
        """The cap entry for ``name`` (any spelling), or None."""
        by_canon = {canonicalize_name(k): v for k, v in self.caps.items()}
        return by_canon.get(canonicalize_name(name))


def load_policy(path: Path) -> DepsPolicy:
    """Read and validate the policy file."""
    with path.open("rb") as handle:
        return DepsPolicy.model_validate(tomllib.load(handle))


# ── Requirement lines ─────────────────────────────────────────────────────────

# name, extras, comma-joined specifiers (no whitespace), environment marker, then
# trailing whitespace and an optional inline comment, kept verbatim.
_REQ_LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?P<extras>\[[^\]]*\])?"
    r"(?P<spec>[^\s;#]*)"
    r"(?P<marker>\s*;[^#]*?)?"
    r"(?P<rest>\s*(?:#.*)?)$"
)
_SPEC = re.compile(r"^(~=|===|==|!=|<=|>=|<|>)(.+)$")
_FLOOR_OPS = frozenset({">=", ">"})


class ReqLine(BaseModel):
    """A requirement line split so it can be re-rendered byte for byte."""

    model_config = ConfigDict(frozen=True)

    name: str
    extras: str
    specs: tuple[str, ...]
    marker: str
    rest: str

    @property
    def canonical(self) -> str:
        """PEP 503 normalized name (``torch_geometric`` -> ``torch-geometric``)."""
        return canonicalize_name(self.name)

    def render(self) -> str:
        """The line text."""
        return f"{self.name}{self.extras}{','.join(self.specs)}{self.marker}{self.rest}"


def _split_spec(spec: str) -> tuple[str, str]:
    match = _SPEC.match(spec)
    if match is None:
        raise ValueError(f"unparseable specifier {spec!r}")
    return match.group(1), match.group(2)


def parse_requirement_line(line: str) -> ReqLine | None:
    """Parse one line; None for a blank or comment line, ValueError for anything else."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("-"):
        raise ValueError(
            f"pip options are not supported in a requirements line: {line!r}"
        )
    match = _REQ_LINE.match(line)
    if match is None:
        raise ValueError(f"unparseable requirement line: {line!r}")
    req = ReqLine(
        name=match.group("name"),
        extras=match.group("extras") or "",
        specs=tuple(s for s in match.group("spec").split(",") if s),
        marker=match.group("marker") or "",
        rest=match.group("rest"),
    )
    for spec in req.specs:
        _split_spec(spec)
    # packaging is the arbiter of validity; the regex only splits.
    Requirement(f"{req.name}{req.extras}{','.join(req.specs)}{req.marker}")
    return req


def is_bounded(req: ReqLine) -> bool:
    """True when the line carries anything other than a floor (``<``, ``==``, ...)."""
    return any(_split_spec(s)[0] not in _FLOOR_OPS for s in req.specs)


def raise_floor(req: ReqLine, tested: str) -> ReqLine:
    """Raise the ``>=`` floor to the tested version's public part; never lower it."""
    target = Version(Version(tested).public)
    specs: list[str] = []
    for spec in req.specs:
        op, version = _split_spec(spec)
        if op == ">=" and target > Version(version):
            specs.append(f">={target}")
        else:
            specs.append(spec)
    return req.model_copy(update={"specs": tuple(specs)})


def strip_cap(req: ReqLine) -> ReqLine:
    """Drop every specifier that is not a floor (upper bounds, pins, exclusions)."""
    kept = tuple(s for s in req.specs if _split_spec(s)[0] in _FLOOR_OPS)
    return req.model_copy(update={"specs": kept})


def requirement_lines(text: str) -> list[ReqLine]:
    """Every requirement in a requirements file, in order."""
    return [r for r in (parse_requirement_line(x) for x in text.split("\n")) if r]


def strip_one_cap(text: str, name: str) -> str:
    """The requirements text with the bound of ``name`` (any spelling) removed."""
    out: list[str] = []
    found = False
    for line in text.split("\n"):
        req = parse_requirement_line(line)
        if req is not None and req.canonical == canonicalize_name(name):
            out.append(strip_cap(req).render())
            found = True
        else:
            out.append(line)
    if not found:
        raise ValueError(f"{name} is not in the requirements text")
    return "\n".join(out)


class FloorRaise(BaseModel):
    """One floor the job raised (or would raise, in a dry run)."""

    name: str
    old: str
    new: str


class RequirementsRewrite(BaseModel):
    """Result of rewriting ``env/requirements.txt``."""

    text: str
    raises: list[FloorRaise]
    stale_comments: list[str]


def _comment_block_above(lines: list[str], index: int) -> list[int]:
    """Indices of the contiguous comment lines directly above ``lines[index]``."""
    block: list[int] = []
    j = index - 1
    while j >= 0 and lines[j].lstrip().startswith("#"):
        block.insert(0, j)
        j -= 1
    return block


def rewrite_requirements(
    text: str, freeze: Mapping[str, str], policy: DepsPolicy, lifted: frozenset[str]
) -> RequirementsRewrite:
    """Raise floors to the tested versions and remove lifted caps.

    Comments and blank lines are kept byte for byte, and so is every requirement line
    the rewrite does not change. ``freeze`` maps canonical names to tested versions and
    must hold every direct dependency. A lifted cap with a ``comment_marker`` takes the
    comment block directly above its line with it; without one the comment is kept and
    the package is listed in ``stale_comments``.
    """
    held = policy.held()
    lines = text.split("\n")
    drop: set[int] = set()
    raises: list[FloorRaise] = []
    stale: list[str] = []
    for i, line in enumerate(lines):
        req = parse_requirement_line(line)
        if req is None:
            continue
        if req.canonical not in freeze:
            raise ValueError(f"{req.name} is a direct dependency but was not installed")
        new = req
        if req.canonical in lifted:
            cap = policy.cap(req.name)
            if cap is None:
                raise ValueError(f"{req.name} was lifted but has no [caps] entry")
            new = strip_cap(new)
            if cap.comment_marker is None:
                stale.append(req.name)
            else:
                block = _comment_block_above(lines, i)
                if not any(cap.comment_marker in lines[j] for j in block):
                    raise ValueError(
                        f"comment_marker {cap.comment_marker!r} of {req.name} is not in "
                        "the comment block above its line"
                    )
                drop.update(block)
        if req.canonical not in held:
            new = raise_floor(new, freeze[req.canonical])
        old_floor = _floor(req)
        new_floor = _floor(new)
        if old_floor is not None and new_floor is not None and new_floor != old_floor:
            raises.append(FloorRaise(name=req.name, old=old_floor, new=new_floor))
        if new != req:
            lines[i] = new.render()
    kept = [line for i, line in enumerate(lines) if i not in drop]
    return RequirementsRewrite(
        text="\n".join(kept), raises=raises, stale_comments=stale
    )


def _floor(req: ReqLine) -> str | None:
    for spec in req.specs:
        op, version = _split_spec(spec)
        if op == ">=":
            return version
    return None


def drop_cap_table(policy_text: str, name: str) -> str:
    """Delete the ``[caps.<name>]`` table (header through the line before the next)."""
    lines = policy_text.split("\n")
    header = f"[caps.{name}]"
    if header not in lines:
        raise ValueError(f"{header} is not in the policy file")
    start = lines.index(header)
    end = start + 1
    while end < len(lines) and not lines[end].startswith("["):
        end += 1
    kept = lines[:start] + lines[end:]
    return "\n".join(kept).rstrip("\n") + "\n"


# ── Freeze, lock, diff ───────────────────────────────────────────────────────


def parse_freeze(text: str) -> dict[str, str]:
    """``name==version`` lines to {canonical name: version}, without the job's tools."""
    out: dict[str, str] = {}
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-e "):
            continue
        name, sep, version = line.partition("==")
        if not sep:
            raise ValueError(f"freeze line is not name==version: {line!r}")
        canon = canonicalize_name(name)
        if canon not in TOOL_PACKAGES:
            out[canon] = version
    return out


def render_lock(freeze: Mapping[str, str]) -> str:
    """The lock file text: the header, then ``name==version`` sorted by name."""
    body = "".join(f"{name}=={freeze[name]}\n" for name in sorted(freeze))
    return LOCK_HEADER + body


class VersionChange(BaseModel):
    """A package whose version moved."""

    old: str
    new: str


class FreezeDiff(BaseModel):
    """Which packages moved between two resolutions."""

    added: dict[str, str]
    removed: dict[str, str]
    changed: dict[str, VersionChange]

    def count(self) -> int:
        """Number of packages that moved in any way."""
        return len(self.added) + len(self.removed) + len(self.changed)


def diff_freeze(old: Mapping[str, str], new: Mapping[str, str]) -> FreezeDiff:
    """Added, removed and changed packages from ``old`` to ``new``."""
    return FreezeDiff(
        added={k: new[k] for k in sorted(set(new) - set(old))},
        removed={k: old[k] for k in sorted(set(old) - set(new))},
        changed={
            k: VersionChange(old=old[k], new=new[k])
            for k in sorted(set(old) & set(new))
            if old[k] != new[k]
        },
    )


# ── Torch selection ──────────────────────────────────────────────────────────

_SCATTER_WHEEL = re.compile(r"torch_scatter-[^\"<>/\s]+-cp313-cp313-linux_x86_64\.whl")


def scatter_wheel_listed(page: str) -> bool:
    """True when a PyG wheel page lists a cp313 linux x86_64 torch_scatter wheel."""
    return _SCATTER_WHEEL.search(page) is not None


def final_releases(pypi_json: Mapping[str, Any]) -> list[Version]:
    """Final (no pre/dev) releases with at least one unyanked file, newest first."""
    out: list[Version] = []
    for key, files in pypi_json["releases"].items():
        try:
            version = Version(key)
        except InvalidVersion:  # a legacy version string; never a candidate
            continue
        if version.is_prerelease or version.is_devrelease:
            continue
        if not any(not f.get("yanked", False) for f in files):
            continue
        out.append(version)
    return sorted(out, reverse=True)


class TorchCheck(BaseModel):
    """One PyG wheel page looked at while picking torch."""

    version: str
    verdict: Literal["torch_scatter wheel listed", "no torch_scatter wheel", "no page"]


class TorchChoice(BaseModel):
    """The torch the job tests, and why."""

    pypi_latest: str
    testable: str
    ci_before: str
    checked: list[TorchCheck]


def select_torch(
    pypi_json: Mapping[str, Any],
    floor: Version,
    ci_before: str,
    fetch_page: Callable[[str], str | None],
) -> TorchChoice:
    """Newest final torch >= ``floor`` whose PyG page lists a torch_scatter wheel.

    ``fetch_page`` returns the page text, or None when the page does not exist (404).
    """
    releases = final_releases(pypi_json)
    checked: list[TorchCheck] = []
    for version in releases:
        if version < floor:
            break
        page = fetch_page(PYG_PAGE.format(version=version))
        if page is None:
            checked.append(TorchCheck(version=str(version), verdict="no page"))
        elif scatter_wheel_listed(page):
            checked.append(
                TorchCheck(version=str(version), verdict="torch_scatter wheel listed")
            )
            return TorchChoice(
                pypi_latest=str(releases[0]),
                testable=str(version),
                ci_before=ci_before,
                checked=checked,
            )
        else:
            checked.append(
                TorchCheck(version=str(version), verdict="no torch_scatter wheel")
            )
    raise CommandFailed(
        f"no torch >= {floor} has a cp313 linux x86_64 torch_scatter wheel on data.pyg.org"
    )


def _fetch(url: str) -> str | None:
    """GET ``url``; None on a 404, CommandFailed on any other failure."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "torchcell-deps-weekly"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body: bytes = response.read()
            return body.decode("utf-8")
    except urllib.error.HTTPError as err:
        if err.code == 404:  # PyG has no page for this torch: data, not a failure
            return None
        raise CommandFailed(f"GET {url}: HTTP {err.code}") from err
    except urllib.error.URLError as err:
        raise CommandFailed(f"GET {url}: {err.reason}") from err


# ── Decisions ────────────────────────────────────────────────────────────────


def commit_subject(requirements_changed: bool, date: str) -> str:
    """``PATCH(deps)`` (a PyPI release once landed) only when the runtime list moved."""
    if requirements_changed:
        return f"PATCH(deps): weekly dependency update {date}"
    return f"build(deps): weekly tested-versions snapshot {date}"


class CheckState(BaseModel):
    """One PR check as ``gh pr checks --json name,bucket`` reports it."""

    name: str
    bucket: str


class LandDecision(BaseModel):
    """Whether the job enqueues its PR for the merge queue."""

    enqueue: bool
    reason: str


def decide_land(
    checks: list[CheckState], timed_out: bool, loop_live: bool
) -> LandDecision:
    """Enqueue only when every check passed (or skipped) and the queue loop is live."""
    if timed_out:
        return LandDecision(enqueue=False, reason="checks did not finish in time")
    if not checks:
        return LandDecision(enqueue=False, reason="no checks reported on the PR")
    failed = sorted(c.name for c in checks if c.bucket in {"fail", "cancel"})
    if failed:
        return LandDecision(enqueue=False, reason=f"checks failed: {', '.join(failed)}")
    pending = sorted(c.name for c in checks if c.bucket not in {"pass", "skipping"})
    if pending:
        return LandDecision(
            enqueue=False, reason=f"checks not done: {', '.join(pending)}"
        )
    if not loop_live:
        return LandDecision(
            enqueue=False, reason="merge-queue loop is not live; PR left open"
        )
    return LandDecision(enqueue=True, reason="all checks passed; enqueued")


class CapOutcome(BaseModel):
    """What the job found for one ``[caps]`` entry."""

    name: str
    line: str
    probe: bool
    capped_version: str | None
    uncapped_version: str | None
    verdict: CapVerdict
    stale_comment: bool = False


def cap_outcomes(
    requirements_text: str,
    policy: DepsPolicy,
    capped: Mapping[str, str],
    probes: Mapping[str, ArmResult],
) -> list[CapOutcome]:
    """One verdict per cap, from the arm that removed that cap alone.

    ``probes`` maps a cap's canonical name to its ``uncapped-<name>`` arm. A cap binds
    when its arm installed a version outside the bound. A binding cap whose arm passed
    is lifted, but only the first such cap in file order: each arm tested one cap
    removed, so two lifted together would be an untested combination; the others are
    deferred to the next run, which tests them on top of this one.
    """
    out: list[CapOutcome] = []
    lifted_one = False
    for req in requirement_lines(requirements_text):
        cap = policy.cap(req.name)
        if cap is None:
            continue
        cv = capped.get(req.canonical)
        arm = probes.get(req.canonical)
        uv = arm.freeze.get(req.canonical) if arm is not None else None
        verdict: CapVerdict
        if not cap.probe:
            verdict = "not probed"
        elif arm is None or uv is None:
            verdict = "kept: not probed in this run"
        elif Version(uv) in SpecifierSet(",".join(req.specs)):
            verdict = "not binding"
        elif not arm.passed:
            verdict = "kept: the suite fails without it"
        elif lifted_one:
            verdict = "deferred: passes alone; one cap is lifted per run"
        else:
            verdict = "lifted"
            lifted_one = True
        out.append(
            CapOutcome(
                name=req.name,
                line=req.render(),
                probe=cap.probe,
                capped_version=cv,
                uncapped_version=uv,
                verdict=verdict,
                stale_comment=verdict == "lifted" and cap.comment_marker is None,
            )
        )
    return out


# ── Report ───────────────────────────────────────────────────────────────────


class ArmResult(BaseModel):
    """One install-and-test arm."""

    name: ArmName
    requirements: str
    stripped_caps: list[str]
    install_seconds: float
    freeze: dict[str, str]
    pytest_exit: int
    pytest_summary: str
    pytest_seconds: float
    log: str

    @property
    def passed(self) -> bool:
        """The suite passed."""
        return self.pytest_exit == 0


class LandOutcome(BaseModel):
    """What ``--land`` did."""

    pr_number: int
    checks: list[CheckState]
    loop_live: bool
    decision: LandDecision


class DepsReport(BaseModel):
    """Everything one run did; dumped as JSON, rendered as the PR and commit body."""

    date: str
    started_at: str
    finished_at: str | None = None
    host: str
    repo: str
    head: str
    python: str
    dry_run: bool
    land_requested: bool
    work_dir: str
    status: Status = "running"
    torch: TorchChoice | None = None
    capped: ArmResult | None = None
    # canonical cap name -> its uncapped-<name> arm
    uncapped: dict[str, ArmResult] = {}
    baseline: str | None = None
    moved: FreezeDiff | None = None
    floor_raises: list[FloorRaise] = []
    caps: list[CapOutcome] = []
    torch_ci_new: str | None = None
    files_changed: list[str] = []
    commit_subject: str | None = None
    branch: str | None = None
    pr_url: str | None = None
    issue_url: str | None = None
    land: LandOutcome | None = None
    error: str | None = None

    def exit_code(self) -> int:
        """0 done, 1 the suite failed on the newest dependencies, 2 the job failed."""
        return {"suite-failed": 1, "job-failed": 2}.get(self.status, 0)


def effective_freeze(report: DepsReport) -> dict[str, str]:
    """The resolution the rewritten files describe: the lifted cap's arm, else capped."""
    if report.capped is None or not report.capped.passed:
        raise ValueError("the capped arm did not pass; there is no tested resolution")
    lifted = [c.name for c in report.caps if c.verdict == "lifted"]
    if len(lifted) > 1:
        raise ValueError(f"more than one cap lifted in one run: {lifted}")
    if lifted:
        return report.uncapped[canonicalize_name(lifted[0])].freeze
    return report.capped.freeze


def _arm_line(arm: ArmResult | None) -> str:
    if arm is None:
        return "not run"
    verdict = "PASS" if arm.passed else f"FAIL (exit {arm.pytest_exit})"
    return (
        f"{verdict}: {arm.pytest_summary}; install {arm.install_seconds:.0f} s, "
        f"tests {arm.pytest_seconds:.0f} s"
    )


def render_markdown(report: DepsReport) -> str:
    """The report as markdown: the PR body, the commit body and the issue body."""
    lines = [f"## deps-weekly {report.date}", "", "| | |", "|---|---|"]
    status = report.status + (" (dry run)" if report.dry_run else "")
    lines.append(f"| status | {status} |")
    lines.append(f"| python | {report.python} |")
    if report.torch is not None:
        t = report.torch
        lines.append(
            f"| torch | tested {t.testable} (PyPI latest {t.pypi_latest}; CI was "
            f"{t.ci_before}) |"
        )
    lines.append(f"| capped arm | {_arm_line(report.capped)} |")
    if not report.uncapped:
        lines.append("| uncapped arms | not run |")
    for cap_name, arm in report.uncapped.items():
        lines.append(f"| uncapped arm ({cap_name}) | {_arm_line(arm)} |")
    lines.append(f"| head | {report.head} |")
    lines.append("")
    lines.append("### Packages that moved")
    lines.append("")
    if report.moved is None:
        n = len(report.capped.freeze) if report.capped else 0
        lines.append(
            f"First snapshot: no `{LOCK}` to compare with ({n} packages resolved)."
        )
    elif report.moved.count() == 0:
        lines.append(f"Nothing moved against `{report.baseline}`.")
    else:
        lines.append(f"Against `{report.baseline}`:")
        lines.append("")
        lines.append("| package | before | after |")
        lines.append("|---|---|---|")
        for name, change in report.moved.changed.items():
            lines.append(f"| {name} | {change.old} | {change.new} |")
        for name, version in report.moved.added.items():
            lines.append(f"| {name} | (new) | {version} |")
        for name, version in report.moved.removed.items():
            lines.append(f"| {name} | {version} | (removed) |")
    lines.append("")
    lines.append("### Floors raised in env/requirements.txt")
    lines.append("")
    if report.floor_raises:
        lines.append("| package | old floor | new floor |")
        lines.append("|---|---|---|")
        for r in report.floor_raises:
            lines.append(f"| {r.name} | {r.old} | {r.new} |")
    else:
        lines.append("None.")
    lines.append("")
    lines.append("### Caps")
    lines.append("")
    if report.caps:
        lines.append("| package | line | capped | uncapped | verdict |")
        lines.append("|---|---|---|---|---|")
        for c in report.caps:
            verdict = c.verdict + (
                " (comment above is now stale)" if c.stale_comment else ""
            )
            lines.append(
                f"| {c.name} | `{c.line}` | {c.capped_version or '-'} | "
                f"{c.uncapped_version or '-'} | {verdict} |"
            )
    else:
        lines.append("Not evaluated.")
    lines.append("")
    lines.append("### Files changed")
    lines.append("")
    lines.extend(f"- `{f}`" for f in report.files_changed)
    if not report.files_changed:
        lines.append("None.")
    if report.torch_ci_new is not None and report.torch is not None:
        lines.append("")
        lines.append(
            f"CI torch moves from {report.torch.ci_before} to {report.torch_ci_new} "
            f"(`{TORCH_CI}`)."
        )
    if report.error:
        lines.append("")
        lines.append(f"Error: {report.error}")
    return "\n".join(lines) + "\n"


def status_line(report: DepsReport) -> str:
    """One line for a log or a jobs panel."""
    moved = "first-snapshot" if report.moved is None else str(report.moved.count())
    lifted = [c.name for c in report.caps if c.verdict == "lifted"]
    parts = [
        f"deps-weekly {report.date}: {report.status.upper()}",
        f"exit={report.exit_code()}",
        f"torch={report.torch.testable if report.torch else '-'}",
        f"capped={'pass' if report.capped and report.capped.passed else ('fail' if report.capped else '-')}",
        "uncapped="
        + (
            ",".join(
                f"{n}:{'pass' if a.passed else 'fail'}"
                for n, a in report.uncapped.items()
            )
            or "-"
        ),
        f"moved={moved}",
        f"floors={len(report.floor_raises)}",
        f"caps-lifted={','.join(lifted) or '-'}",
        f"files={len(report.files_changed)}",
    ]
    if report.dry_run:
        parts.append("dry-run")
    if report.pr_url:
        parts.append(f"pr={report.pr_url}")
    if report.issue_url:
        parts.append(f"issue={report.issue_url}")
    if report.land is not None:
        parts.append(
            f"land={'enqueued' if report.land.decision.enqueue else 'left-open'}"
        )
    return " ".join(parts)


def state_dir() -> Path:
    """``${XDG_STATE_HOME:-~/.local/state}/torchcell/deps-weekly``."""
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "torchcell" / "deps-weekly"


def newest_report(directory: Path) -> Path:
    """The newest ``<date>.json`` report in ``directory``."""
    reports = sorted(directory.glob("*.json"))
    if not reports:
        raise FileNotFoundError(f"no deps-weekly report in {directory}")
    return reports[-1]


# ── External commands ────────────────────────────────────────────────────────


def _tail(path: Path, n: int) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])


def run_logged(cmd: list[str], log: Path, env: Mapping[str, str], cwd: Path) -> float:
    """Run ``cmd`` appending its output to ``log``; raise CommandFailed on non-zero."""
    start = time.monotonic()
    with log.open("a") as handle:
        handle.write(f"\n$ {' '.join(cmd)}\n")
        handle.flush()
        code = subprocess.run(
            cmd, stdout=handle, stderr=subprocess.STDOUT, env=dict(env), cwd=cwd
        ).returncode
    if code != 0:
        raise CommandFailed(
            f"exit {code}: {' '.join(cmd)}\n--- last lines of {log} ---\n{_tail(log, 40)}"
        )
    return time.monotonic() - start


def _capture(cmd: list[str], cwd: Path) -> str:
    """Run a short command and return stdout; raise CommandFailed on non-zero."""
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise CommandFailed(
            f"exit {result.returncode}: {' '.join(cmd)}\n{result.stderr}"
        )
    return result.stdout


_SUMMARY = re.compile(
    r"\d+ (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b.* in "
    r"[\d.]+s"
)


def pytest_summary(log_text: str) -> str:
    """The final pytest summary line, without the ``=`` decoration."""
    for line in reversed(log_text.splitlines()):
        if _SUMMARY.search(line):
            return line.strip().strip("=").strip()
    return "no pytest summary line (see the log)"


# ── The job ──────────────────────────────────────────────────────────────────


class Job:
    """One ``run``: state shared by the steps."""

    def __init__(self, args: argparse.Namespace, repo: Path, date: str) -> None:
        """Hold the CLI args, the worktree and the run date; nothing runs yet."""
        self.args = args
        self.repo = repo
        self.date = date
        self.work: Path = args.work if args.work else DEFAULT_WORK_ROOT / date
        self.report = DepsReport(
            date=date,
            started_at=datetime.now().isoformat(timespec="seconds"),
            host=socket.gethostname(),
            repo=str(repo),
            head="",
            python=platform.python_version(),
            dry_run=args.dry_run,
            land_requested=args.land,
            work_dir=str(self.work),
        )
        self.work_created = False

    # -- preconditions --------------------------------------------------------

    def preflight(self) -> DepsPolicy:
        """Refuse unless the repo is a clean worktree and the work dir is empty."""
        top = _capture(["git", "rev-parse", "--show-toplevel"], self.repo).strip()
        if Path(top).resolve() != self.repo.resolve():
            raise Refusal(f"run from the worktree root ({top}), not {self.repo}")
        dirty = _capture(["git", "status", "--porcelain"], self.repo).strip()
        if dirty:
            raise Refusal(f"the worktree is not clean:\n{dirty}")
        self.report.head = _capture(["git", "rev-parse", "HEAD"], self.repo).strip()
        policy = load_policy(self.repo / POLICY)
        if platform.python_version() != policy.python:
            raise Refusal(
                f"the job runs on Python {policy.python} ({POLICY}); this is "
                f"{platform.python_version()} at {sys.executable}"
            )
        if self.work.exists() and any(self.work.iterdir()):
            raise Refusal(f"work dir {self.work} is not empty")
        self.work.mkdir(parents=True, exist_ok=True)
        self.work_created = True
        for sub in ("uv-cache", "pip-cache", "tmp", "logs"):
            (self.work / sub).mkdir()
        return policy

    # -- environments ---------------------------------------------------------

    def base_env(self) -> dict[str, str]:
        """The parent environment minus anything that leaks another Python setup."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"PYTHONPATH", "TC_DATA_URL", "VIRTUAL_ENV", "PYTHONHOME"}
        }
        env["UV_CACHE_DIR"] = str(self.work / "uv-cache")
        env["PIP_CACHE_DIR"] = str(self.work / "pip-cache")
        env["TMPDIR"] = str(self.work / "tmp")
        return env

    def install_arm(
        self, arm: ArmName, requirements: Path, torch_version: str, log: Path
    ) -> tuple[Path, float]:
        """New venv, installed in the CI order. Returns the venv python and seconds."""
        env = self.base_env()
        venv = self.work / f"venv-{arm}"
        py = venv / "bin" / "python"
        uv = venv / "bin" / "uv"
        constraints = self.work / "constraints-torch.txt"
        constraints.write_text(f"torch=={torch_version}\n")
        seconds = run_logged(
            [sys.executable, "-m", "venv", str(venv)], log, env, self.repo
        )
        seconds += run_logged(
            [str(py), "-m", "pip", "install", "--disable-pip-version-check", "uv"],
            log,
            env,
            self.repo,
        )
        uvi = [str(uv), "pip", "install", "--no-config", "--python", str(py)]
        pin = ["--constraint", str(constraints)]
        steps = [
            [f"torch=={torch_version}", "--index-url", TORCH_CPU_INDEX],
            ["-r", str(requirements), *pin],
            [
                "-r",
                str(REQUIREMENTS_DEPENDENT),
                "--find-links",
                PYG_PAGE.format(version=torch_version),
                "--only-binary",
                "torch-scatter",
                *pin,
            ],
            ["-r", str(REQUIREMENTS_TEST), *pin],
            ["-e", ".", "--no-deps"],
        ]
        for step in steps:
            seconds += run_logged(uvi + step, log, env, self.repo)
        return py, seconds

    def freeze(self, py: Path) -> dict[str, str]:
        """The arm's resolution."""
        uv = py.parent / "uv"
        out = _capture(
            [
                str(uv),
                "pip",
                "freeze",
                "--no-config",
                "--python",
                str(py),
                "--exclude-editable",
            ],
            self.repo,
        )
        return parse_freeze(out)

    def run_pytest(self, arm: ArmName, py: Path, log: Path) -> tuple[int, str, float]:
        """``pytest tests/torchcell`` hermetically, niced; exit, summary, seconds."""
        env = self.base_env()
        data_root = self.work / f"data-root-{arm}"
        data_root.mkdir()
        env["DATA_ROOT"] = str(data_root)
        env["PYTHONHASHSEED"] = "0"
        env["VIRTUAL_ENV"] = str(py.parent.parent)
        env["PATH"] = f"{py.parent}{os.pathsep}{env.get('PATH', '')}"
        threads = str(min(8, os.cpu_count() or 1))
        for var in THREAD_VARS:
            env[var] = threads
        cmd = ["nice", "-n", "10", str(py), "-m", "pytest", "tests/torchcell", "-q"]
        cmd += ["-p", "no:cacheprovider"]
        start = time.monotonic()
        with log.open("w") as handle:
            handle.write(f"$ {' '.join(cmd)}\n")
            handle.flush()
            code = subprocess.run(
                cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=self.repo
            ).returncode
        seconds = time.monotonic() - start
        return code, pytest_summary(log.read_text(errors="replace")), seconds

    def arm(
        self, name: ArmName, requirements: Path, stripped: list[str], torch_version: str
    ) -> ArmResult:
        """Install and test one arm."""
        logs = self.work / "logs"
        print(f"[deps-weekly] arm {name}: installing ({requirements})", flush=True)
        py, install_s = self.install_arm(
            name, requirements, torch_version, logs / f"install-{name}.log"
        )
        frozen = self.freeze(py)
        (logs / f"freeze-{name}.txt").write_text(render_lock(frozen))
        print(f"[deps-weekly] arm {name}: pytest tests/torchcell", flush=True)
        test_log = logs / f"pytest-{name}.log"
        code, summary, test_s = self.run_pytest(name, py, test_log)
        print(f"[deps-weekly] arm {name}: exit {code}: {summary}", flush=True)
        return ArmResult(
            name=name,
            requirements=str(requirements),
            stripped_caps=stripped,
            install_seconds=round(install_s, 1),
            freeze=frozen,
            pytest_exit=code,
            pytest_summary=summary,
            pytest_seconds=round(test_s, 1),
            log=str(test_log),
        )

    # -- the run --------------------------------------------------------------

    def execute(self) -> None:
        """Steps c through k; fills ``self.report``."""
        policy = self.preflight()
        repo = self.repo
        req_text = (repo / REQUIREMENTS).read_text()
        torch_floor = _floor(
            next(r for r in requirement_lines(req_text) if r.canonical == "torch")
        )
        if torch_floor is None:
            raise Refusal(f"torch has no >= floor in {REQUIREMENTS}")
        pypi_text = _fetch(PYPI_TORCH_JSON)
        if pypi_text is None:
            raise CommandFailed(f"GET {PYPI_TORCH_JSON}: HTTP 404")
        pypi = json.loads(pypi_text)
        ci_before = (repo / TORCH_CI).read_text().strip()
        choice = select_torch(pypi, Version(torch_floor), ci_before, _fetch)
        self.report.torch = choice
        print(
            f"[deps-weekly] torch {choice.testable} (PyPI latest {choice.pypi_latest})",
            flush=True,
        )

        capped = self.arm("capped", repo / REQUIREMENTS, [], choice.testable)
        self.report.capped = capped
        if (repo / LOCK).exists():
            self.report.baseline = str(LOCK)
            self.report.moved = diff_freeze(
                parse_freeze((repo / LOCK).read_text()), capped.freeze
            )
        dirty = _capture(["git", "status", "--porcelain"], repo).strip()
        if dirty:
            raise CommandFailed(f"the test suite left files in the worktree:\n{dirty}")

        if not capped.passed:
            self.report.status = "suite-failed"
            if not self.args.dry_run:
                self.report.issue_url = self.file_issue(capped)
            return

        # One arm per probed cap, so a cap that breaks the suite cannot hide another
        # (2026-11-17: bibtexparser 2 failed at conftest import and masked neo4j 6).
        for key in [n for n, c in policy.caps.items() if c.probe]:
            canon = canonicalize_name(key)
            probe_req = self.work / f"requirements.uncapped-{canon}.txt"
            probe_req.write_text(strip_one_cap(req_text, key))
            self.report.uncapped[canon] = self.arm(
                f"uncapped-{canon}", probe_req, [key], choice.testable
            )
        self.report.caps = cap_outcomes(
            req_text, policy, capped.freeze, self.report.uncapped
        )
        self.plan_and_apply(policy, req_text, choice)

    def plan_and_apply(
        self, policy: DepsPolicy, req_text: str, choice: TorchChoice
    ) -> None:
        """Step h-j: compute the new tracked files, write and ship them unless dry."""
        repo = self.repo
        freeze = effective_freeze(self.report)
        if self.report.baseline is not None:  # moved against what the files will say
            self.report.moved = diff_freeze(
                parse_freeze((repo / LOCK).read_text()), freeze
            )
        lifted = frozenset(
            canonicalize_name(c.name) for c in self.report.caps if c.verdict == "lifted"
        )
        rewrite = rewrite_requirements(req_text, freeze, policy, lifted)
        self.report.floor_raises = rewrite.raises
        policy_text = (repo / POLICY).read_text()
        for cap in self.report.caps:
            if cap.verdict == "lifted":
                key = next(
                    k
                    for k in policy.caps
                    if canonicalize_name(k) == canonicalize_name(cap.name)
                )
                policy_text = drop_cap_table(policy_text, key)
        new_files: dict[Path, str] = {
            LOCK: render_lock(freeze),
            REQUIREMENTS: rewrite.text,
            POLICY: policy_text,
        }
        if choice.testable != choice.ci_before:
            new_files[TORCH_CI] = choice.testable + "\n"
            self.report.torch_ci_new = choice.testable
        changed = [
            path
            for path, text in new_files.items()
            if not (repo / path).exists() or (repo / path).read_text() != text
        ]
        self.report.files_changed = [str(p) for p in changed]
        if not changed:
            self.report.status = "no-change"
            return
        self.report.status = "changes"
        self.report.commit_subject = commit_subject(REQUIREMENTS in changed, self.date)
        if self.args.dry_run:
            return
        for path in changed:
            write_tracked(repo, path, new_files[path])
        self.ship(changed)

    def ship(self, changed: list[Path]) -> None:
        """Branch, commit, push, open the PR, and (``--land``) enqueue it."""
        repo = self.repo
        branch = f"deps/weekly-{self.date}"
        subject = self.report.commit_subject
        assert subject is not None
        body = render_markdown(self.report)
        _capture(["git", "checkout", "-b", branch], repo)
        _capture(["git", "add", *[str(p) for p in changed]], repo)
        message = self.work / "commit-message.txt"
        message.write_text(f"{subject}\n\n{body}\n{CO_AUTHOR}\n")
        log = self.work / "logs" / "git.log"
        run_logged(["git", "commit", "-F", str(message)], log, self.base_env(), repo)
        run_logged(["git", "push", "-u", "origin", branch], log, self.base_env(), repo)
        self.report.branch = branch
        pr_body = self.work / "pr-body.md"
        pr_body.write_text(f"{body}\n{PR_FOOTER}\n")
        out = _capture(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                REPO_SLUG,
                "--base",
                "main",
                "--head",
                branch,
                "--title",
                subject,
                "--body-file",
                str(pr_body),
            ],
            repo,
        )
        self.report.pr_url = out.strip().splitlines()[-1]
        if self.args.land:
            self.report.land = self.land(pr_number_from_url(self.report.pr_url), branch)

    def land(self, pr_number: int, branch: str) -> LandOutcome:
        """Wait for the PR's checks; enqueue it when they pass and the loop is live."""
        checks, timed_out = wait_for_checks(
            self.repo, pr_number, self.args.land_timeout
        )
        primary = primary_checkout(self.repo)
        mq = [sys.executable, str(primary / "scripts" / "merge_queue.py")]
        loop_live = subprocess.run([*mq, "loop-status"], cwd=primary).returncode == 0
        decision = decide_land(checks, timed_out, loop_live)
        if decision.enqueue:
            _capture(
                [
                    *mq,
                    "add",
                    branch,
                    "--worktree",
                    str(self.repo),
                    "--by",
                    "deps-weekly",
                    "--note",
                    f"PR #{pr_number}",
                ],
                primary,
            )
        print(f"[deps-weekly] land: {decision.reason}", flush=True)
        return LandOutcome(
            pr_number=pr_number, checks=checks, loop_live=loop_live, decision=decision
        )

    def file_issue(self, capped: ArmResult) -> str:
        """File the suite-failure issue, or comment on the open one with that title."""
        report_md = render_markdown(self.report)
        body = (
            f"{report_md}\n### Last 60 lines of the pytest log\n\n"
            f"`{capped.log}`\n\n```\n{_tail(Path(capped.log), 60)}\n```\n\n{PR_FOOTER}\n"
        )
        body_file = self.work / "issue-body.md"
        body_file.write_text(body)
        listed = _capture(
            [
                "gh",
                "issue",
                "list",
                "--repo",
                REPO_SLUG,
                "--state",
                "open",
                "--search",
                f'"{FAILURE_ISSUE_TITLE}" in:title',
                "--json",
                "number,title,url",
            ],
            self.repo,
        )
        same = [i for i in json.loads(listed) if i["title"] == FAILURE_ISSUE_TITLE]
        if same:
            _capture(
                [
                    "gh",
                    "issue",
                    "comment",
                    str(same[0]["number"]),
                    "--repo",
                    REPO_SLUG,
                    "--body-file",
                    str(body_file),
                ],
                self.repo,
            )
            return str(same[0]["url"])
        out = _capture(
            [
                "gh",
                "issue",
                "create",
                "--repo",
                REPO_SLUG,
                "--title",
                FAILURE_ISSUE_TITLE,
                "--label",
                ISSUE_LABEL,
                "--body-file",
                str(body_file),
            ],
            self.repo,
        )
        return out.strip().splitlines()[-1]

    def cleanup(self) -> None:
        """Move the venvs, caches and temp dirs to the graveyard (a rename on /scratch)."""
        if not self.work_created or self.args.keep_work:
            return
        env = {**os.environ, "DEPRECATED_DIR": DEPRECATED_DIR}
        for sub in sorted(self.work.iterdir()):
            if sub.is_dir() and sub.name != "logs":
                subprocess.run(
                    [
                        "bash",
                        str(self.repo / "scripts" / "deprecate.sh"),
                        str(sub),
                        f"deps-weekly {self.date} scratch (venv, cache or temp)",
                    ],
                    env=env,
                    check=True,
                )


def write_tracked(repo: Path, path: Path, text: str) -> None:
    """Write one tracked file of the job (creating its directory for the lock)."""
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def pr_number_from_url(url: str) -> int:
    """``https://github.com/o/r/pull/123`` -> 123."""
    match = re.search(r"/pull/(\d+)$", url)
    if match is None:
        raise CommandFailed(f"gh pr create did not print a PR URL: {url!r}")
    return int(match.group(1))


def primary_checkout(repo: Path) -> Path:
    """The primary checkout that owns this worktree (where the merge queue lives)."""
    common = _capture(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], repo
    ).strip()
    return Path(common).parent


def read_checks(repo: Path, pr_number: int) -> list[CheckState]:
    """The PR's checks now; empty when none have registered yet."""
    result = subprocess.run(
        [
            "gh",
            "pr",
            "checks",
            str(pr_number),
            "--repo",
            REPO_SLUG,
            "--json",
            "name,bucket",
        ],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    # 0 all passed, 8 some pending, 1 failed checks or none reported yet.
    if result.stdout.strip().startswith("["):
        return [CheckState.model_validate(c) for c in json.loads(result.stdout)]
    if "no checks reported" in result.stderr:
        return []
    raise CommandFailed(
        f"gh pr checks {pr_number}: exit {result.returncode}\n{result.stderr}"
    )


def wait_for_checks(
    repo: Path, pr_number: int, timeout: float
) -> tuple[list[CheckState], bool]:
    """Wait (bounded) for the checks to register, then ``gh pr checks --watch``."""
    deadline = time.monotonic() + timeout
    while not read_checks(repo, pr_number):
        if time.monotonic() + CHECKS_REGISTER_POLL_S > deadline:
            return [], True
        time.sleep(CHECKS_REGISTER_POLL_S)
    watch = [
        "gh",
        "pr",
        "checks",
        str(pr_number),
        "--repo",
        REPO_SLUG,
        "--watch",
        "--fail-fast",
        "--interval",
        "60",
    ]
    try:
        subprocess.run(
            watch,
            cwd=repo,
            stdout=subprocess.DEVNULL,
            timeout=max(1.0, deadline - time.monotonic()),
        )
    except subprocess.TimeoutExpired:  # the bound the caller asked for, not an error
        return read_checks(repo, pr_number), True
    return read_checks(repo, pr_number), False


# ── CLI ──────────────────────────────────────────────────────────────────────


def write_report(report: DepsReport, work: Path | None) -> Path:
    """Write the report to the state dir (and a copy into the work dir)."""
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.date}.json"
    text = report.model_dump_json(indent=2) + "\n"
    path.write_text(text)
    if work is not None:
        (work / "report.json").write_text(text)
    return path


def cmd_run(args: argparse.Namespace) -> int:
    """The weekly job."""
    job = Job(args, Path.cwd(), datetime.now().strftime("%Y.%m.%d"))
    try:
        job.execute()
    except (Refusal, CommandFailed, ValueError) as err:  # job failed: exit 2, report
        job.report.status = "job-failed"
        job.report.error = str(err)
        print(f"[deps-weekly] {type(err).__name__}: {err}", file=sys.stderr, flush=True)
    job.report.finished_at = datetime.now().isoformat(timespec="seconds")
    path = write_report(job.report, job.work if job.work_created else None)
    job.cleanup()
    print(render_markdown(job.report))
    print(f"report: {path}")
    print(status_line(job.report), flush=True)
    return job.report.exit_code()


def cmd_report(args: argparse.Namespace) -> int:
    """Print the newest report (or ``--path``) as a table."""
    path: Path = args.path or newest_report(state_dir())
    report = DepsReport.model_validate_json(path.read_text())
    print(render_markdown(report))
    print(f"report: {path}")
    print(status_line(report))
    return 0


def cmd_write_lock(args: argparse.Namespace) -> int:
    """Write the lock from a report's passing capped arm (the step-h writer).

    The capped arm is the resolution of the requirements as committed. A dry run
    leaves the files unchanged, so a cap it would have lifted is still in them, and
    the lifted arm's resolution would not describe them.
    """
    path: Path = args.report or newest_report(state_dir())
    report = DepsReport.model_validate_json(path.read_text())
    if report.capped is None or not report.capped.passed:
        raise ValueError(f"{path}: the capped arm did not pass; no lock to write")
    write_tracked(Path.cwd(), LOCK, render_lock(report.capped.freeze))
    print(f"wrote {LOCK} from {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The CLI."""
    parser = argparse.ArgumentParser(
        prog="deps_weekly", description="Weekly dependency update job (torchcell)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Install the newest dependencies, test, update.")
    run.add_argument(
        "--work",
        type=Path,
        default=None,
        help=f"work dir (default {DEFAULT_WORK_ROOT}/<date>)",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="change no tracked file, open no PR or issue",
    )
    run.add_argument(
        "--keep-work", action="store_true", help="keep the venvs and caches"
    )
    run.add_argument(
        "--land", action="store_true", help="wait for the PR's checks, then enqueue it"
    )
    run.add_argument(
        "--land-timeout",
        type=float,
        default=LAND_TIMEOUT_S,
        help="seconds to wait for checks",
    )
    run.set_defaults(func=cmd_run)
    rep = sub.add_parser("report", help="Print the newest report as a table.")
    rep.add_argument("--path", type=Path, default=None, help="a specific report JSON")
    rep.set_defaults(func=cmd_report)
    lock = sub.add_parser(
        "write-lock", help=f"Write {LOCK} from a report's passing capped arm."
    )
    lock.add_argument(
        "--report", type=Path, default=None, help="report JSON (default newest)"
    )
    lock.set_defaults(func=cmd_write_lock)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    # An uncaught exception exits 1, which this job reserves for "the suite failed on the
    # newest dependencies"; a crash of the job itself must read as exit 2.
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
