# scripts/test_quality_check.py
# [[scripts.test_quality_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/test_quality_check.py
"""Anti-padding lint for ``tests/``: catch tests that raise coverage without pinning behavior.

Three rules, all decided from the ``ast`` of each ``test_*.py`` file (Decision 19 of
[[plan.test-suite-buildout.2026.09.25]]):

``no-assert``
    A ``test_*`` function with no ``assert`` statement, no ``pytest.raises`` /
    ``pytest.warns`` / ``pytest.fail``, no ``assert*`` call (``torch.testing
    .assert_close``, ``self.assertEqual``, ``np.testing.assert_allclose``) and no call
    to a same-file helper that asserts, proves nothing; it only executes lines.

``truthiness-only``
    A ``@pytest.mark.parametrize``d test whose every assertion is weak: a bare name
    (``assert x``), ``bool(x)``, ``x is not None``, ``isinstance(...)``, or ``len(x) > 0``
    and its variants. A sweep over many inputs that never states an exact expectation
    is the cheapest way to inflate a coverage number.

``unasserted-call``
    A test that calls a name imported from ``torchcell`` and never asserts on anything
    that call touched. "Touched" is tracked through simple name binding (``out = f(...)``,
    then any later ``out = g(out)`` or ``d[k] = out``), ``for`` targets over a derived
    iterable, and the arguments handed to the call (a tensor whose ``.grad`` the call
    fills). ``pytest.raises`` around the call counts as the assertion. Conservative on
    purpose: calls made through fixtures or module-level helpers are not tracked.

A test that legitimately trips a rule (a function whose only contract is "does not
raise", say) carries ``# test-quality: allow <reason>`` on its ``def`` line or a
decorator line, which the lint reads from the source text. Fixtures are never linted.

The lint cannot judge whether an exact value is the right one; that is the reviewer's
quality audit (Decision 18). It is wired to ``make test-quality``, a pre-commit hook on
``^tests/``, and a CI step.

Usage::

    python scripts/test_quality_check.py                 # all of tests/
    python scripts/test_quality_check.py tests/torchcell/losses/test_supcr.py

Exit 1 with one ``path:line: rule: name: message`` per finding. Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOW_MARKER = "test-quality: allow"
PYTEST_ASSERTING = {"raises", "warns", "fail", "deprecated_call"}
WEAK_CALLS = {"isinstance", "bool", "hasattr", "callable"}
LEN_ZERO_COMPARISONS = {
    (ast.Gt, 0),
    (ast.GtE, 1),
    (ast.NotEq, 0),
    (ast.Lt, 1),
    (ast.LtE, 0),
}
Func = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class Finding:
    """One lint hit."""

    path: Path
    line: int
    rule: str
    test: str
    message: str

    def render(self, repo: Path) -> str:
        """``path:line: rule: test: message`` with the path relative to the repo."""
        rel = (
            self.path.relative_to(repo) if self.path.is_relative_to(repo) else self.path
        )
        return f"{rel}:{self.line}: {self.rule}: {self.test}: {self.message}"


def _dotted(node: ast.AST) -> str:
    """``a.b.c`` for an Attribute/Name chain, else ``""``."""
    parts: list[str] = []
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Attribute):
            parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _root(node: ast.AST) -> str:
    return _dotted(node).split(".")[0]


def _torchcell_names(tree: ast.Module) -> set[str]:
    """Local names bound by ``import torchcell...`` / ``from torchcell... import``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torchcell":
                    names.add(alias.asname or alias.name.split(".")[0])
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module.split(".")[0] == "torchcell")
        ):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _nodes(func: Func) -> list[ast.AST]:
    """Every node in the function body, nested defs included, decorators excluded."""
    return [n for stmt in func.body for n in ast.walk(stmt)]


def _is_asserting_call(node: ast.AST, helpers: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    dotted = _dotted(node.func)
    last = dotted.split(".")[-1] if dotted else ""
    if dotted.startswith("pytest.") and last in PYTEST_ASSERTING:
        return True
    return last.startswith("assert") or dotted in helpers


def _raises_assertion_error(node: ast.AST) -> bool:
    if not isinstance(node, ast.Raise) or node.exc is None:
        return False
    exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
    return _dotted(exc) == "AssertionError"


def _is_assertion(node: ast.AST, helpers: set[str]) -> bool:
    return (
        isinstance(node, ast.Assert)
        or _is_asserting_call(node, helpers)
        or _raises_assertion_error(node)
    )


def _weak_assert(test: ast.expr) -> bool:
    """True for an assertion that cannot state an exact expectation."""
    if isinstance(test, ast.Name):
        return True
    if isinstance(test, ast.Call):
        return _dotted(test.func) in WEAK_CALLS
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op, right = test.ops[0], test.comparators[0]
        if isinstance(op, (ast.IsNot, ast.Is)) and (
            isinstance(right, ast.Constant) and right.value is None
        ):
            return True
        left = test.left
        if (
            isinstance(left, ast.Call)
            and _dotted(left.func) == "len"
            and isinstance(right, ast.Constant)
            and (type(op), right.value) in LEN_ZERO_COMPARISONS
        ):
            return True
    return False


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _targets(target: ast.AST) -> set[str]:
    """Names a store target binds or mutates (``d[k] = v`` mutates ``d``)."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_targets(t) for t in target.elts))
    if isinstance(target, ast.Starred):
        return _targets(target.value)
    if isinstance(target, (ast.Subscript, ast.Attribute)):
        return {_root(target)} if _root(target) else set()
    return set()


def _is_parametrized(func: Func) -> bool:
    return any(
        _dotted(d.func if isinstance(d, ast.Call) else d).endswith("parametrize")
        for d in func.decorator_list
    )


def _is_fixture(func: Func) -> bool:
    return any(
        "fixture" in _dotted(d.func if isinstance(d, ast.Call) else d)
        for d in func.decorator_list
    )


def _allowed(source_lines: list[str], func: Func) -> bool:
    """The marker sits anywhere from the first decorator to the end of the signature."""
    first = min([func.lineno, *(d.lineno for d in func.decorator_list)])
    last = func.body[0].lineno if func.body else func.lineno
    return any(ALLOW_MARKER in line for line in source_lines[first - 1 : last])


class _Taint:
    """Names a test derives from its torchcell calls, to a fixpoint."""

    def __init__(self, nodes: list[ast.AST], tc_names: set[str]) -> None:
        self.tc_names = tc_names
        self.derived: set[str] = set()
        binders = [
            n
            for n in nodes
            if isinstance(
                n,
                (
                    ast.Assign,
                    ast.AnnAssign,
                    ast.AugAssign,
                    ast.For,
                    ast.AsyncFor,
                    ast.With,
                    ast.AsyncWith,
                    ast.NamedExpr,
                ),
            )
        ]
        changed = True
        while changed:
            changed = False
            for node in binders:
                for value, target in self._bindings(node):
                    if self.touches(value):
                        names = _targets(target)
                        if not names <= self.derived:
                            self.derived |= names
                            changed = True
            # Arguments handed to a torchcell (or derived) call are touched by it.
            for call in nodes:
                if isinstance(call, ast.Call) and (
                    _root(call.func) in self.tc_names or self.touches(call.func)
                ):
                    names = set().union(
                        *(_names_in(a) for a in call.args),
                        *(_names_in(k.value) for k in call.keywords),
                    )
                    if not names <= self.derived:
                        self.derived |= names
                        changed = True

    @staticmethod
    def _bindings(node: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
        if isinstance(node, ast.Assign):
            return [(node.value, t) for t in node.targets]
        if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            return [(node.value, node.target)] if node.value is not None else []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            return [(node.iter, node.target)]
        if isinstance(node, ast.NamedExpr):
            return [(node.value, node.target)]
        if isinstance(node, (ast.With, ast.AsyncWith)):
            return [
                (i.context_expr, i.optional_vars)
                for i in node.items
                if i.optional_vars is not None
            ]
        return []

    def calls_torchcell(self, node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Call) and _root(n.func) in self.tc_names
            for n in ast.walk(node)
        )

    def touches(self, node: ast.AST) -> bool:
        """The expression calls a torchcell name or mentions a derived name."""
        return self.calls_torchcell(node) or bool(_names_in(node) & self.derived)


def _asserting_helpers(tree: ast.Module) -> set[str]:
    """Module-level functions whose body asserts (transitively through other helpers)."""
    funcs = {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    helpers: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, func in funcs.items():
            if name in helpers or name.startswith("test"):
                continue
            nodes = _nodes(func)
            if any(_is_assertion(n, helpers) for n in nodes):
                helpers.add(name)
                changed = True
    return helpers


def check_function(
    func: Func, tc_names: set[str], helpers: set[str], path: Path, lines: list[str]
) -> list[Finding]:
    """Apply the three rules to one ``test_*`` function."""
    if _is_fixture(func) or _allowed(lines, func):
        return []
    nodes = _nodes(func)
    asserts = [n for n in nodes if isinstance(n, ast.Assert)]
    asserting_calls = [n for n in nodes if _is_asserting_call(n, helpers)]
    raises_assertion = [n for n in nodes if _raises_assertion_error(n)]
    findings: list[Finding] = []

    if not asserts and not asserting_calls and not raises_assertion:
        return [
            Finding(
                path,
                func.lineno,
                "no-assert",
                func.name,
                "no assert, pytest.raises/warns/fail, assert* call, "
                "raise AssertionError, or asserting helper",
            )
        ]

    if (
        _is_parametrized(func)
        and asserts
        and not asserting_calls
        and all(_weak_assert(a.test) for a in asserts)
    ):
        findings.append(
            Finding(
                path,
                func.lineno,
                "truthiness-only",
                func.name,
                "parametrized sweep whose assertions are all bare truthiness, "
                "`is not None`, isinstance, or `len(x) > 0`",
            )
        )

    # A nested helper that calls torchcell stands in for the call (``one = lambda``
    # factories, ``def build(): return Model(...)``), so its name is tracked too.
    local_tc = set(tc_names)
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(n, ast.Call) and _root(n.func) in local_tc for n in _nodes(node)
        ):
            local_tc.add(node.name)
    taint = _Taint(nodes, local_tc)
    if tc_names and any(taint.calls_torchcell(stmt) for stmt in func.body):
        asserted = (
            any(taint.touches(a.test) for a in asserts)
            or any(taint.touches(c) for c in asserting_calls)
            or any(
                _is_asserting_call(item.context_expr, helpers)
                and any(taint.touches(stmt) for stmt in node.body)
                for node in nodes
                if isinstance(node, (ast.With, ast.AsyncWith))
                for item in node.items
            )
        )
        if not asserted:
            findings.append(
                Finding(
                    path,
                    func.lineno,
                    "unasserted-call",
                    func.name,
                    "calls a torchcell name but asserts nothing that call touched",
                )
            )
    return findings


def check_file(path: Path) -> list[Finding]:
    """Lint one test file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    tc_names = _torchcell_names(tree)
    helpers = _asserting_helpers(tree)
    lines = source.splitlines()
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith("test"):
            findings.extend(check_function(node, tc_names, helpers, path, lines))
    return sorted(findings, key=lambda f: (str(f.path), f.line))


def collect(paths: list[Path]) -> list[Path]:
    """``test_*.py`` files under the given paths (directories are walked)."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("test_*.py")))
        elif path.name.startswith("test_") and path.suffix == ".py":
            files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[REPO / "tests"],
        help="test files or directories (default: tests/)",
    )
    args = parser.parse_args(argv)
    files = collect([p if p.is_absolute() else Path.cwd() / p for p in args.paths])
    findings = [f for path in files for f in check_file(path)]
    for finding in findings:
        print(finding.render(REPO))
    if findings:
        print(f"test-quality: {len(findings)} finding(s) in {len(files)} file(s)")
        return 1
    print(f"test-quality: {len(files)} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
