# tests/torchcell/scripts/test_test_quality_check.py
# [[tests.torchcell.scripts.test_test_quality_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_test_quality_check.py
"""``scripts/test_quality_check.py`` on hand-written test files under ``tmp_path``.

Each case is one small file whose expected findings (rule, test name, line) are written
next to it. The three rules: ``no-assert`` (nothing that can fail), ``truthiness-only``
(a parametrized sweep whose assertions are all bare truthiness, ``is not None``,
``isinstance`` or ``len(x) > 0``) and ``unasserted-call`` (a torchcell name is called but
nothing the call touched is asserted, with taint followed through assignments, loops,
``with`` targets, call arguments, subscripts, nested helpers and the capture fixtures).
Fixtures are skipped, the ``# test-quality: allow <reason>`` marker silences a function,
and module-level helpers that assert count as assertions. Every file starts with the same
four-line header, so a body's first line is line 5.
"""

import ast
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import test_quality_check as tqc  # type: ignore[import-not-found]  # noqa: E402

HEADER = "import pytest\nfrom torchcell.models.toy import Model\nimport torchcell.data as tcd\n\n"
FIRST_BODY_LINE = HEADER.count("\n") + 1


def _lint(tmp_path: Path, body: str) -> list[tuple[str, str, int]]:
    path = tmp_path / "test_case.py"
    path.write_text(HEADER + textwrap.dedent(body))
    return [(f.rule, f.test, f.line) for f in tqc.check_file(path)]


def test_no_assert_is_reported_once_at_the_def_line(tmp_path: Path) -> None:
    """A test that only prints has nothing that can fail."""
    body = """\
        def test_nothing():
            value = Model().run()
            print(value)
        """
    assert _lint(tmp_path, body) == [("no-assert", "test_nothing", FIRST_BODY_LINE)]


def test_parametrized_truthiness_only_is_reported_and_the_same_asserts_pass_unparametrized(
    tmp_path: Path,
) -> None:
    """The four weak forms trip the rule only under parametrize; the line is the def, not the decorator."""
    weak = """\
            out = Model(n).run()
            assert out is not None
            assert isinstance(out, int)
            assert len(out) > 0
            assert out
        """
    parametrized = '@pytest.mark.parametrize("n", [1, 2])\ndef test_sweep(n):\n' + weak
    assert _lint(tmp_path, parametrized) == [
        ("truthiness-only", "test_sweep", FIRST_BODY_LINE + 1)
    ]
    plain = "def test_sweep(n=1):\n" + weak
    assert _lint(tmp_path, plain) == []


def test_unasserted_call_when_no_assert_touches_the_torchcell_result(
    tmp_path: Path,
) -> None:
    """The model is built and run, but the assertion is about arithmetic."""
    body = """\
        def test_unasserted():
            model = Model()
            model.run()
            assert 1 + 1 == 2
        """
    assert _lint(tmp_path, body) == [
        ("unasserted-call", "test_unasserted", FIRST_BODY_LINE)
    ]


def test_taint_flows_through_every_binding_form(tmp_path: Path) -> None:
    """Assignment, for-target, with-target, call argument, subscript store and a nested
    factory all carry the torchcell taint to the assertion, so none of these is a finding.
    """
    body = """\
        def test_assignment():
            model = Model()
            result = model.run()
            assert result == 3

        def test_loop():
            for item in tcd.load():
                assert item.ok == 1

        def test_with():
            with Model() as handle:
                assert handle.open == 1

        def test_argument():
            expected = [1, 2]
            tcd.fill(expected)
            assert expected == [1, 2, 3]

        def test_subscript():
            table = {}
            table["k"] = Model().value
            assert table["k"] == 1

        def test_nested_helper():
            def build():
                return Model()
            built = build()
            assert built.value == 1
        """
    assert _lint(tmp_path, body) == []


def test_asserting_helpers_pytest_raises_and_assert_calls_count_as_assertions(
    tmp_path: Path,
) -> None:
    """A module helper that asserts (directly or through another helper), ``pytest.raises``
    around the call, an ``assert_*`` call on the result, and an ``if``-guarded
    ``raise AssertionError`` on the result all satisfy the rule.
    """
    body = """\
        def _check(model):
            assert model.value == 1

        def _via_helper(model):
            _check(model)

        def test_helper():
            _check(Model())

        def test_transitive_helper():
            _via_helper(Model())

        def test_raises():
            with pytest.raises(ValueError):
                Model(-1)

        def test_assert_call():
            import numpy.testing as npt
            npt.assert_equal(Model().value, 1)

        def test_raise_assertion_error():
            if Model().value != 1:
                raise AssertionError("value")
        """
    assert _lint(tmp_path, body) == []


def test_fixtures_are_skipped_and_the_allow_marker_silences_a_function(
    tmp_path: Path,
) -> None:
    """A fixture named test_* is not linted; the marker works on a decorator line or the def line."""
    body = """\
        @pytest.fixture
        def test_data():
            return Model()

        @pytest.mark.slow  # test-quality: allow benchmark, timing only
        def test_benchmark():
            Model().run()

        def test_marked():  # test-quality: allow smoke import only
            Model()

        def test_not_marked():
            Model()
        """
    assert _lint(tmp_path, body) == [
        ("no-assert", "test_not_marked", FIRST_BODY_LINE + 11)
    ]


def test_capture_fixture_output_is_derived_from_the_torchcell_call(
    tmp_path: Path,
) -> None:
    """Asserting on ``capsys.readouterr()`` after a torchcell call is an assertion about that call."""
    body = """\
        def test_prints(capsys):
            tcd.report()
            out = capsys.readouterr().out
            assert out == "ok\\n"

        def test_logs(caplog):
            tcd.report()
            assert caplog.messages == ["ok"]
        """
    assert _lint(tmp_path, body) == []


@pytest.mark.parametrize(
    ("expression", "weak"),
    [
        ("x", True),
        ("x is not None", True),
        ("x is None", True),
        ("isinstance(x, int)", True),
        ("bool(x)", True),
        ("hasattr(x, 'a')", True),
        ("len(x) > 0", True),
        ("len(x) >= 1", True),
        ("len(x) != 0", True),
        ("len(x) == 0", False),
        ("len(x) == 3", False),
        ("x == 1", False),
        ("x.ok", False),
        ("x > 0", False),
        ("x is not other", False),
    ],
)
def test_weak_assert_classification(expression: str, weak: bool) -> None:
    """Fifteen assertion shapes: which can state an expectation and which cannot."""
    assert tqc._weak_assert(ast.parse(expression, mode="eval").body) is weak


def test_torchcell_names_follow_aliases(tmp_path: Path) -> None:
    """``import torchcell.x as y`` binds ``y``; ``from torchcell.x import A as B`` binds ``B``;
    a non-torchcell import binds nothing.
    """
    tree = ast.parse(
        "import torchcell.data as tcd\nimport torchcell\nfrom torchcell.models import A as B, C\nimport numpy as np\n"
    )
    assert tqc._torchcell_names(tree) == {"tcd", "torchcell", "B", "C"}


def test_cli_reports_findings_with_absolute_paths_outside_the_repo_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One bad file and one clean file: the finding line, then the summary, exit 1."""
    bad = tmp_path / "test_bad.py"
    bad.write_text(HEADER + "def test_nothing():\n    Model()\n")
    (tmp_path / "test_good.py").write_text(
        HEADER + "def test_ok():\n    assert Model().v == 1\n"
    )
    (tmp_path / "helper.py").write_text("def test_not_collected():\n    pass\n")
    assert tqc.main([str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        f"{bad}:{FIRST_BODY_LINE}: no-assert: test_nothing: no assert, pytest.raises/warns/fail, "
        "assert* call, raise AssertionError, or asserting helper",
        "test-quality: 1 finding(s) in 2 file(s)",
    ]


def test_cli_is_clean_on_a_clean_directory_and_accepts_a_single_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 0 with the count of files; a single test file path is also accepted."""
    good = tmp_path / "test_good.py"
    good.write_text(HEADER + "def test_ok():\n    assert Model().v == 1\n")
    assert tqc.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "test-quality: 1 file(s) clean"
    assert tqc.main([str(good)]) == 0
    assert capsys.readouterr().out.strip() == "test-quality: 1 file(s) clean"
    assert tqc.collect([tmp_path / "helper.py", tmp_path / "missing"]) == []


def test_finding_render_is_relative_inside_the_repo() -> None:
    """A path under the repo renders repo-relative; the message order is path:line: rule: test."""
    finding = tqc.Finding(
        tqc.REPO / "tests" / "x" / "test_y.py", 7, "no-assert", "test_z", "m"
    )
    assert finding.render(tqc.REPO) == "tests/x/test_y.py:7: no-assert: test_z: m"
