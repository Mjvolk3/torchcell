# tests/torchcell/profiling/test_timing.py
# [[tests.torchcell.profiling.test_timing]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/profiling/test_timing.py
"""``torchcell.profiling.timing`` under a fake clock.

``time_method`` reads the module globals ``_PROFILE_ENABLED`` and ``time`` at call time,
so each test flips the flag with ``monkeypatch`` and replaces ``timing.time`` with a
``FakeClock`` whose ``perf_counter`` pops from a scripted list. Every elapsed value is
therefore exact (``stop - start``), and every table row below is checked as a literal
string against ``capsys``.

Summary fixture: ``Sub.process`` timed at 1 ms and 3 ms (mean 2.0000, population std
1.0000, total 4 ms), ``LazySub.process`` at 2 ms, ``free_fn`` at 10 ms. Sorting is by
total time descending: ``free_fn`` (10) before ``Sub.process`` (4) before
``LazySub.process`` (2). Comparison fixture: methods ``a..f`` with speedups 4.00x (star),
1.50x (check), 0.50x (warning), 1.00x (no indicator), baseline-only ``e`` and
optimized-only ``f``; totals 8.50 ms vs 6.00 ms, change -2.50 ms, speedup 8.5/6 = 1.42x.
"""

from collections.abc import Iterator

import pytest

import torchcell.profiling.timing as timing
from torchcell.profiling.timing import (
    get_timing_summary,
    get_timings,
    print_comparison_table,
    print_timing_summary,
    reset_timings,
    time_method,
)


class FakeClock:
    """``perf_counter`` returns the scripted readings in order."""

    def __init__(self, readings: list[float]) -> None:
        """Script the readings; each is consumed once."""
        self.readings = list(readings)

    def perf_counter(self) -> float:
        """Return the next scripted reading (IndexError when the script is spent)."""
        return self.readings.pop(0)


@pytest.fixture(autouse=True)
def _clean_timings() -> Iterator[None]:
    reset_timings()
    yield
    reset_timings()


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timing, "_PROFILE_ENABLED", True)


@pytest.fixture
def disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timing, "_PROFILE_ENABLED", False)


def _clock(monkeypatch: pytest.MonkeyPatch, readings: list[float]) -> FakeClock:
    clock = FakeClock(readings)
    monkeypatch.setattr(timing, "time", clock)
    return clock


def _seed(entries: dict[str, list[float]]) -> None:
    for name, values in entries.items():
        timing._TIMINGS[name].extend(values)


class Widget:
    """Host for a decorated method; ``__qualname__`` is ``Widget.scale``."""

    @time_method
    def scale(self, x: int, factor: int = 2) -> int:
        """Doubles by default."""
        return x * factor


@time_method
def add(a: int, b: int) -> int:
    """Adds."""
    return a + b


def test_disabled_wrapper_returns_the_result_and_records_nothing(
    disabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With profiling off the clock is never read and _TIMINGS stays empty."""
    clock = _clock(monkeypatch, [])  # any read would IndexError on the empty script
    assert add(2, 3) == 5
    assert Widget().scale(4, factor=3) == 12
    assert clock.readings == []
    assert get_timings() == {}
    assert add.__name__ == "add" and add.__doc__ == "Adds."
    assert Widget.scale.__qualname__ == "Widget.scale"


def test_enabled_wrapper_records_stop_minus_start_under_the_qualname(
    enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readings (10.0, 10.25), (20.0, 20.5), (30.0, 30.125) give 0.25, 0.5 and 0.125 s."""
    clock = _clock(monkeypatch, [10.0, 10.25, 20.0, 20.5, 30.0, 30.125])
    assert add(1, 1) == 2
    assert add(2, 2) == 4
    assert Widget().scale(5) == 10
    assert clock.readings == []
    assert get_timings() == {"add": [0.25, 0.5], "Widget.scale": [0.125]}


def test_get_timings_is_a_shallow_copy() -> None:
    """Adding a key to the returned dict leaves the global alone; the lists are shared."""
    _seed({"a.b": [0.001]})
    snapshot = get_timings()
    snapshot["new"] = [1.0]
    assert get_timings() == {"a.b": [0.001]}
    snapshot["a.b"].append(0.002)
    assert get_timings() == {"a.b": [0.001, 0.002]}


def test_reset_timings_clears_every_entry() -> None:
    """After reset, get_timings is empty and the same defaultdict keeps accepting writes."""
    _seed({"a.b": [0.001], "c.d": [0.002]})
    reset_timings()
    assert get_timings() == {}
    _seed({"e.f": [0.003]})
    assert get_timings() == {"e.f": [0.003]}


def test_get_timing_summary_statistics(enabled: None) -> None:
    """Total = sum * 1000, mean = total / count, min and max in ms, per qualname."""
    _seed({"Sub.process": [0.001, 0.003], "free_fn": [0.010]})
    assert get_timing_summary() == {
        "Sub.process": {
            "total_ms": pytest.approx(4.0),
            "mean_ms": pytest.approx(2.0),
            "count": 2,
            "min_ms": pytest.approx(1.0),
            "max_ms": pytest.approx(3.0),
        },
        "free_fn": {
            "total_ms": pytest.approx(10.0),
            "mean_ms": pytest.approx(10.0),
            "count": 1,
            "min_ms": pytest.approx(10.0),
            "max_ms": pytest.approx(10.0),
        },
    }


def test_get_timing_summary_is_empty_when_disabled_even_with_data(
    disabled: None,
) -> None:
    """Finding: the summary is gated on the flag, not on the data.

    Timings recorded while enabled (or seeded directly) are ignored by
    ``get_timing_summary`` when ``_PROFILE_ENABLED`` is False, while ``get_timings`` and
    ``print_timing_summary`` still report them. The docstring promises a mapping of every
    method's stats; it returns ``{}``.
    """
    _seed({"Sub.process": [0.001]})
    assert get_timing_summary() == {}
    assert get_timings() == {"Sub.process": [0.001]}


def test_print_timing_summary_full_table(
    enabled: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Header, one row per method sorted by total time descending, footer."""
    _seed(
        {"Sub.process": [0.001, 0.003], "LazySub.process": [0.002], "free_fn": [0.010]}
    )
    print_timing_summary()
    rule = "=" * 80
    assert capsys.readouterr().out.split("\n") == [
        "",
        rule,
        "Method Timing Profile",
        rule,
        "Method                                                Calls    Mean (ms)    Std (ms)",
        "-" * 80,
        "free_fn".ljust(50) + "        1      10.0000      0.0000",
        "Sub.process".ljust(50) + "        2       2.0000      1.0000",
        "LazySub.process".ljust(50) + "        1       2.0000      0.0000",
        rule,
        "",
        "",
    ]


def test_print_timing_summary_filter_matches_the_class_prefix_exactly(
    enabled: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """filter_class='Sub' keeps 'Sub.process' and drops 'LazySub.process'; custom title."""
    _seed({"Sub.process": [0.001, 0.003], "LazySub.process": [0.002]})
    print_timing_summary(title="Summary", filter_class="Sub")
    lines = capsys.readouterr().out.split("\n")
    assert lines[2] == "Summary"
    assert lines[6] == "Sub.process".ljust(50) + "        2       2.0000      1.0000"
    assert lines[7] == "=" * 80
    assert not any(line.startswith("LazySub") for line in lines)


def test_print_timing_summary_filter_without_match(
    enabled: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed({"Sub.process": [0.001]})
    print_timing_summary(filter_class="Other")
    assert capsys.readouterr().out == "\n[TIMING] No timing data for class 'Other'\n"


def test_print_timing_summary_with_no_data(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enabled and empty prints the notice; disabled and empty prints nothing."""
    monkeypatch.setattr(timing, "_PROFILE_ENABLED", True)
    print_timing_summary()
    assert (
        capsys.readouterr().out
        == "\n[TIMING] Profiling enabled but no timing data collected\n"
    )
    monkeypatch.setattr(timing, "_PROFILE_ENABLED", False)
    print_timing_summary()
    assert capsys.readouterr().out == ""


def test_print_comparison_table_rows_and_indicators(
    enabled: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every branch of the row formatter, then totals, then the legend.

    ``Other.z`` matches neither class and is ignored; ``g`` has a zero mean on both
    sides, so it is in the method set but gets no row.
    """
    _seed(
        {
            "Base.a": [0.004],
            "Opt.a": [0.001],
            "Base.b": [0.0015],
            "Opt.b": [0.001],
            "Base.c": [0.001],
            "Opt.c": [0.002],
            "Base.d": [0.001],
            "Opt.d": [0.001],
            "Base.e": [0.001],
            "Opt.f": [0.001],
            "Base.g": [0.0],
            "Opt.g": [0.0],
            "Other.z": [0.5],
        }
    )
    print_comparison_table("Base", "Opt", title="T")
    rule = "=" * 100
    assert capsys.readouterr().out.split("\n") == [
        "",
        rule,
        "T",
        rule,
        "Method                                      Base          Opt       Change    Speedup",
        "-" * 100,
        "a".ljust(35) + "        4.00ms        1.00ms       -3.00ms      4.00x ⭐",
        "b".ljust(35) + "        1.50ms        1.00ms       -0.50ms      1.50x ✓",
        "c".ljust(35) + "        1.00ms        2.00ms        1.00ms      0.50x ⚠",
        "d".ljust(35) + "        1.00ms        1.00ms        0.00ms      1.00x ",
        "e".ljust(35) + "        1.00ms          N/A          N/A        N/A",
        "f".ljust(35) + "          N/A        1.00ms          N/A        N/A",
        "-" * 100,
        "TOTAL".ljust(35) + "        8.50ms        6.00ms       -2.50ms      1.42x",
        rule,
        "",
        "Legend: ⭐ = 2x+ speedup, ✓ = 1.2x+ speedup, ⚠ = slowdown",
        "",
    ]


def test_print_comparison_table_with_the_documented_class_names(
    enabled: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: the docstring's own example pair cannot be compared.

    Classes are matched with ``baseline_class in qualname`` and the baseline test runs
    first, so ``"SubgraphRepresentation" in "LazySubgraphRepresentation.process"`` is
    True and every Lazy timing is filed under the baseline, overwriting the real
    baseline entry for the same method name (dict insertion order: Subgraph first, Lazy
    second). The optimized column is always ``N/A`` and the row shows the LAZY time
    (1.00 ms) as the baseline, not the 4.00 ms that was recorded for it.
    """
    _seed(
        {
            "SubgraphRepresentation.process": [0.004],
            "LazySubgraphRepresentation.process": [0.001],
        }
    )
    print_comparison_table("SubgraphRepresentation", "LazySubgraphRepresentation")
    lines = capsys.readouterr().out.split("\n")
    assert lines[2] == "Graph Processor Comparison"
    assert (
        lines[4]
        == "Method                              SubgraphRepr LazySubgraph       Change    Speedup"
    )
    assert (
        lines[6]
        == "process".ljust(35) + "        1.00ms          N/A          N/A        N/A"
    )
    assert (
        lines[8]
        == "TOTAL".ljust(35) + "        1.00ms        0.00ms       -1.00ms      0.00x"
    )


def test_print_comparison_table_with_no_data(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(timing, "_PROFILE_ENABLED", True)
    print_comparison_table("Base", "Opt")
    assert (
        capsys.readouterr().out
        == "\n[TIMING] Profiling enabled but no timing data collected\n"
    )
    monkeypatch.setattr(timing, "_PROFILE_ENABLED", False)
    print_comparison_table("Base", "Opt")
    assert capsys.readouterr().out == ""
