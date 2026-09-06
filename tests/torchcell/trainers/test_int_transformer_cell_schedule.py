"""Tests for RegressionTask's diagnostic-epoch scheduling.

`plot_edge_recovery_every_n_epochs: 0` is the natural way to switch a diagnostic off, and
one of the seven call sites already read it that way. The other six went straight into
`(epoch + 1) % freq`, so a 0 raised ZeroDivisionError from `_shared_step` on the first
VALIDATION batch, three minutes into a 12 h four-GPU run whose model, dataset and sanity
check had all succeeded (job 1599). Nothing about the config was invalid; the crash was
five call sites disagreeing with the sixth about what 0 means.
"""

from typing import Any

import pytest

from torchcell.trainers.int_transformer_cell import RegressionTask


class _Task:
    """Just enough of a task to exercise `_is_scheduled` as an unbound method."""

    def __init__(self, epoch: int) -> None:
        self.current_epoch = epoch

    _is_scheduled: Any = RegressionTask._is_scheduled


@pytest.mark.parametrize("freq", [0, -1, None])
def test_non_positive_frequency_never_fires(freq: Any) -> None:
    """0 means never, everywhere. Previously it divided by zero."""
    for epoch in range(0, 40):
        assert _Task(epoch)._is_scheduled(freq) is False


def test_every_n_epochs_fires_on_the_right_epochs() -> None:
    """Frequency N fires on epochs N-1, 2N-1, ... , counting from 1 as before."""
    fired = [e for e in range(20) if _Task(e)._is_scheduled(5)]
    assert fired == [4, 9, 14, 19]


def test_frequency_one_fires_every_epoch() -> None:
    assert all(_Task(e)._is_scheduled(1) for e in range(10))


def test_large_frequency_is_the_other_way_to_disable() -> None:
    """A big number never fires inside a realistic run, and must not error."""
    assert not any(_Task(e)._is_scheduled(100000) for e in range(500))


def test_no_call_site_still_divides_directly() -> None:
    """Every scheduling decision must route through the helper.

    A new `(self.current_epoch + 1) % <freq>` added later would reintroduce exactly the
    crash this fixes, and it would again survive a smoke test whose CLI overrides set the
    frequency to something positive.
    """
    import inspect

    src = inspect.getsource(RegressionTask)
    occurrences = src.count("(self.current_epoch + 1) %")
    assert occurrences == 1, (
        f"{occurrences} direct modulo sites; only `_is_scheduled` may do the arithmetic"
    )
