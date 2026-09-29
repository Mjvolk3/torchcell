# tests/torchcell/scheduler/test_cosine_annealing_warmup.py
# [[tests.torchcell.scheduler.test_cosine_annealing_warmup]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scheduler/test_cosine_annealing_warmup.py
"""``CosineAnnealingWarmupRestarts`` against its closed-form schedule.

Fixture: ``first_cycle_steps=5, warmup_steps=2, max_lr=0.1, min_lr=0.01,
cycle_mult=2.0, gamma=0.5``. ``base_lr`` is ``min_lr`` for every param group (``init_lr``
overwrites the optimizer's own lr). Constructing the scheduler runs one ``step()``
(``_LRScheduler.__init__`` calls it), leaving ``step_in_cycle=0`` and lr ``min_lr``.

Cycle 0 (length 5, ``max_lr`` 0.1, ``span = max_lr - min_lr = 0.09``):

* step 1, warmup: ``0.09 * 1/2 + 0.01 = 0.055``
* step 2, cosine at 0: ``0.01 + 0.09 * (1 + cos 0)/2 = 0.1``
* step 3: ``0.01 + 0.09 * (1 + cos(pi/3))/2 = 0.01 + 0.09 * 0.75 = 0.0775``
* step 4: ``0.01 + 0.09 * (1 + cos(2pi/3))/2 = 0.01 + 0.09 * 0.25 = 0.0325``

Cycle 1 begins at step 5: ``cur_cycle_steps = int((5 - 2) * 2) + 2 = 8`` and
``max_lr = 0.1 * 0.5 = 0.05`` (span 0.04):

* step 5, warmup position 0: ``0.01``
* step 6: ``0.04 * 1/2 + 0.01 = 0.03``
* step 7: ``0.05``
* step 8: ``0.01 + 0.02 * (1 + cos(pi/6)) = 0.04732050807568878``
* step 9: ``cos(pi/3) = 0.5`` gives ``0.04``; step 10: ``cos(pi/2) = 0`` gives ``0.03``
* step 11: ``cos(2pi/3) = -0.5`` gives ``0.02``
* step 12: ``0.01 + 0.02 * (1 + cos(5pi/6)) = 0.012679491924311226``

Cycle 2 begins at step 13: ``cur_cycle_steps = int((8 - 2) * 2) + 2 = 14``,
``max_lr = 0.025``; step 14 is warmup ``0.015 * 1/2 + 0.01 = 0.0175``.
"""

import math

import pytest
import torch

from torchcell.scheduler.cosine_annealing_warmup import CosineAnnealingWarmupRestarts

EXPECTED_LRS = [
    0.055,
    0.1,
    0.0775,
    0.0325,
    0.01,
    0.03,
    0.05,
    0.01 + 0.02 * (1 + math.cos(math.pi / 6)),
    0.04,
    0.03,
    0.02,
    0.01 + 0.02 * (1 + math.cos(5 * math.pi / 6)),
    0.01,
    0.0175,
]


def _optimizer(n_groups: int = 1) -> torch.optim.Optimizer:
    groups = [
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 0.5}
        for _ in range(n_groups)
    ]
    return torch.optim.SGD(groups)


def _scheduler(
    optimizer: torch.optim.Optimizer, cycle_mult: float = 2.0
) -> CosineAnnealingWarmupRestarts:
    return CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=5,
        warmup_steps=2,
        max_lr=0.1,
        min_lr=0.01,
        cycle_mult=cycle_mult,
        gamma=0.5,
    )


def _lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def test_construction_sets_min_lr_and_starts_at_step_zero() -> None:
    """The optimizer's own lr (0.5) is replaced by min_lr; base_lrs is [min_lr]."""
    optimizer = _optimizer()
    scheduler = _scheduler(optimizer)
    assert _lrs(optimizer) == [0.01]
    assert scheduler.base_lrs == [0.01]
    assert (scheduler.last_epoch, scheduler.step_in_cycle, scheduler.cycle) == (0, 0, 0)
    assert scheduler.cur_cycle_steps == 5
    assert scheduler.max_lr == 0.1


def test_implicit_steps_follow_the_closed_form_schedule() -> None:
    """Fourteen steps reproduce the values worked in the module docstring."""
    optimizer = _optimizer()
    scheduler = _scheduler(optimizer)
    observed: list[float] = []
    for _ in EXPECTED_LRS:
        optimizer.step()
        scheduler.step()
        observed.append(_lrs(optimizer)[0])
    assert observed == pytest.approx(EXPECTED_LRS, abs=1e-12)


def test_restart_bookkeeping_at_cycle_boundaries() -> None:
    """After step 5: cycle 1, length 8, max_lr 0.05; after step 13: cycle 2, length 14."""
    optimizer = _optimizer()
    scheduler = _scheduler(optimizer)
    for _ in range(5):
        scheduler.step()
    assert (scheduler.cycle, scheduler.step_in_cycle, scheduler.cur_cycle_steps) == (
        1,
        0,
        8,
    )
    assert scheduler.max_lr == 0.05
    assert scheduler.last_epoch == 5
    for _ in range(8):
        scheduler.step()
    assert (scheduler.cycle, scheduler.step_in_cycle, scheduler.cur_cycle_steps) == (
        2,
        0,
        14,
    )
    assert scheduler.max_lr == 0.025
    assert scheduler.last_epoch == 13


def test_every_param_group_gets_the_same_schedule() -> None:
    """Two param groups with different initial lrs both follow min_lr -> 0.055 -> 0.1."""
    optimizer = _optimizer(n_groups=2)
    optimizer.param_groups[1]["lr"] = 0.7
    scheduler = _scheduler(optimizer)
    assert _lrs(optimizer) == [0.01, 0.01]
    scheduler.step()
    assert _lrs(optimizer) == pytest.approx([0.055, 0.055])
    scheduler.step()
    assert _lrs(optimizer) == pytest.approx([0.1, 0.1])


def test_explicit_epoch_inside_first_cycle() -> None:
    """step(epoch=3) lands on cycle 0 position 3: lr 0.0775, last_epoch 3."""
    optimizer = _optimizer()
    scheduler = _scheduler(optimizer)
    scheduler.step(epoch=3)
    assert _lrs(optimizer) == pytest.approx([0.0775])
    assert (scheduler.cycle, scheduler.step_in_cycle, scheduler.cur_cycle_steps) == (
        0,
        3,
        5,
    )
    assert scheduler.last_epoch == 3


def test_explicit_epoch_with_cycle_mult_one_uses_modular_arithmetic() -> None:
    """cycle_mult=1: epoch 12 is cycle 12 // 5 = 2, position 2, max_lr 0.1 * 0.5^2 = 0.025.

    Position 2 is the cosine start (cos 0), so lr == max_lr == 0.025.
    """
    optimizer = _optimizer()
    scheduler = _scheduler(optimizer, cycle_mult=1.0)
    scheduler.step(epoch=12)
    assert (scheduler.cycle, scheduler.step_in_cycle, scheduler.cur_cycle_steps) == (
        2,
        2,
        5,
    )
    assert scheduler.max_lr == 0.025
    assert _lrs(optimizer) == pytest.approx([0.025])


def test_explicit_epoch_ignores_warmup_in_the_cycle_length() -> None:
    """Finding: the explicit-epoch path and the implicit path disagree on cycle length.

    Implicit stepping grows a cycle as ``int((cur - warmup) * mult) + warmup``, so with
    ``first=5, warmup=2, mult=2`` the second cycle is 8 steps long and step 8 gives
    lr 0.0473205. ``step(epoch=8)`` instead sets ``cur_cycle_steps = first * mult**n = 10``
    (``n = int(log2(8/5 + 1)) = 1``, ``step_in_cycle = 8 - int(5 * (2 - 1)) = 3``), so the
    cosine is evaluated at ``(3 - 2)/(10 - 2) = 1/8`` of the way:
    ``0.01 + 0.02 * (1 + cos(pi/8)) = 0.048477590650225735``. The two ways of reaching
    epoch 8 yield different learning rates whenever ``warmup_steps > 0`` and
    ``cycle_mult != 1``.
    """
    optimizer = _optimizer()
    scheduler = _scheduler(optimizer)
    scheduler.step(epoch=8)
    assert (scheduler.cycle, scheduler.step_in_cycle, scheduler.cur_cycle_steps) == (
        1,
        3,
        10.0,
    )
    assert _lrs(optimizer) == pytest.approx([0.01 + 0.02 * (1 + math.cos(math.pi / 8))])
    assert _lrs(optimizer)[0] != pytest.approx(EXPECTED_LRS[7])


def test_explicit_epoch_back_into_first_cycle_keeps_the_decayed_max_lr() -> None:
    """Finding: rewinding to an epoch inside cycle 0 does not reset ``cycle``.

    ``step(epoch=8)`` sets ``cycle = 1``. A following ``step(epoch=3)`` takes the
    ``epoch < first_cycle_steps`` branch, which resets ``cur_cycle_steps`` and
    ``step_in_cycle`` but leaves ``cycle`` at 1, so ``max_lr`` stays ``0.05`` and the lr is
    ``0.01 + 0.04 * 0.75 = 0.04`` rather than the ``0.0775`` a fresh scheduler gives at
    epoch 3.
    """
    optimizer = _optimizer()
    scheduler = _scheduler(optimizer)
    scheduler.step(epoch=8)
    scheduler.step(epoch=3)
    assert scheduler.cycle == 1
    assert scheduler.max_lr == 0.05
    assert _lrs(optimizer) == pytest.approx([0.04])


def test_explicit_negative_epoch_returns_base_lrs() -> None:
    """step(epoch=-1) sets step_in_cycle -1, the branch that hands back base_lrs (min_lr)."""
    optimizer = _optimizer()
    scheduler = _scheduler(optimizer)
    scheduler.step(epoch=2)
    assert _lrs(optimizer) == pytest.approx([0.1])
    scheduler.step(epoch=-1)
    assert _lrs(optimizer) == [0.01]
    assert scheduler.step_in_cycle == -1
    assert scheduler.last_epoch == -1


def test_get_last_lr_is_never_populated() -> None:
    """Finding: the overridden ``step`` never sets ``_last_lr``, so ``get_last_lr`` raises.

    ``torch.optim.lr_scheduler.LRScheduler.step`` records ``self._last_lr`` for
    ``get_last_lr()``; this subclass replaces ``step`` wholesale and skips that, so any
    caller relying on the base-class accessor gets ``AttributeError`` even after steps.
    """
    optimizer = _optimizer()
    scheduler = _scheduler(optimizer)
    scheduler.step()
    with pytest.raises(AttributeError, match="_last_lr"):
        scheduler.get_last_lr()


def test_warmup_must_be_shorter_than_the_first_cycle() -> None:
    """warmup_steps == first_cycle_steps trips the constructor assertion."""
    with pytest.raises(AssertionError):
        CosineAnnealingWarmupRestarts(_optimizer(), first_cycle_steps=2, warmup_steps=2)
