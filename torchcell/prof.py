# torchcell/prof.py
# [[torchcell.prof]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/prof.py
# Test file: torchcell/test_prof.py
"""cProfile decorators that dump per-call timing stats to a ``profiles/`` directory."""

import os
import os.path as osp
import time
from collections.abc import Callable
from typing import Any


def prof_input(func: Callable[..., Any]) -> Callable[..., None]:
    """Decorate ``func`` to cProfile it, prompting for an experiment-name prefix."""
    import cProfile
    import datetime
    import pstats

    exp_name = input("Enter name of experiment: ")

    def inner(*args: Any, **kwargs: Any) -> None:
        dir = "profiles"
        if not osp.exists(dir):
            os.mkdir(dir)
        with cProfile.Profile() as pr:
            func(*args, **kwargs)
        stats = pstats.Stats(pr)
        now = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        file_name = f"{exp_name}-{func.__name__}-{now}.prof"
        stats.dump_stats(filename=osp.join(dir, file_name))

    return inner


def prof(func: Callable[..., Any]) -> Callable[..., None]:
    """Decorate ``func`` to cProfile it and dump stats under a fixed name prefix."""
    import cProfile
    import datetime
    import pstats

    def inner(*args: Any, **kwargs: Any) -> None:
        dir = "profiles"
        if not osp.exists(dir):
            os.mkdir(dir)
        with cProfile.Profile() as pr:
            func(*args, **kwargs)
        stats = pstats.Stats(pr)
        now = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        file_name = f"experiment_name-{func.__name__}-{now}.prof"
        stats.dump_stats(filename=osp.join(dir, file_name))

    return inner


def main() -> None:
    """Demonstrate both profiling decorators on dummy sleep functions."""

    @prof
    def test_func_dec() -> None:
        print("test func decorator")
        time.sleep(5)

    @prof_input
    def test_func_dec_in() -> None:
        print("test func decorator")
        time.sleep(5)

    test_func_dec()
    test_func_dec_in()

    # pip install snakeviz
    # In terminal run !snakeviz <path to .prof file>


if __name__ == "__main__":
    main()
