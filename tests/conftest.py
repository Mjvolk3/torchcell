# tests/conftest.py
# [[tests.conftest]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/conftest.py
"""Root pytest configuration: the safe-by-default contract for plain ``pytest``.

Four things live here, in this order because the order is the point:

1. Environment defaults set BEFORE any ``torchcell`` import. ``DATA_ROOT`` gets a fixed
   sentinel path so the twelve ``os.getenv("DATA_ROOT") is None`` gates in the suite
   see a value that exists nowhere on disk; modules that call ``load_dotenv()``
   (``override=False``) cannot replace it, and a shell that exports the real root wins
   because ``setdefault`` never overrides. ``WANDB_MODE`` and ``MPLBACKEND`` are set the
   same way; the matplotlib backend must be chosen before the first ``pyplot`` import.
2. Opt-in flags for the expensive buckets (``--gpu --slow --data --neo4j --network
   --wandb``): a test marked with a bucket is skipped, with a visible reason, unless its
   flag is given. CI runs plain ``pytest tests/torchcell``.
3. Autouse boundary guards that fail loudly when an unmarked test reaches a cluster
   (``sbatch``), a hosting CLI (``gh``, ``ssh``), the network (``socket.getaddrinfo``,
   ``urllib.request.urlopen``, ``requests.Session.request``), a Neo4j driver, or
   ``wandb.init``. Each guard is disabled by its flag. ``socket.socket`` is never
   patched: torch.distributed and wandb offline mode need it.
4. A session-end check that the sentinel data root is still empty, so a test that wrote
   into it is caught instead of leaving a growing directory under ``/tmp``.

Design record: Decisions 6 and 15 of [[plan.test-suite-buildout.2026.09.25]]. The
sentinel is a fixed path, never a per-session ``basetemp``: ``scripts/deprecate.sh``
refuses a graveyard under ``$DATA_ROOT`` by prefix match, and ``scripts/merge_queue.py``
resolves its live queue database under ``$DATA_ROOT`` at import.
"""

import os

SENTINEL_DATA_ROOT = "/tmp/torchcell-test-data-root"
os.environ.setdefault("DATA_ROOT", SENTINEL_DATA_ROOT)
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MPLBACKEND", "Agg")

import socket  # noqa: E402
import subprocess  # noqa: E402
import urllib.request  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

# marker -> (flag, what the flag unlocks)
BUCKETS: dict[str, tuple[str, str]] = {
    "gpu": ("--gpu", "tests that need a CUDA device"),
    "slow": ("--slow", "full dataset builds from the $DATA_ROOT mirrors"),
    "data": ("--data", "tests that read the real $DATA_ROOT"),
    "neo4j": ("--neo4j", "tests that open a Neo4j driver"),
    "network": ("--network", "tests that reach the network, gh, or ssh"),
    "wandb": ("--wandb", "tests that call wandb.init"),
}

CLUSTER_BINARIES = {"sbatch"}
HOSTING_BINARIES = {"gh", "ssh"}


def pytest_addoption(parser: pytest.Parser) -> None:
    """One boolean flag per bucket; plain ``pytest`` runs none of them."""
    for flag, what in BUCKETS.values():
        parser.addoption(flag, action="store_true", default=False, help=f"run {what}")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip every bucket-marked item whose flag is absent, naming the flag."""
    for item in items:
        for marker, (flag, what) in BUCKETS.items():
            if item.get_closest_marker(marker) and not config.getoption(flag):
                item.add_marker(pytest.mark.skip(reason=f"needs {flag} ({what})"))


def _guard(what: str, flag: str | None) -> Callable[..., Any]:
    hint = f"mark it @pytest.mark.{flag[2:]} and pass {flag}" if flag else "patch it"
    marker_hint = f"; {hint}" if flag else ""

    def raise_boundary(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            f"An unmarked test tried to {what}. Tests are hermetic by default"
            f"{marker_hint}."
        )

    return raise_boundary


@pytest.fixture(autouse=True)
def _no_cluster_or_hosting_subprocess(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``subprocess.Popen`` refuses ``sbatch`` always and ``gh``/``ssh`` without --network.

    ``Popen`` is the chokepoint ``run``, ``check_output`` and ``call`` share; every other
    argv (``git``, ``bash``, ``python``) passes straight through.
    """
    network_ok = request.config.getoption("--network")

    class GuardedPopen(subprocess.Popen[Any]):
        def __init__(self, args: Any, *popen_args: Any, **kwargs: Any) -> None:
            argv = args if isinstance(args, (list, tuple)) else [args]
            name = Path(str(argv[0])).name if argv else ""
            if name in CLUSTER_BINARIES:
                raise AssertionError(
                    f"A test tried to run {name!r}: {list(argv)!r}. No test may submit a "
                    "cluster job; patch the submit path."
                )
            if name in HOSTING_BINARIES and not network_ok:
                _guard(f"run {name!r}", "--network")()
            super().__init__(args, *popen_args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", GuardedPopen)


@pytest.fixture(autouse=True)
def _no_external_services(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Network, Neo4j and wandb are unreachable unless the matching flag is given."""
    if not request.config.getoption("--network"):
        monkeypatch.setattr(
            socket, "getaddrinfo", _guard("resolve a hostname", "--network")
        )
        monkeypatch.setattr(
            urllib.request, "urlopen", _guard("open a URL", "--network")
        )
        monkeypatch.setattr(
            "requests.sessions.Session.request",
            _guard("send an HTTP request", "--network"),
        )
    if not request.config.getoption("--neo4j"):
        monkeypatch.setattr(
            "neo4j.GraphDatabase.driver", _guard("open a Neo4j driver", "--neo4j")
        )
    if not request.config.getoption("--wandb"):
        monkeypatch.setattr("wandb.init", _guard("call wandb.init", "--wandb"))


@pytest.fixture(autouse=True, scope="session")
def _sentinel_data_root_stays_empty() -> Iterator[None]:
    """After the session, the sentinel root must not exist or must be empty."""
    yield
    if os.environ.get("DATA_ROOT") != SENTINEL_DATA_ROOT:
        return
    root = Path(SENTINEL_DATA_ROOT)
    if root.exists():
        entries = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
        assert not entries, (
            f"tests wrote into the sentinel DATA_ROOT {SENTINEL_DATA_ROOT}: "
            f"{entries[:20]}. Point the writer at tmp_path or mark the test @pytest.mark.data."
        )
