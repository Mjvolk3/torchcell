# tests/torchcell/scripts/test_ops_sh.py
# [[tests.torchcell.scripts.test_ops_sh]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_ops_sh.py
"""``scripts/ops.sh``: the argument dispatch, which is the only network-free path.

``status``, ``releases`` and ``health`` probe the served Neo4j stores, tc-lit, slurm and
the disks, so they belong to a ``--network`` run; the usage error is what a hermetic test
can pin: an unknown action exits 2 and prints the three actions to stderr.
"""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ops.sh"


def test_unknown_action_exits_two_with_the_usage_line(tmp_path: Path) -> None:
    """The dispatch names the three valid actions and nothing runs."""
    result = subprocess.run(
        ["bash", str(SCRIPT), "bogus"],
        cwd=str(SCRIPT.parents[1]),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert result.returncode == 2
    assert result.stderr.strip() == f"usage: {SCRIPT} {{status|releases|health}}"
    assert result.stdout == ""
