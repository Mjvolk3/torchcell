# tests/torchcell/scripts/test_setup_worktree.py
# [[tests.torchcell.scripts.test_setup_worktree]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_setup_worktree.py
"""``scripts/setup-worktree.sh`` run inside a worktree of a temporary repository.

``PATH`` is ``/usr/bin:/bin`` (no conda, no pre-commit), ``HOME`` and the global git config
are under ``tmp_path``. The script finds the main repository through
``git rev-parse --git-common-dir``, copies its ``.env`` with six paths rewritten to the
worktree, symlinks ``data/`` to the main repository (or creates a local ``data/torchcell``
under ``--data-local`` and points ``DATA_ROOT`` at the worktree), writes ``.env.vscode``,
and registers the weekly-note merge driver in the shared ``.git/config``. Every one of
those is asserted exactly.
"""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "setup-worktree.sh"
ENV_TEMPLATE = (
    'DATA_ROOT="/data/root"\n'
    'ASSET_IMAGES_DIR="/old/images"\n'
    'EXPERIMENT_ROOT="/old/experiments"\n'
    'WORKSPACE_DIR="/old"\n'
    'BIOCYPHER_CONFIG_PATH="/old/bc.yaml"\n'
    'SCHEMA_CONFIG_PATH="/old/schema.yaml"\n'
    'MPLSTYLE_PATH="/old/style"\n'
    'OTHER="kept"\n'
)


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }


def _git(cwd: Path, *args: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def worktree(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """(main clone with a .env, its worktree on plan/x, env)."""
    env = _env(tmp_path)
    main = tmp_path / "torchcell"
    _git(tmp_path, "init", "-q", "-b", "main", str(main), env=env)
    _git(main, "config", "user.email", "t@t", env=env)
    _git(main, "config", "user.name", "t", env=env)
    (main / "README").write_text("base\n")
    _git(main, "add", "README", env=env)
    _git(main, "commit", "-q", "-m", "base", env=env)
    (main / ".env").write_text(ENV_TEMPLATE)
    wt = tmp_path / "torchcell.worktrees" / "plan" / "x"
    _git(main, "worktree", "add", "-q", str(wt), "-b", "plan/x", env=env)
    return main, wt, env


def _run(wt: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(wt),
        capture_output=True,
        text=True,
        env=env,
    )


def test_setup_rewrites_env_paths_links_data_and_registers_the_merge_driver(
    worktree: tuple[Path, Path, dict[str, str]],
) -> None:
    """Six paths point at the worktree, DATA_ROOT is untouched, data/ links to main, driver set."""
    main, wt, env = worktree
    result = _run(wt, env)
    assert result.returncode == 0, result.stderr
    assert "Worktree setup complete!" in result.stdout
    assert (wt / ".env").read_text() == (
        'DATA_ROOT="/data/root"\n'
        f'ASSET_IMAGES_DIR="{wt}/notes/assets/images"\n'
        f'EXPERIMENT_ROOT="{wt}/experiments"\n'
        f'WORKSPACE_DIR="{wt}"\n'
        f'BIOCYPHER_CONFIG_PATH="{wt}/biocypher/config/linux-arm_biocypher_config.yaml"\n'
        f'SCHEMA_CONFIG_PATH="{wt}/biocypher/config/torchcell_schema_config.yaml"\n'
        f'MPLSTYLE_PATH="{wt}/torchcell/torchcell.mplstyle"\n'
        'OTHER="kept"\n'
    )
    assert (wt / "data").is_symlink() and (wt / "data").readlink() == main / "data"
    assert (wt / ".env.vscode").read_text() == f"PYTHONPATH={wt}:${{PYTHONPATH}}\n"
    assert _git(wt, "config", "merge.weeklynote.driver", env=env) == (
        "python3 scripts/git_merge_weekly_note.py %O %A %B %P"
    )
    assert _git(main, "config", "merge.weeklynote.name", env=env) == (
        "union weekly task-note bodies, keep a single frontmatter"
    )
    assert "pre-commit not found in PATH" in result.stdout


def test_setup_is_idempotent_on_a_second_run(
    worktree: tuple[Path, Path, dict[str, str]],
) -> None:
    """A rerun keeps the copied .env (not a symlink) and the existing data symlink."""
    _, wt, env = worktree
    assert _run(wt, env).returncode == 0
    (wt / ".env").write_text((wt / ".env").read_text() + 'ADDED="1"\n')
    result = _run(wt, env)
    assert result.returncode == 0
    assert ".env file already exists (not a symlink)" in result.stdout
    assert "data/ symlink already exists" in result.stdout
    assert 'ADDED="1"\n' in (wt / ".env").read_text()
    assert f'WORKSPACE_DIR="{wt}"\n' in (wt / ".env").read_text()


def test_data_local_points_data_root_at_the_worktree(
    worktree: tuple[Path, Path, dict[str, str]],
) -> None:
    """--data-local rewrites DATA_ROOT and makes a real data/torchcell directory, no symlink."""
    _, wt, env = worktree
    result = _run(wt, env, "--data-local")
    assert result.returncode == 0, result.stderr
    assert f'DATA_ROOT="{wt}"\n' in (wt / ".env").read_text()
    assert (wt / "data" / "torchcell").is_dir() and not (wt / "data").is_symlink()
    assert "Data storage: LOCAL" in result.stdout


def test_unknown_option_exits_one_with_usage(
    worktree: tuple[Path, Path, dict[str, str]],
) -> None:
    """A bad flag is rejected before anything is written."""
    _, wt, env = worktree
    result = _run(wt, env, "--bogus")
    assert result.returncode == 1
    assert result.stdout.splitlines()[:2] == [
        "Unknown option: --bogus",
        "Usage: ./scripts/setup-worktree.sh [--data-local]",
    ]
    assert not (wt / ".env").exists()
