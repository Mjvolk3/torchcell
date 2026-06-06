---
name: ruff
description: Run ruff linter on staged or specified Python files and fix errors following the project's lint strategy.
---

# Ruff Linter

Run ruff check on staged or specified Python files, then fix errors following torchcell's lint conventions.

> **Adoption status.** torchcell is migrating to ruff (it currently runs black + isort + pyupgrade via `.pre-commit-config.yaml`). Ruff subsumes all three plus flake8 + pydocstyle. If `pyproject.toml` has no `[tool.ruff]` table yet, ruff falls back to its defaults -- add the config (see "Project Ruff Configuration" below) and update `.pre-commit-config.yaml` to the ruff hook as part of the switch. Until then this skill still runs ruff with sensible rules.

## Workflow

```
(edit python) -> /ruff -> /mypy -> /update-notes -> /commit
```

## Arguments

- **With arguments** (e.g. `/ruff torchcell/cell.py`): run on those files.
- **No arguments**: run on staged `.py` files under `torchcell/` and `tests/torchcell/`. If nothing staged, target both directories.

## Step 1: Determine target files

- Arguments: use them directly.
- No arguments: `git diff --cached --name-only -- '*.py'`. If none, target `torchcell/` and `tests/torchcell/`.

## Step 2: Run ruff check

```bash
~/miniconda3/envs/torchcell/bin/python -m ruff check <files_or_dirs>
```

If clean, inform the user and stop.

## Step 3: Auto-fix safe errors

```bash
~/miniconda3/envs/torchcell/bin/python -m ruff check --fix <files_or_dirs>
```

Handles import sorting (`I`), unused imports (`F401`), pyupgrade (`UP`), and most style issues.

## Step 4: Fix remaining errors by type

- **`F401` unused import**: remove it; if intentional (side-effect import), `# noqa: F401`.
- **`F841` unused variable**: remove, or prefix `_` if the assignment has a needed side effect.
- **`UP` rules**: modernize syntax (auto-fixed) -- e.g. `dict` over `typing.Dict`, `X | None` over `Optional[X]`.
- **`D` docstring rules**:
  - `D100` module docstring -- for source files use the project's frontmatter pattern:
    ```python
    """
    torchcell/path/module.py
    [[torchcell.path.module]]
    https://github.com/Mjvolk3/torchcell/tree/main/torchcell/path/module.py
    """
    ```
  - `D101`/`D102`/`D103` -- add docstrings. For tests, `D102` is worth keeping (one-line "what this test verifies" improves `pytest -v`); `D103`/`D104` can be per-file-ignored for tests.
- **`E501` line too long**: ignored globally (matches black's wrapping) -- no action.
- **Other `E`**: usually auto-fixed by `--fix`.
- **`F811` redefined**: remove the duplicate. **`F821` undefined**: add the missing import / fix the typo.

## Step 5: Run ruff format (if you made manual edits)

```bash
~/miniconda3/envs/torchcell/bin/python -m ruff format <files_or_dirs>
```

(Equivalent to black; double quotes, skip-magic-trailing-comma matches torchcell's current black config.)

## Step 6: Re-run ruff check

Repeat until clean.

## Step 7: Stage fixed files

`git add <fixed_files>`.

## Project Ruff Configuration (target -- add to pyproject.toml on adoption)

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "D"]
ignore = ["E501", "D205", "D212", "D415"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["D103", "D104"]
"notes/assets/scripts/**" = ["D100", "D103"]
"experiments/**" = ["D100", "D103"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
skip-magic-trailing-comma = true
```

This mirrors the existing black (double quotes, skip-magic-trailing-comma), isort, and pyupgrade (`--py311-plus`) behavior, plus pydocstyle (google).

## Important Rules

- NEVER add a blanket `# noqa` without a specific rule code.
- NEVER disable rules globally in `pyproject.toml` without discussing with the user.
- Prefer fixing the code over suppressing the warning.
- Do NOT ask extra approval questions -- tool approval prompts are the gates.

## Example Invocations

- `/ruff` -- staged Python (or the whole project)
- `/ruff torchcell/cell.py`
- "fix lint errors in dcell.py"
