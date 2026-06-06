---
name: mypy
description: Run mypy on staged or specified Python files and fix errors following the project's type-checking strategy.
---

# Mypy Fixer

Run mypy on staged or specified Python files, then fix errors following torchcell's type-checking strategy.

## torchcell's mypy config (pyproject.toml `[tool.mypy]`)

torchcell is configured strict-ish, with one deliberate relaxation:

- `files = ["torchcell/**/*.py", "tests/**/*.py"]`, `python_version = "3.11"`.
- **`ignore_missing_imports = true`** is set globally -- this is intentional (the scientific-Python stack is largely untyped). Do NOT fight it: you generally will NOT see `[import-untyped]`/`[import-not-found]` errors, and you should NOT add per-import `# type: ignore` for third-party libs.
- Strict flags ON: `disallow_untyped_defs`, `disallow_incomplete_defs`, `disallow_untyped_calls`, `disallow_any_generics`, `no_implicit_optional`, `warn_return_any`, `warn_unused_ignores`, `strict_equality`, etc.

Because `warn_unused_ignores = true`, a stale `# type: ignore` is itself an error -- remove ones that no longer apply.

## Workflow

```
(edit python) -> /mypy -> /ruff -> /update-notes -> /commit
```

## Arguments

- **With arguments** (e.g. `/mypy torchcell/models/dcell.py`): run on those files.
- **No arguments**: run on staged `torchcell/**` + `tests/**` `.py` files. If nothing staged, run the configured `files` set.

## Step 1: Determine target files

- Arguments: use them directly.
- No arguments: `git diff --cached --name-only -- 'torchcell/**/*.py' 'tests/**/*.py'`. If none, run the whole configured set.

## Step 2: Run mypy

Config is picked up from `pyproject.toml`, so just pass the files (no need for `--strict` -- the flags are already set):

```bash
~/miniconda3/envs/torchcell/bin/python -m mypy --show-error-codes <files>
```

If mypy passes clean, inform the user and stop. (A `torchcell/mypy_check.py` helper also exists if you prefer the project's own entry point.)

## Step 3: Fix errors by type

- **`[no-untyped-def]`**: add annotations; `-> None` for non-returning functions.
- **`[union-attr]` / `[attr-defined]`**: add `assert x is not None` (value must not be None here) or `if x is not None:` (None is a valid state). Prefer asserts for invariants.
- **`[type-arg]`**: add explicit type params (`dict[str, Any]`, `Tensor` shapes via annotations) instead of bare generics.
- **`[arg-type]`**: fix the mismatch by narrowing, casting, or correcting the call.
- **`[no-any-return]` / `warn_return_any`**: annotate the return and narrow the value (common with `torch`/`numpy` ops that return `Any`).
- **`[untyped-decorator]`**: for framework decorators mypy can't type, `# type: ignore[misc]` with a one-line reason; for custom decorators, annotate them.
- **Unused ignore**: delete the stale `# type: ignore`.
- **All others**: fix the code. Never add bare `# type: ignore` without a specific code.

## Step 4: Re-run mypy

Repeat until clean.

## Step 5: Stage fixed files

`git add <fixed_files>`.

## Important Rules

- `ignore_missing_imports = true` is already global -- do NOT add per-import ignores for third-party libs, and do NOT remove it.
- NEVER add bare `# type: ignore` without a specific error code.
- `warn_unused_ignores` is on -- remove ignores that no longer apply.
- Prefer fixing the code over suppressing the warning.
- Do NOT ask extra approval questions -- tool approval prompts are the gates.

## Example Invocations

- `/mypy` -- staged Python (or the whole configured set)
- `/mypy torchcell/models/dcell.py`
- "fix type errors in cell.py"
