---
id: 9fqj4hl1g0zthtgk5gpiuxa
title: Test_build_command
desc: ''
updated: 1791270298268
created: 1791270298268
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): the cliff `build` command with the working directory at `tmp_path` (a mirror of the repo's `database/build/` file names, created empty) and `subprocess.Popen` replaced by a recorder. Pinned: parser default and the invalid-choice message, the 0755 chmod, the exact `Popen` call, line streaming (a blank line prints empty), and the returned code.

Finding: `--mode fresh` names `build_image_fresh_linux-arm.sh`, but the repo file is `build-image-fresh_linux-arm.sh`, so fresh mode raises `FileNotFoundError` at `os.chmod` before any process starts (build_command.py:36).
