---
id: 8xnb2yyszeqyw6ll3qoiknt
title: Run in Env
desc: ''
updated: 1790409464291
created: 1790409464291
---

## 2026.09.26 - pre-commit wrapper that puts the torchcell env on PATH

`scripts/run-in-env.sh <cmd> [args...]` prepends `~/miniconda3/envs/torchcell/bin` to `PATH` and execs the command, so a `language: system` pre-commit hook can say `python scripts/x.py` whether or not the env is activated at commit time. Same convention as `scripts/run-mypy.sh`, generalized: the two Phase 0a hooks (`test-quality`, `paired-tests`) use it.
