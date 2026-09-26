---
id: cc972mthzhgic0hgr2xpmp5
title: Test_deprecate_sh
desc: ''
updated: 1790416756553
created: 1790416756553
---

## 2026.09.26 - deprecate.sh moving a path into a graveyard under tmp_path

`DEPRECATED_DIR` names the graveyard and `DATA_ROOT` a sibling directory. The normal move lands the target under `<graveyard>/<timestamp>__<name>/<name>` beside a `DEPRECATION.txt` whose seven fields (`original_path`, `deprecated_at`, `host`, `user`, `git_head`, `size`, `reason`) are asserted line by line (`git_head` is `not-a-git-repo` outside a repository), and the three stdout lines are exact. A graveyard nested inside `DATA_ROOT` exits 2 and moves nothing; a missing target exits 1 naming it; no argument exits 1 with the usage. Phase 4 of [[plan.test-suite-buildout.2026.09.25]].
