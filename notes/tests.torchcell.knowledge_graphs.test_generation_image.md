---
id: b94vkln2lp2ceir51rhq8jf
title: Test_generation_image
desc: ''
updated: 1792057608258
created: 1792057608259
---

## 2026.10.15 - Tag derivation, the loud absent tag, the mode flag, teardown, dry run

`tests/torchcell/knowledge_graphs/test_generation_image.py` runs [[torchcell.knowledge_graphs.generation_image]] against a fake docker (a dict of tag to commit label that answers `image inspect`, `build` and `image rm`): the tag is `tc-neo4j-gen:` plus the first twelve of the full sha and nothing else names an image (`main`, a short sha, `v1.7.4`, uppercase, 41 chars all refused); `image` mode with the tag absent raises `GenerationImageAbsent` naming the tag, the commit, the build command and "no install fallback" after one inspect and no build, and `check` exits 1 with that on stderr; a tag carrying another commit or no label is a `GenerationImageMismatch`; `install` never calls docker, `auto` picks image when present and install otherwise, and `resolve` prints the two lines a shell reads; `ensure` builds only when absent; `remove` is a no-op when absent; `teardown_decision` removes only after a clean merge with zero tasks without `DONE`; `--dry-run` prints `DRY-RUN docker build ...` and `DRY-RUN docker image rm ...` and runs neither. The Dockerfile's text is pinned to the serial container's three install lines, the `direct_url.json` commit check and the commit label.
