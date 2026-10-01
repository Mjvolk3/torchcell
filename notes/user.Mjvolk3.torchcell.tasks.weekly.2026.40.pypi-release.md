---
id: mb0eg1jgw0suqe3dg56tcoy
title: pypi-release
desc: ''
updated: 1790812172761
created: 1790812172761
---

## 2026.09.30

- [x] PR: PyPI publishing rejoins the semantic release (build in the action, dist on the GitHub release, trusted publishing to PyPI); the never-run Dendron `publish.yml` retired; see [[versioning]] (2026.09.30 section) and the contributing guide.
- [ ] Register the trusted publisher on pypi.org: project `torchcell`, owner `Mjvolk3`, repository `torchcell`, workflow `semantic-release.yaml`, environment `pypi`. User-only.

## 2026.10.01

- [x] PR: `REL` commit cutting 1.6.0 so GitHub and PyPI show the same release, plus a manual `publish-tag` job (`gh workflow run semantic-release.yaml -f tag=vX.Y.Z`) for a tag that did not reach PyPI; see [[versioning]] (2026.10.01 section).
