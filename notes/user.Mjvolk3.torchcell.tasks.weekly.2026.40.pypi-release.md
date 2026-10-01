---
id: mb0eg1jgw0suqe3dg56tcoy
title: pypi-release
desc: ''
updated: 1790812172761
created: 1790812172761
---

## 2026.09.30

- [x] PR: PyPI publishing rejoins the semantic release (build in the action, dist on the GitHub release, trusted publishing to PyPI); the never-run Dendron `publish.yml` retired; see [[versioning]] (2026.09.30 section) and the contributing guide.
- [x] Register the trusted publisher on pypi.org: project `torchcell`, owner `Mjvolk3`, repository `torchcell`, workflow `semantic-release.yaml`, environment `pypi`. User-only.

## 2026.10.01

- [x] PR: `REL` commit cutting 1.6.0 so GitHub and PyPI show the same release, plus a manual `publish-tag` job (`gh workflow run semantic-release.yaml -f tag=vX.Y.Z`) for a tag that did not reach PyPI; see [[versioning]] (2026.10.01 section).
- [x] 1.6.0 is on PyPI: the release run (36933195045) bumped, built and made the GitHub release, and its PyPI step was refused (`invalid-publisher`, no publisher registered yet); after the publisher was registered, the manual `publish-tag` run (36936717998) on `v1.6.0` uploaded `torchcell-1.6.0-py3-none-any.whl` (2,458,862 bytes) and `torchcell-1.6.0.tar.gz` (2,059,663 bytes) to PyPI and replaced the GitHub release assets with the same build.
- [x] PR: the README overview figure used a relative path and showed as a broken image on the PyPI 1.6.0 page; it now uses the absolute raw URL, and [[tests.torchcell.test_readme_pypi]] refuses a relative reference. PyPI freezes the description per release, so the page is corrected at the next release.
