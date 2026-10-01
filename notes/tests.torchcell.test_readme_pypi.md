---
id: pncnksec4djf986to7fo32r
title: Test_readme_pypi
desc: ''
updated: 1790895888411
created: 1790895888411
---

## 2026.10.01 - README references must be absolute

The README is the PyPI project description (`pyproject.toml`), and PyPI renders it away from the repository, so a relative path resolves to nothing. Release 1.6.0 showed the overview figure as a broken image because its `src` was `./notes/assets/images/Fig1-torchcell-overview-abc.png`; the logo, already an absolute `raw.githubusercontent.com` URL, rendered.

Three tests: the reference scan returns both the HTML `src`/`href` form and the markdown link form (exact list on a three-reference string); every reference in the README starts with `https://` or `http://` (the list of relative references is exactly `[]`; on the 1.6.0 README it was the overview figure path); and the repository-hosted images are exactly the logo and the overview figure, each an existing file at the path its URL names, so a moved asset fails here and not on the published page.
