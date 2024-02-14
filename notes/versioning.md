---
id: o4ipnw6a881c95lzkx8igcp
title: Versioning
desc: ''
updated: 1707951493042
created: 1707950571933
---
## Semantic-Release for Versioning

[[2024.02.14|dendron://torchcell/user.Mjvolk3.torchcell.tasks#20240214]]

```
python -m pip install python-semantic-release
```

### Semantic-Release for Versioning - pyproject.toml

```toml
[tool.semantic_release]
version_variables = [
    "pyproject.toml:version",
    "torchcell/__version__.py:__version__",
]
branch = "main"
upload_to_pypi = "true"
```

Goes by `MAJOR.MINOR.PATCH` distinction

### Semantic-Release for Versioning - GitHub Actions

We rely on github actions to update versions via assigning a version tag with each release. Any time we want to bump the versioning we can just add one of the following to the next git commit.

- `"major"=="BREAKING CHANGE:"`
- `"minor"=="feat:"`
- `"patch"=="fix:"`

I believe a separate commit is made via github actions which then adds the tag. This means that the tag does not appear immediately in github uppon receiving the push and it is the reason we have resorted to publishing to PyPi from local instead of via git hub actions [[Bash Script used for Publishing on PyPI|dendron://torchcell/pypi-publish#bash-script-used-for-publishing-on-pypi]]. I could not find a way to make the publishing action conditioned on the `".github/workflows/semantic-release.yaml"` action completion. Since we really should base the publishing on the latest version tag, it is not straightforward how to do this.

### Semantic-Release for Versioning - Standard Operating Procedure

- (1) Big brain 🧠 has made progress and wants to update the software version

- (2) Commit with any of the designated text for MAJOR.MINOR.PATCH

Example:

```git
git add .
git commit -m "fix: database to infinity"
```

- (3) Go to [github torchcell](https://github.com/Mjvolk3/torchcell) and check until `".github/workflows/semantic-release.yaml"` has completed and the release version updates accordingly.

![](./assets/images/versioning.md.github-action_semantic-release-version-updating.png)

- (4) Now back on local you will see that the versioning is out of date. You can check this by looking at ay of the files listed in `version_variables` in the `pyproject.toml`. Since github actions bumps the version remotely we now need to sync. `git fetch` and `git merge` to see version update.

Example:

Check these files for current version and you will see mismatch.

```toml
version_variables = [
    "pyproject.toml:version",
    "torchcell/__version__.py:__version__",
]
```


