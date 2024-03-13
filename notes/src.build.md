---
id: rf24oyvklzw6mi39yiev86a
title: Build
desc: ''
updated: 1705660924854
created: 1705660608170
---

## Source Build Instructions

1. Bump version
2. Remove old distribution
3. Build new distribution
4. Upload to PyPI

```bash
bumpver update -p
rm -rf dist/
python -m build
twine upload dist/*
```
