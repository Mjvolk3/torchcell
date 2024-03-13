---
id: 4ohanbz0yfq7ue56lsw6e2l
title: Pypi Publish
desc: ''
updated: 1707950533506
created: 1707950350932
---

## Bash Script used for Publishing on PyPI

```bash
#!/bin/bash
cd /Users/michaelvolk/Documents/projects/torchcell
rm -rf ./dist
eval "$(conda shell.bash hook)"
conda activate torchcell
python -m build
twine upload dist/*
```

We use a VsCode workspace task to easily run this command. This might need `+x` permissions to run, I forget.

```json
{
    "label": "tc: publish pypi",
    "type": "shell",
    "command": "source ${workspaceFolder}/notes/assets/scripts/tc_publish_pypi.sh",
    "problemMatcher": [],
},
```
