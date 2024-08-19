---
id: dq5n5fhanv9sb8edg46tn8f
title: Install
desc: ''
updated: 1717641225428
created: 1717641203129
---

[rapids-ai](https://docs.rapids.ai/install)

```bash
python -m pip install --extra-index-url=https://pypi.nvidia.com \                                        21:32
'cudf-cu12==24.4.*' 'dask-cudf-cu12==24.4.*' 'cuml-cu12==24.4.*' \
'cugraph-cu12==24.4.*' 'cuspatial-cu12==24.4.*' 'cuproj-cu12==24.4.*' \
'cuxfilter-cu12==24.4.*' 'cucim-cu12==24.4.*' 'pylibraft-cu12==24.4.*' \
'raft-dask-cu12==24.4.*' 'cuvs-cu12==24.4.*'
```

Moved into a requirements file:
`torchcell/env/requirements_experiments.txt`

Nvidia has it's own PyPI repository so it is crucial to run the command like this:

```bash
python -m pip install -r /home/michaelvolk/Documents/projects/torchcell/env/requirements_experiments.txt --extra-index-url https://pypi.nvidia.com
```
