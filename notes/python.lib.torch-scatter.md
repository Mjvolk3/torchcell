---
id: 9w6ib0ycqrc2d1lqyry7nmn
title: Torch Scatter
desc: ''
updated: 1728559027718
created: 1728554513409
---

## 2024.10.10 - Fixing Install on Rocky Linux 9

Sometimes we have issues with `torch_scatter.` Just try to import to see what is going on.

```python
import torch_scatter
```

Error

```bash
ImportError: /opt/conda/envs/py35/lib/python3.5/site-packages/torch_cluster/graclus_cpu.cpython-35m-x86_64-linux-gnu.so: undefined symbol: _ZN6caffe26detail37_typeMetaDataInstance_preallocated_32E
```

I first just tried pip and kept getting the same error. I ran `conda update` and pip install again and it worked ✅. I am not sure why.

```bash
conda update -n base -c defaults conda
pip install torch-scatter
```
