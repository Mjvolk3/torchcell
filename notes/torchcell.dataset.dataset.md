---
id: fk8umfllaph5po9aso2jz1y
title: Dataset
desc: ''
updated: 1709700651555
created: 1709699662805
---
## 2024.03.05 Origin - Skip Process Files Exist Check

Added to speed skip one of the checking loops. If this is the only change we can revert to `pyg` `Dataset` class.

```python
# HACK to speed dev
if self.skip_process_file_exist_check:
    return
```

There is another comment which is making me hold out for now. I left this comment but I think it erroneous.

```python
super().__init__()

if isinstance(root, str):
    root = osp.expanduser(osp.normpath(root))

self.root = root
self.transform = transform
self.pre_transform = pre_transform
self.pre_filter = pre_filter
self.log = log
# BOOK once we run self._indices, it computes the len
# if using lmdb this will instantiate the lmdb and cause issues with pickling
self._indices: Sequence | None = None
```

I was able to run [[Experiment_dataset|dendron://torchcell/torchcell.dataset.experiment_dataset]] with the `pyg` `Dataset` so we removed the comment for now. It'll be best to keep using `pyg` when possible.
