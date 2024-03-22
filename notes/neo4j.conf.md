---
id: 88g46j0tyfulkalj986ft57
title: Conf
desc: ''
updated: 1710993353367
created: 1710993353367
---
## 2024.03.20 - Trying to Speed up Querying Locally

Added to base config.

```
dbms.threads.worker_count=10
dbms.logs.query.enabled=verbose
dbms.logs.query.threshold=0
```

These are primarily needed to speed up querying and to diagnose queries.

These are originally commented out. Typically they are dynamically updated.

```
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
```

Comments say set to RAM - Heap

```
dbms.memory.pagecache.size=20G
```

Previously used 28G but failed to start because out of memory. There must be some overhead elsewhere. 20G should help test if this works.

## 2024.03.21 - Specifying Memory

When we specify the memory we are more likely to get and #OOM, therefore we comment them out to update dynamically.

[[2024.03.21 - Setting Heap Causes Query Failure|dendron://torchcell/experiments.smf-dmf-tmf-001.node_removal_domain_overlap#20240321---setting-heap-causes-query-failure]]

## 2024.03.21 - Terminal Commands Copied into neo4j.conf File
