---
id: 0dl36oeyxl1w1ypz5hymt15
title: '203737'
desc: ''
updated: 1736131125348
created: 1736131059554
---
```python
for eid in list(H.edges)[:3]:  # Look at first 3 edges
    print(f"\nEdge ID: {eid}")
    print("Properties:")
    for key, val in H.edges[eid].properties.items():
        print(f"  {key}: {val}")
```