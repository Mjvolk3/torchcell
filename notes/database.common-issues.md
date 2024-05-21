---
id: 21gvicjbpt0qb67vj3r0nao
title: Common Issues
desc: ''
updated: 1715732083063
created: 1715731135136
---
## 2024.03.18 - Proper Indentation

If we are missing data check adapters for proper indentation.

```python
for perturbation in genotype.perturbations:
  ...
  return edges
```

```bash
cat import.report
f4a674caf85adc0800397536db8ed9d7a941e70d8ecf86a3843c25f28516b4a7 (global id space)-[PhenotypeMemberOf]->29d4c34a4008c66eb05e3a34e7366cd05e8fb3ca5b58fd7dae4a215bb201f3ec (global id space) referring to missing node f4a674caf85adc0800397536db8ed9d7a941e70d8ecf86a3843c25f28516b4a7
f4a674caf85adc0800397536db8ed9d7a941e70d8ecf86a3843c25f28516b4a7 (global id space)-[PhenotypeMemberOf]->73ca7d4694085d74c54e04db8956f128262c0ef4cf2bbe46775b97cb170a7813 (global id space) referring to missing node f4a674caf85adc0800397536db8ed9d7a941e70d8ecf86a3843c25f28516b4a7
