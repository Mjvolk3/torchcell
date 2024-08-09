---
id: g7npubvvpk396r8oap7jwey
title: Biocypher_out_combine
desc: ''
updated: 1723177481695
created: 1723177187886
---

## 2024.08.08 - How it works

Pass a list of `biocypher-out` dirs that you want to join. They should be part of the same underlying schema otherwise an error will be thrown. If you have added as in appended to the schema the `combine` should still work. It only fails if you modify key value pairs in the schema. This type of combine is not allowed because it signifies that the underlying data is different and the graphs cannot be combined in principle.

```bash
python -m torchcell.database.biocypher_out_combine \                                                                                                  23:19
"/Users/michaelvolk/Documents/projects/torchcell/database/biocypher-out/2024-08-08_20-47-09" \
"/Users/michaelvolk/Documents/projects/torchcell/database/biocypher-out/2024-08-08_21-24-50" \
--output_base_dir "/Users/michaelvolk/Documents/projects/torchcell/database/biocypher-out" \
--neo4j_yaml "/Users/michaelvolk/Documents/projects/torchcell/biocypher/config/combine_config.yaml"
```

Make the script executable.

```bash
docker exec tc-neo4j /bin/bash chmod +x biocypher-out/2024-08-08_23-16-43_combined/neo4j-admin-import-call.sh   
```

Run the import.

```bash
docker exec tc-neo4j /bin/bash chmod +x biocypher-out/2024-08-08_23-16-43_combined/neo4j-admin-import-call.sh
```

If the import works you will see to import summary.

```bash
IMPORT DONE in 5s 178ms. 
Imported:
  323416 nodes
  841603 relationships
  1826530 properties
Peak memory usage: 1.035GiB
```

Due to the nature of duplication we can get the following error. We can ignore this as it won't affect the final graph.

```bash
(torchcell) michaelvolk@M1-MV torchcell % docker exec tc-neo4j /bin/bash -c "cat /var/lib/neo4j/import.report"                                                                                  23:17
Id 'd2a101f9df3ede07ae68eafc260af9130a519c3739bc93c3087c8d17a1d996b7' is defined more than once in group 'global id space'
Id 'b2529f271f4993d2e965a9d13dc23642c16774495709f069bad9bc22137fa63b' is defined more than once in group 'global id space'
Id '243a6a3cbd010322b45021a8a17247cff1807a9e498b7aefcb1af53c01d03494' is defined more than once in group 'global id space'
...
```
