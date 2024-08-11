---
id: 4yghtip48okuj6fwhbyl1g3
title: Apoc
desc: ''
updated: 1723348252680
created: 1723348126295
---

Downloaded jar from here. [apoc release](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/tag/4.4.0.30)

```cypher
@neo4j> RETURN apoc.version() AS output;
+------------+
| output     |
+------------+
| "4.4.0.30" |
+------------+

1 row
ready to start consuming query after 133 ms, results consumed after another 2 ms
```
