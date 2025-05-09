---
id: 0h00tkee1zuibhn7ryy9r7u
title: Cypher Shell
desc: ''
updated: 1745811905639
created: 1709780790509
---

## 2024.03.06 - Start and use Neo4j Torchcell Database

```bash
neo4j@neo4j> SHOW DATABASES;
neo4j@neo4j> START DATABASE torchcell;
neo4j@neo4j> SHOW DATABASES;
neo4j@neo4j> :use torchcell
```

## 2024.03.20 - Cannot start cypher-shell due to Neo4j lock

- If `cypher-shell` doesn't open you can try to delete the lock, restart the container and try again. I forget what it is called. `store_lock` ? `file_dock`? `data_lock`? `db_lock`? Some sort of lock 🔒... It is `store_lock`.

## 2024.07.06 - Node Count and Edge Count

```bash
MATCH (n)
RETURN count(n) as node_count;
```

```bash
MATCH ()-[r]->()
RETURN count(r) as edge_count;
```

## 2024.07.06 - Create Database

```bash
neo4j@neo4j>
CREATE DATABASE torchcell;
neo4j@neo4j> SHOW DATABASES;
:use torchcell
```
