---
id: h3t0pc4jgpinsa2z9d34all
title: Delta
desc: ''
updated: 1710644184377
created: 1706496737248
---

## Starting Neo4j Database with Apptainer on Delta Interactive CPU

```bash
(base) [mjvolk3@cn034 torchcell]$ hostname -I
172.28.22.97 141.142.144.97 
(base) [mjvolk3@cn034 torchcell]$ export JAVA_HOME=$PWD/jdk-11
(base) [mjvolk3@cn034 torchcell]$ export PATH=$JAVA_HOME/bin:$PATH
(base) [mjvolk3@cn034 torchcell]$ apptainer run --writable-tmpfs \
>   -B $(pwd)/neo4j/data:/data \
>   -B $(pwd)/neo4j/logs:/logs \
>   -B $(pwd)/neo4j/import:/var/lib/neo4j/import \
>   -B $(pwd)/neo4j/plugins:/plugins \
>   neo4j_4.4.30_community.sif
FATAL:   While checking container encryption: could not open image /projects/bbub/mjvolk3/torchcell/neo4j_4.4.30_community.sif: failed to retrieve path for /projects/bbub/mjvolk3/torchcell/neo4j_4.4.30_community.sif: lstat /projects/bbub/mjvolk3/torchcell/neo4j_4.4.30_community.sif: no such file or directory
(base) [mjvolk3@cn034 torchcell]$ ls
bbub                        experiments                                python_app.yaml
config                      MANIFEST.in                                README.md
data                        models                                     relate.project.json
data.csv                    notebooks                                  scripts
data.db.bak                 notes                                      slurm
dendronrc.yml               ontology                                   stubs
dendron.yml                 outputs                                    tb_logs
docker                      package.json                               tests
docker-compose-chatgse.yml  package-lock.json                          torchcell
docker-compose.yml          pods                                       torchcell.code-workspace
Dockerfile                  problematic_mutants_sorted_duplicates.csv  torchcell.egg-info
docker-variables.env        profiles                                   torchcell-remote.code-workspace
docs                        pull_request_template.md                   wandb
env                         pyproject.toml                             yarn.lock
(base) [mjvolk3@cn034 torchcell]$ pwd
/projects/bbub/mjvolk3/torchcell
(base) [mjvolk3@cn034 torchcell]$ cd /scratch/bbub/mjvolk3/torchcell
(base) [mjvolk3@cn034 torchcell]$ apptainer run --writable-tmpfs   -B $(pwd)/neo4j/data:/data   -B $(pwd)/neo4j/logs:/logs   -B $(pwd)/neo4j/import:/var/lib/neo4j/import   -B $(pwd)/neo4j/plugins:/plugins   neo4j_4.4.30_community.sif
[WARN  tini (277288)] Tini is not running as PID 1 and isn't registered as a child subreaper.
Zombie processes will not be re-parented to Tini, so zombie reaping won't work.
To fix the problem, use the -s option or set the environment variable TINI_SUBREAPER to register Tini as a child subreaper, or run Tini as PID 1.
2024-01-29 02:40:24.175+0000 INFO  Starting...
2024-01-29 02:40:24.647+0000 INFO  This instance is ServerId{2d066e2e} (2d066e2e-de04-4212-be8f-e922a1b0fc9e)
2024-01-29 02:40:26.049+0000 INFO  ======== Neo4j 4.4.30 ========
2024-01-29 02:40:27.496+0000 INFO  Performing postInitialization step for component 'security-users' with version 3 and status CURRENT
2024-01-29 02:40:27.496+0000 INFO  Updating the initial password in component 'security-users'
2024-01-29 02:40:28.209+0000 INFO  Bolt enabled on [0:0:0:0:0:0:0:0%0]:7687.
2024-01-29 02:40:29.122+0000 INFO  Remote interface available at http://localhost:7474/
2024-01-29 02:40:29.127+0000 INFO  id: 9FC52821E904C6F6D0042D198579F5651E2ED5FF32BF07A815EB71C25EA6F653
2024-01-29 02:40:29.127+0000 INFO  name: system
2024-01-29 02:40:29.128+0000 INFO  creationDate: 2024-01-29T02:29:51.39Z
2024-01-29 02:40:29.128+0000 INFO  Started.
^C
2024-01-29 02:51:36.504+0000 INFO  Neo4j Server shutdown initiated by request
2024-01-29 02:51:36.505+0000 INFO  Stopping...
2024-01-29 02:51:41.919+0000 INFO  Stopped.
```

## Automated Query to get Schema

Since we don't have the ability with apptainer to do compose for ChatGSE, we can use this shortcut to get the schema to input into an LLM for easier querying. It would be best to somehow expose this to the user.

```bash
// Save this content as print_schema.cypher
CALL db.labels();
CALL db.relationshipTypes();
CALL db.propertyKeys();
CALL db.indexes();
CALL db.constraints();
```

```bash
cypher-shell -u <username> -p <password> -f print_schema.cypher
```
