---
id: oaa6167tsocb57vzku33s9c
title: Docker
desc: ''
updated: 1706953417717
created: 1706953111718
---
## Instructions to Get Image

1. We adapted the [Neo4j-4.4.30-community image](https://github.com/neo4j/docker-neo4j-publish/tree/dae45c73d0c9d68337f01f1711b225a8aef36411/4.4.30/bullseye/community/local-package) by adding python installation where we can install our environment for torchcell.
2. [DownGit](https://minhaskamal.github.io/DownGit/#/home) used for installed the local-package dir.
3. Build the Docker image. We put things into `/database`.

```bash
cd /Users/michaelvolk/Documents/projects/torchcell
docker build --platform linux/amd64 -t michaelvolk/neo4j-community:4.4.30 -f database/Dockerfile.neo4j-4.4.30-community database
```

4. `docker login
5. `

## Docker Commands
