---
id: aa3jczwffj7pbcvjsvhmfu8
title: Docker
desc: ''
updated: 1709223192650
created: 1706953111718
---
## Instructions to Get Image

(1) We adapted the [Neo4j-4.4.30-community image](https://github.com/neo4j/docker-neo4j-publish/tree/dae45c73d0c9d68337f01f1711b225a8aef36411/4.4.30/bullseye/community/local-package) by adding python installation where we can install our environment for torchcell.
(2) [DownGit](https://minhaskamal.github.io/DownGit/#/home) used for installed the local-package dir.
(3) Build the Docker image. We put things into `/database`.

```bash
cd /Users/michaelvolk/Documents/projects/torchcell
docker build --platform linux/amd64 -t michaelvolk/neo4j-community:4.4.30 -f database/Dockerfile.neo4j-4.4.30-community database
```

Alternatively, no cache for complete rebuild.

```bash
docker build --no-cache --platform linux/amd64 -t michaelvolk/tc-neo4j:0.0.1 -f database/Dockerfile.tc-neo4j-4.4.30-community database
```

(4) `docker login``

## 2024.02.08 - Troubleshooting Docker Build Local

Using `database/Dockerfile.tc-neo4j` (name is subject to change). Losing straight forward path to do this... just copying useful commands now and will have to sort things out later.

- We want to be able to build the image for both linux and m1 amd architecture.

```bash
(torchcell) michaelvolk@M1-MV database % cd database                                                                                                11:48
docker buildx build --platform linux/amd64,linux/arm64 -t tc-neo4j-4.4.30-community:0.0.02 -f Dockerfile.tc-neo4j-4.4.30-community --push .
```

The reason we need push

> When using docker buildx to build images, especially with the --platform option for multi-platform builds, the built images are not automatically loaded into your local Docker images list. Instead, they are stored in Docker's build cache. To make the image available locally, you need to use the --load flag when building the image. This flag is necessary when you want the built image to be directly usable on your local Docker daemon. #ChatGPT

```bash
#0 2.050  59800K .......... .......... .......... .......... .......... 99% 88.0M 0s
#0 2.051  59850K .......... .......... .......... .......... .......... 99% 89.4M 0s
#0 2.051  59900K .......... .......... .......... .......... .......... 99% 84.0M 0s
#0 2.052  59950K .......... .......... .......... .......... .......... 99% 98.3M 0s
#0 2.052  60000K .......... .                                          100%  149M=1.7s
#0 2.053 
#0 2.053 2024-02-06 17:49:54 (34.1 MB/s) - '/miniconda.sh' saved [61451533/61451533]
#0 2.053 
#0 2.078 PREFIX=/miniconda
#0 2.266 Unpacking payload ...
#0 2.269 qemu-x86_64: Could not open '/lib64/ld-linux-x86-64.so.2': No such file or directory
#0 2.277 qemu-x86_64: Could not open '/lib64/ld-linux-x86-64.so.2': No such file or directory
------
Dockerfile.tc-neo4j-4.4.30-community:9
--------------------
   8 |     ENV MINICONDA_VERSION 4.9.2
   9 | >>> RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_${MINICONDA_VERSION}-Linux-x86_64.sh -O /miniconda.sh && \
  10 | >>>     /bin/bash /miniconda.sh -b -p /miniconda && \
  11 | >>>     rm /miniconda.sh
  12 |     
--------------------
ERROR: failed to solve: process "/bin/sh -c wget https://repo.anaconda.com/miniconda/Miniconda3-py39_${MINICONDA_VERSION}-Linux-x86_64.sh -O /miniconda.sh &&     /bin/bash /miniconda.sh -b -p /miniconda &&     rm /miniconda.sh" did not complete successfully: exit code: 1
```

- Use `TARGETARCH` for target architecture.

```bash
# Install Miniconda
ARG TARGETARCH
ENV MINICONDA_VERSION 4.9.2

RUN if [ "$TARGETARCH" = "amd64" ]; then \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh; \
elif [ "$TARGETARCH" = "arm64" ]; then \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /miniconda.sh; \
fi && \
/bin/bash /miniconda.sh -b -p /miniconda && \
rm /miniconda.sh

```

- Run build from root

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t tc-neo4j:0.0.3 -f database/Dockerfile.tc-neo4j --push database
```

- Build now successful but we have some permissions error.

```bash
michaelvolk@M1-MV torchcell % docker logs tc-neo4j       
[FATAL tini (7)] exec /startup/docker-entrypoint.sh failed: Permission denied
```

I am not sure where this comes from since we have `database/local-package/docker-entrypoint.sh`... just kidding I found it. `./local-package/* /startup/`. #ChatGPT suggest giving execute permissions since they only had read write permissions. 🐢 The image is starting to take eons to both build and push.

```bash
COPY --chmod=755 ./local-package/* /startup
```

Tagging options so we can just use tag `latest` on [[Delta Build Database from Fresh|dendron://torchcell/database.apptainer#delta-build-database-from-fresh]]

- All in one shot. This is the best option ✅.

```bash
docker login
docker buildx build --platform linux/amd64,linux/arm64 -t michaelvolk/tc-neo4j:0.0.3 -t michaelvolk/tc-neo4j:latest -f database/Dockerfile.tc-neo4j database --push
```

- Tag after creation → ⛔️ this won't work because when we build with `buildx` this images is not part of the list of images... might be in cache, but with push can build image from with a `docker pull michaelvolk/tc-neo4j:0.0.3`

```bash
docker push michaelvolk/tc-neo4j:latest
```

- It is wise to load first then push. This way we can test locally without having to pull, which wastes time... or is wise... ⛔️ cannot do this for multi-platform builds, have to stick with build.

```bash
docker login
docker buildx build --platform linux/amd64,linux/arm64 -t michaelvolk/tc-neo4j:0.0.3 -t michaelvolk/tc-neo4j:latest -f database/Dockerfile.tc-neo4j database --load
```

```bash
docker push michaelvolk/tc-neo4j:0.0.3
docker push michaelvolk/tc-neo4j:latest
```

- [[2024.02.13|dendron://torchcell/user.Mjvolk3.torchcell.tasks#20240213]]

## Docker Image and Container Life Cycle

Turns out I've been an idiot for a few hours and I didn't realize that when I pulled a new image it didn't matter because I had a container that was running the older image so I kept getting the previous results.

- Build image, have to push for multiplatform images

```bash
docker login
docker buildx build --platform linux/amd64,linux/arm64 -t michaelvolk/tc-neo4j:0.0.3 -t michaelvolk/tc-neo4j:latest -f database/Dockerfile.tc-neo4j database --push
```

- For development we should avoid versioning... there should be one version. ⛔️ You should `--no-cache` to ensure saved layers get overwritten for best reproducibility.

```bash
docker buildx build --no-cache --platform linux/amd64,linux/arm64  michaelvolk/tc-neo4j:latest -f database/Dockerfile.tc-neo4j database --push 
```

- Pull image

```bash
docker pull neo4j:latest
```

- Create containers.

```bash
docker run -d --name tc-neo4j -p 7474:7474 -p 7687:7687 michaelvolk/tc-neo4j:latest
```

- Bind `biocypher-out` with volumes.

```bash
docker run -d --name tc-neo4j -p 7474:7474 -p 7687:7687 -v $(pwd)/database/biocypher-out:/database/biocypher-out michaelvolk/tc-neo4j:latest
```

- Bind `/torchcell` so we can call source code that exists in files, not installed libraries.. This should only be used for fast development and I think shoudn not be recommended in general.

```bash
docker run -d --name tc-neo4j -p 7474:7474 -p 7687:7687 -v $(pwd)/database/biocypher-out:/database/biocypher-out -v $(pwd)/torchcell:/torchcell michaelvolk/tc-neo4j:latest
```

- This works for passing in username and password. This way it doesn't need to be interactively changed.

```bash
docker run -d --name tc-neo4j -p 7474:7474 -p 7687:7687 -v $(pwd)/database/biocypher-out:/database/biocypher-out -v $(pwd)/torchcell:/torchcell -e NEO4J_AUTH=neo4j/torchcell michaelvolk/tc-neo4j:latest
```

- For writing the the lmdb database we need access to the `/data/torchcell` dir. ⛔️ `/data` is already being used by neo4j. It contains the dbms. It should not be overwritten.
- We also added the `NEO4J_ACCEPT_LICENSE_AGREEMENT` to the env. This could be added to the base image, but it is nice for now to keep it clear that we are using enterprise as academics.

```bash
docker run --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes -d --name tc-neo4j -p 7474:7474 -p 7687:7687 -v $(pwd)/database/biocypher-out:/database/biocypher-out -v $(pwd)/torchcell:/torchcell -v $(pwd)/data:/torchcell_data -v $(pwd)/database/data:/var/lib/neo4j/data -e NEO4J_AUTH=neo4j/torchcell michaelvolk/tc-neo4j:latest
```

- Trying to map the data in `torchcell` to the data in database with minimal path disturbance.

```bash
docker run --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes -d --name tc-neo4j -p 7474:7474 -p 7687:7687 -v $(pwd)/biocypher-out:/var/lib/neo4j/biocypher-out -v $(pwd)/torchcell:/torchcell -v $(pwd)/data/torchcell:/var/lib/neo4j/data/torchcell -v $(pwd)/database/data:/var/lib/neo4j/data -v $(pwd)/database/.env:/.env -v $(pwd)/biocypher:/var/lib/neo4j/biocypher -e NEO4J_AUTH=neo4j/torchcell michaelvolk/tc-neo4j:latest
```

- We probably don't want biocypher-out to be in the project dir because it contains the bash script that is specific to the environment.

BOOK

```bash
docker run --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes -d --name tc-neo4j -p 7474:7474 -p 7687:7687 -v $(pwd)/database/biocypher-out:/var/lib/neo4j/biocypher-out -v $(pwd)/torchcell:/torchcell -v $(pwd)/data/torchcell:/var/lib/neo4j/data/torchcell -v $(pwd)/database/data:/var/lib/neo4j/data -v $(pwd)/database/.env:/.env -v $(pwd)/biocypher:/var/lib/neo4j/biocypher -e NEO4J_AUTH=neo4j/torchcell michaelvolk/tc-neo4j:latest
```

```bash
docker run -d --name tc-neo4j -p 7474:7474 -p 7687:7687 -v $(pwd)/database/biocypher-out:/database/biocypher-out -v $(pwd)/torchcell:/torchcell -v $(pwd)/data:/torchcell_data -e NEO4J_AUTH=neo4j/torchcell michaelvolk/tc-neo4j:latest
```

- Start container

```bash
docker start tc-neo4j
```

- Running interactive terminal

```bash
docker exec -it tc-neo4j /bin/bash
```

- Stopping and removing containers.

```bash
docker stop tc-neo4j
docker rm tc-neo4j
```

### Docker Image and Container Life Cycle - Local Source with Env Source

Copying local source with volumes is a quick and dirty way to iterate on code without having to rebuild images. The only caveat is that when the source changes the container needs to be stopped and started again.

## GitHub Action Docker Build and Push Template

The reason to do this is we don't need to use buildx which uses emulation for building `amd64` and `arm64`. Github actions supposedly uses native hosts so we don't have to worry about any potential build issues. Also this simplifies moving between local and remote since we just have to push with github to push to docker hub.

```yaml
name: Build and Publish Docker image

on:
  push:
    branches:
      - main

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
    - name: Check out the repo
      uses: actions/checkout@v2

    - name: Log in to the GitHub Container Registry
      uses: docker/login-action@v1
      with:
        registry: ghcr.io
        username: ${{ github.repository_owner }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v2
      with:
        context: .
        file: ./Dockerfile
        push: true
        tags: ghcr.io/${{ github.repository }}/myapp:latest
```

## Docker Update TorchCell Source Without Image Rebuild

We can make small changes to the source code and if the dependencies don't change we should be able to run a fast `python -m pip install torchcell --upgrade` to update the container and run the db build. We really need to only recreate the entire image when dependencies change.  

```bash
docker start tc-neo4j
docker exec -it tc--neo4j python -m pip install torchcell --upgrade
```

```bash
apptainer exec --writable-tmpfs <container_path> python -m pip install torchcell --upgrade
```

## Using cypher-shell to Access torchcell Database

Once the docker container is started you can run the cypher-shell interactively.

```bash
% docker start tc-neo4j
% docker exec -it tc-neo4j /bin/bash
% cypher-shell
# enter username and password
> SHOW DATABASES;
# at first only neo4j and system will appear
> :use system;
> CREATE DATABASE torchcell;
> SHOW DATABASES;
# now torchcell should appear
> MATCH (n)
  RETURN (n) LIMIT (10);
# print out 10 data nodes. 
> STOP DATABASE torchcell;
```

### Using cypher-shell to Access torchcell Database - Stop Database to Allow Bulk Import

The `requestedStatus`, shows if the database has been stopped with the `"online"` or `"offline"` distinction. If we do not do this we get an error that the database is open and needs to be closed for bulk import.

You can also start the database back up after it has been created, and stopped.

```bash
START DATABASE torchcell
```

```BASH
neo4j@neo4j> SHOW DATABASES;
+------------------------------------------------------------------------------------------------------------------------------------+
| name        | aliases | access       | address          | role         | requestedStatus | currentStatus | error | default | home  |
+------------------------------------------------------------------------------------------------------------------------------------+
| "neo4j"     | []      | "read-write" | "localhost:7687" | "standalone" | "online"        | "online"      | ""    | TRUE    | TRUE  |
| "system"    | []      | "read-write" | "localhost:7687" | "standalone" | "online"        | "online"      | ""    | FALSE   | FALSE |
| "torchcell" | []      | "read-write" | "localhost:7687" | "standalone" | "online"        | "online"      | ""    | FALSE   | FALSE |
+------------------------------------------------------------------------------------------------------------------------------------+

3 rows
ready to start consuming query after 685 ms, results consumed after another 61 ms
neo4j@neo4j> STOP DATABASE torchcell;
0 rows
ready to start consuming query after 496 ms, results consumed after another 0 ms
neo4j@neo4j> SHOW DATABASES;
+------------------------------------------------------------------------------------------------------------------------------------+
| name        | aliases | access       | address          | role         | requestedStatus | currentStatus | error | default | home  |
+------------------------------------------------------------------------------------------------------------------------------------+
| "neo4j"     | []      | "read-write" | "localhost:7687" | "standalone" | "online"        | "online"      | ""    | TRUE    | TRUE  |
| "system"    | []      | "read-write" | "localhost:7687" | "standalone" | "online"        | "online"      | ""    | FALSE   | FALSE |
| "torchcell" | []      | "read-write" | "localhost:7687" | "standalone" | "offline"       | "offline"     | ""    | FALSE   | FALSE |
+------------------------------------------------------------------------------------------------------------------------------------+
```

## Docker Common Build Error - Invalid value for option '--nodes'

Looks like this when we try to run the bash script. This error is a sign that there is a path error in the `"neo4j-admin-import-call.sh"`.

```bash
(myenv) root@4c41af09eaab:/var/lib/neo4j# /bin/bash biocypher-out/2024-02-16_01-07-26/neo4j-admin-import-call.sh
Invalid value for option '--nodes' at index 0 ([<label>[:<label>]...=]<files>): Invalid nodes file: /Temperature-header.csv,/Temperature-part.* (java.lang.IllegalArgumentException: File '/Temperature-header.csv' doesn't exist)


USAGE

neo4j-admin import [--expand-commands] [--verbose]
                   [--auto-skip-subsequent-headers[=<true/false>]]
                   [--cache-on-heap[=<true/false>]] [--force[=<true/false>]]
                   [--high-io[=<true/false>]] [--ignore-empty-strings
                   [=<true/false>]] [--ignore-extra-columns[=<true/false>]]
                   [--legacy-style-quoting[=<true/false>]] [--multiline-fields
                   [=<true/false>]] [--normalize-types[=<true/false>]]
                   [--skip-bad-entries-logging[=<true/false>]]
                   [--skip-bad-relationships[=<true/false>]]
                   [--skip-duplicate-nodes[=<true/false>
...
```

## Useful Commands for Checking Source Code Update

This tells us if the changes that we have made on github are actually being reflected in latest github pull for the create knowledge graph script. This is incredibly inefficient and needs to be automated away.

```bash
cd /miniconda/envs/myenv/lib/python3.11/site-packages
cat torchcell/knowledge_graphs/create_scerevisiae_kg_small.py
```

- For `torchcell`

```bash
cat /miniconda/envs/myenv/lib/python3.11/site-packages/torchcell/knowledge_graphs/create_scerevisiae_kg_small.py
```

- For `biocypher`

```bash
cat /miniconda/envs/myenv/lib/python3.11/site-packages/biocypher/_write.py > data/delete.py
```

Looks like you have to exit out of the container for the affects to take place, especially if the update is run from another terminal.

```bash
docker exec -it tc-neo4j python -m pip install git+https://github.com/Mjvolk3/torchcell.git@main
```

## Useful Commands for Profiling Scripts on Delta Interactive Node

We output the profile into `biocypher-out/` since this is mounted. We just need to make sure that the `.pstats` file is outputted to a mounted directory.

```bash
python -m cProfile -o biocypher-out/create_scerevisiae_kg_small.pstats /miniconda/
envs/myenv/lib/python3.11/site-packages/torchcell/knowledge_graphs/create_scerevisiae_kg_small.py
```
