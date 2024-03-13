---
id: a5ssjdm1tqnzd9ullq3eq4v
title: Apptainer
desc: ''
updated: 1709224140184
created: 1707239794928
---

## Delta Build Database from Fresh

Calling this from fresh because it is from the beginning of the entire process. This includes recreating apptainer images.

```bash
apptainer build tc-neo4j.sif docker://michaelvolk/tc-neo4j:latest
```

## 2024.02.13 - Docker image Startup vs Apptainer image startup

Database is already started when starting container, so don't need to work about starting. Apparently this is due to entry point in Dockerfile.tc-neo4j but it doesn't seem to work this way with apptainer. Apptainer might not see entry point?

## 2024.02.21 - Cannot use cypher-shell in Apptainer

```bash
Apptainer> cd /var/lib/neo4j
Apptainer> ls
bin        biocypher-out  conf  import  lib       LICENSES.txt  logs        plugins   README.txt  UPGRADE.txt
biocypher  certificates   data  labs    licenses  LICENSE.txt   NOTICE.txt  products  run
Apptainer> cypher-shell
Connection refused
```

## 2024.02.21 - Building tc-neo4j latest

We get a lot of warnings

```bash
mjvolk3@dt-login02 database % apptainer build -F --disable-cache tc-neo4j.sif docker://michaelvolk/tc-neo4j:late
st

INFO:    Starting build...
2024/02/21 18:11:33  info unpack layer: sha256:31bd5f451a847d651a0996256753a9b22a6ea8c65fefb010e77ea9c839fe2fac
2024/02/21 18:11:33  warn xattr{etc/gshadow} ignoring ENOTSUP on setxattr "user.rootlesscontainers"
2024/02/21 18:11:33  warn xattr{/tmp/build-temp-391887571/rootfs/etc/gshadow} destination filesystem does not support xattrs, further warnings will be suppressed
2024/02/21 18:11:34  info unpack layer: sha256:32b311b806c85db4346e8b1835111a4685f302d3b9df8c823b84513d5a390fa9
2024/02/21 18:11:34  warn xattr{usr/local/share/fonts} ignoring ENOTSUP on setxattr "user.rootlesscontainers"
2024/02/21 18:11:34  warn xattr{/tmp/build-temp-391887571/rootfs/usr/local/share/fonts} destination filesystem does not support xattrs, further warnings will be suppressed
2024/02/21 18:11:35  info unpack layer: sha256:23f2664f4576929643412d268b278955ba16c079dfc8475144b922615506ff44
2024/02/21 18:11:36  info unpack layer: sha256:e028f15ee70b633f7fa9e3e6a8d277d8c06cbe93f30672f1bea10605badd5b67
2024/02/21 18:11:36  info unpack layer: sha256:66b307664f73d473cf6147a8cb791b69511d218aed70986d2609cddb264cacb9
2024/02/21 18:11:36  info unpack layer: sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1
2024/02/21 18:11:36  info unpack layer: sha256:03b8bd5539c62e0fd8050866ea9c7bd4b97a1a3ae95689074a6d0bb3b94bf5a7
2024/02/21 18:11:43  info unpack layer: sha256:f9792d4118c94fa59dc2487f18526687ea79ecae08828dfd2142806eddb101fa
2024/02/21 18:11:47  info unpack layer: sha256:80e9b6e79c7ceaafbf8a7d7588e1ea57086ff8b3f868527183088fb05affab64
2024/02/21 18:11:47  info unpack layer: sha256:d55036009a012788e99e94d92032b31a0a1a927a9a1db4991a951cc532871c75
2024/02/21 18:11:51  warn xattr{var/lib/apt/lists/auxfiles} ignoring ENOTSUP on setxattr "user.rootlesscontainers"
2024/02/21 18:11:51  warn xattr{/tmp/build-temp-391887571/rootfs/var/lib/apt/lists/auxfiles} destination filesystem does not support xattrs, further warnings will be suppressed
2024/02/21 18:11:51  info unpack layer: sha256:8ad41e4cc23e8c92951ec0643eccc6b950d80db09e0f5ad45b20260bc6be6624
2024/02/21 18:13:37  info unpack layer: sha256:0b1edd4228c4a06d0f2ae3bce6a2768c3b9c0c51c3b6c3433eb3d9b631557063
2024/02/21 18:13:37  info unpack layer: sha256:f3aabd6678190972986594b84122ac66a098392e0edbad70c1f468b0a9d312d5
2024/02/21 18:13:41  info unpack layer: sha256:399f3f97107274a060197075a21eeecbd5eae3b7f25e9b2e5ffe89a6d68424a9
2024/02/21 18:13:41  warn xattr{etc/gshadow} ignoring ENOTSUP on setxattr "user.rootlesscontainers"
2024/02/21 18:13:41  warn xattr{/tmp/build-temp-391887571/rootfs/etc/gshadow} destination filesystem does not support xattrs, further warnings will be suppressed
2024/02/21 18:13:41  info unpack layer: sha256:b92b40d41a5467483cea241600a854e12fb3c449989d0e5f91b66e7e9fb3b4c0
2024/02/21 18:13:41  info unpack layer: sha256:5cee011fb80f59f554d18051c5d278f32985c91992cfea54b9f9c88dc9121ced
2024/02/21 18:13:41  info unpack layer: sha256:dbc33d96377805d76c6a361d5b05b62bfbc508e1fb28fd88824b3cebfd486bd8
2024/02/21 18:13:41  warn xattr{data} ignoring ENOTSUP on setxattr "user.rootlesscontainers"
2024/02/21 18:13:41  warn xattr{/tmp/build-temp-391887571/rootfs/data} destination filesystem does not support xattrs, further warnings will be suppressed
2024/02/21 18:13:45  info unpack layer: sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1
INFO:    Creating SIF file...
INFO:    Build complete: tc-neo4j.sif
```

## 2024.02.21 - Build neo4j_4.4.30-enterprise official image

We can get into the apptainer shell, start the database, and run cypher-shell.

There are so warnings that are due to converting the docker images to an apptainer sif.

```bash
mjvolk3@dt-login02 database % apptainer build neo4j_4.4.30-enterprise.sif docker://neo4j:4.4.30-enterprise
INFO:    Starting build...
Getting image source signatures
Copying blob 4f4fb700ef54 skipped: already exists  
Copying blob 5d0aeceef7ee done  
Copying blob 904cfccbb1af done  
Copying blob d6b1435146d7 done  
Copying blob eb34c4b777c6 done  
Copying blob 68e3b493682e done  
Copying config 78253fdb59 done  
Writing manifest to image destination
Storing signatures
2024/02/21 18:06:56  info unpack layer: sha256:5d0aeceef7eeb53c3f853fb229ea7fd13a5a56f4ba371ca48f0477493046b702
2024/02/21 18:06:56  warn xattr{etc/gshadow} ignoring ENOTSUP on setxattr "user.rootlesscontainers"
2024/02/21 18:06:56  warn xattr{/tmp/build-temp-281668347/rootfs/etc/gshadow} destination filesystem does not support xattrs, further warnings will be suppressed
2024/02/21 18:06:57  info unpack layer: sha256:68e3b493682ef0c012693021016cdb635597ad66078e2459b34c1c802d451e43
2024/02/21 18:07:00  info unpack layer: sha256:d6b1435146d7bcae51d98da09c3a494e57b410965054f4a3e55f1dd7a8136855
2024/02/21 18:07:00  warn xattr{etc/gshadow} ignoring ENOTSUP on setxattr "user.rootlesscontainers"
2024/02/21 18:07:00  warn xattr{/tmp/build-temp-281668347/rootfs/etc/gshadow} destination filesystem does not support xattrs, further warnings will be suppressed
2024/02/21 18:07:00  info unpack layer: sha256:904cfccbb1af040ece603b2781a6c70bff35f72d6f65bb2293ccf9213280dfc0
2024/02/21 18:07:00  info unpack layer: sha256:eb34c4b777c6507310d60d8beac64a9c7e025aef534d54754ceac3d10fcdc658
2024/02/21 18:07:00  warn xattr{data} ignoring ENOTSUP on setxattr "user.rootlesscontainers"
2024/02/21 18:07:00  warn xattr{/tmp/build-temp-281668347/rootfs/data} destination filesystem does not support xattrs, further warnings will be suppressed
2024/02/21 18:07:04  info unpack layer: sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1
INFO:    Creating SIF file...
INFO:    Build complete: neo4j_4.4.30-enterprise.sif
mjvolk3@dt-login02 database %
```

At least when you are on the login node exiting the container will lead to the database being stopped. `neo4j stop` automatically called. Checked and it works this away on compute node too 😬, need to run all commands inside withouts exiting.

```bash
Apptainer> neo4j status
Neo4j is running at pid 825056
Apptainer> exit
exit
(torchcell) mjvolk3@dt-login02 torchcell %   apptainer exec --writable-tmpfs \
 
  -B /projects/bbub/mjvolk3/torchcell/biocypher:/var/lib/neo4j/biocypher \
  -B $(pwd)/database/biocypher-out:/var/lib/neo4j/biocypher-out \
  -B $(pwd)/data/torchcell/:/var/lib/neo4j/data/torchcell \
  -B $(pwd)/database/.env:/.env \
  -B $(pwd)/database/conf/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf \
  -B $(pwd)/database/data:/data \
  -B $(pwd)/database/logs:/logs \
  -B $(pwd)/database/import:/var/lib/import \
  -B $(pwd)/database/plugins:/plugins \
  $(pwd)/database/neo4j_4.4.30-enterprise.sif \
  /bin/bash
Apptainer> neo4j status
Neo4j is not running.
Apptainer> 
```

Even when you run commands like this the  database stops.

```bash
[mjvolk3@cn003 torchcell]$ apptainer exec --writable-tmpfs   -B /projects/bbub/mjvolk3/torchcell/biocypher:/var/lib/neo4j/biocypher   -B $(pwd)/database/biocypher-out:/var/lib/neo4j/biocypher-out   -B $(pwd)/data/torchcell/:/var/lib/neo4j/data/torchcell   -B $(pwd)/database/.env:/.env   -B $(pwd)/database/conf/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf   -B $(pwd)/database/data:/data   -B $(pwd)/database/logs:/logs   -B $(pwd)/database/import:/var/lib/import   -B $(pwd)/database/plugins:/plugins   $(pwd)/database/neo4j_4.4.30-enterprise.sif  neo4j start
Directories in use:
home:         /var/lib/neo4j
config:       /var/lib/neo4j/conf
logs:         /var/lib/neo4j/logs
plugins:      /var/lib/neo4j/plugins
import:       /var/lib/neo4j/import
data:         /var/lib/neo4j/data
certificates: /var/lib/neo4j/certificates
licenses:     /var/lib/neo4j/licenses
run:          /var/lib/neo4j/run
Starting Neo4j.
Started neo4j (pid:2234399). It is available at http://0.0.0.0:7474
There may be a short delay until the server is ready.
[mjvolk3@cn003 torchcell]$ apptainer exec --writable-tmpfs   -B /projects/bbub/mjvolk3/torchcell/biocypher:/var/lib/neo4j/biocypher   -B $(pwd)/database/biocypher-out:/var/lib/neo4j/biocypher-out   -B $(pwd)/data/torchcell/:/var/lib/neo4j/data/torchcell   -B $(pwd)/database/.env:/.env   -B $(pwd)/database/conf/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf   -B $(pwd)/database/data:/data   -B $(pwd)/database/logs:/logs   -B $(pwd)/database/import:/var/lib/import   -B $(pwd)/database/plugins:/plugins   $(pwd)/database/neo4j_4.4.30-enterprise.sif  neo4j status
Neo4j is not running.
```

## 2024.02.21 - Comparison between tc-neo4j build and neo4j_4.4.30-enterprise official image

Similar warns but with different paths.

Both logs show a successful build process (INFO: Build complete:), confirming that these warnings did not prevent the creation of the .sif files.

For now these warns will be ignored.

## Fresh Apptainer Build

After doing this we had no issues with database startup and cypher-shell.

```bash
apptainer build -F --disable-cache tc-neo4j.sif docker://michaelvolk/tc-neo4j:latest
```

## 2024.02.29 - Could not install packages due to an OSError

Sometimes we run into this error when we are trying to fresh install `biocypher` and `torchcell`.

```bash
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

I tried to purge pip cache but this did not work.

```bash
pip cache purge
```

The issue was that I only had 16 gb allocated on the `delta` interactive computer node... Also I think that I was trying to install `torchcell` into `(base)` env so essentially installing it twice. I think this is the reason for failure, but regardless I bumped the memory on delta interactive cpu to 32 gb and it now works fine.
