---
id: ndldd0iseolj1vevbu36klr
title: Rsync
desc: ''
updated: 1710644761232
created: 1710644529547
---
## Rsync Example to Copy Data From Delta To Local

Really only need to do this if globus isn't working.

Run command locally.

```bash
rsync -avz -e "ssh -l mjvolk3" mjvolk3@dt-login02.delta.ncsa.illinois.edu:/scratch/bbub/mjvolk3/torchcell/data/sgd /Users/michaelvolk/Documents/projects/torchcell/data/sgd_delta
```

- `-a` - **archive** to copy files recursively and to preserve symbolic links, file permissions, user & group ownerships, and timestamps
- `-v`- **verbose**
- `-z`- **compress** to reduce amount of data sent over network
- `l mjvolk3` - optional since username is specified in SSH target (mjvolk3@...)
  - Shows can specify username in the SSH command if needed.

## Transfer Database with rsync

/scratch/bbub/mjvolk3/torchcelldatabase/data/databases/torchcell

```bash
rsync -avz -e "ssh -l mjvolk3" mjvolk3@dt-login02.delta.ncsa.illinois.edu:/scratch/bbub/mjvolk3/torchcell/database/data/databases/torchcell /Users/michaelvolk/Documents/projects/torchcell/database/data/databases/torchcell_delta
```
