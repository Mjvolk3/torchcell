---
id: d89wpb229frpmqe25uz142u
title: '31'
desc: ''
updated: 1722806053874
created: 1722369588884
---
## 2024.08.01

- 🔲 We keep having permissions issues with root
- [x] #ramble Junyu is working on Beta-Carotene.  

## 2024.07.31

- [x] Troubleshoot speed issue. → it is beyond speed issues the program freezes for interaction datasets. → Not all data was being serialized `publication`, I think this was messing up data transformation in the `cell_adapter`
- [x] Check that the yaml configs for fitness have publication.
- [x] Annual review. Use [[ACCESS Resource Report|dendron://torchcell/access.report.2024.05.15]]
- [x] Build interaction db

## 2024.07.30

- [x] Get biochatter working
- [x] Update `gh_neo4j.conf` so `Connect URL` is autofilled.
- [x] Respond to biocypher PR review with next steps.
- [x] Update edge and node definitions to match biolink more appropriately. → `mentions` somehow not recognized but can use `is_a` and this solves it.
- [x] Document biochatter startup → [[Biochatter|dendron://torchcell/database.docker.biochatter]]
- [x] Update [[torchcell.adapters.cell_adapter]] for new datasets → added interactions.
- [x] Update `torchcell/biocypher/config/torchcell_schema_config.yaml` for new datasets → added interactions.
- [x] Add gene essentiality to schema and clearly differentiated from current fitness. → added to datasets last week [[30|dendron://torchcell/user.Mjvolk3.torchcell.tasks.weekly.2024.30]] but not documented.
- [ ] First interaction db build attempt.

## 2024.07.29

- [x] biochatter trouble shoot
