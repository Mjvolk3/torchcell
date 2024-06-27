---
id: 0j3fly72kssdbapr5eqchti
title: '28'
desc: ''
updated: 1719422972133
created: 1719334149824
---
## 2024.06.26

- [x] `gh` Test build from bash script → worked first time, text again...
- [ ] `gh` Test build on bash script
- [ ] Change logo on docs

## 2024.06.25

- [x] Sync and stop runs on `gh` → also made new plots
- [x] Add gpu to gilahyper, now we have 4 gpus.
- [x] Update cgroup → not cgroup but `gres.conf` and `slurm.conf`
- [x] Bash build → only works inside docker container `docker exec -it tc-neo4j /bin/bash`.

- [x] Get docs going on readthedocs. should should make the update cycles on docs much faster which might get me to document. Next step would be to `make` documentation with every release. → requirements issues with need to install `torchcell` therefore `pytorch-scatter` which needs to be installed before... trying to do in `requirements.txt`. → works and think that it is pushes to read the docs on push.

- [ ] `sbatch` build

- [ ] Troubleshoot why docker container is crashing
- [ ] Verify connection to database from M1

- [ ] Compare GPU accelerated Random Forest v CPU random forest.

- [ ] Per model, per scale, performance v num_params for all models. Double check to see if we can get curving lines, look for examples first.

- [ ] First `Ollama` application to write documentation, write tests. Both should come with summaries. Automatically run tests to inspect which tests fail. Test if new dataset code is adherent to schema or can be made adherent to schema. We should also generate schema expansion recommendations. This part is hard, and really needs a vast amount of knowledge over different experimental methods if we want it to scale to 1000s of studies. This will build our understanding of ontologizing in the domain along with predictions over the ontology. This should be evidence enough for funding parallelized pilot scale reactors. 100 L to one 1000 L reactor. Once the we can reliably use the 1000 L reactor to produce a product at profit, we should be able to achieve a second round of funding for replicating the process. So we can penetrate the market.
