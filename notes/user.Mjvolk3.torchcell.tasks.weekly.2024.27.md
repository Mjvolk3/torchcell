---
id: r62q9b5yqw785xmaesez58p
title: '27'
desc: ''
updated: 1718917940396
created: 1718737771951
---

Changed to weekly because notes were getting to long for quick preview rendering. Also this might give a better feel for weekly progress, more frequent check ups etc. Previous task notes: [[user.mjvolk3.torchcell.tasks.deprecated.2024.06.18|dendron://torchcell/user.Mjvolk3.torchcell.tasks.deprecated.2024.06.18]]

Since futures is short enough we didn't change it. [[torchcell.tasks.future|dendron://torchcell/user.mjvolk3.torchcell.tasks.future]]

## 2024.06.23

## 2024.06.22

## 2024.06.21

## 2024.06.20

- [x] Send email accepting 240 V install
- [x] Send email accepting GPU and 240 V plug
- [x] Launch random forest models both with cpus and gpus → we are doing over `nt` and codon frequency. This should help us determine which jobs are faster although it seems that we do have a termination issue on some of the jobs giving misleading completion times.



- [ ] Build small `db` on `gila`. → started build running `database/build/build-image-fresh_linux-arm.sh`. → This needs to be done with srun.
- [ ] Per model, per scale, performance v num_params for all models. Double check to see if we can get curving lines, look for examples first.

## 2024.06.19

- [x] Solving power issue. → transitioning to 240 V

## 2024.06.18

- [x] Refactor task notes to use weekly.
- [x] Clean harpoon
- [x] Check `gila` runs and sync, restart if necessary. → rerun `esm_all`. → After sync `find /scratch/projects/torchcell/wandb-experiments/gilahyper-43/wandb -type d | wc -l` returns 1889, which seems huge for 3 gpu. Maybe with hyperparameters models are easier to fit? Or is `cuml` provides that substantive a speed up. 3 gpus, 20 hrs, means around 2 mins per run. `20 hr * 3 gpu /1889 runs * 60 min/hr =1.9057702488 gpu * min / run`. Where cpu runs were sometimes taking 24 hours to complete. `cuml` suggests speed up is only 45x which would mean 30 min per run. The mismatch is likely explained in model configuration.
- [x] Take note on which bars got gpu enabled `cuml` RF. → [[experiments.smf-dmf-tmf-001.results|dendron://torchcell/experiments.smf-dmf-tmf-001.results]]
- [x] Plot RF `1e05` → It looks like `intact_mean` mse is worse that for other methods. For a more apples to apples comparison.
- [x] Launch comparable 100 depth `RF` sweep for `1e05` dataset on `gilahyper` → Check on in morning.
- [x] Somehow power went down, looked like machine was down hit start but, it came back up, relogin, slurm is still going. → tomorrow try to sync runs and see  what they say `sacct` for job run info and run finish info. Not clear what happened. We are instlling dedicated 240 V circuit to fix these issues.
- [x] [[paper.recordings.2024.06.18|dendron://torchcell/paper.recordings.2024.06.18]] → summarize with language model. → [[Summary|dendron://torchcell/paper.recordings.2024.06.18.summary]]
- [x] #ramble We need to work out the case where we have many different genomes. → saving until next project.
