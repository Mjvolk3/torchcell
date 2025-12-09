---
id: qy1vsykzz9pyheq2y0cx0j4
title: 224126 Compressing Findings from Sugraph Differences
desc: ''
updated: 1763011589585
created: 1763008927317
---
- Read weekly report @torchcell/notes/user.Mjvolk3.torchcell.tasks.weekly.2025.44.report.md
- We have just finished concluding from all of the experiments that the model architecture from @torchcell/models/hetero_cell_bipartite_dango_gi.py is fundamentally flawed. Since we are using costly sugraph ops, We cannot finishing training for on datasets that have > 100,000 training samples. It simly takes to long. We are now in a state where the git diff is in shambles. I worked with claude code to generate all the tests as quickly as a possibly could. I now want to clean up the repo and commit our findings. The issue is that files are sort of everywhere.
- we have a lot of profile results that are not organized correctly. For example @/home/michaelvolk/Documents/projects/torchcell/profile_results_2025-11-05-03-34-41.txt and similar files. I want all the rsults from profiling, experiments, etc to be located similar to this. @experiments/006-kuzmin-tmi/profiling_results/profiling_087a_2025-11-07-16-45-23/summary_2025-11-07-16-45-23.txt . If we just move files we will have the obvious issue of reproducibility from the original production file. We should add this file linking to weekly report.
- I want to add to @notes/user.Mjvolk3.torchcell.tasks.weekly.2025.44.report.md at the bottom to include a brief description of the related different files that were used to make the conclusions we made in the report. For instance if we want to reproduce this kind of experiment in future there are some useful md's for this. @/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/DDP_DEVICE_FIX.md ... some of the files are more about running tests. Others results. We have been naming files like this @torchcell/notes/experiments.006-kuzmin-tmi.scripts.hetero_cell_bipartite_dango_gi_lazy.2025.11.5.claude-export.gpu-mask-opt.md indiciating claude had something to do with this file, in this case it is a conversation export. We want to do similar things for results or analysis, for instance, claude-results, or claude-analysis.
- I am sort of tempted to just remove a bunch of files now that I have my conclusion but this seems like it would remove the work used to achieve that conclusion. But I was not careful at all to document because I was just racing to the finish. The most important conlcusions on the ones from that report.
- Decided I am just going to check every note and make comment here. For fixup before commit.
- @/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/DANGO_VS_LAZY_HETERO_PROFILE_COMPARISON.md needs to be moves to a dendron note. Just transfer to appropriately named md file based on other dendron notes. We will fixfronmatter later. Don't use all caps... we are going to do this for a few more files follow same pattern.
- This file needs to be moved into notes and referenced in weekly report @/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/analysis/final_preprocessing_solution.md
- All yaml experiment configs can stay.
- All slurm scripts can stay
- All python files can stay but we need to review the changes made ot files in saved in torchcell dir (package files)
- Are there any files that are accessively large we should be worried about commit? Some of the claude-export files. Ultimately, I think we want to keep everything possible because in futurue we want to build testing infrastructure for models of form $f(cell_graph, perturbations)$.
- I want you to know that things got so messy at times that some of the results might be a bit fuzzy because we may have preemptive conclusions in some of the note files, but I am certain of the major conclusions made in @torchcell/notes/user.Mjvolk3.torchcell.tasks.weekly.2025.44.report.md
