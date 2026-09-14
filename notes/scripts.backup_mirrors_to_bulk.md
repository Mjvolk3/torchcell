---
id: oh3iifl1te9cra8867s2znh
title: Backup_mirrors_to_bulk
desc: ''
updated: 1789364461575
created: 1789364461575
---

## 2026.09.14 - Three tiers

Weekly (Sunday 02:00, `scripts/crontab.txt`) non-destructive `rsync -a` of the provenance
tiers on `/scratch` into `/bulk`: `torchcell-library` (papers, OCR, SI, released data),
`torchcell-raw` (the raw files a loader consumed), and, from this date,
`torchcell-genomes` (reference assembly sets with a manifest per set, resolved through
[[torchcell.sequence.genome.registry]]). No `--delete`, so a file removed on `/scratch`
stays in `/bulk`. Symlinks copy as symlinks, which is why the genomes tier holds real
files only. First run with the tier, by hand on 2026-09-14: 17 files, 4,284,249,415 bytes,
exit status 0, tarball sha256 verified on the copy. The cron line runs from the primary
checkout on `main`, so the new tier is covered by cron from the first Sunday after this
change lands.
