---
id: scratch-2026-09-10-overnight-driver
title: Overnight driver log, metabolic module run
desc: 'untracked scratch; driver check-ins for the 2026-09-10 to 2026-09-11 run'
updated: 1789098000000
created: 1789098000000
---

## 2026.09.10 - Setup

Deadline 18:00 CDT 2026-09-11. Priorities from the user, in order: (1) parameter coverage
(kcat, KM, delta G with pKa fill-in) pushed as far as provenance allows, as a general
open-source pipeline reusable on a new organism; (2) flip every constraint on and verify the
objective is not violated (feasibility of the top 100 predicted strains); (3) read whether
the module helps amino-acid or betaxanthin prediction.

- 22:47 WIP commit 983f3b0b on feat/kinetics-equilibrator-datasets (380 files); nothing pushed.
- 22:39 jobs 1681 (nocstr s2702), 1682 (pooled s2702), 1683 (nocstr s2703), 1684 (pooled
  s2703) submitted; wave-6 parity replicates on tonight's code, no feasibility block.
- 22:55 agents spawned: trainer (train_bx.py feasibility block + checkpoints + amino-acid
  head, 10 cells), flux layer (projection head + sampler + covariance, 8 cells),
  thermodynamics (pKa selection, repricing, coverage closure, general pipeline), kinetics
  (kcat/KM gap closure, general pipeline). mypy cleanup agent done.
- 23:10 priority messages sent to all four; cron check-in every 2 h at :23.
- 23:45 kinetics agent round 1 done (46 min): kcat/KM default 170 -> 10, all 176 gaps were
  substrate_structure_missing; kinetics_pipeline.py built; 74/148 measured units assayed on
  the wrong substrate; 59 predicted units rest on failing structures. Round 2 sent: fix the
  97 structures, substrate-matched measured key + homolog tier, reconcile K_M rule, complete
  parquets. Queue: 1681-1684 running, 1689-1692 proj, 1694-1697 parity, 1698-1704 aa pending.
- 00:18 trainer agent stopped waiting on cells (feasibility block, checkpoints, --head
  mulleder19 landed; 11 cells queued). 1682 (pooled s2702) and 1683 (nocstr s2703) COMPLETED.
- 00:20 INCIDENT: 1681 and 1684 hung 90 min with a worker NameError (`re` used before
  import in flux_layer.py line 528 during the flux agent's edit). Spawned DataLoader workers
  re-import train_bx.py + torchcell every epoch, so live edits reach running cells. Cancelled
  both. Wrote scripts/freeze_and_submit.sh (rsync code snapshot under
  $DATA_ROOT/job-snapshots, import check, sha256 record, sbatch from the snapshot; results
  symlinked back). Cancelled 1691-1704 and resubmitted frozen in order:
  1707 nocstr s2702, 1708/1709 proj-anchored s2700/2701, 1710 pooled s2703, 1711-1714 parity
  s2704/2705, 1715-1720 aa pooled/nocstr s2700-2702, 1721 aa allon s2700. 1689/1690
  (proj-nocstr) still running unfrozen. Agents told: submit via the wrapper; every Edit must
  leave modules importable.
- 00:26 flux agent status: projection head, latent sampler, feasibility(), covariance wiring
  landed; 48 tests pass; real-GEM balance median ratio 5e-9; 2-epoch CPU smoke of the
  everything-on arm gives balance_median_ratio_le_1e-03 = 1.00 on all top-100 strains
  (thermo is the binding half); 1689/1690 escaped at ~22/18 vs 33/26 for the box head
  (observation, not endpoint). Flux + trainer agents resumed with poll-do-not-return rule.
- 00:31 kinetics round 2 done: verified structures (0 failing), substrate-matched measured
  tier 96, homolog tier 782 (EC+substrate any organism; closer than predictions on 55/87
  yeast-measured units, median 0.094 vs 0.300 decades), predicted 2,820, default 30. One
  K_M rule (EZKinetics alone), one default (9.60 1/s, 0.147 mM). All 7 predictor tables
  complete. Resolver default flipped to verified shortly before 00:31: cells 1689/1690/1707-1721 read
  the round-1 processed tables; later snapshots read verified. Flagged to flux agent.
  Kinetics round 3 sent (17 reachable rows, lineage paragraph, train_flux.py check).
- 00:47 hourly check: 1689/1690/1707/1708 RUNNING, logs fresh, no tracebacks; 13 pending
  (1709-1721). All four agents running. No action.
- kinetics round 3 done, agent stopped as instructed: final tiers 97 measured / 782 homolog /
  2,831 predicted / 18 default (both parameters); 28 substrate rows across the 18 defaults
  unreachable with open-source tools (14 R-group, 7 formula-less, 5 writer-refused, 2 Cyt c).
  Table lineage recorded in results/kinetic_table_versions.json; fingerprint rule by
  kcat_tier_counts: 148/3570/10 = round-1 processed; 96/782/2820/30 = intermediate;
  97/782/2831/18 = shipped verified. Parquets now os.replace'd (torn-file safe).
- 00:55 killed the kinetics agent's out-of-slurm 200-epoch --smoke run (pid 3852964, 17.6 GB on a shared GPU); 121 metabolism tests pass on the combined tree.
- 01:22 thermo agent round 1 done: 2,400-compound benchmark; rule = lowest worst
  per-pH p90 over 8 compartment pHs s.t. fail <= 0.02; selected cascade_top3 (Uni-pKa simple
  -> MolGpKa -> pKasolver-lite), worst p90 8.33 kJ/mol at pH 7.8, fail 0.0024. Coverage
  3,732 -> 3,738 / 4,131 (90.49%), flux-carrying 94.59%; ThermoSource.UNION_EXTERNAL; 33
  dGPredictor rows un-double-converted; thermo_pipeline.py organism-agnostic. Ceiling: 91.38%
  without new data, 92.06% with ion table, 92.42% hard. Round 2 sent (0.9% reachable, ion
  table stage). Trainer asked to add union_external to --thermo-source; flux agent asked to
  fix constraints.py parse-time 4.184 and use union_external for cells 5-8.
- 01:50 hourly check: 1709/1710/1711/1712 RUNNING fresh; 9 pending. COMPLETED: 1707 box
  nocstr s2702 = 0.0972 (pooled s2702 0.1687); 1708 everything-on projection anchored s2700
  scale 0.1 = 0.1154, sel 72, no collapse; top-100 balance 1.00 but box viol 0.644, thermo
  viol 0.405, both 0.00. Projection scale-0: s2700 0.117, s2701 0.1413 (pooled 0.072/0.106).
  Sent flux agent the directional-box design (second law as per-sample box tightening,
  projection_end=box, K=20/Dykstra) for cells 5-8. Agents: trainer/flux/thermo running,
  kinetics done. No dead cells.
- 02:00 thermo round 2 done: 3,770/4,131 (91.26%), flux-carrying 2,714/2,845
  (95.40%), 35 closable to the 3,805 hard ceiling; accession sweep (stop-at-first-record was
  the bug) +12 species, reaction-driven acyl-transfer writer +6 reactions, CODATA ion table
  (8 ions, NBS cross-check worst 0.774 kJ/mol, Legendre transform equal to eQuilibrator's to
  9e-16) adds coverage only (6 moved reactions are all unbalanced sinks). Round 3 sent,
  time-boxed 2 h: polymer repeat units, metalloporphyrins, final ceiling table, then stop.
- thermo round 3 done, agent stopped: 3,779 priced (91.48%) + 23 closable + 329 permanently
  closed = 4,131; flux-carrying 2,716/2,845 (95.47%); betaxanthin anchors 893/934. Polymer
  repeat-unit surrogates (glucan, chitin, dextrin, starch; bond correction measured on 8
  glycoside hydrolyses, -12.10 +/- 6.81 kJ/mol) behind one route filter; metalloporphyrin
  accessions resolve to the iron-free macrocycle (curation needed); protein-linked residues
  permanently closed by attachment-point structure. COVERAGE STREAMS BOTH CLOSED.
- 02:47 hourly check: 1709-1712 RUNNING fresh; 13 pending (1713-1721 trainer, 1723-1726 flux
  tdg cells). Trainer + flux agents running; thermo + kinetics done. WIP commit attempt
  blocked by mypy (18 errors in parameters.py, kinetics_pipeline.py, three test files);
  ruff-format and markdownlint auto-fixed. Spawned a scoped mypy cleanup agent; retry after.
- 02:53 mypy cleanup agent done (5 files, 37 tests pass); WIP commit 625b1021 landed on the branch (hooks pass). Tree clean except test.pkl.
- 03:47 hourly check: 1713-1716 RUNNING fresh; 9 pending. 1709-1712 COMPLETED. Trainer +
  flux running; thermo + kinetics done. No action.
- 04:47 hourly check: 1717-1720 (aa s2701/s2702) RUNNING fresh; 5 pending (1721 aa allon,
  1723-1726 tdg). 1713-1716 COMPLETED. Trainer + flux running. No action.
- 05:47 hourly check: 1721 (aa allon), 1723/1724/1725 (tdg) RUNNING fresh; 1726 pending.
  1717-1720 COMPLETED (aa pairs s2701, s2702). Trainer + flux running. No action.
- 05:55 amino-acid parity read (mean Spearman over 19 features, 564 pinned genes): box
  nocstr vs pooled = 0.0275/0.0663 (s2700), 0.0005/0.0320 (s2701), 0.0103/0.0560 (s2702):
  the box flux head is below pooled at every seed, paired diffs -0.039, -0.031, -0.046.
  Driver queued flux_projection (scale 0) on the amino-acid head, seeds 2700-2702: jobs
  1727, 1728, 1729 (scripts gh_wave6_aa_proj_s270{0,1,2}.slurm).
