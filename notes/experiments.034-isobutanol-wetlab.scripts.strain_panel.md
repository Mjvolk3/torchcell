---
id: vzug90p4bm0v92bc7156u3l
title: Strain_panel
desc: ''
updated: 1790626258512
created: 1790626258512
---

## 2026.09.28 - What This Script Holds

The eleven strains as pydantic records, each with the verbatim quote that places
it. Emits three LaTeX tables into `notes-tex/034-isobutanol-wetlab/tables/` and
one CSV into `experiments/034-isobutanol-wetlab/results/strain_panel.csv`.

```bash
~/miniconda3/envs/torchcell/bin/python \
    experiments/034-isobutanol-wetlab/scripts/strain_panel.py
```

Outputs:

- `tables/t1-strain-panel.tex` - the eleven tubes, what each is, and its titer
- `tables/t2-one-step-away.tex` - four strains the paper reports that are one
  construction step past a tube in the freezer
- `tables/t3-genotypes.tex` - Supplementary Table 1 genotypes for the eleven
- `results/strain_panel.csv` - the same records with quotes and locators

Two conventions worth knowing before editing it.

**Greek and delta forms are written ASCII in the records.** `bat1D`, `LEU4DS547`,
`d-integration`, `alpha-KIV`. `_tex_escape` turns them into symbols on the way
into LaTeX. The records stay diffable and greppable that way, and the SI writes
them the same way.

**A derived titer is flagged, and there are two.** `Titer.derived` is True for
C0047 (201 mg/L, back-computed as 963 / 4.8) and C0050 (73 mg/L, 227 / 3.1).
Both print with a dagger. Every other titer is printed by the paper.

Three titers that look like they belong to a tube but do not, and are therefore
stored as `None`:

- **C0052 YZy91.** The 378 +/- 10 mg/L of Fig. 3b is YZy91 carrying the
  ILV6 V110E plasmid, a different strain. The host alone has no reported titer.
- **C0053 YZy502.** Fig. 6b plots it but prints no number, and the only ratio
  stated (20-fold) is YZy505 over YZy480, not over YZy502.
- **C0046 YZy148.** It is leu4 delta leu9 delta with no LEU4 plasmid.

See [[experiments.034-isobutanol-wetlab.strain-panel]] for what the panel means.
