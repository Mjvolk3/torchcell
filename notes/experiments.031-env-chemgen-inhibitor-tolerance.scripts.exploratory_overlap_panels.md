---
id: qeo8jau3a6w4486gspju6o4
title: Exploratory_overlap_panels
desc: ''
updated: 1790492159038
created: 1790492159038
---

## 2026.09.27 - Where the five datasets overlap on genotype, chemistry and dose

Eight panels, one figure, all from the served records and the precomputed embeddings.

**Genes overlap heavily, compounds barely.** 98 to 100 percent of Vanacloig's 3,598 genes appear
in each genome-wide partner, and at most 24 percent of its 41 compounds appear in any one of
them. Wildenhain is the exception both ways: 242 genes, 4 percent of the other gene universes,
and 5,170 compounds, more than all the others combined.

**Chemical type.** The partner libraries sit at 20 to 40 heavy atoms and are drug-like; the
hydrolysate panel sits at 3 to 15, in the region where a substructure fingerprint carries almost
no bits. Molecular weight against calculated logP separates them visually. Against the partner
union the Vanacloig panel's median best non-exact Tanimoto is 0.500, with 20 compounds above 0.5
and 12 exact matches, and the compounds with no neighbor are the small ones.

**Concentration ranges.** Hillenmeyer HOM, HET and Hoepfner span roughly nine orders of magnitude
in molar; Wildenhain is a single fixed 20 uM point with zero variance; Vanacloig has no
convertible dose at all. Share of records stating a molar-convertible dose: 84, 96, 100, 100 and
0 percent.

**Shape.** Hoepfner and Hillenmeyer HET are wide in genes and narrow in compounds (about 5,800 x
150 to 290); Wildenhain is the mirror image (242 x 5,170); Vanacloig is the smallest on both
axes. Perturbation class is a clean split: Vanacloig is a 4x deletion on a sensitized host,
Hillenmeyer HOM and Wildenhain are single KanMX deletions, Hillenmeyer HET is engineered copy
number, and Hoepfner carries both classes in one dataset.

**Two plotting lessons recorded here because they cost time.** `savefig_true_size_svg` rescales
the figure from 72 dpi to draw.io's 100 units per inch, so a PNG written AFTER it inherits the
rescaling and every axes shrinks against type still sized in points; write the raster first.
And `constrained_layout` pushes a grid off the canvas when a legend is anchored outside an axes,
so panel h's legend lives in the otherwise empty ninth panel.

Figure: `notes/assets/images/031-env-chemgen-inhibitor-tolerance/exploratory_overlap.svg`.
Result files: `results/chemical_property_table.csv`, `results/genotype_overlap.csv`. Rendered as
Figure 4 of `notes-tex/031-unified-representation`.
