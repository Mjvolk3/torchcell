# Review round 2 -- dispositions

Comments pulled from `microbe-perturb-seq_2026-08-24-14-30-16_fad7a3dc.pdf` (published
version 14) with `notes-tex/common/zotero_comments.py`. Keys are Zotero annotation ids and
are stable, so a follow-up round can refer to them directly.

**45 annotations, all carrying a written comment** -- 42 addressed, 3 needed no action.

All 45 fall on pages 3--6, which is the introduction plus the controlled vocabulary. That
concentration is itself the finding of the round: the argument sections were read and left
alone, and what did not survive contact with a reader was the glossary. Round 1 built it as
a lookup table; round 2 says a lookup table still has to explain, and about half the
comments below are a definition that named a thing without saying why it is that way.

Three changes are structural rather than per-comment, and each closes a class:

- **Notation is now fixed rather than warned about.** `index` means the Illumina index read
  and nothing else; budgets are quoted in cells per target gene and never cells per guide;
  `q` is the per-guide detection probability, so Yao et al.'s sparsity `q` is renamed `nu`
  with their notation recorded where it is defined. Stated in Sec. 2.1 and enforced by the
  glossary being the single source of each. [9], [28]
- **Glossary entries that assert a number now carry a citation** into the table itself,
  rendered from the `citation_key` field that already existed and was never emitted. [12]
- **American spelling is checked as a class.** `check_doc.py` held a flat word list that
  had `permeabilised` but not `permeabilises`, `optimise` but not `factorise`. It is now
  stem-plus-endings for `-ise`/`-yse` and explicit words for `-our`, `-re`, doubled-l and
  the one-offs. [17], [23]

| # | Key | p | Status | Disposition |
|---|---|---|---|---|
| 1 | `V26CUCTW` | 3 | **done** | MAGIC cited in the opening paragraph, and used to make the point rather than only to attribute: the same library returns a fitness list today and would return the effect matrix read out by Perturb-seq. |
| 2 | `73ZEVI2I` | 3 | **done** | The sentence was muddled, not just unclear. It attributed the fitness-neutral advantage to single-cell resolution, which is wrong: that advantage comes from the transcriptional readout. Rewritten as three separate claims -- transcriptional readout scores what growth cannot, single-cell resolution makes the library poolable, the estimate is pseudobulk. |
| 3 | `MIFGPRDI` | 3 | **done** | Verified against the mirror, and the claim does not survive. Brandner et al. scope it to bacteria twice ("single-cell CRISPR screens have not yet been implemented in bacteria to date", paper.md:39). Two microbial pooled screens read out by scRNA-seq predate it: Jackson et al. 2020 (38,285 yeast cells, barcoded deletions) and Nadal-Ribelles et al. 2025 -- the latter cited in the *same sentence*, so it contradicted itself. Now "the first single-cell CRISPR screen in that domain", and CRISPRa/i rather than CRISPRi. |
| 4 | `B2AKTGDP` | 3 | **done** | Cut. |
| 5 | `23E4HVDZ` | 3 | **done** | Rewritten as "Two further questions decide what gets run first, and a cost table answers neither", which says what the sentence was trying to. |
| 6 | `SDINJ85J` | 3 | **done** | Agreed, and the claim was overreaching. Now: quadratic against linear is an argument for spending on environments, explicitly not a verdict, because a gene combination and a growth condition are different kinds of information and a model needs both. |
| 7 | `FY9IB4Y4` | 3 | **done** | Corrected in the introduction and in Sec. 6.2. The distinction the text was missing is published against available: no dCas9 CRISPRi is *published* for this host, but a working system and a JGI assembly are held here. Both carry `\external{in-house, unpublished}`, since an unpublished in-house capability is not a citable one. |
| 8 | `3XY494DT` | 3 | **done** | The rationale is now footing rather than absence: nobody here has run the assay on a walled microbe, so the first run is spent learning it, which is the collaborator's recommendation and is also what the arithmetic supports. The waste hedge is in as well -- spread the run across environments rather than adding guides. Sec. 6.2 keybox and the staging plan both updated. |
| 9 | `S342RPQY` | 3 | **done** | Standardized, see the notation note above. Sec. 2.1 now states the three conventions as a list before the prose. |
| 10 | `QAGS7GWR` | 3 | **done** | Yes, and the glossary now says what it costs an analysis rather than only that it is violated: main effects pooled over multiplexed cells are biased toward zero as k rises, so the 1/k saving is optimistic by whatever the buffering is at that k, unmeasured past k=2. Sec. 4.5 carries the same qualification at the point the 1/k saving is derived. |
| 11 | `HGMTVZZE` | 3 | **done** | The objection is right and "independent of sequencing" was wrong. The floor is not independent at any real depth; it is what the requirement *converges to* as depth grows, the phi term of Eq. (5) with shot noise driven to zero. Rewritten to say that. |
| 12 | `GFN6T74E` | 3 | **done** | Brandner et al. is the source and now renders as a citation in the table. Built the general mechanism rather than patching one row: `rendered_definition()` appends `\citep` for any entry with a `citation_key`, so seven entries now carry one. Code generation was not the obstacle it looked like. |
| 13 | `Y89P3KNE` | 3 | **done** | Because the name records a comparison, not a quantity. Pure counting pins the variance at mu, so a gene varying more than that is dispersed *over* what the counting model allows and phi measures the excess. Now in the glossary and in Sec. 4.1. |
| 14 | `CKQHCDG4` | 3 | **done** | It is a cost advantage, not a throughput one, and the two were running together. Added to the Channel entry and to Sec. 2.2: what caps a split-pool run is cells per RT well (~5,000 measured) and barcode collisions, both of which bind well before price does. |
| 15 | `RE6ZVVCW` | 4 | **done** | p_j was used and never defined. Added as its own entry (transcriptome share, the fraction of a cell's mRNA belonging to gene j), and the depth-sufficiency entry now names both symbols in place. |
| 16 | `ZYSAPQUL` | 4 | **done** | Doublet now covers all three regimes. Under scifi several cells per droplet is the operating point, so a doublet is the narrower event of two cells sharing both round1 well and round2 droplet, which is why overloading stops being ruinous. |
| 17 | `RHSEL2VE` | 4 | **done** | See the spelling note above. |
| 18 | `RYM8WBGW` | 4 | **done** | Yes, on the plasmid. The entry now says it rides on the same plasmid as the guide, transcribed from a separate Pol II cassette, and names the design that makes the barcode be the guide's own spacer. |
| 19 | `CGSX4N2X` | 4 | **done** | It does assume it, and the entry now says so as a property of the definition: pooling over every background is exactly the operation that integrates epistasis away, so a main-effect estimate cannot report an interaction. Sec. 4.5 says the same where the estimand is introduced. |
| 20 | `XJ6MQGAS` | 5 | **done** | Not bacteria-specific. The cause is the read structure, not the organism: read 2 carries fixed linker at defined cycles whatever cell the library came from. Stated explicitly. |
| 21 | `RMTCXX6W` | 5 | **done** | Under the additive model, yes, and the entry now leads with that. Sec. 4.5 adds two paragraphs on what the division rests on. |
| 22 | `D6D2J5DZ` | 5 | **done** | Cut roughly in half and pointed at Sec. 3.3. |
| 23 | `K2CYRLRJ` | 5 | **done** | American spelling by default, everywhere. Already the rule in CLAUDE.md and `notes/writing-style-guide.md`; what was missing was enforcement, which is the checker rewrite above. |
| 24 | `V2MUICZS` | 5 | **done** | Now a range with its cause: capacity depends on how much of the round-1 plate is loaded, 240,000 at Brettner's 48 wells and 480,000 at a full plate. |
| 25 | `8A8LRA5R` | 5 | **done** | Across multiplexed cells. Every cell carrying a guide against that gene whatever else it carries, which is what makes the estimand the main effect. |
| 26 | `XQVEZK55` | 5 | **noop** | Agreement, no change requested. It is the reason Sec. 4.9 sweeps phi rather than fixing it. |
| 27 | `MFX6Z4JW` | 5 | **done** | Answered by the cells-per-target-gene convention, now stated in Sec. 2.1 and in the entry: cells carrying a guide against that gene, pooled over all six of its guides, regardless of plasmid. |
| 28 | `5HV8V74I` | 5 | **done** | Renamed rather than warned about, see the notation note. Yao et al.'s notation is recorded in the glossary entry, in Sec. 4.7 and in the Table 16 caption. |
| 29 | `FVHK2NA7` | 5 | **done** | "the unit every sequencing price in this document is quoted in". |
| 30 | `RN7N2FGQ` | 5 | **done** | Rewritten twice over, in the glossary and in the new Sec. 4.1. mu is a count of molecules of one gene pooled over the cells sharing a perturbation. Two concrete points added, 25% error at 16 UMIs and 5% at 400, and the reason more reads help is now stated: more reads is a larger draw. |
| 31 | `5YUSN46E` | 5 | **done** | Built the scaling argument as a new script function, `derived_values.plate_set_lifetime()`, and a subsection in Sec. 5.1. One plate set is 103 million cells barcoded, about 16 genome-scale screens. The freeze--thaw objection resolves by aliquoting rather than by a shorter plate life: split into w working plates in one thaw and each sees 215/w cycles, so a tolerance of f cycles needs ceil(215/f) plates, 22 at f=10. Flagged that f itself is not in the mirror. |
| 32 | `UM3ZXELV` | 5 | **noop** | Acknowledgement. |
| 33 | `QRC7W7HN` | 5 | **done** | Not the same constraint, and the entry now says so: a sublibrary is an Illumina library, not a droplet channel, bounded below by the collision target and above by PCR complexity, and several are routinely pooled onto one flow cell. |
| 34 | `NR2QZGDS` | 5 | **done** | Rewritten from the ground up in both places, and the missing step was why a sublibrary exists at all. Sec. 2.1 now builds it: a flow cell delivers more reads than one library needs, so indices make a lane divisible; split-pool then reuses that same read as a fourth barcode dimension because the split happens after the last in-cell round. Unique dual indices follow from that rather than being asserted. |
| 35 | `2QUNPFXR` | 5 | **done** | Now cross-references Figs. 2 and 4 by name. |
| 36 | `HHMMBL8B` | 6 | **done** | A real error, and the entry said the wrong thing. Template switching is not always bead-borne: in droplet 5' chemistry the oligo is on the bead and carries the cell barcode, while in split-pool it is added in bulk after lysis and carries no barcode, because the barcode was written three rounds earlier. |
| 37 | `NVMN5MYT` | 6 | **noop** | Acknowledgement. |
| 38 | `WS6NIN6K` | 6 | **done** | Not overlap extension, and saying so explicitly is the fix. It happens within the single RT reaction with no melt and no re-anneal: the enzyme adds the Cs, the TSO base-pairs with them while the enzyme is still engaged, and the enzyme carries on copying the TSO. |
| 39 | `ZJQEUGGR` | 6 | **done** | The arithmetic is now in the entry rather than only in Sec. 5.3: read 2 spends 76 cycles on the composite barcode, three 8 nt barcodes are 24, linkers are the other 52, 52/76 = 68%. The linkers are read because they sit between the barcodes and the sequencer reads through in order. |
| 40 | `Z93IDJ37` | 6 | **done** | QC named concretely: enough UMIs to be a cell rather than ambient RNA, a barcode resolving to exactly one cell, and an identified perturbation. |
| 41 | `5GHTPLPG` | 6 | **done** | By the UMI, and the entry now explains why that works: the UMI is attached before amplification, so every PCR copy inherits it and reads sharing CB, gene and UMI collapse to one count. |
| 42 | `Q7I8ZA85` | 6 | **done** | Per gene is correct, and the entry now says why rather than only asserting it: collapsing keys on CB, gene and UMI together, so two molecules of different genes may share a UMI freely. The comparison is therefore against molecules of one gene in one cell, not against the transcriptome. |
| 43 | `VSGTTCLU` | 6 | **done** | Same rewrite as [34]. |
| 44 | `SB6YSEF6` | 6 | **done** | Cut. The perturbation identifier is now defined as "which perturbation the cell was carrying". |
| 45 | `UAGRWYPN` | 6 | **done** | gRNA throughout, with an entry defining it. `sgRNA` is left standing only inside quotations and vendor documentation, where changing it would falsify the quote. |

## Also in this round, not from a comment

- **Renamed `experiments/019-perturb-seq-costing` to `024`.** The 019 prefix collided with two
  unrelated experiments (`019-simb-multimodal`, `019-echo-crispr-array`) and shared nothing
  with either. Renamed across the experiment directory, `notes/assets/images/`, four Dendron
  notes, every script header, every `%% SOURCE:` line and three draw.io sources.
- **Moved the Poisson primer from Sec. 7.1 to a new Sec. 4.1.** The figure was the last thing
  in the document while the distribution it explains is used in three places, the first of
  which (droplet loading, Sec. 2.2) is 30 pages earlier. Sec. 4.1 now introduces Poisson
  counting once and names all three appearances; Sec. 7.1 is a back-reference. Figure
  numbering shifts by one from Fig. 7 onward, which touches no hardcoded cross-reference
  because the draw.io canvases only refer to Figs. 1--6.
- **Moved the "three properties of a yeast cell" keybox from the head of Sec. 3 to the end of
  Sec. 2.3.** It summarized Sec. 2.3 and was the only keybox in the document whose evidence
  sat four pages behind it.
- **Rebuilt Fig. 7 (the Poisson primer) to the house bar style.** It was the one plot in the
  experiment folder filling bar faces with the pale `PLOT_PALETTE_FILL` colors and drawing
  borders in the line color, which is backwards: line colors are the bar faces and edges are
  black. Also widened from `wide` (118.9 mm) to `full` (179 mm), which is what fixed the
  panel (c) letter being cropped and the panel (b) legend sitting on its own bars, and
  rewrote the mean annotation, which had been rendering as a stray tick between two rules.
- **Retired `GROUPS` / `GROUP_BOUNDS` in `glossary.py`.** They indexed `TERMS` by position, so
  every term added shifted the boundaries silently. Nothing read them.
