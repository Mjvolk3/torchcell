# Review round 1 -- dispositions

Comments pulled from `microbe-perturb-seq_2026-08-21-02-08-50_3e42a6a3.pdf` with
`notes-tex/common/zotero_comments.py`. Keys are Zotero annotation ids and are stable,
so a follow-up round can refer to them directly.

**84 annotations** -- 65 addressed, 3 partial, 9 deferred, 7 needed no action.

Deferred items are almost all draw.io redraws of Figs. 1-2 (panel additions, line
weights, arrows), which are better done in one pass than piecemeal. They are listed
here so none is lost.

| # | Key | p | Status | Disposition |
|---|---|---|---|---|
| 1 | `EWVTJ8GS` | 1 | **done** | Abstract rewritten: 'map perturbations to their transcriptional consequences at dense, transcriptome-wide resolution'. |
| 2 | `DHB3N7P7` | 1 | **done** | 'This document' removed throughout; the abstract and Sec. 1 now state what was established. |
| 3 | `LIRJLFPI` | 1 | **done** | Abstract states both purposes -- method selection AND an explainer whose passages can be reused in a methods write-up. |
| 4 | `XWRVICXE` | 1 | **done** | Collaborator sentence deleted. |
| 5 | `HVCFMPEF` | 1 | **done** | Provenance flags now carry coloured symbols matching the section stoplight, and both are listed in the page-1 key. tcdoc.sty. |
| 6 | `VWHX9DK7` | 1 | **done** | Working notes / scripts / references now on three separate lines. |
| 7 | `9IG2PHUC` | 2 | **done** | Self-reference deleted; the sentence now states the property of the data, not who wants it. Swept the whole document for 'for us' / 'our goal'. |
| 8 | `P8WDHNNE` | 2 | **done** | Qualified: the fitness-neutral advantage is a property of the single-cell measurement and weakens where clonal populations are used to accumulate transcripts. Cross-referenced to Sec. 4.3. |
| 9 | `ZFV4NC65` | 2 | **done** | 'Two developments make' -> 'Three developments have made'. |
| 10 | `C7Q4GHEA` | 2 | **done** | Now reads 'joined in bacteria, the first demonstration in any microbe'. |
| 11 | `23II5BMM` | 2 | **done** | 'This document asks...' -> 'What follows establishes...'. |
| 12 | `6H2URT7R` | 2 | **done** | American spelling applied across prose, tables and the draw.io figures (54 replacements; verbatim source quotes deliberately left as published). |
| 13 | `GLAGGNB2` | 2 | **done** | Glossary is now one flat ALPHABETICAL list with no group-heading rows, which is what made the first definition under each heading look shifted, plus 2.5 pt between rows. |
| 14 | `MC7W58ZV` | 2 | **done** | Same fix as [13]. |
| 15 | `EKVGKWMJ` | 4 | **done** | Rewritten to 'which isolates every molecule that cell releases'. The pattern you are objecting to is a TRAILING RESTATEMENT -- a clause appended to re-say what the sentence already said, in different words ('...because the droplet wall keeps it that way'). Swept for it. |
| 16 | `448C7Q46` | 4 | **done** | Sec. 2.2 now explains the channel concretely: a chip carries several, each priced separately, each recovering up to ~20,000 cells, so two 20,000-cell runs are two channels. 'Channel' added to the glossary; links to Sec. 5.5. |
| 17 | `WHM2QSLN` | 4 | **done** | Spelling. |
| 18 | `W6GVE5CZ` | 4 | **done** | Nature sets in vivo / in vitro / in situ ROMAN, not italic. Kept roman; now consistent. |
| 19 | `PDY6MN29` | 4 | **done** | Sec. 2.2 expands the per-well argument and links to Sec. 5.1 and 5.2. |
| 20 | `RZMGG29X` | 4 | **done** | Spelling. |
| 21 | `C27TSHR5` | 4 | **done** | Rewritten: which oligo carries the barcode fixes WHERE ON THE TRANSCRIPT it lands -- oligo-dT at 3', TSO at 5', Tn5 wherever it cuts. |
| 22 | `I73T23DN` | 4 | **done** | Clarified that this is the plate route specifically; droplets are covered separately. |
| 23 | `YGP6ZSP8` | 4 | **done** | Now states explicitly that the sample index is a sample tag, NOT a UMI, and that a plate method still needs its own UMI. |
| 24 | `5H39TZXX` | 5 | **done** | Spelling, including inside the draw.io sources. |
| 25 | `ZLDZV4NT` | 5 | **noop** | Highlight with no comment. |
| 26 | `V6HK2TF7` | 5 | **done** | Built the mechanism you asked for: notes-tex/common/figure_provenance.py (reusable model + check) and experiments/019-perturb-seq-costing/scripts/figure_sources.py (the records), rendered as Table 15 with citekey, quote and line for all 13 hand-drawn numbers. It immediately caught three bad numbers in the Fig. 1 throughput band -- see [27]. |
| 27 | `T243EKL5` | 5 | **partial** | SPC added to the controlled vocabulary as a fourth isolation principle. Adding it to the figure as a column is a redraw and is deferred to the next round; no yeast SPC study exists yet, so it would enter as a capability, not a data point. |
| 28 | `WGAYNQ29` | 5 | **noop** | Highlight with no comment. |
| 29 | `9M8A9E2M` | 5 | **deferred** | Panel b needs a lead-in on the central challenge. Deferred -- it is a redraw, and it should be done together with [35] and [33]. |
| 30 | `QWWP4QMZ` | 5 | **done** | TSO added to the vocabulary, and the whole table is now alphabetical. |
| 31 | `AGPUY482` | 5 | **done** | 'Mirror of the above' removed from the figure. |
| 32 | `RAWDV3CE` | 5 | **done** | Tagmentation added to the vocabulary, with Tn5 explained inside it. |
| 33 | `K5B6EVQN` | 5 | **partial** | Sec. 2.2 and the new TSO entry now explain that 3'/5' refers to the TRANSCRIPT end the barcode sits beside. The DNA-vs-RNA orientation question is answered in the TSO entry; a figure annotation is deferred with [29]. |
| 34 | `VD79LH6T` | 5 | **done** | Answered in Sec. 2.2 -- the index is a sample tag, a UMI separates molecules within a cell. |
| 35 | `TMI5RHF9` | 5 | **deferred** | Panel d flow chart on when pooling and sequencing happen. Deferred to the next round with [29]. |
| 36 | `CSXID4DM` | 5 | **done** | Yes -- the band refers to the dashed boxes. Caption reworded to say so. |
| 37 | `I87XKAZE` | 5 | **done** | Tn5 is a transposase that cuts and ligates adapters in one step; it is drawn as an enzyme body because it is a protein acting on the molecule, not part of it. In the vocabulary under Tagmentation. |
| 38 | `QUHQR88M` | 5 | **done** | 'Drawn at true Nature print size in <path>' removed from all three captions; they now point at Table 15. |
| 39 | `KA7WW7XC` | 6 | **done** | You were right that it was overstated. Rewritten: digestion is not optional, but its TIMING decides whether the cell survives to be isolated. |
| 40 | `STZUSVM9` | 6 | **done** | Sec. 2.3 now says it plainly -- cells carrying one perturbation are biological replicates of one genotype, the single-cell step does assignment and the pooling does the measurement. |
| 41 | `GBPN2LB3` | 6 | **done** | Answered in Sec. 2.3: rRNA is NOT polyadenylated, so an oligo-dT primer never primes on it and no depletion step is needed. The 93.7% rRNA is caused by Brettner's added random hexamers, not by 3' capture. |
| 42 | `AWGGFPTA` | 6 | **done** | L is now defined as the L distinct perturbations in the library -- one per guide, or one per barcoded deletion strain -- with Z the assignment and Sec. 3.5 named as the problem of recovering it. |
| 43 | `27ES9U9Q` | 6 | **noop** | Resolved, no action. |
| 44 | `73CZND9Z` | 6 | **done** | 'Two families are live options for us' -> 'Two method families are candidates'. |
| 45 | `AS7AXQW8` | 6 | **done** | Added a paragraph answering it directly: precision depends on cells x depth, so 5x less depth is offset by 5x more cells; what breaks the symmetry is the 100-cell biological floor and the fact that rRNA depletion raises usable depth without buying cells. |
| 46 | `2QLFMKFH` | 6 | **done** | Now gives per-run figures: 48 wells x ~5,000 cells = ~240,000 barcoded per run, split into ten sublibraries, two sequenced returning ~5,500 and ~10,000 -- directly comparable with one Chromium channel's ~20,000. |
| 47 | `7J9WLL2R` | 6 | **done** | 'would measure artifacts of its own sample preparation'. |
| 48 | `PZ9PW9ZC` | 6 | **done** | 'Fixing first is what isolates the perturbation's signal from the handling'. |
| 49 | `LA8MI399` | 6 | **done** | Yes -- answered explicitly: sixteen conditions are sixteen wells of one split-pool run, but sixteen channels and sixteen library preps for droplet. |
| 50 | `JEMVWVBI` | 7 | **done** | Both answered. 5,000 falls out of the recipe (25 uL at 200,000 cells/mL); the 48 is unexplained in the paper, and it has a consequence worth carrying -- the realized barcode space for that run was 48x96x96 = 442,368, half the protocol's 96^3, so its collision rate is double the figure in Sec. 5.4. |
| 51 | `JZYEMDII` | 7 | **done** | No. Both round-1 primers carry BC1; the UMI arrives on the round-3 oligo. Stated explicitly. |
| 52 | `FAVECBI2` | 7 | **done** | Explained honestly: PEG works by excluded volume, raising effective local concentration, NOT by improving mixing -- and viscosity does get worse. Noted that the 7.5% is inherited from mammalian SPLiT-seq without justification, so it is a titration candidate. |
| 53 | `FVQGLZJC` | 7 | **done** | It is a lever because it is a RATIO, not a requirement. Brettner et al. say so themselves; their sentence is now quoted. |
| 54 | `DS4LDWHR` | 7 | **done** | Correct -- the hexamer primes anywhere and is not a UMI. Template switching IS used, but later, on the beads after lysis; that is now described in step 5. |
| 55 | `RVSH4BHE` | 7 | **noop** | Agreement. |
| 56 | `NKI3UYEI` | 7 | **done** | Answered in two places: the 5' coverage is what the hexamer buys, and it is unused in a 3'-counting screen; and the rRNA/poly(A) question is answered in Sec. 2.3. |
| 57 | `NBRLU7UK` | 7 | **done** | Filtration removes clumps, not contaminants -- two stuck cells take one path and read as one cell, the split-pool equivalent of a doublet. |
| 58 | `LMXVFSN4` | 7 | **done** | 'Splint' is now a vocabulary entry, defined as the same molecule as the linker and contrasted with a template-switching oligo. |
| 59 | `IB3JZ7KM` | 7 | **deferred** | Carry-over between rounds. Neither Brettner nor Gaisser reports a measurement of it; the blocking strand is the control. Flagged for the pilot rather than answered from literature. |
| 60 | `2PC486GK` | 7 | **done** | Separated the two ideas. Unique dual indices are mandatory because an index hop manufactures a barcode collision (Sec. 5.4); the 94% rRNA / 75% discarded figures are a capture-chemistry issue and are treated in Sec. 3.4. |
| 61 | `5WRWXBNV` | 7 | **done** | 'barcode outboard of the last' rewritten as 'each round ligates its barcode onto the free 5' end, so later barcodes end up further from the cDNA'. |
| 62 | `V3K7XK9G` | 7 | **deferred** | Read-direction arrow inside the figure. Deferred with the other figure work; the prose now states the order and why the reverse is impossible. |
| 63 | `SUNKS8AQ` | 7 | **done** | Trimmed, and now carries the Gaisser citation. |
| 64 | `5B74I6PG` | 8 | **done** | Repo path removed from the figure subtitle; provenance lives in Table 15. |
| 65 | `TIUC2EKG` | 8 | **deferred** | Duplex line-weight convention. Deferred -- it is a redraw. |
| 66 | `69P2UN4H` | 8 | **done** | Spelling. |
| 67 | `U2AECUSJ` | 8 | **deferred** | Grey vs black 3'/5' standardisation. Deferred with [65]. |
| 68 | `NL5YZP33` | 8 | **noop** | Highlight with no comment. |
| 69 | `4DSKKVRE` | 8 | **deferred** | Overhang PCR annotation. Deferred with [65]. |
| 70 | `AN6SYIVR` | 8 | **deferred** | Label BC1 in parentheses on the hexamer. Deferred with [65]; the prose now states it. |
| 71 | `FXXHT8UR` | 8 | **partial** | The two-path point (dT->polyA mRNA, hexamer->everything incl. rRNA) is now explicit in Sec. 2.3 and Sec. 3.1. Drawing both paths is deferred. Your question about barcode length: BC1 does not have to be a hexamer -- the hexamer is the PRIMING sequence and the barcode is separate. |
| 72 | `7XAKXP9U` | 8 | **done** | Yes. 'Splint' and 'linker strand' are the same molecule; now one vocabulary entry saying so. |
| 73 | `LJYDA7CI` | 8 | **deferred** | Blocking-strand placement in the figure. Deferred with [65]. |
| 74 | `6AJKBQCT` | 8 | **done** | Caret fixed -- 4^10 now renders as a real superscript, matching the 96 and 10 exponents beside it. |
| 75 | `9AF9FSEQ` | 8 | **done** | Literal '--' replaced with a real en dash in all three draw.io files. House style is en dash, not em. |
| 76 | `DSWY46K5` | 8 | **done** | Answered properly in a new subsection. Two reasons: it is a FOURTH BARCODE (ten sublibraries multiply the address space by ten without another ligation), and it is the unit of sequencing (lysed, prepped, indexed and loaded independently, so you can sequence two of ten and freeze the rest). Your guess was reason two. |
| 77 | `4F9K9HL2` | 8 | **done** | New paragraph on template switching: RT adds non-templated Cs, the TSO's rGrGrG pairs with them, the polymerase copies the TSO -- so handle R2 is WRITTEN by the polymerase, not ligated. Also in the vocabulary. |
| 78 | `D7PMC5WP` | 8 | **done** | No -- the i7 index is added at index PCR and is a separate sequence from either PCR handle, read on its own index read. Stated in step 5. |
| 79 | `AC94DNVR` | 8 | **done** | Same fix as [76]. |
| 80 | `M8FLFP4H` | 8 | **done** | 'instalments' -> 'installments'. |
| 81 | `TUSQH8FU` | 8 | **noop** | Agreement. |
| 82 | `FHGQAC3B` | 8 | **noop** | Agreement. |
| 83 | `A3I5W3LE` | 8 | **done** | Distinguished in the vocabulary: a splint holds two ends for a LIGASE; a TSO is copied by a POLYMERASE. Different mechanisms. |
| 84 | `3Y32N8CD` | 8 | **done** | Print-size note removed; the draw.io path is retained in Table 15 rather than on the figure. |
