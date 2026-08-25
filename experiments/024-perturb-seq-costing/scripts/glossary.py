# experiments/024-perturb-seq-costing/scripts/glossary.py
# [[experiments.024-perturb-seq-costing.scripts.glossary]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/glossary
"""Controlled vocabulary for the perturb-seq method review.

The document leans on a lot of jargon that is used precisely and that overlaps
confusingly with itself -- four different things are called a "barcode", two
different things are called an "index", and "cells per gene" means something
different from "cells per guide". This module is the single definition of each,
and the table it feeds (Sec. 2.1) is what the rest of the document points at.

Two rules that keep the glossary honest and are enforced by the emitter:

1. **Every term names the section that treats it in depth.** A glossary entry is
   a pointer, not a replacement for the argument. ``where`` is a LaTeX label,
   and a typo in one becomes an undefined reference at build time rather than a
   silent dead end.
2. **A definition that asserts a NUMBER carries provenance.** Most entries are
   definitional and need no source. The few that pin a quantity (UMI length,
   plate size, the collision formula) name where the number comes from, same as
   every other number in this experiment folder.

Ordering is ALPHABETICAL, and this reversed after the first review round. The
table was grouped by theme on the theory that a vocabulary section should teach
the scheme, with the four identifiers side by side. Two things went wrong.
Grouping needs subheading rows, and a subheading leaves the Definition column
blank, so the first entry under each heading reads as though its definition has
slipped a line -- flagged twice in review. And a reader who meets "tagmentation"
in a figure arrives here knowing the word and nothing else; they need lookup, not
pedagogy. Teaching the scheme is what Sec. 2.1's prose is for. So: one flat
alphabetical list, no subheadings, no blank Definition cells.

NOTATION -- made internally consistent in review round 2. Three collisions were
live in the first version, and the round-2 decision was to resolve each rather
than warn about it at every use. Where a source's own symbol is displaced, the
source's notation is recorded at the point of definition:

1. ``index`` means exactly one thing: the Illumina index read added at the final
   PCR. Anything written INTO a cell is a barcode. The field's method name
   "combinatorial indexing" is the single exception and says so where it appears.
2. ``cells per target gene`` is the design unit everywhere. Cells per guide is
   six times smaller in a six-guides-per-gene library and is never quoted as a
   budget figure.
3. ``q`` is the per-guide detection probability and only that. Yao et al.\ write
   $q$ for non-zeros per module in the compressed-sensing bound; that quantity is
   $\\nu$ here.

CITATIONS. ``citation_key`` is rendered as a real ``\\citep`` at the end of the
definition, so a glossary entry that asserts a number carries its source in the
table rather than only in this file. That was a round-2 request and it is also
the rule the rest of the experiment folder already follows.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Term(BaseModel):
    """One vocabulary entry.

    ``abbrev`` is separate from ``term`` so the emitter can render it
    consistently rather than each entry inventing its own parenthetical style.
    """

    term: str
    abbrev: str | None = None
    definition: str
    # LaTeX label (without the \label{}) of the section that treats this term in
    # depth. Rendered as a cross-reference, so a wrong label fails the build.
    where: str
    # Populated only when the definition states a quantity. citation_key is a
    # Zotero citekey; line is the line in that paper's mirrored paper.md.
    citation_key: str | None = None
    line: int | None = None
    quote: str | None = None


# Comment banners below mark what each run of terms is about. They are comments
# and nothing more: rendering is alphabetical, and the GROUPS/GROUP_BOUNDS pair
# that used to sit here was retired in round 2. It indexed TERMS by position, so
# every term added shifted the boundaries silently -- a structure nothing read
# and that could only ever become wrong.
TERMS: list[Term] = [
    # --- Tags on a molecule ---------------------------------------------------
    Term(
        term="Cell barcode",
        abbrev="CB",
        definition=(
            "A sequence shared by every cDNA molecule from one cell and ideally "
            "unique to it; what makes the data single-cell rather than bulk. "
            "Delivered pre-synthesized on a bead in droplet methods, built up "
            "across rounds of well-specific ligation in split-pool."
        ),
        where="sec:identifiers",
    ),
    Term(
        term="Unique molecular identifier",
        abbrev="UMI",
        definition=(
            "A short random sequence attached to each cDNA molecule "
            "\\emph{before} amplification, so every PCR copy inherits it and no "
            "two original molecules are likely to share one. That is how a copy "
            "is told from an original: reads sharing CB, gene and UMI are copies "
            "of a single mRNA and collapse to one count, while reads sharing CB "
            "and gene but differing in UMI are separate molecules. 10\\,nt in the "
            "split-pool protocols here, 10--12\\,nt in droplet."
        ),
        where="sec:identifiers",
        citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
        line=47,
        quote="a 10-base unique molecular identifier (UMI)",
    ),
    Term(
        term="Sample index",
        definition=(
            "The Illumina index read added at final PCR. Its ordinary job is to "
            "let several libraries share one flow cell: each is tagged, sequenced "
            "together, and separated computationally afterwards, which is what "
            "makes a lane divisible between experiments. See \\emph{sublibrary "
            "index} for the extra job it does in split-pool."
        ),
        where="sec:sequencing",
    ),
    Term(
        term="Sublibrary index",
        definition=(
            "A sample index used as a barcode round. Split-pool divides cells "
            "into sublibraries \\emph{after} the last in-cell barcoding round, so "
            "two cells that took identical paths through the plates end up with "
            "identical BC1--BC2--BC3 and are separated only by which sublibrary "
            "they landed in. The index therefore multiplies the barcode space by "
            "the sublibrary count rather than merely naming a sample, and unique "
            "dual indices become mandatory: an index hop moves a cell into "
            "another sublibrary's namespace and manufactures a barcode collision "
            "rather than mere sample bleed."
        ),
        where="sec:collisions",
    ),
    Term(
        term="Guide barcode",
        abbrev="GBC",
        definition=(
            "An expressed, polyadenylated sequence that stands in for the gRNA "
            "itself. It rides on the same plasmid as the guide, transcribed from "
            "a separate Pol\\,II cassette, and the design that removes a whole "
            "class of bookkeeping error is to make the barcode \\emph{be} the "
            "guide's own 20\\,nt spacer placed in that cassette's 3$'$UTR. It is "
            "needed because a Pol\\,III guide cassette carries no poly(A) tail "
            "and is therefore invisible to 3$'$ capture."
        ),
        where="sec:guide-capture",
    ),
    Term(
        term="Barcode collision",
        definition=(
            "Two distinct cells receiving the same full barcode and being merged "
            "into one apparent cell. Rate is "
            "$1-((B-1)/B)^{C-1}$ for $C$ cells in a barcode space of size $B$, "
            "so it is controlled by adding rounds or sublibraries, not by "
            "sequencing harder."
        ),
        where="sec:collisions",
    ),
    Term(
        term="UMI collision",
        definition=(
            "Two distinct molecules of the same gene in the same cell drawing "
            "the same UMI, and being counted once. The comparison that decides "
            "whether this matters is \\emph{per gene per cell}, not against the "
            "whole transcriptome, because collapsing keys on CB, gene and UMI "
            "together -- two molecules of different genes may share a UMI freely. "
            "So a 10\\,nt UMI's $4^{10}\\approx10^{6}$ states are set against at "
            "most a few thousand molecules of any one gene, and the collision "
            "rate is negligible. Not to be confused with barcode collision, "
            "which is a cell-level error."
        ),
        where="sec:identifiers",
    ),
    # --- Cell isolation --------------------------------
    Term(
        term="Droplet isolation",
        definition=(
            "One cell co-encapsulated with one barcoded bead in an emulsion "
            "droplet; the droplet wall is what keeps a cell's molecules "
            "together. Reagent cost scales linearly with cells, because you pay "
            "per channel."
        ),
        where="sec:isolation",
    ),
    Term(
        term="Combinatorial split-pool barcoding",
        definition=(
            "No isolation at all. Fixed, permeabilized cells act as their own "
            "reaction vessels; the population is split across a plate, barcoded "
            "in situ, pooled, and re-split. After $R$ rounds of 96 wells there "
            "are $96^{R}$ possible paths. Cost scales per well, not per cell, "
            "so a plate costs the same for ten thousand cells or a million."
        ),
        where="sec:isolation",
    ),
    Term(
        term="Doublet",
        definition=(
            "Two cells reported as one, and each isolation principle fails into "
            "it differently. In plain droplet capture it is two cells in one "
            "partition, governed by Poisson loading (10x quote $\\sim$8\\% at "
            "20{,}000 cells recovered). In split-pool it is a barcode collision, "
            "two cells taking the same path through the plates. Under "
            "combinatorial fluidic indexing it is neither of those: several cells "
            "per droplet is the operating point, so a doublet is the narrower "
            "event of two cells sharing \\emph{both} their round1 well and their "
            "round2 droplet, which is why overloading stops being ruinous."
        ),
        where="sec:isolation",
    ),
    Term(
        term="Spheroplast",
        definition=(
            "A yeast cell whose $\\beta$-glucan wall has been digested away, "
            "here with zymolyase under osmotic support. Required because no "
            "commercial chemistry can get reagents through an intact fungal "
            "wall; the trade is that a wall-less cell bursts in a normal buffer."
        ),
        where="sec:why-hard",
    ),
    # --- Split-pool chemistry -------------------------------------------------
    Term(
        term="Linker strand",
        definition=(
            "A splint oligo pre-annealed to a barcode, positioning it against "
            "the 5$'$ end of the growing strand so T4 DNA ligase can seal the "
            "nick. Ligation is templated by the linker, which is why a stray "
            "barcode from an earlier well would corrupt the address."
        ),
        where="sec:brettner",
    ),
    Term(
        term="Blocking strand",
        definition=(
            "Added after each ligation round to outcompete the linker and retire "
            "that well's barcode, so it cannot participate in a later round. "
            "Without it, leftover oligos scramble the barcode sequence."
        ),
        where="sec:brettner",
    ),
    Term(
        term="Template switch",
        definition=(
            "The reaction that adds a common handle to the far end of the cDNA, "
            "giving the molecule PCR primer sites at both ends; everything after "
            "it is conventional Illumina library prep. \\textbf{A 3$'$ library "
            "does this too} -- it is how the second handle gets on, and a molecule "
            "handled at one end only cannot be amplified. What 3$'$ and 5$'$ name "
            "is which oligo carries the cell barcode: in 3$'$ the barcode is on "
            "the oligo-dT and the TSO is free and carries only the handle; in "
            "5$'$ they swap, which is what puts the barcode beside the "
            "protospacer and makes a Pol\\,III guide readable "
            "(\\cref{sec:guide-capture})."
        ),
        where="sec:brettner",
    ),
    Term(
        term="Source plate / working plate",
        definition=(
            "A source plate is the pre-diluted barcode-oligo stock as shipped; "
            "working plates are aliquots cycled in daily use. The distinction is "
            "economic: source plates are retired by freeze--thaw damage rather "
            "than by depletion, so they behave as a capital asset."
        ),
        where="sec:startup",
    ),
    Term(
        term="Protocol run",
        definition=(
            "One complete pass through the barcoding workflow -- fix, "
            "permeabilize, three rounds, split into sublibraries. Consumes one "
            "withdrawal from each barcode plate. Capacity is a range, not a "
            "number, because it is set by how much of the round-1 plate is "
            "loaded: Brettner et al.'s published run filled 48 of 96 wells at "
            "$\\sim$5{,}000 cells each and so carried $\\sim$240{,}000 cells, "
            "while a full plate at the same density carries $\\sim$480{,}000. A "
            "unit of \\emph{bench work}, not of sequencing: the sublibraries it "
            "yields are sequenced and priced separately."
        ),
        where="sec:startup",
        citation_key="brettnerUltraHighthroughputMassively2024",
    ),
    Term(
        term="Sublibrary",
        definition=(
            "An aliquot of barcoded cells (5{,}000--20{,}000) split off after the "
            "last in-cell round, lysed and prepared as one Illumina library. It "
            "is not a droplet channel and has no capacity limit of its own: a "
            "run is split into sublibraries because doing so buys barcode space "
            "(see \\emph{sublibrary index}) and because it is the granularity at "
            "which a prep can fail cheaply. The count is bounded below by the "
            "barcode-collision target and above by PCR complexity, and several "
            "sublibraries are routinely pooled onto one flow cell, where their "
            "sample indices separate them again."
        ),
        where="sec:startup",
    ),
    Term(
        term="rRNA depletion",
        definition=(
            "Removal of ribosomal cDNA from the amplified library, after "
            "barcoding and before fragmentation and indexing. It does not "
            "improve capture; it stops $\\sim$94\\% of purchased reads being "
            "spent on rRNA, and is the single highest-leverage change available "
            "to a split-pool screen."
        ),
        where="sec:rrna",
    ),
    # --- Screen design ---------------------------------------
    Term(
        term="Perturb-seq",
        definition=(
            "A pooled genetic screen read out by single-cell RNA-seq, so each "
            "cell yields both a perturbation identity and a whole-transcriptome "
            "phenotype. The transcriptome, not a growth rate, is the phenotype."
        ),
        where="sec:primer",
    ),
    Term(
        term="Plex",
        abbrev="$k$",
        definition=(
            "gRNAs delivered per cell, so $k$ genes are perturbed in that cell at "
            "once. \\emph{Under the additive model} raising $k$ divides the cell "
            "requirement for main effects by $k$, because each cell then reports "
            "on $k$ genes instead of one; the division is a property of that "
            "model, not of the biology, and it is what \\cref{sec:compression} "
            "tests. Raising $k$ does not make genome-wide pairwise interactions "
            "reachable either way."
        ),
        where="sec:multiplex",
    ),
    Term(
        term="Main effect",
        definition=(
            "The average effect of perturbing one gene, pooled over every cell "
            "carrying a guide against it regardless of what else that cell "
            "carries. The pooling is what makes it cheap and is also what makes "
            "it blind: averaging over every genetic background a cell might have "
            "is exactly the operation that integrates epistasis away, so a "
            "main-effect estimate cannot report an interaction and a screen "
            "powered only for main effects is not powered for one. Named "
            "interactions need cells carrying the specific combination, which is "
            "\\cref{eq:pairs}."
        ),
        where="sec:multiplex",
    ),
    Term(
        term="Guide-pooling vs cell-pooling",
        definition=(
            "Two ways to compress a screen. Guide-pooling puts several guides in "
            "one cell; cell-pooling combines separately perturbed cells before "
            "sequencing. Both rely on the perturbations composing additively, "
            "and both are decoded rather than read directly."
        ),
        where="sec:yao",
    ),
    Term(
        term="FR-Perturb",
        definition=(
            "The decoder for a compressed screen: factorizes the observed "
            "expression matrix by sparse PCA, then recovers per-perturbation "
            "effects by LASSO against the pooling design. The recovered effects "
            "are estimates, so power depends on the design matrix as much as on "
            "cell count."
        ),
        where="sec:yao",
    ),
    Term(
        term="Cells per target gene",
        definition=(
            "Cells carrying a guide against that gene, pooled over all of its "
            "guides, and the unit every budget here is quoted in. \\emph{Not} "
            "cells per guide and \\emph{not} cells per plasmid: at six guides per "
            "gene the two differ by a factor of six."
        ),
        where="sec:cells-per-pert",
    ),
    # --- Counting and cost ----------------------------------------------------
    Term(
        term="Shot noise",
        definition=(
            "The irreducible randomness of \\emph{counting}: a run reads a random "
            "sample of the library, so a gene whose expected count is $\\mu$ UMIs "
            "comes back scattered around $\\mu$ with variance also $\\mu$, hence "
            "relative error $1/\\sqrt{\\mu}$. Note that $\\mu$ counts molecules "
            "of one gene pooled over every cell sharing a perturbation, not "
            "molecules in one cell. The only noise buying more reads removes. "
            "Also called counting or sampling noise."
        ),
        where="sec:poisson",
    ),
    Term(
        term="Transcriptome share",
        abbrev="$p_j$",
        definition=(
            "The fraction of a cell's mRNA belonging to gene $j$, so a cell read "
            "to $d$ mRNA UMIs yields $d\\,p_j$ UMIs of that gene. It is what "
            "converts sequencing depth into counts of the gene actually being "
            "measured, and it is per-gene and condition-dependent rather than one "
            "number. Currently assembled from two sources that do not belong "
            "together, which is worth a five-fold spread in cells per "
            "perturbation and is the largest unforced uncertainty in the budget."
        ),
        where="sec:transcript-content",
    ),
    Term(
        term="Biological overdispersion",
        abbrev="$\\varphi_j$",
        definition=(
            "Variance in a gene's expression between genetically identical cells "
            "over and above what counting alone would produce. The \\emph{over} "
            "is the comparison against shot noise, which fixes the variance at "
            "$\\mu$; $\\varphi_j$ measures the excess. From cell-cycle phase, "
            "metabolic state and transcriptional bursting, so reading each cell "
            "more deeply does not reduce it and only more cells do. Equal to the "
            "negative-binomial dispersion, so $\\sqrt{\\varphi_j}$ is the "
            "biological coefficient of variation of the RNA-seq literature."
        ),
        where="sec:poisson",
    ),
    Term(
        term="Depth sufficiency",
        abbrev="$d^{*}$",
        definition=(
            "The per-cell depth $d^{*}=1/(\\varphi_j p_j)$ at which shot noise "
            "and biological overdispersion contribute equally, where $p_j$ is the "
            "target gene's transcriptome share and $\\varphi_j$ its "
            "overdispersion. Below it a screen is measurement-limited and deeper "
            "reads help; above it the requirement is within a factor of two of a "
            "floor no sequencing can cross."
        ),
        where="sec:design-equation",
    ),
    Term(
        term="Pseudobulk",
        definition=(
            "Summing UMIs across all cells sharing a perturbation, to estimate "
            "that perturbation's mean expression. Under multiplexing the pool is "
            "every cell carrying a guide against that gene \\emph{whatever else "
            "it carries}, not only cells with an identical guide set, which is "
            "what makes the estimand the main effect. It is what the whole design "
            "is powered on: single cells are too shallow individually, so the "
            "quantity that matters is cells $\\times$ depth."
        ),
        where="sec:cells-per-pert",
    ),
    Term(
        term="Biological floor",
        definition=(
            "The cells a perturbation needs once sequencing has stopped being the "
            "limitation. It is not independent of sequencing at any real depth; "
            "it is what the requirement converges to as depth grows, the "
            "$\\varphi_j$ term of \\cref{eq:design} with the shot-noise term "
            "driven to zero. Cell-cycle phase and metabolic state have to be "
            "averaged over, and reading one cell more deeply measures that one "
            "cell better rather than averaging anything. The published field "
            "standard is 100 cells."
        ),
        where="sec:cells-per-pert",
        citation_key="brandnerPooledSinglecellCRISPRa2025",
        line=200,
        quote="perturbations are retained only with at least 100 cells",
    ),
    Term(
        term="Usable vs sequenced cell",
        definition=(
            "A usable cell clears three filters -- enough UMIs to be a cell "
            "rather than ambient RNA, a barcode that resolves to exactly one "
            "cell, and an identified perturbation -- while a sequenced cell is "
            "one whose reads were paid for. The two differ by $\\sim$4$\\times$ "
            "in split-pool, which is why a quoted cost per cell understates the "
            "real one."
        ),
        where="sec:cost-per-cell",
    ),
    Term(
        term="Read pair",
        definition=(
            "One paired-end fragment, and the unit every sequencing price in this "
            "document is quoted in. Both method families need paired-end runs, "
            "one read carrying the cDNA and the other the barcodes, but they put "
            "the barcode on opposite reads."
        ),
        where="sec:sequencing",
    ),
    Term(
        term="PhiX spike-in",
        definition=(
            "A fraction of the loaded library that is control phage DNA. At "
            "$\\ge$10\\% it is a split-pool protocol requirement rather than "
            "hygiene, and the reason is the read structure rather than the "
            "organism: read 2 carries fixed linker sequence at defined cycles "
            "whatever cell the library came from, so base diversity collapses at "
            "those cycles and patterned flow cells mis-register clusters without "
            "a diverse library mixed in. It applies to a yeast split-pool run "
            "exactly as it does to a bacterial one. Budget it as a tax on usable "
            "reads."
        ),
        where="sec:sequencing",
        citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
        line=1012,
        quote="Use >=10% of PhiX spike-in.",
    ),
    # --- added after the first review round. Every one of these appeared in a
    # figure or in prose before it was defined anywhere, which is the specific
    # failure a controlled vocabulary exists to prevent.
    Term(
        term="Template-switching oligo",
        abbrev="TSO",
        definition=(
            "The oligo that adds the second PCR handle, in 3$'$ and 5$'$ "
            "chemistries alike, and "
            "\\emph{not} the same mechanism as a linker. It happens \\emph{within "
            "the single reverse-transcription reaction}, with no melting, "
            "re-annealing or second primer extension, which is what distinguishes "
            "it from overlap extension: the enzyme reaches the 5$'$ end of the "
            "mRNA and adds two or three non-templated C residues; the TSO, whose "
            "3$'$ end is rGrGrG, base-pairs with those Cs while the enzyme is "
            "still engaged; and the enzyme carries on, now copying the TSO "
            "instead of the mRNA. The handle is therefore written by the "
            "polymerase starting from the RNA's own 5$'$ end, which is why a "
            "TSO-delivered barcode marks the 5$'$ end of the transcript while an "
            "oligo-dT-delivered one marks the 3$'$ end."
        ),
        where="sec:identifiers",
    ),
    Term(
        term="Splint",
        definition=(
            "The same molecule as the linker strand, named for what it does: "
            "bridge two DNA ends so ligase can seal the nick between them. "
            "Ligation, not polymerization, which is the whole difference from a "
            "template-switching oligo, where a polymerase copies the oligo rather "
            "than a ligase joining to it. The two are not rival chemistries and "
            "not a split-pool against droplet split: split-pool ligates its "
            "barcodes on and template-switches its second handle on, in the same "
            "protocol (\\cref{fig:chemistry}). Sources call the jig a linker "
            "strand or a bridge oligonucleotide; splint is the general name."
        ),
        where="sec:brettner",
    ),
    Term(
        term="Tagmentation",
        definition=(
            "Fragmentation and adapter addition in one step, performed by the Tn5 "
            "transposase. Tn5 is loaded with adapter duplexes, cuts double-stranded "
            "DNA and ligates an adapter to each cut end in the same reaction, so it "
            "replaces a separate shear-then-ligate workflow. \\Cref{fig:splitseq} "
            "and \\cref{fig:scifi} draw it as an enzyme body rather than a "
            "sequence block, because it is a protein acting on the molecule, not "
            "a piece of the molecule."
        ),
        where="sec:identifiers",
    ),
    Term(
        term="Channel",
        definition=(
            "One microfluidic lane of a droplet chip, and the unit droplet reagent "
            "is sold in. A Chromium chip runs several channels side by side; each "
            "is loaded and priced independently and recovers up to $\\sim$20{,}000 "
            "cells, so more cells means proportionally more channels and "
            "proportionally more money. That is what ``cost scales with cells'' "
            "means concretely. A split-pool plate has no equivalent -- it is priced "
            "per plate however many cells pass through it, which is a cost "
            "advantage and not a throughput one: what a plate can carry is capped "
            "by cells per RT well and by barcode collisions, both of which bind "
            "well before price does (\\cref{sec:collisions})."
        ),
        where="sec:jariani",
    ),
    Term(
        term="Semipermeable capsule",
        abbrev="SPC",
        definition=(
            "A hydrogel shell, pore cutoff around 30\\,nm, that confines cells and "
            "their nucleic acids while letting enzymes diffuse in. A fourth "
            "isolation principle alongside droplet, plate and split-pool: like a "
            "droplet it is a physical container, but unlike a droplet it survives "
            "buffer exchange, so a capsule can be washed, sorted and carried "
            "through multi-step chemistry. Small molecules leak, which is what "
            "rules it out for a metabolomic readout."
        ),
        where="sec:vision",
    ),
    Term(
        term="Preindexing",
        definition=(
            "One round of split-pool barcoding done before the cell reaches any "
            "container: a well-specific barcoded primer reverse transcribes in "
            "situ, so the cell arrives carrying a label of its own. Several "
            "preindexed cells can then share a droplet and still be told apart "
            "(\\cref{sec:scifi})."
        ),
        where="sec:scifi",
    ),
    Term(
        term="Combinatorial fluidic indexing",
        abbrev="scifi",
        definition=(
            "Preindexing on a plate followed by droplet capture, so a cell's "
            "identity is the pair (round1 well, round2 droplet). Neither round "
            "identifies a cell on its own. It composes the two isolation "
            "principles rather than choosing between them, and it is what removes "
            "the Poisson loading limit from droplet throughput."
        ),
        where="sec:scifi",
    ),
    Term(
        term="Droplet overloading",
        definition=(
            "Deliberately loading a droplet generator far above the "
            "doublet-avoiding concentration. Ruinous without preindexing, since "
            "every multiply-occupied droplet becomes an unresolvable doublet; the "
            "operating point with it, since those droplets resolve into several "
            "usable cells. Measured at 95.5\\% droplet occupancy and 9.6 nuclei "
            "per droplet at 100-fold overloading."
        ),
        where="sec:scifi",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021",
        line=26,
        quote=(
            "up to a droplet fill rate of 95.5% and an average of 9.6 nuclei per "
            "droplet (1.53 million nuclei per channel)"
        ),
    ),
    Term(
        term="Uninformative sequencing cycle",
        definition=(
            "A sequencing cycle spent reading constant sequence -- a ligation "
            "overhang, a primer binding site, a Tn5 mosaic end -- rather than "
            "barcode or transcript. Paid for at the same rate as an informative "
            "one. Two thirds of SPLiT-seq's composite-barcode cycles are of this "
            "kind, and the reason is the read layout rather than any inefficiency "
            "in the run: the linkers sit between the barcodes on the molecule and "
            "the sequencer reads straight through in order, so 52 of read 2's 76 "
            "barcode cycles are linker. Worked out in \\cref{sec:sequencing}."
        ),
        where="sec:sequencing",
        citation_key="datlingerUltrahighthroughputSinglecellRNA2021",
        line=51,
        quote=(
            "the presence of constant ligation overhangs, primer binding sites "
            "and/or Tn5 mosaic ends in multiround combinatorial indexing renders "
            "a substantial proportion of sequencing cycles uninformative "
            "(sci-RNA-seq v.1: 42% uninformative; sci-RNA-seq v.3 and sci-Plex: "
            "13% uninformative; SPLiT-seq: 67% uninformative)"
        ),
    ),
    Term(
        term="Effect matrix",
        definition=(
            "Perturbations by genes, each entry the log fold change that "
            "perturbation causes in that gene. It is the object a screen "
            "estimates, the $\\boldsymbol{\\Theta}$ of \\cref{eq:theta}, and "
            "an expression compendium of single deletions already is one -- "
            "which is what makes its rank and sparsity measurable before any "
            "screen is run."
        ),
        where="sec:compression",
    ),
    Term(
        term="Composite sample",
        definition=(
            "One measurement of a linear combination of perturbation effects "
            "rather than of a single perturbation: a cell carrying $m$ guides, "
            "or a droplet holding $m$ singly-perturbed cells. Conventional "
            "screening is the case $m=1$. The unit compressed sensing counts, "
            "and the reason its sample requirement can be smaller than the "
            "number of perturbations."
        ),
        where="sec:compression",
    ),
    Term(
        term="Rank and sparsity",
        abbrev="$r$, $\\nu$",
        definition=(
            "The two properties that decide whether an effect matrix can be "
            "recovered from fewer samples than it has columns: $r$ how many "
            "components carry its variance, $\\nu$ how many genes one "
            "perturbation moves. Measured in yeast as $r\\approx177$ at 90\\% of "
            "variance and $\\nu\\approx269$ of 6{,}169 genes. Yao et al.\\ write "
            "$\\nu$ as $q$; it is renamed here because $q$ is the per-guide "
            "detection probability throughout this document and a document "
            "carrying two $q$'s has to warn about it on every use."
        ),
        where="sec:compression",
    ),
    Term(
        term="Guide-detection probability",
        abbrev="$q$",
        definition=(
            "The chance that one gRNA present in a cell is actually seen in that "
            "cell's reads. A $k$-plex cell is usable only if all $k$ are seen, so "
            "the usable fraction is $q^{k}$ and $q$ is the one parameter that "
            "enters the multiplex ceiling exponentially rather than linearly. The "
            "published range is very wide, 21--25\\% in bacteria against "
            "$>$71\\% in the genome-scale yeast screen, and which end applies "
            "here is the single most valuable thing the pilot can measure."
        ),
        where="sec:guide-capture",
        citation_key="nadal-ribellesSinglecellResolvedGenotypephenotype2025",
    ),
    Term(
        term="Guide RNA",
        abbrev="gRNA",
        definition=(
            "The RNA that targets dCas9 to one gene, and the perturbation itself "
            "in a CRISPRi screen. Written \\emph{gRNA} throughout; \\emph{sgRNA} "
            "appears only inside quotations and vendor documentation, where it "
            "means the same molecule. In yeast it is transcribed from a Pol\\,III "
            "promoter, so it is neither capped nor polyadenylated, which is the "
            "whole of why perturbation identity is hard to read out "
            "(\\cref{sec:guide-capture})."
        ),
        where="sec:guide-capture",
    ),
    Term(
        term="Additivity assumption",
        definition=(
            "That two perturbations in one cell move a gene by the sum of what "
            "each moves it alone, in log space. Every pooled decoding needs it, "
            "and it is a model rather than a fact. Yao et al.\\ do not claim "
            "interactions are absent but that they cancel over random "
            "combinations; measured in yeast they do not cancel, they buffer, "
            "with a double producing 0.62 of the sum of its singles. What that "
            "costs an analysis is a systematic direction rather than noise: main "
            "effects pooled over multiplexed cells are biased toward zero as $k$ "
            "rises, so a $1/k$ saving in cells is optimistic by whatever the "
            "buffering is at that $k$, which nobody has measured past $k=2$."
        ),
        where="sec:compression",
    ),
]

def rendered_definition(t: Term) -> str:
    """The definition as it goes into the table, with its source appended.

    An entry that asserts a number has to say where the number came from in the
    table itself, not only in this file -- otherwise a reader who wants to check
    "the field standard is 100 cells" has nowhere to go. Entries with no
    citation_key are definitional and get nothing appended.
    """
    if t.citation_key is None:
        return t.definition
    return t.definition + f"~\\citep{{{t.citation_key}}}"


def alphabetical() -> list[Term]:
    """TERMS sorted by term name, case- and markup-insensitively.

    Sorting on the raw string would let a leading backslash or brace in some
    future entry jump it to the top, so the key strips non-alphanumerics first.
    Abbreviations are deliberately NOT sort keys: a reader looking up UMI finds
    it under "Unique molecular identifier", which is what the See column and the
    prose both call it.
    """
    def key(t: Term) -> str:
        return "".join(ch for ch in t.term.lower() if ch.isalnum())

    return sorted(TERMS, key=key)


if __name__ == "__main__":
    for _t in alphabetical():
        _ab = f" ({_t.abbrev})" if _t.abbrev else ""
        _src = f"  [{_t.citation_key}]" if _t.citation_key else ""
        print(f"  {_t.term}{_ab} -> {_t.where}{_src}")
    print(f"\n{len(TERMS)} terms, "
          f"{sum(1 for t in TERMS if t.citation_key)} carrying a citation")
