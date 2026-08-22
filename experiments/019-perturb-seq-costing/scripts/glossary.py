# experiments/019-perturb-seq-costing/scripts/glossary.py
# [[experiments.019-perturb-seq-costing.scripts.glossary]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/glossary
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

``group`` is kept on each Term because it still records what a term is about,
but it no longer drives rendering.
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


# Groups are rendered as subheading rows inside the table, in this order.
GROUPS: list[str] = [
    "Tags on a molecule",
    "Cell isolation",
    "Split-pool chemistry",
    "Screen design",
    "Counting and cost",
]

TERMS: list[Term] = [
    # --- Tags on a molecule ---------------------------------------------------
    Term(
        term="Cell barcode",
        abbrev="CB",
        definition=(
            "A sequence shared by every cDNA molecule from one cell and ideally "
            "unique to it; what makes the data single-cell rather than bulk. "
            "Delivered pre-synthesised on a bead in droplet methods, built up "
            "across rounds of well-specific ligation in split-pool."
        ),
        where="sec:identifiers",
    ),
    Term(
        term="Unique molecular identifier",
        abbrev="UMI",
        definition=(
            "A short random sequence attached to each cDNA molecule "
            "\\emph{before} amplification. Reads sharing CB, gene and UMI are "
            "PCR copies of one original mRNA, so collapsing on the UMI turns "
            "read counts into molecule counts. 10\\,nt in the split-pool "
            "protocols here, 10--12\\,nt in droplet."
        ),
        where="sec:identifiers",
        citation_key="gaisserHighthroughputSinglecellTranscriptomics2024",
        line=47,
        quote="a 10-base unique molecular identifier (UMI)",
    ),
    Term(
        term="Sublibrary index",
        definition=(
            "The Illumina index read added at final PCR. In split-pool it is not "
            "merely a sample tag: cells are divided into sublibraries "
            "\\emph{after} the last in-cell barcoding round, so the index is a "
            "further barcode dimension and two cells that took identical plate "
            "paths are still resolved if they landed in different sublibraries. "
            "This is why unique dual indices are mandatory -- an index hop "
            "manufactures a barcode collision."
        ),
        where="sec:collisions",
    ),
    Term(
        term="Guide barcode",
        abbrev="GBC",
        definition=(
            "An expressed, polyadenylated sequence that stands in for the guide "
            "RNA itself. A Pol\\,III guide cassette carries no poly(A) tail and "
            "is therefore invisible to 3$'$ capture, so the perturbation "
            "identity has to be re-encoded in something the assay can see."
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
            "the same UMI, and being counted once. Negligible at yeast "
            "expression levels: a 10\\,nt UMI has $4^{10}\\approx10^{6}$ states "
            "against at most a few thousand molecules of any one gene. Not to be "
            "confused with barcode collision, which is a cell-level error."
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
            "Two cells reported as one. In droplet methods it is two cells in "
            "one partition, governed by Poisson loading (10x quote $\\sim$8\\% at "
            "20{,}000 cells recovered); in split-pool the analogous failure is a "
            "barcode collision, which has a different and controllable cause."
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
            "A bead-borne reaction adding a common handle to the far end of the "
            "cDNA, giving the molecule PCR primer sites at both ends. Everything "
            "after it is conventional Illumina library prep."
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
            "permeabilise, three rounds, split into sublibraries. Consumes one "
            "withdrawal from each barcode plate and carries up to $\\sim$480{,}000 "
            "cells. A unit of \\emph{bench work}, not of sequencing: the "
            "sublibraries it yields are sequenced and priced separately."
        ),
        where="sec:startup",
    ),
    Term(
        term="Sublibrary",
        definition=(
            "An aliquot of barcoded cells (5{,}000--20{,}000) split off after the "
            "last in-cell round, lysed and prepared as one Illumina library. "
            "Sublibrary count is set jointly by barcode-collision targets and by "
            "PCR complexity limits."
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
            "Guide RNAs delivered per cell, so $k$ genes are perturbed in that "
            "cell at once. Raising $k$ divides the cell requirement for main "
            "effects by $k$; it does not make genome-wide pairwise interactions "
            "reachable."
        ),
        where="sec:multiplex",
    ),
    Term(
        term="Main effect",
        definition=(
            "The average effect of perturbing one gene, pooled over every cell "
            "carrying a guide against it regardless of what else that cell "
            "carries. Well defined only under an additive (log-additive) model."
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
            "The decoder for a compressed screen: factorises the observed "
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
            "guides. \\emph{Not} cells per guide and \\emph{not} cells per "
            "plasmid -- the distinction is a factor of six here and is the most "
            "common way a screen budget is misread."
        ),
        where="sec:cells-per-pert",
    ),
    # --- Counting and cost ----------------------------------------------------
    Term(
        term="Shot noise",
        definition=(
            "The irreducible randomness of counting. Sequencing samples molecules "
            "at random, so a gene at true rate $\\mu$ is observed with variance "
            "$\\mu$ and relative error $1/\\sqrt{\\mu}$. A property of the "
            "measurement, not of the cell -- and the only noise that buying more "
            "reads removes. Also called counting or sampling noise."
        ),
        where="sec:limits",
    ),
    Term(
        term="Biological overdispersion",
        abbrev="$\\varphi_j$",
        definition=(
            "Variance in a gene's expression between genetically identical cells "
            "in excess of shot noise, from cell-cycle phase, metabolic state and "
            "transcriptional bursting. Reading each cell more deeply does not "
            "reduce it; only more cells do. Equal to the negative-binomial "
            "dispersion, so $\\sqrt{\\varphi_j}$ is the biological coefficient "
            "of variation reported in the RNA-seq literature."
        ),
        where="sec:design-equation",
    ),
    Term(
        term="Depth sufficiency",
        abbrev="$d^{*}$",
        definition=(
            "The per-cell depth $1/(\\varphi_j p_j)$ at which shot noise and "
            "biological overdispersion contribute equally. Below it a screen is "
            "measurement-limited and deeper reads help; above it the requirement "
            "is within a factor of two of a floor no sequencing can cross."
        ),
        where="sec:design-equation",
    ),
    Term(
        term="Pseudobulk",
        definition=(
            "Summing UMIs across all cells sharing a perturbation, to estimate "
            "that perturbation's mean expression. It is what the whole design is "
            "powered on: single cells are too shallow individually, so the "
            "quantity that matters is cells $\\times$ depth."
        ),
        where="sec:cells-per-pert",
    ),
    Term(
        term="Biological floor",
        definition=(
            "The minimum cells a perturbation needs before its mean is "
            "interpretable, independent of sequencing. Cell-cycle phase and "
            "metabolic state must be averaged out, and depth cannot substitute. "
            "The published field standard is 100 cells."
        ),
        where="sec:cells-per-pert",
        citation_key="brandnerPooledSinglecellCRISPRa2025",
        quote="perturbations are retained only with at least 100 cells",
    ),
    Term(
        term="Usable vs sequenced cell",
        definition=(
            "A usable cell survives QC \\emph{and} carries an identified "
            "perturbation; a sequenced cell is one whose reads were paid for. "
            "The two differ by $\\sim$4$\\times$ in split-pool, which is why a "
            "quoted cost per cell understates the real one."
        ),
        where="sec:cost-per-cell",
    ),
    Term(
        term="Read pair",
        definition=(
            "One paired-end fragment, the unit sequencing is priced in here. "
            "Both method families need paired-end runs -- one read carries the "
            "cDNA and the other the barcodes -- but they put the barcode on "
            "opposite reads."
        ),
        where="sec:sequencing",
    ),
    Term(
        term="PhiX spike-in",
        definition=(
            "A fraction of the loaded library that is control phage DNA. At "
            "$\\ge$10\\% it is a split-pool protocol requirement rather than "
            "hygiene: read 2 carries fixed linker sequence at defined cycles, so "
            "base diversity collapses and patterned flow cells mis-register "
            "clusters without it. Budget it as a tax on usable reads."
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
            "The oligo that adds the second PCR handle in 5$'$ chemistries, and "
            "\\emph{not} the same mechanism as a linker. Reverse transcriptase, on "
            "reaching the 5$'$ end of an mRNA, adds two or three non-templated C "
            "residues; the TSO ends in rGrGrG, base-pairs with them, and the "
            "enzyme switches template and copies the TSO onto the cDNA. The handle "
            "is therefore written by the polymerase starting from the RNA's own "
            "5$'$ end, which is why a TSO-delivered barcode marks the 5$'$ end of "
            "the transcript while an oligo-dT-delivered one marks the 3$'$ end."
        ),
        where="sec:identifiers",
    ),
    Term(
        term="Splint",
        definition=(
            "The same molecule as the linker strand, named for what it does: "
            "bridge two DNA ends so ligase can seal the nick between them. "
            "Ligation, not polymerization -- which is the whole difference from a "
            "template-switching oligo, where a polymerase copies the oligo rather "
            "than a ligase joining to it. Split-pool uses splints; droplet 5$'$ "
            "chemistry uses template switching."
        ),
        where="sec:brettner",
    ),
    Term(
        term="Tagmentation",
        definition=(
            "Fragmentation and adapter addition in one step, performed by the Tn5 "
            "transposase. Tn5 is loaded with adapter duplexes, cuts double-stranded "
            "DNA and ligates an adapter to each cut end in the same reaction, so it "
            "replaces a separate shear-then-ligate workflow. Figures draw it as an "
            "enzyme body rather than a sequence block, because it is a protein "
            "acting on the molecule, not a piece of the molecule."
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
            "per plate however many cells pass through it."
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
]

# Which group each term belongs to, by position in TERMS. Kept as explicit
# boundaries rather than a per-term field: the grouping is a property of the
# ORDER (terms are meant to be read in sequence), and a per-term string invites
# a typo that silently creates a fourth group.
GROUP_BOUNDS: dict[str, tuple[int, int]] = {
    "Tags on a molecule": (0, 6),
    "Cell isolation": (6, 10),
    "Split-pool chemistry": (10, 17),
    "Screen design": (17, 23),
    "Counting and cost": (23, len(TERMS)),
}


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
        print(f"  {_t.term}" + (f" ({_t.abbrev})" if _t.abbrev else ""))
    raise SystemExit(0)
    for g, terms in grouped():
        print(f"\n{g}")
        for t in terms:
            ab = f" ({t.abbrev})" if t.abbrev else ""
            src = f"  [{t.citation_key}]" if t.citation_key else ""
            print(f"  {t.term}{ab} -> {t.where}{src}")
    print(f"\n{len(TERMS)} terms")
