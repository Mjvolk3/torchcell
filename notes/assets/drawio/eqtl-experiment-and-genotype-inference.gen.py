# notes/assets/drawio/eqtl-experiment-and-genotype-inference.gen.py
# [[torchcell.datamodels.eqtl-data-model]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes/assets/drawio/eqtl-experiment-and-genotype-inference.gen.py
"""Emit notes/assets/drawio/eqtl-experiment-and-genotype-inference.drawio.

Repetitive cells (matrix grids, marker ticks, the eQTL map) are generated so positions
are exact; the .drawio is committed as the hand-composed schematic and exported to
notes-tex/eqtl-data-model/figures/ by `make figures` there. Panel e states what a
record stores as BUILT for Bloom 2019 (torchcell/datasets/scerevisiae/bloom2019.py,
2026-09-12): a SegregantGenotype sibling class, not a perturbation leaf.

Run from the repo root:
    python notes/assets/drawio/eqtl-experiment-and-genotype-inference.gen.py \
        notes/assets/drawio/eqtl-experiment-and-genotype-inference.drawio
"""

import random

OR, ORF = "#D79B00", "#FFE6CC"     # parent A / orange
RD, RDF = "#B85450", "#F8CECC"     # parent B / red
PU, PUF = "#9673A6", "#E1D5E7"     # inferred / purple
YE, YEF = "#D6B656", "#FFF2CC"     # measured / yellow
BL, BLF = "#6C8EBF", "#DAE8FC"     # emphasis / blue
GY, GYF = "#666666", "#F5F5F5"     # derived / gray

C = []
def cell(i, v, st, x, y, w, h):
    C.append(f'<mxCell id="{i}" value="{v}" style="{st}" vertex="1" parent="1">'
             f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
def box(i, v, x, y, w, h, s, f, fs=8.5):
    cell(i, v, f"rounded=0;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};"
         f"fontFamily=Arial;fontSize={fs};", x, y, w, h)
def txt(i, v, x, y, w, h, col="#666666", al="left", fs=8.5, va="top"):
    cell(i, v, f"text;html=1;whiteSpace=wrap;align={al};verticalAlign={va};"
         f"fontFamily=Arial;fontSize={fs};fontColor={col};", x, y, w, h)
def note(i, v, x, y, w, h, s=BL, f=BLF):
    cell(i, v, f"rounded=0;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};"
         f"fontFamily=Arial;fontSize=8.5;align=left;verticalAlign=top;"
         f"spacingLeft=5;spacingTop=3;spacingRight=5;", x, y, w, h)
def swatch(i, x, y, w, h, s, f):
    cell(i, "", f"rounded=0;html=1;fillColor={f};strokeColor={s};strokeWidth=0.5;", x, y, w, h)
def arrow(i, x1, y1, x2, y2, col=GY, w=1.5, dash=0):
    C.append(f'<mxCell id="{i}" style="endArrow=classic;html=1;strokeWidth={w};'
             f'strokeColor={col};endSize=4;dashed={dash};" edge="1" parent="1">'
             f'<mxGeometry relative="1" as="geometry">'
             f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
             f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')
def line(i, x1, y1, x2, y2, col=GY, w=1, dash=0):
    C.append(f'<mxCell id="{i}" style="endArrow=none;html=1;strokeWidth={w};'
             f'strokeColor={col};dashed={dash};" edge="1" parent="1">'
             f'<mxGeometry relative="1" as="geometry">'
             f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
             f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')
def panel(i, letter, title, x, y, w=560):
    txt(i, f"&lt;b&gt;{letter}&lt;/b&gt;&#160;&#160;{title}", x, y, w, 14,
        col="#000000", fs=10, va="middle")

# ---------------------------------------------------------------- panel a
panel("pa", "a", "The cross, and the two things measured on every segregant", 8, 2)
box("a1", "Parent A&#10;(BY)", 8, 22, 74, 26, OR, ORF)
box("a2", "Parent B&#10;(RM)", 8, 52, 74, 26, RD, RDF)
txt("a3", "deep WGS &gt;100&#215;&#10;every variant and both&#10;alleles known up front&#10;(&#8776;4.5&#215;10&#8308; SNPs)", 8, 82, 92, 44)
arrow("a4", 82, 51, 96, 51)
box("a5", "F1 hybrid&#10;diploid", 96, 39, 66, 26, PU, PUF)
arrow("a6", 162, 51, 176, 51)
box("a7", "sporulate&#10;n segregants", 176, 39, 78, 26, PU, PUF)
txt("a8", "each one a meiotic&#10;MOSAIC of the two", 176, 68, 90, 24)
arrow("a9", 254, 44, 268, 30)
arrow("a10", 254, 58, 268, 86)
box("a11", "genotype assay", 268, 18, 104, 22, YE, YEF)
txt("a12", "SPARSE: markers, or&#10;only expressed SNPs", 268, 42, 108, 24)
box("a13", "transcriptome", 268, 76, 104, 22, YE, YEF)
txt("a14", "DENSE: every gene,&#10;every segregant", 268, 100, 108, 24)
arrow("a15", 372, 29, 386, 29)
arrow("a16", 372, 87, 386, 87)
box("a17", "X&#160;&#160;n &#215; m&#10;genotype POSTERIOR", 386, 18, 118, 30, PU, PUF)
box("a18", "Y&#160;&#160;n &#215; p&#10;expression", 386, 76, 118, 30, YE, YEF)
note("a19", "&lt;b&gt;The asymmetry.&lt;/b&gt; The phenotype is read directly. The genotype never is: it is INFERRED from sparse observations as a mosaic of two known genomes. Panel b is that inference; panels d and e are what it costs.",
     518, 18, 176, 88)

# ---------------------------------------------------------------- panel b
panel("pb", "b", "One segregant, one chromosome: from sparse reads to a posterior", 8, 130)
LX, RX = 132, 694          # plot span
txt("b0", "TRUE variants&#10;in catalog V", 8, 150, 116, 24, al="right", va="middle")
for k in range(70):        # dense true-variant ticks
    swatch(f"bt{k}", LX + 2 + k * 8, 152, 1.6, 12, GY, GY)
txt("b1", "&#8776;1 per 260 bp, none of them read directly", LX, 166, 300, 12)

txt("b2", "OBSERVED&#10;marker calls", 8, 186, 116, 24, al="right", va="middle")
mx = [LX + 10 + k * 46 for k in range(12)]
calls = "AABBBBAAAABB"
for k, (x, c) in enumerate(zip(mx, calls)):
    swatch(f"bm{k}", x, 186, 12, 14, BL, ORF if c == "A" else RDF)
    txt(f"bml{k}", c, x, 186, 12, 14, col="#000000", al="center", va="middle")
txt("b3", "&#8776;1 per 1 kb; each is a read count, not a certainty", LX, 203, 320, 12)

txt("b4", "HMM POSTERIOR&#10;Pr(allele = B)", 8, 222, 116, 26, al="right", va="middle")
# posterior trace: flat at 0 or 1, ramps across the ambiguous boundaries
TY, TH = 222, 26
line("bax", LX, TY + TH, RX, TY + TH, GY, 0.75)
txt("b5", "1", LX - 8, TY - 4, 8, 10, al="right")
txt("b6", "0", LX - 8, TY + TH - 6, 8, 10, al="right")
seg = [(LX, 0), (238, 0), (268, 1), (388, 1), (418, 0), (520, 0), (550, 1), (RX, 1)]
for k in range(len(seg) - 1):
    x1, v1 = seg[k]; x2, v2 = seg[k + 1]
    line(f"bp{k}", x1, TY + TH - v1 * TH, x2, TY + TH - v2 * TH, PU, 1.5)

txt("b7", "INFERRED&#10;blocks", 8, 258, 116, 24, al="right", va="middle")
for k, (x0, x1, p) in enumerate([(LX, 238, "A"), (268, 388, "B"), (418, 520, "A"), (550, RX, "B")]):
    box(f"bb{k}", p, x0, 258, x1 - x0, 20, OR if p == "A" else RD, ORF if p == "A" else RDF)
for k, (x0, x1) in enumerate([(238, 268), (388, 418), (520, 550)]):
    cell(f"bu{k}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={GYF};strokeColor={GY};"
         f"dashed=1;fontFamily=Arial;fontSize=8.5;", x0, 254, x1 - x0, 28)
# a gene conversion tract, invisible between markers
cell("bgc", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={RDF};strokeColor={RD};"
     f"dashed=1;fontFamily=Arial;fontSize=8.5;", 484, 258, 18, 20)
arrow("bgca", 493, 300, 493, 279, RD, 1)
txt("b8", "the ramp IS the uncertainty:&#10;a crossover is placed only to&#10;within one marker interval",
    196, 288, 156, 34)
txt("b9", "a gene conversion tract sits&#10;BETWEEN markers, so no read&#10;reports it and no ramp appears",
    396, 302, 160, 34, col=RD)

note("b10", "&lt;b&gt;Why sparse reads still suffice.&lt;/b&gt; The segregant is never sequenced de novo. At each locus it is ASSIGNED to one of two already-sequenced genomes. Meiosis makes &#8776;90 crossovers, so a genome is &#8776;10&#178; blocks, and one marker fixes every cataloged variant in its block. The task is locating &#8776;10&#178; boundaries, not determining &#8776;10&#8308; values.",
     8, 342, 336, 62)
note("b11", "&lt;b&gt;What no marker density fixes.&lt;/b&gt; Boundary intervals leave &#8776;0.8% of cataloged sites ambiguous. Non-crossover gene conversion (&#8776;46 per meiosis, &#8776;2 kb) affects more bases than that and is invisible to sparse markers. De novo mutation, aneuploidy (a haploid can carry a disomy) and non-reference sequence sit outside the mosaic model entirely.",
     358, 342, 336, 62, GY, GYF)

# ---------------------------------------------------------------- panel c
panel("pc", "c", "What the two matrices look like", 8, 418, 360)
random.seed(7)
# X: rows = segregants, cols = markers, mosaic blocks per row
X0, Y0, CW, CH = 76, 456, 7.2, 7.2
txt("c0", "&lt;b&gt;X&lt;/b&gt;&#160;&#160;n segregants &#215; m markers", X0, Y0 - 16, 160, 12, col="#000000")
for r in range(11):
    c = 0
    cur = random.choice([0, 1])
    while c < 18:
        run = random.randint(5, 11)
        for k in range(c, min(c + run, 18)):
            swatch(f"cx{r}_{k}", X0 + k * CW, Y0 + r * CH, CW - 0.6, CH - 0.6,
                   OR if cur == 0 else RD, ORF if cur == 0 else RDF)
        c += run
        cur = 1 - cur
txt("c3", "blocks, not noise: each row is a&#10;mosaic, so columns are linked", X0, Y0 + 84, 150, 24)

# Y: rows = segregants, cols = genes, graded fill
Y0X = 240
txt("c4", "&lt;b&gt;Y&lt;/b&gt;&#160;&#160;n segregants &#215; p genes", Y0X, Y0 - 16, 160, 12, col="#000000")
GRAD = ["#FFFFFF", "#F5F5F5", "#DAE8FC", "#9DBDE8", "#6C8EBF"]
for r in range(11):
    for k in range(18):
        swatch(f"cy{r}_{k}", Y0X + k * CW, Y0 + r * CH, CW - 0.6, CH - 0.6,
               "#CCCCCC", random.choice(GRAD))
txt("c6", "p &#8776; 5,000 to 6,000; a VECTOR&#10;per strain, not a scalar", Y0X, Y0 + 84, 150, 24)

# ---------------------------------------------------------------- panel d
panel("pd", "d", "The resulting map: cis on the diagonal, trans in bands", 396, 418, 298)
DX, DY, DW, DH = 452, 452, 90, 90
cell("d0", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={GY};"
     f"strokeWidth=0.75;fontFamily=Arial;fontSize=8.5;", DX, DY, DW, DH)
txt("d1", "marker position &#8594;", DX, DY + DH + 2, DW, 10, al="center")
txt("d2", "gene position &#8594;", DX - 76, DY, 72, DH, al="right", va="middle")
for k in range(1, 8):      # faint gridlines, eighths
    grid = "rounded=0;html=1;fillColor=#DDDDDD;strokeColor=none;"
    cell(f"dgv{k}", "", grid, DX + k * DW / 8, DY, 0.5, DH)
    cell(f"dgh{k}", "", grid, DX, DY + k * DH / 8, DW, 0.5)
for k in range(26):        # cis: the diagonal
    f = k / 25.0
    swatch(f"dc{k}", DX + f * (DW - 2.4), DY + DH - 4 - f * (DH - 4), 3, 3, OR, OR)
for hx, cnt in [(0.30, 14), (0.62, 18), (0.85, 9)]:   # trans hotspots
    for k in range(cnt):
        swatch(f"dh{hx}_{k}", DX + hx * (DW - 2.4) + random.uniform(-1.5, 1.5),
               DY + 3 + random.uniform(0, DH - 9), 2.4, 2.4, PU, PU)
for k in range(30):        # scattered weak trans
    swatch(f"ds{k}", DX + random.uniform(1.2, DW - 3), DY + random.uniform(2, DH - 5),
           1.8, 1.8, "#BBBBBB", "#BBBBBB")
txt("d3", "&#9679; cis: a variant acting on its OWN gene, usually in the promoter",
    DX + DW + 10, DY + 2, 140, 24, col=OR)
txt("d4", "&#9679; trans hotspot: one regulatory variant moving hundreds of transcripts",
    DX + DW + 10, DY + 30, 140, 34, col=PU)
txt("d5", "each dot is one significant (gene, marker) pair: a row of the QTL table",
    DX + DW + 10, DY + 68, 140, 24)

# ---------------------------------------------------------------- panel e
panel("pe", "e", "What a record stores, as built for the Bloom 2019 segregant panels", 8, 566)
note("e1", "&lt;b&gt;Stored.&lt;/b&gt; One record per segregant per condition, (genotype, environment) &#8594; phenotype. The genotype is a SegregantGenotype, a sibling of Genotype rather than a perturbation leaf: HaplotypeBlocks of (chromosome, start, end, parent, posterior, n_markers) against two SegregantParents pinned to sha256-anchored assemblies. The variant set is a derived view. Measured: 83 to 143 blocks per segregant (medians by cross), tails to 1,847 from hard calls at posterior 1.0.",
     8, 588, 340, 74)
note("e2", "&lt;b&gt;Not stored, and what stays open.&lt;/b&gt; The QTL table is an estimate conditional on method, threshold and marker density, with no source value to check, so it is not a Phenotype. A mosaic needs no systematic_gene_name, since it is not a GenePerturbation; a cis-eQTL is still usually an intergenic promoter variant with no gene to attach to, so per-variant attribution and the reporter-assay class stay one open decision.",
     358, 588, 336, 74, RD, RDF)

W, H = 702, 672
xml = ('    <!--\n'
       '      Generated by notes/assets/drawio/eqtl-experiment-and-genotype-inference.gen.py,\n'
       '      committed as the hand-composed schematic. Layout comments live in the prolog: a comment\n'
       '      between diagram and mxGraphModel makes drawio fail to read the file.\n'
       '      Palette: slots 1..6 of PLOT_PALETTE. A orange, B red, inferred purple,\n'
       '      measured yellow, derived gray, emphasis blue.\n'
       '    -->\n'
       '<mxfile host="app.diagrams.net" agent="claude-code">\n'
       '  <diagram name="eqtl-experiment" id="eqtl-experiment">\n'
       f'    <mxGraphModel dx="1400" dy="1000" grid="0" page="1" pageWidth="{W}" '
       f'pageHeight="{H}" math="0" shadow="0">\n      <root>\n'
       '        <mxCell id="0" />\n        <mxCell id="1" parent="0" />\n'
       + "\n".join("        " + c for c in C)
       + '\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n')
import sys
open(sys.argv[1], "w").write(xml)
print(f"wrote {len(C)} cells")
