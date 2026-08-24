---
id: 4cgfguczsw3vp3yrd953w6k
title: Betaxanthin Pathway Latex
desc: ''
updated: 1787612268654
created: 1787612268654
---

## 2026.08.02 - Typeset forms of the betaxanthin pathway

Four LaTeX variants of the L-tyrosine to betaxanthin pathway, each compiled and
checked against the manuscript preamble. Related:
[[metabolism.beta-carotene-and-betaxanthin]].

L-tyrosine → L-DOPA → betalamic acid → betaxanthin

## 2026.08.02 - LaTeX version of the pathway

Rendered preview (all four variants typeset, each above its source):
`notes/assets/pdf-output/metabolism.betaxanthin-pathway-latex.pdf`. Scratch PDF, not
committed. If this graduates to a real note or into the paper, the generating `.tex`
has to move into the repo under the experiment/paper folder that owns it.

All variants need only `amsmath`, which `paper/nature-biotech/preamble.tex` already
loads (line 25), so they drop into the manuscript with no new package. Verified by
compiling with tectonic.

### A. Minimal -- the arrow chain as written above

```latex
\[
  \text{L-tyrosine} \longrightarrow \text{L-DOPA} \longrightarrow
  \text{betalamic acid} \longrightarrow \text{betaxanthin}
\]
```

### B. Enzyme-annotated, one line

```latex
\[
  \text{L-tyrosine}
  \xrightarrow{\;\text{CYP76AD1}\;}
  \text{L-DOPA}
  \xrightarrow{\;\text{DOD}\;}
  \text{betalamic acid}
  \xrightarrow[\text{amine}]{\;\text{spontaneous}\;}
  \text{betaxanthin}
\]
```

### C. Full annotation, wrapped (fits a Nature column)

```latex
\begin{equation*}
\begin{aligned}
  \text{L-tyrosine}
    &\xrightarrow[\text{(tyrosine 3-hydroxylase)}]{\text{CYP76AD1}}
     \text{L-DOPA} \\[4pt]
    &\xrightarrow[\text{(4,5-DOPA extradiol dioxygenase)}]{\text{DOD}}
     \text{betalamic acid} \\[4pt]
    &\xrightarrow[\text{amino acid / amine}]{\text{spontaneous}}
     \text{betaxanthin}
\end{aligned}
\end{equation*}
```

### D. Mechanistic -- intermediate and co-substrate made explicit

```latex
\begin{equation*}
\begin{aligned}
  \text{L-tyrosine}
    &\xrightarrow{\text{CYP76AD1}} \text{L-DOPA} \\
  \text{L-DOPA}
    &\xrightarrow{\text{DOD}} \text{4,5-\textit{seco}-DOPA}
     \xrightarrow{\text{spont.}} \text{betalamic acid} \\
  \text{betalamic acid} + \mathrm{R}\text{-}\mathrm{NH}_2
    &\xrightarrow{\text{spont.}} \text{betaxanthin} + \mathrm{H}_2\mathrm{O}
\end{aligned}
\end{equation*}
```

Notes on the chemistry encoded above:

- `\xrightarrow{above}[below]` is the standard amsmath reaction arrow. Convention here
  is enzyme above, co-substrate or qualifier below.
- The DOD step is 4,5-extradiol ring cleavage; the product 4,5-*seco*-DOPA cyclizes
  spontaneously to betalamic acid, which is why D splits that step in two.
- The final condensation is a non-enzymatic Schiff-base formation between betalamic
  acid's aldehyde and any primary amine or amino acid, so the product is a *family*
  of betaxanthins named for the amine (e.g. proline gives indicaxanthin). Labeling
  that arrow "spontaneous" rather than naming an enzyme is deliberate.
- `\text{L-DOPA}` keeps upright roman type; do not write it bare in math mode or the
  letters italicize and read as a product of variables.
