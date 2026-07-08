---
id: mde51luj6nfl8mt78n0vmwr
title: Proof Writing Standard
desc: ''
updated: 1783309924393
created: 1783309924393
---

Canonical standard for writing proofs and formal claims in the Nature Biotech manuscript
(`paper/nature-biotech/`). Refer to this whenever authoring or editing a Supplementary
Note, proposition, or proof. The LaTeX house format is defined in
`paper/nature-biotech/preamble.tex` (see [[paper.nature-biotech.figures]] for the broader
paper workflow). Companion working notes:
[[paper.nature-biotech.fig1.perturbation-operator]], [[paper.nature-biotech.fig1.gat-cgt-equivalence]].

## 2026.07.05 - Proof-writing standard for the CGT paper

### Core principle

**Do not mix the formal claim, its proof, the intuition, and the caveat in one paragraph.**
Separate them into labeled parts. A formal section should read as a sequence of clearly
typed blocks, each answering one question:

- What are the objects? (spaces, maps, assumptions, notation)
- What is the claim? (proposition / lemma)
- Why is it true? (proof, split into named steps)
- What does it mean for the model? (consequence / interpretation)
- What is *not* being claimed? (precision / limitations)

### SI structure and heading hierarchy (Nature)

- **Nature does NOT typeset/subedit the SI** -- it is published as the authors provide it, so
  there is no mandated header hierarchy. The `\section*`/`\subsection*`/`\notesec` scheme is our
  choice; the only requirement is that it be clear, succinct, and match the paper's style
  (verified 2026.07: nature.com/nature/for-authors/formatting-guide and /supp-info; SR guidelines).
- **"Supplementary Note" / "Supplementary Fig." / "Supplementary Table" are the standard
  Nature designations.** Use them (not "Appendix", "SI Section", etc.). Supplementary Figures
  and Tables are numbered S1, S2, ... (we do this). Note: some Nature journals say Supplementary
  Notes need not be numbered and titles are optional; we keep "Supplementary Note 1/2/3" because
  the main text cross-references them by number.
- **Subsection titles must be professional noun phrases**, not informal sentences. Good:
  "Compute and complexity", "Empirical cost analysis", "Limitations of the assumption". Avoid:
  "Where it can break", "The bottleneck is redundant message passing, not loading".
- **Lead each Note with a plain-English paragraph** (the informal statement) before the formal
  setup, so a non-specialist gets the takeaway first. Then \notesec{Setting} for objects/notation.
- **Heading hierarchy mirrors the main text** (`sn-jnl`'s `\bmhead` is only a 10bp level-5
  head, too small to separate Notes -- do not use it for Note titles):
  - `\section*{Supplementary Information}` -- SI title (largest).
  - `\subsection*{Supplementary Note~N: title}` -- each Note; **starts on a new page**
    (`\clearpage` before it).
  - `\notesec{...}` -- run-in bold-upright subsections inside a Note (Setting, Proof,
    Precision, ...).
- **Each Supplementary Note starts a new page** (`\clearpage`). Not strictly mandated by
  Nature, but standard and cleaner.
- **Contents block** at the top of the SI (a bullet-free `tabular`), listing the Notes and
  the figure range. Maintained by hand -- update when a Note or figure block is added.
- **Order:** Notes (text + proofs) first, then ALL figures, then references.

### Skeleton (order within a Supplementary Note)

For a deep-learning paper, use mathematical proof structure but package it for scientific
readers as **Setup → Claim → Proof → Consequence → Interpretation**. Full note order:

1. **Informal statement** — one plain-language sentence of the result (the note title is the
   informal claim).
2. **Setting and notation** — define objects, spaces, maps; point to `Table~\ref{tab:notation}`;
   introduce only note-local symbols.
3. **Assumptions** — state conditions the result needs (A1--A3), explicitly.
4. **Proposition** — the formal claim, in a `proposition` environment.
5. **Proof** — in a `proof` environment, split into named steps.
6. **Corollary / implication for the model** — what the trained model inherits.
7. **Concrete example** — a small toy case showing why the claim matters.
8. **Precision / limitations** — what is exact vs. learned; what is not claimed.
9. **Relation to prior work** — closest precedents, how this differs.

Not every note needs all nine; keep the ones that carry weight, in this order.

### Environment hierarchy (which type to use)

| Type          | Use for                                                              |
|---------------|---------------------------------------------------------------------|
| `definition`  | Introduce objects: graph, perturbation, encoder, function class      |
| assumption    | State conditions needed for the result (plain `\paragraph{Assumptions.}`) |
| `lemma`       | Small technical fact used later                                      |
| `proposition` | Main formal claim in a supplement or theory section                 |
| theorem       | Major result — **only** if central and broad; avoid unless the paper is theoretical |
| `corollary`   | Immediate consequence of a proposition/theorem                      |
| `remark`      | Interpretation, caveat, limitation                                  |
| example       | Concrete toy case showing why the claim matters (`\paragraph{Concrete example.}`) |

For this manuscript prefer **Proposition** for main claims and **Corollary/Remark** for
consequences. **Do not call things Theorems** — the paper is empirical, not a theory paper.

### LaTeX house format (defined in `preamble.tex`)

```latex
% Style: bold head, ROMAN body (no italic statement -- reduces italics density).
\newtheoremstyle{tcplain}{\topsep}{\topsep}{}{}{\bfseries}{.}{.5em}{}
\theoremstyle{tcplain}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
% Run-in bold step header for structured proofs -- USE INSTEAD OF itemize/bullets.
\newcommand{\pfstep}[1]{\par\smallskip\noindent\textbf{#1.}\enspace}
% Run-in subsection header for the notes -- BOLD UPRIGHT. Use \notesec, NOT \paragraph:
% sn-jnl's \paragraph is \bfseries\itshape (bold-italic), which is wrong for Nature.
\newcommand{\notesec}[1]{\par\smallskip\noindent\textbf{#1.}\enspace}
```

Use `\notesec{Setting.}`, `\notesec{Precision.}`, etc. for note subsections (bold upright),
never `\paragraph{...}` (bold-italic in sn-jnl). Keep proof steps to about two `\pfstep`
headers (e.g. Containment / Strictness); more than that reads like a bullet list -- fold the
rest into flowing prose. Each proof should also have a small empirical figure showing the
proof's relevance where possible (see `fig:pert-equivalence-empirical` and
`fig:graphreg-empirical` in `backmatter.tex`).

Rules baked into the format:

- **No bullet lists (`itemize`) inside proofs.** Structure multi-part proofs with `\pfstep{...}`
  run-in headers (bold). This replaces the classic `\emph{Containment.}` italic run-in; we use
  bold to keep italics sparse.
- **Number every nontrivial displayed equation** (`equation`, not `\[...\]`, when it may be referenced).
- **One idea per paragraph.** Keep `\emph` and parentheticals to a minimum; prefer plain
  declarative sentences.
- State each result in a `proposition`/`lemma` environment; prove in the `proof` environment
  (gives the italic "Proof." head and the QED box automatically -- do not hand-write `\hfill$\square$`).

### Canonical template

```latex
\paragraph{Setting.} Define the objects, spaces, maps, assumptions, and notation.

\paragraph{Assumptions.} (A1) ... (A2) ... (A3) ...

\begin{proposition}[Short descriptive name]
\label{prop:key}
Under assumptions A1--A3, $\mathcal{H}_{\mathrm{data}} \subseteq \mathcal{H}_{\mathrm{rep}}$,
with equality if and only if <condition>.
\end{proposition}

\begin{proof}
\pfstep{Containment} ...
\pfstep{Strictness} ...
Therefore <conclusion>.
\end{proof}

\paragraph{Consequence.} Proposition~\ref{prop:key} implies that ...

\paragraph{Interpretation.} In model terms, this means ...
```

### Main text vs. supplement

Keep proofs in **Supplementary Notes**, not the main text (unless a proof is the paper's
central contribution). In the main text, state the intuition in words and cite the
proposition, e.g.:

> The perturbation operator can be viewed as an amortized reparameterization of graph
> rebuilding: any predictor defined on rebuilt perturbed graphs can be written as a predictor
> of the (reference graph, perturbation) pair. We formalize this containment result in
> Supplementary Note 1.

The supplement then carries the full mathematical proof.

### Worked reference

Supplementary Note 1 (`sections/backmatter.tex`, `\begin{proposition}[Reparameterization;
amortization containment]`) and Note 2 (`[Graph attention as the $\lambda\to\infty$ limit]`)
are the current worked examples of this standard. Note 1 informal title:
*"amortized perturbation operators contain graph-rebuilding predictors."*
