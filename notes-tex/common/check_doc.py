#!/usr/bin/env python
# notes-tex/common/check_doc.py
# [[notes-tex.common.check_doc]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/common/check_doc.py
"""Formatting + provenance + status gate for a notes-tex document.

The analogue of the manuscript's `make checkfigs` / `wordcount.py`, but aimed at
the three things that actually go wrong in these documents:

1. **Formatting.** Overfull boxes mean a table or figure is running off the page.
   Our pandoc-built PDFs had exactly this and it was invisible until someone
   looked at page 8. LaTeX reports it; nobody reads the log; so read it here.
2. **Provenance.** A number in the document that has no `%% SOURCE:` line and no
   visible `\\external{}` flag is an unsourced claim. That is the specific defect
   the repo's whole provenance rule exists to prevent, and it is mechanically
   detectable.
3. **Citations.** A `\\cite{key}` whose key is not in references.bib means the work
   is not in the Zotero group collection -- i.e. someone cited a paper we do not
   hold. (This is the check that would have caught Boocock 2023.)

Exit code is 0 on pass, 1 if any ERROR-level finding fires. Warnings never fail
the build -- a `todo` section is supposed to be incomplete.

Usage:  python ../common/check_doc.py main
"""

from __future__ import annotations

import os
import os.path as osp
import re
import sys
from collections import Counter

# A4 minus 2x14 mm margins, in PostScript points. Anything wider than the text
# block will overflow, however good the caption looks in isolation.
TEXT_BLOCK_MM = 182.0
MM_PER_PT = 25.4 / 72.0

RED, YEL, GRN, GRY, RST = "\033[31m", "\033[33m", "\033[32m", "\033[90m", "\033[0m"


def _read(path: str) -> str:
    if not osp.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def tex_sources() -> dict[str, str]:
    """Every .tex file that makes up the document, by path."""
    out = {}
    for root, _dirs, files in os.walk("."):
        if any(p in root for p in (".git", "figures")):
            continue
        for fn in files:
            if fn.endswith(".tex"):
                p = osp.join(root, fn)
                out[p] = _read(p)
    return out


def check_status(srcs: dict[str, str]) -> list[tuple[str, str]]:
    """Status stoplight summary -- what is done, what is not."""
    findings = []
    counts = Counter()
    per_section = []
    # \section{Title\secstatus{todo}} / \subsection{...}
    pat = re.compile(r"\\(sub)*section\*?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
    for path, txt in sorted(srcs.items()):
        for m in pat.finditer(txt):
            title = m.group(2)
            st = re.search(r"\\secstatus\{(\w+)\}", title)
            state = st.group(1) if st else "MISSING"
            counts[state] += 1
            clean = re.sub(r"\\secstatus\{\w+\}", "", title).strip()
            clean = re.sub(r"\\[a-zA-Z]+", "", clean).strip()
            per_section.append((state, clean, path))

    total = sum(counts.values())
    print(f"\n{'='*72}\nSTATUS  ({total} headings)")
    for state, sym, col in (
        ("final", "check", GRN),
        ("tent", "square", YEL),
        ("todo", "X", RED),
    ):
        print(f"  {col}{state:<6}{RST} {counts[state]:>3}  {GRY}{sym}{RST}")
    if counts["MISSING"]:
        print(f"  {RED}no chip{RST} {counts['MISSING']:>3}")
        for state, title, path in per_section:
            if state == "MISSING":
                findings.append(
                    ("WARN", f"heading has no \\secstatus: {title!r} in {path}")
                )
    for state, title, _p in per_section:
        col = {"final": GRN, "tent": YEL, "todo": RED}.get(state, GRY)
        mark = {"final": "v", "tent": "~", "todo": "x"}.get(state, "?")
        print(f"    {col}{mark}{RST} {title[:64]}")

    # Two headings with the same text and nothing between them. This happens when
    # content is inserted ABOVE a heading and the inserted block re-states the
    # heading it was meant to precede -- the build succeeds, the TOC gains a
    # phantom entry, and the numbering of every following section shifts by one.
    # Nothing in the log mentions it; it is only visible by reading the TOC.
    # Only ADJACENT repeats are flagged: a legitimately repeated title elsewhere
    # in a long document (e.g. "Limitations" under two methods) is fine.
    for (s1, t1, p1), (_s2, t2, _p2) in zip(per_section, per_section[1:]):
        if t1 and t1 == t2:
            findings.append(
                ("ERROR", f"duplicated adjacent heading {t1!r} in {p1} -- "
                          "an inserted block probably re-stated the heading it follows")
            )
    return findings


def check_formatting(log: str) -> list[tuple[str, str]]:
    """Overfull/underfull boxes and other layout complaints from the TeX log."""
    findings = []
    # Only overfull boxes matter in practice; underfull hboxes are mostly noise
    # from ragged text, and underfull vboxes from page breaks near floats.
    over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*", log)
    big = [float(x) for x in over if float(x) > 5.0]
    if big:
        worst = max(big)
        lvl = "ERROR" if worst > 20 else "WARN"
        findings.append(
            (
                lvl,
                f"{len(big)} overfull hbox(es) >5pt, worst {worst:.1f}pt "
                f"({worst * MM_PER_PT:.1f} mm past the text block) -- a table or "
                f"figure is running off the page",
            )
        )
    if re.search(r"Overfull \\vbox", log):
        findings.append(("WARN", "overfull vbox -- content pushed past the bottom margin"))
    return findings


def check_refs(log: str) -> list[tuple[str, str]]:
    findings = []
    for key in sorted(set(re.findall(r"Citation `([^']+)' (?:on page \d+ )?undefined", log))):
        findings.append(("ERROR", f"undefined citation: {key} -- not in references.bib"))
    for key in sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log))):
        findings.append(("ERROR", f"undefined reference: {key}"))
    if "There were undefined references" in log:
        findings.append(("WARN", "LaTeX reports undefined references; rerun may be needed"))
    return findings


def check_citations(srcs: dict[str, str]) -> list[tuple[str, str]]:
    """Every cited key must exist in references.bib.

    This is the mechanical form of the group rule that a relevant paper goes into
    the Zotero collection *first*. references.bib is exported from that
    collection, so 'key missing from the bib' means 'we do not hold this paper'.
    """
    findings = []
    bib = _read("references.bib")
    have = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    cited: set[str] = set()
    for txt in srcs.values():
        for m in re.finditer(r"\\(?:cite|citep|citet|mirror)\*?(?:\[[^\]]*\])*\{([^}]*)\}", txt):
            cited.update(k.strip() for k in m.group(1).split(",") if k.strip())
    missing = sorted(cited - have)
    for k in missing:
        findings.append(
            ("ERROR", f"cited but not in references.bib: {k} -- add it to the Zotero collection, then `make bib`")
        )
    unused = sorted(have - cited)
    if unused:
        findings.append(("INFO", f"{len(unused)} bib entries not cited: {', '.join(unused[:6])}{' ...' if len(unused) > 6 else ''}"))

    # \citet is silently broken under the Nature style and must never come back.
    # sn-nature.bst is a numeric style: it emits \bibitem{key} with no author-year
    # label, so natbib has nothing to print for the textual form and renders the
    # literal string "(author?) [12]" into the PDF. It is not an error, not a
    # warning, and not visible in the log -- it only shows up if someone reads the
    # typeset page. House form is to write the name out: "Brettner et al.\
    # \citep{key}", which also matches how the manuscript refers to prior work.
    for path, txt in srcs.items():
        for m in re.finditer(r"\\citet\*?(?:\[[^\]]*\])*\{([^}]*)\}", txt):
            findings.append((
                "ERROR",
                f"{path}: \\citet{{{m.group(1)}}} renders as '(author?)' under "
                "sn-nature.bst -- write the author out and use \\citep instead",
            ))

    print(f"\nCITATIONS  {len(cited)} cited, {len(have)} in bib, {len(missing)} missing")
    return findings


def check_provenance(srcs: dict[str, str]) -> list[tuple[str, str]]:
    """Generated content must name its generating script.

    Mirrors the repo's STRICT RULE: any table/figure/number used in a document
    comes from a committed script. In .tex that shows up as a `%% SOURCE:` line.
    A `\\input{}` of a table without one is the detectable violation.
    """
    findings = []
    n_src, n_ext, n_second = 0, 0, 0
    for path, txt in sorted(srcs.items()):
        n_src += len(re.findall(r"%%\s*SOURCE:", txt))
        n_ext += len(re.findall(r"\\external\{", txt))
        n_second += len(re.findall(r"\\secondhand\{", txt))
        # A tabular environment with no SOURCE comment anywhere in its file.
        if re.search(r"\\begin\{(tabular|longtable)\}", txt) and "%% SOURCE:" not in txt:
            findings.append(
                ("WARN", f"{path} has a table but no '%% SOURCE:' comment -- name the "
                         f"generating script, or say explicitly that it is hand-authored")
            )
    print(
        f"PROVENANCE  {n_src} SOURCE comments, {n_ext} external flags, "
        f"{n_second} second-hand flags"
    )
    return findings


def check_figure_widths(srcs: dict[str, str]) -> list[tuple[str, str]]:
    """Figures placed at true size must fit the text block."""
    findings = []
    refs = set()
    for txt in srcs.values():
        refs.update(re.findall(r"(figures/[A-Za-z0-9._/-]+\.pdf)", txt))
    for rel in sorted(refs):
        if not osp.exists(rel):
            findings.append(("WARN", f"figure referenced but missing: {rel}"))
            continue
        raw = open(rel, "rb").read(400_000)
        boxes = re.findall(rb"/MediaBox\s*\[\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)", raw)
        if not boxes:
            continue
        x0, _y0, x1, _y1 = (float(v) for v in boxes[0])
        mm = (x1 - x0) * MM_PER_PT
        if mm > TEXT_BLOCK_MM + 0.5:
            findings.append(
                ("ERROR", f"{rel} is {mm:.1f} mm wide, text block is {TEXT_BLOCK_MM:.0f} mm "
                          f"-- redraw to <=180 mm or place with \\tcfigfit")
            )
        else:
            print(f"  {GRY}fig{RST} {rel} {mm:.1f} mm")
    return findings



# --- style ------------------------------------------------------------------
# The prose rules from notes/writing-style-guide.md that are mechanically
# checkable. A rule that only lives in a guide gets broken; a rule the build
# checks does not. Everything here EXCLUDES verbatim quotes (``...'') and %%
# comments, because a source quote keeps its own spelling and a comment is not
# prose -- Americanizing a quote would falsify it.

BRITISH = [
    "alphabetised", "permeabilised", "permeabilisation", "organising",
    "organised", "colours", "colour", "instalment", "instalments", "optimised",
    "optimisation", "optimisations", "optimise", "reorganised", "amortised",
    "analysed", "labour", "catalogue", "normalised", "summarised",
    "characterised", "centred", "behaviour", "modelled", "labelled",
    "labelling", "parameterisation", "polymerisation",
]

# Narrating the document, and being self-referential about the group. Both were
# flagged repeatedly in review; both are phrase-level and unambiguous.
NARRATION = [
    r"[Tt]his document (asks|works out|establishes|is a)",
    r"[Tt]his (section|note) asks",
    r"the intended readers?",
]
SELF_REF = [
    r"\bfor us\b", r"\bwe (currently )?prefer\b", r"\bour goal\b",
    r"for a group whose", r"\bwe like\b",
]


def _prose_only(t: str) -> str:
    """Blank out %% comments and ``...'' quotes, preserving offsets."""
    t = re.sub(r"(?m)^%%.*$", lambda m: " " * len(m.group(0)), t)
    t = re.sub(r"``.*?''", lambda m: " " * len(m.group(0)), t, flags=re.S)
    return t


def check_style(srcs: dict[str, str]) -> list[tuple[str, str]]:
    """Enforce notes/writing-style-guide.md where it is machine-checkable."""
    out: list[tuple[str, str]] = []
    n_brit = n_dash = 0
    for name, raw in srcs.items():
        t = _prose_only(raw)
        for m in re.finditer(r"---", t):
            # LaTeX em dash. House style is a spaced en dash or a comma.
            ln = t[: m.start()].count("\n") + 1
            out.append(("ERROR", f"em-dash in {name}:{ln} -- use ` -- ` or a comma"))
            n_dash += 1
        for w in BRITISH:
            for m in re.finditer(rf"\b{w}\b", t):
                ln = t[: m.start()].count("\n") + 1
                out.append(("ERROR", f"British spelling {w!r} in {name}:{ln}"))
                n_brit += 1
        for pat in NARRATION:
            for m in re.finditer(pat, t):
                ln = t[: m.start()].count("\n") + 1
                out.append(("WARN", f"document narration in {name}:{ln}: "
                                    f"{m.group(0)!r} -- state the finding instead"))
        for pat in SELF_REF:
            for m in re.finditer(pat, t):
                ln = t[: m.start()].count("\n") + 1
                out.append(("WARN", f"self-reference in {name}:{ln}: {m.group(0)!r}"))
        # Trailing restatement is NOT checked mechanically, and the attempt is
        # worth recording so nobody retries it. The shape is a sentence-final
        # clause that re-says the main clause, and the obvious discriminator --
        # "adds no number and no cross-reference" -- flags ordinary causal
        # explanation instead ("...bursting, because of cell-cycle phase"). It
        # ran at roughly one true positive in ten, and a gate that cries wolf
        # gets ignored, which costs more than the rule it was enforcing. It is a
        # judgement call; the guide teaches the test, a human applies it.
    tot = n_brit + n_dash
    print(f"{GRY}STYLE{RST}      {tot} hard violation(s) "
          f"(spelling, em-dash); trailing restatement is a manual read -- "
          f"see notes/writing-style-guide.md")
    return out


def main() -> int:
    doc = sys.argv[1] if len(sys.argv) > 1 else "main"
    log = _read(f"{doc}.log")
    srcs = tex_sources()
    if not log:
        print(f"{YEL}no {doc}.log -- build first with `make`{RST}")

    findings: list[tuple[str, str]] = []
    findings += check_status(srcs)
    print()
    findings += check_figure_widths(srcs)
    findings += check_citations(srcs)
    findings += check_provenance(srcs)
    findings += check_style(srcs)
    findings += check_formatting(log)
    findings += check_refs(log)

    print(f"\n{'='*72}")
    errs = [f for f in findings if f[0] == "ERROR"]
    warns = [f for f in findings if f[0] == "WARN"]
    infos = [f for f in findings if f[0] == "INFO"]
    for lvl, msg in errs:
        print(f"{RED}ERROR{RST}  {msg}")
    for lvl, msg in warns:
        print(f"{YEL}WARN {RST}  {msg}")
    for lvl, msg in infos:
        print(f"{GRY}INFO {RST}  {msg}")
    if not findings:
        print(f"{GRN}clean{RST}")
    print(f"{'='*72}\n{len(errs)} error(s), {len(warns)} warning(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
