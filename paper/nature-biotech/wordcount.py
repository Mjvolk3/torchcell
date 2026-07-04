#!/usr/bin/env python3
"""
paper/nature-biotech/wordcount.py

Per-section word-count report against the \\cb{...words} budget tags, so `make paper`
shows how full each section is (actual / budget). Counts BODY PROSE only: figure/table
environments and \\cb{} editing tags are stripped, then texcount (LaTeX-aware) counts the
remaining words. Non-fatal by default -- it informs, it does not gate the build.

  make wordcount        # print the report
  python wordcount.py   # same

Budgets are read from the source (the \\cb{target ~N words} / \\cb{~N words} tags), so the
report stays correct as budgets change. "no limit" sections are counted but not rated.
"""
from __future__ import annotations
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEXCOUNT = os.environ.get("TEXCOUNT", "texcount")

# Ordered (file, is_results) list mirrors content.tex. Results is split by \bmhead.
FILES = ["frontmatter", "introduction", "results", "discussion", "methods"]

CB = re.compile(r"\\cb\{")
FIG = re.compile(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", re.S)


def strip_braced(text: str, macro_open_re: re.Pattern) -> str:
    """Remove `macro{...}` allowing one level of nested braces."""
    out = []
    i = 0
    while i < len(text):
        m = macro_open_re.search(text, i)
        if not m:
            out.append(text[i:]); break
        out.append(text[i:m.start()])
        j = m.end(); depth = 1
        while j < len(text) and depth:
            if text[j] == "{": depth += 1
            elif text[j] == "}": depth -= 1
            j += 1
        i = j
    return "".join(out)


def budget_of(heading: str):
    """Parse a word budget from a heading's \\cb tag. Returns int, 'nolimit', or None."""
    m = re.search(r"\\cb\{([^{}]*)\}", heading)
    if not m: return None
    tag = m.group(1)
    if "limit" in tag: return "nolimit"
    n = re.search(r"([\d,]+)\s*words", tag)
    return int(n.group(1).replace(",", "")) if n else None


FIGPH = re.compile(r"\\figph\{")


def _skip_group(text: str, j: int) -> int:
    """Given j just after an opening '{', return the index just past its matching '}'."""
    depth = 1
    while j < len(text) and depth:
        if text[j] == "{": depth += 1
        elif text[j] == "}": depth -= 1
        j += 1
    return j


def strip_figph(text: str) -> str:
    """Remove \\figph{height}{body} placeholder boxes (both brace groups)."""
    out, i = [], 0
    while i < len(text):
        m = FIGPH.search(text, i)
        if not m:
            out.append(text[i:]); break
        out.append(text[i:m.start()])
        j = _skip_group(text, m.end())          # end of {height}
        while j < len(text) and text[j].isspace():
            j += 1
        if j < len(text) and text[j] == "{":     # {body}
            j = _skip_group(text, j + 1)
        i = j
    return "".join(out)


def count_words(body: str) -> int:
    """Body prose word count via texcount, after stripping floats, placeholders, \\cb tags."""
    body = FIG.sub(" ", body)
    body = strip_figph(body)
    body = strip_braced(body, CB)
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False) as f:
        f.write(body); path = f.name
    try:
        out = subprocess.run([TEXCOUNT, "-1", "-sum=1,0,0", "-q", path],
                             capture_output=True, text=True)
        m = re.search(r"\d+", out.stdout)
        return int(m.group()) if m else 0
    finally:
        os.unlink(path)


# Stable ids used by \wc{id} inline tags in the section headings.
SECTION_ID = {"introduction": "intro", "results": "results",
              "discussion": "discussion", "methods": "methods"}


def rows():
    """Yield (indent, id, name, words, budget) in document order."""
    for stem in FILES:
        path = os.path.join(HERE, "sections", f"{stem}.tex")
        if not os.path.isfile(path): continue
        text = open(path).read()

        if stem == "frontmatter":
            m = re.search(r"\\abstract\{(.*?)\}\s*(?:\n\n|\\keywords)", text, re.S)
            if m:
                b = budget_of(text[text.find("abstractname"):text.find("abstractname")+60]) or 150
                yield 0, "abstract", "Abstract", count_words(m.group(1)), b
            continue

        # Section + \bmhead headings, split the file into heading-bodies.
        heads = [(mm.start(), mm.group(0)) for mm in
                 re.finditer(r"\\section\{[^\n]*\}|\\bmhead\{[^\n]*\}", text)]
        bcount = 0
        for idx, (pos, head) in enumerate(heads):
            is_section = head.startswith("\\section")
            if is_section:
                # Section total spans all its \bmhead subsections (to the next \section/EOF).
                end = len(text)
                for p2, h2 in heads[idx + 1:]:
                    if h2.startswith("\\section"): end = p2; break
                hid = SECTION_ID.get(stem, stem)
            else:
                end = heads[idx + 1][0] if idx + 1 < len(heads) else len(text)
                bcount += 1
                hid = f"r{bcount}"   # results subsections -> r1..r5
            body = text[text.find("}", pos) + 1:end]
            name = re.sub(r"\\cb\{[^{}]*\}", "", head)
            name = re.sub(r"\\(section|bmhead)\{|\}|\\label\{[^}]*\}", "", name).strip()
            name = re.sub(r"\\[a-zA-Z]+|[{}$]", "", name).strip()
            yield (0 if is_section else 1), hid, name, count_words(body), budget_of(head)


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde ")):
        s = s.replace(a, b)
    return s


def _color(words, budget):
    if budget in ("nolimit", None): return "gray"
    r = words / budget
    return "green!55!black" if 0.75 <= r <= 1.10 else ("orange!85!black" if r < 0.75 else "red")


def emit_tex(path: str):
    """Write editing-only macros: inline per-heading \\wc@<id> tags + a \\wcsummarytable.
    Input this file BEFORE the body so the inline macros exist when headings typeset."""
    data = list(rows())
    out = [r"%% AUTO-GENERATED by wordcount.py --tex; do not edit. Input by editing.tex.",
           r"%% Inline per-heading counts: \wc{id} expands these next to each heading."]
    # 1) inline macros keyed by id -> colored "words/budget"
    for _indent, hid, _name, words, budget in data:
        col = _color(words, budget)
        val = f"{words}" if budget in ("nolimit", None) else f"{words}/{budget}"
        out.append(f"\\expandafter\\gdef\\csname wc@{hid}\\endcsname{{\\textcolor{{{col}}}{{{val}}}}}")
    # 2) the full summary table, wrapped so it renders where \wcsummarytable is called
    out += [r"\gdef\wcsummarytable{%",
            r"\clearpage",
            r"{\footnotesize\noindent\textbf{Draft word-budget report}\ "
            r"{\normalfont\small\color{gray}(body prose vs \texttt{\textbackslash cb} budget; "
            r"figures/captions excluded; auto-generated at build)}\par\medskip",
            r"\begin{tabular}{@{}l r r r@{}}", r"\toprule",
            r"Section & Words & Budget & \% \\", r"\midrule"]
    for indent, _hid, name, words, budget in data:
        nm = ("\\quad " if indent else "") + _tex_escape(name)
        if budget in ("nolimit", None):
            out.append(f"{nm} & {words} & \\textemdash & \\textemdash \\\\")
        else:
            r = words / budget
            out.append(f"{nm} & {words} & {budget} & \\textcolor{{{_color(words,budget)}}}{{{r*100:.0f}}} \\\\")
    out += [r"\bottomrule", r"\end{tabular}\par}}"]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {path}")


def main():
    if "--tex" in sys.argv:
        i = sys.argv.index("--tex")
        emit_tex(sys.argv[i + 1] if i + 1 < len(sys.argv) else "wordcount.tex")
        return
    tty = sys.stdout.isatty()
    G = "\033[32m" if tty else ""; Y = "\033[33m" if tty else ""
    R = "\033[31m" if tty else ""; D = "\033[2m" if tty else ""
    B = "\033[1m" if tty else ""; Z = "\033[0m" if tty else ""
    print(f"{B}== Word-count report =={Z}  {D}body prose vs \\cb budget (figures/captions excluded){Z}")
    for indent, _hid, name, words, budget in rows():
        pad = "  " + ("  " if indent else "")
        label = (name[:52] + "…") if len(name) > 53 else name
        if budget == "nolimit":
            print(f"{pad}{D}   --      {words:5d} words   {label}  (no limit){Z}")
        elif budget is None:
            print(f"{pad}   --      {words:5d} words   {label}")
        else:
            ratio = words / budget
            col = G if 0.75 <= ratio <= 1.10 else (Y if ratio < 0.75 else R)
            bar_n = min(int(ratio * 10), 12)
            bar = "#" * bar_n + "." * max(0, 10 - bar_n)
            print(f"{pad}{col}{ratio:4.0%}{Z}  {D}{bar}{Z}  "
                  f"{words:5d} / {budget:<5d}  {label}")
    print(f"  {D}legend: {G}75-110%{Z}{D} on target · {Y}<75%{Z}{D} thin · {R}>110%{Z}{D} over{Z}")


if __name__ == "__main__":
    main()
