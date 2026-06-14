#!/usr/bin/env python3
# paper/nature-biotech/flatten_tex.py
#
# Flatten a multi-file LaTeX manuscript into ONE .tex (no \input / \include), as
# Springer Nature / Nature Biotechnology require at submission. If a matching
# .bbl exists it is inlined in place of \bibliography{} (Springer wants the
# references pasted into the main file).
#
#   python3 flatten_tex.py submission.tex submission-flat.tex
#
# Assumes at most one \input/\include per line (our convention) and skips
# commented-out lines.
import os
import re
import sys

INC = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def expand(path):
    base = os.path.dirname(path)
    out = []
    with open(path) as fh:
        for line in fh:
            m = INC.search(line)
            if m and not line.lstrip().startswith("%"):
                inc = m.group(1)
                if not inc.endswith(".tex"):
                    inc += ".tex"
                inc_path = os.path.join(base, inc)
                if os.path.exists(inc_path):
                    out.append(f"%% --- begin {inc} ---\n")
                    out.append(expand(inc_path))
                    out.append(f"%% --- end {inc} ---\n")
                    continue
            out.append(line)
    return "".join(out)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    flat = expand(src)
    bbl = os.path.splitext(src)[0] + ".bbl"
    if os.path.exists(bbl):
        with open(bbl) as fh:
            flat = re.sub(r"\\bibliography\{[^}]*\}", fh.read(), flat)
        note = f"(with {os.path.basename(bbl)} inlined)"
    else:
        note = "(no .bbl found; \\bibliography{} left as-is -- generate the .bbl and re-run for Springer)"
    with open(dst, "w") as fh:
        fh.write(flat)
    print(f"Wrote {dst} {note}")


if __name__ == "__main__":
    main()
