# experiments/050-ecoli-benchmark/scripts/website_pages.py
# [[experiments.050-ecoli-benchmark.scripts.website_pages]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/050-ecoli-benchmark/scripts/website_pages
"""Render the E. coli benchmark pages for the benchmark website (Docusaurus MDX).

The Sphinx pages under ``docs/source/benchmark/ecoli`` are the source. This script
rewrites each one as MDX (includes inlined, MyST directives replaced, braces and angle
brackets escaped outside code) and adds an examples page built from
``examples/benchmark/*.py`` and the output ``run_examples.py`` recorded. It writes

    experiments/050-ecoli-benchmark/website/benchmark/ecoli/{index,processing,results,examples}.mdx

and the figure beside them. The website branch copies that directory into
``website/docs/benchmark/ecoli/``.

    python experiments/050-ecoli-benchmark/scripts/website_pages.py
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

SOURCE = Path("docs/source/benchmark/ecoli")
OUT = Path("experiments/050-ecoli-benchmark/website/benchmark/ecoli")
EXAMPLES = Path("examples/benchmark")
EXAMPLE_OUTPUT = Path("experiments/050-ecoli-benchmark/results/examples")
GITHUB = "https://github.com/Mjvolk3/torchcell/blob/main"
PAGES = {
    "index": (
        "E. coli tasks",
        "Tasks and splits",
        "Four E. coli benchmark tasks, their splits and the one import that loads them.",
    ),
    "processing": (
        "Processing the E. coli benchmark tasks",
        "Processing notebook",
        "How four E. coli datasets become benchmark tasks, step by step.",
    ),
    "results": (
        "Linear-probe baselines",
        "Linear-probe baselines",
        "Ridge regression on five gene representations for every task and split.",
    ),
}
EXAMPLE_TITLES = {
    "ecoli_growth_rate_campos2018": "Growth rate (Campos 2018)",
    "ecoli_morphology_campos2018": "Cell shape and cell cycle (Campos 2018)",
    "ecoli_gene_interaction_babu2014": "Genetic interactions (Babu 2014)",
    "ecoli_metabolome_fuhrer2017": "Metabolome (Fuhrer 2017)",
}
DIRECTIVE = re.compile(r"^```\{(\w+)\}\s*(.*)$")
LINKS = {"(processing)": "(./processing.mdx)", "(results)": "(./results.mdx)"}


def escape(line: str) -> str:
    """Escape what MDX would read as JSX or an expression, outside inline code."""
    parts = line.split("`")
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("{", "\\{").replace("}", "\\}").replace("<", "&lt;")
    return "`".join(parts)


def convert(path: Path) -> list[str]:
    """One Sphinx markdown file as MDX lines; includes are inlined recursively."""
    out: list[str] = []
    lines = path.read_text(encoding="utf-8").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        match = DIRECTIVE.match(line)
        if match:
            end = lines.index("```", i + 1)
            body = lines[i + 1 : end]
            kind, argument = match.group(1), match.group(2).strip()
            if kind == "include":
                out += convert(path.parent / argument)
            elif kind == "note":
                out += [":::note", "", *[escape(b) for b in body], "", ":::"]
            elif kind == "figure":
                figure = path.parent / argument
                shutil.copyfile(figure, OUT / figure.name)
                caption = " ".join(b for b in body if b and not b.startswith(":"))
                out += [
                    f"![{figure.stem}](./{figure.name})",
                    "",
                    f"*{escape(caption)}*",
                ]
            elif kind != "toctree":
                raise SystemExit(f"{path}: no rule for the directive {kind!r}")
            i = end + 1
            continue
        if line.startswith("```"):
            end = lines.index("```", i + 1)
            out += lines[i : end + 1]
            i = end + 1
            continue
        if not line.startswith("<!--"):
            out.append(escape(line))
        i += 1
    return out


def front_matter(title: str, label: str, description: str) -> list[str]:
    """Docusaurus front matter."""
    return [
        "---",
        f"title: {json.dumps(title)}",
        f"sidebar_label: {json.dumps(label)}",
        f"description: {json.dumps(description)}",
        "---",
        "",
    ]


def main() -> None:
    """Write the four MDX pages and the figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (title, label, description) in PAGES.items():
        body = convert(SOURCE / f"{name}.md")
        first = next(i for i, line in enumerate(body) if line.startswith("# "))
        text = "\n".join(front_matter(title, label, description) + body[first + 1 :])
        for old, new in LINKS.items():
            text = text.replace(old, new)
        (OUT / f"{name}.mdx").write_text(
            re.sub(r"\n{3,}", "\n\n", text).rstrip("\n") + "\n", encoding="utf-8"
        )

    versions = json.loads(
        (EXAMPLE_OUTPUT / "versions.json").read_text(encoding="utf-8")
    )
    page = front_matter(
        "Example scripts",
        "Example scripts",
        "Four complete benchmark runs, one file per task.",
    )
    page += [
        "Four complete benchmark runs, one file per task. Each loads the standardized task, builds features from the protein of each deleted gene (amino-acid composition and log length, 21 numbers), fits a ridge regression, chooses its penalty on validation and scores the test part once. They are starting points to copy: replace `gene_features` and `ridge` with your model and keep the rest.",
        "",
        "```bash",
        'pip install "torchcell>=1.8.8"',
        "python ecoli_growth_rate_campos2018.py",
        "```",
        "",
        "Run a script in an empty directory. The dataset store and the genome are downloaded on the first run. Run them one at a time: two scripts that start together in the same directory download the same genome into the same files.",
        "",
        f"The output shown under each script is one run with torchcell {versions['torchcell']}.",
        "",
    ]
    for stem, title in EXAMPLE_TITLES.items():
        page += [
            f"## {title}",
            "",
            f"[`{stem}.py`]({GITHUB}/{EXAMPLES.as_posix()}/{stem}.py)",
            "",
            "```python",
            (EXAMPLES / f"{stem}.py").read_text(encoding="utf-8").rstrip("\n"),
            "```",
            "",
            "```text",
            (EXAMPLE_OUTPUT / f"{stem}.out").read_text(encoding="utf-8").rstrip("\n"),
            "```",
            "",
        ]
    (OUT / "examples.mdx").write_text("\n".join(page), encoding="utf-8")
    print(f"wrote {sorted(p.name for p in OUT.iterdir())} to {OUT}")


if __name__ == "__main__":
    main()
