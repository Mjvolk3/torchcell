from pathlib import Path
from typing import List, Dict
import re
import subprocess
import sys


def install_mdpdf() -> None:
    """Install mdpdf if not already installed."""
    try:
        subprocess.run(["mdpdf", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("Installing mdpdf...")
        subprocess.run(["npm", "install", "-g", "mdpdf"], check=True)


def combine_readmes(base_dir: str, output_file: str) -> None:
    """Combine all README files into a single markdown file."""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory {base_dir} does not exist")

    pmid_dirs = [d for d in base_path.glob("*") if d.is_dir() and "PMID" in d.name]
    print(f"Found {len(pmid_dirs)} study directories")

    def extract_year(dir_path: Path) -> int:
        year_match = re.search(r"_(\d{4})_", dir_path.name)
        return int(year_match.group(1)) if year_match else 0

    pmid_dirs.sort(key=lambda x: (extract_year(x), x.name))

    content_lines = [
        "# SGD Expression Studies\n\n",
        "This document contains README files from all expression studies in SGD.\n\n",
        "## Table of Contents\n\n",
    ]

    # Generate TOC
    for dir_path in pmid_dirs:
        content_lines.append(f"- [{dir_path.name}](#{dir_path.name.lower()})\n")

    content_lines.append("\n---\n\n")

    # Add content
    for dir_path in pmid_dirs:
        try:
            readme_path = next(dir_path.glob("*.README"))
            content = readme_path.read_text()
            content_lines.extend([f"# {dir_path.name}\n", content.strip(), "\n---\n\n"])
            print(f"Processed {dir_path.name}")
        except Exception as e:
            print(f"Error processing {dir_path.name}: {e}")

    output_path = Path(output_file)
    output_path.write_text("\n".join(content_lines))
    print(f"\nCombined {len(pmid_dirs)} README files into {output_file}")


def convert_to_pdf(markdown_file: str, pdf_file: str) -> None:
    """Convert markdown file to PDF using mdpdf."""
    try:
        install_mdpdf()
        print(f"Converting {markdown_file} to PDF...")
        subprocess.run(["mdpdf", markdown_file, pdf_file], check=True)
        print(f"Successfully created {pdf_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting to PDF: {e}")
        print("You may need to install Node.js first: https://nodejs.org/")


if __name__ == "__main__":
    base_dir = "/Users/michaelvolk/Documents/projects/torchcell/data/all_spell_readmes"
    markdown_file = "combined.md"
    pdf_file = "sgd_expression_studies.pdf"

    # First combine READMEs into markdown
    combine_readmes(base_dir, markdown_file)

    # Then convert to PDF
    convert_to_pdf(markdown_file, pdf_file)
