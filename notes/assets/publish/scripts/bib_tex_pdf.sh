#!/bin/bash

input_file="$1"
output_dir="$2"
output_filename="$3"
header_includes_path="${output_dir}/assets/publish/tex-templates/header-includes.tex"

echo "Edit notes/assets/publish/tex-templates/header-includes.tex for customizing spacing."

cd ./notes && pandoc -F mermaid-filter -s "${input_file}" -o "${output_dir}/assets/pdf-output/${output_filename}.pdf" --pdf-engine=xelatex --citeproc --bibliography assets/publish/bib/bib.bib --metadata csl=assets/publish/bib/nature.csl -V geometry:'top=2cm, bottom=1.5cm, left=2cm, right=2cm' --include-in-header="${header_includes_path}" --strip-comments --dpi=600 && cd ..

output_file_path="${output_dir}/assets/pdf-output/${output_filename}.pdf"
echo "Output file: ${output_file_path}"
