#!/bin/bash

input_file="$1"
output_dir="$2"
output_filename="$3"
reference_docx="/Users/michaelvolk/Documents/projects/torchcell/notes/assets/publish/ms_word_ref/paper-reference.docx"

echo "Using paper-reference.docx template for Word formatting."

# Create docx-output directory if it doesn't exist
mkdir -p "${output_dir}/assets/docx-output"

# Check if reference document exists
if [ ! -f "${reference_docx}" ]; then
  echo "Error: Reference document not found at: ${reference_docx}"
  exit 1
fi

cd ./notes && pandoc -F mermaid-filter \
  --metadata link-citations=true \
  -s "${input_file}" \
  -o "${output_dir}/assets/docx-output/${output_filename}.docx" \
  --reference-doc="${reference_docx}" \
  --citeproc \
  --bibliography assets/publish/bib/bib.bib \
  --metadata csl=assets/publish/bib/nature.csl \
  --strip-comments && cd ..

output_file_path="${output_dir}/assets/docx-output/${output_filename}.docx"
echo "Output file: ${output_file_path}"
