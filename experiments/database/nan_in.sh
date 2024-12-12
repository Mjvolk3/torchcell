#!/bin/bash
for file in /Users/michaelvolk/Documents/projects/torchcell/database/biocypher-out/2024-09-16_19-32-30/*.csv; do
  if grep -q "NaN" "$file"; then
    echo "$file contains NaN"
  fi
done
