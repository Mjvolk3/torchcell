#!/bin/bash
cd /Users/michaelvolk/Documents/projects/torchcell
rm -rf ./dist
eval "$(conda shell.bash hook)"
conda activate torchcell
python -m build
twine upload dist/*