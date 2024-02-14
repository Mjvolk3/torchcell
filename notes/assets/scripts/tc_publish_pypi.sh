#!/bin/bash
rm -rf dist
conda activate torchcell
python -m build
twine upload dist/*
