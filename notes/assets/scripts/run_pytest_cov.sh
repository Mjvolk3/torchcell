#!/bin/bash
source /Users/michaelvolk/opt/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
pytest --cov=torchcell --cov-report html tests/
