# torchcell/multidigraph/uniprot_api_ec.py
# [[torchcell.multidigraph.uniprot_api_ec]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/multidigraph/uniprot_api_ec.py
# Test file: torchcell/multidigraph/test_uniprot_api_ec.py

import json
import os
import os.path as osp
import time
from asyncio import Task
from collections.abc import Callable
from typing import Any, Optional

import aiohttp
import requests
from attrs import define, field
from tqdm import tqdm

from torchcell.multidigraph.validation.locus_related.locus import (
    Alias,
    InteractionOverview,
    LocusData,
    LocusDataUrl,
    PhysicalExperiments,
    Qualities,
    Reference,
    validate_data,
)

# Copy paste from the s288c gff file
# Parent=YBL105C_id001,YBL105C_id002;Name=YBL105C_CDS;orf_classification=Verified;protein_id=UniProtKB:P24583


def get_uniprot_ec(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.txt"
    response = requests.get(url)
    for line in response.text.split("\n"):
        if "EC=" in line:
            return line.split("EC=")[1].split(";")[0].strip()


ec_number = get_uniprot_ec("P24583")
print("EC Number:", ec_number)

# Should output EC Number: 2.7.11.13
# This information is contained in the gbff file also.


def main():
    pass


if __name__ == "__main__":
    main()
