---
id: i0sc3g3kg9yqa2wcywlmwjr
title: Sgd_gene_essentiality
desc: ''
updated: 1721604030269
created: 1721598684492
---
```
[i for i in graph.G_raw.nodes['YBR160W']["phenotype_details"] if ((i['mutant_type'] == "null") and (i['strain']['display_name'] == "S288C") and (i['phenotype']['display_name'] == 'inviable'))]
[{'id': 7621604, 'mutant_type': 'null', 'locus': {...}, 'experiment': {...}, 'experiment_details': None, 'strain': {...}, 'properties': [...], 'note': None, 'phenotype': {...}, 'reference': {...}}]
special variables
function variables
0:
{'id': 7621604, 'mutant_type': 'null', 'locus': {'display_name': 'CDC28', 'id': 1268219, 'link': '/locus/S000000364', 'format_name': 'YBR160W'}, 'experiment': {'display_name': 'systematic mutation set', 'link': None, 'category': 'large-scale survey', 'note': None}, 'experiment_details': None, 'strain': {'display_name': 'S288C', 'link': '/strain/S000203483'}, 'properties': [{...}], 'note': None, 'phenotype': {'display_name': 'inviable', 'link': '/phenotype/inviable', 'id': 1355304}, 'reference': {'display_name': 'Giaever G, et al. (2002)', 'link': '/reference/S000071347', 'pubmed_id': 12140549}}
special variables
function variables
'id':
7621604
'mutant_type':
'null'
'locus':
{'display_name': 'CDC28', 'id': 1268219, 'link': '/locus/S000000364', 'format_name': 'YBR160W'}
'experiment':
{'display_name': 'systematic mutation set', 'link': None, 'category': 'large-scale survey', 'note': None}
'experiment_details':
None
'strain':
{'display_name': 'S288C', 'link': '/strain/S000203483'}
'properties':
[{'class_type': 'BIOITEM', 'bioitem': {...}, 'note': None, 'role': 'Allele'}]
special variables
function variables
0:
{'class_type': 'BIOITEM', 'bioitem': {'display_name': 'cdc28-Δ'}, 'note': None, 'role': 'Allele'}
special variables
function variables
'class_type':
'BIOITEM'
'bioitem':
{'display_name': 'cdc28-Δ'}
'note':
None
'role':
'Allele'
len():
4
len():
1
'note':
None
'phenotype':
{'display_name': 'inviable', 'link': '/phenotype/inviable', 'id': 1355304}
'reference':
{'display_name': 'Giaever G, et al. (2002)', 'link': '/reference/S000071347', 'pubmed_id': 12140549}
special variables
function variables
'display_name':
'Giaever G, et al. (2002)'
'link':
'/reference/S000071347'
'pubmed_id':
12140549
len():
3
len():
10
len():
1
```



```python
# torchcell/graph/graph.py
# [[torchcell.graph.graph]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/graph/graph.py
# Test file: torchcell/graph/test_graph.py

import glob
import gzip
import json
import logging
import os
import os.path as osp
import pickle
import shutil
import tarfile
import time
from collections import defaultdict
from datetime import datetime
from itertools import product
from typing import Set

import gffutils
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import pandas as pd
from attrs import define, field
from Bio import Seq, SeqIO
from Bio.SeqRecord import SeqRecord
from gffutils import FeatureDB
from gffutils.feature import Feature
from matplotlib.patches import Patch
from sortedcontainers import SortedDict, SortedSet
from torch_geometric.data import download_url
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from torchcell.sequence import GeneSet, Genome, ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
import torchcell

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


# BUG switching to genome for GO, this will create training issues with sql,
# but need genome for created graphs.
@define
class SCerevisiaeGraph:
...
def main():
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    print()
```

