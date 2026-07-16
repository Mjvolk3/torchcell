# torchcell/datasets/scerevisiae/gene_name_reconcile
# [[torchcell.datasets.scerevisiae.gene_name_reconcile]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/gene_name_reconcile
"""Shared retain-all gene-name reconciliation for S. cerevisiae dataset loaders.

Wraps the genome's layered resolver (:meth:`SCerevisiaeGenome.resolve_gene_name`) with the
morphology/SCMD retention policy so Ohya 2005 and Ohnuki 2018/2022 share ONE implementation
instead of per-loader copies. Every record is a real measured perturbation, so nothing is
dropped for a naming reason: a source systematic name that resolves to a current R64
identifier (live gene, SGD rename, or valid non-``"gene"`` locus) is remapped to it, UNLESS
that id is already claimed by another record in the same dataset (an SGD merge of two
distinct strains) -- then the original name is kept so the strains stay distinct. Retired /
ambiguous names are kept verbatim.
"""

import logging
import os
import os.path as osp
from collections import Counter

import pandas as pd

from torchcell.sequence.genome.scerevisiae import GeneNameStatus, SCerevisiaeGenome

log = logging.getLogger(__name__)

# Statuses whose resolved systematic id is a valid R64 identifier we prefer over the legacy
# source name (a live gene, an SGD rename, or a valid non-"gene" feature).
_REMAP_STATUSES = (
    GeneNameStatus.CURRENT,
    GeneNameStatus.RENAMED,
    GeneNameStatus.NON_GENE_FEATURE,
)


def default_genome() -> SCerevisiaeGenome:
    """Construct an ``SCerevisiaeGenome`` from ``DATA_ROOT`` (read-only reference use)."""
    data_root = os.environ["DATA_ROOT"]
    return SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )


def reconcile_systematic_names(
    genome: SCerevisiaeGenome, names: pd.Series, *, label: str
) -> pd.Series:
    """Map source systematic names to current R64 ids, retaining every record.

    Collision-safe (an SGD merge of two 2005 ORFs keeps both originals so they stay
    distinct) and drop-free for naming. Logs the status breakdown, the remapped count, the
    merge-collision names kept legacy, and the retired names retained. Returns a new Series
    aligned to ``names``.
    """
    resolutions = {name: genome.resolve_gene_name(name) for name in names.unique()}
    proposed = {
        name: (
            res.systematic_name
            if res.status in _REMAP_STATUSES and res.systematic_name is not None
            else name
        )
        for name, res in resolutions.items()
    }
    proposed_counts = Counter(proposed.values())
    final = {
        name: (name if proposed_counts[prop] > 1 else prop)
        for name, prop in proposed.items()
    }

    remapped = sum(1 for name, f in final.items() if f != name)
    by_status: dict[str, int] = {}
    for res in resolutions.values():
        by_status[res.status.value] = by_status.get(res.status.value, 0) + 1
    collided = sorted(n for n, prop in proposed.items() if proposed_counts[prop] > 1)
    retired = sorted(
        n for n, res in resolutions.items() if res.status == GeneNameStatus.RETIRED
    )
    log.info(
        "%s ORF reconciliation: %d unique names %s; %d remapped to current R64 ids; "
        "%d kept as legacy names on merge-collision %s; %d retained as retired legacy "
        "names %s",
        label,
        len(resolutions),
        by_status,
        remapped,
        len(collided),
        collided,
        len(retired),
        retired,
    )
    return names.map(final)
