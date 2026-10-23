"""The supported queries admit the released-identifier leaf where a loader writes it.

Since PR #1085 (#1045) a symbol-named deletion strain sits on ``ReleasedIdentifierPerturbation``
with the symbol in ``systematic_gene_name``, which a ``systematic_gene_name IN $gene_set`` clause
rejects (Cachera 2023: 1,101 of 4,759 returned, 2026-10-23). The blocks of datasets whose loader
writes that leaf with released kind deletion admit ``released_identifier`` without the gene-set
test; the blocks of datasets that never write it keep the strict clause.
"""

from __future__ import annotations

import re
from pathlib import Path

QUERIES = (
    Path(__file__).resolve().parents[4] / "torchcell" / "knowledge_graphs" / "queries"
)
ADMIT = "perturbation_type = 'released_identifier'"

#: (query file, dataset id) -> whether its block admits the released-identifier leaf.
EXPECTED: dict[tuple[str, str], bool] = {
    ("amino_acid_betaxanthin.cql", "AminoAcidMulleder2016Dataset"): True,
    ("amino_acid_betaxanthin.cql", "BetaxanthinCachera2023Dataset"): True,
    ("amino_acid_betaxanthin.cql", "AminoAcidCooper2010Dataset"): False,
    ("expression_proteome_morphology.cql", "ProteomeMessner2023Dataset"): True,
    ("expression_proteome_morphology.cql", "MicroarrayKemmeren2014Dataset"): False,
    ("expression_proteome_morphology.cql", "ScmdOhya2005Dataset"): False,
    ("essentiality_smf.cql", "SmfCostanzo2016Dataset"): False,
    ("solid_growth_025.cql", "SmfCostanzo2016Dataset"): False,
}


def _block(text: str, dataset_id: str) -> str:
    blocks = [b for b in re.split(r"\n\nUNION ALL\n\n", text) if f"'{dataset_id}'" in b]
    assert len(blocks) == 1, (dataset_id, len(blocks))
    return blocks[0]


def test_blocks_admit_the_released_identifier_leaf_exactly_where_the_loader_writes_it() -> (
    None
):
    for (query, dataset_id), admits in EXPECTED.items():
        block = _block((QUERIES / query).read_text(), dataset_id)
        assert (ADMIT in block) is admits, (query, dataset_id)


def test_an_admitting_block_still_requires_one_deletion_kind_perturbation() -> None:
    """Admission widens the deletion test, it does not drop it: the Cachera block still asks
    for at least one deletion-kind perturbation and keeps the gene-set test for placed ones.
    """
    block = _block(
        (QUERIES / "amino_acid_betaxanthin.cql").read_text(),
        "BetaxanthinCachera2023Dataset",
    )
    assert block.count(ADMIT) == 2  # once in the ALL clause, once in the SIZE clause
    assert "dp.systematic_gene_name IN $gene_set" in block
    assert "| pert]) > 0" in block
