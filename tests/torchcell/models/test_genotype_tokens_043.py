# tests/torchcell/models/test_genotype_tokens_043.py
# [[tests.torchcell.models.test_genotype_tokens_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_genotype_tokens_043.py
"""Shapes and marks of the 043 genotype-token prototype.

Twelve genes on two chromosomes, a 5-wide precomputed encoder projected to d = 8. Three
token sets: a single deletion, an isolate-style perturbation list (variant plus
loss-of-function on one gene, an absence, an accessory ORF off the graph) and a
segregant built from haplotype blocks under the definite assumption (non-reference
parent 2, posterior >= 0.9, whole-gene containment).
"""

import pytest
import torch

from torchcell.datamodels.schema import HaplotypeBlock
from torchcell.models.genotype_tokens_043 import (
    CHANGE_TYPE_INDEX,
    NO_PARENT,
    ChangeType,
    GeneCoordinate,
    GenotypeEncoder,
    GenotypeEncoderConfig,
    GenotypeTokenSet,
    PrecomputedGeneEmbedding,
)

GENES = [f"YA{i:02d}" for i in range(12)]
# six genes per chromosome, 1 kb each, 2 kb apart
COORDS = {
    g: GeneCoordinate(
        chromosome="chrI" if i < 6 else "chrII",
        start=1 + 2000 * (i % 6),
        end=1000 + 2000 * (i % 6),
    )
    for i, g in enumerate(GENES)
}
D_ENC, D = 5, 8


def _encoder(use_position: bool = False) -> GenotypeEncoder:
    torch.manual_seed(0)
    table = torch.randn(len(GENES), D_ENC)
    seq = PrecomputedGeneEmbedding(table, {g: i for i, g in enumerate(GENES)})
    config = GenotypeEncoderConfig(
        d_model=D,
        parent_tokens=("RM",),
        strain_token_vocab=("parent:RM", "dataset:kemmeren2014", "medium:YPD"),
        use_position=use_position,
    )
    positions = None
    if use_position:
        positions = torch.tensor(
            [[1.0 if i < 6 else 2.0, ((i % 6) * 2000 + 500) / 12000] for i in range(12)]
        )
    return GenotypeEncoder(config, GENES, seq, positions=positions)


def _deletion() -> GenotypeTokenSet:
    return GenotypeTokenSet.from_deletions(
        ["YA03"], strain_tokens=("dataset:kemmeren2014", "medium:YPD")
    )


def _isolate() -> GenotypeTokenSet:
    return GenotypeTokenSet.from_perturbations(
        [
            ("YA01", "sequence_variant"),
            ("YA01", "natural_loss_of_function"),
            ("YA05", "sequence_variant"),
            ("YA08", "natural_gene_absence"),
            ("EC1118_1F14_0012g", "natural_gene_presence"),
            ("YA11", "deletion"),
        ]
    )


def _segregant() -> GenotypeTokenSet:
    blocks = [
        # chrI: RM (parent 2) at high posterior over genes 0..2; BY (parent 1) after
        HaplotypeBlock(
            chromosome="chrI", start=1, end=5500, parent=2, posterior=0.98, n_markers=40
        ),
        HaplotypeBlock(
            chromosome="chrI",
            start=6000,
            end=12000,
            parent=1,
            posterior=1.0,
            n_markers=50,
        ),
        # chrII: RM but below threshold over genes 6..8, RM at threshold over 10..11;
        # gene 9 (start 6001) straddles the second block's start and stays reference
        HaplotypeBlock(
            chromosome="chrII", start=1, end=5000, parent=2, posterior=0.6, n_markers=12
        ),
        HaplotypeBlock(
            chromosome="chrII",
            start=6500,
            end=12000,
            parent=2,
            posterior=0.9,
            n_markers=30,
        ),
    ]
    return GenotypeTokenSet.from_segregant_blocks(
        blocks,
        COORDS,
        nonreference_parent=2,
        parent_token="RM",
        strain_tokens=("parent:RM",),
    )


def test_token_sets_hold_the_marks() -> None:
    d = _deletion()
    assert d.gene_ids == ("YA03",)
    assert d.change_types == (ChangeType.deletion,)
    assert d.allele_parent == (NO_PARENT,)

    iso = _isolate()
    marks = dict(zip(iso.gene_ids, iso.change_types))
    assert marks["YA01"] is ChangeType.loss_of_function  # outranks the variant
    assert marks["YA05"] is ChangeType.variant
    assert marks["YA08"] is ChangeType.absent
    assert marks["EC1118_1F14_0012g"] is ChangeType.present
    assert marks["YA11"] is ChangeType.deletion
    assert iso.n_marks == 5

    seg = _segregant()
    assert seg.gene_ids == ("YA00", "YA01", "YA02", "YA10", "YA11")
    assert set(seg.change_types) == {ChangeType.variant}
    assert set(seg.allele_parent) == {"RM"}


def test_encoder_shapes_and_mask() -> None:
    enc = _encoder()
    batch = enc([_deletion(), _isolate(), _segregant()])
    B, G = 3, len(GENES)
    assert batch.X.shape == (G, D)
    assert batch.X_strain.shape == (B, G, D)
    assert batch.change_types.shape == (B, G)
    assert batch.mask.shape == (B, G)
    assert batch.parent.shape == (B, G)
    assert batch.mask.sum(dim=1).tolist() == [1, 4, 5]
    assert batch.perturbation_indices.shape == batch.perturbation_indices_batch.shape
    assert batch.perturbation_indices.numel() == 10
    assert batch.perturbation_indices_batch.tolist() == [0] + [1] * 4 + [2] * 5
    # the deletion row carries the deletion index, the segregant rows parent RM (row 1)
    assert batch.change_types[0, 3].item() == CHANGE_TYPE_INDEX[ChangeType.deletion]
    assert batch.parent[2, 0].item() == 1 and batch.parent[1, 1].item() == 0
    # the accessory ORF is reported, not dropped
    assert batch.off_graph_marks == (
        (),
        (("EC1118_1F14_0012g", ChangeType.present),),
        (),
    )
    # strain tokens: K_max = 2, padded where a strain has fewer
    assert batch.strain_tokens.shape == (B, 2, D)
    assert batch.strain_token_mask.tolist() == [
        [False, False],
        [True, True],
        [False, True],
    ]
    # a reference gene's realized token is its reference token
    ref_rows = ~batch.mask[1]
    assert torch.allclose(batch.X_strain[1][ref_rows], batch.X[ref_rows])
    assert not torch.allclose(batch.X_strain[0, 3], batch.X[3])


def test_position_feature_changes_reference_tokens_only_when_enabled() -> None:
    without = _encoder(use_position=False)
    with_pos = _encoder(use_position=True)
    assert torch.allclose(
        without.reference_tokens(),
        without.proj(without.sequence_encoder.encode_genes(GENES)),
    )
    X = with_pos.reference_tokens()
    assert X.shape == (len(GENES), D)
    assert not torch.allclose(
        X, with_pos.proj(with_pos.sequence_encoder.encode_genes(GENES))
    )


def test_refusals() -> None:
    with pytest.raises(ValueError, match="one mark"):
        GenotypeTokenSet(
            gene_ids=("YA01", "YA01"),
            change_types=(ChangeType.variant, ChangeType.deletion),
            allele_parent=(NO_PARENT, NO_PARENT),
        )
    with pytest.raises(ValueError, match="promoter_sequences"):
        GenotypeTokenSet(
            gene_ids=("YFP",),
            change_types=(ChangeType.promoter,),
            allele_parent=(NO_PARENT,),
        )
    enc = _encoder()
    promoter = GenotypeTokenSet(
        gene_ids=("YFP",),
        change_types=(ChangeType.promoter,),
        allele_parent=(NO_PARENT,),
        promoter_sequences={"YFP": "ACGT" * 20},
    )
    with pytest.raises(ValueError, match="no promoter encoder"):
        enc([promoter])
    with pytest.raises(KeyError):
        GenotypeTokenSet.from_perturbations([("YA01", "crispr_activation")])
