# experiments/019-simb-multimodal/scripts/generate_fig6_cql.py
# [[experiments.019-simb-multimodal.scripts.generate_fig6_cql]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/generate_fig6_cql
"""Generate the Fig-6 production / metabolite build query (WS4).

Emits ``fig6_build.cql`` into ``experiments/019-simb-multimodal/queries`` -- the
UNION ALL of every Fig-6 dataset (two pigment production screens, three
metabolome screens, and the isobutanol production screen + its validated
subset). Fig-6 is the "deletion recommendations for a production target" panel,
paired with WS11b ("does adding a metabolome label improve production
prediction").

Unlike Fig-3 there is NO large-set problem here: the biggest block is ~4.7k
records and the whole union is ~18.9k, all tractable -- so there is a single
canonical query, no "core" pruning variant (per WS4 Design Decision 1 the
isobutanol screen stays IN; the noisy screen vs the validated subset are
separated downstream by split-indices, not by a second build).

Per-modality ``graph_level`` (= CGT head, verified against the live DB):
* ``CarotenoidOzaydin2013Dataset``   -> ``global``     (beta-carotene visual score)
* every metabolite / production set  -> ``metabolism`` (per-metabolite vector head)

DELETION-in-gene_set filter, NOT all-in-gene_set (the load-bearing difference
from Fig-3): the two pigment strains carry heterologous biosynthesis cassettes
as ``gene_addition`` perturbations (beta-carotene = 3 added genes, betaxanthin =
4), and those cassette genes are NOT in the S288C ``gene_set``. Requiring ALL
perturbations in ``gene_set`` (the Fig-3 clause) would drop all ~9.2k pigment
records. So each block instead requires only that every DELETION perturbation's
gene is in ``gene_set`` (so the deleted S288C locus is a real graph node the
``Perturbation`` processor can index) while leaving cassette additions
unrestricted (the processor ignores names absent from the gene graph). At least
one deletion is required so a strain is never all-cassette / effectively WT.

Consequence carried into the census (query_fig6.py): because
``GenotypeAggregator`` keys on the FULL perturbed gene set (deletions +
additions), a cassette-bearing pigment genotype can never share a bucket with a
single-KO metabolome genotype -- betaxanthin / beta-carotene co-locate with
metabolome on ZERO genotypes, whereas the cassette-free isobutanol screen (pure
single-KO) co-locates on thousands. That asymmetry is THE Fig-6 / WS11b finding.

media / temperature are MATCHed but NOT filtered -- the sets span SC-URA / SC /
SM / YPD, all at 30 C; downstream WS6 conditions on the carried env, so no record
is dropped on environment.
"""

from __future__ import annotations

import os.path as osp

# (dataset_id, graph_level)
PIGMENT = [
    ("CarotenoidOzaydin2013Dataset", "global"),  # beta-carotene visual score
    ("BetaxanthinCachera2023Dataset", "metabolism"),  # betaxanthin production
]
METABOLOME = [
    ("AminoAcidMulleder2016Dataset", "metabolism"),  # 19-AA metabolome
    ("MetaboliteZelezniak2018Dataset", "metabolism"),  # kinase-KO metabolome
    ("MetaboliteDaSilveira2014Dataset", "metabolism"),  # lipidome
]
ISOBUTANOL = [
    ("IsobutanolScreenLopez2024Dataset", "metabolism"),  # full YKO screen
    ("IsobutanolValidatedLopez2024Dataset", "metabolism"),  # validated subset
]

ALL_BLOCKS = PIGMENT + METABOLOME + ISOBUTANOL


def _block(dataset_id: str, graph_level: str) -> str:
    """Return one dataset ``MATCH ... RETURN e, ref`` block (deletion-in-gene_set)."""
    return f"""// {dataset_id} (graph_level '{graph_level}')
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = '{dataset_id}'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
WHERE phen.graph_level = '{graph_level}'
 AND ALL(dp IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
 WHERE (NOT dp.perturbation_type ENDS WITH 'deletion')
 OR dp.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert)
 WHERE pert.perturbation_type ENDS WITH 'deletion' | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e, ref"""


def _render(blocks: list[tuple[str, str]], header: str) -> str:
    body = "\n\nUNION ALL\n\n".join(_block(*b) for b in blocks)
    return f"{header}\n\n{body}\n"


def main() -> None:
    here = osp.dirname(osp.abspath(__file__))
    qdir = osp.abspath(osp.join(here, "..", "queries"))

    header = (
        "// Fig-6 production / metabolite union: 2 pigment production screens\n"
        "// (beta-carotene visual score + betaxanthin) + 3 metabolome screens\n"
        "// (Mulleder AA, Zelezniak, da Silveira lipids) + isobutanol screen +\n"
        "// validated subset. deletion-in-gene_set filter keeps heterologous\n"
        "// cassette additions (pigment strains); >=1 deletion required.\n"
        "// Generated by experiments/019-simb-multimodal/scripts/generate_fig6_cql.py"
    )
    with open(osp.join(qdir, "fig6_build.cql"), "w") as f:
        f.write(_render(ALL_BLOCKS, header))
    print(f"wrote {osp.join(qdir, 'fig6_build.cql')}")


if __name__ == "__main__":
    main()
