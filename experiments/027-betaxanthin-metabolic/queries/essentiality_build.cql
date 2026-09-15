// Gene essentiality build (027 wave 9): SGD essentiality ONLY, single deletions.
//
// WHY A DEDICATED QUERY RATHER THAN REUSING AN EXISTING BUILD. Two built CellDatasets
// already contain GeneEssentialitySgdDataset (003-fit-int/001-small-build and
// 025-solid-growth/001-full-build, 1,140 records each), and NEITHER carries the
// `is_essential` label: measured 2026-09-14, their phenotype_label_index keys are `fitness`
// and `gene_interaction` only, because both were opened with a converter that turns the
// boolean into a fitness value (an essentiality-only record in the 003 build reads
// label_name `fitness` with value 0.0). A classifier trained on that is trained on a
// converted label, not on essentiality.
//
// Three properties this query and its build must have, each of which those builds break:
//
//   1. The dataset is opened with `converter=None`. The converter is what replaced
//      `is_essential` with `fitness`.
//   2. SINGLE-deletion genotypes only, `SIZE(...) = 1`. Merzbacher 2025's released split is
//      keyed on one knockout per row, so a multi-deletion genotype has no label in it.
//   3. No co-grouped fitness dataset. With one in the same build the deduplicator merges
//      the boolean into a fitness record again and property 1 is undone.
//
// The phenotype filter is `graph_level = 'node'`, which is what GeneEssentialityPhenotype
// declares (torchcell/datamodels/schema.py: graph_level "node", label_name "is_essential").
// Every other 027 query filters on 'metabolism' or 'global', so this is the only one that
// selects it.
//
// The deletion-in-gene_set guard is carried over from fig6_pigment_transfer.cql verbatim:
// a perturbation that is not a deletion is allowed through (there are none here), and a
// deletion must name a gene the genome carries, so a build cannot silently acquire a
// perturbation the cell graph has no node for.

MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'GeneEssentialitySgdDataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
WHERE phen.graph_level = 'node'
 AND ALL(dp IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
 WHERE (NOT dp.perturbation_type ENDS WITH 'deletion')
 OR dp.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert)
 WHERE pert.perturbation_type ENDS WITH 'deletion' | pert]) = 1
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e, ref
