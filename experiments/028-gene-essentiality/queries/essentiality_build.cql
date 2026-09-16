// Gene essentiality build (027 wave 9): the genotype UNION, single deletions.
//
// WHAT THIS BUILD IS FOR, and what it deliberately is NOT. The essentiality LABEL does not
// come from this graph. It comes from Merzbacher 2025's released split
// (`yeast_essentiality_test_split.csv`), joined per genotype by deleted gene name, which is
// what Flux Cone Learning itself did: it trained on flux cones keyed by knockout with
// labels taken from the literature. This build supplies the GENOTYPE and the cell GRAPH and
// nothing else, and every phenotype column it carries is ignored by the essentiality head.
//
// WHY THE UNION, measured on the served graph 2026-09-15. A build from
// GeneEssentialitySgdDataset alone is SINGLE CLASS: all 1,140 of its records are
// `is_essential: True`, because that dataset IS the essential-gene list. Coverage of the
// released split's two classes, by dataset:
//
//   GeneEssentialitySgdDataset    11/964 nonessential, 155/157 essential
//   SmfCostanzo2016Dataset       820/964 nonessential,   1/157 essential
//   SmfKuzmin2018Dataset         192/964 nonessential,   0/157 essential
//   union of the three           826/964 nonessential, 156/157 essential  (982/1121)
//
// So the positive class is the SGD essential list and the negative class is the viable
// deletion collection, which is what a single-mutant fitness screen is.
//
// THE DEDUPLICATOR'S MERGE IS NOT A PROBLEM HERE, and that is the point of taking the label
// from the CSV. Co-grouping a fitness dataset with the essentiality dataset merges
// `is_essential` into a `fitness` record, which is exactly what made the 003 and 025 builds
// unusable for this endpoint. Since the head reads no phenotype from this build, the merge
// is irrelevant: what survives aggregation is the genotype, which is all that is needed.
//
// SINGLE deletions only. The released split is keyed on one knockout per row, so a
// multi-deletion genotype has no label in it.

MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id IN [
  'GeneEssentialitySgdDataset',
  'SmfCostanzo2016Dataset',
  'SmfKuzmin2018Dataset'
]
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
WHERE ALL(dp IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
 WHERE (NOT dp.perturbation_type ENDS WITH 'deletion')
 OR dp.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert)
 WHERE pert.perturbation_type ENDS WITH 'deletion' | pert]) = 1
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e, ref
